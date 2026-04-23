"""find max-activating examples for each SAE feature across all layers."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sparsify import SparseCoder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from phloem.data_loader import stream_token_batches
from phloem.env import load_env
from phloem.utils.device import resolve_device

load_env()

MODEL_NAME = "google/gemma-4-E2B"
CONTEXT_WINDOW = 20  # tokens of context on each side of the activating token


def collect_max_activations(
    checkpoint: str,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    max_tokens: int = 10_000_000,
    seq_len: int = 1024,
    batch_size: int = 16,
    top_n: int = 20,
    device: str = "cuda",
) -> dict[str, list[list]]:
    """run inference and collect top-N activating examples per feature per layer.

    returns {hookpoint: [[feature_idx, activation, token, context], ...]}
    """
    # load SAEs for all hookpoints
    saes: dict[str, SparseCoder] = {}
    for hp in hookpoints:
        sae_path = f"{checkpoint}/{hp}"
        if not Path(sae_path).exists():
            print(f"  skipping {hp} (no checkpoint found)")
            continue
        saes[hp] = SparseCoder.load_from_disk(sae_path, device=device)
        saes[hp].eval()
    print(f"loaded {len(saes)} SAEs")

    # register hooks on all target modules
    modules = dict(model.base_model.named_modules())
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            captured[name] = output.detach()
        return hook_fn

    handles = []
    for hp in saes:
        handles.append(modules[hp].register_forward_hook(make_hook(hp)))

    # per-feature tracking: for each hookpoint, keep the top-N activation values
    # and the (batch_step, batch_idx, seq_idx) coordinates to reconstruct context later.
    # top_vals[hp] shape: (num_latents, top_n) — the N highest activations seen so far.
    # top_coords[hp] shape: (num_latents, top_n, 3) — (batch_step, batch_idx, seq_idx).
    # we also store all token_ids to look up context at the end.
    top_vals: dict[str, torch.Tensor] = {}
    top_coords: dict[str, torch.Tensor] = {}
    for hp, sae in saes.items():
        top_vals[hp] = torch.full((sae.num_latents, top_n), -float("inf"))
        top_coords[hp] = torch.zeros((sae.num_latents, top_n, 3), dtype=torch.long)

    # store all batch token_ids so we can build context strings at the end.
    # at 10M tokens / 1024 seq_len / 16 batch = 610 batches × 16 × 1024 ints = ~40 MB.
    all_token_ids: list[torch.Tensor] = []

    batch_step = 0
    for batch in tqdm(
        stream_token_batches(tokenizer, batch_size=batch_size, seq_len=seq_len, max_tokens=max_tokens),
        desc="scanning",
        total=max_tokens // (batch_size * seq_len),
    ):
        captured.clear()
        with torch.no_grad():
            model(batch.to(device))

        all_token_ids.append(batch.cpu())

        for hp, sae in saes.items():
            acts = captured[hp]
            if acts.dim() != 3:
                continue
            b, s, d = acts.shape

            with torch.no_grad():
                out = sae(acts.reshape(-1, d).float())

            # out.latent_indices: (b*s, k), out.latent_acts: (b*s, k)
            indices = out.latent_indices.cpu()  # (b*s, k)
            values = out.latent_acts.cpu().float()  # (b*s, k)

            # scatter activations into a dense (num_latents, b*s) matrix, then
            # find per-feature max across all positions in one vectorized pass.
            n_tokens = b * s
            k = indices.shape[1]

            # build per-feature max activation and argmax across this batch.
            # dense_acts[feat, token_pos] = activation if that feature fired, else 0.
            dense_acts = torch.zeros(sae.num_latents, n_tokens)
            feat_flat = indices.reshape(-1)  # (b*s*k,)
            vals_flat = values.reshape(-1)   # (b*s*k,)
            pos_flat = torch.arange(n_tokens).unsqueeze(1).expand(-1, k).reshape(-1)  # (b*s*k,)
            dense_acts[feat_flat, pos_flat] = vals_flat

            # for each feature, get its single best activation in this batch
            best_vals, best_pos = dense_acts.max(dim=1)  # (num_latents,) each

            # check which features have a new activation that beats the current
            # worst in their top-N heap (the min of top_vals[hp][feat]).
            current_mins, current_min_idx = top_vals[hp].min(dim=1)  # (num_latents,)
            improved = best_vals > current_mins  # (num_latents,) bool mask

            if improved.any():
                # build coordinates for the improved features
                improved_pos = best_pos[improved]  # token positions within this batch
                batch_indices = improved_pos // s
                seq_indices = improved_pos % s

                coords = torch.stack([
                    torch.full_like(batch_indices, batch_step),
                    batch_indices,
                    seq_indices,
                ], dim=1)  # (n_improved, 3)

                # replace the worst entry in each improved feature's top-N
                replace_idx = current_min_idx[improved]  # which slot to replace
                feat_mask = improved.nonzero(as_tuple=True)[0]

                top_vals[hp][feat_mask, replace_idx] = best_vals[improved]
                top_coords[hp][feat_mask, replace_idx] = coords

        batch_step += 1

    for h in handles:
        h.remove()

    # convert to result rows, resolving coordinates to context strings
    print("resolving context strings...")
    results = {}
    for hp in saes:
        rows = []
        vals = top_vals[hp]      # (num_latents, top_n)
        coords = top_coords[hp]  # (num_latents, top_n, 3)

        for feat_idx in range(vals.shape[0]):
            for slot in range(top_n):
                val = vals[feat_idx, slot].item()
                if val == -float("inf"):
                    continue
                bs, bi, si = coords[feat_idx, slot].tolist()
                token_ids = all_token_ids[bs][bi]
                ctx = _build_context(tokenizer, token_ids, si)
                rows.append([feat_idx, val, ctx["token"], ctx["context"]])

        # sort by feature then by activation descending
        rows.sort(key=lambda r: (r[0], -r[1]))
        results[hp] = rows
        n_features = len(set(r[0] for r in rows))
        print(f"  {hp}: {len(rows)} examples across {n_features} features")

    return results


def _build_context(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: torch.Tensor,
    position: int,
) -> dict:
    """extract the activating token and its surrounding context."""
    ids = token_ids.tolist()
    start = max(0, position - CONTEXT_WINDOW)
    end = min(len(ids), position + CONTEXT_WINDOW + 1)

    token_str = tokenizer.decode([ids[position]])
    context_ids = ids[start:end]
    context_str = tokenizer.decode(context_ids)

    return {"token": token_str.strip(), "context": context_str.strip()}


def save_parquet(results: dict[str, list[list]], output_dir: str) -> None:
    """save max-activating examples as one parquet file per hookpoint."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("feature", pa.int32()),
        ("activation", pa.float32()),
        ("token", pa.string()),
        ("context", pa.string()),
    ])

    for hp, rows in results.items():
        if not rows:
            continue
        table = pa.table(
            {
                "feature": [r[0] for r in rows],
                "activation": [r[1] for r in rows],
                "token": [r[2] for r in rows],
                "context": [r[3] for r in rows],
            },
            schema=schema,
        )
        filename = hp.replace(".", "_") + ".parquet"
        pq.write_table(table, output_path / filename)
        print(f"  saved {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="find max-activating examples for SAE features")
    parser.add_argument(
        "checkpoint",
        help="checkpoint directory (e.g. models/gemma-4-e2b/checkpoints/all-layers-32x-8B)",
    )
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=20, help="examples to keep per feature")
    parser.add_argument(
        "--layers",
        default="all",
        help="comma-separated layer indices, or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to write parquet files (default: <checkpoint>/max_activations/)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    resolved_device = resolve_device(args.device)

    print(f"loading model: {MODEL_NAME} on {resolved_device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": resolved_device},
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if args.layers == "all":
        hookpoints = [f"language_model.layers.{i}" for i in range(35)]
    else:
        hookpoints = [f"language_model.layers.{i}" for i in args.layers.split(",")]

    output_dir = args.output_dir or f"{args.checkpoint}/max_activations"

    print(f"scanning {args.max_tokens:,} tokens for max-activating examples (top {args.top_n} per feature)")
    results = collect_max_activations(
        checkpoint=args.checkpoint,
        hookpoints=hookpoints,
        tokenizer=tokenizer,
        model=model,
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        top_n=args.top_n,
        device=resolved_device,
    )

    print(f"\nsaving to {output_dir}/")
    save_parquet(results, output_dir)
    print("done.")
