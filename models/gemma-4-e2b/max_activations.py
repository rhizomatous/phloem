"""find max-activating examples for each SAE feature across all layers."""

import argparse
import heapq
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

    # per-feature max-heaps: {hookpoint: {feature_idx: [(act, token_str, context_str), ...]}}
    # using min-heaps of size top_n (heapq is a min-heap, so smallest gets evicted)
    heaps: dict[str, dict[int, list]] = {hp: {} for hp in saes}
    counter = 0  # tiebreaker so the heap never compares dicts

    tokens_seen = 0
    for batch in tqdm(
        stream_token_batches(tokenizer, batch_size=batch_size, seq_len=seq_len, max_tokens=max_tokens),
        desc="scanning",
        total=max_tokens // (batch_size * seq_len),
    ):
        captured.clear()
        with torch.no_grad():
            model(batch.to(device))

        for hp, sae in saes.items():
            acts = captured[hp]
            if acts.dim() == 3:
                b, s, d = acts.shape
            else:
                continue

            with torch.no_grad():
                out = sae(acts.reshape(-1, d).float())

            # out.latent_indices: (b*s, k), out.latent_acts: (b*s, k)
            indices = out.latent_indices.cpu()
            values = out.latent_acts.cpu().float()
            token_ids = batch  # (b, s)

            for pos in range(b * s):
                batch_idx = pos // s
                seq_idx = pos % s

                for j in range(indices.shape[1]):
                    feat = indices[pos, j].item()
                    val = values[pos, j].item()

                    heap = heaps[hp].get(feat)
                    if heap is None:
                        heaps[hp][feat] = []
                        heap = heaps[hp][feat]

                    if len(heap) < top_n:
                        ctx = _build_context(tokenizer, token_ids[batch_idx], seq_idx)
                        heapq.heappush(heap, (val, counter, ctx))
                        counter += 1
                    elif val > heap[0][0]:
                        ctx = _build_context(tokenizer, token_ids[batch_idx], seq_idx)
                        heapq.heapreplace(heap, (val, counter, ctx))
                        counter += 1

        tokens_seen += batch.numel()

    # clean up hooks
    for h in handles:
        h.remove()

    # convert heaps to sorted lists (highest activation first)
    results = {}
    for hp in saes:
        rows = []
        for feat_idx, heap in heaps[hp].items():
            for val, _counter, ctx in sorted(heap, reverse=True):
                rows.append([feat_idx, val, ctx["token"], ctx["context"]])
        results[hp] = rows

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
        # use hookpoint as filename, replacing dots with underscores
        filename = hp.replace(".", "_") + ".parquet"
        pq.write_table(table, output_path / filename)
        n_features = len(set(r[0] for r in rows))
        print(f"  {hp}: {len(rows)} examples across {n_features} features → {filename}")


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
