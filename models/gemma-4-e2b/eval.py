"""quick sanity check for a trained SAE checkpoint."""

import argparse
from collections import Counter

import torch
from sparsify import SparseCoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from phloem.env import load_env
from phloem.utils.device import resolve_device

load_env()

MODEL_NAME = "google/gemma-4-E2B"

TEST_SENTENCES = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The mitochondria is the powerhouse of the cell.",
    "Yesterday I went to the grocery store and bought some apples and oranges.",
    "import torch\nmodel = AutoModelForCausalLM.from_pretrained('gpt2')",
    "The Federal Reserve announced a 25 basis point rate cut on Wednesday.",
]


def eval_sae(
    checkpoint: str,
    hookpoint: str = "language_model.layers.17",
    device: str = "auto",
) -> None:
    resolved_device = resolve_device(device)

    print(f"loading model: {MODEL_NAME} on {resolved_device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": resolved_device},
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"loading SAE from {checkpoint}/{hookpoint}")
    sae = SparseCoder.load_from_disk(
        f"{checkpoint}/{hookpoint}", device=resolved_device
    )
    sae.eval()

    # find the hooked module
    target = dict(model.base_model.named_modules())[hookpoint]

    # run each test sentence and collect stats
    all_fvu = []
    all_l0 = []
    feature_counts: Counter[int] = Counter()

    for sentence in TEST_SENTENCES:
        tokens = tokenizer(sentence, return_tensors="pt").to(resolved_device)
        captured = []

        def hook_fn(module, input, output):
            captured.append(output.detach())

        handle = target.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**tokens)
        handle.remove()

        activations = captured[0]
        if activations.dim() == 3:
            activations = activations.squeeze(0)  # (seq_len, d_in)

        with torch.no_grad():
            out = sae(activations.float())

        fvu = out.fvu.item()
        l0 = out.latent_indices.shape[-1]
        all_fvu.append(fvu)
        all_l0.append(l0)

        for idx in out.latent_indices.flatten().tolist():
            feature_counts[idx] += 1

        # show per-sentence results
        token_strs = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        print(f"\n--- {sentence[:60]}...")
        print(f"    tokens: {len(token_strs)}, FVU: {fvu:.4f}, L0: {l0}")

        # show top-5 most active features for this sentence
        mean_acts = torch.zeros(sae.num_latents, device=resolved_device)
        mean_acts.scatter_add_(
            0,
            out.latent_indices.flatten(),
            out.latent_acts.flatten().float(),
        )
        mean_acts /= activations.shape[0]
        top5 = mean_acts.topk(5)
        top_feats = [
            f"#{idx}({val:.2f})" for idx, val in zip(top5.indices.tolist(), top5.values.tolist())
        ]
        print(f"    top features: {', '.join(top_feats)}")

    # summary
    n_features = sae.num_latents
    n_active = len(feature_counts)
    n_dead = n_features - n_active
    avg_fvu = sum(all_fvu) / len(all_fvu)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  avg FVU:        {avg_fvu:.4f} ({'good' if avg_fvu < 0.05 else 'ok' if avg_fvu < 0.1 else 'high'})")
    print(f"  L0:             {all_l0[0]} (should match k={all_l0[0]})")
    print(f"  total features: {n_features:,}")
    print(f"  active:         {n_active:,} ({100*n_active/n_features:.1f}%)")
    print(f"  dead:           {n_dead:,} ({100*n_dead/n_features:.1f}%)")
    print(f"  (dead count is a lower bound — only {len(TEST_SENTENCES)} sentences tested)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sanity-check a trained SAE")
    parser.add_argument(
        "checkpoint",
        help="checkpoint directory (e.g. models/gemma-4-e2b/checkpoints/my-run)",
    )
    parser.add_argument(
        "--hookpoint",
        default="language_model.layers.17",
        help="which hookpoint's SAE to evaluate (default: language_model.layers.17)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    eval_sae(
        checkpoint=args.checkpoint,
        hookpoint=args.hookpoint,
        device=args.device,
    )
