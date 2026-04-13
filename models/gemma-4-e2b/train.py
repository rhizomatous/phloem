"""train a sparse autoencoder on Gemma 4 E2B with streaming activations via sparsify."""

import argparse

import torch
from datasets import Dataset
from sparsify import SparseCoderConfig, Trainer, TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from phloem.data_loader import stream_token_batches
from phloem.env import load_env
from phloem.utils.device import resolve_device

load_env()

MODEL_NAME = "google/gemma-4-E2B"
HOOK_LAYER = 17
HOOKPOINT = f"language_model.layers.{HOOK_LAYER}"


def build_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    seq_len: int,
    seed: int,
    stream_batch_size: int = 8,
) -> Dataset:
    """materialize streamed RedPajama batches into an in-memory HF Dataset."""
    rows: list[list[int]] = []
    for batch in stream_token_batches(
        tokenizer,
        batch_size=stream_batch_size,
        seq_len=seq_len,
        max_tokens=max_tokens,
    ):
        rows.extend(batch.tolist())
    # sparsify's `Trainer`` does not shuffle internally, and its docstring says the
    # dataset should be shuffled before being passed in. consecutive batches
    # from `stream_token_batches`` would otherwise be within-document-correlated.
    return Dataset.from_dict({"input_ids": rows}).shuffle(seed=seed).with_format("torch")


def train_sae(
    max_tokens: int,
    seq_len: int = 128,
    expansion_factor: int = 8,
    k: int = 100,
    batch_size: int = 16,
    lr: float | None = None,
    device: str = "auto",
    log_to_wandb: bool = False,
    save_dir: str = "models/gemma-4-e2b/checkpoints",
    run_name: str = "gemma-4-e2b-layer17-topk",
    seed: int = 42,
) -> None:
    resolved_device = resolve_device(device)

    print(f"loading model: {MODEL_NAME} on {resolved_device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": resolved_device},
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"streaming {max_tokens:,} tokens from RedPajama v2 (seq_len={seq_len})")
    dataset = build_dataset(
        tokenizer, max_tokens=max_tokens, seq_len=seq_len, seed=seed
    )
    print(f"built dataset: {len(dataset)} sequences × {seq_len} tokens")

    sae_cfg = SparseCoderConfig(
        activation="topk",
        k=k,
        expansion_factor=expansion_factor,
    )
    cfg = TrainConfig(
        sae=sae_cfg,
        batch_size=batch_size,
        hookpoints=[HOOKPOINT],
        lr=lr,
        log_to_wandb=log_to_wandb,
        save_dir=save_dir,
        run_name=run_name,
    )

    trainer = Trainer(cfg, dataset, model)
    trainer.fit()
    print(
        f"training complete. k={k}, expansion={expansion_factor}x. "
        f"checkpoints in {save_dir}/{run_name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train SAE on Gemma 4 E2B via sparsify")
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="tokens to train on (e.g. 10000 for a smoke test, 10000000 for a real run)",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true", dest="log_to_wandb")
    parser.add_argument("--save-dir", default="models/gemma-4-e2b/checkpoints")
    parser.add_argument("--run-name", default="gemma-4-e2b-topk")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_sae(
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
        expansion_factor=args.expansion_factor,
        k=args.k,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        log_to_wandb=args.log_to_wandb,
        save_dir=args.save_dir,
        run_name=args.run_name,
        seed=args.seed,
    )
