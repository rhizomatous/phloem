"""train a sparse autoencoder on Gemma 4 E2B with streaming activations via sparsify."""

import argparse
import tempfile
from pathlib import Path
import numpy as np
import torch
from sparsify import SparseCoderConfig, Trainer, TrainConfig
from sparsify.data import MemmapDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from phloem.data_loader import stream_token_batches
from phloem.env import load_env
from phloem.utils.device import resolve_device

load_env()

MODEL_NAME = "google/gemma-4-E2B"
NUM_LAYERS = 35


def make_hookpoints(layers: list[int]) -> list[str]:
    return [f"language_model.layers.{i}" for i in layers]


def build_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    seq_len: int,
    seed: int,
    data_dir: str | None = None,
    stream_batch_size: int = 8,
) -> MemmapDataset:
    """stream RedPajama tokens into a shuffled numpy memmap on disk.

    if a tokens.bin file already exists at data_dir, reuses it instead of
    re-streaming. collects batches as numpy arrays (4 bytes/token as uint32),
    then writes a flat binary file that sparsify's MemmapDataset reads lazily.
    """
    if data_dir is None:
        data_dir = tempfile.mkdtemp(prefix="phloem-")
    data_path = Path(data_dir) / "tokens.bin"

    if data_path.exists():
        print(f"reusing existing tokenized data at {data_path}")
        return MemmapDataset(str(data_path), ctx_len=seq_len, dtype=np.uint32)

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    chunks: list[np.ndarray] = []
    for batch in stream_token_batches(
        tokenizer,
        batch_size=stream_batch_size,
        seq_len=seq_len,
        max_tokens=max_tokens,
    ):
        chunks.append(batch.numpy().astype(np.uint32))

    all_tokens = np.concatenate(chunks, axis=0)  # (n_rows, seq_len)

    # sparsify's Trainer does not shuffle internally — shuffle before writing.
    rng = np.random.default_rng(seed)
    rng.shuffle(all_tokens)

    # write to flat binary. MemmapDataset reshapes to (-1, ctx_len) on load.
    mmap = np.memmap(data_path, dtype=np.uint32, mode="w+", shape=all_tokens.shape)
    mmap[:] = all_tokens
    mmap.flush()
    del mmap, all_tokens

    return MemmapDataset(str(data_path), ctx_len=seq_len, dtype=np.uint32)


def train_sae(
    max_tokens: int,
    layers: list[int] | None = None,
    seq_len: int = 128,
    expansion_factor: int = 8,
    k: int = 100,
    batch_size: int = 16,
    grad_acc_steps: int = 1,
    lr: float | None = None,
    auxk_alpha: float = 0.0,
    exclude_bos: bool = False,
    device: str = "auto",
    log_to_wandb: bool = False,
    save_dir: str = "models/gemma-4-e2b/checkpoints",
    save_every: int = 100,
    run_name: str = "gemma-4-e2b-topk",
    seed: int = 42,
    data_dir: str | None = None,
    resume: str | None = None,
) -> None:
    if layers is None:
        layers = [17]
    hookpoints = make_hookpoints(layers)
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
        tokenizer,
        max_tokens=max_tokens,
        seq_len=seq_len,
        seed=seed,
        data_dir=data_dir,
    )
    print(f"built dataset: {len(dataset)} sequences × {seq_len} tokens")

    sae_cfg = SparseCoderConfig(
        activation="topk",
        k=k,
        expansion_factor=expansion_factor,
        multi_topk=True,
    )
    exclude_tokens = [2] if exclude_bos else []  # token ID 2 = <bos>

    cfg = TrainConfig(
        sae=sae_cfg,
        batch_size=batch_size,
        grad_acc_steps=grad_acc_steps,
        hookpoints=hookpoints,
        lr=lr,
        auxk_alpha=auxk_alpha,
        exclude_tokens=exclude_tokens,
        log_to_wandb=log_to_wandb,
        save_dir=save_dir,
        save_every=save_every,
        run_name=run_name,
        finetune=resume,
    )

    trainer = Trainer(cfg, dataset, model)
    trainer.fit()
    n_saes = len(trainer.saes)
    print(
        f"training complete. {n_saes} SAE(s), k={k}, expansion={expansion_factor}x. "
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
    parser.add_argument(
        "--layers",
        default="17",
        help="comma-separated layer indices, or 'all' for all 35 layers (default: 17)",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-acc-steps", type=int, default=1, help="gradient accumulation steps (effective batch = batch-size × grad-acc-steps)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--auxk-alpha", type=float, default=0.0, help="weight for AuxK dead-feature loss (e.g. 1/32)")
    parser.add_argument("--exclude-bos", action="store_true", help="exclude BOS token (id=2) from training")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true", dest="log_to_wandb")
    parser.add_argument("--save-dir", default="models/gemma-4-e2b/checkpoints")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--run-name", default="gemma-4-e2b-topk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="directory for tokenized memmap file (default: temp dir, cleaned up by OS)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint directory to resume from (e.g. models/gemma-4-e2b/checkpoints/my-run)",
    )
    args = parser.parse_args()

    if args.layers == "all":
        layers = list(range(NUM_LAYERS))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    train_sae(
        max_tokens=args.max_tokens,
        layers=layers,
        seq_len=args.seq_len,
        expansion_factor=args.expansion_factor,
        k=args.k,
        batch_size=args.batch_size,
        grad_acc_steps=args.grad_acc_steps,
        lr=args.lr,
        auxk_alpha=args.auxk_alpha,
        exclude_bos=args.exclude_bos,
        device=args.device,
        log_to_wandb=args.log_to_wandb,
        save_dir=args.save_dir,
        save_every=args.save_every,
        run_name=args.run_name,
        seed=args.seed,
        data_dir=args.data_dir,
        resume=args.resume,
    )
