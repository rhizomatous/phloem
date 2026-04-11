"""extract activations from Gemma 4 E2B and cache in SAELens-compatible format."""

import argparse
from pathlib import Path

import torch
from datasets import Array2D, Dataset, Features, Sequence, Value
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from phloem.data_loader import stream_token_batches
from phloem.env import load_env
from phloem.utils.device import resolve_device

load_env()

MODEL_NAME = "google/gemma-4-E2B"
HOOK_LAYER = 17  # mid-network (35 layers total)
D_IN = 1536  # hidden dim

# use TransformerLens-style hook name as dataset column key.
# SAELens uses this as a string key, and SAEDashboard/Neuronpedia
# hard-validate against TL naming — so we use it from the start
# even though we extract via raw HF hooks.
HOOK_NAME = f"blocks.{HOOK_LAYER}.hook_resid_post"


def extract_activations(
    output_dir: str = "models/gemma-4-e2b/activations",
    hook_layer: int = HOOK_LAYER,
    max_tokens: int = 100_000,
    batch_size: int = 4,
    seq_len: int = 128,
    dtype: str = "float32",
    device: str = "auto",
    max_docs: int | None = None,
) -> None:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"{output_path} is not empty. delete it or choose a different path."
        )
    output_path.mkdir(parents=True, exist_ok=True)

    storage_dtype = torch.float32 if dtype == "float32" else torch.float16
    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    resolved_device = resolve_device(device)

    print(f"loading model: {MODEL_NAME} on {resolved_device}")
    load_kwargs: dict = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    # on CUDA, "auto" lets transformers shard across multiple GPUs if available
    # (and harmlessly loads to GPU 0 on single-GPU setups). MPS/CPU just take the
    # device name directly since they don't support sharding.
    if resolved_device == "cuda":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = resolved_device

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"hooking layer {hook_layer} (hidden dim {D_IN})")

    # collect activations via forward hook, then short-circuit the forward pass.
    # running the full model through the LM head would waste VRAM on a huge
    # logits tensor (seq_len * vocab_size) that we don't need.
    captured: list[torch.Tensor] = []

    class _StopForward(Exception):
        pass

    def hook_fn(module, input, output):
        # Gemma 4 decoder layers return the hidden state tensor directly
        hidden = output.detach().cpu().to(storage_dtype)
        if hidden.dim() == 2:
            # (seq_len, d_in) -> (1, seq_len, d_in) when batch dim is missing
            hidden = hidden.unsqueeze(0)
        captured.append(hidden)
        raise _StopForward()

    handle = model.model.language_model.layers[hook_layer].register_forward_hook(hook_fn)

    all_activations: list[torch.Tensor] = []
    all_token_ids: list[torch.Tensor] = []
    tokens_collected = 0

    print(f"extracting activations (target: {max_tokens:,} tokens)...")
    try:
        for batch_tokens in tqdm(
            stream_token_batches(
                tokenizer,
                batch_size=batch_size,
                seq_len=seq_len,
                max_tokens=max_tokens,
                max_docs=max_docs,
            ),
            desc="batches",
        ):
            captured.clear()

            input_device = next(model.parameters()).device
            try:
                with torch.no_grad():
                    model(batch_tokens.to(input_device))
            except _StopForward:
                pass

            # captured[0] shape: (batch_size, seq_len, d_in)
            acts = captured[0]
            all_activations.append(acts)
            all_token_ids.append(batch_tokens)
            tokens_collected += batch_tokens.numel()

            if tokens_collected >= max_tokens:
                break
    finally:
        handle.remove()

    print(f"collected {tokens_collected:,} tokens, saving to {output_path}...")

    # stack into (n_sequences, seq_len, d_in) and (n_sequences, seq_len)
    acts_tensor = torch.cat(all_activations, dim=0)
    ids_tensor = torch.cat(all_token_ids, dim=0).to(torch.int32)

    n_seq, context_size, d_in = acts_tensor.shape
    print(f"  activations shape: {acts_tensor.shape}")
    print(f"  token IDs shape: {ids_tensor.shape}")

    # save as HuggingFace Dataset (SAELens-compatible format)
    features = Features(
        {
            hook_name: Array2D(shape=(context_size, d_in), dtype=dtype),
            "token_ids": Sequence(Value(dtype="int32"), length=context_size),
        }
    )

    dataset = Dataset.from_dict(
        {
            hook_name: acts_tensor,
            "token_ids": ids_tensor,
        },
        features=features,
    )

    dataset.save_to_disk(str(output_path))
    print(f"saved {n_seq} sequences to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract Gemma 4 E2B activations")
    parser.add_argument("--output-dir", default="models/gemma-4-e2b/activations")
    parser.add_argument("--hook-layer", type=int, default=HOOK_LAYER)
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    extract_activations(
        output_dir=args.output_dir,
        hook_layer=args.hook_layer,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=args.dtype,
        device=args.device,
        max_docs=args.max_docs,
    )
