"""train a sparse autoencoder on cached DeepSeek-V2-Lite activations."""

import argparse

from sae_lens import (
    BatchTopKTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
)

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"
HOOK_LAYER = 13
D_IN = 2048


def train_sae(
    cached_activations_path: str = "models/deepseek-v2-lite/activations",
    hook_layer: int = HOOK_LAYER,
    d_in: int = D_IN,
    expansion_factor: int = 8,
    k: int = 100,
    training_tokens: int = 10_000_000,
    train_batch_size_tokens: int = 4096,
    context_size: int = 128,
    lr: float = 3e-4,
    device: str = "cuda",
    log_to_wandb: bool = False,
) -> None:
    hook_name = f"model.layers.{hook_layer}"
    d_sae = d_in * expansion_factor

    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
    )

    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=MODEL_NAME,
        hook_name=hook_name,
        context_size=context_size,
        use_cached_activations=True,
        cached_activations_path=cached_activations_path,
        training_tokens=training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        lr=lr,
        device=device,
        model_from_pretrained_kwargs={
            "trust_remote_code": True,
        },
    )
    cfg.logger.log_to_wandb = log_to_wandb

    runner = LanguageModelSAETrainingRunner(cfg)
    sae = runner.run()
    print(f"training complete. SAE d_in={d_in}, d_sae={d_sae}, k={k}")
    return sae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train SAE on DeepSeek-V2-Lite activations")
    parser.add_argument("--cached-activations-path", default="models/deepseek-v2-lite/activations")
    parser.add_argument("--hook-layer", type=int, default=HOOK_LAYER)
    parser.add_argument("--d-in", type=int, default=D_IN)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--training-tokens", type=int, default=10_000_000)
    parser.add_argument("--train-batch-size-tokens", type=int, default=4096)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true", dest="log_to_wandb")
    args = parser.parse_args()

    train_sae(
        cached_activations_path=args.cached_activations_path,
        hook_layer=args.hook_layer,
        d_in=args.d_in,
        expansion_factor=args.expansion_factor,
        k=args.k,
        training_tokens=args.training_tokens,
        train_batch_size_tokens=args.train_batch_size_tokens,
        context_size=args.context_size,
        lr=args.lr,
        device=args.device,
        log_to_wandb=args.log_to_wandb,
    )
