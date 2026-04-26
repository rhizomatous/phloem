"""browse max-activating examples for SAE features."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


# ANSI color ramp: dim → bright for activation intensity
_RESET = "\033[0m"
_COLORS = [
    "\033[38;5;240m",  # dim gray (near zero)
    "\033[38;5;223m",  # light tan
    "\033[38;5;216m",  # salmon
    "\033[38;5;209m",  # orange
    "\033[38;5;196m",  # bright red (max)
]


def _colorize(token: str, activation: float, max_act: float) -> str:
    """color a token by its activation intensity relative to the example's max."""
    if max_act <= 0:
        return f"{_COLORS[0]}{token}{_RESET}"
    level = min(len(_COLORS) - 1, int((activation / max_act) * (len(_COLORS) - 1)))
    return f"{_COLORS[level]}{token}{_RESET}"


def browse(
    parquet_path: str,
    feature: int | None = None,
    top_features: int = 10,
    examples_per_feature: int = 5,
    color: bool = True,
) -> None:
    df = pq.read_table(parquet_path).to_pandas()
    layer = Path(parquet_path).stem.replace("_", ".")

    has_token_acts = "token_activations" in df.columns

    if feature is not None:
        rows = df[df.feature == feature].sort_values("activation", ascending=False)
        if rows.empty:
            print(f"feature {feature} not found in {parquet_path}")
            return
        print(f"=== {layer} feature {feature} ({len(rows)} examples) ===\n")
        for _, r in rows.iterrows():
            _print_example(r, color and has_token_acts)
        return

    # show the top features by peak activation
    peak = df.groupby("feature")["activation"].max().sort_values(ascending=False)
    print(f"{layer}: {df.feature.nunique()} features with data\n")
    print(f"top {top_features} features by peak activation:\n")

    for feat in peak.index[:top_features]:
        rows = df[df.feature == feat].sort_values("activation", ascending=False)
        print(f"--- feature {feat} (peak {peak[feat]:.2f}) ---")
        for _, r in rows.head(examples_per_feature).iterrows():
            _print_example(r, color and has_token_acts)
        print()


def _print_example(row, use_color: bool) -> None:
    """print a single example, optionally with per-token activation coloring."""
    if use_color:
        try:
            token_acts = json.loads(row.token_activations)
            context_tokens = json.loads(row.context_tokens)
            if len(token_acts) == len(context_tokens):
                max_act = max(token_acts) if token_acts else 0
                colored = "".join(
                    _colorize(tok, act, max_act)
                    for tok, act in zip(context_tokens, token_acts)
                )
                print(f"  {row.activation:6.2f}  {colored}")
                return
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    # fallback: plain context with token highlighted
    print(f"  {row.activation:6.2f}  [{row.token}]  {row.context}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="browse max-activating examples")
    parser.add_argument(
        "parquet",
        help="path to a parquet file (e.g. checkpoints/my-run/max_activations/language_model_layers_17.parquet)",
    )
    parser.add_argument(
        "--feature",
        type=int,
        default=None,
        help="show all examples for a specific feature index",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of top features to show (default: 10)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="examples per feature (default: 5)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable per-token activation coloring",
    )
    args = parser.parse_args()

    browse(
        parquet_path=args.parquet,
        feature=args.feature,
        top_features=args.top,
        examples_per_feature=args.examples,
        color=not args.no_color,
    )
