"""browse max-activating examples for SAE features."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def browse(
    parquet_path: str,
    feature: int | None = None,
    top_features: int = 10,
    examples_per_feature: int = 5,
) -> None:
    df = pq.read_table(parquet_path).to_pandas()
    layer = Path(parquet_path).stem.replace("_", ".")

    if feature is not None:
        rows = df[df.feature == feature].sort_values("activation", ascending=False)
        if rows.empty:
            print(f"feature {feature} not found in {parquet_path}")
            return
        print(f"=== {layer} feature {feature} ({len(rows)} examples) ===\n")
        for _, r in rows.iterrows():
            print(f"  {r.activation:6.2f}  [{r.token}]  {r.context}")
        return

    # show the top features by peak activation
    peak = df.groupby("feature")["activation"].max().sort_values(ascending=False)
    print(f"{layer}: {df.feature.nunique()} features with data\n")
    print(f"top {top_features} features by peak activation:\n")

    for feat in peak.index[:top_features]:
        rows = df[df.feature == feat].sort_values("activation", ascending=False)
        print(f"--- feature {feat} (peak {peak[feat]:.2f}) ---")
        for _, r in rows.head(examples_per_feature).iterrows():
            print(f"  {r.activation:6.2f}  [{r.token}]  {r.context}")
        print()


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
    args = parser.parse_args()

    browse(
        parquet_path=args.parquet,
        feature=args.feature,
        top_features=args.top,
        examples_per_feature=args.examples,
    )
