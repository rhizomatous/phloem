"""SAE diagnostics: decoder similarity clustering and feature quality metrics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sparsify import SparseCoder

from phloem.env import load_env
from phloem.utils.device import resolve_device

load_env()


def decoder_cosine_similarity(
    checkpoint: str,
    hookpoint: str,
    device: str = "cpu",
    top_n_pairs: int = 20,
    cluster_threshold: float = 0.9,
) -> dict:
    """compute pairwise cosine similarity on the decoder weight matrix.

    finds duplicate feature families (tight clusters of near-identical decoder
    vectors) and reports the most similar pairs.
    """
    sae_path = f"{checkpoint}/{hookpoint}"
    sae = SparseCoder.load_from_disk(sae_path, device=device)

    # W_dec: (d_sae, d_in) — each row is a feature's decoder direction
    W_dec = sae.W_dec.float()  # (d_sae, d_in)
    W_dec_norm = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # pairwise cosine similarity. this is a (d_sae, d_sae) matrix.
    # at 49K features that's ~9.6 GB in float32. compute in chunks to stay safe.
    n = W_dec_norm.shape[0]
    chunk_size = 1024

    top_sims = []  # (similarity, feat_i, feat_j)

    print(f"  computing pairwise cosine similarity ({n} features)...")
    for i in range(0, n, chunk_size):
        chunk = W_dec_norm[i : i + chunk_size]  # (chunk, d_in)
        sims = chunk @ W_dec_norm.T  # (chunk, n)

        # zero out self-similarity and lower triangle to avoid duplicates
        for local_idx in range(chunk.shape[0]):
            global_idx = i + local_idx
            sims[local_idx, : global_idx + 1] = 0

        # find high-similarity pairs in this chunk
        high_mask = sims > cluster_threshold
        if high_mask.any():
            rows, cols = high_mask.nonzero(as_tuple=True)
            for r, c in zip(rows.tolist(), cols.tolist()):
                top_sims.append((sims[r, c].item(), i + r, c))

    # sort by similarity descending
    top_sims.sort(key=lambda x: -x[0])

    # build clusters via simple union-find on high-similarity pairs
    parent: dict[int, int] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for sim, fi, fj in top_sims:
        union(fi, fj)

    # group features by cluster
    clusters: dict[int, list[int]] = {}
    for sim, fi, fj in top_sims:
        root = find(fi)
        if root not in clusters:
            clusters[root] = set()
        clusters[root].add(fi)
        clusters[root].add(fj)
    clusters = {k: sorted(v) for k, v in clusters.items()}

    # sort clusters by size
    clusters = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))

    return {
        "top_pairs": top_sims[:top_n_pairs],
        "clusters": clusters,
        "n_clustered_features": sum(len(c) for c in clusters.values()),
        "n_clusters": len(clusters),
    }


def token_trigger_concentration(
    parquet_path: str,
    top_n_features: int = 50,
) -> list[dict]:
    """for each feature, measure how concentrated its activations are on
    specific tokens. features that fire on the same 1-2 tokens are likely
    lexical artifacts; features that fire on diverse tokens are more likely
    semantic.
    """
    df = pq.read_table(parquet_path).to_pandas()

    results = []
    for feat_idx in df.feature.unique():
        rows = df[df.feature == feat_idx]
        tokens = rows["token"].tolist()
        n = len(tokens)
        if n == 0:
            continue

        counts = Counter(tokens)
        most_common_token, most_common_count = counts.most_common(1)[0]
        concentration = most_common_count / n
        n_unique = len(counts)

        # entropy over token distribution
        probs = np.array([c / n for c in counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        results.append({
            "feature": feat_idx,
            "n_examples": n,
            "n_unique_tokens": n_unique,
            "concentration": concentration,
            "top_token": most_common_token,
            "top_token_frac": concentration,
            "entropy": entropy,
            "peak_activation": rows["activation"].max(),
        })

    results.sort(key=lambda x: -x["concentration"])
    return results


def print_decoder_report(result: dict, hookpoint: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"DECODER SIMILARITY: {hookpoint}")
    print(f"{'=' * 60}")
    print(f"  features in clusters (cosine > threshold): {result['n_clustered_features']}")
    print(f"  number of clusters: {result['n_clusters']}")

    if result["clusters"]:
        print(f"\n  largest clusters:")
        for root, members in list(result["clusters"].items())[:10]:
            print(f"    cluster ({len(members)} features): {members[:10]}{'...' if len(members) > 10 else ''}")

    if result["top_pairs"]:
        print(f"\n  most similar pairs:")
        for sim, fi, fj in result["top_pairs"][:10]:
            print(f"    features {fi} & {fj}: cosine = {sim:.4f}")


def print_concentration_report(results: list[dict], hookpoint: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"TOKEN-TRIGGER CONCENTRATION: {hookpoint}")
    print(f"{'=' * 60}")

    # most concentrated (likely lexical artifacts)
    print(f"\n  most concentrated (likely lexical/artifact):")
    for r in results[:15]:
        print(
            f"    feature {r['feature']:>6d}: "
            f"{r['concentration']:.0%} on '{r['top_token']}', "
            f"{r['n_unique_tokens']} unique tokens, "
            f"entropy={r['entropy']:.2f}"
        )

    # least concentrated (likely semantic)
    print(f"\n  least concentrated (likely semantic):")
    sorted_by_entropy = sorted(results, key=lambda x: -x["entropy"])
    for r in sorted_by_entropy[:15]:
        print(
            f"    feature {r['feature']:>6d}: "
            f"{r['concentration']:.0%} on '{r['top_token']}', "
            f"{r['n_unique_tokens']} unique tokens, "
            f"entropy={r['entropy']:.2f}"
        )

    # summary stats
    concentrations = [r["concentration"] for r in results]
    high_conc = sum(1 for c in concentrations if c > 0.8)
    med_conc = sum(1 for c in concentrations if 0.3 < c <= 0.8)
    low_conc = sum(1 for c in concentrations if c <= 0.3)
    print(f"\n  distribution:")
    print(f"    >80% concentrated (lexical):  {high_conc} ({100*high_conc/len(results):.1f}%)")
    print(f"    30-80% concentrated (mixed):  {med_conc} ({100*med_conc/len(results):.1f}%)")
    print(f"    <30% concentrated (semantic): {low_conc} ({100*low_conc/len(results):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAE feature diagnostics")
    parser.add_argument(
        "checkpoint",
        help="checkpoint directory (e.g. models/gemma-4-e2b/checkpoints/all-layers-32x-1B)",
    )
    parser.add_argument(
        "--hookpoint",
        default="language_model.layers.17",
        help="hookpoint to analyze, or 'all' for all 35 layers (default: language_model.layers.17)",
    )
    parser.add_argument(
        "--parquet-dir",
        default=None,
        help="max_activations parquet dir (default: <checkpoint>/max_activations/)",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.9,
        help="cosine similarity threshold for clustering (default: 0.9)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    parquet_dir = args.parquet_dir or f"{args.checkpoint}/max_activations"

    if args.hookpoint == "all":
        hookpoints = [f"language_model.layers.{i}" for i in range(35)]
    else:
        hookpoints = [args.hookpoint]

    all_summaries = []

    for hookpoint in hookpoints:
        parquet_file = f"{parquet_dir}/{hookpoint.replace('.', '_')}.parquet"

        print(f"\n{'#' * 60}")
        print(f"# {hookpoint}")
        print(f"{'#' * 60}")

        # decoder similarity
        print("analyzing decoder similarity...")
        sim_result = decoder_cosine_similarity(
            checkpoint=args.checkpoint,
            hookpoint=hookpoint,
            device=args.device,
            cluster_threshold=args.cluster_threshold,
        )
        print_decoder_report(sim_result, hookpoint)

        # token-trigger concentration
        conc_results = []
        if Path(parquet_file).exists():
            print("\nanalyzing token-trigger concentration...")
            conc_results = token_trigger_concentration(parquet_file)
            print_concentration_report(conc_results, hookpoint)
        else:
            print(f"\n  skipping token-trigger analysis: {parquet_file} not found")

        # collect summary for cross-layer comparison
        if conc_results:
            concentrations = [r["concentration"] for r in conc_results]
            all_summaries.append({
                "hookpoint": hookpoint,
                "n_clusters": sim_result["n_clusters"],
                "n_clustered_features": sim_result["n_clustered_features"],
                "n_active_features": len(conc_results),
                "lexical_pct": 100 * sum(1 for c in concentrations if c > 0.8) / len(conc_results),
                "semantic_pct": 100 * sum(1 for c in concentrations if c <= 0.3) / len(conc_results),
                "mean_entropy": np.mean([r["entropy"] for r in conc_results]),
            })

    # cross-layer summary
    if len(all_summaries) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-LAYER SUMMARY")
        print(f"{'=' * 80}")
        print(f"  {'layer':<30s} {'active':>7s} {'clusters':>9s} {'clustered':>10s} {'lexical':>8s} {'semantic':>9s} {'entropy':>8s}")
        for s in all_summaries:
            layer = s["hookpoint"].split(".")[-1]
            print(
                f"  layer {layer:>2s}                        "
                f"{s['n_active_features']:>7d} "
                f"{s['n_clusters']:>9d} "
                f"{s['n_clustered_features']:>10d} "
                f"{s['lexical_pct']:>7.1f}% "
                f"{s['semantic_pct']:>8.1f}% "
                f"{s['mean_entropy']:>8.2f}"
            )
