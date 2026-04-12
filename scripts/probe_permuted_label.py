#!/usr/bin/env python3
"""Permuted-label probe baseline for F1 format identity probe.

Tests whether 100% probe accuracy is an artifact of n≪d (756×3584)
by running the same LogisticRegression with shuffled format labels.

Usage:
    python probe_permuted_label.py \
        --cache_dir /root/autodl-tmp/cache/hidden_states \
        --models coder viscoder2 qwen25 \
        --layers 4 8 12 16 20 24 28 \
        --n_perms 100 \
        --output /root/autodl-tmp/probe_permuted_label_results.json
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from pathlib import Path


def load_hidden_states(cache_dir, model, fmt, layer_idx):
    """Load hidden states for a model/format at a specific layer index."""
    fmt_dir = Path(cache_dir) / model / fmt
    vectors = []
    for pt_file in sorted(fmt_dir.glob("*.pt")):
        tensor = torch.load(pt_file, map_location="cpu", weights_only=True)
        vectors.append(tensor[layer_idx].numpy())
    return np.stack(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/cache/hidden_states")
    parser.add_argument("--models", nargs="+", default=["coder", "viscoder2", "qwen25"])
    parser.add_argument("--formats", nargs="+", default=["svg", "tikz", "asy"])
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24, 28])
    parser.add_argument("--n_perms", type=int, default=100)
    parser.add_argument("--output", default="/root/autodl-tmp/probe_permuted_label_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    layer_indices = list(range(len(args.layers)))

    results = {
        "models": args.models,
        "formats": args.formats,
        "layers": args.layers,
        "n_perms": args.n_perms,
        "seed": args.seed,
        "per_model": {}
    }

    for model in args.models:
        print(f"\n=== Model: {model} ===", flush=True)
        model_results = {}

        for li, layer in zip(layer_indices, args.layers):
            print(f"  [Layer {layer} (index {li})]", flush=True)

            # Load data for all formats
            X_parts = []
            y_parts = []
            for fmt_idx, fmt in enumerate(args.formats):
                X_fmt = load_hidden_states(args.cache_dir, model, fmt, li)
                X_parts.append(X_fmt)
                y_parts.append(np.full(len(X_fmt), fmt_idx))

            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            n_samples, n_features = X.shape
            print(f"    n={n_samples}, d={n_features}, n/d={n_samples/n_features:.3f}", flush=True)

            # Real labels: 5-fold CV
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                     multi_class="multinomial")
            real_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            real_mean = float(np.mean(real_scores))
            real_std = float(np.std(real_scores))
            print(f"    Real: {real_mean:.4f} ± {real_std:.4f}", flush=True)

            # Permuted labels: 100 iterations
            perm_means = []
            for i in range(args.n_perms):
                y_perm = rng.permutation(y)
                perm_scores = cross_val_score(clf, X, y_perm, cv=5, scoring="accuracy")
                perm_means.append(float(np.mean(perm_scores)))
                if (i + 1) % 25 == 0:
                    print(f"    Perm {i+1}/{args.n_perms}: running mean={np.mean(perm_means):.4f}", flush=True)

            perm_arr = np.array(perm_means)
            model_results[f"L{layer}"] = {
                "real_accuracy": real_mean,
                "real_std": real_std,
                "perm_mean": float(np.mean(perm_arr)),
                "perm_std": float(np.std(perm_arr)),
                "perm_max": float(np.max(perm_arr)),
                "perm_min": float(np.min(perm_arr)),
                "perm_95th": float(np.percentile(perm_arr, 95)),
                "n_samples": n_samples,
                "n_features": n_features,
                "real_above_all_perms": bool(real_mean > np.max(perm_arr))
            }
            print(f"    Permuted: {perm_arr.mean():.4f} ± {perm_arr.std():.4f} "
                  f"(max={perm_arr.max():.4f}, 95th={np.percentile(perm_arr, 95):.4f})", flush=True)

        results["per_model"][model] = model_results

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    for model in args.models:
        for layer_key, lr in results["per_model"][model].items():
            gap = lr["real_accuracy"] - lr["perm_mean"]
            print(f"  {model} {layer_key}: real={lr['real_accuracy']:.4f} "
                  f"perm_mean={lr['perm_mean']:.4f} gap={gap:.4f} "
                  f"above_all_perms={lr['real_above_all_perms']}", flush=True)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
