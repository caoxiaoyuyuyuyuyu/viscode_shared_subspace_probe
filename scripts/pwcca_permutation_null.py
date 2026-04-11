#!/usr/bin/env python
"""PWCCA permutation null baseline (D028 W1).

Computes null distribution for PWCCA by shuffling sample pairing
(same logic as A2 CKA permutation). 300 permutations per model.

Usage:
  # Server:
  cd /root/autodl-tmp/viscode_shared_subspace_probe
  python scripts/pwcca_permutation_null.py

  # Local smoke:
  python scripts/pwcca_permutation_null.py --smoke
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

LAYERS = [4, 8, 12, 16, 20, 24, 28]
MODELS = ["coder", "viscoder2", "qwen25"]
FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def pwcca_score(X, Y, k=50):
    """Simplified PWCCA (PCA variance ratio weights)."""
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA

    n, d = X.shape
    k_eff = min(k, n - 1, d)

    pca_x = PCA(n_components=k_eff, random_state=SEED)
    pca_y = PCA(n_components=k_eff, random_state=SEED)
    X_r = pca_x.fit_transform(X - X.mean(axis=0))
    Y_r = pca_y.fit_transform(Y - Y.mean(axis=0))

    n_comp = min(k_eff, n - 1)
    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(X_r, Y_r)
    X_cc, Y_cc = cca.transform(X_r, Y_r)

    corrs = np.array([np.corrcoef(X_cc[:, i], Y_cc[:, i])[0, 1]
                      for i in range(n_comp)])
    corrs = np.abs(np.nan_to_num(corrs, nan=0.0))

    var_x = pca_x.explained_variance_ratio_[:n_comp]
    weights = var_x / var_x.sum()
    return float(np.sum(weights * corrs))


def load_hidden_states(cache_dir, n_triples):
    """Load hidden states into data[model][format] = np.array(n, 7, 3584)."""
    import torch
    data = {}
    for model in MODELS:
        data[model] = {}
        for fmt in FORMATS:
            tensors = []
            d = cache_dir / model / fmt
            for i in range(n_triples):
                t = torch.load(d / f"{i}.pt", map_location="cpu", weights_only=True)
                tensors.append(t.float().numpy())
            data[model][fmt] = np.stack(tensors, axis=0)
            print(f"  {model}/{fmt}: {data[model][fmt].shape}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-perm", type=int, default=300)
    args = parser.parse_args()

    if args.smoke:
        n_triples = 12
        n_perm = 20
        cache_dir = PROJECT_ROOT / "tests" / "fake_hidden_states"
        out_path = PROJECT_ROOT / "artifacts" / "pwcca_perm_null_smoke.json"
    else:
        n_triples = 252
        n_perm = args.n_perm
        cache_dir = Path("/root/autodl-tmp/cache/hidden_states")
        out_path = PROJECT_ROOT / "artifacts" / "pwcca_perm_null.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"=== PWCCA Permutation Null (n_perm={n_perm}) ===")
    print(f"  n_triples={n_triples}, cache={cache_dir}")

    # Load data
    data = load_hidden_states(cache_dir, n_triples)
    n = n_triples
    rng = np.random.RandomState(SEED + 3)

    results = {}
    for model in MODELS:
        print(f"\n--- {model} ---")

        # Observed PWCCA (mean across layers × pairs)
        obs_vals = []
        obs_per_layer = {}
        for li, layer in enumerate(LAYERS):
            layer_vals = []
            for f1, f2 in FORMAT_PAIRS:
                X = data[model][f1][:, li, :]
                Y = data[model][f2][:, li, :]
                val = pwcca_score(X, Y)
                layer_vals.append(val)
                obs_vals.append(val)
            obs_per_layer[str(layer)] = round(float(np.mean(layer_vals)), 6)
        obs_mean = float(np.mean(obs_vals))
        print(f"  Observed mean PWCCA: {obs_mean:.4f}")

        # Null distribution: shuffle sample indices of Y
        null_means = np.empty(n_perm, dtype=np.float64)
        for p_idx in range(n_perm):
            perm = rng.permutation(n)
            perm_vals = []
            for li in range(len(LAYERS)):
                for f1, f2 in FORMAT_PAIRS:
                    X = data[model][f1][:, li, :]
                    Y = data[model][f2][perm, li, :]  # shuffled samples
                    perm_vals.append(pwcca_score(X, Y))
            null_means[p_idx] = np.mean(perm_vals)

            if (p_idx + 1) % 50 == 0:
                print(f"    {model}: {p_idx+1}/{n_perm} (null_mean_so_far={np.mean(null_means[:p_idx+1]):.4f})")

        p_value = float(np.mean(null_means >= obs_mean))
        results[model] = {
            "observed_mean": round(obs_mean, 6),
            "observed_per_layer": obs_per_layer,
            "null_mean": round(float(np.mean(null_means)), 6),
            "null_std": round(float(np.std(null_means)), 6),
            "null_95th": round(float(np.percentile(null_means, 95)), 6),
            "null_99th": round(float(np.percentile(null_means, 99)), 6),
            "p_value": round(p_value, 6),
            "n_permutations": n_perm,
            "ratio_obs_null": round(obs_mean / float(np.mean(null_means)), 3),
        }
        print(f"  Null: {np.mean(null_means):.4f} ± {np.std(null_means):.4f}")
        print(f"  Ratio obs/null: {obs_mean / np.mean(null_means):.3f}x")
        print(f"  p-value: {p_value:.6f}")

    elapsed = time.time() - t_total
    output = {
        "models": results,
        "config": {
            "n_triples": n_triples,
            "n_permutations": n_perm,
            "layers": LAYERS,
            "seed": SEED + 3,
        },
        "elapsed_s": round(elapsed, 1),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n=== Done in {elapsed:.1f}s. Saved to {out_path} ===")


if __name__ == "__main__":
    main()
