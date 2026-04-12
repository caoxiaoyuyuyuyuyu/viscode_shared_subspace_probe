#!/usr/bin/env python
"""Negative Control CKA Analysis: Compare Python-X vs Visual cross-format CKA."""
import json
import numpy as np
import torch
import time
from pathlib import Path

LAYERS = [4, 8, 12, 16, 20, 24, 28]
MODELS = ["coder", "viscoder2", "qwen25"]
VISUAL_FORMATS = ["svg", "tikz", "asy"]
CACHE_DIR = Path("/root/autodl-tmp/cache/hidden_states")
OUT_DIR = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/negative_control")
N_TRIPLES = 252
N_PERM = 5000
SEED = 42

def load_format(model, fmt, n=N_TRIPLES):
    """Load hidden states: returns (n, 7, 3584) float32."""
    tensors = []
    d = CACHE_DIR / model / fmt
    for i in range(n):
        t = torch.load(d / f"{i}.pt", map_location="cpu", weights_only=True)
        tensors.append(t.float().numpy())
    return np.stack(tensors, axis=0)

def center_gram(K):
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    return H @ K @ H

def cka(X, Y):
    """Linear CKA between X(n,d) and Y(n,d)."""
    KX = X @ X.T
    KY = Y @ Y.T
    KX_c = center_gram(KX)
    KY_c = center_gram(KY)
    hsic_xy = np.sum(KX_c * KY_c)
    hsic_xx = np.sum(KX_c * KX_c)
    hsic_yy = np.sum(KY_c * KY_c)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0

def permutation_test(X, Y, n_perm=N_PERM, seed=SEED):
    """Permutation null: shuffle Y sample indices, recompute CKA."""
    observed = cka(X, Y)
    rng = np.random.RandomState(seed)
    null_dist = []
    for _ in range(n_perm):
        perm = rng.permutation(Y.shape[0])
        null_dist.append(cka(X, Y[perm]))
    null_dist = np.array(null_dist)
    p_value = float(np.mean(null_dist >= observed))
    return observed, p_value, float(null_dist.mean()), float(null_dist.std())

def bootstrap_ci(X, Y, n_boot=1000, seed=SEED):
    """Bootstrap 95% CI for CKA."""
    rng = np.random.RandomState(seed + 1)
    n = X.shape[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boots.append(cka(X[idx], Y[idx]))
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load all data
    print("Loading hidden states...")
    data = {}
    for model in MODELS:
        data[model] = {}
        for fmt in VISUAL_FORMATS + ["python"]:
            data[model][fmt] = load_format(model, fmt)
            print(f"  {model}/{fmt}: {data[model][fmt].shape}")

    # Define pairs
    python_pairs = [("python", "svg"), ("python", "tikz"), ("python", "asy")]
    visual_pairs = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]

    results = {"python_x": [], "visual_x": [], "summary": {}}

    # Compute CKA for all pairs at all layers
    for pair_type, pairs, result_key in [
        ("python-X", python_pairs, "python_x"),
        ("visual-X", visual_pairs, "visual_x"),
    ]:
        for model in MODELS:
            for li, layer in enumerate(LAYERS):
                for f1, f2 in pairs:
                    X = data[model][f1][:, li, :]
                    Y = data[model][f2][:, li, :]

                    obs, p_val, null_mean, null_std = permutation_test(X, Y)
                    ci_lo, ci_hi = bootstrap_ci(X, Y)

                    row = {
                        "model": model, "layer": layer,
                        "format_pair": f"{f1}-{f2}",
                        "cka": round(obs, 4),
                        "p_value": p_val,
                        "null_mean": round(null_mean, 4),
                        "null_std": round(null_std, 4),
                        "ci_95_lo": round(ci_lo, 4),
                        "ci_95_hi": round(ci_hi, 4),
                    }
                    results[result_key].append(row)
                    print(f"  [{pair_type}] {model} L{layer} {f1}-{f2}: CKA={obs:.4f} p={p_val:.4f} null={null_mean:.4f}+-{null_std:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}]")

    # Summary: mean CKA at L28 for python-X vs visual-X
    for model in MODELS:
        py_l28 = [r["cka"] for r in results["python_x"] if r["model"] == model and r["layer"] == 28]
        vis_l28 = [r["cka"] for r in results["visual_x"] if r["model"] == model and r["layer"] == 28]
        results["summary"][model] = {
            "python_x_mean_L28": round(np.mean(py_l28), 4),
            "visual_x_mean_L28": round(np.mean(vis_l28), 4),
            "ratio": round(np.mean(py_l28) / np.mean(vis_l28), 4) if np.mean(vis_l28) > 0 else None,
        }
        print(f"\n[SUMMARY] {model}: Python-X mean CKA@L28={np.mean(py_l28):.4f}, Visual-X mean CKA@L28={np.mean(vis_l28):.4f}, ratio={np.mean(py_l28)/np.mean(vis_l28):.2f}")

    elapsed = time.time() - t0
    results["elapsed_s"] = round(elapsed, 1)
    results["n_perm"] = N_PERM
    results["n_triples"] = N_TRIPLES

    out_path = OUT_DIR / "negative_control_cka.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path} ({elapsed:.1f}s)")

if __name__ == "__main__":
    main()
