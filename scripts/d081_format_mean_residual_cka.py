#!/usr/bin/env python
"""D081 Task 1A: Format-mean residualized CKA + per-pair permutation p-values.

Subtracts per-format centroid from hidden states, then recomputes CKA
with per-pair per-layer permutation tests (for BH-FDR in Task 4).

Usage:
  # All 6 models:
  python scripts/d081_format_mean_residual_cka.py \
      --models coder viscoder2 qwen25 codestral starcoder2 deepseek \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --n-perm 1000

  # Smoke test:
  python scripts/d081_format_mean_residual_cka.py \
      --models coder --cache-dir /tmp/smoke_hs --smoke
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

np.seterr(over="raise", invalid="raise")
sys.stdout.reconfigure(line_buffering=True)

FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ── Helpers ──────────────────────────────────────────────────────────
def _save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**3
    except ImportError:
        return 0.0


def read_model_meta(cache_dir, model):
    for fmt in FORMATS:
        summary_path = cache_dir / model / fmt / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            layers = s.get("layers_resolved") or s.get("layers")
            hidden_dim = s["hidden_dim"]
            n_saved = s["n_saved"]
            return {"layers": layers, "hidden_dim": hidden_dim, "n_saved": n_saved}
    raise FileNotFoundError(f"No summary.json for {model} in {cache_dir}")


def load_hidden_states(cache_dir, model, layers, hidden_dim, n_triples):
    n_layers = len(layers)
    data = {}
    for fmt in FORMATS:
        tensors = []
        d = cache_dir / model / fmt
        for i in range(n_triples):
            pt_path = d / f"{i}.pt"
            t = torch.load(pt_path, map_location="cpu", weights_only=True)
            assert t.shape == (n_layers, hidden_dim), \
                f"{pt_path}: expected ({n_layers}, {hidden_dim}), got {t.shape}"
            tensors.append(t.float().numpy())
        arr = np.stack(tensors, axis=0)
        data[fmt] = arr
        print(f"  {model}/{fmt}: {arr.shape}")
    return data


# ── CKA Core ─────────────────────────────────────────────────────────
def _center_gram(K):
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    return H @ K @ H


def _cka_from_centered(KX_c, KY_c):
    KX64 = KX_c.astype(np.float64, copy=False)
    KY64 = KY_c.astype(np.float64, copy=False)
    hsic_xy = np.float64(np.sum(KX64 * KY64))
    hsic_xx = np.float64(np.sum(KX64 * KX64))
    hsic_yy = np.float64(np.sum(KY64 * KY64))
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ── Format-Mean Residualization ──────────────────────────────────────
def residualize_format_mean(data, layer_idx, n_triples):
    """Subtract per-format centroid: X_res[fmt] = X[fmt] - mean(X[fmt])."""
    res = {}
    for fmt in FORMATS:
        X = data[fmt][:n_triples, layer_idx, :]  # (n, hidden_dim)
        X_mean = X.mean(axis=0, keepdims=True)  # (1, hidden_dim)
        res[fmt] = X - X_mean
    return res


# ── Per-Pair Permutation Test ────────────────────────────────────────
def per_pair_permutation(res_data, n_triples, n_perm):
    """Per-pair permutation test on residualized hidden states.

    Returns dict: {pair: {observed_cka, null_mean, p_value}}.
    """
    n = n_triples
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    rng = np.random.RandomState(SEED + 7)

    # Compute Gram matrices on residualized data
    grams = {fmt: res_data[fmt] @ res_data[fmt].T for fmt in FORMATS}
    centered = {fmt: _center_gram(grams[fmt]) for fmt in FORMATS}

    # Pre-compute HSIC_xx
    hsic_xx = {}
    for fmt in FORMATS:
        hsic_xx[fmt] = float(np.sum(centered[fmt].astype(np.float64) ** 2))

    results = {}
    for f1, f2 in FORMAT_PAIRS:
        pair_name = f"{f1}-{f2}"
        obs_cka = _cka_from_centered(centered[f1], centered[f2])

        null_ckas = np.empty(n_perm, dtype=np.float32)
        for p in range(n_perm):
            perm = rng.permutation(n)
            KY_perm = grams[f2][np.ix_(perm, perm)]
            KY_c = H @ KY_perm @ H
            hsic_xy = np.float64(np.sum(centered[f1].astype(np.float64) * KY_c.astype(np.float64)))
            hsic_yy = np.float64(np.sum(KY_c.astype(np.float64) ** 2))
            denom = np.sqrt(hsic_xx[f1] * hsic_yy)
            null_ckas[p] = hsic_xy / denom if denom > 1e-12 else 0.0

        p_value = float(np.mean(null_ckas >= obs_cka))
        results[pair_name] = {
            "observed_cka": round(obs_cka, 6),
            "null_mean": round(float(np.mean(null_ckas)), 6),
            "null_std": round(float(np.std(null_ckas)), 6),
            "p_value": round(p_value, 6),
        }
    return results


# ── Also compute original (non-residualized) per-pair p-values ───────
def per_pair_permutation_original(data, layer_idx, n_triples, n_perm):
    """Per-pair permutation test on ORIGINAL (non-residualized) hidden states."""
    n = n_triples
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    rng = np.random.RandomState(SEED + 3)

    grams = {}
    for fmt in FORMATS:
        X = data[fmt][:n_triples, layer_idx, :]
        grams[fmt] = X @ X.T
    centered = {fmt: _center_gram(grams[fmt]) for fmt in FORMATS}

    hsic_xx = {}
    for fmt in FORMATS:
        hsic_xx[fmt] = float(np.sum(centered[fmt].astype(np.float64) ** 2))

    results = {}
    for f1, f2 in FORMAT_PAIRS:
        pair_name = f"{f1}-{f2}"
        obs_cka = _cka_from_centered(centered[f1], centered[f2])

        null_ckas = np.empty(n_perm, dtype=np.float32)
        for p in range(n_perm):
            perm = rng.permutation(n)
            KY_perm = grams[f2][np.ix_(perm, perm)]
            KY_c = H @ KY_perm @ H
            hsic_xy = np.float64(np.sum(centered[f1].astype(np.float64) * KY_c.astype(np.float64)))
            hsic_yy = np.float64(np.sum(KY_c.astype(np.float64) ** 2))
            denom = np.sqrt(hsic_xx[f1] * hsic_yy)
            null_ckas[p] = hsic_xy / denom if denom > 1e-12 else 0.0

        p_value = float(np.mean(null_ckas >= obs_cka))
        results[pair_name] = {
            "observed_cka": round(obs_cka, 6),
            "null_mean": round(float(np.mean(null_ckas)), 6),
            "p_value": round(p_value, 6),
        }
    return results


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["coder", "viscoder2", "qwen25",
                                 "codestral", "starcoder2", "deepseek"])
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/cache/hidden_states")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-triples", type=int, default=252)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    n_triples = 8 if args.smoke else args.n_triples
    n_perm = 20 if args.smoke else args.n_perm
    out_dir = PROJECT_ROOT / "artifacts" / "d081_format_mean_residual_cka"

    print(f"=== D081 Task 1A: Format-Mean Residualized CKA ===")
    print(f"  models: {args.models}")
    print(f"  n_triples={n_triples}, n_perm={n_perm}")
    print(f"  output: {out_dir}")
    t0_total = time.time()

    all_results = {
        "metadata": {
            "method": "format_mean_subtraction",
            "n_perm": n_perm,
            "n_triples": n_triples,
            "description": "Subtract per-format centroid, recompute CKA + per-pair permutation",
        },
        "models": {},
    }

    # Also collect all per-pair p-values for BH-FDR (Task 4)
    all_pvalues = []

    for model in args.models:
        print(f"\n--- {model} ---")
        meta = read_model_meta(cache_dir, model)
        layers = meta["layers"]
        hidden_dim = meta["hidden_dim"]
        n_saved = meta["n_saved"]
        n_actual = min(n_triples, n_saved)
        print(f"  layers={layers}, hidden_dim={hidden_dim}, n_saved={n_saved}")

        data = load_hidden_states(cache_dir, model, layers, hidden_dim, n_actual)

        model_results = {}
        for li, layer in enumerate(layers):
            layer_key = f"L{layer}"
            print(f"  Layer {layer} ({li+1}/{len(layers)})...")

            # Original CKA + per-pair permutation
            orig_results = per_pair_permutation_original(data, li, n_actual, n_perm)

            # Residualized CKA + per-pair permutation
            res_data = residualize_format_mean(data, li, n_actual)
            res_results = per_pair_permutation(res_data, n_actual, n_perm)

            layer_out = {}
            for pair_name in [f"{f1}-{f2}" for f1, f2 in FORMAT_PAIRS]:
                layer_out[pair_name] = {
                    "original_cka": orig_results[pair_name]["observed_cka"],
                    "original_p": orig_results[pair_name]["p_value"],
                    "residual_cka": res_results[pair_name]["observed_cka"],
                    "residual_p": res_results[pair_name]["p_value"],
                    "residual_null_mean": res_results[pair_name]["null_mean"],
                }
                # Collect for BH-FDR
                all_pvalues.append({
                    "model": model, "layer": layer_key, "pair": pair_name,
                    "type": "visual_cross_format",
                    "p_original": orig_results[pair_name]["p_value"],
                    "p_residualized": res_results[pair_name]["p_value"],
                })

            model_results[layer_key] = layer_out
            print(f"    orig: {[orig_results[f'{f1}-{f2}']['observed_cka'] for f1, f2 in FORMAT_PAIRS]}")
            print(f"    res:  {[res_results[f'{f1}-{f2}']['observed_cka'] for f1, f2 in FORMAT_PAIRS]}")

            del res_data
            gc.collect()

        all_results["models"][model] = model_results
        del data
        gc.collect()

    elapsed = time.time() - t0_total
    all_results["metadata"]["elapsed_s"] = round(elapsed, 1)

    # Save main results
    _save_json(out_dir / "results.json", all_results)
    print(f"\nSaved: {out_dir / 'results.json'}")

    # Save per-pair p-values for BH-FDR
    _save_json(out_dir / "all_pvalues_for_fdr.json", {
        "description": "Per-pair per-layer p-values for BH-FDR correction",
        "n_perm": n_perm,
        "tests": all_pvalues,
    })
    print(f"Saved: {out_dir / 'all_pvalues_for_fdr.json'}")

    # Print summary
    print(f"\n=== Summary (elapsed {elapsed:.0f}s) ===")
    for model in args.models:
        if model in all_results["models"]:
            md = all_results["models"][model]
            orig_ckas = []
            res_ckas = []
            for layer_data in md.values():
                for pair_data in layer_data.values():
                    orig_ckas.append(pair_data["original_cka"])
                    res_ckas.append(pair_data["residual_cka"])
            print(f"  {model}: orig_mean={np.mean(orig_ckas):.4f}, "
                  f"res_mean={np.mean(res_ckas):.4f}, "
                  f"ratio={np.mean(res_ckas)/np.mean(orig_ckas):.3f}")


if __name__ == "__main__":
    main()
