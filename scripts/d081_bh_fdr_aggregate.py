#!/usr/bin/env python
"""D081 Task 4: BH-FDR global correction aggregator.

Reads per-pair p-values from Task 1A output (all_pvalues_for_fdr.json),
applies Benjamini-Hochberg FDR correction at α=0.05, outputs comparison table.

Can run locally (no GPU needed) once p-values are available.

Usage:
  python scripts/d081_bh_fdr_aggregate.py \
      --input artifacts/d081_format_mean_residual_cka/all_pvalues_for_fdr.json \
      --output artifacts/d081_bh_fdr_correction/results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _save_json(path, obj):
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Returns (significant_mask, q_values).
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n

    # Find largest k where p_(k) <= k * alpha / n
    rejected = sorted_p <= thresholds
    if not rejected.any():
        # No rejections
        qvals = np.minimum.accumulate(
            (sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
        qvals_orig = np.empty(n)
        qvals_orig[sorted_idx] = qvals
        return np.zeros(n, dtype=bool), qvals_orig

    max_k = np.max(np.where(rejected)[0])
    sig = np.zeros(n, dtype=bool)
    sig[sorted_idx[:max_k + 1]] = True

    # Compute adjusted q-values (Benjamini-Yekutieli step-up)
    qvals = np.minimum.accumulate(
        (sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    qvals = np.minimum(qvals, 1.0)
    qvals_orig = np.empty(n)
    qvals_orig[sorted_idx] = qvals

    return sig, qvals_orig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=str(PROJECT_ROOT / "artifacts" /
                                    "d081_format_mean_residual_cka" /
                                    "all_pvalues_for_fdr.json"))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "artifacts" /
                                    "d081_bh_fdr_correction" / "results.json"))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    print(f"=== D081 Task 4: BH-FDR Global Correction ===")
    print(f"  input: {args.input}")
    print(f"  alpha: {args.alpha}")

    with open(args.input) as f:
        data = json.load(f)

    tests = data["tests"]
    n_perm = data["n_perm"]
    print(f"  Total tests: {len(tests)}, n_perm: {n_perm}")

    # Extract p-values
    p_original = np.array([t["p_original"] for t in tests])
    p_residualized = np.array([t.get("p_residualized", t["p_original"]) for t in tests])

    # Apply BH-FDR to original p-values
    sig_orig, q_orig = bh_fdr(p_original, args.alpha)
    sig_res, q_res = bh_fdr(p_residualized, args.alpha)

    # Also apply Bonferroni for comparison
    bonf_orig = p_original * len(p_original)
    bonf_sig = bonf_orig <= args.alpha

    # Build output
    per_test = []
    for i, t in enumerate(tests):
        per_test.append({
            "model": t["model"],
            "layer": t["layer"],
            "pair": t["pair"],
            "type": t.get("type", "visual_cross_format"),
            "p_uncorrected": round(float(p_original[i]), 6),
            "q_bh": round(float(q_orig[i]), 6),
            "significant_uncorrected": bool(p_original[i] < 0.05),
            "significant_bh": bool(sig_orig[i]),
            "significant_bonferroni": bool(bonf_sig[i]),
            "p_residualized": round(float(p_residualized[i]), 6),
            "q_bh_residualized": round(float(q_res[i]), 6),
            "significant_bh_residualized": bool(sig_res[i]),
        })

    # Summary statistics
    models = sorted(set(t["model"] for t in tests))
    summary = {
        "total_tests": len(tests),
        "alpha": args.alpha,
        "n_perm": n_perm,
        "original": {
            "significant_uncorrected": int(np.sum(p_original < 0.05)),
            "significant_bh": int(np.sum(sig_orig)),
            "significant_bonferroni": int(np.sum(bonf_sig)),
        },
        "residualized": {
            "significant_uncorrected": int(np.sum(p_residualized < 0.05)),
            "significant_bh": int(np.sum(sig_res)),
        },
        "by_model": {},
    }

    for model in models:
        mask = np.array([t["model"] == model for t in tests])
        n_model = int(mask.sum())
        summary["by_model"][model] = {
            "n_tests": n_model,
            "sig_uncorr": int(np.sum(p_original[mask] < 0.05)),
            "sig_bh": int(np.sum(sig_orig[mask])),
            "sig_bonf": int(np.sum(bonf_sig[mask])),
            "sig_res_uncorr": int(np.sum(p_residualized[mask] < 0.05)),
            "sig_res_bh": int(np.sum(sig_res[mask])),
        }

    results = {
        "metadata": {
            "method": "benjamini_hochberg",
            "alpha": args.alpha,
            "total_tests": len(tests),
            "n_perm": n_perm,
        },
        "summary": summary,
        "per_test": per_test,
    }

    _save_json(Path(args.output), results)
    print(f"\nSaved: {args.output}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"  Total tests: {len(tests)}")
    print(f"  Original p-values:")
    print(f"    Significant (uncorrected p<0.05): "
          f"{summary['original']['significant_uncorrected']}/{len(tests)}")
    print(f"    Significant (BH FDR q<0.05):      "
          f"{summary['original']['significant_bh']}/{len(tests)}")
    print(f"    Significant (Bonferroni):          "
          f"{summary['original']['significant_bonferroni']}/{len(tests)}")
    print(f"  Residualized p-values:")
    print(f"    Significant (uncorrected p<0.05): "
          f"{summary['residualized']['significant_uncorrected']}/{len(tests)}")
    print(f"    Significant (BH FDR q<0.05):      "
          f"{summary['residualized']['significant_bh']}/{len(tests)}")

    print(f"\n  Per-model breakdown:")
    print(f"  {'Model':<12} {'Tests':>5} {'Uncorr':>6} {'BH':>4} {'Bonf':>4} "
          f"{'Res-Unc':>7} {'Res-BH':>6}")
    for model in models:
        m = summary["by_model"][model]
        print(f"  {model:<12} {m['n_tests']:>5} {m['sig_uncorr']:>6} "
              f"{m['sig_bh']:>4} {m['sig_bonf']:>4} "
              f"{m['sig_res_uncorr']:>7} {m['sig_res_bh']:>6}")


if __name__ == "__main__":
    main()
