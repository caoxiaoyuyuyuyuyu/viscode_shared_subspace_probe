#!/usr/bin/env python
"""Verify: does format classifier still work on rank-2 residualized data?

If accuracy remains high after projecting out the LR coef_ subspace,
the rank-2 projection is insufficient to remove format information.

Also implements iterative residualization: repeatedly project out format
subspace until format classifier drops below threshold.

Usage:
  # Quick verification:
  python scripts/verify_residualized_accuracy.py \
      --models coder \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --mode verify

  # Iterative residualization (full pipeline):
  python scripts/verify_residualized_accuracy.py \
      --models coder \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --mode iterative \
      --n-perm 5000 --n-bootstrap 1000

  # Smoke test:
  python scripts/verify_residualized_accuracy.py \
      --models coder --cache-dir /tmp/smoke_hs --mode verify --smoke
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.stdout.reconfigure(line_buffering=True)

FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42
CHANCE_LEVEL = 1.0 / len(FORMATS)  # 0.333


def _save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


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
    raise FileNotFoundError(f"No summary.json for {model}")


def load_hidden_states(cache_dir, model, layers, hidden_dim, n_triples):
    n_layers = len(layers)
    data = {}
    for fmt in FORMATS:
        tensors = []
        d = cache_dir / model / fmt
        for i in range(n_triples):
            t = torch.load(d / f"{i}.pt", map_location="cpu", weights_only=True)
            assert t.shape == (n_layers, hidden_dim)
            tensors.append(t.float().numpy())
        data[fmt] = np.stack(tensors, axis=0)
        print(f"  {model}/{fmt}: {data[fmt].shape}")
    return data


def fit_classifier(X, y):
    """Train LR, return fitted model + 5-fold CV accuracy."""
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED, C=1.0)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    clf.fit(X, y)  # fit on full data for coef_
    return clf, float(np.mean(scores)), float(np.std(scores))


def project_out(H, W):
    """Project H to orthogonal complement of row space of W."""
    WWT = W @ W.T
    WWT_inv = np.linalg.inv(WWT)
    P = W.T @ WWT_inv @ W
    return H - H @ P


# ── CKA functions ──────────────────────────────────────────────────
def _center_gram(K):
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    return H @ K @ H


def _cka_from_centered(KX_c, KY_c):
    hsic_xy = np.sum(KX_c * KY_c)
    hsic_xx = np.sum(KX_c * KX_c)
    hsic_yy = np.sum(KY_c * KY_c)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0


def compute_cka_from_data(fmt_data, n_triples):
    """Compute mean CKA across format pairs from hidden state matrices."""
    grams = {}
    for fmt in FORMATS:
        X = fmt_data[fmt]
        grams[fmt] = X @ X.T

    centered = {fmt: _center_gram(grams[fmt]) for fmt in FORMATS}
    cka_vals = []
    for f1, f2 in FORMAT_PAIRS:
        cka_vals.append(_cka_from_centered(centered[f1], centered[f2]))
    return float(np.mean(cka_vals)), cka_vals


def run_bootstrap_ci(fmt_data, n_triples, n_bootstrap, frac=0.8):
    """Bootstrap CI for CKA."""
    n = n_triples
    m = int(n * frac)
    H = np.eye(m, dtype=np.float32) - np.float32(1.0 / m)
    rng = np.random.RandomState(SEED)

    grams = {}
    for fmt in FORMATS:
        X = fmt_data[fmt]
        grams[fmt] = X @ X.T

    boot_means = np.empty(n_bootstrap, dtype=np.float32)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=m, replace=False)
        pair_ckas = []
        for f1, f2 in FORMAT_PAIRS:
            KX = grams[f1][np.ix_(idx, idx)]
            KY = grams[f2][np.ix_(idx, idx)]
            KX_c = H @ KX @ H
            KY_c = H @ KY @ H
            hsic_xy = np.sum(KX_c * KY_c)
            hsic_xx = np.sum(KX_c * KX_c)
            hsic_yy = np.sum(KY_c * KY_c)
            denom = np.sqrt(hsic_xx * hsic_yy)
            pair_ckas.append(hsic_xy / denom if denom > 1e-12 else 0.0)
        boot_means[b] = np.mean(pair_ckas)

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": round(float(np.mean(boot_means)), 6),
        "ci_low": round(float(ci_low), 6),
        "ci_high": round(float(ci_high), 6),
    }


def run_a2_perm(fmt_data, n_triples, n_perm):
    """A2 permutation test on given data."""
    n = n_triples
    H_center = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    rng = np.random.RandomState(SEED + 2)

    grams = {fmt: fmt_data[fmt] @ fmt_data[fmt].T for fmt in FORMATS}
    centered = {fmt: H_center @ grams[fmt] @ H_center for fmt in FORMATS}
    hsic_xx = {fmt: float(np.sum(centered[fmt] ** 2)) for fmt in FORMATS}

    obs_vals = []
    for f1, f2 in FORMAT_PAIRS:
        obs_vals.append(_cka_from_centered(centered[f1], centered[f2]))
    obs_mean = float(np.mean(obs_vals))

    null_means = np.empty(n_perm, dtype=np.float32)
    for p_idx in range(n_perm):
        perm = rng.permutation(n)
        null_vals = []
        for f1, f2 in FORMAT_PAIRS:
            KY_perm = grams[f2][np.ix_(perm, perm)]
            KY_c = H_center @ KY_perm @ H_center
            hsic_xy = np.sum(centered[f1] * KY_c)
            hsic_yy = np.sum(KY_c * KY_c)
            denom = np.sqrt(hsic_xx[f1] * hsic_yy)
            null_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
        null_means[p_idx] = np.mean(null_vals)
        if (p_idx + 1) % 500 == 0:
            print(f"      perm {p_idx+1}/{n_perm}")

    p_value = float(np.mean(null_means >= obs_mean))
    return {
        "observed": round(obs_mean, 6),
        "null_mean": round(float(np.mean(null_means)), 6),
        "null_std": round(float(np.std(null_means)), 6),
        "null_95th": round(float(np.percentile(null_means, 95)), 6),
        "p_value": round(p_value, 6),
        "n_perm": n_perm,
    }


# ── Verify Mode ────────────────────────────────────────────────────
def run_verify(model, data, layers, n_triples, out_dir):
    """Quick check: format accuracy on rank-2 residualized data."""
    print(f"\n  === VERIFY: format accuracy on residualized data ===")
    results = []

    for li, layer in enumerate(layers):
        # Build X, y
        X = np.concatenate([data[fmt][:n_triples, li, :] for fmt in FORMATS], axis=0)
        y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])

        # Original accuracy
        _, orig_acc, orig_std = fit_classifier(X, y)

        # Project out rank-2
        clf_orig = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED, C=1.0)
        clf_orig.fit(X, y)
        W = clf_orig.coef_.astype(np.float32)
        X_res = project_out(X, W)

        # Residualized accuracy
        _, res_acc, res_std = fit_classifier(X_res, y)

        results.append({
            "layer": layer,
            "original_accuracy": round(orig_acc, 4),
            "residualized_accuracy": round(res_acc, 4),
            "residualized_std": round(res_std, 4),
            "W_rank": int(np.linalg.matrix_rank(W)),
        })
        print(f"    L{layer}: orig={orig_acc:.4f}, residualized={res_acc:.4f} (±{res_std:.4f})")

    _save_json(out_dir / model / "verify_accuracy.json", results)
    return results


# ── Iterative Mode ─────────────────────────────────────────────────
def run_iterative(model, data, layers, n_triples, out_dir, n_perm, n_bootstrap,
                  acc_threshold=0.50, max_iterations=50):
    """Iteratively project out format subspace until classifier drops below threshold."""
    print(f"\n  === ITERATIVE RESIDUALIZATION ===")
    print(f"  Threshold: format accuracy < {acc_threshold}")
    print(f"  Max iterations: {max_iterations}")

    all_layer_results = {}

    for li, layer in enumerate(layers):
        print(f"\n  --- Layer L{layer} ---")

        # Per-format data at this layer
        fmt_data = {fmt: data[fmt][:n_triples, li, :].copy() for fmt in FORMATS}

        # Original CKA
        orig_cka, _ = compute_cka_from_data(fmt_data, n_triples)

        # Original variance
        orig_var = sum(np.var(fmt_data[fmt]) for fmt in FORMATS)

        iterations = []
        total_dims_removed = 0
        iteration = 0

        while iteration < max_iterations:
            # Build classification dataset from current fmt_data
            X = np.concatenate([fmt_data[fmt] for fmt in FORMATS], axis=0)
            y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])

            # Train classifier
            clf, cv_acc, cv_std = fit_classifier(X, y)
            W = clf.coef_.astype(np.float32)
            W_rank = int(np.linalg.matrix_rank(W))

            # Compute CKA on current data
            cka_mean, cka_per_pair = compute_cka_from_data(fmt_data, n_triples)

            # Variance retained
            cur_var = sum(np.var(fmt_data[fmt]) for fmt in FORMATS)
            var_retained = cur_var / orig_var if orig_var > 0 else 0

            iter_result = {
                "iteration": iteration,
                "format_accuracy_cv": round(cv_acc, 4),
                "format_accuracy_std": round(cv_std, 4),
                "W_rank": W_rank,
                "total_dims_removed": total_dims_removed,
                "cka_mean": round(cka_mean, 6),
                "cka_per_pair": {f"{f1}-{f2}": round(v, 6)
                                 for (f1, f2), v in zip(FORMAT_PAIRS, cka_per_pair)},
                "variance_retained": round(float(var_retained), 4),
            }
            iterations.append(iter_result)

            print(f"    iter {iteration}: acc={cv_acc:.4f}(±{cv_std:.4f}), "
                  f"CKA={cka_mean:.4f}, dims_removed={total_dims_removed}, "
                  f"var_ret={var_retained:.4f}")

            # Check stopping condition
            if cv_acc < acc_threshold:
                print(f"    → STOPPED: accuracy {cv_acc:.4f} < threshold {acc_threshold}")
                break

            # Project out format subspace from each format's data
            for fmt in FORMATS:
                fmt_data[fmt] = project_out(fmt_data[fmt], W)

            total_dims_removed += W_rank
            iteration += 1

        # Final stats after convergence
        final_cka = iterations[-1]["cka_mean"]
        final_acc = iterations[-1]["format_accuracy_cv"]

        # Bootstrap CI on final residualized data (if in iterative+full mode)
        boot_ci = None
        a2_result = None
        if n_bootstrap > 0:
            print(f"    Computing bootstrap CI on final residualized data...")
            boot_ci = run_bootstrap_ci(fmt_data, n_triples, n_bootstrap)
            print(f"    Bootstrap: {boot_ci['mean']:.4f} [{boot_ci['ci_low']:.4f}, {boot_ci['ci_high']:.4f}]")

        if n_perm > 0:
            print(f"    Computing A2 permutation ({n_perm} perms)...")
            a2_result = run_a2_perm(fmt_data, n_triples, n_perm)
            print(f"    A2: obs={a2_result['observed']:.4f}, null={a2_result['null_mean']:.4f}, "
                  f"p={a2_result['p_value']}")

        layer_result = {
            "layer": layer,
            "n_iterations": len(iterations),
            "total_dims_removed": total_dims_removed,
            "final_format_accuracy": final_acc,
            "original_cka": round(orig_cka, 6),
            "final_cka": round(final_cka, 6),
            "cka_delta_abs": round(final_cka - orig_cka, 6),
            "cka_delta_pct": round((final_cka - orig_cka) / orig_cka * 100, 1)
                if orig_cka > 0 else 0,
            "cka_retained_pct": round(final_cka / orig_cka * 100, 1)
                if orig_cka > 0 else 0,
            "final_variance_retained": iterations[-1]["variance_retained"],
            "iterations": iterations,
        }
        if boot_ci:
            layer_result["bootstrap_ci"] = boot_ci
        if a2_result:
            layer_result["a2_permutation"] = a2_result

        all_layer_results[layer] = layer_result

    # Save
    _save_json(out_dir / model / "iterative_residualization.json", all_layer_results)

    # A2 aggregate across layers (using final residualized data)
    if n_perm > 0:
        print(f"\n  === Aggregate A2 permutation (all layers, final residualized) ===")
        # Rebuild final residualized data for all layers
        all_fmt_data = {}
        for li, layer in enumerate(layers):
            fmt_data_layer = {fmt: data[fmt][:n_triples, li, :].copy() for fmt in FORMATS}
            n_iters = all_layer_results[layer]["n_iterations"]
            for it in range(n_iters - 1):  # last iteration is the check, not a projection
                X = np.concatenate([fmt_data_layer[fmt] for fmt in FORMATS], axis=0)
                y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])
                clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED, C=1.0)
                clf.fit(X, y)
                W = clf.coef_.astype(np.float32)
                for fmt in FORMATS:
                    fmt_data_layer[fmt] = project_out(fmt_data_layer[fmt], W)
            all_fmt_data[li] = fmt_data_layer

        # Aggregate A2
        n = n_triples
        H_center = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
        rng = np.random.RandomState(SEED + 2)

        obs_vals = []
        all_centered = {}
        all_grams = {}
        all_hsic_xx = {}
        for li in range(len(layers)):
            all_grams[li] = {fmt: all_fmt_data[li][fmt] @ all_fmt_data[li][fmt].T for fmt in FORMATS}
            all_centered[li] = {fmt: H_center @ all_grams[li][fmt] @ H_center for fmt in FORMATS}
            all_hsic_xx[li] = {fmt: float(np.sum(all_centered[li][fmt] ** 2)) for fmt in FORMATS}
            for f1, f2 in FORMAT_PAIRS:
                obs_vals.append(_cka_from_centered(all_centered[li][f1], all_centered[li][f2]))
        obs_mean = float(np.mean(obs_vals))

        null_means = np.empty(n_perm, dtype=np.float32)
        for p_idx in range(n_perm):
            perm = rng.permutation(n)
            null_vals = []
            for li in range(len(layers)):
                for f1, f2 in FORMAT_PAIRS:
                    KY_perm = all_grams[li][f2][np.ix_(perm, perm)]
                    KY_c = H_center @ KY_perm @ H_center
                    hsic_xy = np.sum(all_centered[li][f1] * KY_c)
                    hsic_yy = np.sum(KY_c * KY_c)
                    denom = np.sqrt(all_hsic_xx[li][f1] * hsic_yy)
                    null_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
            null_means[p_idx] = np.mean(null_vals)
            if (p_idx + 1) % 500 == 0:
                print(f"    agg perm {p_idx+1}/{n_perm}")

        agg_p = float(np.mean(null_means >= obs_mean))
        agg_result = {
            "observed": round(obs_mean, 6),
            "null_mean": round(float(np.mean(null_means)), 6),
            "null_std": round(float(np.std(null_means)), 6),
            "null_95th": round(float(np.percentile(null_means, 95)), 6),
            "p_value": round(agg_p, 6),
            "n_perm": n_perm,
        }
        print(f"    Aggregate A2: obs={obs_mean:.4f}, null={np.mean(null_means):.4f}, p={agg_p}")
        _save_json(out_dir / model / "iterative_a2_aggregate.json", agg_result)

    # Summary table
    print(f"\n  {'='*70}")
    print(f"  ITERATIVE RESIDUALIZATION SUMMARY: {model}")
    print(f"  {'='*70}")
    print(f"  {'Layer':<8} {'Iters':>6} {'Dims':>6} {'FinalAcc':>10} {'OrigCKA':>10} "
          f"{'FinalCKA':>10} {'Retained':>10}")
    print(f"  {'-'*68}")
    for layer, r in all_layer_results.items():
        print(f"  L{layer:<6} {r['n_iterations']:>6} {r['total_dims_removed']:>6} "
              f"{r['final_format_accuracy']:>10.4f} {r['original_cka']:>10.4f} "
              f"{r['final_cka']:>10.4f} {r['cka_retained_pct']:>9.1f}%")

    return all_layer_results


def main():
    parser = argparse.ArgumentParser(description="Verify/Iterative Residualized CKA")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/cache/hidden_states")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--mode", choices=["verify", "iterative"], default="verify")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-triples", type=int, default=None)
    parser.add_argument("--acc-threshold", type=float, default=0.50,
                        help="Stop iterating when format accuracy drops below this")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "format_residualized_cka"

    if args.smoke:
        args.n_perm = 50
        args.n_bootstrap = 10
        args.n_triples = args.n_triples or 12

    print(f"Mode: {args.mode}")
    print(f"Models: {args.models}")

    for model in args.models:
        meta = read_model_meta(cache_dir, model)
        layers = meta["layers"]
        hidden_dim = meta["hidden_dim"]
        n_triples = args.n_triples or meta["n_saved"]

        print(f"\n  Loading {model}...")
        data = load_hidden_states(cache_dir, model, layers, hidden_dim, n_triples)

        if args.mode == "verify":
            run_verify(model, data, layers, n_triples, out_dir)
        elif args.mode == "iterative":
            run_iterative(model, data, layers, n_triples, out_dir,
                          n_perm=args.n_perm, n_bootstrap=args.n_bootstrap,
                          acc_threshold=args.acc_threshold)

        del data
        gc.collect()

    print(f"\nDONE. Results in {out_dir}")


if __name__ == "__main__":
    main()
