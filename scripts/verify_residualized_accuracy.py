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

  # Random baseline (requires iterative results as reference):
  python scripts/verify_residualized_accuracy.py \
      --models coder \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --mode random-baseline \
      --iterative-ref-dir /path/to/iterative/output \
      --n-repeats 50

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
    """Project H to orthogonal complement of row space of W.

    Uses QR decomposition for numerical stability (avoids inv of near-singular
    WWT after many iterations of residualization).
    """
    # QR on W^T gives orthonormal basis Q for column space of W^T = row space of W
    Q, _ = np.linalg.qr(W.T, mode="reduced")  # Q: (d, k) orthonormal
    # Project out: H_res = H - H @ Q @ Q^T
    HQ = H @ Q          # (n, k)
    return H - HQ @ Q.T  # (n, d)


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
                  acc_threshold=0.40, max_iterations=50, hidden_dim=None):
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

            dims_ratio = total_dims_removed / hidden_dim if hidden_dim else 0
            iter_result = {
                "iteration": iteration,
                "format_accuracy_cv": round(cv_acc, 4),
                "format_accuracy_std": round(cv_std, 4),
                "W_rank": W_rank,
                "total_dims_removed": total_dims_removed,
                "dims_removed_ratio": round(dims_ratio, 4),
                "cka_mean": round(cka_mean, 6),
                "cka_per_pair": {f"{f1}-{f2}": round(v, 6)
                                 for (f1, f2), v in zip(FORMAT_PAIRS, cka_per_pair)},
                "variance_retained": round(float(var_retained), 4),
            }
            iterations.append(iter_result)

            print(f"    iter {iteration}: acc={cv_acc:.4f}(±{cv_std:.4f}), "
                  f"CKA={cka_mean:.4f}, dims_removed={total_dims_removed}"
                  f"({dims_ratio:.1%}), var_ret={var_retained:.4f}")

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

        final_dims_ratio = total_dims_removed / hidden_dim if hidden_dim else 0
        layer_result = {
            "layer": layer,
            "n_iterations": len(iterations),
            "total_dims_removed": total_dims_removed,
            "dims_removed_ratio": round(final_dims_ratio, 4),
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

        # Incremental save after each layer
        _save_json(out_dir / model / "iterative_residualization.json", all_layer_results)
        print(f"    [saved incremental results to iterative_residualization.json]")

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
    print(f"  {'Layer':<8} {'Iters':>6} {'Dims':>6} {'Dim%':>7} {'FinalAcc':>10} {'OrigCKA':>10} "
          f"{'FinalCKA':>10} {'Retained':>10}")
    print(f"  {'-'*75}")
    for layer, r in all_layer_results.items():
        dim_pct = r.get('dims_removed_ratio', 0) * 100
        flag = " ⚠" if dim_pct > 30 else ""
        print(f"  L{layer:<6} {r['n_iterations']:>6} {r['total_dims_removed']:>6} "
              f"{dim_pct:>6.1f}% {r['final_format_accuracy']:>10.4f} {r['original_cka']:>10.4f} "
              f"{r['final_cka']:>10.4f} {r['cka_retained_pct']:>9.1f}%{flag}")

    return all_layer_results


# ── Random Baseline Mode ──────────────────────────────────────────
def run_random_baseline(model, data, layers, n_triples, out_dir, hidden_dim,
                        iterative_ref_dir=None, n_repeats=50):
    """Random projection baseline: project out same number of random dims as iterative.

    For each layer, sample random orthonormal directions matching the number of
    dims removed by iterative residualization, project to orthogonal complement,
    and compute CKA, variance_retained, format_accuracy. Repeat n_repeats times.
    """
    # Load iterative results to get dims_removed per layer
    ref_dir = iterative_ref_dir or out_dir
    ref_path = ref_dir / model / "iterative_residualization.json"
    if not ref_path.exists():
        print(f"  ERROR: iterative results not found at {ref_path}")
        print(f"  Run --mode iterative first to generate reference dims_removed per layer.")
        return None

    with open(ref_path) as f:
        iterative_results = json.load(f)

    print(f"\n  === RANDOM BASELINE (n_repeats={n_repeats}) ===")
    print(f"  Reference: {ref_path}")

    rng = np.random.RandomState(SEED + 100)
    all_layer_results = {}

    for li, layer in enumerate(layers):
        layer_key = str(layer)
        if layer_key not in iterative_results:
            print(f"  WARNING: Layer {layer} not in iterative results, skipping")
            continue

        n_dims = iterative_results[layer_key]["total_dims_removed"]
        iter_cka = iterative_results[layer_key]["final_cka"]
        iter_var = iterative_results[layer_key]["final_variance_retained"]
        iter_acc = iterative_results[layer_key]["final_format_accuracy"]

        print(f"\n  --- Layer L{layer}: projecting out {n_dims}/{hidden_dim} random dims "
              f"({n_dims/hidden_dim:.1%}) × {n_repeats} repeats ---")

        if n_dims == 0:
            print(f"    Skipping (0 dims removed in iterative)")
            continue

        # Per-format data at this layer
        fmt_data_orig = {fmt: data[fmt][:n_triples, li, :].copy() for fmt in FORMATS}

        cka_samples = []
        var_samples = []
        acc_samples = []

        orig_var = sum(np.var(fmt_data_orig[fmt]) for fmt in FORMATS)

        for rep in range(n_repeats):
            # Sample random orthonormal directions
            # Generate random matrix and QR-decompose for orthonormal basis
            R = rng.randn(hidden_dim, n_dims).astype(np.float32)
            Q, _ = np.linalg.qr(R, mode="reduced")  # Q: (hidden_dim, n_dims) orthonormal

            # Project out: H_res = H - H @ Q @ Q^T
            fmt_data_rand = {}
            for fmt in FORMATS:
                HQ = fmt_data_orig[fmt] @ Q
                fmt_data_rand[fmt] = fmt_data_orig[fmt] - HQ @ Q.T

            # CKA
            cka_mean, _ = compute_cka_from_data(fmt_data_rand, n_triples)
            cka_samples.append(cka_mean)

            # Variance retained
            cur_var = sum(np.var(fmt_data_rand[fmt]) for fmt in FORMATS)
            var_samples.append(cur_var / orig_var if orig_var > 0 else 0)

            # Format accuracy
            X = np.concatenate([fmt_data_rand[fmt] for fmt in FORMATS], axis=0)
            y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])
            _, acc, _ = fit_classifier(X, y)
            acc_samples.append(acc)

            if (rep + 1) % 10 == 0:
                print(f"    repeat {rep+1}/{n_repeats}")

        cka_arr = np.array(cka_samples)
        var_arr = np.array(var_samples)
        acc_arr = np.array(acc_samples)

        layer_result = {
            "layer": layer,
            "n_dims_projected": n_dims,
            "dims_ratio": round(n_dims / hidden_dim, 4),
            "n_repeats": n_repeats,
            "random_cka_mean": round(float(np.mean(cka_arr)), 6),
            "random_cka_std": round(float(np.std(cka_arr)), 6),
            "random_var_retained_mean": round(float(np.mean(var_arr)), 4),
            "random_var_retained_std": round(float(np.std(var_arr)), 4),
            "random_accuracy_mean": round(float(np.mean(acc_arr)), 4),
            "random_accuracy_std": round(float(np.std(acc_arr)), 4),
            "iterative_cka": iter_cka,
            "iterative_var_retained": iter_var,
            "iterative_accuracy": iter_acc,
            "theoretical_var_retained": round(1.0 - n_dims / hidden_dim, 4),
        }
        all_layer_results[layer] = layer_result

        print(f"    Random:    CKA={np.mean(cka_arr):.4f}±{np.std(cka_arr):.4f}, "
              f"var_ret={np.mean(var_arr):.4f}±{np.std(var_arr):.4f}, "
              f"acc={np.mean(acc_arr):.4f}±{np.std(acc_arr):.4f}")
        print(f"    Iterative: CKA={iter_cka:.4f}, var_ret={iter_var:.4f}, acc={iter_acc:.4f}")
        print(f"    Theoret:   var_ret={1.0 - n_dims/hidden_dim:.4f}")

    _save_json(out_dir / model / "random_baseline.json", all_layer_results)

    # Summary table
    print(f"\n  {'='*90}")
    print(f"  RANDOM BASELINE vs ITERATIVE SUMMARY: {model}")
    print(f"  {'='*90}")
    print(f"  {'Layer':<7} {'Dims':>5} {'Dim%':>6} | {'RandCKA':>10} {'IterCKA':>10} | "
          f"{'RandVar':>10} {'IterVar':>10} {'TheoVar':>10} | {'RandAcc':>10} {'IterAcc':>10}")
    print(f"  {'-'*88}")
    for layer, r in all_layer_results.items():
        dim_pct = r['dims_ratio'] * 100
        print(f"  L{layer:<5} {r['n_dims_projected']:>5} {dim_pct:>5.1f}% | "
              f"{r['random_cka_mean']:>10.4f} {r['iterative_cka']:>10.4f} | "
              f"{r['random_var_retained_mean']:>10.4f} {r['iterative_var_retained']:>10.4f} "
              f"{r['theoretical_var_retained']:>10.4f} | "
              f"{r['random_accuracy_mean']:>10.4f} {r['iterative_accuracy']:>10.4f}")

    return all_layer_results


# ── PCA Baseline Mode ─────────────────────────────────────────────
def run_pca_baseline(model, data, layers, n_triples, out_dir, hidden_dim,
                     iterative_ref_dir=None):
    """PCA variance-matched baseline: project out top-k PCs to match iterative var_retained.

    For each layer, find the number of top principal components whose removal
    produces approximately the same variance_retained as iterative residualization.
    Then compute CKA and format_accuracy on this PCA-residualized data.

    This controls for variance amount (not dimension count), complementing the
    random baseline which controls for dimension count (not variance amount).
    """
    ref_dir = iterative_ref_dir or out_dir
    ref_path = ref_dir / model / "iterative_residualization.json"
    if not ref_path.exists():
        print(f"  ERROR: iterative results not found at {ref_path}")
        return None

    with open(ref_path) as f:
        iterative_results = json.load(f)

    print(f"\n  === PCA VARIANCE-MATCHED BASELINE ===")
    print(f"  Reference: {ref_path}")

    all_layer_results = {}

    for li, layer in enumerate(layers):
        layer_key = str(layer)
        if layer_key not in iterative_results:
            print(f"  WARNING: Layer {layer} not in iterative results, skipping")
            continue

        iter_var_ret = iterative_results[layer_key]["final_variance_retained"]
        iter_cka = iterative_results[layer_key]["final_cka"]
        iter_acc = iterative_results[layer_key]["final_format_accuracy"]
        iter_dims = iterative_results[layer_key]["total_dims_removed"]

        print(f"\n  --- Layer L{layer}: target var_retained ≈ {iter_var_ret:.4f} ---")

        # Per-format data at this layer
        fmt_data_orig = {fmt: data[fmt][:n_triples, li, :].copy() for fmt in FORMATS}
        orig_var = sum(np.var(fmt_data_orig[fmt]) for fmt in FORMATS)

        # Compute PCA on concatenated data
        X_all = np.concatenate([fmt_data_orig[fmt] for fmt in FORMATS], axis=0)
        X_mean = X_all.mean(axis=0, keepdims=True)
        X_centered = X_all - X_mean

        # SVD for PCA
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        explained_var = S ** 2 / X_centered.shape[0]
        total_var = float(np.sum(explained_var))
        cumvar = np.cumsum(explained_var) / total_var

        # Find k such that removing top-k PCs gives var_retained ≈ iter_var_ret
        # Removing top-k means retaining 1 - cumvar[k-1] fraction
        target_retain = iter_var_ret
        k_match = 1
        for k in range(1, len(cumvar)):
            if (1.0 - cumvar[k - 1]) <= target_retain:
                k_match = k
                break
        else:
            k_match = len(cumvar)

        # Also check k_match-1 for closer match
        if k_match > 1:
            var_ret_k = 1.0 - cumvar[k_match - 1]
            var_ret_km1 = 1.0 - cumvar[k_match - 2]
            if abs(var_ret_km1 - target_retain) < abs(var_ret_k - target_retain):
                k_match = k_match - 1

        actual_var_retained_pca = 1.0 - cumvar[k_match - 1]

        print(f"    PCA: removing top-{k_match} PCs → var_retained={actual_var_retained_pca:.4f} "
              f"(target={target_retain:.4f}, iterative dims={iter_dims})")

        # Project out top-k PCs
        # Top-k PC directions are rows of Vt[:k_match]
        Q_pca = Vt[:k_match].T  # (hidden_dim, k_match)

        fmt_data_pca = {}
        for fmt in FORMATS:
            centered = fmt_data_orig[fmt] - X_mean
            HQ = centered @ Q_pca
            fmt_data_pca[fmt] = centered - HQ @ Q_pca.T + X_mean  # add mean back

        # Verify actual variance retained
        pca_var = sum(np.var(fmt_data_pca[fmt]) for fmt in FORMATS)
        actual_var_check = pca_var / orig_var if orig_var > 0 else 0

        # CKA
        cka_mean, cka_per_pair = compute_cka_from_data(fmt_data_pca, n_triples)

        # Format accuracy
        X_pca = np.concatenate([fmt_data_pca[fmt] for fmt in FORMATS], axis=0)
        y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])
        _, pca_acc, pca_acc_std = fit_classifier(X_pca, y)

        layer_result = {
            "layer": layer,
            "n_pcs_removed": k_match,
            "pca_var_retained": round(float(actual_var_check), 4),
            "pca_cka": round(cka_mean, 6),
            "pca_cka_per_pair": {f"{f1}-{f2}": round(v, 6)
                                 for (f1, f2), v in zip(FORMAT_PAIRS, cka_per_pair)},
            "pca_accuracy": round(pca_acc, 4),
            "pca_accuracy_std": round(pca_acc_std, 4),
            "iterative_dims_removed": iter_dims,
            "iterative_var_retained": iter_var_ret,
            "iterative_cka": iter_cka,
            "iterative_accuracy": iter_acc,
        }
        all_layer_results[layer] = layer_result

        print(f"    PCA:       CKA={cka_mean:.4f}, var_ret={actual_var_check:.4f}, "
              f"acc={pca_acc:.4f}, k={k_match}")
        print(f"    Iterative: CKA={iter_cka:.4f}, var_ret={iter_var_ret:.4f}, "
              f"acc={iter_acc:.4f}, dims={iter_dims}")

    _save_json(out_dir / model / "pca_baseline.json", all_layer_results)

    # Summary table
    print(f"\n  {'='*95}")
    print(f"  PCA BASELINE vs ITERATIVE SUMMARY: {model}")
    print(f"  {'='*95}")
    print(f"  {'Layer':<7} {'PCA_k':>6} {'IterDim':>8} | {'PCA_CKA':>9} {'IterCKA':>9} | "
          f"{'PCA_Var':>9} {'IterVar':>9} | {'PCA_Acc':>9} {'IterAcc':>9}")
    print(f"  {'-'*85}")
    for layer, r in all_layer_results.items():
        print(f"  L{layer:<5} {r['n_pcs_removed']:>6} {r['iterative_dims_removed']:>8} | "
              f"{r['pca_cka']:>9.4f} {r['iterative_cka']:>9.4f} | "
              f"{r['pca_var_retained']:>9.4f} {r['iterative_var_retained']:>9.4f} | "
              f"{r['pca_accuracy']:>9.4f} {r['iterative_accuracy']:>9.4f}")

    return all_layer_results


def run_pca_mink(model, data, layers, n_triples, out_dir, hidden_dim,
                 n_perm, n_bootstrap, acc_threshold=0.50, max_k=100):
    """PCA format-removal: find minimum k top-PCs to drop format accuracy below threshold.

    For each layer:
    1. Compute SVD on concatenated [svg+tikz+asy] data
    2. Incrementally remove top-k PCs (k=1,2,3,...) until format acc < threshold
    3. At that k, compute CKA + bootstrap CI + A2 permutation test
    """
    print(f"\n  === PCA FORMAT-REMOVAL (min-k, threshold={acc_threshold}) ===")

    all_layer_results = {}

    for li, layer in enumerate(layers):
        print(f"\n  --- Layer L{layer} ---")

        fmt_data_orig = {fmt: data[fmt][:n_triples, li, :].copy() for fmt in FORMATS}
        orig_var = sum(np.var(fmt_data_orig[fmt]) for fmt in FORMATS)

        # Original CKA
        orig_cka, _ = compute_cka_from_data(fmt_data_orig, n_triples)

        # SVD on concatenated data
        X_all = np.concatenate([fmt_data_orig[fmt] for fmt in FORMATS], axis=0)
        X_mean = X_all.mean(axis=0, keepdims=True)
        X_centered = X_all - X_mean
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Incrementally remove top-k PCs
        k_found = None
        k_search_log = []

        for k in range(1, min(max_k, len(S)) + 1):
            Q_pca = Vt[:k].T  # (hidden_dim, k)

            fmt_data_pca = {}
            for fmt in FORMATS:
                centered = fmt_data_orig[fmt] - X_mean
                HQ = centered @ Q_pca
                fmt_data_pca[fmt] = centered - HQ @ Q_pca.T + X_mean

            # Format accuracy
            X_pca = np.concatenate([fmt_data_pca[fmt] for fmt in FORMATS], axis=0)
            y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])
            _, pca_acc, pca_std = fit_classifier(X_pca, y)

            k_search_log.append({"k": k, "acc": round(pca_acc, 4), "std": round(pca_std, 4)})
            print(f"    k={k}: acc={pca_acc:.4f} (±{pca_std:.4f})")

            if pca_acc < acc_threshold:
                k_found = k
                break

        if k_found is None:
            print(f"    WARNING: format acc never dropped below {acc_threshold} up to k={max_k}")
            k_found = k  # use last k tested

        # Recompute final projection at k_found
        Q_pca = Vt[:k_found].T
        fmt_data_final = {}
        for fmt in FORMATS:
            centered = fmt_data_orig[fmt] - X_mean
            HQ = centered @ Q_pca
            fmt_data_final[fmt] = centered - HQ @ Q_pca.T + X_mean

        # Final metrics
        pca_var = sum(np.var(fmt_data_final[fmt]) for fmt in FORMATS)
        var_retained = pca_var / orig_var if orig_var > 0 else 0

        cka_mean, cka_per_pair = compute_cka_from_data(fmt_data_final, n_triples)

        X_final = np.concatenate([fmt_data_final[fmt] for fmt in FORMATS], axis=0)
        y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])
        _, final_acc, final_std = fit_classifier(X_final, y)

        # Bootstrap CI
        boot_ci = None
        if n_bootstrap > 0:
            print(f"    Computing bootstrap CI (B={n_bootstrap})...")
            boot_ci = run_bootstrap_ci(fmt_data_final, n_triples, n_bootstrap)
            print(f"    Bootstrap: {boot_ci['mean']:.4f} [{boot_ci['ci_low']:.4f}, {boot_ci['ci_high']:.4f}]")

        # A2 permutation
        a2_result = None
        if n_perm > 0:
            print(f"    Computing A2 permutation ({n_perm} perms)...")
            a2_result = run_a2_perm(fmt_data_final, n_triples, n_perm)
            print(f"    A2: obs={a2_result['observed']:.4f}, null={a2_result['null_mean']:.4f}, p={a2_result['p_value']}")

        layer_result = {
            "layer": layer,
            "k_min": k_found,
            "k_ratio": round(k_found / hidden_dim, 6),
            "final_accuracy": round(final_acc, 4),
            "final_accuracy_std": round(final_std, 4),
            "original_cka": round(orig_cka, 6),
            "pca_cka": round(cka_mean, 6),
            "cka_delta_pct": round((cka_mean - orig_cka) / orig_cka * 100, 1) if orig_cka > 0 else 0,
            "pca_cka_per_pair": {f"{f1}-{f2}": round(v, 6)
                                 for (f1, f2), v in zip(FORMAT_PAIRS, cka_per_pair)},
            "variance_retained": round(float(var_retained), 4),
            "k_search_log": k_search_log,
        }
        if boot_ci:
            layer_result["bootstrap_ci"] = boot_ci
        if a2_result:
            layer_result["a2_permutation"] = a2_result

        all_layer_results[layer] = layer_result

        # Incremental save
        _save_json(out_dir / model / "pca_mink.json", all_layer_results)

        print(f"    RESULT: k={k_found}, CKA={cka_mean:.4f} (orig={orig_cka:.4f}, "
              f"delta={layer_result['cka_delta_pct']:+.1f}%), var_ret={var_retained:.4f}, "
              f"acc={final_acc:.4f}")

    # Summary table
    print(f"\n  {'='*80}")
    print(f"  PCA FORMAT-REMOVAL (min-k) SUMMARY: {model}")
    print(f"  {'='*80}")
    print(f"  {'Layer':<7} {'k':>5} {'k%':>7} | {'OrigCKA':>9} {'PCA_CKA':>9} {'Delta%':>8} | "
          f"{'VarRet':>8} {'Acc':>8}")
    print(f"  {'-'*72}")
    for layer, r in all_layer_results.items():
        print(f"  L{layer:<5} {r['k_min']:>5} {r['k_ratio']*100:>6.2f}% | "
              f"{r['original_cka']:>9.4f} {r['pca_cka']:>9.4f} {r['cka_delta_pct']:>+7.1f}% | "
              f"{r['variance_retained']:>8.4f} {r['final_accuracy']:>8.4f}")

    return all_layer_results


def main():
    parser = argparse.ArgumentParser(description="Verify/Iterative Residualized CKA")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/cache/hidden_states")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--mode", choices=["verify", "iterative", "random-baseline",
                                          "pca-baseline", "pca-mink"], default="verify")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-triples", type=int, default=None)
    parser.add_argument("--acc-threshold", type=float, default=0.40,
                        help="Stop iterating when format accuracy drops below this")
    parser.add_argument("--n-repeats", type=int, default=50,
                        help="Number of random repeats for random-baseline mode")
    parser.add_argument("--iterative-ref-dir", type=str, default=None,
                        help="Directory with iterative results for random-baseline reference")
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
        args.n_repeats = min(args.n_repeats, 5)

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
                          acc_threshold=args.acc_threshold,
                          hidden_dim=hidden_dim)
        elif args.mode == "random-baseline":
            ref_dir = Path(args.iterative_ref_dir) if args.iterative_ref_dir else None
            run_random_baseline(model, data, layers, n_triples, out_dir,
                                hidden_dim=hidden_dim,
                                iterative_ref_dir=ref_dir,
                                n_repeats=args.n_repeats)
        elif args.mode == "pca-baseline":
            ref_dir = Path(args.iterative_ref_dir) if args.iterative_ref_dir else None
            run_pca_baseline(model, data, layers, n_triples, out_dir,
                             hidden_dim=hidden_dim,
                             iterative_ref_dir=ref_dir)
        elif args.mode == "pca-mink":
            run_pca_mink(model, data, layers, n_triples, out_dir,
                         hidden_dim=hidden_dim,
                         n_perm=args.n_perm, n_bootstrap=args.n_bootstrap,
                         acc_threshold=args.acc_threshold)

        del data
        gc.collect()

    print(f"\nDONE. Results in {out_dir}")


if __name__ == "__main__":
    main()
