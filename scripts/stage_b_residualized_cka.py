#!/usr/bin/env python
"""Format-Residualized CKA: project out format subspace, then recompute CKA.

Uses the F1 format classifier (LogisticRegression) weights to define the
format-discriminative subspace. Projects hidden states to the orthogonal
complement and recomputes CKA + bootstrap CI + A2 permutation null.

Usage:
  # Pilot (single model):
  python scripts/stage_b_residualized_cka.py \
      --models coder \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --n-perm 5000 --n-bootstrap 1000

  # Full (all 6 models):
  python scripts/stage_b_residualized_cka.py \
      --models coder viscoder2 qwen25 codestral starcoder2 deepseek \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --n-perm 5000 --n-bootstrap 1000

  # Smoke test:
  python scripts/stage_b_residualized_cka.py \
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

np.seterr(over='raise', invalid='raise')
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(line_buffering=True)

FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42


# ── Helpers (same as multimodel) ──────────────────────────────────
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
    raise FileNotFoundError(f"No summary.json found for model {model} in {cache_dir}")


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
        arr = np.stack(tensors, axis=0)  # (n, n_layers, hidden_dim)
        data[fmt] = arr
        print(f"  {model}/{fmt}: {arr.shape}")
    return data


# ── CKA Core (same as multimodel) ─────────────────────────────────
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


# ── Format Subspace Extraction ─────────────────────────────────────
def fit_format_classifier(data, layer_idx, n_triples):
    """Train LogisticRegression to classify format, return coef_ matrix."""
    X = np.concatenate([data[fmt][:n_triples, layer_idx, :] for fmt in FORMATS], axis=0)
    y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED, C=1.0)
    clf.fit(X, y)
    accuracy = clf.score(X, y)

    # coef_ shape: (n_classes, hidden_dim) = (3, hidden_dim)
    W = clf.coef_.astype(np.float32)
    return W, accuracy


def project_out_format_subspace(H, W):
    """Project hidden states to orthogonal complement of format subspace.

    H: (n, hidden_dim) — hidden state matrix for one format at one layer
    W: (k, hidden_dim) — format classifier weight rows (k format directions)

    Returns H_residual = H - H @ W^T @ (W @ W^T)^{-1} @ W
    """
    # W^T: (hidden_dim, k)
    # W @ W^T: (k, k)
    # (W @ W^T)^{-1}: (k, k)
    # Projection matrix P = W^T @ (W @ W^T)^{-1} @ W: (hidden_dim, hidden_dim)
    # H_residual = H - H @ P = H @ (I - P)

    WWT = W @ W.T  # (k, k)
    WWT_inv = np.linalg.inv(WWT)  # (k, k)
    P = W.T @ WWT_inv @ W  # (hidden_dim, hidden_dim) — format projection matrix

    H_residual = H - H @ P
    return H_residual


def residualize_data(data, W, layer_idx, n_triples):
    """Apply format subspace projection to all formats at one layer.

    Returns residualized data dict with same structure.
    """
    res_data = {}
    for fmt in FORMATS:
        H = data[fmt][:n_triples, layer_idx, :]  # (n, hidden_dim)
        H_res = project_out_format_subspace(H, W)
        res_data[fmt] = H_res
    return res_data


# ── CKA on Residualized Data ──────────────────────────────────────
def compute_residualized_cka(res_data):
    """Compute CKA for one layer from residualized hidden states."""
    grams = {}
    for fmt in FORMATS:
        grams[fmt] = res_data[fmt] @ res_data[fmt].T  # (n, n)

    centered = {fmt: _center_gram(grams[fmt]) for fmt in FORMATS}
    results = []
    for f1, f2 in FORMAT_PAIRS:
        cka_val = _cka_from_centered(centered[f1], centered[f2])
        results.append({"pair": f"{f1}-{f2}", "cka": round(cka_val, 6)})
    return results, grams


def run_residualized_bootstrap(res_grams, n_triples, n_bootstrap, frac=0.8):
    """Bootstrap CI for residualized CKA at one layer."""
    n = n_triples
    m = int(n * frac)
    H = np.eye(m, dtype=np.float32) - np.float32(1.0 / m)
    rng = np.random.RandomState(SEED)

    boot_means = np.empty(n_bootstrap, dtype=np.float32)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=m, replace=False)
        pair_ckas = []
        for f1, f2 in FORMAT_PAIRS:
            KX = res_grams[f1][np.ix_(idx, idx)]
            KY = res_grams[f2][np.ix_(idx, idx)]
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


def run_residualized_a2_perm(data, W, layer_idx, n_triples, n_perm):
    """A2 permutation test on residualized hidden states at one layer.

    Permutes sample indices of the second format's hidden states BEFORE
    residualization, to preserve the null hypothesis structure.
    """
    n = n_triples
    rng = np.random.RandomState(SEED + 2)

    # Observed: residualize then compute CKA
    res_data = residualize_data(data, W, layer_idx, n_triples)
    obs_vals = []
    for f1, f2 in FORMAT_PAIRS:
        KX = res_data[f1] @ res_data[f1].T
        KY = res_data[f2] @ res_data[f2].T
        KX_c = _center_gram(KX)
        KY_c = _center_gram(KY)
        obs_vals.append(_cka_from_centered(KX_c, KY_c))
    obs_mean = float(np.mean(obs_vals))

    # Null distribution: permute sample indices
    H_center = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    null_means = np.empty(n_perm, dtype=np.float32)
    for p_idx in range(n_perm):
        perm = rng.permutation(n)
        null_vals = []
        for f1, f2 in FORMAT_PAIRS:
            # f1 stays, f2 permuted — both residualized with same W
            KX = res_data[f1] @ res_data[f1].T
            # Permute the residualized f2
            f2_perm = res_data[f2][perm, :]
            KY = f2_perm @ f2_perm.T
            KX_c = H_center @ KX @ H_center
            KY_c = H_center @ KY @ H_center
            hsic_xy = np.sum(KX_c * KY_c)
            hsic_xx = np.sum(KX_c * KX_c)
            hsic_yy = np.sum(KY_c * KY_c)
            denom = np.sqrt(hsic_xx * hsic_yy)
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


# ── Per-Layer A2 Permutation (aggregate across layers) ─────────────
def run_residualized_a2_perm_aggregate(data, format_weights, layers, n_triples, n_perm):
    """A2 permutation test aggregated across all layers (matches original A2 design)."""
    n = n_triples
    rng = np.random.RandomState(SEED + 2)

    # Residualize all layers
    all_res_data = {}
    for li, layer in enumerate(layers):
        W = format_weights[li]
        all_res_data[li] = residualize_data(data, W, li, n_triples)

    # Observed
    obs_vals = []
    for li in range(len(layers)):
        for f1, f2 in FORMAT_PAIRS:
            KX = all_res_data[li][f1] @ all_res_data[li][f1].T
            KY = all_res_data[li][f2] @ all_res_data[li][f2].T
            KX_c = _center_gram(KX)
            KY_c = _center_gram(KY)
            obs_vals.append(_cka_from_centered(KX_c, KY_c))
    obs_mean = float(np.mean(obs_vals))

    # Pre-compute observed gram matrices for f1 (stays fixed)
    H_center = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    obs_grams_centered = {}
    obs_hsic_xx = {}
    for li in range(len(layers)):
        obs_grams_centered[li] = {}
        obs_hsic_xx[li] = {}
        for fmt in FORMATS:
            K = all_res_data[li][fmt] @ all_res_data[li][fmt].T
            Kc = H_center @ K @ H_center
            obs_grams_centered[li][fmt] = Kc
            obs_hsic_xx[li][fmt] = float(np.sum(Kc ** 2))

    # Null
    null_means = np.empty(n_perm, dtype=np.float32)
    for p_idx in range(n_perm):
        perm = rng.permutation(n)
        null_vals = []
        for li in range(len(layers)):
            for f1, f2 in FORMAT_PAIRS:
                KX_c = obs_grams_centered[li][f1]
                f2_perm = all_res_data[li][f2][perm, :]
                KY = f2_perm @ f2_perm.T
                KY_c = H_center @ KY @ H_center
                hsic_xy = np.sum(KX_c * KY_c)
                hsic_yy = np.sum(KY_c * KY_c)
                denom = np.sqrt(obs_hsic_xx[li][f1] * hsic_yy)
                null_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
        null_means[p_idx] = np.mean(null_vals)

        if (p_idx + 1) % 500 == 0:
            print(f"    perm {p_idx+1}/{n_perm}")

    p_value = float(np.mean(null_means >= obs_mean))
    return {
        "observed": round(obs_mean, 6),
        "null_mean": round(float(np.mean(null_means)), 6),
        "null_std": round(float(np.std(null_means)), 6),
        "null_95th": round(float(np.percentile(null_means, 95)), 6),
        "p_value": round(p_value, 6),
        "n_perm": n_perm,
    }


# ── Main Pipeline ──────────────────────────────────────────────────
def analyze_one_model(model, cache_dir, out_dir, n_perm, n_bootstrap, n_triples):
    print(f"\n{'='*60}")
    print(f"  Format-Residualized CKA: {model}")
    print(f"{'='*60}")

    meta = read_model_meta(cache_dir, model)
    layers = meta["layers"]
    hidden_dim = meta["hidden_dim"]
    n_saved = meta["n_saved"]
    print(f"  Layers: {layers} (hidden_dim={hidden_dim}, n_saved={n_saved})")

    if n_triples is None:
        n_triples = n_saved
    assert n_triples <= n_saved

    model_out = out_dir / model
    model_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n  Loading hidden states...")
    data = load_hidden_states(cache_dir, model, layers, hidden_dim, n_triples)

    # Step 1: Train format classifiers per layer, extract weights
    print(f"\n  === Step 1: Format Classifier (per-layer) ===")
    format_weights = {}  # li -> W (k, hidden_dim)
    classifier_info = []
    for li, layer in enumerate(layers):
        W, acc = fit_format_classifier(data, li, n_triples)
        format_weights[li] = W
        rank = np.linalg.matrix_rank(W)
        classifier_info.append({
            "layer": layer,
            "accuracy": round(acc, 4),
            "W_shape": list(W.shape),
            "W_rank": int(rank),
        })
        print(f"    L{layer}: acc={acc:.4f}, W shape={W.shape}, rank={rank}")
    _save_json(model_out / "format_classifier_info.json", classifier_info)

    # Step 2: Compute residualized CKA per layer
    print(f"\n  === Step 2: Residualized CKA (per-layer) ===")
    cka_rows = []
    bootstrap_rows = []
    for li, layer in enumerate(layers):
        W = format_weights[li]

        # Residualize
        res_data = residualize_data(data, W, li, n_triples)

        # Variance retained
        orig_var = sum(np.var(data[fmt][:n_triples, li, :]) for fmt in FORMATS)
        res_var = sum(np.var(res_data[fmt]) for fmt in FORMATS)
        var_retained = res_var / orig_var if orig_var > 0 else 0

        # CKA
        cka_results, res_grams = compute_residualized_cka(res_data)
        mean_cka = np.mean([r["cka"] for r in cka_results])
        for r in cka_results:
            r["layer"] = layer
        cka_rows.extend(cka_results)

        # Bootstrap CI
        boot = run_residualized_bootstrap(res_grams, n_triples, n_bootstrap)
        boot["layer"] = layer
        boot["variance_retained"] = round(float(var_retained), 4)
        bootstrap_rows.append(boot)

        print(f"    L{layer}: res_CKA_mean={mean_cka:.4f}, "
              f"bootstrap={boot['mean']:.4f} [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}], "
              f"var_retained={var_retained:.4f}")

        del res_data, res_grams
        gc.collect()

    _save_json(model_out / "residualized_cka_per_layer.json", cka_rows)
    _save_json(model_out / "residualized_bootstrap_ci.json", bootstrap_rows)

    # Step 3: A2 permutation test (aggregate across layers)
    print(f"\n  === Step 3: A2 Permutation ({n_perm} perms, aggregate) ===")
    a2_result = run_residualized_a2_perm_aggregate(
        data, format_weights, layers, n_triples, n_perm
    )
    print(f"    obs={a2_result['observed']:.4f}, null={a2_result['null_mean']:.4f}, "
          f"p={a2_result['p_value']:.6f}")
    _save_json(model_out / "residualized_a2_permutation.json", a2_result)

    elapsed = time.time() - t0

    # Summary
    summary = {
        "model": model,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "n_triples": n_triples,
        "n_perm": n_perm,
        "n_bootstrap": n_bootstrap,
        "elapsed_s": round(elapsed, 1),
        "format_classifier_accuracy": {
            ci["layer"]: ci["accuracy"] for ci in classifier_info
        },
        "residualized_cka_per_layer": {
            b["layer"]: {
                "bootstrap_mean": b["mean"],
                "ci_low": b["ci_low"],
                "ci_high": b["ci_high"],
                "variance_retained": b["variance_retained"],
            }
            for b in bootstrap_rows
        },
        "residualized_cka_mean_all": round(float(np.mean([r["cka"] for r in cka_rows])), 6),
        "a2_observed": a2_result["observed"],
        "a2_null_mean": a2_result["null_mean"],
        "a2_p_value": a2_result["p_value"],
        "a2_obs_null_ratio": round(a2_result["observed"] / a2_result["null_mean"], 4)
            if a2_result["null_mean"] > 0 else None,
    }
    _save_json(model_out / "summary.json", summary)

    print(f"\n  {model} DONE in {elapsed:.1f}s")
    print(f"  Residualized CKA mean={summary['residualized_cka_mean_all']:.4f}")
    print(f"  A2: obs={a2_result['observed']:.4f}, null={a2_result['null_mean']:.4f}, "
          f"p={a2_result['p_value']}, ratio={summary['a2_obs_null_ratio']}")

    del data
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Format-Residualized CKA Analysis")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/cache/hidden_states")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-triples", type=int, default=None)
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

    print(f"Format-Residualized CKA Analysis")
    print(f"  Models: {args.models}")
    print(f"  Cache: {cache_dir}")
    print(f"  Output: {out_dir}")
    print(f"  n_perm={args.n_perm}, n_bootstrap={args.n_bootstrap}")

    all_summaries = {}
    for model in args.models:
        summary = analyze_one_model(
            model, cache_dir, out_dir,
            n_perm=args.n_perm,
            n_bootstrap=args.n_bootstrap,
            n_triples=args.n_triples,
        )
        all_summaries[model] = summary

    # Comparison table
    print(f"\n{'='*60}")
    print(f"  FORMAT-RESIDUALIZED CKA COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Res CKA':>10} {'A2 obs':>10} {'A2 null':>10} {'ratio':>8} {'p':>8}")
    print(f"{'-'*63}")
    for model, s in all_summaries.items():
        ratio_str = f"{s['a2_obs_null_ratio']:.2f}x" if s['a2_obs_null_ratio'] else "N/A"
        print(f"{model:<15} {s['residualized_cka_mean_all']:>10.4f} "
              f"{s['a2_observed']:>10.4f} {s['a2_null_mean']:>10.4f} "
              f"{ratio_str:>8} {s['a2_p_value']:>8.4f}")

    _save_json(out_dir / "residualized_summary.json", all_summaries)
    print(f"\nALL DONE. Results in {out_dir}")


if __name__ == "__main__":
    main()
