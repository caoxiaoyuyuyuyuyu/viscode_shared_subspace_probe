#!/usr/bin/env python
"""Stage B Cross-Family CKA Analysis: per-model CKA + permutation + PWCCA.

Reads hidden states from {cache_dir}/{model}/{format}/{i}.pt,
auto-detects layers and hidden_dim from summary.json.

Usage:
  # Production (server):
  python scripts/stage_b_analysis_multimodel.py \
      --models codestral starcoder2 deepseek \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --n-perm 5000 --n-bootstrap 1000

  # Smoke test:
  python scripts/stage_b_analysis_multimodel.py \
      --models codestral starcoder2 deepseek \
      --cache-dir /tmp/smoke_hs --smoke
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

sys.stdout.reconfigure(line_buffering=True)

FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42


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
    """Read layers and hidden_dim from summary.json of first available format."""
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


# ── Data Loader ──────────────────────────────────────────────────────
def load_hidden_states(cache_dir, model, layers, hidden_dim, n_triples):
    """Load hidden states for one model: data[format] = np.array(n, n_layers, hidden_dim)."""
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
        print(f"  {model}/{fmt}: {arr.shape} {arr.dtype}")
    return data


# ── CKA Core ─────────────────────────────────────────────────────────
def _center_gram(K):
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    return H @ K @ H


def _cka_from_centered(KX_c, KY_c):
    hsic_xy = np.sum(KX_c * KY_c)
    hsic_xx = np.sum(KX_c * KX_c)
    hsic_yy = np.sum(KY_c * KY_c)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_raw_grams(data, n_layers):
    """Compute uncentered Gram matrices: grams[fmt][li] = (n,n)."""
    grams = {}
    for fmt in FORMATS:
        grams[fmt] = {}
        for li in range(n_layers):
            X = data[fmt][:, li, :]
            grams[fmt][li] = X @ X.T
    return grams


# ── C. CKA ───────────────────────────────────────────────────────────
def run_cka(raw_grams, layers):
    rows = []
    for li, layer in enumerate(layers):
        centered = {fmt: _center_gram(raw_grams[fmt][li]) for fmt in FORMATS}
        for f1, f2 in FORMAT_PAIRS:
            cka_val = _cka_from_centered(centered[f1], centered[f2])
            rows.append({"layer": layer, "pair": f"{f1}-{f2}", "cka": round(cka_val, 6)})
        del centered
    return rows


# ── D. Bootstrap CI ──────────────────────────────────────────────────
def run_bootstrap_ci(raw_grams, layers, n_triples, n_bootstrap, frac=0.8):
    n = n_triples
    m = int(n * frac)
    H = np.eye(m, dtype=np.float32) - np.float32(1.0 / m)
    rng = np.random.RandomState(SEED)
    rows = []

    for li, layer in enumerate(layers):
        boot_means = np.empty(n_bootstrap, dtype=np.float32)
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=m, replace=False)
            pair_ckas = []
            for f1, f2 in FORMAT_PAIRS:
                KX = raw_grams[f1][li][np.ix_(idx, idx)]
                KY = raw_grams[f2][li][np.ix_(idx, idx)]
                KX_c = H @ KX @ H
                KY_c = H @ KY @ H
                hsic_xy = np.sum(KX_c * KY_c)
                hsic_xx = np.sum(KX_c * KX_c)
                hsic_yy = np.sum(KY_c * KY_c)
                denom = np.sqrt(hsic_xx * hsic_yy)
                pair_ckas.append(hsic_xy / denom if denom > 1e-12 else 0.0)
            boot_means[b] = np.mean(pair_ckas)

        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        rows.append({
            "layer": layer,
            "mean": round(float(np.mean(boot_means)), 6),
            "ci_low": round(float(ci_low), 6),
            "ci_high": round(float(ci_high), 6),
            "ci_width": round(float(ci_high - ci_low), 6),
        })
        print(f"    L{layer}: {np.mean(boot_means):.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    return rows


# ── F. A2 Permutation Test ───────────────────────────────────────────
def run_a2_permutation(raw_grams, layers, n_triples, n_perm):
    n = n_triples
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    rng = np.random.RandomState(SEED + 2)

    # Pre-compute centered Grams
    centered = {}
    hsic_xx_cache = {}
    for fmt in FORMATS:
        centered[fmt] = {}
        for li in range(len(layers)):
            centered[fmt][li] = H @ raw_grams[fmt][li] @ H
            hsic_xx_cache[(fmt, li)] = float(np.sum(centered[fmt][li] ** 2))

    # Observed CKA
    obs_vals = []
    obs_per_pair = {}
    for li in range(len(layers)):
        for f1, f2 in FORMAT_PAIRS:
            v = _cka_from_centered(centered[f1][li], centered[f2][li])
            obs_vals.append(v)
            key = f"{f1}-{f2}"
            if key not in obs_per_pair:
                obs_per_pair[key] = []
            obs_per_pair[key].append(v)
    obs_mean = float(np.mean(obs_vals))

    # Null distribution
    null_means = np.empty(n_perm, dtype=np.float32)
    for p_idx in range(n_perm):
        perm = rng.permutation(n)
        null_vals = []
        for li in range(len(layers)):
            for f1, f2 in FORMAT_PAIRS:
                KX_c = centered[f1][li]
                KY_perm = raw_grams[f2][li][np.ix_(perm, perm)]
                KY_c = H @ KY_perm @ H
                hsic_xy = np.sum(KX_c * KY_c)
                hsic_yy = np.sum(KY_c * KY_c)
                denom = np.sqrt(hsic_xx_cache[(f1, li)] * hsic_yy)
                null_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
        null_means[p_idx] = np.mean(null_vals)

        if (p_idx + 1) % 500 == 0:
            print(f"    perm {p_idx+1}/{n_perm} (RSS={_rss_gb():.2f} GB)")

    p_value = float(np.mean(null_means >= obs_mean))
    result = {
        "observed_cka_mean": round(obs_mean, 6),
        "null_mean": round(float(np.mean(null_means)), 6),
        "null_std": round(float(np.std(null_means)), 6),
        "null_95th": round(float(np.percentile(null_means, 95)), 6),
        "null_99th": round(float(np.percentile(null_means, 99)), 6),
        "p_value": round(p_value, 6),
        "n_permutations": n_perm,
        "observed_per_pair": {k: round(float(np.mean(v)), 6) for k, v in obs_per_pair.items()},
    }
    return result


# ── Robustness: PWCCA ────────────────────────────────────────────────
def pwcca_score(X, Y, k=50):
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

    corrs = np.array([np.corrcoef(X_cc[:, i], Y_cc[:, i])[0, 1] for i in range(n_comp)])
    corrs = np.abs(np.nan_to_num(corrs, nan=0.0))

    var_x = pca_x.explained_variance_ratio_[:n_comp]
    weights = var_x / var_x.sum()

    return round(float(np.sum(weights * corrs)), 6)


def run_pwcca(data, layers):
    rows = []
    for li, layer in enumerate(layers):
        for f1, f2 in FORMAT_PAIRS:
            X = data[f1][:, li, :]
            Y = data[f2][:, li, :]
            val = pwcca_score(X, Y)
            rows.append({"layer": layer, "pair": f"{f1}-{f2}", "pwcca": val})
        vals = [r["pwcca"] for r in rows if r["layer"] == layer]
        print(f"    L{layer}: mean PWCCA = {np.mean(vals):.4f}")
    return rows


# ── PWCCA Permutation Null ───────────────────────────────────────────
def run_pwcca_perm_null(data, layers, n_perm=300):
    """Quick permutation null for PWCCA (per-model aggregate)."""
    rng = np.random.RandomState(SEED + 3)
    n = data[FORMATS[0]].shape[0]

    # Observed
    obs_vals = []
    for li in range(len(layers)):
        for f1, f2 in FORMAT_PAIRS:
            obs_vals.append(pwcca_score(data[f1][:, li, :], data[f2][:, li, :]))
    obs_mean = float(np.mean(obs_vals))

    # Null
    null_means = []
    for p_idx in range(n_perm):
        perm = rng.permutation(n)
        perm_vals = []
        for li in range(len(layers)):
            for f1, f2 in FORMAT_PAIRS:
                perm_vals.append(pwcca_score(data[f1][:, li, :], data[f2][perm, li, :]))
        null_means.append(float(np.mean(perm_vals)))
        if (p_idx + 1) % 50 == 0:
            print(f"    PWCCA perm {p_idx+1}/{n_perm}")

    null_arr = np.array(null_means)
    p_value = float(np.mean(null_arr >= obs_mean))
    return {
        "observed": round(obs_mean, 6),
        "null_mean": round(float(np.mean(null_arr)), 6),
        "null_std": round(float(np.std(null_arr)), 6),
        "p_value": round(p_value, 6),
        "n_perm": n_perm,
    }


# ── Main ─────────────────────────────────────────────────────────────
def analyze_one_model(model, cache_dir, out_dir, n_perm, n_bootstrap, n_triples):
    """Full analysis pipeline for one model."""
    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"{'='*60}")

    # Read metadata
    meta = read_model_meta(cache_dir, model)
    layers = meta["layers"]
    hidden_dim = meta["hidden_dim"]
    n_saved = meta["n_saved"]
    print(f"  Layers: {layers} (hidden_dim={hidden_dim}, n_saved={n_saved})")

    if n_triples is None:
        n_triples = n_saved
    assert n_triples <= n_saved, f"Requested {n_triples} but only {n_saved} saved"

    model_out = out_dir / model
    model_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load data
    print(f"\n  Loading hidden states...")
    data = load_hidden_states(cache_dir, model, layers, hidden_dim, n_triples)

    # Raw Gram matrices
    print(f"  Computing Gram matrices...")
    raw_grams = compute_raw_grams(data, len(layers))

    # CKA
    print(f"\n  === CKA ===")
    cka_rows = run_cka(raw_grams, layers)
    for li, layer in enumerate(layers):
        vals = [r["cka"] for r in cka_rows if r["layer"] == layer]
        print(f"    L{layer}: mean={np.mean(vals):.4f}  {vals}")
    _save_json(model_out / "cka_per_layer_per_pair.json", cka_rows)

    # Bootstrap CI
    print(f"\n  === Bootstrap CI ({n_bootstrap} samples) ===")
    bootstrap_rows = run_bootstrap_ci(raw_grams, layers, n_triples, n_bootstrap)
    _save_json(model_out / "bootstrap_ci.json", bootstrap_rows)

    # A2 Permutation
    print(f"\n  === A2 Permutation ({n_perm} perms) ===")
    perm_result = run_a2_permutation(raw_grams, layers, n_triples, n_perm)
    print(f"    obs={perm_result['observed_cka_mean']:.4f}, "
          f"null={perm_result['null_mean']:.4f}, p={perm_result['p_value']:.6f}")
    _save_json(model_out / "a2_permutation.json", perm_result)

    # Free Gram matrices
    del raw_grams
    gc.collect()

    # PWCCA
    print(f"\n  === PWCCA ===")
    pwcca_rows = run_pwcca(data, layers)
    _save_json(model_out / "pwcca_per_layer_per_pair.json", pwcca_rows)

    # PWCCA permutation null (quick, 300 perms)
    print(f"\n  === PWCCA Permutation Null (300 perms) ===")
    pwcca_perm = run_pwcca_perm_null(data, layers, n_perm=300)
    print(f"    obs={pwcca_perm['observed']:.4f}, null={pwcca_perm['null_mean']:.4f}, "
          f"p={pwcca_perm['p_value']:.6f}")
    _save_json(model_out / "pwcca_perm_null.json", pwcca_perm)

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
        "cka_mean_all_layers": round(float(np.mean([r["cka"] for r in cka_rows])), 6),
        "cka_per_pair_mean": perm_result["observed_per_pair"],
        "a2_p_value": perm_result["p_value"],
        "a2_observed_vs_null": f"{perm_result['observed_cka_mean']:.4f} vs {perm_result['null_mean']:.4f}",
        "pwcca_mean": round(float(np.mean([r["pwcca"] for r in pwcca_rows])), 6),
        "pwcca_perm_p": pwcca_perm["p_value"],
    }
    _save_json(model_out / "summary.json", summary)
    print(f"\n  {model} DONE in {elapsed:.1f}s")
    print(f"  CKA mean={summary['cka_mean_all_layers']:.4f}, "
          f"A2 p={summary['a2_p_value']}, "
          f"PWCCA mean={summary['pwcca_mean']:.4f}")

    del data
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-family CKA analysis")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names (matching hidden_states subdirectories)")
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/cache/hidden_states",
                        help="Hidden states cache directory")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: artifacts/stage_b_analysis_multimodel/)")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-triples", type=int, default=None,
                        help="Number of triples (default: auto from summary.json)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "stage_b_analysis_multimodel"

    if args.smoke:
        args.n_perm = 50
        args.n_bootstrap = 10
        args.n_triples = args.n_triples or 12

    print(f"Cross-Family CKA Analysis")
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

    # Cross-model comparison table
    print(f"\n{'='*60}")
    print(f"  CROSS-FAMILY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'CKA mean':>10} {'A2 p':>8} {'PWCCA':>10} {'PWCCA p':>8}")
    print(f"{'-'*55}")
    for model, s in all_summaries.items():
        print(f"{model:<15} {s['cka_mean_all_layers']:>10.4f} {s['a2_p_value']:>8.4f} "
              f"{s['pwcca_mean']:>10.4f} {s['pwcca_perm_p']:>8.4f}")

    # Qwen reference (if available)
    qwen_models = ["coder", "viscoder2", "qwen25"]
    qwen_available = all(
        (cache_dir / m / "svg" / "summary.json").exists() for m in qwen_models
    )
    if qwen_available:
        print(f"\n  Qwen family reference (from prior analysis):")
        print(f"  coder:     CKA@L28=0.149, A2 p=0.0")
        print(f"  viscoder2: CKA@L28=0.137, A2 p=0.0")
        print(f"  qwen25:    CKA@L28=0.076, A2 p=0.0")

    _save_json(out_dir / "cross_family_summary.json", all_summaries)
    print(f"\nALL DONE. Results in {out_dir}")


if __name__ == "__main__":
    main()
