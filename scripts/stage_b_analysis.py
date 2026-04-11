#!/usr/bin/env python
"""Stage B Analysis v1: Probe fit / CKA / Bootstrap CI / A1 power sim / A2 permutation.

Pre-reg v3.3 + D026 amendment compliant:
  - NO quantitative predictivity claim
  - NO decoupling claim
  - N=18 contains VisCoder2 as "exploratory view with known memorization floor"
  - A1 power sim: N=12 AND N=18
  - Main metric: CKA (ρ), Procrustes + PWCCA as robustness

Usage:
  # Production (server):
  export PATH=/root/miniconda3/bin:$PATH
  cd /root/autodl-tmp/viscode_shared_subspace_probe
  python scripts/stage_b_analysis.py

  # Smoke test (local, no GPU needed):
  python scripts/stage_b_analysis.py --smoke

  # Force rerun (ignore checkpoints):
  python scripts/stage_b_analysis.py --smoke --force-rerun
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.stdout.reconfigure(line_buffering=True)

# ── Constants ─────────────────────────────────────────────────────────
LAYERS = [4, 8, 12, 16, 20, 24, 28]
MODELS = ["coder", "viscoder2", "qwen25"]
FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ── Helpers ───────────────────────────────────────────────────────────
def mem_gb():
    """Current RSS in GB."""
    return psutil.Process().memory_info().rss / 1024**3


def _save_ckpt(path, obj):
    """Atomic JSON checkpoint write with fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def _load_ckpt(path):
    with open(path) as f:
        return json.load(f)


# ── Smoke Data Generation ────────────────────────────────────────────
def generate_smoke_data(cache_dir, n_triples):
    """Generate fake .pt files for local smoke testing."""
    rng = np.random.RandomState(SEED + 99)
    any_created = False
    for model in MODELS:
        for fmt in FORMATS:
            d = cache_dir / model / fmt
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_triples):
                pt_path = d / f"{i}.pt"
                if not pt_path.exists():
                    t = torch.tensor(rng.randn(len(LAYERS), 3584).astype(np.float32))
                    torch.save(t, pt_path)
                    any_created = True
    if any_created:
        print(f"  Generated fake .pt data in {cache_dir}")


# ── A. Data Loader ───────────────────────────────────────────────────
def load_hidden_states(cache_dir, n_triples):
    """Load hidden states into data[model][format] = np.array(n, 7, 3584) float32."""
    print(f"=== A. Loading hidden states === (RSS={mem_gb():.2f} GB)")
    t0 = time.time()

    data = {}
    detected_dtype = None
    for model in MODELS:
        data[model] = {}
        for fmt in FORMATS:
            tensors = []
            d = cache_dir / model / fmt
            for i in range(n_triples):
                pt_path = d / f"{i}.pt"
                t = torch.load(pt_path, map_location="cpu", weights_only=True)
                assert t.shape == (len(LAYERS), 3584), f"{pt_path}: shape {t.shape}"
                if detected_dtype is None:
                    detected_dtype = str(t.dtype)
                # Enforce float32 — avoid float64 blowup
                tensors.append(t.float().numpy())
            arr = np.stack(tensors, axis=0)  # (n, 7, 3584) float32
            assert arr.dtype == np.float32
            data[model][fmt] = arr
            print(f"  {model}/{fmt}: {arr.shape} {arr.dtype}")

    elapsed = time.time() - t0
    print(f"  Data loading: {elapsed:.1f}s (RSS={mem_gb():.2f} GB)")
    return data, detected_dtype, elapsed


# ── Raw Gram Matrices (shared by C/D/E/F) ────────────────────────────
def compute_raw_grams(data):
    """Compute uncentered Gram matrices: raw_grams[model][fmt][li] = (n,n) float32."""
    print(f"  Computing raw Gram matrices... (RSS={mem_gb():.2f} GB)")
    raw_grams = {}
    for model in MODELS:
        raw_grams[model] = {}
        for fmt in FORMATS:
            raw_grams[model][fmt] = {}
            for li in range(len(LAYERS)):
                X = data[model][fmt][:, li, :]  # (n, 3584) float32
                raw_grams[model][fmt][li] = X @ X.T  # (n, n) float32
    print(f"  Raw Gram matrices done. (RSS={mem_gb():.2f} GB)")
    return raw_grams


# ── B. Probe Fit (Format Classifier) ────────────────────────────────
def run_probe_fit(data, n_triples):
    """Per-model, per-layer format classification (5-fold CV LogReg)."""
    print(f"\n=== B. Probe Fit (Format Classifier) === (RSS={mem_gb():.2f} GB)")
    t0 = time.time()

    rows = []
    for model in MODELS:
        for li, layer in enumerate(LAYERS):
            X = np.concatenate([data[model][fmt][:, li, :] for fmt in FORMATS], axis=0)
            y = np.concatenate([np.full(n_triples, i) for i, _ in enumerate(FORMATS)])

            clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            acc = float(scores.mean())
            n_total = len(y)
            n_test = n_total // 5
            n_train = n_total - n_test
            rows.append({
                "model": model, "layer": layer, "accuracy": round(acc, 4),
                "std": round(float(scores.std()), 4),
                "n_train": n_train, "n_test": n_test,
                "split_strategy": "5-fold-cv",
            })
            print(f"  {model} layer {layer}: acc={acc:.4f} (±{scores.std():.4f})")
            del X, y, clf, scores
    gc.collect()

    elapsed = time.time() - t0
    print(f"  Probe fit: {elapsed:.1f}s (RSS={mem_gb():.2f} GB)")
    return {"results": rows, "elapsed_s": round(elapsed, 1)}


# ── C. CKA (Main Metric ρ) ──────────────────────────────────────────
def _center_gram_f32(K):
    """HSIC centering of a Gram matrix, float32."""
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    return H @ K @ H


def _cka_from_centered(KX_c, KY_c):
    """CKA from pre-centered Gram matrices using trace trick."""
    hsic_xy = np.sum(KX_c * KY_c)
    hsic_xx = np.sum(KX_c * KX_c)
    hsic_yy = np.sum(KY_c * KY_c)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def run_cka(raw_grams):
    """CKA for each (model, layer, format_pair)."""
    print(f"\n=== C. CKA Computation === (RSS={mem_gb():.2f} GB)")
    t0 = time.time()

    rows = []
    for model in MODELS:
        for li, layer in enumerate(LAYERS):
            # Center Gram matrices on the fly
            centered = {}
            for fmt in FORMATS:
                centered[fmt] = _center_gram_f32(raw_grams[model][fmt][li])

            pair_vals = []
            for f1, f2 in FORMAT_PAIRS:
                cka_val = _cka_from_centered(centered[f1], centered[f2])
                rows.append({
                    "model": model, "layer": layer,
                    "pair": f"{f1}-{f2}", "cka": round(cka_val, 6),
                })
                pair_vals.append(cka_val)

            # Sanity: self-CKA
            self_cka = _cka_from_centered(centered["svg"], centered["svg"])
            if abs(self_cka - 1.0) > 0.01:
                print(f"  WARNING: self-CKA {model}/svg layer {layer} = {self_cka:.4f}")

            del centered
        gc.collect()

    for model in MODELS:
        print(f"  {model}:")
        for li, layer in enumerate(LAYERS):
            vals = [r["cka"] for r in rows
                    if r["model"] == model and r["layer"] == layer]
            print(f"    layer {layer}: mean CKA = {np.mean(vals):.4f} ({vals})")

    elapsed = time.time() - t0
    print(f"  CKA: {elapsed:.1f}s (RSS={mem_gb():.2f} GB)")
    return {"results": rows, "elapsed_s": round(elapsed, 1)}


# ── D. Bootstrap CI ─────────────────────────────────────────────────
def run_bootstrap_ci(raw_grams, n_triples, n_bootstrap):
    """Bootstrap 95% CI for CKA per (model, layer)."""
    print(f"\n=== D. Bootstrap CI ({n_bootstrap} samples) === (RSS={mem_gb():.2f} GB)")
    t0 = time.time()

    n = n_triples
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    rng = np.random.RandomState(SEED)
    rows = []

    for model in MODELS:
        for li, layer in enumerate(LAYERS):
            boot_means = np.empty(n_bootstrap, dtype=np.float32)
            for b in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                pair_ckas = []
                for f1, f2 in FORMAT_PAIRS:
                    KX = raw_grams[model][f1][li][np.ix_(idx, idx)]
                    KY = raw_grams[model][f2][li][np.ix_(idx, idx)]
                    KX_c = H @ KX @ H
                    KY_c = H @ KY @ H
                    hsic_xy = np.sum(KX_c * KY_c)
                    hsic_xx = np.sum(KX_c * KX_c)
                    hsic_yy = np.sum(KY_c * KY_c)
                    denom = np.sqrt(hsic_xx * hsic_yy)
                    pair_ckas.append(hsic_xy / denom if denom > 1e-12 else 0.0)
                    del KX, KY, KX_c, KY_c
                boot_means[b] = np.mean(pair_ckas)

                if (b + 1) % 100 == 0:
                    print(f"    {model} L{layer}: {b+1}/{n_bootstrap} (RSS={mem_gb():.2f} GB)")

            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            mean_val = float(np.mean(boot_means))
            se_val = float(np.std(boot_means))
            rows.append({
                "model": model, "layer": layer,
                "mean": round(mean_val, 6),
                "ci_low": round(float(ci_low), 6),
                "ci_high": round(float(ci_high), 6),
                "ci_width": round(float(ci_high - ci_low), 6),
                "se": round(se_val, 6),
            })
            print(f"  {model} layer {layer}: {mean_val:.4f} [{ci_low:.4f}, {ci_high:.4f}] (w={ci_high-ci_low:.4f})")
            del boot_means
            gc.collect()

    elapsed = time.time() - t0
    print(f"  Bootstrap CI: {elapsed:.1f}s (RSS={mem_gb():.2f} GB)")
    return {"results": rows, "n_bootstrap": n_bootstrap, "elapsed_s": round(elapsed, 1)}


# ── E. A1 Power Simulation ──────────────────────────────────────────
def run_a1_power_sim(raw_grams, n_triples, n_power_iter):
    """Monte Carlo power simulation for CKA detection at N=12 and N=18."""
    print(f"\n=== E. A1 Power Simulation ({n_power_iter} iterations) === (RSS={mem_gb():.2f} GB)")
    t0 = time.time()

    rng = np.random.RandomState(SEED + 1)
    rep_layer_idx = 5  # layer 24 = index 5
    n = n_triples
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    results = {}

    configs = {
        "N12": {"models": ["coder", "qwen25"], "n_cells": 12},
        "N18": {"models": MODELS, "n_cells": 18},
    }

    for config_name, cfg in configs.items():
        print(f"\n  {config_name} (models={cfg['models']}, cells={cfg['n_cells']}):")

        # Pre-compute centered Grams at representative layer
        centered = {}
        for model in cfg["models"]:
            centered[model] = {}
            for fmt in FORMATS:
                centered[model][fmt] = H @ raw_grams[model][fmt][rep_layer_idx] @ H

        # Step 1: Null distribution (500 permutations)
        n_null = 500
        null_dist = np.empty(n_null, dtype=np.float32)
        for p_idx in range(n_null):
            perm = rng.permutation(n)
            null_vals = []
            for model in cfg["models"]:
                for f1, f2 in FORMAT_PAIRS:
                    KX_c = centered[model][f1]
                    KY_perm = raw_grams[model][f2][rep_layer_idx][np.ix_(perm, perm)]
                    KY_c = H @ KY_perm @ H
                    null_vals.append(_cka_from_centered(KX_c, KY_c))
                    del KY_perm, KY_c
            null_dist[p_idx] = np.mean(null_vals)
        null_95 = float(np.percentile(null_dist, 95))
        print(f"    Null distribution: mean={np.mean(null_dist):.4f}, 95th={null_95:.4f}")

        # Step 2: Bootstrap power
        n_sig = 0
        observed_ckas = np.empty(n_power_iter, dtype=np.float32)
        for it in range(n_power_iter):
            idx = rng.choice(n, size=n, replace=True)
            obs_vals = []
            n_b = len(idx)
            Hb = np.eye(n_b, dtype=np.float32) - np.float32(1.0 / n_b)
            for model in cfg["models"]:
                for f1, f2 in FORMAT_PAIRS:
                    KX = raw_grams[model][f1][rep_layer_idx][np.ix_(idx, idx)]
                    KY = raw_grams[model][f2][rep_layer_idx][np.ix_(idx, idx)]
                    KX_c = Hb @ KX @ Hb
                    KY_c = Hb @ KY @ Hb
                    hsic_xy = np.sum(KX_c * KY_c)
                    hsic_xx = np.sum(KX_c * KX_c)
                    hsic_yy = np.sum(KY_c * KY_c)
                    denom = np.sqrt(hsic_xx * hsic_yy)
                    obs_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
                    del KX, KY, KX_c, KY_c
            obs_mean = float(np.mean(obs_vals))
            observed_ckas[it] = obs_mean
            if obs_mean > null_95:
                n_sig += 1
            if (it + 1) % 200 == 0:
                print(f"    iter {it+1}/{n_power_iter}, running power={n_sig/(it+1):.3f} (RSS={mem_gb():.2f} GB)")

        power = n_sig / n_power_iter
        results[config_name] = {
            "power": round(power, 4),
            "n_iter": n_power_iter,
            "n_null": n_null,
            "n_cells": cfg["n_cells"],
            "representative_layer": LAYERS[rep_layer_idx],
            "mean_observed_cka": round(float(np.mean(observed_ckas)), 6),
            "null_95th": round(float(null_95), 6),
            "null_mean": round(float(np.mean(null_dist)), 6),
        }
        print(f"    power = {power:.4f}, mean_obs_CKA = {np.mean(observed_ckas):.4f}")
        del centered, null_dist, observed_ckas
        gc.collect()

    elapsed = time.time() - t0
    print(f"  A1 Power Sim: {elapsed:.1f}s (RSS={mem_gb():.2f} GB)")
    results["elapsed_s"] = round(elapsed, 1)
    return results


# ── F. A2 Permutation Test ───────────────────────────────────────────
def run_a2_permutation(raw_grams, n_triples, n_perm):
    """Permutation test for CKA: per-model, shuffle sample indices."""
    print(f"\n=== F. A2 Permutation Test ({n_perm} permutations) === (RSS={mem_gb():.2f} GB)")
    t0 = time.time()

    n = n_triples
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    rng = np.random.RandomState(SEED + 2)
    results = {}

    for model in MODELS:
        # Pre-compute centered Grams
        centered = {}
        for fmt in FORMATS:
            centered[fmt] = {}
            for li in range(len(LAYERS)):
                centered[fmt][li] = H @ raw_grams[model][fmt][li] @ H

        # Pre-compute HSIC_xx (invariant under permutation of Y)
        hsic_xx_cache = {}
        for fmt in FORMATS:
            for li in range(len(LAYERS)):
                hsic_xx_cache[(fmt, li)] = float(np.sum(centered[fmt][li] ** 2))

        # Observed CKA
        obs_vals = []
        for li in range(len(LAYERS)):
            for f1, f2 in FORMAT_PAIRS:
                obs_vals.append(_cka_from_centered(centered[f1][li], centered[f2][li]))
        obs_mean = float(np.mean(obs_vals))

        # Null distribution
        null_means = np.empty(n_perm, dtype=np.float32)
        for p_idx in range(n_perm):
            perm = rng.permutation(n)
            null_vals = []
            for li in range(len(LAYERS)):
                for f1, f2 in FORMAT_PAIRS:
                    KX_c = centered[f1][li]
                    KY_perm = raw_grams[model][f2][li][np.ix_(perm, perm)]
                    KY_c = H @ KY_perm @ H
                    hsic_xy = np.sum(KX_c * KY_c)
                    hsic_yy = np.sum(KY_c * KY_c)
                    denom = np.sqrt(hsic_xx_cache[(f1, li)] * hsic_yy)
                    null_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
                    del KY_perm, KY_c
            null_means[p_idx] = np.mean(null_vals)

            if (p_idx + 1) % 500 == 0:
                print(f"    {model}: {p_idx+1}/{n_perm} permutations (RSS={mem_gb():.2f} GB)")

        p_value = float(np.mean(null_means >= obs_mean))
        results[model] = {
            "observed_cka_mean": round(float(obs_mean), 6),
            "null_mean": round(float(np.mean(null_means)), 6),
            "null_std": round(float(np.std(null_means)), 6),
            "null_95th": round(float(np.percentile(null_means, 95)), 6),
            "null_99th": round(float(np.percentile(null_means, 99)), 6),
            "p_value": round(p_value, 6),
            "n_permutations": n_perm,
        }
        print(f"  {model}: obs={obs_mean:.4f}, null_mean={np.mean(null_means):.4f}, p={p_value:.6f}")
        del centered, hsic_xx_cache, null_means
        gc.collect()

    elapsed = time.time() - t0
    print(f"  A2 Permutation: {elapsed:.1f}s (RSS={mem_gb():.2f} GB)")
    results["elapsed_s"] = round(elapsed, 1)
    return results


# ── Robustness: Procrustes ──────────────────────────────────────────
def procrustes_score(X, Y):
    """1 - Procrustes disparity (higher = more similar)."""
    from scipy.spatial import procrustes as sp_procrustes
    from sklearn.decomposition import PCA
    k = min(50, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, random_state=SEED)
    X_r = pca.fit_transform(X - X.mean(axis=0))
    Y_r = pca.fit_transform(Y - Y.mean(axis=0))
    _, _, disparity = sp_procrustes(X_r, Y_r)
    return round(1.0 - disparity, 6)


def pwcca_score(X, Y, k=50):
    """Projection-Weighted CCA (Morcos et al. 2018).

    SVD-truncates to rank min(n-1, k) to avoid n<<d CCA degeneracy,
    then weights canonical correlations by explained variance.
    Raw CCA gives trivial ρ=1.0 when n<<d (Ramsay et al. 2005);
    PWCCA resolves this via dimensionality reduction + variance weighting.
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA

    n, d = X.shape
    k_eff = min(k, n - 1, d)

    # SVD truncation via PCA (same k as Procrustes for comparability)
    pca_x = PCA(n_components=k_eff, random_state=SEED)
    pca_y = PCA(n_components=k_eff, random_state=SEED)
    X_r = pca_x.fit_transform(X - X.mean(axis=0))
    Y_r = pca_y.fit_transform(Y - Y.mean(axis=0))

    # CCA on reduced representations (well-conditioned: k_eff < n)
    n_comp = min(k_eff, n - 1)
    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(X_r, Y_r)
    X_cc, Y_cc = cca.transform(X_r, Y_r)

    # Canonical correlations
    corrs = np.array([np.corrcoef(X_cc[:, i], Y_cc[:, i])[0, 1]
                      for i in range(n_comp)])
    corrs = np.abs(np.nan_to_num(corrs, nan=0.0))

    # PWCCA weights: project CCA directions back and weight by
    # total absolute projection onto original neuron activations.
    # Simplified: use X-side PCA explained variance as proxy weights.
    var_x = pca_x.explained_variance_ratio_[:n_comp]
    weights = var_x / var_x.sum()

    return round(float(np.sum(weights * corrs)), 6)


def run_robustness(data):
    """Procrustes (PCA→k=50) + PWCCA (Morcos et al. 2018) as robustness checks."""
    print(f"\n=== Robustness: Procrustes + PWCCA === (RSS={mem_gb():.2f} GB)")
    t0 = time.time()

    proc_rows = []
    pwcca_rows = []
    for model in MODELS:
        for li, layer in enumerate(LAYERS):
            for f1, f2 in FORMAT_PAIRS:
                X = data[model][f1][:, li, :]
                Y = data[model][f2][:, li, :]
                proc_val = procrustes_score(X, Y)
                proc_rows.append({"model": model, "layer": layer,
                                  "pair": f"{f1}-{f2}", "procrustes": round(proc_val, 6)})
                pwcca_val = pwcca_score(X, Y)
                pwcca_rows.append({"model": model, "layer": layer,
                                   "pair": f"{f1}-{f2}", "pwcca": round(pwcca_val, 6)})

        for li, layer in enumerate(LAYERS):
            proc_vals = [r["procrustes"] for r in proc_rows
                         if r["model"] == model and r["layer"] == layer]
            pwcca_vals = [r["pwcca"] for r in pwcca_rows
                          if r["model"] == model and r["layer"] == layer]
            print(f"  {model} L{layer}: Procrustes={np.mean(proc_vals):.4f}  PWCCA={np.mean(pwcca_vals):.4f}")

    elapsed = time.time() - t0
    print(f"  Robustness: {elapsed:.1f}s (RSS={mem_gb():.2f} GB)")
    return {"procrustes": proc_rows, "pwcca": pwcca_rows, "elapsed_s": round(elapsed, 1)}


# ── Save Outputs (CSVs, Figures, Report) ─────────────────────────────
def save_outputs(ckpt_dir, out_dir, n_bootstrap, n_power_iter, n_perm):
    """Assemble final report from checkpoint JSONs + generate CSVs/figures."""
    print(f"\n=== Saving outputs === (RSS={mem_gb():.2f} GB)")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all checkpoints
    a = _load_ckpt(ckpt_dir / "a_load.json")
    b = _load_ckpt(ckpt_dir / "b_probe_fit.json")
    c = _load_ckpt(ckpt_dir / "c_cka.json")
    d = _load_ckpt(ckpt_dir / "d_bootstrap.json")
    e = _load_ckpt(ckpt_dir / "e_a1_power.json")
    f = _load_ckpt(ckpt_dir / "f_a2_permutation.json")

    # Robustness checkpoint (optional)
    rob_path = ckpt_dir / "g_robustness.json"
    rob = _load_ckpt(rob_path) if rob_path.exists() else None

    # ── Combined report JSON ──
    report = {"a_load": a, "b_probe_fit": b, "c_cka": c,
              "d_bootstrap": d, "e_a1_power": e, "f_a2_permutation": f}
    if rob:
        report["g_robustness"] = rob
    _save_ckpt(out_dir / "stage_b_analysis_report.json", report)
    print("  Saved stage_b_analysis_report.json")

    # ── CSVs ──
    # 1. probe_format_accuracy.csv
    with open(out_dir / "probe_format_accuracy.csv", "w") as fh:
        fh.write("model,layer,accuracy\n")
        for r in b["results"]:
            fh.write(f"{r['model']},{r['layer']},{r['accuracy']}\n")
    print(f"  Saved probe_format_accuracy.csv ({len(b['results'])} rows)")

    # 2. cka_per_layer_per_pair.csv
    with open(out_dir / "cka_per_layer_per_pair.csv", "w") as fh:
        fh.write("model,layer,format_pair,cka\n")
        for r in sorted(c["results"], key=lambda x: (x["model"], x["layer"], x["pair"])):
            fh.write(f"{r['model']},{r['layer']},{r['pair']},{r['cka']}\n")
    print(f"  Saved cka_per_layer_per_pair.csv ({len(c['results'])} rows)")

    # 3. cka_bootstrap_ci.csv
    with open(out_dir / "cka_bootstrap_ci.csv", "w") as fh:
        fh.write("model,layer,mean,ci_low,ci_high,ci_width\n")
        for r in sorted(d["results"], key=lambda x: (x["model"], x["layer"])):
            fh.write(f"{r['model']},{r['layer']},{r['mean']},{r['ci_low']},{r['ci_high']},{r['ci_width']}\n")
    print(f"  Saved cka_bootstrap_ci.csv ({len(d['results'])} rows)")

    # 4. a1_power_sim.json
    _save_ckpt(out_dir / "a1_power_sim.json", e)
    print("  Saved a1_power_sim.json")

    # 5. a2_permutation.json
    _save_ckpt(out_dir / "a2_permutation.json", f)
    print("  Saved a2_permutation.json")

    # ── Figures ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _generate_figures(b, c, fig_dir, plt)
    except ImportError:
        print("  WARNING: matplotlib not available, skipping figures")

    # ── Stats report ──
    report_md = _generate_stats_report(b, c, d, e, f, rob, n_bootstrap, n_power_iter, n_perm)
    with open(out_dir / "stats_report.md", "w") as fh:
        fh.write(report_md)
    print("  Saved stats_report.md")


def _generate_figures(b, c, fig_dir, plt):
    """Generate PDF figures from checkpoint data."""
    # Build lookup dicts from checkpoint rows
    probe_acc = {}
    for r in b["results"]:
        probe_acc.setdefault(r["model"], {})[r["layer"]] = r["accuracy"]

    cka_lookup = {}
    for r in c["results"]:
        cka_lookup[(r["model"], r["layer"], r["pair"])] = r["cka"]

    # Figure 1: probe accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        accs = [probe_acc[model][l] for l in LAYERS]
        ax.plot(LAYERS, accs, "o-", label=model, linewidth=2)
    ax.axhline(1/3, color="gray", linestyle="--", label="chance (1/3)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Format Classification Accuracy (5-fold CV)")
    ax.set_title("Format Probe Accuracy by Layer")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "probe_accuracy_curve.pdf", dpi=150)
    plt.close(fig)
    print("  Saved figures/probe_accuracy_curve.pdf")

    # Figure 2: CKA layer curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        means = []
        for layer in LAYERS:
            vals = [cka_lookup[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            means.append(np.mean(vals))
        ax.plot(LAYERS, means, "o-", label=model, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean CKA (across format pairs)")
    ax.set_title("Cross-Format CKA by Layer")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "cka_layer_curve.pdf", dpi=150)
    plt.close(fig)
    print("  Saved figures/cka_layer_curve.pdf")

    # Figure 3: CKA heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, model in zip(axes, MODELS):
        mat = np.zeros((len(LAYERS), len(FORMAT_PAIRS)))
        for li, layer in enumerate(LAYERS):
            for pi, (f1, f2) in enumerate(FORMAT_PAIRS):
                mat[li, pi] = cka_lookup[(model, layer, f"{f1}-{f2}")]
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels(LAYERS)
        ax.set_xticks(range(len(FORMAT_PAIRS)))
        ax.set_xticklabels([f"{a}-{b}" for a, b in FORMAT_PAIRS], rotation=45)
        ax.set_ylabel("Layer")
        ax.set_title(model)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        color="white" if mat[i,j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="CKA")
    fig.suptitle("Cross-Format CKA Heatmap")
    fig.tight_layout()
    fig.savefig(fig_dir / "cka_heatmap.pdf", dpi=150)
    plt.close(fig)
    print("  Saved figures/cka_heatmap.pdf")


def _generate_stats_report(b, c, d, e, f, rob, n_bootstrap, n_power_iter, n_perm):
    """Generate comprehensive stats report markdown."""
    # Build lookups
    probe_acc = {}
    for r in b["results"]:
        probe_acc.setdefault(r["model"], {})[r["layer"]] = r["accuracy"]

    cka_lookup = {}
    for r in c["results"]:
        cka_lookup[(r["model"], r["layer"], r["pair"])] = r["cka"]

    boot_lookup = {}
    for r in d["results"]:
        boot_lookup[(r["model"], r["layer"])] = r

    lines = []
    lines.append("# Stage B Analysis v1 — Statistical Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"Layers: {LAYERS}, Models: {MODELS}")
    lines.append(f"Bootstrap: {n_bootstrap}, Permutations: {n_perm}, Power iterations: {n_power_iter}")

    lines.append("\n## Pre-Registration Compliance")
    lines.append("- [x] Main metric: CKA (linear kernel)")
    lines.append("- [x] Procrustes + PWCCA as robustness checks")
    lines.append("- [x] A1 power sim: N=12 AND N=18")
    lines.append("- [x] A2 permutation test per model")
    lines.append("- [x] NO quantitative predictivity claim")
    lines.append("- [x] NO decoupling claim")
    lines.append("- [x] N=18 VisCoder2 as 'exploratory view with known memorization floor'")

    # Probe accuracy table
    lines.append("\n## B. Format Probe Accuracy (5-fold CV)")
    lines.append("\n| Model | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
    lines.append("|-------|" + "|".join("------" for _ in LAYERS) + "|")
    for model in MODELS:
        vals = " | ".join(f"{probe_acc[model][l]:.4f}" for l in LAYERS)
        lines.append(f"| {model} | {vals} |")

    # CKA table
    lines.append("\n## C. CKA (Mean Across Format Pairs)")
    lines.append("\n| Model | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
    lines.append("|-------|" + "|".join("------" for _ in LAYERS) + "|")
    for model in MODELS:
        vals = []
        for layer in LAYERS:
            pair_vals = [cka_lookup[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            vals.append(f"{np.mean(pair_vals):.4f}")
        lines.append(f"| {model} | {' | '.join(vals)} |")

    # CKA per pair
    lines.append("\n### CKA Per Format Pair")
    for model in MODELS:
        lines.append(f"\n**{model}**:")
        lines.append("| Layer | svg-tikz | svg-asy | tikz-asy |")
        lines.append("|-------|----------|---------|----------|")
        for layer in LAYERS:
            v1 = cka_lookup[(model, layer, "svg-tikz")]
            v2 = cka_lookup[(model, layer, "svg-asy")]
            v3 = cka_lookup[(model, layer, "tikz-asy")]
            lines.append(f"| {layer} | {v1:.4f} | {v2:.4f} | {v3:.4f} |")

    # Bootstrap CI
    lines.append("\n## D. Bootstrap 95% CI")
    lines.append("\n| Model | Layer | Mean | CI Low | CI High | Width |")
    lines.append("|-------|-------|------|--------|---------|-------|")
    for model in MODELS:
        for layer in LAYERS:
            v = boot_lookup[(model, layer)]
            lines.append(f"| {model} | {layer} | {v['mean']:.4f} | {v['ci_low']:.4f} | {v['ci_high']:.4f} | {v['ci_width']:.4f} |")

    # A1 Power
    lines.append("\n## E. A1 Power Simulation")
    for name in ["N12", "N18"]:
        if name in e:
            v = e[name]
            lines.append(f"\n**{name}** (cells={v['n_cells']}, iter={v['n_iter']}):")
            lines.append(f"- Power: **{v['power']:.4f}**")
            lines.append(f"- Mean observed CKA: {v['mean_observed_cka']:.4f}")
            lines.append(f"- Null mean: {v['null_mean']:.4f}")

    # A2 Permutation
    lines.append("\n## F. A2 Permutation Test")
    lines.append("\n| Model | Observed CKA | Null Mean | Null Std | p-value |")
    lines.append("|-------|-------------|-----------|----------|---------|")
    for model in MODELS:
        if model in f:
            v = f[model]
            lines.append(f"| {model} | {v['observed_cka_mean']:.4f} | {v['null_mean']:.4f} | {v['null_std']:.4f} | {v['p_value']:.6f} |")

    # Robustness
    if rob:
        proc_lookup = {}
        for r in rob["procrustes"]:
            proc_lookup[(r["model"], r["layer"], r["pair"])] = r["procrustes"]

        lines.append("\n## Robustness: Procrustes + PWCCA")
        lines.append("\n### Mean Procrustes per Model x Layer")
        lines.append("| Model | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
        lines.append("|-------|" + "|".join("------" for _ in LAYERS) + "|")
        for model in MODELS:
            vals = []
            for layer in LAYERS:
                pair_vals = [proc_lookup.get((model, layer, f"{f1}-{f2}"), 0.0)
                             for f1, f2 in FORMAT_PAIRS]
                vals.append(f"{np.mean(pair_vals):.4f}")
            lines.append(f"| {model} | {' | '.join(vals)} |")

        # PWCCA table
        pwcca_lookup = {}
        if "pwcca" in rob:
            for r in rob["pwcca"]:
                pwcca_lookup[(r["model"], r["layer"], r["pair"])] = r["pwcca"]

            lines.append("\n### Mean PWCCA per Model x Layer")
            lines.append("*PWCCA (Morcos et al. 2018): SVD-truncated CCA weighted by explained variance.*")
            lines.append("*Raw CCA degenerates to ρ=1.0 when n<<d (Ramsay et al. 2005); PWCCA avoids this.*")
            lines.append("\n| Model | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
            lines.append("|-------|" + "|".join("------" for _ in LAYERS) + "|")
            for model in MODELS:
                vals = []
                for layer in LAYERS:
                    pair_vals = [pwcca_lookup.get((model, layer, f"{f1}-{f2}"), 0.0)
                                 for f1, f2 in FORMAT_PAIRS]
                    vals.append(f"{np.mean(pair_vals):.4f}")
                lines.append(f"| {model} | {' | '.join(vals)} |")

    # Sanity checks
    lines.append("\n## Sanity Checks")
    for model in MODELS:
        accs = [probe_acc[model][l] for l in LAYERS]
        monotonic = all(accs[i] <= accs[i+1] for i in range(len(accs)-1))
        lines.append(f"- {model} probe monotonic increasing: {'YES' if monotonic else 'NO'} (min={min(accs):.4f}, max={max(accs):.4f})")

    ci_widths = [boot_lookup[(m, l)]["ci_width"] for m in MODELS for l in LAYERS]
    lines.append(f"- Bootstrap CI widths: min={min(ci_widths):.4f}, max={max(ci_widths):.4f}, mean={np.mean(ci_widths):.4f}")
    lines.append(f"- All CI widths < 0.1: {'YES' if max(ci_widths) < 0.1 else 'NO'}")

    for model in MODELS:
        if model in f:
            p = f[model]["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            lines.append(f"- {model} A2 p-value: {p:.6f} ({sig})")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage B Analysis v1")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test with fake data (N=12, reduced iterations)")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignore existing checkpoints and rerun all modules")
    args = parser.parse_args()

    np.random.seed(SEED)
    t_total = time.time()

    # ── Config ──
    if args.smoke:
        n_triples = 12
        n_bootstrap = 10
        n_power_iter = 10
        n_perm = 50
        cache_dir = PROJECT_ROOT / "tests" / "fake_hidden_states"
        ckpt_dir = PROJECT_ROOT / "artifacts" / "stage_b_analysis_v1_checkpoints_smoke"
        out_dir = PROJECT_ROOT / "artifacts" / "stage_b_analysis_v1_smoke"
    else:
        n_triples = 252
        n_bootstrap = 1000
        n_power_iter = 1000
        n_perm = 5000
        cache_dir = Path("/root/autodl-tmp/cache/hidden_states")
        ckpt_dir = PROJECT_ROOT / "artifacts" / "stage_b_analysis_v1_checkpoints"
        out_dir = PROJECT_ROOT / "artifacts" / "stage_b_analysis_v1"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Stage B Analysis v1 {'(SMOKE)' if args.smoke else ''}")
    print(f"  n_triples={n_triples}, bootstrap={n_bootstrap}, power={n_power_iter}, perm={n_perm}")
    print(f"  checkpoints: {ckpt_dir}")
    print(f"  RSS={mem_gb():.2f} GB")
    print("=" * 60)

    # Generate fake data for smoke
    if args.smoke:
        generate_smoke_data(cache_dir, n_triples)

    # ── Determine which modules need running ──
    module_names = ["a_load", "b_probe_fit", "c_cka", "d_bootstrap",
                    "e_a1_power", "f_a2_permutation", "g_robustness"]
    need_run = {}
    for name in module_names:
        ckpt_path = ckpt_dir / f"{name}.json"
        need_run[name] = args.force_rerun or not ckpt_path.exists()
        if not need_run[name]:
            print(f"  skipping module {name} (checkpoint exists)")

    need_data = any(need_run[m] for m in module_names)
    need_raw_grams = any(need_run[m] for m in ["c_cka", "d_bootstrap", "e_a1_power", "f_a2_permutation"])

    # ── Load data if needed ──
    data = None
    raw_grams = None

    if need_data:
        data, detected_dtype, load_time = load_hidden_states(cache_dir, n_triples)

        # Save A checkpoint
        if need_run["a_load"]:
            sample_shape = list(data[MODELS[0]][FORMATS[0]].shape)
            _save_ckpt(ckpt_dir / "a_load.json", {
                "n_triples": n_triples,
                "layers": LAYERS,
                "models": MODELS,
                "formats": FORMATS,
                "pt_dtype": detected_dtype,
                "loaded_dtype": "float32",
                "sample_shape": sample_shape,
                "load_time_s": round(load_time, 1),
            })
            print(f"  [checkpoint] a_load.json saved")

        # Compute shared raw Gram matrices
        if need_raw_grams:
            raw_grams = compute_raw_grams(data)

    # ── B. Probe Fit ──
    if need_run["b_probe_fit"]:
        b_result = run_probe_fit(data, n_triples)
        _save_ckpt(ckpt_dir / "b_probe_fit.json", b_result)
        print(f"  [checkpoint] b_probe_fit.json saved")
        del b_result
        gc.collect()

    # ── G. Robustness (run before freeing data) ──
    if need_run["g_robustness"]:
        rob_result = run_robustness(data)
        _save_ckpt(ckpt_dir / "g_robustness.json", rob_result)
        print(f"  [checkpoint] g_robustness.json saved")
        del rob_result
        gc.collect()

    # ── Free data (no longer needed; C-F use raw_grams) ──
    if data is not None:
        del data
        gc.collect()
        print(f"  [memory] data freed (RSS={mem_gb():.2f} GB)")

    # ── C. CKA ──
    if need_run["c_cka"]:
        c_result = run_cka(raw_grams)
        _save_ckpt(ckpt_dir / "c_cka.json", c_result)
        print(f"  [checkpoint] c_cka.json saved")
        del c_result
        gc.collect()

    # ── D. Bootstrap CI ──
    if need_run["d_bootstrap"]:
        d_result = run_bootstrap_ci(raw_grams, n_triples, n_bootstrap)
        _save_ckpt(ckpt_dir / "d_bootstrap.json", d_result)
        print(f"  [checkpoint] d_bootstrap.json saved")
        del d_result
        gc.collect()

    # ── E. A1 Power Sim ──
    if need_run["e_a1_power"]:
        e_result = run_a1_power_sim(raw_grams, n_triples, n_power_iter)
        _save_ckpt(ckpt_dir / "e_a1_power.json", e_result)
        print(f"  [checkpoint] e_a1_power.json saved")
        del e_result
        gc.collect()

    # ── F. A2 Permutation ──
    if need_run["f_a2_permutation"]:
        f_result = run_a2_permutation(raw_grams, n_triples, n_perm)
        _save_ckpt(ckpt_dir / "f_a2_permutation.json", f_result)
        print(f"  [checkpoint] f_a2_permutation.json saved")
        del f_result
        gc.collect()

    # ── Free raw_grams ──
    if raw_grams is not None:
        del raw_grams
        gc.collect()
        print(f"  [memory] raw_grams freed (RSS={mem_gb():.2f} GB)")

    # ── Assemble report from checkpoints ──
    save_outputs(ckpt_dir, out_dir, n_bootstrap, n_power_iter, n_perm)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Total elapsed: {elapsed:.1f}s")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Final RSS: {mem_gb():.2f} GB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
