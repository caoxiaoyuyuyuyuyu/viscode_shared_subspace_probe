#!/usr/bin/env python
"""Stage B Analysis v1: Probe fit / CKA / Bootstrap CI / A1 power sim / A2 permutation.

Pre-reg v3.3 + D026 amendment compliant:
  - NO quantitative predictivity claim
  - NO decoupling claim
  - N=18 contains VisCoder2 as "exploratory view with known memorization floor"
  - A1 power sim: N=12 AND N=18
  - Main metric: CKA (ρ), CCA/procrustes as robustness

Usage:
  export PATH=/root/miniconda3/bin:$PATH
  cd /root/autodl-tmp/viscode_shared_subspace_probe
  python scripts/stage_b_analysis.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.stdout.reconfigure(line_buffering=True)

# ── Config ─────────────────────────────────────────────────────────────
LAYERS = [4, 8, 12, 16, 20, 24, 28]
MODELS = ["coder", "viscoder2", "qwen25"]
FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]

CACHE_DIR = Path("/root/autodl-tmp/cache/hidden_states")
TRIPLES_PATH = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/stage_a/sbert_triples.json")
OUT_DIR = Path("/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/stage_b_analysis_v1")
FIG_DIR = OUT_DIR / "figures"

N_BOOTSTRAP = 1000
N_PERMUTATION = 5000
N_POWER_ITER = 1000
SEED = 42

np.random.seed(SEED)


# ── A. Data Loader ─────────────────────────────────────────────────────
def load_hidden_states():
    """Load all hidden states into data[model][format] = np.array(252, 7, 3584)."""
    print("=== A. Loading hidden states ===")
    t0 = time.time()

    with open(TRIPLES_PATH) as f:
        triples = json.load(f)
    n_triples = len(triples)
    print(f"  sbert_triples.json: {n_triples} triples")
    assert n_triples == 252, f"Expected 252 triples, got {n_triples}"

    data = {}
    for model in MODELS:
        data[model] = {}
        for fmt in FORMATS:
            tensors = []
            d = CACHE_DIR / model / fmt
            for i in range(n_triples):
                pt_path = d / f"{i}.pt"
                t = torch.load(pt_path, map_location="cpu", weights_only=True)
                assert t.shape == (7, 3584), f"{pt_path}: shape {t.shape}"
                tensors.append(t.numpy())
            arr = np.stack(tensors, axis=0)  # (252, 7, 3584)
            data[model][fmt] = arr
            print(f"  {model}/{fmt}: {arr.shape}")

    elapsed = time.time() - t0
    print(f"  Data loading: {elapsed:.1f}s")
    return data, triples


# ── B. Probe Fit (Format Classifier) ──────────────────────────────────
def run_probe_fit(data):
    """Per-model, per-layer format classification (5-fold CV LogReg)."""
    print("\n=== B. Probe Fit (Format Classifier) ===")
    t0 = time.time()

    results = {}  # model -> layer_idx -> accuracy
    for model in MODELS:
        results[model] = {}
        for li, layer in enumerate(LAYERS):
            # Stack all 3 formats: (756, 3584)
            X = np.concatenate([data[model][fmt][:, li, :] for fmt in FORMATS], axis=0)
            y = np.concatenate([np.full(252, i) for i, _ in enumerate(FORMATS)])

            clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial", random_state=SEED)
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            acc = scores.mean()
            results[model][layer] = round(float(acc), 4)
            print(f"  {model} layer {layer}: acc={acc:.4f} (±{scores.std():.4f})")

    elapsed = time.time() - t0
    print(f"  Probe fit: {elapsed:.1f}s")
    return results


# ── C. CKA (Main Metric ρ) ────────────────────────────────────────────
def linear_cka(X, Y):
    """Centered Kernel Alignment with linear kernel.
    X, Y: (n, d) arrays.
    """
    n = X.shape[0]
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Gram matrices
    KX = X @ X.T
    KY = Y @ Y.T
    # HSIC
    hsic_xy = np.trace(KX @ KY) / ((n - 1) ** 2)
    hsic_xx = np.trace(KX @ KX) / ((n - 1) ** 2)
    hsic_yy = np.trace(KY @ KY) / ((n - 1) ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def run_cka(data):
    """CKA for each (model, layer, format_pair)."""
    print("\n=== C. CKA Computation ===")
    t0 = time.time()

    results = {}  # (model, layer, pair_str) -> cka_value
    for model in MODELS:
        for li, layer in enumerate(LAYERS):
            for f1, f2 in FORMAT_PAIRS:
                X = data[model][f1][:, li, :]  # (252, 3584)
                Y = data[model][f2][:, li, :]
                cka_val = linear_cka(X, Y)
                key = (model, layer, f"{f1}-{f2}")
                results[key] = round(cka_val, 6)

            # Sanity: self-CKA should be ~1.0
            X_svg = data[model]["svg"][:, li, :]
            self_cka = linear_cka(X_svg, X_svg)
            if abs(self_cka - 1.0) > 0.01:
                print(f"  WARNING: self-CKA {model}/svg layer {layer} = {self_cka:.4f}")

    # Print per-layer mean CKA
    for model in MODELS:
        print(f"  {model}:")
        for li, layer in enumerate(LAYERS):
            vals = [results[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            print(f"    layer {layer}: mean CKA = {np.mean(vals):.4f} ({vals})")

    elapsed = time.time() - t0
    print(f"  CKA: {elapsed:.1f}s")
    return results


# ── D. Bootstrap CI ───────────────────────────────────────────────────
def run_bootstrap_ci(data):
    """Bootstrap 95% CI for CKA per (model, layer)."""
    print(f"\n=== D. Bootstrap CI ({N_BOOTSTRAP} samples) ===")
    t0 = time.time()

    rng = np.random.RandomState(SEED)
    results = {}  # (model, layer) -> {mean, ci_low, ci_high, per_pair: ...}

    for model in MODELS:
        for li, layer in enumerate(LAYERS):
            boot_means = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.choice(252, size=252, replace=True)
                pair_ckas = []
                for f1, f2 in FORMAT_PAIRS:
                    X = data[model][f1][idx, li, :]
                    Y = data[model][f2][idx, li, :]
                    pair_ckas.append(linear_cka(X, Y))
                boot_means.append(np.mean(pair_ckas))

            boot_means = np.array(boot_means)
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            mean_val = np.mean(boot_means)
            results[(model, layer)] = {
                "mean": round(float(mean_val), 6),
                "ci_low": round(float(ci_low), 6),
                "ci_high": round(float(ci_high), 6),
                "ci_width": round(float(ci_high - ci_low), 6),
            }
            print(f"  {model} layer {layer}: {mean_val:.4f} [{ci_low:.4f}, {ci_high:.4f}] (w={ci_high-ci_low:.4f})")

    elapsed = time.time() - t0
    print(f"  Bootstrap CI: {elapsed:.1f}s")
    return results


# ── E. A1 Power Simulation ────────────────────────────────────────────
def run_a1_power_sim(data):
    """Monte Carlo power simulation for CKA detection at N=12 and N=18.

    For each iteration:
    - Sample N triples from the 252 real triples
    - Compute observed CKA (mean across format pairs and layers)
    - Generate null CKA by permuting format labels
    - Record if observed > 95th percentile of null
    """
    print(f"\n=== E. A1 Power Simulation ({N_POWER_ITER} iterations) ===")
    t0 = time.time()

    rng = np.random.RandomState(SEED + 1)
    results = {}

    # Models for N=12: coder + qwen25 only (no viscoder2)
    # Models for N=18: all 3
    configs = {
        "N12": {"n_triples": 252, "models": ["coder", "qwen25"], "n_cells": 12},
        "N18": {"n_triples": 252, "models": MODELS, "n_cells": 18},
    }

    for config_name, cfg in configs.items():
        print(f"\n  {config_name} (models={cfg['models']}, cells={cfg['n_cells']}):")
        n_sig = 0
        observed_ckas = []
        null_p_values = []

        for it in range(N_POWER_ITER):
            # Sample a subset of triples
            n_sample = min(cfg["n_triples"], 252)
            idx = rng.choice(252, size=n_sample, replace=True)

            # Compute observed CKA across all models×layers×pairs in this config
            obs_cka_vals = []
            for model in cfg["models"]:
                for li in range(len(LAYERS)):
                    for f1, f2 in FORMAT_PAIRS:
                        X = data[model][f1][idx, li, :]
                        Y = data[model][f2][idx, li, :]
                        obs_cka_vals.append(linear_cka(X, Y))
            obs_mean = np.mean(obs_cka_vals)
            observed_ckas.append(obs_mean)

            # Null: permute format labels within each triple
            null_ckas = []
            for _ in range(100):  # 100 null draws per iteration
                perm_idx = rng.permutation(n_sample)
                null_vals = []
                for model in cfg["models"]:
                    for li in range(len(LAYERS)):
                        for f1, f2 in FORMAT_PAIRS:
                            X = data[model][f1][idx, li, :]
                            Y = data[model][f2][perm_idx, li, :]
                            null_vals.append(linear_cka(X, Y))
                null_ckas.append(np.mean(null_vals))

            null_95 = np.percentile(null_ckas, 95)
            if obs_mean > null_95:
                n_sig += 1

            p_val = np.mean(np.array(null_ckas) >= obs_mean)
            null_p_values.append(p_val)

        power = n_sig / N_POWER_ITER
        results[config_name] = {
            "power": round(power, 4),
            "n_iter": N_POWER_ITER,
            "n_cells": cfg["n_cells"],
            "mean_observed_cka": round(float(np.mean(observed_ckas)), 6),
            "mean_null_p_value": round(float(np.mean(null_p_values)), 6),
        }
        print(f"    power = {power:.4f}, mean_obs_CKA = {np.mean(observed_ckas):.4f}, mean_null_p = {np.mean(null_p_values):.4f}")

    elapsed = time.time() - t0
    print(f"  A1 Power Sim: {elapsed:.1f}s")
    return results


# ── F. A2 Permutation Test ────────────────────────────────────────────
def run_a2_permutation(data):
    """Permutation test for CKA: per-model, shuffle format labels 5000 times."""
    print(f"\n=== F. A2 Permutation Test ({N_PERMUTATION} permutations) ===")
    t0 = time.time()

    rng = np.random.RandomState(SEED + 2)
    results = {}

    for model in MODELS:
        # Observed CKA: mean across layers and format pairs
        obs_vals = []
        for li in range(len(LAYERS)):
            for f1, f2 in FORMAT_PAIRS:
                X = data[model][f1][:, li, :]
                Y = data[model][f2][:, li, :]
                obs_vals.append(linear_cka(X, Y))
        obs_mean = np.mean(obs_vals)

        # Null distribution: permute sample indices for one format
        null_means = []
        for _ in range(N_PERMUTATION):
            perm = rng.permutation(252)
            null_vals = []
            for li in range(len(LAYERS)):
                for f1, f2 in FORMAT_PAIRS:
                    X = data[model][f1][:, li, :]
                    Y = data[model][f2][perm, li, :]  # permute Y samples
                    null_vals.append(linear_cka(X, Y))
            null_means.append(np.mean(null_vals))

        null_means = np.array(null_means)
        p_value = float(np.mean(null_means >= obs_mean))

        results[model] = {
            "observed_cka_mean": round(float(obs_mean), 6),
            "null_mean": round(float(np.mean(null_means)), 6),
            "null_std": round(float(np.std(null_means)), 6),
            "null_95th": round(float(np.percentile(null_means, 95)), 6),
            "null_99th": round(float(np.percentile(null_means, 99)), 6),
            "p_value": round(p_value, 6),
            "n_permutations": N_PERMUTATION,
        }
        print(f"  {model}: obs={obs_mean:.4f}, null_mean={np.mean(null_means):.4f}, p={p_value:.6f}")

    elapsed = time.time() - t0
    print(f"  A2 Permutation: {elapsed:.1f}s")
    return results


# ── Robustness: CCA + Procrustes ──────────────────────────────────────
def cca_score(X, Y, n_components=10):
    """Mean canonical correlation (top-k)."""
    from sklearn.cross_decomposition import CCA
    k = min(n_components, X.shape[0] - 1, X.shape[1], Y.shape[1])
    cca = CCA(n_components=k, max_iter=1000)
    try:
        X_c, Y_c = cca.fit_transform(X, Y)
        corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(k)]
        return float(np.mean(corrs))
    except Exception:
        return float("nan")


def procrustes_score(X, Y):
    """1 - Procrustes disparity (higher = more similar)."""
    from scipy.spatial import procrustes as sp_procrustes
    # Reduce dims for stability
    from sklearn.decomposition import PCA
    k = min(50, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, random_state=SEED)
    X_r = pca.fit_transform(X - X.mean(axis=0))
    Y_r = pca.fit_transform(Y - Y.mean(axis=0))
    _, _, disparity = sp_procrustes(X_r, Y_r)
    return round(1.0 - disparity, 6)


def run_robustness(data):
    """CCA and Procrustes as robustness checks."""
    print("\n=== Robustness: CCA + Procrustes ===")
    t0 = time.time()

    cca_results = {}
    proc_results = {}
    for model in MODELS:
        for li, layer in enumerate(LAYERS):
            for f1, f2 in FORMAT_PAIRS:
                X = data[model][f1][:, li, :]
                Y = data[model][f2][:, li, :]
                cca_val = cca_score(X, Y)
                proc_val = procrustes_score(X, Y)
                cca_results[(model, layer, f"{f1}-{f2}")] = round(cca_val, 6)
                proc_results[(model, layer, f"{f1}-{f2}")] = round(proc_val, 6)

        # Print summary per model
        for li, layer in enumerate(LAYERS):
            cca_vals = [cca_results[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            proc_vals = [proc_results[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            print(f"  {model} L{layer}: CCA={np.nanmean(cca_vals):.4f}, Procrustes={np.mean(proc_vals):.4f}")

    elapsed = time.time() - t0
    print(f"  Robustness: {elapsed:.1f}s")
    return cca_results, proc_results


# ── Save outputs ──────────────────────────────────────────────────────
def save_outputs(probe_results, cka_results, bootstrap_results, power_results,
                 perm_results, cca_results, proc_results):
    """Save all CSVs, JSONs, figures, and stats report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. probe_format_accuracy.csv
    with open(OUT_DIR / "probe_format_accuracy.csv", "w") as f:
        f.write("model,layer,accuracy\n")
        for model in MODELS:
            for layer in LAYERS:
                f.write(f"{model},{layer},{probe_results[model][layer]}\n")
    print(f"  Saved probe_format_accuracy.csv ({len(MODELS)*len(LAYERS)} rows)")

    # 2. cka_per_layer_per_pair.csv
    with open(OUT_DIR / "cka_per_layer_per_pair.csv", "w") as f:
        f.write("model,layer,format_pair,cka\n")
        for (model, layer, pair), val in sorted(cka_results.items()):
            f.write(f"{model},{layer},{pair},{val}\n")
    print(f"  Saved cka_per_layer_per_pair.csv ({len(cka_results)} rows)")

    # 3. cka_bootstrap_ci.csv
    with open(OUT_DIR / "cka_bootstrap_ci.csv", "w") as f:
        f.write("model,layer,mean,ci_low,ci_high,ci_width\n")
        for (model, layer), v in sorted(bootstrap_results.items()):
            f.write(f"{model},{layer},{v['mean']},{v['ci_low']},{v['ci_high']},{v['ci_width']}\n")
    print(f"  Saved cka_bootstrap_ci.csv ({len(bootstrap_results)} rows)")

    # 4. a1_power_sim.json
    with open(OUT_DIR / "a1_power_sim.json", "w") as f:
        json.dump(power_results, f, indent=2)
    print("  Saved a1_power_sim.json")

    # 5. a2_permutation.json
    with open(OUT_DIR / "a2_permutation.json", "w") as f:
        json.dump(perm_results, f, indent=2)
    print("  Saved a2_permutation.json")

    # 6. Figure: probe_accuracy_curve.pdf
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        accs = [probe_results[model][l] for l in LAYERS]
        ax.plot(LAYERS, accs, "o-", label=model, linewidth=2)
    ax.axhline(1/3, color="gray", linestyle="--", label="chance (1/3)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Format Classification Accuracy (5-fold CV)")
    ax.set_title("Format Probe Accuracy by Layer")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "probe_accuracy_curve.pdf", dpi=150)
    plt.close(fig)
    print("  Saved figures/probe_accuracy_curve.pdf")

    # 7. Figure: cka_layer_curve.pdf
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        means = []
        for li, layer in enumerate(LAYERS):
            vals = [cka_results[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            means.append(np.mean(vals))
        ax.plot(LAYERS, means, "o-", label=model, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean CKA (across format pairs)")
    ax.set_title("Cross-Format CKA by Layer")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cka_layer_curve.pdf", dpi=150)
    plt.close(fig)
    print("  Saved figures/cka_layer_curve.pdf")

    # 8. Figure: cka_heatmap.pdf (one subplot per model)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, model in zip(axes, MODELS):
        mat = np.zeros((len(LAYERS), len(FORMAT_PAIRS)))
        for li, layer in enumerate(LAYERS):
            for pi, (f1, f2) in enumerate(FORMAT_PAIRS):
                mat[li, pi] = cka_results[(model, layer, f"{f1}-{f2}")]
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
    fig.savefig(FIG_DIR / "cka_heatmap.pdf", dpi=150)
    plt.close(fig)
    print("  Saved figures/cka_heatmap.pdf")

    # 9. stats_report.md
    report = generate_stats_report(probe_results, cka_results, bootstrap_results,
                                   power_results, perm_results, cca_results, proc_results)
    with open(OUT_DIR / "stats_report.md", "w") as f:
        f.write(report)
    print("  Saved stats_report.md")


def generate_stats_report(probe_results, cka_results, bootstrap_results,
                          power_results, perm_results, cca_results, proc_results):
    """Generate comprehensive stats report."""
    lines = []
    lines.append("# Stage B Analysis v1 — Statistical Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"N_triples: 252, Layers: {LAYERS}, Models: {MODELS}")
    lines.append(f"Bootstrap: {N_BOOTSTRAP}, Permutations: {N_PERMUTATION}, Power iterations: {N_POWER_ITER}")

    # Pre-reg compliance
    lines.append("\n## Pre-Registration Compliance")
    lines.append("- [x] Main metric: CKA (linear kernel)")
    lines.append("- [x] CCA + Procrustes as robustness checks")
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
        vals = " | ".join(f"{probe_results[model][l]:.4f}" for l in LAYERS)
        lines.append(f"| {model} | {vals} |")

    # CKA table
    lines.append("\n## C. CKA (Mean Across Format Pairs)")
    lines.append("\n| Model | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
    lines.append("|-------|" + "|".join("------" for _ in LAYERS) + "|")
    for model in MODELS:
        vals = []
        for layer in LAYERS:
            pair_vals = [cka_results[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            vals.append(f"{np.mean(pair_vals):.4f}")
        lines.append(f"| {model} | {' | '.join(vals)} |")

    # CKA per pair detail
    lines.append("\n### CKA Per Format Pair")
    for model in MODELS:
        lines.append(f"\n**{model}**:")
        lines.append("| Layer | svg-tikz | svg-asy | tikz-asy |")
        lines.append("|-------|----------|---------|----------|")
        for layer in LAYERS:
            v1 = cka_results[(model, layer, "svg-tikz")]
            v2 = cka_results[(model, layer, "svg-asy")]
            v3 = cka_results[(model, layer, "tikz-asy")]
            lines.append(f"| {layer} | {v1:.4f} | {v2:.4f} | {v3:.4f} |")

    # Bootstrap CI
    lines.append("\n## D. Bootstrap 95% CI")
    lines.append("\n| Model | Layer | Mean | CI Low | CI High | Width |")
    lines.append("|-------|-------|------|--------|---------|-------|")
    for model in MODELS:
        for layer in LAYERS:
            v = bootstrap_results[(model, layer)]
            lines.append(f"| {model} | {layer} | {v['mean']:.4f} | {v['ci_low']:.4f} | {v['ci_high']:.4f} | {v['ci_width']:.4f} |")

    # A1 Power
    lines.append("\n## E. A1 Power Simulation")
    for name, v in power_results.items():
        lines.append(f"\n**{name}** (cells={v['n_cells']}, iter={v['n_iter']}):")
        lines.append(f"- Power: **{v['power']:.4f}**")
        lines.append(f"- Mean observed CKA: {v['mean_observed_cka']:.4f}")
        lines.append(f"- Mean null p-value: {v['mean_null_p_value']:.4f}")

    # A2 Permutation
    lines.append("\n## F. A2 Permutation Test")
    lines.append("\n| Model | Observed CKA | Null Mean | Null Std | p-value |")
    lines.append("|-------|-------------|-----------|----------|---------|")
    for model in MODELS:
        v = perm_results[model]
        lines.append(f"| {model} | {v['observed_cka_mean']:.4f} | {v['null_mean']:.4f} | {v['null_std']:.4f} | {v['p_value']:.6f} |")

    # Robustness
    lines.append("\n## Robustness: CCA + Procrustes")
    lines.append("\n### Mean CCA per Model × Layer")
    lines.append("| Model | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
    lines.append("|-------|" + "|".join("------" for _ in LAYERS) + "|")
    for model in MODELS:
        vals = []
        for layer in LAYERS:
            pair_vals = [cca_results[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            vals.append(f"{np.nanmean(pair_vals):.4f}")
        lines.append(f"| {model} | {' | '.join(vals)} |")

    lines.append("\n### Mean Procrustes per Model × Layer")
    lines.append("| Model | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
    lines.append("|-------|" + "|".join("------" for _ in LAYERS) + "|")
    for model in MODELS:
        vals = []
        for layer in LAYERS:
            pair_vals = [proc_results[(model, layer, f"{f1}-{f2}")] for f1, f2 in FORMAT_PAIRS]
            vals.append(f"{np.mean(pair_vals):.4f}")
        lines.append(f"| {model} | {' | '.join(vals)} |")

    # Sanity checks
    lines.append("\n## Sanity Checks")
    # Check probe accuracy monotonicity
    for model in MODELS:
        accs = [probe_results[model][l] for l in LAYERS]
        monotonic = all(accs[i] <= accs[i+1] for i in range(len(accs)-1))
        lines.append(f"- {model} probe monotonic increasing: {'YES' if monotonic else 'NO'} (min={min(accs):.4f}, max={max(accs):.4f})")

    # CI width check
    ci_widths = [bootstrap_results[(m, l)]["ci_width"] for m in MODELS for l in LAYERS]
    lines.append(f"- Bootstrap CI widths: min={min(ci_widths):.4f}, max={max(ci_widths):.4f}, mean={np.mean(ci_widths):.4f}")
    lines.append(f"- All CI widths < 0.1: {'YES' if max(ci_widths) < 0.1 else 'NO'}")

    # A2 significance
    for model in MODELS:
        p = perm_results[model]["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        lines.append(f"- {model} A2 p-value: {p:.6f} ({sig})")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    print("=" * 60)
    print("Stage B Analysis v1")
    print("=" * 60)

    data, triples = load_hidden_states()
    probe_results = run_probe_fit(data)
    cka_results = run_cka(data)
    bootstrap_results = run_bootstrap_ci(data)
    power_results = run_a1_power_sim(data)
    perm_results = run_a2_permutation(data)
    cca_results, proc_results = run_robustness(data)

    print("\n=== Saving outputs ===")
    save_outputs(probe_results, cka_results, bootstrap_results, power_results,
                 perm_results, cca_results, proc_results)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Total elapsed: {elapsed:.1f}s")
    print(f"Output dir: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
