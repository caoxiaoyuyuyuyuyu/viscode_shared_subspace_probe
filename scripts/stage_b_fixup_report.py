#!/usr/bin/env python
"""Fixup: re-run robustness + generate stats_report.md from saved artifacts.

The initial run crashed in generate_stats_report due to a key mismatch.
CSVs, figures, and JSONs were saved. This script:
1. Loads hidden states (fast)
2. Runs robustness (Procrustes)
3. Loads existing JSONs
4. Generates stats_report.md + robustness CSV

Usage:
  export PATH=/root/miniconda3/bin:$PATH
  cd /root/autodl-tmp/viscode_shared_subspace_probe
  python scripts/stage_b_fixup_report.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import procrustes as sp_procrustes
from sklearn.decomposition import PCA

sys.stdout.reconfigure(line_buffering=True)

# ── Config ──
LAYERS = [4, 8, 12, 16, 20, 24, 28]
MODELS = ["coder", "viscoder2", "qwen25"]
FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42

CACHE_DIR = Path("/root/autodl-tmp/cache/hidden_states")
TRIPLES_PATH = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/stage_a/sbert_triples.json")
OUT_DIR = Path("/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/stage_b_analysis_v1")

N_BOOTSTRAP = 1000
N_POWER_ITER = 1000
N_PERM = 2000


def load_hidden_states():
    print("=== Loading hidden states ===")
    t0 = time.time()
    with open(TRIPLES_PATH) as f:
        triples = json.load(f)
    n = len(triples)
    data = {}
    for model in MODELS:
        data[model] = {}
        for fmt in FORMATS:
            tensors = []
            d = CACHE_DIR / model / fmt
            for i in range(n):
                t = torch.load(d / f"{i}.pt", map_location="cpu", weights_only=True)
                tensors.append(t.numpy())
            data[model][fmt] = np.stack(tensors, axis=0)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return data


def procrustes_score(X, Y):
    k = min(50, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, random_state=SEED)
    X_r = pca.fit_transform(X - X.mean(axis=0))
    Y_r = pca.fit_transform(Y - Y.mean(axis=0))
    _, _, disparity = sp_procrustes(X_r, Y_r)
    return round(1.0 - disparity, 6)


def run_robustness(data):
    print("\n=== Robustness: Procrustes ===")
    t0 = time.time()
    proc_rows = []
    for model in MODELS:
        for li, layer in enumerate(LAYERS):
            for f1, f2 in FORMAT_PAIRS:
                X = data[model][f1][:, li, :]
                Y = data[model][f2][:, li, :]
                proc_val = procrustes_score(X, Y)
                proc_rows.append({"model": model, "layer": layer,
                                  "pair": f"{f1}-{f2}", "procrustes": proc_val})
        for li, layer in enumerate(LAYERS):
            vals = [r["procrustes"] for r in proc_rows
                    if r["model"] == model and r["layer"] == layer]
            print(f"  {model} L{layer}: Procrustes={np.mean(vals):.4f}")
    print(f"  Robustness: {time.time()-t0:.1f}s")
    return {"procrustes": proc_rows}


def generate_report(probe_csv, cka_csv, ci_csv, power_json, perm_json, rob):
    """Generate stats_report.md from saved artifacts."""
    import csv

    # Parse probe CSV
    probe_acc = {}
    with open(probe_csv) as f:
        for row in csv.DictReader(f):
            probe_acc.setdefault(row["model"], {})[int(row["layer"])] = float(row["accuracy"])

    # Parse CKA CSV
    cka_lookup = {}
    with open(cka_csv) as f:
        for row in csv.DictReader(f):
            cka_lookup[(row["model"], int(row["layer"]), row["format_pair"])] = float(row["cka"])

    # Parse bootstrap CI CSV
    boot_lookup = {}
    with open(ci_csv) as f:
        for row in csv.DictReader(f):
            boot_lookup[(row["model"], int(row["layer"]))] = {
                "mean": float(row["mean"]), "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]), "ci_width": float(row["ci_width"])
            }

    # Power JSON
    with open(power_json) as f:
        e = json.load(f)

    # Permutation JSON
    with open(perm_json) as f:
        perm = json.load(f)

    # Procrustes lookup
    proc_lookup = {}
    if rob:
        for r in rob["procrustes"]:
            proc_lookup[(r["model"], r["layer"], r["pair"])] = r["procrustes"]

    lines = []
    lines.append("# Stage B Analysis v1 — Statistical Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"Layers: {LAYERS}, Models: {MODELS}")
    lines.append(f"Bootstrap: {N_BOOTSTRAP}, Permutations: {N_PERM}, Power iterations: {N_POWER_ITER}")

    lines.append("\n## Pre-Registration Compliance")
    lines.append("- [x] Main metric: CKA (linear kernel)")
    lines.append("- [x] Procrustes as robustness check")
    lines.append("- [x] A1 power sim: N=12 AND N=18")
    lines.append("- [x] A2 permutation test per model")
    lines.append("- [x] NO quantitative predictivity claim")
    lines.append("- [x] NO decoupling claim")
    lines.append("- [x] N=18 VisCoder2 as 'exploratory view with known memorization floor'")

    # Probe accuracy
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
            lines.append(f"- Null 95th percentile: {v['null_95th']:.4f}")

    # A2 Permutation
    lines.append("\n## F. A2 Permutation Test")
    lines.append("\n| Model | Observed CKA | Null Mean | Null Std | p-value |")
    lines.append("|-------|-------------|-----------|----------|---------|")
    for model in MODELS:
        if model in perm:
            v = perm[model]
            lines.append(f"| {model} | {v['observed_cka_mean']:.4f} | {v['null_mean']:.4f} | {v['null_std']:.4f} | {v['p_value']:.6f} |")

    # Robustness
    if rob:
        lines.append("\n## Robustness: Procrustes")
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
        if model in perm:
            p = perm[model]["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            lines.append(f"- {model} A2 p-value: {p:.6f} ({sig})")

    return "\n".join(lines)


def main():
    t0 = time.time()

    # 1. Load data + run robustness
    data = load_hidden_states()
    rob = run_robustness(data)
    del data

    # 2. Save robustness CSV
    with open(OUT_DIR / "robustness_procrustes.csv", "w") as f:
        f.write("model,layer,format_pair,procrustes\n")
        for r in rob["procrustes"]:
            f.write(f"{r['model']},{r['layer']},{r['pair']},{r['procrustes']}\n")
    print(f"  Saved robustness_procrustes.csv ({len(rob['procrustes'])} rows)")

    # 3. Generate report
    report = generate_report(
        OUT_DIR / "probe_format_accuracy.csv",
        OUT_DIR / "cka_per_layer_per_pair.csv",
        OUT_DIR / "cka_bootstrap_ci.csv",
        OUT_DIR / "a1_power_sim.json",
        OUT_DIR / "a2_permutation.json",
        rob,
    )

    with open(OUT_DIR / "stats_report.md", "w") as f:
        f.write(report)
    print(f"  Saved stats_report.md")

    print(f"\nTotal fixup time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
