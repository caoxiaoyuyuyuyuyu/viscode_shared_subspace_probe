#!/usr/bin/env python
"""D081 Task 4: BH-FDR global correction across per-pair per-layer permutation tests.

For each (model, layer, pair), compute an individual permutation p-value by
permuting sample indices of the second format and recomputing CKA. Apply
Benjamini-Hochberg FDR at α=0.05, either globally or by type (visual/python_x).

Usage:
  python scripts/d081_bh_fdr_correction.py [--n-perm 5000] [--pool-scope by_type] [--cache-dir ...]
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

SEED = 42
N_TRIPLES = 252
ALPHA = 0.05

VISUAL_FORMATS = ["svg", "tikz", "asy"]
VISUAL_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
PYTHON_X_PAIRS = [("python", "svg"), ("python", "tikz"), ("python", "asy")]

QWEN_LAYERS = [4, 8, 12, 16, 20, 24, 28]
QWEN_HIDDEN = 3584
QWEN_MODELS = {"coder", "viscoder2", "qwen25"}

DEFAULT_CACHE = Path("/root/autodl-tmp/cache/hidden_states")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "artifacts" / "d081_bh_fdr_correction"


# ── Meta resolution ──────────────────────────────────────────────────
def resolve_meta(cache_dir: Path, model: str):
    """Return dict with layers, hidden_dim for a model."""
    if model in QWEN_MODELS:
        return {"layers": QWEN_LAYERS, "hidden_dim": QWEN_HIDDEN}
    for fmt in VISUAL_FORMATS + ["python"]:
        p = cache_dir / model / fmt / "summary.json"
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            layers = s.get("layers_resolved") or s.get("layers")
            hidden_dim = s["hidden_dim"]
            return {"layers": list(layers), "hidden_dim": int(hidden_dim)}
    raise FileNotFoundError(f"No summary.json for {model}")


def format_has_data(cache_dir: Path, model: str, fmt: str, n_triples: int):
    d = cache_dir / model / fmt
    if not d.is_dir():
        return False
    return (d / f"{n_triples - 1}.pt").exists()


def discover_models(cache_dir: Path, n_triples: int):
    models = []
    if not cache_dir.is_dir():
        return models
    for entry in sorted(cache_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Require all three visual formats
        if all(format_has_data(cache_dir, entry.name, f, n_triples) for f in VISUAL_FORMATS):
            models.append(entry.name)
    return models


# ── Data loader ──────────────────────────────────────────────────────
def load_hidden_states(cache_dir: Path, model: str, formats, layers, hidden_dim, n_triples):
    n_layers = len(layers)
    data = {}
    for fmt in formats:
        tensors = []
        d = cache_dir / model / fmt
        for i in range(n_triples):
            pt = d / f"{i}.pt"
            t = torch.load(pt, map_location="cpu", weights_only=True)
            assert t.shape == (n_layers, hidden_dim), \
                f"{pt}: expected ({n_layers},{hidden_dim}), got {tuple(t.shape)}"
            tensors.append(t.float().numpy())
        data[fmt] = np.stack(tensors, axis=0)  # (n, n_layers, hidden_dim) f32
    return data


# ── CKA core ─────────────────────────────────────────────────────────
def center_gram(K):
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    return H @ K @ H


def cka_from_centered(KX_c, KY_c):
    hsic_xy = float(np.sum(KX_c.astype(np.float64) * KY_c.astype(np.float64)))
    hsic_xx = float(np.sum(KX_c.astype(np.float64) ** 2))
    hsic_yy = float(np.sum(KY_c.astype(np.float64) ** 2))
    denom = (hsic_xx * hsic_yy) ** 0.5
    return (hsic_xy / denom) if denom > 1e-12 else 0.0


def compute_raw_grams(data, n_layers):
    grams = {}
    for fmt, arr in data.items():
        grams[fmt] = {}
        for li in range(n_layers):
            X = arr[:, li, :]
            grams[fmt][li] = X @ X.T  # (n, n) f32
    return grams


# ── Per-pair per-layer permutation test ─────────────────────────────
def perm_pvalue_pair(raw_gram_x, raw_gram_y, n_triples, n_perm, rng):
    """Individual permutation test for one (layer, pair).

    Observed: CKA(center(Kx), center(Ky)).
    Null: permute rows/cols of Ky, recenter, recompute CKA.
    p = (#{null >= obs} + 1) / (n_perm + 1).
    """
    n = n_triples
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    KX_c = H @ raw_gram_x @ H
    KY_c_obs = H @ raw_gram_y @ H
    obs = cka_from_centered(KX_c, KY_c_obs)

    hsic_xx = float(np.sum(KX_c.astype(np.float64) ** 2))
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        KY_perm = raw_gram_y[np.ix_(perm, perm)]
        KY_c = H @ KY_perm @ H
        hsic_xy = float(np.sum(KX_c.astype(np.float64) * KY_c.astype(np.float64)))
        hsic_yy = float(np.sum(KY_c.astype(np.float64) ** 2))
        denom = (hsic_xx * hsic_yy) ** 0.5
        null_v = (hsic_xy / denom) if denom > 1e-12 else 0.0
        if null_v >= obs:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return obs, p


# ── BH-FDR ───────────────────────────────────────────────────────────
def bh_fdr(pvals: np.ndarray, alpha=0.05):
    n = len(pvals)
    sorted_idx = np.argsort(pvals, kind="stable")
    sorted_p = pvals[sorted_idx]
    # Adjusted q-values (monotone from the tail)
    qvals_sorted = np.minimum.accumulate(
        (sorted_p * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    qvals_sorted = np.clip(qvals_sorted, 0.0, 1.0)
    qvals = np.empty(n)
    qvals[sorted_idx] = qvals_sorted
    sig = qvals <= alpha
    return sig, qvals


# ── Main driver ──────────────────────────────────────────────────────
def run_model(cache_dir, model, n_triples, n_perm, seed):
    meta = resolve_meta(cache_dir, model)
    layers = meta["layers"]
    hidden_dim = meta["hidden_dim"]

    formats = list(VISUAL_FORMATS)
    has_python = format_has_data(cache_dir, model, "python", n_triples)
    if has_python:
        formats.append("python")

    t0 = time.time()
    data = load_hidden_states(cache_dir, model, formats, layers, hidden_dim, n_triples)
    grams = compute_raw_grams(data, len(layers))
    del data
    gc.collect()

    rng = np.random.RandomState(seed)
    rows = []
    pair_sets = [("visual", VISUAL_PAIRS)]
    if has_python:
        pair_sets.append(("python_x", PYTHON_X_PAIRS))

    for type_name, pairs in pair_sets:
        for li, layer in enumerate(layers):
            for (f1, f2) in pairs:
                obs, p = perm_pvalue_pair(
                    grams[f1][li], grams[f2][li], n_triples, n_perm, rng
                )
                rows.append({
                    "model": model,
                    "layer": f"L{layer}",
                    "pair": f"{f1}-{f2}",
                    "type": type_name,
                    "observed_cka": round(float(obs), 6),
                    "p_uncorrected": round(float(p), 6),
                })
    elapsed = time.time() - t0
    print(f"  {model}: {len(rows)} tests in {elapsed:.1f}s (python={'y' if has_python else 'n'})",
          flush=True)
    del grams
    gc.collect()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-triples", type=int, default=N_TRIPLES)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pool-scope", choices=["global", "by_type"], default="by_type",
                    help="BH-FDR pool scope: global (all tests) or by_type (visual/python_x separate)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Explicit list; default = auto-discover")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    cache_dir = args.cache_dir
    if args.models:
        models = args.models
    else:
        models = discover_models(cache_dir, args.n_triples)
    print(f"[D081] cache_dir={cache_dir}")
    print(f"[D081] models={models}")
    print(f"[D081] n_perm={args.n_perm}, n_triples={args.n_triples}, alpha={args.alpha}")

    all_rows = []
    for mi, m in enumerate(models):
        print(f"\n=== [{mi+1}/{len(models)}] {m} ===", flush=True)
        all_rows.extend(run_model(cache_dir, m, args.n_triples, args.n_perm,
                                  args.seed + mi * 1000))

    # Apply BH-FDR with chosen pool scope
    if args.pool_scope == "global":
        pvals = np.array([r["p_uncorrected"] for r in all_rows], dtype=np.float64)
        sig_bh, qvals = bh_fdr(pvals, alpha=args.alpha)
        for r, q, s in zip(all_rows, qvals, sig_bh):
            r["q_bh"] = round(float(q), 6)
            r["significant_bh"] = bool(s)
            r["significant_uncorrected"] = r["p_uncorrected"] <= args.alpha
    else:  # by_type
        for r in all_rows:
            r["significant_uncorrected"] = r["p_uncorrected"] <= args.alpha
        for type_name in {r["type"] for r in all_rows}:
            indices = [i for i, r in enumerate(all_rows) if r["type"] == type_name]
            pvals = np.array([all_rows[i]["p_uncorrected"] for i in indices], dtype=np.float64)
            sig_bh_t, qvals_t = bh_fdr(pvals, alpha=args.alpha)
            for idx, q, s in zip(indices, qvals_t, sig_bh_t):
                all_rows[idx]["q_bh"] = round(float(q), 6)
                all_rows[idx]["significant_bh"] = bool(s)

    total = len(all_rows)
    sig_uncorr = int(sum(r["significant_uncorrected"] for r in all_rows))
    sig_bh_ct = int(sig_bh.sum())
    by_type = {}
    for r in all_rows:
        t = r["type"]
        b = by_type.setdefault(t, {"total": 0, "sig_uncorr": 0, "sig_bh": 0})
        b["total"] += 1
        b["sig_uncorr"] += int(r["significant_uncorrected"])
        b["sig_bh"] += int(r["significant_bh"])

    out = {
        "metadata": {
            "n_perm": args.n_perm,
            "alpha": args.alpha,
            "method": "benjamini_hochberg",
            "pool_scope": args.pool_scope,
            "n_triples": args.n_triples,
            "models": models,
            "total_tests": total,
            "p_value_formula": "(#{null_cka >= obs} + 1) / (n_perm + 1)",
        },
        "per_test": all_rows,
        "summary": {
            "total_tests": total,
            "significant_uncorrected": sig_uncorr,
            "significant_bh": sig_bh_ct,
            "by_type": by_type,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "results.json"
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(out_path)

    print("\n========== SUMMARY ==========")
    print(json.dumps(out["summary"], indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
