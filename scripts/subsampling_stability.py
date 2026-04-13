#!/usr/bin/env python
"""Subsampling stability for format-residualized CKA (v2).

Sweeps 6 models x {100,150,200,252} x n_seeds. Each entry reports:
  - residualized_cka (mean over 3 format pairs)
  - format_classifier_acc_train (fit accuracy)
  - format_classifier_acc_holdout (5-fold CV mean)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_b_residualized_cka import (
    FORMATS, read_model_meta, load_hidden_states,
    fit_format_classifier, residualize_data, compute_residualized_cka,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.stdout.reconfigure(line_buffering=True)

SEED = 42


def subsample_data(data, idx):
    return {fmt: data[fmt][idx] for fmt in FORMATS}


def holdout_acc(data, layer_idx, n_triples):
    X = np.concatenate([data[fmt][:n_triples, layer_idx, :] for fmt in FORMATS], axis=0)
    y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED, C=1.0)
    # cv=5 stratified by default
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return float(scores.mean())


def run(model, cache_dir, out_dir, sample_sizes, n_seeds):
    meta = read_model_meta(cache_dir, model)
    layers = meta["layers"]
    hidden_dim = meta["hidden_dim"]
    n_full = meta["n_saved"]
    print(f"Model {model}: layers={layers}, hidden_dim={hidden_dim}, n_full={n_full}")

    data = load_hidden_states(cache_dir, model, layers, hidden_dim, n_full)

    results = {"model": model, "layers": layers, "n_full": n_full,
               "sample_sizes": sample_sizes, "n_seeds": n_seeds,
               "records": []}
    t0 = time.time()
    for li, layer in enumerate(layers):
        for n in sample_sizes:
            if n > n_full:
                print(f"  skip layer={layer} n={n} > n_full={n_full}")
                continue
            for s in range(n_seeds):
                rng = np.random.RandomState(1000 * s + n)
                idx = rng.choice(n_full, size=n, replace=False)
                sub = subsample_data(data, idx)
                W, acc_train = fit_format_classifier(sub, li, n)
                acc_hold = holdout_acc(sub, li, n)
                res_data = residualize_data(sub, W, li, n)
                rows, _ = compute_residualized_cka(res_data)
                cka_mean = float(np.mean([r["cka"] for r in rows]))
                rec = {
                    "layer": int(layer),
                    "n": int(n),
                    "seed": int(s),
                    "residualized_cka": round(cka_mean, 6),
                    "format_classifier_acc_train": round(float(acc_train), 4),
                    "format_classifier_acc_holdout": round(acc_hold, 4),
                }
                results["records"].append(rec)
                print(f"  layer={layer} n={n} seed={s}: cka={cka_mean:.4f} "
                      f"acc_train={acc_train:.3f} acc_hold={acc_hold:.3f}")
    results["elapsed_s"] = round(time.time() - t0, 1)

    out_path = Path(out_dir) / model / "stability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}  elapsed={results['elapsed_s']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["coder", "viscoder2", "qwen25", "codestral", "starcoder2", "deepseek"])
    ap.add_argument("--cache-dir", default="/root/autodl-tmp/cache/hidden_states")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/cache/subsampling_stability")
    ap.add_argument("--n-list", type=int, nargs="+", default=[100, 150, 200, 252])
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--layers", default="all", help="(currently ignored; always all)")
    args = ap.parse_args()
    for model in args.models:
        run(model, Path(args.cache_dir), args.out_dir, args.n_list, args.n_seeds)


if __name__ == "__main__":
    main()
