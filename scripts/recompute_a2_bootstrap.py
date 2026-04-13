#!/usr/bin/env python
"""Recompute A2 permutation (n_perm=5000) + bootstrap CI (B=1000) for iterative results.

Replays the deterministic iterative projections to recover residualized states,
then recomputes only A2 and bootstrap without re-running the full pipeline.

Usage:
  python scripts/recompute_a2_bootstrap.py \
      --models deepseek viscoder2 qwen25 codestral starcoder2 \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --results-dir /root/autodl-tmp/cache/format_residualized_cka \
      --n-perm 5000 --n-bootstrap 1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

np.seterr(over='raise', invalid='raise')
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(line_buffering=True)

FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42


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
    import torch
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


def project_out(H, W):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    tol = S[0] * 1e-6
    mask = S > tol
    V = Vt[mask]
    return H - H @ V.T @ V


def _center_gram(K):
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    return H @ K @ H


def _cka_from_centered(KX_c, KY_c):
    hsic_xy = np.float64(np.sum(KX_c.astype(np.float64) * KY_c.astype(np.float64)))
    hsic_xx = np.float64(np.sum(KX_c.astype(np.float64) ** 2))
    hsic_yy = np.float64(np.sum(KY_c.astype(np.float64) ** 2))
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0


def run_bootstrap_ci(fmt_data, n_triples, n_bootstrap, frac=0.8):
    n = n_triples
    m = int(n * frac)
    H = np.eye(m, dtype=np.float32) - np.float32(1.0 / m)
    rng = np.random.RandomState(SEED)

    grams = {fmt: fmt_data[fmt] @ fmt_data[fmt].T for fmt in FORMATS}

    boot_means = np.empty(n_bootstrap, dtype=np.float32)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=m, replace=False)
        pair_ckas = []
        for f1, f2 in FORMAT_PAIRS:
            KX = grams[f1][np.ix_(idx, idx)]
            KY = grams[f2][np.ix_(idx, idx)]
            KX_c = H @ KX @ H
            KY_c = H @ KY @ H
            hsic_xy = np.float64(np.sum(KX_c.astype(np.float64) * KY_c.astype(np.float64)))
            hsic_xx = np.float64(np.sum(KX_c.astype(np.float64) ** 2))
            hsic_yy = np.float64(np.sum(KY_c.astype(np.float64) ** 2))
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
    n = n_triples
    H_center = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    rng = np.random.RandomState(SEED + 2)

    grams = {fmt: fmt_data[fmt] @ fmt_data[fmt].T for fmt in FORMATS}
    centered = {fmt: H_center @ grams[fmt] @ H_center for fmt in FORMATS}
    hsic_xx = {fmt: float(np.sum(centered[fmt].astype(np.float64) ** 2)) for fmt in FORMATS}

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
            hsic_xy = np.float64(np.sum(centered[f1].astype(np.float64) * KY_c.astype(np.float64)))
            hsic_yy = np.float64(np.sum(KY_c.astype(np.float64) ** 2))
            denom = np.sqrt(hsic_xx[f1] * hsic_yy)
            null_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
        null_means[p_idx] = np.mean(null_vals)
        if (p_idx + 1) % 1000 == 0:
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


def replay_iterative_projections(data, layer_idx, n_triples, n_iters):
    """Replay deterministic iterative projections to recover residualized states."""
    fmt_data = {fmt: data[fmt][:n_triples, layer_idx, :].copy() for fmt in FORMATS}

    for it in range(n_iters - 1):  # last iteration is the check, not a projection
        X = np.concatenate([fmt_data[fmt] for fmt in FORMATS], axis=0)
        y = np.concatenate([np.full(n_triples, i) for i in range(len(FORMATS))])
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED, C=1.0)
        clf.fit(X, y)
        W = clf.coef_.astype(np.float32)
        for fmt in FORMATS:
            fmt_data[fmt] = project_out(fmt_data[fmt], W)

    return fmt_data


def main():
    parser = argparse.ArgumentParser(description="Recompute A2 + bootstrap for iterative results")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/cache/hidden_states")
    parser.add_argument("--results-dir", type=str,
                        default="/root/autodl-tmp/cache/format_residualized_cka")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)

    print(f"=== Recompute A2 (n_perm={args.n_perm}) + Bootstrap (B={args.n_bootstrap}) ===")
    print(f"Models: {args.models}")

    for model in args.models:
        t0_model = time.time()
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        # Load existing iterative results
        iter_path = results_dir / model / "iterative_residualization.json"
        if not iter_path.exists():
            print(f"  ERROR: {iter_path} not found, skipping")
            continue
        with open(iter_path) as f:
            iter_results = json.load(f)

        # Read model meta
        meta = read_model_meta(cache_dir, model)
        layers = meta["layers"]
        hidden_dim = meta["hidden_dim"]
        n_triples = meta["n_saved"]

        print(f"  layers={layers}, hidden_dim={hidden_dim}, n_triples={n_triples}")

        # Load hidden states
        data = load_hidden_states(cache_dir, model, layers, hidden_dim, n_triples)

        # Per-layer: replay projections, recompute A2 + bootstrap
        for li, layer in enumerate(layers):
            layer_key = str(layer)
            if layer_key not in iter_results:
                print(f"  WARNING: layer {layer} not in results, skipping")
                continue

            n_iters = iter_results[layer_key]["n_iterations"]
            print(f"\n  Layer L{layer}: replaying {n_iters} iterations...")

            t0 = time.time()
            fmt_data = replay_iterative_projections(data, li, n_triples, n_iters)

            # Verify CKA matches (inline)
            grams = {fmt: fmt_data[fmt] @ fmt_data[fmt].T for fmt in FORMATS}
            centered = {fmt: _center_gram(grams[fmt]) for fmt in FORMATS}
            cka_vals = []
            for f1, f2 in FORMAT_PAIRS:
                cka_vals.append(_cka_from_centered(centered[f1], centered[f2]))
            replay_cka = float(np.mean(cka_vals))
            saved_cka = iter_results[layer_key]["final_cka"]
            cka_diff = abs(replay_cka - saved_cka)
            print(f"    CKA check: replay={replay_cka:.6f}, saved={saved_cka:.6f}, diff={cka_diff:.6f}")
            if cka_diff > 0.01:
                print(f"    WARNING: CKA mismatch > 1%, check reproducibility!")

            # Bootstrap CI
            print(f"    Computing bootstrap CI (B={args.n_bootstrap})...")
            boot_ci = run_bootstrap_ci(fmt_data, n_triples, args.n_bootstrap)
            print(f"    Bootstrap: {boot_ci['mean']:.4f} [{boot_ci['ci_low']:.4f}, {boot_ci['ci_high']:.4f}]")

            # A2 permutation
            print(f"    Computing A2 permutation (n_perm={args.n_perm})...")
            a2_result = run_a2_perm(fmt_data, n_triples, args.n_perm)
            print(f"    A2: obs={a2_result['observed']:.4f}, null={a2_result['null_mean']:.4f}, "
                  f"p={a2_result['p_value']}")

            # Update results
            iter_results[layer_key]["bootstrap_ci"] = boot_ci
            iter_results[layer_key]["a2_permutation"] = a2_result

            # Incremental save
            _save_json(iter_path, iter_results)
            elapsed = time.time() - t0
            print(f"    [saved, elapsed={elapsed:.1f}s]")

        # Recompute aggregate A2
        print(f"\n  === Recomputing aggregate A2 (all layers) ===")
        t0 = time.time()

        all_fmt_data = {}
        for li, layer in enumerate(layers):
            layer_key = str(layer)
            n_iters = iter_results[layer_key]["n_iterations"]
            all_fmt_data[li] = replay_iterative_projections(data, li, n_triples, n_iters)

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

        null_means = np.empty(args.n_perm, dtype=np.float32)
        for p_idx in range(args.n_perm):
            perm = rng.permutation(n)
            null_vals = []
            for li in range(len(layers)):
                for f1, f2 in FORMAT_PAIRS:
                    KY_perm = all_grams[li][f2][np.ix_(perm, perm)]
                    KY_c = H_center @ KY_perm @ H_center
                    hsic_xy = np.float64(np.sum(all_centered[li][f1].astype(np.float64) * KY_c.astype(np.float64)))
                    hsic_yy = np.float64(np.sum(KY_c.astype(np.float64) ** 2))
                    denom = np.sqrt(all_hsic_xx[li][f1] * hsic_yy)
                    null_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
            null_means[p_idx] = np.mean(null_vals)
            if (p_idx + 1) % 1000 == 0:
                print(f"    agg perm {p_idx+1}/{args.n_perm}")

        agg_p = float(np.mean(null_means >= obs_mean))
        agg_result = {
            "observed": round(obs_mean, 6),
            "null_mean": round(float(np.mean(null_means)), 6),
            "null_std": round(float(np.std(null_means)), 6),
            "null_95th": round(float(np.percentile(null_means, 95)), 6),
            "p_value": round(agg_p, 6),
            "n_perm": args.n_perm,
        }
        agg_path = results_dir / model / "iterative_a2_aggregate.json"
        _save_json(agg_path, agg_result)
        elapsed = time.time() - t0
        print(f"    Aggregate A2: obs={obs_mean:.4f}, null={np.mean(null_means):.4f}, p={agg_p}")
        print(f"    [saved aggregate, elapsed={elapsed:.1f}s]")

        del data, all_fmt_data
        import gc
        gc.collect()

        model_elapsed = time.time() - t0_model
        print(f"\n  {model} DONE in {model_elapsed:.1f}s")

    print(f"\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
