#!/usr/bin/env python
"""D081 Task 5: Token-shared null baseline.

Computes BPE token Jaccard similarity between all sample pairs across
formats, then runs a token-weighted permutation null: samples are only
swapped if their token Jaccard exceeds a threshold, controlling for
tokenizer artifacts in the CKA signal.

Usage:
  python scripts/d081_token_shared_null.py \
      --models coder viscoder2 qwen25 codestral starcoder2 deepseek \
      --cache-dir /root/autodl-tmp/cache/hidden_states \
      --data-dir /root/autodl-tmp/viscode_shared_subspace_probe/artifacts \
      --n-perm 1000

  # Smoke test:
  python scripts/d081_token_shared_null.py --models coder --smoke
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

np.seterr(over="raise", invalid="raise")
sys.stdout.reconfigure(line_buffering=True)

FORMATS = ["svg", "tikz", "asy"]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ── Helpers ──────────────────────────────────────────────────────────
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
    raise FileNotFoundError(f"No summary.json for {model} in {cache_dir}")


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
        arr = np.stack(tensors, axis=0)
        data[fmt] = arr
        print(f"  {model}/{fmt}: {arr.shape}")
    return data


# ── CKA Core ─────────────────────────────────────────────────────────
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


# ── BPE Jaccard Computation ──────────────────────────────────────────
def compute_bpe_jaccard_matrix(code_strings_f1, code_strings_f2, tokenizer):
    """Compute pairwise BPE token Jaccard between samples of two formats.

    Returns (n, n) matrix where J[i,j] = |tokens(f1[i]) ∩ tokens(f2[j])| / |tokens(f1[i]) ∪ tokens(f2[j])|
    """
    n = len(code_strings_f1)
    assert len(code_strings_f2) == n

    # Tokenize all samples
    token_sets_f1 = []
    token_sets_f2 = []
    for i in range(n):
        ids1 = set(tokenizer.encode(code_strings_f1[i], add_special_tokens=False))
        ids2 = set(tokenizer.encode(code_strings_f2[i], add_special_tokens=False))
        token_sets_f1.append(ids1)
        token_sets_f2.append(ids2)

    # Compute pairwise Jaccard
    J = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            inter = len(token_sets_f1[i] & token_sets_f2[j])
            union = len(token_sets_f1[i] | token_sets_f2[j])
            J[i, j] = inter / union if union > 0 else 0.0
    return J


def compute_bpe_jaccard_diagonal(code_strings_f1, code_strings_f2, tokenizer):
    """Compute diagonal BPE Jaccard (matched pairs only)."""
    n = len(code_strings_f1)
    diag = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ids1 = set(tokenizer.encode(code_strings_f1[i], add_special_tokens=False))
        ids2 = set(tokenizer.encode(code_strings_f2[i], add_special_tokens=False))
        inter = len(ids1 & ids2)
        union = len(ids1 | ids2)
        diag[i] = inter / union if union > 0 else 0.0
    return diag


# ── Token-Constrained Permutation ────────────────────────────────────
def token_constrained_permutation(n, jaccard_diag, threshold, rng):
    """Generate a permutation that only swaps samples with Jaccard > threshold.

    For samples below threshold, they stay in place (identity).
    For samples above threshold, they are permuted among themselves.
    """
    eligible = np.where(jaccard_diag >= threshold)[0]
    perm = np.arange(n)
    if len(eligible) > 1:
        perm[eligible] = rng.permutation(eligible)
    return perm


def token_weighted_permutation(n, jaccard_matrix, rng):
    """Weighted permutation: sample j replaces i with prob ∝ Jaccard(i,j).

    Uses rejection sampling approach: for each position, sample replacement
    weighted by Jaccard similarity.
    """
    perm = np.arange(n)
    available = list(range(n))
    rng.shuffle(available)

    # Simple approach: sort by Jaccard-weighted random priority
    for i in range(n):
        weights = jaccard_matrix[i, available].copy()
        weights_sum = weights.sum()
        if weights_sum < 1e-8:
            # Uniform if no overlap
            idx = rng.randint(len(available))
        else:
            probs = weights / weights_sum
            idx = rng.choice(len(available), p=probs)
        perm[i] = available[idx]
        available.pop(idx)
        if not available:
            break
    return perm


# ── Main Pipeline ────────────────────────────────────────────────────
def run_token_null_for_pair(grams_f1, grams_f2, jaccard_diag, jaccard_matrix,
                             n, n_perm, threshold=0.3):
    """Run 3 null variants for one format pair at one layer."""
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    rng_global = np.random.RandomState(SEED + 10)
    rng_thresh = np.random.RandomState(SEED + 11)
    rng_weight = np.random.RandomState(SEED + 12)

    KX_c = _center_gram(grams_f1)
    KY_c = _center_gram(grams_f2)
    obs_cka = _cka_from_centered(KX_c, KY_c)

    hsic_xx = float(np.sum(KX_c.astype(np.float64) ** 2))

    null_global = np.empty(n_perm, dtype=np.float32)
    null_thresh = np.empty(n_perm, dtype=np.float32)
    null_weight = np.empty(n_perm, dtype=np.float32)

    for p in range(n_perm):
        # Global null (standard)
        perm_g = rng_global.permutation(n)
        # Threshold null
        perm_t = token_constrained_permutation(n, jaccard_diag, threshold, rng_thresh)
        # Weighted null
        perm_w = token_weighted_permutation(n, jaccard_matrix, rng_weight)

        for null_arr, perm in [(null_global, perm_g), (null_thresh, perm_t), (null_weight, perm_w)]:
            KY_perm = grams_f2[np.ix_(perm, perm)]
            KY_pc = H @ KY_perm @ H
            hsic_xy = np.float64(np.sum(KX_c.astype(np.float64) * KY_pc.astype(np.float64)))
            hsic_yy = np.float64(np.sum(KY_pc.astype(np.float64) ** 2))
            denom = np.sqrt(hsic_xx * hsic_yy)
            cka = hsic_xy / denom if denom > 1e-12 else 0.0
            null_arr[p] = cka

    return {
        "observed_cka": round(obs_cka, 6),
        "global_null_mean": round(float(np.mean(null_global)), 6),
        "global_p": round(float(np.mean(null_global >= obs_cka)), 6),
        "threshold_null_mean": round(float(np.mean(null_thresh)), 6),
        "threshold_p": round(float(np.mean(null_thresh >= obs_cka)), 6),
        "threshold_jaccard": threshold,
        "weighted_null_mean": round(float(np.mean(null_weight)), 6),
        "weighted_p": round(float(np.mean(null_weight >= obs_cka)), 6),
        "mean_jaccard_diag": round(float(np.mean(jaccard_diag)), 4),
    }


def load_code_strings(data_dir, n_triples):
    """Load code strings for BPE tokenization.

    Looks for probe pool JSON or generation artifacts.
    """
    # Try multiple possible locations
    candidates = [
        data_dir / "probe_pool.json",
        data_dir / "stage_a_gen" / "probe_pool.json",
        data_dir / "stage_a_gen_v3" / "probe_pool.json",
    ]

    for path in candidates:
        if path.exists():
            print(f"  Loading code strings from: {path}")
            with open(path) as f:
                pool = json.load(f)
            codes = {}
            for fmt in FORMATS:
                if fmt in pool:
                    codes[fmt] = [item.get("code", item.get("content", ""))
                                  for item in pool[fmt][:n_triples]]
                elif "triples" in pool:
                    codes[fmt] = [t[fmt] for t in pool["triples"][:n_triples]]
            if all(fmt in codes and len(codes[fmt]) == n_triples for fmt in FORMATS):
                return codes
            print(f"    Incomplete: {[(f, len(codes.get(f, []))) for f in FORMATS]}")

    # Try loading from individual format files
    for fmt in FORMATS:
        gen_dirs = sorted((data_dir).glob(f"*gen*"))
        for gd in gen_dirs:
            fmt_file = gd / f"{fmt}_codes.json"
            if fmt_file.exists():
                print(f"  Found: {fmt_file}")

    raise FileNotFoundError(
        f"Cannot find code strings in any of: {[str(p) for p in candidates]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["coder", "viscoder2", "qwen25",
                                 "codestral", "starcoder2", "deepseek"])
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/cache/hidden_states")
    parser.add_argument("--data-dir", type=str,
                        default="/root/autodl-tmp/viscode_shared_subspace_probe/artifacts")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-triples", type=int, default=252)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Jaccard threshold for constrained permutation")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    data_dir = Path(args.data_dir)
    n_triples = 8 if args.smoke else args.n_triples
    n_perm = 20 if args.smoke else args.n_perm
    out_dir = PROJECT_ROOT / "artifacts" / "d081_token_shared_null"

    print(f"=== D081 Task 5: Token-Shared Null Baseline ===")
    print(f"  models: {args.models}, n_perm={n_perm}, threshold={args.threshold}")
    t0 = time.time()

    # Load tokenizer (Qwen2.5 tokenizer, shared across Qwen family)
    print("\n--- Loading tokenizer ---")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load code strings
    print("\n--- Loading code strings ---")
    codes = load_code_strings(data_dir, n_triples)
    for fmt in FORMATS:
        print(f"  {fmt}: {len(codes[fmt])} samples, "
              f"mean len={np.mean([len(c) for c in codes[fmt]]):.0f} chars")

    # Pre-compute Jaccard matrices for each format pair
    print("\n--- Computing BPE Jaccard matrices ---")
    jaccard_data = {}
    for f1, f2 in FORMAT_PAIRS:
        pair_name = f"{f1}-{f2}"
        print(f"  {pair_name}...")
        J = compute_bpe_jaccard_matrix(codes[f1], codes[f2], tokenizer)
        diag = np.diag(J)  # matched-pair Jaccard
        jaccard_data[pair_name] = {
            "matrix": J,
            "diagonal": diag,
            "mean_diag": float(np.mean(diag)),
            "mean_offdiag": float((J.sum() - np.trace(J)) / (n_triples * (n_triples - 1))),
        }
        print(f"    mean_diag={np.mean(diag):.4f}, "
              f"mean_offdiag={jaccard_data[pair_name]['mean_offdiag']:.4f}")

    # Run for each model
    all_results = {
        "metadata": {
            "n_perm": n_perm,
            "n_triples": n_triples,
            "threshold": args.threshold,
            "tokenizer": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "description": "3-way null comparison: global, Jaccard-threshold, Jaccard-weighted",
        },
        "jaccard_summary": {
            pair: {k: v for k, v in jd.items() if k not in ("matrix", "diagonal")}
            for pair, jd in jaccard_data.items()
        },
        "models": {},
    }

    for model in args.models:
        print(f"\n=== {model} ===")
        meta = read_model_meta(cache_dir, model)
        layers = meta["layers"]
        hidden_dim = meta["hidden_dim"]
        n_actual = min(n_triples, meta["n_saved"])

        data = load_hidden_states(cache_dir, model, layers, hidden_dim, n_actual)

        model_results = {}
        for li, layer in enumerate(layers):
            layer_key = f"L{layer}"
            print(f"  Layer {layer}...")

            # Compute Gram matrices
            grams = {}
            for fmt in FORMATS:
                X = data[fmt][:n_actual, li, :]
                grams[fmt] = X @ X.T

            layer_out = {}
            for f1, f2 in FORMAT_PAIRS:
                pair_name = f"{f1}-{f2}"
                jd = jaccard_data[pair_name]
                result = run_token_null_for_pair(
                    grams[f1], grams[f2],
                    jd["diagonal"][:n_actual],
                    jd["matrix"][:n_actual, :n_actual],
                    n_actual, n_perm, args.threshold,
                )
                layer_out[pair_name] = result

            model_results[layer_key] = layer_out
            del grams
            gc.collect()

        all_results["models"][model] = model_results
        del data
        gc.collect()

    elapsed = time.time() - t0
    all_results["metadata"]["elapsed_s"] = round(elapsed, 1)

    _save_json(out_dir / "results.json", all_results)
    print(f"\nSaved: {out_dir / 'results.json'}")

    # Print 3-way comparison summary
    print(f"\n=== 3-Way Null Comparison (elapsed {elapsed:.0f}s) ===")
    print(f"{'Model':<12} {'Pair':<10} {'Obs CKA':>8} {'Global p':>9} {'Thresh p':>9} {'Weight p':>9}")
    for model in args.models:
        if model not in all_results["models"]:
            continue
        md = all_results["models"][model]
        # Show first layer as sample
        first_layer = list(md.keys())[0]
        for pair, res in md[first_layer].items():
            print(f"{model:<12} {pair:<10} {res['observed_cka']:>8.4f} "
                  f"{res['global_p']:>9.4f} {res['threshold_p']:>9.4f} "
                  f"{res['weighted_p']:>9.4f}")


if __name__ == "__main__":
    main()
