#!/usr/bin/env python
"""Negative Control CKA Analysis: Compare Python-X vs Visual cross-format CKA."""
import json
import os
import numpy as np

np.seterr(over='raise', invalid='raise')
import torch
import time
import traceback
import sys
from pathlib import Path

CKPT_FILE = Path("/root/autodl-tmp/logs/ckpt_python_neg.txt")


def atomic_write_json(path: Path, obj):
    """Atomic JSON write: tmp file + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_ckpt_line(line: str):
    """Append a line to CKPT_FILE with fsync."""
    CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CKPT_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_set():
    """Parse CKPT_FILE → set of (pair_type, model, layer) triples already done."""
    done = set()
    if not CKPT_FILE.exists():
        return done
    for raw in CKPT_FILE.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        # Format: "python-X {model} L{layer} t={ts}"
        parts = raw.split()
        if len(parts) < 3:
            continue
        pair_type, model, layer_tok = parts[0], parts[1], parts[2]
        if not layer_tok.startswith("L"):
            continue
        try:
            layer = int(layer_tok[1:])
        except ValueError:
            continue
        done.add((pair_type, model, layer))
    return done

LAYERS = [4, 8, 12, 16, 20, 24, 28]
MODELS = ["coder", "viscoder2", "qwen25"]
VISUAL_FORMATS = ["svg", "tikz", "asy"]
CACHE_DIR = Path("/root/autodl-tmp/cache/hidden_states")
OUT_DIR = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/negative_control")
N_TRIPLES = 252
N_PERM = 5000
SEED = 42

def load_format(model, fmt, n=N_TRIPLES):
    """Load hidden states: returns (n, 7, 3584) float32."""
    tensors = []
    d = CACHE_DIR / model / fmt
    for i in range(n):
        t = torch.load(d / f"{i}.pt", map_location="cpu", weights_only=True)
        tensors.append(t.float().numpy())
    return np.stack(tensors, axis=0)

def center_gram(K):
    n = K.shape[0]
    H = np.eye(n, dtype=np.float32) - np.float32(1.0 / n)
    return H @ K @ H

def cka(X, Y):
    """Linear CKA between X(n,d) and Y(n,d)."""
    KX = X @ X.T
    KY = Y @ Y.T
    KX_c = center_gram(KX)
    KY_c = center_gram(KY)
    hsic_xy = np.float64(np.sum(KX_c.astype(np.float64) * KY_c.astype(np.float64)))
    hsic_xx = np.float64(np.sum(KX_c.astype(np.float64) ** 2))
    hsic_yy = np.float64(np.sum(KY_c.astype(np.float64) ** 2))
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0

def permutation_test(X, Y, n_perm=N_PERM, seed=SEED):
    """Permutation null: shuffle Y sample indices, recompute CKA."""
    observed = cka(X, Y)
    rng = np.random.RandomState(seed)
    null_dist = []
    for _ in range(n_perm):
        perm = rng.permutation(Y.shape[0])
        null_dist.append(cka(X, Y[perm]))
    null_dist = np.array(null_dist)
    p_value = float(np.mean(null_dist >= observed))
    return observed, p_value, float(null_dist.mean()), float(null_dist.std())

def bootstrap_ci(X, Y, n_boot=1000, seed=SEED, frac=0.8):
    """Subsampling 95% CI for CKA (without replacement to avoid Gram diagonal inflation)."""
    rng = np.random.RandomState(seed + 1)
    n = X.shape[0]
    m = int(n * frac)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, m, replace=False)
        boots.append(cka(X[idx], Y[idx]))
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load all data
    print("Loading hidden states...")
    data = {}
    for model in MODELS:
        data[model] = {}
        for fmt in VISUAL_FORMATS + ["python"]:
            data[model][fmt] = load_format(model, fmt)
            print(f"  {model}/{fmt}: {data[model][fmt].shape}")

    # Define pairs
    python_pairs = [("python", "svg"), ("python", "tikz"), ("python", "asy")]
    visual_pairs = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]

    # Resume support: load partial results + done-set
    out_path = OUT_DIR / "negative_control_cka.json"
    done_set = load_done_set()
    if out_path.exists() and done_set:
        try:
            results = json.loads(out_path.read_text())
            # Ensure keys exist
            results.setdefault("python_x", [])
            results.setdefault("visual_x", [])
            results.setdefault("summary", {})
            print(f"[RESUME] loaded partial results: py_x={len(results['python_x'])} vis_x={len(results['visual_x'])} done_set={len(done_set)}")
        except Exception as e:
            print(f"[RESUME] failed to load partial results ({e}); starting fresh")
            results = {"python_x": [], "visual_x": [], "summary": {}}
            done_set = set()
    else:
        results = {"python_x": [], "visual_x": [], "summary": {}}

    # Compute CKA for all pairs at all layers
    for pair_type, pairs, result_key in [
        ("python-X", python_pairs, "python_x"),
        ("visual-X", visual_pairs, "visual_x"),
    ]:
        total = len(MODELS) * len(LAYERS)
        for model in MODELS:
            for li, layer in enumerate(LAYERS):
                if (pair_type, model, layer) in done_set:
                    print(f"[RESUME] skip pair_type={pair_type} model={model} layer={layer}", flush=True)
                    continue
                # Display: layer={actual_idx}(li={li}/{len(LAYERS)-1}, max_L={LAYERS[-1]})
                print(f"[CKPT] {pair_type} computing model={model} layer_idx={layer} li={li}/{len(LAYERS)-1} t={time.time()}", flush=True)
                for f1, f2 in pairs:
                    X = data[model][f1][:, li, :]
                    Y = data[model][f2][:, li, :]

                    obs, p_val, null_mean, null_std = permutation_test(X, Y)
                    ci_lo, ci_hi = bootstrap_ci(X, Y)

                    row = {
                        "model": model, "layer": layer,
                        "format_pair": f"{f1}-{f2}",
                        "cka": round(obs, 4),
                        "p_value": p_val,
                        "null_mean": round(null_mean, 4),
                        "null_std": round(null_std, 4),
                        "ci_95_lo": round(ci_lo, 4),
                        "ci_95_hi": round(ci_hi, 4),
                    }
                    results[result_key].append(row)
                    print(f"  [{pair_type}] {model} L{layer} {f1}-{f2}: CKA={obs:.4f} p={p_val:.4f} null={null_mean:.4f}+-{null_std:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}]")

                # Atomic checkpoint: rewrite partial results JSON + append CKPT line
                try:
                    atomic_write_json(out_path, results)
                    append_ckpt_line(f"{pair_type} {model} L{layer} t={time.time()}")
                    done_set.add((pair_type, model, layer))
                except Exception as e:
                    print(f"[CKPT] atomic write failed: {e}", flush=True)

    # Summary: mean CKA at L28 for python-X vs visual-X
    for model in MODELS:
        py_l28 = [r["cka"] for r in results["python_x"] if r["model"] == model and r["layer"] == 28]
        vis_l28 = [r["cka"] for r in results["visual_x"] if r["model"] == model and r["layer"] == 28]
        results["summary"][model] = {
            "python_x_mean_L28": round(np.mean(py_l28), 4),
            "visual_x_mean_L28": round(np.mean(vis_l28), 4),
            "ratio": round(np.mean(py_l28) / np.mean(vis_l28), 4) if np.mean(vis_l28) > 0 else None,
        }
        print(f"\n[SUMMARY] {model}: Python-X mean CKA@L28={np.mean(py_l28):.4f}, Visual-X mean CKA@L28={np.mean(vis_l28):.4f}, ratio={np.mean(py_l28)/np.mean(vis_l28):.2f}")

    elapsed = time.time() - t0
    results["elapsed_s"] = round(elapsed, 1)
    results["n_perm"] = N_PERM
    results["n_triples"] = N_TRIPLES

    atomic_write_json(out_path, results)
    print(f"\nSaved to {out_path} ({elapsed:.1f}s)")

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
