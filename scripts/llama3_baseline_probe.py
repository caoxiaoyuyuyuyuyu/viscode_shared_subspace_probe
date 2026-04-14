#!/usr/bin/env python
"""Llama-3-8B code-naive baseline: hidden state extraction + cross-format CKA.

Llama-3-8B-base has no specialized code training, serving as a code-naive
baseline to contrast with code-specialized models (Qwen2.5-Coder, VisCoder2, etc.).

Architecture: 32 layers, hidden_dim=4096.
Sample layers: L4, L18, L31 (depth ~13%, 56%, 97%) — matched to Qwen's
L4/L16/L28 (14%/57%/100%) by relative depth.

Phase 1: Extract hidden states for all 4 formats (svg, tikz, asy, python)
Phase 2: Compute cross-format CKA with permutation test + bootstrap CI

Usage:
  # Phase 1: extract hidden states (requires GPU)
  CUDA_VISIBLE_DEVICES=0 python scripts/llama3_baseline_probe.py --phase extract

  # Phase 2: compute CKA (CPU-only, after extraction)
  python scripts/llama3_baseline_probe.py --phase cka

  # Both phases
  CUDA_VISIBLE_DEVICES=0 python scripts/llama3_baseline_probe.py --phase all
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

np.seterr(over="raise", invalid="raise")

# ── Config ─────────────────────────────────────────────────────────────
MODEL_NAME = "NousResearch/Meta-Llama-3-8B"
MODEL_LOCAL_PATH = "/root/autodl-tmp/models/Meta-Llama-3-8B"
MODEL_TYPE = "base"  # not instruct — plain caption + "\n" prompt
LAYERS = [4, 8, 12, 16, 20, 24, 28]  # 7 equidistant layers matching Qwen protocol
HIDDEN_DIM = 4096
MAX_SEQ_LEN = 4096
N_TRIPLES = 252

VISUAL_FORMATS = ["svg", "tikz", "asy"]
ALL_FORMATS = ["svg", "tikz", "asy", "python"]
FORMAT_PAIRS = [
    ("svg", "tikz"), ("svg", "asy"), ("tikz", "asy"),
    ("python", "svg"), ("python", "tikz"), ("python", "asy"),
]

CACHE_DIR = Path("/root/autodl-tmp/cache/hidden_states/llama3")
OUT_DIR = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/llama3_baseline")
RESOLVED_TRIPLES = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/stage_a/resolved_triples.json")
PYTHON_SNIPPETS = Path("/root/autodl-tmp/cache/hidden_states/python_snippets.json")
CKPT_FILE = Path("/root/autodl-tmp/logs/ckpt_llama3_baseline.txt")

N_PERM = 5000
N_BOOT = 1000
SEED = 42


# ── Checkpoint helpers ─────────────────────────────────────────────────
CKPT_VERSION = "# version=1 llama3_baseline format=fmt sample_idx t=ts"


def atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_ckpt(line: str):
    CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    need_header = not CKPT_FILE.exists() or CKPT_FILE.stat().st_size == 0
    with open(CKPT_FILE, "a") as f:
        if need_header:
            f.write(CKPT_VERSION + f" t={time.time()}\n")
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_extract_done():
    """Return set of (fmt, sample_idx) already extracted."""
    done = set()
    if not CKPT_FILE.exists():
        return done
    lines = CKPT_FILE.read_text().splitlines()
    if not lines or not lines[0].startswith("# version=1"):
        return done
    for raw in lines[1:]:
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) >= 2:
            fmt, idx_str = parts[0], parts[1]
            try:
                done.add((fmt, int(idx_str)))
            except ValueError:
                continue
    return done


# ── Prompt builder ─────────────────────────────────────────────────────
def build_prompt(caption):
    """Base model prompt: caption + newline."""
    return caption + "\n"


# ── Hidden state extraction ────────────────────────────────────────────
def extract_hidden_states(model, tokenizer, prompt_text, code_text, layers, device):
    """Teacher-force prompt+code, mean-pool code-token hidden states per layer."""
    import torch

    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]

    full_text = prompt_text + code_text
    full_ids = tokenizer.encode(full_text, return_tensors="pt")

    if full_ids.shape[1] > MAX_SEQ_LEN:
        full_ids = full_ids[:, :MAX_SEQ_LEN]

    code_start = prompt_len
    code_end = full_ids.shape[1]

    if code_end <= code_start:
        return None

    full_ids = full_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=full_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
    vectors = []
    for layer_idx in layers:
        hs = hidden_states[layer_idx]
        code_hs = hs[0, code_start:code_end, :]
        pooled = code_hs.float().mean(dim=0)
        vectors.append(pooled)

    return torch.stack(vectors, dim=0)  # (n_layers, hidden_dim)


def run_extraction():
    """Phase 1: extract hidden states for all formats."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load data
    with open(RESOLVED_TRIPLES) as f:
        triples = json.load(f)
    with open(PYTHON_SNIPPETS) as f:
        snippets = json.load(f)

    print(f"[extract] {len(triples)} triples, {len(snippets)} python snippets")
    assert len(triples) == N_TRIPLES, f"Expected {N_TRIPLES} triples, got {len(triples)}"

    # Resume support
    done_set = load_extract_done()
    if done_set:
        print(f"[RESUME] {len(done_set)} samples already extracted")

    # Load model
    device = torch.device("cuda:0")
    print(f"[extract] Loading {MODEL_NAME} (fp16)...")
    t0 = time.perf_counter()

    model_path = MODEL_LOCAL_PATH if os.path.isdir(MODEL_LOCAL_PATH) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[extract] Loaded in {time.perf_counter() - t0:.1f}s")

    # Verify architecture
    n_layers = model.config.num_hidden_layers
    h_dim = model.config.hidden_size
    print(f"[extract] num_hidden_layers={n_layers}, hidden_size={h_dim}")
    assert n_layers == 32, f"Expected 32 layers, got {n_layers}"
    assert h_dim == HIDDEN_DIM, f"Expected hidden_dim={HIDDEN_DIM}, got {h_dim}"

    # Extract for each format
    for fmt in ALL_FORMATS:
        out_dir = CACHE_DIR / fmt
        out_dir.mkdir(parents=True, exist_ok=True)

        n_ok, n_skip = 0, 0
        t_start = time.perf_counter()

        for i in range(N_TRIPLES):
            if (fmt, i) in done_set:
                n_ok += 1
                continue

            # Get caption and code
            if fmt == "python":
                caption = triples[i]["svg"]["caption"]
                code = snippets[i]["code"]
            else:
                caption = triples[i][fmt]["caption"]
                code = triples[i][fmt]["code"]

            if not code or not code.strip():
                n_skip += 1
                continue

            prompt_text = build_prompt(caption)
            result = extract_hidden_states(
                model, tokenizer, prompt_text, code, LAYERS, device
            )

            if result is None:
                print(f"  [SKIP] {fmt}/{i}: no code tokens after prompt")
                n_skip += 1
                continue

            save_path = out_dir / f"{i}.pt"
            torch.save(result.cpu(), save_path)
            append_ckpt(f"{fmt} {i} t={time.time()}")
            n_ok += 1

            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (N_TRIPLES - i - 1) / rate if rate > 0 else 0
                print(f"  [{fmt}] {i+1}/{N_TRIPLES} {rate:.1f} s/s ok={n_ok} skip={n_skip} ETA={eta:.0f}s")

        elapsed = time.perf_counter() - t_start
        print(f"[extract] {fmt}: {n_ok} saved, {n_skip} skipped, {elapsed:.1f}s")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Sanity check
    for fmt in ALL_FORMATS:
        sample = CACHE_DIR / fmt / "0.pt"
        if sample.exists():
            t = torch.load(sample, map_location="cpu", weights_only=True)
            assert t.shape == (len(LAYERS), HIDDEN_DIM), f"{fmt}/0.pt shape {t.shape}"
    print("[extract] Sanity check passed")


# ── CKA functions ──────────────────────────────────────────────────────
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
    """Permutation null: shuffle Y rows, recompute CKA."""
    observed = cka(X, Y)
    rng = np.random.RandomState(seed)
    null_dist = []
    for _ in range(n_perm):
        perm = rng.permutation(Y.shape[0])
        null_dist.append(cka(X, Y[perm]))
    null_dist = np.array(null_dist)
    p_value = float(np.mean(null_dist >= observed))
    return observed, p_value, float(null_dist.mean()), float(null_dist.std())


def bootstrap_ci(X, Y, n_boot=N_BOOT, seed=SEED, frac=0.8):
    """Subsampling 95% CI for CKA (without replacement)."""
    rng = np.random.RandomState(seed + 1)
    n = X.shape[0]
    m = int(n * frac)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, m, replace=False)
        boots.append(cka(X[idx], Y[idx]))
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_cka_analysis():
    """Phase 2: compute cross-format CKA for all pairs and layers."""
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load hidden states
    print("[cka] Loading hidden states...")
    data = {}
    for fmt in ALL_FORMATS:
        tensors = []
        d = CACHE_DIR / fmt
        for i in range(N_TRIPLES):
            t = torch.load(d / f"{i}.pt", map_location="cpu", weights_only=True)
            tensors.append(t.float().numpy())
        data[fmt] = np.stack(tensors, axis=0)  # (252, 3, 4096)
        print(f"  {fmt}: {data[fmt].shape}")

    # CKA for all pairs x layers
    results = []
    total = len(FORMAT_PAIRS) * len(LAYERS)
    done = 0

    for f1, f2 in FORMAT_PAIRS:
        for li, layer in enumerate(LAYERS):
            t0 = time.time()
            X = data[f1][:, li, :]  # (252, 4096)
            Y = data[f2][:, li, :]

            obs, p_val, null_mean, null_std = permutation_test(X, Y)
            ci_lo, ci_hi = bootstrap_ci(X, Y)

            row = {
                "format_pair": f"{f1}-{f2}",
                "layer": layer,
                "layer_depth_pct": round(layer / 32 * 100, 1),
                "cka": round(obs, 4),
                "p_value": p_val,
                "null_mean": round(null_mean, 4),
                "null_std": round(null_std, 4),
                "ci_95_lo": round(ci_lo, 4),
                "ci_95_hi": round(ci_hi, 4),
            }
            results.append(row)
            done += 1
            elapsed = time.time() - t0
            print(f"  [{done}/{total}] {f1}-{f2} L{layer}: CKA={obs:.4f} p={p_val:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] ({elapsed:.1f}s)")

    # Summary stats
    visual_ckas = [r["cka"] for r in results if r["format_pair"] in ("svg-tikz", "svg-asy", "tikz-asy")]
    python_ckas = [r["cka"] for r in results if r["format_pair"].startswith("python-")]

    summary = {
        "model": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "num_layers": 32,
        "hidden_dim": HIDDEN_DIM,
        "sampled_layers": LAYERS,
        "n_triples": N_TRIPLES,
        "n_perm": N_PERM,
        "n_boot": N_BOOT,
        "visual_cross_format_cka_mean": round(float(np.mean(visual_ckas)), 4),
        "python_cross_format_cka_mean": round(float(np.mean(python_ckas)), 4),
        "n_significant_visual": sum(1 for r in results if r["format_pair"] in ("svg-tikz", "svg-asy", "tikz-asy") and r["p_value"] < 0.05),
        "n_significant_python": sum(1 for r in results if r["format_pair"].startswith("python-") and r["p_value"] < 0.05),
        "run_ts": datetime.now(timezone.utc).isoformat(),
    }

    output = {"summary": summary, "results": results}
    atomic_write_json(OUT_DIR / "llama3_baseline_cka.json", output)
    print(f"\n[cka] Results saved to {OUT_DIR / 'llama3_baseline_cka.json'}")
    print(f"  Visual cross-format CKA mean: {summary['visual_cross_format_cka_mean']}")
    print(f"  Python cross-format CKA mean: {summary['python_cross_format_cka_mean']}")
    print(f"  Significant (p<0.05): visual={summary['n_significant_visual']}/9, python={summary['n_significant_python']}/9")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Llama-3-8B code-naive baseline probe")
    parser.add_argument("--phase", required=True, choices=["extract", "cka", "all"],
                        help="extract=hidden states, cka=CKA analysis, all=both")
    args = parser.parse_args()

    t_total = time.time()

    if args.phase in ("extract", "all"):
        run_extraction()

    if args.phase in ("cka", "all"):
        run_cka_analysis()

    print(f"\n[done] Total elapsed: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
