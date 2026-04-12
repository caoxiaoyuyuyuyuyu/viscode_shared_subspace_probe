#!/usr/bin/env python
"""Stage B: HF teacher-forcing probe — hidden state extraction.

For each (model, format, triple), teacher-forces reference code through
the model and extracts residual stream at equidistant layers.
Mean-pools over code-token positions → saves [n_layers_sampled, hidden_dim] per sample.

Supports multi-model extension: layers are auto-computed as equidistant
points from the model's num_hidden_layers (default 7 points).

Pre-reg: §3 (probe pool), §5 (aggregation), §12 (tool-stack split).

Usage:
  # Step 1: resolve triples (once, any GPU)
  python stage_b_probe.py --resolve-only \
      --triples-path outputs/stage_a/sbert_triples.json \
      --out-dir /root/autodl-tmp/cache/hidden_states

  # Step 2: extract hidden states (auto layers for any model)
  python stage_b_probe.py --model starcoder2 --format svg --gpu 0 \
      --triples-path outputs/stage_a/sbert_triples.json \
      --out-dir /root/autodl-tmp/cache/hidden_states

  # Step 2 (alt): explicit layer override
  python stage_b_probe.py --model codestral --format tikz --gpu 0 \
      --layers 5,10,15,20,25,30,35 \
      --triples-path outputs/stage_a/sbert_triples.json \
      --out-dir /root/autodl-tmp/cache/hidden_states
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force HuggingFace offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")

sys.stdout.reconfigure(line_buffering=True)


# ── Config ─────────────────────────────────────────────────────────────
DEFAULT_N_LAYER_POINTS = 7  # number of equidistant layers to sample
LEGACY_LAYERS = [4, 8, 12, 16, 20, 24, 28]  # original Qwen2.5-7B (28 layers) config

MODEL_REGISTRY = {
    # Original Qwen2.5 family (28 layers, hidden_dim=3584)
    "coder":       {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "type": "chat",
                    "layers": [4, 8, 12, 16, 20, 24, 28]},
    "viscoder2":   {"name": "TIGER-Lab/VisCoder2-7B",          "type": "chat",
                    "layers": [4, 8, 12, 16, 20, 24, 28]},
    "qwen25":      {"name": "Qwen/Qwen2.5-7B",                 "type": "base",
                    "layers": [4, 8, 12, 16, 20, 24, 28]},
    # Multi-model extension (D035/D036/D037)
    "codestral":   {"name": "mistralai/Codestral-22B-v0.1",         "type": "chat",
                    "layers": [8, 16, 24, 32, 40, 48, 56]},       # 56L
    "starcoder2":  {"name": "bigcode/starcoder2-15b-instruct-v0.1", "type": "chat",
                    "layers": [6, 12, 16, 23, 28, 34, 40]},       # 40L
    "deepseek":    {"name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "type": "chat",
                    "layers": [4, 8, 12, 15, 19, 23, 27]},        # 27L
}


def compute_equidistant_layers(n_layers, n_points=DEFAULT_N_LAYER_POINTS):
    """Compute n_points equidistant 1-indexed layer numbers spanning 1..n_layers."""
    return [round(n_layers * (i + 1) / n_points) for i in range(n_points)]

# Data paths on server (matching step2_sbert_matching.py)
DATA_ROOT = "/root/autodl-tmp/viscode_shared_subspace_probe/data"
VCM_PATH = f"{DATA_ROOT}/VisCode_filtered/"
DATIKZ_V2_PATH = f"{DATA_ROOT}/datikz_v2/"

# Sampling constants — must match step2_sbert_matching.py exactly
STEP2_SEED = 20260410
PROBE_EMBED_SAMPLE_SVG = 50_000


def parse_args():
    p = argparse.ArgumentParser(description="Stage B: HF teacher-forcing probe")
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                   help="Model to probe (omit with --resolve-only)")
    p.add_argument("--format", choices=["svg", "tikz", "asy", "python"],
                   help="Format to probe (omit with --resolve-only)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index for logging/audit only; actual device chosen via external CUDA_VISIBLE_DEVICES")
    p.add_argument("--triples-path", required=True,
                   help="Path to sbert_triples.json")
    p.add_argument("--out-dir", required=True,
                   help="Output root for hidden state tensors")
    p.add_argument("--resolved-cache", default="",
                   help="Path to resolved triples JSON (auto-derived if empty)")
    p.add_argument("--resolve-only", action="store_true",
                   help="Only resolve triples (no model loading)")
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Max sequence length (truncate code if exceeded)")
    p.add_argument("--layers", type=str, default="",
                   help="Explicit comma-separated layer indices (e.g. '4,8,12,16,20,24,28'). "
                        "If empty, auto-computes equidistant layers from model config.")
    p.add_argument("--n-layer-points", type=int, default=DEFAULT_N_LAYER_POINTS,
                   help=f"Number of equidistant layers to sample (default {DEFAULT_N_LAYER_POINTS}). "
                        "Ignored if --layers is set.")
    return p.parse_args()


# ── Triple resolution ──────────────────────────────────────────────────
# Resolves step2 triple indices → (caption, code) for each format.
# Must reproduce step2_sbert_matching.py sampling to map indices back
# to original dataset rows and extract reference code.

def resolve_probe_triples(triples_path, cache_path):
    """Resolve triple indices to actual (caption, code) pairs."""
    import datasets

    with open(triples_path) as f:
        triples = json.load(f)
    print(f"[resolve] {len(triples)} triples to resolve")

    # ── SVG: SVGX-Core-250k ──
    print("[resolve] Loading SVGX-Core-250k...")
    svgx_ds = datasets.load_dataset("xingxm/SVGX-Core-250k", split="train")
    svg_valid_indices = []
    for i, cap in enumerate(svgx_ds["qwen_caption"]):
        if cap and isinstance(cap, str) and cap.strip():
            svg_valid_indices.append(i)
    print(f"  {len(svg_valid_indices)}/{len(svgx_ds)} valid captions")

    # Reproduce step2 sampling (random.Random with same seed)
    rng = random.Random(STEP2_SEED)
    if len(svg_valid_indices) > PROBE_EMBED_SAMPLE_SVG:
        svg_sampled = rng.sample(svg_valid_indices, PROBE_EMBED_SAMPLE_SVG)
    else:
        svg_sampled = svg_valid_indices
    print(f"  SVG sampled: {len(svg_sampled)}")

    # ── TikZ: DaTikZ v2 train (no sampling — PROBE_EMBED_SAMPLE_TIKZ=0) ──
    print("[resolve] Loading DaTikZ v2 train...")
    tikz_ds = datasets.load_dataset("nllg/datikz-v2", split="train")
    tikz_captions = tikz_ds["caption"]  # columnar access (avoid O(n) row copy)
    tikz_valid_indices = []
    for i, cap in enumerate(tikz_captions):
        if cap and isinstance(cap, str) and cap.strip():
            tikz_valid_indices.append(i)
    print(f"  {len(tikz_valid_indices)}/{len(tikz_ds)} valid captions")

    # ── Asy: VCM-asy (no sampling — PROBE_EMBED_SAMPLE_ASY=0) ──
    print("[resolve] Loading VCM-asy...")
    vcm = datasets.load_from_disk(VCM_PATH)
    asy_ds = vcm.filter(lambda x: x["language"] == "asymptote", num_proc=1)
    asy_messages = asy_ds["messages"]  # columnar access (avoid O(n) row copy)
    asy_valid_indices = []
    asy_code_cache = {}
    for i, msgs in enumerate(asy_messages):
        caption, code = "", ""
        for m in msgs:
            if m["role"] == "user":
                caption = m["content"]
            elif m["role"] == "assistant":
                code = m["content"]
        if caption and caption.strip():
            asy_valid_indices.append(i)
            asy_code_cache[i] = code
    print(f"  {len(asy_valid_indices)}/{len(asy_ds)} valid captions")

    # ── Resolve each triple ──
    resolved = []
    for i, t in enumerate(triples):
        tid = i  # sbert_triples.json has no triple_id field; use enumerate index

        # SVG: svg_idx → index into svg_sampled → original dataset index
        svg_orig_idx = svg_sampled[t["svg_idx"]]
        svg_code = svgx_ds[svg_orig_idx]["svg_code"]

        # TikZ: tikz_idx → index into tikz_valid_indices → original dataset index
        tikz_orig_idx = tikz_valid_indices[t["tikz_idx"]]
        tikz_code = tikz_ds[tikz_orig_idx]["code"]

        # Asy: asy_idx → index into asy_valid_indices → original dataset index
        asy_orig_idx = asy_valid_indices[t["asy_idx"]]
        asy_code = asy_code_cache[asy_orig_idx]

        resolved.append({
            "triple_id": tid,
            "min_cosine": t["min_cosine"],
            "svg": {"caption": t["svg_caption"], "code": svg_code,
                    "orig_idx": svg_orig_idx},
            "tikz": {"caption": t["tikz_caption"], "code": tikz_code,
                     "orig_idx": tikz_orig_idx},
            "asy": {"caption": t["asy_caption"], "code": asy_code,
                    "orig_idx": asy_orig_idx},
        })

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(resolved, f, ensure_ascii=False)
    print(f"[resolve] Saved {len(resolved)} resolved triples → {cache_path}")

    del svgx_ds, tikz_ds, asy_ds, vcm
    gc.collect()
    return resolved


# ── Hidden state extraction ───────────────────────────────────────────

def build_prompt(caption, model_type, tokenizer):
    """Build teacher-forcing prompt (caption only, per pre-reg §3)."""
    if model_type == "chat":
        messages = [{"role": "user", "content": caption}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return caption + "\n"


def extract_hidden_states(model, tokenizer, prompt_text, code_text,
                          layers, max_seq_len, device):
    """Teacher-force prompt+code, extract and mean-pool code-token hidden states.

    Returns tensor [len(layers), hidden_dim] or None if no code tokens.
    """
    import torch

    # Tokenize prompt to find boundary
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]

    # Tokenize full sequence (prompt + code)
    full_text = prompt_text + code_text
    full_ids = tokenizer.encode(full_text, return_tensors="pt")

    # Truncate if exceeds max_seq_len
    if full_ids.shape[1] > max_seq_len:
        full_ids = full_ids[:, :max_seq_len]

    code_start = prompt_len
    code_end = full_ids.shape[1]

    if code_end <= code_start:
        return None

    full_ids = full_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=full_ids, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, [0]=embedding, [i]=layer i
    hidden_states = outputs.hidden_states

    vectors = []
    for layer_idx in layers:
        if layer_idx >= len(hidden_states):
            raise ValueError(
                f"layer {layer_idx} OOB, model has {len(hidden_states)} hidden states "
                f"(valid range 0..{len(hidden_states)-1}). Pre-reg LAYERS must not exceed model depth."
            )
        hs = hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
        code_hs = hs[0, code_start:code_end, :]  # [code_len, hidden_dim]
        pooled = code_hs.float().mean(dim=0)  # [hidden_dim] in fp32
        vectors.append(pooled)

    return torch.stack(vectors, dim=0)  # [n_layers, hidden_dim]


def main():
    args = parse_args()

    # ── Resolve triples ──
    cache_path = args.resolved_cache or os.path.join(
        os.path.dirname(args.triples_path), "resolved_triples.json"
    )

    if os.path.exists(cache_path):
        print(f"[stage_b] Loading resolved triples from {cache_path}")
        with open(cache_path) as f:
            resolved = json.load(f)
        print(f"[stage_b] {len(resolved)} resolved triples")
    else:
        print(f"[stage_b] Resolving triples (first run)...")
        resolved = resolve_probe_triples(args.triples_path, cache_path)

    if args.resolve_only:
        print("[stage_b] --resolve-only: done.")
        return

    # ── Validate required args ──
    if not args.model or not args.format:
        print("ERROR: --model and --format required unless --resolve-only", file=sys.stderr)
        sys.exit(1)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda:0")

    model_cfg = MODEL_REGISTRY[args.model]
    fmt = args.format

    print(f"[stage_b] model={args.model} format={fmt} gpu={args.gpu} "
          f"triples={len(resolved)}")

    # ── Load model (fp16, no quantization — pre-reg §12) ──
    print(f"[stage_b] Loading {model_cfg['name']} (fp16)...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    load_time = time.perf_counter() - t0
    print(f"[stage_b] Loaded in {load_time:.1f}s — "
          f"layers={n_layers}, hidden_dim={hidden_dim}")

    # Determine which layers to probe (priority: CLI > registry > auto-compute)
    if args.layers:
        LAYERS = [int(x.strip()) for x in args.layers.split(",")]
        print(f"[stage_b] Using explicit CLI layers: {LAYERS}")
    elif "layers" in model_cfg:
        LAYERS = model_cfg["layers"]
        print(f"[stage_b] Using registry layers for {args.model}: {LAYERS}")
    else:
        LAYERS = compute_equidistant_layers(n_layers, args.n_layer_points)
        print(f"[stage_b] Auto-computed {args.n_layer_points} equidistant layers "
              f"for {n_layers}-layer model: {LAYERS}")

    # Validate requested layers exist (fail-fast, no silent clamp)
    for l in LAYERS:
        if l > n_layers:
            raise ValueError(
                f"LAYERS config error: layer {l} > model n_layers={n_layers}. "
                f"Pre-reg LAYERS must not exceed model depth."
            )

    # ── Process all triples for this format ──
    out_dir = Path(args.out_dir) / args.model / fmt
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    n_total = len(resolved)
    t_start = time.perf_counter()

    for i, triple in enumerate(resolved):
        tid = triple["triple_id"]
        entry = triple[fmt]
        caption = entry["caption"]
        code = entry["code"]

        if not code or not code.strip():
            print(f"  [SKIP] triple {tid}: empty code")
            n_skip += 1
            continue

        prompt_text = build_prompt(caption, model_cfg["type"], tokenizer)

        result = extract_hidden_states(
            model, tokenizer, prompt_text, code,
            LAYERS, args.max_seq_len, device
        )

        if result is None:
            print(f"  [SKIP] triple {tid}: no code tokens after prompt")
            n_skip += 1
            continue

        # Save [7, hidden_dim] tensor
        save_path = out_dir / f"{tid}.pt"
        torch.save(result.cpu(), save_path)
        n_ok += 1

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_total}] {rate:.1f} samples/s, "
                  f"ok={n_ok} skip={n_skip}, ETA {eta:.0f}s")

    elapsed = time.perf_counter() - t_start
    print(f"\n[stage_b] Done: {n_ok} saved, {n_skip} skipped, "
          f"{elapsed:.1f}s ({n_ok/elapsed:.1f} samples/s)" if elapsed > 0
          else f"\n[stage_b] Done: {n_ok} saved, {n_skip} skipped")

    # ── Post-run pairwise row distinctness sanity assertion ──
    sample_pts = sorted(out_dir.glob("*.pt"))[:3]
    for pt in sample_pts:
        t = torch.load(pt)  # expected [7, hidden_dim]
        assert t.shape[0] == len(LAYERS), (
            f"Sanity FAIL: {pt.name} shape {t.shape}, expected [{len(LAYERS)}, *]"
        )
        L = t.shape[0]
        for i in range(L):
            for j in range(i + 1, L):
                diff = (t[i] - t[j]).abs().sum().item()
                assert diff > 1e-3, (
                    f"Sanity FAIL: {pt.name} row {i}==row {j} (L1 diff={diff})"
                )
    if sample_pts:
        print(f"[sanity] {len(sample_pts)} files × {L*(L-1)//2} pairs all distinct ✓",
              flush=True)

    # ── Write summary ──
    summary = {
        "model": args.model,
        "model_name": model_cfg["name"],
        "format": fmt,
        "layers": LAYERS,
        "layers_resolved": list(LAYERS),
        "model_n_layers": n_layers,
        "n_layers_model": n_layers,
        "hidden_dim": hidden_dim,
        "n_triples": n_total,
        "n_saved": n_ok,
        "n_skipped": n_skip,
        "elapsed_s": round(elapsed, 1),
        "tensor_shape": [len(LAYERS), hidden_dim],
        "max_seq_len": args.max_seq_len,
        "gpu": args.gpu,
        "assigned_gpu": args.gpu,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "torch_current_device": torch.cuda.current_device() if torch.cuda.is_available() else -1,
        "run_ts": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stage_b] Summary → {summary_path}")

    # ── Cleanup ──
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
