#!/usr/bin/env python
"""Negative Control Probe: Extract hidden states for Python code snippets.

Uses the same 3 models as stage_b_probe.py but with non-visual Python code
instead of visual code formats (SVG/TikZ/Asy). Captions come from the
corresponding SVG entry in resolved_triples.json.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/negative_control_probe.py --model coder
  CUDA_VISIBLE_DEVICES=0 python scripts/negative_control_probe.py --model viscoder2
  CUDA_VISIBLE_DEVICES=0 python scripts/negative_control_probe.py --model qwen25
"""

import argparse
import gc
import json
import os
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
LAYERS = [4, 8, 12, 16, 20, 24, 28]

MODEL_REGISTRY = {
    "coder":     {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "type": "chat"},
    "viscoder2": {"name": "TIGER-Lab/VisCoder2-7B",          "type": "chat"},
    "qwen25":    {"name": "Qwen/Qwen2.5-7B",                 "type": "base"},
}

CACHE_DIR = Path("/root/autodl-tmp/cache/hidden_states")
PYTHON_SNIPPETS = CACHE_DIR / "python_snippets.json"
RESOLVED_TRIPLES = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/stage_a/resolved_triples.json")
MAX_SEQ_LEN = 4096


def build_prompt(caption, model_type, tokenizer):
    """Build teacher-forcing prompt (caption only, per pre-reg)."""
    if model_type == "chat":
        messages = [{"role": "user", "content": caption}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return caption + "\n"


def extract_hidden_states(model, tokenizer, prompt_text, code_text, layers, device):
    """Teacher-force prompt+code, extract and mean-pool code-token hidden states."""
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

    hidden_states = outputs.hidden_states
    vectors = []
    for layer_idx in layers:
        hs = hidden_states[layer_idx]
        code_hs = hs[0, code_start:code_end, :]
        pooled = code_hs.float().mean(dim=0)
        vectors.append(pooled)

    return torch.stack(vectors, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Load data
    with open(PYTHON_SNIPPETS) as f:
        snippets = json.load(f)
    with open(RESOLVED_TRIPLES) as f:
        triples = json.load(f)

    n_samples = len(snippets)
    print(f"[neg_ctrl] {n_samples} Python snippets, {len(triples)} triples")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda:0")
    model_cfg = MODEL_REGISTRY[args.model]

    print(f"[neg_ctrl] Loading {model_cfg['name']} (fp16)...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"[neg_ctrl] Loaded in {load_time:.1f}s")

    # Output directory
    out_dir = CACHE_DIR / args.model / "python"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    t_start = time.perf_counter()

    for i, snippet in enumerate(snippets):
        tid = snippet["triple_id"]
        code = snippet["code"]

        if not code or not code.strip():
            n_skip += 1
            continue

        # Use caption from the SVG entry of the corresponding triple
        caption = triples[tid]["svg"]["caption"]
        prompt_text = build_prompt(caption, model_cfg["type"], tokenizer)

        result = extract_hidden_states(model, tokenizer, prompt_text, code, LAYERS, device)

        if result is None:
            print(f"  [SKIP] {tid}: no code tokens after prompt")
            n_skip += 1
            continue

        save_path = out_dir / f"{tid}.pt"
        torch.save(result.cpu(), save_path)
        n_ok += 1

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_samples}] {rate:.1f} samples/s, ok={n_ok} skip={n_skip}, ETA {eta:.0f}s")

    elapsed = time.perf_counter() - t_start
    print(f"[neg_ctrl] Done: {n_ok} saved, {n_skip} skipped, {elapsed:.1f}s")

    # Sanity check
    sample_pts = sorted(out_dir.glob("*.pt"))[:3]
    for pt in sample_pts:
        t = torch.load(pt)
        assert t.shape == (len(LAYERS), 3584), f"Shape mismatch: {pt.name} {t.shape}"
    if sample_pts:
        print(f"[sanity] {len(sample_pts)} files shape OK")

    # Summary
    summary = {
        "model": args.model, "format": "python", "n_samples": n_samples,
        "n_saved": n_ok, "n_skipped": n_skip, "elapsed_s": round(elapsed, 1),
        "layers": LAYERS, "run_ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
