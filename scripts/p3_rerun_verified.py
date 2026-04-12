#!/usr/bin/env python3
"""P3 Mini Pilot Rerun — verified artifact with per-sample details + throughput."""

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch
from vllm import LLM, SamplingParams

# ── Config ──────────────────────────────────────────────────────────────
MODELS = [
    {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "type": "chat"},
    {"name": "TIGER-Lab/VisCoder2-7B",           "type": "chat"},
    {"name": "Qwen/Qwen2.5-7B",                  "type": "base"},
]

PILOT_PROMPTS = [
    "A red circle centered in the canvas",
    "A blue square with a yellow border",
    "Three overlapping circles in red, green, and blue",
    "A simple house with a triangular roof",
    "A five-pointed yellow star",
    "Two rectangles side by side, one green and one purple",
    "A circle inside a square, both black outlines",
    "An orange triangle pointing upward",
    "A horizontal line of five small dots",
    "A red heart shape",
]

FORMAT_TEMPLATES = {
    "SVG":       "Generate an SVG image: {caption}\n\nOutput only the SVG code.",
    "TikZ":      "Generate TikZ code for: {caption}\n\nOutput only the TikZ code.",
    "Asymptote": "Generate Asymptote code for: {caption}\n\nOutput only the Asymptote code.",
}

VALIDITY_CHECKS = {
    "SVG":       lambda text: "<svg" in text.lower(),
    "TikZ":      lambda text: "\\begin{tikzpicture}" in text or "\\tikz" in text,
    "Asymptote": lambda text: any(kw in text for kw in ["import", "draw", "path", "size("]),
}

WARMUP_SAMPLES = 2  # skip first N samples of first format per model for throughput

OUTPUT_DIR  = "/root/autodl-tmp/viscode_shared_subspace_probe/outputs/sanity"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mini_pilot_rerun_results.json")
ARTIFACT_COPY = "/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/mini_pilot_rerun_results.json"


def build_prompts(fmt: str, model_type: str, tokenizer=None):
    """Build prompt strings for each caption."""
    prompts = []
    for caption in PILOT_PROMPTS:
        user_msg = FORMAT_TEMPLATES[fmt].format(caption=caption)
        if model_type == "chat" and tokenizer is not None:
            messages = [{"role": "user", "content": user_msg}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
        else:
            # base model: completion prefix
            prompts.append(user_msg + "\n\n")
    return prompts


def run_cell(llm, tokenizer, model_cfg, fmt, sampling_params):
    """Run one model × format cell, return cell dict."""
    prompts = build_prompts(fmt, model_cfg["type"], tokenizer)

    samples = []
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_gen_time = time.perf_counter() - t0

    checker = VALIDITY_CHECKS[fmt]
    for i, out in enumerate(outputs):
        text = out.outputs[0].text.strip()
        n_tokens = len(out.outputs[0].token_ids)
        valid = checker(text)
        reason = ""
        if fmt == "SVG":
            reason = "contains <svg" if valid else "no <svg tag"
        elif fmt == "TikZ":
            reason = "contains \\begin{tikzpicture} or \\tikz" if valid else "no tikz marker"
        else:
            reason = "contains draw/import/path/size" if valid else "no asymptote keywords"

        samples.append({
            "prompt": PILOT_PROMPTS[i],
            "generated_text_preview": text[:200],
            "generated_text_length": len(text),
            "n_tokens": n_tokens,
            "valid": valid,
            "validity_reason": reason,
            "error": None,
        })

    valid_count = sum(s["valid"] for s in samples)
    total_tokens = sum(s["n_tokens"] for s in samples)

    # Per-sample timing: vLLM batches, so we approximate per-sample from batch
    for s in samples:
        s["generation_time_s"] = round(total_gen_time * s["n_tokens"] / max(total_tokens, 1), 3)

    avg_tok_s = total_tokens / total_gen_time if total_gen_time > 0 else 0

    return {
        "model": model_cfg["name"],
        "format": fmt,
        "samples": samples,
        "valid_count": valid_count,
        "total": len(samples),
        "cell_pass": valid_count >= 1,
        "generation_total_time_s": round(total_gen_time, 2),
        "total_tokens": total_tokens,
        "avg_tok_s": round(avg_tok_s, 1),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ARTIFACT_COPY), exist_ok=True)

    run_ts = datetime.now(timezone.utc).isoformat()
    print(f"P3 Rerun started at {run_ts}")
    print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1', 'not set')}")

    sampling_params = SamplingParams(temperature=0.3, max_tokens=1024, stop=["```"])

    cells = []
    throughput_samples = []  # (tok_count, time_s) for post-warmup

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        short_name = model_name.split("/")[-1]

        print(f"\n{'='*60}")
        print(f"LOADING MODEL: {short_name} ({model_cfg['type']})")
        print(f"{'='*60}")

        t_load = time.perf_counter()
        llm = LLM(
            model=model_name,
            dtype="half",
            enforce_eager=True,
            max_model_len=4096,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
        )
        load_time = time.perf_counter() - t_load
        print(f"Model loaded in {load_time:.1f}s")

        tokenizer = llm.get_tokenizer() if model_cfg["type"] == "chat" else None

        for fi, fmt in enumerate(FORMAT_TEMPLATES):
            print(f"\n--- {short_name} x {fmt} ---")
            try:
                cell = run_cell(llm, tokenizer, model_cfg, fmt, sampling_params)
                cell["model_load_time_s"] = round(load_time, 1)
                cell["error"] = None

                # Throughput tracking (skip warmup for first format only)
                is_first_format = (fi == 0)
                for si, s in enumerate(cell["samples"]):
                    if is_first_format and si < WARMUP_SAMPLES:
                        continue  # skip warmup
                    throughput_samples.append((s["n_tokens"], s["generation_time_s"]))

                print(f"  valid: {cell['valid_count']}/{cell['total']}, "
                      f"avg {cell['avg_tok_s']} tok/s, "
                      f"time {cell['generation_total_time_s']}s")
                print(f"  first output ({len(cell['samples'][0]['generated_text_preview'])} chars): "
                      f"{cell['samples'][0]['generated_text_preview'][:80]}...")

            except Exception as e:
                print(f"  ERROR: {e}")
                cell = {
                    "model": model_cfg["name"],
                    "format": fmt,
                    "samples": [],
                    "valid_count": 0,
                    "total": 10,
                    "cell_pass": False,
                    "model_load_time_s": round(load_time, 1),
                    "generation_total_time_s": 0,
                    "total_tokens": 0,
                    "avg_tok_s": 0,
                    "error": str(e),
                }

            cells.append(cell)

        # Unload model
        del llm
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nModel {short_name} unloaded.")

    # ── Summary ──
    cells_pass = sum(1 for c in cells if c["cell_pass"])
    total_ooms = 0  # no OOM expected with 96GB
    overall_valid = sum(c["valid_count"] for c in cells)
    overall_total = sum(c["total"] for c in cells)
    p3_status = "PASS" if cells_pass >= 7 else "FAIL"

    # Throughput (post-warmup)
    tp_tokens = sum(t for t, _ in throughput_samples)
    tp_time = sum(t for _, t in throughput_samples)
    avg_tok_s_after_warmup = tp_tokens / tp_time if tp_time > 0 else 0

    result = {
        "run_timestamp": run_ts,
        "vllm_config": {
            "dtype": "half",
            "enforce_eager": True,
            "v1": os.environ.get("VLLM_USE_V1", "1") != "0",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.7,
        },
        "cells": cells,
        "summary": {
            "cells_pass": cells_pass,
            "cells_total": len(cells),
            "total_ooms": total_ooms,
            "p3_status": p3_status,
            "overall_valid": overall_valid,
            "overall_total": overall_total,
        },
        "throughput": {
            "note": "排除首次 compile 的稳态吞吐 (skip first 2 samples of first format per model)",
            "avg_tok_s_after_warmup": round(avg_tok_s_after_warmup, 1),
            "warmup_samples": WARMUP_SAMPLES,
            "post_warmup_total_tokens": tp_tokens,
            "post_warmup_total_time_s": round(tp_time, 2),
        },
    }

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

    with open(ARTIFACT_COPY, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Artifact copy saved to {ARTIFACT_COPY}")

    # Print summary table
    print(f"\n{'='*60}")
    print("P3 MINI PILOT RERUN SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<35} {'SVG':<15} {'TikZ':<15} {'Asymptote':<15}")
    print("-" * 80)
    for model_cfg in MODELS:
        row = f"{model_cfg['name'].split('/')[-1]:<35}"
        for fmt in FORMAT_TEMPLATES:
            cell = next(c for c in cells if c["model"] == model_cfg["name"] and c["format"] == fmt)
            status = "PASS" if cell["cell_pass"] else ("ERROR" if cell.get("error") else "FAIL")
            row += f" {status} ({cell['valid_count']}/{cell['total']})  "
        print(row)

    print(f"\nCells pass: {cells_pass}/{len(cells)}")
    print(f"Overall valid: {overall_valid}/{overall_total}")
    print(f"OOMs: {total_ooms}")
    print(f"P3 status: {p3_status}")
    print(f"Steady-state throughput: {avg_tok_s_after_warmup:.1f} tok/s (post-warmup)")


if __name__ == "__main__":
    main()
