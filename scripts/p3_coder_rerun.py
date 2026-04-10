#!/usr/bin/env python3
"""P3 Coder Rerun — fix stop token bug, format-specific stops.

Only runs Qwen2.5-Coder-7B-Instruct × 3 formats.
- SVG: stop=["</svg>"], post-process append </svg>
- TikZ: stop=["\\end{tikzpicture}"], post-process append \\end{tikzpicture}
- Asymptote: no stop token, max_tokens=2048
"""

import gc
import json
import os
import time
from datetime import datetime, timezone

import torch
from vllm import LLM, SamplingParams

# ── Config ──────────────────────────────────────────────────────────────
MODEL = {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "type": "chat"}

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

FORMAT_CONFIGS = {
    "SVG": {
        "template": "Generate an SVG image: {caption}\n\nOutput only the SVG code.",
        "sampling_params": SamplingParams(
            temperature=0.3, max_tokens=1024, stop=["</svg>"]
        ),
        "post_process": lambda text: text.rstrip() + "\n</svg>" if "</svg>" not in text else text,
        "validity": lambda text: "<svg" in text.lower(),
    },
    "TikZ": {
        "template": "Generate TikZ code for: {caption}\n\nOutput only the TikZ code.",
        "sampling_params": SamplingParams(
            temperature=0.3, max_tokens=1024, stop=["\\end{tikzpicture}"]
        ),
        "post_process": lambda text: text.rstrip() + "\n\\end{tikzpicture}" if "\\end{tikzpicture}" not in text else text,
        "validity": lambda text: "\\begin{tikzpicture}" in text or "\\tikz" in text,
    },
    "Asymptote": {
        "template": "Generate Asymptote code for: {caption}\n\nOutput only the Asymptote code.",
        "sampling_params": SamplingParams(
            temperature=0.3, max_tokens=2048
        ),
        "post_process": lambda text: text,
        "validity": lambda text: any(kw in text for kw in ["import", "draw", "path", "size("]),
    },
}

WARMUP_SAMPLES = 2
OUTPUT_DIR = "/root/autodl-tmp/viscode_shared_subspace_probe/outputs/sanity"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "p3_coder_rerun_results.json")
ARTIFACT_COPY = "/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/p3_coder_rerun_results.json"


def build_prompts(template, tokenizer):
    prompts = []
    for caption in PILOT_PROMPTS:
        user_msg = template.format(caption=caption)
        messages = [{"role": "user", "content": user_msg}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


def run_cell(llm, tokenizer, fmt, cfg):
    prompts = build_prompts(cfg["template"], tokenizer)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, cfg["sampling_params"])
    total_gen_time = time.perf_counter() - t0

    samples = []
    for i, out in enumerate(outputs):
        raw_text = out.outputs[0].text.strip()
        text = cfg["post_process"](raw_text)
        n_tokens = len(out.outputs[0].token_ids)
        valid = cfg["validity"](text)

        if fmt == "SVG":
            reason = "contains <svg" if valid else "no <svg tag"
        elif fmt == "TikZ":
            reason = "contains \\begin{tikzpicture} or \\tikz" if valid else "no tikz marker"
        else:
            reason = "contains draw/import/path/size" if valid else "no asymptote keywords"

        samples.append({
            "prompt": PILOT_PROMPTS[i],
            "generated_text_preview": text[:300],
            "generated_text_length": len(text),
            "n_tokens": n_tokens,
            "valid": valid,
            "validity_reason": reason,
            "error": None,
        })

    valid_count = sum(s["valid"] for s in samples)
    total_tokens = sum(s["n_tokens"] for s in samples)

    for s in samples:
        s["generation_time_s"] = round(total_gen_time * s["n_tokens"] / max(total_tokens, 1), 3)

    avg_tok_s = total_tokens / total_gen_time if total_gen_time > 0 else 0

    return {
        "model": MODEL["name"],
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
    print(f"P3 Coder Rerun started at {run_ts}")
    print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1', 'not set')}")
    print(f"Fix: format-specific stop tokens (no global stop=['```'])")

    t_load = time.perf_counter()
    llm = LLM(
        model=MODEL["name"],
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
    )
    load_time = time.perf_counter() - t_load
    print(f"Model loaded in {load_time:.1f}s")

    tokenizer = llm.get_tokenizer()

    cells = []
    throughput_samples = []

    for fi, (fmt, cfg) in enumerate(FORMAT_CONFIGS.items()):
        print(f"\n--- Qwen2.5-Coder x {fmt} ---")
        try:
            cell = run_cell(llm, tokenizer, fmt, cfg)
            cell["model_load_time_s"] = round(load_time, 1)
            cell["error"] = None

            is_first_format = (fi == 0)
            for si, s in enumerate(cell["samples"]):
                if is_first_format and si < WARMUP_SAMPLES:
                    continue
                throughput_samples.append((s["n_tokens"], s["generation_time_s"]))

            print(f"  valid: {cell['valid_count']}/{cell['total']}, "
                  f"avg {cell['avg_tok_s']} tok/s, "
                  f"time {cell['generation_total_time_s']}s")
            for si, s in enumerate(cell["samples"]):
                tag = "OK" if s["valid"] else "FAIL"
                print(f"  [{tag}] #{si} ({s['n_tokens']} tok): {s['generated_text_preview'][:80]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            cell = {
                "model": MODEL["name"],
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

    # Summary
    cells_pass = sum(1 for c in cells if c["cell_pass"])
    overall_valid = sum(c["valid_count"] for c in cells)
    overall_total = sum(c["total"] for c in cells)

    tp_tokens = sum(t for t, _ in throughput_samples)
    tp_time = sum(t for _, t in throughput_samples)
    avg_tok_s = tp_tokens / tp_time if tp_time > 0 else 0

    result = {
        "run_timestamp": run_ts,
        "fix_description": "format-specific stop tokens: SVG→</svg>, TikZ→\\end{tikzpicture}, Asymptote→none(max_tokens=2048)",
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
            "p3_status": "PASS" if cells_pass >= 2 else "FAIL",
            "overall_valid": overall_valid,
            "overall_total": overall_total,
        },
        "throughput": {
            "note": "steady-state throughput (skip first 2 samples of first format)",
            "avg_tok_s": round(avg_tok_s, 1),
            "post_warmup_total_tokens": tp_tokens,
            "post_warmup_total_time_s": round(tp_time, 2),
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

    with open(ARTIFACT_COPY, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Artifact copy saved to {ARTIFACT_COPY}")

    # Print summary
    print(f"\n{'='*60}")
    print("P3 CODER RERUN SUMMARY (stop token fix)")
    print(f"{'='*60}")
    print(f"{'Format':<15} {'Valid':<15} {'tok/s':<15} {'Time':<15}")
    print("-" * 60)
    for c in cells:
        print(f"{c['format']:<15} {c['valid_count']}/{c['total']:<13} {c['avg_tok_s']:<15} {c['generation_total_time_s']}s")
    print(f"\nOverall: {overall_valid}/{overall_total} valid")
    print(f"Steady-state throughput: {avg_tok_s:.1f} tok/s")


if __name__ == "__main__":
    main()
