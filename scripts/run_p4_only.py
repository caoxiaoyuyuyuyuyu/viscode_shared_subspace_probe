#!/usr/bin/env python3
"""P4 CLIPScore validation only. Uses Qwen2.5-Coder to generate SVGs, then CLIP to score."""

import gc
import io
import json
import os
import sys
import time
import numpy as np

P4_PROMPTS = [
    "A red circle",
    "A blue square",
    "Three green triangles in a row",
    "A yellow star with five points",
    "Two overlapping rectangles",
]

SVG_INSTRUCTION = "Generate an SVG image of: {caption}\n\nRespond with ONLY the SVG code. Do not include any explanation, markdown formatting, or code fences. Start directly with <svg and end with </svg>."


def extract_svg(text: str) -> str:
    lower = text.lower()
    if "<svg" not in lower:
        return ""
    start = lower.index("<svg")
    svg = text[start:]
    end_lower = svg.lower()
    if "</svg>" in end_lower:
        end = end_lower.index("</svg>") + len("</svg>")
        svg = svg[:end]
    else:
        svg += "</svg>"
    return svg


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
    import torch
    import cairosvg
    from PIL import Image

    t_start = time.time()
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Generate SVGs
    print("Loading model for SVG generation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(model=model_name, dtype="half", max_model_len=4096,
              gpu_memory_utilization=0.7, trust_remote_code=True, enforce_eager=True)

    prompts_text = []
    for p in P4_PROMPTS:
        msg = SVG_INSTRUCTION.format(caption=p)
        messages = [{"role": "user", "content": msg}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_text.append(text)

    print("Generating 5 SVGs...", flush=True)
    params = SamplingParams(temperature=0.3, max_tokens=2048)
    outputs = llm.generate(prompts_text, params)
    svg_texts = [out.outputs[0].text.strip() for out in outputs]

    for i, txt in enumerate(svg_texts):
        has_svg = "<svg" in txt.lower()
        print(f"  [{i}] '{P4_PROMPTS[i]}': {'valid' if has_svg else 'INVALID'} ({len(txt)} chars)", flush=True)

    # Unload vLLM
    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    print("vLLM unloaded.", flush=True)

    # Render SVGs
    images = []
    valid_indices = []
    for i, txt in enumerate(svg_texts):
        svg_str = extract_svg(txt)
        if not svg_str:
            print(f"  [{i}] no valid SVG")
            continue
        try:
            png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"),
                                          output_width=224, output_height=224)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f"  [{i}] render failed: {e}")

    if len(valid_indices) < 5:
        print(f"Only {len(valid_indices)}/5 rendered, P4 FAIL")
        sys.exit(1)

    # Load CLIP (no cache_dir, rely on HF_HOME)
    print("Loading CLIP model...", flush=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device).eval()

    # Matched scores
    matched_scores = []
    for idx, img in zip(valid_indices, images):
        inputs = clip_proc(text=[P4_PROMPTS[idx]], images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip_model(**inputs)
        matched_scores.append(out.logits_per_image.item() / 100.0)

    # Shuffled scores
    shuffled_scores = []
    for i, (idx, img) in enumerate(zip(valid_indices, images)):
        shuffled_idx = valid_indices[(i + 1) % len(valid_indices)]
        inputs = clip_proc(text=[P4_PROMPTS[shuffled_idx]], images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip_model(**inputs)
        shuffled_scores.append(out.logits_per_image.item() / 100.0)

    del clip_model, clip_proc
    gc.collect()
    torch.cuda.empty_cache()

    diffs = [m - s for m, s in zip(matched_scores, shuffled_scores)]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-8
    snr = mean_diff / std_diff if std_diff > 0 else 0.0
    p4_pass = all(d > 0 for d in diffs) and snr > 2.0

    print(f"\nP4 RESULTS:", flush=True)
    print(f"  Matched:  {[round(s, 4) for s in matched_scores]}")
    print(f"  Shuffled: {[round(s, 4) for s in shuffled_scores]}")
    print(f"  Diffs:    {[round(d, 4) for d in diffs]}")
    print(f"  SNR:      {round(snr, 4)}")
    print(f"  Status:   {'PASS' if p4_pass else 'FAIL'}")
    print(f"  Runtime:  {(time.time()-t_start)/60:.1f} min", flush=True)

    result = {
        "matched_scores": [round(s, 4) for s in matched_scores],
        "shuffled_scores": [round(s, 4) for s in shuffled_scores],
        "paired_diffs": [round(d, 4) for d in diffs],
        "snr": round(snr, 4),
        "status": "PASS" if p4_pass else "FAIL",
        "svg_texts": svg_texts,
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/p4_clipscore_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to artifacts/p4_clipscore_results.json", flush=True)

    sys.exit(0 if p4_pass else 1)

if __name__ == "__main__":
    main()
