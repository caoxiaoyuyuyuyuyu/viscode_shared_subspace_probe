#!/usr/bin/env python3
"""P4 CLIPScore validation with 20 prompts (5 dimensions × 4)."""

import json, time, gc, io, re, sys
import torch
import numpy as np

PROMPTS_20 = [
    # Color (4)
    "A solid red circle",
    "A filled blue rectangle",
    "A green triangle with green fill",
    "A yellow five-pointed star",
    # Shape count (4)
    "A single large hexagon",
    "Two circles side by side",
    "Three squares in a horizontal row",
    "Five small diamonds arranged in a line",
    # Quadrant (4)
    "A small circle in the top-left corner",
    "A small square in the top-right corner",
    "A small triangle in the bottom-left corner",
    "A small diamond in the bottom-right corner",
    # Spatial (4)
    "Two overlapping circles",
    "A square next to a triangle",
    "A small circle inside a large square",
    "A circle and a square far apart",
    # Stroke-fill (4)
    "A filled red square with no border",
    "A circle with only a black outline and no fill",
    "A blue rectangle with a red border",
    "Three hollow circles with thick outlines",
]

DIMENSIONS = ["Color", "Shape_count", "Quadrant", "Spatial", "Stroke_fill"]

def extract_svg(text: str) -> str:
    """Extract SVG from model output."""
    m = re.search(r"(<svg[\s\S]*?</svg>)", text, re.IGNORECASE)
    return m.group(1) if m else text.strip()

def svg_to_pil(svg_str: str, size: int = 224):
    """Render SVG to PIL Image via cairosvg."""
    import cairosvg
    from PIL import Image
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"),
                                  output_width=size, output_height=size)
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")

def main():
    t0 = time.time()
    N = len(PROMPTS_20)

    # ── Step 1: Generate SVGs with vLLM ──
    print("=== Loading vLLM ===", flush=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=0.6, max_model_len=4096)
    sampling = SamplingParams(temperature=0.3, max_tokens=2048)

    prompts_formatted = []
    for cap in PROMPTS_20:
        messages = [{"role": "user",
                     "content": f"Generate an SVG image: {cap}\n\nOutput only the SVG code."}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        prompts_formatted.append(text)

    print("=== Generating 20 SVGs ===", flush=True)
    outputs = llm.generate(prompts_formatted, sampling)
    svgs = [extract_svg(o.outputs[0].text) for o in outputs]

    # Unload vLLM
    print("=== Unloading vLLM ===", flush=True)
    del llm, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    print(f"GPU memory after unload: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # ── Step 2: Render SVGs to images ──
    print("=== Rendering SVGs ===", flush=True)
    images = []
    valid_indices = []
    render_errors = []
    for i, svg in enumerate(svgs):
        try:
            img = svg_to_pil(svg)
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            render_errors.append({"index": i, "prompt": PROMPTS_20[i], "error": str(e)})
            print(f"  Render FAIL [{i}]: {e}", flush=True)

    print(f"  Valid renders: {len(valid_indices)}/{N}", flush=True)

    # ── Step 3: CLIP scoring ──
    print("=== Loading CLIP ===", flush=True)
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    matched_scores = []
    shuffled_scores = []
    per_sample = []

    with torch.no_grad():
        for idx_pos, i in enumerate(valid_indices):
            # matched: image_i vs caption_i
            inputs_m = clip_proc(text=[PROMPTS_20[i]], images=[images[idx_pos]],
                                  return_tensors="pt", padding=True)
            inputs_m = {k: v.cuda() for k, v in inputs_m.items()}
            out_m = clip_model(**inputs_m)
            sc_matched = out_m.logits_per_image[0, 0].item()

            # shuffled: image_i vs caption_{(i+1) % N}
            j = (i + 1) % N
            inputs_s = clip_proc(text=[PROMPTS_20[j]], images=[images[idx_pos]],
                                  return_tensors="pt", padding=True)
            inputs_s = {k: v.cuda() for k, v in inputs_s.items()}
            out_s = clip_model(**inputs_s)
            sc_shuffled = out_s.logits_per_image[0, 0].item()

            matched_scores.append(sc_matched)
            shuffled_scores.append(sc_shuffled)
            per_sample.append({
                "index": i,
                "prompt": PROMPTS_20[i],
                "matched": round(sc_matched, 4),
                "shuffled": round(sc_shuffled, 4),
                "diff": round(sc_matched - sc_shuffled, 4),
            })

    # ── Step 4: Compute statistics ──
    matched_arr = np.array(matched_scores)
    shuffled_arr = np.array(shuffled_scores)
    diffs = matched_arr - shuffled_arr

    snr = float(np.mean(diffs) / np.std(diffs)) if np.std(diffs) > 0 else float("inf")

    # Per-dimension SNR (4 samples each, by valid_indices)
    dim_snr = {}
    for d_idx, d_name in enumerate(DIMENSIONS):
        dim_indices = set(range(d_idx * 4, d_idx * 4 + 4))
        dim_diffs = [diffs[k] for k, vi in enumerate(valid_indices) if vi in dim_indices]
        if len(dim_diffs) >= 2:
            dd = np.array(dim_diffs)
            dim_snr[d_name] = round(float(np.mean(dd) / np.std(dd)) if np.std(dd) > 0 else float("inf"), 4)
        else:
            dim_snr[d_name] = None

    elapsed = (time.time() - t0) / 60.0

    if snr >= 2.0:
        status = "PASS"
    elif snr >= 1.0:
        status = "MARGINAL"
    else:
        status = "FAIL"

    result = {
        "per_sample_table": per_sample,
        "matched_mean": round(float(np.mean(matched_arr)), 4),
        "shuffled_mean": round(float(np.mean(shuffled_arr)), 4),
        "diff_mean": round(float(np.mean(diffs)), 4),
        "diff_std": round(float(np.std(diffs)), 4),
        "snr": round(snr, 4),
        "negative_diffs_count": int(np.sum(diffs < 0)),
        "per_dimension_snr": dim_snr,
        "status": status,
        "valid_renders": f"{len(valid_indices)}/{N}",
        "render_errors": render_errors,
        "runtime_min": round(elapsed, 2),
    }

    print("\n=== RESULTS ===", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    # Save to file
    out_path = "/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/p4_clipscore_20_results.json"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
