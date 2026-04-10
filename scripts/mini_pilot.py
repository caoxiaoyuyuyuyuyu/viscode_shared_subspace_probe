#!/usr/bin/env python3
"""Mini pilot: vLLM SVG generation → cairosvg render → CLIPScore + shuffle SNR baseline."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

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


def generate_svgs(model_name: str, cache_dir: str) -> list[dict]:
    """Generate SVGs for all pilot prompts using vLLM."""
    from vllm import LLM, SamplingParams

    print(f"Loading model: {model_name} (4-bit via vLLM)")
    llm = LLM(
        model=model_name,
        download_dir=cache_dir,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        max_model_len=4096,
        gpu_memory_utilization=0.8,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

    results = []
    for i, prompt in enumerate(PILOT_PROMPTS):
        print(f"  [{i+1}/{len(PILOT_PROMPTS)}] {prompt}")
        messages = [{"role": "user", "content": prompt}]
        outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
        text = outputs[0].outputs[0].text
        results.append({"prompt": prompt, "generated_text": text})

    del llm
    import torch
    torch.cuda.empty_cache()

    return results


def extract_svg(text: str) -> str | None:
    """Extract first <svg>...</svg> block from text."""
    start = text.find("<svg")
    end = text.find("</svg>")
    if start != -1 and end != -1:
        return text[start : end + len("</svg>")]
    return None


def render_pngs(results: list[dict], output_dir: Path) -> list[dict]:
    """Render SVGs to PNGs via cairosvg."""
    import cairosvg

    rendered = []
    for i, r in enumerate(results):
        svg = extract_svg(r["generated_text"])
        if svg is None:
            print(f"  [SKIP] No SVG found for prompt {i}: {r['prompt'][:40]}")
            continue
        png_path = output_dir / f"pilot_{i:02d}.png"
        try:
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), output_width=224, output_height=224)
            rendered.append({**r, "svg": svg, "png_path": str(png_path)})
        except Exception as e:
            print(f"  [SKIP] Render failed for prompt {i}: {e}")

    print(f"  Rendered {len(rendered)}/{len(results)} SVGs")
    return rendered


def compute_clip_scores(rendered: list[dict], cache_dir: str) -> np.ndarray:
    """Compute CLIPScore for each (caption, rendered PNG) pair."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-large-patch14"
    print(f"Loading CLIP: {model_id}")
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    scores = []
    for r in rendered:
        image = Image.open(r["png_path"]).convert("RGB")
        inputs = processor(text=[r["prompt"]], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # CLIPScore = cosine similarity * 100
        score = outputs.logits_per_image.item()
        scores.append(score)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(scores)


def shuffle_baseline(scores: np.ndarray, rendered: list[dict], cache_dir: str, num_permutations: int) -> np.ndarray:
    """Compute shuffle baseline: randomize caption-PNG pairings."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n = len(rendered)
    captions = [r["prompt"] for r in rendered]
    images = [Image.open(r["png_path"]).convert("RGB") for r in rendered]

    shuffle_means = []
    rng = np.random.default_rng(42)

    for perm_i in range(num_permutations):
        perm = rng.permutation(n)
        shuffled_captions = [captions[j] for j in perm]

        perm_scores = []
        for cap, img in zip(shuffled_captions, images):
            inputs = processor(text=[cap], images=img, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            perm_scores.append(outputs.logits_per_image.item())

        shuffle_means.append(np.mean(perm_scores))

        if (perm_i + 1) % 100 == 0:
            print(f"  Shuffle permutation {perm_i + 1}/{num_permutations}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(shuffle_means)


def main():
    parser = argparse.ArgumentParser(description="Mini pilot: SVG generation + CLIPScore + SNR")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/.hf_cache")
    parser.add_argument("--output_dir", default="artifacts/mini_pilot")
    parser.add_argument("--num_permutations", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate SVGs
    print("\n=== Step 1: vLLM SVG Generation ===")
    results = generate_svgs(args.model_name, args.cache_dir)

    # Step 2: Render PNGs
    print("\n=== Step 2: cairosvg Rendering ===")
    rendered = render_pngs(results, output_dir)
    if len(rendered) < 2:
        print("[FAIL] Too few successful renders to compute CLIPScore")
        sys.exit(1)

    # Step 3: CLIPScore
    print("\n=== Step 3: CLIPScore (real pairs) ===")
    real_scores = compute_clip_scores(rendered, args.cache_dir)
    print(f"  Real scores: mean={real_scores.mean():.4f}, std={real_scores.std():.4f}")

    # Step 4: Shuffle baseline
    print(f"\n=== Step 4: Shuffle Baseline ({args.num_permutations} permutations) ===")
    shuffle_means = shuffle_baseline(real_scores, rendered, args.cache_dir, args.num_permutations)

    # Step 5: SNR
    mean_real = real_scores.mean()
    mean_shuffle = shuffle_means.mean()
    pooled_std = np.sqrt((real_scores.std() ** 2 + shuffle_means.std() ** 2) / 2)
    snr = (mean_real - mean_shuffle) / pooled_std if pooled_std > 0 else float("inf")

    print(f"\n=== Results ===")
    print(f"  Mean real CLIPScore:    {mean_real:.4f}")
    print(f"  Mean shuffle CLIPScore: {mean_shuffle:.4f}")
    print(f"  Pooled std:             {pooled_std:.4f}")
    print(f"  SNR:                    {snr:.4f}")

    # Step 6: Save JSON
    output = {
        "model": args.model_name,
        "num_prompts": len(PILOT_PROMPTS),
        "num_rendered": len(rendered),
        "num_permutations": args.num_permutations,
        "real_scores": real_scores.tolist(),
        "real_mean": float(mean_real),
        "real_std": float(real_scores.std()),
        "shuffle_mean": float(mean_shuffle),
        "shuffle_std": float(shuffle_means.std()),
        "pooled_std": float(pooled_std),
        "snr": float(snr),
        "per_prompt": [
            {"prompt": r["prompt"], "clip_score": float(s)}
            for r, s in zip(rendered, real_scores)
        ],
    }

    json_path = output_dir / "mini_pilot_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    if snr > 0:
        print("\n[PASS] Mini pilot — positive SNR")
    else:
        print("\n[WARN] Mini pilot — SNR <= 0, signal may be weak")


if __name__ == "__main__":
    main()
