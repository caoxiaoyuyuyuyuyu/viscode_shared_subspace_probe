#!/usr/bin/env python3
"""Mini pilot: 10 prompts × 1 model × SVG → CLIPScore + SNR baseline.

Computes CLIPScore (ViT-L/14) for generated SVGs and a shuffle-permutation
SNR baseline per pre_registration §4.
"""

import argparse
import json
import os
import sys
import time
import io
import numpy as np

DEFAULT_CACHE_DIR = "/root/autodl-tmp/.hf_cache"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

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


def generate_svgs(model_name: str, cache_dir: str, prompts: list[str]) -> list[dict]:
    """Generate SVGs using vLLM."""
    from vllm import LLM, SamplingParams

    print(f"Loading {model_name} with vLLM...")
    llm = LLM(
        model=model_name,
        download_dir=cache_dir,
        quantization="awq_marlin",
        dtype="half",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
    )

    system_msg = "You are an SVG code generator. Output only valid SVG code, nothing else."
    formatted = []
    for p in prompts:
        formatted.append(f"Generate an SVG image: {p}\n\nOutput only the SVG code:")

    params = SamplingParams(temperature=0.3, max_tokens=1024, stop=["```"])

    print(f"Generating {len(prompts)} SVGs...")
    outputs = llm.generate(formatted, params)

    results = []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text.strip()
        # Extract SVG content
        if "<svg" in text:
            svg_start = text.index("<svg")
            svg_end = text.rfind("</svg>")
            if svg_end > svg_start:
                text = text[svg_start:svg_end + 6]
            else:
                text = text[svg_start:] + "</svg>"
        results.append({
            "prompt": prompts[i],
            "svg": text,
            "n_tokens": len(out.outputs[0].token_ids),
        })

    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    return results


def render_svgs(generations: list[dict], output_dir: str) -> list[dict]:
    """Render SVGs to PNG using cairosvg."""
    import cairosvg
    from PIL import Image

    rendered = []
    for i, gen in enumerate(generations):
        png_path = os.path.join(output_dir, f"pilot_{i:02d}.png")
        try:
            png_data = cairosvg.svg2png(
                bytestring=gen["svg"].encode(),
                output_width=224,
                output_height=224,
            )
            with open(png_path, "wb") as f:
                f.write(png_data)
            img = Image.open(io.BytesIO(png_data)).convert("RGB")
            rendered.append({**gen, "png_path": png_path, "image": img, "valid": True})
            print(f"  [{i}] Rendered: {gen['prompt'][:40]}...")
        except Exception as e:
            print(f"  [{i}] RENDER FAILED: {e}")
            rendered.append({**gen, "png_path": None, "image": None, "valid": False})

    valid = sum(1 for r in rendered if r["valid"])
    print(f"\nRendered {valid}/{len(rendered)} SVGs successfully")
    return rendered


def compute_clip_scores(rendered: list[dict], cache_dir: str) -> np.ndarray:
    """Compute CLIPScore for each (image, caption) pair."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    model_id = "openai/clip-vit-large-patch14"
    print(f"\nLoading CLIP model ({model_id})...")
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    scores = []
    for r in rendered:
        if not r["valid"]:
            scores.append(0.0)
            continue

        inputs = processor(
            text=[r["prompt"]],
            images=[r["image"]],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # CLIPScore = cosine similarity × 100
            score = outputs.logits_per_image.item()

        scores.append(score)

    del model, processor
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return np.array(scores)


def compute_snr(
    rendered: list[dict],
    real_scores: np.ndarray,
    cache_dir: str,
    n_permutations: int = 1000,
) -> dict:
    """Compute shuffle-permutation SNR baseline per pre_registration §4."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    valid_indices = [i for i, r in enumerate(rendered) if r["valid"]]
    if len(valid_indices) < 2:
        return {"error": "Too few valid renders for SNR computation"}

    valid_scores = real_scores[valid_indices]
    valid_images = [rendered[i]["image"] for i in valid_indices]
    valid_prompts = [rendered[i]["prompt"] for i in valid_indices]

    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"\nComputing shuffle baseline ({n_permutations} permutations)...")
    rng = np.random.default_rng(42)
    shuffle_means = []

    for perm_i in range(n_permutations):
        shuffled_prompts = rng.permutation(valid_prompts).tolist()
        perm_scores = []
        for img, prompt in zip(valid_images, shuffled_prompts):
            inputs = processor(
                text=[prompt], images=[img], return_tensors="pt", padding=True
            ).to(device)
            with torch.no_grad():
                score = model(**inputs).logits_per_image.item()
            perm_scores.append(score)
        shuffle_means.append(np.mean(perm_scores))

        if (perm_i + 1) % 100 == 0:
            print(f"  Permutation {perm_i + 1}/{n_permutations}")

    del model, processor
    import gc; gc.collect()
    torch.cuda.empty_cache()

    mean_real = float(np.mean(valid_scores))
    std_real = float(np.std(valid_scores, ddof=1))
    mean_shuffle = float(np.mean(shuffle_means))
    std_shuffle = float(np.std(shuffle_means, ddof=1))
    pooled_std = float(np.sqrt((std_real**2 + std_shuffle**2) / 2))

    snr = (mean_real - mean_shuffle) / pooled_std if pooled_std > 0 else float("inf")

    return {
        "mean_real": round(mean_real, 4),
        "std_real": round(std_real, 4),
        "mean_shuffle": round(mean_shuffle, 4),
        "std_shuffle": round(std_shuffle, 4),
        "pooled_std": round(pooled_std, 4),
        "SNR": round(snr, 4),
        "n_valid": len(valid_indices),
        "n_permutations": n_permutations,
        "gate_decision": "CLIPScore" if snr >= 1.0 else "attribute-match",
    }


def main():
    parser = argparse.ArgumentParser(description="Mini pilot: CLIPScore + SNR baseline")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output_dir", default="/root/autodl-tmp/viscode_shared_subspace_probe/outputs/pilot")
    parser.add_argument("--num_permutations", type=int, default=1000)
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, load existing SVGs from output_dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Generate SVGs
    gen_path = os.path.join(args.output_dir, "generations.json")
    if args.skip_generation and os.path.exists(gen_path):
        print("Loading existing generations...")
        with open(gen_path) as f:
            generations = json.load(f)
    else:
        generations = generate_svgs(args.model_name, args.cache_dir, PILOT_PROMPTS)
        with open(gen_path, "w") as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)

    # Step 2: Render to PNG
    rendered = render_svgs(generations, args.output_dir)

    # Step 3: CLIPScore
    print("\nComputing CLIPScores...")
    real_scores = compute_clip_scores(rendered, args.cache_dir)
    for i, (r, s) in enumerate(zip(rendered, real_scores)):
        print(f"  [{i}] CLIPScore={s:.2f} | {r['prompt'][:50]}")

    print(f"\nMean CLIPScore: {real_scores.mean():.4f} ± {real_scores.std():.4f}")

    # Step 4: SNR baseline
    snr_result = compute_snr(rendered, real_scores, args.cache_dir, args.num_permutations)

    print(f"\n{'=' * 60}")
    print("SNR BASELINE RESULT")
    print(f"{'=' * 60}")
    print(f"  mean_real:    {snr_result.get('mean_real')}")
    print(f"  mean_shuffle: {snr_result.get('mean_shuffle')}")
    print(f"  pooled_std:   {snr_result.get('pooled_std')}")
    print(f"  SNR:          {snr_result.get('SNR')}")
    print(f"  Gate:         {snr_result.get('gate_decision')}")

    # Save results
    full_results = {
        "model": args.model_name,
        "prompts": PILOT_PROMPTS,
        "clip_scores": real_scores.tolist(),
        "mean_clip_score": float(real_scores.mean()),
        "std_clip_score": float(real_scores.std()),
        "snr_baseline": snr_result,
        "n_valid_renders": sum(1 for r in rendered if r["valid"]),
    }

    out_path = os.path.join(args.output_dir, "mini_pilot_results.json")
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
