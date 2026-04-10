#!/usr/bin/env python3
"""Dual-stack sanity check: vLLM generation + HF teacher-forcing hidden states.

Part 1: vLLM free generation → SVG output + tok/s
Part 2: HF transformers teacher-forcing → hidden_states shape at layer 16
"""

import argparse
import time
import json
import os
import sys

DEFAULT_CACHE_DIR = "/root/autodl-tmp/.hf_cache"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
PROMPT = "Draw a simple red circle as an SVG image. Output only the SVG code, nothing else."
REFERENCE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="80" fill="red"/>
</svg>"""


def vllm_sanity(model_name: str, cache_dir: str, output_dir: str) -> dict:
    """Part 1: vLLM free generation."""
    print("\n" + "=" * 60)
    print("PART 1: vLLM Free Generation Sanity")
    print("=" * 60)

    result = {"part": "vllm", "status": "FAIL", "error": None}
    try:
        from vllm import LLM, SamplingParams

        print(f"Loading model {model_name} with vLLM (4-bit)...")
        t0 = time.time()
        llm = LLM(
            model=model_name,
            download_dir=cache_dir,
            quantization="awq_marlin",
            dtype="half",
            max_model_len=4096,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
        )
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s")

        params = SamplingParams(temperature=0.3, max_tokens=512, stop=["</svg>"])

        t0 = time.time()
        outputs = llm.generate([PROMPT], params)
        gen_time = time.time() - t0

        text = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)
        tok_s = n_tokens / gen_time if gen_time > 0 else 0

        print(f"\nGenerated {n_tokens} tokens in {gen_time:.2f}s ({tok_s:.1f} tok/s)")
        print(f"Output:\n{text[:500]}")

        # Try rendering with cairosvg
        try:
            import cairosvg
            svg_text = text if "</svg>" in text else text + "</svg>"
            png_path = os.path.join(output_dir, "vllm_sanity.png")
            cairosvg.svg2png(bytestring=svg_text.encode(), write_to=png_path)
            print(f"\nRendered PNG saved to {png_path}")
            result["rendered"] = True
        except Exception as e:
            print(f"\nRendering failed (non-critical): {e}")
            result["rendered"] = False

        result.update({
            "status": "PASS",
            "load_time_s": round(load_time, 1),
            "gen_time_s": round(gen_time, 2),
            "n_tokens": n_tokens,
            "tok_s": round(tok_s, 1),
            "output_preview": text[:200],
        })

        # Free GPU memory
        del llm
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)
        print(f"\nvLLM FAILED: {e}")
        import traceback; traceback.print_exc()

    print(f"\n>>> Part 1 result: {result['status']}")
    return result


def hf_teacher_forcing_sanity(model_name: str, cache_dir: str, output_dir: str) -> dict:
    """Part 2: HF transformers teacher-forcing hidden states."""
    print("\n" + "=" * 60)
    print("PART 2: HF Teacher-Forcing Hidden States Sanity")
    print("=" * 60)

    result = {"part": "hf_teacher_forcing", "status": "FAIL", "error": None}
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading tokenizer {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )

        print(f"Loading model {model_name} (4-bit)...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True,
        )
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s")
        print(f"Model layers: {model.config.num_hidden_layers}")

        # Build teacher-forcing input: caption + reference code
        caption = "A red circle centered in the canvas"
        messages = [{"role": "user", "content": caption}]

        # Tokenize caption part
        caption_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        caption_ids = tokenizer.encode(caption_text, return_tensors="pt")
        caption_len = caption_ids.shape[1]

        # Tokenize full sequence: caption + reference code
        full_text = caption_text + REFERENCE_SVG
        full_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)
        total_len = full_ids.shape[1]
        code_token_start = caption_len
        code_token_end = total_len

        print(f"\nCaption tokens: {caption_len}")
        print(f"Code tokens: {code_token_end - code_token_start}")
        print(f"Total tokens: {total_len}")

        # Forward pass with hidden states
        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_ids=full_ids, output_hidden_states=True)
        fwd_time = time.time() - t0

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1,) tensors
        n_layers = len(hidden_states) - 1  # exclude embedding layer

        # Extract layer 16 (or closest available)
        target_layer = min(16, n_layers)
        layer_hs = hidden_states[target_layer]  # [1, seq_len, hidden_dim]

        # Mean-pool over code token positions only
        code_hs = layer_hs[0, code_token_start:code_token_end, :]  # [code_len, hidden_dim]
        pooled = code_hs.mean(dim=0)  # [hidden_dim]

        print(f"\nForward pass: {fwd_time:.2f}s")
        print(f"Total layers (excl embedding): {n_layers}")
        print(f"Target layer: {target_layer}")
        print(f"Layer {target_layer} hidden_states shape: {layer_hs.shape}")
        print(f"Code tokens range: [{code_token_start}, {code_token_end})")
        print(f"Code hidden_states shape: {code_hs.shape}")
        print(f"Mean-pooled vector shape: {pooled.shape}")
        print(f"Mean-pooled dtype: {pooled.dtype}")
        print(f"Mean-pooled norm: {pooled.float().norm().item():.4f}")

        result.update({
            "status": "PASS",
            "load_time_s": round(load_time, 1),
            "fwd_time_s": round(fwd_time, 2),
            "n_layers": n_layers,
            "target_layer": target_layer,
            "hidden_dim": pooled.shape[0],
            "hidden_states_shape": list(layer_hs.shape),
            "code_tokens": code_token_end - code_token_start,
            "pooled_shape": list(pooled.shape),
            "pooled_dtype": str(pooled.dtype),
            "pooled_norm": round(pooled.float().norm().item(), 4),
        })

        del model, tokenizer
        import gc; gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)
        print(f"\nHF Teacher-Forcing FAILED: {e}")
        import traceback; traceback.print_exc()

    print(f"\n>>> Part 2 result: {result['status']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Dual-stack sanity check")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output_dir", default="/root/autodl-tmp/viscode_shared_subspace_probe/outputs/sanity")
    parser.add_argument("--skip_vllm", action="store_true", help="Skip vLLM part")
    parser.add_argument("--skip_hf", action="store_true", help="Skip HF part")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    if not args.skip_vllm:
        results["vllm"] = vllm_sanity(args.model_name, args.cache_dir, args.output_dir)
    if not args.skip_hf:
        results["hf"] = hf_teacher_forcing_sanity(args.model_name, args.cache_dir, args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for part, res in results.items():
        print(f"  {part}: {res['status']}")

    overall = "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL"
    print(f"\n  OVERALL: {overall}")

    out_path = os.path.join(args.output_dir, "dual_stack_sanity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
