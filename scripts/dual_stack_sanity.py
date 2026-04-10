#!/usr/bin/env python3
"""Dual-stack sanity check: vLLM generation + HF teacher-forcing hidden state extraction."""

import argparse
import os
import sys
import time
from pathlib import Path


def part1_vllm_sanity(model_name: str, cache_dir: str, output_dir: str) -> bool:
    """Part 1: vLLM 4-bit generation sanity check."""
    print("\n" + "=" * 60)
    print("Part 1 — vLLM Sanity Check")
    print("=" * 60)

    try:
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

        prompt = "Draw a simple red circle"
        messages = [{"role": "user", "content": prompt}]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

        t0 = time.perf_counter()
        outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
        elapsed = time.perf_counter() - t0

        generated_text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0

        print(f"\nPrompt: {prompt}")
        print(f"Generated tokens: {num_tokens}")
        print(f"Throughput: {tok_per_sec:.1f} tok/s")
        print(f"\n--- Generated text (first 500 chars) ---")
        print(generated_text[:500])

        # Optional: cairosvg render verification
        svg_start = generated_text.find("<svg")
        svg_end = generated_text.find("</svg>")
        if svg_start != -1 and svg_end != -1:
            svg_content = generated_text[svg_start : svg_end + len("</svg>")]
            try:
                import cairosvg

                out_path = Path(output_dir) / "part1_render.png"
                cairosvg.svg2png(bytestring=svg_content.encode(), write_to=str(out_path))
                print(f"\nSVG rendered to: {out_path}")
            except Exception as e:
                print(f"\ncairosvg render skipped: {e}")
        else:
            print("\nNo complete <svg>...</svg> found in output.")

        # Cleanup vLLM to free GPU memory
        del llm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        print("\n[PASS] Part 1 — vLLM sanity")
        return True

    except Exception as e:
        print(f"\n[FAIL] Part 1 — vLLM sanity: {e}")
        return False


def part2_hf_teacher_forcing(model_name: str, cache_dir: str, output_dir: str) -> bool:
    """Part 2: HF teacher-forcing hidden state extraction."""
    print("\n" + "=" * 60)
    print("Part 2 — HF Teacher-Forcing Hidden State Extraction")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading model: {model_name} (4-bit via BitsAndBytes)")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # Sample (caption, reference_svg) pair
        caption = "A red circle centered in the canvas"
        reference_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
            '<circle cx="100" cy="100" r="50" fill="red"/>'
            "</svg>"
        )

        # Build chat template with caption + reference code
        messages = [
            {"role": "user", "content": caption},
            {"role": "assistant", "content": reference_svg},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        print(f"Input length: {inputs['input_ids'].shape[1]} tokens")

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract layer 16 hidden states
        target_layer = 16
        if target_layer >= len(outputs.hidden_states):
            target_layer = len(outputs.hidden_states) - 1
            print(f"Warning: model has {len(outputs.hidden_states)} layers, using layer {target_layer}")

        hidden = outputs.hidden_states[target_layer]  # (1, seq_len, hidden_dim)

        # Identify code token positions (tokens from the reference_svg portion)
        ref_tokens = tokenizer(reference_svg, return_tensors="pt", add_special_tokens=False)
        ref_len = ref_tokens["input_ids"].shape[1]
        total_len = inputs["input_ids"].shape[1]

        # Code tokens are the last ref_len tokens before any trailing special tokens
        code_start = total_len - ref_len
        code_end = total_len
        code_hidden = hidden[:, code_start:code_end, :]  # (1, ref_len, hidden_dim)

        # Mean pool over code token positions
        pooled = code_hidden.mean(dim=1)  # (1, hidden_dim)

        print(f"\nLayer {target_layer} hidden states shape: {hidden.shape}")
        print(f"Code token range: [{code_start}, {code_end}) ({ref_len} tokens)")
        print(f"Mean-pooled code embedding shape: {pooled.shape}")
        print(f"Mean-pooled code embedding dtype: {pooled.dtype}")

        # Save for verification
        out_path = Path(output_dir) / "part2_hidden.pt"
        torch.save({"pooled": pooled.cpu(), "layer": target_layer, "shape": pooled.shape}, str(out_path))
        print(f"Saved to: {out_path}")

        # Cleanup
        del model, outputs, hidden
        torch.cuda.empty_cache()

        print("\n[PASS] Part 2 — HF teacher-forcing")
        return True

    except Exception as e:
        print(f"\n[FAIL] Part 2 — HF teacher-forcing: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Dual-stack sanity check")
    parser.add_argument(
        "--model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model name or path"
    )
    parser.add_argument(
        "--cache_dir", default="/root/autodl-tmp/.hf_cache", help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--output_dir", default="artifacts/sanity", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    p1 = part1_vllm_sanity(args.model_name, args.cache_dir, args.output_dir)
    p2 = part2_hf_teacher_forcing(args.model_name, args.cache_dir, args.output_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Part 1 (vLLM):             {'PASS' if p1 else 'FAIL'}")
    print(f"Part 2 (HF teacher-force): {'PASS' if p2 else 'FAIL'}")

    if not (p1 and p2):
        sys.exit(1)


if __name__ == "__main__":
    main()
