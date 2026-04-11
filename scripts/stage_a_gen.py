#!/usr/bin/env python3
"""Stage A vLLM generation — D025 v3.

3 models x 3 formats x 200 prompts x 2 shot regimes (0-shot + 3-shot) = 3600.
Data-parallel via shard-id; each process pins one GPU.
n=1 per prompt; ICL regimes via prompt construction, NOT SamplingParams.n.

Per-format SamplingParams (from p3_coder_rerun.py):
  SVG:       stop=["</svg>"],                max_tokens=1024, T=0.3
  TikZ:      stop=["\\end{tikzpicture}"],    max_tokens=1024, T=0.3
  Asymptote: stop=[],                        max_tokens=2048, T=0.3
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

# Force HuggingFace offline mode + point to local cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")


# ── Config ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "coder":     {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "type": "chat"},
    "viscoder2": {"name": "TIGER-Lab/VisCoder2-7B",          "type": "chat"},
    "qwen25":    {"name": "Qwen/Qwen2.5-7B",                 "type": "base"},
}

FORMAT_CONFIG = {
    "svg":  {"stop": ["</svg>"],             "max_tokens": 1024},
    "tikz": {"stop": ["\\end{tikzpicture}"], "max_tokens": 1024},
    "asy":  {"stop": [],                      "max_tokens": 2048},
}

VALIDITY_CHECKS = {
    "svg":  lambda text: "<svg" in text.lower(),
    "tikz": lambda text: "\\begin{tikzpicture}" in text or "\\tikz" in text,
    "asy":  lambda text: any(kw in text for kw in ["import", "draw", "path", "size("]),
}

FORMAT_LABEL = {"svg": "SVG", "tikz": "TikZ", "asy": "Asymptote"}

EVAL_POOL_FILE = {"svg": "svg.jsonl", "tikz": "tikz.jsonl", "asy": "asymptote.jsonl"}
ICL_EXEMPLAR_FILE = {"svg": "svg.jsonl", "tikz": "tikz.jsonl", "asy": "asymptote.jsonl"}


def parse_args():
    p = argparse.ArgumentParser(description="Stage A generation v3 (D025)")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--format", required=True, choices=list(FORMAT_CONFIG.keys()))
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--shot-regime", choices=["0-shot", "3-shot", "both"], default="both",
                   help="ICL prompting regime: 0-shot, 3-shot, or both")
    p.add_argument("--icl-exemplars-dir", required=True,
                   help="Dir with {svg,tikz,asymptote}.jsonl exemplar files")
    p.add_argument("--eval-pool-dir", required=True,
                   help="Dir with eval pool JSONL files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-prompts", type=int, default=200,
                   help="Max prompts to load from eval pool (default 200)")
    p.add_argument("--seed-base", type=int, default=20260411)
    return p.parse_args()


def load_eval_pool(eval_pool_dir, fmt, limit):
    path = os.path.join(eval_pool_dir, EVAL_POOL_FILE[fmt])
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def load_icl_exemplars(exemplars_dir, fmt):
    path = os.path.join(exemplars_dir, ICL_EXEMPLAR_FILE[fmt])
    exemplars = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exemplars.append(json.loads(line))
    return exemplars


def build_prompt_0shot(caption, fmt):
    label = FORMAT_LABEL[fmt]
    return f"Generate {label} code for the following description:\n{caption}"


def build_prompt_3shot(caption, fmt, exemplars):
    label = FORMAT_LABEL[fmt]
    parts = [f"Generate {label} code for the following description."]
    for i, ex in enumerate(exemplars, 1):
        parts.append(f"\nExample {i}:\nDescription: {ex['caption']}\nCode:\n{ex['code']}")
    parts.append(f"\nNow generate code for:\nDescription: {caption}\nCode:")
    return "\n".join(parts)


def wrap_chat(user_msg, model_type, tokenizer):
    if model_type == "chat" and tokenizer is not None:
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_msg


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model_cfg = MODEL_REGISTRY[args.model]
    fmt_cfg = FORMAT_CONFIG[args.format]

    print(f"[stage_a_gen v3] model={args.model} format={args.format} "
          f"shard={args.shard_id}/{args.num_shards} gpu={args.gpu} "
          f"shot_regime={args.shot_regime}")

    # ── Load eval pool ────────────────────────────────────────────────
    pool = load_eval_pool(args.eval_pool_dir, args.format, args.max_prompts)
    pool = [r for i, r in enumerate(pool) if i % args.num_shards == args.shard_id]
    print(f"[stage_a_gen v3] {len(pool)} prompts in this shard")

    # ── Load ICL exemplars ────────────────────────────────────────────
    exemplars = []
    if args.shot_regime in ("3-shot", "both"):
        exemplars = load_icl_exemplars(args.icl_exemplars_dir, args.format)
        print(f"[stage_a_gen v3] Loaded {len(exemplars)} ICL exemplars")
        for ex in exemplars:
            print(f"  exemplar {ex['id']}: {ex['caption'][:40]}...")

    # ── Determine regimes ─────────────────────────────────────────────
    regimes = []
    if args.shot_regime in ("0-shot", "both"):
        regimes.append("0-shot")
    if args.shot_regime in ("3-shot", "both"):
        regimes.append("3-shot")

    # ── Load model ────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams

    print(f"[stage_a_gen v3] Loading {model_cfg['name']}...")
    t_load = time.perf_counter()
    llm = LLM(
        model=model_cfg["name"],
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    print(f"[stage_a_gen v3] Model loaded in {time.perf_counter() - t_load:.1f}s")

    tokenizer = llm.get_tokenizer() if model_cfg["type"] == "chat" else None

    # ── Generate ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model}_{args.format}_s{args.shard_id}.jsonl")
    run_ts = datetime.now(timezone.utc).isoformat()
    n_written = 0
    t_all = time.perf_counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(pool):
            prompt_id = row.get("id", f"{args.format}_{idx}")
            caption = row["caption"]

            for regime in regimes:
                # Build prompt
                if regime == "0-shot":
                    user_msg = build_prompt_0shot(caption, args.format)
                    exemplar_ids = []
                else:
                    user_msg = build_prompt_3shot(caption, args.format, exemplars)
                    exemplar_ids = [ex["id"] for ex in exemplars]

                prompt_text = wrap_chat(user_msg, model_cfg["type"], tokenizer)

                # Seed: offset 3-shot by 10000 to avoid collisions
                seed = args.seed_base + idx + (10000 if regime == "3-shot" else 0)
                sp = SamplingParams(
                    n=1,
                    temperature=0.3,
                    max_tokens=fmt_cfg["max_tokens"],
                    stop=fmt_cfg["stop"],
                    include_stop_str_in_output=True,
                    seed=seed,
                )

                try:
                    t0 = time.perf_counter()
                    outs = llm.generate([prompt_text], sp, use_tqdm=False)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    print(f"[stage_a_gen v3] FAIL id={prompt_id} regime={regime}: {e}")
                    continue

                o = outs[0].outputs[0]
                checker = VALIDITY_CHECKS[args.format]
                rec = {
                    "model": args.model,
                    "model_name": model_cfg["name"],
                    "format": args.format,
                    "prompt_id": prompt_id,
                    "prompt_idx": idx,
                    "prompt": caption,
                    "shot_regime": regime,
                    "icl_exemplar_ids": exemplar_ids,
                    "generation": o.text,
                    "finish_reason": o.finish_reason,
                    "stop_reason": getattr(o, "stop_reason", None),
                    "n_tokens": len(o.token_ids),
                    "latency_ms": round(latency_ms, 2),
                    "seed": seed,
                    "truncated": o.finish_reason == "length",
                    "valid": checker(o.text),
                    "run_ts": run_ts,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1

            if (idx + 1) % 10 == 0:
                print(f"[stage_a_gen v3] progress: {idx+1}/{len(pool)} prompts, {n_written} records")

    total_time = time.perf_counter() - t_all
    print(f"[stage_a_gen v3] done: {n_written} records in {total_time:.1f}s -> {out_path}")

    # ── Cleanup ───────────────────────────────────────────────────────
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
