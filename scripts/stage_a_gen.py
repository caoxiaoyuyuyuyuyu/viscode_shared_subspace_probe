#!/usr/bin/env python3
"""Stage A vLLM generation — D024 v2.

3 models × 3 formats × 200 prompts × n_shots=2 = 3600 samples total.
Data-parallel via shard-id; each process pins one GPU and one model/format.

Per-format SamplingParams (from p3_coder_rerun.py, NOT p3_rerun_verified.py):
  SVG:       stop=["</svg>"],                max_tokens=1024, T=0.3
  TikZ:      stop=["\\end{tikzpicture}"],    max_tokens=1024, T=0.3
  Asymptote: stop=[],                        max_tokens=2048, T=0.3

n_shots=2 via SamplingParams(n=2, seed=...) — shares prefix KV cache.
If vLLM degenerates (both shots identical), switch fallback to two generate()
calls with offset seed (base+pid vs base+pid+10000).
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

# ── Force HuggingFace offline mode (Reviewer 二审 critical fix) ─────────
# Hardcoded at module level so a single missing env var in launcher cannot
# trigger 7-min metadata retry per model load (× 18 launches = 2.1h waste).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# ── CLI ─────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "coder":     {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "type": "chat"},
    "viscoder2": {"name": "TIGER-Lab/VisCoder2-7B",          "type": "chat"},
    "qwen25":    {"name": "Qwen/Qwen2.5-7B",                 "type": "base"},
}

FORMAT_CONFIG = {
    "svg":  {"stop": ["</svg>"],             "max_tokens": 1024,
             "template": "Generate an SVG image: {caption}\n\nOutput only the SVG code."},
    "tikz": {"stop": ["\\end{tikzpicture}"], "max_tokens": 1024,
             "template": "Generate TikZ code for: {caption}\n\nOutput only the TikZ code."},
    "asy":  {"stop": [],                      "max_tokens": 2048,
             "template": "Generate Asymptote code for: {caption}\n\nOutput only the Asymptote code."},
}

# Validity checks (from p3_rerun_verified.py)
VALIDITY_CHECKS = {
    "svg":  lambda text: "<svg" in text.lower(),
    "tikz": lambda text: "\\begin{tikzpicture}" in text or "\\tikz" in text,
    "asy":  lambda text: any(kw in text for kw in ["import", "draw", "path", "size("]),
}

# eval pool file name for each format (asy → asymptote.jsonl)
EVAL_POOL_DIR = "/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/stage_a/eval_pool/v3_4"
EVAL_POOL_FILE = {"svg": "svg.jsonl", "tikz": "tikz.jsonl", "asy": "asymptote.jsonl"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--format", required=True, choices=list(FORMAT_CONFIG.keys()))
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--n-shots", type=int, default=2)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=20260411)
    p.add_argument("--fallback-double-gen", action="store_true",
                   help="Disable n=2 shared KV; run two generate() calls with offset seeds.")
    return p.parse_args()


def load_eval_pool(fmt: str, limit: int):
    path = os.path.join(EVAL_POOL_DIR, EVAL_POOL_FILE[fmt])
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


def build_prompt(caption: str, fmt: str, model_type: str, tokenizer):
    user_msg = FORMAT_CONFIG[fmt]["template"].format(caption=caption)
    if model_type == "chat" and tokenizer is not None:
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # base model: completion prefix
    return user_msg + "\n\n"


def main():
    args = parse_args()

    # ── Pin GPU BEFORE importing vLLM ───────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from vllm import LLM, SamplingParams

    model_cfg = MODEL_REGISTRY[args.model]
    fmt_cfg = FORMAT_CONFIG[args.format]

    # ── Load + shard eval pool ─────────────────────────────────────────
    all_rows = load_eval_pool(args.format, args.limit)
    shard_rows = [
        (idx, row) for idx, row in enumerate(all_rows)
        if idx % args.num_shards == args.shard_id
    ]
    print(f"[stage_a_gen] model={args.model} format={args.format} "
          f"shard={args.shard_id}/{args.num_shards} gpu={args.gpu} "
          f"n_shots={args.n_shots} total={len(all_rows)} shard_size={len(shard_rows)}")

    # ── Load vLLM model ─────────────────────────────────────────────────
    t_load = time.perf_counter()
    llm = LLM(
        model=model_cfg["name"],
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
    )
    load_time = time.perf_counter() - t_load
    print(f"[stage_a_gen] model loaded in {load_time:.1f}s")

    tokenizer = llm.get_tokenizer() if model_cfg["type"] == "chat" else None

    # ── Output file ─────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.format}.shard{args.shard_id}.jsonl")
    # When num_shards == 1 (dry-run), drop the shard suffix
    if args.num_shards == 1:
        out_path = os.path.join(args.out_dir, f"{args.format}.jsonl")

    # ── Generate one prompt at a time (seed is per-prompt) ──────────────
    run_ts = datetime.now(timezone.utc).isoformat()
    n_written = 0
    t_all = time.perf_counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, row in shard_rows:
            prompt_id = row["id"]
            caption = row["caption"]
            try:
                prompt_text = build_prompt(caption, args.format, model_cfg["type"], tokenizer)
            except Exception as e:
                print(f"[stage_a_gen] prompt build failed id={prompt_id}: {e}")
                continue

            base_seed = args.seed_base + idx

            shots = []  # list of (shot_idx, text, finish_reason, stop_reason, n_tokens, seed, latency_ms)

            try:
                if args.fallback_double_gen:
                    # Fallback: two generate() calls with offset seeds
                    for shot_idx in range(args.n_shots):
                        seed = base_seed + (shot_idx * 10000)
                        sp = SamplingParams(
                            n=1,
                            temperature=0.3,
                            max_tokens=fmt_cfg["max_tokens"],
                            stop=fmt_cfg["stop"],
                            include_stop_str_in_output=True,
                            seed=seed,
                        )
                        t0 = time.perf_counter()
                        outs = llm.generate([prompt_text], sp, use_tqdm=False)
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        o = outs[0].outputs[0]
                        shots.append((shot_idx, o.text, o.finish_reason,
                                      getattr(o, "stop_reason", None),
                                      len(o.token_ids), seed, latency_ms))
                else:
                    # Primary: n=n_shots in one call, shared prefix KV
                    sp = SamplingParams(
                        n=args.n_shots,
                        temperature=0.3,
                        max_tokens=fmt_cfg["max_tokens"],
                        stop=fmt_cfg["stop"],
                        include_stop_str_in_output=True,
                        seed=base_seed,
                    )
                    t0 = time.perf_counter()
                    outs = llm.generate([prompt_text], sp, use_tqdm=False)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    per_shot_latency = latency_ms / max(len(outs[0].outputs), 1)
                    for shot_idx, o in enumerate(outs[0].outputs):
                        shots.append((shot_idx, o.text, o.finish_reason,
                                      getattr(o, "stop_reason", None),
                                      len(o.token_ids), base_seed, per_shot_latency))
            except Exception as e:
                print(f"[stage_a_gen] generation failed id={prompt_id}: {e}")
                continue

            checker = VALIDITY_CHECKS[args.format]
            for shot_idx, text, finish_reason, stop_reason, n_tokens, seed, latency_ms in shots:
                rec = {
                    "model": args.model,
                    "model_name": model_cfg["name"],
                    "format": args.format,
                    "prompt_id": prompt_id,
                    "prompt_idx": idx,
                    "prompt": caption,
                    "shot_idx": shot_idx,
                    "generation": text,
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                    "n_tokens": n_tokens,
                    "latency_ms": round(latency_ms, 2),
                    "seed": seed,
                    "truncated": finish_reason == "length",
                    "valid": checker(text),
                    "run_ts": run_ts,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1

    total_time = time.perf_counter() - t_all
    print(f"[stage_a_gen] done: {n_written} records in {total_time:.1f}s → {out_path}")

    # ── Cleanup ────────────────────────────────────────────────────────
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
