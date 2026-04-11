#!/usr/bin/env python3
"""Select ICL exemplars from training datasets for 3-shot prompting (D025).

Uses pyarrow compute for fast vectorized filtering.
Sources: SVGX-Core-250k, DaTikZ v2 train, VisCode-Multi asymptote subset.
Output: artifacts/stage_a/icl_exemplars/{svg,tikz,asymptote}.jsonl
"""

import json
import os
import random
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

PROJECT = "/root/autodl-tmp/viscode_shared_subspace_probe"
SEED = 20260411

# Format-specific filter ranges
# Asy captions are GPT-generated descriptions (typically 1000-1500 chars),
# so we relax the upper bound while keeping code length consistent.
CAPTION_RANGE = {"svg": (10, 100), "tikz": (10, 100), "asy": (10, 1500)}
CODE_RANGE = (200, 800)  # same for all formats


def load_arrow_dir(path):
    """Load all arrow IPC stream files from a HF datasets cache directory."""
    arrow_files = sorted(
        f for f in os.listdir(path)
        if f.startswith("data-") and f.endswith(".arrow")
    )
    tables = []
    for fname in arrow_files:
        fpath = os.path.join(path, fname)
        with open(fpath, "rb") as f:
            reader = ipc.open_stream(f)
            tables.append(reader.read_all())
    return pa.concat_tables(tables)


def main():
    t0 = time.time()

    # ── Load probe pool indices ──
    with open(os.path.join(PROJECT, "outputs/stage_a/sbert_triples.json")) as f:
        triples = json.load(f)
    svg_probe = {t["svg_idx"] for t in triples}
    tikz_probe = {t["tikz_idx"] for t in triples}
    asy_probe = {t["asy_idx"] for t in triples}
    print(f"Probe pool: SVG={len(svg_probe)}, TikZ={len(tikz_probe)}, Asy={len(asy_probe)}")

    # ── Load eval pool content for dedup ──
    eval_captions = set()
    eval_codes = set()
    pool_dir = os.path.join(PROJECT, "artifacts/stage_a/eval_pool/v3_4")
    for fname in ["svg.jsonl", "tikz.jsonl", "asymptote.jsonl"]:
        with open(os.path.join(pool_dir, fname)) as f:
            for line in f:
                row = json.loads(line.strip())
                if row.get("caption"):
                    eval_captions.add(row["caption"].strip())
                if row.get("code"):
                    eval_codes.add(row["code"].strip())
    print(f"Eval pool dedup: {len(eval_captions)} captions, {len(eval_codes)} codes")
    sys.stdout.flush()

    out_dir = os.path.join(PROJECT, "artifacts/stage_a/icl_exemplars")
    os.makedirs(out_dir, exist_ok=True)

    def select_and_write(name, fmt, table, caption_col, code_col, probe_idxs, out_file):
        cap_lo, cap_hi = CAPTION_RANGE[fmt]
        code_lo, code_hi = CODE_RANGE
        print(f"\n[{name}] {len(table)} total rows (caption {cap_lo}-{cap_hi}, code {code_lo}-{code_hi})")
        sys.stdout.flush()

        captions = table.column(caption_col)
        codes = table.column(code_col)

        # Vectorized length filtering
        cap_lens = pc.utf8_length(captions)
        code_lens = pc.utf8_length(codes)
        mask = pc.and_(
            pc.and_(pc.greater_equal(code_lens, code_lo), pc.less_equal(code_lens, code_hi)),
            pc.and_(pc.greater_equal(cap_lens, cap_lo), pc.less_equal(cap_lens, cap_hi))
        )
        indices = pc.filter(pa.array(range(len(table))), mask).to_pylist()
        print(f"  Length filter: {len(indices)} pass")
        sys.stdout.flush()

        # Fine-grained exclusion on filtered candidates only
        candidates = []
        for i in indices:
            if i in probe_idxs:
                continue
            cap = captions[i].as_py()
            code = codes[i].as_py()
            if cap is None or code is None:
                continue
            if cap.strip() in eval_captions or code.strip() in eval_codes:
                continue
            candidates.append({"id": f"{name}_{i}", "caption": cap, "code": code})

        print(f"  After exclusions: {len(candidates)} candidates")
        sys.stdout.flush()

        rng = random.Random(SEED)
        selected = rng.sample(candidates, min(3, len(candidates)))
        with open(os.path.join(out_dir, out_file), "w") as f:
            for s in selected:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        for s in selected:
            print(f"  -> {s['id']}: caption={s['caption'][:50]}... code_len={len(s['code'])}")
        sys.stdout.flush()
        return selected

    # ── SVG: SVGX-Core-250k ──
    print("\n[SVG] Loading SVGX-Core-250k_text...")
    sys.stdout.flush()
    svg_table = load_arrow_dir(os.path.join(PROJECT, "data/SVGX-Core-250k_text"))
    select_and_write("svgx", "svg", svg_table, "blip_caption", "svg_code", svg_probe, "svg.jsonl")
    del svg_table

    # ── TikZ: DaTikZ v2 train ──
    print("\n[TikZ] Loading DaTikZ v2 train...")
    sys.stdout.flush()
    tikz_table = load_arrow_dir(os.path.join(PROJECT, "data/datikz/train"))
    select_and_write("datikz", "tikz", tikz_table, "caption", "code", tikz_probe, "tikz.jsonl")
    del tikz_table

    # ── Asy: VisCode-Multi asymptote subset ──
    print("\n[Asy] Loading VisCode-Multi...")
    sys.stdout.flush()
    asy_table = load_arrow_dir(os.path.join(PROJECT, "data/VisCode_filtered"))
    print(f"  Full dataset: {len(asy_table)} rows")
    sys.stdout.flush()

    # Filter for asymptote language
    lang_col = asy_table.column("language")
    asy_mask = pc.equal(lang_col, "asymptote")
    asy_orig_indices = pc.filter(pa.array(range(len(asy_table))), asy_mask).to_pylist()
    print(f"  Asymptote entries: {len(asy_orig_indices)}")
    sys.stdout.flush()

    # Extract caption/code from messages column
    messages_col = asy_table.column("messages")
    cap_lo, cap_hi = CAPTION_RANGE["asy"]
    code_lo, code_hi = CODE_RANGE
    candidates = []
    for asy_idx, orig_idx in enumerate(asy_orig_indices):
        if asy_idx in asy_probe:
            continue
        msgs = messages_col[orig_idx].as_py()
        caption = ""
        code = ""
        for msg in msgs:
            if msg["role"] == "user":
                caption = msg["content"]
            elif msg["role"] == "assistant":
                code = msg["content"]
        if not (code_lo <= len(code) <= code_hi and cap_lo <= len(caption) <= cap_hi):
            continue
        if caption.strip() in eval_captions or code.strip() in eval_codes:
            continue
        candidates.append({"id": f"vcm_asy_{asy_idx}", "caption": caption, "code": code})

    print(f"  After exclusions: {len(candidates)} candidates")
    sys.stdout.flush()

    rng = random.Random(SEED)
    selected = rng.sample(candidates, min(3, len(candidates)))
    with open(os.path.join(out_dir, "asymptote.jsonl"), "w") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    for s in selected:
        print(f"  -> {s['id']}: caption={s['caption'][:50]}... code_len={len(s['code'])}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
