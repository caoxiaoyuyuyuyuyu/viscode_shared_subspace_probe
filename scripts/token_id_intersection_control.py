#!/usr/bin/env python3
"""Token ID intersection control — tokenization artifact evidence for format probe 100%.

Downloads code samples from HF datasets + Qwen2 tokenizer family locally.
Computes pairwise Jaccard, three-way intersection, format-exclusive tokens.

No GPU needed. Pure CPU.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────
TARGET_N = 252  # match probe pool size
SEED = 20260410

# Tokenizer models to check
TOKENIZER_MODELS = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "TIGER-Lab/VisCoder2-7B",
    "Qwen/Qwen2.5-7B",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_svg_codes(n=TARGET_N):
    """Load SVG code from SVGX-Core-250k."""
    from datasets import load_dataset
    print(f"[data] Loading SVG codes from SVGX-Core-250k (streaming, n={n})...")
    ds = load_dataset("xingxm/SVGX-Core-250k", split="train", streaming=True)
    codes = []
    for row in ds:
        code = row.get("svg_code", "")
        if code and isinstance(code, str) and len(code.strip()) >= 50:
            codes.append(code.strip())
            if len(codes) >= n:
                break
    print(f"  SVG: {len(codes)} codes collected")
    return codes


def load_tikz_codes(n=TARGET_N):
    """Load TikZ code from DaTikZ v2."""
    from datasets import load_dataset
    print(f"[data] Loading TikZ codes from DaTikZ v2 (streaming, n={n})...")
    ds = load_dataset("nllg/datikz-v2", split="train", streaming=True)
    codes = []
    for row in ds:
        code = row.get("code", "")
        if code and isinstance(code, str) and len(code.strip()) >= 50:
            codes.append(code.strip())
            if len(codes) >= n:
                break
    print(f"  TikZ: {len(codes)} codes collected")
    return codes


def load_asy_codes(n=TARGET_N):
    """Load Asymptote code from TIGER-Lab/VisCode-Multi-679K."""
    from datasets import load_dataset
    print(f"[data] Loading Asy codes from VisCode-Multi-679K (streaming, n={n})...")
    ds = load_dataset("TIGER-Lab/VisCode-Multi-679K", split="train", streaming=True)
    codes = []
    for row in ds:
        if row.get("language", "") != "asymptote":
            continue
        msgs = row.get("messages", [])
        code = ""
        for m in msgs:
            if m["role"] == "assistant":
                code = m["content"]
                break
        if code and isinstance(code, str) and len(code.strip()) >= 50:
            codes.append(code.strip())
            if len(codes) >= n:
                break
    print(f"  Asy: {len(codes)} codes collected")
    return codes


def load_tokenizers():
    """Load tokenizers, detecting if vocabs are identical."""
    from transformers import AutoTokenizer
    tokenizers = {}
    vocab_hashes = {}

    for model_name in TOKENIZER_MODELS:
        short = model_name.split("/")[-1]
        print(f"[tokenizer] Loading {model_name}...")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizers[model_name] = tok
        # Hash vocab for identity check
        vocab_keys = tuple(sorted(tok.get_vocab().items()))
        vh = hash(vocab_keys)
        vocab_hashes[model_name] = vh
        print(f"  vocab_size={tok.vocab_size}, hash={vh}")

    return tokenizers, vocab_hashes


def compute_metrics(tokenizer, svg_codes, tikz_codes, asy_codes):
    """Compute token set metrics for one tokenizer."""
    format_sets = {"svg": set(), "tikz": set(), "asy": set()}

    for fmt, codes in [("svg", svg_codes), ("tikz", tikz_codes), ("asy", asy_codes)]:
        for code in codes:
            ids = tokenizer.encode(code, add_special_tokens=False)
            format_sets[fmt].update(ids)

    results = {
        "format_vocab_sizes": {fmt: len(s) for fmt, s in format_sets.items()},
        "pairwise_jaccard": {},
        "pairwise_intersection_size": {},
        "pairwise_union_size": {},
    }

    pairs = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]
    for a, b in pairs:
        inter = len(format_sets[a] & format_sets[b])
        union = len(format_sets[a] | format_sets[b])
        jaccard = inter / union if union > 0 else 0.0
        key = f"{a}_{b}"
        results["pairwise_jaccard"][key] = round(jaccard, 4)
        results["pairwise_intersection_size"][key] = inter
        results["pairwise_union_size"][key] = union

    # Three-way
    triple_inter = format_sets["svg"] & format_sets["tikz"] & format_sets["asy"]
    triple_union = format_sets["svg"] | format_sets["tikz"] | format_sets["asy"]
    results["triple_intersection_size"] = len(triple_inter)
    results["triple_union_size"] = len(triple_union)
    results["triple_coverage_fraction"] = round(
        len(triple_inter) / len(triple_union) if len(triple_union) > 0 else 0.0, 4
    )

    # Format-exclusive tokens
    exclusive_count = {}
    exclusive_frac = {}
    for fmt in ["svg", "tikz", "asy"]:
        others = set()
        for other_fmt in ["svg", "tikz", "asy"]:
            if other_fmt != fmt:
                others |= format_sets[other_fmt]
        excl = format_sets[fmt] - others
        exclusive_count[fmt] = len(excl)
        exclusive_frac[fmt] = round(
            len(excl) / len(format_sets[fmt]) if len(format_sets[fmt]) > 0 else 0.0, 4
        )
    results["format_exclusive_token_count"] = exclusive_count
    results["format_exclusive_fraction"] = exclusive_frac

    return results


def main():
    t0 = time.time()

    # Load data
    svg_codes = load_svg_codes()
    tikz_codes = load_tikz_codes()
    asy_codes = load_asy_codes()

    n_codes = len(svg_codes) + len(tikz_codes) + len(asy_codes)
    print(f"\n[data] Total: {n_codes} code strings "
          f"({len(svg_codes)} SVG + {len(tikz_codes)} TikZ + {len(asy_codes)} Asy)")

    # Load tokenizers
    tokenizers, vocab_hashes = load_tokenizers()

    # Check vocab identity
    hash_list = list(vocab_hashes.values())
    all_same = all(h == hash_list[0] for h in hash_list)
    print(f"\n[tokenizer] All vocabs identical: {all_same}")

    # Compute metrics per tokenizer
    tokenizer_results = []
    primary_done = False
    for model_name in TOKENIZER_MODELS:
        tok = tokenizers[model_name]
        if primary_done and vocab_hashes[model_name] == vocab_hashes[TOKENIZER_MODELS[0]]:
            tokenizer_results.append({
                "model": model_name,
                "vocab_equal_to": TOKENIZER_MODELS[0],
                "note": "Identical tokenizer vocab — metrics same as above"
            })
            print(f"\n[{model_name}] Vocab identical to {TOKENIZER_MODELS[0]}, skipping")
            continue

        print(f"\n[compute] {model_name}...")
        metrics = compute_metrics(tok, svg_codes, tikz_codes, asy_codes)
        metrics["model"] = model_name
        metrics["vocab_size"] = tok.vocab_size
        tokenizer_results.append(metrics)
        primary_done = True

        # Print summary
        print(f"  Pairwise Jaccard: {metrics['pairwise_jaccard']}")
        print(f"  Triple coverage: {metrics['triple_coverage_fraction']}")
        print(f"  Exclusive fractions: {metrics['format_exclusive_fraction']}")

    # Verdict
    primary = tokenizer_results[0]
    all_jaccard_below = all(v < 0.2 for v in primary["pairwise_jaccard"].values())
    triple_below = primary["triple_coverage_fraction"] < 0.2
    artifact_confirmed = all_jaccard_below and triple_below

    if artifact_confirmed:
        interpretation = (
            "Token distributions are near-orthogonal across formats. A linear classifier "
            "can achieve 100% format accuracy using token identity alone, without any "
            "mechanistic representation of visual semantics. This confirms that the 100% "
            "accuracy reported in the surface format probe is a tokenization artifact."
        )
    else:
        interpretation = (
            "Token distributions show substantial overlap. The 100% accuracy may not be "
            "fully explained by token identity alone; representation-level factors "
            "(e.g. positional patterns, token sequence structure) likely contribute."
        )

    verdict = {
        "all_jaccard_below_0.2": all_jaccard_below,
        "triple_coverage_below_0.2": triple_below,
        "artifact_confirmed": artifact_confirmed,
        "interpretation": interpretation,
    }

    elapsed = time.time() - t0

    # ── Save JSON ─────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    json_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_svg_codes": len(svg_codes),
        "n_tikz_codes": len(tikz_codes),
        "n_asy_codes": len(asy_codes),
        "n_code_strings": n_codes,
        "data_sources": {
            "svg": "xingxm/SVGX-Core-250k (first 252 valid, svg_code field)",
            "tikz": "nllg/datikz-v2 train (first 252 valid, code field)",
            "asy": "TIGER-Lab/VisCode-Multi-679K (first 252 asymptote, assistant content)",
        },
        "all_tokenizer_vocabs_identical": all_same,
        "tokenizers_checked": tokenizer_results,
        "verdict": verdict,
    }

    json_path = ARTIFACTS_DIR / "stage_b_token_id_intersection_control.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n[save] JSON → {json_path}")

    # ── Save Markdown ─────────────────────────────────────────────────────
    p = primary  # primary tokenizer metrics
    md = f"""# Token ID Intersection Control — Format Probe 100% Artifact Evidence

**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
**Purpose**: Provide quantitative token-level evidence that the 100% accuracy of
the surface format probe (stage_b_analysis.py Section B) is a tokenization
artifact, not a mechanistic finding. Supports paper Section 5.3 Controls.

## Setup
- {len(svg_codes)} SVG + {len(tikz_codes)} TikZ + {len(asy_codes)} Asy = {n_codes} code strings
- SVG source: SVGX-Core-250k | TikZ source: DaTikZ v2 | Asy source: VisCode-Multi-679K
- Tokenizers: Qwen2.5-Coder-7B-Instruct / VisCoder2-7B / Qwen2.5-7B (all Qwen2 family)
- All tokenizer vocabs identical: **{all_same}**

## Results

### Pairwise Jaccard Similarity ({primary['model']} tokenizer)

| Pair | |A ∩ B| | |A ∪ B| | Jaccard |
|------|--------|--------|---------|
| SVG vs TikZ | {p['pairwise_intersection_size']['svg_tikz']} | {p['pairwise_union_size']['svg_tikz']} | {p['pairwise_jaccard']['svg_tikz']:.4f} |
| SVG vs Asy | {p['pairwise_intersection_size']['svg_asy']} | {p['pairwise_union_size']['svg_asy']} | {p['pairwise_jaccard']['svg_asy']:.4f} |
| TikZ vs Asy | {p['pairwise_intersection_size']['tikz_asy']} | {p['pairwise_union_size']['tikz_asy']} | {p['pairwise_jaccard']['tikz_asy']:.4f} |

### Three-way intersection
- |SVG ∩ TikZ ∩ Asy| = {p['triple_intersection_size']}
- |SVG ∪ TikZ ∪ Asy| = {p['triple_union_size']}
- Coverage fraction = {p['triple_coverage_fraction']:.4f}

### Format-exclusive tokens (held only by 1 format)
- SVG-exclusive: {p['format_exclusive_token_count']['svg']} ({p['format_exclusive_fraction']['svg']:.1%} of SVG vocab)
- TikZ-exclusive: {p['format_exclusive_token_count']['tikz']} ({p['format_exclusive_fraction']['tikz']:.1%} of TikZ vocab)
- Asy-exclusive: {p['format_exclusive_token_count']['asy']} ({p['format_exclusive_fraction']['asy']:.1%} of Asy vocab)

### Cross-tokenizer consistency
- All 3 tokenizers (Qwen2.5-Coder / VisCoder2 / Qwen2.5-7B) share identical vocab: **{all_same}**

## Verdict

- All pairwise Jaccard < 0.2: **{all_jaccard_below}**
- Three-way coverage < 0.2: **{triple_below}**
- **Artifact confirmed: {artifact_confirmed}**

{interpretation}

## Paper Section 5.3 Integration

Supplement surface format probe disclosure (§5.1) with:
- "Token ID Jaccard between format pairs: SVG–TikZ {p['pairwise_jaccard']['svg_tikz']:.3f}, SVG–Asy {p['pairwise_jaccard']['svg_asy']:.3f}, TikZ–Asy {p['pairwise_jaccard']['tikz_asy']:.3f}"
- "Three-way token coverage: {p['triple_coverage_fraction']:.3f}"
- "This quantifies that SVG/TikZ/Asy token distributions are {'near-orthogonal' if artifact_confirmed else 'substantially overlapping'}, which {'explains' if artifact_confirmed else 'does not fully explain'} the 100% probe accuracy at a pre-representation level."

## Data Note

Code samples sourced from the same HuggingFace datasets used in the study's probe pool
(SVGX-Core-250k for SVG, DaTikZ v2 for TikZ, VisCode-Multi-679K for Asymptote).
{len(svg_codes)} samples per format, matching the N=252 triple count.
"""

    md_path = ARTIFACTS_DIR / "stage_b_token_id_intersection_control.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"[save] Markdown → {md_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Jaccard: svg_tikz={p['pairwise_jaccard']['svg_tikz']}, "
          f"svg_asy={p['pairwise_jaccard']['svg_asy']}, "
          f"tikz_asy={p['pairwise_jaccard']['tikz_asy']}")
    print(f"Triple coverage: {p['triple_coverage_fraction']}")
    print(f"Exclusive fractions: {p['format_exclusive_fraction']}")
    print(f"Artifact confirmed: {artifact_confirmed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
