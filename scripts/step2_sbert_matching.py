#!/usr/bin/env python3
"""Stage A Step 2 v2: SBERT cross-format probe triple matching.

Produces sbert_triples.json from probe pools:
  - SVG probe: SVGX-Core-250k (full)
  - TikZ probe: DaTikZ v2 train 94,532 rows only (R18 pivot)
  - Asy probe: VCM-asy from local Arrow (VisCoder2 SFT)

Rules (D017 + Director warnings):
  - cosine >= 0.70 HARD threshold, NO fallback/degradation
  - Each tikz_idx and asy_idx used at most ONCE (unique constraint)
  - Caption truncated to 50 tokens before embedding
  - If < 500 triples at 0.70, report actual count (do NOT lower threshold)
"""

import json
import os
import random
import hashlib
import datetime
from pathlib import Path

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
SEED = 20260410
COSINE_THRESHOLD = 0.70  # D017: hard, no degradation
TRIPLE_TARGET = 500
CAPTION_MAX_TOKENS = 50  # D017: truncate before embedding
PROBE_EMBED_SAMPLE = 10_000  # per format

DATA_ROOT = "/root/autodl-tmp/viscode_shared_subspace_probe/data"
VCM_PATH = f"{DATA_ROOT}/VisCode_filtered/"
DATIKZ_V2_PATH = f"{DATA_ROOT}/datikz_v2/"  # R18: DaTikZ v2 train only
OUT_DIR = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/stage_a")

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def truncate_caption(text: str, max_tokens: int = CAPTION_MAX_TOKENS) -> str:
    """Truncate caption to max_tokens words (whitespace-split as token proxy)."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def load_svgx_probe():
    """Load SVGX-Core-250k as SVG probe pool."""
    from datasets import load_dataset
    print("[1/3] Loading SVGX-Core-250k for SVG probe...")
    ds = load_dataset("xingxm/SVGX-Core-250k", split="train")
    items = []
    for i, cap in enumerate(ds["qwen_caption"]):
        if cap and isinstance(cap, str) and cap.strip():
            items.append((i, truncate_caption(cap.strip())))
    print(f"  SVGX probe: {len(items)}/{len(ds)} valid captions")
    return items


def load_datikz_v2_probe():
    """Load DaTikZ v2 train as TikZ probe pool (R18 pivot)."""
    import datasets
    print("[2/3] Loading DaTikZ v2 train for TikZ probe...")
    # Try loading from disk first, fallback to HF
    if os.path.exists(DATIKZ_V2_PATH):
        ds = datasets.load_from_disk(DATIKZ_V2_PATH)
        if isinstance(ds, datasets.DatasetDict):
            ds = ds["train"]
    else:
        ds = datasets.load_dataset("nllg/datikz-v2", split="train")
    items = []
    for i in range(len(ds)):
        row = ds[i]
        cap = row.get("caption", "")
        if cap and isinstance(cap, str) and cap.strip():
            items.append((i, truncate_caption(cap.strip())))
    print(f"  DaTikZ v2 train probe: {len(items)}/{len(ds)} valid captions")
    return items


def load_vcm_asy_probe():
    """Load VCM-asy from local Arrow as Asy probe pool."""
    import datasets
    print("[3/3] Loading VCM-asy from local Arrow for Asy probe...")
    vcm = datasets.load_from_disk(VCM_PATH)
    ds = vcm.filter(lambda x: x["language"] == "asymptote", num_proc=1)
    items = []
    for i in range(len(ds)):
        row = ds[i]
        msgs = row["messages"]
        caption = ""
        for m in msgs:
            if m["role"] == "user":
                caption = m["content"]
                break
        if caption and caption.strip():
            items.append((i, truncate_caption(caption.strip())))
    print(f"  VCM-asy probe: {len(items)}/{len(ds)} valid captions")
    return items


def sbert_embed(captions, model, batch_size=64):
    """Embed captions using SBERT model."""
    return model.encode(captions, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)


def greedy_match_triples_unique(svg_caps, svg_embs, tikz_caps, tikz_embs,
                                asy_caps, asy_embs, threshold):
    """Greedy match with UNIQUE constraint on tikz_idx and asy_idx.

    For each SVG caption, find best available TikZ and Asy match.
    Once a tikz_idx or asy_idx is used, it cannot be reused.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    print("  Computing cosine similarities...")
    sim_st = cosine_similarity(svg_embs, tikz_embs)  # (n_svg, n_tikz)
    sim_sa = cosine_similarity(svg_embs, asy_embs)    # (n_svg, n_asy)

    # Compute candidate score for each SVG: min(best_tikz_cos, best_asy_cos)
    best_tikz_cos = np.max(sim_st, axis=1)
    best_asy_cos = np.max(sim_sa, axis=1)
    candidate_scores = np.minimum(best_tikz_cos, best_asy_cos)

    # Sort SVG indices by candidate score descending (greedy: best first)
    svg_order = np.argsort(-candidate_scores)

    used_tikz = set()
    used_asy = set()
    triples = []

    for svg_i in svg_order:
        # Find best available tikz
        tikz_ranked = np.argsort(-sim_st[svg_i])
        best_tikz = None
        best_tikz_cos_val = 0.0
        for t_idx in tikz_ranked:
            t_idx = int(t_idx)
            if t_idx not in used_tikz:
                best_tikz = t_idx
                best_tikz_cos_val = float(sim_st[svg_i, t_idx])
                break
        if best_tikz is None:
            continue

        # Find best available asy
        asy_ranked = np.argsort(-sim_sa[svg_i])
        best_asy = None
        best_asy_cos_val = 0.0
        for a_idx in asy_ranked:
            a_idx = int(a_idx)
            if a_idx not in used_asy:
                best_asy = a_idx
                best_asy_cos_val = float(sim_sa[svg_i, a_idx])
                break
        if best_asy is None:
            continue

        min_cos = min(best_tikz_cos_val, best_asy_cos_val)
        if min_cos < threshold:
            continue  # Below threshold, skip

        used_tikz.add(best_tikz)
        used_asy.add(best_asy)
        triples.append({
            "svg_idx": int(svg_i),
            "tikz_idx": best_tikz,
            "asy_idx": best_asy,
            "svg_caption": svg_caps[svg_i],
            "tikz_caption": tikz_caps[best_tikz],
            "asy_caption": asy_caps[best_asy],
            "min_cosine": round(min_cos, 4),
            "svg_tikz_cos": round(best_tikz_cos_val, 4),
            "svg_asy_cos": round(best_asy_cos_val, 4),
        })

    # Sort by min_cosine descending
    triples.sort(key=lambda x: x["min_cosine"], reverse=True)
    return triples


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load probe pools ───────────────────────────────────────────
    svg_probe = load_svgx_probe()
    tikz_probe = load_datikz_v2_probe()
    asy_probe = load_vcm_asy_probe()

    # ── Sample for embedding ───────────────────────────────────────
    rng = random.Random(SEED)

    def sample_probe(items, n, label):
        if len(items) <= n:
            print(f"  {label}: using all {len(items)} (< {n})")
            return items
        sampled = rng.sample(items, n)
        print(f"  {label}: sampled {n}/{len(items)}")
        return sampled

    print("\nSampling probe pools...")
    svg_sample = sample_probe(svg_probe, PROBE_EMBED_SAMPLE, "SVG")
    tikz_sample = sample_probe(tikz_probe, PROBE_EMBED_SAMPLE, "TikZ")
    asy_sample = sample_probe(asy_probe, PROBE_EMBED_SAMPLE, "Asy")

    svg_caps = [x[1] for x in svg_sample]
    tikz_caps = [x[1] for x in tikz_sample]
    asy_caps = [x[1] for x in asy_sample]

    # ── SBERT embedding ────────────────────────────────────────────
    print("\nLoading SBERT model...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding probe captions...")
    svg_embs = sbert_embed(svg_caps, sbert)
    print(f"  SVG: {svg_embs.shape}")
    tikz_embs = sbert_embed(tikz_caps, sbert)
    print(f"  TikZ: {tikz_embs.shape}")
    asy_embs = sbert_embed(asy_caps, sbert)
    print(f"  Asy: {asy_embs.shape}")

    # ── Greedy matching (unique constraint) ────────────────────────
    print(f"\nGreedy matching with threshold={COSINE_THRESHOLD} (HARD, unique idx)...")
    triples = greedy_match_triples_unique(
        svg_caps, svg_embs, tikz_caps, tikz_embs,
        asy_caps, asy_embs, COSINE_THRESHOLD
    )

    print(f"\n  Result: {len(triples)} triples at cosine >= {COSINE_THRESHOLD}")
    if len(triples) < TRIPLE_TARGET:
        print(f"  ⚠️ BELOW TARGET: {len(triples)}/{TRIPLE_TARGET}")
        print(f"  D017: threshold is HARD at {COSINE_THRESHOLD}, NOT lowering.")
        print(f"  Reporting actual count for Director decision.")

    # ── Verify unique constraint ───────────────────────────────────
    tikz_indices = [t["tikz_idx"] for t in triples]
    asy_indices = [t["asy_idx"] for t in triples]
    assert len(tikz_indices) == len(set(tikz_indices)), "UNIQUE VIOLATION: duplicate tikz_idx!"
    assert len(asy_indices) == len(set(asy_indices)), "UNIQUE VIOLATION: duplicate asy_idx!"
    print(f"  ✓ Unique constraint verified: {len(set(tikz_indices))} tikz, {len(set(asy_indices))} asy")

    # ── Save ───────────────────────────────────────────────────────
    with open(OUT_DIR / "sbert_triples.json", "w") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    print(f"  Saved sbert_triples.json")

    # ── Stats ──────────────────────────────────────────────────────
    if triples:
        cosines = [t["min_cosine"] for t in triples]
        stats = {
            "count": len(triples),
            "target": TRIPLE_TARGET,
            "threshold": COSINE_THRESHOLD,
            "mean_cosine": round(float(np.mean(cosines)), 4),
            "median_cosine": round(float(np.median(cosines)), 4),
            "min_cosine": round(float(np.min(cosines)), 4),
            "max_cosine": round(float(np.max(cosines)), 4),
            "p25_cosine": round(float(np.percentile(cosines, 25)), 4),
            "unique_tikz": len(set(tikz_indices)),
            "unique_asy": len(set(asy_indices)),
            "caption_max_tokens": CAPTION_MAX_TOKENS,
        }
    else:
        stats = {"count": 0, "target": TRIPLE_TARGET, "threshold": COSINE_THRESHOLD}

    report = {
        "probe_pools": {
            "svg": {"source": "SVGX-Core-250k", "total": len(svg_probe), "sampled": len(svg_sample)},
            "tikz": {"source": "DaTikZ-v2-train", "total": len(tikz_probe), "sampled": len(tikz_sample)},
            "asy": {"source": "VCM-asy-local-Arrow", "total": len(asy_probe), "sampled": len(asy_sample)},
        },
        "sbert_triples": stats,
        "seed": SEED,
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(OUT_DIR / "step2_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved step2_report.json")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 2 v2 SUMMARY")
    print(f"{'=' * 60}")
    print(f"Probe pools: SVG={len(svg_probe)}, TikZ={len(tikz_probe)}, Asy={len(asy_probe)}")
    print(f"SBERT triples: {stats.get('count', 0)}/{TRIPLE_TARGET} (threshold={COSINE_THRESHOLD})")
    if stats.get("count", 0) > 0:
        print(f"  cosine: mean={stats['mean_cosine']}, median={stats['median_cosine']}, "
              f"min={stats['min_cosine']}, max={stats['max_cosine']}")
    print(f"Unique constraint: tikz={stats.get('unique_tikz', 0)}, asy={stats.get('unique_asy', 0)}")
    if stats.get("count", 0) < TRIPLE_TARGET:
        print(f"\n⚠️ DIRECTOR ACTION NEEDED: only {stats.get('count', 0)}/{TRIPLE_TARGET} triples.")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
