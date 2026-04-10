#!/usr/bin/env python3
"""Sentence-BERT cross-format triple matching for probe pool.

Matches (SVG, TikZ, Asymptote) caption triples using all-MiniLM-L6-v2,
filters by cosine ≥ threshold, keeps top-K triples.
Per pre_registration §3-quater.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path


def load_captions(data_path: str, caption_field: str = "caption") -> list[str]:
    """Load captions from JSON, JSONL, CSV, or HF dataset identifier."""
    if data_path.startswith("hf://") or "/" in data_path and not os.path.exists(data_path):
        # Try loading from HuggingFace datasets
        from datasets import load_dataset
        ds_name = data_path.replace("hf://", "")
        parts = ds_name.split(":")
        ds_id = parts[0]
        split = parts[1] if len(parts) > 1 else "train"
        print(f"  Loading HF dataset: {ds_id} split={split}")
        ds = load_dataset(ds_id, split=split, cache_dir=os.environ.get("HF_HOME"))
        if caption_field in ds.column_names:
            return ds[caption_field]
        # Try common field names
        for field in ["caption", "text", "description", "prompt", "instruction"]:
            if field in ds.column_names:
                print(f"  Using field '{field}' (requested '{caption_field}' not found)")
                return ds[field]
        raise ValueError(f"No caption field found in {ds.column_names}")

    path = Path(data_path)
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            return [d[caption_field] for d in data]
        return [d[caption_field] for d in data.values()]

    elif path.suffix == ".jsonl":
        captions = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                captions.append(d[caption_field])
        return captions

    elif path.suffix == ".csv":
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            return [row[caption_field] for row in reader]

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def encode_captions(captions: list[str], model_name: str, cache_dir: str) -> np.ndarray:
    """Encode captions using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"  Encoding {len(captions)} captions with {model_name}...")
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    embeddings = model.encode(captions, show_progress_bar=True, batch_size=64)
    return np.array(embeddings)


def greedy_match_triples(
    svg_captions: list[str],
    tikz_captions: list[str],
    asym_captions: list[str],
    svg_embs: np.ndarray,
    tikz_embs: np.ndarray,
    asym_embs: np.ndarray,
    threshold: float = 0.7,
    top_k: int = 500,
) -> list[dict]:
    """Greedy match triples maximizing average pairwise cosine."""
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"\nComputing pairwise cosine similarities...")
    # Normalize for cosine
    svg_norm = svg_embs / np.linalg.norm(svg_embs, axis=1, keepdims=True)
    tikz_norm = tikz_embs / np.linalg.norm(tikz_embs, axis=1, keepdims=True)
    asym_norm = asym_embs / np.linalg.norm(asym_embs, axis=1, keepdims=True)

    # Pairwise cosine: svg-tikz, svg-asym, tikz-asym
    sim_st = cosine_similarity(svg_norm, tikz_norm)  # [n_svg, n_tikz]
    sim_sa = cosine_similarity(svg_norm, asym_norm)   # [n_svg, n_asym]
    sim_ta = cosine_similarity(tikz_norm, asym_norm)  # [n_tikz, n_asym]

    print(f"  SVG×TikZ: {sim_st.shape}, SVG×Asym: {sim_sa.shape}, TikZ×Asym: {sim_ta.shape}")

    # Greedy search: for each SVG caption, find best (TikZ, Asym) pair
    # Score = mean of 3 pairwise cosines; all 3 must be ≥ threshold
    candidates = []
    used_tikz = set()
    used_asym = set()

    # Pre-compute combined scores for efficiency
    n_svg = len(svg_captions)
    n_tikz = len(tikz_captions)
    n_asym = len(asym_captions)

    print(f"  Searching {n_svg} × {n_tikz} × {n_asym} space...")

    # For each SVG, find top TikZ matches, then for each pair find top Asym
    for s_idx in range(n_svg):
        best_score = -1
        best_triple = None

        # Get top-K tikz candidates for this SVG
        tikz_scores = sim_st[s_idx]
        tikz_candidates = np.argsort(tikz_scores)[::-1][:50]  # top 50

        for t_idx in tikz_candidates:
            if t_idx in used_tikz:
                continue
            st_sim = tikz_scores[t_idx]
            if st_sim < threshold:
                break  # sorted, no more valid

            # Get top Asym candidates for this (SVG, TikZ) pair
            asym_scores_s = sim_sa[s_idx]
            asym_scores_t = sim_ta[t_idx]
            avg_asym = (asym_scores_s + asym_scores_t) / 2
            asym_candidates = np.argsort(avg_asym)[::-1][:20]

            for a_idx in asym_candidates:
                if a_idx in used_asym:
                    continue
                sa_sim = asym_scores_s[a_idx]
                ta_sim = asym_scores_t[a_idx]

                # All three pairwise cosines must be ≥ threshold
                min_sim = min(st_sim, sa_sim, ta_sim)
                if min_sim < threshold:
                    continue

                avg_sim = (st_sim + sa_sim + ta_sim) / 3
                if avg_sim > best_score:
                    best_score = avg_sim
                    best_triple = (s_idx, t_idx, a_idx, st_sim, sa_sim, ta_sim, avg_sim)

        if best_triple is not None:
            s_idx, t_idx, a_idx, st, sa, ta, avg = best_triple
            used_tikz.add(t_idx)
            used_asym.add(a_idx)
            candidates.append({
                "triple_id": len(candidates),
                "svg_idx": int(s_idx),
                "tikz_idx": int(t_idx),
                "asym_idx": int(a_idx),
                "svg_caption": svg_captions[s_idx],
                "tikz_caption": tikz_captions[t_idx],
                "asym_caption": asym_captions[a_idx],
                "cos_svg_tikz": round(float(st), 4),
                "cos_svg_asym": round(float(sa), 4),
                "cos_tikz_asym": round(float(ta), 4),
                "avg_cosine": round(float(avg), 4),
                "min_cosine": round(float(min(st, sa, ta)), 4),
            })

    # Sort by average cosine, keep top-K
    candidates.sort(key=lambda x: x["avg_cosine"], reverse=True)
    triples = candidates[:top_k]

    # Re-index
    for i, t in enumerate(triples):
        t["triple_id"] = i

    print(f"\n  Found {len(candidates)} valid triples (threshold={threshold})")
    print(f"  Kept top {len(triples)} (requested {top_k})")
    if triples:
        cosines = [t["avg_cosine"] for t in triples]
        print(f"  Avg cosine range: [{min(cosines):.4f}, {max(cosines):.4f}]")

    return triples


def sample_sanity(triples: list[dict], n: int = 10) -> list[dict]:
    """Random sample N triples for human sanity check."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(triples), size=min(n, len(triples)), replace=False)
    samples = [triples[i] for i in sorted(indices)]

    print(f"\n{'=' * 70}")
    print(f"SANITY CHECK: {len(samples)} random triples")
    print(f"{'=' * 70}")
    for s in samples:
        print(f"\n  Triple #{s['triple_id']} (avg_cos={s['avg_cosine']:.4f}):")
        print(f"    SVG:  {s['svg_caption'][:80]}")
        print(f"    TikZ: {s['tikz_caption'][:80]}")
        print(f"    Asym: {s['asym_caption'][:80]}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Sentence-BERT cross-format triple matching")
    parser.add_argument("--svg_data", required=True, help="SVG captions (JSON/JSONL/CSV or hf://repo:split)")
    parser.add_argument("--tikz_data", required=True, help="TikZ captions")
    parser.add_argument("--asymptote_data", required=True, help="Asymptote captions")
    parser.add_argument("--caption_field", default="caption", help="Field name for captions")
    parser.add_argument("--output_path", default="/root/autodl-tmp/viscode_shared_subspace_probe/outputs/sbert/triples.json")
    parser.add_argument("--threshold", type=float, default=0.7, help="Min pairwise cosine threshold")
    parser.add_argument("--top_k", type=int, default=500, help="Number of triples to keep")
    parser.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/.hf_cache")
    parser.add_argument("--sanity_n", type=int, default=10, help="Number of triples to sample for sanity")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load captions
    print("Loading captions...")
    svg_captions = load_captions(args.svg_data, args.caption_field)
    tikz_captions = load_captions(args.tikz_data, args.caption_field)
    asym_captions = load_captions(args.asymptote_data, args.caption_field)

    print(f"  SVG: {len(svg_captions)}, TikZ: {len(tikz_captions)}, Asymptote: {len(asym_captions)}")

    # Encode
    print("\nEncoding with sentence-BERT...")
    svg_embs = encode_captions(svg_captions, args.sbert_model, args.cache_dir)
    tikz_embs = encode_captions(tikz_captions, args.sbert_model, args.cache_dir)
    asym_embs = encode_captions(asym_captions, args.sbert_model, args.cache_dir)

    # Match
    triples = greedy_match_triples(
        svg_captions, tikz_captions, asym_captions,
        svg_embs, tikz_embs, asym_embs,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    if not triples:
        print("\nERROR: No valid triples found!")
        sys.exit(1)

    # Sanity sample
    samples = sample_sanity(triples, args.sanity_n)

    # Save
    output = {
        "metadata": {
            "sbert_model": args.sbert_model,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "n_svg": len(svg_captions),
            "n_tikz": len(tikz_captions),
            "n_asymptote": len(asym_captions),
            "n_triples_found": len(triples),
        },
        "triples": triples,
        "sanity_samples": samples,
    }

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(triples)} triples to {args.output_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
