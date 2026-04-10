#!/usr/bin/env python3
"""Sentence-BERT greedy triplet matching across SVG/TikZ/Asymptote datasets."""

import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_captions(data_path: str) -> list[str]:
    """Load captions from HF dataset identifier or local JSON/CSV file."""
    path = Path(data_path)

    if path.exists():
        # Local file
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                if isinstance(data[0], str):
                    return data
                # List of dicts — look for caption/text field
                for key in ("caption", "text", "prompt", "description"):
                    if key in data[0]:
                        return [d[key] for d in data]
                raise ValueError(f"Cannot find caption field in JSON. Keys: {list(data[0].keys())}")
            raise ValueError(f"Unexpected JSON structure in {path}")

        elif path.suffix == ".csv":
            import csv
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for key in ("caption", "text", "prompt", "description"):
                if key in rows[0]:
                    return [r[key] for r in rows]
            raise ValueError(f"Cannot find caption field in CSV. Columns: {list(rows[0].keys())}")

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    else:
        # HF dataset identifier (e.g., "username/dataset_name")
        from datasets import load_dataset

        ds = load_dataset(data_path)
        # Try train or first available split
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        split = ds[split_name]

        for key in ("caption", "text", "prompt", "description"):
            if key in split.column_names:
                return split[key]

        raise ValueError(
            f"Cannot find caption field in HF dataset {data_path}. "
            f"Columns: {split.column_names}"
        )


def encode_captions(captions: list[str], cache_dir: str) -> np.ndarray:
    """Encode captions using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir)
    embeddings = model.encode(captions, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def greedy_triplet_matching(
    svg_caps: list[str],
    tikz_caps: list[str],
    asym_caps: list[str],
    svg_embs: np.ndarray,
    tikz_embs: np.ndarray,
    asym_embs: np.ndarray,
    threshold: float,
    top_k: int,
) -> list[dict]:
    """Greedy matching to form (svg, tikz, asymptote) triplets maximizing avg pairwise cosine."""
    from sklearn.metrics.pairwise import cosine_similarity

    print("Computing pairwise cosine similarities...")
    # svg vs tikz
    sim_st = cosine_similarity(svg_embs, tikz_embs)
    # svg vs asymptote
    sim_sa = cosine_similarity(svg_embs, asym_embs)
    # tikz vs asymptote
    sim_ta = cosine_similarity(tikz_embs, asym_embs)

    n_svg, n_tikz, n_asym = len(svg_caps), len(tikz_caps), len(asym_caps)

    # Build candidate triplets with scores
    print("Building candidate triplets (greedy)...")
    candidates = []

    # For each SVG caption, find best tikz and asymptote matches
    used_tikz = set()
    used_asym = set()

    # Sort SVG indices by max potential score (heuristic: sum of best tikz + best asym sim)
    svg_priority = np.argsort(-(sim_st.max(axis=1) + sim_sa.max(axis=1)))

    for si in svg_priority:
        # Find best available tikz
        tikz_order = np.argsort(-sim_st[si])
        best_ti = None
        for ti in tikz_order:
            if ti not in used_tikz:
                best_ti = ti
                break
        if best_ti is None:
            continue

        # Find best available asymptote
        asym_order = np.argsort(-(sim_sa[si] + sim_ta[best_ti]) / 2)
        best_ai = None
        for ai in asym_order:
            if ai not in used_asym:
                best_ai = ai
                break
        if best_ai is None:
            continue

        # Compute pairwise cosines
        cos_st = float(sim_st[si, best_ti])
        cos_sa = float(sim_sa[si, best_ai])
        cos_ta = float(sim_ta[best_ti, best_ai])
        min_cos = min(cos_st, cos_sa, cos_ta)
        avg_cos = (cos_st + cos_sa + cos_ta) / 3

        # Filter: min pairwise cosine >= threshold
        if min_cos >= threshold:
            candidates.append({
                "svg_idx": int(si),
                "tikz_idx": int(best_ti),
                "asym_idx": int(best_ai),
                "svg_caption": svg_caps[si],
                "tikz_caption": tikz_caps[best_ti],
                "asymptote_caption": asym_caps[best_ai],
                "cos_svg_tikz": cos_st,
                "cos_svg_asym": cos_sa,
                "cos_tikz_asym": cos_ta,
                "min_pairwise_cos": min_cos,
                "avg_pairwise_cos": avg_cos,
            })
            used_tikz.add(best_ti)
            used_asym.add(best_ai)

        if len(candidates) >= top_k:
            break

    # Sort by avg pairwise cosine descending
    candidates.sort(key=lambda x: x["avg_pairwise_cos"], reverse=True)
    candidates = candidates[:top_k]

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Sentence-BERT triplet matching")
    parser.add_argument("--svg_data", required=True, help="SVG captions: HF dataset ID or local JSON/CSV")
    parser.add_argument("--tikz_data", required=True, help="TikZ captions: HF dataset ID or local JSON/CSV")
    parser.add_argument("--asymptote_data", required=True, help="Asymptote captions: HF dataset ID or local JSON/CSV")
    parser.add_argument("--output_path", default="artifacts/sbert_triplets.json")
    parser.add_argument("--threshold", type=float, default=0.7, help="Min pairwise cosine threshold")
    parser.add_argument("--top_k", type=int, default=500, help="Max triplets to keep")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/.hf_cache")
    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load captions
    print("=== Loading captions ===")
    print(f"  SVG: {args.svg_data}")
    svg_caps = load_captions(args.svg_data)
    print(f"    → {len(svg_caps)} captions")

    print(f"  TikZ: {args.tikz_data}")
    tikz_caps = load_captions(args.tikz_data)
    print(f"    → {len(tikz_caps)} captions")

    print(f"  Asymptote: {args.asymptote_data}")
    asym_caps = load_captions(args.asymptote_data)
    print(f"    → {len(asym_caps)} captions")

    # Encode
    print("\n=== Encoding captions ===")
    print("  Encoding SVG captions...")
    svg_embs = encode_captions(svg_caps, args.cache_dir)
    print("  Encoding TikZ captions...")
    tikz_embs = encode_captions(tikz_caps, args.cache_dir)
    print("  Encoding Asymptote captions...")
    asym_embs = encode_captions(asym_caps, args.cache_dir)

    # Match
    print(f"\n=== Greedy triplet matching (threshold={args.threshold}, top_k={args.top_k}) ===")
    triplets = greedy_triplet_matching(
        svg_caps, tikz_caps, asym_caps,
        svg_embs, tikz_embs, asym_embs,
        args.threshold, args.top_k,
    )
    print(f"  Matched {len(triplets)} triplets")

    # Sample 10 for inspection
    if triplets:
        print("\n=== Sample triplets (10 random) ===")
        sample = random.sample(triplets, min(10, len(triplets)))
        for i, t in enumerate(sample):
            print(f"\n  Triplet {i+1} (avg cos: {t['avg_pairwise_cos']:.3f}, min cos: {t['min_pairwise_cos']:.3f}):")
            print(f"    SVG:       {t['svg_caption']}")
            print(f"    TikZ:      {t['tikz_caption']}")
            print(f"    Asymptote: {t['asymptote_caption']}")

    # Save
    output = {
        "config": {
            "svg_data": args.svg_data,
            "tikz_data": args.tikz_data,
            "asymptote_data": args.asymptote_data,
            "threshold": args.threshold,
            "top_k": args.top_k,
        },
        "stats": {
            "svg_count": len(svg_caps),
            "tikz_count": len(tikz_caps),
            "asymptote_count": len(asym_caps),
            "matched_triplets": len(triplets),
        },
        "triplets": triplets,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(triplets)} triplets to: {args.output_path}")


if __name__ == "__main__":
    main()
