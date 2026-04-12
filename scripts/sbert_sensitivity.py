"""SBERT threshold sensitivity analysis (C3).

For each threshold in {0.60, 0.65, 0.70, 0.75, 0.80}, filter triples by
min_cosine >= threshold, load layer-28 hidden states, compute mean linear CKA
across 3 format pairs per model.
"""
import os, json, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

# ── paths ──
ROOT = "/root/autodl-tmp/viscode_shared_subspace_probe"
CACHE = "/root/autodl-tmp/cache/hidden_states"
TRIPLES = os.path.join(ROOT, "outputs/stage_a/sbert_triples.json")
OUT_DIR = os.path.join(ROOT, "outputs/sbert_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["coder", "viscoder2", "qwen25"]
FORMATS = ["svg", "tikz", "asy"]
LAYER_IDX = 6  # layer 28 = index 6 in LAYERS=[4,8,12,16,20,24,28]
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
FORMAT_PAIRS = [("svg", "tikz"), ("svg", "asy"), ("tikz", "asy")]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    n = X.shape[0]
    H = np.eye(n) - 1.0 / n
    KX = H @ (X @ X.T) @ H
    KY = H @ (Y @ Y.T) @ H
    hsic_xy = np.sum(KX * KY)
    hsic_xx = np.sum(KX * KX)
    hsic_yy = np.sum(KY * KY)
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def load_hidden(model: str, fmt: str, idx: int) -> np.ndarray:
    """Load layer-28 hidden state for one (model, format, triple_idx)."""
    path = os.path.join(CACHE, model, fmt, f"{idx}.pt")
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t[LAYER_IDX].numpy()  # shape [hidden_dim]


def main():
    with open(TRIPLES) as f:
        triples = json.load(f)
    print(f"Total triples: {len(triples)}")

    # Pre-load all hidden states (252 × 3 models × 3 formats)
    # Each is a 1-D vector [hidden_dim]
    all_hidden = {}
    for model in MODELS:
        for fmt in FORMATS:
            for idx in range(len(triples)):
                all_hidden[(model, fmt, idx)] = load_hidden(model, fmt, idx)
    print("Hidden states loaded.")

    results = {
        "thresholds": THRESHOLDS,
        "n_triples": [],
        "cka_by_threshold": {},
        "cka_by_threshold_pair": {},  # detailed per-pair breakdown
    }

    for thr in THRESHOLDS:
        passing = [i for i, t in enumerate(triples) if t["min_cosine"] >= thr]
        n_pass = len(passing)
        results["n_triples"].append(n_pass)
        print(f"\nThreshold {thr:.2f}: {n_pass} triples")

        if n_pass < 2:
            results["cka_by_threshold"][str(thr)] = {m: None for m in MODELS}
            results["cka_by_threshold_pair"][str(thr)] = {m: {} for m in MODELS}
            continue

        model_cka = {}
        model_pair_cka = {}
        for model in MODELS:
            # Build matrices [n_pass, hidden_dim] per format
            mats = {}
            for fmt in FORMATS:
                mats[fmt] = np.stack([all_hidden[(model, fmt, i)] for i in passing])

            pair_ckas = {}
            for f1, f2 in FORMAT_PAIRS:
                c = linear_cka(mats[f1], mats[f2])
                pair_ckas[f"{f1}-{f2}"] = round(c, 4)

            mean_cka = np.mean(list(pair_ckas.values()))
            model_cka[model] = round(float(mean_cka), 4)
            model_pair_cka[model] = pair_ckas
            print(f"  {model}: mean_cka={model_cka[model]:.4f}  {pair_ckas}")

        results["cka_by_threshold"][str(thr)] = model_cka
        results["cka_by_threshold_pair"][str(thr)] = model_pair_cka

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
