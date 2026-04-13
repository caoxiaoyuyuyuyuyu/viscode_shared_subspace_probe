#!/usr/bin/env python3
"""BPE Token-Type Jaccard Overlap between format pairs.

Computes Jaccard similarity J(A,B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|
for 6 format pairs across 252 prompts, then correlates with CKA values.

No GPU required — tokenizer only.
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)

# --- Paths (AutoDL) ---
RESOLVED_TRIPLES = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/stage_a/resolved_triples.json")
PYTHON_SNIPPETS = Path("/root/autodl-tmp/cache/hidden_states/python_snippets.json")
CKA_RESULTS = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/negative_control/negative_control_cka.json")
OUT_DIR = Path("/root/autodl-tmp/viscode_shared_subspace_probe/outputs/tokenizer_overlap")

# Qwen2.5-Coder tokenizer — try local cache first, then HF
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

PAIRS = [
    ("svg", "tikz"),
    ("svg", "asy"),
    ("tikz", "asy"),
    ("python", "svg"),
    ("python", "tikz"),
    ("python", "asy"),
]


def load_data():
    """Load code strings for all 4 formats, aligned by triple_id."""
    print("Loading resolved triples...")
    triples = json.loads(RESOLVED_TRIPLES.read_text())
    print(f"  Loaded {len(triples)} triples")

    print("Loading python snippets...")
    py_snippets = json.loads(PYTHON_SNIPPETS.read_text())
    print(f"  Loaded {len(py_snippets)} python snippets")

    n = min(len(triples), len(py_snippets), 252)
    codes = {"svg": [], "tikz": [], "asy": [], "python": []}
    for i in range(n):
        codes["svg"].append(triples[i]["svg"]["code"])
        codes["tikz"].append(triples[i]["tikz"]["code"])
        codes["asy"].append(triples[i]["asy"]["code"])
        codes["python"].append(py_snippets[i]["code"])

    print(f"  Using {n} aligned prompts")
    return codes, n


def compute_jaccard(tokenizer, codes, n):
    """Compute BPE token-type Jaccard for all 6 pairs across n prompts."""
    print("Tokenizing all code strings...")
    # Tokenize all codes — get token IDs (types = unique set per string)
    token_sets = {}
    for fmt in ["svg", "tikz", "asy", "python"]:
        token_sets[fmt] = []
        for i in range(n):
            ids = tokenizer.encode(codes[fmt][i], add_special_tokens=False)
            token_sets[fmt].append(set(ids))
        print(f"  {fmt}: median token count = {np.median([len(s) for s in token_sets[fmt]]):.0f}")

    # Compute Jaccard per pair per prompt
    results = {}
    for f1, f2 in PAIRS:
        pair_name = f"{f1}-{f2}"
        jaccards = []
        for i in range(n):
            s1 = token_sets[f1][i]
            s2 = token_sets[f2][i]
            union = len(s1 | s2)
            if union == 0:
                jaccards.append(0.0)
            else:
                jaccards.append(len(s1 & s2) / union)
        results[pair_name] = jaccards
        mean_j = np.mean(jaccards)
        std_j = np.std(jaccards)
        print(f"  {pair_name}: mean Jaccard = {mean_j:.4f} +/- {std_j:.4f}")

    return results


def load_cka_data():
    """Load CKA results and build (pair, model, layer) -> cka dict."""
    print("Loading CKA results...")
    cka = json.loads(CKA_RESULTS.read_text())

    cka_lookup = {}
    for row in cka.get("python_x", []):
        key = (row["format_pair"], row["model"], row["layer"])
        cka_lookup[key] = row["cka"]
    for row in cka.get("visual_x", []):
        key = (row["format_pair"], row["model"], row["layer"])
        cka_lookup[key] = row["cka"]

    print(f"  Loaded {len(cka_lookup)} CKA cells")
    return cka_lookup


def compute_spearman(jaccard_results, cka_lookup):
    """Spearman rank correlation: mean Jaccard rank vs CKA value across 126 observations."""
    # Map pair names between Jaccard and CKA formats
    pair_map = {
        "svg-tikz": "svg-tikz",
        "svg-asy": "svg-asy",
        "tikz-asy": "tikz-asy",
        "python-svg": "python-svg",
        "python-tikz": "python-tikz",
        "python-asy": "python-asy",
    }

    models = ["coder", "viscoder2", "qwen25"]
    layers = [4, 8, 12, 16, 20, 24, 28]

    jaccard_vals = []
    cka_vals = []
    labels = []

    for pair_name in pair_map:
        cka_pair = pair_map[pair_name]
        mean_j = float(np.mean(jaccard_results[pair_name]))
        for model in models:
            for layer in layers:
                key = (cka_pair, model, layer)
                if key in cka_lookup:
                    jaccard_vals.append(mean_j)
                    cka_vals.append(cka_lookup[key])
                    labels.append(f"{pair_name}/{model}/L{layer}")

    n_obs = len(jaccard_vals)
    print(f"\nSpearman correlation: {n_obs} observations")

    if n_obs < 10:
        print("  ERROR: too few observations for correlation")
        return None

    rho, p_val = stats.spearmanr(jaccard_vals, cka_vals)
    print(f"  rho = {rho:.4f}, p = {p_val:.2e}")

    # Also compute per-pair mean CKA for ranking comparison
    print("\n  Per-pair summary (mean Jaccard | mean CKA across models×layers):")
    for pair_name in pair_map:
        cka_pair = pair_map[pair_name]
        mean_j = float(np.mean(jaccard_results[pair_name]))
        cka_list = []
        for model in models:
            for layer in layers:
                key = (cka_pair, model, layer)
                if key in cka_lookup:
                    cka_list.append(cka_lookup[key])
        mean_cka = np.mean(cka_list) if cka_list else float("nan")
        print(f"    {pair_name:15s}: Jaccard={mean_j:.4f}  CKA={mean_cka:.4f}")

    return {"rho": float(rho), "p_value": float(p_val), "n_observations": n_obs}


def main():
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    codes, n = load_data()
    jaccard_results = compute_jaccard(tokenizer, codes, n)
    cka_lookup = load_cka_data()
    spearman = compute_spearman(jaccard_results, cka_lookup)

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "n_prompts": n,
        "tokenizer": TOKENIZER_NAME,
        "pair_jaccard": {
            pair: {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "median": round(float(np.median(vals)), 4),
            }
            for pair, vals in jaccard_results.items()
        },
        "spearman": spearman,
    }
    out_path = OUT_DIR / "tokenizer_overlap_jaccard.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print verdict
    if spearman:
        rho = spearman["rho"]
        p = spearman["p_value"]
        if rho > 0.7 and p < 0.01:
            print(f"\n>>> VERDICT: rho={rho:.3f} > 0.7, p={p:.2e} < 0.01 → CKA asymmetry explained by token overlap → F3 should be 'tokenizer effect'")
        elif rho < 0.5:
            print(f"\n>>> VERDICT: rho={rho:.3f} < 0.5 → token overlap does NOT explain CKA → F3 'syntax family' attribution can be retained")
        else:
            print(f"\n>>> VERDICT: rho={rho:.3f} in [0.5, 0.7] → partial overlap effect → F3 needs nuanced framing")


if __name__ == "__main__":
    main()
