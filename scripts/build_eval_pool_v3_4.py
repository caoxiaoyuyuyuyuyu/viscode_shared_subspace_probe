#!/usr/bin/env python3
"""
build_eval_pool_v3_4.py — Build eval pools per v3.4 §3 spec.

Produces 3 JSONL files (svg/tikz/asymptote, 200 lines each) + build_report.json.
Follows pre_registration_v3_4_skeleton.md §3 locked spec:
  - SVG  = VisPlotBench svg 65 (all) + VCM-svg 135 (random sample after filter)
  - TikZ = DaTikZ v1 test 200 (random sample after filter)
  - Asy  = VisPlotBench asy 92 (all) + VCM-asy 108 (random sample after filter)

Filter rules (applied to code-having sources: VCM, DaTikZ):
  1. code length ∈ [50, 2000]
  2. caption length ∈ [10, 300]
  3. hash dedup (sha256 of code)
  4. seed = 20260410

VisPlotBench entries are prompt-only (no reference code); included as-is per "全用" spec.
"""

import json
import hashlib
import datetime
import sys
import os

import numpy as np
import datasets

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 20260410
DATA_ROOT = "/root/autodl-tmp/viscode_shared_subspace_probe/data"
VCM_PATH = f"{DATA_ROOT}/VisCode_filtered/"
VPB_SVG_PATH = f"{DATA_ROOT}/VisPlotBench_svg/"
VPB_ASY_PATH = f"{DATA_ROOT}/VisPlotBench_asymptote/"
DATIKZ_PATH = f"{DATA_ROOT}/datikz/"
OUT_DIR = "/root/autodl-tmp/viscode_shared_subspace_probe/artifacts/stage_a/eval_pool/v3_4/"

TARGET = {"svg": 200, "tikz": 200, "asymptote": 200}


# ── Helpers ───────────────────────────────────────────────────────────────────
def code_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def passes_length_filter(code: str, caption: str) -> bool:
    return 50 <= len(code) <= 2000 and 10 <= len(caption) <= 300


def make_entry(id_str: str, fmt: str, source: str, code: str, caption: str) -> dict:
    return {
        "id": id_str,
        "format": fmt,
        "source": source,
        "code": code,
        "caption": caption,
        "code_sha256": code_sha256(code) if code else "",
        "code_len": len(code),
        "caption_len": len(caption),
    }


def extract_vcm_fields(row: dict) -> tuple:
    """Extract (caption, code) from VCM messages format."""
    msgs = row["messages"]
    caption = ""
    code = ""
    for m in msgs:
        if m["role"] == "user":
            caption = m["content"]
        elif m["role"] == "assistant":
            code = m["content"]
    return caption, code


def write_jsonl(entries: list, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return file_sha256(path)


# ── Validate paths ────────────────────────────────────────────────────────────
for p in [VCM_PATH, VPB_SVG_PATH, VPB_ASY_PATH, DATIKZ_PATH]:
    if not os.path.exists(p):
        raise SystemExit(f"FAIL: source path not found: {p}")

os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
seen_hashes: set = set()
report = {
    "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "seed": SEED,
    "source_counts": {},
    "filter_rejection_counts": {},
    "sha256": {},
    "raised": False,
}


# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")
vcm = datasets.load_from_disk(VCM_PATH)
vpb_svg = datasets.load_from_disk(VPB_SVG_PATH)["test"]
vpb_asy = datasets.load_from_disk(VPB_ASY_PATH)["test"]
datikz = datasets.load_from_disk(DATIKZ_PATH)["test"]

print(f"  VCM: {len(vcm)} rows")
print(f"  VisPlotBench SVG: {len(vpb_svg)} rows")
print(f"  VisPlotBench ASY: {len(vpb_asy)} rows")
print(f"  DaTikZ v1 test: {len(datikz)} rows")

# Pre-filter VCM by language (much faster than row-by-row)
print("  Pre-filtering VCM by language...")
vcm_svg_ds = vcm.filter(lambda x: x["language"] == "svg", num_proc=1)
vcm_asy_ds = vcm.filter(lambda x: x["language"] == "asymptote", num_proc=1)
print(f"  VCM-svg: {len(vcm_svg_ds)}, VCM-asy: {len(vcm_asy_ds)}")


# ── Helper: filter + sample from VCM ─────────────────────────────────────────
def filter_vcm_ds(ds):
    """Apply length + dedup filters to pre-filtered VCM split. Returns list of (code, caption)."""
    filtered = []
    rejected = 0
    for i in range(len(ds)):
        row = ds[i]
        caption, code = extract_vcm_fields(row)
        if not passes_length_filter(code, caption):
            rejected += 1
            continue
        h = code_sha256(code)
        if h in seen_hashes:
            rejected += 1
            continue
        seen_hashes.add(h)
        filtered.append((code, caption))
    return filtered, rejected


def filter_datikz(ds):
    """Filter DaTikZ by length + dedup. Returns list of (code, caption)."""
    filtered = []
    rejected = 0
    for i in range(len(ds)):
        row = ds[i]
        code = row["code"]
        caption = row["caption"]
        if not passes_length_filter(code, caption):
            rejected += 1
            continue
        h = code_sha256(code)
        if h in seen_hashes:
            rejected += 1
            continue
        seen_hashes.add(h)
        filtered.append((code, caption))
    return filtered, rejected


def sample_n(filtered: list, n: int) -> list:
    """Random sample n items from filtered list using global rng."""
    if len(filtered) < n:
        return filtered  # take all available (will fail target check later)
    indices = rng.choice(len(filtered), size=n, replace=False)
    indices.sort()
    return [filtered[i] for i in indices]


# ── Build SVG pool ────────────────────────────────────────────────────────────
print("\nBuilding SVG pool...")
svg_pool = []

# VisPlotBench SVG: all 65
for i in range(len(vpb_svg)):
    r = vpb_svg[i]
    caption = r["task__plot_description"]
    entry = make_entry(f"svg_VisPlotBench_{i}", "svg", "VisPlotBench", "", caption)
    svg_pool.append(entry)
print(f"  VisPlotBench SVG: {len(vpb_svg)} entries (all)")

# VCM-SVG: filter + sample 135
print("  Filtering VCM-svg...")
vcm_svg_filtered, vcm_svg_rejected = filter_vcm_ds(vcm_svg_ds)
print(f"  VCM-svg after filter: {len(vcm_svg_filtered)} (rejected {vcm_svg_rejected})")
vcm_svg_sampled = sample_n(vcm_svg_filtered, 135)
for j, (code, caption) in enumerate(vcm_svg_sampled):
    entry = make_entry(f"svg_VCM-svg_{j}", "svg", "VCM-svg", code, caption)
    svg_pool.append(entry)
print(f"  VCM-svg sampled: {len(vcm_svg_sampled)}")

report["source_counts"]["svg_VisPlotBench"] = len(vpb_svg)
report["source_counts"]["svg_VCM"] = len(vcm_svg_sampled)
report["filter_rejection_counts"]["vcm_svg"] = vcm_svg_rejected


# ── Build TikZ pool ──────────────────────────────────────────────────────────
print("\nBuilding TikZ pool...")
tikz_pool = []

print("  Filtering DaTikZ v1 test...")
datikz_filtered, datikz_rejected = filter_datikz(datikz)
print(f"  DaTikZ after filter: {len(datikz_filtered)} (rejected {datikz_rejected})")
datikz_sampled = sample_n(datikz_filtered, 200)
for j, (code, caption) in enumerate(datikz_sampled):
    entry = make_entry(f"tikz_DaTikZ-v1-test_{j}", "tikz", "DaTikZ-v1-test", code, caption)
    tikz_pool.append(entry)
print(f"  DaTikZ sampled: {len(datikz_sampled)}")

report["source_counts"]["tikz_DaTikZ"] = len(datikz_sampled)
report["filter_rejection_counts"]["datikz"] = datikz_rejected


# ── Build Asymptote pool ─────────────────────────────────────────────────────
print("\nBuilding Asymptote pool...")
asy_pool = []

# VisPlotBench Asy: all 92
for i in range(len(vpb_asy)):
    r = vpb_asy[i]
    caption = r["task__plot_description"]
    entry = make_entry(f"asymptote_VisPlotBench_{i}", "asymptote", "VisPlotBench", "", caption)
    asy_pool.append(entry)
print(f"  VisPlotBench ASY: {len(vpb_asy)} entries (all)")

# VCM-Asy: filter + sample 108
print("  Filtering VCM-asy...")
vcm_asy_filtered, vcm_asy_rejected = filter_vcm_ds(vcm_asy_ds)
print(f"  VCM-asy after filter: {len(vcm_asy_filtered)} (rejected {vcm_asy_rejected})")
vcm_asy_sampled = sample_n(vcm_asy_filtered, 108)
for j, (code, caption) in enumerate(vcm_asy_sampled):
    entry = make_entry(f"asymptote_VCM-asy_{j}", "asymptote", "VCM-asy", code, caption)
    asy_pool.append(entry)
print(f"  VCM-asy sampled: {len(vcm_asy_sampled)}")

report["source_counts"]["asy_VisPlotBench"] = len(vpb_asy)
report["source_counts"]["asy_VCM"] = len(vcm_asy_sampled)
report["filter_rejection_counts"]["vcm_asy"] = vcm_asy_rejected


# ── Target check (fail-fast) ─────────────────────────────────────────────────
pools = {"svg": svg_pool, "tikz": tikz_pool, "asymptote": asy_pool}
for fmt, pool in pools.items():
    if len(pool) < TARGET[fmt]:
        report["raised"] = True
        report_path = os.path.join(OUT_DIR, "build_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        raise SystemExit(f"FAIL: {fmt} pool only {len(pool)}/{TARGET[fmt]}")


# ── Write JSONL files ─────────────────────────────────────────────────────────
print("\nWriting JSONL files...")
for fmt, pool in pools.items():
    path = os.path.join(OUT_DIR, f"{fmt}.jsonl")
    sha = write_jsonl(pool, path)
    report["sha256"][fmt] = sha
    report["source_counts"][f"{fmt}_total"] = len(pool)
    print(f"  {path}: {len(pool)} lines, SHA256={sha}")


# ── Write build report ────────────────────────────────────────────────────────
report_path = os.path.join(OUT_DIR, "build_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nBuild report: {report_path}")
print("BUILD SUCCESS")
