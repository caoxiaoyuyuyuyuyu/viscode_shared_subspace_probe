#!/usr/bin/env python
"""Local dry-run: verify stage_b_probe.py resolve logic against actual sbert_triples.json schema.

Tests:
1. Mock sbert_triples.json matches step2 output schema (no triple_id field)
2. resolve_probe_triples() handles enumerate-based triple_id correctly
3. Output resolved_triples.json has expected structure
4. All idx lookups within bounds

No GPU needed. Uses mock datasets to avoid downloading real data.
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Create mock sbert_triples.json (matching step2 schema exactly) ──
MOCK_TRIPLES = [
    {
        "svg_idx": 0,
        "tikz_idx": 0,
        "asy_idx": 0,
        "svg_caption": "A bar chart showing revenue",
        "tikz_caption": "A diagram of a neural network",
        "asy_caption": "A 3D surface plot",
        "min_cosine": 0.82,
        "svg_tikz_cos": 0.85,
        "svg_asy_cos": 0.82,
    },
    {
        "svg_idx": 1,
        "tikz_idx": 1,
        "asy_idx": 1,
        "svg_caption": "A pie chart of expenses",
        "tikz_caption": "A flowchart diagram",
        "asy_caption": "A geometric shape",
        "min_cosine": 0.75,
        "svg_tikz_cos": 0.78,
        "svg_asy_cos": 0.75,
    },
    {
        "svg_idx": 2,
        "tikz_idx": 2,
        "asy_idx": 2,
        "svg_caption": "A line graph of temperature",
        "tikz_caption": "A tree structure",
        "asy_caption": "A contour plot",
        "min_cosine": 0.71,
        "svg_tikz_cos": 0.73,
        "svg_asy_cos": 0.71,
    },
]

# Verify mock matches step2 schema
STEP2_KEYS = {"svg_idx", "tikz_idx", "asy_idx", "svg_caption", "tikz_caption",
              "asy_caption", "min_cosine", "svg_tikz_cos", "svg_asy_cos"}

for i, t in enumerate(MOCK_TRIPLES):
    assert set(t.keys()) == STEP2_KEYS, f"Triple {i} keys mismatch: {set(t.keys())} vs {STEP2_KEYS}"
    assert "triple_id" not in t, f"Triple {i} should NOT have triple_id (step2 doesn't produce it)"

print("✓ Mock triples match step2 schema (no triple_id)")


# ── Mock datasets ──
def make_mock_svgx(n=10):
    """Mock SVGX-Core-250k with svg_code and qwen_caption."""
    rows = []
    for i in range(n):
        rows.append({
            "qwen_caption": f"Caption SVG {i}",
            "svg_code": f"<svg><circle r='{i}'/></svg>",
        })
    return rows


def make_mock_tikz(n=10):
    """Mock DaTikZ v2 with caption and code."""
    rows = []
    for i in range(n):
        rows.append({
            "caption": f"Caption TikZ {i}",
            "code": f"\\begin{{tikzpicture}}\\draw ({i},0);\\end{{tikzpicture}}",
        })
    return rows


def make_mock_asy(n=10):
    """Mock VCM-asy with messages format."""
    rows = []
    for i in range(n):
        rows.append({
            "language": "asymptote",
            "messages": [
                {"role": "user", "content": f"Caption Asy {i}"},
                {"role": "assistant", "content": f"draw((0,0)--({i},{i}));"},
            ]
        })
    return rows


class MockDataset:
    """Minimal mock for HF dataset."""
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]

    def filter(self, fn, **kwargs):
        return MockDataset([r for r in self._rows if fn(r)])


# ── Patch datasets and run resolve ──
with tempfile.TemporaryDirectory() as tmpdir:
    # Write mock triples
    triples_path = os.path.join(tmpdir, "sbert_triples.json")
    with open(triples_path, "w") as f:
        json.dump(MOCK_TRIPLES, f)

    cache_path = os.path.join(tmpdir, "resolved_triples.json")

    # Prepare mock datasets
    mock_svgx = MockDataset(make_mock_svgx(100))
    mock_tikz = MockDataset(make_mock_tikz(100))
    mock_vcm = MockDataset(make_mock_asy(100))

    # Patch imports
    mock_datasets = MagicMock()
    mock_datasets.load_dataset = MagicMock(side_effect=lambda name, split=None: {
        "xingxm/SVGX-Core-250k": mock_svgx,
        "nllg/datikz-v2": mock_tikz,
    }[name])
    mock_datasets.load_from_disk = MagicMock(return_value=mock_vcm)

    # Add stage_b_probe to path
    sys.path.insert(0, str(Path(__file__).parent))

    # Import and patch
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        import importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "stage_b_probe",
            str(Path(__file__).parent / "stage_b_probe.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Override constants that reference server paths
        mod.VCM_PATH = "/mock/vcm"
        mod.PROBE_EMBED_SAMPLE_SVG = 50  # smaller for test

        # Run resolve
        print("\n── Running resolve_probe_triples() ──")
        resolved = mod.resolve_probe_triples(triples_path, cache_path)

    # ── Verify output ──
    print("\n── Verifying output ──")

    assert os.path.exists(cache_path), "resolved_triples.json not created!"
    with open(cache_path) as f:
        loaded = json.load(f)

    assert len(loaded) == len(MOCK_TRIPLES), f"Expected {len(MOCK_TRIPLES)} triples, got {len(loaded)}"

    REQUIRED_KEYS = {"triple_id", "min_cosine", "svg", "tikz", "asy"}
    FORMAT_KEYS = {"caption", "code", "orig_idx"}

    for i, r in enumerate(loaded):
        assert set(r.keys()) == REQUIRED_KEYS, f"Triple {i} keys: {set(r.keys())} != {REQUIRED_KEYS}"
        assert r["triple_id"] == i, f"Triple {i} triple_id should be {i}, got {r['triple_id']}"
        assert isinstance(r["min_cosine"], float), f"Triple {i} min_cosine not float"

        for fmt in ["svg", "tikz", "asy"]:
            entry = r[fmt]
            assert set(entry.keys()) == FORMAT_KEYS, f"Triple {i} {fmt} keys: {set(entry.keys())} != {FORMAT_KEYS}"
            assert entry["caption"], f"Triple {i} {fmt} caption empty"
            assert entry["code"], f"Triple {i} {fmt} code empty"
            assert isinstance(entry["orig_idx"], int), f"Triple {i} {fmt} orig_idx not int"

    print(f"✓ {len(loaded)} resolved triples verified")
    print(f"  triple_ids: {[r['triple_id'] for r in loaded]}")
    print(f"  Sample triple 0:")
    t0 = loaded[0]
    for fmt in ["svg", "tikz", "asy"]:
        print(f"    {fmt}: caption='{t0[fmt]['caption'][:40]}...' code='{t0[fmt]['code'][:40]}...' orig_idx={t0[fmt]['orig_idx']}")

    print("\n✅ DRY-RUN PASS — resolve pipeline verified against actual sbert_triples.json schema")
