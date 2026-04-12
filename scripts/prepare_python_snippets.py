#!/usr/bin/env python
"""Prepare Python code snippets for negative control experiment (C1).

Extracts 252 non-visual Python code snippets from the Python standard library.
These are ordinary algorithmic/data-processing functions — no graphics code.

Output: python_snippets.json in CACHE_DIR, format:
  [{"triple_id": 0, "code": "def foo(): ...", "source": "os.path"}, ...]

Usage:
  python scripts/prepare_python_snippets.py
"""

import ast
import json
import os
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

N_SNIPPETS = 252
SEED = 20260412
MIN_LINES = 5
MAX_LINES = 150
MIN_CHARS = 100
MAX_CHARS = 4000  # rough match to visual code length distribution

CACHE_DIR = Path("/root/autodl-tmp/cache/hidden_states")

# Modules to EXCLUDE (visual/graphics/GUI related)
EXCLUDE_MODULES = {
    "tkinter", "turtle", "turtledemo", "idlelib", "curses",
    "PIL", "matplotlib", "pyglet", "pygame", "cairo",
    # Also exclude test modules (too meta)
    "test", "unittest", "__pycache__", "site-packages",
}


def is_excluded(path_str: str) -> bool:
    parts = Path(path_str).parts
    return any(exc in parts for exc in EXCLUDE_MODULES)


def extract_functions_from_file(filepath: str):
    """Extract top-level function and class method source code from a .py file."""
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    snippets = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
            func_lines = lines[start:end]
            code = "\n".join(func_lines)

            n_lines = len(func_lines)
            if n_lines < MIN_LINES or n_lines > MAX_LINES:
                continue
            if len(code) < MIN_CHARS or len(code) > MAX_CHARS:
                continue

            # Skip if it looks like a test
            if node.name.startswith("test_") or node.name.startswith("_test"):
                continue

            snippets.append({
                "code": code,
                "source": f"{Path(filepath).stem}.{node.name}",
                "n_lines": n_lines,
            })

    return snippets


def find_stdlib_path():
    """Find Python standard library path."""
    import sysconfig
    stdlib = sysconfig.get_path("stdlib")
    if stdlib and os.path.isdir(stdlib):
        return stdlib
    # Fallback: try common conda paths
    for p in ["/root/miniconda3/lib/python3.12", "/root/miniconda3/lib/python3.11",
              "/root/miniconda3/lib/python3.10", "/usr/lib/python3.12",
              "/usr/lib/python3.11", "/usr/lib/python3.10"]:
        if os.path.isdir(p):
            return p
    raise RuntimeError("Cannot find Python stdlib path")


def main():
    stdlib_path = find_stdlib_path()
    print(f"[prep] Scanning stdlib: {stdlib_path}")

    # Collect all .py files
    py_files = []
    for root, dirs, files in os.walk(stdlib_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_MODULES and not d.startswith("__")]
        if is_excluded(root):
            continue
        for f in files:
            if f.endswith(".py") and not f.startswith("test_"):
                py_files.append(os.path.join(root, f))

    print(f"[prep] Found {len(py_files)} .py files")

    # Extract all function snippets
    all_snippets = []
    for pf in py_files:
        snippets = extract_functions_from_file(pf)
        all_snippets.extend(snippets)

    print(f"[prep] Extracted {len(all_snippets)} function snippets")

    if len(all_snippets) < N_SNIPPETS:
        raise RuntimeError(
            f"Only {len(all_snippets)} snippets found, need {N_SNIPPETS}. "
            f"Loosen MIN_LINES/MAX_LINES filters."
        )

    # Random sample
    rng = random.Random(SEED)
    selected = rng.sample(all_snippets, N_SNIPPETS)

    # Format output
    output = []
    for i, s in enumerate(selected):
        output.append({
            "triple_id": i,
            "code": s["code"],
            "source": s["source"],
            "n_lines": s["n_lines"],
        })

    # Save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "python_snippets.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Stats
    lengths = [len(s["code"]) for s in output]
    print(f"[prep] Saved {len(output)} snippets → {out_path}")
    print(f"[prep] Code length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.0f}")
    print(f"[prep] Line count: min={min(s['n_lines'] for s in output)}, "
          f"max={max(s['n_lines'] for s in output)}")

    # Verify no visual code leaked through
    visual_keywords = {"svg", "tikz", "canvas", "draw", "paint", "render", "pixel",
                       "matplotlib", "pyplot", "figure", "plot(", "imshow"}
    flagged = 0
    for s in output:
        code_lower = s["code"].lower()
        for kw in visual_keywords:
            if kw in code_lower:
                flagged += 1
                print(f"  [WARN] {s['source']} contains '{kw}'")
                break

    if flagged > 0:
        print(f"[prep] WARNING: {flagged} snippets contain visual keywords — review manually")
    else:
        print("[prep] No visual keywords detected ✓")


if __name__ == "__main__":
    main()
