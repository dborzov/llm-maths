#!/usr/bin/env python3
"""
run_plots.py — Pre-build script for Ye Olde LLM Maths Almanac.

Finds all ```pyplot {id="..."} ``` fenced code blocks in content Markdown files,
executes them in an isolated uv environment (numpy + matplotlib), saves PNG output,
and caches results to skip unchanged blocks on subsequent runs.

Usage:
    uv run scripts/run_plots.py [--force] [--with-torch]
    uv run scripts/run_plots.py --help

Author code contract:
    - `plt` (matplotlib.pyplot) and `np` (numpy) are pre-imported.
    - Do NOT call plt.show() or plt.savefig() — the wrapper handles that.
    - The last active matplotlib figure is what gets saved.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root — script is run from repo root via uv run)
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).parent.parent
CONTENT_DIR  = REPO_ROOT / "content"
STATIC_DIR   = REPO_ROOT / "static"
PLOTS_DIR    = STATIC_DIR / "plots"
CACHE_FILE   = Path(__file__).parent / ".plot_cache.json"

# ---------------------------------------------------------------------------
# Regex for ```pyplot {id="..." caption="..."} ... ``` blocks
# ---------------------------------------------------------------------------
PYPLOT_PATTERN = re.compile(
    r'^```pyplot\s*\{([^}]*)\}\s*\n([\s\S]*?)^```',
    re.MULTILINE,
)

ATTR_PATTERN = re.compile(r'(\w+)=["\']([^"\']*)["\']')


def parse_attrs(attr_string: str) -> dict:
    """Parse key="value" pairs from a code block attribute string."""
    return dict(ATTR_PATTERN.findall(attr_string))


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, sort_keys=True, indent=2))


def make_wrapper_script(code: str, output_path: Path) -> str:
    """Wrap user code with the matplotlib setup boilerplate."""
    # Escape the output path for embedding in Python string
    out_str = str(output_path).replace("\\", "\\\\")
    return f"""\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__ALMANAC_OUTPUT__ = {out_str!r}

# ---- user code ----
{code}
# ---- end user code ----

plt.savefig(__ALMANAC_OUTPUT__, dpi=150, bbox_inches="tight")
plt.close("all")
"""


def run_plot(code: str, output_path: Path, with_torch: bool = False) -> bool:
    """Execute plot code in an isolated uv environment. Returns True on success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="almanac_plot_", delete=False
    ) as f:
        f.write(make_wrapper_script(code, output_path))
        tmp_path = f.name

    try:
        cmd = [
            "uv", "run",
            "--isolated",
            "--with", "numpy",
            "--with", "matplotlib",
        ]
        if with_torch:
            cmd += ["--with", "torch"]
        cmd += ["python", tmp_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        if result.returncode != 0:
            print(f"    ERROR: {result.stderr.strip()}", file=sys.stderr)
            return False

        if not output_path.exists():
            print(f"    ERROR: plot file was not created at {output_path}", file=sys.stderr)
            return False

        return True
    finally:
        os.unlink(tmp_path)


def collect_pyplot_blocks(md_file: Path) -> list[dict]:
    """Find all pyplot blocks in a markdown file."""
    text = md_file.read_text(encoding="utf-8")
    blocks = []
    for ordinal, match in enumerate(PYPLOT_PATTERN.finditer(text)):
        attrs = parse_attrs(match.group(1))
        block_id = attrs.get("id", str(ordinal))
        code = match.group(2).rstrip()
        blocks.append({
            "id":      block_id,
            "caption": attrs.get("caption", ""),
            "code":    code,
            "attrs":   attrs,
        })
    return blocks


def file_slug(md_file: Path) -> str:
    """Derive the plot subdirectory path from the content file path.
    e.g. content/articles/backprop.md -> articles/backprop
    """
    rel = md_file.relative_to(CONTENT_DIR)
    return str(rel.with_suffix(""))


def main():
    parser = argparse.ArgumentParser(description="Generate matplotlib plots for almanac articles.")
    parser.add_argument("--force", action="store_true", help="Ignore cache, re-run all plots")
    parser.add_argument("--with-torch", action="store_true", help="Include torch in plot environment")
    args = parser.parse_args()

    cache = {} if args.force else load_cache()
    new_cache = {}

    md_files = list(CONTENT_DIR.rglob("*.md"))
    if not md_files:
        print("No markdown files found.")
        return

    # Track all valid (slug, id) pairs for orphan cleanup
    valid_keys: set[str] = set()

    n_generated = n_cached = n_errors = 0

    for md_file in sorted(md_files):
        blocks = collect_pyplot_blocks(md_file)
        if not blocks:
            continue

        slug = file_slug(md_file)
        print(f"  {md_file.relative_to(REPO_ROOT)}: {len(blocks)} plot(s)")

        for block in blocks:
            block_key = f"{slug}/{block['id']}"
            valid_keys.add(block_key)

            h = code_hash(block["code"])
            output_path = PLOTS_DIR / slug / f"{block['id']}.png"

            # Check cache
            if not args.force and cache.get(block_key) == h and output_path.exists():
                print(f"    [cached] {block['id']}")
                new_cache[block_key] = h
                n_cached += 1
                continue

            print(f"    [run]    {block['id']} ...", end="", flush=True)
            success = run_plot(block["code"], output_path, with_torch=args.with_torch)
            if success:
                print(" ok")
                new_cache[block_key] = h
                n_generated += 1
            else:
                print(" FAILED", file=sys.stderr)
                n_errors += 1

    # Orphan cleanup: remove PNGs whose source block no longer exists
    if PLOTS_DIR.exists():
        for png in PLOTS_DIR.rglob("*.png"):
            rel = png.relative_to(PLOTS_DIR)
            # Convert path like 'articles/backprop/fig1.png' -> 'articles/backprop/fig1'
            key = str(rel.with_suffix(""))
            if key not in valid_keys:
                print(f"  [orphan] removing {png.relative_to(REPO_ROOT)}")
                png.unlink()

    save_cache(new_cache)

    print(f"\nPlots: {n_generated} generated, {n_cached} cached, {n_errors} errors")

    if n_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
