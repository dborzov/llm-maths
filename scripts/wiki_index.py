#!/usr/bin/env python3
"""
wiki_index.py — derived SQLite index of the wiki + CLI for AI agents.

The markdown files under content/wiki/ are the source of truth.
This script rebuilds data/wiki.db from scratch on every run by:
  1. Parsing wiki page front matter (slug, title, aliases, category, sot, related)
  2. Scanning all non-wiki articles for {{< wiki "slug" >}} usages
  3. Storing everything in a lightweight SQLite DB

CLI:
  uv run scripts/wiki_index.py rebuild        # (re)build data/wiki.db
  uv run scripts/wiki_index.py search <term>  # fuzzy-search concepts
  uv run scripts/wiki_index.py show <slug>    # full record for a slug
  uv run scripts/wiki_index.py backrefs <slug># articles that use this slug
  uv run scripts/wiki_index.py lint           # report issues (orphan usages, missing SOT)
  uv run scripts/wiki_index.py add-usage <slug> <article_path>  # record a usage manually

Exits 0 on success; exits 1 if lint finds errors (but warnings do not block).
Run via `make wiki` or called from validate.py.
"""

from __future__ import annotations

import re
import sqlite3
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WIKI_CONTENT = REPO_ROOT / "content" / "wiki"
CONTENT_ISSUES = REPO_ROOT / "content" / "issues"
DB_PATH = REPO_ROOT / "scripts" / "wiki.db"

WIKI_USAGE_RE = re.compile(r'\{\{[<\s%]+wiki\s+"([^"]+)"')


# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        raw = tomllib.loads(text[3:end].replace(":", "=", 0))  # YAML not TOML
        # Hugo uses YAML front matter; tomllib won't work directly.
        # Fall back to a minimal YAML-ish parser for the fields we need.
        return _parse_yaml_fm(text[3:end])
    except Exception:
        return _parse_yaml_fm(text[3:end])


def _parse_yaml_fm(block: str) -> dict:
    """Minimal YAML front-matter parser for the fields wiki pages use."""
    result: dict = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\w[\w_-]*):\s*(.*)', line)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("["):
            # Inline YAML list: ["a", "b", "c"]
            items_raw = re.findall(r'"([^"]+)"', val)
            if not items_raw:
                items_raw = re.findall(r"'([^']+)'", val)
            if not items_raw and val.strip("[]").strip():
                items_raw = [v.strip().strip('"\'') for v in val.strip("[]").split(",")]
            result[key] = [x for x in items_raw if x]
        elif val == "" and i + 1 < len(lines) and lines[i + 1].startswith("  -"):
            # Block list
            items = []
            i += 1
            while i < len(lines) and lines[i].startswith("  -"):
                items.append(lines[i][3:].strip().strip('"\''))
                i += 1
            result[key] = items
            continue
        elif val.lower() in ("true", "false"):
            result[key] = val.lower() == "true"
        elif val.startswith('"') and val.endswith('"'):
            result[key] = val[1:-1]
        elif val.startswith("'") and val.endswith("'"):
            result[key] = val[1:-1]
        else:
            result[key] = val
        i += 1
    return result


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS concepts (
            slug        TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            category    TEXT,
            sot_path    TEXT,
            sot_title   TEXT,
            wiki_path   TEXT,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS aliases (
            alias   TEXT NOT NULL,
            slug    TEXT NOT NULL REFERENCES concepts(slug),
            PRIMARY KEY (alias, slug)
        );
        CREATE TABLE IF NOT EXISTS related (
            slug        TEXT NOT NULL REFERENCES concepts(slug),
            related_slug TEXT NOT NULL,
            PRIMARY KEY (slug, related_slug)
        );
        CREATE TABLE IF NOT EXISTS usages (
            slug         TEXT NOT NULL REFERENCES concepts(slug),
            article_path TEXT NOT NULL,
            PRIMARY KEY (slug, article_path)
        );
    """)
    conn.commit()


def rebuild_db(conn: sqlite3.Connection) -> tuple[int, int]:
    """Drop all rows and repopulate from markdown. Returns (pages, usages)."""
    conn.execute("DELETE FROM usages")
    conn.execute("DELETE FROM related")
    conn.execute("DELETE FROM aliases")
    conn.execute("DELETE FROM concepts")
    conn.commit()

    page_count = 0
    for md_file in sorted(WIKI_CONTENT.rglob("*.md")):
        if md_file.name.startswith("_index"):
            continue
        fm = parse_frontmatter(md_file)
        slug = fm.get("slug") or md_file.stem
        if not slug:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO concepts VALUES (?,?,?,?,?,?,?)",
            (
                slug,
                fm.get("title", slug),
                fm.get("category", ""),
                fm.get("source_of_truth", ""),
                fm.get("source_of_truth_title", ""),
                str(md_file.relative_to(REPO_ROOT)),
                fm.get("description", ""),
            ),
        )
        for alias in fm.get("also_known_as", []):
            if alias:
                conn.execute(
                    "INSERT OR IGNORE INTO aliases VALUES (?,?)", (alias, slug)
                )
        for rel in fm.get("related", []):
            if rel:
                conn.execute(
                    "INSERT OR IGNORE INTO related VALUES (?,?)", (slug, rel)
                )
        page_count += 1

    conn.commit()

    # Scan article content for {{< wiki "slug" >}} usages
    usage_count = 0
    all_slugs = {row["slug"] for row in conn.execute("SELECT slug FROM concepts")}
    for md_file in sorted(CONTENT_ISSUES.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        article_rel = str(md_file.relative_to(REPO_ROOT))
        for slug in set(WIKI_USAGE_RE.findall(text)):
            if slug in all_slugs:
                conn.execute(
                    "INSERT OR IGNORE INTO usages VALUES (?,?)", (slug, article_rel)
                )
                usage_count += 1

    conn.commit()
    return page_count, usage_count


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_rebuild(conn: sqlite3.Connection, _args: list[str]) -> int:
    init_schema(conn)
    pages, usages = rebuild_db(conn)
    print(f"wiki: rebuilt {pages} pages, {usages} usages → {DB_PATH}")
    return 0


def cmd_search(conn: sqlite3.Connection, args: list[str]) -> int:
    if not args:
        print("usage: wiki_index.py search <term>", file=sys.stderr)
        return 1
    term = args[0].lower()
    rows = conn.execute(
        """
        SELECT DISTINCT c.slug, c.title, c.category, c.sot_path
        FROM concepts c
        LEFT JOIN aliases a ON a.slug = c.slug
        WHERE lower(c.slug) LIKE ? OR lower(c.title) LIKE ? OR lower(a.alias) LIKE ?
        ORDER BY c.category, c.title
        """,
        (f"%{term}%", f"%{term}%", f"%{term}%"),
    ).fetchall()
    if not rows:
        print(f"no results for '{term}'")
        return 0
    for r in rows:
        sot = f"  → {r['sot_path']}" if r["sot_path"] else "  (no SOT)"
        print(f"[{r['category']}] {r['slug']:30s} {r['title']}{sot}")
    return 0


def cmd_show(conn: sqlite3.Connection, args: list[str]) -> int:
    if not args:
        print("usage: wiki_index.py show <slug>", file=sys.stderr)
        return 1
    slug = args[0]
    row = conn.execute("SELECT * FROM concepts WHERE slug=?", (slug,)).fetchone()
    if not row:
        print(f"no wiki page for slug '{slug}'", file=sys.stderr)
        return 1
    print(f"slug:        {row['slug']}")
    print(f"title:       {row['title']}")
    print(f"category:    {row['category']}")
    print(f"description: {row['description']}")
    print(f"sot_path:    {row['sot_path']}")
    print(f"sot_title:   {row['sot_title']}")
    print(f"wiki_path:   {row['wiki_path']}")
    aliases = conn.execute("SELECT alias FROM aliases WHERE slug=?", (slug,)).fetchall()
    print(f"aliases:     {', '.join(r['alias'] for r in aliases)}")
    related = conn.execute("SELECT related_slug FROM related WHERE slug=?", (slug,)).fetchall()
    print(f"related:     {', '.join(r['related_slug'] for r in related)}")
    usages = conn.execute("SELECT article_path FROM usages WHERE slug=?", (slug,)).fetchall()
    print(f"usages ({len(usages)}):")
    for u in usages:
        print(f"  {u['article_path']}")
    return 0


def cmd_backrefs(conn: sqlite3.Connection, args: list[str]) -> int:
    if not args:
        print("usage: wiki_index.py backrefs <slug>", file=sys.stderr)
        return 1
    slug = args[0]
    usages = conn.execute(
        "SELECT article_path FROM usages WHERE slug=?", (slug,)
    ).fetchall()
    if not usages:
        print(f"no usages found for '{slug}'")
        return 0
    for u in usages:
        print(u["article_path"])
    return 0


def cmd_lint(conn: sqlite3.Connection, _args: list[str]) -> int:
    warnings = []
    errors = []

    # Usages pointing to nonexistent slugs (can't happen after rebuild, but check)
    orphan = conn.execute(
        "SELECT DISTINCT slug FROM usages WHERE slug NOT IN (SELECT slug FROM concepts)"
    ).fetchall()
    for r in orphan:
        errors.append(f"usage of unknown slug '{r['slug']}'")

    # Wiki pages without a source_of_truth
    no_sot = conn.execute(
        "SELECT slug, title FROM concepts WHERE sot_path IS NULL OR sot_path=''"
    ).fetchall()
    for r in no_sot:
        warnings.append(f"wiki/{r['slug']}: no source_of_truth — add one when the article exists")

    # Wiki pages with 0 usages (may just be new/unused)
    unused = conn.execute(
        """SELECT c.slug, c.title FROM concepts c
           LEFT JOIN usages u ON u.slug = c.slug
           WHERE u.slug IS NULL"""
    ).fetchall()
    for r in unused:
        warnings.append(f"wiki/{r['slug']}: never used in any article (add {{{{< wiki \"{r['slug']}\" >}}}} on first mention)")

    for w in warnings:
        print(f"  WIKI WARN  {w}")
    for e in errors:
        print(f"  WIKI ERROR {e}", file=sys.stderr)

    if warnings:
        print(f"\nwiki: {len(warnings)} warning(s), {len(errors)} error(s)")
    return 1 if errors else 0


def cmd_add_usage(conn: sqlite3.Connection, args: list[str]) -> int:
    if len(args) < 2:
        print("usage: wiki_index.py add-usage <slug> <article_path>", file=sys.stderr)
        return 1
    slug, article_path = args[0], args[1]
    row = conn.execute("SELECT slug FROM concepts WHERE slug=?", (slug,)).fetchone()
    if not row:
        print(f"unknown slug '{slug}'", file=sys.stderr)
        return 1
    conn.execute("INSERT OR IGNORE INTO usages VALUES (?,?)", (slug, article_path))
    conn.commit()
    print(f"recorded usage: {slug} ← {article_path}")
    return 0


COMMANDS = {
    "rebuild": cmd_rebuild,
    "search": cmd_search,
    "show": cmd_show,
    "backrefs": cmd_backrefs,
    "lint": cmd_lint,
    "add-usage": cmd_add_usage,
}


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] not in COMMANDS:
        print(f"usage: wiki_index.py <{'|'.join(COMMANDS)}> [args...]")
        print()
        print("  rebuild             (re)build data/wiki.db from markdown")
        print("  search <term>       fuzzy-search concepts by name/alias")
        print("  show <slug>         full record for a slug")
        print("  backrefs <slug>     articles that use this slug")
        print("  lint                report warnings (missing SOT, unused slugs)")
        print("  add-usage <slug> <path>  manually record a usage")
        return 0

    cmd = args[0]
    rest = args[1:]

    # Always rebuild DB before non-rebuild commands (fast, derived)
    conn = open_db()
    init_schema(conn)
    if cmd != "rebuild":
        rebuild_db(conn)

    return COMMANDS[cmd](conn, rest)


if __name__ == "__main__":
    sys.exit(main())
