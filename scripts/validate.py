#!/usr/bin/env python3
"""
validate.py — convention linter for LLM Maths Comics.

Runs four checks across the repo's content + data:

  1. Front-matter contract for articles inside an issue section:
     required keys present, valid types, valid enums.
  2. Tech-tree consistency: every issue cover's referenced techtree data
     file exists; every non-external techNode in an article matches a
     node id in that data file; every techtree node that has a `link`
     points to an actual article in the issue.
  3. Cross-link integrity: every Markdown link of the form `../slug/`
     resolves to a real article in the same issue.
  4. Pyplot block rules: unique ids per article, no forbidden imports,
     no plt.show/plt.savefig calls.

Outputs a colored report. Exits non-zero if any errors are found. Run via
`make validate` or directly as `uv run scripts/validate.py`.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CONTENT_ISSUES = REPO_ROOT / "content" / "comicbook"
DATA_TECHTREES = REPO_ROOT / "data" / "techtrees"
DATA_TIMELINES = REPO_ROOT / "data" / "timelines"
STATIC_HEADER_IMGS = REPO_ROOT / "static" / "header-illustrations"

REQUIRED_ARTICLE_FIELDS = {
    "title", "description", "topics", "tags", "theme",
    "math", "draft", "date", "issue", "weight", "techKind", "techNode",
}
VALID_THEMES = {"cream", "teal"}
VALID_TECHKINDS = {"mainline", "primer", "boss", "external"}

# Modules allowed inside pyplot blocks — must match the isolated env in run_plots.py
PYPLOT_ALLOWED_TOP_LEVEL_IMPORTS = {
    # The wrapper imports these; user code shouldn't re-import them but it's
    # only a soft warning if they do.
    "numpy", "np", "matplotlib", "plt",
    # Standard library is always fine; we don't enumerate it. We only flag
    # third-party packages that the isolated env does NOT install.
}
PYPLOT_FORBIDDEN_IMPORTS = {
    # Pyplot blocks run in an isolated uv venv with only numpy + matplotlib.
    # Anything else needs an explicit run_plots.py update.
    "scipy", "torch", "pandas", "sklearn", "sympy", "polars",
}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def print_summary(self) -> int:
        BOLD = "\033[1m"
        RED = "\033[31m"
        YEL = "\033[33m"
        GRN = "\033[32m"
        DIM = "\033[2m"
        RST = "\033[0m"

        if self.errors:
            print(f"{BOLD}{RED}errors ({len(self.errors)}):{RST}")
            for e in self.errors:
                print(f"  {RED}✗{RST} {e}")
        if self.warnings:
            print(f"{BOLD}{YEL}warnings ({len(self.warnings)}):{RST}")
            for w in self.warnings:
                print(f"  {YEL}!{RST} {w}")
        if not self.errors and not self.warnings:
            print(f"{BOLD}{GRN}✓ validate: all checks passed{RST}")
        elif not self.errors:
            print(f"{BOLD}{GRN}✓ validate: no errors{DIM} ({len(self.warnings)} warnings){RST}")
        return 1 if self.errors else 0


# --------------------------------------------------------------------- #
# YAML front-matter mini-parser (avoids a PyYAML dependency for one job)
# --------------------------------------------------------------------- #

def parse_front_matter(text: str, path: Path, report: Report) -> dict | None:
    """Extract YAML-ish front matter from a Markdown file. Best-effort but
    strict on the bits we care about (key: value, lists)."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        report.err(f"{rel(path)}: malformed front matter (no closing ---)")
        return None
    body = text[4:end]
    out: dict = {}
    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        # Strip surrounding quotes
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        elif v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
        # Lists like [a, b, c]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            out[k] = [x.strip().strip("'\"") for x in inner.split(",")] if inner else []
            continue
        if v.lower() in {"true", "false"}:
            out[k] = (v.lower() == "true")
            continue
        if re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
            continue
        out[k] = v
    return out


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------- #
# Issue discovery
# --------------------------------------------------------------------- #

@dataclass
class Article:
    path: Path
    fm: dict
    slug: str  # the directory-relative filename without extension


@dataclass
class IssueSection:
    """A new-format issue (a directory with _index.md + articles)."""
    dir: Path
    cover: Path
    cover_fm: dict
    articles: list[Article]
    techtree_name: str | None
    techtree_path: Path | None
    techtree_data: dict | None


def discover_issues(report: Report) -> list[IssueSection]:
    issues: list[IssueSection] = []
    for child in sorted(CONTENT_ISSUES.iterdir()):
        if not child.is_dir():
            continue
        cover = child / "_index.md"
        if not cover.exists():
            report.err(f"{rel(child)}: issue directory missing _index.md")
            continue
        cover_text = cover.read_text()
        cover_fm = parse_front_matter(cover_text, cover, report) or {}

        # Find the {{< techtree name="..." >}} in cover body
        tt_match = re.search(r'\{\{<\s*techtree\s+name="([^"]+)"', cover_text)
        techtree_name = tt_match.group(1) if tt_match else None
        techtree_path = DATA_TECHTREES / f"{techtree_name}.toml" if techtree_name else None
        techtree_data: dict | None = None
        if techtree_path and techtree_path.exists():
            try:
                techtree_data = tomllib.loads(techtree_path.read_text())
            except tomllib.TOMLDecodeError as e:
                report.err(f"{rel(techtree_path)}: TOML parse error — {e}")
        elif techtree_path:
            report.err(f"{rel(cover)}: references techtree '{techtree_name}' but {rel(techtree_path)} does not exist")

        articles: list[Article] = []
        for art_path in sorted(child.glob("*.md")):
            if art_path.name.startswith("_"):
                continue
            fm = parse_front_matter(art_path.read_text(), art_path, report) or {}
            articles.append(Article(path=art_path, fm=fm, slug=art_path.stem))

        issues.append(IssueSection(
            dir=child, cover=cover, cover_fm=cover_fm, articles=articles,
            techtree_name=techtree_name, techtree_path=techtree_path,
            techtree_data=techtree_data,
        ))
    return issues


# --------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------- #

def check_front_matter(issue: IssueSection, report: Report) -> None:
    for art in issue.articles:
        missing = REQUIRED_ARTICLE_FIELDS - set(art.fm.keys())
        if missing:
            report.err(f"{rel(art.path)}: missing front-matter fields: {sorted(missing)}")
        if "theme" in art.fm and art.fm["theme"] not in VALID_THEMES:
            report.err(f"{rel(art.path)}: theme={art.fm['theme']!r} not in {VALID_THEMES}")
        if "techKind" in art.fm and art.fm["techKind"] not in VALID_TECHKINDS:
            report.err(f"{rel(art.path)}: techKind={art.fm['techKind']!r} not in {VALID_TECHKINDS}")
        for bool_key in ("math", "draft"):
            if bool_key in art.fm and not isinstance(art.fm[bool_key], bool):
                report.err(f"{rel(art.path)}: {bool_key}={art.fm[bool_key]!r} must be true/false")
        for int_key in ("issue", "weight"):
            if int_key in art.fm and not isinstance(art.fm[int_key], int):
                report.err(f"{rel(art.path)}: {int_key}={art.fm[int_key]!r} must be an integer")

    # Weights should be unique within an issue
    seen_weights: dict[int, str] = {}
    for art in issue.articles:
        w = art.fm.get("weight")
        if isinstance(w, int):
            if w in seen_weights:
                report.err(
                    f"{rel(issue.dir)}: duplicate weight {w} on "
                    f"{art.slug} and {seen_weights[w]}"
                )
            seen_weights[w] = art.slug

    # All articles should share the same issue number
    issue_nums = {a.fm.get("issue") for a in issue.articles if "issue" in a.fm}
    if len(issue_nums) > 1:
        report.err(f"{rel(issue.dir)}: articles disagree on issue number: {sorted(issue_nums)}")


def check_techtree(issue: IssueSection, report: Report) -> None:
    if issue.techtree_data is None:
        # Already reported in discover_issues if it was supposed to exist
        return

    nodes = issue.techtree_data.get("nodes", [])
    edges = issue.techtree_data.get("edges", [])
    node_by_id = {n["id"]: n for n in nodes}

    # Every techNode in articles should match a tree node (unless techKind=external)
    article_tech_nodes: dict[str, Article] = {}
    for art in issue.articles:
        kind = art.fm.get("techKind")
        tn = art.fm.get("techNode")
        if not tn:
            continue
        if kind == "external":
            continue
        article_tech_nodes[tn] = art
        if tn not in node_by_id:
            # Allow articles deliberately not in the tree (like the cold-open)
            # to be flagged as warnings rather than errors when techKind=mainline.
            # But primer/boss articles *must* be in the tree.
            if kind in {"primer", "boss"}:
                report.err(
                    f"{rel(art.path)}: techNode={tn!r} not found in "
                    f"{rel(issue.techtree_path)} (techKind={kind} requires a tree node)"
                )
            else:
                report.warn(
                    f"{rel(art.path)}: techNode={tn!r} not in tree "
                    f"(ok for mainline cold-opens, otherwise add a node)"
                )

    # Every tree node with a link should resolve to an article in the issue
    for node in nodes:
        link = node.get("link", "").strip()
        if not link or node.get("kind") == "external":
            continue
        # Normalize: strip leading/trailing slashes; expected form "slug/" or "slug"
        target_slug = link.rstrip("/").lstrip("/")
        target_path = issue.dir / f"{target_slug}.md"
        if not target_path.exists():
            report.err(
                f"{rel(issue.techtree_path)}: node {node['id']!r} link={link!r} "
                f"does not match any article in {rel(issue.dir)}"
            )

    # Edges must reference valid nodes
    for edge in edges:
        for end in ("from", "to"):
            v = edge.get(end)
            if v and v not in node_by_id:
                report.err(
                    f"{rel(issue.techtree_path)}: edge {end}={v!r} references unknown node id"
                )


LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def check_cross_links(issue: IssueSection, report: Report) -> None:
    """Every (../slug/) link in an article should resolve to a real article in the same issue."""
    article_slugs = {a.slug for a in issue.articles}
    for art in issue.articles:
        text = art.path.read_text()
        for match in LINK_RE.finditer(text):
            target = match.group(2).strip()
            # only care about ../slug or ../slug/ — siblings in the same issue
            m = re.fullmatch(r"\.\./([a-z0-9][a-z0-9-]*)/?(?:#[^)]*)?", target)
            if not m:
                continue
            slug = m.group(1)
            if slug not in article_slugs:
                report.err(
                    f"{rel(art.path)}: broken cross-link {target!r} — "
                    f"no sibling article {slug}.md in {rel(issue.dir)}"
                )


PYPLOT_BLOCK_RE = re.compile(
    r"^```pyplot\s*\{([^}]*)\}\s*\n([\s\S]*?)^```",
    re.MULTILINE,
)
PYPLOT_ID_RE = re.compile(r'id=["\']([^"\']+)["\']')
IMPORT_RE = re.compile(r"^\s*(?:from\s+(\w+)|import\s+(\w+))", re.MULTILINE)


def check_pyplot_blocks(issue: IssueSection, report: Report) -> None:
    for art in issue.articles:
        text = art.path.read_text()
        seen_ids: set[str] = set()
        for attr_str, code in PYPLOT_BLOCK_RE.findall(text):
            m = PYPLOT_ID_RE.search(attr_str)
            if not m:
                report.err(f"{rel(art.path)}: pyplot block missing id attribute")
                continue
            block_id = m.group(1)
            if block_id in seen_ids:
                report.err(f"{rel(art.path)}: duplicate pyplot id {block_id!r}")
            seen_ids.add(block_id)
            # No plt.show/savefig
            if re.search(r"\bplt\.(show|savefig)\b", code):
                report.err(
                    f"{rel(art.path)}: pyplot block {block_id!r} calls plt.show/savefig "
                    f"(the wrapper handles it — see run_plots.py)"
                )
            # Forbidden imports
            for top, plain in IMPORT_RE.findall(code):
                mod = (top or plain).split(".")[0]
                if mod in PYPLOT_FORBIDDEN_IMPORTS:
                    report.err(
                        f"{rel(art.path)}: pyplot block {block_id!r} imports "
                        f"{mod!r} — not in the isolated env "
                        f"(only numpy+matplotlib are guaranteed). "
                        f"Rewrite to use numpy, or update run_plots.py deps."
                    )


def check_header_images(issues: list[IssueSection], report: Report) -> None:
    """Every header: field in _index.md and article front matter must resolve to a real file."""
    for issue in issues:
        # Issue cover
        img = issue.cover_fm.get("header")
        if img and not (STATIC_HEADER_IMGS / img).exists():
            report.err(
                f"{rel(issue.cover)}: header={img!r} not found in "
                f"static/header-illustrations/"
            )
        # Individual articles
        for art in issue.articles:
            img = art.fm.get("header")
            if img and not (STATIC_HEADER_IMGS / img).exists():
                report.err(
                    f"{rel(art.path)}: header={img!r} not found in "
                    f"static/header-illustrations/"
                )


def check_timeline_data(report: Report) -> None:
    """Verify timeline TOML files parse and never exceed 5 events."""
    if not DATA_TIMELINES.exists():
        return
    for tl_path in sorted(DATA_TIMELINES.glob("*.toml")):
        try:
            data = tomllib.loads(tl_path.read_text())
        except tomllib.TOMLDecodeError as e:
            report.err(f"{rel(tl_path)}: TOML parse error — {e}")
            continue
        events = data.get("events", [])
        if len(events) > 5:
            report.warn(
                f"{rel(tl_path)}: has {len(events)} events; "
                f"only the first 5 will render (5-event cap is enforced — see CLAUDE.md)"
            )


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #

def check_wiki(report: Report) -> None:
    """Run wiki_index lint and surface warnings. Errors from wiki are promoted to errors."""
    import subprocess
    result = subprocess.run(
        ["uv", "run", "scripts/wiki_index.py", "lint"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("WIKI ERROR"):
            report.err(line)
        elif line.startswith("WIKI WARN"):
            report.warn(line)
    if result.returncode != 0 and result.stderr:
        report.warn(f"wiki_index lint failed: {result.stderr.strip()[:200]}")


def main() -> int:
    report = Report()
    issues = discover_issues(report)
    for issue in issues:
        if not issue.articles:
            continue  # legacy single-file issues won't have an articles list
        check_front_matter(issue, report)
        check_techtree(issue, report)
        check_cross_links(issue, report)
        check_pyplot_blocks(issue, report)
    check_header_images(issues, report)
    check_timeline_data(report)
    check_wiki(report)
    return report.print_summary()


if __name__ == "__main__":
    sys.exit(main())
