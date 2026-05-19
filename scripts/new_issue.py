#!/usr/bin/env python3
"""
new_issue.py — scaffold a fresh issue or a new article for LLM Maths Comics.

Usage:
    uv run scripts/new_issue.py issue NN slug "Title"
    uv run scripts/new_issue.py article ISSUE_DIR slug "Title" [--kind primer|mainline|boss]

The issue subcommand creates:
    content/comicbook/NN-slug/_index.md            (cover stub with techtree shortcode)
    data/techtrees/issueNN.toml                  (techtree stub with example node + legend)

The article subcommand creates a single article file with the standard front matter,
auto-incrementing the file's NN prefix to the next available within the issue, and a
weight 10 above the previous article. Pass --kind to set techKind.

Both commands are idempotent: they refuse to overwrite an existing file.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CONTENT_ISSUES = REPO_ROOT / "content" / "comicbook"
DATA_TECHTREES = REPO_ROOT / "data" / "techtrees"

VALID_KINDS = ("mainline", "primer", "boss", "external")

ISSUE_COVER_TEMPLATE = """\
---
title: "{title}"
description: ""
issue: {nn_int}
layout: issue-cover
theme: cream
math: false
header: default.png
date: {date}
---

## The Mystery

[Open with the framing question — the specific moment, the human stakes, what made the field move. See `content/comicbook/03-quantization/_index.md` for the canonical example.]

## How To Read This Issue

Start with the cold-open. The **tech tree** below is the table of contents — each node is an article, arrows show dependencies.

{{{{< techtree name="issue{nn}" >}}}}

## What You Will Walk Away With

By the end of this issue you should be able to:

- ...

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
"""

TECHTREE_TEMPLATE = """\
# Schema and conventions: see docs/techtrees-schema.md.
# Canonical example: data/techtrees/issue03.toml.

title  = "Tech tree — {title}"
width  = 1280
height = 740
node_w = 200
node_h = 70

# ---------------------------------------------------------------
# Layout convention: arrows point UPWARD from prerequisite to user.
# Put boss/mainline nodes at the top (small y), primers at the bottom.
#
# These stub nodes are placeholders. Replace them with the real tree
# before scaffolding articles — `make validate` will complain if any
# node `link` does not match an article in the issue directory.
# ---------------------------------------------------------------

[[nodes]]
id    = "example-primer"
label = "Example\\nPrimer"
kind  = "primer"
x     = 400
y     = 540
# link  = "NN-slug/"      # uncomment and point to a real article slug
sub   = "ch.?"

[[nodes]]
id    = "example-mainline"
label = "Example\\nMainline"
kind  = "mainline"
x     = 400
y     = 220
# link  = "NN-slug/"      # uncomment and point to a real article slug
sub   = "ch.?"

[[edges]]
from = "example-primer"
to   = "example-mainline"

[[legend]]
kind  = "boss"
label = "Boss capstone"

[[legend]]
kind  = "mainline"
label = "Mainline narrative"

[[legend]]
kind  = "primer"
label = "Tutorial primer"

caption = 'Arrows point <em>upward</em> from prerequisite to user.'
"""

ARTICLE_TEMPLATE = """\
---
title: "{title}"
description: ""
topics: []
tags: []
theme: {theme}
math: true
draft: false
date: {date}
issue: {issue_int}
weight: {weight}
techKind: {kind}
techNode: {tech_node}
header: default.png
---

## [Open With A Human Moment]

[A year, a person, a specific failed approach — never "in this article we will…". See `content/comicbook/03-quantization/01-cold-open.md` for the canonical opening.]

## [The Setup]

[Concrete, anchored example. Mathematics, then what the math *feels like*.]

## [The Aha Moment]

[The naive approach falling over → the surprising trick that fixes it. Use a `pyplot` block here if a picture helps.]

## What To Remember

1. ...
2. ...

<!-- TODO: cliffhanger forward link to the next article in the issue.
     Write it as: Continue to + bold + arrow + linked next-article title
     + em-dash + one-sentence hook. See any article in issue 03 for the
     exact format. Do not leave this comment in place once you've written
     the real cliffhanger — make validate will be happier. -->

"""


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def today() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S-04:00")


def cmd_issue(args: argparse.Namespace) -> None:
    nn = args.nn.zfill(2)
    try:
        nn_int = int(nn)
    except ValueError:
        fail(f"NN must be an integer (got {args.nn!r})")
    slug = args.slug.strip("-/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        fail(f"slug must be lowercase alphanumeric with dashes (got {slug!r})")

    issue_dir = CONTENT_ISSUES / f"{nn}-{slug}"
    cover_path = issue_dir / "_index.md"
    tree_path = DATA_TECHTREES / f"issue{nn}.toml"

    if issue_dir.exists():
        fail(f"issue directory already exists: {issue_dir.relative_to(REPO_ROOT)}")
    if tree_path.exists():
        fail(f"techtree data file already exists: {tree_path.relative_to(REPO_ROOT)}")

    issue_dir.mkdir(parents=True)
    cover_path.write_text(
        ISSUE_COVER_TEMPLATE.format(
            title=args.title, nn=nn, nn_int=nn_int, date=today()
        )
    )
    DATA_TECHTREES.mkdir(parents=True, exist_ok=True)
    tree_path.write_text(TECHTREE_TEMPLATE.format(title=args.title))

    print(f"created  {cover_path.relative_to(REPO_ROOT)}")
    print(f"created  {tree_path.relative_to(REPO_ROOT)}")
    print()
    print("next steps:")
    print(f"  1. edit {cover_path.relative_to(REPO_ROOT)} — set description, fill in the mystery prose")
    print(f"  2. edit {tree_path.relative_to(REPO_ROOT)} — define the actual tech-tree nodes and edges")
    print(f"  3. uv run scripts/new_issue.py article {nn}-{slug} cold-open 'Title' --kind mainline")


def find_next_article_index(issue_dir: Path) -> int:
    """Return the next zero-padded numeric prefix for a new article in this issue."""
    used = []
    for p in issue_dir.glob("*.md"):
        if p.name.startswith("_"):
            continue
        m = re.match(r"^(\d+)-", p.name)
        if m:
            used.append(int(m.group(1)))
    return max(used) + 1 if used else 1


def find_max_weight(issue_dir: Path) -> int:
    """Scan article weights so the new one slots in at the end."""
    max_w = 0
    for p in issue_dir.glob("*.md"):
        if p.name.startswith("_"):
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        m = re.search(r"^weight:\s*(\d+)", text, re.MULTILINE)
        if m:
            max_w = max(max_w, int(m.group(1)))
    return max_w


def cmd_article(args: argparse.Namespace) -> None:
    issue_dir = CONTENT_ISSUES / args.issue_dir
    if not issue_dir.is_dir():
        fail(f"issue directory not found: {issue_dir.relative_to(REPO_ROOT)}")
    cover_path = issue_dir / "_index.md"
    if not cover_path.exists():
        fail(f"issue cover not found: {cover_path.relative_to(REPO_ROOT)}")

    slug = args.slug.strip("-/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        fail(f"slug must be lowercase alphanumeric with dashes (got {slug!r})")
    if args.kind not in VALID_KINDS:
        fail(f"--kind must be one of {VALID_KINDS} (got {args.kind!r})")

    # Pull the issue number from the cover's front matter.
    cover_text = cover_path.read_text()
    m = re.search(r"^issue:\s*(\d+)", cover_text, re.MULTILINE)
    if not m:
        fail(f"cover {cover_path.relative_to(REPO_ROOT)} is missing 'issue: N' front matter")
    issue_int = int(m.group(1))

    nn = f"{find_next_article_index(issue_dir):02d}"
    weight = find_max_weight(issue_dir) + 10
    article_path = issue_dir / f"{nn}-{slug}.md"
    if article_path.exists():
        fail(f"article already exists: {article_path.relative_to(REPO_ROOT)}")

    # Alternate theme cream/teal based on article index parity for visual variety.
    theme = "cream" if int(nn) % 2 == 1 else "teal"

    # techNode defaults to the slug (without index prefix).
    tech_node = args.tech_node or slug

    article_path.write_text(
        ARTICLE_TEMPLATE.format(
            title=args.title,
            theme=theme,
            date=today(),
            issue_int=issue_int,
            weight=weight,
            kind=args.kind,
            tech_node=tech_node,
        )
    )

    print(f"created  {article_path.relative_to(REPO_ROOT)}")
    print()
    print("front matter set:")
    print(f"  weight    = {weight}     (slots in at the end of issue {issue_int})")
    print(f"  techKind  = {args.kind}")
    print(f"  techNode  = {tech_node}  (make sure this matches a node id in data/techtrees/issue{issue_int:02d}.toml)")
    print(f"  theme     = {theme}      (auto-alternated)")


def main() -> int:
    parser = argparse.ArgumentParser(prog="new_issue.py", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_issue = sub.add_parser("issue", help="scaffold a new issue (directory + cover + techtree stub)")
    p_issue.add_argument("nn", help="two-digit issue number, e.g. 04")
    p_issue.add_argument("slug", help="lowercase slug, e.g. attention-anatomy")
    p_issue.add_argument("title", help='display title, e.g. "Attention, Anatomized"')
    p_issue.set_defaults(func=cmd_issue)

    p_article = sub.add_parser("article", help="scaffold a new article inside an existing issue")
    p_article.add_argument("issue_dir", help="issue directory name, e.g. 03-quantization")
    p_article.add_argument("slug", help="article slug, e.g. cold-open")
    p_article.add_argument("title", help='display title, e.g. "The Cold Open"')
    p_article.add_argument("--kind", default="primer", choices=VALID_KINDS,
                           help="techKind value (default: primer)")
    p_article.add_argument("--tech-node", default=None,
                           help="techNode id (default: same as slug)")
    p_article.set_defaults(func=cmd_article)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
