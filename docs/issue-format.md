# Issue and Article Format Spec

**When to read this:** You need to know directory layout, file naming conventions, or the exact front matter fields for articles and issue covers. Also read this when `make validate` is failing on a front matter error.

---

## Directory Layout

Each issue is a Hugo section — a directory containing an `_index.md` (the cover page) plus article files:

```
content/issues/
  _index.md                         ← top-level issues list
  01_embeddings.md                  ← legacy single-article issue (leaf)
  02_logits.md                      ← legacy single-article issue (leaf)
  03-sixteen-numbers/               ← new-format multi-article issue — CANONICAL
    _index.md                       ← cover page (tech tree TOC + intro text)
    01-cold-open.md                 ← mainline article
    02-numbers-in-boxes.md          ← primer article
    ...
```

**File naming inside an issue:** `NN-slug.md` where `NN` is a zero-padded order index. The number drives sort order on the cover page and `PrevInSection` / `NextInSection` navigation.

Articles stay **flat** within their issue directory — do NOT nest further.

---

## Front Matter: Article

```yaml
---
title: ""               # the article's pop-art display title
description: ""         # one sentence shown in cards & issue cover list
topics: []              # broad: [embeddings], [quantization], [optimization]
tags: []                # specific: [numpy, gptq, fp4]
theme: cream            # or: teal
math: true              # set false only if no KaTeX in the article
draft: false
date: 2026-MM-DDT09:00:00-04:00
issue: 3                # the issue number this article belongs to
weight: 10              # ordering within the issue (steps of 10 — leave gaps)
techKind: primer        # mainline | primer | boss | external
techNode: numbers       # id matching the tech tree node in data/techtrees/issueNN.toml
header: default.webp    # optional hero image — see docs/components/hero.md
---
```

- `weight` controls order on the issue cover and in prev/next navigation. Use **steps of 10** so you can insert later without renumbering.
- `techNode` must match a node `id` in the corresponding `data/techtrees/issueNN.toml`. Exception: cold-open articles with `techKind: mainline` are not required to be in the tree.
- `make validate` enforces every field; mismatches fail the build.

---

## Front Matter: Issue Cover (`_index.md`)

```yaml
---
title: ""               # the playful issue title
description: ""         # one-sentence hook
issue: 3
layout: issue-cover     # tells Hugo to use the issue-cover layout
theme: cream
math: false             # cover usually has no KaTeX
header: default.webp    # optional hero image — see docs/components/hero.md
---
```

The body of `_index.md` should:
1. Open with a few paragraphs setting up the mystery / cold-open framing.
2. Embed the tech-tree graph: `{{< techtree name="issueNN" >}}`.
3. Optionally end with a short reading-order note. The article list renders automatically below.

---

## The Two Themes

| Theme  | Background   | Primary Accent         |
|--------|--------------|------------------------|
| `cream` | `#FDF5E6`   | `#FF007F` Pop Pink     |
| `teal`  | `#007A7A`   | `#FFD700` Pop Yellow   |

The scaffolding script auto-alternates the theme per-article for visual variety. Theme data files: `data/themes/cream.toml` and `data/themes/teal.toml`.

---

## KaTeX Math

Set `math: true` in front matter. Use `$...$` for inline, `$$...$$` for display.

The Goldmark `passthrough` extension (configured in `hugo.toml`) protects these delimiters from Markdown processing.

---

## Full File Structure Reference

```
content/issues/                   → ALL issues (mainline content)
  _index.md                       → top-level issues list
  NN_legacy.md                    → legacy single-article issues
  NN-slug/                        → new-format multi-article issues
    _index.md                     → issue cover with tech tree TOC
    NN-article.md                 → individual articles
content/docs/                     → meta: contribute guide, stack docs
prompts/                          → authoring prompts (REPO ROOT, NOT content/) — never rendered
archetypes/                       → Hugo `hugo new content` templates
  issue.md                        → issue cover template
  article.md                      → article template
data/themes/                      → CSS variable definitions per theme
data/techtrees/                   → tech tree node/edge data per issue
data/timelines/                   → timeline event data per issue
docs/                             → repo-local authoring docs (NOT rendered)
  components/README.md            → component-library index
  components/<name>.md            → per-component reference (load on demand)
  techtrees-schema.md             → tech-tree TOML schema reference
  timelines-schema.md             → timeline TOML schema reference
themes/almanac/                   → custom Hugo theme
  assets/css/main.css             → all CSS in one file
  layouts/                        → Hugo templates
  layouts/_default/issue-cover.html
  layouts/_default/_markup/render-codeblock-pyplot.html
  layouts/shortcodes/             → all component shortcodes
static/js/fold.js                 → auto-wraps H2 sections in <details>
static/js/toc.js                  → TOC scroll-spy
static/js/pyplot.js               → lightbox for pyplot blocks + figure shortcode
static/header-illustrations/      → article hero images (.webp, ≤1600px wide, ≤300KB)
static/plots/                     → generated PNGs (gitignored)
scripts/run_plots.py              → pre-build pyplot executor
scripts/new_issue.py              → scaffold new issues / articles
scripts/validate.py               → convention linter
```
