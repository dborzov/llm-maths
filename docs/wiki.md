# Wiki System

**When to read this:** You are writing or editing any article and need to link a recurring concept to its canonical definition, OR you are adding/updating wiki pages.

---

## What the Wiki Is For

Every non-trivial term that recurs across multiple issues has a **stub wiki page** at `/wiki/<category>/<slug>/`. Each stub has one job: point readers (and AI agents) to the article where the concept is properly defined.

**What a wiki page contains:**
- `source_of_truth` — the canonical article where the concept is introduced
- `also_known_as` — all aliases, alternate names, and symbols for the concept
- `related` — slugs of related wiki pages
- A minimal body: a short table of symbols/definitions if useful

**What a wiki page does NOT contain:**
- A re-explanation of the concept — that lives in the source-of-truth article
- Narrative prose — keep stubs short

---

## Using the Wiki in Articles

On the first mention of a recurring concept in an article, link it to its wiki page using the `wiki` shortcode:

```markdown
The {{< wiki "attention" >}}multi-head attention{{< /wiki >}} block computes…
The {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} grows linearly with sequence length.
The {{< wiki "number-formats" >}}BF16{{< /wiki >}} format trades mantissa precision for range.
```

The slug must match the `slug` field in a wiki page's front matter. Missing slugs render plain text and emit a build warning — `make validate` will surface them.

Re-mentions in the same article do not need to be linked, but it is harmless if they are.

---

## Adding a New Wiki Page

1. Create `content/wiki/<category>/<slug>.md` with this front matter:

```yaml
---
title: "Human-readable name"
slug: "machine-slug"         # must be unique across all wiki pages
description: "One sentence."
category: "transformer"      # matches the directory name
also_known_as:               # all names, aliases, symbols for this concept
  - "alternate name"
  - "W_K"
source_of_truth: "/comicbook/05-microgpt/06-qkv-projections/"
source_of_truth_title: "ch.6 Q, K, V (microGPT)"
related:
  - "attention"              # slugs of related wiki pages
draft: false
---
```

2. Run `uv run scripts/wiki_index.py rebuild` to update `data/wiki.db`.

3. Run `make validate` — it will warn if the slug is unused in any article.

**Grouping rule:** Err on the side of fewer pages. Group related individual terms into one page (e.g., all weight matrices in one page, not one page per matrix). Split later if a page becomes unwieldy.

---

## Category Structure

| Category (directory) | What goes here |
|---|---|
| `transformer/` | Attention, weights, KV cache, hyperparameters, residual stream, etc. |
| `quantization/` | Number formats, quantization schemes, calibration |
| `long-context-eval/` | Benchmarks, evaluation vocabulary |
| `models/` | Open-source LLM families |
| `people/` | Researchers and paper authors |

New categories are fine — create the directory and an `_index.md`.

---

## AI Agent CLI

The Python indexer provides a CLI for agents to interact with the wiki without reading every markdown file:

```bash
# Search for a concept by name or alias
uv run scripts/wiki_index.py search "key projection"
uv run scripts/wiki_index.py search "BF16"

# Show full record for a slug (aliases, SOT, usages)
uv run scripts/wiki_index.py show attention

# Show all articles that use a slug
uv run scripts/wiki_index.py backrefs kv-cache

# Check for problems (called automatically by make validate)
uv run scripts/wiki_index.py lint

# Rebuild the index after editing wiki pages
uv run scripts/wiki_index.py rebuild
```

The SQLite DB at `data/wiki.db` is **derived** — do not edit it directly. Edit the markdown files and rebuild.

---

## Validate Integration

`make validate` runs `wiki_index.py lint` and surfaces:
- **WIKI WARN** — wiki page has no `source_of_truth`, or a slug that is never linked from any article.
- **WIKI ERROR** — a `{{< wiki >}}` shortcode references a slug that has no wiki page.

Warnings do not block the build. An AI agent running `make validate` should notice wiki warnings and fix them before proceeding (add missing SOT links, create missing wiki pages, or add first-mention shortcodes to articles).

---

## Source of Truth vs Wiki Page

The wiki page is a **graph node** — it points at a source-of-truth article.  
The source-of-truth article is where the concept is **introduced and explained**.

| What you want | Where it lives |
|---|---|
| Understand the concept | Source-of-truth article (`source_of_truth` link on the wiki page) |
| Find the canonical name/slug | Wiki page |
| Find all places the concept is used | `uv run scripts/wiki_index.py backrefs <slug>` |
| Add a first-mention link in a new article | `{{< wiki "slug" >}}text{{< /wiki >}}` shortcode |
