# LLM Maths Comics — Claude Code Instructions

## What This Project Is

A static website: a serialized collection of long-form deep-learning mathematics articles, structured as **comic-book issues**. Each issue groups multiple articles around a single theme and presents them via a **tech tree graph** as table of contents. Pop-art neubrutalist visual design. Hugo + custom theme + Python pre-build script.

Live at: `https://borzov.ca/llm-maths/`

**Canonical reference**: `content/issues/03-sixteen-numbers/` is the reference implementation for the new issue format. When in doubt, copy what issue 03 does.

**Canonical vocabulary for LLM architecture**: `content/issues/05-microgpt-unfolded/` introduces a single 60-line plain-Python transformer (microGPT) that this entire project uses as its shared reference implementation. Variable names from microGPT — `wte`, `wpe`, `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`, `mlp_fc1`, `mlp_fc2`, `lm_head`, `q`, `k`, `v`, `head_dim`, `n_layer`, `n_embd`, `block_size`, `n_head`, `keys[li]`, `values[li]`, `x_residual`, `attn_logits`, `attn_weights`, `head_out`, `x_attn`, the `prefill` and `decode` phases — are the **canonical names** every article in this project should use. See [the microGPT terminology contract](#the-microgpt-terminology-contract) below for the full enforcement rules.

## The Issue Format (READ THIS FIRST)

An **issue** is the unit of release. It contains **multiple articles** sharing one theme:

- A short, playful **issue title** that hooks readers.
- A **mystery / cold open** article that sets up the question the whole issue answers.
- A handful of **mainline narrative chapters** that progress the story.
- A handful of **primer / tech-tree chapters** — standalone tutorials on each prerequisite mathematical concept, written in the spirit of James Burke's *Connections* (connecting the idea to history, everyday objects, and other applications the reader probably already knows).
- A **tech tree graph** rendered on the issue cover page, acting as the table of contents — readers see the dependency structure visually and click into nodes.
- **Heavy internal linking**: every time a mainline chapter mentions a load-bearing concept, link to the primer. Every primer reciprocally links forward to the mainline chapter that uses it.

Audience level: comfortable with intro linear algebra / calculus / statistics / probability, basic ML & LLM architecture, numpy + pytorch. Tone: Radiolab / Planet Money / James Burke — narrate through a mystery, anchor abstractions in human stories and specific examples, surface "aha" moments, use napkin-math / Fermi estimates, lean on concrete metaphors.

## Authoring A New Issue — The Fast Path

Given only a topic from the user, here is the **complete recipe**:

```bash
# 1. Pick a playful title and a slug. Sketch the dependency DAG on paper.
make new-issue NN=04 SLUG=attention-anatomy TITLE='Attention, Anatomized'
# → creates content/issues/04-attention-anatomy/_index.md
# → creates data/techtrees/issue04.toml (stub with example nodes)

# 2. Edit data/techtrees/issue04.toml — define the actual nodes and edges
#    (schema reference at docs/techtrees-schema.md)

# 3. Scaffold each article in the order the reader will encounter them.
#    The script auto-increments NN and weight, and alternates the theme.
make new-article ISSUE=04-attention-anatomy SLUG=cold-open      TITLE='The Cold Open'         KIND=mainline
make new-article ISSUE=04-attention-anatomy SLUG=dot-products   TITLE='Dot Products Revisited' KIND=primer
make new-article ISSUE=04-attention-anatomy SLUG=softmax-primer TITLE='Softmax & Friends'      KIND=primer
# ... etc

# 4. Fill in each article. Match the canonical style.
# 5. Validate continuously while you work.
make validate

# 6. Preview and ship.
make preview
```

The scaffolding scripts live at `scripts/new_issue.py`; the linter is `scripts/validate.py`. Both are idempotent — they refuse to overwrite existing files.

## Quality Bar — Every Article Must Have

This is the per-article checklist. If any item is missing, the article isn't done.

- [ ] **Opens with a human moment.** A specific year, a specific person, a specific failed approach. Never "In this article we will…".
- [ ] **One concrete example carried through.** Better to anchor one well than five poorly. Resist the "also see" pile-up.
- [ ] **Math + intuition + picture.** Whenever a load-bearing equation appears, immediately follow with what the math *feels like*, and where possible a pyplot block that shows it.
- [ ] **An "aha" turn.** Set up a question, demonstrate the naive approach failing, then deliver the surprising trick.
- [ ] **Napkin math.** Fermi estimates for the key quantities (parameters, bytes, FLOPs, tokens).
- [ ] **At least 2 pyplot blocks** for mainline articles; **1+ pyplot block** for primers where appropriate. Use theme colors: `#FF007F`, `#00A8A8`, `#FFD700`, `#FF8C00`.
- [ ] **Cross-links to siblings.** Every concept that has a primer must link to that primer with `[label](../slug-of-primer/)`. Every primer should link forward to the mainline article that uses it.
- [ ] **Pop-art formatting variety.** Mix `**bold**`, `*emphasis*`, tables, blockquotes, `<details>` blocks, fenced code, inline HTML/SVG. Reach for the [component library](docs/components/README.md) — pullquotes, callouts, margin notes, crossheads — to break up uniform pages. Don't let the page be one font weight on one background.
- [ ] **Closes with a forward link.** A "Continue to → [Next Article]" line at the bottom, written as a cliffhanger that pulls the reader onward.
- [ ] **Uses microGPT terminology where applicable.** Any time the article touches a transformer-internal concept that already has a name in microGPT, *use that name* and link to the matching primer in issue 5. The terminology contract is enforced in the next section.

## The microGPT Terminology Contract

Issue 5 (`content/issues/05-microgpt-unfolded/`) introduces a 60-line plain-Python transformer that is the shared **reference implementation** for everything else in this project. Whenever a later article (in any issue) discusses transformer internals, it should refer to those internals using **microGPT's variable names and structure**, and link back to the issue 5 primer that explains the concept.

This keeps the project coherent: a reader who has read issue 5 once can pick up *any* article in *any* later issue and know that `attn_wk` always means the same thing.

### The Canonical Names

| Symbol | What it is | Where in microGPT | Primer to link |
|---|---|---|---|
| `wte` | token embedding table | `state_dict['wte']` | [ch.3 Tokens & Positions](../05-microgpt-unfolded/03-embeddings/) |
| `wpe` | positional embedding table | `state_dict['wpe']` | [ch.3 Tokens & Positions](../05-microgpt-unfolded/03-embeddings/) |
| `attn_wq` / `attn_wk` / `attn_wv` | per-layer Q/K/V projections | `state_dict[f'layer{li}.attn_w?']` | [ch.6 Q, K, V](../05-microgpt-unfolded/06-qkv-projections/) |
| `attn_wo` | attention output projection | `state_dict[f'layer{li}.attn_wo']` | [ch.9 Multi-Head Attention](../05-microgpt-unfolded/09-multi-head/) |
| `mlp_fc1` / `mlp_fc2` | MLP fatten / skinny matrices | `state_dict[f'layer{li}.mlp_fc?']` | [ch.11 The MLP Block](../05-microgpt-unfolded/11-mlp-block/) |
| `lm_head` | vocab projection | `state_dict['lm_head']` | [ch.15 LM Head & Sampling](../05-microgpt-unfolded/15-sampling/) |
| `q`, `k`, `v` | current-token query/key/value | local in `gpt()` | [ch.6](../05-microgpt-unfolded/06-qkv-projections/) |
| `q_h`, `k_h`, `v_h` | per-head slices | local in head loop | [ch.9](../05-microgpt-unfolded/09-multi-head/) |
| `attn_logits` | raw `q · k / √d` scores | local | [ch.8 Scaled Dot-Product Attention](../05-microgpt-unfolded/08-attention/) |
| `attn_weights` | softmaxed scores | local | [ch.8](../05-microgpt-unfolded/08-attention/) |
| `head_out` | per-head output vector | local | [ch.8](../05-microgpt-unfolded/08-attention/) |
| `x_attn` | concatenated head outputs | local | [ch.9](../05-microgpt-unfolded/09-multi-head/) |
| `x_residual` | residual-stream copy held aside | local | [ch.10 The Residual Stream](../05-microgpt-unfolded/10-residual-stream/) |
| `keys[li]`, `values[li]` | KV cache for layer `li` | function argument | [ch.13 The KV Cache](../05-microgpt-unfolded/13-kv-cache/) |
| `n_layer`, `n_embd`, `block_size`, `n_head`, `head_dim` | model hyperparameters | `const.py` | [ch.2 The State Dict](../05-microgpt-unfolded/02-state-dict/) |
| **prefill** / **decode** | the two inference phases | driver loop | [ch.14 Prefill vs Decode](../05-microgpt-unfolded/14-prefill-decode/) |
| KV cache shape `(2, L, H, T, D)` | the 5D tensor view of the cache | implicit | [ch.17 The Three Axes](../05-microgpt-unfolded/17-kv-axes/) |
| `n_kv_head`, `group_size` | GQA/MQA sharing factor | (extension) | [ch.18 Grouped-Query Attention](../05-microgpt-unfolded/18-gqa/) |
| `kv_down`, `d_c`, latent `c` | MLA low-rank cache form | (extension) | [ch.19 Multi-head Latent Attention](../05-microgpt-unfolded/19-mla/) |
| sliding window `W` | per-layer attention window | (extension) | [ch.20 Sliding-Window Attention](../05-microgpt-unfolded/20-sliding-window/) |
| `state[li]` for SSM layers | fixed-size recurrent state replacing KV | (extension) | [ch.21 State-Space Hybrids](../05-microgpt-unfolded/21-ssm-hybrids/) |

### What This Means In Practice

When writing a *new* article that touches any of these concepts:

1. **Use the microGPT name on first mention** rather than coining a synonym. Write "the `attn_wk` projection" rather than "the key matrix `W_K`" or "the key weights". If the source paper uses different notation, introduce both: *"the K projection (`attn_wk` in our reference listing)"*.
2. **Link to the issue 5 primer** the first time a microGPT name appears in the article. Use a sibling-relative link like `[the `attn_wk` projection](../../05-microgpt-unfolded/06-qkv-projections/)` from another issue, or `../06-qkv-projections/` from within issue 5.
3. **Show the relevant slice of the listing** when the article is talking about one specific line. Copy-paste the 1–4 line excerpt from microGPT verbatim, do *not* rewrite it into a different style. Consistency is the entire value proposition.
4. **For shape questions, defer to the table in [ch.1 Sixty Lines, One LLM](../05-microgpt-unfolded/01-cold-open/#the-names-you-should-tattoo)**. Don't re-derive `wte` is `vocab_size × n_embd` in every article — point at the canonical table.

### Exception: When NOT To Use microGPT Terms

MicroGPT is intentionally *minimal*. It does not cover:
- **GLU-family activations** (SwiGLU, GeGLU) — the listing only has `relu`. If you need to discuss gated MLPs, name the gate variables yourself and explicitly note the divergence from the listing.
- **Rotary or ALiBi positions** — the listing uses additive `wpe`. If you need RoPE, name the rotation parameters yourself.
- **Multi-query or grouped-query attention** — the listing has `n_head` independent heads. Note when you're describing GQA/MQA explicitly.
- **Layer-norm variants beyond RMSNorm** — the listing uses `rmsnorm`.
- **Anything from the training loop** — microGPT is inference-only.

When you cross one of these boundaries, **say so explicitly** in the article: *"microGPT uses a plain additive `wpe`; modern models use rotary position embeddings (RoPE) — see [chapter X] for the substitution."*

## Primer vs Mainline vs Boss — How To Decide

When sketching the tech tree, each article slots into one of three roles. Pick deliberately.

| If the article is...                                                       | Use `techKind` | Tree node color |
|----------------------------------------------------------------------------|----------------|------------------|
| The narrative entry point (cold open). No prereqs.                         | `mainline`     | not in tree      |
| A storyline chapter advancing the plot ("X happened, then Y broke things") | `mainline`     | pink             |
| A standalone tutorial on one math concept that other chapters build on     | `primer`       | cream            |
| The capstone "everything builds to this" finale                            | `boss`         | yellow halftone  |
| Cross-issue reference to a concept covered elsewhere                       | `external`     | faint teal       |

Heuristic: **mainline chapters advance the story. Primer chapters can be read in any order. Boss chapters are the destination.**

## Directory Layout For An Issue

Each issue is a **Hugo section** — a directory containing an `_index.md` (the cover page) plus the article files.

```
content/issues/
  _index.md                         ← top-level issues list
  01_embeddings.md                  ← legacy single-article issue (leaf)
  02_logits.md                      ← legacy single-article issue (leaf)
  03-sixteen-numbers/               ← new-format multi-article issue (section) — CANONICAL
    _index.md                       ← cover page (tech tree TOC + intro text)
    01-cold-open.md                 ← mainline article
    02-numbers-in-boxes.md          ← primer article
    ...
```

**File naming inside an issue**: `NN-slug.md` where `NN` is a zero-padded order index. The number drives sort order in the cover-page article list and `PrevInSection` / `NextInSection` navigation.

Articles **stay flat** within their issue directory — do NOT nest further.

### Required Front Matter For An Article

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

`weight` controls article order on the issue cover and in prev/next navigation. **Use steps of 10** so you can insert later without renumbering.

`make validate` enforces every field above; mismatches fail the build.

### Required Front Matter For An Issue Cover (`_index.md`)

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
2. Embed the tech-tree graph via `{{< techtree name="issueNN" >}}`.
3. Optionally end with a short reading-order note. The article list is rendered automatically below.

## The Component Library

Every reusable visual building block — pullquotes, callouts, margin notes, crossheads, figures, pyplot blocks, code blocks, hero images, timelines, tech trees, and interactive infographics — is documented at **[`docs/components/README.md`](docs/components/README.md)**.

Read that index file before writing or editing an article. It is intentionally short (~80 lines): a catalog table that points you at the per-component detail files so you don't load every component's reference into context unless you actually need it.

The human-facing showcase with live examples lives at `/llm-maths/docs/components/` (source: `content/docs/components.md`).

**One-screen cheat sheet:**

| You want… | Reach for |
|---|---|
| The top-of-article banner | `header:` front-matter field (see `docs/components/hero.md`) |
| Profound conclusion / theorem | `{{% pullquote %}}` |
| Tangent / tip / warning / definition | `{{% callout %}}` |
| Citation or single-sentence aside | `{{% marginnote %}}` |
| Mid-section signpost (not a TOC entry) | `{{< crosshead >}}` |
| Third-party image with caption + lightbox | `{{< figure src=... caption=... >}}` |
| Our own matplotlib plot | ` ```pyplot ` fenced block |
| Five-event chronology | `{{< timeline name="..." >}}` |
| Issue-cover DAG TOC | `{{< techtree name="..." >}}` |
| Interactive panel instead of a wall-of-numbers table | `{{< infographic >}}` |

## Validation

`make validate` runs `scripts/validate.py`, which checks four invariants:

1. **Front matter contract**: every article has all required fields with valid types and enum values.
2. **Tech tree consistency**: every issue's `{{< techtree >}}` shortcode references a real data file; every article's `techNode` (unless `techKind=mainline` for cold-open) matches a node id; every tree-node `link` resolves to an actual article file.
3. **Cross-link integrity**: every `[text](../slug/)` link inside an article resolves to a real sibling article in the same issue.
4. **Pyplot rules**: unique ids, no forbidden imports (scipy/torch/etc.), no `plt.show`/`plt.savefig`.

`make build` and `make preview` both run validate first. A failed validate fails the build.

## KaTeX Math

Set `math: true` in front matter. Use `$...$` for inline, `$$...$$` for display.

The Goldmark `passthrough` extension (configured in `hugo.toml`) protects these delimiters from Markdown processing.

## Pitfalls We Already Hit — Don't Re-Discover Them

These are mistakes from prior sessions that took non-trivial time to debug. They're written down so they stay solved.

1. **Shortcode `relURL` is rooted at the site, not the calling page.** If you write `{{ .link | relURL }}` in a shortcode, the link will resolve to `/llm-maths/06-foo/` rather than `/llm-maths/issues/03-NN/06-foo/`. **Fix**: capture `.Page.RelPermalink` (or `.Page.Parent.RelPermalink` if the calling page is an article) into a local variable *before* entering the range loop, and prepend it to relative links. See `techtree.html` and `timeline.html` for the pattern.

2. **Inside Go template `range` blocks, `$` is the root context — NOT the parent iterator.** `$.x` will fail if `.` was rebound. **Fix**: capture the current item with `{{- $node := . -}}` immediately upon entering the range, then use `$node.x` instead.

3. **`_build.render: never` in `cascade` is unreliable across Hugo versions.** Trying to hide a content section via cascade left the child pages building anyway. **Fix**: keep non-publishable files OUTSIDE `content/`. Authoring prompts live at the repo root in `prompts/`, not in `content/prompts/`.

4. **Pyplot isolated env has only numpy + matplotlib.** Importing `scipy.stats.norm` will pass local linting but fail the plot run. **Fix**: use numpy-only equivalents (e.g. hardcoded quantile values from the published source rather than computing them with scipy). `make validate` now catches this preemptively.

5. **Hugo template variable `:=` re-declaration in the same scope is an error.** If you copy a `{{ $x := ... }}` from one block to another in the same template, the second one fails. **Fix**: declare once at the top, reuse below.

6. **Hugo `Parent.RelPermalink` for the top-level issues section is `/llm-maths/issues/`, NOT `/issues/`.** Comparing against literal `/issues/` will fail for the breadcrumb logic. **Fix**: compare against `(site.GetPage "/issues").RelPermalink` — Hugo will give you the correctly-prefixed path.

## File Structure Reference

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
data/timelines/                   → timeline event data per use
docs/                             → repo-local authoring docs (NOT rendered)
  components/README.md            → component-library index (READ FIRST)
  components/<name>.md            → per-component reference (load on demand)
  techtrees-schema.md             → tech-tree TOML schema reference
  timelines-schema.md             → timeline TOML schema reference
themes/almanac/                   → custom Hugo theme
  assets/css/main.css             → all CSS in one file (see "COMPONENT LIBRARY" section)
  layouts/                        → Hugo templates
  layouts/_default/issue-cover.html         → issue cover layout
  layouts/_default/_markup/render-codeblock-pyplot.html  → pyplot hook
  layouts/shortcodes/             → all component shortcodes live here
static/js/fold.js                 → auto-wraps H2 sections in <details>
static/js/toc.js                  → TOC scroll-spy
static/js/pyplot.js               → lightbox for pyplot blocks + figure shortcode
static/header-illustrations/      → article hero images (always .webp, ≤1600px wide, ≤300KB — see docs/components/hero.md)
static/plots/                     → generated PNGs (gitignored)
scripts/run_plots.py              → pre-build pyplot executor
scripts/new_issue.py              → scaffold new issues / articles
scripts/validate.py               → convention linter (front matter, tech tree, links, pyplot)
```

## Essential Commands

```bash
make help            # list all targets with their purposes
make preview         # validate, run plots, serve at http://localhost:1313/llm-maths/
make build           # validate, run plots, build minified site to public/
make validate        # run the linter on its own
make plots           # only run pyplot executor (skips unchanged blocks)
make plots-force     # re-run all pyplot blocks regardless of cache
make clean           # delete public/ + resources/
make clean-all       # + delete plot cache and generated PNGs

make new-issue NN=04 SLUG=foo TITLE='Foo'
make new-article ISSUE=04-foo SLUG=cold-open TITLE='Cold Open' KIND=mainline
```

**Python tooling:** always use `uv` — never bare `python` or `pip`.

## Sanity-Check Protocol

After every significant change (CSS, templates, content, scripts):

1. `make validate` — must exit clean.
2. `uv run scripts/run_plots.py` — must exit with 0 errors.
3. `hugo` — must build cleanly.
4. `make preview` and visually inspect affected pages at `http://localhost:1313/llm-maths/`.

`make build` chains 1–3 in order, so it's the single command to run for a full pre-deploy check.

## The Two Themes

| Theme | Background | Primary Accent | Source |
|---|---|---|---|
| `cream` | `#FDF5E6` | `#FF007F` Pop Pink | StyleGuide component in `styles/` |
| `teal`  | `#007A7A` | `#FFD700` Pop Yellow | WikiMockup component in `styles/` |

Theme data files: `data/themes/cream.toml` and `data/themes/teal.toml`. The scaffolding script auto-alternates the theme per-article for visual variety.

## Hugo Notes

- `enableGitInfo = false` in `hugo.toml`. CI passes `--enableGitInfo` explicitly.
- `baseURL = "https://borzov.ca/llm-maths/"` — URLs include the `/llm-maths/` prefix.
- CSS via Hugo Pipes: `resources.Get | minify | fingerprint`.
- The prompts section lives **outside** `content/` (in `prompts/` at repo root) — Hugo never sees it.

## Design System

From the `styles/` app (React/Vite reference of the visual language):

- **No rounded corners.** Zero `border-radius` everywhere.
- **Hard shadows.** `4px 4px 0px #1A1A1A` — no blur.
- **Thick borders.** `3px solid #1A1A1A` on structural elements.
- **Halftone dots.** `radial-gradient(#1A1A1A 1px, transparent 0); background-size: 5px 5px`
- **Bangers** (Google Fonts) for display/headings, **Space Grotesk** for body text.
- No JavaScript framework on the site — vanilla JS only.

## GitHub Actions

`.github/workflows/deploy.yml` — triggers on push to `main`:
1. `uv run scripts/run_plots.py`
2. `hugo --minify --enableGitInfo`
3. Deploy to GitHub Pages via `actions/deploy-pages@v4`

## Dependency Management

```bash
uv run <command>          # runs in project venv
uv run --isolated ...     # throwaway venv (used by run_plots.py internally)
uv add <package>          # add to pyproject.toml
```

Never use bare `pip install`. Never use `apt` for Python packages.

## When You Are Stuck

- "What does a good article look like?" → read `content/issues/03-sixteen-numbers/06-outliers.md` (mainline) and `content/issues/03-sixteen-numbers/03-lloyd-max.md` (primer).
- "Which component do I use here?" → `docs/components/README.md`. Then drill into the per-component file.
- "What does a good tech tree look like?" → see `data/techtrees/issue03.toml`; schema in `docs/techtrees-schema.md`.
- "What does a good timeline look like?" → see `data/timelines/quantization2019to2026.toml`; schema in `docs/timelines-schema.md`.
- "What conventions am I forgetting?" → run `make validate`. It tells you.
