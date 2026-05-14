# LLM Maths Comics — Claude Code Instructions

## What This Project Is

A static website: a serialized collection of long-form deep-learning mathematics articles, structured as **comic-book issues**. Each issue groups multiple articles around a single theme and presents them via a **tech tree graph** as table of contents. Pop-art neubrutalist visual design. Hugo + custom theme + Python pre-build script.

Live at: `https://borzov.ca/llm-maths/`

**Canonical reference**: `content/issues/03-sixteen-numbers/` is the reference implementation for the new issue format. When in doubt, copy what issue 03 does.

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
- [ ] **Pop-art formatting variety.** Mix `**bold**`, `*emphasis*`, tables, blockquotes, `<details>` blocks, fenced code, inline HTML/SVG. Don't let the page be one font weight on one background.
- [ ] **Closes with a forward link.** A "Continue to → [Next Article]" line at the bottom, written as a cliffhanger that pulls the reader onward.

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
header: default.png     # optional; defaults to header-illustrations/default.png
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
header: default.png     # optional; large illustration on the cover
---
```

The body of `_index.md` should:
1. Open with a few paragraphs setting up the mystery / cold-open framing.
2. Embed the tech-tree graph via `{{< techtree name="issueNN" >}}`.
3. Optionally end with a short reading-order note. The article list is rendered automatically below.

## The Tech Tree Widget (Reusable)

**Where**: shortcode at `themes/almanac/layouts/shortcodes/techtree.html`. Data at `data/techtrees/<name>.toml`. **Full schema in `docs/techtrees-schema.md`** — that's where you should look first when authoring a tree.

**Layout convention**: Y-axis is the dependency direction — boss/mainline at the **top**, prerequisites at the **bottom**. Arrows point UPWARD (from prerequisite to user).

**Link resolution**: relative `link` values in the TOML resolve against the **issue cover's RelPermalink**, so the shortcode works whether called from the cover or from inside an article.

## The Timeline Widget (Reusable)

**Where**: shortcode at `themes/almanac/layouts/shortcodes/timeline.html`. Data at `data/timelines/<name>.toml`. **Full schema in `docs/timelines-schema.md`**.

**Hard cap of 5 events** — extra events are silently dropped with a build warning. This is intentional: timelines exist to show the *big picture*, not the full bibliography. If you have more than five events, split into multiple timelines or write prose.

## Pyplot Executable Code Blocks

The almanac's signature feature. Mark code blocks with `pyplot` to execute them during build, with the matplotlib plot shown side-by-side with the code.

**Syntax:**

````markdown
```pyplot {id="unique-id" caption="Optional caption"}
x = np.linspace(-3, 3, 200)
plt.plot(x, np.tanh(x), color='#FF007F', linewidth=2)
plt.title("tanh")
```
````

**Strict rules** (enforced by `make validate`):
- `plt` and `np` are pre-imported — do NOT import them.
- Do NOT call `plt.show()` or `plt.savefig()`.
- `id` must be unique within the article.
- **Pyplot blocks run in an isolated `uv run --isolated` venv with ONLY `numpy` and `matplotlib`.** `scipy`, `torch`, `pandas`, `sklearn`, `sympy` are NOT available. Use numpy-only equivalents. If you genuinely need another package, update `scripts/run_plots.py`'s isolated-env dependency list AND document it here.
- PNG output: `static/plots/{section}/{article-slug}/{id}.png`.

**Script:** `scripts/run_plots.py` — reads `scripts/.plot_cache.json` (SHA-256 hashes) to skip unchanged blocks.

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
  techtrees-schema.md             → tech-tree TOML schema reference
  timelines-schema.md             → timeline TOML schema reference
themes/almanac/                   → custom Hugo theme
  assets/css/main.css             → all CSS in one file
  layouts/                        → Hugo templates
  layouts/_default/issue-cover.html         → issue cover layout
  layouts/_default/_markup/render-codeblock-pyplot.html  → pyplot hook
  layouts/shortcodes/techtree.html          → tech tree widget
  layouts/shortcodes/timeline.html          → timeline widget
static/js/fold.js                 → auto-wraps H2 sections in <details>
static/js/toc.js                  → TOC scroll-spy
static/header-illustrations/      → article header images
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
- "What does a good tech tree look like?" → see `data/techtrees/issue03.toml`; schema in `docs/techtrees-schema.md`.
- "What does a good timeline look like?" → see `data/timelines/quantization2019to2026.toml`; schema in `docs/timelines-schema.md`.
- "What conventions am I forgetting?" → run `make validate`. It tells you.
