# Ye Olde LLM Maths Almanac — Claude Code Instructions

## What This Project Is

A static website: a wiki-style collection of long-form deep-learning mathematics articles. Pop-art neubrutalist visual design. Built with Hugo + a custom theme + a Python pre-build script.

Live at: `https://borzov.ca/llm-maths/`

## Essential Commands

```bash
make preview      # run pyplot pre-build (cached) + hugo serve --buildDrafts
make plots        # only run pyplot executor (skips unchanged blocks)
make plots-force  # re-run all pyplot blocks regardless of cache
make build        # full production build to public/
make clean        # delete public/ and resources/
make clean-all    # + delete plot cache and generated PNGs
```

**Python tooling:** always use `uv` — never bare `python` or `pip`.

## Adding a New Chapter

```bash
hugo new content chapters/NN_topic-name.md
```

Front matter required fields:

```yaml
---
title: ""
description: ""   # one sentence, shown in article cards
topics: []        # broad: [embeddings], [probability], [optimization]
tags: []          # specific: [numpy, matplotlib, pytorch]
theme: cream      # or: teal
math: true        # or false — controls KaTeX loading
draft: false
---
```

Place the file in `content/chapters/`. Numeric prefix (`01_`, `02_`) controls ordering.

## The Two Themes

| Theme | Background | Primary Accent | Source |
|---|---|---|---|
| `cream` | `#FDF5E6` | `#FF007F` Pop Pink | StyleGuide component in `styles/` |
| `teal` | `#007A7A` | `#FFD700` Pop Yellow | WikiMockup component in `styles/` |

Theme data files: `data/themes/cream.toml` and `data/themes/teal.toml`.

To add a new theme, create `data/themes/newname.toml` with CSS variable overrides. See the existing files for the full list of variables.

## pyplot Executable Code Blocks

The almanac's signature feature. Authors mark code blocks with `pyplot` to have them executed during the build, with the matplotlib plot shown side-by-side with the code.

**Syntax:**

````markdown
```pyplot {id="unique-id" caption="Optional caption"}
x = np.linspace(-3, 3, 200)
plt.plot(x, np.tanh(x), color='#FF007F', linewidth=2)
plt.title("tanh")
```
````

**Rules:**
- `plt` and `np` are pre-imported — do NOT import them
- Do NOT call `plt.show()` or `plt.savefig()`
- `id` must be unique within the article
- PNG output: `static/plots/{section}/{article-slug}/{id}.png`
- Run `make plots` after editing a pyplot block

**Script:** `scripts/run_plots.py` — reads `scripts/.plot_cache.json` (SHA-256 hashes) to skip unchanged blocks.

## KaTeX Math

Set `math: true` in front matter. Use `$...$` for inline, `$$...$$` for display.

The Goldmark `passthrough` extension (configured in `hugo.toml`) protects these delimiters from Markdown processing.

## File Structure Reference

```
content/chapters/       → Articles (the main content)
content/docs/           → Meta: contribute guide, stack docs
content/prompts/        → Article generation prompts shown on site
data/themes/            → CSS variable definitions for each theme
themes/almanac/         → The custom Hugo theme (DO NOT edit node_modules-style)
  assets/css/main.css   → All CSS in one file, ~700 lines
  layouts/              → Hugo templates
  layouts/_default/_markup/render-codeblock-pyplot.html  → pyplot hook
static/js/fold.js       → Auto-wraps H2 sections in <details>
static/js/toc.js        → TOC scroll-spy via IntersectionObserver
static/plots/           → Generated PNGs (gitignored, rebuilt by run_plots.py)
scripts/run_plots.py    → Pre-build pyplot executor
scripts/.plot_cache.json → Cache: code-hash → skip re-execution
```

## Hugo Notes

- `enableGitInfo = false` in `hugo.toml` (not a git repo locally). CI passes `--enableGitInfo` explicitly.
- `baseURL = "https://borzov.ca/llm-maths/"` — all URLs include the `/llm-maths/` prefix.
- CSS is processed by Hugo Pipes: `resources.Get | minify | fingerprint`.
- Sections: `chapters/`, `docs/`, `prompts/`. All use the same `single.html` / `list.html` layouts.

## Design System

From the `styles/` app (React/Vite reference implementation of the visual language):

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

PR builds run steps 1–2 but not deployment.

## Dependency Management

Python deps in `pyproject.toml`. Always use:

```bash
uv run <command>          # runs in project venv
uv run --isolated ...     # throwaway venv (used by run_plots.py internally)
uv add <package>          # add to pyproject.toml
```

Never use bare `pip install`. Never use `apt` for Python packages.
