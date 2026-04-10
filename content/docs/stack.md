---
title: "Tech Stack & Build System"
description: "A detailed breakdown of how the almanac is built — Hugo, the custom theme, the two visual styles, pyplot execution, and GitHub Pages deployment."
topics: [meta]
tags: [hugo, python, uv, matplotlib, katex, github-pages]
theme: cream
math: false
draft: false
---

## Overview

The almanac is a **static website**. There is no server, no database, no JavaScript framework. Every page is a pre-built HTML file served directly from GitHub Pages.

The build pipeline has two stages:

```
1. Python pre-build  →  executes pyplot code blocks, saves PNG plots
2. Hugo build        →  renders Markdown → HTML, bundles CSS, fingerprints assets
```

Both stages run locally with a single `make preview` command. In CI they run sequentially in GitHub Actions before deployment.


## Hugo

[Hugo](https://gohugo.io) (v0.123.7+extended) is the static site generator.

- **Configuration:** `hugo.toml`
- **Theme:** `themes/almanac/` — fully custom, no third-party theme dependency
- **Content:** `content/` — Markdown files with YAML front matter
- **Data:** `data/` — TOML files consumed by templates (theme definitions live here)
- **Static assets:** `static/` — JS files, generated plots

Hugo was chosen for: speed (sub-100ms builds), excellent Markdown processing via Goldmark, built-in syntax highlighting via Chroma, and robust template language.

### Key Hugo Configuration (`hugo.toml`)

```toml
baseURL = "https://borzov.ca/llm-maths/"
theme   = "almanac"
enableGitInfo = false   # enabled in CI via --enableGitInfo flag

[markup.goldmark.extensions.passthrough]
  enable = true
  [markup.goldmark.extensions.passthrough.delimiters]
    block  = [["$$", "$$"]]
    inline = [["$", "$"]]

[markup.highlight]
  style     = "monokai"
  noClasses = true        # inline styles — no separate chroma CSS needed
```

The `passthrough` extension is critical: it tells Goldmark to leave `$...$` and `$$...$$` delimiters untouched so KaTeX can render them client-side. Without it, Goldmark would eat the `$` signs.

### Directory Structure

```
/
├── hugo.toml                    ← Site config
├── Makefile                     ← Developer shortcuts
├── pyproject.toml               ← Python dep spec (uv)
│
├── archetypes/
│   └── chapters.md              ← Template for new articles
│
├── content/
│   ├── _index.md                ← Homepage content
│   ├── chapters/                ← The main articles
│   ├── docs/                    ← Meta: this page, contribute guide
│   └── prompts/                 ← Article generation prompts
│
├── data/
│   └── themes/
│       ├── cream.toml           ← Cream theme CSS variable values
│       └── teal.toml            ← Teal theme CSS variable values
│
├── static/
│   ├── js/
│   │   ├── fold.js              ← Auto-foldable H2 sections
│   │   └── toc.js               ← TOC scroll-spy
│   └── plots/                   ← Generated PNGs (gitignored)
│
├── scripts/
│   └── run_plots.py             ← pyplot pre-build executor
│
├── themes/almanac/
│   ├── assets/css/main.css      ← All CSS (~700 lines)
│   └── layouts/
│       ├── _default/
│       │   ├── baseof.html
│       │   ├── single.html
│       │   ├── list.html
│       │   └── _markup/render-codeblock-pyplot.html
│       ├── index.html
│       └── partials/
│           ├── head.html
│           ├── header.html
│           ├── footer.html
│           ├── toc.html
│           ├── theme-vars.html
│           └── katex.html
│
└── .github/workflows/deploy.yml ← CI/CD
```


## The Custom `almanac` Theme

The theme is built from scratch — no Bootstrap, no Tailwind, no external CSS framework. It faithfully implements the **pop-art neubrutalist** design language established in the `styles/` reference app.

Design principles, directly from the style guide:

- **No rounded corners.** Sharp 90° angles everywhere.
- **Thick black borders.** `3px solid #1A1A1A` on structural elements.
- **Hard offset shadows.** `4px 4px 0px #1A1A1A` — no blur radius.
- **High-saturation solid colours.** Large blocks of Pop Pink, Teal, Yellow, Orange.
- **Bangers** (display/heading) + **Space Grotesk** (body) — loaded from Google Fonts.
- **Visible structure.** Halftone dot patterns, checkered textures as decorative overlays.

### CSS Architecture

All CSS lives in `themes/almanac/assets/css/main.css`, processed by Hugo Pipes:

```html
{{ $css := resources.Get "css/main.css" | minify | fingerprint }}
<link rel="stylesheet" href="{{ $css.RelPermalink }}">
```

The file is structured into clearly labelled sections:

| Section | Purpose |
|---|---|
| Design tokens (`:root`) | CSS custom properties with cream defaults |
| Reset | Minimal `box-sizing`, margin/padding zero |
| Base typography | Body font, link colours, `code`, `hr` |
| Utility classes | `.card-pop`, `.halftone`, `.checkered`, `.label` |
| Site header / nav | Sticky nav bar, brand, links, search |
| Article layout | CSS Grid two-column: TOC sidebar + content |
| TOC sidebar | Sticky positioning, scroll-spy highlight |
| Article content | Prose: headings, lists, blockquotes, tables, images |
| Code blocks | Dark background, Monokai, highlight wrapper |
| pyplot blocks | Side-by-side flex: code left, plot right |
| Foldable sections | `<details>`/`<summary>` styles |
| Math (KaTeX) | Display math container, overflow handling |
| Homepage | Hero, article grid, cards |
| Footer | Dark ink background, brand, links |
| Responsive | Mobile collapse at 900px and 600px |


## The Two Visual Themes

Each article declares its visual theme in front matter (`theme: cream` or `theme: teal`). The system has three parts:

### 1. Data files (`data/themes/*.toml`)

Each theme is a TOML file of CSS variable values:

```toml
# data/themes/teal.toml
bg-page      = "#007A7A"
bg-surface   = "#FDF5E6"
bg-nav       = "#005858"
accent-1     = "#FFD700"
text-nav     = "#FDF5E6"
```

### 2. The `theme-vars.html` partial

This partial reads the TOML and emits an inline `<style>` block in `<head>`:

```html
<style>
:root {
  --bg-page: #007A7A;
  --bg-surface: #FDF5E6;
  --accent-1: #FFD700;
  /* ... */
}
</style>
```

Because it appears *after* the linked stylesheet, it wins in cascade order without `!important`.

### 3. CSS uses only variables

The stylesheet never has a raw colour value — only `var(--accent-1)`, `var(--bg-page)` etc. Switching the variables via the `<style>` block changes the entire page's colour scheme.

The `<html data-theme="teal">` attribute is also set (by `baseof.html`) for any CSS or JS that needs to branch on the current theme.

**Adding a new theme** is one file. Create `data/themes/violet.toml` with the variable overrides, and any article can use `theme: violet`.


## pyplot — Executable Code Blocks

This is the most unusual part of the build. Articles can contain fenced code blocks that are *executed* during the build, with their matplotlib output embedded next to the code.

### Author Syntax

````markdown
```pyplot {id="activation-fns" caption="tanh vs sigmoid"}
x = np.linspace(-4, 4, 300)
plt.plot(x, np.tanh(x), label='tanh', color='#FF007F', linewidth=2)
plt.plot(x, 1/(1+np.exp(-x)), label='sigmoid', color='#00A8A8', linewidth=2)
plt.legend()
plt.tight_layout()
```
````

`plt` and `np` are pre-imported. Do not call `plt.savefig()` or `plt.show()`.

### Pre-build Script (`scripts/run_plots.py`)

The script is run before Hugo, via `uv run scripts/run_plots.py`.

**What it does:**

1. Walks all `.md` files in `content/`
2. Finds `pyplot {id="..."}` fenced blocks using a regex
3. Computes a SHA-256 hash of each code block's body
4. Compares to `scripts/.plot_cache.json` — skips blocks whose hash matches (nothing changed)
5. For each stale/new block, writes a wrapper script:
   ```python
   import matplotlib; matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   import numpy as np
   __ALMANAC_OUTPUT__ = "static/plots/chapters/slug/id.png"
   # ... user code ...
   plt.savefig(__ALMANAC_OUTPUT__, dpi=150, bbox_inches="tight")
   plt.close("all")
   ```
6. Runs `uv run --isolated --with numpy --with matplotlib python /tmp/script.py`
7. Saves the PNG to `static/plots/{section}/{article-slug}/{id}.png`
8. Updates the cache; deletes orphaned PNGs for removed blocks
9. Exits 1 if any plot failed (fails the CI build)

The `--isolated` flag means uv creates an ephemeral environment without touching the project venv. This keeps plot dependencies clean and reproducible.

### Hugo Render Hook

Hugo intercepts `pyplot` fences via a code block render hook at:
`themes/almanac/layouts/_default/_markup/render-codeblock-pyplot.html`

It constructs the PNG path from the page's file path and the block's `id` attribute, then renders:

```html
<div class="pyplot-block">
  <div class="pyplot-code">{{ highlight .Inner "python" "" }}</div>
  <figure class="pyplot-output">
    <img src="/llm-maths/plots/chapters/slug/id.png" loading="lazy">
    <figcaption>caption text</figcaption>
  </figure>
</div>
```

CSS makes `.pyplot-block` a two-column grid (code left, plot right), collapsing to single column on mobile.

### Plot Output Path Convention

| Content file | PNG output |
|---|---|
| `content/chapters/02_logits.md`, id `sigmoid` | `static/plots/chapters/02_logits/sigmoid.png` |
| `content/chapters/sub/deep-dive.md`, id `fig1` | `static/plots/chapters/sub/deep-dive/fig1.png` |

The path is derived from the `.md` file's path relative to `content/`, with `.md` stripped.


## Foldable Sections (`fold.js`)

`static/js/fold.js` runs on `DOMContentLoaded`. It finds all `<h2>` elements inside `.article-content` and wraps each one and its following siblings (until the next `<h2>`) in a `<details><summary>` pair. The first section starts `open`.

- Heading IDs (used for anchor links from the TOC) are moved from the `<h2>` to the `<details>` element so in-page navigation continues to work.
- No authoring changes required — the transformation is purely client-side.


## TOC Scroll-Spy (`toc.js`)

`static/js/toc.js` uses the browser's `IntersectionObserver` API to monitor headings as they scroll. When a heading enters the top quarter of the viewport, the corresponding TOC link gets class `toc-active`, which the CSS highlights in `--accent-1` colour.

The TOC itself is generated by Hugo from H2 and H3 headings, configured in `hugo.toml`:

```toml
[markup.tableOfContents]
  startLevel = 2
  endLevel   = 3
```


## Math Rendering (KaTeX)

KaTeX v0.16.11 is loaded from CDN via the `katex.html` partial, only on pages where `math: true` is set in front matter (avoids loading KaTeX on pages that don't need it).

The `auto-render` extension scans the document for `$...$` and `$$...$$` delimiters after the page loads. The Goldmark `passthrough` extension ensures these delimiters survive Markdown processing untouched.


## Python Dependency Management (uv)

[uv](https://docs.astral.sh/uv/) manages all Python tooling. The project's `pyproject.toml` declares:

```toml
[project]
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
]
```

`uv run scripts/run_plots.py` creates a `.venv` on first run and installs dependencies. Subsequent runs use the cached venv.

For plot execution, `uv run --isolated` creates throwaway environments per-execution, keeping the project venv clean.

PyTorch is available in plots via `uv run scripts/run_plots.py --with-torch` (or `make plots-torch` if you add that target).


## Local Development

```bash
# First-time setup (uv auto-creates .venv)
make preview          # run plots + hugo serve --buildDrafts

# Subsequent workflows
make plots            # run changed pyplot blocks only (cached)
make plots-force      # re-run all pyplot blocks
make build            # production build to public/
make clean            # delete public/ and resources/
make clean-all        # + delete plot cache and generated PNGs
```

`hugo serve` watches for file changes and rebuilds automatically. Note: pyplot PNGs are not regenerated on save — run `make plots` after editing a pyplot block, then the server will pick up the new PNG.


## GitHub Actions CI/CD (`.github/workflows/deploy.yml`)

```
push to main
  └── build job
        ├── actions/checkout@v4      (fetch-depth: 0 for git history)
        ├── peaceiris/actions-hugo@v3
        ├── astral-sh/setup-uv@v5
        ├── uv run scripts/run_plots.py
        ├── hugo --minify --enableGitInfo
        └── upload-pages-artifact
  └── deploy job (main only)
        └── actions/deploy-pages@v4
```

PR builds run the `build` job but not `deploy` — catching errors before they reach `main`.

The `--enableGitInfo` flag is only passed in CI (where a full git history is available from `fetch-depth: 0`). Locally, `enableGitInfo = false` in `hugo.toml` avoids errors in non-git directories.


## Deployment Target

The site deploys to **GitHub Pages** at `https://borzov.ca/llm-maths/`.

Hugo's `baseURL = "https://borzov.ca/llm-maths/"` ensures all asset URLs (CSS, JS, plots, internal links) are correctly prefixed with `/llm-maths/` in the generated HTML.

The custom domain (`borzov.ca`) is configured in the GitHub repository's Pages settings, pointing to the `gh-pages` branch (or Actions deployment environment, depending on setup).
