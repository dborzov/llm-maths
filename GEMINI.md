# GEMINI.md - Ye Olde LLM Maths Almanac

This project is a serialized collection of long-form deep-learning mathematics articles, structured as "comic-book issues" with a pop-art neubrutalist visual design.

## Project Overview

- **Goal:** Provide rigorous, visual, and intuitive explanations of LLM mathematics without "hand-waving."
- **Structure:** Content is organized into **Issues** (Hugo sections). Each issue contains multiple **Articles** (mainline narrative, primers, or capstone "boss" chapters).
- **Visual Style:** "Pop-art neubrutalist" — no rounded corners, thick borders, hard shadows, halftone dots, and specific high-contrast color themes (`cream` and `teal`).
- **Interactive Elements:** Tech tree graphs (table of contents), timelines, and executable `pyplot` code blocks that render as images.

## Tech Stack

- **Static Site Generator:** [Hugo](https://gohugo.io/) (v0.123.7+extended).
- **Theme:** Custom `almanac` theme (located in `themes/almanac/`).
- **Math:** KaTeX (configured via Hugo passthrough).
- **Plots:** Python (Matplotlib + Numpy) executed during pre-build via `scripts/run_plots.py`.
- **Package Management:** 
    - [uv](https://docs.astral.sh/uv/) for Python dependencies (`pyproject.toml`).
    - `npm` for the design system reference in `styles/`.
- **Deployment:** GitHub Actions to GitHub Pages.

## Building and Running

### Prerequisites
- Hugo (extended version)
- Python 3.11+ and `uv`

### Common Commands
- `make preview`: Runs validation, generates/updates plots, and starts the Hugo dev server at `http://localhost:1313/llm-maths/`.
- `make build`: Validates, runs plots, and builds the production site to `public/`.
- `make validate`: Runs the project-specific linter (`scripts/validate.py`) to check front matter, links, and conventions.
- `make plots`: Executes `pyplot` blocks in markdown files (cached in `scripts/.plot_cache.json`).
- `make test-ui-install`: One-time install of Playwright + chromium under `tests/`.
- `make test-ui`: Runs the Playwright UI test suite (mobile / tablet / desktop). Required after any CSS, JS, template, or shortcode change. Loop until 0 failed — see [`docs/dev.md`](docs/dev.md) → "Testing".
- `make clean-all`: Resets the build environment, including plot caches.

## Development Conventions

### 1. Canonical Terminology (The microGPT Contract)
All articles must use the variable names and architectural structure of the **microGPT** reference implementation (`content/comicbook/05-microgpt/`).
- **Symbols:** `wte`, `wpe`, `attn_wq/k/v`, `attn_wo`, `mlp_fc1/2`, `lm_head`, `q`, `k`, `v`, `attn_logits`.
- **Phases:** `prefill` and `decode`.
- **Action:** On first mention of a canonical term, link to its corresponding primer in Issue 05.

### 2. Article Authoring
- **Scaffolding:** Use `make new-issue NN=NN SLUG=slug TITLE="Title"` and `make new-article ISSUE=NN-slug SLUG=slug TITLE="Title"`.
- **Types (`techKind`):**
    - `mainline`: Narrative chapters advancing the story.
    - `primer`: Standalone tutorials on prerequisites.
    - `boss`: Capstone articles.
- **Content Checklist:**
    - Open with a "human moment" (story/history).
    - Use at least one concrete example.
    - Math + Intuition + Picture (Pyplot).
    - Cross-link siblings within the issue.

### 3. Pyplot Blocks
- Syntax: ` ```pyplot {id="unique-id" caption="Caption"} `
- Rules: `plt` and `np` are pre-imported. Do NOT call `plt.show()` or `plt.savefig()`.
- Environment: Isolated `uv` venv with ONLY `numpy` and `matplotlib`.

### 4. Visuals and Assets
- **Themes:** Specified in front matter (`theme: cream` or `theme: teal`).
- **Component Library:** Every reusable visual block (pullquote, callout, margin note, crosshead, figure, hero, timeline, tech tree, infographic, pyplot) is catalogued at **[`docs/components/README.md`](docs/components/README.md)** — read that index first, then drill into per-component reference files only as needed. Human-facing showcase at `/llm-maths/docs/components/`.
- **Images:** Hero images (the article banner set via `header:` front matter) MUST be **WebP**, max 1600px wide, and < 300KB.
- **Design System:** Reference available in `styles/` (run `npm run dev` there to view).

## Project Structure
- `content/comicbook/`: The core articles and issue covers.
- `content/docs/components.md`: Human-facing showcase of every reusable component.
- `data/techtrees/`: TOML definitions for the issue-level tech trees.
- `data/timelines/`: TOML definitions for the timeline widgets.
- `docs/components/`: AI-agent reference for the component library (read `README.md` first).
- `themes/almanac/`: Hugo layouts, shortcodes, and CSS (`assets/css/main.css`).
- `scripts/`: Python tools for validation, plotting, and scaffolding.
- `static/header-illustrations/`: WebP hero images (see `docs/components/hero.md`).
- `static/plots/`: Generated PNGs from `pyplot` blocks (gitignored).
- `tests/`: Playwright UI test suite (mobile / tablet / desktop). See [`tests/README.md`](tests/README.md).

## Validation
`make validate` is mandatory before committing. It enforces front matter fields, tech tree integrity, cross-link resolution, and `pyplot` constraints.

## UI Testing (Playwright)
`make test-ui` is mandatory before committing any change to CSS, JS, templates, or shortcodes. The suite pins down tap, click, keyboard, and viewport-breakpoint behaviour across mobile / tablet / desktop. When fixing a UI bug, add a test that reproduces the bug before fixing it. Loop until 0 failed — never weaken or skip assertions to make the red go away. Full operating procedure (when to add tests, what to add, work-until-green loop) in [`docs/dev.md`](docs/dev.md) → "Testing" and [`tests/README.md`](tests/README.md).
