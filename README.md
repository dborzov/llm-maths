# Ye Olde LLM Maths Almanac

A wiki-style collection of long-form deep-learning mathematics articles. Pop-art neubrutalist visual design. No hand-waving. Just the maths, done carefully.

**Live site:** [borzov.ca/llm-maths](https://borzov.ca/llm-maths/)

---

## Quick Start

**Prerequisites:** [Hugo](https://gohugo.io/installation/) (v0.123.7+extended), [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
make preview        # generate plots (cached) + start dev server
```

Open `http://localhost:1313/llm-maths/` in your browser.

Other commands:

```bash
make build          # full production build → public/
make plots          # re-run only changed pyplot blocks
make plots-force    # re-run all pyplot blocks
make clean          # delete public/ and resources/
make clean-all      # + delete plot cache and generated PNGs
```

---

## Adding a Chapter

```bash
hugo new content chapters/03_your-topic.md
```

Edit the front matter:

```yaml
---
title: "Your Title"
description: "One sentence for the article card."
topics: [attention]
tags: [numpy, pytorch]
theme: cream          # or: teal
math: true
draft: false
---
```

Then write your article. For executable plots, use the `pyplot` block syntax:

````markdown
```pyplot {id="my-plot" caption="Optional caption"}
x = np.linspace(-3, 3, 200)
plt.plot(x, np.tanh(x))
plt.title("tanh")
```
````

`plt` and `np` are pre-imported. Do not call `plt.show()` or `plt.savefig()`.

Run `make plots` to execute and preview the plot.

→ Full contributor guide: [borzov.ca/llm-maths/docs/contribute/](https://borzov.ca/llm-maths/docs/contribute/)

---

## Visual Themes

Each article specifies a visual theme in its front matter. Two themes are built in:

| Theme | Background | Accent | Source |
|---|---|---|---|
| `cream` | `#FDF5E6` Pop Cream | `#FF007F` Pop Pink | StyleGuide component in `styles/` |
| `teal` | `#007A7A` deep teal | `#FFD700` Pop Yellow | WikiMockup + concept-art2 in `styles/` |

The `styles/` directory contains a Vite + React app that renders the design system reference. Run `npm run dev` inside `styles/` to explore it.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Static site generator | Hugo v0.123.7+extended |
| Theme | Custom `almanac` theme (no third-party) |
| CSS | Single-file pop-art design system (`themes/almanac/assets/css/main.css`) |
| Math | KaTeX (CDN, loaded per-page) |
| Plots | Python + matplotlib via `scripts/run_plots.py` |
| Python deps | uv (`pyproject.toml`) |
| CI/CD | GitHub Actions → GitHub Pages |

→ Full stack breakdown: [borzov.ca/llm-maths/docs/stack/](https://borzov.ca/llm-maths/docs/stack/)

---

## Project Structure

```
content/chapters/       Articles
content/docs/           Contributor guide, stack docs
content/prompts/        Article generation prompts
data/themes/            CSS variable definitions (cream.toml, teal.toml)
themes/almanac/         Custom Hugo theme
scripts/run_plots.py    pyplot pre-build executor
static/js/              fold.js (foldable sections), toc.js (scroll-spy)
styles/                 Visual design reference app (Vite + React)
```

---

## Deployment

Pushes to `main` trigger GitHub Actions:
1. `uv run scripts/run_plots.py` — generate matplotlib plots
2. `hugo --minify --enableGitInfo` — build site
3. Deploy to GitHub Pages at `borzov.ca/llm-maths/`

Pull requests build but do not deploy.

---

## Contributing

Open an issue or pull request. See the [contributor guide](https://borzov.ca/llm-maths/docs/contribute/) for article format, pyplot syntax, theme selection, and LLM-assisted authoring tips.
