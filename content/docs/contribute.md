---
title: "Contributing Articles"
description: "How to write, format, and submit articles to the almanac — including the art of prompting an LLM to do the heavy lifting."
topics: [meta]
tags: [contributing, format, pyplot, llm-authoring]
theme: cream
math: true
draft: false
---

## Philosophy

Every article in the almanac follows the same north star: make the reader *grok* the idea, not just memorise it.

That means:

- **Historical narrative.** Start with a real person, a real problem, and the dead ends they hit before the insight. Walk in their shoes.
- **Concrete toy examples first.** Before the general formula, show a specific case with real numbers. $n = 4$, not $n$.
- **Working code.** Every claim that can be computed, should be computed. Python + NumPy is the baseline. PyTorch where relevant.
- **No hand-waving.** If something "can be shown," show it.
- **Plots that earn their place.** A graph should reveal something that prose cannot.


## Article Front Matter

Every article begins with a YAML front matter block:

```yaml
---
title: "The Attention Mechanism: A Complete Mathematical Treatment"
description: "Scaled dot-product attention from first principles to matrix form."
topics: [attention, transformers]
tags: [attention, linear-algebra, numpy, pytorch]
theme: cream
math: true
draft: false
---
```

| Field | Required | Values | Notes |
|---|---|---|---|
| `title` | yes | string | The article title shown in nav and cards |
| `description` | yes | string | One-sentence summary; shown in article cards |
| `topics` | yes | list | Broad category: `[embeddings]`, `[probability]`, `[optimization]`, etc. |
| `tags` | yes | list | Specific tags for navigation |
| `theme` | yes | `cream` or `teal` | Visual style — see [Choosing a Theme](#choosing-a-theme) |
| `math` | yes | `true` / `false` | Set `true` whenever the article uses $\LaTeX$ math |
| `draft` | yes | `true` / `false` | Set `false` when ready to publish |


## File Location and Naming

Place chapters in `content/chapters/`:

```
content/
└── chapters/
    ├── 01_embeddings.md
    ├── 02_logits.md
    └── 03_your_new_article.md
```

Use a numeric prefix (`03_`) to control ordering in the chapters list. Lowercase, underscores, descriptive.

Create a new chapter skeleton with:

```bash
hugo new content chapters/03_your_topic.md
```

This uses the archetype at `archetypes/chapters.md` to pre-fill the front matter.


## Section Structure

Articles are divided by `##` H2 headings. Each H2 becomes a foldable section in the rendered article — the reader can collapse sections they've already understood, keeping long articles navigable.

The table of contents is built automatically from H2 and H3 headings and appears in the sticky left sidebar.

**Best practice:** Aim for 5–10 H2 sections per article. Each section should be self-contained enough that a reader can orient themselves from the heading alone.

```markdown
## The Problem With One-Hot Vectors

*Opening of section…*

### Why Zero Dot Products Are Catastrophic

*Sub-section…*

## The Skip-Gram Objective

*Next main section…*
```


## Python Code Blocks

Standard fenced code blocks render with syntax highlighting (Monokai):

````markdown
```python
import numpy as np

x = np.linspace(-3, 3, 200)
y = 1 / (1 + np.exp(-x))   # sigmoid
```
````

For output / terminal snippets, use an unlabelled fence:

````markdown
```
array([0.047, 0.119, 0.269, 0.500, 0.731, 0.881, 0.953])
```
````


## pyplot Blocks — Executable Code with Embedded Plots

The almanac's signature feature. Mark a code block as `pyplot` and the pre-build script runs it, saves the matplotlib figure, and renders code and plot side-by-side.

**Syntax:**

````markdown
```pyplot {id="sigmoid-curve" caption="The logistic sigmoid function"}
import numpy as np

x = np.linspace(-6, 6, 400)
plt.figure(figsize=(7, 4))
plt.plot(x, 1 / (1 + np.exp(-x)), color='#FF007F', linewidth=2.5)
plt.axhline(0.5, color='#1A1A1A', linewidth=1, linestyle='--', alpha=0.4)
plt.axvline(0, color='#1A1A1A', linewidth=1, linestyle='--', alpha=0.4)
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Logistic Sigmoid')
plt.tight_layout()
```
````

**Rules for pyplot code:**

- `plt` (matplotlib.pyplot) and `np` (numpy) are **pre-imported** — do not import them yourself.
- Do **not** call `plt.show()` or `plt.savefig()` — the build script handles saving.
- The **last active figure** is what gets saved. If you create multiple figures, only the last one is captured.
- The `id` attribute must be **unique within the article**.
- `caption` is optional but recommended.
- `torch` is available if you pass `--with-torch` to `run_plots.py`.

**Run plots locally:**

```bash
make plots          # run only changed blocks (cached by code hash)
make plots-force    # re-run all blocks regardless of cache
```

Generated PNGs land at `static/plots/chapters/{slug}/{id}.png`.


## Math

The almanac uses KaTeX. Inline math with `$...$`, display math with `$$...$$`:

```markdown
The sigmoid function maps any real number to $(0, 1)$:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Its derivative is famously self-referential: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.
```

Set `math: true` in front matter to load KaTeX for that page.


## Choosing a Theme

Each article specifies its own visual theme. Currently two are defined:

| Theme | Background | Primary Accent | When to Use |
|---|---|---|---|
| `cream` | `#FDF5E6` Pop Cream | `#FF007F` Pop Pink | Default. Probability, statistics, ML theory |
| `teal` | `#007A7A` deep teal | `#FFD700` Pop Yellow | Geometry, linear algebra, transformers |

The choice is aesthetic — use whichever feels right for the article's subject matter.

Adding a new theme is a single file: create `data/themes/mytheme.toml` with CSS variable overrides. See `data/themes/teal.toml` for the format.


## Using an LLM to Write Articles

The almanac was designed with LLM-assisted authorship in mind. The chapters you see were themselves generated by LLM and then edited. This is the recommended workflow:

### Step 1 — Write the prompt

A good almanac prompt has these ingredients:

1. **Topic and arc** — what the article is about, from what angle
2. **Narrative style** — historical walk-through, "walking in the shoes of the inventor"
3. **Toy examples** — explicitly ask for small, concrete worked examples with real numbers
4. **Code requirements** — Python, NumPy, matplotlib; pyplot blocks for plots
5. **Audience calibration** — assume: intro stats, comfortable with NumPy/PyTorch, wants to grok not just memorise
6. **Output format** — a markdown file with Hugo front matter, `##` sections, `pyplot` blocks

See the [example prompts]({{< relref "/prompts" >}}) for reference — those are the exact prompts that generated the existing chapters.

### Step 2 — Generate the draft

Paste the prompt into Claude (or GPT-4, Gemini, etc.). Ask for a complete markdown file. Request the full front matter block.

**Tips for better output:**

- Tell the model the article will have foldable sections (H2 headings), so it should make each section self-contained
- Ask it to use `pyplot {id="..."}` syntax for any plot
- Specify preferred figure size (`figsize=(8, 5)` or `(10, 4)`) for consistent visuals
- Ask for `# ---- user code ----` style comments inside pyplot blocks for readability
- If the article is long, generate section-by-section and stitch together

### Step 3 — Edit and validate

1. Add the file to `content/chapters/`
2. Run `make plots` — fix any Python errors in pyplot blocks
3. Run `make preview` — review in browser
4. Edit prose, adjust plots, tighten the narrative
5. Set `draft: false` when satisfied

### Example Prompt Template

```
Write a chapter for the LLM Maths Almanac on [TOPIC].

Arc: [ONE SENTENCE DESCRIBING THE NARRATIVE ARC]

Style requirements:
- Open with a historical anecdote: a specific person, a specific problem they were stuck on, the
  wrong turns they took before the insight. Make it feel like a detective story.
- Every concept must be demonstrated with a concrete toy example (real numbers, small matrices,
  readable arrays). Show the idea before naming it.
- Python code: NumPy and matplotlib. Use the pyplot block syntax for any figure:
  ```pyplot {id="unique-id" caption="Description"}
  (code — plt and np are pre-imported, do not call plt.savefig or plt.show)
  ```
- Math: KaTeX inline ($...$) and display ($$...$$).
- Structure: 6-8 H2 sections. Each section should be self-contained enough to read in isolation.
- Audience: comfortable with intro statistics and NumPy/PyTorch. Wants to grok intuition,
  not memorise formulas.

Output: a complete markdown file with Hugo front matter:
---
title: "..."
description: "..."
topics: [...]
tags: [...]
theme: cream   # or teal
math: true
draft: false
---

Topic: [TOPIC IN DETAIL]
```


## Submitting

This is an open source repository on GitHub. The standard flow:

1. Fork the repository
2. Write your chapter in `content/chapters/`
3. Run `make plots && make preview` to verify locally
4. Open a pull request

For plot generation in CI, the GitHub Actions workflow runs `uv run scripts/run_plots.py` automatically before the Hugo build. Generated PNGs are **not committed** — they are always regenerated from source.
