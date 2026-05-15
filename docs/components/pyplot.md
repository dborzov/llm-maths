# Pyplot Block

**Executable matplotlib code fence.** Python source is rendered as a PNG at build time, the source listing sits in a collapsible details element below the image, and clicks open the shared lightbox. Pyplot blocks are the canonical way to ship plots in this project.

## How to invoke

Use a fenced code block with `pyplot` as the language and an attribute block:

````markdown
```pyplot {id="tanh-curve" caption="The tanh activation, our reference S-curve"}
x = np.linspace(-3, 3, 200)
plt.plot(x, np.tanh(x), color='#FF007F', linewidth=2)
plt.title("tanh")
```
````

## Required attributes

| Attribute | Required | Description |
|---|---|---|
| `id` | **yes** | Unique within the article. Used as the filename of the generated PNG and as the DOM id. |
| `caption` | recommended | Short uppercase caption rendered under the image. |

## Strict rules (enforced by `make validate`)

- **`plt` and `np` are pre-imported** by the wrapper. Do NOT import them inside the block.
- **Do NOT call `plt.show()` or `plt.savefig()`.** The wrapper does this.
- **`id` must be unique** within the article.
- **Isolated env has only `numpy` and `matplotlib`.** No `scipy`, `torch`, `pandas`, `sklearn`, `sympy`. If you genuinely need another package, update `scripts/run_plots.py`'s isolated-env dependency list — but try numpy-only equivalents first.
- **Theme colors only:** `#FF007F` (pink), `#00A8A8` (teal), `#FFD700` (yellow), `#FF8C00` (orange), plus the ink `#1A1A1A`. Don't pull from `matplotlib`'s default cycle.

## Output

- PNG saved to `static/plots/{section}/{article-slug}/{id}.png`.
- Hash cache at `scripts/.plot_cache.json` skips unchanged blocks on rebuild.
- `make plots-force` re-runs every block regardless of cache.

## Rendering

The Hugo hook at `themes/almanac/layouts/_default/_markup/render-codeblock-pyplot.html` produces a vertical card:

1. The plot image at the top, full container width. Click to open the lightbox.
2. The optional `caption=` text underneath (uppercase, italic-feeling).
3. A `<details>` collapsible holding the Python source — collapsed by default, so the picture leads.

## When to use

- **Always** when the plot is reproducible from numpy. Pyplot is preferred over `figure` because:
  - It's regenerable — change the formula, get the new plot for free.
  - The source sits next to the image — readers see *how* the plot is made.
  - It's cached, so rebuilds are fast.

## Per-article quantity

- **Mainline articles:** at least 2.
- **Primer articles:** 1+ where appropriate.
- **Issue covers:** 0 (covers shouldn't have body plots — they're table-of-contents pages).

## Where it renders

- Hugo hook: `themes/almanac/layouts/_default/_markup/render-codeblock-pyplot.html`
- Python runner: `scripts/run_plots.py`
- Validation: `scripts/validate.py` (see `check_pyplot_blocks`)
- CSS: search `PYPLOT BLOCKS` in `themes/almanac/assets/css/main.css`
- JS (lightbox): `static/js/pyplot.js`
