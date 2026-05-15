# Figure

**Image + caption with click-to-fullscreen lightbox.** Use for third-party diagrams, screenshots, hand-drawn explanations — anything you want a reader to be able to zoom into. For our own matplotlib plots, use the [pyplot block](pyplot.md) instead.

## How to invoke

Use the **`<>` delimiter form** (parameters only — no body):

```markdown
{{< figure src="diagrams/attention-map.webp"
           alt="Heatmap of attention scores"
           caption="Attention scores over a 2048-token window. Notice the spike at the first 4 tokens (the *attention sinks*)." >}}
```

## Parameters

| Param | Required | Description |
|---|---|---|
| `src` | **yes** | Image URL or path. Absolute URLs (`http://...`) and rooted paths (`/llm-maths/...`) are used verbatim; relative paths are run through `relURL`. Prefer paths under `static/`. |
| `alt` | recommended | A11y description. Falls back to `caption` if omitted, but you should provide it explicitly. |
| `caption` | recommended | Figcaption text. Rendered via the inline Markdown renderer, so `*italic*`, `**bold**`, and `[links](...)` work. |
| `credit` | no | Photographer / source attribution. Renders small and uppercase to the right of the caption. |
| `width` | no | CSS max-width override (e.g. `60%`, `400px`). Defaults to full content width. |
| `id` | no | Anchor id for deep-linking. |

## File contract

- **Format:** WebP preferred; PNG/SVG allowed if vector content. JPEG only for photos.
- **Storage:** Under `static/figures/<issue-slug>/<article-slug>/<name>.<ext>` for issue-specific images, or `static/figures/shared/<name>.<ext>` for reused diagrams.
- **Size:** keep under 400 KB.

## Lightbox behaviour

Clicking the image (or its corner zoom hint) opens the **shared pyplot lightbox** — same `<dialog>` that pyplot blocks use, defined in `static/js/pyplot.js`. Inside the lightbox:
- ESC or backdrop-click closes.
- Click the image to toggle "fit-to-viewport" vs "actual pixels" (scrolling within the wrapper).
- The caption text appears at the bottom of the lightbox.

No-JS fallback: the `<a href="...">` still opens the raw image at full size in a new tab.

## When to use

- A third-party paper figure (with attribution via `credit`).
- A hand-drawn diagram or whiteboard photo.
- A screenshot of a tool, terminal, or model output.

## When NOT to use

- Our own matplotlib plot — use a [pyplot block](pyplot.md). Pyplot blocks are reproducible, cached, and have the source listing right next to the image.
- The article's hero — use the `header:` front-matter field instead. See [hero.md](hero.md).
- Decorative imagery with no caption. If a reader doesn't need the caption, they probably don't need the image.

## Where it renders

- Shortcode: `themes/almanac/layouts/shortcodes/figure.html`
- CSS: search `FIGURE` in `themes/almanac/assets/css/main.css`
- JS (lightbox wiring): `static/js/pyplot.js` — the `figures` selector at the top.
