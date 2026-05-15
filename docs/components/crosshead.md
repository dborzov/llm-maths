# Crosshead

**Mid-section signpost heading.** A short, punchy label used to break up long columns of running prose. Visually distinct from real `<h2>` so it can punctuate the *interior* of a section without claiming a TOC entry.

## How to invoke

Use the **`<>` delimiter form** (body is a short plain-text label, no Markdown):

```markdown
{{< crosshead >}}The Heavy-Hitter Hypothesis{{< /crosshead >}}
```

## Parameters

| Param | Required | Values | Default |
|---|---|---|---|
| `id` | no | explicit anchor id (rarely needed) | — |

## Rendering

- Pop-display font (Bangers), ~1.6rem.
- Yellow underline stripe running through the lower half of the text.
- Pink corner-tick to the left.
- **Not** included in the article's TOC sidebar.
- **Not** semantic — emits `<p class="crosshead">`, not `<h*>`. This is intentional: crossheads are typographic, not structural.

## When to use

- Every 5–8 paragraphs of unbroken running prose. If a reader could lose their place skimming, drop a crosshead.
- To make a beat shift visible within a single section — "now we look at the *implementation*."

## When NOT to use

- As a real subsection heading. If you want a TOC entry, write `## H2`.
- Adjacent to an H2 or H3. Pick one — they fight visually.
- In short articles (<1000 words). They feel like decoration without long prose underneath.
- More than ~3 per article.

## Where it renders

- Shortcode: `themes/almanac/layouts/shortcodes/crosshead.html`
- CSS: search `CROSSHEAD` in `themes/almanac/assets/css/main.css`
