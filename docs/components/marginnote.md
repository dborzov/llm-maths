# Margin Note

**Tufte-style sidenote.** Floats in the right margin on wide screens; inlines as a small annotated block on tablets/phones. Use for citations, corollaries, brief clarifications, and anything that's genuinely optional reading.

## How to invoke

Place the shortcode inline inside a sentence. Use the **`%%` delimiter form** so Markdown inside renders:

```markdown
We could afford to store the entire context.
{{% marginnote %}}
At 32k context on Llama-3 70B the KV cache exceeds **16 GB per user** — see [Dao 2023](https://arxiv.org/abs/2307.08691).
{{% /marginnote %}}
But in 2024 context is the new scarcity.
```

The marginnote is wrapped in a `<span>`, so it can sit inside a `<p>` without breaking nesting.

## Parameters

| Param | Required | Values | Default |
|---|---|---|---|
| `label` | no | text shown on the mobile inline label / hover tooltip | `note` |
| `id` | no | anchor id | — |

## Rendering rules

- **Desktop (≥1280px):** floats absolutely into the right margin (240px column), italic and ~78% opacity. Hover restores full opacity.
- **Mobile / tablet (<1280px):** renders as a small teal-bordered block inline, with a tiny uppercase label above.
- **Trigger marker:** a small pink ⊕ glyph appears inline in the prose where the note was placed.

The two forms exist because Tufte-style margin notes are useless on phones. Don't try to hide the marker on mobile — the marker is the affordance that tells the reader "there's an aside here."

## When to use

- A citation that would interrupt the sentence: paper title, link, page number.
- A quick clarification: "We use base-2 logs throughout."
- A corollary or "by the way" that's a single sentence.

## When NOT to use

- Anything longer than ~2 sentences. Use a [callout](callout.md).
- A definition that every reader will need. Inline it or use a `note` callout.
- More than 3 in one article. They visually clutter the right rail.

## Where it renders

- Shortcode: `themes/almanac/layouts/shortcodes/marginnote.html`
- CSS: search `MARGIN NOTE` in `themes/almanac/assets/css/main.css`

## Pitfalls

- **Place the shortcode where you want the inline marker — not where you want the note text to "land."** On desktop the note floats up to align with its marker; on mobile it appears right where you wrote it.
- **The aside is wrapped in `<span>`, not `<div>`.** This is intentional (so it can sit in a `<p>`). If you put block-level Markdown inside (headings, multiple paragraphs, tables) the HTML will be malformed.
