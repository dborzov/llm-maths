# Pullquote

**Stage-stealing excerpt panel.** Use to lift a profound conclusion, a counter-intuitive result, or a famous author's voice out of running prose. **One per article maximum.** The rarity is the punch.

## How to invoke

Use the **`%%` delimiter form** so Markdown inside the body renders:

```markdown
{{% pullquote author="Grace Hopper" type="profound" %}}
The most dangerous phrase in the language is, "We've always done it this way."
{{% /pullquote %}}
```

## Parameters

| Param | Required | Values | Default |
|---|---|---|---|
| `type` | no | `profound` \| `technical` \| `counter-intuitive` | `profound` |
| `author` | no | free text — rendered as small-caps attribution | — |
| `id` | no | anchor id for deep-linking | — |

## Variants

| Type | Background | When to use |
|---|---|---|
| `profound` | Pop pink | Big claims, big punches. Author quotes. The "thesis statement" of the article. |
| `technical` | Pop teal | Theorem statements. Hard quantitative results. "The KV cache scales linearly with context length but quadratically with batch." |
| `counter-intuitive` | Pop orange | Surprises, contrarian takes, "wait — really?" moments. The unintuitive conclusion the article is building to. |

## When to use

- Once you've earned it. After two paragraphs of setup, lift the punchline so a skimmer who scrolls past prose can still find the point.
- At a section transition. The pullquote stops the reader, then the next H2 starts the new beat.
- To attribute a famous figure's voice — a sentence by Shannon, Tukey, Hopper, etc.

## When NOT to use

- As decoration. If the same sentence works inline, leave it inline.
- For self-attributed authorial commentary. The pullquote is for high-stakes statements; use a [callout](callout.md) for "here's my opinion" notes.
- Stacked. Two pullquotes in one article dilutes both.

## Where it renders

- Shortcode: `themes/almanac/layouts/shortcodes/pullquote.html`
- CSS: search `PULLQUOTE` in `themes/almanac/assets/css/main.css`
