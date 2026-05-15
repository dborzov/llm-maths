# Timeline

**A five-event horizontal milestone strip.** Use it once per article — at the start, to give readers the big-picture chronology of an idea before you dive in.

Hard cap: **5 events.** This is enforced both by `make validate` and by the shortcode itself (which silently drops events past index 4 with a build warning). The cap is intentional: timelines exist to show the big picture, not the full bibliography.

## How to invoke

```markdown
{{< timeline name="quantization2019to2026" >}}
```

The `name` argument selects a TOML file at `data/timelines/<name>.toml`.

## Data schema

```toml
title   = "QUANTIZATION (2019 → 2026)"   # optional caption header
caption = "Five papers, one bottleneck"   # optional footer (safeHTML)

[[events]]
date = "2022"
title = "LLM.int8()"
who   = "Dettmers et al."
body  = "Outlier-aware 8-bit inference for OPT-175B."
kind  = "paper"             # paper | hardware | format | release
link  = "06-outliers/"      # optional, resolves against issue cover

[[events]]
# … up to 5 entries total
```

`kind` drives the dot colour:
- `paper` → pop pink (default)
- `hardware` → pop teal
- `format` → pop yellow
- `release` → pop orange

## When to use

Timelines work best when:
- The article opens with "field history" — three to five papers that bracket the topic.
- The story you're telling is **rate of change**, not "everything that ever happened."
- The reader can be expected to know roughly when transformers / GPT-3 / Llama happened — the timeline anchors *your* narrative within that.

Timelines do NOT work when:
- The dates are arbitrary (e.g. "Step 1 / Step 2 / Step 3"). Use a numbered list instead.
- You need >5 events. Either split into two timelines (rarely a good sign) or write prose.
- The chronological order isn't the point. Use a [techtree](techtree.md) for dependency order or a [pullquote](pullquote.md) for a single quote.

## Where it renders

- Shortcode: `themes/almanac/layouts/shortcodes/timeline.html`
- CSS: search `TIMELINE WIDGET` in `themes/almanac/assets/css/main.css`
- Full schema reference: [`docs/timelines-schema.md`](../timelines-schema.md)
- Existing examples: `data/timelines/quantization2019to2026.toml`, `data/timelines/longcontext-bench-2023to2026.toml`

## Pitfalls

- **Link resolution is rooted at the *issue cover*, not the article calling the shortcode.** This means a relative `link = "06-outliers/"` in the TOML will resolve to `.../<issue-slug>/06-outliers/` whether the shortcode is rendered on the cover or inside an article. This is intentional — don't try to fix it.
- **`body` is rendered as `safeHTML`.** Keep it plain text; if you must use HTML, escape carefully.
