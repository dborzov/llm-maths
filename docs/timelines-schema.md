# `data/timelines/` — timeline data files

Data backing the `{{< timeline name="..." >}}` shortcode. The shortcode lives in `themes/almanac/layouts/shortcodes/timeline.html`; this directory only holds data.

## Canonical Example

See `quantization2019to2026.toml` — five turning points spanning 2018–2025, used in `content/comicbook/03-quantization/12-hardware-horizon.md`.

## The Five-Event Rule (Hard Cap)

Timelines **MUST have at most 5 events**. This is enforced: extras are silently dropped at render time with a build warning, and `make validate` flags any TOML with >5 events.

The point of a timeline is *the big picture, not the bibliography*. If your story needs more than five inflection points, write the additional context as prose around the timeline, or split into two timelines.

When choosing five events, ask: *if I removed this event, would the rest of the story not have happened?* If the answer is "the rest would have happened anyway", it doesn't belong on the timeline.

## Schema (TOML)

```toml
title   = "How LLMs Got Small — 2019 to 2026 in five turning points"   # optional
caption = "Five inflection points. <em>Hundreds of papers omitted.</em>"  # optional, HTML allowed

[[events]]
date  = "2022"                                # any short string — "2022", "Q3 2024", "May 2023"
title = "LLM.int8()"                          # what happened
who   = "Dettmers, Lewis, Belkada, Zettlemoyer"   # who did it (optional)
body  = "Outlier-aware INT8 inference for OPT-175B."  # one-sentence description
kind  = "paper"                               # paper | hardware | format | release
link  = "06-outliers/"                        # optional, resolves to issue cover dir
```

## Visual Kinds

| `kind`     | Dot color   | Meaning                                |
|------------|-------------|----------------------------------------|
| `paper`    | pink        | A specific paper or algorithmic move   |
| `hardware` | teal        | Silicon release (GPU, accelerator)     |
| `format`   | yellow      | Number-format / standard ratification  |
| `release`  | orange      | A specific model/system shipping       |

## How Links Resolve

Same convention as the techtree shortcode: `link` values without a leading `/` resolve against the **issue cover's RelPermalink**. So a timeline embedded in an article correctly links sideways to other articles in the same issue.
