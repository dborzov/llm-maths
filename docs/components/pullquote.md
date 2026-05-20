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

## Style: headline, not prose

A pullquote should read like a headline, not a paragraph. The goal is that a skimmer who scrolls past all the prose still walks away with the key idea.

**Rules:**
- Short, dense, information-maximising. No running sentences.
- Bold the numbers and key terms: `**91% → 13%**`, `**2,000×**`, `**GQA**`.
- Bullet lists work well when the point is a stack of moves or a contrast set. Use `-` not `*`.
- Strip qualifiers that belong in prose ("at long context, on the most popular model on the planet"). The surrounding text handles context; the pullquote handles the punch.
- If the original sentence had a soft setup clause ("The answer is not in any single mechanism. The answer is in the *stack* —"), cut it. Start with the stack.

**Headline style with a list (canonical example):**

```markdown
{{% pullquote type="counter-intuitive" %}}
The DeepSeek V4 attention stack:

- **MLA** — shrinks the KV cache
- **DSA** — lightning indexer shrinks the score matrix
- **CSA** — compresses the sequence before scoring
- **HCA** — throws away even the sparsity bookkeeping
{{% /pullquote %}}
```

**Headline style with a single dramatic statistic:**

```markdown
{{% pullquote type="counter-intuitive" %}}
Efficient LLM inference isn't an ML problem. It's grappling with one physical fact:

**A GPU core takes ~2,000× longer to load two numbers from HBM than to multiply them in a register.**
{{% /pullquote %}}
```

**Headline style with a contrast count:**

```markdown
{{% pullquote type="counter-intuitive" %}}
Since 2023, KV cache optimizations that conquered production: **GQA**, **MQA**, **MLA**, **FlashAttention**.

As of 2026: [96 papers on KV cache pruning](https://github.com/October2001/Awesome-KV-Cache-Compression).

Saw production use: **0**.
{{% /pullquote %}}
```

**Moving thrown-out content into prose:** when you strip a qualifier or setup clause from the pullquote, make sure its meaning lands somewhere in the surrounding paragraph before or after the pullquote — not just deleted.

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
