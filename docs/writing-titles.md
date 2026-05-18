# Writing Article Titles and Blurbs

**When to read this:** You are writing or editing the `title`, `short_title`, `description`, or `blurb` fields of an article or comic book cover. These are the first things a potential reader sees.

---

## Intent

The title and blurb form a card in a feed of dozens. The reader gives it ~1–2 seconds while scanning. Your job is **not** to summarize the post and **not** to attract every reader. Your job is to let the *right* reader self-identify and decide it's worth their click.

Think of it as a billboard for a sophisticated technical audience (Hacker News / lobste.rs crowd):

- **Fixed real estate, small budget.** Title ≤ ~10 words. Blurb ≤ ~40 words, 3–4 short bullets.
- **One hook, not five.** Cognitive load must stay low.
- **Spend the budget on the target reader.** Don't try to be legible to everyone.
- **Sophisticated audience, not marketing voice.** Information-dense, specific, technical. Allergic to fluff, hedging, adjectives, and "you'll learn…" framing.

---

## The Pattern

**Title:** `Keyword: hook`

- **Keyword** anchors the reader's context in the first word or two — the right reader sees a familiar term and knows the domain instantly. Use the most recognizable umbrella term, not the most precise one.
- **Hook** (after the colon) is short, often a question or a striking claim. Questions provoke curiosity; the blurb answers nothing, the article does.
- Total: ≤ 10 words.

**`short_title`:** The Keyword part only (text before `:`). Used in tech tree labels. E.g., title `"Kahan Summation: a 1965 fix…"` → short_title `"Kahan Summation"`.

**`description`:** One clean sentence. No HTML. Used in article cards (compact list) and SEO meta. No hedging verbs.

**Blurb:** 3–4 short bullets.

- Each bullet is one striking idea: a surprising fact, a sharp question, a reframing, a stakes-laden setup.
- Short. Plain text (markdown OK, inline code OK).
- Leave the resolution inside the post. Setups and questions, not conclusions.
- Each bullet pulls its weight independently.

---

## Worked Example

**The post:** a deep dive into Kahan summation — floating-point non-associativity, the O(n·ε) error accumulation, Wilkinson's 1960 diagnosis, Kahan's 1965 algorithm, and how the same bug appeared in vLLM's FP8 attention kernel decades later.

**Before:**
```yaml
title: "The Sum Is Not What You Think"
description: "Floating-point summation is non-associative. The error grows with how many terms you add. James Wilkinson at NPL in 1960, William Kahan at Berkeley in 1965, and the compensated-summation algorithm that fixed it."
```

What's wrong: the title is a generic mystery hook with no keyword anchor — the right reader can't self-identify from it. The description is a wall of text that lists sections rather than landing a single hook.

**After:**
```yaml
title: "Kahan Summation: a 1965 fix inside your 2026 GPU"
short_title: "Kahan Summation"
description: "Floating-point addition is non-associative. The error grows with n. Wilkinson named it in 1960; Kahan fixed it in 1965. The same class of bug lived in vLLM's attention kernel."
blurb:
  - "(a + b) + c ≠ a + (b + c) in floating point. This is always true."
  - "Sum n numbers naively: error grows as O(n·ε). At n = 128,000, that error swamps the result."
  - "James Wilkinson at NPL named it in 1960. William Kahan fixed it in 1965."
  - "The same class of bug sat in vLLM's FP8 attention kernel — waiting for context windows to grow long enough."
```

Why this works:

- **"Kahan Summation"** is the recognizable umbrella term. The right reader sees it and knows this is relevant.
- **"a 1965 fix inside your 2026 GPU"** is the hook — surprising time gap, specific, implies "this is more relevant than you'd think."
- **Bullets** each carry one idea: the non-associativity fact (bullet 1), the error growth formula (bullet 2), the historical attribution (bullet 3), the payoff connection to the present (bullet 4).
- **`description`** is one sentence — the same information, compressed, for the card view.

---

## More Examples

```yaml
title: "Hopper FP8: when the FP32 accumulator stops accumulating"
short_title: "Hopper FP8"
description: "NVIDIA documents FP8 tensor cores as accumulating into FP32 registers. At long contraction dimensions the precision quietly evaporates — the same bug DeepSeek-V3 hit during training five months earlier."
blurb:
  - "NVIDIA docs: FP8 tensor cores accumulate into FP32 registers. Mostly true."
  - "At long contraction dimensions, the effective precision quietly evaporates."
  - "DeepSeek-V3 hit the same hardware quirk five months earlier, during training."
  - "The question nobody asked for three years: *\"Is the FP32 accumulator actually FP32?\"*"
```

```yaml
title: "Sliding Window Attention: why FP8 broke even at 741K tokens"
short_title: "Sliding Window Attention"
description: "gpt-oss-20b's FP8 break-even was 741,565 tokens — the flag was a no-op. Sliding-window layers have bounded caches; the fix was a per-layer skip flag."
blurb:
  - "gpt-oss-20b: FP8 slope was 96% of BF16. The bandwidth saving was essentially zero."
  - "Sliding-window layers have tiny, bounded caches — FP8 saves almost nothing on them."
  - "In hybrid attention models, these layers dominate the count."
  - "The fix: a per-layer skip flag drops the break-even from 741,565 tokens to 7,659."
```

---

## Failure Modes to Avoid

| Bad | Better |
|---|---|
| "explores," "examines," "looks at," "discusses" | Say what the article *actually does* or cut |
| "This post explains," "we'll cover," "join me as" | Cut completely |
| "comprehensive," "fascinating," "deep dive into the world of" | Cut adjectives; let content earn the reaction |
| Promising the reader's emotion ("mind-blowing," "you'll never see X the same way") | Let the content earn the reaction |
| Kitchen-sink title enumerating every section | Pick one hook |
| First bullet restates the title | The blurb is *additional* information |
| Vague title with no keyword anchor ("The Sum Is Not What You Think") | Start with the recognizable term |

---

## Cold-Open Articles

Cold-open articles (typically `01-cold-open.md`) don't fit the pure `Keyword: hook` pattern because they're narrative entrances, not standalone technical primers. For these:

- Use the issue's main topic as the keyword: `"FP8 KV Cache: 91% to 13%"`, `"LLM Quantization: OPT-175B in 2022"`.
- The hook can be the inciting incident or the first number that doesn't make sense.
- `short_title` should be the issue's main keyword (same as the comic book title keyword).

---

## Front Matter Reference

```yaml
title: "Keyword: hook question or statement"   # ≤ 10 words
short_title: "Keyword"                          # text before ":" — used in tech tree labels
description: "One clean sentence."             # for cards and SEO meta, no HTML
blurb:
  - "First bullet — one specific idea."
  - "Second bullet."
  - "Third bullet."
  - "Fourth bullet (optional)."
```

Full front matter spec: [`docs/issue-format.md`](issue-format.md).
