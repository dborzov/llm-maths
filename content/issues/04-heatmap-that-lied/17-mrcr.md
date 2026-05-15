---
title: "The Fourth Poem"
description: "A long conversation. Eight similar requests scattered through 200,000 tokens. The question: reproduce the fourth poem about tapirs. MRCR — the benchmark that defined 'retrieval with bookkeeping' and became long-context evaluation's de facto retrieval yardstick."
topics: [evaluation, long-context, methodology]
tags: [mrcr, mrcr-v2, vodrahalli, michelangelo, openai-mrcr, long-context-benchmarks, retrieval, ordinal]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 95
techKind: primer
techNode: mrcr
header: 17-mrcr.webp
---

## Autumn 2024, Mountain View

**Kiran Vodrahalli** is designing what will become Michelangelo's best-known task, and he is trying to articulate something that feels obvious once you name it but is surprisingly hard to specify precisely.

NIAH is broken. The community knows it. A model retrieving the phrase "a sandwich in Dolores Park" from 200,000 tokens of Paul Graham essays has proven — what exactly? That it can find *one fact* in a large context. One needle. Dropped in. Find it.

The problem is that production use cases rarely ask for *one* needle from a context that contains *one* needle. They ask for *the relevant one* from a context that contains *many similar things*. A customer-service model fielding a long conversation has to track which billing issue came first, which technical complaint was escalated, which product request the user mentioned in passing six hours ago. There is no unique needle. There are many needles that look alike, and the model has to count which one is the fourth.

Vodrahalli's insight: **the hard part of long-context retrieval isn't distance — it's disambiguation under ordinal constraint**. This leads to {{< wiki "long-context-benchmarks" >}}MRCR{{< /wiki >}}: Multi-Round Co-reference Resolution.

## The Scenario

The MRCR prompt is a long naturalistic dialogue — a realistic-looking conversation between a fictional user and assistant, spanning tens of thousands of tokens, padded with unrelated filler. Embedded in this dialogue are **eight similar requests** from the user, each followed by an assistant response:

```
user: write me a short story about a parakeet
assistant: <story, ~300 words>

[thousands of tokens of unrelated conversation]

user: now write a poem about frogs
assistant: <poem, ~100 words>

[thousands more tokens]

user: write a story about tapirs
assistant: <story, ~300 words>

[thousands more tokens]

user: write a poem about tapirs
assistant: <poem, ~100 words>

[thousands more tokens]

user: another story, this time about cats
assistant: <story, ~300 words>

[…and so on through eight total requests…]
```

The eight requests span two **forms** (poem, short story) and several **topics** (parakeet, frog, tapir, cat, elephant, hedgehog…). Topics repeat across forms deliberately: tapirs appear as both a story topic *and* a poem topic. The final question, appended after all the padding:

> *"Reproduce verbatim the second poem about tapirs."*

Or: *"the fourth story"*. Or: *"the third request that mentioned an animal beginning with T."* The exact query changes per instance, but the structure is always the same: **find the $k$-th item matching (form, topic)**.

To answer correctly, the model must:

1. **Scan** the entire dialogue for items matching the target form (poem, not story).
2. **Filter** by topic (tapirs — not frogs, not elephants, not parakeets; even though tapirs share many tokens with "story about tapirs" vs "poem about tapirs").
3. **Count** within the matching subset (first poem about tapirs, second poem about tapirs — stop at $k$).
4. **Reproduce** the verbatim content at that ordinal position.

Any of the four steps failing produces a wrong answer. A model that retrieves the *first* tapir poem instead of the *second* fails on step 3. A model that grabs a tapir *story* instead of a tapir *poem* fails on step 2. A model that returns a poem about elephants fails on step 1. The test is specifically designed so that all common failure modes produce a plausible-looking wrong answer — which is what makes it hard.

{{% callout type="definition" title="MRCR: Multi-Round Co-reference Resolution" %}}
**MRCR** presents the model with a long dialogue containing $N$ similar user requests (8 in the hard variant). Each request varies form (poem vs story) and topic (tapirs, frogs, elephants…). The task: reproduce the *k*-th item matching a specified (form, topic) pair. Requires disambiguation, ordinal counting within the matching subset, and verbatim reproduction.
{{% /callout %}}

## The Verification Trick

Running MRCR at 1M tokens across hundreds of instances is expensive. Prefill costs scale with context length — at 1M tokens per instance, even a single evaluation run is a significant budget item. Vodrahalli's engineering solution is elegant: **random string verification**.

Before generating the retrieved text, the model is instructed to first output a specific random string — something like `7f3a9c2d` — that appears nowhere in the context. This does two things:

**Commitment signal.** The random string forces the model to "stake a claim" before outputting content. A model that outputs the wrong random string has already failed before producing any poem text, making scoring fast.

**Prefix caching.** Because all instances that share the same long dialogue haystack only differ in the final query ("second poem about tapirs" vs "third poem about tapirs"), the entire haystack can be **KV-cached once** and reused across all $k$-variant queries. The marginal cost of the second, third, and fourth question about the same dialogue is near-zero once the cache is warm. At 1M tokens, this turns an 8× prefill (8 questions × 1M tokens) into approximately 1× (one cache fill) plus 8 very short completions. Without prefix caching, large-scale MRCR evaluation at 1M context would be financially prohibitive for most research groups.

The random string is not magic — it's a pragmatic device to make systematic long-context evaluation economically feasible.

## MRCR v1 and v2

The original **MRCR v1** appeared in the Michelangelo paper (Vodrahalli et al., September 2024, arXiv:2409.12640). It spread quickly because it was the cleanest available test for retrieval-with-bookkeeping, and because the Google DeepMind team made the evaluation framework openly available.

When OpenAI launched GPT-4.1 in April 2025, they released their own cleaned implementation as a public dataset: **OpenAI-MRCR**. The public release exposed several methodology gaps in v1: the F1 scoring function misbehaved on edge cases where the ground truth was empty (an instance where the requested item didn't actually exist in the context). Disambiguation criteria were inconsistently applied across instances. Some generated content straddled "poem" and "story" and could be scored either way.

**MRCR v2**, shipped by OpenAI on December 5, 2025, addressed all of these:

- **F1 for empty ground truth**: correctly scores confident wrong outputs when the answer is "not present in context."
- **Tighter disambiguation criteria**: form and topic labels applied rigorously during data generation.
- **Cleaner haystack generation**: dialogue chunks of more uniform length, removing accidental artifacts from v1.

{{% marginnote %}}Anthropic's Opus 4.6 system card notes: *"this result is not reproducible via the public API, as half the problems exceed its 1M token limit."* MRCR v2 1M results are evaluated with special API access; cross-lab comparisons should account for potential methodology drift.{{% /marginnote %}}

Every number in this article refers to **MRCR v2** unless noted. The v1-to-v2 shift is large enough that cross-version comparisons are unreliable.

## The 9-Cell Grid

MRCR v2's standard evaluation is a **3 × 3 grid**: three needle-count variants crossed with three context lengths.

| | 2-needle | 4-needle | 8-needle |
|---|---|---|---|
| **128K** | easy | medium | hard |
| **256K** | medium | hard | very hard |
| **1M** | hard | very hard | **hardest** |

The **8-needle, 1M** cell — eight similar requests scattered across a million-token context, asking for the $k$-th of them — is the standard headline number. When a system card reports "MRCR score," they almost always mean this cell.

```pyplot {id="mrcr-9cell-grid" caption="The MRCR v2 9-cell grid. Each cell is one benchmark configuration; values are representative best-frontier performance in early 2026. The 8-needle 1M bottom-right corner is the headline cell. Difficulty degrades cleanly in both dimensions: longer context and more needles each independently cost accuracy."}
needle_labels = ['2-needle', '4-needle', '8-needle']
ctx_labels    = ['128K', '256K', '1M']

# Representative 2026 best-frontier-model scores per cell
scores = np.array([
    [92, 84, 76],   # 128K row
    [82, 71, 54],   # 256K row
    [81, 62, 42],   # 1M row (the hard row)
])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
im = ax.imshow(scores, aspect='auto', cmap='RdYlGn', vmin=10, vmax=100, origin='upper')
ax.set_xticks(range(3))
ax.set_xticklabels(needle_labels, fontsize=10)
ax.set_yticks(range(3))
ax.set_yticklabels(ctx_labels, fontsize=10)
ax.set_title('MRCR v2: best-frontier accuracy (%) per cell, early 2026')
ax.set_xlabel('difficulty (needles in prompt)')
ax.set_ylabel('context length')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{scores[i,j]}%", ha='center', va='center',
                fontweight='bold', fontsize=12, color='#1A1A1A')
plt.colorbar(im, ax=ax, label='accuracy (%)')
ax.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor='#FF007F', lw=3))
ax.text(2, 2.7, '← headline cell', fontsize=8, ha='center', color='#FF007F')

ax2 = axes[1]
hard_col = scores[:, 2]
bar_colors = ['#FFD700', '#FF8C00', '#FF007F']
bars = ax2.bar(ctx_labels, hard_col, color=bar_colors, edgecolor='#1A1A1A', linewidth=1.4)
for bar, val in zip(bars, hard_col):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5,
             f"{val}%", ha='center', fontweight='bold', fontsize=12)
ax2.set_title('8-needle column: how performance falls with context length')
ax2.set_ylabel('accuracy (%)')
ax2.set_ylim(0, 100)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

print("The 8-needle column (hardest per context length) in early 2026:")
print("  128K: ~76%   (two consecutive frontier models differ by ~20 pp)")
print("  256K: ~54%   (below-human-expert threshold for most tasks)")
print("  1M  : ~42%   (Claude Opus 4.6 is the only model above 70%)")
print("MRCR v2 headline cell = 8-needle, 1M.")
```

The gradient across the grid is clean. Context length and needle count each degrade performance roughly independently — doubling context costs about as much as doubling needle count. This regularity is part of what makes MRCR useful as a calibration tool: you can read a model's performance across the grid and infer where its effective context window actually ends.

## The Lab Numbers

The most striking entry in the MRCR record is the gap between two consecutive Anthropic models.

```pyplot {id="mrcr-lab-scores" caption="MRCR v2 8-needle 1M scores across the post-2025 frontier. This is the headline cell. The Sonnet 4.5 → Opus 4.6 jump is the largest single-version improvement ever recorded on this benchmark. The Opus 4.7 regression is real — Anthropic's own release notes confirm it."}
models = [
    ("Claude Sonnet 4.5\nSep 2025",  18.5, '#FF8C00'),
    ("GPT-4.1\nApr 2025",            61.7, '#FF007F'),
    ("Gemini 3 Pro\nMar 2026",       26.3, '#00A8A8'),
    ("Claude Opus 4.6\nFeb 2026",    76.0, '#FF8C00'),
    ("Claude Opus 4.7\nApr 2026",    32.2, '#FF8C00'),
]

names  = [m[0] for m in models]
scores = [m[1] for m in models]
colors = [m[2] for m in models]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(range(len(models)), scores, color=colors,
              edgecolor='#1A1A1A', linewidth=1.5)
for i, (bar, s) in enumerate(zip(bars, scores)):
    ax.text(bar.get_x() + bar.get_width()/2, s + 1.5,
            f"{s:.1f}%", ha='center', fontweight='bold', fontsize=11)

ax.set_xticks(range(len(models)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("MRCR v2 8-needle 1M accuracy (%)")
ax.set_title("MRCR v2 (8-needle, 1M): the headline retrieval cell, 2025–2026")
ax.set_ylim(0, 90)
ax.spines[['top', 'right']].set_visible(False)

# Annotate the Opus 4.6→4.7 regression
ax.annotate("",
    xy=(4, 32.2 + 1), xytext=(3, 76.0 - 1),
    arrowprops=dict(arrowstyle="->", color='#1A1A1A', lw=1.8, linestyle='dashed'))
ax.text(3.6, 55, "regression:\n4.7 < 4.6\nat 1M", fontsize=8, ha='center',
        style='italic', color='#1A1A1A')

print("Claude Sonnet 4.5 → Opus 4.6: +57.5 pp on 8-needle 1M  (the '4x jump')")
print("Claude Opus 4.6  → Opus 4.7 : -43.8 pp on 8-needle 1M  (the regression)")
print("Anthropic Apr 2026 release notes: 'use Opus 4.6 for RAG-shaped workloads.'")
```

**The Opus 4.7 regression.** Claude Opus 4.7, released April 2026, scored 32.2% on MRCR v2 8-needle 1M — down from Opus 4.6's 76%. Anthropic acknowledged this in their release notes and explicitly recommended *"use Opus 4.6 for RAG-shaped workloads, Opus 4.7 for code and reasoning tasks."* The regression is believed to reflect a training-objective shift toward code and extended reasoning, which displaced some of the long-context bookkeeping capability Opus 4.6 had specifically optimised. **Model capability is not monotone across generations.** A newer model number does not guarantee better MRCR performance.

**The Gemini cliff.** Gemini 3 Pro scores strongly on MRCR v2 at 128K, but at 1M falls to 26.3% — a 50-point drop across a context extension of 7.8×. The 1M context window exists; whether it is *usable* is the question MRCR answers. Gemini 3 Pro's answer at 1M is: not reliably.

The Opus 4.7 regression and the Gemini cliff together make the same point: **MRCR scores should always specify context length**. A quoted number without a length qualifier is essentially uninterpretable.

## Single-Pass Solvability — The Architectural Point

[Vodrahalli's Chisel](../09-latent-structure/) introduced the conceptual point; this is the place to make it precise.

MRCR is **single-pass solvable**. Here is a concrete algorithm that solves any MRCR instance deterministically:

```python
def solve_mrcr(dialogue, form, topic, k):
    count = 0
    for turn in dialogue:
        if turn["role"] == "assistant":
            if matches_form(turn["content"], form) and matches_topic(turn["content"], topic):
                count += 1
                if count == k:
                    return turn["content"]
    return "NOT_FOUND"
```

One linear scan. One counter. No backtracking. A human speed-reading the conversation can do this with a finger, a tally mark, and a highlighter. No graph traversal required; no visiting a position in the context twice.

This is the precise statement in OpenAI's GPT-4.1 launch blog: *"A model (or even a human) could theoretically solve an OpenAI-MRCR problem by doing one pass or read-through of the prompt."* GraphWalks cannot be solved this way — it requires revisiting multiple positions in the prompt non-linearly (read node $X$'s children, then jump to wherever each child is to read *their* children). MRCR does not need the jump.

{{% pullquote type="profound" %}}
MRCR's hard part is not logic — it is **memory**. A running count over millions of tokens, updated correctly with each match, read off at the right position. A counter, held in attention across a very long context.
{{% /pullquote %}}

**What this means architecturally.** A transformer processing a 1M-token context "sees" all positions in a single forward pass, so in principle it can implement the linear scan in one go. The question is whether its attention mechanism has effectively learned to maintain a counter — "three tapir poems seen so far, waiting for the fourth" — across millions of tokens without losing it. Empirically: Claude Sonnet 4.5, at 18.5% on the hard cell, has not. Claude Opus 4.6, at 76%, largely has. The counter is learnable, but it requires specific training incentives to acquire robustly at long context.

## What MRCR Measures — and What It Does Not

MRCR measures **retrieval with bookkeeping**: the ability to track many similar items across a long context and retrieve one by ordinal index. It is the right benchmark if your application involves:

- Finding the $k$-th version of a document in a long edit history.
- Retrieving the $n$-th occurrence of a log pattern across a large log file.
- Answering "what did the third customer review say about shipping time?"

MRCR does *not* measure:

- **Multi-hop reasoning.** Following a chain of logical dependencies between facts in different parts of the context. That requires GraphWalks ([One Pass Isn't Enough](../08-needle-to-graph/)).
- **Aggregation.** Tallying many independent facts and computing a statistic across all of them. That requires OOLONG ([The Tallying Problem](../18-oolong/)).
- **Absence detection.** Noticing what's missing from an expected set. AbsenceBench.
- **Realistic task performance.** Correlation with production RAG quality. Requires HELMET, LongBench v2, or domain-specific suites.

The 2026 practitioner model treats MRCR as the **sanity-check tier** for retrieval capacity. If a model fails MRCR v2 2-needle at 128K, its context window is not functional for any retrieval use. If it passes 2-needle at 128K but fails 8-needle at 1M, you know the upper bound of its useful window. If it passes 8-needle at 1M (above ~70%), move up to GraphWalks and OOLONG to probe the harder capabilities. Each tier tells you something different.

## What To Remember

1. **MRCR asks**: in a long conversation with 8 similar requests, reproduce the *k*-th request of type (form, topic). The challenge is disambiguation, ordinal counting, and verbatim recall — simultaneously.
2. **Random string verification** + prefix caching makes large-scale 1M-token evaluation financially feasible.
3. **MRCR v2** (OpenAI, December 2025) tightened scoring methodology and disambiguation criteria. All comparisons must use the same version.
4. **The 9-cell grid** (2/4/8 needles × 128K/256K/1M) is the standard surface; the 8-needle 1M cell is the headline number.
5. **Claude Sonnet 4.5: 18.5%; Claude Opus 4.6: 76%.** The 4× jump story is in [The 4× Jump](../11-context-rot-fix/). Claude Opus 4.7 *regressed* to 32.2%; use Opus 4.6 for RAG workloads.
6. **MRCR is single-pass solvable**: a linear scan with a counter suffices. This is its key distinction from GraphWalks (non-linear traversal required).
7. **MRCR is retrieval-with-bookkeeping**, not reasoning or aggregation.

**Continue to** → [The Tallying Problem](../18-oolong/) — the capability MRCR does not probe. OOLONG asks not "which one is the fourth?" but "how many are there altogether?" — and every frontier model in 2025 fails at 128K.
