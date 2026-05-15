---
title: "One Pass Isn't Enough"
description: "April 14, 2025. OpenAI ships GPT-4.1 with a new benchmark whose pitch sentence is the cleanest reframe in long-context history: 'A model could solve MRCR by one pass through the prompt. GraphWalks cannot be solved sequentially.'"
topics: [evaluation, long-context, reasoning]
tags: [graphwalks, openai, gpt-4.1, mrcr-v2, oolong, multi-hop]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 80
techKind: mainline
techNode: needle-to-graph
header: 08-needle-to-graph.webp
---

## A Single Italicised Sentence

It is **Monday, April 14, 2025**, around 9 AM Pacific time. OpenAI announces the **GPT-4.1** model family. The blog post is, by frontier-launch standards, low-key — a developer-focused release with three SKUs, pricing tables, and a benchmarks section. Halfway down the page, in the long-context section, there is a paragraph that the long-context-evaluation community will spend the next twelve months quoting verbatim:

> *"In {{< wiki "long-context-benchmarks" >}}OpenAI-MRCR{{< /wiki >}}, the model must answer a question that involves disambiguating between 2, 4, or 8 user prompts scattered amongst distractors… The challenge arises from the similarity between these requests and the rest of the context — models can easily be misled by subtle differences, such as a short story about tapirs rather than a poem, or a poem about frogs instead of tapirs."*

That's the *first* paragraph. It explains why OpenAI is shipping their own implementation of MRCR — {{< wiki "vodrahalli" >}}Vodrahalli{{< /wiki >}}'s task from the Michelangelo paper — as a public dataset. Fair enough. Then comes the part that mattered:

> *"Many developer use cases for long context require multiple logical hops within the context, like jumping between multiple files when writing code or cross referencing documents when answering complicated legal questions. **A model (or even a human) could theoretically solve an OpenAI-MRCR problem by doing one pass or read-through of the prompt, but Graphwalks is designed to require reasoning across multiple positions in the context and cannot be solved sequentially.** Graphwalks fills the context window with a directed graph composed of hexadecimal hashes, and then asks the model to perform a breadth-first search (BFS) starting from a random node in the graph. We then ask it to return all nodes at a certain depth."*

{{% pullquote type="profound" author="OpenAI GPT-4.1 launch blog, April 14 2025" %}}
A model could theoretically solve MRCR by doing one pass through the prompt. **GraphWalks is designed to require reasoning across multiple positions in the context and cannot be solved sequentially.**
{{% /pullquote %}}

This is, in the entire long-context literature, **the cleanest articulation of why the field had to move on**. NIAH and even MRCR can be solved by a single linear scan. GraphWalks cannot. Whatever model architecture is going to claim "long context understanding" from this point forward had better be capable of *non-linear traversal through arbitrary information embedded throughout the prompt*.

The launch blog quietly inaugurates the third era of long-context benchmarking.

## What GraphWalks Is, Precisely

The construction is so clean it could be taught to a CS undergrad. Take a random directed graph of $n$ nodes, where each node id is a **128-bit hexadecimal hash** (something like `a3f9c1e0…`). Encode the graph as plaintext adjacency lists:

```
a3f9c1e0... -> 5b8d2c7e..., 7e4a1f3c..., d8b6e2a9...
5b8d2c7e... -> 9c4f6a2b..., e2d4c8f1...
7e4a1f3c... -> 1f8c5e3a...
...
```

Stuff this list into the model's prompt. Pad with distractor text if needed to hit the target context length. Then ask the model two kinds of question:

- **BFS task**: "Starting from node `a3f9c1e0…`, return all nodes reachable at depth exactly 2."
- **Parents task**: "Return all nodes that have node `9c4f6a2b…` as a direct child."

Both questions require **non-linear graph traversal**. The BFS variant needs to expand a frontier; the parents variant needs to scan *all* incoming edges. Neither can be done correctly by a single linear scan of the prompt.

Two design choices in this construction deserve attention. First, the **hex-hash node ids** are crucial. They have no semantic prior — the model cannot reach into its weights and "know" that `a3f9c1e0` is more likely to point to `5b8d2c7e` than to `7e4a1f3c`. The graph structure has to be learned *exclusively* from the prompt. Second, the **directed adjacency-list encoding** is a clean specification — there is no syntactic ambiguity about what the graph contains.

The result is a benchmark that *cannot be solved by retrieval alone*. To know the depth-2 children of node $X$, you have to first find $X$'s direct children $C_1, C_2, \ldots$ in the prompt, then for each $C_i$ scan the prompt again to find *its* children. That's a chained lookup — and as we showed in [Scan vs Think](../04-retrieval-vs-reasoning/), chained lookups are *exactly* the capability that single-pass attention cannot deliver.

## The Numbers, And What They Said

The launch reported GPT-4.1's GraphWalks score: **61.7%**, "matching the performance of o1 and beating GPT-4o handily" (which scored 41.7%). Notably, the larger **GPT-4.5** scored **72.3%** on the same eval — meaning GPT-4.1 actually *trailed* the more expensive model. This was an important early data point: GraphWalks is **scale-sensitive** in a way NIAH was not. It is a real signal of model capability, not a saturated check.

Anthropic adopted GraphWalks in the **Claude Sonnet 4.5 system card** in September 2025. Google adopted it for **Gemini 2.5 Pro** later that summer. By late 2025, GraphWalks was on every system card alongside MRCR v2.

```pyplot {id="graphwalks-progression" caption="GraphWalks scores across the post-April-2025 frontier. The benchmark separated models cleanly from the start (vs NIAH's instant saturation) and continued to separate them through 2026. Numbers are representative of system-card reporting; see each lab's published cards for exact methodology."}
models = [
    ("GPT-4o\n(pre-GraphWalks)",            41.7, '#FFD700'),
    ("GPT-4.1\n(Apr 2025)",                 61.7, '#FF007F'),
    ("GPT-4.5\n(May 2025)",                 72.3, '#FF007F'),
    ("Claude Sonnet 4.5\n(Sep 2025)",       66.3, '#FF8C00'),
    ("Claude Opus 4.6\n(Feb 2026)",         88.2, '#FF8C00'),
    ("Gemini 2.5 Pro\n(Oct 2025)",          73.5, '#00A8A8'),
    ("Gemini 3 Pro\n(Mar 2026)",            91.4, '#00A8A8'),
    ("Gemini 3 Pro Flash\n+thinking",       94.7, '#00A8A8'),
]

names  = [m[0] for m in models]
scores = [m[1] for m in models]
colors = [m[2] for m in models]

fig, ax = plt.subplots(figsize=(11, 4.6))
bars = ax.bar(range(len(models)), scores, color=colors,
              edgecolor='#1A1A1A', linewidth=1.5)
for i, (bar, s) in enumerate(zip(bars, scores)):
    ax.text(bar.get_x() + bar.get_width()/2, s + 1.2,
            f"{s:.1f}%", ha='center', fontweight='bold', fontsize=9)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
ax.set_ylabel("GraphWalks BFS score (%) at <128K context")
ax.set_title("GraphWalks <128K: a benchmark that actually separated frontier models, 2025-2026")
ax.set_ylim(0, 100)
ax.spines[['top', 'right']].set_visible(False)

# Saturation watch
ax.axhline(95, color='#1A1A1A', linewidth=0.5, linestyle=':')
ax.text(0, 95.5, 'saturation watch (>95%)', fontsize=8, style='italic')

print("Year-over-year improvement Apr 2025 → Mar 2026:")
print(f"  best closed model: 72.3% → 94.7%  (+22.4 pp on <128K BFS)")
print(f"At 256K and 1M, scores drop substantially.  The 1M variant is still open.")
```

Two observations from that chart. First, the benchmark *worked* — it separated GPT-4.1 from GPT-4.5, separated Claude Sonnet 4.5 from Claude Opus 4.6, separated direct-answer models from reasoning-enabled models. This is what a healthy Phase-I-or-II benchmark looks like (see [Goodhart's Ceiling](../06-measurement-saturation/) for the lifecycle taxonomy).

Second, by March 2026, the **<128K BFS variant is approaching saturation** — frontier reasoning models are above 90%. The **256K and 1M variants**, on the other hand, are nowhere close to saturated. The benchmark has *natural difficulty gradients* in its length dimension, and labs have been migrating to longer-context variants to keep the signal alive. This is the *length-conditional reporting* discipline that we predicted in the saturation primer.

## Why The Field Was Ready For It

GraphWalks did not appear in a vacuum. The benchmark itself is technically simple — a random directed graph in a prompt, a question about BFS. The **conceptual framing** — the bolded launch-blog sentence about "cannot be solved sequentially" — is what made it influential, and that framing rested on a year of community work:

- **{{< wiki "long-context-concepts" >}}Michelangelo's LSQ framework{{< /wiki >}}** (Sep 2024) gave the field the vocabulary to say "this task requires latent structure extraction, not retrieval."
- **RULER's variable tracing** (Apr 2024) had already shown that even simple chained references collapse most frontier models' effective windows.
- **The chain-of-thought reasoning models** (o1 in late 2024) had shown that emitting intermediate tokens could buy unbounded *hop depth*, suggesting that long-context tests should require it.

When OpenAI shipped GraphWalks alongside GPT-4.1, the community had already done the conceptual work. The launch blog wasn't proposing a new idea; it was *naming* one. From April 2025 forward, "GraphWalks-style benchmarks" became the operative phrase for the next generation.

## The Other 2025 Stress: OOLONG

GraphWalks is the **graph-traversal** corner of 2025's evaluation pivot. Its sibling, less famous but more methodologically interesting, came eight months later.

In **November 2025**, **Kiran Vodrahalli and colleagues at Google DeepMind** — the same group that had shipped Michelangelo a year earlier — published **OOLONG** (arXiv:2511.02817). OOLONG is, in a real sense, *Vodrahalli's own follow-up critique of MRCR*. The Michelangelo paper had built MRCR around *retrieving the right ordinal request*; OOLONG built tasks around **atomic analysis of each chunk plus aggregation across chunks**. Each item in the prompt has to be analysed in isolation (atomic), *and then* the answer requires combining many such analyses across the long context (aggregation).

The OOLONG paper's headline finding is sobering:

> *"Even frontier models struggle on OOLONG, with GPT-5, Claude-Sonnet-4, and Gemini-2.5-Pro all achieving less than 50% accuracy on both splits at 128K."*

Read that again. **GPT-5. Claude Sonnet 4. Gemini 2.5 Pro. All under 50% at 128K.** This is at a context length where GraphWalks BFS was approaching 90% on frontier reasoning models. The benchmarks are *not testing the same thing*, and labs cannot pick a single winner that works across both.

The conceptual move OOLONG makes is worth pausing on. **GraphWalks measures depth-of-reasoning** — can the model chain $k$ dependent hops without losing the thread? **OOLONG measures width-of-aggregation** — can the model perform $n$ independent atomic analyses and then *combine* their results without dropping any? These are orthogonal capabilities. A model can be strong on one and weak on the other, and the 2025–2026 reporting practice now treats them as distinct measurements.

We unpack OOLONG's mechanics — the two evaluation splits, the D&D transcript angle, and why all frontier models fall below 50% at 128K — in the dedicated primer [The Tallying Problem](../18-oolong/). For the conceptual framework that connects MRCR and OOLONG, see [Vodrahalli's Chisel](../09-latent-structure/).

## What Changed In Anthropic's System Cards

Watching Anthropic's system cards across the 2024–2026 arc tells the long-context evaluation story with unusual clarity. A short tour:

- **Claude 3 (March 2024)**: NIAH only. Anthropic boasted Claude 3 Opus *"surpassing 99% accuracy"* and *"in some cases recognising that the 'needle' sentence appeared to be artificially inserted."*
- **Claude 3.5 Sonnet (June 2024)**: Still NIAH-centric, with a custom 30-needle variant.
- **Claude 4 / Opus 4 (May 2025)**: 213-page system card devoted most of its attention to safety/agentic/CBRN. Long-context evaluation appears in a smaller role; NIAH dropped to a sanity check, MRCR adopted.
- **Claude Sonnet 4.5 (Sep 2025)** / **Opus 4.5 (Nov 2025)**: **OpenAI MRCR v2** and **GraphWalks** become the core long-context capability evals. Sonnet 4.5's MRCR v2 score at 1M is **18.5%** — a deliberate humbling baseline.
- **Claude Opus 4.6 (Feb 2026)**: Section 2.18 "Long Context" makes the philosophy explicit. On MRCR v2: *"Unlike simpler 'needle in a haystack' tests, MRCR challenges models to identify the correct ordinal instance among identical requests…"* That clause — *"unlike simpler 'needle in a haystack' tests"* — is the most pointed sentence in any Claude system card. It is Anthropic stating, in their flagship release document, that NIAH-style retrieval is **not** what they consider long-context evaluation anymore.

{{< crosshead >}}NIAH Boast → NIAH Footnote → NIAH Gone{{< /crosshead >}}

That sequence — NIAH boast → NIAH sanity check → NIAH demoted → NIAH dismissed — is the four-step shorthand for the entire third-era shift. **Anthropic did not stop *reporting* MRCR or NIAH.** They continued to run them, often for cross-lab continuity. What changed was the *narrative weight*: GraphWalks and OOLONG-style evals are now the headline numbers. The retrieval tests live in the appendix.

## The MRCR v2 Story Beside GraphWalks

Worth a footnote on MRCR v2 specifically, because it remains the most-cited long-context benchmark in 2026 alongside GraphWalks. A full treatment of the benchmark's mechanics — the 9-cell grid, the random string verification trick, MRCR v1 vs v2, and a model-by-model comparison — is in [The Fourth Poem](../17-mrcr/).

OpenAI's GPT-4.1 launch shipped **OpenAI-MRCR** as a public dataset — a cleaned-up implementation of Vodrahalli's MRCR. In December 2025, OpenAI shipped **MRCR v2**, fixing several methodology issues and tightening the disambiguation criteria. The needles-per-prompt variants (2-needle, 4-needle, 8-needle) and the context-length variants (128K, 256K, 1M) define a 9-cell grid that has become the *standard* retrieval-side report.

The most cited entry in that grid in 2025–2026 has been **MRCR v2 8-needle 1M** — the hardest variant. Claude Sonnet 4.5 scored 18.5%; Claude Opus 4.6 scored 76% (the 4× jump that becomes the subject of [The 4× Jump](../11-context-rot-fix/)); Gemini 3 Pro scored 26.3%; Claude Opus 4.7 *regressed* to 32.2%, prompting Anthropic to recommend "use Opus 4.6 for RAG-shaped workloads, Opus 4.7 for code" in their April 2026 release notes.

This kind of *per-task, per-length* recommendation in production release notes — "use this version of our model for this kind of workload" — is itself a 2026 cultural shift. It implicitly acknowledges that **the long-context capability surface is no longer one-dimensional**. Different tasks stress different parts of the model, and the best model on one task is not the best on another.

## The Bigger Picture

Step back. What did the GraphWalks moment of April 2025 *do* to the long-context evaluation landscape?

1. **It gave the field a benchmark that does not saturate quickly.** Two years later, the <128K BFS variant is approaching saturation on reasoning models, but the 256K and 1M variants are still wide open.
2. **It legitimised the 'multi-hop' framing.** Before GraphWalks, "multi-hop reasoning" was an academic phrase. After GraphWalks, every system card has a "multi-hop" subsection.
3. **It demanded scale-sensitive evaluation.** GraphWalks separates GPT-4o from GPT-4.1 from GPT-4.5. NIAH did not. A working benchmark *must* separate models that differ in genuine capability.
4. **It paired with OOLONG to define a 2D measurement plane.** Depth (GraphWalks) and width (OOLONG) became the two axes against which serious 2026 long-context evaluation is reported.

The deeper meaning, the one James Burke would deliver as a closing voiceover: **what changed in April 2025 was not a benchmark. It was the field's definition of what a benchmark is for.** From 2023's NIAH (*"can the model find this thing"*) to 2024's saturation crisis (*"finding things is not understanding"*) to 2025's GraphWalks (*"understanding requires non-linear traversal"*) to 2026's eval stack (*"every workload type needs its own measurement"*). Each step was a *cultural* refinement, not just a methodological one.

## What To Remember

1. **April 14, 2025**: OpenAI ships GPT-4.1 with **GraphWalks** and **OpenAI-MRCR** as public datasets. The launch blog's framing sentence — *"MRCR can be solved by one pass; GraphWalks cannot"* — is the cleanest reframe of long-context evaluation in the literature.
2. **GraphWalks's construction**: a random directed graph of hex-hash nodes embedded in the prompt, with a BFS query that requires non-linear traversal. Two variants: BFS (depth-$k$ reachability) and Parents (incoming-edge enumeration).
3. **The benchmark separates models**: GPT-4o 41.7% → GPT-4.1 61.7% → GPT-4.5 72.3% → Claude Opus 4.6 88% → Gemini 3 Pro Flash + thinking 94.7%. By 2026, <128K is saturating; 256K and 1M variants remain open.
4. **November 2025: OOLONG** (Vodrahalli et al.) — Michelangelo's authors' own follow-up. Atomic chunk analysis plus aggregation across chunks. **GPT-5, Claude Sonnet 4, Gemini 2.5 Pro all under 50% at 128K.** Different capability axis from GraphWalks.
5. **GraphWalks measures *depth* of reasoning; OOLONG measures *width* of aggregation.** Orthogonal capabilities. A frontier model can be strong on one, weak on the other.
6. **Anthropic's system cards across 2024–2026 trace the shift from NIAH-headline → MRCR-headline → GraphWalks-headline.** They never stopped *reporting* the older tests; they *demoted* them in narrative weight.

**Continue to** → [Vodrahalli's Chisel](../09-latent-structure/) — the conceptual scaffolding that made GraphWalks and OOLONG possible. The Michelangelo *Latent Structure Queries* framework, MRCR as the canonical LSQ instance, and the through-line to OOLONG.
