---
title: "The Chroma Measurement"
description: "July 14, 2025. A small RAG company called Chroma quietly publishes a 90-page report measuring 18 frontier models. The phrase 'context rot' enters the literature with a number attached, and the long-context conversation will never sound the same."
topics: [evaluation, long-context, methodology]
tags: [chroma, context-rot, nolima, absencebench, oolong, hong-2025]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 120
techKind: primer
techNode: context-rot
header: default.webp
---

## A Quiet Report From An Unexpected Lab

It is **July 14, 2025**, a Monday. **Chroma** — a small RAG-database startup, of all places — quietly publishes a report titled *"Context Rot: How Increasing Input Tokens Impacts LLM Performance"*. The authors are **Kelly Hong**, **Anton Troynikov**, and **Jeff Huber**. The report is 90 pages long, methodologically careful, and (uncommonly for the era) released as a website rather than an arXiv paper. It tests **18 frontier models** — GPT-4.1, Claude 4, Gemini 2.5, Qwen3, and 14 others — across multiple long-context tasks at lengths from 1K to 1M tokens.

The Chroma team's framing is, at the start, quite specific. *"Our results reveal that models do not use their context uniformly; instead, their performance grows increasingly unreliable as input length grows."* The phrase that catches on, the one that ends up in every long-context system card from late 2025 onward:

> **"Context rot"**

Within two weeks, AI Twitter is using the phrase as if it were a CS-101 term. Within two months, Anthropic's Opus 4.5 release blog cites the Chroma report. Within six months, Anthropic's Opus 4.6 launch blog adopts "context rot" as a *phenomenon to be fixed* (covered in [The 4× Jump](../11-context-rot-fix/)). A whole vocabulary has emerged out of one small report from a database company.

This primer is about what the Chroma study actually measured, why it stuck, and how its findings connect to the larger 2025 stress-test ecosystem (NoLiMa, AbsenceBench, OOLONG).

## The Chroma Construction

What the Chroma team did, mechanically, is something nobody else had done at that scale. They took **{{< wiki "long-context-benchmarks" >}}NIAH{{< /wiki >}}**, **NoLiMa** (a no-token-overlap variant), and an MRCR-style focused retrieval task, and they ran them **across 18 frontier models** at **8 context lengths** from 1K to 1M tokens — with the *position of the needle controlled and reported separately*. The result is a multi-dimensional measurement grid: (model × benchmark × context length × needle position). The point is not the headline number but the *shape* of the surface.

Three findings dominated the report.

### Finding 1 — Degradation Is Universal And Non-Linear

For every model, on every test, accuracy degraded as input length grew. Not gracefully — non-linearly. The shape is consistent: a slow decline up to ~25% of the advertised window, then a sharp drop somewhere between 50% and 75% of the window, then often a partial recovery very near the maximum length (a kind of recency reflex).

```pyplot {id="chroma-degradation-shape" caption="The canonical 'context rot' curve from the Chroma report, schematically reproduced. Each line is one frontier model on a single task; the x-axis is fraction of advertised window. Three patterns dominate: a slow decline near 0, a steep drop near 0.5-0.75, and partial recovery at the very end (recency reflex)."}
positions = np.linspace(0.02, 1.0, 30)

def context_rot_curve(p, plateau=0.95, knee=0.5, drop=0.5, recovery=0.15):
    # plateau early, drop after the knee, partial recovery at the end
    early = plateau * (1 - 0.1 * p)
    mid_drop = drop / (1 + np.exp(-20 * (p - knee)))
    recency = recovery * np.exp(-((p - 1.0) * 10) ** 2)
    return np.clip(early - mid_drop + recency, 0.05, 1.0)

models = {
    "GPT-4.1":         (0.96, 0.55, 0.45, 0.12),
    "Claude 4.5":      (0.97, 0.62, 0.30, 0.18),
    "Gemini 2.5 Pro":  (0.95, 0.65, 0.35, 0.20),
    "Qwen3-72B":       (0.92, 0.48, 0.55, 0.10),
    "Llama-3.1-405B":  (0.93, 0.45, 0.60, 0.08),
}
colors = ['#FF007F', '#00A8A8', '#FFD700', '#FF8C00', '#1A1A1A']

fig, ax = plt.subplots(figsize=(9, 4.3))
for (name, params), c in zip(models.items(), colors):
    y = context_rot_curve(positions, *params)
    ax.plot(positions, y, marker='o', linewidth=1.8, color=c, label=name,
            markeredgecolor='#1A1A1A', markersize=4)

ax.set_xlabel("position in context (fraction of advertised window)")
ax.set_ylabel("retrieval accuracy")
ax.set_title("'Context rot' - degradation curves on Chroma's stress tests")
ax.set_ylim(0.2, 1.05)
ax.legend(loc='lower left', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.15)

# Annotate the regions
ax.axvspan(0.0, 0.25, alpha=0.05, color='#00A8A8')
ax.axvspan(0.5, 0.75, alpha=0.05, color='#FF007F')
ax.axvspan(0.9, 1.0, alpha=0.05, color='#FFD700')
ax.text(0.12, 0.25, "plateau", fontsize=8, ha='center', alpha=0.7)
ax.text(0.62, 0.25, "drop zone", fontsize=8, ha='center', alpha=0.7)
ax.text(0.95, 0.25, "recency\nreflex", fontsize=8, ha='center', alpha=0.7)

print("Chroma's universal observation:")
print("  Plateau (0-25% of window):     models hold ~95% accuracy")
print("  Drop zone (50-75% of window):  most models fall 20-50 pp")
print("  Recency reflex (>90% window):  partial recovery to ~50-70%")
print("  -> the 'rot' is a *band* in the middle-late context, not a uniform decline.")
```

This is a *generalisation* of the U-curve from [The U-Curve](../07-lost-in-the-middle/). The Liu et al. 2023 study found a smile in retrieval accuracy across position. Chroma 2025 found the same smile, but **measured the smile's shape on every frontier model** and showed the **shape varies systematically between labs**. Anthropic models, in 2025, had a deeper recency recovery (the right side of the U was higher). OpenAI models had a steeper drop zone. Qwen models had the flattest drop but the lowest plateau. **The U is everyone's, but its parameters fingerprint each lab.**

### Finding 2 — Anthropic Abstains; OpenAI Confabulates

The second Chroma finding became a meme: when context rot kicks in, **models from different labs fail in characteristically different ways**.

- **Anthropic models** (Claude 3.5, 4.0, 4.5) tend to *abstain* — they say "I don't know" or refuse to commit to an answer. Hallucination rate goes *down* under context rot for Anthropic.
- **OpenAI models** (GPT-4, 4.1, 4.5) tend to *confabulate* — they produce a confident-sounding wrong answer. Hallucination rate goes *up*.
- **Google's Gemini family** sits between the two, leaning toward confabulation but with higher abstention rates than OpenAI.

This is a *training-objective* fingerprint. Anthropic's safety/honesty pipeline penalises overconfident wrong answers more than abstentions; the model has internalised "if you're not sure, abstain." OpenAI's pipeline penalises both wrong answers and unhelpful abstentions, with weights that produce more confident-but-wrong outputs at long context. **Same architecture family, different training values, different failure mode.**

For practitioners this is hugely consequential. A RAG pipeline built around an Anthropic model is more likely to fail *silently* (with an abstention or a "this context does not support an answer"); a pipeline built around an OpenAI model is more likely to fail *loudly* (with a confident hallucination). You design your safety net differently depending on which failure mode dominates.

### Finding 3 — The Test Is Easier Than It Looks On Paper

Chroma's most subtle finding, and the one most underplayed in secondary reporting: **even on the easiest possible long-context test (NIAH-style retrieval with surface-token overlap), models degrade significantly at moderate context lengths.** It is not that models fail at "complex reasoning at 1M tokens." They fail at *simple retrieval at 100K tokens*.

This is in some ways more damning than the 2024 multi-hop critiques. The whole point of NIAH was that retrieval was the *easy* test. If frontier models can't reliably retrieve a sentence at 100K, then the marketing claim "200K context window" is straightforwardly misleading — long before you start asking about reasoning.

## NoLiMa — The Token-Overlap Trick

Worth a short detour on **NoLiMa** (Modarressi et al., February 2025), because it appears alongside NIAH in the Chroma stress suite.

NoLiMa stands for **"No Literal Match"**. The construction takes the standard NIAH-style needle-in-a-haystack setup and adds one constraint: **the question and the needle share *no* literal token overlap**. The model can't find the needle by string-matching the query against the haystack; it has to do *semantic* retrieval.

For example: the needle might be *"Maya purchased a violin from a luthier in Cremona,"* and the question might be *"Who acquired a stringed instrument abroad?"* No shared tokens. The model has to perform semantic equivalence — *purchased ≈ acquired*, *violin ≈ stringed instrument*, *Cremona ≈ abroad* — to recognise the needle as the answer.

NoLiMa's finding was striking: **every frontier model lost 20-50 percentage points** when token overlap was removed. The biggest drops happened at long context — at 32K, GPT-4 lost 30 pp on NoLiMa vs NIAH; at 128K, 50 pp. **A significant fraction of NIAH's apparent success at long context was actually surface-token matching, not semantic retrieval.**

This finding combines with the Chroma report's broader degradation curves to paint a sobering picture: a frontier model claiming 99% on NIAH at 128K may, with the surface-token shortcut removed, score 50% on the same task. The benchmark wasn't measuring "long-context retrieval"; it was measuring "long-context *surface-token* retrieval."

## AbsenceBench — Reversing NIAH

A different angle from the same era. **AbsenceBench** (Fu et al., 2025) inverts the NIAH construction: instead of inserting a needle and asking the model to find it, AbsenceBench *removes* one item from a known list embedded in the prompt, and asks the model to identify what's missing.

For example: present the model with a long list of 50 specific named entities (cities, products, people), then ask "what's missing from the standard list of 51?" The latent structure is *the gap* in an ordered set.

Frontier models perform astonishingly badly on AbsenceBench. The reason connects to the Michelangelo IDK task ([Vodrahalli's Chisel](../09-latent-structure/)): **detecting absence requires a different mode of {{< wiki "attention" >}}attention{{< /wiki >}} than detecting presence.** A NIAH-style retrieval lets attention concentrate probability mass on a single match; an absence query requires comparing the input set to an expected set and *noticing the diff*. The transformer is not natively built for this. Even at moderate context lengths, AbsenceBench scores trail NIAH scores by 30-50 pp.

This is, again, evidence that *the long-context capability surface is multi-dimensional*. A model that aces NIAH may fail AbsenceBench. A model that aces both may fail OOLONG. There is no single "long-context ability" anymore.

## OOLONG — The Aggregation Stress

We've covered OOLONG in detail in [One Pass Isn't Enough](../08-needle-to-graph/) and [Vodrahalli's Chisel](../09-latent-structure/). It deserves a brief reappearance here because **OOLONG is the standard 2026 test for the *width* axis of context rot**.

Recall the OOLONG construction: each chunk in the prompt encodes a small atomic fact (a number, a category, a sentiment). The question requires per-chunk classification *plus* aggregation across chunks ("how many tickets are about billing with sentiment < 3?"). This stresses *working memory across the context*, not depth of reasoning.

Chroma did not run OOLONG (it came out four months after their report), but the {{< wiki "vodrahalli" >}}Vodrahalli{{< /wiki >}} et al. OOLONG paper is in some sense a *spiritual successor* to the Chroma study. Both ask: when does the model start dropping pieces? Chroma measures *retrieval rot*. OOLONG measures *aggregation rot*. Together they bracket the practical limit of long-context capability in 2026.

The OOLONG headline number: **GPT-5, Claude Sonnet 4, Gemini 2.5 Pro all under 50% at 128K.** This number anchors the practitioner's mental model: even when the model can find each fact, it cannot reliably combine many facts. The capability surface has a width axis that is much worse than the depth axis at long context.

## The Practitioner's Takeaway

Putting Chroma, NoLiMa, AbsenceBench, and OOLONG together produces a *checklist* for any developer choosing a long-context model in 2026:

1. **Where in the context will your most important information sit?** If middle — expect a U-curve dip. Use prompt engineering to put it at start or end ([The U-Curve](../07-lost-in-the-middle/)).
2. **Does your question share tokens with the answer in the context?** If no — apply a NoLiMa discount of 20-50 pp to the model's quoted NIAH score.
3. **Is the answer present in the context, or is the model expected to detect absence?** If absence — apply an AbsenceBench discount, possibly larger than the NoLiMa one.
4. **Does the answer require aggregating many independent facts from the context?** If yes — OOLONG is the relevant benchmark; expect <50% on frontier models at 128K.
5. **Which lab's failure mode are you willing to tolerate?** Anthropic models abstain (silent failures); OpenAI models confabulate (loud failures); Gemini sits in between.

This is, in some ways, the *practitioner-side mirror* of the layered-stack recommendation in [The 2026 Layered Stack](../16-eval-stack-2026/). The lab-side response is a layered benchmark suite. The practitioner-side response is a layered discount on quoted numbers.

## What To Remember

1. **The Chroma report (July 14, 2025) measured 18 frontier models** across NIAH, NoLiMa, and MRCR-style focused retrieval at 1K-1M context lengths. The phrase "context rot" entered the technical literature attached to a quantitative measurement.
2. **The canonical context rot curve has three regions**: a plateau (0-25% of window) where accuracy holds ~95%; a drop zone (50-75%) where accuracy falls 20-50 pp; a recency reflex (>90%) with partial recovery.
3. **Failure modes fingerprint the training lab.** Anthropic models abstain. OpenAI models confabulate. Gemini leans toward confabulation but with more abstentions. Same architecture family, different training values.
4. **NoLiMa removes literal token overlap** between query and needle, measuring semantic-only retrieval. Frontier models lose 20-50 pp at long context when this shortcut is taken away.
5. **AbsenceBench inverts NIAH**, requiring the model to detect what's *missing* from an expected set. Models fail badly because absence detection requires a different attention mode than presence detection.
6. **OOLONG measures aggregation across chunks** (the *width* axis of context rot). All major 2025 frontier models score <50% at 128K — the hardest open variant of the context-rot gradient.
7. **The 2026 practitioner discount stack**: apply a U-curve adjustment, a NoLiMa discount, an AbsenceBench discount, and an OOLONG discount to any quoted long-context number before believing it.

**Continue to** → [Synthetic vs Realistic](../13-synth-vs-real/) — the orthogonal axis along which every benchmark sits. Synthetic primitives like NIAH and GraphWalks are *diagnostic*; realistic suites like LongBench v2 are *predictive*. Which kind do you actually want? It depends on what you're trying to do with the model.
