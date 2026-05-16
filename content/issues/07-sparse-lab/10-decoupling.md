---
title: "Decoupling Memory From Time"
description: "Boss capstone. MLA shrinks cache per token to constant. CSA + HCA shrink compute per token to constant. The cost of context per inference query becomes LINEAR in T. The 10-year quadratic ceiling on long-context LLMs is gone. What does the world look like on the other side?"
topics: [attention, deepseek, economics, long-context, theory]
tags: [boss, decoupling, mla, dsa, csa, hca, linear-context, agents]
theme: cream
math: true
draft: false
date: 2026-05-16T10:30:00-04:00
issue: 7
weight: 100
techKind: boss
techNode: decoupling
header: default.png
---

## The Stack, One Last Time

[Open with the cumulative picture. MLA + DSA + CSA + HCA. Five years of work compressed into a single attention block.]

[The four cuts, restated as theorems:]

**Theorem 1 (MLA, 2024).** *Cache bytes per token per layer become constant in $T$.*
[ MLA stores one short latent $c_t$ per token, regardless of how many heads or how deep the head dimension. Cache footprint stops scaling with $H \cdot D$.]

**Theorem 2 (DSA, 2025).** *Attention scores per decode step become constant in $T$.*
[The lightning indexer scores $T$ tokens (linear) but only attends to top-$k$ (constant). For decode, this is a step function: 1 token in, $O(k)$ attention out.]

**Theorem 3 (CSA, 2026).** *Effective sequence length is reduced by compression rate $m$ before the indexer runs.*
[The indexer scores $T/m$ entries instead of $T$. With $m=4$, this is a 4× constant-factor cut on top of DSA.]

**Theorem 4 (HCA, 2026).** *Half the layers replace sparse selection with dense attention over heavily compressed entries.*
[Compute drops further; the indexer is removed from those layers entirely.]

## The Composition Theorem

[The big one. Multiply through:

Cost-per-decode-step for V4-Pro at context $T$:
- MLA cache: $O(d_c + d_R)$ per token per layer → linear in $T$ for the *cache size*, but constant in $T$ for the *per-token* read cost (since each cached entry is constant size).
- CSA: $O(T/m \cdot n^I_h \cdot c_I)$ for the indexer, $O(k \cdot n_h \cdot c)$ for the attended path.
- HCA: $O((T/m') \cdot n_h \cdot c)$.

For each layer, the per-step cost is **at most linear in $T$**, and for HCA + CSA's attended path, **constant in $T$**.

The query-side cost is the indexer + the heavy-compression dense attention. Both scale linearly in $T$ with very small constants ($1/m$, $1/m'$). At $T = 1M$ on V4-Pro: ~$10^{10}$ FLOPs per decode step. Compare to dense MHA at the same $T$: ~$10^{12}$. **100× cheaper.**

The cumulative ratio between V3.2 (which has MLA + DSA) and V4 (which adds CSA + HCA) is the 3.7× headline number from Figure 1. The cumulative ratio between V4 and a 2023-era vanilla MHA dense baseline is **~100×**.]

```pyplot {id="cumulative-architecture-cost" caption="CUMULATIVE COST PER DECODE STEP AT 1M CONTEXT, BY ARCHITECTURE GENERATION. EACH BAR ADDS ONE MORE CUT TO THE STACK ABOVE."}
import numpy as np
import matplotlib.pyplot as plt

stages = [
    ("vanilla MHA\n(2017)", 1.0),
    ("+ GQA\n(2023)", 0.13),  # roughly 8× compression on H-axis
    ("+ MLA\n(V2, 2024)",  0.04),  # ~30× on D-axis
    ("+ DSA\n(V3.2, 2025)", 0.012),  # ~3× compute cut
    ("+ CSA\n(V4 sparse layers)", 0.004),  # ~3× more
    ("+ HCA\n(V4 dense compress)", 0.001),  # half the layers cheaper still
]

names = [s[0] for s in stages]
costs = np.array([s[1] for s in stages])

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(range(len(stages)), costs, color=['#FF007F','#FF8C00','#FFD700','#00A8A8','#00A8A8','#1A1A1A'])
ax.set_yscale('log')
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('relative attention FLOPs per decode step (log)')
ax.set_title('Each architecture generation cuts attention cost by ~3-30×. Cumulative: ~1000×.')
ax.grid(True, alpha=0.3, axis='y', which='both')
for bar, c in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width()/2, c * 1.3, f'{c:.4g}', ha='center', fontsize=9)
ax.spines[['top','right']].set_visible(False)
```

## What This Buys

[The economic argument. Inference cost is dominated by attention at long context. Cutting attention cost by 100× cuts per-token serving cost by something like 5-30×, depending on the workload (more for long-context, less for short).

DeepSeek's API prices through this arc:
- V3 (Dec 2024): ~$0.14 / 1M input tokens
- V3.2-Exp (Sept 2025): ~$0.07 / 1M (50% cut)
- V4-Flash (2026): ~$0.04 / 1M input
- V4-Pro: more expensive per token but cheaper per long-context query

Compare to GPT-4-class APIs at $10 / 1M for input tokens. The factor is more than 100×.]

## The Applications That Become Possible

[Three categories that were latency- or cost-prohibitive before V4:

**Long-form coding agents.** A 1M-token context can hold an entire repository. Agents that read, edit, test, and re-read a codebase across sessions now have working memory at the scale of an actual code review.

**Legal document analysis.** A 500-page deposition + 20 prior depositions + relevant case law = ~1M tokens. Previously this required RAG retrieval over chunks. Now it can be one query.

**Genomics, scientific text mining.** Entire papers, plus references, plus prior papers in the same line of research. The 1M context becomes a "load this entire field" button.

The point is not that 1M tokens is the new normal everywhere — short-context queries remain short-context. The point is that *the option exists* and that the unit economics work.]

## The Question The Paper Does Not Answer

[V4 stacks four orthogonal cuts. The cuts compose multiplicatively because they attack different axes:
- MLA: D-axis (cache dimension)
- GQA (inherited): H-axis (heads)
- DSA: T-axis (selection)
- CSA: T-axis (compression before selection)
- HCA: T-axis (compression alone, no selection)

What is the *next* axis? The V4 paper hints at "model sparsity along new dimensions, such as more sparse embedding modules". The Conclusion lists:
- Embedding-level sparsity
- Low-latency architecture work
- Long-horizon multi-round agentic tasks
- Multimodal integration

None of these are attention-internal. The *attention* problem may, for the first time in a decade, be solved enough to move on to.]

## What Comes After Attention

[Speculation, clearly labeled as such. The next major LLM bottleneck, once attention's quadratic ceiling is gone, is likely:
1. **MoE routing cost.** Currently the second-largest compute term after attention. Will become first.
2. **Long-horizon state persistence.** V4 supports 1M tokens *per query*. The question becomes how to maintain working memory across queries, sessions, and weeks.
3. **Training efficiency at this architectural complexity.** V4 needed Muon + mHC + Anticipatory Routing + SwiGLU Clamping to train stably. The papers from 2026-2027 will probably be about simplifying this stack.]

## A Final Picture

[Restate the issue in one paragraph. The decade of quadratic attention is ending. A small lab in Hangzhou, stacking four bets one at a time, took it apart. The result is that the unit cost of context becomes linear. Every product that depends on long context — and that is, increasingly, every interesting product — runs on different math now.]

## What To Remember

1. **Four orthogonal cuts compose multiplicatively.** MLA (D-axis) × DSA (T-axis selection) × CSA (T-axis compression) × HCA (T-axis dense compression).
2. **Per-decode-step cost at 1M context drops by ~100×** vs. vanilla MHA, ~3.7× vs. V3.2.
3. **API prices drop by ~50% per generation.** Cumulative ~10× cut from V3 to V4.
4. **Million-token contexts are now economically viable** for legal, code, and scientific applications.
5. **The next bottleneck is not attention.** It's somewhere else now — MoE routing, training stability, multi-query state.

---

That is the issue. Six years of attention research, one Hangzhou lab, four stacked architectural moves. Whatever the next generation of LLMs looks like, it lives on the other side of this curve.
