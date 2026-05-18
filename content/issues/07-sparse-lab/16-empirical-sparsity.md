---
title: "Empirical Sparsity: 5% of tokens, 90% of the mass"
short_title: "Empirical Sparsity"
description: "In trained transformers, 5–10% of past tokens absorb 90% of attention mass per query — a power-law distribution with three structural clusters: attention sinks, the local sliding window, and scattered heavy-hitter content tokens."
blurb:
  - "Plot attention weights for a single query over 4K tokens. The heatmap is almost entirely black."
  - "Three warm clusters: the first 1–3 tokens (sinks), the recent ~128 tokens (sliding window), and ~50 scattered heavy hitters."
  - "The model is simply not reading most of the context — and it doesn't need to, because the information isn't uniformly distributed."
  - "Power-law distribution per query, per head, per layer: the sparsity structure is not a bug the indexer introduces, it is a fact the indexer exploits."
topics: [attention, sparsity, empirical, primer]
tags: [empirical-sparsity, attention-weights, power-law, heavy-tail, attention-sinks]
theme: cream
math: true
draft: false
date: 2026-05-16T11:50:00-04:00
issue: 7
weight: 160
techKind: primer
techNode: empirical-sparsity
header: 16-empirical-sparsity.webp
---

## The Heatmap

Plot the {{< wiki "attention" >}}attention{{< /wiki >}} weights for a single query over a 4K-token past. The picture is striking: the heatmap is almost entirely black. What little heat there is clusters in three places:

1. The first 1-3 tokens — the attention sinks.
2. The recent ~128 tokens — the local sliding window.
3. A sparse scatter of "heavy hitter" content tokens spread through the middle of the sequence.

Everything else is cold. The model is simply not reading most of the context.

```pyplot {id="attention-heatmap" caption="STYLIZED ATTENTION HEATMAP FOR A SINGLE QUERY OVER A 4K-TOKEN PAST. HEAVY HITTERS AT POSITIONS 0-3 (SINKS), THE LOCAL WINDOW (RIGHT-MOST 128), AND A SCATTER OF MID-CONTEXT TOKENS."}
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)
T = 4096
# Build a stylized attention distribution
weights = np.zeros(T)
# Sinks at first 4 positions
weights[:4] = np.array([0.18, 0.12, 0.08, 0.04])
# Sliding window: last 128 tokens with decaying weight
sw = np.exp(-np.arange(128) * 0.02)
sw = sw / sw.sum() * 0.35
weights[-128:] = sw[::-1]
# Heavy hitters: ~50 scattered positions with elevated weight
hh = np.random.choice(np.arange(100, T-200), size=50, replace=False)
weights[hh] = np.random.exponential(0.005, size=50)
# Background noise
weights += 0.0002 * np.random.rand(T)
weights = weights / weights.sum()

fig, ax = plt.subplots(figsize=(11, 2.5))
ax.imshow(weights[None, :], aspect='auto', cmap='hot', vmax=np.percentile(weights, 99))
ax.set_yticks([])
ax.set_xlabel('token position (0 = start of context, 4096 = current query)')
ax.set_title('attention weights for one query over 4K-token past: sparse, with structure')

# Cumulative plot below
fig2, ax2 = plt.subplots(figsize=(8, 4))
sorted_w = np.sort(weights)[::-1]
cum = np.cumsum(sorted_w)
ax2.plot(np.arange(1, T + 1), cum, color='#FF007F', linewidth=2)
ax2.axhline(0.9, color='#1A1A1A', linewidth=1, linestyle='--', label='90% mass')
n_for_90 = (cum >= 0.9).argmax() + 1
ax2.axvline(n_for_90, color='#00A8A8', linewidth=1.5, linestyle=':', label=f'{n_for_90} tokens')
ax2.set_xscale('log')
ax2.set_xlabel('top-N tokens (sorted by weight, log scale)')
ax2.set_ylabel('cumulative attention mass')
ax2.set_title(f'90% of attention mass goes to top {n_for_90} of {T} tokens (~{100*n_for_90/T:.1f}%)')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')
ax2.spines[['top','right']].set_visible(False)
```

## The Power Law

Across many empirical studies — H₂O (2023), KVzip (2025), the NSA paper (2025) — attention weights follow a power law:

$$p(w \ge t) \propto t^{-\alpha}$$

with $\alpha \approx 1.5$ for most heads of most layers. That exponent has teeth:

- Top 1% of tokens absorb 50% of attention mass.
- Top 5% absorb 80-90%.
- Top 10% absorb 95%+.

The exact thresholds vary per head, per layer, per query — but not by much. The *qualitative shape* is universal. Heavy-tail distributions concentrate. This one concentrates hard.

## What Is The 90% Made Of

Three components, in roughly this order of importance:

1. **Attention sinks at positions 0-3.** [Chapter 17](../17-attention-sink/) covers this in detail: Xiao et al. (StreamingLLM, 2023) showed these are not informational — they are a *{{< wiki "softmax" >}}softmax{{< /wiki >}} escape valve*. The model dumps "I have nothing to attend to" probability mass at the start of the sequence. Always there. Always large.

2. **The local window of the last ~128 tokens.** Linguistic locality. Recent context matters more, and the model has learned to exploit it.

3. **A sparse scatter of content tokens** distributed through the middle. These are the "heavy hitters" — task-relevant tokens that the model actually needs to do the job.

A sparse-attention method has to capture all three, or quality collapses. CSA does this with three branches: an indexer that catches the content scatter, a sliding window that catches the local region, and an attention sink mechanism that catches the first few tokens. Three branches, three components of the 90%.

## The Cross-Issue Connection

The empirical sparsity of attention is the *same* observation that drives [Issue 6 — Eviction Notice](/issues/06-eviction-notice/). {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} pruning reads the same heavy-hitter data and draws the same conclusion: most tokens are cold. Heavy hitters get evicted last; cold tokens get evicted first. KV pruning *deletes* the cold tokens from the cache; sparse attention *doesn't read* them in the first place. Same empirical fact, two different mechanisms.

The strategies also compose. You can prune the KV cache (Issue 6) and then sparse-attend over what remains (Issue 7). Both layers of optimization are justified by the same underlying power law.

## When Sparsity Breaks Down

The power law is real, but it is not a law of nature. Three failure cases surface in practice:

1. **Synthetic needle-in-haystack tasks.** When the relevant token can sit *anywhere* in the context, the model cannot predict its position from the query alone. The indexer sees no signal; scores degrade toward uniform noise. The sparse method misses the needle.
2. **Very short contexts.** With ~100 tokens of context, full attention is cheap and sparsity overhead isn't worth it. The regime where sparsity pays starts around 4K tokens and up.
3. **Training distribution shift.** Sparsity patterns learned on natural language don't automatically transfer to code, structured data, or other domains with very different locality statistics.

Production methods (DSA, CSA) survive these failure modes by maintaining the sliding window branch and the attention sink mechanism as backstops. When the indexer is confused, the window and the sinks still catch the structure that was always there.

## What To Remember

1. **5-10% of past tokens absorb 90% of attention mass.** This is empirically robust across heads, layers, models.
2. **The 90% has structure.** Sinks (positions 0-3), local window (last ~128), scattered heavy hitters in the middle.
3. **Sparse attention exploits the same empirical fact as KV pruning.** Different mechanism, same opportunity.
4. **The structure can break.** Needle-in-haystack tasks defeat naïve sparsity; backstops are needed.

**Continue to** → [Sparse Attention's Lost Decade](../04-sparse-detour/) and [The NSA Blueprint](../05-nsa-paper/), where this empirical observation gets converted into architecture.
