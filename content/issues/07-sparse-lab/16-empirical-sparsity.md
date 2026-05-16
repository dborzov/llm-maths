---
title: "Where Attention Is Sparse"
description: "Primer. Empirical data on attention weights from trained transformers: 5-10% of past tokens absorb 90% of attention mass. Power-law distribution per query, per head, per layer. The structural foundation every sparse-attention method exploits."
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
header: default.png
---

## The Heatmap

[Open with the empirical picture. Show a real attention heatmap from a trained model on a 4K-token sequence. The hot regions cluster in:
1. The first 1-3 tokens (attention sinks).
2. The recent ~128 tokens (local window).
3. A small number of "heavy hitter" content tokens scattered through the middle.
The rest is cold.]

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

[Across many empirical studies (H₂O 2023, KVzip 2025, the NSA paper 2025), attention weights follow a power law:

$$p(w \ge t) \propto t^{-\alpha}$$

with $\alpha \approx 1.5$ for most heads of most layers. This means:
- Top 1% of tokens absorb 50% of attention mass.
- Top 5% absorb 80-90%.
- Top 10% absorb 95%+.

The exact thresholds vary per head, per layer, per query. The *qualitative shape* is universal.]

## What Is The 90% Made Of

[Three components, in roughly this order:

1. **Attention sinks at positions 0-3.** Xiao et al. (StreamingLLM, 2023) showed these are not informational — they are a *softmax escape valve*. The model dumps "I have nothing to attend to" probability mass at the start of the sequence. Always there. Always large.

2. **The local window of the last ~128 tokens.** Linguistic locality. Recent context matters more.

3. **A sparse scatter of content tokens** distributed through the middle. These are the "heavy hitters" — task-relevant tokens that the model actually needs.

A sparse-attention method has to keep all three. CSA does this with: indexer (catches content scatter) + sliding window (catches local) + attention sink mechanism (catches sinks). Three branches mapping to the three components of the 90%.]

## The Cross-Issue Connection

[The empirical sparsity of attention is the *same* observation that drives KV cache pruning (Issue 6). Heavy hitters get evicted last; cold tokens get evicted first. KV pruning *deletes* the cold; sparse attention *doesn't read* the cold. Same data, two strategies.

The strategies compose. You can prune (Issue 6) and then sparse-attend over what's left (Issue 7).]

## When Sparsity Breaks Down

[Failure cases:
1. **Synthetic needle-in-haystack tasks.** When the relevant token is *anywhere* in the context, the model can't predict its position. Indexer scores degrade.
2. **Very short queries.** With ~100 tokens of context, full attention is fine and sparsity overhead isn't worth it.
3. **Training distribution shift.** Sparsity patterns learned on natural text don't always transfer to code.

Production methods (DSA, CSA) survive these by maintaining the sliding window branch and the attention sink mechanism — backstops for when the indexer is confused.]

## What To Remember

1. **5-10% of past tokens absorb 90% of attention mass.** This is empirically robust across heads, layers, models.
2. **The 90% has structure.** Sinks (positions 0-3), local window (last ~128), scattered heavy hitters in the middle.
3. **Sparse attention exploits the same empirical fact as KV pruning.** Different mechanism, same opportunity.
4. **The structure can break.** Needle-in-haystack tasks defeat naïve sparsity; backstops are needed.

**Continue to** → [Sparse Attention's Lost Decade](../04-sparse-detour/) and [The NSA Blueprint](../05-nsa-paper/), where this empirical observation gets converted into architecture.
