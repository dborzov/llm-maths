---
title: "The NSA Blueprint"
description: "February 16, 2025. DeepSeek publishes 'Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention'. Three branches — compression, selection, sliding window — that compose. The paper that becomes the blueprint for DSA, CSA, and HCA all at once."
topics: [attention, sparse-attention, deepseek]
tags: [nsa, native-sparse-attention, deepseek, hardware-aligned, three-branch]
theme: cream
math: true
draft: false
date: 2026-05-16T09:40:00-04:00
issue: 7
weight: 50
techKind: mainline
techNode: nsa-paper
header: default.png
---

## A Preprint, Not A Product

**February 16, 2025.** Eight DeepSeek authors post a paper to arXiv: *"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"*. No model release. No weights. Just the architecture.

The paper is structured as a critique of every sparse-attention attempt that came before — the criteria from [the last chapter](../04-sparse-detour/) appear almost verbatim. The contribution is a single new architecture, **NSA**, that the authors claim passes all the criteria. Three branches, one composite output, fully trainable from scratch.

The paper is doing two things at once. It is a research contribution — a real algorithm with real ablations. But it is also a *blueprint*. Read carefully, and you can see the lab telegraphing what they are about to ship.

## The Three Branches

[The architecture: every layer's attention is split into three parallel paths, summed at the output.

**Branch 1: Compression.** Pool every $m$ tokens into one summary KV pair (a learned aggregation). Attend densely over the compressed sequence. Provides global context.

**Branch 2: Selection.** For each query, score every past *block*, keep the top-$k$ blocks, attend over them. Provides fine-grained retrieval.

**Branch 3: Sliding window.** Attend to the last $W$ tokens. Provides local detail.

Concatenate or sum the outputs of the three branches. Done.]

[Show the diagram from the NSA paper — or rebuild it with our own pyplot.]

```pyplot {id="nsa-three-branches" caption="NSA'S THREE-BRANCH ARCHITECTURE. EACH QUERY READS FROM ALL THREE PATHS. THE COMPRESSION BRANCH IS GLOBAL, THE SELECTION BRANCH IS FINE-GRAINED, THE SLIDING WINDOW IS LOCAL."}
# Schematic diagram showing the three branches feeding into a sum/concat
# at the output. Color-code: pink=compression, teal=selection, yellow=sliding window.
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5))
# Draw three boxes for branches with arrows feeding into a join node
# This is a schematic, so manual placement is fine
boxes = [
    (0.1, 0.65, 'compression\n(every m tokens → 1)', '#FF007F'),
    (0.1, 0.4, 'selection\n(top-k blocks)', '#00A8A8'),
    (0.1, 0.15, 'sliding window\n(last W tokens)', '#FFD700'),
]
for (x, y, label, c) in boxes:
    ax.add_patch(plt.Rectangle((x, y), 0.35, 0.18, facecolor=c, alpha=0.6, edgecolor='#1A1A1A', linewidth=2))
    ax.text(x + 0.175, y + 0.09, label, ha='center', va='center', fontsize=11)
    ax.annotate('', xy=(0.7, 0.5), xytext=(x + 0.35, y + 0.09),
                arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.5))

ax.add_patch(plt.Rectangle((0.7, 0.41), 0.18, 0.18, facecolor='#FF8C00', alpha=0.5, edgecolor='#1A1A1A', linewidth=2))
ax.text(0.79, 0.5, 'sum /\nconcat', ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title("NSA: three parallel attention paths, one summed output")
```

## What Each Branch Solves

[Walk through what each branch fixes from the criteria matrix in the previous chapter:

- **Compression** is data-dependent and trainable (vs. fixed patterns) — but it's coarse-grained.
- **Selection** is fine-grained and data-dependent — but it's discrete (top-k, not differentiable) — solved by straight-through estimator (see [topk primer](../13-topk-and-routing/)).
- **Sliding window** is the production-friendly fallback that ensures local coherence.

Together, all three feed *every* query, all three are differentiable, and the kernel design makes all three FlashAttention-compatible.]

## The Hardware Alignment

[The "hardware-aligned" in the paper title is doing real work. NSA picks block sizes aligned to GPU SM tile sizes (typically 128 or 64). Top-$k$ values are picked to be multiples of warp size (32). The block compressor uses a simple linear pooling that maps onto a single tile.

This is the difference between a research paper and a product. Reformer's LSH bucketing was mathematically clean; the kernel was a research project. NSA's three branches are designed *backwards from the kernel*.]

## What Was Missing

[Why was this a preprint and not a product? Two things:
1. **Training-from-scratch experiments only.** No demonstration of converting an existing dense model. For DeepSeek to ship NSA in a model that competed with V3, they would need to *retrain*, which is expensive.
2. **Three branches per layer is operationally heavier than one.** Each query reads from three KV sources. The kernel works, but it's not as fast as a single-branch design could be.

The paper hints at both. Six months later, the lab solves both by stacking everything onto MLA's latent — and dropping the compression branch.]

## A Blueprint, Not A Product

[Recap. NSA is the conceptual frame. The three branches recur in DSA (only the selection branch survives, with compression folded into MLA), and again in CSA (compression returns explicitly), and again in HCA (only compression, no selection).

Read NSA as a research note from a lab telling you what they are about to ship. The next paper is the product.]

## What To Remember

1. **NSA = three branches:** compression (global), selection (fine-grained, learned top-k), sliding window (local).
2. **Hardware alignment is what made it real.** Block sizes match SM tiles. Top-k matches warp size. Kernels work.
3. **It was a research preprint.** No model shipped on it. The lab was building toward something.
4. **All three branches recur downstream.** DSA inherits selection. CSA reintroduces compression. HCA inherits compression alone. Sliding window survives everywhere as a supplementary branch.

**Continue to** → [Lightning Strikes Twice](../06-lightning-indexer/) — September 29, 2025. V3.2-Exp ships. The selection branch survives as the lightning indexer; compression is folded into MLA's latent. The 50% price cut.
