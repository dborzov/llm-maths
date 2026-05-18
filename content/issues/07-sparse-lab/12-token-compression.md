---
title: "Token Compression: why learned weights beat average pool"
short_title: "Token Compression"
description: "CSA and HCA both compress m tokens into one using a softmax-weighted blend where weights are produced by the data itself — letting the model concentrate weight on a single high-information token rather than averaging it away."
blurb:
  - "Average pool weights every token equally. One function word alongside a key content word pulls the compressed vector away from what the model needs."
  - "Strided conv uses learned weights but applies the same schedule regardless of which token in the block is critical."
  - "Softmax-weighted pool: a block dominated by one high-information token can concentrate nearly all weight there."
  - "CSA's two-stream overlap means each compressed entry draws from 2m=8 tokens, not m=4 — boundary information is never lost."
topics: [attention, compression, primer]
tags: [token-compression, block-pooling, softmax-weighted, csa, hca, learned-pooling]
theme: cream
math: true
draft: false
date: 2026-05-16T11:10:00-04:00
issue: 7
weight: 120
techKind: primer
techNode: token-compression
header: 12-token-compression.webp
---

## The Operator

Both [CSA](../07-csa/) and [HCA](../08-hca/) collapse a window of $m$ token hidden states into a single compressed {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} entry using the same core operator. Given $m$ hidden states $h_j$, the compressor projects each one into KV space ($C_j = h_j W^{KV}$) and into a compression-weight space ($Z_j = h_j W^Z$), then blends them with {{< wiki "softmax" >}}softmax{{< /wiki >}}-normalized weights:

$$S = \text{Softmax}(Z + B), \quad C^{\text{Comp}} = \sum_j S_j \odot C_j$$

Each compressed entry is a softmax-weighted *element-wise* blend of $m$ raw KV vectors. The weights are *learned* and *positional*: $B$ is a learnable positional bias, one scalar per intra-block slot, so the model can systematically up- or down-weight a token based on its position within the block independent of content.

## Three Ways To Compress

Three designs compete for this slot, and the differences are not subtle.

**1. Average pool.** $C^{\text{Comp}} = \frac{1}{m} \sum_j C_j$. No parameters, zero overhead — and exactly the problem. Every token is weighted equally regardless of its content, so a function word like "the" averaged alongside a key content word pulls the compressed vector away from the thing the model actually needs to retrieve.

**2. Strided conv.** $C^{\text{Comp}} = \text{Conv1D}(C, \text{stride}=m)$. Learnable weights, which is progress — but the kernel is *position-independent*, fixed the same across every block. Whether the critical token is first or last in the block, the kernel applies the same schedule.

**3. Softmax-weighted pool (the V4 choice).** The compression weights are produced by the data itself ($Z = HW^Z$), so each block is compressed differently based on what is actually in that block. A block dominated by a single high-information token can concentrate nearly all weight there; a uniformly mixed block spreads weight more evenly.

```pyplot {id="compression-comparison" caption="THREE COMPRESSORS ON A TOY SEQUENCE WHERE TOKEN 3 IS A KEY CONTENT WORD AND OTHERS ARE FUNCTION WORDS. AVERAGE POOL FLATTENS IT. STRIDED CONV PARTIALLY PRESERVES IT. SOFTMAX-WEIGHTED LEARNS TO UPWEIGHT IT."}
import numpy as np
import matplotlib.pyplot as plt

# Toy: 4 tokens, one is "important" (3rd token has large norm)
m = 4
C = np.array([
    [0.1, 0.0, 0.2, -0.1],  # token 0 (function)
    [-0.1, 0.1, 0.0, 0.2],  # token 1 (function)
    [0.0, -0.1, 0.1, 0.0],  # token 2 (function)
    [0.8, -0.9, 1.2, 0.5],  # token 3 (content)
])

# Three compressors
avg = C.mean(axis=0)
# Strided conv with a learned kernel that just happens to upweight position 3
conv_w = np.array([0.1, 0.2, 0.2, 0.5])
strided = conv_w @ C
# Softmax-weighted: each row gets its own weight, learned to attend to the high-norm one
z = np.array([0.0, 0.0, 0.0, 3.0])  # learned to upweight position 3
s = np.exp(z) / np.exp(z).sum()
softw = s @ C

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(4)
w = 0.25
ax.bar(x - w, avg, w, color='#FFD700', label='average pool')
ax.bar(x, strided, w, color='#00A8A8', label='strided conv')
ax.bar(x + w, softw, w, color='#FF007F', label='softmax-weighted (V4 choice)')
ax.set_xticks(x)
ax.set_xticklabels(['dim 0', 'dim 1', 'dim 2', 'dim 3'])
ax.set_ylabel('compressed entry value')
ax.set_title('only softmax-weighted recovers the content token (row 3 of C)')
ax.legend()
ax.spines[['top','right']].set_visible(False)
ax.axhline(0, color='#1A1A1A', linewidth=0.5)
```

## The Two-Stream Overlap (CSA only)

The [CSA chapter](../07-csa/) runs *two* parallel compressor streams, $C^a$ and $C^b$, with the $b$ stream offset by half a block. Compressed entry $i$ therefore pools from two non-overlapping windows:

- $C^a$: tokens $[mi,\, m(i+1) - 1]$
- $C^b$: tokens $[m(i-1),\, mi - 1]$

Each compressed entry effectively covers $2m$ raw tokens, but adjacent compressed entries share $m$ of those tokens — a sliding-window effect at the compressed level. This makes information flow smoothly across block boundaries, which matters when a sentence or semantic unit straddles an arbitrary $m$-stride cut.

The [HCA chapter](../08-hca/) uses a single stream with no overlap. At $m'=128$, blocks are large enough that boundary artifacts are diluted across many tokens, so the extra stream's parameter cost is not justified by the marginal quality gain.

## What Compression Destroys

Compressing $m$ tokens into one is irreversible: token-level identity is gone, and you cannot in general recover the original $m$ KV vectors from $C^{\text{Comp}}$. The architecture absorbs this loss through three complementary mechanisms rather than trying to prevent it.

First, the sliding window branch keeps the last $n_{\text{win}}=128$ tokens fully uncompressed, so recent context is always exact. Second, the lightning indexer operates on compressed entries but is trained to preserve *relative ranking* across blocks — coarse retrieval over a long history does not require fine-grained reconstruction. Third, multiple {{< wiki "attention" >}}attention{{< /wiki >}} layers, each with their own compressor, can collectively re-distribute information: an early layer compresses a block, a later layer reads the compressed representation and decides which signals to carry forward. The loss at each individual layer is small; it does not compound catastrophically.

## What To Remember

1. **The compressor is softmax-weighted pooling.** $S = \text{Softmax}(Z + B)$, $C^{\text{Comp}} = \sum S_j \odot C_j$.
2. **The compression weights are learned per-token.** This is the difference from average pool and strided conv.
3. **CSA uses two overlapping streams. HCA uses one.** Trade-off: smoothness across block boundaries vs. parameter count.
4. **Compression is lossy.** Sliding window + multiple layers + the indexer absorb the loss.

**Continue to** → [Compressed Sparse Attention](../07-csa/), where the compressor sits at the bottom of Figure 3.
