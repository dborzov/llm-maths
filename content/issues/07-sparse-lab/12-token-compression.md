---
title: "Compressing m Tokens Into One"
description: "Primer. The softmax-weighted block compressor at the heart of CSA and HCA. Why a learned-weight pooling beats average-pool or strided convolution. The two-stream overlap trick. What gets lost, what gets preserved."
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
header: default.png
---

## The Operator

[The CSA/HCA compressor takes a window of $m$ token hidden states $h_j$, projects them to KV space $C_j = h_j W^{KV}$ and compression-weight space $Z_j = h_j W^Z$, then computes:

$$S = \text{Softmax}(Z + B), \quad C^{\text{Comp}} = \sum_j S_j \odot C_j$$

Plain English: each compressed entry is a softmax-weighted *element-wise* blend of $m$ raw KV vectors. The weights are *learned* and *positional* — $B$ is a learnable positional bias per intra-block slot.]

## Three Ways To Compress

[Compare three candidate compressors:

**1. Average pool.** $C^{\text{Comp}} = \frac{1}{m} \sum_j C_j$. Cheap, no params. But: every token weighted equally, regardless of importance. A "the" averaged in with a content word at equal weight wrecks the content word.

**2. Strided conv.** $C^{\text{Comp}} = \text{Conv1D}(C, \text{stride}=m)$. Learnable weights, but the weights are *position-independent* — every $m$-token block uses the same kernel. Cannot adapt to whether the first or last token of the block is more important *for that token's content*.

**3. Softmax-weighted pool (the V4 choice).** Each block's compression weights are produced by the data itself ($Z = HW^Z$). The model learns to compress *each block differently* based on what is in the block.]

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

[CSA has *two* parallel compressor streams, $C^a$ and $C^b$, with the $b$ stream offset by half a block. The compressed entry $i$ pools from:
- $C^a$ over tokens $[mi, m(i+1) - 1]$
- $C^b$ over tokens $[m(i-1), mi - 1]$

So each compressed entry covers $2m$ raw tokens, but adjacent compressed entries share $m$ of those tokens. This makes information flow smoothly across block boundaries — important when content doesn't respect the arbitrary $m$-stride alignment.

HCA does *not* do this — single stream, no overlap. The reason: at $m'=128$, blocks are large enough that boundary effects matter less, and the extra parameter cost of two streams isn't justified.]

## What Compression Destroys

[Hard truth: compressing $m$ tokens into one *loses information*. The token-level identity is gone. You cannot, in general, recover from $C^{\text{Comp}}$ the original $m$ KV vectors.

The model compensates in three ways:
1. **The sliding window branch** keeps the last $n_{\text{win}}=128$ tokens uncompressed.
2. **The lightning indexer** (in CSA) operates on compressed entries but can still discriminate which block is relevant — coarse retrieval over a long history.
3. **Multiple layers**, each with their own compressor, can collectively *re-distribute* information by reading-then-re-compressing at different points in the depth.]

## What To Remember

1. **The compressor is softmax-weighted pooling.** $S = \text{Softmax}(Z + B)$, $C^{\text{Comp}} = \sum S_j \odot C_j$.
2. **The compression weights are learned per-token.** This is the difference from average pool and strided conv.
3. **CSA uses two overlapping streams. HCA uses one.** Trade-off: smoothness across block boundaries vs. parameter count.
4. **Compression is lossy.** Sliding window + multiple layers + the indexer absorb the loss.

**Continue to** → [Compressed Sparse Attention](../07-csa/), where the compressor sits at the bottom of Figure 3.
