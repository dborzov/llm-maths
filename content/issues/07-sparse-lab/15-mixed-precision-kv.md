---
title: "Mixed-Precision KV Cache"
description: "Primer. V4 stores KV in three precisions at once: BF16 for the RoPE-rotated channels, FP8 for the rest of the KV body, FP4 for the indexer's QK path. Why each region gets the bit budget it deserves — and why this is the natural endpoint of Issue 3's quantization arc."
topics: [quantization, attention, kv-cache, primer]
tags: [mixed-precision, fp4, fp8, bf16, kv-cache, deepseek-v4, quantization-aware-training]
theme: teal
math: true
draft: false
date: 2026-05-16T11:40:00-04:00
issue: 7
weight: 150
techKind: primer
techNode: mixed-precision-kv
header: default.png
---

## The Three Regions

[From the V4 paper §2.3.4 (Efficiency Discussion):

> *"We adopt a mixed storage format for KV entries: BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage. Second, attention computation within the lightning indexer is performed in FP4 precision."*

Three regions. Three precisions. The bit budget is allocated to where it actually matters.]

| Region | Precision | Bits/elem | Why |
|---|---|---|---|
| RoPE-rotated dimensions | BF16 | 16 | Position math is precision-sensitive; rotation amplifies errors |
| Body of KV (non-RoPE) | FP8 | 8 | Most of the dimensions; large dynamic range from {{< wiki "number-formats" >}}FP8 E4M3{{< /wiki >}} handles attention values |
| Lightning indexer QK | FP4 | 4 | Only the ranking matters; tiny precision suffices |

## Why The RoPE Channels Need More Bits

[The RoPE rotation $R_\theta$ applies a position-dependent rotation matrix to each pair of channels. Small quantization errors get *rotated through trigonometric functions* and accumulate in unpredictable directions across positions.

In contrast, the non-RoPE channels are just dot-producted directly. Quantization errors stay where you put them.

This is the empirical finding from QuaRot (2024) and Atom (2024) that DeepSeek operationalizes in V4: RoPE channels need full precision; the rest can compress aggressively.]

## Why The Indexer Tolerates FP4

[The indexer's job is *ranking*, not numerical correctness. As long as the relative ordering of scores is approximately preserved, the top-k operator will pick the same tokens.

Quantization noise in the indexer's QK path adds ~0.3 to log-perplexity in pre-training studies, which is absorbed by the downstream attention. The compute saving (4× lower) is direct.

This is the same logic Issue 3 ch.6 walks through for INT8 weights: precision is needed where amplification happens; it is not needed where downstream layers absorb the error.]

```pyplot {id="kv-cache-bit-budget" caption="V4's KV CACHE BIT BUDGET, BY REGION. BF16 ON 64 ROPE DIMS, FP8 ON THE REMAINING NON-ROPE DIMS, FP4 ON THE INDEXER QK. NET: ~5.5 BITS PER STORED FLOAT, DOWN FROM 16."}
import numpy as np
import matplotlib.pyplot as plt

# V4-Pro CSA per-token KV bit budget:
# - Compressed KV entry: c=512 dims, of which 64 are RoPE (BF16) and 448 are body (FP8)
# - Indexer key: c_I=128 dims, FP4
rope_bits = 64 * 16
body_bits = 448 * 8
indexer_bits = 128 * 4

total = rope_bits + body_bits + indexer_bits
bf16_baseline = (512 + 128) * 16

fig, ax = plt.subplots(figsize=(9, 4))
labels = ['RoPE (64 dims × 16b)', 'KV body (448 dims × 8b)', 'Indexer key (128 dims × 4b)']
values = [rope_bits, body_bits, indexer_bits]
colors = ['#FF007F', '#FFD700', '#00A8A8']
left = 0
for label, v, c in zip(labels, values, colors):
    ax.barh(0, v, left=left, color=c, label=label, edgecolor='#1A1A1A')
    ax.text(left + v/2, 0, f'{v}', ha='center', va='center', fontsize=10, color='#1A1A1A')
    left += v
ax.axvline(bf16_baseline, color='#1A1A1A', linewidth=2, linestyle='--')
ax.text(bf16_baseline + 50, 0.4, f'BF16 baseline:\n{bf16_baseline} bits', fontsize=9)
ax.set_xlim(0, bf16_baseline + 1000)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_xlabel('bits per token per layer (KV side only)')
ax.set_title(f'V4 stores per-token KV in {total} bits vs. BF16 {bf16_baseline} = {bf16_baseline/total:.1f}× compression')
ax.legend(loc='upper right', fontsize=9)
ax.spines[['top','right','left']].set_visible(False)
```

## Quantization-Aware Training

[V4 trains with QAT for the FP4 indexer path. From the paper §5.2.1, FP4 quantization-aware training is applied during post-training. The model adapts to the quantization noise during fine-tuning rather than seeing it for the first time at inference.

This is the same QAT pattern from Issue 3 ch.6b, but applied to a smaller and more error-tolerant submodule.]

## Cross-Link To Issue 3

[Issue 3 ("Sixteen Numbers Walk Into A GPU") covers FP4, FP8, BF16, NF4, and the quantization landscape in depth. The key bridge:

- [Numbers In Boxes](/issues/03-sixteen-numbers/02-numbers-in-boxes/) — how FP8 and BF16 differ
- [Hardware Horizon](/issues/03-sixteen-numbers/12-hardware-horizon/) — Blackwell native FP4

V4's mixed-precision KV cache is the **applied** version of Issue 3's theory. The theory said "different distributions need different precisions"; V4 applies this within a single model.]

## What To Remember

1. **V4 stores KV in three precisions at once.** BF16 RoPE, FP8 body, FP4 indexer.
2. **Net: ~3× compression** vs. pure BF16 storage.
3. **Precision is allocated to amplification, not to importance.** RoPE rotations amplify errors; indexers absorb them.
4. **QAT is the bridge.** The model is fine-tuned with quantization noise so the FP4 path works at inference.

**Continue to** → [Compressed Sparse Attention](../07-csa/) — where the FP4 indexer and FP8 KV body live side by side.
