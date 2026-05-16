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

The V4 {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} is not stored at a single precision — it is partitioned into three regions, each with the bit budget it actually needs. The paper (§2.3.4, Efficiency Discussion) states it directly:

> *"We adopt a mixed storage format for KV entries: BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage. Second, attention computation within the lightning indexer is performed in FP4 precision."*

Three regions, three precisions. The argument is not that some parts of the cache are more important — it is that different parts are differentially *sensitive* to quantization error, and the allocation tracks sensitivity rather than importance.

| Region | Precision | Bits/elem | Why |
|---|---|---|---|
| RoPE-rotated dimensions | BF16 | 16 | Position math is precision-sensitive; rotation amplifies errors |
| Body of KV (non-RoPE) | FP8 | 8 | Most of the dimensions; large dynamic range from {{< wiki "number-formats" >}}FP8 E4M3{{< /wiki >}} handles attention values |
| Lightning indexer QK | FP4 | 4 | Only the ranking matters; tiny precision suffices |

## Why The RoPE Channels Need More Bits

{{< wiki "rope" >}}RoPE{{< /wiki >}} is the culprit. The rotation $R_\theta$ applies a position-dependent rotation matrix to each pair of channels, and that rotation is not free with respect to quantization noise. A small error introduced at storage time gets *rotated through trigonometric functions* at every subsequent position, accumulating in directions that cannot be predicted or corrected by downstream layers.

The non-RoPE channels face no such amplification. They pass directly into the dot-product of the attention score; a small quantization offset stays where you put it and averages out across the $d_k$ dimensions. QuaRot (2024) and Atom (2024) documented this asymmetry empirically, and DeepSeek operationalizes it in V4: keep the RoPE channels at BF16 where rotation-amplified error would be destructive; compress everything else to FP8 where errors stay bounded.

## Why The Indexer Tolerates FP4

The [lightning indexer](../06-lightning-indexer/) does not need to compute accurate attention scores — it needs to *rank* blocks correctly so the top-k selection lands on the right tokens. Those are very different requirements. As long as the relative ordering of scores is approximately preserved, the top-k operator selects the same blocks regardless of whether the absolute score values are off by a few percent.

Quantization noise in the indexer's QK path adds roughly 0.3 to log-perplexity in pre-training studies; downstream {{< wiki "attention" >}}attention{{< /wiki >}} over the retrieved tokens absorbs that residual error cleanly. The compute saving is direct: 4× fewer bits to load and multiply per score. Issue 3 ch.6 walks through the same logic for INT8 weights — precision is needed where error amplifies, not where downstream operations absorb it.

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

V4 does not just apply FP4 at inference — it trains with the quantization noise present from the start of post-training. From the paper (§5.2.1), FP4 quantization-aware training is applied during fine-tuning, so the [CSA](../07-csa/) indexer path adapts its weights to the reduced precision rather than encountering it for the first time at serving time. The model that ships is a model that already knows how to work with FP4 scores.

This is the same QAT pattern Issue 3 ch.6b covers for weight quantization, applied here to a smaller and more naturally error-tolerant submodule — which is why it works cleanly without the elaborate calibration steps required for full-model quantization.

## Cross-Link To Issue 3

Issue 3 ("Sixteen Numbers Walk Into A GPU") covers the full quantization landscape — FP4, FP8, BF16, NF4, the number-format zoo — in depth. Two chapters are load-bearing for understanding what V4 does:

- [Numbers In Boxes](/issues/03-sixteen-numbers/02-numbers-in-boxes/) — how FP8 and BF16 represent values differently and what that means for rounding behavior
- [Hardware Horizon](/issues/03-sixteen-numbers/12-hardware-horizon/) — Blackwell's native FP4 support, which is what makes the indexer's FP4 path practical at scale

V4's mixed-precision KV cache is the **applied** version of Issue 3's central argument. Issue 3 showed that different numerical distributions call for different format choices; V4 takes that principle and deploys it within a single model's KV cache, assigning precision by region sensitivity rather than by a uniform global choice.

## What To Remember

1. **V4 stores KV in three precisions at once.** BF16 RoPE, FP8 body, FP4 indexer.
2. **Net: ~3× compression** vs. pure BF16 storage.
3. **Precision is allocated to amplification, not to importance.** RoPE rotations amplify errors; indexers absorb them.
4. **QAT is the bridge.** The model is fine-tuned with quantization noise so the FP4 path works at inference.

**Continue to** → [Compressed Sparse Attention](../07-csa/) — where the FP4 indexer and FP8 KV body live side by side.
