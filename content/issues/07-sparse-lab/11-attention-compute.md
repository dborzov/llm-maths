---
title: "The Compute Bill of Attention"
description: "Primer. The exact FLOP counts for QK^T, the softmax, and attention·V — derived from the microGPT listing, then scaled to a 70B model at 128K context. Prefill vs decode regimes, arithmetic intensity, and why FlashAttention is a memory trick, not a compute trick."
topics: [attention, flops, hardware, primer]
tags: [attention-flops, prefill, decode, arithmetic-intensity, flash-attention]
theme: teal
math: true
draft: false
date: 2026-05-16T11:00:00-04:00
issue: 7
weight: 110
techKind: primer
techNode: attention-compute
header: default.png
---

## Where The FLOPs Live

[Open with the microGPT attention listing from Issue 5 ch.8. Walk through the three matrix products:
1. `attn_logits = q @ k.T / sqrt(head_dim)` — O(T² · head_dim) per head per layer.
2. `attn_weights = softmax(attn_logits)` — O(T²) per head per layer.
3. `head_out = attn_weights @ v` — O(T² · head_dim) per head per layer.

Total per head per layer per forward pass: $4 T^2 D$ FLOPs.
For prefill of a full sequence: $4 H L T^2 D$ FLOPs across the whole model.
For decode of one token at step $T$: $4 H L T D$ FLOPs.]

## Napkin Math: Llama-3-70B At 128K

[Concrete:
- H = 8 (GQA), L = 80, D = 128, T = 128K
- Prefill: $4 \cdot 8 \cdot 80 \cdot (128{,}000)^2 \cdot 128 \approx 4.3 \times 10^{14}$ FLOPs
- That's 430 TFLOPs — at H100 peak FP8 of 1.0 PFLOPs, ~430 ms of pure compute just for attention prefill.

For decode at step T=128K:
- $4 \cdot 8 \cdot 80 \cdot 128{,}000 \cdot 128 \approx 3.4 \times 10^{10}$ FLOPs/step
- ~34 GFLOPs per generated token, which is ~34 microseconds at H100 peak.

But: decode is memory-bound, not compute-bound. The 34 GFLOPs are bottlenecked by *loading the 4 GB KV cache* (Llama-3-70B GQA-8) from HBM. HBM bandwidth at 3 TB/s: ~1.3 ms per decode step *for cache loads alone*. So a single decode step is ~1300 µs limited by memory, not 34 µs limited by compute.]

```pyplot {id="attention-arithmetic-intensity" caption="ARITHMETIC INTENSITY (FLOP/BYTE) FOR PREFILL AND DECODE ACROSS CONTEXT LENGTHS. PREFILL CROSSES THE H100 'RIDGE POINT' (~80 FLOP/BYTE) AT MEDIUM T. DECODE NEVER DOES."}
import numpy as np
import matplotlib.pyplot as plt

T = np.logspace(np.log10(1024), np.log10(1_048_576), 200)
H = 8
L = 80
D = 128

# Prefill: compute scales as T^2, memory traffic as T (cache writes only)
# For QK^T: write T x D for queries, read T x D for keys. Compute T^2 D per head per layer.
prefill_flops = 4 * H * L * T**2 * D
prefill_bytes = 2 * H * L * T * D * 2  # K and V written once each, bf16
prefill_AI = prefill_flops / prefill_bytes

# Decode: compute T D per step per head per layer, memory T D per step
decode_flops = 4 * H * L * T * D
decode_bytes = 2 * H * L * T * D * 2  # read whole cache per step
decode_AI = decode_flops / decode_bytes

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.semilogx(T / 1000, prefill_AI, color='#FF007F', linewidth=2.5, label='prefill (compute-bound regime)')
ax.semilogx(T / 1000, decode_AI, color='#00A8A8', linewidth=2.5, label='decode (memory-bound regime)')
ax.axhline(80, color='#FF8C00', linewidth=1.5, linestyle='--', label="H100 ridge (~80 FLOP/byte)")
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('arithmetic intensity (FLOPs / byte)')
ax.set_title('Prefill crosses the ridge at medium T. Decode stays below.', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)
```

## Why FlashAttention Is Memory, Not Compute

[FlashAttention's trick is tiling the attention computation so the T × T score matrix never materializes in HBM. The compute is identical to naive attention — the same $4 H L T^2 D$ FLOPs. The savings are in *not paying* the memory bandwidth to write and re-read a T × T matrix.

This is why FlashAttention plus a longer context = faster prefill, but plus a quadratically-growing model size still hits the compute wall. FlashAttention does not solve the $O(T^2)$ scaling. It removes a constant factor.]

## What Sparse Attention Actually Cuts

[Forward link: sparse attention (DSA, CSA) doesn't compute the full T × T score matrix at all. The indexer scores T entries linearly. Then top-k means dense attention is over k entries, not T.

Compute drops from $T^2$ to $T + k^2 \approx T$ for large T. This is the linear-in-T regime.]

## What To Remember

1. **Attention is $4 H L T^2 D$ FLOPs for prefill.** $T^2$ is the term that dominates at long context.
2. **Decode is memory-bound.** Compute = $T$, memory traffic = $T$, but the constants make memory bandwidth the binding constraint.
3. **FlashAttention is a memory optimization.** It doesn't cut compute.
4. **Sparse attention cuts the $T^2$ term itself.** This is the move issue 7's mainline is about.

**Continue to** → [The Other Wall](../03-quadratic-wall/), where these FLOP counts power the mainline argument about why DSA had to exist.
