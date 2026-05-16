---
title: "The Other Wall"
description: "MLA cut the KV cache by 30×. It did not cut a single attention FLOP. At 128K context the T×T score matrix is 16 billion entries per layer per head — and computing those scores is what was actually expensive."
topics: [attention, long-context, compute]
tags: [deepseek, quadratic, prefill, decode, flops, kv-cache]
theme: cream
math: true
draft: false
date: 2026-05-16T09:20:00-04:00
issue: 7
weight: 30
techKind: mainline
techNode: quadratic-wall
header: default.png
---

## A Different Bottleneck

[Open with a moment from the V3.1 technical report (Aug 2025): the team admits openly that long-context prefill is compute-bound, not memory-bound. The wall has moved. MLA solved the wrong half of the problem.]

## The T × T Matrix Nobody Materializes (But Everybody Pays For)

[The attention score matrix is $T \times T$ per head per layer. At T=128K, that's 16 billion entries per layer per head. You never store it — FlashAttention streams it. But you still *compute* it. The cost is $O(H \cdot L \cdot T^2 \cdot D)$ for prefill.]

[Napkin math:
- Llama-3 70B, T=128K, H=8 (GQA), L=80, D=128: 
  $80 \cdot 8 \cdot (128{,}000)^2 \cdot 128 \cdot 2 \approx 2.7 \times 10^{17}$ FLOPs just for QK^T per prefill.
- An H100 does ~1000 TFLOPs (FP8). That's ~270 seconds *just for the attention scores* at 128K. The full forward includes value-side and MLP too.]

```pyplot {id="prefill-vs-mlp-flops" caption="ATTENTION FLOPS VS MLP FLOPS, AS A FUNCTION OF CONTEXT LENGTH. MLP IS LINEAR IN T. ATTENTION IS QUADRATIC. CROSSOVER HAPPENS AT ~16K FOR TYPICAL CONFIGS."}
import numpy as np
import matplotlib.pyplot as plt

# Llama-3 70B-ish dims
L, H, D, d_model, d_ff = 80, 8, 128, 8192, 28672
T = np.logspace(np.log10(1024), np.log10(1_048_576), 200)

# Attention QK^T + attn·V FLOPs, both ~ 2 H T^2 D per layer per pass
attn_flops_per_layer = 4 * H * T**2 * D
# MLP FLOPs ~ 2 T d_model d_ff per layer (forward only)
mlp_flops_per_layer = 4 * T * d_model * d_ff

# Total
attn_total = L * attn_flops_per_layer / 1e12
mlp_total = L * mlp_flops_per_layer / 1e12

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.loglog(T / 1000, attn_total, color='#FF007F', linewidth=2.5, label='attention (QK^T + attn·V)')
ax.loglog(T / 1000, mlp_total, color='#00A8A8', linewidth=2.5, label='MLP')
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('prefill TFLOPs (Llama-70B-ish)')
ax.set_title('attention is O(T²); MLP is O(T). past ~16K, attention dominates.', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, which='both')
ax.spines[['top','right']].set_visible(False)
```

## Prefill vs Decode: Two Different Walls

[The decode-time picture is different. Decode generates one token at a time, so per-step you compute Q against all T cached keys: $O(T)$ per step, but $O(T)$ cache loads dominate the wall-clock — memory-bound, not compute-bound. This is where issue 6's KV pruning thread lived.

But for prefill, all T queries × T keys = $O(T^2)$ compute. This is the wall MLA didn't touch.]

[Show the regime crossover. Decode = memory-bound (handled by MLA and KV pruning). Prefill = compute-bound (the wall sparse attention has to solve).]

```pyplot {id="prefill-decode-regimes" caption="PREFILL IS COMPUTE-BOUND. DECODE IS MEMORY-BOUND. MLA SOLVED HALF OF BOTH. DSA SOLVES THE OTHER HALF."}
# Two-panel figure: x=T, y=arithmetic intensity (FLOP/byte)
# prefill: scales with T (compute-bound at large T)
# decode: roughly constant in T (memory-bound)
```

## The Cost of Long Prompts

[Concrete economic argument. A 128K-token document analysis service. Per-query cost breakdown. Why GPT-4-class APIs charge ~$10 per 1M input tokens — most of it is prefill compute. Why long-context apps stayed expensive even after MLA made the cache fit.]

## The Question The Field Was Avoiding

[Sparse attention had been *attempted* for years. The community had given up. The next chapter is the catalogue of failures.]

## What To Remember

1. **MLA cut cache, not compute.** Attention's prefill FLOPs are $O(T^2)$ regardless of how you store the cache.
2. **The T × T score matrix is the bottleneck.** At 128K context on a 70B-scale model, computing attention scores is ~270 H100-seconds per prefill.
3. **Prefill and decode have different walls.** Decode is memory-bound (cache loads). Prefill is compute-bound (score matrix). MLA addressed the decode wall. Something else has to address the prefill wall.
4. **The field knew this in 2021.** Sparse attention is six years old. It just didn't ship.

**Continue to** → [Sparse Attention's Lost Decade](../04-sparse-detour/) — the forensic tour of why six years of efficient-transformer research failed to produce a single shipping production model.
