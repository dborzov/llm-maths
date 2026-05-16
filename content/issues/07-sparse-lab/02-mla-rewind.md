---
title: "MLA: The First Cut"
description: "May 2024. DeepSeek-V2 ships Multi-head Latent Attention. We re-tell the algebra from Issue 5 ch.19 with a different audience in mind: not how MLA works, but why it was the first move in a longer game."
topics: [attention, kv-cache, deepseek]
tags: [mla, deepseek-v2, latent-attention, absorption, rope]
theme: teal
math: true
draft: false
date: 2026-05-16T09:10:00-04:00
issue: 7
weight: 20
techKind: mainline
techNode: mla-rewind
header: default.png
---

## Why We Are Doing This Again

This is the **second** issue on this site that begins with MLA. The [first](/issues/05-microgpt-unfolded/19-mla/) — Issue 5, chapter 19 — taught MLA *as a microGPT variation*. K and V folded into a latent $c_t$, an absorption identity that makes the read-time decompression free, the RoPE-vs-NOPE split, the algebra of the cache savings. If that chapter is fresh in your head, this one is going to feel familiar for the first three sections.

We are going to walk that math again. But the frame is different.

In Issue 5, MLA was a *technique* — a clever low-rank cache. In this issue, MLA is the *opening move* in DeepSeek's six-bet arc. It sets up the next bet, which sets up the next bet, which sets up DSA, which sets up V4. Every design choice we make in MLA reappears two papers later as a constraint that drives the next move. The RoPE-vs-NOPE split, in particular, is the seed of the lightning indexer.

If you read Issue 5 ch.19 already, skim sections 1–4 below for the new framing. If you didn't, read this chapter for the algebra and then go to Issue 5 ch.19 for the microGPT-level implementation.

## Hangzhou, May 6, 2024

[A scene-setting opening. Liang Wenfeng. The hedge fund roots. The 50-page paper drop on Hugging Face. The Figure 3 diagram. The footnote on the absorption trick. The reaction from inference teams.]

[Key facts to include:]
- DeepSeek-V2: 236B parameters, MoE (21B active), 128K context.
- Trained for $5.6M of compute (per the technical report).
- API price at release: ~$0.14 per million input tokens (cache hit), ~$0.28 per million output. Roughly 1/100 of GPT-4 pricing.
- The paper title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model".
- Open weights, MIT license, full technical report.

## The KV Cache Problem MLA Was Built For

[Recap the (2, L, H, T, D) cache shape and the napkin math. Llama-3 with GQA at 128K. The MLA promise of ~30x smaller.]

```pyplot {id="kv-axes-recap" caption="THE FIVE AXES OF THE KV CACHE. MLA ATTACKS D. GQA ATTACKS H. EVICTION ATTACKS T. QUANTIZATION ATTACKS THE BIT-WIDTH."}
# Recap chart: KV cache size vs context length for vanilla MHA, GQA, and MLA.
# Annotate which axis each method shrinks.
import numpy as np
import matplotlib.pyplot as plt

T = np.arange(1024, 131072, 2048)
L = 60     # DeepSeek-V2 layers
H = 128    # DeepSeek-V2 heads
D = 128    # head dim
d_c = 512  # MLA latent
bytes_per = 2

mha = 2 * L * H * D * T * bytes_per / 1e9
gqa = 2 * L * (H // 8) * D * T * bytes_per / 1e9
mla = L * (d_c + 64) * T * bytes_per / 1e9  # latent + rope channel

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(T / 1000, mha, color='#FF007F', linewidth=2.5, label='MHA (baseline)')
ax.plot(T / 1000, gqa, color='#00A8A8', linewidth=2.5, label='GQA-8 (H-axis cut)')
ax.plot(T / 1000, mla, color='#FFD700', linewidth=2.5, label='MLA (D-axis cut)')
ax.set_xlabel('context length (k tokens)')
ax.set_ylabel('KV cache (GB), DeepSeek-V2 scale')
ax.set_title('MLA shrinks the D-axis. GQA shrinks the H-axis. They compose.')
ax.legend()
ax.spines[['top','right']].set_visible(False)
```

## The Absorption Trick (Recap)

[The algebraic identity:
$q^\top (W_{uk} c) = (W_{uk}^\top q)^\top c$
Once per decode step instead of once per cached token. Same trick for V via linearity of the sum. See Issue 5 ch.19 for the full derivation. This time we are going to focus on what the identity DOES — pre-fold the cache expansion into the query — because that lens makes the RoPE split unavoidable in the next section.]

## The RoPE Compromise — And Why It Matters Two Papers Later

[The conflict: RoPE rotation depends on the absolute position of K, which is fixed at write time. But MLA's absorption trick means K is never materialized at write time. So:
- Split each head into k_nope (goes through latent) and k_rope (per-token, full width, small dim).
- DeepSeek-V2: D_R = 64, D_total = 128. Half the head dimension bypasses MLA.

The point of this section: the RoPE channel is a *seed*. It is the first time DeepSeek introduces a SECONDARY attention path alongside the main one. The pattern of "split the dimensions into two channels, one for the heavy lifting, one for a side calculation" is the architectural pattern that, eight months later, becomes the lightning indexer's bilinear scoring head. The lightning indexer is, viewed correctly, the third instance of this pattern.]

## What MLA Did Not Do

[The setup for the next chapter. MLA solved cache MEMORY. It did not touch:
- Attention FLOPs (still O(T^2) per layer).
- Prefill compute cost at long context.
- The score matrix size: T x T per layer, blown up but never stored.

Napkin math: at T=128K, the score matrix is ~16B entries per layer per head. Computing those scores is the actual cost driver in long-context prefill. MLA leaves this completely unsolved.]

```pyplot {id="mla-doesnt-cut-flops" caption="MLA CUTS CACHE BUT NOT FLOPS. THE FLOPS CURVE STILL SCALES AS T². THIS IS THE WALL THE NEXT CHAPTER WALKS INTO."}
T = np.arange(1024, 131072, 2048)
H = 128
D = 128
d_c = 512
L = 60

# Cache bytes per layer
cache_mha = 2 * H * D * T * 2 / 1e9
cache_mla = (d_c + 64) * T * 2 / 1e9

# FLOPs per layer for attention scores: roughly H * T * T * D
flops_mha = H * T * T * D / 1e12  # TFLOPs
flops_mla = H * T * T * D / 1e12  # SAME — MLA doesn't cut FLOPs

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(T / 1000, cache_mha, color='#FF007F', linewidth=2.5, label='MHA cache')
axes[0].plot(T / 1000, cache_mla, color='#FFD700', linewidth=2.5, label='MLA cache')
axes[0].set_xlabel('context length (k tokens)')
axes[0].set_ylabel('cache GB / layer')
axes[0].set_title('cache: MLA wins ~30x')
axes[0].legend()
axes[0].spines[['top','right']].set_visible(False)

axes[1].plot(T / 1000, flops_mha, color='#FF007F', linewidth=2.5, label='MHA flops')
axes[1].plot(T / 1000, flops_mla, color='#00A8A8', linewidth=2.5, label='MLA flops (identical)', linestyle='--')
axes[1].set_xlabel('context length (k tokens)')
axes[1].set_ylabel('attention TFLOPs / layer / step')
axes[1].set_title('flops: MLA changes nothing')
axes[1].legend()
axes[1].spines[['top','right']].set_visible(False)
plt.tight_layout()
```

## What To Remember

1. **MLA is the first cut.** Cache memory per token drops ~30× — the first axis of cost that decouples from $T$ at constant scale.
2. **The absorption identity does the work.** $q^\top (W_{uk} c) = (W_{uk}^\top q)^\top c$ means the cache's expansion to K can be pre-folded into Q. The compute stays put; the storage shrinks.
3. **The RoPE split introduces a side channel.** A small per-token vector $k_\text{rope}$ flows alongside the latent. This pattern — main path + small side channel — recurs in NSA's branches and again in DSA's lightning indexer.
4. **MLA leaves attention FLOPs untouched.** The $T \times T$ score matrix per layer is still computed in full. At 128K context this is the next wall.

**Continue to** → [The Other Wall](../03-quadratic-wall/) — the compute side of long-context attention. The wall MLA didn't touch, and the reason DSA had to exist.
