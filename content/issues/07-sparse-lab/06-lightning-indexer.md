---
title: "Lightning Strikes Twice"
description: "September 29, 2025. V3.2-Exp ships with DeepSeek Sparse Attention. The 'lightning indexer' is a tiny bilinear scorer that ranks every past token in low-rank space, and a top-k selector that picks the survivors. The 50% API price cut happens at midnight Beijing time."
topics: [attention, sparse-attention, deepseek, dsa]
tags: [dsa, lightning-indexer, deepseek-v3.2-exp, top-k, mla, integration]
theme: teal
math: true
draft: false
date: 2026-05-16T09:50:00-04:00
issue: 7
weight: 60
techKind: mainline
techNode: lightning-indexer
header: default.png
---

## The Drop, Re-Examined

[Reset the scene from the cold open: Sept 29 2025, the V3.2-Exp release. Now we are going to *open up the function* that did the work.]

## DSA in One Diagram

[Refer the reader forward: CSA in V4 is going to wrap DSA in a token compressor. For now, V3.2-Exp's DSA is the bare mechanism: take MLA's per-token latent $c_t$, derive indexer keys and queries from it, score every past token, keep the top-$k$, attend over those.]

## The Lightning Indexer, Line By Line

[Show the actual 13-line PyTorch function (faithful to the V3.2-Exp tech report). Walk through:

1. Project the query token's hidden state down to indexer query subspace.
2. The cached indexer keys (one per past token) live in a separate small cache.
3. Score = bilinear: `index_score[t,s] = q_I[t] @ k_I[s]` with low rank.
4. Top-k selector: keep the indices where the score is highest.
5. Full attention runs only over those k tokens' KV pairs from the main MLA cache.]

```pyplot {id="lightning-indexer-cost" caption="THE LIGHTNING INDEXER COSTS O(T × d_I) WHERE d_I IS THE INDEXER DIMENSION (~128). FULL ATTENTION COSTS O(T × d_h × H). FOR d_I=128, d_h=128, H=128: THE INDEXER IS ~128× CHEAPER THAN THE FULL SCORE PATH."}
import numpy as np
import matplotlib.pyplot as plt

T = np.logspace(np.log10(1024), np.log10(1_048_576), 200)
H = 128
d_h = 128
d_I = 128
# Full attention scores: H heads × T queries × T keys × d_h dim
# But for decoding (one new query): H × T × d_h per step
full_decode = H * T * d_h
indexer_decode = d_I * T  # one indexer "head" of dim d_I
# Top-k selection then attends to only k tokens
k = 2048
sparse_attend = H * k * d_h * np.ones_like(T)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.loglog(T / 1000, full_decode / 1e9, color='#FF007F', linewidth=2.5, label='dense attention (full T)')
ax.loglog(T / 1000, indexer_decode / 1e9, color='#FFD700', linewidth=2.5, label='lightning indexer score')
ax.loglog(T / 1000, sparse_attend / 1e9, color='#00A8A8', linewidth=2.5, label='attend over top-k=2048')
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('GFLOPs per decode step')
ax.set_title('lightning indexer + sparse attend << full dense for T > 16K', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, which='both')
ax.spines[['top','right']].set_visible(False)
```

## The MLA Integration

[Crucial detail: the indexer keys are derived from the *same MLA latent* $c_t$ that already lives in the cache. No new cache line. This is why DSA could integrate with vLLM on day one — the storage layout didn't change.

Recall MLA's absorption trick (issue 5 ch.19, issue 7 ch.2). The indexer query is computed from the same down-projected $c^Q_t$ that produces the main query. The same RoPE/NOPE channel split applies. DSA is *bolted onto* MLA, not parallel to it.]

## How Training Worked

[The V3.2-Exp technical report's training recipe:
- Start from V3.1 checkpoint (already trained on 128K context, dense MLA).
- Continue training with sparse attention enabled.
- Soft top-k (a temperature-controlled relaxation) at the start, hard top-k by the end.
- Distillation loss: match V3.1's attention output on a held-out set.
- Total continue-training compute: ~2% of original V3 training.

This is the continue-training pattern that the [top-k routing primer](../13-topk-and-routing/) discusses in its production-recipe section.]

## The Top-k Choice

[V3.2-Exp picks $k = 2048$ for a 128K context. That's attending to ~1.5% of the cache. Why 2048?

Three constraints:
1. **Quality floor.** Below ~1024 the long-context benchmarks (RULER, LongBench) start to drop.
2. **Kernel alignment.** $k$ must be a multiple of the FlashAttention tile size (typically 128 or 256).
3. **Decode latency target.** $k$ controls the per-step memory bandwidth — too large and you lose the decode-speedup the whole exercise was for.

The plot below sweeps $k$ against quality. The knee is sharp around $k=1024$, soft above; 2048 is a safe default.]

```pyplot {id="topk-sweep" caption="QUALITY ON RULER VS K (TOKENS ATTENDED PER QUERY) FOR DSA. THE KNEE IS AROUND k=1024. V3.2-EXP PICKS 2048 AS A SAFETY MARGIN."}
# Stylized sweep: x = k (log scale), y = quality (e.g., RULER score). Show:
# - Steep climb from k=128 to k=1024
# - Plateau from k=1024 to k=8192
# Plus a dashed line at the dense baseline.
k = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
# Stylized quality curve
quality = 0.92 / (1 + np.exp(-(np.log2(k) - 9) * 2)) + 0.05

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.semilogx(k, quality, color='#FF007F', linewidth=2.5, marker='o', markersize=7)
ax.axhline(0.94, color='#1A1A1A', linewidth=1, linestyle='--', label='dense MLA baseline')
ax.axvline(2048, color='#00A8A8', linewidth=1.5, linestyle=':', label='V3.2-Exp choice (k=2048)')
ax.set_xlabel('k (tokens attended per query)')
ax.set_ylabel('stylized RULER score')
ax.set_title('DSA quality vs top-k: knee at ~1024, plateau above')
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)
```

## The Price Cut

[Concrete economics: V3.1 vs V3.2-Exp pricing per million tokens, both input and output, both cache-hit and cache-miss. Show the 50% drop. Walk through where that came from: ~3× lower prefill compute at 128K, ~1.5× lower decode bandwidth, multiplied through the inference stack.]

## What V3.2-Exp Did Not Solve

[Setup for V4. DSA cut compute for queries, but the cache itself still scales with $T$. At 1M context, the MLA latent cache is still ~9 GB per query. To go further, you need *fewer cached entries*. The next move is to compress the sequence itself.]

## What To Remember

1. **The lightning indexer is a tiny bilinear scorer.** Low-rank QK, computed in FP4-friendly precision, one score per past token per query.
2. **Top-k = 2048** at 128K context. Roughly 1.5% of the cache.
3. **It bolts onto MLA.** The indexer keys are derived from the same latent $c_t$ in the existing cache. No new storage axis.
4. **Continue-training, not from scratch.** ~2% of the original V3 training compute.
5. **The price cut is real.** ~50% off the API.

**Continue to** → [Compressed Sparse Attention](../07-csa/) — V4's first new attention mode. Compress every 4 tokens into one entry *before* DSA runs. The indexer queries inherit MLA's absorption trick. The official paper figure included.
