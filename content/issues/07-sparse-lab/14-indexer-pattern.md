---
title: "The Indexer Pattern"
description: "Primer. A tiny model scores items for a big model. Speculative decoding, KVzap's surrogate, MoE routers, DSA's lightning indexer, CSA's compressed indexer. Five places the same architectural pattern shows up. Why it works."
topics: [attention, architecture, primer]
tags: [indexer-pattern, speculative-decoding, kvzap, moe-routing, lightning-indexer, surrogate-model]
theme: cream
math: true
draft: false
date: 2026-05-16T11:30:00-04:00
issue: 7
weight: 140
techKind: primer
techNode: indexer-pattern
header: default.png
---

## The Pattern, Stated Once

> **A small model produces a score over a large set of items. A bigger model uses the top-scoring subset.**

[That's it. The size asymmetry is the whole point. The small model is cheap enough to run over *everything*. The big model is expensive enough that you only want to run it on the items the small model says matter.]

## Five Instances

### 1. Speculative Decoding (Leviathan, Chen et al., 2023)

[A small draft model generates $k$ candidate tokens. The big target model verifies them in parallel. Accepts the prefix that matches; falls back to one big-model token on the first mismatch. Speedup: 2-4× for typical pairs.

The indexer here: the draft model is the scorer; the target model is the executor. The "score" is the proposed token; the "selection" is the verification accept-reject.]

### 2. KVzap Surrogate (NVIDIA, 2026)

[A tiny MLP (or linear layer) on the hidden state predicts a per-token importance score. KV cache entries below a threshold get evicted. The surrogate is ~1M parameters; the model it serves is 70B+. R² of the surrogate's prediction is 0.67-0.77.

Issue 6 (the previous issue on this site) covers this in depth. See [the boss chapter](/issues/06-eviction-notice/06-kvzap/).]

### 3. MoE Router (Switch, DeepSeekMoE, Mixtral)

[A linear layer scores each token against $E$ expert ids. Top-1 or top-2 experts are routed; the others get zero.

The router is one matrix multiply; the experts are the bulk of the model. The asymmetry is even larger here than in DSA — the experts are 99% of the parameters, the router is 0.01%.]

### 4. DSA's Lightning Indexer (DeepSeek, 2025)

[Issue 7 ch.6. The indexer is a bilinear scorer over MLA latents. Scores every past token; top-k goes to full attention.]

### 5. CSA's Compressed Indexer (DeepSeek, 2026)

[Issue 7 ch.7. Same lightning indexer, but it now scores *compressed* entries instead of raw tokens. Indexer queries are derived from the same latent as the main attention queries.]

## Why It Works

[Three structural reasons:

**1. Memory bandwidth dominates inference.** GPUs are starved for bandwidth, swimming in compute. The indexer adds compute to idle cycles while the main attention waits on HBM. The extra FLOPs cost wall-clock zero.

**2. Predictability of the score is "good enough".** R² doesn't have to be 1.0. As long as the *ranking* of the top-k is approximately correct, the downstream attention can recover. Issue 6's KVzap surrogate gets 67% R² and that's enough to evict 70% of the cache.

**3. The big model can absorb mistakes.** Sparse attention misses some relevant tokens; the multi-layer stack and the sliding-window branch cover the gap. Speculative decoding falls back to the target model on misses. The indexer doesn't have to be *right*; it has to be *cheap and approximately right*.]

```pyplot {id="indexer-asymmetry" caption="THE COMPUTE ASYMMETRY OF THE INDEXER PATTERN. INDEXERS ARE 100-10000× CHEAPER THAN THE MODELS THEY SERVE. WHITE-COLLAR ATTENTION OF FUTURE HARDWARE."}
import numpy as np
import matplotlib.pyplot as plt

methods = ['speculative\ndecoding', 'KVzap\nsurrogate', 'MoE\nrouter', 'DSA\nindexer', 'CSA\nindexer']
indexer_size = np.array([7e9, 5e6, 1e6, 2e6, 3e6])  # params
exec_size = np.array([70e9, 70e9, 671e9, 671e9, 1.6e12])  # params

ratio = exec_size / indexer_size

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars = ax.bar(range(len(methods)), ratio, color=['#FF007F', '#00A8A8', '#FFD700', '#FF8C00', '#1A1A1A'])
ax.set_yscale('log')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('executor / indexer parameter ratio (log)')
ax.set_title('the indexer is 10-500K× smaller than the model it serves')
for bar, r in zip(bars, ratio):
    ax.text(bar.get_x() + bar.get_width()/2, r * 1.3, f'{r:.0f}×', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y', which='both')
ax.spines[['top','right']].set_visible(False)
```

## When The Pattern Fails

[Three failure modes:

1. **Score-loss mismatch.** If the indexer's scores correlate with the wrong thing (e.g., raw attention weight when the downstream task cares about contribution norm), the selected items are wrong. Issue 6 ch.8 covers this.

2. **Indexer parameter starvation.** Too small and the indexer can't learn anything useful. Below ~1M params, lightning indexers stop discriminating.

3. **Distribution shift.** Indexer trained on one regime, served on another. Continue-training is the standard fix.]

## The Pattern Going Forward

[Speculation. The indexer pattern is going to recur in every part of the inference stack where item selection happens:
- Tool selection in agentic LLMs ("which tool should I call?")
- Context-window curation across sessions ("which past sessions are relevant?")
- Multi-token prediction (MTP) verification heads

Wherever a model has to pick a few items from many, expect a tiny scorer + big executor.]

## What To Remember

1. **Tiny scorer, big executor.** The pattern.
2. **Memory bandwidth makes it free.** Indexer compute fills bandwidth-stalled cycles.
3. **Approximately-right is enough.** Multi-layer stacks absorb indexer mistakes.
4. **Five instances and counting.** Speculative decoding, KVzap, MoE routing, DSA, CSA.

**Continue to** → [Lightning Strikes Twice](../06-lightning-indexer/), where the pattern hits the attention stack.
