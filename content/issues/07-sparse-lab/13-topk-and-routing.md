---
title: "Hard Top-k Without Crying"
description: "Primer. Top-k is a discrete operator that is not differentiable. Yet every sparse-attention method trains the upstream scorer via gradient descent. The straight-through estimator, soft top-k relaxations, and why the boring-but-correct answer is also the production answer."
topics: [attention, sparse-attention, routing, primer]
tags: [top-k, straight-through-estimator, gumbel-softmax, soft-top-k, routing, moe]
theme: teal
math: true
draft: false
date: 2026-05-16T11:20:00-04:00
issue: 7
weight: 130
techKind: primer
techNode: topk-routing
header: 13-topk-and-routing.webp
---

## The Differentiability Problem

The top-k operator $\text{Top-k}(I_{t,:})$ returns a discrete set. Its derivative is zero almost everywhere and undefined at the boundaries. You cannot backpropagate through it.

Yet DSA, CSA, and NSA all *train* the lightning indexer via gradient descent on the downstream {{< wiki "attention" >}}attention{{< /wiki >}} loss. How?

## The Straight-Through Estimator (STE)

The forward pass uses hard top-k. The backward pass pretends the operator was identity. Gradients flow through the selected tokens as if they were soft-weighted; the unselected tokens get zero gradient.

In code:

```python
def hard_topk_ste(scores, k):
    # Forward: hard select
    top_vals, top_idx = scores.topk(k, dim=-1)
    mask = torch.zeros_like(scores).scatter_(-1, top_idx, 1.0)
    # Backward: pass scores through as if identity, gated by mask
    return mask + (scores - scores.detach()) * mask
```

The trick: `mask + (scores - scores.detach()) * mask` evaluates to `mask` in the forward pass, but `(scores - scores.detach()) * mask` has gradient `mask` w.r.t. `scores` in the backward pass.

This is the workhorse of every production sparse-selection method.

## Soft Top-k Relaxations

The alternative to STE is replacing top-k with a differentiable approximation entirely.

**Gumbel-{{< wiki "softmax" >}}Softmax{{< /wiki >}} / Concrete:** Add Gumbel noise, temperature-anneal a softmax that asymptotically becomes a one-hot. Pros: smooth. Cons: noisy at high temperature, slow to converge at low temperature.

**Sinkhorn for entropic optimal transport:** Solve a regularized assignment problem. Pros: theoretically clean. Cons: needs an iterative solver, expensive.

**Differentiable sorting networks:** Replace argmax with a differentiable sort. Pros: math is beautiful. Cons: kernels don't exist; would be slow.

DeepSeek's choice across NSA, DSA, and V4: **STE for the hard top-k**, plus an *auxiliary loss* on the indexer to teach it to match the dense attention pattern. The auxiliary loss is the differentiability that matters; STE is the runtime mechanism.

```pyplot {id="ste-vs-soft" caption="STE PASSES SCORE GRADIENT THROUGH THE TOP-K SELECTOR. SOFT TOP-K RELAXATIONS USE A SHARPNESS-CONTROLLED SOFTMAX THAT BECOMES ONE-HOT AS TEMPERATURE → 0."}
import numpy as np
import matplotlib.pyplot as plt

scores = np.array([0.1, 0.3, -0.2, 0.9, 0.4, -0.6, 0.5, 0.1, 0.8, 0.2])
k = 3

# Hard top-k
top_idx = np.argsort(scores)[-k:]
hard = np.zeros_like(scores); hard[top_idx] = 1.0

# Soft top-k via softmax at temperature 0.2, 1.0, 5.0
def soft(scores, T):
    s = np.exp(scores / T)
    return s / s.sum()

soft02 = soft(scores, 0.2) * k
soft10 = soft(scores, 1.0) * k
soft50 = soft(scores, 5.0) * k

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(scores))
ax.bar(x - 0.3, hard, 0.18, color='#FF007F', label='hard top-3 (STE forward)')
ax.bar(x - 0.1, soft02, 0.18, color='#00A8A8', label='soft (T=0.2)')
ax.bar(x + 0.1, soft10, 0.18, color='#FFD700', label='soft (T=1.0)')
ax.bar(x + 0.3, soft50, 0.18, color='#FF8C00', label='soft (T=5.0)')
ax.set_xticks(x)
ax.set_xlabel('token index')
ax.set_ylabel('selection weight')
ax.set_title('hard vs soft top-k: temperature anneals soft toward hard')
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
```

## The Production Recipe

What DeepSeek actually does, drawn from the V3.2-Exp and V4 technical reports:

1. Pre-train with **dense attention** for the first ~1T tokens.
2. **Warm up the lightning indexer**: a short stage where the indexer is trained with a soft loss to predict the dense attention's top-k pattern.
3. **Switch to hard top-k via STE**: the indexer's gradients now come from the downstream LM loss directly.
4. **Auxiliary indexer loss** continues: a small term that encourages the indexer's scores to correlate with what the dense attention would have weighted.

No exotic relaxation. STE plus an auxiliary loss is what works — as used in [the lightning indexer](../06-lightning-indexer/) and [CSA](../07-csa/).

## The Routing Connection

Top-k selection is also the heart of MoE routing. Switch Transformer, DeepSeekMoE, Mixtral — every MoE router is a top-k operator over expert scores.

Same problem, same solution: STE for the hard routing, plus auxiliary loss for load balancing.

The pattern transfers cleanly between MoE and sparse attention. Anyone who has trained an MoE router has effectively trained 80% of a lightning indexer.

## What Goes Wrong

Three failure modes surface consistently in production:

1. **Stuck routing.** If the auxiliary loss is too weak, the indexer collapses to selecting the same tokens always — the "router collapse" problem. The model still runs; it just quietly ignores most of the context.
2. **Dead tokens.** Tokens that are *never* selected get no gradient, ever. Once a token is frozen out of the top-k, no signal arrives to reconsider it. Initialization matters a lot.
3. **Distribution shift.** If the indexer was trained on 64K context and you serve at 1M, the score distribution shifts in ways the indexer was never calibrated for. Continue-training on long-context data helps.

## What To Remember

1. **Top-k is not differentiable. STE makes it work anyway.**
2. **Auxiliary loss is what teaches the scorer.** Not the relaxation, not the temperature.
3. **The same pattern runs MoE routers.** Sparse attention training is a transfer from MoE training.
4. **Watch for router collapse and dead tokens.** Both kill quality silently.

**Continue to** → [Lightning Strikes Twice](../06-lightning-indexer/), where the lightning indexer uses exactly this recipe, and [Compressed Sparse Attention](../07-csa/), where top-k selection filters the compressed sequence.
