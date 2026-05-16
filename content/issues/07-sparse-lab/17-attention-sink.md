---
title: "Attention Sinks Revisited"
description: "Primer. From StreamingLLM (2023) to V4's learnable sink logits. The softmax has to send mass somewhere; sinks are the escape valve. V4 turns the implicit sink mechanism into an explicit learnable parameter per head."
topics: [attention, sinks, softmax, primer]
tags: [attention-sinks, streamingllm, softmax, learnable-sinks, deepseek-v4, no-op-attention]
theme: teal
math: true
draft: false
date: 2026-05-16T12:00:00-04:00
issue: 7
weight: 170
techKind: primer
techNode: attention-sink
header: 17-attention-sink.webp
---

## The Strange Behavior

Xiao et al. (2023) made an unsettling discovery while building StreamingLLM: when you slide a window over a transformer's context — keeping only the last $W$ tokens — quality collapses *unless you also keep the first 4 tokens*. The model had "decided" that positions 0–3 were critical to its function.

The surprising part: it has nothing to do with what's in those tokens. It happens with *any* 4 tokens in those positions. Replace them all with `<BOS>` four times and the model runs fine. The model is using early positions as a *sink* for {{< wiki "attention" >}}attention{{< /wiki >}} probability mass, not as content carriers.

## Why Softmax Needs A Sink

The {{< wiki "softmax" >}}softmax{{< /wiki >}} normalizes over all keys. The sum of attention weights is *forced* to be 1.0. The model has no mechanism to say "I have nothing useful to attend to right now" — every query must distribute its full probability mass somewhere.

The emergent solution: dump it into tokens that don't matter. Positions 0–3 appear early and are always present, so the model learns to use them as no-op tokens — structural sinks that absorb whatever mass can't go anywhere useful.

Xiao et al. confirm this empirically: zero out attention to positions 0–3, and attention weights to *other* tokens spike disproportionately. The model wasn't attending to those early positions for their content. They were the constant in the denominator of a disguised division operation.

## V4's Explicit Sink

V4 formalizes this. Instead of relying on the model to discover sink positions by accident, V4 introduces a learnable sink logit $z'_h$ per attention head. From the V4 paper §2.3.3:

> *"In the core attention of CSA and HCA, we employ the trick of attention sink. To be specific, we set a series of learnable sink logits $\{z'_1, z'_2, \ldots, z'_{n_h}\}$. For the $h$-th attention head, $\text{Exp}(z'_h)$ will be added to the denominator of the attention score:*
> 
> $$s_{h,i,j} = \frac{\text{Exp}(z_{h,i,j})}{\sum_k \text{Exp}(z_{h,i,k}) + \text{Exp}(z'_h)}$$
> 
> *This technique allows each query head to adjust its total attention scores to be not equal to 1, and even to be near 0."*

The learnable $z'_h$ is a per-head escape valve. When a head has nothing relevant to attend to, it cranks $z'_h$ up; the denominator inflates, the attention weights collapse toward zero, and the head's output contribution is effectively nil. No wasted cache slots, no implicit sink tokens — just a single learned scalar per head doing the job explicitly.

## What This Frees Up

The learnable sink frees up two things that the implicit-sink approach could not.

1. **No "wasted" position-0 tokens.** V4 doesn't need to reserve any KV cache entries for sink behavior. The mechanism lives in the model weights, not in the context window. Sliding-window deployments no longer need to pin early tokens.

2. **Per-head abstention.** A head can effectively disable itself for a particular query by driving its sink logit high. This is *attention sparsity at the head granularity* — on top of the token-level sparsity that DSA provides.

The combination stacks cleanly: per-head abstention via the sink logit, plus per-query token selection via DSA, plus compressed sequence representation via CSA. Three orthogonal sparsity mechanisms, all active simultaneously.

```pyplot {id="sink-effect" caption="ATTENTION WEIGHT DISTRIBUTION FOR A SINGLE HEAD, WITH AND WITHOUT A LEARNED SINK LOGIT. THE SINK ALLOWS THE WEIGHTS TO SUM TO < 1, MEANING THE HEAD CAN ABSTAIN."}
import numpy as np
import matplotlib.pyplot as plt

# Set up a stylized logit vector
np.random.seed(7)
logits = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1])

# Without sink: standard softmax
w_no_sink = np.exp(logits) / np.exp(logits).sum()

# With sink logit z' = 3 (large): denominator inflates, attention drops
z_sink = 3.0
denom = np.exp(logits).sum() + np.exp(z_sink)
w_with_sink = np.exp(logits) / denom

fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(logits))
w = 0.35
ax.bar(x - w/2, w_no_sink, w, color='#FF007F', label='no sink: sum = 1.0')
ax.bar(x + w/2, w_with_sink, w, color='#00A8A8', label=f'sink logit z\'=3: sum = {w_with_sink.sum():.2f}')
ax.set_xticks(x)
ax.set_xlabel('past token index')
ax.set_ylabel('attention weight')
ax.set_title('learnable sink lets the head abstain — total mass falls below 1')
ax.legend()
ax.spines[['top','right']].set_visible(False)
```

## The Connection To Sparse Attention

Sparse attention sharpens the problem. When you sparsify to top-$k$ and renormalize over those $k$ tokens, you lose the ability to "vote for none of the above." If a head's actual best top-$k$ candidates are all useless for this query — say the head is specialized for long-range syntactic agreement and the current query has no relevant syntax — the renormalization still forces large attention weights over useless tokens. The head can't abstain; it can only mis-attend.

The learnable sink logit closes that gap. The head's logit vector now includes the sink term in the denominator. When the top-$k$ logits are all low, the sink dominates, the attention weights collapse, and the head output approaches zero. The model learns to route "nothing to say here" through the sink rather than through a random token.

This is why the [CSA chapter](../07-csa/) and the [HCA chapter](../08-hca/) both treat sink logits as non-optional. Sparsity without abstention concentrates mass on the wrong tokens; abstention without sparsity can't scale. The two mechanisms close the loop together.

## A Brief History

The mechanism developed quickly once people knew what they were looking for.

- **2023, June** — Xiao et al., StreamingLLM. First empirical observation that models implicitly learn attention sinks at early positions.
- **2023–2024** — Implicit sinks (pinning the first 4 tokens) become a standard engineering workaround in sliding-window deployments.
- **2024** — OpenAI's GPT-OSS paper formalizes attention sinks as a learnable scalar parameter added to the softmax denominator, removing the need for pinned tokens.
- **2026** — V4 adopts the OpenAI formulation with per-head learnable sink logits in both CSA and HCA.

The core observation is two-and-a-half years old. V4's contribution isn't the idea — it's making explicit learnable sinks standard equipment in a state-of-the-art production model.

## What To Remember

1. **Softmax has to normalize to 1.** Without a sink, the model dumps probability mass on the first 4 tokens by default.
2. **A learnable sink logit per head** is the clean fix. The head can drive the denominator up to abstain.
3. **Sparse attention needs sinks.** Otherwise, sparse-renormalization concentrates weight on irrelevant tokens.
4. **V4 ships with sink logits in both CSA and HCA.** Standard equipment going forward.

**Continue to** → [Compressed Sparse Attention](../07-csa/) or [Heavily Compressed Attention](../08-hca/), where the sink logit is one of the "other details" you'll find in the V4 paper's §2.3.3.
