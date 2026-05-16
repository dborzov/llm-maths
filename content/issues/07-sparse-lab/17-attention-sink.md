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
header: default.png
---

## The Strange Behavior

[Open with the StreamingLLM observation (Xiao et al., 2023): when you slide a window over a transformer's context (keeping only the last $W$ tokens), quality collapses *unless you also keep the first 4 tokens*. The model has "decided" that positions 0-3 are critical to its function.

This is not because the first 4 tokens contain critical content. It happens with ANY 4 tokens. Replace them with `<BOS>` four times — works fine. The model is using them as a *sink*, not as content.]

## Why Softmax Needs A Sink

[The softmax normalizes over all keys. The sum of attention weights is *forced* to be 1.0. The model has no way to say "I have nothing to attend to right now" — every token must absorb some probability mass.

Solution: dump the probability mass into a token that doesn't matter (positions 0-3, the early sequence). The model effectively learns these as no-op tokens.

The Xiao et al. paper shows this empirically: zero out the attention to positions 0-3, and the attention weights to *other* tokens skyrocket disproportionately. The model wasn't using positions 0-3 informationally — it was using them as the constant in a division.]

## V4's Explicit Sink

[V4 formalizes this. From the V4 paper §2.3.3:

> *"In the core attention of CSA and HCA, we employ the trick of attention sink. To be specific, we set a series of learnable sink logits $\{z'_1, z'_2, \ldots, z'_{n_h}\}$. For the $h$-th attention head, $\text{Exp}(z'_h)$ will be added to the denominator of the attention score:*
> 
> $$s_{h,i,j} = \frac{\text{Exp}(z_{h,i,j})}{\sum_k \text{Exp}(z_{h,i,k}) + \text{Exp}(z'_h)}$$
> 
> *This technique allows each query head to adjust its total attention scores to be not equal to 1, and even to be near 0."*

The learnable $z'_h$ is a per-head escape valve. When a head has nothing relevant to attend to, it cranks $z'_h$ up, the denominator inflates, the attention weights collapse toward zero, and the head's output is effectively zero.]

## What This Frees Up

[Two things:

1. **No "wasted" position-0 tokens.** V4 doesn't need to dedicate any cache entries to sink behavior. The mechanism is structural.

2. **Per-head abstention.** A head can effectively disable itself for a particular query. This is *attention sparsity at the head granularity*, on top of the token-level sparsity DSA provides.

The combination is powerful: per-head abstention via sink logit + per-query token selection via DSA + compressed sequence via CSA. Three orthogonal sparsity mechanisms.]

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

[Why does this matter for an issue about sparse attention?

When you sparsify attention to top-$k$, you renormalize over those $k$. If the head's actual best top-$k$ are all *useless* (e.g., the head is specialized for syntax and there's no relevant syntax in this query), the renormalization still produces large attention weights over useless tokens.

The learnable sink logit fixes this. The head sees `useless_logit_1 + useless_logit_2 + ... + sink`. The sink wins, attention to the useless tokens collapses, the head abstains.

This is why V4's CSA and HCA both ship with sink logits as a default. Sparsity + abstention together close the loop.]

## A Brief History

[Timeline:
- **2023, June** — Xiao et al., StreamingLLM. First empirical observation of attention sinks.
- **2023-2024** — implicit sinks (the first 4 tokens) become standard in sliding-window deployments.
- **2024** — OpenAI's GPT-OSS paper formalizes attention sinks as a learnable parameter.
- **2026** — V4 adopts the OpenAI formulation, per-head learnable sink logits.

The mechanism is two-and-a-half years old. V4's contribution is making it standard equipment.]

## What To Remember

1. **Softmax has to normalize to 1.** Without a sink, the model dumps probability mass on the first 4 tokens by default.
2. **A learnable sink logit per head** is the clean fix. The head can drive the denominator up to abstain.
3. **Sparse attention needs sinks.** Otherwise, sparse-renormalization concentrates weight on irrelevant tokens.
4. **V4 ships with sink logits in both CSA and HCA.** Standard equipment going forward.

**Continue to** → [Compressed Sparse Attention](../07-csa/) or [Heavily Compressed Attention](../08-hca/), where the sink logit is one of the "other details" you'll find in the V4 paper's §2.3.3.
