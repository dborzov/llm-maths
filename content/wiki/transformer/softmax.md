---
title: "Softmax"
slug: "softmax"
description: "The softmax function: converts a vector of logits into a probability distribution, used in attention and language model heads."
category: "transformer"
also_known_as: ["softmax function", "attention softmax", "temperature", "softmax temperature", "log-softmax"]
source_of_truth: "/comicbook/05-microgpt/07-softmax/"
source_of_truth_title: "ch.7 Softmax (microGPT)"
related: ["attention", "hyperparameters"]
draft: false
---

Softmax converts a vector of raw scores (logits) into a probability distribution:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

| Term | Meaning |
|---|---|
| logits | Raw unnormalized scores before softmax |
| temperature | Scalar that divides logits before softmax; lower → sharper distribution |
| attention softmax | Applied to `attn_logits` scaled by `1/√head_dim` → `attn_weights` |

**Sparsity consequence** — In transformers, softmax attention produces distributions where a few tokens capture most of the weight mass. This winner-take-most behavior is the mathematical basis for KV cache pruning; see [ch.7 — Softmax's Long Tail](/comicbook/06-kvcache-pruning/07-attention-sparsity/).
