---
title: "Logit"
slug: "logit"
description: "The logit function (log-odds) and pre-softmax raw scores. In statistics: log(p/(1-p)). In deep learning: the raw model outputs before softmax normalization."
category: "transformer"
also_known_as:
  - "logits"
  - "log-odds"
  - "log odds"
  - "pre-softmax scores"
  - "raw scores"
  - "sigmoid inverse"
  - "lm_head output"
source_of_truth: "/issues/02-logit-wager/04-log-odds/"
source_of_truth_title: "ch.4 The Language of Risk (Issue 02)"
related:
  - "softmax"
  - "embeddings"
draft: false
---

The word "logit" has two related but distinct uses:

| Context | Meaning |
|---|---|
| Statistics | `logit(p) = log(p / (1−p))` — the log-odds of a probability `p`. Inverse of the sigmoid/logistic function. |
| Deep learning | The raw, unnormalized output scores of a model before softmax normalization. |

In LLMs, "logits" are the output of `lm_head` — a `(vocab_size,)` vector of unnormalized scores from which a probability distribution over the vocabulary is obtained by applying softmax.

**Why "logit"?** The connection is that passing the pre-softmax score through softmax is the multi-class generalization of passing log-odds through the sigmoid: both convert unbounded reals to probabilities.
