---
title: "Entropy, Cross-Entropy, KL Divergence & Mutual Information"
slug: "information-measures"
description: "The four log-based information measures: entropy (the fair price), cross-entropy (the bill a model pays, i.e. the training loss), KL divergence (the overcharge), and mutual information (the discount context buys)."
category: "information-theory"
also_known_as:
  - "entropy"
  - "Shannon entropy"
  - "H(p)"
  - "cross-entropy"
  - "cross-entropy loss"
  - "CrossEntropyLoss"
  - "F.cross_entropy"
  - "negative log-likelihood"
  - "NLL"
  - "log loss"
  - "binary cross-entropy"
  - "BCE"
  - "training loss"
  - "perplexity"
  - "PPL"
  - "bits per character"
  - "nats"
  - "KL divergence"
  - "Kullback-Leibler divergence"
  - "relative entropy"
  - "D_KL"
  - "forward KL"
  - "reverse KL"
  - "F.kl_div"
  - "mutual information"
  - "MI"
  - "I(X;Y)"
  - "conditional entropy"
  - "pointwise mutual information"
  - "PMI"
  - "weight of evidence"
  - "deciban"
source_of_truth: "/comicbook/02-logits/07-cross-entropy/"
source_of_truth_title: "ch.7 Cross-Entropy: the bill, the overcharge, and the discount (Issue 02)"
related:
  - "softmax"
  - "logit"
draft: false
---

| Quantity | Formula | Reading | Where it shows up |
|---|---|---|---|
| Entropy | $H(p) = -\sum p \log p$ | The fair price: fewest bits any code can achieve | Chinchilla's irreducible loss $E$ |
| Cross-entropy | $H(p,q) = -\sum p \log q$ | The bill a model $q$ pays on data from $p$ | The training loss; perplexity $= e^{H(p,q)}$ |
| KL divergence | $D_{\mathrm{KL}}(p \parallel q) = H(p,q) - H(p)$ | The overcharge; also expected evidence per sample | Distillation, RLHF penalty, quantization evaluation |
| Mutual information | $I(X;Y) = H(X) - H(X \mid Y) = D_{\mathrm{KL}}(p(x,y) \parallel p(x)p(y))$ | The discount context buys | What context is worth; PMI in word2vec; InfoNCE |

Asymmetry: KL and cross-entropy are not symmetric (forward KL is mass-covering, reverse KL is mode-seeking). Mutual information is symmetric.

PyTorch reports `loss` in **nats** (natural log). 1 nat ≈ 1.443 bits.
