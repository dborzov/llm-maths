---
title: "Token Embeddings"
slug: "embeddings"
description: "Dense vector representations mapping tokens to real-valued vectors: the lookup table that converts discrete tokens into geometry the model can reason over."
category: "transformer"
also_known_as:
  - "word embeddings"
  - "wte"
  - "embedding layer"
  - "word2vec"
  - "skip-gram"
  - "token lookup table"
  - "dense vectors"
  - "embedding dimension"
  - "vocabulary embedding"
source_of_truth: "/comicbook/01-word2vec/"
source_of_truth_title: "The Geometry of Meaning (Issue 01)"
related:
  - "transformer-weights"
  - "hyperparameters"
draft: false
---

The embedding layer maps a discrete token index to a dense vector in ℝⁿ. In microGPT this is `wte` (word token embedding), a matrix of shape `(vocab_size, n_embd)`.

| Symbol | What it is |
|---|---|
| `wte` | Token embedding table: `(vocab_size, n_embd)` |
| `wpe` | Position embedding table: `(block_size, n_embd)` — GPT-2/3 style; replaced by RoPE in modern models |
| `n_embd` | Embedding dimension (also called `d_model` in the literature) |
| cosine similarity | The primary distance metric for comparing embedding vectors |

**Why high-dimensional vectors?** Two random vectors in 300-dimensional space are nearly orthogonal — the space has room for every token to have its own direction. See [The Stranger Country](/comicbook/01-word2vec/05-high-dimensional/) for the geometric argument.

**In the transformer forward pass** the embedding lookup is the very first operation: `x = wte[token_id]`. In GPT-2/3-style models `wpe[pos_id]` was then added; in modern models (including microGPT) position is injected later via {{< wiki "rope" >}}RoPE{{< /wiki >}} on `q` and `k` inside each attention layer. The resulting vector enters the residual stream.
