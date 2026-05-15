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
source_of_truth: "/issues/01_embeddings/"
source_of_truth_title: "Word2Vec and The Geometry of Meaning (Issue 01)"
related:
  - "transformer-weights"
  - "hyperparameters"
draft: false
---

The embedding layer maps a discrete token index to a dense vector in ℝⁿ. In microGPT this is `wte` (word token embedding), a matrix of shape `(vocab_size, n_embd)`.

| Symbol | What it is |
|---|---|
| `wte` | Token embedding table: `(vocab_size, n_embd)` |
| `wpe` | Position embedding table: `(block_size, n_embd)` |
| `n_embd` | Embedding dimension (also called `d_model` in the literature) |
| cosine similarity | The primary distance metric for comparing embedding vectors |

**Why high-dimensional vectors?** Two random vectors in 300-dimensional space are nearly orthogonal — the space has room for every token to have its own direction. See [Issue 01](/issues/01_embeddings/) for the geometric argument.

**In the transformer forward pass** the embedding lookup is the very first operation: `h = wte[token_ids] + wpe[positions]`. The resulting vector `h` enters the residual stream.
