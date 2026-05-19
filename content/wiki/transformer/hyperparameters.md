---
title: "Model Hyperparameters"
slug: "hyperparameters"
description: "The integer constants that define a model's architecture: n_layer, n_embd, n_head, block_size, head_dim, n_kv_head, group_size."
category: "transformer"
also_known_as: ["n_layer", "n_embd", "block_size", "n_head", "head_dim", "n_kv_head", "group_size", "d_model", "d_head", "context length", "number of layers", "embedding dimension"]
source_of_truth: "/comicbook/05-microgpt/02-state-dict/"
source_of_truth_title: "ch.2 The State Dict (microGPT)"
related: ["transformer-weights", "attention", "kv-cache"]
draft: false
---

Defined in `const.py` in the microGPT reference.

| Symbol | What it is |
|---|---|
| `n_layer` | Number of transformer blocks stacked |
| `n_embd` | Embedding dimension (also called `d_model` in papers) |
| `block_size` | Maximum sequence length / context window |
| `n_head` | Number of attention heads |
| `head_dim` | Dimension per head — equals `n_embd / n_head` |
| `n_kv_head` | Number of KV heads in GQA/MQA (equals `n_head` for standard MHA) |
| `group_size` | Queries-per-KV-head in GQA — equals `n_head / n_kv_head` |
