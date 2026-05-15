---
title: "Residual Stream"
slug: "residual-stream"
description: "The residual stream: the hidden state vector h that accumulates information through transformer layers via skip connections."
category: "transformer"
also_known_as: ["hidden state", "x_residual", "h", "residual connection", "skip connection", "residual pathway"]
source_of_truth: "/issues/05-microgpt-unfolded/10-residual-stream/"
source_of_truth_title: "ch.10 The Residual Stream (microGPT)"
related: ["attention", "kv-cache", "transformer-weights"]
draft: false
---

The residual stream is the vector `x_residual` that flows through every transformer layer. Each layer adds its output to the stream rather than replacing it:

```
x_residual = x_residual + attn_output   # attention sublayer
x_residual = x_residual + mlp_output    # MLP sublayer
```

| Symbol | What it is |
|---|---|
| `x_residual` / `h` | Hidden state at a given layer and position |
| `h_j` | Hidden state at sequence position j |
| `h_j^out` | Hidden state after the attention sublayer update |
| `D_h` | Hidden dimension (embedding size, `n_embd`) |

**Why it matters for KV pruning** — In [Issue 06](/issues/06-eviction-notice/05-ghost-state/), the hidden state `h_t` is used as a feature to predict which KV pairs can safely be evicted, because it accumulates everything the model has computed about position t up to that layer.
