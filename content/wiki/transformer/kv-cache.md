---
title: "KV Cache"
slug: "kv-cache"
description: "The key/value cache that accumulates past-token projections across decode steps: keys[li], values[li], the (2,L,H,T,D) shape, prefill, and decode."
category: "transformer"
also_known_as: ["keys[li]", "values[li]", "key cache", "value cache", "prefill", "decode", "inference phases", "KV cache shape"]
source_of_truth: "/issues/05-microgpt-unfolded/13-kv-cache/"
source_of_truth_title: "ch.13 The KV Cache (microGPT)"
related: ["attention", "hyperparameters"]
draft: false
---

During autoregressive inference the model reuses past K and V projections rather than recomputing them.

| Symbol | What it is |
|---|---|
| `keys[li]`, `values[li]` | Accumulated K/V tensors for layer `li` |
| KV cache shape `(2, L, H, T, D)` | 5D view: 2 (K+V) × layers × heads × sequence × head_dim |
| **prefill** | Processing the entire prompt in one forward pass — cache is populated |
| **decode** | Generating tokens one at a time — cache is extended per step |

- Cache shape details → [ch.17 The Three Axes](/issues/05-microgpt-unfolded/17-kv-axes/)
- Prefill vs decode → [ch.14](/issues/05-microgpt-unfolded/14-prefill-decode/)
- KV cache in quantization context → [Issue 03 ch.10](/issues/03-sixteen-numbers/10-kv-cache/)
