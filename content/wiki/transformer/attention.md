---
title: "Attention"
slug: "attention"
description: "Multi-head attention: Q/K/V tensors, per-head computation, attention logits/weights, output, and attention variants (GQA, MQA, MLA, sliding window, sparse attention)."
category: "transformer"
also_known_as: ["multi-head attention", "MHA", "q", "k", "v", "q_h", "k_h", "v_h", "attn_logits", "attn_weights", "head_out", "x_attn", "attention heads", "n_head", "head_dim", "GQA", "MQA", "MLA", "grouped-query attention", "multi-query attention", "multi-head latent attention", "sliding window attention", "sparse attention", "DSA", "DeepSeek Sparse Attention", "lightning indexer", "CSA", "Compressed Sparse Attention", "HCA", "Heavily Compressed Attention", "NSA", "Native Sparse Attention", "attention sink", "sink logit"]
source_of_truth: "/issues/05-microgpt-unfolded/08-attention/"
source_of_truth_title: "ch.8 Scaled Dot-Product Attention (microGPT)"
related: ["kv-cache", "transformer-weights", "hyperparameters"]
draft: false
---

The core operation of the transformer. Within a single forward pass:

| Symbol | What it is |
|---|---|
| `q`, `k`, `v` | Current-token query/key/value vectors (after projection through `attn_wq/wk/wv`) |
| `q_h`, `k_h`, `v_h` | Per-head slices of `q`, `k`, `v` |
| `attn_logits` | Raw dot-product scores: `q · k / √head_dim` |
| `attn_weights` | Softmaxed scores — probability distribution over past tokens |
| `head_out` | Per-head output vector (weighted sum of values) |
| `x_attn` | Concatenated head outputs, fed into `attn_wo` |

**Multi-head attention** — [ch.9](/issues/05-microgpt-unfolded/09-multi-head/): the Q/K/V computation runs `n_head` times in parallel, each on a `head_dim`-wide slice.

**Attention variants:**
- **GQA / MQA** — grouped-query / multi-query: multiple query heads share fewer K/V heads (`n_kv_head`, `group_size`). See [ch.18](/issues/05-microgpt-unfolded/18-gqa/).
- **MLA** — multi-head latent attention: low-rank KV via `kv_down`, latent `c`, `d_c`. See [ch.19](/issues/05-microgpt-unfolded/19-mla/).
- **Sliding window** — per-layer attention window `W`. See [ch.20](/issues/05-microgpt-unfolded/20-sliding-window/).
- **DSA** — DeepSeek Sparse Attention (V3.2-Exp). Lightning indexer scores past tokens; top-$k$ go to full attention. See [Issue 7 ch.6](/issues/07-sparse-lab/06-lightning-indexer/).
- **CSA** — Compressed Sparse Attention (V4). Token-level compressor pools $m$ tokens into one entry, then DSA selects top-$k$ compressed entries. See [Issue 7 ch.7](/issues/07-sparse-lab/07-csa/).
- **HCA** — Heavily Compressed Attention (V4). Single compressor with $m'=128$, dense attention over the compressed sequence, no sparse selection. See [Issue 7 ch.8](/issues/07-sparse-lab/08-hca/).
- **Attention sinks** — Per-head learnable sink logit added to the softmax denominator, allowing a head to abstain. See [Issue 7 ch.17](/issues/07-sparse-lab/17-attention-sink/).
