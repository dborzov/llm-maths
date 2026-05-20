---
title: "Transformer Weight Matrices"
slug: "transformer-weights"
description: "The named weight matrices in microGPT: embedding tables, Q/K/V/O projections, MLP matrices, and lm_head."
category: "transformer"
also_known_as: ["wte", "wpe", "attn_wq", "attn_wk", "attn_wv", "attn_wo", "mlp_fc1", "mlp_fc2", "lm_head", "state dict"]
source_of_truth: "/comicbook/05-microgpt/02-state-dict/"
source_of_truth_title: "ch.2 The State Dict (microGPT)"
related: ["attention", "hyperparameters"]
draft: false
---

The persistent parameters that survive between forward passes. All live in `state_dict`.

| Symbol | What it is |
|---|---|
| `wte` | Token embedding table — shape `vocab_size × n_embd` |
| `wpe` | Positional embedding table — shape `block_size × n_embd` (GPT-2/3 style; not in microGPT, which uses RoPE) |
| `attn_wq` / `attn_wk` / `attn_wv` | Per-layer Q/K/V projection matrices |
| `attn_wo` | Attention output projection |
| `mlp_fc1` / `mlp_fc2` | MLP "fatten" and "skinny" matrices |
| `lm_head` | Vocab projection at the output — shape `n_embd × vocab_size` |

Shape reference lives in [ch.1 — don't re-derive per article](/comicbook/05-microgpt/01-cold-open/).
