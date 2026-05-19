---
title: "Rotary Position Encoding"
slug: "rope"
description: "RoPE (Rotary Position Encoding): encodes absolute positions as rotations of query/key vectors, giving relative position information to attention while supporting extrapolation."
category: "transformer"
also_known_as:
  - "RoPE"
  - "rotary position embedding"
  - "rotary encoding"
  - "position encoding"
  - "position embedding"
  - "RoPE theta"
  - "rope_theta"
source_of_truth: "/comicbook/04-long-context-bench/02-context-window/"
source_of_truth_title: "ch.2 What 1M Tokens Actually Costs (Issue 04)"
related:
  - "attention"
  - "kv-cache"
  - "hyperparameters"
draft: false
---

RoPE encodes token positions by rotating query and key vectors before the dot-product attention score is computed. A token at position `t` gets rotated by angle `t·θ` in each 2D subspace of the head, where `θ` varies across subspaces to create different "wavelengths."

| Term | Meaning |
|---|---|
| `rope_theta` | Base wavelength (default 10,000 in GPT; 500,000–10M in long-context models) |
| Long-context extension | Increasing `rope_theta` extends the effective context window |
| Relative-position property | The dot product `q · k` depends only on position *difference* `t - s`, not absolute positions |

**Why it matters for long context** — a model trained with `rope_theta = 10,000` will see position encodings it never trained on when given inputs beyond its training context length. Scaling `rope_theta` (YaRN, LongRoPE, etc.) is a key technique for extending context windows without full retraining.
