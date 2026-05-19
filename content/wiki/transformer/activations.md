---
title: "Activation Functions"
slug: "activations"
description: "Nonlinear activation functions used in transformer MLP blocks: ReLU, GELU, SiLU, SwiGLU, and their effect on weight distributions."
category: "transformer"
also_known_as:
  - "activation function"
  - "nonlinearity"
  - "ReLU"
  - "GELU"
  - "SiLU"
  - "SwiGLU"
  - "gated linear unit"
  - "GLU"
  - "Swish"
  - "post-activation distribution"
source_of_truth: "/comicbook/05-microgpt/12-activations/"
source_of_truth_title: "ch.12 Activation Functions (microGPT)"
related:
  - "mlp-block"
  - "residual-stream"
draft: false
---

Activation functions sit between the two linear layers of the MLP block. They introduce nonlinearity — without them, stacked linear layers collapse to a single linear transformation.

| Function | Formula | Property |
|---|---|---|
| ReLU | `max(0, x)` | Simple, one-sided (output ≥ 0) |
| GELU | `x · Φ(x)` | Smooth gating by normal CDF |
| SiLU/Swish | `x · σ(x)` | Similar to GELU, cheaper |
| SwiGLU | `SiLU(xW₁) ⊙ (xV)` | Gated: multiplies two projections element-wise |

**One-sided outputs matter for quantization.** Post-ReLU activations are non-negative, making symmetric (absmax) quantization wasteful — zero-point quantization handles one-sided distributions more efficiently.
