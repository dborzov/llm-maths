---
title: "MLP Block"
slug: "mlp-block"
description: "The MLP (feed-forward) sub-block in each transformer layer: two linear projections with a nonlinear activation function between them."
category: "transformer"
also_known_as:
  - "mlp"
  - "feed-forward network"
  - "FFN"
  - "mlp_fc1"
  - "mlp_fc2"
  - "SwiGLU"
  - "gated MLP"
  - "MLP sublayer"
  - "feed-forward block"
  - "FC1"
  - "FC2"
source_of_truth: "/comicbook/05-microgpt/11-mlp-block/"
source_of_truth_title: "ch.11 The MLP Block (microGPT)"
related:
  - "transformer-weights"
  - "residual-stream"
  - "activations"
draft: false
---

Each transformer layer contains an MLP sub-block that runs after attention. The pattern is: expand → activate → contract.

| Symbol | What it is |
|---|---|
| `mlp_fc1` | First projection: `(n_embd, 4·n_embd)` — expands the residual stream 4× |
| `mlp_fc2` | Second projection: `(4·n_embd, n_embd)` — contracts back |
| Activation | Nonlinearity between the two linear layers (ReLU, GELU, SiLU/SwiGLU) |

```python
# microGPT MLP block (simplified)
h = mlp_fc1 @ x_residual    # expand
h = activation(h)
h = mlp_fc2 @ h              # contract
x_residual = x_residual + h  # residual add
```

**SwiGLU / gated variants** split `mlp_fc1` into two parallel projections and multiply them element-wise before `mlp_fc2`. Used in most modern LLMs (LLaMA, Mistral, Qwen, DeepSeek).
