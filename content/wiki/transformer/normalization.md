---
title: "Layer Normalization"
slug: "normalization"
description: "Normalization layers in transformers: RMSNorm (used in modern LLMs) and LayerNorm (used in original transformers). Applied before attention and MLP blocks."
category: "transformer"
also_known_as:
  - "RMSNorm"
  - "LayerNorm"
  - "layer norm"
  - "layer normalization"
  - "rmsnorm"
  - "pre-norm"
  - "post-norm"
  - "root mean square normalization"
source_of_truth: "/issues/05-microgpt-unfolded/05-rmsnorm/"
source_of_truth_title: "ch.5 RMSNorm (microGPT)"
related:
  - "residual-stream"
  - "transformer-weights"
  - "hyperparameters"
draft: false
---

Normalization layers stabilize training by rescaling activations before each sub-block. Modern LLMs use **pre-norm** architecture: normalize before attention/MLP, then add the residual.

| Variant | Formula | Used in |
|---|---|---|
| LayerNorm | `(x − μ) / σ · γ + β` | Original Transformer (2017), GPT-2 |
| RMSNorm | `x / RMS(x) · γ` | LLaMA, Mistral, Qwen, DeepSeek |

**RMSNorm drops the mean-centering** (the `− μ` and `+ β` terms), making it cheaper while retaining the variance-scaling benefit.

**Effect on quantization:** post-normalization activations have roughly zero mean and unit variance, making them more amenable to uniform quantization. The outlier problem (see Issue 03) primarily arises *before* normalization, in the residual stream.
