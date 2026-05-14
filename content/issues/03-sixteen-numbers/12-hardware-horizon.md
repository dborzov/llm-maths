---
title: "Hardware Horizon"
description: "When the silicon joins the conversation. FP8 on Hopper, MXFP4 on Blackwell, the OCP microscaling standard, and a five-event timeline of how we got here."
topics: [quantization, hardware]
tags: [hopper, blackwell, fp8, mxfp4, nvidia, ocp]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 120
techKind: boss
techNode: hardware
header: default.png
---

## Numbers Become Hardware

For most of the story we've told in this issue, *quantization is a software trick*. A clever calibration pass, a Cholesky factor, a per-block scale — all running on top of GPU hardware that fundamentally does FP16 (or FP32) math. The hardware doesn't know about INT4. It just multiplies floats. We dequantize on the fly.

In 2023, that changed. NVIDIA's **Hopper** architecture introduced native **FP8 tensor cores** — silicon-level multiplication and accumulation in 8-bit floating point, without a dequantization detour. In 2025, **Blackwell** doubled down and added native **FP4** (specifically **MXFP4**, with hardware-aware per-block scales).

This is the moment quantization stopped being a workaround and became a *first-class hardware feature*. The implications are still working themselves out.

## Hopper: FP8 In Two Flavours

Hopper's tensor cores support two FP8 formats, both following the IEEE 754-style geometry from [Numbers In Boxes](../02-numbers-in-boxes/):

- **E4M3**: 1 sign + 4 exponent + 3 mantissa = 8 bits. Range ±448. Best for weights and activations (more precision, less dynamic range).
- **E5M2**: 1 sign + 5 exponent + 2 mantissa = 8 bits. Range ±57,344. Best for gradients (more range, less precision).

The split is deliberate. *During training*, you want E4M3 for the **forward pass** (where precision matters for activations) and E5M2 for the **backward pass** (where gradient magnitudes can range across many orders). The split is the FP8 analog of the [BF16 vs FP16 choice](../02-numbers-in-boxes/): range here, precision there.

The throughput gain is dramatic. Hopper's FP8 tensor core runs at **2× the rate of BF16** for the same chip area. Combined with halving the memory footprint, you get **4× the effective training throughput** of BF16-based training. The first papers using FP8 (Micikevicius et al. 2022; later H100-specific work from NVIDIA and DeepSeek) reported essentially no loss in pretraining accuracy.

## Blackwell And MXFP4: The OCP Microscaling Spec

Blackwell pushes further. It supports **MXFP4** (and **MXFP6**, **MXFP8**, **MXINT8**) — the "MX" family from the **Open Compute Project's Microscaling specification**, ratified in 2023.

MXFP4 is exactly what [Calibration & Blocks](../11-calibration-and-blocks/) described, baked into the format:

- **32 FP4 values** per block. (Recall: FP4 has [exactly 16 distinct values](../02-numbers-in-boxes/) — its specific levels are $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ and their negatives.)
- **1 FP8 (E8M0) shared scale** per block. The scale is a pure-exponent format (no mantissa) — it's just a power of two. This is intentional: multiplying by a power-of-two scale is essentially free in hardware.
- **Total**: 32 × 4 + 8 = 136 bits per block of 32 values = **4.25 bits/value**.

The format encodes per-block scaling *in the storage layout itself*. There is no separate scale tensor to look up. The tensor cores read the block scale alongside the data and apply it during the multiply-accumulate.

This is the trick that makes FP4 viable. The 16 representable values per slot are useless on their own. With a per-block scale, those 16 values can be *placed wherever the block's weights need them* — and the placement comes for free, because the scale is right there in the format.

NVIDIA reports that Blackwell's MXFP4 tensor cores run at **roughly 2× the rate of FP8** for the same chip area. Combined with the halved memory and bandwidth costs, FP4 training is **~4× more efficient than FP8** and **~16× more efficient than BF16** at the same hardware budget.

## What FP4 Training Looks Like In Practice

Frontier labs are now training in FP4 routinely. The recipe is roughly:

1. **Pretrain in FP4** for most of the schedule, using MXFP4 for forward weights/activations and MXFP6 for gradients (more range needed).
2. **Stochastic rounding** during weight updates (see [Calibration & Blocks](../11-calibration-and-blocks/)).
3. **Periodic master-weight refresh** in FP32 to prevent drift.
4. **Final fine-tune in FP8 or BF16** for the last few percent of accuracy.

The loss curve, by all reports, is *indistinguishable* from FP8 training of the same model. We have arrived at the conclusion: **neural networks don't need most of the precision we used to give them**. They never did. The previous decade of FP16/BF16 training was a historical accident of the hardware being available.

## The Quantization-Hardware Co-Evolution

The story over 2019–2026 is one of software and hardware *teaching each other*:

- **2019–2021**: Software learns to quantize CNNs and small transformers to INT8. Hardware mostly does FP16/BF16.
- **2022**: Software (LLM.int8, GPTQ, AWQ) cracks LLM quantization to 4–8 bits. Hardware still does FP16 — quantization happens via dequantization-on-the-fly.
- **2023**: Hopper ships native FP8. Software starts to train *in* FP8 rather than just inference-quantize.
- **2024**: The OCP MX standard is ratified. Software adopts per-block scaling as universal.
- **2025**: Blackwell ships native MXFP4. Frontier labs train in FP4.
- **2026**: The frontier shifts to even-finer formats — MXFP3? Block-floating-point with adaptive precision? — and the entire training stack is rebuilt around hardware-native low-precision.

The five biggest inflection points, distilled:

{{< timeline name="quantization2019to2026" >}}

## What's Missing From The Timeline (And Why)

The OCP timeline above leaves out a lot. **GPTQ**, **AWQ**, **SmoothQuant**, **QLoRA** — none of them appear as separate dots. That is on purpose. The narrative is at the level of *industrial inflection points*, not specific paper releases. Within the bucket labeled "GPTQ & QLoRA" lives an entire ecosystem of related methods. Within "Blackwell MXFP4" lives a whole standards-body conversation that took years.

The discipline of stopping at five forces a perspective: which five things, if you removed them, would the rest of the story not have happened?

- Without **mixed-precision training**, FP16 doesn't establish itself as the deep-learning standard, and the 2020s don't have a default precision to negotiate down from.
- Without **LLM.int8**, the field doesn't realize outliers are real and treatable; everyone keeps trying QAT with diminishing returns.
- Without **GPTQ + QLoRA**, 4-bit inference and fine-tuning don't reach the open-source community in 2023, and the whole "run LLMs on consumer hardware" wave never crests.
- Without **H100 FP8**, the hardware floor for sub-INT8 stays at FP16, and the software/hardware co-evolution doesn't get its first joint round.
- Without **Blackwell MXFP4**, FP4 stays exotic and only inference-side; training in FP4 doesn't become normal.

Each is a different *kind* of inflection — a technique, an architecture, a hardware-software joint move. Together they describe the shape of the field.

## What This Issue Did Not Cover

A complete tour of LLM quantization in 2026 would also include:

- **Sparsity** + quantization (Wanda, SparseGPT, 2:4 structured sparsity)
- **Diffusion-model quantization**, which has different statistics
- **Quantization-aware training** (QAT), which has had a recent revival
- **Mixture-of-experts quantization** challenges (per-expert outliers, activation routing)
- The **GGUF** format and llama.cpp's K-quant variants in detail
- Quantization for **multimodal models** and vision encoders

Each is its own issue. The next one in the *LLM Maths Comics* line will pick something else from this list — let me know which one would be most useful to you, and it will be.

## What To Take Away

If you read only one sentence from this issue, make it this:

> Neural network weights and activations have **structure** that arbitrary lossy compression schemes do not anticipate — outlier dimensions, Gaussian bulk, second-order curvature — and exploiting that structure lets you use a number system with **sixteen distinct values** to represent something that used to take **65,536**.

Every chapter of this issue is a different facet of that idea. Lloyd-Max's centroid condition tells you where to put the 16 values. Hessian-aware quantization tells you which of them to be careful with. Outlier-aware methods tell you when to break the rules. Microscaling tells the hardware how to fit it all together.

We have come a long way from the 2017 mixed-precision paper. We are not done. The next decade will keep finding structure to exploit, and the formats will keep getting smaller. Somewhere there is a researcher right now, in a lab or a basement, looking at an LLM and seeing a property of the weights that nobody has noticed before. That property will become next year's quantization trick. That's how this story keeps moving.

---

**That closes Issue #3.**

If you want to read a sibling issue, the [back catalog](/issues/) is one click away. If you found this useful, the [contribute guide](/docs/contribute/) tells you how to draft your own issue using the same recipe. And if you want the seed prompt that produced this very issue, it lives in the repository's `prompts/` directory — outside the published site, kept in-repo for posterity.

Thanks for reading.
