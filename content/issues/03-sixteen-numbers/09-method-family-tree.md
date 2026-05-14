---
title: "The Method Family Tree"
description: "Six modern quantization methods side by side. What trick each one is making, where they agree, where they actively disagree, and which to pick when."
topics: [quantization]
tags: [gptq, awq, smoothquant, qlora, nf4, hqq]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 90
techKind: boss
techNode: method-family
header: default.png
---

## The Shelf

By mid-2024 the open-source LLM ecosystem had converged on a shelf of six quantization methods, each with a different angle and a different best-use case. This article puts them next to each other so you can see what each one is *actually* claiming.

We'll cover:

1. **LLM.int8()** — Dettmers, 2022. Mixed-precision split for activation outliers.
2. **GPTQ** — Frantar/Alistarh, 2022. Second-order layer-wise compensation.
3. **AWQ** — Lin et al., 2023. Protect weights that touch large activations.
4. **SmoothQuant** — Xiao et al., 2022. Migrate outliers from activations to weights.
5. **QLoRA / NF4** — Dettmers et al., 2023. Quantile-spaced 4-bit + double quantization for fine-tuning.
6. **HQQ** — Badri/Shaji, 2023. No calibration data needed — closed-form, fast.

Each has a one-page entry. At the bottom is a quick selection table.

---

## 1. LLM.int8() — The Outlier Decomposition

**Year:** August 2022.
**The trick:** Split activations into a "regular" path (INT8 + vector-wise scales) and an "outlier" path (FP16). Recombine for the matmul. See [The 1% That Ruins Everything](../06-outliers/) for the full story.

**When to use:** When you absolutely need *zero* accuracy degradation and don't mind a 3–4× slowdown vs FP16.

**Where it lives in the family:** The grandparent. Every later method either inherits the "outliers need special treatment" insight or builds an alternative way to handle them.

**Key trade-off:** Accuracy is excellent. Throughput is poor. Memory is halved. Calibration data: not needed.

---

## 2. GPTQ — Second-Order Compensation

**Year:** October 2022.
**The trick:** Layer-by-layer, quantize weights one column at a time, compensating each error against the remaining un-quantized weights using the Cholesky factor of the calibration Hessian $X^\top X$. See [Brain Surgery Returns](../08-brain-surgery/).

**When to use:** You want a *pre-quantized model* that runs fast at inference with no runtime overhead, you have ~128 samples for calibration, and you can spend ~30 minutes on a one-time calibration pass.

**Where it lives:** The workhorse of pure-weight quantization. Activations stay in FP16; only weights become INT4/INT3.

**Key trade-off:** Best-in-class 4-bit accuracy. Calibration takes time. Mainly weight-only.

---

## 3. AWQ — Activation-Aware Weight Quantization

**Year:** June 2023, by **Ji Lin** and collaborators at MIT.
**The trick:** Don't waste precision on *all* weights uniformly. Use the *activation magnitudes* to identify which weight columns are most often *multiplied by big activations* — those columns need to be quantized most carefully. Apply a per-channel scale to protect them.

In math: given activation $X$ with per-channel statistics $\bar{s}_i = \text{mean}_t |X_{ti}|$, AWQ finds a rescaling diagonal $S$ such that we replace

$$
Y = X W^\top \quad \text{with} \quad Y = (X S^{-1}) \cdot (S W^\top)
$$

The product is mathematically identical, but $(SW^\top)$ has been rescaled to make weights that interact with large activations *larger* relative to the quantization grid, so they survive rounding better. The "search" for $S$ is a grid search over a single scalar exponent.

**Why it's clever:** It needs almost no compute (no Hessian, no Cholesky, no sequential updates). Just one calibration pass to measure activation magnitudes, then a closed-form rescale. AWQ tends to *match GPTQ* in accuracy on most benchmarks while being **5–10× faster to calibrate**.

**When to use:** When you want GPTQ-like quality but without the second-order machinery. Especially good for very large models where Cholesky is annoying.

**Where it lives:** Sibling to GPTQ. They're often interchangeable for 4-bit; GPTQ tends to edge out at 3-bit, AWQ tends to edge out for 4-bit MoE models.

---

## 4. SmoothQuant — Migrate The Outliers Away

**Year:** November 2022, by **Guangxuan Xiao** and collaborators.
**The trick:** Take the *exact* algebraic identity from AWQ above — $Y = (X S^{-1}) \cdot (S W^\top)$ — and use it to make **both activations and weights** quantizable, by moving the outlier magnitudes from $X$ to $W$.

The intuition: outliers live in the activations, where they're hard to quantize. But weights are well-behaved (Gaussian-ish). So we use the diagonal scale $S$ to *push* the activation outliers into the weights:

- Pre-multiply each activation channel $i$ by $1/s_i$ — this **shrinks** the outliers in $X$.
- Pre-multiply each weight column $i$ by $s_i$ — this absorbs the displacement.
- Now **both** $XS^{-1}$ (activations) and $SW^\top$ (weights) are quantizable in INT8!

The choice of $s_i$ is a balance: $s_i = \max(|X_i|)^\alpha / \max(|W_i|)^{1-\alpha}$, with $\alpha = 0.5$ as a default. You're splitting the difficulty between activations and weights.

**Why it matters:** SmoothQuant is the first method that does **W8A8** quantization — both weights and activations in INT8, no FP16 path, no runtime mixed precision. Inference is fast (because everything is INT8) without LLM.int8's slowdown.

**When to use:** When you need *static* INT8 inference on hardware optimized for INT8 (TPUs, edge accelerators, older GPUs without FP8 support).

**Where it lives:** A bridge — the first method that took the outlier problem and dissolved it, rather than working around it. Conceptual parent of all the "smooth"/"migration" follow-ups.

---

## 5. QLoRA / NF4 — Fine-Tuning On A Single GPU

**Year:** May 2023, by Tim Dettmers (again).
**The trick:** Not one trick — three. (a) **NF4** as the weight format (Lloyd-Max-optimal for Gaussians, see [Lloyd-Max Bargain](../03-lloyd-max/)). (b) **Double quantization** — quantize the per-block scale factors *themselves* with FP8, saving another ~0.4 bits per weight. (c) **Paged optimizers** to swap optimizer state to CPU when GPU memory is tight.

**What QLoRA is for:** Not inference. **Fine-tuning.** You load the base model in 4-bit NF4 (frozen), and you train a small adapter (LoRA) on top in FP16. The base model never moves out of 4-bit, so the memory cost is dominated by the adapter, which is tiny.

Effect: you can fine-tune a 65B-parameter model on a **single 48 GB GPU**. This was, prior to QLoRA, considered impossible.

**Key insight reused:** The "NF4" format is the Lloyd-Max quantizer for a standard normal. It's a direct application of [Lloyd's 1957 result](../03-lloyd-max/). The double-quantization trick is a Russian doll: quantize the data, then quantize the quantization metadata.

**Where it lives:** Sibling-of-a-different-kind. QLoRA is the method that opened fine-tuning to academics and hobbyists. NF4 itself is now also used in pure-inference settings (e.g., as bitsandbytes 4-bit).

---

## 6. HQQ — Half-Quadratic Quantization

**Year:** October 2023, by **Hicham Badri** and **Appu Shaji** at Mobius Labs.
**The trick:** *No calibration data needed.* HQQ formulates quantization as a sparse-regularized optimization problem solved by **half-quadratic splitting** — a classical optimization technique from signal processing. The minimization runs entirely from the weight matrix alone, no $X$.

**Why this matters:** Calibration data is annoying to obtain ethically, especially for fine-tuned or proprietary models. HQQ skips it entirely.

**Trade-off:** Slightly worse accuracy than GPTQ/AWQ at 4-bit, but **calibration in seconds** (because there's nothing to calibrate). Excellent at 8-bit and 3-bit, where the gap to GPTQ is negligible.

**When to use:** When you're quantizing private models, when calibration data is unavailable, or when you simply want the fastest-possible quantization pass.

**Where it lives:** Stands a bit apart from the others. Methodologically aligned with the long tradition of signal-processing quantization theory rather than the deep-learning lineage.

---

## The Selection Table

| Method | Weights | Activations | Calibration | Inference Speed | Best Bit Width | Special Case |
|---|---|---|---|---|---|---|
| LLM.int8 | INT8 | INT8 + FP16 outliers | none | 3-4× slower | 8 | First wave |
| GPTQ | INT4/3 | FP16 | 128 samples | ~FP16 | 4 | Sequential compensation |
| AWQ | INT4 | FP16 | 32 samples | ~FP16 | 4 | Simpler than GPTQ |
| SmoothQuant | INT8 | INT8 | activation stats | full INT8 fast | 8 | W8A8 |
| QLoRA / NF4 | NF4 | FP16 | none (for inf) | ~FP16 | 4 | Fine-tuning |
| HQQ | INT4/3/2 | FP16 | **none** | ~FP16 | 4 | Data-free |

## What They All Share

Reading down the table you can pull out the things every successful method has in common.

**Block-wise scales.** Every method uses a per-group scale factor (typically blocks of 128 weights). Not per-tensor, not per-row. Per-block. The reason is in [Calibration & Blocks](../11-calibration-and-blocks/) — it's the simplest way to handle the heterogeneity of weight magnitudes within a layer.

**Weight-only when possible.** All the methods that achieve <FP16 size with minimal accuracy loss leave activations in FP16. The exception is SmoothQuant, which makes activations quantizable by *engineering* them to be so.

**Outliers as a first-class concern.** Whether you handle outliers by mixed-precision (LLM.int8), by activation-aware re-scaling (AWQ), by migration (SmoothQuant), or by quantile-aware levels (NF4), every method has an *outlier strategy*. None of them ignore the problem.

**No retraining required.** All six methods are **post-training quantization** (PTQ). You take a frozen FP16 model, run a calibration pass (or skip it for HQQ), and get a quantized model out. The model never sees its loss function again. This is in stark contrast to **quantization-aware training** (QAT), which was the dominant approach before 2022 and required full retraining at substantial cost.

## What They All Miss

Two things.

**1. Activations.** Most modern methods are weight-only. Activations live in FP16. This means the KV cache — which is *activations* in disguise — is unaffected. As long-context inference becomes the dominant cost, the KV cache becomes the dominant memory bill. We deal with it in [KV Cache Tyranny](../10-kv-cache/).

**2. The hardware floor.** All these methods are clever software tricks running on hardware that fundamentally does FP16 math. The next leap is when the *silicon itself* learns to do INT4 and FP4 math natively — see [Hardware Horizon](../12-hardware-horizon/). At that point, the software-quantization tricks become *cooperators* with hardware quantization, not workarounds for the lack of it.

## How To Pick

A practical decision tree if you're about to quantize a model:

1. **Need to fine-tune?** Use QLoRA/NF4. Nothing else.
2. **Have hardware with FP8 support (Hopper or newer)?** Use FP8 directly; quantization may not even be needed.
3. **Want INT8 inference?** Use SmoothQuant (W8A8). If accuracy is critical, fall back to LLM.int8.
4. **Want INT4 inference, have calibration data?** Use GPTQ for the very last bit of accuracy, AWQ if you want simpler.
5. **Want INT4 inference, no calibration data?** Use HQQ.
6. **On a phone or edge accelerator?** Use whatever the runtime supports — typically GGUF/GGML's K-quant variants, which combine ideas from GPTQ and HQQ.

Most users will, in practice, pick the format their inference engine supports — **GGUF** for `llama.cpp`-based stacks, **GPTQ** or **AWQ** for `vLLM` and TensorRT, **bitsandbytes** for `transformers`-native loading. The engine choice has come to matter as much as the algorithm.

**Continue to** → [KV Cache Tyranny](../10-kv-cache/) for the other half of the modern memory bill.
