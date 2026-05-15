---
title: "Calibration & Blocks"
description: "A primer on the machinery that makes everything work: per-group scales, why 128 is the magic block size, and stochastic vs deterministic rounding."
topics: [quantization]
tags: [calibration, block-quantization, group-quantization, stochastic-rounding]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 110
techKind: primer
techNode: calibration
header: default.webp
---

## The Quiet Trick Behind Every Method

Read any modern quantization paper and you'll find a parameter labeled `group_size` or `block_size`, almost always set to 32, 64, or 128. It's mentioned in passing as if it were a footnote. It is not a footnote. **The block size is the difference between a working quantization scheme and a broken one.**

This article is the primer on what blocking actually buys you, why the number is what it is, and the calibration practices that go with it.

## The Problem With One Scale

Recall the basic INT-quantization picture. To represent a vector of real values $w$ with $n$ levels, you pick a scale $s$ and represent each $w_i$ as $w_i \approx s \cdot q_i$ where $q_i$ is an integer.

The question is: how do you pick $s$?

**Per-tensor scale**: one $s$ for the whole matrix. Simplest, smallest metadata, but fails badly when different rows have different magnitudes. This is what naïve {{< wiki "number-formats" >}}INT8{{< /wiki >}} does — and what makes it crumble on LLMs.

**Per-row scale**: one $s$ per row of the matrix. Better — adjusts for row-to-row heterogeneity. Standard since AlexNet-era quantization.

**Per-block scale**: one $s$ per *block* of consecutive elements within a row. The block is typically 32 to 128 elements wide. The size of the metadata grows, but the precision available to each block grows much faster.

```pyplot {id="block-vs-row-scale" caption="Same data, three scaling strategies. Per-block scaling adapts to local heterogeneity that per-row scaling misses."}
np.random.seed(42)
W = np.random.randn(8, 256) * 0.1
# Inject heterogeneity: each block of 64 in row 0 has a different scale
W[0, 0:64] *= 0.5
W[0, 64:128] *= 5.0
W[0, 128:192] *= 0.3
W[0, 192:256] *= 2.0

def quantize(x, scale, n_levels=15):
    q = np.clip(np.round(x / scale * n_levels), -n_levels, n_levels)
    return q * scale / n_levels

# Three scaling strategies
W_tensor = quantize(W, np.abs(W).max())
W_row = np.zeros_like(W)
for i in range(W.shape[0]):
    W_row[i] = quantize(W[i], np.abs(W[i]).max())
W_block = np.zeros_like(W)
for i in range(W.shape[0]):
    for b in range(0, 256, 64):
        W_block[i, b:b+64] = quantize(W[i, b:b+64], np.abs(W[i, b:b+64]).max())

err_tensor = ((W - W_tensor)**2).mean()
err_row = ((W - W_row)**2).mean()
err_block = ((W - W_block)**2).mean()
print(f"per-tensor MSE: {err_tensor:.6f}")
print(f"per-row MSE:    {err_row:.6f}  ({err_tensor/err_row:.1f}x better)")
print(f"per-block MSE:  {err_block:.6f}  ({err_tensor/err_block:.1f}x better)")

fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
data = [W[0], W_tensor[0], W_row[0], W_block[0]]
titles = ['Original (row 0)', 'INT4 per-tensor scale', 'INT4 per-row scale', 'INT4 per-block scale (block=64)']
colors = ['#1A1A1A', '#FF8C00', '#00A8A8', '#FF007F']
for ax, d, t, c in zip(axes, data, titles, colors):
    ax.plot(d, color=c, linewidth=0.8)
    ax.set_title(t, fontsize=10, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0, color='#1A1A1A', linewidth=0.3, alpha=0.5)
plt.tight_layout()
```

The per-tensor result is a wreck. Per-row is much better, but in the regions where the magnitude is small (block 0:64, scale 0.5; block 128:192, scale 0.3), the levels are wider than needed. Per-block adapts on a finer scale and gives near-original reconstruction.

## Why 128?

A common block size across GPTQ, AWQ, GGUF K-quant, and bitsandbytes NF4 is **128 elements per block**. Why this specific number?

Three forces converge on it:

1. **Memory locality.** A 128-element block in INT4 is exactly **64 bytes**, which is one cache line on most GPU architectures. Aligning blocks to cache lines means dequantization is fast.

2. **Metadata overhead.** Each block needs at least one scale (FP16 = 2 bytes) and often a zero-point. At block size 128, the metadata is 2 bytes / 128 values = **0.125 bits/value** on top of the 4 bits/value, for a total of ~4.125. At block size 32 it's 0.5 bits/value of metadata — a meaningful tax. At block size 1024 it's 0.016 bits/value but the precision benefits diminish.

3. **Empirical sweet spot.** Below 64 you start to overfit to per-block noise; above 256 you lose adaptiveness to within-row heterogeneity. 128 happens to land in the trough of the perplexity-vs-block-size curve for most LLM architectures. (This is not a deep mathematical truth — it's an empirical convergence across many papers.)

## Calibration: What Is It, And Why It's Mostly OK

Methods like [GPTQ](../08-brain-surgery/) and [AWQ](../09-method-family-tree/) require a **calibration set** — a few hundred sample inputs run through the model to gather activation statistics. Two questions:

**What size?** Astonishingly small. **128 samples** is the default for GPTQ. AWQ uses **32**. Even 16 often works. The reason is that the *layer Hessian* $X^\top X$ is a $d \times d$ covariance matrix, and the statistics we need from it (the diagonal, plus large-eigenvalue directions) stabilize with very few samples. We're not training, we're estimating an empirical covariance.

**Does it matter what you calibrate on?** Surprisingly little, within domain. Calibrating on C4 (a general web corpus) gives quantized models that work fine on Wikipedia, code, dialogue, etc. There is some degradation when the calibration set is *radically* different from the use case (e.g., calibrating only on code and then deploying on natural language), but it's typically <1% perplexity.

This is one of the lucky empirical facts of LLM quantization: it's not actually data-hungry. The activation statistics generalize across domains, because LLM activations are mostly determined by the *architecture*, not the specific input.

## Symmetric vs Asymmetric Quantization

Within each block, you can choose between two simple quantization schemes:

**Symmetric**: $w \approx s \cdot q$ where $q \in [-k, k]$. The grid is symmetric around zero. Used for weight quantization in GPTQ, AWQ, NF4.

**Asymmetric**: $w \approx s \cdot (q - z)$ where $q \in [0, 2k-1]$ and $z$ is a per-block "zero point" offset. The grid is shifted to fit the block's range. Used for activation quantization in SmoothQuant, ZeroQuant, and most KV-cache quantization.

Why the asymmetry of choices? Because **weights are roughly symmetric around zero** (see [Geometry Of Weights](../05-geometry-of-weights/)), so symmetric quantization wastes nothing. **Activations are often non-negative** (especially after ReLU/GeLU/SiLU), so symmetric INT4 would waste half the grid on values that never appear. Asymmetric quantization with a zero-point lets the grid sit where the data sits.

It's a small choice but real: symmetric quantization saves the cost of storing $z$, asymmetric quantization recovers the range. Most modern stacks let you mix-and-match.

## Stochastic Rounding: An 1822 Idea

When you round a real number $w$ to its nearest representable level $\hat{w}$, you can do it in two ways:

**Round-to-nearest (RTN):** $\hat{w} = $ closest level. Standard.

**Stochastic rounding (SR):** $\hat{w} = $ floor with probability $1 - (w - \lfloor w \rfloor)$, ceil with probability $w - \lfloor w \rfloor$. Random, but unbiased: $\mathbb{E}[\hat{w}] = w$.

Stochastic rounding has a beautiful property: **summed errors don't grow without bound**. For a deterministic round-to-nearest, error can accumulate consistently in one direction. For stochastic rounding, errors are mean-zero and add by random walk — bounded.

This idea is *old*. Charles Babbage's 1822 Difference Engine used a mechanical version of stochastic rounding to keep tabulation errors from accumulating. Hartley & Heath proposed it for digital computing in 1952. It fell out of favor when floats with adequate precision became cheap.

It came back with a vengeance for low-precision training. When you train a network and your gradients are FP8 or BF16, RTN biases the accumulated weights in a direction that hurts convergence. Stochastic rounding is essentially free precision — at the cost of one PRNG sample per multiplication.

Modern Hopper and Blackwell silicon support stochastic rounding in hardware. For pure-inference quantization, RTN is usually fine (you're not accumulating in a loop). For training in FP8/FP4, stochastic rounding is mandatory.

## A Closer Look At Per-Block Metadata

When you say "GPTQ at 4 bits, group_size=128", what's actually stored on disk?

- **Quantized values**: 4 bits × 128 values = 64 bytes per block
- **Scale**: 1 FP16 value per block = 2 bytes
- **Zero point (if asymmetric)**: 4 bits per block = 0.5 byte
- **Total**: 66.5 bytes for 128 weights = **4.16 bits/weight effective**

For a 7B model, that's 7B × 4.16 bits = ~3.6 GB. Compared to 14 GB at FP16 — a clean 4× compression, even after metadata.

You can push further. **Double quantization** (QLoRA) quantizes the *scale factors themselves*. Take all the FP16 scales for a tensor, group them into super-blocks of 256, and quantize those super-block scales with FP8. The metadata-on-metadata costs 8 bits / 256 = 0.03 bits/weight. Total saving: each block scale was 2 bytes = 16 bits / 128 = 0.125 bits/weight, now down to ~0.03 + (one super-scale per 256 sub-scales). Net: roughly 0.4 bits/weight saved.

It's a small saving in absolute terms but a meaningful win when every bit counts.

## Why Block-Wise Matters Beyond Weights

{{< wiki "kv-cache" >}}KV cache{{< /wiki >}} quantization (see [KV Cache Tyranny](../10-kv-cache/)) uses the same block idea, but on a different axis. KIVI quantizes K **per-channel block** (because outliers cluster per channel) and V **per-token block** (because outliers cluster per token). The block size still matters; the axis-of-blocking matters more.

For activations in general, **per-token quantization** is the standard. Each token gets its own scale, computed on the fly during inference. This costs a single max-reduction per token but means each token's activations are independently fit to the grid. It's how SmoothQuant and ZeroQuant achieve INT8 activation quantization without LLM.int8's mixed-precision split.

## Microscaling: Block Scales In Hardware

Software has been doing per-block scaling for a few years. In 2024, **NVIDIA Hopper** and the **OCP Microscaling spec** put block scales into the *number format itself*. The MXFP4 format consists of:

- 32 FP4 values per block
- 1 FP8 (E8M0) shared scale per block

The shared scale lives alongside the values in memory; the hardware knows to apply it during matmul without dispatching a separate kernel. This is the trick that makes FP4 actually viable as a hardware format despite having only 16 representable values per slot — the per-block scale slides those 16 dots to where the weights actually are.

We see this play out in [Hardware Horizon](../12-hardware-horizon/).

## Recap

1. **Per-block scaling** is the load-bearing trick of every modern method. Block sizes 32–128 are typical.
2. **Calibration sets** are small (32–128 samples), and **calibration is data-cheap**: statistics generalize across domains.
3. **Symmetric vs asymmetric** scaling matches the data's symmetry. Weights are symmetric; activations often aren't.
4. **Stochastic rounding** is 200 years old, free precision under accumulation, and has just returned for low-precision training.
5. **Microscaling** is per-block scaling baked into silicon — the next-generation format family.

**Continue to** → [Hardware Horizon](../12-hardware-horizon/) for where it all lands: native FP8 and FP4 on modern GPUs, and the timeline that got us here.
