---
title: "FP8 E4M3: three mantissa bits in attention math"
short_title: "FP8 E4M3"
description: "The E4M3 layout, why softmax is forgiving of precision loss, and what per-tensor scale = 1.0 assumes about the KV cache."
blurb:
  - "1 sign bit, 4 exponent bits, 3 mantissa bits — 16 representable values per binade."
  - "Softmax is unusually forgiving: relative order of scores matters more than absolute precision."
  - "The default vLLM scale factor is literally 1.0. No calibration, no per-head tuning."
  - "Why this works on most models — and what assumption it quietly makes about KV activation distributions."
topics: [quantization, primer]
tags: [fp8, e4m3, softmax, ieee754, number-formats]
theme: cream
math: true
draft: false
date: 2026-05-17T09:00:00-04:00
issue: 9
weight: 30
techKind: primer
techNode: fp8-format
header: default.webp
---

## A Number With 16 Mantissa Rungs Per Octave

If you have read [Numbers In Boxes](/comicbook/03-quantization/02-numbers-in-boxes/) from Issue 3, the geometry of floats is already in your head. This primer is shorter than that one. It exists for two reasons: (a) to nail down *exactly* which FP8 variant the vLLM `--kv-cache-dtype fp8` flag uses (there are two), and (b) to explain why a number system with eight bits of precision can possibly be used for attention math without the model falling over.

The format in question is **FP8 E4M3**. The name is a layout. One bit of sign, four bits of exponent, three bits of mantissa, eight bits total:

$$
\underbrace{\text{S}}_{1\text{ bit}} \;|\; \underbrace{\text{EEEE}}_{4\text{ bits}} \;|\; \underbrace{\text{MMM}}_{3\text{ bits}}
$$

The {{< wiki "number-formats" >}}number-formats{{< /wiki >}} primer in Issue 3 covers the IEEE 754 geometry — the rule that a normal floating-point number with exponent $e$ and mantissa $m$ represents $(-1)^s \cdot 2^{e - \text{bias}} \cdot (1 + m / 2^{\text{mantissa\_bits}})$. We will not re-derive that here. The numerical specifics for E4M3, as actually used on Hopper FP8 tensor cores:

- **Exponent bias**: 7 (so the exponent field $E \in [0, 15]$ represents unbiased $e \in [-7, 8]$).
- **Range**: roughly $\pm 448$ (the largest finite value is $2^8 \cdot (1 + 7/8) = 480$; E4M3 has a NaN sequester at the all-ones-exponent slot that reduces this slightly).
- **Smallest positive normal**: $2^{-6} \approx 0.0156$.
- **Smallest positive subnormal**: $2^{-9} \approx 0.00195$.
- **Mantissa rungs per binade**: $2^3 = 8$ representable points between any power of two and the next.

That last bullet is the one to internalize. Between, say, $1.0$ and $2.0$, FP8 E4M3 has eight representable values: $1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875$. Each step is $0.125$. Between $2.0$ and $4.0$, the rungs are spaced $0.25$ apart. Between $4.0$ and $8.0$, $0.5$ apart. And so on logarithmically. This is the same geometry as FP16 and BF16, but with **far fewer rungs**: where BF16 has 128 rungs per binade and FP16 has 1024, FP8 E4M3 has 8.

```pyplot {id="fp8-rungs" caption="What FP8 E4M3 looks like vs BF16 in the [0.5, 8] range. Each tick mark is a representable value. FP8's 8-rungs-per-binade is sparse but evenly distributed in log space — and on a per-binade basis, the relative spacing is the same as BF16's."}
import numpy as np
import matplotlib.pyplot as plt

# Enumerate all FP8 E4M3 values in [0, 16]
fp8_values = []
for e in range(0, 9):
    base = 2.0 ** e
    for m in range(8):
        v = base * (1 + m / 8)
        if v < 16:
            fp8_values.append(v)
fp8_values = np.array(fp8_values)

# Approximate BF16 in the same range (mantissa=7 bits → 128 rungs per binade)
bf16_values = []
for e in range(-1, 4):
    base = 2.0 ** e
    for m in range(128):
        v = base * (1 + m / 128)
        if 0.5 < v < 16:
            bf16_values.append(v)
bf16_values = np.array(bf16_values)

fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)

# BF16
axes[0].vlines(bf16_values, 0, 1, color='#FF007F', linewidth=0.4)
axes[0].set_xlim(0.5, 16)
axes[0].set_xscale('log', base=2)
axes[0].set_yticks([])
axes[0].set_title(f'BF16  —  {len(bf16_values)} representable values in [0.5, 16]',
                  fontsize=10, fontweight='bold')
axes[0].spines[['top', 'right', 'left']].set_visible(False)

# FP8 E4M3
axes[1].vlines(fp8_values[(fp8_values >= 0.5) & (fp8_values < 16)], 0, 1,
               color='#00A8A8', linewidth=1.4)
axes[1].set_xlim(0.5, 16)
axes[1].set_xscale('log', base=2)
axes[1].set_yticks([])
axes[1].set_xticks([0.5, 1, 2, 4, 8, 16])
axes[1].set_xticklabels(['0.5', '1', '2', '4', '8', '16'])
axes[1].set_xlabel('value (log scale, base 2)')
axes[1].set_title(f'FP8 E4M3  —  {sum((fp8_values >= 0.5) & (fp8_values < 16))} representable values in [0.5, 16]',
                  fontsize=10, fontweight='bold')
axes[1].spines[['top', 'right', 'left']].set_visible(False)

plt.tight_layout()
```

Eight values per octave is not a lot. It is enough to be the *base layer* of a quantization scheme that gets cleverer at the per-block level (MXFP4 does this for 4-bit; the [Hardware Horizon](/comicbook/03-quantization/12-hardware-horizon/) covers it). For FP8 KV-cache in vLLM, though, the math is much simpler: the bits are stored *as-is*, with a single per-tensor scale that defaults to **1.0**, and the attention kernel reads them directly.

## Why E4M3 And Not E5M2

Hopper FP8 ships with two variants, both 8 bits but with different exponent/mantissa splits:

| Format | Exponent | Mantissa | Range | Mantissa rungs/binade | Typical use |
|---|---|---|---|---|---|
| **FP8 E4M3** | 4 | 3 | ±448 | 8 | weights, activations (incl. KV cache) |
| **FP8 E5M2** | 5 | 2 | ±57,344 | 4 | gradients |

The split reflects what each tensor is used for. **E4M3** has more mantissa bits — more *precision* — at the cost of less *dynamic range*. **E5M2** has more exponent bits — much wider dynamic range — at the cost of mantissa precision. Gradients during training can span many orders of magnitude (a vanishing gradient and an exploding gradient differ by $10^8$ or more), so they want E5M2. Activations and KV cache values during inference are usually bounded — *bounded by what?* turns out to be the interesting question for calibration — so E4M3's narrower range is enough, and the extra precision matters.

vLLM's `--kv-cache-dtype fp8` is E4M3 by default. The flag also accepts an explicit `fp8_e4m3` value if you want to be unambiguous, and historically there was a `fp8_e5m2` mode that was less popular and is now mostly deprecated for KV-cache use.

For the rest of this issue, "FP8" means E4M3 unless stated otherwise.

## Per-Tensor Scale = 1.0: What Does That Assume?

Here is the thing about FP8 KV-cache that is easy to gloss over and important to understand: the scale used to convert from the underlying FP16/BF16 value to the stored FP8 value is, by default, **1.0**. No calibration. No per-channel anything. Just a direct cast from BF16 to FP8.

Mechanically, the cast is:

```python
def bf16_to_fp8_uncalibrated(x_bf16):
    # Clip to FP8 range, then cast. No scaling.
    x_clipped = clip(x_bf16, -448.0, 448.0)
    return cast_to_fp8_e4m3(x_clipped)   # rounds to nearest representable value
```

That's the entire quantization. Two lines. The implicit assumption is that the K and V tensor values during attention are *already in roughly the same magnitude range that FP8 E4M3 can represent*. If your K values typically live in $[-2, 2]$, FP8's $[-448, 448]$ range is enormous overkill on the high end but provides plenty of mantissa precision. If your K values happen to spike to $10{,}000$ (unlikely but check), they'll be clipped to $\pm 448$, which is a quantization error you didn't want.

In practice the assumption holds for almost every well-validated model. Look at any of the K-value histograms from [Inside K and V](/comicbook/03-quantization/16-kv-distribution/) and you'll see that the bulk of values sits in $[-5, 5]$, with rare outliers up to maybe $\pm 50$. Everything fits comfortably in E4M3 with no scaling needed.

The outlier discussion from Issue 3 is worth recalling here. The whole reason **per-channel K, per-token V** asymmetric quantization works for low-bit (INT4, 2-bit) is that outliers ruin per-tensor scaling at those bit widths. At FP8, the format has so much dynamic range — three orders of magnitude — that even with the outliers, the bulk values still get usable precision *without per-channel scaling*. This is the geometric reason FP8 KV is the *one* point on the [KV method family tree](/comicbook/03-quantization/15-kv-method-family/) where calibration is genuinely optional.

{{% pullquote type="counter-intuitive" %}}
FP8 KV-cache is the rare quantization method where "scale = 1.0, no calibration" is a defensible default. INT4 needs per-channel scales. INT8 needs at least per-tensor calibration. FP8's range is wide enough that the bulk just *fits*.
{{% /pullquote %}}

This will be the recurring trade-off in the rest of the issue. The simplicity buys you a flag-flip deployment story. It also means you have *no slack* if your model happens to put values outside the comfortable range — and a small number of models do. That is the story of [Chapter 10 — Scale Equals One](../10-calibrate/), where the FlashMLA backend for Kimi-K2.5 turns out to show a consistent downward shift in long-context accuracy that calibration recovers.

## Why Softmax Forgives FP8 Noise

There is a deeper reason the FP8 bargain works at all, and it's worth stating up front because it explains *why* a number system with 8 mantissa rungs per binade can carry attention math at all. The reason is {{< wiki "softmax" >}}softmax{{< /wiki >}}.

Attention computes scores $s_i = q^\top k_i$ and then normalizes them through softmax:

$$
\text{softmax}(s_i) \;=\; \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

The exponential function is **convex and steep**. A score of $5.1$ produces $e^{5.1} \approx 164$. A score of $5.0$ produces $e^{5.0} \approx 148$. The relative weight is $164 / (164 + 148 + \ldots)$ versus $148 / (\ldots)$ — a small difference. Now compare to a much larger score: $7.0$ produces $e^{7.0} \approx 1097$ — about seven times either of the others. The softmax doesn't really care about the precision of $5.0$ vs $5.1$ when $7.0$ dominates. It only really cares about whether one score is **a lot bigger** than another.

This is the heuristic that makes attention forgiving of input-level numerical noise. **The softmax exponentially squashes small score differences.** If your FP8 quantization adds noise of magnitude $0.1$ to your scores, but the scores you actually care about differ from the noise floor by $3$ or $4$, that noise gets exponentially squashed below the relevant signal. It doesn't matter that you've lost 7 bits of mantissa precision on the input side; the softmax only cares about the top of the score distribution, and the top is well-separated from the bulk.

The same heuristic does *not* apply universally. There are two specific places where FP8 noise *can* damage the math, and both of them show up in this issue:

1. **The accumulator inside the attention matmul.** When you're computing $QK^\top$ across thousands of K vectors, you're summing thousands of FP8 multiplications into a running scalar. If the running scalar quietly loses precision once the sum gets big enough, your score estimate is wrong *before* it ever sees the softmax. The softmax can't squash precision loss that's already in the input.
   This is the bug from [the cold open](../01-cold-open/) — and the next two chapters ([The Sum Is Not What You Think](../04-summation-history/) and [The Accumulator Lie](../05-accumulator-lie/)) are the autopsy.

2. **The attention sinks.** {{< wiki "attention" >}}Attention sinks{{< /wiki >}} from [Issue 3's deep dive](/comicbook/03-quantization/16-kv-distribution/) are the few tokens (usually BOS) that receive a disproportionate share of softmax mass. If those specific tokens' K or V vectors get quantized poorly, the sink behavior changes — and downstream attention shifts in ways the model wasn't trained for. Most production FP8 KV deployments keep the first 4-16 tokens at full precision for this reason.

The forgiveness of softmax is a *general* property; the two exceptions are *specific* failure modes. Both matter. The first one is the entire detective story of this issue. The second is part of the calibration story.

## A Worked Example: The Cast In Three Steps

To anchor this, here is a concrete walk-through of what happens when a single Llama-3.1-8B K vector gets stored to FP8 KV cache and read back.

```python
import numpy as np

# A toy K vector for one head, one token. d=128.
# Bulk is roughly N(0, 0.5) with a couple of outlier channels.
np.random.seed(7)
k_bf16 = np.random.randn(128).astype(np.float32) * 0.5
k_bf16[12] = 2.7   # outlier channel a (typical of Llama K)
k_bf16[97] = -3.1  # outlier channel b

# Step 1: clip to FP8 E4M3 range. (Not actually triggered here — bulk is small.)
fp8_max = 448.0
k_clipped = np.clip(k_bf16, -fp8_max, fp8_max)

# Step 2: quantize to nearest FP8 E4M3 representable value.
def quantize_to_fp8_e4m3(x):
    """Return the nearest E4M3 representable value as a float32, with NaN for too-large."""
    out = np.zeros_like(x)
    for i, v in enumerate(x):
        if v == 0.0:
            out[i] = 0.0
            continue
        sign = 1.0 if v > 0 else -1.0
        absv = abs(v)
        # Find binade
        e = int(np.floor(np.log2(absv)))
        e = max(-9, min(e, 8))         # clip to representable exponents
        base = 2.0 ** e
        mantissa = (absv / base - 1.0) * 8
        mantissa = round(mantissa) / 8
        out[i] = sign * base * (1.0 + mantissa)
    return out

k_fp8 = quantize_to_fp8_e4m3(k_clipped)

# Step 3: read back. FP8 is stored. To attend with it, the kernel just multiplies in FP8.
# Quantization error per element:
err = k_fp8 - k_bf16
print(f"max abs error: {np.max(np.abs(err)):.4f}")
print(f"mean abs error: {np.mean(np.abs(err)):.4f}")
print(f"signal RMS:     {np.sqrt(np.mean(k_bf16**2)):.4f}")
print(f"noise RMS:      {np.sqrt(np.mean(err**2)):.4f}")
print(f"SNR (dB):       {20*np.log10(np.sqrt(np.mean(k_bf16**2)) / np.sqrt(np.mean(err**2))):.1f}")
```

The output of this on a typical K vector: max absolute error around $0.03$, mean absolute error around $0.01$, signal-to-noise ratio around 30 dB. **Pleasant**. The signal dominates the noise by a factor of about 30. For the bulk of K values, FP8 E4M3 is effectively lossless within the precision a soft-max-driven attention head cares about.

The outlier channels (channels 12 and 97 in the example) get the *most* relative error — their magnitudes are bigger, they land in a coarser binade — but the absolute error is still small relative to the signal. The bulk channels have the *least* relative error because they live in the tight $[-1, 1]$ range where FP8 still has 8 mantissa rungs per binade.

This is the per-element cost of FP8 KV. It is, indeed, small. The bug that drove this issue is not at this layer. It is at the *aggregation* layer — when those small per-element errors get summed across hundreds of thousands of multiplications inside the attention matmul. Per-element FP8 conversion is fine. The accumulation of FP8 products into a running scalar is where the precision is silently lost.

That's the territory of the next chapter.

## What To Remember

1. **FP8 E4M3** is 1 sign + 4 exponent + 3 mantissa, range ±448, eight mantissa rungs per binade. The format the vLLM `--kv-cache-dtype fp8` flag uses.
2. **E4M3 vs E5M2** is a precision-vs-range trade. E4M3 wins for activations and KV; E5M2 wins for gradients. vLLM KV cache is E4M3.
3. **Per-tensor scale = 1.0** is the uncalibrated default — direct cast from BF16. Defensible because FP8's dynamic range is so wide that the bulk just fits, even with outliers. INT4/INT8 cannot get away with this.
4. **Softmax forgives input noise** because the exponential squashes small score differences. The relevant signal is the *gap* between top scores and bulk scores, not the per-element precision.
5. **Softmax does not forgive accumulation noise.** Precision loss inside the matmul shows up in the input to softmax, where the exponential cannot recover what's been lost. This is the bug we are about to chase.

**Continue to** → [The Sum Is Not What You Think](../04-summation-history/) — the 1965 numerical-analysis primer that explains why summing 100,000 small numbers into a single float can quietly lose most of your precision, and the William Kahan paper that solved it.
