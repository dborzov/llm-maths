---
title: "Numbers In Boxes"
description: "A primer on float formats: why FP16 is not just half of FP32, why representable numbers cluster near zero, and what \"FP4 has 16 values\" actually means."
topics: [quantization, number-formats]
tags: [ieee-754, fp16, bf16, fp8, fp4]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 20
techKind: primer
techNode: numbers
header: default.webp
---

## The Berkeley Cellar

In 1976, in a basement office at UC Berkeley, **William Kahan** was angry. He was angry because every computer manufacturer had its own incompatible floating-point implementation, and he had personally seen students lose months of work to "the same calculation" producing different answers on a DEC VAX, an IBM 360, and a Cray-1.

So he agreed to chair a committee. The committee — IEEE P754 — would produce a *single* standard for binary floating-point. It would take six years and a fight with the entire computer industry, but in 1985 they published **IEEE 754**, the geometry of computer numbers that every CPU and GPU has used ever since.

Kahan won a Turing Award for it in 1989. He also, almost by accident, locked in a peculiar choice that quantum-leaped into significance four decades later when neural networks got too big to fit on a graphics card.

The choice was this: **the spacing between consecutive representable numbers gets smaller near zero, and larger far from zero**.

This sounds obvious — of course you want more precision near zero, that's where most numbers in physics live. But it has a consequence so important to LLMs that you can spend a whole career missing it. *Floats are not a uniform grid.* They are a logarithmic grid, with two intersecting populations: the **subnormals** crammed near zero, and the **normal range** that doubles in spacing every time the exponent ticks up.

## The Anatomy Of A Float

Every IEEE 754 float, regardless of width, has the same three parts:

```
[ sign | exponent | mantissa ]
```

- The **sign** bit picks positive or negative.
- The **exponent** is a small integer that scales by powers of two.
- The **mantissa** ("significand") is the fractional part inside the chosen power-of-two interval.

The value is:

$$
(-1)^{\text{sign}} \cdot 2^{\text{exponent} - \text{bias}} \cdot (1 + \text{mantissa})
$$

Three formats matter for LLMs:

| Format | Sign | Exponent | Mantissa | Total | Range | Precision |
|---|---|---|---|---|---|---|
| **FP32** | 1 | 8 | 23 | 32 | ±10³⁸ | ~7 decimals |
| **FP16** | 1 | 5 | 10 | 16 | ±65,504 | ~3 decimals |
| **BF16** | 1 | 8 | 7 | 16 | ±10³⁸ | ~2 decimals |
| **FP8 E4M3** | 1 | 4 | 3 | 8 | ±448 | ~1 decimal |
| **FP8 E5M2** | 1 | 5 | 2 | 8 | ±57,344 | ½ decimal |
| **FP4 E2M1** | 1 | 2 | 1 | 4 | ±6 | 16 values total |

Notice that FP16 and BF16 both use 16 bits. **They are not the same format.** FP16 spends more bits on the mantissa (precision), BF16 spends more on the exponent (range). For neural networks, BF16 won the format war for *training* because gradient magnitudes can swing wildly across many orders of magnitude — having dynamic range matters more than mantissa precision. For *inference*, FP16 is still common because activations live in a narrower band.

## What Representable Numbers Actually Look Like

Let's plot every positive FP8-E4M3 value on a number line. There are only 240-ish of them. The pattern is the entire story.

```pyplot {id="fp8-number-line" caption="Every positive FP8-E4M3 representable value. The grid gets twice as dense each time you cross a power-of-two boundary toward zero."}
def fp8_e4m3_values():
    """Enumerate all positive non-NaN representable FP8-E4M3 values."""
    out = []
    # Subnormal range: exponent bits = 0
    for m in range(8):
        if m == 0:
            continue  # zero handled separately
        out.append(2**(-6) * (m / 8))
    # Normal range: exponent bits 1..15 (15 reserved for NaN by convention, but include it)
    for e in range(1, 15):
        for m in range(8):
            val = 2**(e - 7) * (1 + m/8)
            out.append(val)
    return out

vals = fp8_e4m3_values()
print(f"total positive representable values: {len(vals)}")

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.scatter(vals, [0]*len(vals), s=12, color='#FF007F', zorder=3)
for x in [1/64, 1/8, 1, 8, 64, 256]:
    ax.axvline(x, color='#1A1A1A', linewidth=0.5, alpha=0.3)
ax.set_xscale('log')
ax.set_yticks([])
ax.set_xlabel("value")
ax.set_title("Every positive FP8-E4M3 representable value")
ax.spines[['top', 'right', 'left']].set_visible(False)
```

The dots are **uniformly spaced inside each octave** (each power-of-two interval) — but the octaves themselves double in width. So 1/64 to 1/32 has the same number of representable values as 64 to 128. The format puts its representable values where the values *are*: most densely near zero, where small differences matter.

This is profound. **You're not allowed to "use" a number that isn't a dot.** Every multiplication and addition in your model will be silently snapped to the nearest dot. The art of choosing a number format is the art of choosing where the dots land.

## "FP4 Has Sixteen Values" — Literally

FP4 (the OCP "E2M1" format) uses **2 exponent bits and 1 mantissa bit**, plus the sign. The total count of distinct values is exactly:

$$
2 \text{ signs} \times 2^2 \text{ exponents} \times 2^1 \text{ mantissas} = 16
$$

Let's see them all.

```pyplot {id="fp4-all-values" caption="All 16 values of FP4 E2M1. That is the entire format."}
import itertools

def fp4_e2m1_values():
    out = set()
    out.add(0.0)
    # exponent bits = 0 -> subnormal: 0.5 * (m/2)
    for m in range(2):
        if m == 0:
            continue
        out.add(0.5 * (m / 2))
    # exponent bits 1..3 -> normal: 2^(e-1) * (1 + m/2)
    for e in range(1, 4):
        for m in range(2):
            val = 2**(e - 1) * (1 + m / 2)
            out.add(val)
    # symmetrize
    pos = sorted(out)
    full = sorted([-v for v in pos if v > 0] + pos)
    return full

vals = fp4_e2m1_values()
print(f"total FP4 values: {len(vals)}")
print(f"values: {vals}")

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.scatter(vals, [0]*len(vals), s=80, color='#FF007F', zorder=3)
for v in vals:
    ax.annotate(f"{v}", (v, 0), xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=8)
ax.set_yticks([])
ax.set_xlabel("value")
ax.set_title("Every value of FP4 E2M1 (n=16)")
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.axhline(0, color='#1A1A1A', linewidth=0.5)
```

That's it. The entire number system on which 2025-era frontier models are *trained* has fewer values than the number of letters in the alphabet. Half of them are in [-2, 2]. The largest is 6. The smallest non-zero positive value is 0.25.

You should be alarmed.

How does this possibly work? The answer is **microscaling** — every block of 32 weights shares a separate FP8 scale factor, so the 16 representable values get *stretched and shifted* to land where the weights in that block actually live. We get to that idea properly in [Calibration & Blocks](../11-calibration-and-blocks/), and we see the silicon side of it in [Hardware Horizon](../12-hardware-horizon/).

## The Integer Cousins: Two's Complement And Scales

Floats are not the only option. The other family is **integer** quantization: pick a scale factor $s$ and a zero point $z$, and represent $x \approx s \cdot (q - z)$ where $q$ is a small integer.

For INT8 you have 256 levels (typically $-128 \le q \le 127$). For INT4: 16 levels. So **INT4 and FP4 have the same number of distinct values**. They differ in how those values are *spaced*:

- **INT4** spaces them uniformly.
- **FP4** spaces them logarithmically (denser near zero).

```pyplot {id="int4-vs-fp4" caption="INT4 (top) vs FP4 (bottom). Both have 16 values. Notice the dramatically different density near zero."}
vals_fp4 = sorted([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
vals_fp4 = sorted([-v for v in vals_fp4 if v > 0] + vals_fp4)

# INT4 with scale chosen to match FP4 max (6)
vals_int4 = [(q - 0) * (6 / 7) for q in range(-8, 8)]  # 16 levels symmetric

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
for ax, vals, title, color in [(ax1, vals_int4, 'INT4 (uniform)', '#00A8A8'),
                                (ax2, vals_fp4, 'FP4 (logarithmic)', '#FF007F')]:
    ax.scatter(vals, [0]*len(vals), s=70, color=color, zorder=3)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, loc='left')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.axhline(0, color='#1A1A1A', linewidth=0.5)
ax2.set_xlabel("value")
plt.tight_layout()
```

Which one is *better* depends entirely on what you're representing. Uniform spacing is optimal if your data is uniformly distributed (no real data is). Logarithmic spacing is optimal if your data follows a roughly Gaussian or Laplacian distribution centered on zero — and as we'll see in [Geometry Of Weights](../05-geometry-of-weights/), that's almost exactly what neural network weights look like.

But "almost exactly" isn't the same as "exactly". The *optimal* placement of $n$ levels for a given distribution is the subject of [The Lloyd-Max Bargain](../03-lloyd-max/) — and the answer turns out not to be uniform *or* logarithmic, but something more interesting.

## What FP16 Costs You

It is tempting, at this point, to ask: if FP4 only has 16 values, why was FP16 the *training* default for so long? Why not just use INT8 from day one?

The answer is twofold:

1. **Training requires huge dynamic range.** Gradients during backprop can be as small as 10⁻⁸ and as large as 10⁵. INT8 has at most ~256 levels of range. FP16 has roughly five orders of magnitude (~10⁻⁵ to 10⁵). That's *barely* enough — the original mixed-precision training paper (Micikevicius et al., 2018) introduces a "loss scale" trick precisely because FP16 gradients underflow without it.

2. **Multiplications need precision in the mantissa.** The reason FP16's mantissa is 10 bits, not 5, is that when you multiply two FP16 numbers, the precision of the result is bounded by the smaller of the two mantissas.

When **BF16** arrived (Google, 2017), it traded mantissa precision for FP32-equivalent range. The bet was: for neural network training, you need more range than you need precision. The bet paid off. By 2020, BF16 was the default training format on TPUs and high-end GPUs.

This question — *which bits should I spend where, range or precision?* — is the entire design space of low-bit number formats. It's why we have **E4M3** (more precision, smaller range) for weights and activations and **E5M2** (more range, less precision) for gradients in FP8 training. It's why **FP4 E2M1** picks 2 exponent bits, not 1 (a choice with massive consequences). It's why **NF4** ("Normal Float 4", invented for QLoRA) abandons IEEE 754 geometry entirely and just picks 16 values from the quantiles of a standard normal distribution — and we'll meet that idea soon.

## What To Remember

1. Floats are a **logarithmic** grid, not a uniform one. Spacing doubles each octave.
2. FP16 vs BF16 vs FP8 are not "smaller versions of FP32" — they are different geometric objects, each with different range/precision trade-offs.
3. **FP4 has exactly 16 distinct values.** This works only because of **scale factors** that are applied per-block to position those 16 dots where the weights actually are.
4. Integer quantization spaces its levels *uniformly*; float quantization spaces them *logarithmically*. Which is better depends on the data distribution.

This last question — what is the actual distribution of weights and activations in a trained LLM? — is what we tackle next, after one more theoretical detour into the math of *optimal* quantization.

**Continue to** → [The Lloyd-Max Bargain](../03-lloyd-max/) for a 1957 result from Bell Labs that tells us where to put the dots.
