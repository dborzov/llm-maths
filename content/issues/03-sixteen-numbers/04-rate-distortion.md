---
title: "The Rate-Distortion Bridge"
description: "Shannon's lossy-compression frontier — the bend where you stop paying for bits and start paying for fidelity. The single curve that connects MP3, JPEG, and your LLM."
topics: [quantization, information-theory]
tags: [shannon, rate-distortion, entropy]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 40
techKind: primer
techNode: rate-distortion
header: default.webp
---

## A Short Primer, Because You Already Know The Pieces

You're assumed to already know information theory in its lossless flavour: a source with entropy $H(X)$ bits per symbol cannot be losslessly compressed below $H(X)$ bits per symbol on average. That's [Shannon 1948](https://en.wikipedia.org/wiki/A_Mathematical_Theory_of_Communication), the founding theorem.

This article is a five-minute upgrade to the **lossy** version. We need it because LLM quantization *is* lossy compression: you are throwing information away on purpose. Shannon's lossy theorem tells you the floor of how much fidelity you have to sacrifice for a given bitrate. The curve has a name (**rate-distortion**) and a shape (always convex, decreasing) that explains, in one picture, why all the quantization methods you'll meet in this issue make the trade-offs they do.

## The Setup

You have a source $X$ with distribution $p(x)$. You will encode it as a discrete representation using $R$ bits per symbol. After decoding you get $\hat{X}$. Pick a **distortion measure** — usually squared error:

$$
D = \mathbb{E}\big[(X - \hat{X})^2\big]
$$

**Shannon's question:** for a given rate budget $R$, what is the smallest achievable distortion $D$?

**Shannon's answer:** the **rate-distortion function** $R(D)$, defined as

$$
R(D) = \min_{p(\hat{x}|x): \mathbb{E}[(X-\hat{X})^2] \le D} I(X; \hat{X})
$$

— the minimum mutual information between $X$ and any reconstruction $\hat{X}$ achieving distortion at most $D$. This is achievable (you can build codes that get arbitrarily close) and unbeatable (no code can do strictly better). It is the lossy analog of $H(X)$.

For Shannon's two foundational results to be friends, think of it this way:

| | Lossless | Lossy |
|---|---|---|
| Source bound | $H(X)$ bits/symbol | $R(D)$ bits/symbol |
| Quantity allowed | none — must be exact | up to $D$ distortion |
| Bound from below by | entropy | mutual information minimum |

## The Gaussian Result You Should Remember

For a Gaussian source with variance $\sigma^2$ and squared-error distortion, the rate-distortion function has a clean closed form:

$$
R(D) = \frac{1}{2} \log_2 \frac{\sigma^2}{D} \quad \text{for } 0 < D < \sigma^2
$$

Rearrange: $D = \sigma^2 \cdot 2^{-2R}$. Every additional bit cuts distortion by a factor of 4.

```pyplot {id="rd-curve" caption="The rate-distortion curve for a Gaussian source. Each bit halves the standard deviation of the error."}
sigma2 = 1.0
R = np.linspace(0.05, 6, 200)
D = sigma2 * 2**(-2 * R)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.fill_between(R, D, sigma2, color='#FFD700', alpha=0.35, label='unachievable region')
ax.plot(R, D, color='#FF007F', linewidth=2.5, label='R(D) — Shannon limit')
# Mark some operating points
for r, d in [(1, sigma2 * 2**(-2)), (2, sigma2 * 2**(-4)),
             (4, sigma2 * 2**(-8)), (6, sigma2 * 2**(-12))]:
    ax.scatter([r], [d], s=70, color='#1A1A1A', zorder=4)
    ax.annotate(f"R={r}", (r, d), xytext=(8, -6), textcoords='offset points',
                fontsize=9, fontweight='bold')
ax.set_yscale('log')
ax.set_xlabel("Rate R (bits per symbol)")
ax.set_ylabel("Distortion D (MSE)")
ax.set_title("Shannon's rate-distortion frontier for a Gaussian source")
ax.legend(loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.3)
```

This curve is the **frontier of all possible lossy codes**. Every Lloyd-Max quantizer, every JPEG, every NF4 weight scheme — all of them are *points below the frontier*, each chasing the limit but never quite reaching it.

## Where Lloyd-Max Sits On The Curve

A scalar Lloyd-Max quantizer with $n$ levels achieves a rate of exactly $\log_2 n$ bits per symbol. So 4 bits → 16 levels, 8 bits → 256 levels. The distortion you get is determined by the data distribution and the optimal level placement.

For Gaussian sources, the scalar Lloyd-Max quantizer hits roughly **1.5 dB above** Shannon's frontier. That gap — called the **space-filling loss** — is what *vector* quantization recovers. Linde-Buzo-Gray (1980) generalized Lloyd-Max to higher dimensions and showed you can squeeze closer to Shannon by quantizing $d$ samples jointly. The gap closes as $d \to \infty$.

For LLM weight quantization in 2026, almost everyone uses scalar quantizers (with per-block scales). They give up the ~1.5 dB. The reason is **hardware**: GPU matmul kernels are scalar. You can't economically do vector decoding inside a dot-product pipeline. So we accept the scalar penalty and recover what we can through **block-wise scales** (covered in [Calibration & Blocks](../11-calibration-and-blocks/)).

## Why This Matters For LLMs

Here is the practical, useful summary you should take away.

1. **The bit budget is fixed and tight.** A 70B-parameter model at 16 bits is 140 GB. At 4 bits, it's 35 GB. There is no cheating — those bits are the rate.

2. **Distortion in weights $\ne$ distortion in model output.** Squared error on the weights themselves is not what we ultimately care about — we care about KL-divergence on the model's output distribution. Modern quantization methods (especially GPTQ — see [Brain Surgery Returns](../08-brain-surgery/)) explicitly optimize a *weighted* squared error that approximates the loss-function distortion, not raw weight-MSE. This is one reason GPTQ outperforms plain Lloyd-Max round-to-nearest.

3. **The Gaussian frontier is a benchmark, not the truth.** Real LLM weights are not Gaussian. They have heavier tails. The actual rate-distortion function for them is steeper than the Gaussian one — *less forgiving*. This is part of why the "1% problem" of [outliers](../06-outliers/) is so brutal: a heavy-tailed source has rate-distortion that is dominated by the tails.

4. **Diminishing returns.** Every bit you remove past about 4 hurts disproportionately more. The rate-distortion curve is convex — bending downward fast near the limit. This is why we keep stopping near 4 bits, not 2 bits. (Researchers have tried 2 and even 1-bit quantization with mixed success; the math says it gets exponentially harder.)

## The Bridge

Rate-distortion is the bridge between **lossless** information theory (which you already know) and the practical quantization theory of the next several chapters. Whenever someone shows you a new "PPL versus bits-per-weight" curve in a quantization paper, what they are really plotting is *an upper bound on the achievable rate-distortion curve* for that particular model and dataset. The whole field is, in some sense, a race to bend that curve down toward Shannon.

**Continue to** → [The Geometry Of Weights](../05-geometry-of-weights/) for the empirical distribution that all this theory has to actually live with.
