---
title: "The Lloyd-Max Bargain"
description: "Bell Labs, 1957. Stuart Lloyd asks: given n quantization levels and a signal distribution, where should the levels go? The answer reappears 65 years later, inside your LLM."
topics: [quantization, theory]
tags: [lloyd-max, k-means, vector-quantization, bell-labs]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 30
techKind: primer
techNode: lloyd-max
header: default.png
---

## A Telephone Engineer's Question

In **1957**, at Bell Labs in Murray Hill, New Jersey, an engineer named **Stuart Lloyd** wrote a memo titled "Least Squares Quantization in PCM." Bell Labs was, at the time, busy figuring out how to turn analog voice signals into digital ones — a process called Pulse Code Modulation. The question Lloyd was being paid to answer was concrete: if I have a continuous signal (a person's voice) and I have to represent each sample with one of $n$ discrete levels, *where should those levels go*?

The answer Lloyd produced was so good that Bell Labs treated the memo as proprietary. He never formally published it. Three years later, an MIT graduate student named **Joel Max** independently derived the same result and *did* publish, in the 1960 IEEE Transactions on Information Theory. The world called it "the Max quantizer" until 1982, when Bell finally let Lloyd's memo into the open literature. Today we credit both: the **Lloyd-Max quantizer**.

The result they share is the foundation of every lossy compression scheme you have ever used. JPEG. MP3. AAC. H.264. And — as we'll see in [The 1% That Ruins Everything](../06-outliers/) and [Calibration & Blocks](../11-calibration-and-blocks/) — modern LLM quantization, too.

## The Setup

You have a real-valued signal $X$ with probability density $p(x)$ — the distribution of values it takes. You must represent each sample by one of $n$ levels $\hat{x}_1, \dots, \hat{x}_n$ (the **codebook**), choosing the nearest level. There are boundaries $b_0 < b_1 < \dots < b_n$ that partition the real line into $n$ intervals, where everything in interval $i$ gets mapped to level $\hat{x}_i$.

You want to **minimize the expected squared error**:

$$
D = \mathbb{E}[(X - \hat{X})^2] = \sum_{i=1}^{n} \int_{b_{i-1}}^{b_i} (x - \hat{x}_i)^2 \, p(x) \, dx
$$

Lloyd's question: given $p(x)$ and $n$, what choices of $\{b_i\}$ and $\{\hat{x}_i\}$ minimize $D$?

## Two Conditions That Have To Hold

Lloyd took the partial derivatives. The minimum has to satisfy two **necessary conditions**:

**Condition 1 — Nearest-neighbor:** Each boundary $b_i$ sits at the midpoint of two adjacent levels:

$$
b_i = \frac{\hat{x}_i + \hat{x}_{i+1}}{2}
$$

This is intuitive: given fixed levels, every $x$ should map to the *closest* level. Boundaries are midpoints.

**Condition 2 — Centroid:** Each level $\hat{x}_i$ is the conditional mean of $X$ over its interval:

$$
\hat{x}_i = \mathbb{E}[X \mid b_{i-1} \le X < b_i] = \frac{\int_{b_{i-1}}^{b_i} x \, p(x) \, dx}{\int_{b_{i-1}}^{b_i} p(x) \, dx}
$$

This one is *less* intuitive at first but obvious in hindsight. If you've already committed to the partition, the best representative for an interval is its centre of mass under the data distribution — the value that minimizes squared error to all the points inside.

## Lloyd's Iteration: A Loop That Always Works

These two conditions are *coupled*: each depends on the other. Lloyd's algorithm just alternates them.

1. Pick $n$ initial levels.
2. Compute the boundaries as midpoints of adjacent levels (Condition 1).
3. Update each level to the centroid of its interval (Condition 2).
4. Repeat until levels stop moving.

This loop is **monotonically non-increasing** in $D$ — every step either improves or holds. It converges to a *local* minimum (not necessarily global, but always at least a saddle-stable solution).

If you have seen **$k$-means clustering**, you have already seen this algorithm. $k$-means *is* Lloyd's algorithm, generalized to $\mathbb{R}^d$. Lloyd ran it in 1D on a voice signal; Stuart Lloyd's intellectual descendants run it in 768-dimensional space on token embeddings. The math is identical.

## Where The Levels Go For Real Distributions

Let's run Lloyd's iteration on a Gaussian, which is close to the marginal distribution of trained LLM weights.

```pyplot {id="lloyd-iteration" caption="Lloyd's iteration converges to optimal level placement for a Gaussian. The levels concentrate where probability mass concentrates."}
np.random.seed(0)
X = np.random.randn(50000)  # standard normal samples

n_levels = 8
# Initialize with uniformly-spaced levels in [-3, 3]
levels = np.linspace(-3, 3, n_levels)

# Run Lloyd's iteration
history = [levels.copy()]
for it in range(15):
    boundaries = (levels[:-1] + levels[1:]) / 2
    boundaries = np.concatenate(([-np.inf], boundaries, [np.inf]))
    new_levels = np.zeros_like(levels)
    for i in range(n_levels):
        mask = (X >= boundaries[i]) & (X < boundaries[i+1])
        if mask.any():
            new_levels[i] = X[mask].mean()
        else:
            new_levels[i] = levels[i]
    levels = new_levels
    history.append(levels.copy())

print(f"final levels: {np.round(levels, 3)}")

fig, ax = plt.subplots(figsize=(8, 4))
xs = np.linspace(-4, 4, 400)
ax.fill_between(xs, np.exp(-xs**2/2) / np.sqrt(2*np.pi),
                color='#FFD700', alpha=0.45, zorder=1, label='Gaussian density')
for i, lv in enumerate(history):
    ys = np.full_like(lv, -0.02 - 0.012*i)
    ax.scatter(lv, ys, s=18,
               color='#FF007F' if i == len(history)-1 else '#1A1A1A',
               alpha=1.0 if i == len(history)-1 else 0.2,
               zorder=3)
ax.set_yticks([])
ax.set_xlabel("x")
ax.set_title("Lloyd's iteration: 8 levels on a standard Gaussian (top: density, bottom: levels over iterations)")
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.legend()
```

The final levels are not uniformly spaced. They cluster near zero (where the Gaussian has mass) and spread out in the tails. **For a standard normal**, the 8 optimal levels are approximately at:

$$
\pm 0.245,\ \pm 0.756,\ \pm 1.344,\ \pm 2.152
$$

These are called the **Lloyd-Max levels for the Gaussian**. They are not arbitrary — they are the *uniquely optimal* placement of 8 levels under squared error. And if this list looks suspiciously like quantile-based levels, that's because it almost is: the levels are very close to the conditional means within equal-probability bins.

## NF4: Lloyd's Idea, Born Again In 2023

Here's the thing. When **Tim Dettmers** and the QLoRA team needed a 4-bit number format for fine-tuning 65B models in 2023, they didn't pick FP4 (logarithmic), and they didn't pick INT4 (uniform). They picked something called **NF4** — "Normal Float 4". Here's how they defined it: take the **quantiles** of a standard normal distribution at 16 equally-spaced probability values, and use *those* as your representable levels.

```pyplot {id="nf4-vs-int4" caption="NF4's 16 levels (top) versus uniformly-spaced INT4 (bottom). NF4 puts representable values where weights actually live."}
# NF4 levels from the QLoRA paper (Dettmers et al. 2023): the 16 conditional
# centroids of a standard normal split into equal-probability bins, normalised
# so the extremes are exactly ±1. These are baked into bitsandbytes.
nf4_levels = np.array([
    -1.0, -0.6961928, -0.5250730, -0.39491748,
    -0.28444138, -0.18477343, -0.09105056, 0.0,
    0.07958029, 0.16093975, 0.24611230, 0.33791524,
    0.44070983, 0.56261307, 0.72295684, 1.0,
])

int4_levels = np.linspace(-1, 1, 16)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
xs = np.linspace(-1, 1, 400)
ax1.fill_between(xs, np.exp(-xs**2*4) / 0.9, color='#FFD700', alpha=0.35)
ax1.scatter(nf4_levels, [0]*16, s=60, color='#FF007F', zorder=3, label='NF4')
ax1.set_title("NF4 — quantile-spaced levels (normalized)", fontsize=10, loc='left')
ax1.set_yticks([])
ax1.spines[['top','right','left']].set_visible(False)
ax1.axhline(0, color='#1A1A1A', linewidth=0.5)

ax2.fill_between(xs, np.exp(-xs**2*4) / 0.9, color='#FFD700', alpha=0.35)
ax2.scatter(int4_levels, [0]*16, s=60, color='#00A8A8', zorder=3, label='INT4')
ax2.set_title("INT4 — uniformly-spaced levels", fontsize=10, loc='left')
ax2.set_yticks([])
ax2.set_xlabel("normalized value")
ax2.spines[['top','right','left']].set_visible(False)
ax2.axhline(0, color='#1A1A1A', linewidth=0.5)

plt.tight_layout()
```

This is **Lloyd's centroid condition, applied to weights**. The QLoRA team assumed (correctly, as we'll see in [Geometry Of Weights](../05-geometry-of-weights/)) that LLM weight blocks are approximately Gaussian after normalization. Given that, the optimal placement of 16 levels — the *Lloyd-Max placement* — is what NF4 implements.

When you read the QLoRA paper and it says "NF4 is information-theoretically optimal for normally distributed weights" — that's not marketing copy. It's a direct statement about Lloyd's 1957 theorem.

## The Limit Of Scalar Quantization

Lloyd-Max is *one-dimensional*. It quantizes each weight independently. But weights are not independent — adjacent weights in a layer are correlated. You can do strictly better by quantizing *blocks of weights jointly* in $\mathbb{R}^d$. This is **vector quantization** (VQ), and it has its own beautiful history: the **Linde-Buzo-Gray algorithm** (1980), Shannon's foreshadowing in the 1948 channel coding theorem, and a connection to $k$-means we've already noted.

But VQ is computationally expensive, and modern LLM hardware is built around scalar arithmetic. So everyone has converged on **scalar quantization with extremely fine-grained scale factors** — quantize each weight independently, but use a different scale for every block of 32 or 128 weights. This gives most of the benefit of VQ without the cost. We unpack this trick in [Calibration & Blocks](../11-calibration-and-blocks/).

## What The Bargain Actually Is

The "Lloyd-Max bargain" is this: **you do not get to pick how dense your representable values are. Your data does.** The optimal placement of $n$ levels depends entirely on the probability density $p(x)$.

- If your weights are uniform, you want uniform spacing (INT4).
- If your weights are Gaussian, you want centroid-of-bin spacing (NF4).
- If your weights are heavy-tailed, you want logarithmic spacing (FP4).
- If your weights have a wild distribution unique to your particular layer, you want **Lloyd's iteration** run on actual samples from that layer.

The reason this matters for LLMs is that **none of the off-the-shelf formats are quite right**. Real weights have outliers (a heavier tail than Gaussian); real activations have systematic giant spikes that break every standard assumption. So practical methods (GPTQ, AWQ, SmoothQuant) all add tricks on top of base quantization — but those tricks make most sense when you know what they are *correcting for*, which is the gap between "off-the-shelf format" and "what Lloyd-Max would say is optimal for this layer."

Lloyd's 1957 question is still, in 2026, the right question to ask.

**Continue to** → [The Rate-Distortion Bridge](../04-rate-distortion/) for the broader information-theoretic frame that Lloyd-Max sits inside.
