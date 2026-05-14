---
title: "The Axis Question"
description: "Per-tensor, per-row, per-channel, per-token, per-block — every quantization method has to pick an axis, and the choice is the difference between a working scheme and a broken one. A primer on what each axis means, when it's right, and the modern stack's emerging convention of mixing them."
topics: [quantization]
tags: [per-channel, per-token, per-block, per-tensor, axes, scaling-granularity]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 180
techKind: primer
techNode: quantization-axes
header: default.webp
---

## A Picture That Keeps Coming Back

Walk through this issue and the same panel keeps showing up — sometimes for weights, sometimes for activations, sometimes for K, sometimes for V. The panel is always: *here's a 2D matrix of values; the heterogeneity lives along this particular axis; therefore quantize along that axis.*

It shows up in [The 1% That Ruins Everything](../06-outliers/) as the moment Dettmers realizes per-tensor scaling crushes the bulk and per-row scaling rescues it. It shows up in [Calibration & Blocks](../11-calibration-and-blocks/) as the per-block-vs-per-row trade-off. It shows up in [KV Cache Tyranny](../10-kv-cache/) as the per-channel-K-vs-per-token-V asymmetry. It shows up implicitly in every method in [The Method Family Tree](../09-method-family-tree/) and [The KV Method Family Tree](../15-kv-method-family/).

This article is the explicit primer on the move. **What does it mean to quantize "along an axis"? Why does the axis choice matter so much? And how do you pick the right one?**

The short answer: the axis you quantize along should match the axis along which the data is *heterogeneous*. Get the match right and 4-bit works. Get it wrong and even 8-bit collapses.

## The Whole Game In One Sentence

Every quantization scheme is fundamentally a choice of:

> **A partition of the tensor into groups, where each group gets one shared scale (and possibly one shared zero-point).**

That's it. The bits-per-value question (4 vs 8) is one knob; the partition is the other. The two are *separate*. INT4 with a smart partition can outperform INT8 with a dumb partition.

The axis question is just: **how do you partition?**

## A Tour Of The Standard Partitions

Take a generic 2D weight matrix $W$ of shape $(m, n)$ — say $m$ output dimensions × $n$ input dimensions. There are six standard ways to partition it.

### 1. Per-Tensor

One scale for the whole thing. Every value $W_{ij}$ is quantized using the same scale $s$.

$$
s = \frac{\max_{ij} |W_{ij}|}{q_\max}, \qquad \hat W_{ij} = \text{round}(W_{ij} / s) \cdot s
$$

**Cost:** 1 scalar of metadata. Cheapest possible.
**Failure mode:** If any row or column has a much larger range than the others, that row/column dominates the scale and everything else gets crushed.
**Where used:** Almost nowhere in modern LLM quantization. Per-tensor was the default for INT8 CNN inference circa 2018; it broke as soon as transformers showed up. Survives today only as the absolute-baseline RTN configuration nobody actually deploys.

### 2. Per-Row (Per-Output-Channel)

One scale per row of $W$. Each output dimension gets its own scale.

$$
s_i = \frac{\max_j |W_{ij}|}{q_\max}, \qquad \hat W_{ij} = \text{round}(W_{ij} / s_i) \cdot s_i
$$

**Cost:** $m$ scalars of metadata. Negligible.
**When right:** When different output dimensions have different overall magnitudes (typical for trained weights). Per-row absorbs that heterogeneity at almost no cost.
**Where used:** The default for *weight* quantization in essentially every method. GPTQ, AWQ, and friends all do per-row at minimum.

### 3. Per-Column (Per-Input-Channel)

One scale per column of $W$. Each input dimension gets its own scale.

$$
s_j = \frac{\max_i |W_{ij}|}{q_\max}, \qquad \hat W_{ij} = \text{round}(W_{ij} / s_j) \cdot s_j
$$

**Cost:** $n$ scalars of metadata. Negligible.
**When right:** When the input feature dimensions have different statistics — the natural axis for *activations* (where outliers cluster per-channel; see [The Geometry of Weights](../05-geometry-of-weights/)).
**Where used:** Standard for activation quantization in SmoothQuant, ZeroQuant. AWQ uses per-column statistics on activations to *choose* per-row scales on weights (a clever cross-axis move).

### 4. Per-Token

For activations specifically. The activation tensor is shape $(T, d)$ where $T$ is tokens and $d$ is channels. **Per-token** is the same as per-row of the activation matrix: one scale per sequence position.

**Cost:** $T$ scalars of metadata, computed online during inference.
**When right:** When different tokens have wildly different magnitudes. Crucial for V vectors (KIVI's V-side observation; see [Inside K and V](../16-kv-distribution/)).
**Where used:** Per-token activation quantization in SmoothQuant. Per-token V quantization in KIVI, KVQuant, QServe.

### 5. Per-Block (1D Group)

Take a row of $W$ and split it into contiguous blocks of $g$ values (typically $g = 32, 64, 128$). Each block gets its own scale.

$$
s_{i, b} = \frac{\max_{j \in B_b} |W_{ij}|}{q_\max}
$$

**Cost:** $\lceil n/g \rceil \cdot m$ scalars. For $g = 128, m = 4096, n = 4096$: $32 \cdot 4096 = 131{,}072$ scalars per layer. At FP16 = 256 KB. Compared to the underlying 4-bit matrix (8 MB), that's 3% overhead.
**When right:** *Almost always*, on top of per-row. Per-block captures local heterogeneity within a row (some 128-element windows are tighter than others). The per-block trick is what lets 4-bit weight quantization actually work.
**Where used:** Universal. GPTQ, AWQ, NF4, GGUF — all use per-block scales with $g \in \{32, 64, 128\}$.

We discussed why $g = 128$ is the conventional sweet spot in [Calibration & Blocks](../11-calibration-and-blocks/) — it's the cache-line-aligned, metadata-cheap, accuracy-preserving local minimum.

### 6. Per-Block (2D Tile)

A genuinely 2D partition: tile $W$ into $g \times g$ blocks, one scale per tile. Or non-square ($g_1 \times g_2$).

**Cost:** $\frac{mn}{g_1 g_2}$ scalars. For $g_1 = g_2 = 32$: 16K scalars per layer. Cheap.
**When right:** When *both* row and column axes have local heterogeneity. Useful for KV cache and for some attention patterns.
**Where used:** Less common than 1D blocks; some GGUF K-quant variants and some KV methods. A frontier area.

## A Pictorial Walk Through The Same Tensor

Let's quantize the same matrix six different ways and look at what happens.

```pyplot {id="six-axes" caption="The same matrix quantized to INT4 with six different partitions. Per-tensor crushes everything; per-row helps; per-block recovers most of the original."}
np.random.seed(2)
m, n = 8, 64
W = np.random.randn(m, n) * 0.1

# Inject row-level heterogeneity (some rows much bigger)
for i in [2, 5]:
    W[i, :] *= 6
# Inject column-level heterogeneity (some cols much bigger)
for j in [10, 35, 50]:
    W[:, j] *= 8
# Inject within-row block heterogeneity
W[0, :16] *= 4
W[0, 32:48] *= 0.2

def q_int4(W, scale_fn):
    """Generic quantizer: scale_fn(W) returns a same-shape array of scales."""
    s = scale_fn(W)
    s = np.clip(s, 1e-9, None)
    return np.clip(np.round(W / s * 7), -7, 7) * s / 7

# Six scaling strategies
def per_tensor(W):     return np.full_like(W, np.abs(W).max())
def per_row(W):        return np.broadcast_to(np.abs(W).max(axis=1, keepdims=True), W.shape).copy()
def per_col(W):        return np.broadcast_to(np.abs(W).max(axis=0, keepdims=True), W.shape).copy()
def per_block_1d(W, g=16):
    out = np.zeros_like(W)
    for i in range(W.shape[0]):
        for b in range(0, W.shape[1], g):
            out[i, b:b+g] = np.abs(W[i, b:b+g]).max()
    return out
def per_block_2d(W, g1=4, g2=16):
    out = np.zeros_like(W)
    for i0 in range(0, W.shape[0], g1):
        for j0 in range(0, W.shape[1], g2):
            out[i0:i0+g1, j0:j0+g2] = np.abs(W[i0:i0+g1, j0:j0+g2]).max()
    return out

results = {
    "per-tensor":     q_int4(W, per_tensor),
    "per-row":        q_int4(W, per_row),
    "per-column":     q_int4(W, per_col),
    "per-block 1D":   q_int4(W, per_block_1d),
    "per-block 2D":   q_int4(W, per_block_2d),
}

fig, axes = plt.subplots(3, 2, figsize=(11, 8))
axes_flat = axes.flatten()

for ax, (name, Wq) in zip(axes_flat, [("original (FP)", W)] + list(results.items())):
    err = ((W - Wq) ** 2).mean() if name != "original (FP)" else 0
    im = ax.imshow(np.abs(Wq), aspect='auto', cmap='magma', vmin=0, vmax=np.abs(W).max())
    ax.set_title(f"{name}    MSE={err:.4f}" if err else name)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()

print("Quantization MSE summary:")
for name, Wq in results.items():
    err = ((W - Wq) ** 2).mean()
    print(f"  {name:14s}: {err:.5f}")
```

The MSE column tells the whole story. Per-tensor is catastrophic. Per-row is much better but doesn't catch *within*-row heterogeneity. Per-column is better than per-tensor but worse than per-row for this kind of data. Per-block 1D and 2D are both substantial improvements. The 2D block sometimes wins, sometimes loses to 1D depending on the structure.

## How To Pick

Rule of thumb: **the partition should match the axis along which the data is heterogeneous.**

Practical translation:

| Tensor type | Heterogeneous along… | Right partition |
|---|---|---|
| Trained weights $W$ | rows + within-row | per-row + per-block(1D) |
| FFN activations | feature dims (channels) | per-column + per-token |
| K (per layer) | feature dims (channels) | per-channel |
| V (per layer) | sequence positions (tokens) | per-token |
| Embedding tables | both rows and columns | per-block(2D) sometimes |
| Gradient tensors | wide dynamic range | per-tensor with FP8 (range matters more than partition) |

The rule is empirical, not theoretical — but it has held up across two years of method-paper publications. The methods that work are the ones that match axes.

## Why "Per-Row + Per-Block" Won For Weights

The dominant convention for weight quantization in 2026 is **per-row scales for the *coarse* placement, plus per-block scales for the *fine* placement, with block size 32 to 128**. Every modern weight quantizer uses some flavor of this.

Why this combination?

- **Per-row** alone gets you about 60% of the way to FP16 quality at 4-bit, by absorbing inter-row scale heterogeneity.
- **Per-block within row** gets you another 35% by absorbing intra-row heterogeneity.
- The two stack additively because they capture different statistical properties.
- The metadata cost is modest: per-row scales are $m$ FP16 values (~8 KB for a 4096-row layer); per-block scales add ~256 KB at $g=128$, on top of 8 MB of underlying 4-bit weights.

What about per-column? Almost nobody uses per-column for weights. The reason: weight columns correspond to *input features*, and those features are already going to be normalized by an upstream LayerNorm, so per-column heterogeneity has been pre-absorbed. Per-row corresponds to *output features*, which haven't been normalized yet, and where heterogeneity persists.

For activations the picture is reversed: per-column (= per-channel = per-feature) is right because the activation outliers cluster per-channel. We covered why in [The Geometry of Weights](../05-geometry-of-weights/) and [The 1% That Ruins Everything](../06-outliers/).

## What Rotations Buy You: A Different Angle On The Same Question

A subtle interpretation of [the rotation methods](../17-rotations/) is that they make *the partition choice less important*. After a Hadamard rotation, no axis is particularly outlier-bearing — the outliers are spread out — so per-tensor or per-row scaling works almost as well as per-block.

This is, in some sense, the opposite move from "match the partition to the heterogeneity": rotation methods *eliminate* the heterogeneity, so you don't have to match anything. Both moves are valid; they trade off differently. Block-axes-aware methods like KIVI keep the original basis but match the partition to its quirks. Rotation-axes-blind methods like QuaRot move into a basis where any partition works.

For the next generation of quantization methods, you'll see *both* layered: rotate first, then partition cleverly within the rotated basis. The two moves compose.

## A Practical Selection Heuristic

If you are designing a new quantization scheme today, here's a starting checklist:

1. **What's heterogeneous?** Look at min/max per row, per column, per block. Heatmap it. The biggest spread tells you the natural partition axis.
2. **Is the heterogeneity persistent across inputs?** If yes (like K's outlier channels), commit to a per-channel scale at calibration time. If no (like V's outlier tokens), compute scales online per-token.
3. **Can you afford per-block?** If your bit budget is below 8-bit, almost always yes — the metadata cost is small relative to the bit savings. Default to per-row + per-block(128).
4. **Is rotation an option?** If your stack supports kernel-fused Hadamards, rotate first. The post-rotation heterogeneity is usually low enough that even per-row works fine.
5. **Mix the partitions.** Some layers want per-channel, some want per-token, some want both. Layer-adaptive bit allocation (KVTuner-style) extends to layer-adaptive *partition* choice.

## What To Remember

1. **The partition is half the quantization decision.** Bits-per-value is the other half. They are independent.
2. **Match the partition axis to the heterogeneity axis.** Per-row for weights; per-channel for activation outlier dims; per-token for V; per-channel for K.
3. **Per-row + per-block(128) is the universal weight default.** Almost every modern method uses it.
4. **Per-token is the activation default.** Computed online; one scale per sequence position.
5. **Rotations make partition less important** by eliminating the heterogeneity in the first place. Stack with care.
6. The **axis question** is the single thread that runs through every method in [the weight family](../09-method-family-tree/) and [the KV family](../15-kv-method-family/). It is the cross-cutting concept that explains why two methods with the same bit budget can differ by 3× in perplexity.

**Continue to** → [Hardware Horizon](../12-hardware-horizon/) — the closing chapter, where everything in this issue lands on actual silicon. Native FP8 and FP4 tensor cores, microscaling block formats baked into hardware, and the timeline of how the silicon caught up with all the quantization tricks the software community had been spinning for half a decade.
