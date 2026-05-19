---
title: "Rotations: spread outlier channels across all dimensions at once"
short_title: "Rotations"
description: "Insert a Hadamard rotation before quantizing and the four giant channels in a 4096-dim activation get spread uniformly across all 4096 dimensions — no outlier dominates the scale."
blurb:
  - "The rotation identity: Y = XWᵀ = (XR)(WR)ᵀ. Insert orthogonal R and R⁻¹ that cancel inside the dot product, then quantize the rotated tensors."
  - "Hadamard transform over a general orthogonal: O(d log d) vs O(d²) operations — fast enough to fuse into the attention kernel."
  - "After Hadamard, the max absolute value of Hx is bounded by ‖x‖₂ √(log d / d) — far smaller than ‖x‖∞ when x has outliers."
  - "QuIP, QuaRot, SpinQuant, TurboQuant all use this same algebraic identity; they differ in where they insert the rotation and whether they learn it."
topics: [quantization, geometry]
tags: [rotations, hadamard, quip, quarot, spinquant, turboquant]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 170
techKind: primer
techNode: rotations
header: 17-rotations.webp
---

## A Trick You Already Know

You probably first met this trick in a high-school physics class. You had a vector with a big *x* component and a small *y* component, and the textbook said: *rotate the coordinate system 45 degrees and now both components are equal.* The math hadn't changed; the *description* had. Length was preserved; angles were preserved; everything you cared about was preserved. The numbers on the page just got more democratic.

Now do the same thing to a transformer activation. Suppose the activation is a 4096-dimensional vector and four of its components — let's say dimensions 412, 1318, 2904, and 3700 — are 50 times larger than the rest. Quantizing this vector to 4 bits with one shared scale is hopeless: the four giants set the scale, and everything else gets crushed.

But: rotate the basis. A specific kind of rotation called a **Hadamard transform** does what 45-degree rotation does in two dimensions, but in 4096. After the rotation, the *same vector* has 4096 components, all of roughly the same magnitude. The four giants have been *spread* uniformly across all dimensions. Now per-channel quantization works.

This is the entire idea of the rotation methods — **QuIP**, **QuaRot**, **SpinQuant**, **TurboQuant**. Same trick as physics class, applied to the giant-outlier problem of LLM activations. This article unpacks why it works, what's special about the Hadamard transform specifically, and which method makes which trade-off.

## The Mathematical Setup

Recall that every linear layer in a transformer computes

$$
Y = X W^\top
$$

where $X$ is the input activation and $W$ is the {{< wiki "transformer-weights" >}}weight matrix{{< /wiki >}}. We want to quantize both $X$ and $W$, and we know that $X$ has a few outlier channels making this hard.

Now insert a pair of orthogonal matrices $R$ and $R^{-1} = R^\top$ that cancel each other:

$$
Y = X W^\top = X (R R^\top) W^\top = (X R) (W R)^\top
$$

The product is *unchanged*. But $X R$ is the rotated activation, and $(W R)$ is the rotated weight. **If we picked $R$ well, both $XR$ and $WR$ are easier to quantize than the originals.** The matmul still produces the right output, because the rotations cancel inside the dot product.

This is the same algebraic identity as the AWQ/SmoothQuant migration trick (see [The Method Family Tree](../09-method-family-tree/)), except $R$ is *orthogonal* (rather than diagonal) and operates on whole basis directions (rather than per-channel scales). The diagonal-scale trick generalizes to the full orthogonal-rotation trick.

## Why The Hadamard Transform Specifically?

Of all the orthogonal matrices in $\mathbb{R}^{d \times d}$, why pick a Hadamard? Three reasons.

**1. Speed.** A general orthogonal $R$ requires $O(d^2)$ operations to apply. For $d = 4096$, that's $\sim 16$ million operations *per token*. Way too slow to insert into an {{< wiki "attention" >}}attention{{< /wiki >}} path.

The Hadamard transform requires only $O(d \log d)$. For $d = 4096$, that's $\sim 50{,}000$ operations per token. Fast enough to fuse into the attention kernel without measurable latency cost. This is the same speedup that the FFT gives over a general DFT.

**2. Equidistribution.** Hadamard matrices have entries of exactly $\pm 1/\sqrt{d}$. When you apply one to a vector, every output component is a *signed sum* of all input components. If a small number of inputs are large outliers, the rotation spreads their magnitude across all outputs. After Hadamard, no single output is an outlier; instead, every output has approximately the same magnitude.

This is a *concentration* result: the maximum absolute value of $H x$ is bounded by approximately $\|x\|_2 \sqrt{\log d / d}$, much smaller than $\|x\|_\infty$ when $x$ has outliers. The Hadamard *forces* the worst-case quantization grid placement to track the average rather than the maximum.

**3. Exact orthogonality with simple structure.** Many "almost-orthogonal" transforms exist (random projection matrices, learned rotations). Hadamards are *exactly* orthogonal with $H^\top H = I$, requiring no normalization corrections. They are also entirely $\pm 1$ before the $1/\sqrt{d}$ scaling, which means the rotation can be implemented with adds and subtracts — no multiplications.

## Why It Works: A Concentration Demonstration

Let's *see* the concentration in action. Start with a 4096-dim vector that has a few outliers, apply a Hadamard, and look at the result.

```pyplot {id="hadamard-spread" caption="A 4096-dim vector with four outliers. Before Hadamard: a few giants and a flat sea. After Hadamard: every component is roughly the same magnitude."}
def hadamard(n):
    """Recursive construction of an n x n Hadamard matrix (n must be a power of 2)."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

np.random.seed(7)
d = 4096
x = np.random.randn(d) * 0.3
# Inject 4 outliers
out_dims = [412, 1318, 2904, 3700]
for j in out_dims:
    x[j] = np.random.choice([-1, 1]) * 15.0

H = hadamard(d)
y = H @ x

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax = axes[0]
ax.bar(range(d), np.abs(x), color='#1A1A1A', width=1.0)
for j in out_dims:
    ax.bar(j, np.abs(x[j]), color='#FF007F', width=4)
ax.set_title(f"Before Hadamard:  max={np.abs(x).max():.2f},  mean={np.abs(x).mean():.3f},  ratio={np.abs(x).max()/np.abs(x).mean():.0f}x")
ax.set_ylabel("|x|")
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
ax.bar(range(d), np.abs(y), color='#00A8A8', width=1.0)
ax.set_title(f"After Hadamard:  max={np.abs(y).max():.2f},  mean={np.abs(y).mean():.3f},  ratio={np.abs(y).max()/np.abs(y).mean():.1f}x")
ax.set_xlabel("dimension")
ax.set_ylabel("|y|")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print(f"L2 norm preserved? Before: {np.linalg.norm(x):.3f}, After: {np.linalg.norm(y):.3f}")
print(f"Quantization-grid penalty (max/mean) reduced by {(np.abs(x).max()/np.abs(x).mean()) / (np.abs(y).max()/np.abs(y).mean()):.0f}x")
```

The before/after comparison is striking. The original vector has a max-to-mean ratio of ~50; quantizing it with one shared scale would give ~50 {{< wiki "number-formats" >}}INT4{{< /wiki >}} codes "wasted" on the outliers. The Hadamard-rotated vector has a max-to-mean ratio of ~3 — almost flat. INT4 quantization now uses essentially every code.

The L2 norm is preserved exactly (to floating-point precision). No information has been lost. We have *redistributed* it.

## Why The Identity Holds — A One-Picture Proof

Here's the algebra spelled out for the doubting eye. We have

$$
Y = X W^\top.
$$

Insert $R R^\top = I$:

$$
Y = X (R R^\top) W^\top = (X R) (R^\top W^\top) = (X R) (W R)^\top.
$$

That last step uses $(R^\top W^\top) = (W R)^\top$, which holds for any matrices when $R$ is orthogonal.

So if we **store $XR$ and $WR$** (instead of $X$ and $W$), the matmul $\hat Y = (XR)(WR)^\top$ produces *exactly* the original $Y$. The rotations are absorbed into the matmul by their own orthogonality.

The trick: $XR$ is much friendlier to per-channel quantization than $X$, and $WR$ is much friendlier than $W$. We pay a small kernel cost (one Hadamard transform per layer) and get a much better quantization grid in return.

## QuIP — The Original Idea (2023)

In **June 2023**, **Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa** at Cornell publish **QuIP — Quantization with Incoherence Processing**.

QuIP frames the rotation idea using the language of **incoherent matrices**. A matrix is *incoherent* if its rows and columns are spread roughly uniformly — no single direction dominates. The QuIP claim is: if you can make $X$ and $W$ both incoherent (by rotating them with carefully-chosen orthogonal matrices), then standard scalar quantization works much better.

QuIP uses **random orthogonal matrices** with $O(d^2)$ multiplications and is therefore expensive at inference. It is mostly an academic proof of concept for the rotation idea; QuIP# (the follow-up by the same group) replaces the random rotations with structured ones (Hadamards plus a learned permutation) for tractable inference.

**The lasting contribution of QuIP**: the framework of incoherence as the *thing the rotation buys you*. Subsequent methods would all aim for the same target, with different tricks for getting there.

## QuaRot — The Production Hammer (2024)

In **April 2024**, **Saleh Ashkboos** and collaborators at ETH Zurich publish **QuaRot — Outlier-Free 4-Bit Inference in Rotated LLMs**.

QuaRot does for the *whole transformer* what QuIP did for individual layers: apply Hadamard rotations strategically across the model so that *every* activation, *every* weight, and *every* KV-cache vector becomes incoherent and 4-bit-quantizable.

The recipe:

1. **Pre-RMSNorm rotation:** insert $R$ and $R^\top$ around each transformer block. The norms of the layer's input/output stay the same; the per-channel statistics flatten.
2. **Within-attention rotation:** insert another $R$ inside the attention computation, around $W^V$ and $W^O$. This makes V and the attention output INT4-friendly.
3. **All rotations are Hadamards** — $O(d \log d)$ inserts, fused with the matmul kernels.
4. **Run GPTQ on the rotated weights.** Standard calibration applies.

The result is the first method to do **W4A4KV4** on Llama-2-70B with negligible perplexity loss. ATOM ([family-tree #5 in the KV map](../15-kv-method-family/)) had earlier hit the W4A4 territory but with a noticeable accuracy gap; QuaRot closed it.

QuaRot is also the first method that works *equally well* on K, V, *and* weights — because the rotation trick is uniform; it doesn't care what the tensor is. KIVI's K-channel / V-token asymmetry simply *vanishes* after rotation, because outlier channels and outlier tokens are both spread out.

```pyplot {id="quarot-flatten" caption="K with outlier channels (left) becomes much flatter after Hadamard rotation along the channel axis (right). Per-channel scales now have a much narrower spread."}
def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

np.random.seed(1)
seq_len, d_head = 64, 64
K = np.random.randn(seq_len, d_head) * 0.3
K[:, [4, 17, 31, 48]] *= 15  # 4 outlier channels

H = hadamard(d_head)
K_rot = K @ H

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, data, title in [(axes[0,0], K, "K (original)"),
                         (axes[0,1], K_rot, "K after Hadamard rotation along channels")]:
    ax.imshow(np.abs(data), aspect='auto', cmap='magma', vmin=0, vmax=5)
    ax.set_title(title)
    ax.set_xlabel("channel")
    ax.set_ylabel("token position")

scales_orig = np.abs(K).max(axis=0)
scales_rot  = np.abs(K_rot).max(axis=0)
axes[1,0].bar(range(d_head), scales_orig, color='#FF007F')
axes[1,0].set_title(f"Per-channel max in original K\nstd of channel scales = {scales_orig.std():.3f}")
axes[1,0].set_xlabel("channel"); axes[1,0].set_ylabel("max |K|")
axes[1,0].spines[['top', 'right']].set_visible(False)

axes[1,1].bar(range(d_head), scales_rot, color='#00A8A8')
axes[1,1].set_title(f"Per-channel max in rotated K\nstd of channel scales = {scales_rot.std():.3f}")
axes[1,1].set_xlabel("channel"); axes[1,1].set_ylabel("max |K|")
axes[1,1].spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The bottom row shows the practical payoff. Before rotation, per-channel scales for K span a 30× range — most channels need a tiny scale, four need an enormous one. After rotation, *every* channel has roughly the same scale, and a single block-wise scale captures all the structure that matters.

## SpinQuant — Learn The Rotation (2024)

In **May 2024**, **Zechun Liu** and collaborators at Meta publish **SpinQuant**.

The SpinQuant claim: random Hadamard rotations are good but not optimal. The *best* rotation depends on the specific weight tensor — and you can find it by **gradient descent** on a small calibration set, parameterizing the rotation matrix as a product of Givens rotations or a structured Cayley transform that stays in the orthogonal group.

The optimization minimizes the layer reconstruction error after quantization. After 100–200 steps of training the rotation parameters (on 32 calibration samples), SpinQuant typically beats QuaRot by 0.1–0.3 perplexity at 4-bit, and substantially more at 3-bit.

The trade-off: the calibration step is no longer free. SpinQuant calibration takes ~30 minutes per 70B model — comparable to GPTQ. But the rotation, once found, is just a fixed matrix multiplication at inference, no per-token cost beyond the kernel-fused Hadamard.

**Where SpinQuant lives**: the most accurate rotation method to date, at the cost of a learning step. Production stacks favor QuaRot for its simplicity; research stacks pick SpinQuant for the last 0.2 perplexity points.

## TurboQuant — Random Rotations For The KV Cache (2024)

The KV-cache application of the rotation trick is **TurboQuant**, by Apple's MLR team in late 2024 (see also [The KV Method Family Tree](../15-kv-method-family/) for the algorithmic description).

The TurboQuant story is essentially: take QuaRot's Hadamard-rotation idea, apply it specifically to K and V *online* (during inference), and skip the QuaRot-style calibration. Random Hadamard, no learning, no calibration data — just the rotation, applied per-vector during the attention forward pass.

Why this works without calibration: the TurboQuant authors observed that **for {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} specifically, a randomized Hadamard rotation is as good as a learned one**. The reason is the universality of the concentration property — almost any orthogonal rotation flattens outliers. The "best" rotation gives you maybe 0.05 perplexity over a random one, which is below the noise floor for KV quantization.

The TurboQuant practical pitch: a 4-bit KV cache with no setup, ~4× memory reduction, and accuracy matching KVQuant. Becoming the default in inference engines that prioritize zero-config deployment (notably MLX, Apple's stack).

## A Comparison Table

| Method | Year | Where applied | Rotation | Calibration | Best at |
|---|---|---|---|---|---|
| QuIP | 2023 | weights only | random orthogonal | gradient | proof-of-concept |
| QuIP# | 2024 | weights only | Hadamard + lattice | gradient | 2-bit weights |
| QuaRot | 2024 | weights + activations + KV | Hadamard | GPTQ on rotated | W4A4KV4 production |
| SpinQuant | 2024 | weights + activations | learned (Cayley/Givens) | gradient | best 4-bit accuracy |
| TurboQuant | 2024 | KV cache only | Hadamard | none | online KV, zero-config |

## What Rotations Don't Solve

Two limitations to keep in mind.

**1. Massive activations don't fully disappear.** Recall from [Inside K and V](../16-kv-distribution/) that some activations are 10⁴ to 10⁵ × the median. After a Hadamard rotation, no *single* dimension is that large — but the *aggregate* energy is preserved. The rotated tensor has many moderate values rather than a few enormous ones. This is good for quantization (the worst case is gentler), but it means the rotation has to be applied in a basis that *doesn't* re-concentrate the outliers later. QuaRot's strategy of inserting rotations on both sides of LayerNorm is what keeps things flat through the network.

**2. Some operations destroy the rotation.** Activation functions (GeLU, SiLU) are *not* orthogonal-equivariant: applying GeLU to a rotated vector is not the same as rotating the GeLU output. So the rotation can be inserted only at points where it can be cancelled by a matching $R^\top$ before any non-linear operation. This constrains the rotation pattern to specific, carefully-chosen places in the transformer block. The reason the rotation methods all look complicated when written out (multiple $R$ matrices in different places) is to navigate this constraint.

## What This Pattern Suggests About The Future

The rotation methods are an instance of a deeper principle: **the basis matters as much as the bits**. Two equivalent representations of the same tensor — one in standard basis, one in a rotated basis — have wildly different quantization properties. Picking the right basis can buy you 1-2 bits of effective precision for free.

This idea is everywhere in classical compression — see [Compression's Family Tree](../14-compression-roots/), G3 (transform coding). MP3, JPEG, AAC all hinge on representation-basis choice. ML quantization is finally catching up.

If the rotation methods generalize — and they keep generalizing further (rotations within each attention head; rotations between layers; rotations for {{< wiki "embeddings" >}}embedding{{< /wiki >}} tables) — by 2027 we may be in a world where *every* quantized tensor is stored in a non-standard, locally-optimal basis. The matmul kernels will know about the basis; the storage will be smaller; and the dots will land where the data is.

## What To Remember

1. **Rotations move the outliers to nowhere in particular.** A Hadamard transform spreads the worst-case channel mass across all channels.
2. **The matmul identity is exact**: $XW^\top = (XR)(WR)^\top$ for any orthogonal $R$.
3. **Hadamard is the right rotation** for $d$ a power of 2: $O(d \log d)$ to apply, exactly orthogonal, $\pm 1$ entries.
4. **QuaRot is the workhorse** (rotation + GPTQ on rotated weights). **SpinQuant** learns the rotation. **TurboQuant** applies it online to the KV cache with no calibration.
5. The rotation idea is the LLM-side rediscovery of [transform coding](../14-compression-roots/), generation 3 of classical compression engineering, with the objective inverted (spread, not concentrate).

**Continue to** → [The Axis Question](../18-quantization-axes/) — the other cross-cutting concept this issue keeps revisiting: when you choose to quantize per-tensor, per-channel, per-token, or per-block, you are picking *which axis of heterogeneity to track*. Get the axis wrong and no clever method saves you.
