---
title: "The KV Cache Method Family Tree"
description: "Eight modern KV-cache quantization methods on one shelf. KIVI, KVQuant, GEAR, ATOM, QServe, KVTuner, TurboQuant, and the FP8-native option. Which trick each one is making, and which to pick when."
topics: [quantization, attention]
tags: [kv-cache, kivi, kvquant, gear, atom, qserve, turboquant, kvtuner]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 150
techKind: boss
techNode: kv-method-family
header: default.webp
---

## The Other Family Tree

[The Method Family Tree](../09-method-family-tree/) covered the **weight-quantization** half of modern LLM compression: GPTQ, AWQ, SmoothQuant, QLoRA/NF4, HQQ, LLM.int8. Six methods, one shelf, well-mapped territory.

Now meet its sibling. The **KV-cache** quantization shelf had no equivalent map until late 2024, because the KV-cache problem hadn't fully arrived yet. Once weights got small enough, attention's running buffer of keys and values became the new memory bill — bigger than the weights themselves at long context, and *growing linearly with sequence length*. By 2025, every serious inference engine has a KV-cache quantization story; by 2026, there are at least eight distinct approaches in production.

This article is the field guide. We tour each method, note what trick it makes, and end with the same one-page selection table that closes [the weight family tree](../09-method-family-tree/). The cast:

1. **FP8 KV** — the hardware-native baseline.
2. **KIVI** — per-channel K, per-token V, 2-bit.
3. **KVQuant** — non-uniform 4-bit + per-channel K rotations + outlier isolation.
4. **GEAR** — low-rank + sparse + quantized residual.
5. **ATOM** — W4A4KV4, the "everything is 4-bit" recipe.
6. **QServe** — W4A8KV4 with smoothing for serving.
7. **KVTuner** — layer-adaptive bit allocation.
8. **TurboQuant** — random rotations to flatten outliers.

Each gets a one-page entry. The lessons emerge after.

## Why KV Quantization Is A Different Problem From Weight Quantization

Before the methods, the framing. KV quantization is *not* just "weight quantization, but for keys and values." The constraints are different:

- **No offline calibration.** The {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} is built *during inference* — you cannot pre-quantize it on a calibration set. Whatever you do has to happen on the fly, ideally in less than a token's worth of compute.
- **Asymmetric structure.** As we saw in [KV Cache Tyranny](../10-kv-cache/) and unpack in [Inside K and V](../16-kv-distribution/), keys have outlier *channels* while values have outlier *tokens*. Same tensor shape, different quantization axes. This rules out symmetric W-style methods.
- **Streaming workload.** Tokens arrive one at a time. The quantizer has to amortize across the sequence, not the calibration set.
- **Bigger blast radius for errors.** A single mis-quantized K vector affects all *future* {{< wiki "attention" >}}attention{{< /wiki >}} queries against that token. A weight error gets averaged over many activations; a KV error compounds across the rest of the sequence.

These constraints push the KV side of the field toward **online, statistics-driven, axis-aware** methods. None of the major weight quantization methods (GPTQ, AWQ, SmoothQuant) translates directly.

## 1. FP8 KV — The Hardware Baseline

**Year:** 2023 (Hopper); 2024 widespread.
**The trick:** Just store K and V in {{< wiki "number-formats" >}}FP8{{< /wiki >}} instead of FP16. No clever scaling, no axis tricks, no calibration.

This is the simplest possible KV quantization, available on every Hopper-or-newer GPU. The hardware can read FP8 from the cache and dequantize-on-the-fly to FP16 for the attention matmul, with no runtime cost to speak of.

**Why it works at all:** FP8 has roughly the same dynamic range as FP16 (E5M2 has a *bigger* range than FP16, in fact). The precision drop from FP16's 10-bit mantissa to FP8's 2- or 3-bit mantissa is barely felt by attention, because the {{< wiki "softmax" >}}softmax{{< /wiki >}} in attention is itself a noise-suppressor: small differences between large {{< wiki "logit" >}}logits{{< /wiki >}} get exponentially flattened. Most attention computations don't need the precision FP16 was giving them.

**Why it's not enough:** At long context, FP8 is still *only 2× compression*. For Llama-2-70B at 128K context, FP8 KV cache is 80 GB — half of what FP16 demanded, but still bigger than the quantized weights. To get below 4 bits per KV element you need cleverer methods.

**When to use:** Always, as the floor. Most engines (vLLM, SGLang, TensorRT-LLM) ship FP8 KV as a one-flag toggle. It's the first thing to enable.

## 2. KIVI — Per-Channel K, Per-Token V

**Year:** June 2023, by **Yujun Liu, Hongyu Wang**, and collaborators at MIT.
**The trick:** quantize K **per-channel** (one scale per feature dim) and V **per-token** (one scale per sequence position), exploiting the empirical observation that K's outliers cluster in specific channels and V's in specific tokens. See [Inside K and V](../16-kv-distribution/) for the deep dive on why this asymmetry exists.

**The recipe:**

1. For each K vector arriving from the projection, look up the per-channel scale (computed once per layer at calibration, kept fixed).
2. Quantize K to **2-bit asymmetric** using those scales.
3. For each V vector, compute a per-token absmax scale on the fly.
4. Quantize V to 2-bit using the per-token scale.
5. At decode time, dequantize the relevant K, V vectors before the attention matmul.

**Why it gets to 2 bits without breaking:** Because per-channel scaling on K *isolates* the outlier channels — they get their own scale, the bulk channels get tighter scales, the bulk dimensions effectively use the full 2-bit range. Per-token V scaling does the same trick on V's outlier rows.

**Reported numbers:** Llama-2-7B WikiText perplexity 5.47 (FP16) → 5.55 (KIVI 2-bit). Eight-fold cache compression for ~0.1 perplexity. Industry-foundational result.

**Where it lives:** The grandparent of the modern KV family. Every later method either inherits the axis-aware insight or finds a different way around it.

```pyplot {id="kivi-axis-recap" caption="The KIVI insight in one picture: K outliers are vertical (per-channel), V outliers are horizontal (per-token). One axis per tensor."}
np.random.seed(2)
seq_len, d_head = 64, 32

K = np.random.randn(seq_len, d_head) * 0.3
K[:, [5, 18, 27]] *= 12  # outlier channels

V = np.random.randn(seq_len, d_head) * 0.3
V[[10, 33, 50], :] *= 12  # outlier tokens

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, data, title in [(axes[0], K, "K — quantize PER CHANNEL"),
                         (axes[1], V, "V — quantize PER TOKEN")]:
    ax.imshow(np.abs(data), aspect='auto', cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("feature dim")
    ax.set_ylabel("token position")
plt.tight_layout()
```

## 3. KVQuant — Non-Uniform 4-Bit + Outlier Isolation

**Year:** January 2024, by **Coleman Hooper** and collaborators at UC Berkeley.
**The trick:** Three stacked tricks: (a) **per-channel pre-RoPE rotation** for K, (b) **non-uniform quantization** (NF4-style levels chosen for the empirical K and V distributions), and (c) **explicit outlier separation** (top-1% values stored in FP16, the rest at 4-bit).

**Why "pre-RoPE"?** This is a specifically transformer-architecture observation. **{{< wiki "rope" >}}RoPE{{< /wiki >}}** (Rotary Positional Embeddings) is the position-encoding scheme used by Llama-family models: it applies a position-dependent rotation to K and Q vectors *after* the linear projection. KIVI's per-channel scales are computed on *post*-RoPE K — but RoPE *changes* the channel statistics depending on the token position. KVQuant moves the quantization to *before* RoPE: the per-channel statistics there are stable (RoPE hasn't shuffled them yet), so the scales fit better.

**Why non-uniform levels?** The same Lloyd-Max insight from [the lloyd-max bargain](../03-lloyd-max/). The empirical K and V distributions aren't uniform; they aren't even Gaussian; they're heavy-tailed. KVQuant fits a custom 4-bit grid to each layer's K and V distributions during a one-time calibration, NF4-style.

**Why outlier isolation?** Same lesson as [LLM.int8](../06-outliers/): a tiny fraction of the values are giant. Store them in FP16, store the rest at 4-bit, do the matmul in two pieces.

**Reported numbers:** Llama-2 perplexity within 0.1 of FP16 at 4-bit; within 0.3 at 3-bit. Significantly better than KIVI at the same bitwidth, but with substantially more calibration machinery.

**Where it lives:** The "complete pipeline" KV method. If KIVI is one trick, KVQuant is three. It is the GPTQ of the KV family — the workhorse-quality method, with non-trivial setup cost.

## 4. GEAR — Decompose, Then Quantize Only The Residual

**Year:** March 2024, by **Hao Kang** and collaborators at Georgia Tech.
**The trick:** Decompose K and V as **low-rank + sparse + residual**. Quantize *only the residual*, which is small and well-behaved. Keep the low-rank part in FP16 (it's small) and the sparse outlier part in FP16 (it's truly tiny).

In math: for a KV tensor $X$, find $L$ (low-rank, rank $r$), $S$ (sparse, $k$ entries) such that

$$
X = L + S + R, \quad R \text{ small Gaussian-ish residual}
$$

Then quantize $R$ to 4-bit (or 2-bit) — easy, because $R$ has no outliers and no global structure left to preserve.

**Why this works:** The decomposition extracts the "hard parts" of the KV tensor (the global low-rank structure that softmax cares about, and the per-token outliers) into two separate, small, FP16-stored tensors. What's left is statistically homogeneous and quantizes cleanly.

**Cost:** The decomposition step is non-trivial (essentially a streaming SVD plus outlier detection). It runs per-layer at fill time, but isn't free.

**Where it lives:** The "principled decomposition" method. Theoretically beautiful (it's a direct application of low-rank+sparse matrix recovery). Practically a bit heavy — the decomposition adds latency that some serving stacks can't afford.

## 5. ATOM — W4A4KV4

**Year:** October 2023, by **Yilong Zhao** and collaborators.
**The trick:** Quantize *everything* to 4-bit — weights, activations, *and* KV cache. The first method to do all three at once with usable accuracy.

The recipe is a careful integration of techniques developed for each piece separately:

- **Weights:** GPTQ-style sequential compensation at 4-bit, with per-group scaling.
- **Activations:** A SmoothQuant-style migration that pushes outliers from activations into weights, then INT4 the activations.
- **KV cache:** Per-channel K, per-token V (KIVI-style) at INT4 with asymmetric grids.
- **Outlier handling:** A "dynamic outlier extraction" step that identifies the top ~1% of activations per token and routes them through a parallel FP16 path (LLM.int8 lineage).

**Why it matters:** ATOM was the first paper to demonstrate that full INT4 inference (including KV cache) is *workable* on Llama-family models with a 1–2% accuracy loss. It's the proof-of-concept for the modern "everything is 4-bit" inference stack, including QServe.

**Where it lives:** The integration paper. Not best-in-class on any single axis, but the first end-to-end demonstration that you can run an LLM where every tensor is 4-bit.

## 6. QServe — W4A8KV4 With Smoothing

**Year:** May 2024, by the same MIT group as KIVI/AWQ.
**The trick:** A *production-engineering* trade-off: keep weights at 4-bit, activations at 8-bit (not 4 — the accuracy cost is too high), KV cache at 4-bit. Then engineer the kernels to minimize dequantization overhead.

Why this asymmetry? **Activations are harder to quantize than weights.** Their outliers are larger and more concentrated. ATOM's W4A4 works on Llama, but at the edge of accuracy; for a production-quality serving stack you want a margin. QServe sacrifices the activation bits to get back the margin, then makes up the latency cost with kernel optimizations.

The KV part of QServe uses a SmoothQuant-style channel migration on K (smoothing K's outlier channels into the W^K projection so what gets cached is well-behaved) plus per-token V scales. The result is a KV cache that's INT4 with per-block scaling and minimal accuracy loss, with kernel paths optimized for hot decode.

**Reported numbers:** ~1.2× higher decode throughput than TensorRT-LLM at FP8 on Llama-3-70B, with negligible perplexity loss.

**Where it lives:** The serving-engineering method. Probably what your inference engine actually runs in 2026 if you flag "fast quantization mode."

## 7. KVTuner — Layer-Adaptive KV Bits

**Year:** 2024, by various groups (the technique is more an idea than a single paper).
**The trick:** **Different layers need different bit widths** for the KV cache. Some early layers tolerate 2-bit fine; some middle layers need 4-bit; a handful of attention-sink layers need 8-bit. Allocate the bit budget per-layer to minimize end-to-end perplexity loss.

The motivation: the KV-cache memory bill is *sum* over layers, but the *accuracy cost* of quantizing each layer's KV is non-uniform. If you have a 4-bit average budget and 32 layers, it might be optimal to put 6 bits in 8 critical layers and 3 bits in the other 24, summing to the same average but losing less accuracy.

**The selection algorithm:** A small calibration sweep — for each (layer, bitwidth) combination, measure the perplexity contribution. Solve a knapsack to maximize quality at fixed average bit budget. Some implementations use a Hessian-based sensitivity score to skip the sweep.

**Reported gains:** ~30% memory reduction at the same perplexity vs. uniform-bit KV. Becoming standard practice in mixed-precision inference engines.

**Where it lives:** The "meta-method" — pick the best of the others, layer by layer. Stacks cleanly on top of any uniform-bit method (KIVI, KVQuant, etc.).

## 8. TurboQuant — Random Rotations Spread Outliers

**Year:** 2024, by **Apple's MLR team** (Bai et al.).
**The trick:** Apply a **random orthogonal rotation** (specifically, a fast Hadamard transform) to K (and to V) before quantization. The rotation spreads the outlier channels' mass across all channels, so per-channel quantization sees nearly-uniform statistics. After dequantization at attention time, apply the inverse rotation. Math is preserved exactly because the rotation is orthogonal.

This is the [transform-coding generation 3 idea](../14-compression-roots/) — exactly. We pull the technique apart in [The Rotation Trick](../17-rotations/), so here we focus on what TurboQuant specifically claims.

**The KV-specific argument:** K's outlier channels are the bottleneck for low-bit K quantization. KIVI handled them by giving each its own scale; KVQuant by isolating them in FP16; TurboQuant by *rotating them away*. After a random Hadamard rotation, no single channel is large; instead, *every* channel has Gaussian-bulk + small-residual structure, which quantizes cleanly with per-channel or even per-block scales.

**Why "fast Hadamard"?** Because matmul. A Hadamard transform on a $d$-dimensional vector takes $O(d \log d)$ operations and can be fused into the attention kernel. A random orthogonal rotation in general would take $O(d^2)$ — too slow. Hadamards have the right algebraic structure (orthogonal, simple binary entries, FFT-style fast multiplication) and empirically distribute outlier mass nearly as well as truly random rotations.

**Reported numbers (TurboQuant for KV cache):** Llama-2-7B at 4-bit KV with TurboQuant achieves perplexity within 0.05 of FP16, beating KIVI at the same bitwidth and matching KVQuant *without* the calibration step. The "no calibration" property is what makes it attractive for serving: you can flip it on without a setup pass.

**Where it lives:** The "geometric" method. A spiritual cousin of QuaRot/SpinQuant on the weight side (which we cover in [Rotations](../17-rotations/)). Likely to dominate the next generation of online KV quantization.

## The Selection Table

| Method | Bits | Calibration | Online cost | KV speed-up | Best at | Failure mode |
|---|---|---|---|---|---|---|
| FP8 KV | 8 | none | nil | 2× | always-on baseline | bigger context still pricey |
| KIVI | 2 | per-layer K-channel scales | nil | 8× | 2-bit ceiling | RoPE drift on long context |
| KVQuant | 4 | non-uniform grids per layer | nil | 4× | 4-bit accuracy ceiling | calibration-heavy |
| GEAR | 4 | streaming SVD | moderate | 4× | best 4-bit accuracy | adds decode latency |
| ATOM | 4 | end-to-end W4A4KV4 | nil online | 4× | full-INT4 inference | edge of accuracy |
| QServe | 4 | smoothing | nil | 4× | production serving | serving-tuned, less general |
| KVTuner | 2-8 mix | sensitivity sweep | nil | 4-6× avg | best memory-quality trade | layered on top of other methods |
| TurboQuant | 4 | **none** | small (Hadamard) | 4× | calibration-free 4-bit | needs hardware-fused Hadamard |

## What They All Share

Three convergent themes, mirroring [the weight family tree](../09-method-family-tree/):

**Per-block / per-axis scales.** All methods use *some* form of grouped scaling. The axis varies (per-channel, per-token, per-rotated-block) but the principle is universal: a single tensor-wide scale fails because the data is heterogeneous.

**Outlier as first-class citizen.** Every method has an outlier strategy: KIVI uses per-channel scales (outlier dimensions get their own grids); KVQuant explicitly extracts the top 1% in FP16; GEAR pulls outliers into a sparse component; TurboQuant rotates them away. None ignore the problem.

**Asymmetry of K and V.** Every method that beats FP8 explicitly treats K and V differently. The KIVI insight — K has outlier channels, V has outlier tokens — has held up across two years of follow-up work, with one wrinkle (attention sinks) detailed in [Inside K and V](../16-kv-distribution/).

## What's Next

Two open frontiers:

**1. KV cache *during* training.** Almost all KV quantization is inference-side. Training-time KV quantization (relevant for very long context fine-tuning) is a near-empty area. Expect FP4/FP6 KV training within a year.

**2. Vector-quantized KV.** Same gap as on the weight side: nobody is doing real vector quantization of K/V. AQLM-style approaches haven't been adapted to KV yet. Hardware permitting, this could be a significant gain.

## How To Pick

Practical decision tree if you're staring at an inference engine:

1. **On Hopper or newer with no setup time?** Enable FP8 KV. It's free.
2. **Need 4-bit, no calibration data, want it on a flag flip?** TurboQuant.
3. **Need 4-bit and have time for a calibration sweep?** KVQuant.
4. **Need 2-bit?** KIVI is the only widely-supported option; expect ~0.1 PPL loss.
5. **Memory-constrained, want the best perplexity per byte?** KVTuner on top of KIVI or KVQuant.
6. **Building a serving stack?** QServe's W4A8KV4 is the production sweet spot.

Most users will, as with weight quantization, end up using *whatever their inference engine ships*. By 2026, vLLM and SGLang ship KIVI + KVQuant + FP8 as one-flag toggles; QServe is bundled into TensorRT-LLM; TurboQuant is showing up in MLX (Apple's stack). The framework choice picks the method.

**Continue to** → [Inside K and V](../16-kv-distribution/) — the deep observation that this entire shelf rests on: *why* keys have outlier channels and values have outlier tokens, and the timeline of what we've learned about it.
