---
title: "Compression's Family Tree"
description: "Pulse code modulation. Vector quantization. Transform coding. Predictive coding. Eighty years of lossy compression engineering, and where each thread surfaces inside a modern LLM."
topics: [quantization, history, compression]
tags: [pcm, dpcm, lbg, jpeg, mp3, predictive-coding, vector-quantization]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 140
techKind: primer
techNode: compression-roots
header: default.webp
---

## A Field That Forgot Its Inheritance

Walk into a 2024 ML conference and you can spend three days hearing about quantization without anyone uttering the words "**vector quantization**" or "**linear predictive coding**" or even "**transform coding**". Walk into the lobby of a 1985 ICASSP and those three phrases are *literally on the lapel pins*. The two communities — ML quantization and classical compression — built much of the same machinery, half a century apart, with almost no cross-pollination.

That gap is starting to close. **NF4** is a Lloyd-Max quantizer for a Gaussian source, a problem solved at Bell Labs in 1957. **GPTQ**'s sequential compensation is the linear-predictive-coding loop from speech vocoders, with the prediction filter replaced by a Hessian factor. **AWQ** and **SmoothQuant** (see [The Method Family Tree](../09-method-family-tree/)) apply a diagonal-rescaling identity that JPEG implementers have been using to balance brightness and chrominance scales since 1992. **QuaRot** and **SpinQuant** are subband-coding's transform-then-quantize step, with Hadamard rotations standing in for the discrete cosine transform.

The point of this article is not to rename anything or take credit away from anyone. The modern LLM quantization community made these methods *work* on objects (LLM weights, KV caches) that the classical community never imagined. But standing back to see the family tree clarifies *why* certain things work, *what's still missing*, and *what the next generation of methods will probably look like*. If you only know modern ML, you are missing thirty years of bug-fixes already paid for in another field.

## A Five-Generation Family Tree

Every modern quantization technique descends, structurally, from one of five compression-engineering generations. They emerged roughly a decade apart, each generation adding a new mathematical tool.

| Generation | Era | Core idea | Modern LLM cousin |
|---|---|---|---|
| **G1** Sample-then-quantize (PCM) | 1937–1957 | snap each sample to a fixed grid | Absmax / RTN |
| **G2** Predictive coding (DPCM, LPC) | 1952–1972 | quantize the *residual* after prediction | GPTQ sequential compensation |
| **G3** Transform coding (DCT, MDCT) | 1974–1985 | rotate to a basis where coefficients are sparse, then quantize | QuaRot, SpinQuant, QuIP |
| **G4** Vector quantization (LBG, TSVQ) | 1980–1990 | quantize *blocks* of values jointly to a learned codebook | NF4 codebook (degenerate 1-D case) |
| **G5** Perceptually-weighted optimization (CELP, AAC) | 1985–2000 | optimize codes against a perceptually-weighted error | OmniQuant, AdaRound |

The thread is consistent: **each generation adds a new way of using prior knowledge** about the source signal. Generation 1 assumed nothing. Generation 2 assumed temporal correlation. Generation 3 assumed sparsity in some basis. Generation 4 assumed clusters. Generation 5 assumed a perceptual error model.

LLM quantization has, as of 2026, *nearly* recapitulated this entire arc. Generation 4 (true vector quantization) is the one we mostly haven't picked up — and the reason is hardware, which we'll come back to.

## Generation 1: PCM And The Origin Story

In **1937**, an engineer at AT&T's research labs named **Alec Reeves** files a patent in France for a system he calls *Pulse Code Modulation*. The idea is laughably simple in retrospect: instead of transmitting a continuous voltage down a wire (vulnerable to attenuation and crosstalk), sample the voltage 8,000 times a second, snap each sample to one of 128 fixed levels, and transmit the seven-bit code for the level. At the receiver, regenerate the voltage from the codes. The signal is now *digital*, immune to transmission noise, and re-amplifiable without distortion drift.

PCM is **uniform quantization**. The 128 levels are equally spaced. There is no per-block scaling, no calibration, no adaptive logic. It works because human voices, when properly amplified, span a known voltage range. You set the levels once, by hand, at deployment time.

This is the direct ancestor of [absmax](../02a-absmax/): pick a fixed scale, snap to the grid. The *only* difference is that absmax computes the scale per-tensor at runtime, while PCM hardcoded it at the factory.

```pyplot {id="pcm-vs-absmax" caption="A toy speech-style waveform, quantized first by 1937-style PCM (fixed levels) and then by 2022-style absmax (per-window scaling). Same number of bits — but absmax adapts to local amplitude."}
np.random.seed(1)
t = np.linspace(0, 4*np.pi, 800)
sig = np.sin(t) * (0.3 + 0.7*np.exp(-((t-6)**2)/3))  # an envelope-modulated tone

n_levels = 7
fixed_scale = 1.0  # PCM-style fixed range

def quantize(x, scale):
    q = np.clip(np.round(x / scale * n_levels), -n_levels, n_levels)
    return q * scale / n_levels

q_pcm = quantize(sig, fixed_scale)

# Absmax in non-overlapping windows of 100 samples
window = 100
q_abs = np.zeros_like(sig)
for i in range(0, len(sig), window):
    blk = sig[i:i+window]
    s = max(np.abs(blk).max(), 1e-9)
    q_abs[i:i+window] = quantize(blk, s)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax = axes[0]
ax.plot(sig, color='#1A1A1A', linewidth=0.6, label='original')
ax.plot(q_pcm, color='#FF8C00', linewidth=1.0, label=f'PCM (fixed scale=1.0)')
ax.set_title("Generation 1: fixed grid (PCM, 1937)")
ax.set_ylabel("amplitude")
ax.legend(fontsize=8, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
ax.plot(sig, color='#1A1A1A', linewidth=0.6, label='original')
ax.plot(q_abs, color='#FF007F', linewidth=1.0, label='absmax (per-window)')
ax.set_title("Generation 1.1: per-window adaptive scale (absmax, 2022)")
ax.set_xlabel("sample index")
ax.set_ylabel("amplitude")
ax.legend(fontsize=8, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

You can already see the move that will repeat throughout the family tree. **A higher-information signal is encoded with a smarter calibration**. PCM ignored the signal's local amplitude. Absmax scales to it. *Per-block* absmax (the next decade's contribution) goes one finer. The trajectory is the trajectory.

## Generation 2: Predictive Coding And The GPTQ Connection

In **1952**, an engineer at Bell Labs named **C. Chapin Cutler** files a patent for **DPCM — Differential Pulse Code Modulation**. The idea: most speech samples are close to the previous sample. Don't transmit the absolute amplitude; transmit the *difference* from a predicted value, where the prediction is some simple function of recent samples (often just "the previous sample"). The differences have **smaller variance** than the originals, so the same number of bits gives you finer resolution.

Twenty years later, **Bishnu Atal** at Bell Labs generalizes this to **Linear Predictive Coding (LPC)**: predict the next speech sample as a linear combination of the previous $N$ samples, fit the coefficients to the speaker's vocal tract, and quantize only the *residual*. This is the foundation of every speech codec from GSM to Opus.

The key idea — the one that will matter for LLMs — is the **encoder loop**:

1. Predict.
2. Quantize the residual.
3. **Update the prediction model** using the quantized residual (so the decoder can do the same).
4. Repeat.

The residual gets smaller as the prediction gets better. The quantization gets effectively higher-precision because it now lives on a smaller dynamic range.

This is **exactly** the structure of [GPTQ's sequential compensation](../08-brain-surgery/), with one variable renamed. In GPTQ:

1. Pick the next weight column to quantize.
2. Snap to the nearest grid point (this is the "quantize the residual" step).
3. **Update the *remaining* weight columns** so the *layer output* is restored — using the Cholesky factor of the Hessian as the prediction model.
4. Repeat.

The Cholesky factor is the analogue of the LPC predictor. Both encode "given what we already know about this signal, what would we *expect* the next sample to look like, so we can compensate when reality differs from prediction." In LPC the model is the speaker's vocal tract; in GPTQ the model is the layer's input covariance. The math of the update is structurally identical.

```pyplot {id="dpcm-residual-shrinks" caption="DPCM's central idea: prediction shrinks the variance, so the same quantizer gets higher effective resolution. The same logic underwrites GPTQ's compensation: every quantized weight reduces the variance of the remaining weights' optimal corrections."}
np.random.seed(0)
n = 256
sig = np.cumsum(np.random.randn(n) * 0.3)  # random walk: smooth speech-like signal

# DPCM with a one-step predictor: predict sample[i] = sample[i-1]
residual = np.diff(np.concatenate([[0], sig]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.hist(sig, bins=40, color='#FF8C00', alpha=0.7, edgecolor='#1A1A1A', linewidth=0.3)
ax.set_title(f"Original signal\nstd = {sig.std():.2f}")
ax.set_xlabel("amplitude")
ax.set_ylabel("count")
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
ax.hist(residual, bins=40, color='#FF007F', alpha=0.7, edgecolor='#1A1A1A', linewidth=0.3)
ax.set_title(f"DPCM residual (after prediction)\nstd = {residual.std():.2f}  ({sig.std()/residual.std():.1f}x narrower)")
ax.set_xlabel("residual amplitude")
ax.set_ylabel("count")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The compression community had a name for this idea by 1980: **prediction-error quantization**. Modern LLM quantization rediscovered it, called it "Hessian-aware compensation," and won the day with it. Same idea, same math, fifty years later.

## Generation 3: Transform Coding And The Rotation Methods

In **1974**, **Nasir Ahmed** publishes the **Discrete Cosine Transform (DCT)**. Two years later, the same idea — *transform a signal into a basis where it's sparse, then quantize the coefficients* — becomes the foundation of subband and transform coding. The DCT has a magic property: when you apply it to natural images, the result has **most of its energy in a few low-frequency coefficients**. You can quantize the high-frequency coefficients to near-zero precision because they barely matter.

By 1992, this becomes **JPEG**. By 1993, **MP3** (using a relative of the DCT called the MDCT). The pattern: apply an orthogonal transform; coefficients in the new basis are concentrated; quantize cheaply because most coefficients are tiny.

Why does this work? Because images, audio, and video all have a property called **basis-coherent structure**. The signal isn't sparse in the time domain (most pixels are non-zero, most audio samples are non-zero), but it *is* sparse in some other basis (frequency, wavelet, etc.). Transform coding finds that basis and exploits it.

LLM activations have an *opposite* problem. They aren't sparse — they have outliers. A handful of channels have giant magnitudes; the rest are bulk Gaussian. The transform-coding move applied to LLM tensors has to *spread the outliers out* across many channels rather than concentrate energy in a few. Same trick, opposite direction.

That's exactly what [QuIP, QuaRot, SpinQuant, and TurboQuant do](../17-rotations/): apply an orthogonal rotation (typically a fast Hadamard transform) to make outliers *less concentrated*, so per-channel quantization works. The orthogonality preserves the matmul; the rotation re-parameterizes the data into a more quantization-friendly basis. We unpack this as a standalone concept in [The Rotation Trick](../17-rotations/).

The lineage is unmistakable. Transform coding picks a basis to make the signal *more compressible*. The rotation methods pick a basis to make the signal *more uniformly compressible*. Different objective, same machinery.

## Generation 4: Vector Quantization And The Road Not Taken

In **1980**, **Yoseph Linde, Andrés Buzo, and Robert Gray** publish what becomes known as the **LBG algorithm** — the multidimensional generalization of [Lloyd-Max](../03-lloyd-max/) to vector quantization. Instead of quantizing each scalar independently, partition $\mathbb{R}^d$ into a Voronoi diagram of $K$ codebook centroids, encode each $d$-dimensional vector by its centroid index, and decode by looking up the centroid. The codebook is *learned* (often by k-means) on training data.

VQ is information-theoretically *better* than scalar quantization. Recall from [the rate-distortion bridge](../04-rate-distortion/) that scalar Lloyd-Max sits about 1.5 dB above the Shannon frontier, and that gap closes only with vector quantization. VQ can squeeze closer to the limit because it exploits *joint* structure across multiple samples.

LLM weight quantization has *almost entirely ignored* this. Almost every modern method — GPTQ, AWQ, NF4, HQQ — is **scalar**: each weight is independently snapped to a 1-D grid (with a per-block shared scale). The 1.5 dB of gap to Shannon is **left on the table**.

Why? **Hardware.** GPU matmul kernels do scalar arithmetic. They cannot, economically, do a "look up the centroid index in a codebook" operation inside a dot-product loop without dispatching to a separate kernel that destroys throughput. Vector quantization requires the multiplication to know which centroid is being used, which requires a memory load that scalar quantization avoids.

The exceptions to this rule are interesting. **Product quantization (PQ)**, used in vector-search libraries like FAISS, decomposes high-dimensional vectors into many low-dimensional sub-vectors, each independently VQ'd. PQ has been used for retrieval, embedding compression, and (recently, in research) LLM weight compression. **AQLM** (Egiazarian et al., 2024) uses additive quantization (a PQ relative) to reach 2-bit on Llama with surprisingly little quality loss. These are the early signs of generation-4 methods entering the LLM stack.

```pyplot {id="vq-vs-scalar" caption="Scalar 4-bit quantization (16 levels per axis = 256 cells) vs. vector 8-bit quantization (256 codebook centroids in 2D). Same bit budget; vector quantization adapts to the data shape, scalar does not."}
np.random.seed(1)
n = 600
# Correlated 2D Gaussian
data = np.random.randn(n, 2) @ np.array([[1.0, 0.6], [0.3, 0.7]])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Scalar quantization: 16 levels per axis
ax = axes[0]
ax.scatter(data[:, 0], data[:, 1], s=8, color='#1A1A1A', alpha=0.4)
for v in np.linspace(-3, 3, 16):
    ax.axvline(v, color='#FF8C00', linewidth=0.4, alpha=0.6)
    ax.axhline(v, color='#FF8C00', linewidth=0.4, alpha=0.6)
ax.set_title("Scalar 4-bit per axis (256 cells)\nGrid is axis-aligned, not data-aligned")
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.spines[['top', 'right']].set_visible(False)

# Vector quantization: 256 codebook centroids learned from data (mini k-means)
ax = axes[1]
K = 256
# crude k-means: initialize from random data points, refine 8 steps
centroids = data[np.random.choice(n, K, replace=False)].copy()
for _ in range(8):
    dists = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    assign = np.argmin(dists, axis=1)
    for k in range(K):
        if (assign == k).any():
            centroids[k] = data[assign == k].mean(axis=0)
ax.scatter(data[:, 0], data[:, 1], s=8, color='#1A1A1A', alpha=0.4)
ax.scatter(centroids[:, 0], centroids[:, 1], s=14, color='#FF007F', alpha=0.8)
ax.set_title("Vector 8-bit (256 codebook centroids)\nGrid follows the data shape")
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Visually you can see what scalar quantization is leaving on the table: many scalar grid cells contain *no data*, and many regions of data are split across multiple grid cells. Vector quantization aligns the cells to the data manifold. For correlated data — which LLM weights certainly are — this matters.

The reason the field hasn't gone hard on VQ for weights is hardware, but the reason might evaporate. Specialized accelerators that *can* do codebook lookup (like Cerebras's wafer-scale or some future generations of Blackwell) would unlock VQ as a first-class option. AQLM is the canary; expect more.

## Generation 5: Perceptual Optimization And OmniQuant

In **1985**, **Schroeder and Atal** publish **CELP** — Code-Excited Linear Prediction. The encoder maintains a small codebook of "excitation" patterns. For each block, it iteratively searches for the codebook entry that, when passed through the LPC prediction filter, produces the closest match to the original block under a **perceptually-weighted error**. The perceptual weighting is a frequency-dependent reshape that emphasizes errors humans hear and ignores errors humans don't.

This is the first quantization method that **optimized against an error model that wasn't squared error**. The MSE-vs-perceptual gap is the difference between MP3 (which sounds great) and a naive PCM downsampling (which sounds awful). The bits saved by ignoring inaudible errors are bits you can spend on audible ones.

LLM quantization has the same idea, with "perceptually weighted" replaced by "**Hessian weighted**" or "**output-reconstruction weighted**". A weight quantization error projected onto a flat direction of the loss landscape is "inaudible" to the model; a tiny error projected onto a steep direction is catastrophic. GPTQ-family methods use the layer Hessian as the weighting matrix. **OmniQuant** and **AdaRound** go further and *learn* the per-block placement parameters by gradient descent against the layer's reconstruction loss.

This is generation 5 in the LLM stack. We have not yet built the analogue of MPEG-4 AAC — a fully perceptual, end-to-end-optimized quantizer for LLMs — but OmniQuant and EfficientQAT are the first steps.

## What's On The Tree, What Isn't, And What That Predicts

Mapping the existing LLM techniques onto the family tree:

| LLM technique | Compression generation | Notes |
|---|---|---|
| Absmax / RTN | G1 | Pure PCM with adaptive scale |
| NF4 | G1 + G4 (degenerate) | A 1-D Gaussian-optimal scalar quantizer |
| GPTQ / SparseGPT | G2 | Predictive (Hessian) compensation |
| AWQ / SmoothQuant | G2.5 | Activation-statistic-driven scale; no full prediction loop |
| QuIP, QuaRot, SpinQuant, TurboQuant | G3 | Transform → quantize |
| AQLM | G4 | True vector quantization, additive form |
| OmniQuant, AdaRound | G5 | Learned perceptual (loss-weighted) parameters |
| QAT (LLM-QAT) | G5+ | Full perceptual loop with task loss |

Two observations.

**First, generation 4 is underdeveloped.** Almost no LLM method is doing real vector quantization. The 1.5 dB compression-frontier gap is sitting there, ready to be picked up by hardware that can do codebook lookups in a matmul. **Predict the next decade**: the first inference accelerator to do native VQ matmul will unlock a 0.5-bit-per-weight reduction across the entire stack.

**Second, no current LLM method is doing real *combined* G2+G3.** Speech codecs combine prediction *and* transform coding (e.g. AAC's MDCT followed by predictive coding of the coefficients). For LLMs, you could imagine GPTQ-on-rotated-weights (apply Hadamard rotation, then run GPTQ). A few research papers have started doing this — QuaRot composes with GPTQ — but it's not the default. The "stack the generations" recipe has decades of speech-codec evidence behind it; expect it to dominate LLM quantization by ~2027.

**Third, generation 6 is on the horizon.** In speech and audio, the post-2010 trend was **learned end-to-end neural codecs** (SoundStream, Encodec). They beat all the classical methods at low bitrates. The LLM analogue is **learning a quantizer for the model jointly with the model**. EfficientQAT and the FP4-pretraining experiments coming out of frontier labs are early steps. The cleanest version of this — a transformer that *trains itself* to be quantization-friendly — has not been built yet, but probably will be.

## A Recap In One Diagram

The shortest version of this article: **almost everything modern LLM quantization does was first done by speech and image codec engineers, between 1937 and 2000**.

```pyplot {id="generations-radial" caption="Five generations of compression engineering, with the modern LLM quantization analogue mapped onto each."}
import numpy as np
fig, ax = plt.subplots(figsize=(9.5, 5))

generations = [
    ("G1: PCM\n(1937–57)",            "Absmax, RTN, NF4 placement"),
    ("G2: Predictive\n(1952–72)",     "GPTQ, SparseGPT"),
    ("G3: Transform\n(1974–85)",      "QuaRot, SpinQuant, TurboQuant"),
    ("G4: Vector Q\n(1980–90)",       "AQLM, FAISS-PQ, (mostly missing)"),
    ("G5: Perceptual\n(1985–2000)",   "OmniQuant, AdaRound, QAT"),
]

ys = np.linspace(0.85, 0.15, len(generations))
ax.set_xlim(0, 10); ax.set_ylim(0, 1)
ax.axis('off')
for (gen, mod), y in zip(generations, ys):
    ax.add_patch(plt.Rectangle((0.3, y-0.06), 3.0, 0.12, color='#FFD700', ec='#1A1A1A', lw=2))
    ax.text(1.8, y, gen, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(4.3, y), xytext=(3.4, y),
                arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=2))
    ax.add_patch(plt.Rectangle((4.4, y-0.06), 5.2, 0.12, color='#FF007F', ec='#1A1A1A', lw=2))
    ax.text(7.0, y, mod, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
ax.text(1.8, 0.97, "Compression engineering generation", ha='center', fontweight='bold', fontsize=11)
ax.text(7.0, 0.97, "Modern LLM quantization cousin", ha='center', fontweight='bold', fontsize=11)
```

## What To Remember

1. Modern LLM quantization recapitulates **five generations of speech and image compression**, in fast forward.
2. **Predictive coding** (DPCM, LPC) is the structural ancestor of GPTQ's sequential compensation. The Cholesky factor *is* the LPC predictor.
3. **Transform coding** (DCT, MDCT) is the ancestor of the [rotation methods](../17-rotations/). The objective is opposite (spread, not concentrate) but the machinery is the same.
4. **True vector quantization** is mostly missing from the LLM stack. The 1.5 dB gap to Shannon is real, hardware-bounded, and will probably be picked up in the next generation of accelerators.
5. **Perceptually-weighted optimization** (CELP) is the ancestor of OmniQuant and Hessian-aware methods. We're early on this curve.
6. The next LLM-quantization breakthroughs will, with high probability, come from **stacking generations**: rotate-then-predict-then-vector-quantize, with all three stages learned end-to-end. The compression people did this in the 1990s. The ML people are halfway there.

**Continue to** → [The KV Cache Method Family Tree](../15-kv-method-family/) — the second great quantization map of this issue, this one for the activation side of the bargain.
