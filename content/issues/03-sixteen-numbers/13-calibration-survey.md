---
title: "Calibration, A Field Guide"
description: "The other half of every quantization method: how it picks its scales. A taxonomy of calibration strategies, the tradeoff each one makes, and the eighty years of compression work they all rest on."
topics: [quantization, calibration]
tags: [calibration, ptq, qat, hqq, gptq, awq, smoothquant, omniquant, history]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 130
techKind: primer
techNode: calibration-survey
header: default.webp
---

## A Phone Call From Murray Hill

Sometime in **early 1972**, an engineer at **Bell Northern Research** named **Patrick Crochiere** picks up a long-distance call. The voice on the other end is degraded — not noise exactly, but a wobble, a kind of electronic *unease*. He recognizes it immediately. His company has just rolled out a new digital trunk line that compresses each voice sample using **adaptive differential PCM**: instead of storing the absolute voltage of every sample, the line stores only the *difference* from a running prediction, with a quantizer whose **step size adapts to the magnitude of recent samples**.

The wobble is the adaptation working too slowly. Each time a speaker shifts from a quiet sibilant to a loud vowel, the quantizer takes about three milliseconds to **re-calibrate**. For three milliseconds, the quantization is wrong: the step size is sized for the previous regime, not the current one. The brain hears it as a soft, rolling distortion.

Crochiere's fix is one of the foundational papers of the field. He proposes that the encoder transmit a small *header* before each block of samples — twenty-four bits of metadata announcing the step size for the next block. Decoders receive the header, set their reconstruction grid to match, and process the block. Calibration becomes **explicit, periodic, and free of feedback delay**.

The paper is published in **1976**. Forty-six years later, when **Elias Frantar** sits down to write GPTQ, his calibration has the same shape: read a small block of data, decide on a per-block scale, encode that scale, encode the values. The difference is that Crochiere's "block" was a hundred milliseconds of speech and Frantar's was a hundred and twenty-eight columns of a Llama {{< wiki "transformer-weights" >}}weight matrix{{< /wiki >}}. The bones of the technique are identical.

This article is the field guide to **calibration** — the half of every quantization method that decides where the dots go. We will tour the modern menagerie (data-free, statistics-only, Hessian-aware, gradient-based) with one eye on the long lineage that all of these methods inherit from, and one eye on the practical question of *which one you actually want to use*.

## What Even Is Calibration?

Strip a quantization method to its bones and you have two choices to make:

1. **The grid:** which set of representable values? (Picked once, often by the format — {{< wiki "number-formats" >}}INT8{{< /wiki >}}, FP4, NF4.)
2. **The placement:** where does that grid sit, relative to the data? (Picked per tensor / per row / per block — the **scale** and possibly the **zero-point**.)

**Calibration is the second choice.** It is the act of looking at some sample of the data — or no data at all — and producing the parameters that locate the grid. Everything from "use the maximum absolute value" ([absmax](../02a-absmax/)) to "solve a constrained optimization with second-order information" ([GPTQ](../08-brain-surgery/)) is a calibration strategy. The *only* difference between a five-line absmax script and a hundred-thousand-line OmniQuant codebase is **how much work is being done at calibration time, and how much information is being used to do it**.

The reason this matters is empirical: the *grid* matters less than people think, and the *placement* matters more. Two methods using identical 4-bit grids with different calibration can differ by **3× in perplexity**. Two methods using radically different grids (NF4 vs INT4) with identical calibration often differ by less than 1%.

> The art of low-bit quantization is mostly the art of calibration.

That sentence is the entire intellectual content of this article. The rest is a tour.

## A Taxonomy In Five Buckets

The space of calibration strategies has converged, by 2026, on five distinct families. Each is defined by **what kind of information it consumes** to set the scales.

| Bucket | Information used | Cost | Representative methods |
|---|---|---|---|
| **0. Static** | None — fixed by the format | $0$ | NF4 grid (the *grid*, not the placement) |
| **1. Data-free** | Only the weights themselves | seconds | RTN, HQQ, OBQ-data-free |
| **2. Statistics-only** | First-order activation statistics on a calibration set | minutes | SmoothQuant, AWQ, ZeroQuant |
| **3. Second-order** | Activation Hessians on a calibration set | minutes–hours | GPTQ, OBQ, SqueezeLLM, SparseGPT |
| **4. Gradient-based** | Backpropagated gradients of a reconstruction loss | hours–days | OmniQuant, AdaRound, LSQ, BRECQ |
| **5. Full QAT** | Backpropagated gradients of the *task* loss | days–weeks | LLM-QAT, EfficientQAT |

The buckets are roughly ordered by **information consumed** and therefore by **cost**, and roughly inversely by **how much accuracy you can squeeze out of a given bit budget**. Move down the list and you spend more compute and more data, and you generally get a better quantization. The sweet spot for production-scale LLM weight quantization in 2026 is **bucket 3**: GPTQ-family methods. For inference engines that need fast turnaround, bucket 1 (HQQ) wins. For training-time integration, you have no choice but bucket 5.

We will visit each bucket.

## Bucket 0 — Static, Data-Free Grids

This is the boundary case. **No calibration at all.** The number format itself fixes both the grid and the placement.

The cleanest example is the [Lloyd-Max picture](../03-lloyd-max/): pick 16 levels at the conditional centroids of a *standard* normal distribution. This is the **NF4** grid. It is computed *once*, in a notebook, and burned into a lookup table. Every quantizer that uses NF4 uses the *same* sixteen real numbers:

$$
\{-1.0,\ -0.6962,\ -0.5251,\ -0.3949,\ -0.2844,\ -0.1848,\ -0.0911,\ 0,\ 0.0796,\ 0.1609,\ 0.2461,\ 0.3379,\ 0.4407,\ 0.5626,\ 0.7230,\ 1.0\}
$$

At calibration time, the *placement* is still chosen — typically a per-block absmax scale stretches the [-1, 1] grid to fit the block — so NF4 isn't *fully* calibration-free. But the grid itself never depends on data. It is a pure consequence of an assumption (weights are unit-normal-ish) and a theorem (Lloyd's centroid condition).

**Why mention it?** Because it isolates a lesson: a clever *static* grid plus a tiny per-block placement step can beat a clumsy *data-driven* grid plus an elaborate calibration. The format is doing real work even before any data shows up.

## Bucket 1 — Data-Free Calibration: HQQ And The OBQ Family

In this bucket, you have the full weight matrix in front of you, but **no activations and no calibration data**. The only thing you can reason about is the *weights themselves*.

The naivest member is **round-to-nearest** (RTN): set $s = \max|w| / q_{\max}$ per row or per block, snap each weight to its nearest grid point. This is what [absmax](../02a-absmax/) does, applied at whatever blocking granularity you like. It is the *baseline* every modern method must beat.

The 2023 surprise was **HQQ — Half-Quadratic Quantization** by Hicham Badri and Appu Shaji at Mobius Labs. HQQ formulates quantization as the optimization

$$
\min_{\hat W} \|W - \hat W\|_p^p + \lambda \cdot \mathbb{1}[\hat W \in \mathcal{G}]
$$

— minimize the $L_p$ distance from the original weights to a $\hat W$ constrained to lie on the chosen grid $\mathcal{G}$. They use $p = 0.5$, which up-weights small errors and down-weights large ones, making the solution implicitly **outlier-aware** *without* ever consulting an activation. The half-quadratic splitting algorithm — a 1990s technique from image deblurring (Geman & Yang, 1995) — solves this efficiently.

```pyplot {id="hqq-vs-rtn-toy" caption="A toy weight row with one outlier. HQQ-style outlier-down-weighting picks scales that fit the bulk; RTN's max-driven scale wastes precision on the outlier."}
np.random.seed(0)
n = 256
w = np.random.randn(n) * 0.10
w[42] = 2.5  # one outlier

def quantize_int4(w, scale):
    q = np.clip(np.round(w / scale * 7), -7, 7)
    return q * scale / 7

# RTN: scale picked from absmax
s_rtn = np.abs(w).max()
w_rtn = quantize_int4(w, s_rtn)

# HQQ-flavored: pick a smaller scale that down-weights the outlier
# (a true HQQ run is iterative; this is just the steady-state intuition)
s_hqq = np.percentile(np.abs(w), 99.0)
w_hqq = quantize_int4(np.clip(w, -s_hqq, s_hqq), s_hqq)

err_rtn_bulk = np.std(w[w != 2.5] - w_rtn[w != 2.5])
err_hqq_bulk = np.std(w[w != 2.5] - w_hqq[w != 2.5])

print(f"RTN scale = {s_rtn:.3f},  bulk error std = {err_rtn_bulk:.4f}")
print(f"HQQ scale = {s_hqq:.3f},  bulk error std = {err_hqq_bulk:.4f}  ({err_rtn_bulk/err_hqq_bulk:.1f}x better on bulk)")

fig, ax = plt.subplots(figsize=(8.5, 4))
ax.scatter(range(n), w, s=12, color='#1A1A1A', alpha=0.6, label='original')
ax.scatter(range(n), w_rtn, s=14, color='#FF8C00', alpha=0.7, marker='x', label=f'RTN (scale={s_rtn:.2f})')
ax.scatter(range(n), w_hqq, s=14, color='#FF007F', alpha=0.7, marker='+', label=f'HQQ-style (scale={s_hqq:.2f})')
ax.axhline(s_rtn, color='#FF8C00', linestyle=':', alpha=0.4)
ax.axhline(-s_rtn, color='#FF8C00', linestyle=':', alpha=0.4)
ax.set_xlabel("weight index")
ax.set_ylabel("value")
ax.set_title("Same row, two scales — HQQ-style ignores the outlier and fits the bulk")
ax.legend(loc='lower right', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
```

The HQQ trick is **outlier robustness without seeing any data**. You can do this because outliers in the *weights themselves* are statistically detectable — a single isolated giant value is almost certainly more important to leave alone than to crush the bulk for. Activations are a separate story (they have their *own* outliers, much more dramatic), but for the weight half of the bargain, HQQ extracts surprising mileage from no calibration data at all.

A related family is **Optimal Brain Quantization (OBQ)**: the same Hessian-compensation idea as GPTQ, but with $H$ replaced by the identity (so each weight is treated independently). OBQ runs without calibration data and gives slightly worse results than GPTQ, slightly better than HQQ. The ranking is consistent across many models.

**When to use bucket 1:** when you cannot ethically obtain a calibration set (proprietary data, fine-tuned models with sensitive distributions, on-device fine-tuning). When you need a quantized model in **seconds**, not minutes.

## Bucket 2 — Statistics-Only Calibration: SmoothQuant, AWQ, ZeroQuant

Step one rung up the ladder: now you allow the calibration step to **forward-pass a small batch of inputs** through the model and gather **first-order statistics** about activations. No backward pass, no Hessians.

The statistics are typically simple:

- **per-channel mean absolute activation**: $\bar{s}_i = \mathbb{E}_t |x_{ti}|$
- **per-channel max activation**: $m_i = \max_t |x_{ti}|$
- **outlier indicator**: $\mathbb{1}[m_i > \alpha]$ for some threshold $\alpha$

That's it. A **single forward pass over 128 samples** suffices. No gradient tape, no optimizer, no calibration loop. It runs in minutes even on a 70B-parameter model.

The two canonical members of this bucket are the migration-identity twins — **SmoothQuant** and **AWQ** (both covered in [The Method Family Tree](../09-method-family-tree/)) — which use the per-channel activation magnitudes to choose a diagonal rescaling matrix $S$ via the algebraic identity $Y = (XS^{-1})(SW^\top)$:

- **SmoothQuant** uses $S$ to push outliers from activations *into* the weights, where they're easier to quantize, so that **both** $X$ and $W$ end up INT8-quantizable.
- **AWQ** uses $S$ to *protect* weight columns that interact with large activations, leaving the activations in FP16 but choosing the weight scales so that the most-important weights survive 4-bit rounding cleanly.

The two methods use the *same* statistic (per-channel activation magnitudes) and the *same* algebraic identity ($Y = (XS^{-1})(SW^\top)$), but apply it for opposite ends. They are calibration-equivalent — the work happens in choosing $S$, and once $S$ is chosen the rest is straightforward absmax.

**ZeroQuant** (Yao et al., 2022) is an earlier and more general version of the same family — per-channel scales for activations, per-row scales for weights, fine-tuning-free, INT8 throughout. It came out a few months before SmoothQuant and shares the same intellectual lineage.

```pyplot {id="awq-stat-grid" caption="Per-channel activation statistics on a synthetic 12-sample calibration batch. AWQ-style methods use this single picture as input to choose weight scales."}
np.random.seed(3)
batch, dim = 12, 64
X = np.random.randn(batch, dim) * 0.4
outlier_dims = [7, 23, 41, 55]
X[:, outlier_dims] *= 25

mean_abs = np.abs(X).mean(axis=0)
max_abs = np.abs(X).max(axis=0)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(dim), mean_abs, color='#00A8A8', label='mean |X[:,j]|', edgecolor='#1A1A1A', linewidth=0.3)
ax.bar(range(dim), max_abs - mean_abs, bottom=mean_abs,
       color='#FF007F', label='max - mean (extra reach of outliers)', edgecolor='#1A1A1A', linewidth=0.3)
for j in outlier_dims:
    ax.annotate('OUTLIER', xy=(j, max_abs[j]), xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=7, color='#FF007F', fontweight='bold')
ax.set_yscale('log')
ax.set_xlabel("activation channel j")
ax.set_ylabel("|activation| (log)")
ax.set_title("Activation statistics from a 12-sample calibration batch — the entire input to AWQ")
ax.legend(loc='upper left', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
```

The lesson: **a tiny amount of activation data carries an enormous amount of calibration signal**. Twelve samples already reveal the outlier channels with high confidence. AWQ in production uses 32; GPTQ uses 128; nobody uses more than 1024. There are no diminishing returns past about 256.

**When to use bucket 2:** when you want the SmoothQuant or AWQ recipe (the most popular weight-quantization defaults of 2024–2026), and you can run a forward pass over a representative sample.

## Bucket 3 — Second-Order Calibration: GPTQ, SqueezeLLM, SparseGPT

Now things get interesting. Allow the calibration step to compute the **layer Hessian** $H = X^\top X$ — a $d \times d$ matrix of second-order activation correlations. This adds substantial cost (one matrix multiply per layer plus a Cholesky) but unlocks a dramatically more powerful technique: [error compensation](../08-brain-surgery/).

The GPTQ recipe in one paragraph: process weight columns left-to-right; for each column, snap to the nearest grid point; compute the resulting reconstruction error; **distribute** the error across the as-yet-unquantized columns using the Cholesky factor of $H^{-1}$. The end result is that the *output* of the layer matches the original far more tightly than naïve rounding would allow, even though the per-weight rounding error is the same.

The deep reason this works is that **the layer Hessian encodes which features cause large output changes**. Quantization errors that project onto small-eigenvalue directions of $H$ are *cheap*; errors along large-eigenvalue directions are *expensive*. The Cholesky compensation rotates errors away from expensive directions and into cheap ones. We unpacked this geometry in the [Taylor & Hessians primer](../07-taylor-and-hessians/).

The whole family inherits this trick:

| Method | Year | Twist on GPTQ |
|---|---|---|
| **GPTQ** | 2022 | original; sequential RTN with Hessian compensation |
| **SparseGPT** | 2023 | same machinery, but for *pruning* ("set this weight to zero") instead of quantizing |
| **SqueezeLLM** | 2023 | uses Hessian to choose a non-uniform grid (sensitivity-weighted Lloyd-Max) |
| **OWQ** | 2023 | mixed-precision: high-sensitivity columns kept in FP16, the rest GPTQ |
| **SpQR** | 2023 | GPTQ + per-row outlier extraction; ~2-bit avg with FP16 outliers |

What unites them: **they all turn the layer Hessian into the calibration signal**. Once you have $H$ and its Cholesky, you can do a remarkable amount of compensation, error analysis, and sensitivity-weighted decision-making — none of which is possible with bucket-2 statistics alone.

**Cost note:** the Hessian for a layer with input width $d$ takes $O(d^2 \cdot N_\text{calib})$ flops to assemble and $O(d^3)$ to Cholesky. For $d = 4096$ and $N = 128$: about 2 GFLOPs to assemble, 70 GFLOPs to factor. On an A100 that's a few seconds per layer. Times 80 layers in a 70B model: a few minutes. Well worth it.

**When to use bucket 3:** when the model is going to be deployed widely and you want every percentage point of accuracy back, and you have tens of minutes of GPU time to spend up front.

## Bucket 4 — Gradient-Based Calibration: OmniQuant, AdaRound, LSQ

Step up another rung: allow the calibration step to **back-propagate** through a reconstruction loss and learn the calibration parameters as if they were model weights. This is sometimes called **PTQ with Adaptive Calibration** or **mini-QAT**. The model itself is *not* fine-tuned — only the per-block scales and zero-points are trained, on a small calibration set, against a layerwise reconstruction loss.

The two foundational papers are:

- **AdaRound** (Nagel et al., 2020) — for each weight $w$, instead of rounding to the nearest grid point, *learn* a binary "round up vs. round down" decision. The decision is parametrized by a sigmoid and trained to minimize layer reconstruction error. Beats GPTQ on CNNs; competitive on LLMs.
- **OmniQuant** (Shao et al., 2023) — learn the entire per-channel rescaling matrix $S$ and per-block clipping factors via gradient descent on a layer reconstruction loss. State-of-the-art on Llama-family models at 2-3 bit precision.

The trick is **what exactly is being optimized**. It's not the model weights. It's the *placement parameters* — scales, zero-points, smoothing factors. Two or three parameters per block, trained for a few hundred steps on a calibration set. The model itself stays frozen.

```pyplot {id="learned-vs-fixed-scale" caption="A toy single-block quantizer. Round-to-nearest with absmax-scaling (RTN) vs. learned-scale gradient descent. The learned scale shrinks slightly to better fit the bulk."}
np.random.seed(0)
n = 64
w = np.concatenate([np.random.randn(n-2) * 0.3, [2.0, -1.8]])

def quantize_int4(w, scale):
    q = np.clip(np.round(w / scale * 7), -7, 7)
    return q * scale / 7

def loss(scale):
    return ((w - quantize_int4(w, scale))**2).mean()

scales = np.linspace(0.05, 3.5, 200)
losses = np.array([loss(s) for s in scales])
s_rtn = np.abs(w).max()
s_learned = scales[np.argmin(losses)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(scales, losses, color='#1A1A1A', linewidth=1.5)
ax.axvline(s_rtn, color='#FF8C00', linewidth=2, label=f'RTN scale = {s_rtn:.2f}')
ax.axvline(s_learned, color='#FF007F', linewidth=2, label=f'learned scale = {s_learned:.2f}')
ax.set_xlabel("scale")
ax.set_ylabel("reconstruction MSE")
ax.set_title("Loss landscape over scale")
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
ax.scatter(range(n), w, s=14, color='#1A1A1A', alpha=0.6, label='original')
ax.scatter(range(n), quantize_int4(w, s_rtn), s=12, color='#FF8C00', marker='x', label=f'RTN, MSE={loss(s_rtn):.4f}')
ax.scatter(range(n), quantize_int4(w, s_learned), s=12, color='#FF007F', marker='+', label=f'learned, MSE={loss(s_learned):.4f}')
ax.set_xlabel("weight index")
ax.set_ylabel("value")
ax.set_title("Quantized weights under each scale")
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

**Cost note:** OmniQuant trains for ~40 epochs over 128 calibration samples on each layer. On a 70B model this is ~6 hours of GPU time. Per percentage point of accuracy recovered, this is ~20× more expensive than GPTQ. Worth it at 2-bit, often not at 4-bit.

**When to use bucket 4:** when you are quantizing to 2-bit or 3-bit and the GPTQ-family is leaving accuracy on the floor. When you're doing extreme low-bit research.

## Bucket 5 — Quantization-Aware Training (QAT)

The maximum-information bucket: allow gradients to flow through the *whole task loss*, with the **quantization simulated** in the forward pass and a straight-through estimator passing gradients backward. The model itself is fine-tuned to be friendly to its quantization grid.

For LLMs this is rare, expensive, and reserved for two cases:

- **Pretraining-time QAT** (LLM-QAT, EfficientQAT-Pre): the model is trained from scratch with quantized forward passes. Used by some frontier labs for FP8/FP4 pretraining; see [Hardware Horizon](../12-hardware-horizon/).
- **Final-stage fine-tune-aware quantization**: take a pre-trained model, apply GPTQ to get a 4-bit checkpoint, then fine-tune for a few hundred steps with QAT to recover the last percentage point. Common in production.

QAT is by far the most expensive calibration strategy, and produces the best accuracy-per-bit. But it requires the *whole training pipeline*, the *whole dataset*, the *whole compute budget*. For most practitioners it's out of reach. For frontier labs it is the default.

## A Visual Of The Cost-Quality Frontier

```pyplot {id="calib-cost-quality" caption="Five calibration buckets, schematic placement on the cost vs. quality axis. Each subsequent bucket pays more compute for diminishing accuracy returns."}
methods = [
    ("RTN",          1,    9.5, '#FF8C00'),
    ("HQQ",          5,    7.0, '#FF8C00'),
    ("AWQ",          50,   5.5, '#00A8A8'),
    ("SmoothQuant",  60,   5.4, '#00A8A8'),
    ("GPTQ",         300,  4.8, '#FF007F'),
    ("SpQR",         400,  4.7, '#FF007F'),
    ("OmniQuant",    20000, 4.5, '#FFD700'),
    ("AdaRound",     5000, 4.6, '#FFD700'),
    ("QAT",          500000, 4.2, '#1A1A1A'),
]
fig, ax = plt.subplots(figsize=(9, 4.8))
for name, cost, ppl, color in methods:
    ax.scatter([cost], [ppl], s=110, color=color, edgecolor='#1A1A1A', linewidth=1.5, zorder=3)
    ax.annotate(name, (cost, ppl), xytext=(8, 4), textcoords='offset points',
                fontsize=9, fontweight='bold')
ax.set_xscale('log')
ax.set_xlabel("calibration cost (GPU-seconds, schematic)")
ax.set_ylabel("perplexity at 4-bit (schematic, lower better)")
ax.set_title("Cost–quality frontier of quantization calibration")
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.3, linewidth=0.4)
ax.axhline(4.0, color='#1A1A1A', linewidth=0.4, linestyle=':')
ax.text(1.5, 4.05, 'FP16 baseline', fontsize=8, alpha=0.5)
```

The plot is schematic, not measured — the exact perplexity numbers depend on model and dataset. But the *shape* is real and persistent: roughly an exponential trade-off between calibration compute and accuracy. Each new bucket buys perhaps 0.3 perplexity units at ~10× the cost of the previous one.

The two natural operating points are:

1. **HQQ / RTN** if you want a quantized model in seconds.
2. **GPTQ / AWQ** if you want best-in-class quality with minutes of work.

Almost everyone lives at one of these two points. Buckets 4 and 5 are research territory.

## How Big A Calibration Set Do You Need?

A persistent surprise of LLM quantization is that the calibration set is *small*. You can quantize a 70B model with **128 samples**. AWQ uses **32**. The famous "sample-efficient" finding is one of the more counterintuitive empirical facts of the field.

```pyplot {id="calib-size-saturation" caption="Schematic perplexity as a function of calibration set size for a GPTQ-style method. Saturation around 128 samples is a robust empirical pattern."}
n_samples = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
ppl = 5.4 + 4 / (n_samples + 0.5)  # synthetic: 1/n decay
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(n_samples, ppl, 'o-', color='#FF007F', linewidth=2, markersize=8,
        markeredgecolor='#1A1A1A', markeredgewidth=1)
ax.axhline(5.4, color='#1A1A1A', linewidth=0.4, linestyle=':')
ax.text(800, 5.42, 'asymptotic best', fontsize=8, alpha=0.6)
ax.axvspan(100, 200, alpha=0.15, color='#FFD700')
ax.text(150, 7.5, '"production sweet spot"\n(128 samples)', fontsize=9, ha='center', fontweight='bold')
ax.set_xscale('log')
ax.set_xlabel("calibration set size")
ax.set_ylabel("perplexity (schematic)")
ax.set_title("Calibration is sample-efficient: 128 samples ≈ 1024 samples")
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.3)
```

Why is calibration so cheap? Three reasons:

1. **The activation Hessian $X^\top X$ is the empirical estimator of a population covariance.** For a $d$-dim covariance, $O(d)$ samples suffice for a full-rank estimate; in practice the **leading eigenvectors** (the only thing GPTQ uses) stabilize even faster.
2. **Outlier channels are stable across inputs.** They show up in dimension 412 every time, in dimension 8194 every time. A handful of samples is enough to identify them.
3. **The relevant statistics are second-moment, not distributional.** GPTQ doesn't care about the *shape* of activations, just their magnitude correlations. Magnitudes converge fast.

The corollary: **the choice of calibration domain matters less than people think**. C4, Wikipedia, code, dialogue — all give nearly identical quantizations. There is some degradation when you radically out-of-distribution it (calibrate on Chinese text, deploy on English; calibrate on JavaScript, deploy on math), but it is typically <1% perplexity. For most production deployments a handful of C4 samples is fine.

## The Compression-Algorithm Heritage

Now stand back. The five-bucket taxonomy above is *exactly* the same taxonomy the speech-coding community converged on in the 1980s and 1990s, with renamed buckets:

| LLM Bucket | Speech-coding analogue | Year |
|---|---|---|
| Static grid | $\mu$-law / A-law PCM | 1972 |
| Data-free | DPCM with fixed step | 1952 |
| Statistics-only | Adaptive DPCM (ADPCM) | 1976 (Crochiere) |
| Second-order | Linear predictive coding (LPC) | 1967 (Atal) |
| Gradient-based | Code-excited linear prediction (CELP) | 1985 (Schroeder/Atal) |

We're walking the same path. The motivation is even similar: more bits at the same rate require more *cleverness*, and the cleverness comes from using more information about the signal. We covered the foundational compression-history thread separately in [Compression's Family Tree](../14-compression-roots/).

The structural identity is striking. ADPCM transmits a per-block step-size header — exactly what GPTQ stores as a per-block FP16 scale. CELP iteratively refines per-block parameters by minimizing a perceptually-weighted reconstruction error — exactly what OmniQuant does, with "perceptually-weighted" replaced by "Hessian-weighted". The vocabulary is different. The math is, in places, *identical*.

## A Calibration Choose-Your-Own Decision Tree

Final recap, framed as the practical choice you have to make.

1. **Do you have access to the training data, the gradients, and a few days of GPU time?** → use QAT (bucket 5).
2. **Are you quantizing to 2-bit and the standard methods are losing accuracy you can't accept?** → use OmniQuant (bucket 4).
3. **Do you have ~32 calibration samples and a few minutes?** → use AWQ or SmoothQuant (bucket 2). Industry default for 4-bit weight quantization.
4. **Do you want the absolute best quality at 3-bit or 4-bit and are willing to spend ~30 minutes of GPU?** → use GPTQ (bucket 3). The accuracy ceiling for PTQ.
5. **Do you have no calibration data at all, or need a quantized model in seconds?** → use HQQ (bucket 1).
6. **Do you just want the quickest possible thing that doesn't catastrophically break?** → use round-to-nearest with per-block scaling (bucket 0/1). It's the baseline; surprisingly hard to beat at 8-bit.

For the [{{< wiki "kv-cache" >}}KV cache{{< /wiki >}}](../10-kv-cache/), the buckets shift around: most KV methods are bucket 1 (data-free, or per-token online) because there's no time to calibrate at inference. The handful that are bucket 2 (KIVI calibrates outlier-channel masks once per layer) need only seconds. The full bucket-3 GPTQ machinery does not really have a KV analogue — KV calibration is a different game, with its own family tree, in [The KV Method Family Tree](../15-kv-method-family/).

## What To Remember

1. Calibration is the half of every quantization method that **places** the grid. It is more important than the *choice* of grid.
2. The taxonomy lives on a **compute-vs-quality frontier**: data-free → statistics-only → second-order → gradient-based → QAT, each ~10× more expensive than the previous.
3. The two natural operating points are **HQQ** (seconds, no data) and **GPTQ/AWQ** (minutes, 32–128 samples).
4. **128 calibration samples is plenty.** Activations are stable enough that you don't need more.
5. The whole structure has a precise analogue in 1970s–80s **speech coding**: ADPCM ≈ AWQ, LPC ≈ GPTQ, CELP ≈ OmniQuant. We are walking a path that compression engineers already mapped.
6. For [KV cache](../10-kv-cache/), the calibration story is different — see the [KV method family](../15-kv-method-family/).

**Continue to** → [Compression's Family Tree](../14-compression-roots/) — the deeper history of how lossy compression became a discipline, and where every modern LLM quantization trick has a 50-year-old cousin.
