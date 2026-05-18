---
title: "FP8 Calibration: why scale = 1.0 usually works"
description: "Every other KV quantization method requires calibration data. FP8 doesn't — by default. Why E4M3's dynamic range is usually wide enough, and when it isn't."
blurb:
  - "KIVI, KVQuant, GEAR, ATOM, QServe all require calibration. FP8 KV in vLLM doesn't, by default."
  - "E4M3's three orders of dynamic range are wide enough that most KV activations just fit."
  - "Kimi-K2.5 on FlashMLA shows a consistent downward accuracy shift — the exception that proves the rule."
  - "LLM-Compressor: what to reach for when scale = 1.0 isn't enough."
topics: [quantization, calibration, primer]
tags: [fp8, calibration, llm-compressor, per-tensor, per-head, kimi-k2, flashmla]
theme: teal
math: true
draft: false
date: 2026-05-18T09:00:00-04:00
issue: 9
weight: 100
techKind: primer
techNode: calibrate
header: default.webp
---

## The Default Is Audacious

Every other quantization method in [Issue 3's KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/) requires calibration. KIVI does a per-layer K-channel calibration. KVQuant fits non-uniform 4-bit grids per layer using a calibration dataset. GEAR does a streaming SVD per fill. ATOM does end-to-end W4A4KV4 calibration. QServe migrates outliers via SmoothQuant. The list goes on.

The vLLM FP8 KV-cache default does **none of this**. It uses a single per-tensor scale, fixed at **1.0**, no calibration data, no per-head tuning. The K and V tensors are stored directly in FP8 E4M3, with the values just cast from their BF16 originals (clipped to the FP8 range if needed). Read back, dequantize, attend.

This is audacious in a way that easy to miss. For INT8 or INT4 quantization, a per-tensor scale of 1.0 would catastrophically lose precision on the bulk values, because the integer grid's range is small. For FP8 with its three orders of magnitude of dynamic range, the same default works because *the bulk just fits*. The geometry of E4M3 is wide enough that even with K and V activation distributions varying across layers, the data lands somewhere usable on the 16-rungs-per-binade grid.

But "wide enough for most models" is not "wide enough for every model". This chapter is about what happens when the default isn't enough, and what to reach for when it isn't.

## What "Calibration" Actually Means For FP8 KV

For FP8 KV-cache, calibration means computing a per-tensor (or per-head, or per-channel) **scale factor** $s$ that the kernel applies during quantization:

$$
\text{stored\_FP8} \;=\; \text{cast\_to\_FP8\_E4M3}\!\left( \frac{\text{value\_BF16}}{s} \right)
$$

and the inverse during readback:

$$
\text{dequantized\_BF16} \;=\; \text{cast\_to\_BF16}(\text{stored\_FP8}) \times s
$$

If $s = 1.0$ (the default), this is the no-op cast from [Chapter 3](../03-what-is-fp8-here/) — store the FP8 representation of the value as-is, read it back as-is. If $s > 1.0$, you're dividing the input by $s$ before storage, which compresses the dynamic range — useful if your values are larger than what FP8 E4M3 can represent. If $s < 1.0$, you're amplifying before storage, which is useful if your values are systematically small and you want to use more of FP8's mantissa precision.

The right scale, for any given tensor, is roughly:

$$
s^* \;\approx\; \frac{\max |x|}{448}
$$

where 448 is the maximum representable value in E4M3. Choosing $s$ this way ensures that the largest values land near the top of E4M3's range, using the full mantissa precision available.

The reason `scale = 1.0` works as a default is that, for most well-trained transformer K and V tensors, $\max|x|$ is already in the right neighborhood of FP8's representable range. Llama, Qwen, most Mistral, most open-weight models — their K and V tensors typically have max magnitudes in the 10-100 range, which sits comfortably below E4M3's 448 ceiling without needing rescaling. The simplicity is real.

## When The Default Fails: The Kimi-K2.5 Case

The team ran their accuracy sweep across many models and found one model where the default produced a *consistent* accuracy regression that two-level accumulation could not fix: **Kimi-K2.5**.

Kimi-K2.5 is a Moonshot AI model that uses the **FlashMLA** attention backend (their multi-head latent attention variant of FlashAttention). The FlashMLA kernel is a different code path than the FA3 kernel; it inherits the FP8 quantization conceptually but is implemented separately.

On the same long-context benchmark (MRCR with 4 needles, sequence lengths up to 1M tokens), Kimi-K2.5 with FP8 KV cache showed a small but **systematic** downward shift across every context bucket. Not the catastrophic 91→13% collapse the cold open opened with — more like 2-4 percentage points lower at every bucket, with the gap not closing at any context length. The aggregate AUC drop was modest, but the curve was clearly biased low.

```pyplot {id="kimi-shift" caption="Synthetic illustration of the Kimi-K2.5 / FlashMLA accuracy pattern. Not a catastrophe, but a consistent downward bias across context lengths. The shift is systematic enough that uncalibrated FP8 is leaving accuracy on the table for this model + backend combination."}
import numpy as np
import matplotlib.pyplot as plt

contexts = np.array([8, 16, 32, 64, 128, 256, 512, 1024])

# BF16 baseline (slightly noisy)
bf16 = np.array([0.92, 0.91, 0.89, 0.87, 0.84, 0.81, 0.78, 0.74])
# FP8 (calibrated, near baseline)
fp8_cal = bf16 - 0.01
# FP8 uncalibrated on Kimi-K2.5 / FlashMLA — consistent downward shift
fp8_uncal = bf16 - 0.04

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(contexts, bf16, color='#00A8A8', linewidth=2.6,
        marker='o', markersize=6, markerfacecolor='#FFD700',
        markeredgecolor='#1A1A1A', markeredgewidth=1.0,
        label='BF16 baseline')
ax.plot(contexts, fp8_cal, color='#FFD700', linewidth=2.2,
        marker='s', markersize=6, markeredgecolor='#1A1A1A',
        markeredgewidth=1.0, linestyle='--',
        label='FP8 with LLM-Compressor calibration')
ax.plot(contexts, fp8_uncal, color='#FF007F', linewidth=2.6,
        marker='^', markersize=7, markerfacecolor='#FFD700',
        markeredgecolor='#1A1A1A', markeredgewidth=1.0,
        label='FP8 uncalibrated (Kimi-K2.5 / FlashMLA)')

ax.fill_between(contexts, fp8_uncal, bf16, color='#FF007F', alpha=0.12)
ax.text(128, 0.85, 'systematic ~3-4 pt shift\nat every context length',
        ha='center', fontsize=9, style='italic')

ax.set_xscale('log')
ax.set_xlabel('context length (k tokens, log scale)')
ax.set_ylabel('MRCR pass@1')
ax.set_ylim(0.65, 1.0)
ax.set_title('When uncalibrated FP8 leaves accuracy on the table',
             fontsize=11, fontweight='bold')
ax.legend(loc='lower left', framealpha=1, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.15)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

This is a **calibration-shaped** failure mode, not an accumulator-shaped failure mode. The two-level fix has nothing to say here — the accumulator is fine on FlashMLA, and Blackwell-style precision issues aren't the cause. The cause is that Kimi-K2.5's K and V tensor distributions, on FlashMLA's specific code path, sit at a position in E4M3's grid where uncalibrated `scale = 1.0` is *systematically biased*. A calibrated scale recovers most of the lost accuracy.

The diagnostic pattern is informative. **A flat downward shift** across all context lengths usually means a quantization bias — the values are being systematically quantized in the wrong direction. **A divergence at long context** (like the cold-open 91→13% catastrophe) usually means an accumulator problem — the error compounds with the contraction dimension. The two failure modes look different on the accuracy-vs-context plot, and they have different fixes.

## When To Reach For Calibration

The decision rule, in three cases:

**Case 1: well-validated model + FA3 (Hopper) or FlashInfer (Blackwell).** This is the majority of production deployments — Llama family, Qwen family, Mistral family, most open-weight models. The team's exhaustive sweep showed that uncalibrated FP8 + two-level accumulation gets within 1-3 percentage points of BF16 on every benchmark. **Use the default.** Don't bother with calibration.

**Case 2: non-standard attention backend.** Kimi-K2.5 with FlashMLA is the canonical example. The kernel path hasn't inherited the same level of optimization as FA3 or FlashInfer, and the FP8 quantization may interact with the kernel's internal numerics differently. **Calibrate** — usually the simplest per-tensor scales suffice to close the gap.

**Case 3: novel model family or unusual training recipe.** Models with weird activation distributions (some FP8-pretrained models, models that have been heavily fine-tuned with techniques that change activation statistics) may have K and V distributions that don't fit E4M3's default range. **Sweep a few scale values** as a sanity check; if you see consistent biases across benchmarks, calibrate.

The case-by-case decision is, in practice, a flowchart with one fast path (use the default) and one slow path (calibrate if you observe a consistent issue).

## How LLM-Compressor Works

When you do need to calibrate, the recommended tool is **[`vllm-project/LLM-Compressor`](https://github.com/vllm-project/llm-compressor)** — vLLM's official quantization tooling. The high-level workflow:

```python
from llmcompressor import calibrate_kv_cache_scales

# 1. Provide calibration data
calibration_data = load_dataset('your-calibration-set')

# 2. Run calibration
scales = calibrate_kv_cache_scales(
    model='Kimi-K2.5',
    backend='flashmla',
    granularity='per-tensor',   # or 'per-head', 'per-layer'
    calibration_data=calibration_data,
    n_samples=512,              # forward-pass through 512 examples
)

# 3. Save scales to disk
scales.save('kimi-k2.5-fp8-kv-scales.safetensors')

# 4. Serve with calibrated scales
# vllm serve Kimi-K2.5 --kv-cache-dtype fp8 \
#                       --kv-cache-scales-path kimi-k2.5-fp8-kv-scales.safetensors
```

The calibration step runs a forward pass through N representative examples (typically 256-1024), records the maximum and standard deviation of the K and V tensor at each layer / head / channel, and computes the optimal scale for each granularity. The output is a small `safetensors` file with the scale tensors. At serve time, vLLM loads the scales and applies them during the FP8 quantization step.

The calibration granularity is a knob:

| Granularity | Scales | Accuracy | Bookkeeping |
|---|---:|---|---|
| Per-tensor | 1 per layer | Coarsest | Trivial |
| Per-head | $H$ per layer | Medium | Small |
| Per-channel | $D$ per layer | Finest | Largest |

For most models that need calibration, **per-head scales** are the sweet spot. They capture most of the activation distribution variation (different heads attend to different patterns and have different value distributions) while keeping the bookkeeping small. Per-channel scales are rarely worth it for FP8 — the marginal accuracy gain over per-head doesn't justify the extra book-keeping and the slight runtime cost.

For Kimi-K2.5, the team's experiments showed that per-tensor calibration recovered most of the 3-4 point shift; per-head calibration closed the rest. The full calibration step takes about an hour on a small calibration set, and the resulting scales file is a few megabytes — well within practical deployment budgets.

## The Calibration Lineage

Calibration as a concept is old. It is one of the central topics in [Issue 3's calibration field guide](/issues/03-sixteen-numbers/13-calibration-survey/), which traces five families of calibration approaches:

1. **Data-free** — analytic methods that compute scales without any input data (Lloyd-Max for uniform quantization, the symmetric scaling of [absmax](/issues/03-sixteen-numbers/02a-absmax/)).
2. **Statistics-only** — collect summary statistics from a small sample and use them (the LLM-Compressor approach).
3. **Second-order** — use Hessian information to identify sensitive directions (GPTQ for weights).
4. **Gradient-based** — backpropagate through quantization to fine-tune scales.
5. **Full QAT** — train (or fine-tune) the model with quantization noise present from the start.

FP8 KV-cache calibration is firmly in family 2: **statistics-only**. Run a forward pass, collect max-and-variance per layer / head, fit a scale. The reason it's so simple is that FP8's wide dynamic range makes the scale-fitting problem easy — there isn't a Lloyd-Max optimization to solve, because FP8's grid is already roughly the right shape. You're really just *centering* the data on FP8's range, not designing the grid.

This is the simplest possible end of the calibration spectrum. INT4 calibration requires gradient-based or second-order methods because the grid is too small for statistics alone to fit. FP8 calibration is one Numpy line of code per layer.

{{% callout type="info" title="Why FP8 Calibration Is Easy" %}}
FP8 calibration doesn't require gradient methods, Hessian information, or QAT. The grid is already wide enough that you only need to *position* the data correctly within it. A per-tensor scale collected from a few hundred forward passes is sufficient for most cases; per-head scales close the remaining gap on tricky backends like FlashMLA.

This is the architectural reason FP8 KV is the "easy" entry on [the KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/) — the format gives you most of the slack for free.
{{% /callout %}}

## What Per-Head Scales Buy You

The per-head scale machinery — wired into vLLM through [`vllm#30833`](https://github.com/vllm-project/vllm/pull/30833) and [`vllm#30141`](https://github.com/vllm-project/vllm/pull/30141), as discussed in [the tile-sizes chapter](../07-tile-sizes/) — deserves a brief expansion here, because it's the calibration knob most users will actually engage.

Within a single attention layer, different heads can have substantially different K and V distributions. Some heads attend to local patterns (small K vector magnitudes); others attend to specific position embeddings or BOS-like sinks (large K magnitudes in specific channels). A single per-tensor scale forces all heads to share a quantization range that may be too wide for some and too narrow for others.

Per-head scales let each head pick the scale that matches its own distribution. The cost is $H$ scales per layer instead of 1, which is negligible storage. The benefit is that each head's quantization noise is bounded by the local mantissa precision, not by a global compromise. On models where head distributions vary widely (often empirically true for grouped-query attention models with few KV heads), the accuracy gain from per-head scales over per-tensor is 0.5-1 percentage point.

In practice, the team's recommendation in 2026 is:

- Per-tensor scales (the default) for most production deployments.
- Per-head scales for models that show systematic per-head variance in calibration.
- Per-channel scales rarely worth it for FP8 (more useful for INT4, which is a different chapter).

## Calibration As Insurance

The honest framing of calibration is **insurance**. The simple default works on most production models, most of the time, with no setup cost. Calibration adds a calibration step (which has compute cost and operational complexity), produces a scales artifact (which must be versioned and tracked), and recovers a small accuracy increment that may or may not matter for your application.

For most engineering teams, the question is whether the operational cost of calibration is justified by the accuracy gain on your workload. If your workload is well-served by the default (and the team's exhaustive sweep suggests this covers most cases), don't bother. If you're seeing measurable accuracy degradation on your own benchmarks, the calibration workflow is straightforward and the gains are real.

The point this issue keeps returning to: **the FP8 KV-cache bargain works because the default is good enough for most cases**. The team's job over the past three months has been to *make* the default actually good enough — which required the [accumulator fix](../05-accumulator-lie/), the [tile-size optimization](../07-tile-sizes/), the [skip-SW flag](../09-sliding-window-puzzle/), and the per-head scale wiring. Calibration is the last-resort tool for the cases where even the polished default doesn't quite reach.

## What To Remember

1. **The default in vLLM is `scale = 1.0`, no calibration.** This works because FP8 E4M3's dynamic range is wide enough that most K and V distributions fit without rescaling.
2. **A consistent downward shift across context lengths** is the signature of an uncalibrated FP8 scale issue — different from the long-context accumulator collapse from earlier chapters.
3. **Kimi-K2.5 + FlashMLA** is the canonical case where the default doesn't work. Per-tensor calibration recovers most of the gap.
4. **LLM-Compressor** is the tooling. Run forward passes, collect statistics, fit scales, save a small artifact. Per-tensor → per-head → per-channel is the granularity ladder; per-head is the typical sweet spot.
5. **Treat calibration as insurance**, not as a routine step. The default covers most models; calibrate when you see measurable accuracy degradation on your own workload.

**Continue to** → [The State of FP8 KV](../11-state-of-fp8/) — the boss capstone. The full report card across models and benchmarks, the three places FP8 KV still doesn't fit, and the connection back to [Issue 3's KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/).
