---
title: "Thirteen Percent"
description: "March 2026. A routine validation sweep at AWS turns up a 78-point accuracy drop on a flag that's been shipping in vLLM for nearly three years. The detective story begins."
topics: [quantization, inference, kv-cache]
tags: [fp8, vllm, flash-attention-3, needle-in-a-haystack, long-context]
theme: cream
math: true
draft: false
date: 2026-05-17T09:00:00-04:00
issue: 9
weight: 10
techKind: mainline
techNode: cold-open
header: default.webp
---

## Santa Clara / Boston, March 2026

The number on **Jonas Kübler**'s terminal does not make sense.

He runs the script again, just to be sure. The configuration is boring: a single H100 80GB, `vllm bench serve` with concurrency 1, the `--kv-cache-dtype fp8` flag enabled, Llama-3.1-8B as the model, OpenAI's `mrcr` long-context evaluation suite as the workload. He has run this exact configuration hundreds of times in the past two months. The point of the sweep is to certify that the FP8 KV-cache feature — available in vLLM for almost three years, recommended in every "speed up your inference" tutorial, deployed in production by an unknown but probably-large number of users — does what the README claims it does. Halves your {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} memory. Lowers your decode latency. Keeps your model accuracy within rounding error of the BF16 baseline.

At 128,000 tokens of context, the BF16 baseline scores **91%** on the needle-in-a-haystack benchmark. The same model with FP8 KV-cache scores **13%**.

Kübler is an applied scientist at **AWS**. He has been running this sweep with **Eldar Kurtić** at **Red Hat AI** in Boston, who is on Slack three time zones away. Kübler pastes the number. Kurtić replies with the only reasonable response.

> *what.*

Three years of "just enable the flag" advice in vLLM tutorials, and on the most popular open-weight model on the planet, at the context lengths that long-context evaluation actually cares about, the flag is silently destroying long-document recall by seventy-eight points.

## The Benchmark That Caught It

To appreciate the drop you have to know what the needle-in-a-haystack task actually does. The team is using a long-context variant where a single short sentence — *the needle* — is buried somewhere in a 128,000-token document — *the haystack* — and the model is asked a question whose answer depends on retrieving that exact sentence. You score the model by what fraction of the time it retrieves the needle correctly. There is no soft credit; either it found the sentence or it did not.

```pyplot {id="niah-illustration" caption="The needle-in-a-haystack task at 128k context. The full 91→13% catastrophe in one picture. BF16 finds the needle 91% of the time across positions; FP8 KV (before the fix) collapses to 13%."}
import numpy as np
import matplotlib.pyplot as plt

# Synthetic version of the headline curve: accuracy by needle position
positions = np.linspace(0, 128, 32)  # tokens (in thousands)

# Roughly: BF16 stays ~90% with mild dip in the middle.
bf16 = 0.93 - 0.05 * np.exp(-((positions - 64)**2) / (40**2))
bf16 += np.random.RandomState(1).normal(0, 0.02, size=positions.size)
bf16 = np.clip(bf16, 0, 1)

# FP8 (broken): collapses past ~30k as the contraction dim grows
fp8_broken = np.where(
    positions < 30,
    0.85 + np.random.RandomState(2).normal(0, 0.04, size=positions.size),
    0.20 + np.random.RandomState(3).normal(0, 0.06, size=positions.size)
)
fp8_broken[positions > 80] *= 0.6
fp8_broken = np.clip(fp8_broken, 0, 1)

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(positions, bf16 * 100, color='#00A8A8', linewidth=2.8,
        marker='o', markersize=6, markerfacecolor='#FFD700',
        markeredgecolor='#1A1A1A', markeredgewidth=1.0,
        label='BF16 KV (baseline)  ~91%')
ax.plot(positions, fp8_broken * 100, color='#FF007F', linewidth=2.8,
        marker='s', markersize=6, markerfacecolor='#FFD700',
        markeredgecolor='#1A1A1A', markeredgewidth=1.0,
        label='FP8 KV (broken)  ~13%')

ax.axhline(91, color='#00A8A8', linewidth=1, linestyle=':')
ax.axhline(13, color='#FF007F', linewidth=1, linestyle=':')

ax.set_xlabel('needle position (thousands of tokens)')
ax.set_ylabel('retrieval accuracy (%)')
ax.set_ylim(0, 100)
ax.set_title('Llama-3.1-8B, 128k context — the regression that started the investigation',
             fontsize=11, fontweight='bold')
ax.legend(loc='center right', framealpha=1, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.15)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Notice the *shape* of the regression. Up to about 30,000 tokens of context, FP8 KV is fine — it tracks BF16 within a couple of points. Past that, the curve falls off a cliff. Past about 80,000, it goes essentially to noise. The cutoff is sharp, the cutoff is reproducible, and the cutoff is *exactly* the region long-context users actually deploy these models in.

This is the part that scared the team. If FP8 KV had been uniformly bad, somebody would have caught it years ago. If it had been mildly bad — losing a couple of points across the range — it would be a known trade-off, the kind of thing you document in a tutorial. But this — *perfectly fine at short context, then catastrophic at long context* — was a specific kind of bug. It had the fingerprint of something accumulating.

## What FP8 KV-Cache Is Supposed To Do

Before chasing the bug, briefly: what is the flag actually doing?

In vanilla BF16 inference, every key and value vector produced by every attention layer is stored in the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} as 16-bit floats. Two bytes per element. For Llama-3.1-8B at 128k context, the cache footprint is roughly:

$$
32 \text{ layers} \times 2 \text{ (K and V)} \times 8 \text{ heads} \times 128 \text{ head\_dim} \times 128{,}000 \text{ tokens} \times 2 \text{ bytes} \approx 17 \text{ GB}
$$

The `--kv-cache-dtype fp8` flag changes one thing: keys and values are stored in {{< wiki "number-formats" >}}FP8 E4M3{{< /wiki >}} instead. One byte per element. Cache footprint halves to ~8.5 GB. The {{< wiki "attention" >}}attention{{< /wiki >}} kernel — {{< wiki "flash-attention" >}}FlashAttention-3{{< /wiki >}} on Hopper, FlashInfer on Blackwell — reads the FP8 values straight from cache and runs the entire $QK^\top$ and $\text{softmax}(QK^\top) \cdot V$ matmul in **FP8 arithmetic**, accumulating into FP32 registers. No dequantization detour. The accumulator is supposed to keep precision.

In return for one byte per element instead of two, you expect:

1. **2× less HBM traffic per attention step** during decode (the cache is the dominant read).
2. **2× longer contexts** at the same memory budget — or the same context with room left over for more concurrent requests.
3. **Lower {{< wiki "softmax" >}}softmax{{< /wiki >}} arithmetic** because the FP8 path on Hopper runs at 2× the FLOPS of BF16.

Three apparent wins for one apparent loss of precision (mantissa shrinks from 7 bits to 3). The bet — explicit in every paper and tutorial that recommends the flag — is that attention is *forgiving* of FP8 noise, because softmax exponentially squashes small score differences anyway. Most of the precision in BF16 was being wasted. The 2× compression is essentially free.

That is the bet the team thought they were validating.

{{% pullquote type="counter-intuitive" %}}
The bet was correct. The compression was free. The bug was that the attention math itself, on Hopper, was silently losing precision in a different layer of the stack — and nobody had noticed because nobody had specifically benchmarked long-context recall.
{{% /pullquote %}}

## Why Nobody Had Noticed

This is the embarrassing part. The team checks Git: the `--kv-cache-dtype fp8` flag has been in vLLM since **vLLM 0.2.x**, late 2023. The flag is recommended in the official vLLM docs. It is in dozens of MLOps tutorials. It is the standard advice on the `r/LocalLLaMA` Discord. Some unknown but probably-large number of production deployments are running this flag right now.

How did three years of users miss a 78-point accuracy regression?

The honest answer is: **nobody runs needle-in-a-haystack benchmarks on their own deployment.** They run a couple of golden prompts, eyeball the output, and ship. The benchmarks that *are* commonly reported — MMLU, HellaSwag, GSM8K — are all *short-context* benchmarks. Their average input length is 2,000 tokens, often well below 1,000. At that context length, FP8 KV-cache really is fine. The regression doesn't kick in until the input length pushes the **contraction dimension** of the attention matmul past about 30,000 tokens.

Long-context evaluation — `openai/mrcr`, `RULER`, `LongBench` — is a relatively new discipline. The benchmarks that catch the bug only exist because of the long-context arms race of 2024-2025. Before those benchmarks existed, the bug existed but was invisible.

This is a recurring pattern. **You measure where you look.** Tooling that doesn't exist yet hides bugs that nobody knows to look for. The FP8 KV regression is in some sense a sibling of the [PagedAttention KV-fragmentation crisis](/issues/08-anatomy-of-a-token/09-kv-fragmentation/) from Issue 8: a problem that was always there, that nobody noticed until somebody built a tool that surfaced it. In both cases, the *measurement* was the discovery.

## What The Team Did Next

The first two weeks of the investigation are mostly debugging dead ends. The team suspects:

- **Numerical noise in the softmax.** Reasonable — softmax with FP8 inputs has been a known wrinkle since FA3 shipped. They isolate softmax, swap it back to FP32, no change.
- **Quantization scale mismatch.** Reasonable — per-tensor uncalibrated FP8 (`scale = 1.0`) is a strong default. They sweep through calibrated scales using `LLM-Compressor`, get marginal improvements, nothing that closes a 78-point gap. (We will return to calibration in [Scale Equals One](../10-calibrate/).)
- **A bug in the vLLM scheduler under long-context.** Reasonable — long contexts hit edge cases in continuous batching. They run the same prompt outside vLLM, directly against the FA3 kernel. The kernel reproduces the bug.
- **A miscompiled FA3 kernel for the specific Hopper SKU.** Reasonable — these kernels are tuned per-architecture. They rebuild from clean source. The bug survives.

What finally cracks it is a different question. Instead of "what is wrong with the FP8 path?", they ask "what is the BF16 path doing that the FP8 path isn't?" — and trace the difference all the way down to a single line in the FA3 source. The FP32 accumulator that the documentation promises is supposed to be a 32-bit floating-point register sitting inside the tensor core, summing dot-product partial results. They probe it. It is not 32 bits in any meaningful sense. It loses precision once the partial sum has been added into more than about 100,000 times.

That is the moment the team realizes they have a **hardware bug**, not a software bug. Or rather: a *hardware behavior*, not documented as a bug, that produces software-visible numerical errors at the context lengths nobody had stress-tested.

And then they go looking for the same fingerprint elsewhere — and they find it, almost immediately, in **Figure 7(b)** of the DeepSeek-V3 technical report, published in **December 2024**, fifteen months earlier, in a footnote section about training precision.

## What This Issue Will Do

This is a detective story with the murderer revealed in the cold open: the FP32 accumulator in the FA3 FP8 path was lying about its precision. What's left to explain is the *body* of the story — every step of how the team got there, and every step of what they did after.

In rough order:

- **[Chapter 2 — The Halving](../02-bandwidth-bargain/)** is the bargain that made FP8 KV-cache worth doing in the first place. Linear ITL model, bandwidth-bound decode, why the 2× memory cut translates into a real latency win.
- **[Chapter 3 — E4M3 In Three Steps](../03-what-is-fp8-here/)** is the specific FP8 format and what "per-tensor scale = 1.0" means.
- **[Chapter 4 — The Sum Is Not What You Think](../04-summation-history/)** is the numerical-analysis backbone: William Kahan at Berkeley in 1965, why floating-point addition is non-associative, what compensated summation looks like.
- **[Chapter 5 — The Accumulator Lie](../05-accumulator-lie/)** is the autopsy of the bug. What the FA3 FP8 inner loop actually does, why the FP32 register isn't doing what its name says, why the DeepSeek-V3 team saw the same thing during training, and what the precision loss looks like at the values that matter.
- **[Chapter 6 — Two Levels of Honesty](../06-two-level-fix/)** is the fix: SageAttention2's two-level accumulation, adapted to the FA3 FP8 path. 91% accuracy recovered to 89%. Costs register pressure.
- **[Chapter 7 — Register Wars](../07-tile-sizes/)** is the cost of the fix: how the extra register pressure spilled to local memory, why this hurt prefill especially at `head_dim = 256`, and the tile-size optimization PRs that partially recovered the loss.
- **[Chapter 8 — Slope vs Intercept](../08-itl-slope-model/)** is the analytical model used to characterize every benchmark in this issue: `ITL = slope × T + intercept`, what "break-even" means, what the numbers actually look like.
- **[Chapter 9 — The Window That Wouldn't Save](../09-sliding-window-puzzle/)** is the second-order problem: gpt-oss-20b and other hybrid-attention models whose sliding-window layers refuse to amortize FP8's fixed overhead, and the new `--kv-cache-dtype-skip-layers sliding_window` flag that fixed it.
- **[Chapter 10 — Scale Equals One](../10-calibrate/)** is the calibration question: when uncalibrated FP8 is enough and when you need `LLM-Compressor`.
- **[Chapter 11 — The State of FP8 KV](../11-state-of-fp8/)** is the boss capstone: the report card across every benchmark and every model, the three remaining places FP8 KV doesn't fit, and the connection back to [Issue 3's KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/) — where the simplest entry on the shelf turned out to take the longest to finish properly.

## What To Carry Forward

One sentence, before the chapters start: **the 78-point regression was not in the model**. It was not in the quantization scheme. It was not in vLLM. It was in a single arithmetic register, inside a tensor core, on a piece of silicon that has been shipping since March 2023. That register was doing exactly what its design intended. The design intent did not match the documentation, and the documentation did not match the production workloads. Three layers of mismatch, two and a half years of users not noticing, and one weekend of patient debugging to spot it.

This is, if you squint at it from far enough away, a story about *what it costs to actually validate a feature*. The flag had been "shipping" for three years. The flag had not been *finished* for three years. The difference is the difference between *available* and *correct*. It is also, less politely, the difference between a research idea and a production system.

We will spend the next ten chapters carefully not crossing that line again.

**Continue to** → [The Halving — The Bandwidth Bargain](../02-bandwidth-bargain/) — why FP8 KV-cache is irresistible in the first place, and what "irresistible" actually translates to in milliseconds of latency.
