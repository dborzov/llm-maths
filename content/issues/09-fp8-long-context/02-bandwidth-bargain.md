---
title: "HBM Bandwidth: why you'd halve the KV cache"
short_title: "HBM Bandwidth"
description: "The bandwidth wall during decode, the linear ITL model, and the napkin math that makes FP8 KV irresistible — before anyone checked whether it still worked."
blurb:
  - "H100 decode is bandwidth-bound: 3.35 TB/s is the constraint, not FLOPs."
  - "The KV cache is the largest thing crossing that bus — every token, every step."
  - "Halve the cache: halve the bandwidth pressure, halve the latency slope."
  - "The math checked out. For three years, nobody checked whether the model stayed accurate."
topics: [quantization, inference, kv-cache, performance]
tags: [fp8, vllm, decode, itl, bandwidth, hopper]
theme: teal
math: true
draft: false
date: 2026-05-17T09:00:00-04:00
issue: 9
weight: 20
techKind: mainline
techNode: bandwidth-bargain
header: default.webp
---

## Why The Flag Existed In The First Place

Before the bug, the bargain. Why did anyone write `--kv-cache-dtype fp8` into vLLM in the first place? What did the people enabling it think they were buying?

The short answer is: **decode latency**. The longer answer requires sitting with a single number on an H100 spec sheet and watching it bend the rest of the inference stack around it. The number is the HBM bandwidth: **3.35 terabytes per second**. That sounds enormous. It is, in fact, the entire reason the FP8 KV-cache flag exists.

This chapter sets up the bargain. The next chapter ([E4M3 In Three Steps](../03-what-is-fp8-here/)) explains the specific FP8 format being used, and the chapter after that ([The Sum Is Not What You Think](../04-summation-history/)) sets up the numerical analysis we will need to understand why the bargain failed at long context. But first: the math that made everyone want this flag.

## The Two Phases And Why They Behave Differently

Modern LLM inference runs in two phases: **prefill** (process the input prompt, populate the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}) and **decode** (generate tokens one at a time, extending the cache with each step). [Issue 8's primer](/issues/08-anatomy-of-a-token/07-prefill-vs-decode/) walks through both at length, so here is the one-paragraph summary needed for the bargain.

**Prefill** processes many tokens at once. Each token contributes a row to the $QK^\top$ score matrix; many tokens × many keys means a large matrix multiplication; matrix multiplications are *compute-bound* on modern GPUs. You are limited by FLOPs, not by memory bandwidth.

**Decode** is the opposite. One token is being generated. That token contributes *one* row to $QK^\top$. To compute the attention output for that one row, the GPU has to **read the entire KV cache** from HBM — every K and V vector from every prior position, for every layer, for every head. The math being done with those bytes is tiny: a single $(1 \times T) \times (T \times D)$ matmul per head. The bytes being moved are enormous: the whole cache.

This is the **arithmetic intensity** picture that drove FlashAttention in the first place. [Issue 8's roofline primer](/issues/08-anatomy-of-a-token/04-roofline/) and [Issue 6's bandwidth wall](/issues/06-eviction-notice/11-bandwidth-wall/) both unpack the math: decode attention sits at roughly $I \approx 1$ FLOP per byte, while the H100 ridge point is roughly $\sim 200$ FLOPs per byte. Decode is bandwidth-bound by a factor of **~200×**. Most of the tensor core silicon is sitting idle during decode, waiting for the cache to land in SRAM.

The implication, said plainly: **if you can move the KV cache through the HBM bus twice as fast, you decode twice as fast.** And the easiest way to do that — given that bandwidth in TB/s is fixed by the silicon — is to **make the cache half as big in bytes**.

## The Linear ITL Model

Engineers like to fit a line to a curve when the line is good enough. For decode latency, it is. The standard model used in this issue — and in essentially every modern LLM benchmark paper — is:

$$
\text{ITL}(T) \;=\; \text{slope} \times T \;+\; \text{intercept}
$$

where `ITL` is **inter-token latency** (the time from emitting one token to emitting the next), `T` is **context length** (how many tokens are already in the cache), and the two parameters are fit from a sweep across input lengths.

[Chapter 8 — Slope vs Intercept](../08-itl-slope-model/) goes deep on this model, including what "break-even" means and how the slope and intercept are actually fit from data. For this chapter you only need to know what each parameter physically represents.

- **slope** (ms / token). This is the per-cached-token cost of attention. Every additional token in the cache adds `slope` milliseconds to every subsequent decode step. The slope is dominated by **memory bandwidth**: each new cache entry adds bytes that have to be re-read from HBM on every decode step.
- **intercept** (ms). This is the fixed cost per decode step independent of context length — kernel launch overhead, the non-attention parts of the forward pass (MLP, LayerNorm, the linear projections), and any per-token overhead in the attention path itself (quantization fusion, etc.).

The slope is what gets killed by halving the cache. If FP8 KV halves the memory traffic of attention, then in a purely bandwidth-bound regime, the slope should also halve. In practice it doesn't *quite* halve — there is some fixed work per element regardless of precision, and the FP8 conversions cost a small fraction of a tensor core cycle — but the **theoretical floor** for the FP8 slope is **50% of the BF16 slope**. The team's headline number — Llama-3.1-8B FP8 slope at **54%** of BF16 — is essentially this ceiling, achieved.

```pyplot {id="itl-vs-context" caption="The bargain in a single picture. Linear ITL model fit to real numbers from the vLLM benchmark. BF16 KV's slope at 4.37e-5 ms/token vs FP8 KV at 2.37e-5 ms/token. They share an intercept; FP8 wins by halving the slope."}
import numpy as np
import matplotlib.pyplot as plt

# Real fit parameters from the vLLM blog (Llama-3.1-8B, H100)
bf16_slope, bf16_intercept = 4.37e-5, 6.44       # ms/token, ms
fp8_slope,  fp8_intercept  = 2.37e-5, 6.58       # ms/token, ms

T = np.linspace(0, 128_000, 400)
itl_bf16 = bf16_slope * T + bf16_intercept
itl_fp8  = fp8_slope  * T + fp8_intercept

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(T / 1000, itl_bf16, color='#FF007F', linewidth=2.6,
        label=f'BF16 KV   (slope {bf16_slope:.2e})')
ax.plot(T / 1000, itl_fp8,  color='#00A8A8', linewidth=2.6,
        label=f'FP8 KV    (slope {fp8_slope:.2e})')

# Break-even point: where do the two lines cross?
T_be = (fp8_intercept - bf16_intercept) / (bf16_slope - fp8_slope)
ax.axvline(T_be / 1000, color='#FFD700', linewidth=2, linestyle='--',
           label=f'break-even ~{T_be/1000:.1f}k tokens')

# Annotate
ax.scatter([T_be / 1000], [bf16_slope * T_be + bf16_intercept],
           s=100, color='#FFD700', edgecolor='#1A1A1A', zorder=5)
ax.annotate('past here, FP8 is strictly faster',
            xy=(T_be / 1000, bf16_slope * T_be + bf16_intercept),
            xytext=(40, 10),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A'),
            fontsize=9)

ax.set_xlabel('context length T (thousands of tokens)')
ax.set_ylabel('inter-token latency (ms)')
ax.set_title('Linear ITL model: halving the cache halves the slope', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', framealpha=1, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.15)
ax.spines[['top', 'right']].set_visible(False)

print(f'BF16 ITL at 128k: {bf16_slope*128_000 + bf16_intercept:.1f} ms')
print(f'FP8  ITL at 128k: {fp8_slope*128_000  + fp8_intercept:.1f} ms')
print(f'speedup at 128k:  {(bf16_slope*128_000 + bf16_intercept)/(fp8_slope*128_000 + fp8_intercept):.2f}x')
```

Two parallel lines that almost share an intercept and have wildly different slopes. The break-even point — where the FP8 line ducks under the BF16 line — sits around **7,000 tokens** for Llama-3.1-8B. Past that, FP8 is strictly faster per token. By 128k tokens, the FP8 line is at roughly **12 ms** per token versus the BF16 line at **20 ms**. A **40% reduction in inter-token latency** — meaning at the same token-per-second budget, you can serve roughly 70% more concurrent users.

That is the bargain, in one picture and three numbers. **Halve the cache, halve the slope, win at long context.** Engineers love bargains this clean.

## Napkin Math: Why The Slope Reduction Translates To Throughput

The slope reduction alone is impressive. But the reason the FP8 flag is *also* the throughput flag in practice — the reason the AWS team's headline benchmark shows **14.9% higher output throughput** on Llama-3.1-8B under load — is that the memory savings *also* let the vLLM scheduler pack more concurrent requests onto the same GPU.

The arithmetic is straightforward. An H100 80GB hosting Llama-3.1-8B-FP16 in BF16 uses roughly 16 GB for the model weights, leaving 64 GB for KV cache. At BF16, that's ~32k tokens per concurrent user across the maximum concurrency the scheduler can fit. Switching to FP8 KV cuts that per-user bytes in half, so the same 64 GB now holds either **64k tokens per user** at the same concurrency or **2× the concurrent users** at the same per-user context length.

In practice, vLLM's continuous-batching scheduler does both: it accepts more incoming requests *and* permits longer per-request contexts. The throughput gain that comes out the other end is the product of the per-token speedup (~40% at 20k context) and the concurrency gain (~70%) — minus overhead — which lands at the observed 14.9% throughput improvement on the AWS team's 150-request, concurrency-8 sweep.

| Config | Median TTFT | Median ITL | Total duration | Output tok/s |
|---|---:|---:|---:|---:|
| BF16 KV   | 763.6 ms  | 15.18 ms | 672.6 s | 450.3 |
| **FP8 KV** | **742.8 ms** | **12.93 ms** | **585.2 s** | **517.5** |

The same numbers, looked at three ways:

1. **Time-to-first-token is unchanged.** Prefill is compute-bound, not bandwidth-bound, so the FP8 weight savings don't show up here — and the team's earlier fix added some prefill overhead (we'll see why in [Two Levels Of Honesty](../06-two-level-fix/)).
2. **Inter-token latency drops 15%.** Direct consequence of the slope reduction at the test's ~20k-token average input length.
3. **Output throughput rises 15%.** Decode is where the model spends most of its serving time at long contexts, so the ITL improvement translates directly to throughput.

15% throughput, "for free", from one flag. This is why people *wanted* this flag to work. This is why three years of vLLM users enabled it and trusted the README. This is why the 78-point accuracy regression in the cold open was such a betrayal of expectations.

{{% callout type="theorem" title="The Decode-Bound Decision Rule" %}}
If your workload is dominated by **decode** at **long contexts**, the only thing that matters is reducing **bytes per cached element**. Every other inference optimization (FlashAttention, PagedAttention, speculative decoding) is downstream of this single fact. FP8 KV-cache delivers the cleanest possible 2× — every byte you don't store is a byte you don't read on every subsequent decode step.
{{% /callout %}}

## What The Bargain Looked Like Before The Fix

Before the team's intervention, the same Llama-3.1-8B benchmark told a different story. The slope ratio of FP8 to BF16 was **63%**, not the post-fix 54%. The break-even point was at **~25,000 tokens**, not the post-fix 7,000. The bargain was still real, but it was muddier — and on Llama-3.1-8B, it was the *good* case.

The bad case, on a hybrid-attention model like gpt-oss-20b, was much worse: the pre-fix FP8 slope was **96%** of BF16. Functionally identical. The break-even point was at **~741,565 tokens**. That is past the maximum context length the model supports. *FP8 KV cache on gpt-oss-20b was, before the fix, never actually faster than BF16.* The flag was advertised as a 2× memory and latency win. In practice it was a 2× memory win and a 0% latency win.

How can a 2× memory savings translate into 0% latency improvement? The reason is that **memory savings don't help if the kernel can't reach the bandwidth ceiling.** If your attention kernel has enough fixed per-token overhead that it never becomes bandwidth-bound at the context lengths people actually use, you might as well not have halved the cache. The bandwidth was never the bottleneck.

This is the subtle part of the bargain that nobody emphasized in 2023-2025: **FP8 KV is a bandwidth-bound optimization. You need to be in the bandwidth-bound regime for it to help.** On Llama-class models, you are. On hybrid-attention models with small sliding windows, you might not be. [Chapter 9 — The Window That Wouldn't Save](../09-sliding-window-puzzle/) is the long form of why that is, and how the team's `--kv-cache-dtype-skip-layers sliding_window` flag fixed it.

```pyplot {id="slope-ratios-comparison" caption="FP8 slope as a percentage of BF16 slope. Closer to 50% is better — FP8 is fully amortizing its memory savings. Closer to 100% means FP8 is paying its overhead without realizing the bandwidth win. Pre-fix gpt-oss-20b at 96% means the flag was functionally a no-op."}
import numpy as np
import matplotlib.pyplot as plt

configs = [
    ('Llama-3.1-8B\nbefore fix',      63, '#FF007F'),
    ('Llama-3.1-8B\nafter fix',       54, '#00A8A8'),
    ('gpt-oss-20b\nbefore fix',       96, '#FF007F'),
    ('gpt-oss-20b\nfull FP8',         80, '#FFD700'),
    ('gpt-oss-20b\nskip-SW',          71, '#00A8A8'),
]

labels = [c[0] for c in configs]
values = [c[1] for c in configs]
colors = [c[2] for c in configs]

fig, ax = plt.subplots(figsize=(10, 4.8))
bars = ax.bar(range(len(configs)), values, color=colors, edgecolor='#1A1A1A', linewidth=1.5)

ax.axhline(50, color='#1A1A1A', linewidth=1.5, linestyle='--', alpha=0.7)
ax.text(len(configs) - 0.4, 51, 'theoretical floor', fontsize=8, ha='right', style='italic')

ax.axhline(100, color='#FF8C00', linewidth=1.5, linestyle='--', alpha=0.7)
ax.text(len(configs) - 0.4, 101, 'BF16 (no benefit)', fontsize=8, ha='right', style='italic')

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f'{v}%',
            ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(configs)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('FP8 slope (% of BF16)')
ax.set_ylim(0, 115)
ax.set_title('How close to the 50% bandwidth floor does each FP8 config get?',
             fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

This chart is the cliffhanger for the rest of the issue. The team's job in chapters 5-10 is to turn the pink bars into the teal bars. Each chapter is one piece of the journey: the accumulator fix, the tile-size optimization, the per-head scales, the layer-skip flag for sliding-window models.

## Why The Bargain Made The Bug So Costly

Step back from the math for a moment.

The thing that makes the 91→13% accuracy collapse from [the cold open](../01-cold-open/) painful is precisely that the bargain was *real*. Users were getting a 40% latency improvement and a 70% concurrency boost. Even if the accuracy regression had been visible — even if every tutorial had warned "FP8 KV breaks long context recall, use at your own risk" — many production users would have shipped it anyway. The bargain was that good.

But the bargain was hidden behind a bug nobody had named. For three years, the people enabling the flag were getting *both* the win (lower latency, more concurrency) *and* the loss (long-context recall destroyed), and they didn't know they were paying the second half. They were running short-context benchmarks, seeing the latency win, and shipping. Some of them, presumably, were running long-context production workloads where the accuracy was silently bad.

This is the part of the story that has a moral. **A bargain that's too good to be true probably has a bug somewhere.** And until the bug is named, the bargain is dangerous in proportion to how good it looks.

The bug, when the team found it, lived inside a register the size of a postage stamp inside a tensor core inside Hopper. Hopper has been shipping since March 2023. The register had been wrong about its own precision for the entire time. We will spend the next three chapters chasing it.

## What To Remember

1. **Decode is bandwidth-bound** at roughly $I \approx 1$ FLOP/byte. The H100 ridge point is around 200 FLOPs/byte. The KV cache is the dominant memory cost during decode.
2. **The linear ITL model** is `ITL = slope × T + intercept`. The slope reflects per-cached-token attention cost; halving the cache halves the slope.
3. **FP8 KV's theoretical ceiling** is a 50% slope reduction relative to BF16. Llama-3.1-8B after the fix hits 54% — essentially optimal.
4. **The throughput gain is multiplicative**: lower ITL × more concurrent users (because the cache holds 2× more tokens) = 15% throughput at the AWS team's serving benchmark.
5. **The bargain only holds in the bandwidth-bound regime.** Hybrid-attention models with small sliding windows have bounded KV caches that don't reach the bandwidth ceiling — and for them, the FP8 flag is closer to a no-op than a 2× win. That's the [Chapter 9 cliffhanger](../09-sliding-window-puzzle/).

**Continue to** → [E4M3 In Three Steps](../03-what-is-fp8-here/) — the specific FP8 format used in `--kv-cache-dtype fp8`, what its 16 mantissa values feel like, and why per-tensor scale = 1.0 is the right starting point.
