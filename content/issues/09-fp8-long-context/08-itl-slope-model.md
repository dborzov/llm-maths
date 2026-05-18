---
title: "Decode Latency: the linear model behind FP8's break-even"
description: "The two-parameter ITL model: slope measures bandwidth pressure, intercept measures overhead, and their ratio defines the context length where FP8 starts winning."
blurb:
  - "ITL(T) = slope × context_length + intercept. Two numbers predict decode latency."
  - "Slope is the bandwidth tax per token — FP8 cuts it. Intercept is conversion overhead — FP8 raises it."
  - "Below the break-even: FP8 is slower. Above it: FP8 wins."
  - "One model's break-even was 741,565 tokens — meaning the flag was, in practice, a no-op."
topics: [primer, performance, inference]
tags: [itl, ttft, benchmarking, linear-regression, decode, vllm]
theme: teal
math: true
draft: false
date: 2026-05-18T09:00:00-04:00
issue: 9
weight: 80
techKind: primer
techNode: itl-model
header: default.webp
---

## The Two-Parameter Model In One Picture

Every benchmark in this issue is described by a two-parameter linear model:

$$
\text{ITL}(T) \;=\; \text{slope} \times T \;+\; \text{intercept}
$$

where `ITL` is **inter-token latency** (the time from one generated token to the next), `T` is **context length** (how many tokens are already in the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} when this token is generated), and the two parameters are *fit from a sweep* across input lengths.

This primer is short because the model is simple. What it does for you: it gives you a clean language for reasoning about FP8 KV-cache's benefit, identifies the **break-even point** where FP8 starts winning over BF16, and exposes the two failure modes you have to watch out for (intercept growth from FP8 conversion overhead, slope failures from non-bandwidth-bound regimes).

The companion model for prefill is **quadratic** in context length:

$$
\text{TTFT}(T) \;=\; a \cdot T^2 \;+\; b \cdot T \;+\; c
$$

`TTFT` is **time-to-first-token** (how long until the model emits its first output after receiving the prompt). The quadratic comes from the prefill phase, where every input token attends to every preceding input token — $O(T^2)$ work per layer.

For most of this issue we will be talking about the linear ITL model. The quadratic TTFT model shows up in [Chapter 7's tile-size discussion](../07-tile-sizes/) and in the `head_dim = 256` regression. This primer focuses on the linear model.

## What Each Parameter Actually Measures

**Slope** (ms / token of context). Every additional token in the cache adds `slope` milliseconds to *every subsequent decode step*. This is **per-cached-token cost** and it is dominated by memory bandwidth. The math, for a single attention head:

- Each decode step reads `T` cached K vectors and `T` cached V vectors from HBM.
- At BF16, each vector is `head_dim × 2 = 256` bytes (for `head_dim = 128`). At FP8, half that.
- Per layer, per head, the read traffic is roughly `T × 4 × head_dim` bytes (K + V, BF16) or half that for FP8.
- Multiply by layers and heads, divide by HBM bandwidth, and you get the slope contribution from attention reads.

For Llama-3.1-8B at BF16: 32 layers × 8 KV heads × 256 bytes per K + 256 bytes per V × T tokens / 3.35 TB/s ≈ $4 \times 10^{-5}$ ms/token. This is exactly the measured BF16 slope from [Chapter 2](../02-bandwidth-bargain/). The napkin math is the model: the slope is bandwidth divided by per-token byte cost.

**Intercept** (ms). The fixed cost per decode step that doesn't depend on context length. This includes:

- Kernel launch overhead for each operation in the forward pass.
- The non-attention work — MLP, LayerNorm, the linear projections from residual stream to Q/K/V/O, the unembedding.
- Any per-token preamble in the attention path itself (Q quantization, position embedding, etc.).
- The constant part of the attention computation (one new Q vector, one new K and V vector to write to cache).

For Llama-3.1-8B, the intercept is roughly $6.5$ ms. The non-attention work dominates — the MLP is doing about $32 \times 2 \times d^2$ FLOPs per step, which at $d = 4096$ is $\sim 10^9$ FLOPs, comparable to the attention work at moderate context lengths.

```pyplot {id="itl-decomposition" caption="Decomposing total ITL into the slope-contribution from attention vs the intercept-contribution from everything else. At long context, the slope dominates; at short context, the intercept dominates. The break-even point is where this picture flips."}
import numpy as np
import matplotlib.pyplot as plt

T = np.linspace(100, 128_000, 400)

bf16_slope = 4.37e-5
bf16_intercept = 6.44

fp8_slope = 2.37e-5
fp8_intercept = 6.58

# Decompose the BF16 line into slope-part and intercept-part
bf16_slope_part = bf16_slope * T
bf16_intercept_part = np.ones_like(T) * bf16_intercept

# Same for FP8
fp8_slope_part = fp8_slope * T
fp8_intercept_part = np.ones_like(T) * fp8_intercept

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

# Left: BF16 stack
axes[0].fill_between(T / 1000, 0, bf16_intercept_part, color='#FFD700', alpha=0.6, label='intercept (~6.4 ms)')
axes[0].fill_between(T / 1000, bf16_intercept_part, bf16_intercept_part + bf16_slope_part,
                     color='#FF007F', alpha=0.6, label='attention slope × T')
axes[0].plot(T / 1000, bf16_intercept_part + bf16_slope_part, color='#1A1A1A', linewidth=2)
axes[0].set_title('BF16 KV cache', fontweight='bold')
axes[0].set_xlabel('context length T (k tokens)')
axes[0].set_ylabel('ITL (ms)')
axes[0].legend(loc='upper left', framealpha=1, edgecolor='#1A1A1A')
axes[0].spines[['top', 'right']].set_visible(False)
axes[0].grid(True, alpha=0.15)

# Right: FP8 stack
axes[1].fill_between(T / 1000, 0, fp8_intercept_part, color='#FFD700', alpha=0.6, label='intercept (~6.6 ms)')
axes[1].fill_between(T / 1000, fp8_intercept_part, fp8_intercept_part + fp8_slope_part,
                     color='#00A8A8', alpha=0.6, label='attention slope × T (halved)')
axes[1].plot(T / 1000, fp8_intercept_part + fp8_slope_part, color='#1A1A1A', linewidth=2)
axes[1].set_title('FP8 KV cache', fontweight='bold')
axes[1].set_xlabel('context length T (k tokens)')
axes[1].legend(loc='upper left', framealpha=1, edgecolor='#1A1A1A')
axes[1].spines[['top', 'right']].set_visible(False)
axes[1].grid(True, alpha=0.15)

plt.tight_layout()
```

The picture makes the trade-off visible. Yellow is the same in both — the intercept is essentially identical for BF16 and FP8, because FP8 KV doesn't change the non-attention work. Pink (BF16 attention) and teal (FP8 attention) are the part that scales with context, and FP8 cuts that contribution roughly in half. At small `T`, the yellow dominates and the two columns are essentially the same height. At large `T`, the pink/teal difference makes the FP8 column substantially shorter.

## The Break-Even Arithmetic

The break-even point is the context length at which the FP8 line crosses below the BF16 line:

$$
T^* \;=\; \frac{\text{intercept}_\text{FP8} - \text{intercept}_\text{BF16}}{\text{slope}_\text{BF16} - \text{slope}_\text{FP8}}
$$

A small algebraic exercise. For Llama-3.1-8B after the fix:

$$
T^* \;=\; \frac{6.58 - 6.44}{4.37 \times 10^{-5} - 2.37 \times 10^{-5}} \;=\; \frac{0.14}{2.00 \times 10^{-5}} \;=\; 7{,}000 \text{ tokens}
$$

The team's reported break-even of "~7k tokens" is exactly this fraction. Past 7,000 tokens of context, FP8 is strictly faster per generated token. Below 7,000, BF16 is marginally faster because FP8's slightly higher intercept (the cost of the quantization preamble that wasn't fully fused at the time) outweighs the slope reduction.

The break-even point is the most useful summary statistic in the whole issue. It is a single number that tells you "at what context length does this optimization become worth it?". A small break-even (like 7k) means the optimization is broadly applicable. A huge break-even (like 740k, which is what gpt-oss-20b had before the layer-skip fix) means the optimization is theoretically real but practically never realized.

The team's headline progress numbers, from [Chapter 2](../02-bandwidth-bargain/), can be rewritten as break-even movements:

| Config | Before | After | Why |
|---|---:|---:|---|
| Llama-3.1-8B FP8 | 24,889 | **7,010** | Two-level fix + tile-size opt + Q-fusion |
| gpt-oss-20b FP8 (full) | 741,565 | **22,109** | Tile-size opt + Q-fusion |
| gpt-oss-20b FP8 skip-SW | n/a | **7,659** | + layer-skip flag for sliding-window |

Three orders of magnitude improvement on gpt-oss-20b. From "FP8 never wins" to "FP8 wins past 7.6k tokens".

## How To Fit The Line From A Benchmark Sweep

The benchmark is conceptually simple. Run `vllm bench serve` with concurrency 1, fixed output length (say 128 tokens), and sweep input length from 256 to 125,000 tokens in maybe 20 steps. For each input length, record the median ITL across the 128 output tokens. Fit a line.

```python
import numpy as np

def fit_itl_line(input_lengths, observed_itls):
    """
    Fit ITL = slope * T + intercept by least squares.
    Returns (slope, intercept) tuple in same units as observed_itls.
    """
    T = np.asarray(input_lengths)
    y = np.asarray(observed_itls)
    A = np.vstack([T, np.ones_like(T)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept

# Worked example
input_lengths = [256, 1000, 4000, 16000, 64000, 125000]
itl_bf16      = [6.45, 6.49, 6.61, 7.14, 9.24, 11.90]
itl_fp8       = [6.59, 6.61, 6.68, 6.96, 8.10, 9.54]

slope_bf16, int_bf16 = fit_itl_line(input_lengths, itl_bf16)
slope_fp8,  int_fp8  = fit_itl_line(input_lengths, itl_fp8)

print(f"BF16: slope = {slope_bf16:.2e} ms/token, intercept = {int_bf16:.2f} ms")
print(f"FP8:  slope = {slope_fp8:.2e} ms/token, intercept = {int_fp8:.2f} ms")
print(f"break-even: {(int_fp8 - int_bf16) / (slope_bf16 - slope_fp8):,.0f} tokens")
```

Two things to note about the fitting procedure.

First, the fit is **linear in T**, not in $\log T$. The model assumes that doubling the context doubles the per-step attention work, which is correct for full attention (every cached token contributes one K + V load per step). For models with sliding windows, this assumption breaks past the window size, and the linear fit will systematically under-estimate the slope at long context. See [Chapter 9](../09-sliding-window-puzzle/).

Second, the fit is **per concurrency level**. At concurrency 1, the ITL is the pure single-request latency. At higher concurrency, the scheduler interleaves requests, batches them at each step, and the per-token timing gets messier. For batch comparison, the throughput metrics (output tokens / second per GPU) are usually more meaningful than the per-token ITL. The fit-a-line procedure is for *characterizing* the FP8 vs BF16 gap on a controlled benchmark; the at-load throughput numbers are for *predicting* how that gap translates to serving performance.

## When The Linear Model Fails

Three named failure modes for the linear fit.

**1. Sliding-window attention.** When the attention mechanism only attends to the last $W$ tokens (the sliding window), the per-step attention cost is bounded by $W$, not $T$. The ITL is linear in `min(T, W)`, which means the slope-of-T flattens to zero once $T > W$. A linear fit to the full sweep will under-estimate the slope at small $T$ and over-estimate at large $T$. The right fit is a piecewise model:

$$
\text{ITL}(T) \;=\; \text{slope} \times \min(T, W) \;+\; \text{intercept}
$$

For hybrid attention models (mix of global and sliding-window layers), the relationship is somewhere in between, dominated by the global layers' slope but with the sliding-window layers contributing a smaller flat term.

**2. Compute-bound attention.** At very short contexts, attention is compute-bound rather than bandwidth-bound, because the per-step FLOPs are large enough that you don't see the bandwidth ceiling. The ITL in this regime is determined by FLOPs/s, not bytes/s, and the model is less useful. In practice this regime is below T ≈ 100 for most models, well below where anyone deploys.

**3. KV-cache spilling.** At very long contexts, the cache may exceed GPU memory and the system has to fall back to either CPU RAM, paged KV (which has its own latency profile, see [Issue 8 ch.10](/issues/08-anatomy-of-a-token/10-paged-attention/)), or eviction (see [Issue 6](/issues/06-eviction-notice/)). All three add per-step costs that aren't linear in T, so the model breaks at the spill point.

The take-away is that the linear fit is a *summary*, not a physical law. It's correct in the regime where the kernel is doing full attention on a cache that fits in HBM. Outside that regime, you have to use a different model.

## What Break-Even Means Operationally

Stating the break-even is one number. Acting on it is a different exercise.

The right way to think about break-even is: **at what context length does enabling FP8 KV become strictly better than BF16 for my workload?** For a workload where the average context is 30k tokens, a break-even of 7k means you should always enable FP8. For a workload where the average context is 2k tokens, a break-even of 7k means FP8 is slightly *worse* on average — you should leave it off.

Most production workloads are a *mixture* of context lengths. A chatbot might have a long-tail distribution: most requests are short (5-50 tokens), but the long tail can hit 100k+. In that case, the right policy is *enable FP8 if the average context is above the break-even*, weighted by the fraction of compute spent at each context length. For a typical chatbot, that policy says "enable FP8" — the long-tail requests are where most of the latency budget lives.

For deployments serving specifically *short-context* workloads (think a code-completion endpoint with 1k-token average prompts), BF16 may genuinely be better. The break-even framework gives you a principled way to decide, instead of "FP8 is always on" or "FP8 is always off".

```pyplot {id="break-even-decision-table" caption="Decision table for FP8 KV based on your workload's average context length. Below break-even, the slope advantage of FP8 doesn't compensate for the small intercept overhead. Above, FP8 is strictly better. Most production workloads sit comfortably above 7k average context."}
import numpy as np
import matplotlib.pyplot as plt

break_even = 7000

workloads = [
    ('Code completion\n(1k avg)',         1000,  'BF16 wins'),
    ('Chatbot short\n(3k avg)',           3000,  'BF16 wins'),
    ('Chatbot mixed\n(15k avg)',          15000, 'FP8 wins'),
    ('Document QA\n(40k avg)',            40000, 'FP8 wins'),
    ('Long-doc analysis\n(80k avg)',      80000, 'FP8 wins'),
    ('Million-token agent\n(500k avg)',   500000,'FP8 wins big'),
]

fig, ax = plt.subplots(figsize=(11, 4.6))

for i, (label, T, verdict) in enumerate(workloads):
    color = '#FF007F' if T < break_even else ('#00A8A8' if T < 100_000 else '#FFD700')
    ax.barh(i, T / 1000, color=color, edgecolor='#1A1A1A', linewidth=1.4)
    ax.text(T / 1000 + 10, i, f'  {verdict}', va='center', fontsize=10)

ax.axvline(break_even / 1000, color='#1A1A1A', linewidth=2, linestyle='--')
ax.text(break_even / 1000 + 2, len(workloads) - 0.3, 'break-even (~7k)',
        fontsize=9, style='italic')

ax.set_yticks(range(len(workloads)))
ax.set_yticklabels([w[0] for w in workloads])
ax.set_xscale('log')
ax.set_xlabel('average context length (k tokens, log scale)')
ax.set_xlim(0.5, 1500)
ax.set_title('When to enable FP8 KV-cache, by workload average context',
             fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.invert_yaxis()
plt.tight_layout()
```

The decision table tells you what to do with the break-even number. Above ~7k average context (which most modern long-context deployments are), enable it. Below, leave it off. The next chapter — [The Window That Wouldn't Save](../09-sliding-window-puzzle/) — is the case where the break-even moves so high (700k tokens, pre-fix) that "enable" becomes a synonym for "no benefit", and the team had to add a separate flag to recover the bargain on hybrid-attention models.

## What To Remember

1. **The linear ITL model** is `ITL = slope × T + intercept`. Slope is per-cached-token attention cost (bandwidth-bound). Intercept is per-step overhead independent of context.
2. **FP8 KV halves the slope** in the bandwidth-bound regime. Llama-3.1-8B sits at 54% (essentially the floor). Pre-fix gpt-oss-20b sat at 96% (essentially no benefit).
3. **Break-even** = `(intercept_FP8 - intercept_BF16) / (slope_BF16 - slope_FP8)`. The single most useful summary number for deciding whether FP8 helps.
4. **Linear model breaks** for sliding-window models, compute-bound short contexts, and cache-spilled long contexts. Use a piecewise or quadratic model where appropriate.
5. **Operational decision: above ~7k average context, enable FP8.** Below, leave it off. Most production workloads in 2026 sit above 7k.

**Continue to** → [The Window That Wouldn't Save](../09-sliding-window-puzzle/) — the gpt-oss-20b case study, where sliding-window layers refused to amortize the FP8 overhead and the team had to add a per-layer skip flag.
