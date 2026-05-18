---
title: "Sliding Window Attention: why FP8 broke even at 741K tokens"
description: "gpt-oss-20b's FP8 break-even was 741,565 tokens — the flag was a no-op. Sliding-window layers have bounded caches; the fix was a per-layer skip flag."
blurb:
  - "gpt-oss-20b: FP8 slope was 96% of BF16. The bandwidth saving was essentially zero."
  - "Sliding-window layers have tiny, bounded caches — FP8 saves almost nothing on them."
  - "In hybrid attention models, these layers dominate the count."
  - "The fix: a per-layer skip flag drops the break-even from 741,565 tokens to 7,659."
topics: [quantization, kv-cache, attention, hybrid-attention]
tags: [fp8, sliding-window, gpt-oss-20b, gemma, layer-skip, hybrid-attention]
theme: cream
math: true
draft: false
date: 2026-05-18T09:00:00-04:00
issue: 9
weight: 90
techKind: mainline
techNode: sliding-puzzle
header: default.webp
---

## The Second Cliffhanger

By April 2026, the two-level accumulation fix from [Chapter 6](../06-two-level-fix/) has shipped. Long-context accuracy is restored. The team turns to characterizing the speedup across the model zoo, expecting clean wins everywhere.

The first model that doesn't cooperate is **gpt-oss-20b**.

This is a model OpenAI released as an open-weight variant of their production stack — 20 billion parameters, hybrid attention architecture, native support for long context. It is exactly the kind of model FP8 KV-cache should be a slam dunk on. The team enables `--kv-cache-dtype fp8`, runs the same sweep they ran on Llama-3.1-8B, and gets numbers that look like this:

- BF16 slope: $8.94 \times 10^{-6}$ ms/token
- FP8 slope:  $8.60 \times 10^{-6}$ ms/token *(pre-fix)*

The FP8 slope is **96% of BF16 slope**. The improvement is essentially nothing. Plug this into the [break-even formula from Chapter 8](../08-itl-slope-model/):

$$
T^* \;=\; \frac{\text{intercept}_\text{FP8} - \text{intercept}_\text{BF16}}{\text{slope}_\text{BF16} - \text{slope}_\text{FP8}} \;=\; \frac{0.04}{8.94 \times 10^{-6} - 8.60 \times 10^{-6}} \;=\; \frac{0.04}{3.4 \times 10^{-7}} \;\approx\; 117{,}000 \text{ tokens?}
$$

The team's actual fitted break-even number, before the eventual fix, was **741,565 tokens**. The exact value depends on which fit is run and what input lengths the sweep includes. Either way, the answer is *past the maximum context length the model supports*. The FP8 flag, on this model, was never actually faster than BF16 in any realistic deployment.

This was not a software bug. The kernel was working as intended. The two-level accumulation fix did its job. The accuracy was fine — the model scored well on every benchmark in both BF16 and FP8 modes. The flag was simply *not buying any latency*.

The team's job is to figure out why, and then to fix it. The answer is a property of the model architecture, not the kernel.

## Hybrid Attention: A Different Cache Shape

gpt-oss-20b has a hybrid attention architecture: some layers use **global attention** (every token attends to every preceding token), others use **sliding-window attention** (every token attends only to the most recent $W$ tokens). The sliding-window variant in gpt-oss-20b has $W = 128$ — a relatively small window. Gemma's variant uses $W = 512$, and other models use everything from 256 to 4096.

For sliding-window layers, the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} size is bounded:

$$
\text{KV cache per sliding layer} \;=\; H \times W \times D \times 2 \text{ (K + V)} \times \text{bytes per element}
$$

For gpt-oss-20b's sliding layers at $H = 8$ heads, $W = 128$, $D = 128$ head dim, BF16: **524 KB per layer**. Constant in $T$. Bound the cache, bound the read traffic, bound the per-step attention cost. The global layers' cache, by contrast, scales linearly with $T$.

The team's mistake was assuming that halving the KV-cache memory per layer (FP8 vs BF16) would translate uniformly to a halving of the per-step attention cost. For the global layers, it does. For the sliding-window layers, it doesn't — and gpt-oss-20b has *more* sliding-window layers than global layers, so the total attention cost is dominated by the part that doesn't scale.

```pyplot {id="sliding-vs-global-scaling" caption="Per-step attention cost vs context length. Global attention scales with T. Sliding-window with W=128 stays bounded — quantizing it to FP8 doesn't grow with T, so the fixed FP8 conversion overhead never gets amortized past a tiny constant savings."}
import numpy as np
import matplotlib.pyplot as plt

T = np.linspace(1000, 200_000, 400)
W = 128

# Per-step attention cost, normalized to (cost-per-token-of-cache × tokens read)
# Global: linear in T (full attention)
global_bf16 = 1.0 * T
global_fp8  = 0.5 * T  # FP8 halves the cost

# Sliding-window: flat past W
sw_bf16 = np.minimum(T, W) * 1.0
sw_fp8  = np.minimum(T, W) * 0.5 + 0.5  # tiny constant FP8 overhead added

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
ax.plot(T / 1000, global_bf16, color='#FF007F', linewidth=2.6, label='BF16 global')
ax.plot(T / 1000, global_fp8,  color='#00A8A8', linewidth=2.6, label='FP8 global (-50%)')
ax.set_title('Global attention layer:\nFP8 halves the per-step cost', fontweight='bold')
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('relative attention cost')
ax.legend(framealpha=1, edgecolor='#1A1A1A')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.15)

ax = axes[1]
ax.plot(T / 1000, sw_bf16, color='#FF007F', linewidth=2.6, label='BF16 SW (W=128)')
ax.plot(T / 1000, sw_fp8,  color='#00A8A8', linewidth=2.6, label='FP8 SW (no benefit)')
ax.set_title('Sliding-window attention layer (W=128):\nbounded → FP8 doesn’t amortize', fontweight='bold')
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('relative attention cost')
ax.legend(framealpha=1, edgecolor='#1A1A1A')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.15)

plt.tight_layout()
```

The picture tells the whole story. On global-attention layers, the BF16-vs-FP8 gap grows linearly with context. On sliding-window layers, the gap is a constant — the FP8 conversion overhead per token is paid in full, and the memory benefit is bounded by $W$, which is tiny. The bargain from [Chapter 2](../02-bandwidth-bargain/) **does not apply to sliding-window layers**.

This is structural. Even with perfect kernel engineering, you cannot make FP8 win on a bounded-memory operation. The whole point of FP8 KV is that it halves bandwidth-bound work, and a sliding-window layer at $W = 128$ is not bandwidth-bound for the same reason a small MLP isn't bandwidth-bound: there isn't enough memory traffic to matter.

## Why "Why Not Both" Was The Right Answer

The team's fix, in [`vllm#33695`](https://github.com/vllm-project/vllm/pull/33695), is a new flag: **`--kv-cache-dtype-skip-layers sliding_window`**. The flag does what it says: for each attention layer in the model, if the layer is a sliding-window layer, keep its KV cache in BF16; otherwise, quantize to FP8.

The mechanism requires some plumbing. vLLM's previous design allowed a single KV-cache dtype for the entire model. To support per-layer dtypes, the team had to generalize:

- The block manager (responsible for paged KV-cache allocation, see [Issue 8 ch.11](/issues/08-anatomy-of-a-token/11-block-manager/)) had to track dtype per layer, allocating different-sized blocks for BF16 layers and FP8 layers.
- The attention backend dispatch had to choose the right kernel per layer (the FP8 path for FP8 layers, the BF16 path for BF16 layers).
- The `reshape_and_cache_flash` operation, which writes new K and V vectors into the cache, had to apply different quantization (or no quantization) depending on the destination layer.

None of these changes are deep, but together they unmake the assumption baked into earlier vLLM versions that the cache is uniform in precision. Once that assumption is broken, more general configurations become possible — per-layer FP4 (for super-aggressive compression of insensitive layers), mixed-precision per-head, etc. The sliding-window flag is the first instance of the broader pattern.

The expression "hybrid precision KV cache" is, in fact, the recurring theme of modern long-context inference. [DeepSeek-V4's mixed-precision policy from Issue 7](/issues/07-sparse-lab/15-mixed-precision-kv/) — BF16 RoPE channels, FP8 KV body, FP4 indexer — is the same idea, more extreme. The principle: different sub-regions of the cache have different accuracy/bandwidth trade-offs, and a fixed cache dtype is leaving performance on the table.

## The Numbers After The Flag

With `--kv-cache-dtype-skip-layers sliding_window` enabled on gpt-oss-20b:

- BF16 slope: $8.94 \times 10^{-6}$ ms/token
- FP8 slope (full FP8): $7.14 \times 10^{-6}$ — break-even at ~22,000 tokens
- FP8 slope (skip-SW):  $6.34 \times 10^{-6}$ — break-even at **~7,659 tokens**

The skip-SW variant gets to **71% of BF16 slope** — close to the Llama-3.1-8B 54%, far better than the gpt-oss-20b 96% without the flag. The break-even drops from 740k tokens to 7.7k. The flag *does* what the original flag was supposed to do — provide a meaningful latency win at production context lengths.

```pyplot {id="gpt-oss-improvement" caption="The skip-SW flag's effect on gpt-oss-20b. Three break-even points: 740k tokens (pre-fix), 22k tokens (with two-level fix, full FP8), 7.7k tokens (with skip-SW). Each is roughly one order of magnitude improvement over the previous."}
import numpy as np
import matplotlib.pyplot as plt

configs = ['Pre-fix\nfull FP8', 'Post-fix\nfull FP8', 'Post-fix\nskip-SW']
break_evens = [741565, 22109, 7659]
colors = ['#FF007F', '#FFD700', '#00A8A8']

fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(range(3), break_evens, color=colors, edgecolor='#1A1A1A', linewidth=1.5)

for bar, v in zip(bars, break_evens):
    ax.text(bar.get_x() + bar.get_width() / 2, v * 1.4, f'{v:,}\ntokens',
            ha='center', fontsize=10, fontweight='bold')

ax.set_yscale('log')
ax.set_ylim(1000, 2_000_000)
ax.set_xticks(range(3))
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel('break-even point (tokens, log scale)')
ax.set_title('gpt-oss-20b FP8 break-even — three orders of magnitude improvement',
             fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# Reference lines
ax.axhline(128_000, color='#1A1A1A', linewidth=1.0, linestyle=':', alpha=0.6)
ax.text(0.15, 130_000, '128k (typical long-context limit)', fontsize=8, style='italic')
ax.axhline(10_000, color='#1A1A1A', linewidth=1.0, linestyle=':', alpha=0.6)
ax.text(0.15, 10_300, '10k (typical production avg)', fontsize=8, style='italic')

plt.tight_layout()
```

The intermediate number — break-even at 22k tokens with full FP8 post-fix but without skip-SW — is interesting. Even with the kernel improvements, leaving the sliding-window layers in FP8 keeps the break-even at the high end of what production workloads typically run. The skip-SW flag, adding one more layer of policy, is what gets it back to the comfortable 7k zone.

This is also the configuration the team recommends as the **default** for gpt-oss-20b deployments. The base FP8 flag is now correct on the global layers; the skip-SW flag adds the right policy for the sliding-window layers; together they recover the bargain.

## Throughput Under Load, With Skip-SW

The single-request fit-the-line numbers above tell you about ITL behavior. The serving question is: how does this translate to throughput when the system is loaded?

The team's serving benchmark on gpt-oss-20b: 150 requests, concurrency 8, ~20k input tokens, ~2k output tokens each.

| Config | Median TTFT | Median ITL | Total duration | Output tok/s |
|---|---:|---:|---:|---:|
| BF16          | 468.9 ms | 8.09 ms | 364.2 s | 831.6 |
| FP8 (full)    | 451.7 ms | 7.90 ms | 355.1 s | 853.0 |
| **FP8 skip-SW** | **456.4 ms** | **7.70 ms** | **347.4 s** | **871.8** |

Full FP8 buys ~2.6% throughput; skip-SW buys ~4.8%. Both are small compared to Llama-3.1-8B's 14.9% throughput win, and that's the point: hybrid-attention models with small sliding windows have *less to gain* from FP8 KV in the first place, because the sliding-window layers' KV cache isn't a big share of the bandwidth bill. The flag still helps, just less.

This is the second-order lesson of the issue. **FP8 KV's benefit is proportional to how much of your KV cache is unbounded.** A pure-global-attention model (Llama) sees the full benefit. A hybrid model with mostly bounded layers (gpt-oss-20b with $W = 128$) sees a fraction of the benefit. A model that is pure sliding-window everywhere ($W = $ everything) would see essentially no benefit. The arithmetic is one chart away:

```pyplot {id="benefit-vs-bounded-fraction" caption="The benefit of FP8 KV scales with the fraction of the KV cache that grows unboundedly with context. Pure global → full benefit. Pure sliding-window → no benefit. Hybrid models sit in between."}
import numpy as np
import matplotlib.pyplot as plt

bounded_fraction = np.linspace(0, 1, 200)
# Throughput benefit scales (1 - bounded_fraction) roughly
benefit = (1 - bounded_fraction) * 15  # peak benefit ~15%

fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(bounded_fraction * 100, benefit, color='#00A8A8', linewidth=2.6)
ax.fill_between(bounded_fraction * 100, benefit, 0, color='#00A8A8', alpha=0.15)

# Annotate models
models = [
    ('Llama-3.1-8B', 0, 15),       # pure global, 0% sliding-window
    ('gpt-oss-20b', 65, 5),        # ~65% sliding-window layers
    ('Hypothetical\npure-SW model', 100, 0),
]
for label, x, y in models:
    ax.scatter([x], [y], s=100, color='#FFD700',
               edgecolor='#1A1A1A', linewidth=1.5, zorder=5)
    ax.annotate(label, xy=(x, y), xytext=(x + 3, y + 0.8),
                fontsize=9.5)

ax.set_xlabel('% of KV cache that is bounded (sliding-window or other)')
ax.set_ylabel('FP8 KV throughput benefit (%)')
ax.set_title('FP8 KV benefit scales with unbounded cache fraction',
             fontsize=11, fontweight='bold')
ax.set_xlim(-5, 105)
ax.set_ylim(0, 17)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.15)
plt.tight_layout()
```

This is, in some sense, a reason for the sliding-window architectural trend in modern models. Bounded-memory layers are *intrinsically more efficient* at long context, because they don't have to read a growing cache on every step. The cost is that they can't attend to arbitrarily-far-back tokens. Hybrid architectures are a compromise: enough sliding-window layers to bound the memory, enough global layers to preserve long-range attention.

The FP8 KV story on these models inherits the compromise. The sliding-window layers don't benefit from FP8 because they're already efficient; the global layers benefit fully. Skip-SW recognizes this asymmetry and quantizes only where the benefit exists.

## The Gemma Counter-Example

To complicate the story slightly: **Gemma-4-E2B** is also a hybrid attention model — three out of four layers are sliding-window, similar fraction to gpt-oss-20b — but Gemma's window size is **512**, four times larger than gpt-oss-20b's 128.

A bigger window means the bounded-memory layer's cache is bigger, which means quantizing it to FP8 has more to give. The Gemma team's measurements:

- BF16 slope: $5.30 \times 10^{-5}$ ms/token
- FP8 slope: $3.60 \times 10^{-5}$ ms/token — break-even comparable to Llama's

For Gemma, **full FP8 is the right choice** — even on the sliding-window layers. The skip-SW flag is *not* needed because there's enough data within each window to amortize the FP8 overhead. The window size matters, and the threshold of "is the window big enough to make FP8 worth it?" lives somewhere between 128 and 512.

This gives the team a rule of thumb: **enable skip-SW if your sliding-window size is ≤ ~256**, and let full FP8 ride otherwise. The exact threshold depends on the model's other parameters (head dim, number of heads, etc.), so the recommendation is more nuanced than a single number, but the qualitative pattern is correct.

For models with very large windows (Mistral-style $W = 4096$ and up), the sliding-window layers behave almost like global layers from the FP8 perspective: there's plenty of cache to amortize the conversion, and skip-SW provides no benefit. The flag is helpful in the *small-window* regime; for everyone else, the original `--kv-cache-dtype fp8` is fine.

{{% callout type="info" title="The Rule Of Thumb" %}}
For hybrid-attention models, enable `--kv-cache-dtype-skip-layers sliding_window` when:
- Sliding-window size W ≤ ~256 (gpt-oss-20b, similar models)

For W ≥ ~512 (Gemma, Mistral, etc.), full FP8 is fine — there's enough data per window to amortize.

The flag is opt-in and harmless; if in doubt, enable it and measure.
{{% /callout %}}

## The Broader Pattern: Per-Layer Precision

The skip-SW flag is the simplest example of a broader pattern that this issue has been circling: **different layers in the same model can profitably use different KV-cache precisions**. The full version of this pattern shows up across the recent literature:

- DeepSeek-V4's [mixed-precision KV cache from Issue 7](/issues/07-sparse-lab/15-mixed-precision-kv/) (BF16 RoPE / FP8 KV body / FP4 indexer).
- [KVTuner from Issue 3's KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/) (per-layer adaptive bit allocation: 6 bits in critical layers, 3 bits in less-critical layers).
- Per-head scales from [the previous chapter](../07-tile-sizes/) (one scale per attention head instead of a single scale per layer).

Each is a different granularity at which precision can be allocated. The skip-SW flag is *layer-grain by attention type* — coarse but easy to opt into. KVTuner is *layer-grain by empirical sensitivity* — finer-grained but requires a calibration sweep. DeepSeek-V4's policy is *region-grain by mathematical role* — finest, requires model-specific tuning.

The arc, across the whole literature, is from "one precision for the whole cache" toward "many precisions, allocated by who needs what". FP8 KV-cache in vLLM is on that arc; skip-SW is the first move, and more granular policies are likely to follow as inference engines mature.

## What Did Not Work: Other Possible Fixes

Three alternatives the team considered and rejected.

**Quantize the sliding-window layers more aggressively (e.g., INT4).** This was the natural counter-suggestion: if FP8 doesn't help on sliding-window layers because there isn't enough data to amortize, maybe INT4 would help more by compressing harder. The team tried it. The result was that the accuracy started to suffer (per-tensor INT4 on small per-layer caches is a stretch) without much latency benefit (the bounded cache still doesn't dominate bandwidth). The right answer was the simpler one: just leave those layers in BF16.

**Special-case the sliding-window layers to a different attention kernel.** Possible in principle, but the engineering cost is high — you'd need a separate optimized kernel path for bounded-attention layers, and the benefit would be a few extra percent of latency. Not worth it.

**Re-architect the model to use fewer sliding-window layers.** Out of scope. The vLLM team doesn't get to change the model; their job is to serve what people give them.

The eventual fix — `--kv-cache-dtype-skip-layers sliding_window` — is the right level of intervention: a policy flag that recognizes the architectural asymmetry without trying to paper over it.

## What To Remember

1. **Sliding-window attention layers have bounded KV cache** (size $\propto W$, not $T$). FP8 doesn't amortize because there's no growing memory to compress.
2. **Hybrid-attention models** (gpt-oss-20b, Gemma) mix global and sliding-window layers. FP8 helps fully on the global layers, marginally or not at all on the sliding-window layers.
3. **`--kv-cache-dtype-skip-layers sliding_window`** keeps sliding-window layers in BF16, quantizes global layers to FP8. Brings gpt-oss-20b's break-even from 22k tokens down to 7.7k.
4. **Window size matters.** $W \le 256$: enable skip-SW. $W \ge 512$: full FP8 is fine. The threshold is empirical.
5. **The broader pattern is per-layer precision allocation.** Skip-SW is the simplest instance; KVTuner and DeepSeek-V4 are finer-grained examples. The trend is toward heterogeneous-precision caches.

**Continue to** → [Scale Equals One](../10-calibrate/) — when uncalibrated per-tensor FP8 (`scale = 1.0`) is enough, when it isn't, and how the Kimi-K2.5 / FlashMLA case study made calibration suddenly relevant again.
