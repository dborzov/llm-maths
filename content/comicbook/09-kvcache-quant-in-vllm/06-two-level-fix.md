---
title: "FlashAttention-3 FP8: the two-level accumulator fix"
short_title: "FlashAttention-3 FP8"
description: "SageAttention2's two-level accumulation adapted to FA3: inner fast register, outer true FP32, periodic flush. Accuracy from 13% back to 89% — and the register pressure that followed."
blurb:
  - "Accumulate into the fast tensor-core register; every N steps, flush to a real FP32 register and reset."
  - "SageAttention2 published the same idea in November 2024 for diffusion model kernels."
  - "Accuracy recovered: 91% → 13% → 89%. Close enough to ship."
  - "The cost: register pressure. At head_dim = 256, prefill regressed ~60%."
topics: [quantization, inference, kv-cache, kernel-engineering]
tags: [fp8, flash-attention-3, sageattention2, two-level-accumulation, kahan, register-pressure]
theme: teal
math: true
draft: false
date: 2026-05-18T09:00:00-04:00
issue: 9
weight: 60
techKind: mainline
techNode: two-level-fix
header: default.webp
---

## The Fix In One Sentence

> *Inside the FA3 FP8 inner loop, keep accumulating into the fast tensor-core register; every $N$ multiply-accumulates, copy the partial sum into a real FP32 register, reset the inner accumulator, and continue.*

That is, mechanically, the entire fix. The same {{< wiki "kahan" >}}Kahan{{< /wiki >}}-family algorithm from [the previous primer](../04-summation-history/), tiled along the inner contraction loop, adapted for the constraints of a Hopper tensor core. The PR is **[flash-attention#104](https://github.com/vllm-project/flash-attention/pull/104)**, merged in early April 2026, roughly two weeks after the team confirmed the [accumulator was the cause](../05-accumulator-lie/).

This chapter walks through the implementation in detail, explains why two-level accumulation is the right granularity (not single-step Kahan, not whole-matmul software FP32), reports the headline accuracy numbers, and then turns to the engineering reality: the fix isn't free. Two-level accumulation costs **register pressure**, which spills to local memory at large `head_dim`, which slows down prefill. The next chapter ([Register Wars](../07-tile-sizes/)) is the optimization story for that downstream problem. This chapter ends with the trade-off as it stood when the PR merged: accuracy fully recovered, prefill latency at `head_dim = 256` regressed by ~60% in the worst case.

## Where The Idea Came From: SageAttention2

The two-level accumulation idea is not new to vLLM. It was published in November 2024 by **Jintao Zhang et al.** in [**SageAttention2**](https://arxiv.org/abs/2411.10958), a paper specifically about accelerating attention inference at low precision. SageAttention2 was originally focused on INT8 and INT4 attention kernels for diffusion models, where the same accumulator-precision-loss problem shows up for the same reason — long contractions with low-precision inputs.

The SageAttention2 algorithm, in their notation, is essentially:

```
for inner_block in blocks_of_inner_loop:
    acc_inner = 0   # fast on-chip register
    for k in inner_block:
        acc_inner += Q[i] @ K[k].T
    acc_outer += acc_inner    # promote to real FP32 register
    # reset acc_inner implicitly at next iteration
```

The granularity (`block_size`) is a tuning parameter. SageAttention2 used $N = 32$ in most experiments. The vLLM team adopted the same general structure with `N` chosen per tile shape to balance accuracy against register count.

The intellectual lineage is worth highlighting. SageAttention2's authors cite [Kahan summation](../04-summation-history/) directly. The structural equivalence is: SageAttention2's `acc_inner` is the fast-but-lossy accumulator, the periodic `acc_outer += acc_inner` is the same idea as Kahan's `s = t` after compensation, and the implicit reset of `acc_inner` between blocks is the moral equivalent of zeroing `c` and starting over with a fresh imprecise accumulator. The Kahan-style compensation term `c` is implicit in this version — the inner accumulator's error is bounded by being small, so no explicit compensation is needed across blocks.

Compared to literal per-add Kahan compensation, two-level accumulation is the *tiled* version: it sacrifices a small amount of error guarantee for a much better register footprint and a much simpler kernel. For tensor-core workloads, this is the right trade.

## What The FA3 Inner Loop Looks Like After The Fix

Here is a stripped-down version of the FA3 FP8 inner loop with the two-level fix applied. The real kernel is written in CUTLASS C++; this is a Python-equivalent skeleton.

```python
def fa3_fp8_inner_loop_two_level(Q_tile, K_tiles, V_tiles, *, n_promote=4):
    """
    Q_tile : (Bq, D) in FP8
    K_tiles, V_tiles : list of (Bk, D) tiles in FP8
    n_promote : promote acc_O from the fast register to true FP32 every n_promote KV tiles.
    """
    # Online softmax statistics
    m = -inf * ones(Bq)
    ell = zeros(Bq)
    # Two-level output accumulator
    acc_O_fast = zeros((Bq, D), dtype='wgmma_internal_fp32')   # the lossy register
    acc_O_true = zeros((Bq, D), dtype='real_fp32')             # the slow true-FP32 register

    for tile_idx, (K_tile, V_tile) in enumerate(zip(K_tiles, V_tiles)):
        # 1. Compute partial scores. The Q@K^T contraction is per-tile (size Bk), short.
        S_tile = wgmma_fp8_to_fp32(Q_tile @ K_tile.T) / sqrt(D)

        # 2. Online softmax bookkeeping
        m_b = S_tile.max(axis=1)
        P_tile = exp(S_tile - m_b[:, None])
        ell_b = P_tile.sum(axis=1)

        # 3. Merge online softmax statistics
        m_new = maximum(m, m_b)
        alpha = exp(m - m_new)
        beta = exp(m_b - m_new)
        ell = alpha * ell + beta * ell_b

        # 4. Update output accumulator. This is the FA second matmul:
        #    O += P @ V. Contraction dim = Bk (one tile), not full T.
        update_term = beta[:, None] * wgmma_fp8_to_fp32(P_tile.to_fp8() @ V_tile)
        acc_O_fast = alpha[:, None] * acc_O_fast + update_term

        # 5. Two-level promotion. Every n_promote tiles, copy fast -> true, reset fast.
        if (tile_idx + 1) % n_promote == 0:
            acc_O_true = acc_O_true + acc_O_fast
            acc_O_fast = zeros_like(acc_O_fast)

        m = m_new

    # Final promotion
    acc_O = acc_O_true + acc_O_fast
    O_tile = acc_O / ell[:, None]
    return O_tile
```

Compare against the pre-fix version, which only had `acc_O_fast`. The change is two lines: declare a second accumulator (`acc_O_true`), and every $n$ tiles, dump the fast one into the slow one and reset.

The choice of `n_promote` matters. Too small and you're promoting on every tile, defeating the point of using the fast register. Too large and the fast accumulator drifts before promotion. The vLLM team settled on `n_promote = 4` for most tile shapes after some empirical sweeps — meaning the fast register accumulates roughly four KV tiles of contribution before being honest about its bookkeeping. For `Bk = 64` (a typical KV tile width), this corresponds to promoting every 256 summands, well below the threshold at which the alignment shift bug becomes serious. The accumulator-drift error is then bounded by the per-block accuracy of summing 256 FP8 products into an FP32-storage-but-not-quite-FP32-add register, which empirically is fine.

## The Accuracy Recovery

The headline number from `flash-attention#104`: on the 128k needle-in-a-haystack task on Llama-3.1-8B that the cold open opened with, two-level accumulation brought FP8 KV-cache accuracy from **13% back to 89%** — almost completely closing the gap to the 91% BF16 baseline.

```pyplot {id="accuracy-recovery" caption="The headline accuracy recovery. Before/after for the cold-open needle-in-a-haystack at 128k context. BF16 was 91% the whole time; FP8 was 13% before the fix, 89% after. The gap to BF16 is now within benchmark noise."}
import numpy as np
import matplotlib.pyplot as plt

configs = ['BF16\nbaseline', 'FP8 KV\nbefore fix', 'FP8 KV\nafter two-level']
values  = [91, 13, 89]
colors  = ['#00A8A8', '#FF007F', '#FFD700']

fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(range(3), values, color=colors, edgecolor='#1A1A1A', linewidth=1.8)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f'{v}%',
            ha='center', fontsize=14, fontweight='bold')

ax.axhline(91, color='#00A8A8', linewidth=1.2, linestyle=':', alpha=0.7)
ax.axhline(13, color='#FF007F', linewidth=1.2, linestyle=':', alpha=0.4)

# Recovery arrow
ax.annotate('', xy=(2.0, 86), xytext=(1.0, 17),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=2))
ax.text(1.5, 50, '+76 pts\nrecovered', ha='center', fontweight='bold',
        fontsize=11, color='#1A1A1A')

ax.set_xticks(range(3))
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel('needle-in-a-haystack accuracy (%)')
ax.set_ylim(0, 105)
ax.set_title('Llama-3.1-8B at 128k context — the fix in one chart',
             fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

A two-point gap from BF16 baseline is, for a quantization scheme, *excellent*. Earlier KV quantization methods from [Issue 3's KV method family tree](/comicbook/03-quantization/15-kv-method-family/) — KIVI at 2-bit, KVQuant at 4-bit — quote similar gaps but require explicit per-channel scales and a calibration pass. FP8 KV with two-level accumulation requires neither. It is the simplest possible KV quantization configuration, with the smallest possible accuracy loss, on the most popular open-weight model on the planet, at the most realistic long-context evaluation. The bargain from [Chapter 2](../02-bandwidth-bargain/) is back.

The same recovery shows up on essentially every long-context benchmark the team ran:

| Model | Task | BF16 | FP8 (before fix) | FP8 (after fix) | Recovery |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | NIAH @128k | 91% | 13% | 89% | 98% |
| Llama-3.3-70B-Instruct | MRCR @128k AUC | 100% | — | 97-98% | 97-98% |
| Qwen3-30B-A3B-Instruct | MRCR @256k AUC | 100% | — | 94-98% | 94-98% |
| Qwen3.5-27B | MRCR @1M AUC | 100% | — | 100% | 100% |
| Qwen3-30B-A3B-Thinking | AIME25 | — | — | -1 to -2 points | 97-99% |

The Qwen3.5-27B result at 1M tokens is particularly striking: full AUC recovery on a 1-million-token benchmark, using nothing but per-tensor uncalibrated FP8 KV cache plus the two-level accumulation fix. This is the configuration that would have been considered laughably aggressive in 2024.

## What Two-Level Costs: Register Pressure

Now the cost. The two-level fix is not free. The cost is **register pressure**.

To understand why, recall that a Hopper tensor core operates over a *warp group* — four warps cooperating on a single matrix tile. Each warp has access to a fixed-size register file. The FA3 kernel's inner loop is highly tuned for occupancy: every register matters, and the kernel is engineered to keep its working set inside the register file so that it doesn't spill to **local memory** (which is high-latency, defeats the point of using a tensor core).

Adding a second accumulator — the `acc_O_true` true-FP32 register — costs additional registers per warp. For `head_dim = 64` and `head_dim = 128`, the register file has slack and the cost is invisible. For `head_dim = 256` (Gemma family, some specialized models), the kernel was *already* near the register-file ceiling, and adding the true-FP32 accumulator pushes it over. The compiler then spills some of the inner-loop's working set to local memory, which slows the kernel.

The slowdown is specifically on **prefill**, not decode. Prefill is the phase where the kernel is doing many query rows in parallel — each query row needs its own accumulator state, and the register pressure scales with the number of concurrent rows. Decode has one query row, so the absolute register count is much smaller, and the two-level accumulator fits even at large `head_dim`.

The team's measurements, on Gemma-4-E2B with `head_dim = 256`:

- **TTFT quadratic coefficient before fix**: $6.93 \times 10^{-7}$ ms/token²
- **TTFT quadratic coefficient after fix**: $1.12 \times 10^{-6}$ ms/token²
- **Ratio**: ~1.6× slower prefill at long contexts

For a model where decode is the bottleneck, this is a fine trade. For a model where prefill latency directly impacts user time-to-first-token, it's painful. The team's response — `flash-attention#122` and `flash-attention#125`, both still in flight as of the issue's publication — is to *partially* amortize the two-level overhead via better tile-size selection and by promoting the accumulator only every $N$ tiles for adaptive $N$. The next chapter ([Register Wars](../07-tile-sizes/)) is the engineering story.

```pyplot {id="prefill-cost-by-headdim" caption="The cost of the fix lives in prefill at large head_dim. head_dim=64 and head_dim=128 absorb the extra register; head_dim=256 spills to local memory and prefill slows down."}
import numpy as np
import matplotlib.pyplot as plt

head_dims  = ['64', '128', '256']
prefill_before  = [1.0, 1.0, 1.0]
prefill_after   = [1.02, 1.04, 1.60]   # relative TTFT vs pre-fix

x = np.arange(len(head_dims))
w = 0.36

fig, ax = plt.subplots(figsize=(8, 4.4))
bars_b = ax.bar(x - w/2, prefill_before, w, color='#00A8A8',
                edgecolor='#1A1A1A', linewidth=1.4, label='before two-level fix')
bars_a = ax.bar(x + w/2, prefill_after, w, color='#FF007F',
                edgecolor='#1A1A1A', linewidth=1.4, label='after two-level fix')

for b in bars_b:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
            f'{b.get_height():.2f}', ha='center', fontsize=10)
for b in bars_a:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
            f'{b.get_height():.2f}', ha='center', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(head_dims)
ax.set_xlabel('head_dim')
ax.set_ylabel('relative TTFT (pre-fix = 1.0)')
ax.set_ylim(0, 2.0)
ax.set_title('Two-level fix is free on head_dim 64/128; spills at 256',
             fontsize=11, fontweight='bold')
ax.legend(framealpha=1, edgecolor='#1A1A1A')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Three bars per group is enough to tell the story. For `head_dim = 64` (Llama family small) and `head_dim = 128` (Llama family standard, Qwen3 family), the prefill cost is essentially zero. For `head_dim = 256` (Gemma-4-E2B, some research models), prefill is 60% slower. The team's pragmatic response is **make two-level accumulation the default** (because the accuracy recovery is essential), and provide an opt-out (`disable-two-level-accumulation`) for users on `head_dim = 256` who have done their own accuracy testing.

## Why Not Just Use Single-Step Kahan?

A reasonable question: if Kahan summation provides $O(\epsilon)$ error independent of $n$, why doesn't FA3 just do literal per-add Kahan compensation? The fix described in [the previous primer](../04-summation-history/) was four lines of code per add — surely the tensor core can do that?

It cannot, and the reason matters. Kahan compensation requires an *additional* register per accumulator slot (the `c` correction variable) and *three* additional operations per add (the `y - c`, the `c = (t - s) - y`, the final `s = t`). The tensor core's WGMMA instruction is a single-cycle multiply-accumulate with no slot for a second accumulator and no spare instruction bandwidth for the Kahan bookkeeping. To do single-step Kahan, you'd have to do the multiply-accumulate in *software* — separate multiply, separate add — which defeats the entire purpose of using a tensor core.

Two-level accumulation is the *engineering compromise*. It gives you most of Kahan's accuracy benefit (the inner block sums are small enough that the imprecision doesn't accumulate), keeps the tensor core's fast path intact for the inner loop, and pays only one extra register (the `acc_O_true` slot) plus one promote-and-reset per $N$ tiles. The accuracy guarantee is weaker than Kahan's — the error is bounded by $O(\epsilon)$ within a block plus $O(N \cdot \epsilon)$ across blocks — but the latter term is tiny for sensible $N$, and the structure preserves the tensor core's throughput.

This is the *generic structural pattern* of all hardware-aware numerical optimizations: take a theoretically-optimal algorithm, find the granularity at which it fits the hardware's register and instruction budget, accept a small loss in the asymptotic guarantee in exchange for a much better constant factor in practice.

{{% pullquote type="profound" %}}
Two-level accumulation is what Kahan summation looks like after a hardware compiler has had its way with it. Same algebraic family. Different granularity. Same effect on the error bound at the contraction sizes that matter.
{{% /pullquote %}}

## What Did Not Get Fixed By The Two-Level Patch

Two-level accumulation solves the accuracy bug, partially recovers the prefill bug, and is now the default FA3 FP8 path. What it *does not* fix:

- **Sliding-window models like gpt-oss-20b.** The accumulator wasn't the problem here — the FP8 path was already accuracy-fine on these models, even before the fix. The problem was *performance*: the FP8 break-even point was at 700k tokens because the sliding-window layers couldn't amortize the FP8 fixed overhead. The two-level fix has nothing to say about this. The fix for sliding-window models is layer-skipping, in [Chapter 9](../09-sliding-window-puzzle/).
- **`head_dim = 256` prefill latency.** Partially addressed by tile-size optimization (next chapter). The team also provides an opt-out for users who need pure prefill speed and have validated accuracy on their own workload.
- **Models with non-standard attention backends.** Kimi-K2.5 with FlashMLA shows a different failure mode that the two-level fix doesn't reach. This is the calibration story, in [Chapter 10](../10-calibrate/).
- **Blackwell.** The bug never existed on B200; the two-level fix is a no-op there. (See [the previous chapter](../05-accumulator-lie/) — the hardware was redesigned.)

The two-level fix is the central fix of this issue. It is not the only fix. The next four chapters cover the remaining problems that two-level accumulation revealed by making the underlying accuracy issue go away.

## What To Remember

1. **Two-level accumulation** is the SageAttention2 algorithm: fast tensor-core inner accumulator, true-FP32 outer accumulator, periodic promotion. Structurally a tiled Kahan-family algorithm.
2. **Accuracy recovery is essentially complete.** 91→89% on Llama-3.1-8B 128k needle-in-a-haystack; full AUC recovery on Qwen3.5-27B at 1M tokens; near-baseline on every model the team tested.
3. **Cost is register pressure** — invisible at `head_dim = 64` and `head_dim = 128`, ~1.6× prefill regression at `head_dim = 256` (Gemma-4-E2B). Decode is unaffected.
4. **Decode-bound users always want two-level accumulation on.** Prefill-bound users on `head_dim = 256` may want to opt out after their own accuracy testing.
5. **Two-level is not single-step Kahan.** It is the hardware-aware version: keep the inner loop fast, pay an extra register and an occasional add for the outer correction. The error bound is slightly weaker than full Kahan but the constant factors are much better.

**Continue to** → [Register Wars: Tile Sizes](../07-tile-sizes/) — the optimization story for the prefill regression. Why `head_dim = 256` spills, what tile-size sweeps recovered, and what the open `flash-attention#125` PR is still working on.
