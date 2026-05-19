---
title: "Hopper FP8: when the FP32 accumulator stops accumulating"
short_title: "Hopper FP8"
description: "NVIDIA documents FP8 tensor cores as accumulating into FP32 registers. At long contraction dimensions the precision quietly evaporates — the same bug DeepSeek-V3 hit during training five months earlier."
blurb:
  - "NVIDIA docs: FP8 tensor cores accumulate into FP32 registers. Mostly true."
  - "At long contraction dimensions, the effective precision quietly evaporates."
  - "DeepSeek-V3 hit the same hardware quirk five months earlier, during training."
  - "The question nobody asked for three years: *\"Is the FP32 accumulator actually FP32?\"*"
topics: [quantization, hardware, kv-cache, attention]
tags: [fp8, hopper, tensor-cores, fp32-accumulator, flash-attention-3, deepseek-v3]
theme: cream
math: true
draft: false
date: 2026-05-18T09:00:00-04:00
issue: 9
weight: 50
techKind: mainline
techNode: accumulator-lie
header: default.webp
---

## The Question The Team Stopped Avoiding

By the third week of debugging, **Jonas Kübler** and **Eldar Kurtić** have ruled out every software cause they can think of. The vLLM scheduler is innocent. The Triton kernel paths are innocent. The Python glue is innocent. The model itself is the same model — it scores 91% in BF16, 13% in FP8, on the same prompts, on the same GPU.

The cause has to be in the FP8 path itself. And the FP8 path's central object is the **FA3 FP8 attention kernel** — the fused {{< wiki "flash-attention" >}}FlashAttention-3{{< /wiki >}} kernel for Hopper that reads K and V from cache in FP8, computes $QK^\top$ in FP8, and computes $\text{softmax}(QK^\top) \cdot V$ in FP8, accumulating partial results into FP32 registers. That last clause — *accumulating partial results into FP32 registers* — is the line in the NVIDIA documentation everyone has been quoting for three years. FP32 accumulators were supposed to make FP8 attention numerically robust.

The question the team stops avoiding in the third week is:

> *Is the FP32 accumulator actually FP32?*

The answer, when they find it, is: *kind of, sort of, in a way that matters at small contraction dimensions and fails at large ones.* The hardware reality is more nuanced than the documentation suggests. And the moment the team articulates this question, two things happen in quick succession. First, the fix becomes obvious — it's the [Kahan algorithm from 1965](../04-summation-history/) tiled differently. Second, they find the same observation, made by a different team for a different reason, sitting in **Figure 7(b)** of the **DeepSeek-V3 technical report** from December 2024, five months earlier.

This chapter is the autopsy. We will spend it understanding exactly what the FA3 FP8 inner loop does, why the FP32 accumulator quietly stops being effective at long context, and what the DeepSeek footnote tells us about the same bug appearing in training.

## What The FA3 FP8 Path Actually Does

To follow the bug you have to know what the kernel is doing in some detail. [Issue 8's primer on FlashAttention](/comicbook/08-vLLM/06-flash-attention/) covers the full tiling story; the relevant inner loop for our purposes is the one that computes a single row of the implicit $QK^\top$ score matrix.

In pseudocode, for a single Q tile in the outer loop and a single KV tile in the inner loop:

```python
# Inner loop iteration: process one (Q tile, KV tile) pair.
# Q_tile is (Bq, D), K_tile is (Bk, D), V_tile is (Bk, D).
# All three are FP8 E4M3 in this kernel.

S_tile = wgmma_fp8_to_fp32(Q_tile @ K_tile.T) / sqrt(D)
# S_tile is (Bq, Bk) in FP32. Lives in SRAM, never written to HBM.

# Online softmax updates omitted — see issue 08 for the full algorithm.
m_b   = S_tile.max(axis=1)
P_tile = exp(S_tile - m_b)
ell_b = P_tile.sum(axis=1)

# The key second matmul.
O_tile = wgmma_fp8_to_fp32(P_tile.to_fp8() @ V_tile)
# O_tile is (Bq, D) in FP32. Accumulates across inner loop iterations.
```

The two important operations are the two `wgmma_fp8_to_fp32` calls. WGMMA — **W**arp **G**roup **M**atrix **M**ultiply-**A**ccumulate — is Hopper's tensor-core instruction. It takes two FP8 matrix tiles, multiplies them, and accumulates the result into an FP32 register. The documentation describes this as a fused operation: each multiply-accumulate is a single FMA cycle, the products are in FP8, and the partial sum is held in FP32 throughout.

This is what the documentation says. The reality is more delicate.

## The Hardware Reality, Spelled Out

Inside an H100's tensor core, the actual data path for a WGMMA FP8-to-FP32 operation is roughly:

1. Two FP8 source operands flow into the multiplier.
2. The multiplier produces an *intermediate* product. NVIDIA does not fully publicly specify the bit-width of this intermediate, but it is *not* FP32 — it is narrower, and it is enough to hold an exact product of two FP8 values (which is bounded in mantissa width because the inputs themselves are narrow).
3. The intermediate product is *added* into the running accumulator slot.
4. The accumulator slot is the architectural FP32 register that the documentation refers to.

The trouble is at step 3. The **adder pathway** inside the tensor core has a fixed precision budget that does *not* fully match a software FP32 add. Specifically, when the running accumulator has grown large (in magnitude) and the incoming product is small, the alignment shift in the adder drops bits the way [the previous chapter's naive summation](../04-summation-history/) drops bits. The accumulator's *storage* is FP32, but the *operation that updates it* is not a full-precision FP32 add.

NVIDIA's documentation says "accumulates into FP32". This is technically true — the destination register is FP32. The documentation does not say "with full FP32 arithmetic precision per add", because internally it isn't quite. The team's bug lives in the gap between those two statements.

For most workloads, the gap is invisible. A typical matmul in a transformer's MLP has a contraction dimension of $4096$ or $8192$. Summing $8192$ FP8 products into an FP32 accumulator works fine — the accumulator never grows large enough relative to the per-add increments for the alignment shift to drop meaningful bits. The same goes for short-context attention.

The bug appears when the *contraction dimension is the context length*. Specifically, during the $\text{softmax}(QK^\top) \cdot V$ matmul during decode, the contraction dimension is $T$ — the number of tokens already in the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}. At $T = 100{,}000$, you're summing 100,000 FP8 products into the accumulator. By the time you've added the first 30,000, the accumulator has grown large enough that subsequent FP8 increments are partially being shifted off the end of the alignment.

```pyplot {id="accumulator-precision-vs-contraction" caption="Simulated relative error of an FP32 accumulator with a Hopper-like adder precision budget, as a function of contraction dimension. Up to ~30k summands, error is invisible. Past that, it grows linearly. The pink region is where the FP8 KV cache regression lives."}
import numpy as np
import matplotlib.pyplot as plt

# Simulate a "leaky" FP32 accumulator that quantizes additions to a coarser grid.
# This is a simplified stand-in for the actual hardware behavior.
np.random.seed(3)

def leaky_add(s, x, leak_bits=4):
    """Simulate an add where the smaller magnitude loses `leak_bits` bits during align."""
    if abs(s) < 1e-30:
        return s + x
    # Compute exponent difference
    e_s = np.floor(np.log2(abs(s) + 1e-30))
    e_x = np.floor(np.log2(abs(x) + 1e-30))
    shift = max(0, int(e_s - e_x))
    if shift > 0:
        # Mask off the bottom `leak_bits` bits of x
        scale = 2 ** (leak_bits - max(0, 23 - shift))
        x = np.round(x * scale) / scale
    return s + x

contractions = np.unique(np.logspace(2, 6, 60).astype(int))
relative_errors_naive = []
relative_errors_block = []

for T in contractions:
    # Synthetic product stream: ~N(1, 0.3) so the accumulator grows
    prods = np.random.RandomState(0).randn(T).astype(np.float32) * 0.3 + 1.0
    exact = np.sum(prods.astype(np.float64))

    # Naive leaky FP32 accumulator
    s = np.float64(0.0)
    for p in prods:
        s = leaky_add(s, p, leak_bits=8)
    relative_errors_naive.append(abs(s - exact) / abs(exact))

    # Block-accumulated (two-level): every 64 elements, dump to true FP32
    s_outer = np.float64(0.0)
    block = 64
    for start in range(0, T, block):
        s_inner = np.float64(0.0)
        for p in prods[start:start + block]:
            s_inner = leaky_add(s_inner, p, leak_bits=8)
        s_outer = s_outer + s_inner
    relative_errors_block.append(abs(s_outer - exact) / abs(exact))

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(contractions, relative_errors_naive, color='#FF007F', linewidth=2.6,
        label='naive accumulator (FA3 FP8 default before fix)')
ax.plot(contractions, relative_errors_block, color='#00A8A8', linewidth=2.6,
        label='two-level (after fix, block=64)')

ax.axvspan(3e4, 1.3e5, color='#FF007F', alpha=0.1)
ax.text(6e4, 1e-1, 'long-context\nregression\nzone', ha='center',
        fontsize=9, color='#FF007F', fontweight='bold')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('contraction dimension (number of summands)')
ax.set_ylabel('relative error of accumulator')
ax.set_title('Why the bug only shows up past ~30k tokens of context',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper left', framealpha=1, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.15, which='both')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The pink curve grows linearly past the regression zone. The teal curve stays flat. The structural difference is exactly the difference between [naive summation and Kahan-family compensated summation](../04-summation-history/), at the granularity of tensor-core block sizes. The bug's shape predicts the fix's shape.

## How The Team Found It

The detective work, in three rough phases.

**Phase 1: Probe the kernel.** The team writes a controlled benchmark that calls the FA3 FP8 kernel directly with a synthetic prompt of known structure — a needle they themselves chose, at a position they themselves chose. They compare the attention scores produced by the FP8 kernel against the same scores computed in pure FP32 outside the kernel. At short context (say, $T = 4096$), the FP8 scores match the FP32 reference to within a small relative error. At long context (say, $T = 128{,}000$), the FP8 scores are systematically off, and the error grows with $T$.

This pinpoints the bug as living inside the kernel's accumulation, not in the FP8 quantization itself. (The per-element quantization error from [Chapter 3](../03-what-is-fp8-here/) is well-bounded and doesn't grow with $T$.)

**Phase 2: Replace the kernel's accumulator with a software FP32 sum.** They modify a debug build of the FA3 kernel to use a deliberately slow, software-emulated FP32 accumulator — essentially summing the FP8 products in a Python-equivalent loop that uses a true full-precision add at every step. The bug disappears. The 13% accuracy comes back to 91%. This confirms the kernel's *logic* is right; the kernel's *accumulator* is the problem.

**Phase 3: Reproduce the same precision loss in isolation.** They write a minimal CUDA program that does nothing but call WGMMA FP8-to-FP32 in a long-contraction-dimension matmul, and compare against a CUDA reference using FP32-to-FP32 cores. The same precision loss appears, isolated from any of the FA3 complexity. At this point the bug has nothing to do with vLLM, has nothing to do with attention, and has nothing to do with the model. It is purely a property of the tensor core's behavior on long contractions of FP8 inputs.

This is the moment the team realizes they are not looking at a software bug. They are looking at a *hardware behavior whose documentation is imprecise*, surfacing as a software bug at exactly the workload sizes nobody had specifically stressed.

{{% callout type="warning" title="The Doc Sheet Was Not Wrong" %}}
NVIDIA's documentation for Hopper FP8 WGMMA does say the accumulator is FP32. What it does not say — clearly, in the place the inference engineers would have looked — is that the *adder pathway* inside the tensor core does not fully match software FP32 add semantics at large operand magnitude ratios. The information was probably available to someone who read the hardware spec at the bit-level; it was not available to the people writing the kernel from the API documentation.

This is not a NVIDIA failure. It is a *layer interface* failure. The hardware ABI was correct; the user-facing description was the level of abstraction at which the precision contract should have been stated, and wasn't. The fix lives in software because the hardware is what it is.
{{% /callout %}}

## The DeepSeek-V3 Footnote

Five months before the AWS / Red Hat team starts this investigation, **DeepSeek-AI** publishes the [**DeepSeek-V3 technical report**](https://arxiv.org/abs/2412.19437). The report is famous for its FP8 mixed-precision training recipe, which is the first widely-cited demonstration that a frontier-scale MoE model can be pretrained almost entirely in FP8.

Buried in **Section 3.3** of the V3 report is a discussion of FP32 accumulator precision. The DeepSeek engineers found, during their pretraining runs, that the Hopper FP8 tensor cores' FP32 accumulators were losing precision at large contraction dimensions during *training* — the same bug, surfacing in *backward-pass* matmuls where the contraction dimension was the model's hidden dimension multiplied by the sequence length, well into the hundreds of thousands. Their **Figure 7(b)** shows the relative precision of an FP8 matmul as a function of the accumulator's contraction dimension, looking eerily similar to the pink curve in the chart above: flat below a threshold, growing linearly past it.

The V3 team's solution was a custom CUDA kernel that mimicked compensated accumulation: after every $N$ multiply-accumulates inside the tensor core, the accumulator was dumped to a software FP32 register and reset. They called it *promotion accumulation*. It's structurally identical to the two-level accumulation in [SageAttention2](https://arxiv.org/abs/2411.10958) that the vLLM team adopts a year later.

The discovery is, in some sense, *the same discovery made twice* — once by a training team and once by an inference team. The training team published it as a side note in the technical report and moved on. The inference team, working in a different framework on a different workload, hit the same wall, found the V3 footnote during their literature search, and built the fix on top of it.

This is a small but characteristic pattern in modern ML systems: **the same hardware bug surfaces in different layers of the stack at different times, depending on whose workload happens to stress the precision contract first**. Training engineers and inference engineers rarely read each other's papers. When they do, the same bug-and-fix pair often resolves both problems.

{{% pullquote type="profound" %}}
The DeepSeek-V3 team and the vLLM team both, independently, hit the same FP32-accumulator precision loss in Hopper tensor cores. The first team pretrained around it. The second team inferenced around it. The fix that ships in both places is, algebraically, the algorithm William Kahan published in 1965.
{{% /pullquote %}}

## Why The Bug Is Specific To Decode-Heavy Long Context

To put a sharp point on it: the bug is not a generic FP8 problem. It is a *long-contraction* problem. Three pieces have to line up for the bug to bite:

1. **The matmul has to be FP8 in, FP32 accumulator out.** This is the FA3 FP8 path. If you accumulated in a wider register, the bug pushes out further. If you accumulated in BF16, the bug bites much sooner. FP32 accumulation is the standard, and the bug lives at the threshold where it stops being enough.
2. **The contraction dimension has to be large.** Roughly $> 30{,}000$ in single precision, possibly less depending on the input distribution. This is the context length during the $\text{softmax}(QK^\top) \cdot V$ matmul in decode.
3. **The accumulator has to be summing values of varying magnitudes.** If every product were the same size, the alignment shift wouldn't drop meaningful bits. In real attention, the product magnitudes vary substantially across the sequence (some tokens are sinks, some are bulk), so the alignment shifts are non-trivial.

All three conditions are met in long-context decode. The first two are met in prefill, too, but the regression is less obvious because prefill is compute-bound (not bandwidth-bound) and the per-token output is summed across many query positions, which averages over noise patterns. Decode is the place where one query attends to a hundred thousand keys and the precision loss in that single dot product becomes visible in the model's behavior.

The Llama-3.1-8B numbers from the cold open are a clean illustration:
- At $T = 30{,}000$: FP8 needle accuracy ~85%, BF16 ~92%. Bug exists but is small.
- At $T = 60{,}000$: FP8 ~50%, BF16 ~91%. Bug is severe.
- At $T = 128{,}000$: FP8 ~13%, BF16 ~91%. Bug has taken over.

Everywhere FP8 KV-cache was being benchmarked at typical context lengths (4k-16k), the bug was invisible. Everywhere it was being deployed at production long contexts (32k+), the bug was destroying recall. Three years of users in the second regime, with nobody in the first regime knowing the second regime existed.

## The Hopper-Specific Caveat

One more thing the team confirms before moving on to the fix: the bug is **specific to Hopper**. They re-run the same FP8 KV-cache benchmark on **Blackwell (B200)** using the FlashInfer backend. No regression. The FP32 accumulator on Blackwell's tensor cores does not exhibit the same alignment-shift precision loss.

NVIDIA does not publicly document the Blackwell tensor core's accumulator pathway in fine-grained detail. But the empirical observation is consistent: the team's needle-in-a-haystack benchmark on B200 / FlashInfer / Llama-3.1-8B reaches BF16-parity accuracy even at 128k context, without any of the two-level accumulation machinery. The hardware-level precision loss is, apparently, fixed.

This is the silicon-software co-evolution story we have been tracking since [Issue 3's Hardware Horizon](/comicbook/03-quantization/12-hardware-horizon/). Software discovers a precision bug. Hardware silently fixes it in the next generation. Software writes a workaround for the current generation. The workaround lives in code essentially forever, because somebody is always going to be running the previous generation, but the architectural pressure that produced the workaround is gone.

This is a Hopper-only story now. By 2027 it will be a legacy story. The two-level accumulation code will still ship in vLLM in 2030 for the same reason `KAHAN_SUM` lives in BLAS today: because somewhere, somebody is still running the older hardware, and the fix doesn't cost anything on the newer hardware.

## What To Remember

1. **Hopper's FP8 tensor cores accumulate into an FP32 register**, as documented. The adder pathway inside the tensor core is *not* full-precision FP32 — it drops mantissa bits during alignment shifts when the running accumulator is large relative to the incoming product.
2. **The bug bites past ~30,000 summands.** Specifically: the contraction dimension of the $\text{softmax}(QK^\top) \cdot V$ matmul during decode, which is the context length.
3. **Llama-3.1-8B's accuracy collapse** (91→13% on the 128k needle-in-a-haystack) is the direct fingerprint of this bug.
4. **DeepSeek-V3 hit the same bug during training** five months earlier, documented it in Figure 7(b) of their technical report, and shipped their own version of compensated accumulation as a fix.
5. **Blackwell does not have the bug.** The accumulator pathway was redesigned. The two-level accumulation fix is, going forward, a *Hopper-legacy* workaround.

**Continue to** → [Two Levels Of Honesty](../06-two-level-fix/) — how SageAttention2's two-level accumulation algorithm got wired into the FA3 FP8 kernel, the 91→89% accuracy recovery it delivered, and the register-pressure cost it triggered.
