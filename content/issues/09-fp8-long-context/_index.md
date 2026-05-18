---
title: "The Long-Context Lie"
description: "An issue on FP8 KV-cache quantization in vLLM — the 91→13% accuracy catastrophe nobody noticed for two years, the 1965 numerical-analysis trick that fixed it, and the production engineering that turned a flag into a default."
issue: 9
layout: issue-cover
theme: cream
math: false
header: default.webp
date: 2026-05-17T09:00:00-04:00
---

## The Mystery

**March 2026.** A team at AWS and Red Hat AI is running a routine validation sweep on vLLM, the open-source inference engine that quietly serves a large fraction of the world's open-weight LLM traffic. They are testing one specific feature: the `--kv-cache-dtype fp8` flag, available in vLLM for nearly three years. The flag halves the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} memory footprint by storing keys and values in {{< wiki "number-formats" >}}FP8 E4M3{{< /wiki >}} instead of BF16, and runs the entire {{< wiki "attention" >}}attention{{< /wiki >}} computation — the $QK^\top$ and the $\text{softmax} \cdot V$ — directly in FP8. On {{< wiki "flash-attention" >}}FlashAttention-3{{< /wiki >}} on Hopper, the math is supposed to accumulate into FP32 registers. The model is supposed to be fine.

The benchmark they pick is a **needle-in-a-haystack** task at 128,000 tokens of context. Llama-3.1-8B with BF16 KV cache scores 91% — bury a sentence in the middle of a 128k-token document, ask a question about it, expect a correct answer. Same model, same hardware, same prompt, with FP8 KV cache enabled: **13%**.

A seventy-eight-point regression. From a feature that was supposed to be free.

{{% pullquote type="counter-intuitive" %}}
Three years of "just enable the flag" advice in vLLM tutorials, and at long context, on the most popular open-weight model on the planet, the flag was silently destroying long-document recall.
{{% /pullquote %}}

What follows is the detective story of how the team traced the catastrophe to a hardware spec sheet that wasn't lying so much as carefully not telling the full truth, found the same bug independently documented in a DeepSeek-V3 training report from five months earlier, fixed it with a numerical-analysis idea **William Kahan** published in 1965 for very different reasons, and then re-fixed it three more times because the first fix triggered a register-spill cascade in the kernel, the second fix didn't help sliding-window models, and the third fix exposed a per-tensor scale issue that some attention backends really, really did not like.

Today, three months later, FP8 KV-cache is *finally* what the tutorials always promised: a one-flag toggle that nearly halves your memory bill, recovers 97-99% of BF16 accuracy on every standard benchmark, and lowers decode latency by ~14% at typical serving load.

This is the story of how it got there.

## The Cast

| Who | Where | What they did |
|---|---|---|
| **Jonas Kübler** | AWS | Caught the 13% number on the needle-in-a-haystack sweep. |
| **Eldar Kurtić** | Red Hat AI | Co-lead on the accuracy investigation. |
| **Lucas Wilkinson** | Red Hat AI | Optimized the FA3 FP8 tile sizes for memory-bound decode. |
| **Matthew Bonanni** | Red Hat AI | Per-head scales, query quantization fusion in vLLM. |
| **Michael Goin** | Red Hat AI | LLM-Compressor calibration integration. |
| **DeepSeek-V3 team** | Hangzhou | Documented the same FP32-accumulator precision loss during pretraining, five months earlier, in a footnote (Fig. 7b) of their technical report. |
| **Jintao Zhang et al. (SageAttention2)** | — | Published the two-level accumulation trick the vLLM team adopted as the fix. |
| **William Kahan** | UC Berkeley, 1965 | Wrote the original compensated-summation paper whose algebra underlies the fix. |

## How To Read This Issue

Start with the [cold open](01-cold-open/) — it puts you in front of the 13% number on the morning Kübler first reported it. After that, the **tech tree** below is the table of contents. Each node is an article; arrows show which articles build on which.

{{< techtree name="issue09" >}}

The four **pink mainline chapters** tell the story in order. *The Halving* explains why FP8 KV is irresistible in the first place — the linear ITL model, the bandwidth wall, the 2× memory win that should have been a free lunch. *The Accumulator Lie* is the detective story: how the team traced a 78-point accuracy drop to a quietly-imprecise FP32 accumulator inside Hopper's FP8 tensor cores. *Two Levels Of Honesty* is the fix: compensated summation, vintage 1965, adapted to a 2024 attention kernel. *The Window That Wouldn't Save* is the next problem the fix exposed — hybrid-attention models like gpt-oss-20b whose sliding-window layers refused to amortize FP8's fixed overhead.

The five **cream primer chapters** give depth on demand: read them when a mainline chapter refers to them, or read all of them first if you prefer foundations before narrative. *E4M3 In Three Steps* is the specific FP8 format used here and why per-tensor scale = 1.0 is the default. *The Sum Is Not What You Think* is Kahan's 1965 paper, the numerical-analysis backbone. *Slope vs Intercept* is the two-parameter ITL model and what "break-even at 7K tokens" actually means. *Register Wars* is the tile-size engineering reality — why head_dim=256 broke things. *Scale Equals One* is the calibration question — when uncalibrated FP8 is fine and when it isn't.

The **yellow boss chapter** — *The State Of FP8 KV* — is the destination, where all four mainlines and all five primers reconverge into the report card: which models, which benchmarks, which workloads ship FP8 KV today, and the three places it still doesn't fit.

If you are in a hurry: the mainline (pink nodes) is a three-hour read. If you want the full numerical-analysis treatment — Wilkinson at NPL, Kahan summation, the DeepSeek-V3 connection, why tensor cores lie about precision, and the open questions that still face vLLM 2026 — read everything in numerical order.

## What You Will Walk Away With

By the end of this issue you should be able to answer, with confidence and napkin math:

### The bargain and the lie

- Why FP8 KV-cache should be a **free 2× memory win** on any Hopper-or-newer GPU, what the ITL break-even point actually is, and why "free" was wishful thinking for long-context workloads.
- The exact arithmetic of the **FP8 E4M3** format (1 sign + 4 exponent + 3 mantissa bits, range ±448, log-spaced grid) and why {{< wiki "softmax" >}}softmax{{< /wiki >}} is unusually forgiving of FP8 noise.
- Why **uncalibrated per-tensor scale = 1.0** is the default in vLLM and what that choice is implicitly assuming about the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}'s value distribution.
- The detective story of the **91→13%** accuracy collapse on a 128k-token needle-in-a-haystack — what the symptom looked like, what the team initially blamed, and how they finally pinned it on the hardware.
- Why NVIDIA's Hopper documentation says FP8 tensor cores accumulate in FP32 and why, in practice, the accumulator runs out of meaningful precision when the contraction dimension exceeds ~100,000.

### The numerical-analysis backbone

- Why floating-point summation is **non-associative**: $(a + b) + c \ne a + (b + c)$ in finite precision, and what catastrophic cancellation looks like with FP32 dot products of ~100k summands.
- **Kahan summation** (1965): the one-loop compensated-summation algorithm that recovers most of the precision a naive accumulator throws away, and why it costs three extra FLOPs per addition for a guarantee of $O(\epsilon)$ error regardless of $n$.
- The **two-level accumulation** trick from SageAttention2 (2024): keep the inner accumulator in fast tensor-core registers, periodically dump to a true FP32 register, reset the inner accumulator. The 2024 version of Kahan's 1965 idea, in the same algebraic family.
- Why the same precision-loss footprint shows up in **DeepSeek-V3's training stack** (Fig. 7b of their technical report) for the same reason — and why this is the rare case where inference and training engineers found the same bug from opposite directions.

### The kernel engineering reality

- Why two-level accumulation **breaks even on accuracy** (91% recovered to 89%) but **costs prefill speed**: the extra register pressure spills to local memory at large `head_dim`, especially `head_dim = 256`.
- The **tile size optimization story** for FA3 FP8: why `head_dim = 64` and `head_dim = 128` got speedups while `head_dim = 256` (Gemma-4-E2B) is currently a regression on prefill.
- **Query quantization fusion**: why moving the per-token Q quantization out of the attention backend and into a `torch.compile`-fusable preamble eliminated a fixed per-token overhead.
- **Per-head scales**: why FA3 supports an array of scales (one per KV head) instead of a single scalar, what it took to wire that through vLLM's `reshape_and_cache_flash` kernel, and what it buys you when activation distributions vary per head.

### The hybrid-attention puzzle

- Why **sliding-window attention layers** (gpt-oss-20b, Gemma) refuse to amortize FP8's fixed per-token overhead — their KV cache is bounded by the window size, so the 2× memory win doesn't grow with context.
- The arithmetic: gpt-oss-20b's FP8 break-even was at **741,565 tokens** of context with full FP8, dropping to **7,659 tokens** with the new `--kv-cache-dtype-skip-layers sliding_window` flag.
- Why this is the architectural argument for **hybrid precision** in serving stacks: bounded-memory layers stay in BF16, long-context global-attention layers go FP8.

### The calibration question

- What "uncalibrated" means in this context: per-tensor scale fixed at 1.0, no calibration data, no per-head tuning. The simplest possible FP8 configuration.
- Why this works on most well-validated paths (FA3, FlashInfer) but produces a **consistent downward shift** on **Kimi-K2.5** with the FlashMLA backend — and what that says about which backends inherit the long-context fixes and which ones don't.
- When to reach for **LLM-Compressor** and per-head scales: the data-collection workflow, the residual accuracy you can recover, the production engineering trade-off.

### The state of FP8 in 2026

- Why **FP8 KV-cache is now the recommended default** for many long-context vLLM deployments, with three named exceptions (short contexts < 7k, `head_dim = 256` prefill-heavy workloads, custom attention backends without the FA3 fixes).
- Why **Blackwell (B200)** does not need the two-level accumulation fix — what changed in the silicon, and what that says about the silicon-software co-evolution we've been tracing since [Issue 3's Hardware Horizon](/issues/03-sixteen-numbers/12-hardware-horizon/).
- How FP8 KV fits into the broader [KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/) from Issue 3 — and why the simplest method on the shelf became the one that took two years and a numerical-analysis paper to finish properly.

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
