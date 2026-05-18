---
title: "FP8 KV in vLLM: the 2026 report card"
description: "Four model families, two GPU generations, one verdict: mostly safe to ship. Three named exceptions — and where FP8 fits in the broader KV quantization family tree."
blurb:
  - "Four model families, two GPU generations (Hopper and Blackwell): mostly safe to enable."
  - "Three named exceptions where FP8 KV still doesn't fit — and what to use instead."
  - "The flag that silently broke recall for three years is now the recommended default."
  - "How FP8 sits among the eight modern KV quantization methods (Comic Book #3)."
topics: [quantization, kv-cache, inference, production]
tags: [fp8, vllm, state-of-the-art, benchmarks, hopper, blackwell, capstone]
theme: cream
math: true
draft: false
date: 2026-05-18T09:00:00-04:00
issue: 9
weight: 110
techKind: boss
techNode: state-of-fp8
header: default.webp
---

## The Final Report Card

Three months after **Jonas Kübler**'s 13% number lit up the Slack channel, the FP8 KV-cache story has a clean ending. The two-level accumulation fix [recovered the accuracy](../06-two-level-fix/). Tile-size optimization [recovered most of the prefill latency](../07-tile-sizes/) at the painful `head_dim = 256` case. The layer-skip flag [fixed sliding-window hybrid models](../09-sliding-window-puzzle/). Per-head scales and query quantization fusion landed as smaller engineering wins. The calibration workflow [is documented](../10-calibrate/) for the few cases where the default isn't enough.

This is the report card. The full state of FP8 KV-cache in vLLM as of mid-2026, across every model and benchmark the AWS / Red Hat team ran. The takeaway up front: **FP8 KV-cache is now the recommended default starting point for many long-context vLLM deployments**, with three named exceptions that we will name carefully at the end.

This capstone has three parts. First, the full benchmark table — what FP8 KV-cache does on each model the team validated. Second, the three exceptions — when not to enable it. Third, the connection back to [Issue 3's KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/) and the broader story of where FP8 fits among the eight modern KV quantization methods.

## What Ships, Per Model

The team validated FP8 KV-cache across four model families, on both Hopper (H100) with FA3 and Blackwell (B200) with FlashInfer. The headline accuracy numbers — pass@1 on reasoning benchmarks, AUC on long-context benchmarks — are summarized below.

| Model | Backend | Reasoning recovery | Long-context recovery |
|---|---|---:|---:|
| Llama-3.1-8B | FA3 (H100) | n/a (no reasoning evals) | needle@128k: 89% (vs 91% BF16) |
| Llama-3.3-70B-Instruct | FA3 (H100) | n/a | MRCR@128k AUC: 97-98% |
| Qwen3-30B-A3B-Instruct-2507 | FA3 (H100) | n/a | MRCR@256k AUC: 94-98% |
| Qwen3-30B-A3B-Instruct-2507 | FlashInfer (B200) | n/a | MRCR@256k AUC: 93-96% |
| Qwen3-30B-A3B-Thinking-2507 | FA3 (H100) | 97-99% across 4 benchmarks | n/a |
| Qwen3-30B-A3B-Thinking-2507 | FlashInfer (B200) | ~98% across 4 benchmarks | n/a |
| Qwen3.5-27B | FA3 (H100) | 99-100% | MRCR@1M AUC: **100%** |
| gpt-oss-20b | FA3 (H100) + skip-SW | n/a | within 1 point of BF16 |
| Kimi-K2.5 | FlashMLA | needs per-tensor calibration | shifted ~3-4 pts uncalibrated |

The pattern across the table: on every well-validated path (FA3 with two-level accumulation, FlashInfer on B200), FP8 KV-cache recovers 94-100% of the BF16 baseline AUC at every context length up to 1M tokens. The exception — Kimi-K2.5 with FlashMLA — is the calibration case from [Chapter 10](../10-calibrate/), where the issue is the backend's specific code path rather than FP8 itself.

The Qwen3.5-27B result deserves a special mention. **At one million tokens of context, with per-tensor uncalibrated FP8 KV cache plus FP8 attention, the model achieves 100% of its BF16 AUC.** This is the configuration that would have been considered "obviously broken" in 2024 — a one-flag toggle, no calibration, a million-token context — and in 2026 it works. The fact that this passes the stress test is, in some sense, the strongest possible validation of the team's fixes from the past three months.

## Performance: Throughput And Latency

The accuracy side is one half of the report card. The performance side is the other half. Two summary tables, both from the team's serving benchmarks: 150 requests, concurrency 8, ~20k input tokens, ~2k output tokens.

### Llama-3.1-8B

| Config | Median TTFT | Median ITL | Total duration | Output tok/s |
|---|---:|---:|---:|---:|
| BF16    | 763.6 ms | 15.18 ms | 672.6 s | 450.3 |
| **FP8** | **742.8 ms** | **12.93 ms** | **585.2 s** | **517.5** |

**+14.9% throughput, -14.8% median ITL, -3% TTFT, no accuracy regression.** The bargain from [Chapter 2](../02-bandwidth-bargain/) realized in full. This is what users were expecting when they flipped the flag in 2023; this is what they're finally getting in 2026.

### gpt-oss-20b

| Config | Median TTFT | Median ITL | Total duration | Output tok/s |
|---|---:|---:|---:|---:|
| BF16          | 468.9 ms | 8.09 ms | 364.2 s | 831.6 |
| FP8 (full)    | 451.7 ms | 7.90 ms | 355.1 s | 853.0 |
| **FP8 skip-SW** | **456.4 ms** | **7.70 ms** | **347.4 s** | **871.8** |

**+4.8% throughput from skip-SW vs +2.6% from plain FP8.** Smaller win than Llama because of the hybrid attention architecture (most layers are sliding-window and don't benefit from FP8), but still a real win. The skip-SW flag captures another 2 percentage points beyond what plain FP8 delivers.

### Blackwell (B200)

The same kind of pattern shows up on Blackwell with FlashInfer, but without the need for the two-level accumulation workaround (because B200's tensor cores don't have the precision bug). Llama-3.1-8B on B200:

- BF16 slope: $1.80 \times 10^{-5}$ ms/token
- FP8 slope:  $9.72 \times 10^{-6}$ ms/token — break-even ~4k tokens

The FP8/BF16 slope ratio on B200 is also 54% — the same theoretical floor Llama hits on Hopper. The break-even is slightly lower because the kernel intercept overhead is smaller. The bargain is, if anything, slightly cleaner on B200 than on H100.

```pyplot {id="state-of-fp8-summary" caption="The headline numbers in one chart. Accuracy recovery (% of BF16 baseline AUC) on the x-axis, throughput gain (% over BF16) on the y-axis. Everything in the upper-right corner is FP8 KV doing its job. Bigger marker = more model parameters."}
import numpy as np
import matplotlib.pyplot as plt

models = [
    ('Llama-3.1-8B',                 14.9, 98, 8,    '#FF007F', 'H100'),
    ('Llama-3.3-70B-Instruct',       12,   97, 70,   '#FF007F', 'H100'),
    ('Qwen3-30B-A3B-Inst-2507',      10,   96, 30,   '#00A8A8', 'H100'),
    ('Qwen3-30B-A3B-Thinking-2507',  9,    98, 30,   '#00A8A8', 'H100'),
    ('Qwen3.5-27B',                  11,  100, 27,   '#00A8A8', 'H100'),
    ('gpt-oss-20b (skip-SW)',         4.8, 99, 20,   '#FFD700', 'H100'),
    ('Kimi-K2.5 (FlashMLA, uncal.)',  9,   93, 70,   '#FF8C00', 'H100'),
]

fig, ax = plt.subplots(figsize=(10, 6))
for name, throughput, accuracy, params, color, _ in models:
    ax.scatter(accuracy, throughput,
               s=100 + params * 5,
               color=color, edgecolor='#1A1A1A', linewidth=1.5,
               alpha=0.75, zorder=5)
    ax.annotate(name, (accuracy, throughput),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8.5)

ax.axhspan(0, 20, color='#00A8A8', alpha=0.05)
ax.axvspan(95, 101, color='#00A8A8', alpha=0.05)
ax.text(95.1, 0.5, 'production-ready zone',
        fontsize=9, color='#1A1A1A', style='italic')

ax.set_xlim(91, 101)
ax.set_ylim(0, 18)
ax.set_xlabel('accuracy recovery (% of BF16 baseline)')
ax.set_ylabel('throughput gain (% over BF16)')
ax.set_title('The state of FP8 KV-cache, after the fixes', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.15)
plt.tight_layout()
```

Everyone except Kimi is comfortably in the "production-ready zone" — > 95% accuracy recovery, positive throughput gain. Kimi sits just outside because of the uncalibrated FlashMLA shift, which a calibration step fixes (moving the orange marker to the right).

## The Three Places FP8 KV Does Not Fit

Three named exceptions. The team is explicit about these because the *failure mode* of FP8 KV-cache when it's the wrong fit is "no benefit at the cost of operational complexity" rather than "model breaks" — but it's still the wrong choice for these workloads.

### Exception 1: Short Contexts (< ~7k tokens)

The break-even arithmetic from [Chapter 8](../08-itl-slope-model/) tells you when FP8 KV starts winning. Below the break-even point, FP8 is marginally *slower* than BF16 because the small intercept overhead (the conversion preamble, the per-token quantization work) outweighs the slope reduction. For workloads where the average context is below 7,000 tokens — short-prompt code completion, simple chatbots, tight Q&A pipelines — BF16 is the right default.

Most production workloads are not in this regime, because long contexts dominate the latency budget even when they're rare. But for specifically short-context workloads, the FP8 flag is not the right move.

### Exception 2: `head_dim = 256` Models With Prefill-Sensitive Workloads

The two-level accumulation fix adds register pressure that spills at `head_dim = 256`, slowing prefill by ~1.6× even after the tile-size optimization in PR #125. For workloads where prefill latency directly impacts user-visible time-to-first-token — chat applications, interactive editors, anything where the user is staring at a blank screen during prefill — this regression matters.

The team provides an opt-out (`disable-two-level-accumulation`) that recovers prefill speed at the cost of accuracy on long-context workloads. The opt-out is reasonable for users who have validated accuracy on their own specific workload and confirmed that the kernel's precision drift doesn't manifest there. It is *not* a recommended default, because the accuracy risk is real on long-context workloads.

For most users on `head_dim = 256` models who want both prefill speed and long-context accuracy, the right answer in 2026 is: wait for [`flash-attention#122`](https://github.com/vllm-project/flash-attention/pull/122) (which makes the accumulation promote-every-N-steps adaptive) to land, or move to Blackwell where the issue doesn't exist.

### Exception 3: Non-Standard Attention Backends Without The Fixes

The team's optimizations are tightly coupled to specific kernel paths — FA3 on Hopper, FlashInfer on Blackwell, plus a smaller set of backend variants. Some models use other backends:

- Kimi-K2.5 uses FlashMLA (and benefits from calibration, per [Chapter 10](../10-calibrate/)).
- Some research models use vLLM's older Triton backends, which don't have the FA3 FP8 fixes.
- Custom inference engines (TensorRT-LLM, SGLang, exotic in-house stacks) have their own FP8 KV paths that may or may not have shipped equivalent fixes.

The team's recommendation for these cases is to **measure**. The accuracy fingerprint of the precision issue (catastrophic drop at long context) is distinctive; the fingerprint of the calibration issue (consistent shift) is distinctive; and the fingerprint of "actually fine" is also distinctive. Run your own benchmarks. If you see the long-context collapse, you have an accumulator problem that needs a fix in your backend; if you see a flat shift, calibrate; if you see neither, you're fine.

## Where FP8 KV Lives In The Broader Method Family

Back to [Issue 3's KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/). That tree, in May 2026, lists eight methods: FP8, KIVI (2-bit), KVQuant (4-bit), GEAR (low-rank + sparse + 4-bit), ATOM (W4A4KV4), QServe (W4A8KV4), KVTuner (per-layer adaptive bits), TurboQuant (random rotations). The selection table at the end of that article asks: which one do you actually use?

FP8 KV's position on the tree is the "always-on baseline" — the floor of compression you can flip on with no calibration, no setup, no operational complexity, on every modern GPU. The other methods compress harder (4-bit, 2-bit) but require more bookkeeping. FP8 is the *first* thing you enable, not the *only* thing you enable.

The May 2026 update to the selection table, in light of this issue:

| Method | Bits | Calibration | Online cost | KV speed-up | Best at | Failure mode |
|---|---|---|---|---|---|---|
| **FP8 KV (post-fixes)** | 8 | none (usually) | nil | 2× | always-on baseline | head_dim=256 prefill, hybrid SW models |
| KIVI | 2 | per-layer K-channel | nil | 8× | 2-bit ceiling | RoPE drift on long context |
| KVQuant | 4 | non-uniform grids | nil | 4× | 4-bit accuracy ceiling | calibration-heavy |
| GEAR | 4 | streaming SVD | moderate | 4× | best 4-bit accuracy | adds decode latency |
| ATOM | 4 | end-to-end W4A4KV4 | nil online | 4× | full-INT4 inference | edge of accuracy |
| QServe | 4 | smoothing | nil | 4× | production serving | serving-tuned |
| KVTuner | 2-8 mix | sensitivity sweep | nil | 4-6× avg | best memory-quality | layered on others |
| TurboQuant | 4 | none | small (Hadamard) | 4× | calibration-free 4-bit | needs fused Hadamard |

The first row's "failure mode" column is the contribution of this issue. We now *know* — empirically, with benchmarks — where FP8 KV doesn't fit. Two years ago, that column would have said "long-context accuracy?" with a question mark. After this issue, the question mark is gone.

The interesting consequence: **FP8 KV-cache and the more aggressive quantization methods now compose**. You can run FP8 KV on the global-attention layers, KVTuner on some layers for more aggressive 4-bit quantization where the model tolerates it, leave specific outlier-sensitive layers in BF16, and combine the whole stack into a hybrid policy that achieves better effective compression than any single method. The vLLM stack is getting there — the per-layer skip flag from [Chapter 9](../09-sliding-window-puzzle/) is the first move; the per-head scales are the next move; the long-term direction is finer-grained policy that allocates precision by sensitivity.

## What FP8 KV Looks Like Beside Its Cousins

Three brief comparisons.

**FP8 KV vs KIVI (2-bit).** KIVI achieves 4× more compression (2-bit vs 8-bit) at the cost of calibration and per-channel scaling. For workloads that are *memory-bound at long context* and where you need every byte of cache savings, KIVI is more aggressive. For workloads where 2× is enough and you want flag-flip operations, FP8 is simpler. In 2026, vLLM ships both — the choice depends on how aggressive you need to be.

**FP8 KV vs DeepSeek-V4's mixed-precision policy.** [V4's policy](/issues/07-sparse-lab/15-mixed-precision-kv/) — BF16 RoPE / FP8 KV body / FP4 indexer — is the *fine-grained* version of what the skip-SW flag is doing coarsely. Both recognize that the cache has heterogeneous sub-regions that benefit from different precisions. V4 is a model-specific design choice. FP8 KV in vLLM is a model-agnostic flag. Both are correct at their level of abstraction; they will increasingly converge as inference engines get smarter.

**FP8 KV vs KV pruning (Issue 6).** [KV pruning from Issue 6](/issues/06-eviction-notice/) is a different kind of compression: instead of storing the same tokens at lower precision, you *delete* tokens entirely. The two are complementary. A typical 2026 production stack will quantize the KV cache to FP8 *and* prune low-importance tokens with KVzap *and* sparse-attend over the rest with DSA-style indexers. Compression is multiplicative across layers of the stack.

## What's Next

Three open frontiers visible from where the issue ends.

**1. FP4 KV-cache.** The natural next step on the same arc. FP4 is the format Blackwell shipped natively in 2025 ([Hardware Horizon](/issues/03-sixteen-numbers/12-hardware-horizon/)), and KV-cache is the obvious target for it. The same questions this issue chased through FP8 — accumulator precision, calibration, hybrid policies — will need to be re-answered for FP4. The DeepSeek-V4 indexer's FP4 QK path is the first widely-deployed example; vLLM doesn't yet have a flag for FP4 KV but probably will by 2027.

**2. Hardware-native compensated accumulation.** The two-level accumulation fix is a software workaround for a hardware behavior. Future tensor core generations could expose compensated accumulation as a native instruction — a Kahan-style add-with-correction — and the software workaround would become unnecessary. Blackwell already partially addresses this by redesigning the accumulator. Whether NVIDIA exposes it as a first-class instruction or keeps it internal is an open question.

**3. Heterogeneous-precision KV policies.** The arc from "one precision for the whole cache" to "different precisions per layer" is partway done (skip-SW). The next step is per-layer bit allocation (KVTuner-style), then per-head, then per-channel, and eventually per-region (V4-style). Each step adds bookkeeping complexity but recovers accuracy. vLLM's roadmap explicitly mentions this direction.

There is also a meta-question that this issue keeps returning to but doesn't resolve: **how much of the inference stack is still validated only at short context?** The cold open's bug was hidden for three years because nobody had run a long-context benchmark on the FP8 path. Similar bugs may exist elsewhere — in older attention backends, in the long-context optimizations themselves, in the interaction between PagedAttention and FP8, in speculative decoding's interactions with quantized caches. The pattern is: **build long-context benchmarks first, run them everywhere, then trust the optimizations**. The team's experience over the past three months is the cautionary tale.

## One Lesson From The Whole Story

This issue is, in a structural sense, the same story as [Issue 6's KVzap journey](/issues/06-eviction-notice/) and [Issue 7's DeepSeek-V4 arc](/issues/07-sparse-lab/). Each one is about an inference optimization that *appeared* to work for years and was eventually discovered to be incomplete in some specific way, and each one is about the careful engineering work required to make the optimization actually-work, not just plausibly-work.

The pattern: a clever idea ships. The clever idea is partially right and partially wrong. Production users get the partially-right benefit and (often quietly) eat the partially-wrong cost. Eventually somebody runs a benchmark that surfaces the wrong half. The wrong half gets fixed, sometimes by importing an algorithm from a different decade. The flag goes from "available" to "default" once the fixes have shipped.

FP8 KV-cache is the 2026 instance of this pattern. The clever idea was 2023's Hopper FP8 silicon plus FA3's fused FP8 attention path. The partially-wrong part was the FP32 accumulator's long-contraction precision loss. The fix was Kahan's 1965 compensated summation, tiled for tensor cores via SageAttention2's two-level accumulation. The flag is now a default.

We will see the same pattern again with FP4 KV. We will see it with whatever comes after. The lesson, repeated for the *n*th time: **a flag exists is not a flag is correct**. The discipline of validating every optimization at the workload sizes it will actually be deployed at — long context, real prompts, real concurrency — is what separates production engineering from research. The vLLM FP8 KV journey is the recent canonical example of doing that discipline right.

## Where To Read Further

The vLLM blog post that is the source for this issue: [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://blog.vllm.ai/2026/04/22/fp8-kvcache.html). Read it for the full benchmark tables, the kernel PR list, the calibration recipes, and the official maintainer recommendations.

The two papers that this issue most directly leans on:

- [**SageAttention2** (Zhang et al., 2024)](https://arxiv.org/abs/2411.10958) — the two-level accumulation algorithm the team adopted.
- [**DeepSeek-V3 Technical Report**, especially Section 3.3](https://arxiv.org/abs/2412.19437) — the prior independent discovery of the FP32-accumulator precision loss during training.

The supporting infrastructure:

- [**vllm-project/flash-attention**](https://github.com/vllm-project/flash-attention) — the FA3 fork where the kernel-side fixes landed (PRs #91, #96, #104, #122, #125).
- [**vllm-project/vllm**](https://github.com/vllm-project/vllm) — the engine-side wiring (PRs #24914, #30141, #30833, #33695).
- [**vllm-project/LLM-Compressor**](https://github.com/vllm-project/llm-compressor) — the calibration tooling for the cases that need it.

And the prior issues that this one builds on:

- [Issue 3 — Sixteen Numbers Walk Into A GPU](/issues/03-sixteen-numbers/) for the full quantization landscape, the [KV method family tree](/issues/03-sixteen-numbers/15-kv-method-family/), the [number-format primer](/issues/03-sixteen-numbers/02-numbers-in-boxes/), and the [hardware horizon](/issues/03-sixteen-numbers/12-hardware-horizon/) that places Hopper and Blackwell in context.
- [Issue 6 — The Eviction Notice](/issues/06-eviction-notice/) for the complementary KV-pruning story and the [bandwidth wall primer](/issues/06-eviction-notice/11-bandwidth-wall/).
- [Issue 7 — The Sparse Lab](/issues/07-sparse-lab/) for DeepSeek-V4's mixed-precision KV policy and the broader picture of long-context inference economics.
- [Issue 8 — Anatomy of a Token](/issues/08-anatomy-of-a-token/) for the [FlashAttention primer](/issues/08-anatomy-of-a-token/06-flash-attention/), the [roofline model](/issues/08-anatomy-of-a-token/04-roofline/), and the [prefill / decode split](/issues/08-anatomy-of-a-token/07-prefill-vs-decode/) that frame the bandwidth-bound regime FP8 KV-cache lives in.

## The Five-Event Timeline

To close the issue, the timeline that frames it:

{{< timeline name="fp8-kvcache-history" >}}

Five turning points across three years. **Hopper FP8** ships the silicon that makes FP8 attention possible. **FlashAttention-3** ships the kernel that uses it. **DeepSeek-V3's footnote** documents the precision bug from the training side. **The AWS / Red Hat investigation** discovers it from the inference side. **The two-level accumulation fix** closes the loop. The whole arc is a single hardware-software co-evolution cycle: silicon arrives with a quirk, software bumps into the quirk at unexpected workload sizes, software designs a workaround using a 60-year-old algorithm, the workaround becomes the default, the next generation of silicon quietly fixes the underlying quirk.

This is how this kind of engineering works. Slowly, in production, with a flag that turns out to mean something more nuanced than the documentation said. Three months of debugging to make a three-year-old flag actually do what it claimed.

The flag is now a default. The KV cache is half the size. The decode is 15% faster. Llama-3.1-8B at 128k context finds the needle 89% of the time, almost as well as it ever did. The bargain is real.

That closes the issue.

---

**Thanks for reading.** If you found this useful, the [back catalog of issues](/issues/) is one click away. If you want to draft your own issue, the [contribute guide](/docs/contribute/) tells you how the recipe works. If you want to know how this very issue was authored — the prompts, the source vLLM blog post, the iteration — that's in the repository's `prompts/` directory, kept in-repo for posterity.
