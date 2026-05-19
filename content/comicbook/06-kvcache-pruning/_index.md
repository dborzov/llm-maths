---
title: "KV Cache Pruning: from Heavy Hitters to shipping KVzap"
description: "KV cache pruning for LLMs — from the Heavy Hitter Oracle (2023) to KVzap (2026), and why the first method to actually ship in production took three years to arrive."
issue: 6
layout: issue-cover
theme: cream
math: true
header: 06-kvcache-pruning-cover.webp
date: 2026-05-14T23:01:06-04:00
---

## The Mystery

**January 2026.** Simon Jégou and Matthijs Jebleck, two researchers at NVIDIA, submit a paper with a deceptively simple claim: they have written a 14-line function that tells a language model which of its past tokens it can safely forget. The function runs in under 1.1% of total inference compute. It achieves 2.7–3.5× compression of the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} — the data structure that, at 128,000 tokens of context, consumes 335 gigabytes of GPU memory for Llama-65B.

The KV cache problem has been a known crisis since 2023. The math is unforgiving: cache size scales as $(2,\, L,\, H,\, T,\, D)$ — two tensors, $L$ layers, $H$ heads, $T$ token positions, $D$ head dimension. Every dimension except $T$ is baked in at training time. But $T$ — the sequence length — grows without bound as users push context windows from 4K to 32K to 128K to one million tokens. One axis, unbounded.

{{% pullquote type="counter-intuitive" %}}
By late 2025, the community had catalogued over twenty KV pruning methods. Zero were integrated in vLLM, SGLang, or TensorRT-LLM — the three inference engines that run production traffic.
{{% /pullquote %}}

{{% callout type="theorem" title="The Four Production Requirements" %}}
Every KV pruning method before January 2026 failed at least one:

1. **Fast** — adds less than ~1% to total inference latency
2. **Phase-agnostic** — works identically during prefill *and* decoding
3. **Optimization-friendly** — composable with flash attention, tensor parallelism, PagedAttention
4. **Faithful** — preserves actual output quality, not just reported benchmark scores

KVzap passes all four.
{{% /callout %}}

The key observation: a tiny surrogate trained on the {{< wiki "residual-stream" >}}hidden states{{< /wiki >}} $h_t$ can predict, with $R^2 = 0.67$–$0.77$, which tokens will matter for future computation. Hidden states are produced by the forward pass anyway — the surrogate runs on arithmetic cycles that were already going to waste. During autoregressive decoding, a GPU running a 70B model uses about 0.3% of its advertised FLOP throughput. The bottleneck is memory bandwidth: loading the KV cache from HBM for every generated token. KVzap's surrogate adds arithmetic to idle compute units while the HBM bus is stalled. The extra FLOPs cost nothing.

## Three Years, Five Methods

{{< timeline name="kv-pruning-methods" >}}

## How To Read This Issue

Start with the [cold open](01-cold-open/) — it sets the stakes and introduces the moment the field realized the T-axis was a different kind of problem. After that, the **tech tree** below is the table of contents. Each node is an article; arrows show which articles build on which.

{{< techtree name="issue06" >}}

The four **pink mainline chapters** tell the story in order: the T-axis memory crisis, the Heavy Hitter Oracle, the pretext trick, and the surrogate learning discovery. The **yellow boss chapter** — *KVzap: The Final Zap* — is the destination, where every thread converges and the method is dissected on the KVpress Leaderboard. The **cream primer chapters** give depth on demand: read them when a mainline chapter refers to them, or read all of them first if you prefer foundations before narrative.

If you are in a hurry: the mainline (pink nodes) is a four-hour read. If you want the full treatment — attention sparsity from first principles, the log-space threshold intuition, the von Neumann bottleneck at GPU scale — read everything in numerical order.

## What You Will Walk Away With

By the end of this issue you should be able to answer, with confidence and the napkin math to back it up:

### The T-axis problem

- Why Llama-65B at 128K context needs **335 GB** just for its KV cache — and the arithmetic: $2 \times 80 \times 64 \times 131072 \times 128 \times 2$ bytes.
- Why the **T-axis was the last axis nobody cracked** — every other dimension of the cache shape ($L$, $H$, $D$) is fixed at model design time, but $T$ grows with the user's prompt.
- Why **GPU decode is memory-bandwidth-bound** at ~1 FLOP/byte — ninety-three times below the H100 ridge point — and how this makes KV cache compression especially valuable.

### The pruning family tree

- What **"heavy hitters"** are: why 80% of {{< wiki "attention" >}}attention{{< /wiki >}} mass goes to 20% of tokens — and why the H₂O paper from UT Austin in 2023 was the first to exploit this systematically.
- How **KVzip's copy-and-paste pretext task** scores token importance by asking the model to predict each token from context — and why the 2× prefill cost made it unshippable despite near-perfect accuracy.
- Why **attention weight ≠ contribution**: the KVzip+ normalization insight, the $\|W_O v_i\| / \|h_j\|$ factor, and why a token that gets heavily attended to can still contribute nothing to the residual update.
- The full **KV pruning family tree from H₂O (2023) to KVzap (2026)**, with the four-criteria filter that explains every production rejection.

### The KVzap solution

- That **hidden states predict token importance with $R^2 = 0.67$–$0.77$** — and why this is surprising: the {{< wiki "residual-stream" >}}residual stream{{< /wiki >}} at position $t$ has not yet seen the future queries that will determine whether token $t$ matters.
- The difference between **threshold-based and top-k eviction** — and why the threshold is strictly better: $\tau = -4$ gives 74% compression on RULER (synthetic) and 66% on LongBench (real-world) without any per-dataset tuning, because the score distribution adapts to information density automatically.

---

{{% callout type="tangent" title="Two strategies, one observation" %}}
This issue is about **deleting** tokens from the cache. The complement strategy — keeping every token in the cache but **not attending to most of them** — is the subject of [Issue 7: The Sparse Lab](/comicbook/07-deepseek-attn/), which traces DeepSeek's MLA → DSA → CSA/HCA arc from May 2024 through V4 in 2026.

Both strategies exploit the same empirical fact: 5-10% of past tokens absorb 90% of attention mass (see [Where Attention Is Sparse](/comicbook/07-deepseek-attn/16-empirical-sparsity/)). Issue 6 turns this into eviction; Issue 7 turns it into selection. They compose — production systems quantize, prune, *and* sparse-attend.
{{% /callout %}}

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
