---
title: "The Eviction Notice"
description: "KV cache pruning for LLMs — from Heavy Hitters to KVzap, and why the first method to actually ship took three years to arrive."
issue: 6
layout: issue-cover
theme: cream
math: false
header: 06-eviction-notice-cover.webp
date: 2026-05-14T23:01:06-04:00
---

## The Mystery

**January 2026.** Simon Jégou and Matthijs Jebleck, two researchers at NVIDIA, submit a paper with a deceptively simple claim: they have written a 14-line function that tells a language model which of its past tokens it can safely forget. The function runs in under 1.1% of total inference compute. It achieves 2.7–3.5× compression of the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} — the data structure that, at 128,000 tokens of context, consumes 335 gigabytes of GPU memory for Llama-65B. And it works during both prefill *and* decoding, without modifying a single line of the attention kernel.

The KV cache problem has been a known crisis since 2023. The math is unforgiving: cache size scales as $(2, L, H, T, D)$ — two tensors, $L$ layers, $H$ heads, $T$ token positions, $D$ head dimension. Every dimension except $T$ is a property of the model architecture, baked in at training time. But $T$ — the sequence length — grows without bound as users push context windows from 4K to 32K to 128K to one million tokens. The cache that ate the server: one axis, unbounded.

By late 2025, the `NVIDIA/kvpress` repository listed more than twenty KV pruning methods. The `Awesome-KV-Cache-Compression` list maintained by the community had dozens more. Every method came with ablation tables, arXiv papers, and a claimed compression ratio. And yet: zero of them were integrated in vLLM, zero in SGLang, zero in TensorRT-LLM — the three inference engines that actually run production traffic. The engineers who evaluated these methods kept declining. Not because the research was bad. Because production inference has four requirements — fast, phase-agnostic, optimization-friendly, faithful — and every method before January 2026 failed at least one.

Jégou and Jebleck's paper passes all four. The key observation: a tiny surrogate model trained on the {{< wiki "residual-stream" >}}hidden states{{< /wiki >}} $h_t$ can predict, with $R^2 = 0.67$–$0.77$, which tokens will matter for future computation — and hidden states are produced by the forward pass anyway, so the surrogate runs on compute cycles that were already going to waste. During autoregressive decoding, a GPU running a 70B model uses about 0.3% of its advertised FLOP throughput. The bottleneck is memory bandwidth: loading the KV cache from HBM for every generated token. KVzap's surrogate adds arithmetic to idle compute units while the HBM bus is stalled. The extra FLOPs cost nothing.

## How To Read This Issue

Start with the [cold open](01-cold-open/) — it sets the stakes and introduces the moment the field realized the T-axis was a different kind of problem. After that, the **tech tree** below is the table of contents. Each node is an article; arrows show which articles build on which.

{{< techtree name="issue06" >}}

The five **pink mainline chapters** tell the story in order: the T-axis memory crisis, the Heavy Hitter Oracle, the pretext trick, the surrogate learning discovery, and the final KVzap design. The **yellow boss chapter** — *KVzap: The Final Zap* — is the destination, where every thread converges and the method is dissected on the KVpress Leaderboard. The **cream primer chapters** give depth on demand: read them when a mainline chapter refers to them, or read them all first if you prefer foundations before narrative. There is no wrong order.

If you are in a hurry: the mainline (pink nodes) is a four-hour read. If you want the full treatment — attention sparsity from first principles, the log-space threshold intuition, the von Neumann bottleneck at GPU scale — read everything in numerical order.

## What You Will Walk Away With

By the end of this issue you should be able to answer, with confidence and the napkin math to back it up:

- Why Llama-65B at 128K context needs **335 GB** just for its KV cache — and the arithmetic: $2 \times 80 \times 64 \times 131072 \times 128 \times 2$ bytes.
- Why the **T-axis was the last axis nobody cracked** — every other axis of the KV cache shape is fixed at model design time, but T grows with the user's prompt.
- What **"heavy hitters"** are: why, empirically, 80% of {{< wiki "attention" >}}attention{{< /wiki >}} mass goes to 20% of tokens — and why the H₂O paper from UT Austin in 2023 was the first to exploit this systematically.
- How **KVzip's copy-and-paste pretext task** scores token importance by asking the model to predict each token from context — and why the 2× prefill cost made it impossible to ship despite near-perfect scores.
- Why **attention weight ≠ contribution**: the KVzip+ normalization insight, the $\|W_O v_i\| / \|h_j\|$ factor, and why a token that gets attended to can still contribute nothing to the output.
- That **hidden states predict token importance with $R^2 = 0.67$–$0.77$** — and why this is surprising: the residual stream at position $t$ has not yet seen the future queries that will determine whether token $t$ matters.
- The difference between **threshold-based and top-k eviction** — and why the threshold is strictly better: the same $\tau = -4$ gives 74% compression on RULER (synthetic) and 66% on LongBench (real-world), not through tuning, but because the score distribution adapts to information density automatically.
- Why **GPU decode is memory-bandwidth-bound** at ~1 FLOP/byte — ninety-three times below the H100 ridge point — and how this makes KVzap's surrogate arithmetic genuinely free.
- The full **KV pruning family tree from H₂O (2023) to KVzap (2026)**, with the four-criteria filter (fast / phase-agnostic / optimization-friendly / faithful) that explains every production rejection.

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
