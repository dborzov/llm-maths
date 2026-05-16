---
title: "Two Phases, Two Personalities"
description: "Prefill is compute-bound; decode is bandwidth-bound. They have opposite hardware personalities, opposite bottlenecks, and opposite optimal batch sizes — and every major inference engine decision flows from that asymmetry."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T12:00:00-04:00
issue: 8
weight: 70
techKind: mainline
techNode: prefill-vs-decode
header: default.png
---


>
> microGPT issue 05 already has [Prefill vs Decode](/llm-maths/issues/05-microgpt-unfolded/14-prefill-decode/) as a transformer-internals chapter. This article is the **systems-engineering re-framing** of the same split: two workloads with opposite arithmetic intensity, fighting over the same hardware, with different latency metrics.

## Anchor and Frame

- **Open with a profile trace.** One GPU, one Llama-70B, one user asking a 4K-token question. Show the trace: 600 ms of dense prefill, then 100 decode steps of ~25 ms each, streaming. Two visually different bars on the same plot.
- **Anchor metric pair.** TTFT (time to first token) — dominated by prefill. ITL (inter-token latency) / TPOT (time per output token) — dominated by decode. **These two metrics are in tension. Optimizing one usually hurts the other.**

## Outline

### What Prefill Does

- Takes the prompt of length $T$, runs one forward pass over all $T$ tokens in parallel.
- Builds the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} for those $T$ positions in one shot.
- Arithmetic intensity ≈ $D$ (hidden dim). Compute-bound. The tensor cores are saturated.

### What Decode Does

- Takes the last token of the running sequence, runs one forward pass over **one** token.
- Appends one $(K_t, V_t)$ pair per layer to the cache.
- Arithmetic intensity ≈ 1. Memory-bound. The tensor cores idle while HBM streams.

### Why They Are In Tension

- Long prefills hog the GPU. A user mid-decode waits for the prefill to finish. **Head-of-line blocking.**
- Batching helps decode (more work per HBM byte) but doesn't help prefill (already compute-bound).
- Foreshadow: chunked prefill and continuous batching are both attempts to resolve this tension.

### The Latency Budget

- Worked example. Llama-70B on H100 TP=4. Prefill at $T = 4096$: ~340 ms. Per-token decode: ~24 ms. So generating 200 tokens takes 4.8 s, of which ~7% is prefill.
- For long-context queries (32K prompt → 200 tokens out), the ratio flips: 3 s prefill, 4.8 s decode.

### The Two Metrics, Drawn

- Pyplot: two-axis plot of TTFT vs throughput as you crank batch size. TTFT rises (long prefills compete), throughput rises (more decodes per step). The SLA is a constraint line on this plot.

### What Comes Next

- The scheduler must reconcile these. The KV cache must serve both workloads. Continuous batching mixes them.

## Connections

- ← [The Roofline](../04-roofline/) — the two workloads sit on opposite sides of the ridge.
- ← [Attention in SRAM](../06-flash-attention/) — Flash is the prefill kernel.
- → [The Conveyor Belt](../08-continuous-batching/), [The KV Cache Is a Heap](../09-kv-fragmentation/), [Slicing the Prefill](../13-chunked-prefill/).
- ← Cross-link to microGPT [Prefill vs Decode](/llm-maths/issues/05-microgpt-unfolded/14-prefill-decode/).

## What To Remember

1. **Prefill and decode are not stages of one workload. They are two different workloads** that happen to share a GPU and a KV cache.
2. **Two metrics, two regimes.** TTFT is dominated by prefill (compute-bound). ITL is dominated by decode (memory-bound). Every optimization in the rest of the issue is judged on its effect on this pair.
3. **The prefill/decode split was the field's organizing principle for three years** — and then vLLM V1 ([→ ch.14](../14-scheduler/)) quietly merged them back together. Hold that tension; the merging is the payoff later.

**Continue to → [The Conveyor Belt](../08-continuous-batching/)** — the 2022 paper that made it possible to serve more than one user at a time without throwing away most of the GPU.

