---
title: "The Full Anatomy of a Token"
description: "The cold-open trace, re-annotated end to end: the same packet, the same H200 box, the same user — but every layer now named, every component explained, every latency budget itemized. Everything the issue introduced, assembled into one complete, labelled picture."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T17:30:00-04:00
issue: 8
weight: 180
techKind: boss
techNode: full-anatomy
header: default.png
---


## Frame

- Return to chapter 1. Anya, 4:12 PM, "Why did the Roman Empire fall?". One TCP packet hits the load balancer. The next 18 paragraphs are the next 5 seconds.
- Goal: every link in the previous 17 chapters lights up. The reader sees the whole tree and *feels* it as one system.

## The Annotated Timeline

For each step, give: wall-clock budget, component, chapter link, what physically happens, one sentence on why.

### t = 0.0 ms — TCP arrives

- HTTPS terminated at the edge, request hops to vLLM's OpenAI-compatible HTTP server.

### t = 0.3 ms — Tokenizer (AsyncLLM)

- BPE tokenizer turns 47 bytes → 12 token ids.
- IPC to EngineCore over a ZMQ socket (multi-process to dodge GIL contention).

### t = 0.7 ms — Engine ingests request

- A new `Request` object joins the `waiting` queue.

### t = 1.0 ms — Scheduler step ([ch.14](../14-scheduler/))

- Token budget for this step: 8192. Already running: 53 requests, allocating ~270 of those tokens.
- Anya's request is admitted. The block manager ([ch.11](../11-block-manager/)) walks her 12 tokens through the prefix cache hash table ([ch.12](../12-prefix-caching/)). 8 tokens hit the cached system prompt; 4 fresh tokens need prefill.
- Plan: `{anya: 4, others: ...}`.

### t = 1.3 ms — Model runner assembles inputs

- For each TP rank, the runner builds the input tensor for the captured CUDA-graph bucket ([ch.5](../05-cuda-graphs/)) one size up from "active batch + Anya".

### t = 1.5 ms — Forward pass starts

- CUDA graph replay. The kernel sequence for 80 transformer layers is queued in one driver call.
- Inside each layer: a TP-sharded QKV matmul, then **PagedFlashAttention** ([ch.10](../10-paged-attention/), [ch.6](../06-flash-attention/)) reading K/V through Anya's block table, then output projection (all-reduce on the TP group, [ch.16](../16-tp-pp/)), then MLP (column-parallel up, row-parallel down).
- Anya's tokens are part of a *mixed* batch: some prefill chunks ([ch.13](../13-chunked-prefill/)), most decodes ([ch.7](../07-prefill-vs-decode/)).

### t = 6 ms — Layer 1 KV streams to the decode cluster

- (If running disagg.) NIXL one-sided RDMA write from prefill machine's HBM → decode machine's HBM, block-aligned ([ch.17](../17-disagg-pd/)).

### t = 280 ms — First token sampled

- The logits for Anya's last prefill position land on EngineCore. Top-p sampling, the chosen id is detokenized into the first reply token (probably *"The"*).
- Streamed back: AsyncLLM → HTTP/2 response chunk → load balancer → Anya's phone. **TTFT = 280 ms.**

### t = 280..310 ms — First decode step ([ch.8](../08-continuous-batching/))

- Anya joins the *decode set*. The scheduler emits `{anya: 1, …}`.
- A draft model emits 4 candidate tokens ([ch.15](../15-speculative-decoding/)). The target verifies all 4 in one memory pass. Acceptance: 3. Net tokens this step: 3.

### t = 310..6000 ms — The stream

- 80 more decode steps. Average ITL: 28 ms (with spec decode amortization). Sometimes a step does 1 token, sometimes 5.
- Mid-stream: Anya's KV cache crosses a block boundary; the block manager allocates a fresh block from the free queue.
- Once: the free queue runs low. A long-idle request is preempted; its blocks go back to the pool.

### t = 6.1 s — `<|eot|>` sampled

- Sampler emits the end-of-turn token. Engine marks request done. Block manager decrements refcounts on Anya's blocks; the prefix-cache portion stays warm (still ref'd by the system-prompt cache entry).
- Final response chunk goes out. Connection stays open for the next turn (the prefix cache will hit again).

## The Diagram

- A full-width, top-to-bottom illustration: time axis on the left (ms), components as horizontal swimlanes, arrows showing data hand-offs. Mark every chapter link as a colored badge on the relevant arrow. This is the issue's signature illustration.

## Counting The Wins

- Without paged attention: KV waste would force batch ≤ ~120. With it: batch = 850 in the same memory.
- Without prefix caching: Anya's prefill would be 1,500 tokens instead of 4. **375× saving on prefill compute** for this turn.
- Without continuous batching + V1 scheduler: head-of-line blocking on a 64K prefill from another user would inject seconds into Anya's ITL.
- Without chunked prefill: the same.
- Without spec decode: ITL ~45 ms instead of 28 ms.
- Without CUDA graphs: per-step CPU overhead ~2 ms × 200 steps = 400 ms of throttled latency.
- Without disagg P/D: prefill compute would compete for Anya's decode bandwidth.

**Roughly: one TCP packet → six layers of OS-style abstractions and three years of distributed-systems research → a 280 ms TTFT and a fluent paragraph.**

## Where The Story Goes Next

- **2026 onward.** Multi-tier KV cache fabrics (Mooncake, LMCache). Disaggregated *experts* (MoE inference across machines). Long-context inference with sparse attention (forward-link → [Issue 07: The Sparse Lab](/llm-maths/issues/07-sparse-lab/)). Quantization down to FP4 weights and KV (→ [Issue 03: Sixteen Numbers](/llm-maths/issues/03-sixteen-numbers/)). On-device inference with the same architecture pattern reduced to one chip.
- Every one of those threads pulls on the same architectural primitives: paged KV, token-budget scheduling, FlashAttention tiling, RDMA-fabric KV transfer. The vocabulary is settling.

## What To Remember

1. **Modern LLM inference is an operating system.** Virtual memory (paged KV), scheduler (V1 token budget), processes (requests), IPC (engine ↔ runner), networking (RDMA KV transfer). The CS-101 vocabulary is the right vocabulary.
2. **Every major optimization is a forced move from the roofline.** Decode is bandwidth-bound; spec decode, paged attention, FlashAttention, KV quantization, prefix caching all attack the bandwidth term.
3. **The stack is young.** Half of these techniques didn't exist three years ago. The architectural pattern is stabilizing — but the next chapter of the story (multi-tier KV, MoE serving, sparse long-context) is already being written.

*This is the boss. There is no next chapter — the issue is done. If you want where the story goes next, the index page lists the issues that pick up each thread.*

