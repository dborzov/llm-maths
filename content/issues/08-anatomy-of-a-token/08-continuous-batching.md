---
title: "The Conveyor Belt"
description: "Static batching wastes most of the GPU whenever sequences finish at different lengths — the GPU idles waiting for the longest sequence. Orca's iteration-level scheduling (OSDI 2022) fixed this: swap finished requests out and new ones in at every decode step, not at batch boundaries."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T12:30:00-04:00
issue: 8
weight: 80
techKind: mainline
techNode: continuous-batching
header: default.png
---


## Anchor and Frame

- **Cold open.** Imagine a 2022 chatbot that batches 32 requests into one forward pass. The shortest reply is 12 tokens; the longest is 480. The batch runs at the speed of the slowest. When the 12-token reply finishes, its slot sits idle for 468 more steps. **The GPU is 90% padding.**
- **History.** Gyeong-In Yu et al. at Seoul National + FriendliAI, OSDI 2022. "Orca: A Distributed Serving System for Transformer-Based Generative Models." Insight: schedule at the *iteration* level, not the *request* level.

## Outline

### Static Batching, Drawn

- Diagram (or pyplot): 32 rows, each a sequence. Color the cells: prefill, decode, padding. The padding triangle is huge.

### Dynamic Batching

- The intermediate idea: form a new batch every N ms or M requests, whichever comes first. Reduces queue time but does nothing about the padding triangle once a batch is running.

### Iteration-Level Scheduling (Orca)

- The trick: **rebuild the batch after every forward pass**. Finished sequences are removed; new requests can join. The batch is fluid, not fixed.
- Diagram: same 32 rows but now each row ends when its sequence ends, and a new row starts in its place mid-figure.

### Why It Took A Paper

- The transformer kernels of 2021–22 assumed dense rectangular batches. Orca contributed *selective batching*: a per-operation choice of which subset of the batch participates. Attention is sliced per-sequence (different KV lengths); linear layers are batched flat.
- Foreshadow PagedAttention: the KV layout problem is what Orca didn't solve — it punted to padding/over-allocation. PagedAttention finishes the job.

### Throughput-Latency Trade Studies

- Pyplot: throughput vs concurrent users for static batching, dynamic batching, continuous batching. Curves cross at different points; continuous batching wins by 5–20× at typical loads.

### The New Bottleneck

- Once you can keep the GPU fed, the next bottleneck is memory: how do you allocate KV cache for a batch where sequences come and go every step?
- Naive answer: pre-allocate max-length blocks per slot. Wastes 60–80% of KV memory. This is the problem [→ ch.9](../09-kv-fragmentation/) opens with.

## Connections

- ← [Two Phases, Two Personalities](../07-prefill-vs-decode/) — Orca's insight is to mix prefill and decode in one batch.
- → [The KV Cache Is a Heap](../09-kv-fragmentation/) — the new bottleneck Orca exposed.
- → [The Token Budget](../14-scheduler/) — vLLM V1 generalizes Orca's iteration-level scheduling.

## What To Remember

1. **Iteration-level scheduling.** Rebuild the batch every forward pass. Finished requests leave; new ones join. The GPU never waits for the slowest member of a fixed batch.
2. **Selective batching is the kernel-level enabler.** Attention runs per-sequence; everything else runs flat-batched. This is why naive PyTorch loops cannot do this — you need a custom attention dispatch.
3. **Continuous batching turned the next bottleneck into memory.** Once you fix scheduling, the GPU is happy but the allocator is now your problem. That's the rest of the issue.

**Continue to → [The KV Cache Is a Heap](../09-kv-fragmentation/)** — the allocator hell that Orca exposed and PagedAttention finally cleaned up.

