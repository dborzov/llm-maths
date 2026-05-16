---
title: "The KV Cache Is A Heap"
description: "The KV cache grows one token at a time, to unpredictable lengths, and must be contiguous in naive implementations. The result is a fragmented heap that wastes 60–80 percent of HBM in the worst case. This is the problem PagedAttention was built to solve."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T13:00:00-04:00
issue: 8
weight: 90
techKind: mainline
techNode: kv-fragmentation
header: 09-kv-fragmentation.webp
---


## Anchor and Frame

- **Anchor case.** A batch of 32 sequences with max-length 4096. Default allocation: a contiguous $(32, 4096, L, H, D)$ tensor. The actual sequences range from 100 to 3,800 tokens. **Average utilization: 38%.**
- **The PagedAttention paper's measurement.** Kwon et al. report that production serving systems before vLLM achieved 20–40% effective KV memory utilization — i.e. 60–80% of HBM was holding padding.

## Outline

### The Allocator's Three Bad Choices

Every classical KV allocator has to pick one of these:

1. **Pre-allocate max-length** per slot. Worst-case bounded; average-case ruinous. (HF Transformers default.)
2. **Pre-allocate current length** in contiguous tensors, copy-and-grow when full. Lots of memory traffic; doesn't compose with iteration-level scheduling.
3. **Dynamic allocation in fragments.** Solves utilization but breaks every CUDA kernel that assumes contiguous keys/values.

### Why Cache Allocation is Hard

- Variable lengths (prompts: 50 → 50,000 tokens; outputs: 5 → 4,000 tokens).
- Unpredictable lifetime (the model decides when to emit `<|eot|>`).
- Massive size (335 GB at 128K context for Llama-65B — see [Issue 06](/llm-maths/issues/06-eviction-notice/02-kv-crisis/) for the napkin math).
- Hot working set: attention must visit *every* KV position every decode step.

### Internal vs External Fragmentation

- Define both, with the classical OS analogies (Knuth, *Art of Computer Programming* Vol. 1, §2.5).
- **Internal:** slots over-reserved; unused but allocated.
- **External:** free space exists in non-contiguous chunks; can't allocate a contiguous block that fits.
- KV allocation suffers both, simultaneously, at scale.

### A Picture

- Pyplot: HBM memory map for a 32-slot batch over 50 generation steps. Color cells "live KV" / "reserved padding" / "free". Watch the utilization curve sag.

### Why You Can't Just Use `cudaMalloc` In A Loop

- CUDA allocator latency is microseconds per call. Decode steps are tens of milliseconds. Calling the allocator on every step burns the latency budget.
- And contiguity matters for the attention kernel — until it doesn't.

### Connections Backwards And Forwards

- The OS people *solved this exact problem* sixty years ago. The next chapter is the field rediscovering that.

## Connections

- ← [The Conveyor Belt](../08-continuous-batching/) — Orca exposed this problem.
- ← [The Pyramid of Speed](../03-memory-hierarchy/) — bandwidth + capacity make HBM precious, which makes utilization existential.
- → [Borrowing from 1965](../10-paged-attention/) — the answer.
- Cross-link to {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} (microGPT ch.13) for the shape definition.

## What To Remember

1. **The KV cache is a heap, not a stack.** Sequences are born and die at unpredictable times; their sizes are unknown in advance. This is not a CUDA-friendly access pattern.
2. **Naive contiguous allocation wastes 60–80% of HBM** in production serving workloads. That is more memory than every other inference optimization combined could ever save.
3. **The OS literature solved this in 1965.** Virtual memory, page tables, demand paging. The trick is mapping it onto an attention kernel without losing the contiguity that GPUs love.

**Continue to → [Borrowing from 1965](../10-paged-attention/)** — the moment Kwon, Li, Zhuang, and Sheng saw the page-table analogy and built vLLM around it.

