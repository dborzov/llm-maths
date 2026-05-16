---
title: "Borrowing From 1965"
description: ""
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T05:05:21-04:00
issue: 8
weight: 100
techKind: mainline
techNode: paged-attention
header: default.png
---

> **Scaffold note.** The marquee chapter. The reader should leave this chapter understanding *why the operating-systems analogy is exact*, *what the block table physically is*, and *what changes inside the attention kernel*. This is the central trick of the entire field.

## Anchor and Frame

- **History bridge.** 1965, MIT. Multics. Fernando Corbató introduces *virtual memory*: per-process page tables map logical pages onto physical frames. Programs see contiguous address space; the OS scatters their pages across RAM. The problem being solved: fragmentation, multi-tenancy, on-demand growth.
- **2023, UC Berkeley.** Woosuk Kwon and team realize they are facing exactly that problem with KV caches — and that the answer was on the bookshelf. Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.

## Outline

### The Analogy, Drawn

| OS / virtual memory | LLM serving |
|---|---|
| Process | In-flight request (sequence) |
| Logical address | Token position in sequence |
| Physical page (4 KB) | KV block (16 tokens × K-or-V × dim × dtype) |
| Page table | Block table (per-request) |
| Free frame pool | Free block pool |
| Demand paging | Allocate-on-append (per decode step) |
| Copy-on-write (fork) | Beam-search / parallel sampling branching |
| Shared library page | Prefix-cached shared blocks |
| Swap to disk | KV offload to CPU/SSD/peer-GPU |

Make this table the visual centerpiece.

### Block Anatomy

- A *block* is the unit of paging. vLLM defaults: 16 tokens per block. Per layer per head, a block stores 16 K-vectors and 16 V-vectors. For Llama-70B-FP8 with $H = 8$ KV-heads and $D = 128$: one block is $80 \text{ layers} \times 8 \times 16 \times 128 \times 2 \text{ (K+V)} \times 1 \text{ byte} \approx$ 327 KB. Roughly the size of a Linux *huge page*.
- Why 16? It's a tile size that fits one warp's worth of work cleanly and amortizes the indirection cost.

### The Block Table

- A per-request array of physical block ids: `block_table[i] = phys_id`. To attend to logical position $t$, walk to logical block $\lfloor t/16 \rfloor$, look up the physical id, jump to that block in the global KV pool.
- Show as both a data structure and a memory map.

### What Changes Inside The Attention Kernel

- Recall the Flash inner loop (→ [Attention in SRAM](../06-flash-attention/)): outer loop over Q-tiles, inner loop over KV-tiles.
- Paged-Flash modifies the inner loop to *load each KV-tile through the block table*. One extra layer of indirection at tile boundaries.
- Cost: tiny — block boundaries are infrequent and the indirection lookup pipelines with the load.
- Benefit: 60–80% of wasted HBM is recovered. Effective batch size jumps 2–4×.

### Three Things Paging Enables For Free

1. **Prefix caching** — different requests share a prefix's physical blocks via the block table. Foreshadow [→ ch.12](../12-prefix-caching/).
2. **Copy-on-write branching** — beam search forks share blocks until they diverge.
3. **Tiered storage** — physical blocks can live on CPU or NVMe and be paged in on demand (KV offload).

### Pyplot

- Effective batch size vs HBM utilization, for naive contiguous vs paged. The paged curve sits ~3× higher across the full range.

## Connections

- ← [The KV Cache Is a Heap](../09-kv-fragmentation/) — the problem this chapter solves.
- ← [Attention in SRAM](../06-flash-attention/) — the kernel skeleton being modified.
- → [The Block Manager](../11-block-manager/) — the actual data structure.
- → [Reusing the Prologue](../12-prefix-caching/) — paging's first big payoff.
- → [Two Houses, Divided](../17-disagg-pd/) — paged KV is what you can ship over RDMA between machines.

## What To Remember

1. **PagedAttention is virtual memory for KV caches.** Logical positions through a per-request page table; physical blocks pooled and reused; fragmentation eliminated by paging exactly the same way Multics eliminated it in 1965.
2. **The block table is the entire mechanism.** One extra level of indirection in the attention kernel. Everything downstream — prefix caching, CoW branching, offloading — is built on it.
3. **Three free wins:** sharing prefixes, sharing beam-search forks, sharing across tiers. None of these were the goal; all of them fall out of the page-table abstraction.

**Continue to → [The Block Manager](../11-block-manager/)** — the data structure that holds all this together.

