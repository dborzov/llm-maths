---
title: "The Block Manager"
description: ""
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T05:05:29-04:00
issue: 8
weight: 110
techKind: primer
techNode: block-manager
header: default.png
---

> **Scaffold note.** A primer for the curious. The block manager is the engineering inside the paging abstraction. Cover the data structures and the O(1) operations that make it fast enough to run on the critical path.

## Anchor and Frame

- **Anchor case.** A single decode step. 512 in-flight requests. Each step: ~20 of them need to grow by one block. ~5 finish (releasing all their blocks). The block manager must do allocate/release in nanoseconds — *per request, per step* — without ever copying memory.
- **Source.** Walk the `vllm.v1.core.kv_cache_manager` and `vllm.v1.core.block_pool` modules as the running example.

## Outline

### Three Data Structures

1. **BlockPool** — global slab of physical blocks (a giant contiguous KV tensor on each GPU, conceptually indexed `[0..N)`).
2. **FreeKVCacheBlockQueue** — doubly-linked free list of physical block ids. O(1) pop and push. Free order matters: it's the *LRU* tail that gets evicted first when prefix caching is active.
3. **KVCacheManager** — per-request orchestrator. Owns the block table per request id. Calls into BlockPool to get/return blocks.

### The Operations

For each, give the operation and its big-O:

- `allocate_initial(seq_len)` — at prefill: get `ceil(T/16)` blocks. O(blocks).
- `append_token()` — at decode: maybe append one block. Amortized O(1) per token.
- `free(request_id)` — at completion: return all blocks to free pool. O(blocks).
- `cow_branch(parent, n_children)` — beam search/parallel sampling: bump ref count on parent's last block, allocate fresh blocks for each child as soon as they write. O(1) up front, lazy.
- `lookup_prefix(hash)` — prefix caching: hash → block id. O(1) average via dict.

### Reference Counting + LRU

- Every physical block carries a refcount. Drops to 0 → eligible for eviction, but not immediately freed (LRU keeps it around in case a future request hashes to the same prefix).
- When the free pool is exhausted, the LRU tail of the zero-refcount blocks is evicted to satisfy a new allocation.

### Copy-on-Write In Practice

- Two beam-search candidates share blocks for the prompt. When candidate A diverges, its next-block write triggers a CoW: copy the parent's last block, decrement parent ref, allocate fresh.
- vLLM's design notes from the paper.

### Diagram

- Pyplot or SVG: an animated-style snapshot of the BlockPool over three decode steps. Color: free, in-use, prefix-shared, decrement-pending.

### Why This Belongs On The Critical Path

- The block manager runs every iteration. If a `dict[hash]` lookup were 10 µs per request × 500 requests, that's 5 ms — already a fifth of the decode-step budget. vLLM optimizes Python overhead heavily here (object pooling, `__slots__`, avoiding dict allocations in the hot loop).

## Connections

- ← [Borrowing from 1965](../10-paged-attention/) — the page-table abstraction.
- → [Reusing the Prologue](../12-prefix-caching/) — the LRU+refcount story is the foundation of prefix caching.
- → [The Token Budget](../14-scheduler/) — the scheduler asks the block manager whether each step's plan fits.

## What To Remember

1. **The free queue is a doubly-linked list.** Push and pop in O(1). LRU eviction is the tail. Reference counting is per block.
2. **CoW is lazy.** Two sequences share blocks until one writes; the write triggers a copy. Beam search and parallel sampling are free until divergence.
3. **The block manager lives on the per-iteration critical path.** Every microsecond of Python overhead here taxes every decode step.

**Continue to → [Reusing the Prologue](../12-prefix-caching/)** — the highest-leverage optimization the block manager unlocks.

