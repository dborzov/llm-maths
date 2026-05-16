---
title: "Reusing The Prologue"
description: "Every conversation in a deployment starts with the same system prompt. Hashing prefix blocks and re-using their physical memory turns a full prefill into a cache hit — the single highest-leverage optimization in modern LLM serving, and an almost embarrassingly simple one in hindsight."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T14:30:00-04:00
issue: 8
weight: 120
techKind: mainline
techNode: prefix-caching
header: 12-prefix-caching.webp
---


## Anchor and Frame

- **Anchor case.** A production chatbot ships a 1,500-token system prompt with every request. A user sends a 50-token question. Naive: prefill 1,550 tokens. With prefix cache: prefill 50 tokens. **31× less prefill compute, ~30× lower TTFT.**
- **The empirical observation.** Across real production logs, *most prompts are mostly prefix*. System prompts, few-shot examples, RAG context, conversation history — these are all bytes the model has seen many times before.

## Outline

### What Is A Prefix

- A *prefix* is a span of tokens at the start of two requests that is byte-identical. Because attention is causal, the KV vectors at positions $0..k$ depend only on tokens $0..k$. If two requests share the first $k$ tokens, they have the same KV vectors there.

### Hashing The Block

- Hash each fully-filled KV block by the tuple `(prefix_hash, token_ids[i:i+16])`. Chained: block $j$'s hash incorporates block $j-1$'s hash. This way, equal hashes are equal-prefix.
- Hash table: `block_hash → physical_block_id`. Populated when a request commits its first generation token.

### The Cache Hit Path

- On prefill, walk the new request's tokens 16-at-a-time, hash each block, look up. Hits ⇒ point the request's block table at the existing physical block, bump refcount.
- Stop on first miss. The remaining tokens go through a *truncated* prefill.

### Pyplot

- TTFT and prefill cost as a function of prefix-hit rate. Real production logs (synthetic but plausible) show 60–90% prefix coverage on chat workloads.

### Tricky Bits

- **Eviction policy.** Prefix-cached blocks are pinned (ref ≥ 1 while a request is using them), eligible for LRU eviction when their refcount drops to zero. The free queue handles this naturally (→ [The Block Manager](../11-block-manager/)).
- **Hash collisions.** Cryptographic-strength is overkill; vLLM uses xxHash. Cross-fingers on collisions; the probability is bounded by $2^{-64}$ × number of blocks.
- **Tokenization sensitivity.** A change of one byte at position zero invalidates every downstream hash. This is why some serving stacks pre-canonicalize whitespace and BOS handling.

### Beyond Server-Side: Hydragen, Cascade Inference

- The idea generalizes. Hydragen (2024) fuses attention across a *batch* with shared prefix, reading the shared KV blocks once for all batch members. Cascade attention does the same hierarchically.

### Cross-System

- LMCache, SGLang's RadixAttention, TRT-LLM's KV reuse — all converged on the same idea independently. The cache lives at the granularity of paged blocks because that's the granularity at which sharing physically works.

## Connections

- ← [Borrowing from 1965](../10-paged-attention/), [The Block Manager](../11-block-manager/) — the data structure is the prefix-cache.
- → [The Token Budget](../14-scheduler/) — the scheduler asks "how many prefix blocks does this request need?" before deciding what to admit.
- → [Two Houses, Divided](../17-disagg-pd/) — prefix caches on the prefill cluster save the most expensive work in the entire stack.

## What To Remember

1. **Equal prefix ⇒ equal KV.** This is the entire premise. Causal attention means prefix positions don't see future tokens, so they cache.
2. **Hash 16 tokens at a time, chained.** A block's hash includes its predecessor's. Same hash ⇒ same prefix back to position 0.
3. **In production, this is the optimization with the largest user-visible effect.** TTFT collapses; throughput rises; the GPU stops re-doing work that's been done a thousand times.

**Continue to → [Slicing the Prefill](../13-chunked-prefill/)** — even when there's no prefix to reuse, the prefill itself can be made gentler on everyone else in the batch.

