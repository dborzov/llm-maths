---
title: "Slicing The Prefill"
description: ""
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T05:05:29-04:00
issue: 8
weight: 130
techKind: mainline
techNode: chunked-prefill
header: default.png
---

> **Scaffold note.** Chunked prefill is the *fairness* mechanism — without it, one long-context request can wreck inter-token latency for every other user in the batch. With it, every step has bounded compute and predictable wall time.

## Anchor and Frame

- **Anchor case.** User A sends a 64K-token document for summarization. User B is mid-decode, expecting tokens every 30 ms. Naive serving: A's prefill takes 5 seconds; B's stream stalls for the entire 5 seconds. **Tail latency catastrophe.**
- **Fix.** Slice A's prefill into 32 chunks of 2K tokens. Run one chunk per scheduler step, alongside B's decode. A's prefill now takes 32 steps × ~150 ms each ≈ 4.8 s (slightly more wall time) — but B never sees a gap > 150 ms.

## Outline

### The Problem, Quantified

- Prefill is compute-bound, scaling linearly in $T$ for fixed model. At $T = 64K$, that's hundreds of times one decode step.
- A naive scheduler that batches A's full prefill with B's decode picks the slowest member: the prefill. Decode latency follows prefill latency.

### Slice The Work

- Prefill is *associative* over the token axis: producing the KV blocks for tokens $0..T$ can be decomposed into producing $0..T_1$, then $T_1..T_2$, etc. Each chunk does a prefill over $T_1..T_2$ with attention attending to all already-cached KV blocks.
- This requires the attention kernel to accept "the queries are a chunk; the keys/values are the chunk plus everything cached so far". Flash + Paged attention already does exactly this.

### What Chunk Size To Pick

- vLLM's `long_prefill_token_threshold`: typically 2048–8192. Trade-off:
  - Smaller chunks → finer-grained interleaving, better tail latency.
  - Larger chunks → fewer kernel launches, higher arithmetic intensity per step.
- The right size depends on hardware (H100 vs H200) and target ITL.

### The Pyplot

- Tail latency (p99 ITL) vs prefill length, with and without chunking. Without: hockey stick. With: flat.

### Mixed Step Composition

- A scheduler step's batch now contains both prefill chunks (variable per-request token counts) and decode steps (one token per request). The new token budget is `prefill_chunk_size + n_decoding`.
- This composes naturally with the V1 scheduler's `{request_id: num_tokens}` representation (→ [The Token Budget](../14-scheduler/)).

### Sarathi / Sarathi-Serve

- History: Agrawal et al., Microsoft Research India, "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023, ASPLOS 2024). The paper that crystallized the technique. vLLM and SGLang adopted it within months.

### When To Disable It

- Tiny prefills (< chunk size) skip chunking entirely.
- Prefix-cache hits can leave a tiny tail; that tail is one chunk.

## Connections

- ← [Two Phases, Two Personalities](../07-prefill-vs-decode/) — chunking dissolves the visible boundary.
- ← [The Conveyor Belt](../08-continuous-batching/) — chunking makes a *fair* continuous batch possible.
- → [The Token Budget](../14-scheduler/) — chunked prefill is the operational reason the scheduler unifies prefill and decode.

## What To Remember

1. **Prefill is associative over the token axis.** A long prefill is N short prefills with attention attending to growing context. The attention kernel already supports this.
2. **One long request can starve everyone else.** Chunked prefill bounds the size of any single scheduler step → bounds tail ITL.
3. **The right chunk size is a tuning knob, not a constant.** It trades fine-grained fairness for kernel-launch amortization.

**Continue to → [The Token Budget](../14-scheduler/)** — once chunking exists, the prefill/decode distinction stops being interesting and the scheduler can treat them as one workload.

