---
title: "The Token Budget"
description: "vLLM V1's scheduler is deliberately simple: allocate a fixed token budget per step, let each request spend it on prefill or decode tokens, and let the prefill-versus-decode distinction dissolve. This chapter traces the years of layered complexity that one clean abstraction quietly deletes."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T15:30:00-04:00
issue: 8
weight: 140
techKind: mainline
techNode: scheduler
header: default.png
---


## Anchor and Frame

- **History.** vLLM V0 (2023–24): two distinct schedulers — a prefill scheduler and a decode scheduler — coordinating through a request lifecycle of `WAITING → RUNNING_PREFILL → RUNNING_DECODE → DONE`. Stateful, fragile, full of edge cases.
- **vLLM V1, January 2025.** Woosuk Kwon and team rewrite EngineCore. The scheduler is now a function that produces a dict: `{request_id: num_tokens_this_step}`. That's it. The prefill/decode distinction has disappeared from the scheduler's vocabulary.

## Outline

### The Insight

- A prefill step is "produce KV for $N$ new tokens of this request". A decode step is "produce KV for 1 new token of this request". They are the **same operation** with different $N$.
- Once chunked prefill exists ([→ ch.13](../13-chunked-prefill/)), the prefill $N$ can be any value. The boundary between "prefill" and "decode" is now arbitrary — just where you cut the chunk.

### The Algorithm, In Pseudocode

```python
def schedule_step(self):
    budget = self.token_budget        # e.g. 8192
    plan = {}                         # request_id -> num_tokens

    # 1. Continue running requests in priority order
    for req in self.running:
        n = min(req.tokens_remaining_to_prefill or 1, budget)
        if not self.block_manager.can_allocate(req, n):
            self.preempt(req); continue
        plan[req.id] = n
        budget -= n
        if budget == 0: return plan

    # 2. Admit waiting requests
    while self.waiting and budget > 0:
        req = self.waiting.popleft()
        # Apply prefix-cache lookup; reduces n
        n = min(req.tokens_to_prefill_after_cache, budget)
        if not self.block_manager.can_allocate(req, n):
            self.waiting.appendleft(req); break
        plan[req.id] = n
        self.running.add(req); budget -= n

    return plan
```

Walk through every line. Each line corresponds to a chapter:
- `block_manager.can_allocate` → [The Block Manager](../11-block-manager/).
- `tokens_to_prefill_after_cache` → [Reusing the Prologue](../12-prefix-caching/).
- `min(..., budget)` → [Slicing the Prefill](../13-chunked-prefill/).
- The plan is then handed to the model runner, which assembles inputs and launches the captured CUDA graph for batch size ≥ next bucket → [Launches Aren't Free](../05-cuda-graphs/).

### Preemption

- Running requests can be evicted to free up KV blocks (their cache is dropped; on next run they recompute or restore from CPU offload). The policy is configurable; FCFS-with-priority is the default.

### Why The Token Budget Matters

- It's the unit of fairness: every request gets at most $N$ tokens per step. It's the unit of bounded latency: any single step is $\le$ budget × per-token cost.
- It's also what makes graph capture practical: the engine captures one graph per (admitted batch size, total tokens) bucket and replays the closest one.

### The V1 vs V0 Comparison

- Pyplot or table: lines of scheduler code, scheduling latency per step, P99 tail latency. V1 is shorter, faster, and lower-tail.

### Async Engine + Multi-Process Architecture

- Worth a callout: the scheduler runs inside the `EngineCore` process. The API server (`AsyncLLM`) is a separate process talking over a ZMQ bus. This isolates Python GIL contention and pipeline lag between request ingest and engine work.

## Connections

- ← [The Conveyor Belt](../08-continuous-batching/), [Reusing the Prologue](../12-prefix-caching/), [Slicing the Prefill](../13-chunked-prefill/), [Borrowing from 1965](../10-paged-attention/), [Launches Aren't Free](../05-cuda-graphs/) — every previous chapter shows up as one expression in `schedule_step`.
- → [The Draft Trick](../15-speculative-decoding/) — the scheduler is what dispatches speculative-decode candidate verification.
- → [Two Houses, Divided](../17-disagg-pd/) — disagg P/D splits the scheduler into two coordinating instances.

## What To Remember

1. **The V1 scheduler is a dict-producing function.** `{request_id: num_tokens}` per step. Prefill and decode are both "produce some KV". The distinction lives only in how big $N$ is.
2. **The token budget is the central knob.** Per-step bounded work; predictable latency; clean graph-capture buckets.
3. **Every previous chapter is one line of the scheduler.** Block manager, prefix cache, chunking — they all surface as method calls. That's why this chapter is the engine half's climax.

**Continue to → [The Draft Trick](../15-speculative-decoding/)** — once the scheduler can dispatch one token of work per step, dispatching *several candidate tokens* costs almost the same and gets you most of them for free.

