---
title: "The Cold Open"
description: "One HTTP packet, one H200 box, one streamed reply — a five-minute tour of every layer the token will visit, every component named, no detail explained yet. The whole issue is the explanation."
topics: [inference, vllm]
tags: [vllm, gpu, inference, serving, cold-open]
theme: cream
math: true
draft: false
date: 2026-05-16T09:00:00-04:00
issue: 8
weight: 10
techKind: mainline
techNode: cold-open
header: 01-cold-open-08.webp
---


## Anchor and Frame

- **One specific box.** An 8×H200 server in some us-east-2 zone, June 2026. Make it concrete: chassis, NVLink switch, NICs.
- **One specific user.** Anya, somewhere on a phone, types: *"Why did the Roman Empire fall?"* (twelve tokens after tokenization). She is user #347 on this box right now.
- **One specific model.** Llama-3-70B in FP8. Sliced TP=4 across half the chassis. The other half is doing a different model entirely.

## The Tour (Outline)

Walk through the entire stack at the level of *names*, not *mechanisms*. Each named component links forward to the chapter that unpacks it.

### Step 0 — Wire

- TCP packet hits an HTTPS load balancer, arrives at the OpenAI-compatible API server inside vLLM. *Out of scope; we open the box at the API.*

### Step 1 — Tokenizer and AsyncLLM

- Bytes → tokens (BPE/Tiktoken). 47 chars become 12 tokens.
- `AsyncLLM` wraps the request, ships it across an inter-process bus to **EngineCore**.

### Step 2 — Scheduler picks a slot

- Forward link → [The Token Budget](../14-scheduler/).
- The scheduler sees 53 in-flight requests already; finds room in this iteration's *token budget*; tags Anya's request with 12 prefill tokens.

### Step 3 — KV cache allocation

- Forward link → [Borrowing From 1965](../10-paged-attention/), [The Block Manager](../11-block-manager/).
- 12 tokens × 80 layers × 8 KV-heads × 128 dim × 2 (K+V) × 1 byte (FP8) ≈ **246 KB**. Allocated as **1 block of 16 slots** in the paged KV pool.

### Step 4 — Prefix cache lookup

- Forward link → [Reusing The Prologue](../12-prefix-caching/).
- The first eight tokens are the standard system prompt. Hash hit! Anya's request inherits a 19-block prefix from a cache entry that has been resident since 6 AM.

### Step 5 — The forward pass

- Forward link → [Inside the Silicon](../02-gpu-anatomy/), [The Pyramid of Speed](../03-memory-hierarchy/), [Attention in SRAM](../06-flash-attention/).
- Model weights stream from HBM → SRAM through 132 SMs. {{< wiki "attention" >}}Attention{{< /wiki >}} is a tiled FlashAttention kernel; the score matrix never lands in HBM.
- The forward pass is launched as a single **CUDA graph replay** (200 µs of CPU overhead, not 2 ms).

### Step 6 — Sampling + return-of-token

- The logits for token-12 land on the engine process. Top-p sampling picks the first reply token: `The`.
- It is streamed back through AsyncLLM → API server → HTTP/2 frame → Anya's phone. **TTFT ≈ 280 ms.**

### Step 7 — Decode loop

- Forward link → [Two Phases, Two Personalities](../07-prefill-vs-decode/), [The Conveyor Belt](../08-continuous-batching/), [The Draft Trick](../15-speculative-decoding/).
- Anya's request now joins the *decode set*. Every 30 ms a new token is appended. The engine is doing continuous batching with 53 other users — some prefilling, most decoding.
- A draft model is speculating 4 tokens ahead; the target model accepts 3 of them in the same memory pass.

### Step 8 — Disaggregated case

- Forward link → [Two Houses, Divided](../17-disagg-pd/).
- *In a future deployment*, Anya's prefill ran on a different physical machine than her decode. The KV cache was shipped over NIXL/Mooncake at wire speed. Three hops, one hidden state, zero milliseconds of bullshit.

### Step 9 — Done

- Anya's request emits `<|eot|>` after 247 tokens. Block IDs returned to the free pool. Prefix-cache entry reference-decremented. The 53rd slot is open again.

## What To Remember

1. **Every engineering decision in modern inference is downstream of one fact**: at decode time, the GPU is loading bytes faster than it is multiplying them. The rest of the issue is consequences.
2. **Every layer in the stack is named after a specific 2022–2026 paper**: Orca (continuous batching), PagedAttention (vLLM), Hydragen (prefix caching), DistServe / Mooncake (disagg P/D), EAGLE (spec decode). The history is recent and traceable.
3. **The tech-tree on the cover is the map.** Read primers if names like *streaming multiprocessor* or *HBM* feel hand-wavy; otherwise jump to *prefill vs decode*.

**Continue to → [Inside the Silicon](../02-gpu-anatomy/)** — before we can dissect the stack, we need to know what kind of machine is on the operating table.

