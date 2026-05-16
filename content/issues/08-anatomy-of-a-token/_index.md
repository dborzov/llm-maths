---
title: "Anatomy of a Token"
description: "From the TCP packet to the streamed character on your screen — a first-principles tour of every layer of a modern LLM inference engine, with vLLM as the worked example and a GPU as the patient on the table."
issue: 8
layout: issue-cover
theme: cream
math: false
header: default.webp
date: 2026-05-16T09:00:00-04:00
---

## The Mystery

**Spring 2026.** You open the ChatGPT app on your phone, type a question, and tap send. Three hundred milliseconds later, the first character of the answer appears. Two seconds later, you're reading a fluent paragraph streaming in at sixty words per second. Somewhere in a data center, a 600-billion-parameter mixture-of-experts model just produced a sequence of conditional probability distributions, sampled from them, and pushed bytes back to your phone — all while serving roughly **eight hundred other users** off the same eight-GPU box.

Run the napkin math. A 600B model in FP8 is 600 GB. An H200 has 141 GB of HBM. The model has been sliced eight ways. The {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} for eight hundred concurrent users at an average of 4,000 tokens of context is **another 250 GB of working memory**, allocated and freed at every step. At decode time, every one of those eight GPUs is fetching tens of gigabytes per second across an HBM bus, with a peak bandwidth of 4.8 TB/s. The model weights stream through SRAM. The {{< wiki "attention" >}}attention{{< /wiki >}} kernel is reading scattered keys and values from a paged address space. Tokens are being scheduled across users at every iteration, sharing a finite token budget. Somewhere a draft model is speculating four tokens ahead and the target model is verifying its guesses in a single forward pass. Two physical machines are passing KV cache to each other over RDMA.

None of this existed five years ago. Most of it didn't exist three years ago.

{{% pullquote type="counter-intuitive" %}}
The hardest problems in modern LLM inference are not about machine learning. They are about virtual memory, scheduling, queueing theory, and the fact that loading a byte from HBM is roughly two thousand times slower than multiplying two numbers in a register.
{{% /pullquote %}}

This issue takes apart a single inference request and walks the reader through every layer the token passes through on its way out of the box. The starting point is bare silicon: streaming multiprocessors, the memory hierarchy, the arithmetic intensity that determines whether a kernel is doing real work or sitting on the HBM bus. Then we build upward: the attention kernel, the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} and why its allocation pattern is a hell of fragmentation, the operating-system trick from 1965 that fixed it, the scheduler that decides who gets cycles, and the modern serving topologies (continuous batching, chunked prefill, prefix caching, disaggregated prefill/decode, speculative decoding) that turn a four-GPU machine into something that can credibly handle a thousand concurrent users.

The canonical reference implementation throughout is **vLLM** (V1 engine, late 2025), because it is open source, widely deployed, and an unusually clean instance of the architectural pattern that every serving system — vLLM, SGLang, TensorRT-LLM, TGI — converges on. The cover image is the inference stack as a comic-book heist: a token enters from the bottom, gets passed hand-to-hand up through eighteen layers of machinery, and exits from the top with a probability mass next to it.

## How To Read This Issue

Start with the [cold open](01-cold-open/) — it follows one HTTP request through one specific box, and names every layer the token touches. After that, the **tech tree** below is the table of contents.

{{< techtree name="issue08" >}}

The five **GPU primers** at the bottom (`gpu-anatomy`, `memory-hierarchy`, `roofline`, `cuda-graphs`, `flash-attention`) are the foundation. If you already know what a streaming multiprocessor is, what HBM bandwidth costs, and why FlashAttention is mostly about not writing the score matrix to memory — skip them and pick up the mainline at `prefill-vs-decode`. If those phrases were vague, read them in order; everything downstream stops making sense without them.

The **pink mainline** is the actual narrative. It starts where the canonical [microGPT](/llm-maths/issues/05-microgpt-unfolded/) issue ended — the [prefill / decode](/llm-maths/issues/05-microgpt-unfolded/14-prefill-decode/) split — and walks through the year-by-year invention of the modern inference stack: continuous batching (Orca, 2022), PagedAttention (vLLM, 2023), prefix caching (2024), chunked prefill (2024), the V1 unified scheduler (2025), speculative decoding (2023–25), and disaggregated prefill/decode (DistServe / Mooncake / NIXL, 2024–26). The **boss** chapter at the top, *[The Full Anatomy of a Token](18-full-anatomy/)*, returns to the cold-open packet and re-annotates it with every component the tree has introduced.

Two cream primers — `block-manager` and `tp-pp` — sit beside the mainline. Read them when the mainline points to them, or read them up front if you prefer foundations before narrative.

## What You Will Walk Away With

By the end of this issue you should be able to answer, with confidence and napkin math:

- Why a modern GPU is structured as a few-hundred **streaming multiprocessors** running thousands of threads in parallel, and why its **memory hierarchy** — registers, shared memory, L2, HBM — has bandwidths that span four orders of magnitude.
- Why the **roofline model** says decode attention sits at roughly 1 FLOP/byte and is therefore **~90× below the H100 ridge point** — and what that single fact predicts about the entire shape of the inference stack.
- Why **kernel launch latency** matters at decode time, what a **CUDA graph** actually is, and why vLLM's piecewise compilation captures only the long stable suffix of the forward pass.
- The exact arithmetic of **FlashAttention's tiling**: how online softmax lets you compute attention without ever materializing the score matrix in HBM, and why this is the single most important inference kernel of the decade.
- Why naive **static batching** wastes most of the GPU when sequences finish at different times, why **continuous batching** (Orca, 2022) fixed it by swapping requests in and out at every iteration, and what the new bottleneck became.
- Why the **KV cache is a memory-allocator nightmare**: variable-length sequences, unpredictable growth, fragmentation, the worst-case over-allocation that classical CUDA allocators were forced into.
- The **PagedAttention** trick: how a 1965 operating-systems idea (virtual memory + page tables) maps onto KV cache management, what a **block table** is, and what changes inside the attention kernel when keys and values live at non-contiguous physical addresses.
- How vLLM's **block manager** turns the block table into a real data structure: free queues, reference counting, copy-on-write, and the O(1) LRU eviction policy that makes prefix caching cheap.
- Why **prefix caching** is one of the highest-leverage optimizations in the entire stack — every conversation shares a long system prompt, and hashing it lets every request after the first skip prefill entirely.
- Why long prefills hurt **time-to-first-token** for everyone in the batch, and how **chunked prefill** slices a 100K-token prompt into pieces that interleave with decode steps.
- The **V1 unified scheduler**'s surprising insight: prefill and decode are the same operation differing only in token count, so the scheduler just allocates a token budget per step — `{request_id: num_tokens}` — and the prefill/decode dichotomy quietly dies.
- Why **speculative decoding** works exactly because decode is bandwidth-bound: verifying thirty-two candidate tokens costs the same memory pass as verifying one, so you can run a small draft model and get a free 2–3× speedup if your draft is even modestly accurate.
- The mechanics of **tensor parallelism** (column-parallel + row-parallel linear layers, one all-reduce per block) and **pipeline parallelism** (one stage per GPU, micro-batches), and the rule of thumb for which to reach for at which model scale.
- Why the field is splitting **prefill machines and decode machines** into physically separate clusters connected by an RDMA KV-cache fabric — and what NIXL and Mooncake actually transport.
- The full annotated **life of a token**, end to end, with every component named, every latency budget itemized, and every line of pseudocode pointing to the chapter where it was unpacked.

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
