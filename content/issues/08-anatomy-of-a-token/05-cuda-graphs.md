---
title: "Launches Aren't Free"
description: "A decode-step forward pass launches hundreds of tiny CUDA kernels. The Python scheduling overhead can eat the entire latency budget before the GPU starts. CUDA graphs capture the launch sequence and replay it as a single GPU command, dropping CPU-side overhead from milliseconds to microseconds."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T11:00:00-04:00
issue: 8
weight: 50
techKind: primer
techNode: cuda-graphs
header: 05-cuda-graphs.webp
---


## Anchor and Frame

- **Open with a profile screenshot.** A naive Hugging Face decode loop on Llama-3-8B: GPU utilization 8%, ITL 60 ms. Same code with CUDA graphs: 38%, ITL 22 ms. The kernels did the same work.
- **The anchor stat.** A single kernel launch costs ~5–10 µs of CPU time (PyTorch overhead + driver). A 32-layer Llama decode step has ~250 kernel launches. That's **~2 ms** before the GPU has computed anything.

## Outline

### Why Launches Are Slow

- A "kernel launch" is: Python → C++ binding → cuLaunchKernel → driver enqueues onto a stream → GPU eventually picks it up. Each step costs microseconds.
- During *prefill* with batch=64, sequence=4096, the GPU work is so large that launch overhead disappears in the noise. During *decode*, the kernels are tiny — the launch is the kernel.

### CUDA Graphs

- A CUDA graph is a *recording* of a sequence of kernel launches with their topology and arguments. Capture once, replay many times.
- A graph replay is one driver call with O(1) overhead, regardless of how many kernels it contains.
- Constraints: all tensor shapes, pointers, and arguments must be identical at replay time. This is the source of every painful corner case.

### Piecewise Compilation

- vLLM V1 uses `torch.compile` to **partition** the forward pass into a *piecewise* graph: a small dynamic prelude (attention metadata, dispatch on batch size), then a long stable suffix (the per-layer transformer blocks with fixed shapes) captured as a graph.
- Kaichao You's design: graphs are captured at a small set of batch-size "buckets" — 1, 2, 4, 8, … 512. At decode time the engine pads the batch up to the next bucket and replays that bucket's graph.

### The Padding Cost

- Worked example: 53 in-flight decodes get padded to 64. The 11 wasted slots cost some FLOPs and some HBM bandwidth but save thousands of microseconds of launch overhead. **Padding to the next bucket is a near-universal win at decode.**

### What Stays Eager

- Attention is *not* captured into the graph — its shape depends on the per-request KV-cache length and block table. It's invoked via a callable that the graph branches into. This is the *piecewise* in piecewise compilation.
- Sampling, log-prob extraction, structured-output mask application: usually eager.

### Pyplot

- Bar chart: ITL with eager vs graph capture for batch sizes 1–256. Show the gap closes as batch grows (because GPU work grows past CPU overhead).

## Connections

- ← [Inside the Silicon](../02-gpu-anatomy/) — the SMs are waiting on the host loop.
- → [The Token Budget](../14-scheduler/) — the scheduler has to respect graph-capture buckets when shaping each step's batch.
- → [Two Phases, Two Personalities](../07-prefill-vs-decode/) — the launch-overhead story is the *other* half of why prefill and decode are different (the first half is arithmetic intensity).

## What To Remember

1. **At decode time the CPU can be the bottleneck**, not the GPU. ~5 µs per launch × hundreds of layers × thousands of tokens/sec is hundreds of milliseconds of pure overhead.
2. **CUDA graphs replay a captured launch sequence as a single driver call** with O(1) overhead. vLLM applies this in *pieces* — stable suffix captured, dynamic prelude eager.
3. **Padding to the nearest captured batch-size bucket is nearly always worth it.** Wasted FLOPs are cheap when you live in the bandwidth-bound regime.

**Continue to → [Attention in SRAM](../06-flash-attention/)** — the last GPU primer before the engine takes over. The single most important inference kernel of the decade.

