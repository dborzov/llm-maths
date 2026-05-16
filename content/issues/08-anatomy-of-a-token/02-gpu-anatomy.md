---
title: "Inside the Silicon"
description: "A GPU is a throughput machine: 132 streaming multiprocessors, thousands of simultaneous threads, and a memory hierarchy that spans four orders of magnitude. Understanding the hardware is the foundation for every inference optimization that follows."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T09:30:00-04:00
issue: 8
weight: 20
techKind: primer
techNode: gpu-anatomy
header: 02-gpu-anatomy.webp
---


## Anchor and Frame

- **Open with a comparison.** A 64-core CPU has ~64 truly parallel things. An H100 has **~16,896 CUDA cores** organized into **132 streaming multiprocessors** and runs ~270,000 threads simultaneously. That is a four-orders-of-magnitude difference in parallelism.
- **Anchor example:** add two billion-element vectors. CPU: 64 cores, ~1.5 GB/s/core, ~15 ms. H200: 1 kernel launch, 80 GB at 4.8 TB/s, ~3 ms. The *kernel* takes longer to launch than to run.

## Outline

### A Brief Hardware History

- 1999 GeForce 256, 2006 G80/CUDA, 2017 Volta tensor cores, 2022 Hopper, 2024 Blackwell. The story is: shaders → general compute → matrix engines → mixed-precision factories.

### The Streaming Multiprocessor

- An H100 SM = 4 sub-partitions × (16 FP32 + 16 INT32 + 8 FP64 lanes) + 1 tensor core + 65,536 32-bit registers + 228 KB combined L1/shared.
- Why "warp": 32 threads execute in lockstep (SIMT). Divergent branches serialize.
- Show a diagram (pyplot or SVG) of one SM.

### Threads, Warps, Blocks, Grids

- The CUDA execution hierarchy. Anchor with a 1024-thread vector add: launch grid = (N/1024 blocks, 1024 threads/block), each thread handles one element.

### Tensor Cores

- One tensor core does a 4×4 (or larger) matmul accumulate in *one* clock. The actual workhorse of LLM forward passes. Quote real numbers: H100 FP16 = 989 TFLOP/s on tensor cores vs ~67 TFLOP/s on CUDA cores.
- Why this matters for the rest of the issue: prefill keeps tensor cores busy; decode does not.

### Kernel Launch: The CPU-Side Cost

- Each kernel launch is a few microseconds of CPU overhead. Foreshadow [Launches Aren't Free](../05-cuda-graphs/).

### Napkin Math

- H100: 132 SMs × 4 sub-partitions × 32 threads × 32 register width ≈ 540,000 simultaneous register slots. The GPU is, mechanically, a *very wide* register file.
- Pyplot: bar chart of FLOP/s per chip (CPU vs GPU, 2010→2026) showing the order-of-magnitude divergence.

### Anchor Closing

- Restate: a GPU's job is to *hide latency with parallelism*. It does not make individual operations faster; it makes *many* operations happen at once. Every optimization in this issue is a consequence of that.

## Connections

- → [The Pyramid of Speed](../03-memory-hierarchy/) — once you know what the cores are, the next question is how they get fed.
- ← [The Cold Open](../01-cold-open/) — Anya's request lives on 132 of these SMs.
- External: AMD MI300, Apple M-series GPUs, TPUs — same throughput-machine philosophy, different surface details.

## What To Remember

1. A GPU is a SIMT processor: thousands of threads in lockstep groups of 32 ("warps") over hundreds of streaming multiprocessors.
2. The real arithmetic muscle is the **tensor core**, not the scalar CUDA core. Modern attention and matmul kernels live on it.
3. Throughput, not latency. Every architectural choice — wide registers, deep pipelines, big shared memory — sacrifices single-operation speed for parallelism.

**Continue to → [The Pyramid of Speed](../03-memory-hierarchy/)** — the SMs are hungry, and the rest of the issue is the story of feeding them.

