---
title: "The Pyramid of Speed"
description: "Registers, shared memory, L2, and HBM: four orders of magnitude of bandwidth separate the fastest storage from the slowest. Every inference optimization is a trade along this pyramid — capacity for speed, or speed for capacity."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T10:00:00-04:00
issue: 8
weight: 30
techKind: primer
techNode: memory-hierarchy
header: default.png
---


## Anchor and Frame

- **Cold open:** Seymour Cray's quote — *"You can't fake what you don't have. Memory bandwidth is the only thing that matters."* Show that a 2024 H200 has more HBM bandwidth (4.8 TB/s) than the entire combined memory bandwidth of every commercial system Cray ever shipped.
- **Anchor example:** loading a single 70B-parameter model in BF16 — 140 GB. Streamed through SRAM one tile at a time during every forward pass.

## Outline

### The Pyramid

A levelled table — capacity ↓ as bandwidth and proximity ↑. (Show as a stair-stepped chart, not a table; that's the visual hook of the article.)

| Level | Per SM | Total | Bandwidth | Latency | Lives |
|---|---|---|---|---|---|
| Registers | 256 KB | ~34 MB | ~20 TB/s | ~1 cycle | per-thread, per-warp |
| Shared / L1 (SRAM) | 228 KB | ~30 MB | ~19 TB/s | ~30 cyc | per-block |
| L2 cache | — | 50 MB | ~6 TB/s | ~200 cyc | chip-wide |
| HBM3 | — | 80–141 GB | 3.35–4.8 TB/s | ~600 cyc | chip-wide |
| NVLink to peer GPU | — | — | 900 GB/s | µs | chassis |
| PCIe to host | — | — | 64 GB/s | µs | host |

Each rung is ~10× slower and ~100× larger than the one above. Four rungs, four orders of magnitude.

### Why HBM Exists

- A short history of why GPUs moved from GDDR to HBM (2015 Fiji, then HBM2/3/3e). Stacked silicon with through-silicon vias. The dies sit on top of the GPU package on a silicon interposer — not connected by board traces — which is the only way to hit terabyte-per-second bandwidths.

### Why SRAM Is Tiny

- SRAM is six transistors per bit. DRAM is one transistor plus a capacitor. The area trade-off is *brutal* — that's why 30 MB of SRAM and 80 GB of DRAM on the same chip.

### The Cost of a Load

- Napkin math: an HBM byte costs ~600 cycles of latency to arrive. An FMA on a tensor core costs ~1 cycle. **The same time budget loads one byte or runs ~600 FMAs.** This number is the entire reason the rest of the issue exists.
- Pyplot: log-scale bar chart of bandwidth per level. Highlight the four-orders gap.

### Implications, Forward-Linked

- **{{< wiki "attention" >}}Attention{{< /wiki >}} kernels** — must tile through SRAM (→ [Attention in SRAM](../06-flash-attention/)).
- **Roofline** — arithmetic intensity is the single number predicting which level you live in (→ [The Roofline](../04-roofline/)).
- **KV cache** — lives in HBM, must be fetched anew every decode step (→ [The KV Cache Is a Heap](../09-kv-fragmentation/)).

## Connections

- ← [Inside the Silicon](../02-gpu-anatomy/) — the cores are useless without this hierarchy.
- → [The Roofline](../04-roofline/), [Attention in SRAM](../06-flash-attention/).
- Cross-issue: Issue 06's [Bandwidth Wall](/llm-maths/issues/06-eviction-notice/11-bandwidth-wall/) deepens the same idea for decode-time KV pressure.

## What To Remember

1. **Four orders of magnitude.** Register bandwidth ≫ SRAM ≫ L2 ≫ HBM ≫ off-chip. Performance follows from how high in the pyramid you can keep your hot working set.
2. **Bandwidth, not capacity, is the binding constraint** for LLM inference. Capacity gates *whether* a request fits. Bandwidth gates *how fast* every step runs.
3. **Every important kernel is a tiling story** — Flash, Paged, fused-MLP, fused-rotary. They all reduce to "keep the tile in SRAM for as long as possible".

**Continue to → [The Roofline](../04-roofline/)** — the napkin-math diagram that compresses this whole hierarchy onto two axes.

