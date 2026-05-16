---
title: "FlashAttention"
slug: "flash-attention"
description: "FlashAttention: an IO-aware exact attention algorithm that tiles the softmax computation to avoid materializing the full T×T attention score matrix in HBM. Same FLOPs as naive attention, but 2-4× faster in practice by eliminating intermediate memory writes."
category: "transformer"
also_known_as:
  - "FlashAttention"
  - "FlashAttention-2"
  - "FlashAttention-3"
  - "IO-aware attention"
  - "fused attention kernel"
  - "tiled attention"
source_of_truth: "/issues/07-sparse-lab/11-attention-compute/"
source_of_truth_title: "ch.11 The Compute Bill of Attention (Issue 07)"
related:
  - "attention"
  - "kv-cache"
draft: false
---

FlashAttention (Dao et al., 2022) reformulates the standard attention computation to minimize HBM (GPU high-bandwidth memory) traffic without changing the mathematical result.

| Term | Meaning |
|---|---|
| Naive attention | Materializes the full $T \times T$ score matrix in HBM; two HBM reads/writes per layer per forward pass |
| FlashAttention | Tiles Q, K, V into SRAM blocks; computes softmax incrementally; never writes the $T \times T$ matrix to HBM |
| Arithmetic intensity | FlashAttention raises AI from ~1 FLOP/byte (memory-bound) toward the hardware ridge point |
| FLOPs | Identical to naive attention — same $4HLT^2D$ total; speedup is purely from reduced IO |

**Why it matters for sparse attention** — FlashAttention eliminates a major memory bottleneck in *prefill* (compute-bound regime) but does not reduce FLOPs. Sparse attention methods like DSA and CSA target *decode* (memory-bound, one token at a time), where FlashAttention's prefill optimization doesn't apply. The two techniques are complementary, not competing.

**Versions:**
- FlashAttention (2022): original SRAM-tiling algorithm for A100
- FlashAttention-2 (2023): improved parallelism over sequence length; 2× faster
- FlashAttention-3 (2024): exploits Hopper (H100) async data movement and FP8 paths

See [ch.11 — The Compute Bill of Attention](/issues/07-sparse-lab/11-attention-compute/) for the full arithmetic intensity analysis and where FlashAttention fits in the optimization landscape.
