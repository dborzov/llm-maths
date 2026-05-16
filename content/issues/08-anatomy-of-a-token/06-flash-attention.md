---
title: "Attention In SRAM"
description: ""
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T05:05:21-04:00
issue: 8
weight: 60
techKind: primer
techNode: flash-attention
header: default.png
---

> **Scaffold note.** The last GPU primer. FlashAttention is the *kernel* that the rest of the issue takes for granted. It deserves one chapter because (a) it's the most influential single kernel in transformer history and (b) PagedAttention is built on the same tiling skeleton with paged indirection added.

## Anchor and Frame

- **History.** Tri Dao, Daniel Y. Fu et al., Stanford, May 2022. FlashAttention. Then FA-2 (2023), FA-3 (2024 Hopper), FA-4 (2025 Blackwell). Each generation widens the gap between fused and naive attention.
- **Anchor problem.** Score matrix size for a sequence of length $T$: $T^2$. At $T = 32\text{K}$, $T^2 = 10^9$ entries. Even in FP16 that's 2 GB **per head**, **per layer**. Naive attention writes and re-reads this matrix to HBM. It is unbearable.

## Outline

### The Naive Pipeline

- Compute $S = QK^\top$ → write S to HBM → read S → compute $P = \text{softmax}(S)$ → write P → read P → compute $O = PV$.
- Three HBM round-trips for the full $T \times T$ matrix. $O(T^2)$ HBM traffic. Show the napkin math: for $T = 8192$, that's tens of GB of HBM traffic *per layer*.

### The Insight: Online Softmax

- The softmax normalizer is a *running* statistic — you can compute it block by block if you keep a running max $m$ and a running sum $\ell$.
- Walk through the update equations:
  $$ m_\text{new} = \max(m_\text{old}, m_\text{block}) $$
  $$ \ell_\text{new} = e^{m_\text{old}-m_\text{new}}\,\ell_\text{old} + e^{m_\text{block}-m_\text{new}}\,\ell_\text{block} $$
- This is the single mathematical ingredient that makes the rest possible.

### Tiling

- Load tiles of K and V from HBM → SRAM. For each tile, compute partial $QK^\top$, update the running softmax, accumulate into $O$. Never write the score matrix anywhere except inside SRAM.
- HBM traffic drops from $O(T^2)$ to $O(T)$. **That single change is the entire 2–4× speedup.**

### The Loop Skeleton

- Provide actual pseudocode that mirrors the FA-2 paper. Two nested loops: outer over Q-tiles (queries), inner over KV-tiles (keys/values).

### FA-2, FA-3, FA-4

- FA-2 (2023): reorder loops so warps work on rows of Q rather than rows of S — improves occupancy on Ampere/Hopper.
- FA-3 (2024 Hopper): TMA-driven loads + warp-specialization with the WGMMA matrix engine.
- FA-4 (2025 Blackwell): block-FP4 tensor cores, tcgen05 instructions.

### Pyplot

- HBM bytes vs sequence length for naive vs Flash: a parabola vs a line. The cross-over is at $T \approx 100$; by $T = 1024$ Flash is dominating; at $T = 32\text{K}$ the gap is 1000×.

### Foreshadow Paging

- The Flash inner loop indexes K and V at sequential strides. PagedAttention modifies the indexing to walk a *block table* — a level of indirection — without changing the tiling or softmax structure. **Flash + paging = vLLM's actual decode kernel.**

## Connections

- ← [The Pyramid of Speed](../03-memory-hierarchy/), [Inside the Silicon](../02-gpu-anatomy/).
- → [Borrowing from 1965](../10-paged-attention/) — paged attention literally edits this kernel.
- → [Two Phases, Two Personalities](../07-prefill-vs-decode/) — Flash is the prefill workhorse; the decode special case is "outer loop has one row".
- Cross-link to {{< wiki "attention" >}}attention{{< /wiki >}} (microGPT ch.8).

## What To Remember

1. **The score matrix never has to land in HBM.** That single trick — online softmax over SRAM tiles — is the entire FlashAttention contribution.
2. **HBM traffic drops from $O(T^2)$ to $O(T)$.** Naive attention is unworkable at long contexts not because of arithmetic but because of bytes.
3. **Every modern attention variant** — paged, sliding-window, MLA, NSA, sparse — is a wrinkle on the same tiled inner loop. Master Flash and you've mastered 90% of attention engineering.

**Continue to → [Two Phases, Two Personalities](../07-prefill-vs-decode/)** — the GPU primers are done. Now the engine takes over.

