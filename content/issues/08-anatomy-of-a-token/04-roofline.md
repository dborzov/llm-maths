---
title: "The Roofline"
description: "The roofline model tells you in one number — arithmetic intensity — whether a kernel is compute-bound or memory-bound. Decode attention sits at roughly 1 FLOP/byte, catastrophically to the left of the ridge point. That single fact predicts the shape of every optimization in modern LLM serving."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T10:30:00-04:00
issue: 8
weight: 40
techKind: primer
techNode: roofline
header: 04-roofline.webp
---


>
> The companion plot in [Issue 06's Bandwidth Wall](/llm-maths/issues/06-eviction-notice/11-bandwidth-wall/) already exists. Here the focus is the *roofline model itself* as a piece of intellectual machinery, not just one labelled dot on it.

## Anchor and Frame

- **History.** Sam Williams, Andrew Waterman, David Patterson at Berkeley, 2008. The roofline model was invented to reason about scientific HPC kernels on multicore CPUs. It survives as the canonical tool for thinking about LLM inference fifteen years later because the underlying truth — "compute or bandwidth, pick one bottleneck" — has not changed.
- **Anchor example:** plot the H100 roofline. Drop three dots onto it: (i) a 4K×4K matmul tile, (ii) a prefill forward pass, (iii) a decode forward pass. Show that they sit in radically different regimes.

## Outline

### Arithmetic Intensity, Defined

- The single number that decides everything: $I = \text{FLOPs} / \text{bytes moved}$.
- Worked examples: SAXPY ($I \approx 0.25$), dense matmul ($I \to N/3$ as $N$ grows), softmax-attention naive ($I \approx 1$).

### Building the Diagram

- Plot $I$ vs achievable FLOP/s on a log-log axis. Two ceilings:
  - Horizontal: peak compute.
  - Diagonal: peak bandwidth × $I$.
- The break-even point — *ridge point* — is at $I = \text{compute} / \text{bandwidth}$. For H100: $312 \times 10^{12} / 3.35 \times 10^{12} \approx 93$ FLOP/byte. For H200: ~75. For B200 FP8: ~150.

### Where The LLM Kernels Land

- **Prefill matmul** (large $T$): $I \approx D$ where $D$ is hidden dim → for D=8192, that's well past the ridge. Compute-bound.
- **Decode attention** ($T$ keys, 1 query): $I \approx 1$ FLOP/byte. ~93× below ridge. Catastrophically memory-bound.
- **Decode FFN/MLP**: $I \approx 1$ FLOP/byte (the weights have to be reloaded each step, multiplied by one token). Memory-bound.
- **Spec-decode verification of K candidate tokens**: $I \approx K$ FLOP/byte. *Roughly K times more useful work per HBM byte loaded.* Foreshadow [The Draft Trick](../15-speculative-decoding/).

### Pyplot

- Reuse and refine the existing plot from Issue 06 — but with three labeled dots: prefill, decode, decode+spec(K=4).

### The Forced Moves

- If you sit *under* the ridge, throwing more compute at the problem accomplishes nothing.
- The only levers that move you toward the ridge are:
  1. **Reduce bytes.** (Quantization → Issue 03; KV pruning → Issue 06; KV-cache MLA-style compression → Issue 05 ch.19.)
  2. **Increase FLOPs per byte loaded.** (Batching, speculative decoding, multi-token prediction.)
  3. **Move the byte once, use it many times.** (FlashAttention tiling — [→ ch.6](../06-flash-attention/).)
- Everything in this issue is one of those three.

## Connections

- ← [The Pyramid of Speed](../03-memory-hierarchy/) — the bandwidth term is HBM bandwidth, not abstract.
- → [Two Phases, Two Personalities](../07-prefill-vs-decode/) — the prefill/decode split is literally a left-of-ridge / right-of-ridge story.
- → [The Draft Trick](../15-speculative-decoding/) — the cleanest exploitation of the memory-bound regime.

## What To Remember

1. **Arithmetic intensity is destiny.** One number — FLOPs per HBM byte — predicts whether your kernel is compute-bound or bandwidth-bound.
2. **Prefill is right of the ridge; decode is far, far left.** Their performance profiles are not slightly different; they are categorically different.
3. **There are three levers to move toward the ridge** — reduce bytes, raise FLOPs-per-byte, or amortize a loaded byte over many operations. The rest of this issue is variations on those three themes.

**Continue to → [Launches Aren't Free](../05-cuda-graphs/)** — even when the GPU is busy with bytes, the CPU launching the kernels can still wreck your latency budget.

