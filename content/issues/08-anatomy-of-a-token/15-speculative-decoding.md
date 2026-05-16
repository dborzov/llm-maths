---
title: "The Draft Trick"
description: "Decode is so bandwidth-bound that verifying 32 candidate tokens costs the same memory pass as verifying 1. Run a small draft model to speculate several tokens ahead, then let the target model verify all of them in a single forward pass — a 2–3× throughput gain if the draft is even modestly accurate."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T16:00:00-04:00
issue: 8
weight: 150
techKind: mainline
techNode: speculative-decoding
header: default.png
---


## Anchor and Frame

- **History line.** Leviathan et al., DeepMind, *"Fast Inference from Transformers via Speculative Decoding"* (Nov 2022). Chen, Borgeaud et al. (DeepMind, Feb 2023). Then SpecInfer, Medusa, EAGLE, EAGLE-2, EAGLE-3, P-EAGLE, MLPSpeculator… a *family* of methods.
- **Anchor case.** Llama-3.1-8B decode at batch 4 on H100 ITL: 18 ms. EAGLE-3 speculation, $K=4$ draft tokens, ~70% acceptance: ITL 7.8 ms. **2.3× speedup on a workload the GPU was barely using.**

## Outline

### Why It Works — The Roofline Re-Stated

- Decode is at $I \approx 1$ FLOP/byte. The HBM bus is the bottleneck.
- *Verifying* $K$ candidate tokens in one forward pass costs $K\times$ the FLOPs but only $1\times$ the HBM bandwidth for weights (you load the weights once and use them on $K$ tokens). Arithmetic intensity rises to $\approx K$ FLOP/byte.
- As long as $K < \text{ridge point}$ (~93), the pass is still memory-bound and *nearly free per extra token*.
- **The only catch:** you have to *guess right* about which $K$ tokens to verify.

### The Two-Model Setup

- Target model $M$ (the real, slow one). Draft model $m$ (small, fast, hopefully aligned).
- Step: draft samples $K$ tokens autoregressively. Target verifies all $K$ in one forward pass. For each position, target accepts if $r < p_M(t)/p_m(t)$ (rejection sampling). On rejection: emit a corrected token from $p_M - p_m$, discard remaining drafts.
- This is *provably exact*: the output distribution equals the target's own. Free speedup, zero quality loss (modulo numerical noise).

### The Mechanic Inside vLLM

- The scheduler treats the speculation as a single multi-token request. Block allocation accounts for $K$ provisional tokens. On rejection, the unused blocks are released. Foreshadow: this composes with prefix caching, chunked prefill, paged attention with no extra machinery.

### EAGLE & Friends

- **Medusa** (2024) adds $K$ extra prediction heads to the target model itself — no separate draft model. Tree attention verifies all combinations.
- **EAGLE-1/2/3** (2024–25) trains a tiny autoregressive draft head that lives in feature space, fed by the target's hidden states. Gets to ~70% token acceptance with 1% of target compute.
- **MTP / Multi-Token Prediction** (DeepSeek V3, 2024) bakes the draft into training.

### The Acceptance Curve

- Pyplot: ITL vs $K$ for different acceptance rates (0.5, 0.7, 0.9). There's an optimum: too small a $K$ underuses the memory pass; too large eats latency on guesses that get rejected.

### When Spec Decode *Doesn't* Help

- **Heavy batches.** As batch grows, arithmetic intensity grows naturally and the headroom disappears. Spec decode is a single-/few-stream win.
- **Compute-bound steps.** Right of the ridge, more FLOPs are not free anymore.

## Connections

- ← [The Roofline](../04-roofline/) — the chapter that earns this one its rent.
- ← [The Token Budget](../14-scheduler/) — the scheduler dispatches the verify pass.
- ← [Borrowing from 1965](../10-paged-attention/) — block allocation for $K$ provisional tokens.
- Cross-link: → [Issue 06](/llm-maths/issues/06-eviction-notice/) — KV pruning gives spec decode more room by freeing KV memory.

## What To Remember

1. **Verifying K candidates costs ~1× the HBM pass.** That's the whole trick. Memory-bound regime makes extra FLOPs cheap.
2. **Rejection sampling makes it exact.** The output distribution is provably identical to the target's. No quality trade-off, just engineering complexity.
3. **The best draft model lives inside the target model.** EAGLE/Medusa/MTP all share weights or features with the target, getting both speed and high acceptance rate.

**Continue to → [Splitting the Model](../16-tp-pp/)** — speculative decoding is a single-machine optimization. Splitting the model across machines opens a whole different scaling axis.

