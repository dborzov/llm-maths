---
title: "Five Paragraphs From Hangzhou"
description: "September 29, 2025. DeepSeek ships V3.2-Exp with a 13-line scoring function called the lightning indexer. The API price drops 50% overnight. Six years of sparse-attention research had failed to do that. What did this lab figure out?"
topics: [attention, long-context, sparse-attention]
tags: [deepseek, dsa, v3.2-exp, mla, history]
theme: cream
math: true
draft: false
date: 2026-05-16T09:00:00-04:00
issue: 7
weight: 10
techKind: mainline
techNode: cold-open
header: 01-cold-open-07.webp
---

## The Drop

**Monday, September 29, 2025. 03:17 UTC.** A new repository goes live on Hugging Face under the user **`deepseek-ai`**. The model name is `DeepSeek-V3.2-Exp`. The model card is short. Five paragraphs.

The first paragraph announces the model. The second paragraph describes the parameter count and the MoE configuration — same as DeepSeek-V3, released ten months earlier. The third paragraph is the one that, by sunrise on the US East Coast, will be quoted in every group chat in San Francisco:

> *We introduce **DeepSeek Sparse Attention (DSA)**, a fine-grained sparse attention mechanism that selects, for each query token, a small set of key-value pairs from the KV cache using a learned lightning indexer. As a result, the API price for v3.2-exp has been reduced by approximately 50% relative to v3.1, effective immediately.*

The fourth paragraph links to a technical report. The fifth links to weights, inference code, and a vLLM patch that has, somehow, already been merged upstream.

By 06:00 UTC the first independent benchmarks are circulating. By noon UTC every major inference team in the US has the model running. By the evening, OpenAI, Anthropic, and Google have engineers reading the technical report carefully. The price cut is real. The benchmarks are intact. The kernel is open-source and FlashAttention-compatible.

There has been a six-year arms race in academia for an "efficient transformer" — a way to make {{< wiki "attention" >}}attention{{< /wiki >}}'s $O(T^2)$ cost scale better. **None of the academic proposals shipped to production.** Longformer, BigBird, Reformer, Linformer, Performer, Routing Transformer, FlashAttention's block-sparse mode, Mistral's sliding window, H₂O's heavy-hitter eviction, StreamingLLM's attention sinks, SnapKV's clustering, DuoAttention's per-head heterogeneity — each was credible on paper, each broke on a different production requirement. A small Chinese lab nobody had heard of two years ago just made the entire conversation moot.

What did DeepSeek figure out?

## The Lab Behind The Lab

Before we follow the algorithm, follow the money.

**DeepSeek-AI** is, despite the model card, not primarily a research lab. It is the AI division of **High-Flyer** — 幻方量化, "Magic Square Quant" — a quantitative hedge fund founded in **Hangzhou** in **2016** by **Liang Wenfeng**. High-Flyer's business is statistical arbitrage on the Chinese A-share market. The job description is what an American would call "renaissance technologies for Shanghai": train enormous statistical models on enormous tick datasets, find correlations, place bets, repeat.

The relevant detail is that, between 2019 and 2022, High-Flyer accumulated approximately **10,000 NVIDIA A100 GPUs** — *for backtesting*. Not for AI research. For trading. By the time the **October 2022 US export controls** capped what Chinese firms could buy from NVIDIA, High-Flyer was already past the cap. They had the largest GPU cluster of any Chinese hedge fund. Then, two months later, ChatGPT launched.

Liang Wenfeng spun out **DeepSeek** as a separate entity in July 2023, with the hedge fund's GPUs and a charter that read, more or less: *use these for general AI research, publish openly, do not charge market rates*. The first products — DeepSeek-Coder, DeepSeek-LLM — landed in late 2023 and were forgettable. The lab was, by its own admission, learning.

Then they shipped MLA.

## A Seven-Bet Arc

Every release from DeepSeek since May 2024 has been an architectural bet on attention. The cumulative effect is what this issue is about, but the individual bets are worth listing now so the rest of the narrative has hooks to attach to:

| Date | Release | The bet |
|---|---|---|
| **May 6, 2024** | DeepSeek-V2 | **Multi-head Latent Attention** (MLA). Cache one short latent vector $c_t$ per token instead of full K and V. Read-time decompression made free by an absorption identity. Cache footprint cut ~30×. |
| **Dec 26, 2024** | DeepSeek-V3 | 671B-param MoE on top of MLA, trained for ~$6M of compute. Sets the world's open-weight quality bar. |
| **Jan 20, 2025** | DeepSeek-R1 | Pure-RL reasoning fine-tune of V3. Same MLA backbone. Demonstrates that the architecture composes with long-form chain-of-thought. |
| **Feb 16, 2025** | NSA paper | "**Native Sparse Attention** — Hardware-Aligned and Natively Trainable Sparse Attention". A research preprint. Three branches: compression, selection, sliding window. Not shipped in any model. Yet. |
| **Aug 12, 2025** | DeepSeek-V3.1 | 128K-token context. MLA scaled up. Compute-bound at long context — and the team admits it openly in the technical report. |
| **Sept 29, 2025** | DeepSeek-V3.2-Exp | **DSA** — DeepSeek Sparse Attention. The lightning indexer. 50% API price cut. |
| **2026** | DeepSeek-V4 | **CSA + HCA**. V4-Pro (1.6T params, 49B active) and V4-Flash (284B params, 13B active). One million tokens of native context. 27% of V3.2's per-token FLOPs at that length; 10% of the KV cache. |

Each release fixes a problem the previous one created. MLA shrank the KV cache so aggressively that the *memory* bottleneck moved off the critical path — and exposed, for the first time, that attention's actual cost-driver in long-context decoding was the *compute* of computing $T \cdot T$ score matrices, not the memory of storing them. The team's response was the NSA paper. The NSA paper's response, six months later in production form, was DSA. DSA's response, eight months after *that*, is V4 — which doesn't just iterate on DSA but introduces two distinct attention modes (**Compressed Sparse Attention** for sparsity layers and **Heavily Compressed Attention** for the rest) and interleaves them.

The arc is the point. No single one of these moves would have worked on its own. The architecture compounds.

## What This Issue Is Going To Do

The rest of this issue follows the architectural arc, with stops along the way for the mathematics that makes each move land. The full map is on the [cover](./), but here is the route at a glance.

### The first question: what was MLA, exactly?

[Chapter 2 — MLA: The First Cut](../02-mla-rewind/) — we already met MLA in [Issue 5 ch.19](/issues/05-microgpt-unfolded/19-mla/), where it lived as a microGPT variation. This time we go deeper on the *historical* MLA: the people who built it, the algebra that makes the absorption trick legal, and the choice DeepSeek made about RoPE that becomes, two papers later, the seed of the lightning indexer.

### The second question: if the cache is tiny, why is long-context still expensive?

[Chapter 3 — The Other Wall](../03-quadratic-wall/) — the napkin math: at 128K context, attention scores alone are a 16-billion-entry matrix per layer. Computing them dominates prefill. Storing them — wait, you don't store them, but you do store the KV cache they read from — dominates decode. MLA solved one of those. The other was waiting.

### The third question: hadn't sparse attention been tried already?

[Chapter 4 — Sparse Attention's Lost Decade](../04-sparse-detour/) — a forensic tour of the 2019-2024 "efficient transformer" literature, organized by which production requirement each method broke. The pattern that emerges is the pattern DeepSeek will eventually beat.

### The fourth question: what was NSA?

[Chapter 5 — The NSA Blueprint](../05-nsa-paper/) — February 2025. The DeepSeek paper that proposed *trainable* sparse attention with three composable branches. Why it was a research preprint and not yet a product, and what was missing.

### The fifth question: how does DSA actually work?

[Chapter 6 — Lightning Strikes Twice](../06-lightning-indexer/) — the lightning indexer mechanism in full, line-by-line. The bilinear score head, the top-$k$ rule, the kernel-aligned block layout, and the trick that integrates the whole thing with MLA's latent cache.

### The sixth and seventh questions: what does V4 do?

[Chapter 7 — Compressed Sparse Attention](../07-csa/) — V4's main attention mode. Compress every $m=4$ tokens into one entry *first*, then run DSA's top-$k$ selection over the compressed entries. The indexer queries share the same latent $c^Q_t$ as the main queries; the compressor is a softmax-weighted block mixer. Includes the official Figure 3 from the V4 paper.

[Chapter 8 — Heavily Compressed Attention](../08-hca/) — V4's complement mode. Compress every $m'=128$ tokens into one entry. Run dense attention. No sparse selection at all. Why does this work on some layers? Includes the official Figure 4 from the V4 paper.

### The eighth question: why interleave?

[Chapter 9 — The Hybrid Pattern](../09-hybrid-pattern/) — V4-Flash uses pure sliding window for the first two layers, then interleaves CSA and HCA. V4-Pro uses HCA for the first two layers, then interleaves. What does interleaving buy you that pure CSA or pure HCA does not?

### And the ninth, biggest question: what does this *mean*?

[Chapter 10 — Decoupling Memory From Time](../10-decoupling/) — the boss capstone. Once memory per token is constant in $T$ (MLA + compression) and compute per token is constant in $T$ (sparse selection + compression), the entire economic shape of long-context LLM inference changes. The unit cost of context becomes linear. The agents that need 10M-token working memories become buildable. The decade-long quadratic ceiling on the entire field lifts.

## What You Need To Know Going In

Three pieces of context will make the rest of the issue legible.

**First**, you should already be comfortable with [multi-head attention](/issues/05-microgpt-unfolded/08-attention/) at the microGPT level — Q, K, V, the score matrix, the softmax, the per-head output. If those are wobbly, take the detour through Issue 5 chapters 8 and 9 first.

**Second**, you should know that [MLA](/issues/05-microgpt-unfolded/19-mla/) exists as a memory-saving variant of attention. [Chapter 2](../02-mla-rewind/) recaps the algebra, but the issue assumes you already believe the absorption trick is real.

**Third**, you should know that the [KV cache](/issues/05-microgpt-unfolded/13-kv-cache/) is a 5-dimensional tensor of shape $(2, L, H, T, D)$, that $T$ — the token axis — is the one that grows without bound, and that [pruning along the T-axis](/issues/06-eviction-notice/) is its own field of research. Issue 6 was about evicting whole tokens from the cache. This issue is about *not attending* to most of the cache in the first place, which is a different move with a different mathematics.

## A Final Note Before We Start

DeepSeek publishes their papers in English, under permissive licenses, with code. There is a real argument that the lab's most important contribution to the field is not any single one of MLA, NSA, or DSA — it is the publication discipline. They write 50-page technical reports. They release the kernels. They merge upstream into vLLM. They make it possible to write articles like this one without speculation, because the work is on the table.

This is what we know about how DeepSeek bent attention. Let us follow the bend.

**Continue to** → [MLA: The First Cut](../02-mla-rewind/) — the May 2024 release, the algebraic identity behind the absorption trick, and the design decision about positional encoding that, three papers later, becomes the seed of the lightning indexer and CSA's compressed-key path.
