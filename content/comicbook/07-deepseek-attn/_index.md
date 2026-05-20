---
title: "Sparse Attention: from DeepSeek MLA to V4's million-token CSA+HCA"
description: "How a Hangzhou quant fund quietly built four generations of sparse attention — from MLA's 30× memory cut to DSA's lightning indexer to V4's CSA+HCA hybrid running million-token context at 10% of yesterday's compute."
issue: 7
layout: issue-cover
theme: cream
math: false
header: 07-sparse-lab-cover.webp
date: 2026-05-16T09:00:00-04:00
---

## The Mystery

**Late September 2025.** A Chinese AI lab named **DeepSeek** publishes a model called **DeepSeek-V3.2-Exp** on Hugging Face. The release notes are five paragraphs long. The third paragraph mentions, almost in passing, that the model uses something called **DeepSeek Sparse Attention** — DSA — and that as a result, the inference API price has been cut by **roughly 50%** as of midnight Beijing time.

Six months later, **DeepSeek-V4** drops. The model card opens with a number nobody has put in print before: **one million tokens of context, supported natively, at 27% of V3.2's per-token FLOPs and 10% of V3.2's KV cache**. Two attention modes — Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) — interleave across the 61 transformer blocks. The headline plot is the one below: KV cache size vs. context length, with three curves and the V3.2 baseline crushed underneath the V4-Flash line at 1M tokens.

{{< figure src="/figures/07-deepseek-attn/v4-paper/deepseek-v4-figure1-bench-and-efficiency.webp"
           alt="DeepSeek V4 paper Figure 1: benchmark performance bars on the left, single-token FLOPs and accumulated KV cache size vs. token position on the right, comparing V3.2 against V4-Pro and V4-Flash."
           caption="**Figure 1 of the DeepSeek-V4 paper.** *Left:* benchmark performance of V4-Pro-Max vs frontier models on SimpleQA, HLE, Apex, Codeforces, SWE-Verified, Terminal-Bench 2.0, and Toolathlon. *Right:* single-token inference FLOPs and accumulated KV cache size as context grows from 0 to 1M tokens. At 1M context, V4-Pro uses **3.7×** fewer FLOPs and **9.5×** less KV cache than V3.2; V4-Flash uses **9.8×** fewer FLOPs and **13.7×** less KV cache. This is the picture the rest of the issue explains."
           credit="Reproduced from DeepSeek-AI, DeepSeek-V4 Technical Report (2026), Fig. 1." >}}

Sparse attention had been tried — and had failed in production — for six years before V3.2-Exp. Longformer, BigBird, Reformer, Linformer, Performer, Routing Transformer, FlashAttention's block-sparse mode, Mistral's sliding window, the H₂O eviction policy, StreamingLLM's attention sinks, SnapKV's clustering, DuoAttention's per-head heterogeneity. Each was credible on paper, each broke on a different production requirement. None of them shipped at the frontier.

Then DeepSeek shipped four attention variants in twenty-three months, each one stacked on top of the last, and the inference cost of a million-token context fell by an order of magnitude.

What did this lab figure out that ten years of efficient-transformer research did not?

{{% pullquote type="counter-intuitive" %}}
The DeepSeek V4 attention stack:

- **MLA** — shrinks the KV cache
- **DSA** — lightning indexer shrinks the score matrix
- **CSA** — compresses the sequence before scoring
- **HCA** — throws away even the sparsity bookkeeping
{{% /pullquote %}}

## The Lab Behind The Lab

To follow this story you need to know who is telling it. **DeepSeek-AI** is not, despite the model card, primarily a research lab. It is the AI arm of **High-Flyer** — 幻方量化, "Magic Square Quant" — a quantitative hedge fund founded in **Hangzhou** in **2016** by **Liang Wenfeng**. High-Flyer's business is statistical arbitrage on the Chinese A-share market. By 2022, they had accumulated approximately **10,000 NVIDIA A100 GPUs** for backtesting trading strategies. The 2022 US export controls capped what Chinese firms could buy from NVIDIA. High-Flyer was already past the cap.

Liang Wenfeng spun out **DeepSeek** as a separate entity in July 2023, with the hedge fund's GPUs and a charter that read, more or less: *use these for general AI research, publish openly, do not charge market rates*.

The arc of the bets they made next is the arc of this issue.

| Date | Release | The architectural move |
|---|---|---|
| **May 2024** | DeepSeek-V2 | **MLA** — Multi-head Latent Attention. D-axis surgery on the KV cache. Cache footprint cut ~30×. |
| **Dec 2024** | DeepSeek-V3 | 671B-param MoE on top of MLA. The platform. |
| **Jan 2025** | DeepSeek-R1 | Pure-RL reasoning fine-tune. Same MLA backbone. |
| **Feb 2025** | NSA paper | **Native Sparse Attention** — Hardware-Aligned and Natively Trainable Sparse Attention. The blueprint. |
| **Aug 2025** | DeepSeek-V3.1 | 128K context. Compute-bound at long context. The bottleneck the next paper attacks. |
| **Sept 2025** | DeepSeek-V3.2-Exp | **DSA** — the lightning indexer ships. ~50% API price cut. |
| **2026** | DeepSeek-V4 | **CSA + HCA + mHC + Muon**. Million-token native context. V4-Pro (1.6T params, 49B active) and V4-Flash (284B params, 13B active). |

Each release fixes a bottleneck the previous one exposed. MLA cut the *memory* footprint of the KV cache. That cut surfaced the *compute* bottleneck. NSA was the research proposal; DSA was the product. DSA's lightning indexer cut the compute of long-context attention. CSA stacked compression on top of sparsity. HCA showed that on some layers you don't even need sparsity — extreme compression alone is enough. V4 interleaves CSA and HCA, and the four moves compose into the curve in Figure 1.

## How To Read This Issue

Start with the [cold open](01-cold-open/) — it puts you in front of the V3.2-Exp release announcement on the morning it shipped, and walks through what changed in the inference economy that day. After that, the **tech tree** below is the table of contents. Each node is an article; arrows show which articles build on which.

{{< techtree name="issue07" >}}

The eight **pink mainline chapters** tell the story in order: MLA's first cut, the compute wall it left standing, a decade of sparse-attention attempts that didn't work, the NSA blueprint, the lightning indexer, then V4's CSA and HCA in their own chapters, then the hybrid pattern that interleaves them. The **yellow boss chapter** — *Decoupling Memory From Time* — is the destination, where the four architectural moves are stacked and the next decade of LLM inference economics falls out of the algebra.

The **cream primer chapters** give depth on demand: read them when a mainline chapter refers to them, or read all of them first if you prefer foundations before narrative. The two new primers for the V4 era are **Compressing m Tokens Into One** (the softmax-weighted block compressor used by both CSA and HCA) and **Mixed-Precision KV Cache** (FP4 for the indexer, FP8 for the KV body, BF16 for the rotary channel — V4's per-region precision policy).

If you are in a hurry: the mainline (pink nodes) is a six-hour read. If you want the full treatment — the algebraic identity behind MLA's absorption trick, the empirical evidence that attention has been sparse all along, the lineage of "tiny scorer + big executor" architectures from speculative decoding through KVzap to DSA — read everything in numerical order.

## What You Will Walk Away With

By the end of this issue you should be able to answer, with confidence and the napkin math to back it up:

### The architectural arc

- Why **MLA** attacks the *D-axis* of the KV cache and **GQA** attacks the *H-axis*, why the two compose, and what the algebraic absorption trick actually does.
- Why MLA solved the *memory* wall of long-context inference but left the *compute* wall — the $O(T^2)$ scaling of attention scores — completely intact.
- Why every sparse-attention attempt from 2019 to 2024 failed at least one of the four production criteria (training stability, kernel efficiency, quality preservation, decode compatibility).
- What **NSA** ("Native Sparse Attention") proposed and why the three-branch structure (compression / selection / sliding window) was the conceptual breakthrough.
- How the **lightning indexer** in DSA actually works: the bilinear scoring head, the top-$k$ selection rule, the integration with MLA's latent cache, the kernel-aligned block layout.
- How **CSA** (Compressed Sparse Attention) extends DSA by compressing every $m$ tokens into one entry *before* scoring, and how the indexer queries are derived from the same latent $c^Q_t$ as the main queries.
- How **HCA** (Heavily Compressed Attention) drops the sparsity machinery entirely, uses compression rate $m' \gg m$ (V4 picks $m' = 128$, $m = 4$), and runs full dense attention over the heavily compressed sequence.
- Why **interleaving CSA and HCA** layers works — and what that says about which kinds of computation benefit from sparsity vs. compression.

### The reusable patterns

- The **indexer pattern**: a tiny model scores items for a big model. Speculative decoding (Leviathan/Chen, 2023), KVzap surrogate (NVIDIA, 2026), MoE routers, DSA's lightning indexer, CSA's indexer-with-compression — they all live in this family.
- The **block compressor**: a softmax-weighted mixer that pools $m$ contiguous tokens into a single KV entry. Used identically in CSA and HCA, with only the compression rate $m$ differing.
- The **soft-then-hard top-k** trick for differentiable selection during training, and why straight-through estimators are the boring-but-correct answer.
- The **mixed-precision KV cache** policy: BF16 for RoPE channels, FP8 for the rest of the KV, FP4 for the indexer's QK path. The bit budget is allocated to where precision actually matters.
- The **learnable attention sink**: a per-head logit added to the softmax denominator, letting a head abstain from attending.

### The economics and what comes next

- Why the inference cost of long-context LLMs has historically scaled with $T^2$ (compute) plus $T$ (memory), and how MLA + DSA + CSA + HCA together collapse this to roughly $T \cdot k$ where $k$ is a constant chosen at training time.
- Why V4-Pro at 1M context uses only **27% of V3.2's per-token FLOPs and 10% of its KV cache**, and what this means for serving cost per query.
- The **decoupling theorem** at the boss capstone: once memory per token is constant in $T$ (MLA + compression) and compute per token is constant in $T$ (sparse selection + compression), the cost of *context* per inference query becomes linear in $T$. The economics of every product built on long-context LLMs — agents, code review, legal analysis, genomics pipelines — pivots on this single fact.

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
