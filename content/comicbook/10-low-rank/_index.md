---
title: "Low-Rank Magic: how a 1936 theorem powers LoRA and MLA"
description: "SVD's rank-r approximation theorem (Eckart-Young, 1936) is the same piece of math under LoRA's training-time fine-tuning and DeepSeek MLA's inference-time KV cache compression. This issue walks the lineage."
issue: 10
layout: issue-cover
theme: cream
math: false
header: default.webp
date: 2026-05-19T09:00:00-04:00
---

## The Mystery

**June 2021.** Edward Hu and seven coauthors at Microsoft Research upload a paper called *"LoRA: Low-Rank Adaptation of Large Language Models"* to arXiv. The headline number: GPT-3 175B fine-tuned with **10,000× fewer trainable parameters** and roughly **3× less GPU memory**, at essentially zero accuracy cost. Within eighteen months, every open-source fine-tuning recipe on the internet — Alpaca, Vicuna, the entire Hugging Face PEFT library — is using LoRA by default.

**May 2024.** A Chinese lab called DeepSeek-AI uploads a 236-billion-parameter mixture-of-experts model. The buried headline: inference is roughly **30× cheaper** on the KV cache, because a new architecture called **Multi-head Latent Attention** routes the K and V vectors of every layer through a narrow shared bottleneck before writing them to the cache. The technique enables a million-token context at a tenth of the prior cost.

Two papers, three years apart, two different parts of the LLM pipeline — fine-tuning on one end, inference on the other. Different teams. Different countries. Different objectives. And yet, when you stare at the math on the whiteboard for both, you see **the exact same trick**: replace a big matrix $W$ with the product of two skinny matrices $BA$ where $A$ and $B$ live in a tiny shared dimension $r$.

The trick is older than LLMs, older than backprop, older than computer science. It was proven optimal in **1936** by two Yale statisticians — Carl Eckart and Gale Young — for an entirely different problem: how do you best approximate a noisy data matrix with one of lower rank?

This issue walks the lineage. We will start in 1936 with Eckart-Young's theorem, follow the math through three decades of forgotten applications, watch it suddenly reappear in 2020 when Armen Aghajanyan proved that fine-tuning happens in a tiny subspace, watch LoRA productionalize that observation in 2021, and finally watch DeepSeek bake the same factorization directly into the architecture of the transformer in 2024.

## How To Read This Issue

Start with the [cold open](01-cold-open/) — it puts you in front of the LoRA paper's release on the morning it dropped, walks through what changed in fine-tuning economics that week, and connects the dots to MLA three years later. After that, the **tech tree** below is the table of contents.

{{< techtree name="issue10" >}}

The mainline (pink) is a three-chapter arc: **LoRA** (the original training-time trick) → **MLA, re-read as a LoRA** (the architectural pivot) → **the unified low-rank thesis** (the boss capstone).

The cream primers give depth on demand. **SVD & Eckart-Young** is the foundation — read it first if linear algebra feels rusty. **The Low-Rank Hypothesis** documents the empirical observation that trained neural-network matrices have rapidly decaying singular values. **Intrinsic Dimension** unpacks the 2020 Aghajanyan paper that *proved* fine-tuning is low-rank. **The LoRA Zoo** is a field guide to QLoRA, DoRA, AdaLoRA and friends.

Two **external cross-references** point at existing chapters elsewhere on the site: the [microGPT MLA chapter](/comicbook/05-microgpt/19-mla/) gives the bottom-up implementation view, and [Issue 7 — Sparse Attention](/comicbook/07-deepseek-attn/) covers what DeepSeek built on top of MLA. The bridge chapter (ch.7) cross-references both heavily; read them as a triangle.

## What You Will Walk Away With

By the end of this issue you should be able to answer, with confidence and the napkin math to back it up:

- **What SVD actually proves.** Eckart-Young: among all rank-$r$ matrices, the one that minimizes $\Vert W - W_r \Vert_F$ is the truncated SVD $U_r \Sigma_r V_r^\top$. The keep-the-top-$r$-singular-values recipe is *provably* optimal — not a heuristic.
- **Why neural networks are low-rank in practice.** When you take a trained transformer's weight matrices and run SVD, the singular values fall off a cliff. The empirical fact, the candidate explanations (lottery ticket, NTK, optimization bias), and what fails to explain it.
- **What "intrinsic dimension" means.** Aghajanyan et al, 2020: fine-tuning BERT-large from the pre-trained checkpoint to MRPC accuracy can be done in a subspace of dimension ~200. The 110-million-parameter update lives in a 200-dimensional sub-manifold. This *is* LoRA, two years before the LoRA paper.
- **The LoRA factorization.** Replace a weight update $\Delta W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ with $BA$ where $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$. Parameter count drops from $d_\text{out} d_\text{in}$ to $r(d_\text{out} + d_\text{in})$. For Llama2-7B with $d=4096$ and $r=16$, that's a 128× shrink per layer. At inference, you fold $BA$ back into $W$; the runtime cost is zero.
- **Why LoRA saves *optimizer* memory, not just *weight* memory.** Adam needs first and second moments per trainable parameter, in fp32. If only $0.1\%$ of parameters are trainable, optimizer state shrinks by $1000\times$. This is the dominant savings on the training side, not the weight savings.
- **The LoRA → MLA pivot.** LoRA factors the *weight update* $\Delta W$. MLA factors the *weight itself* $W_K$, baking the factorization into the architecture from initialization. Same algebra, different scope, different stage of the model life-cycle. Once you see this, the absorption trick and the rope-vs-nope split fall out for free.
- **The unified thesis.** Every "compression" trick in modern LLMs is, at root, an exploitation of the fact that real-world weight matrices and activation matrices are *low-rank*. Quantization exploits low-rank-in-magnitude. Pruning exploits low-rank-in-support. LoRA exploits low-rank-in-direction. MLA bakes that exploitation into the architecture. The boss chapter draws the family tree.

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
