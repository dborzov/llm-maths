---
title: "Compressed Sparse Attention (V4)"
description: "DeepSeek V4's first attention mode. Compress every m=4 tokens into one entry, then run DSA's top-k selection over the compressed entries. The indexer queries share the same latent vector as the main queries. This is the chapter where we read Figure 3 of the V4 paper, line by line."
topics: [attention, sparse-attention, deepseek, csa, v4]
tags: [csa, deepseek-v4, compressed-sparse-attention, lightning-indexer, token-compression, mqa]
theme: cream
math: true
draft: false
date: 2026-05-16T10:00:00-04:00
issue: 7
weight: 70
techKind: mainline
techNode: csa
header: default.png
---

## Figure 3, As Promised

This chapter is built around one diagram — Figure 3 of the V4 technical report. Look at it once, then we will spend the rest of the chapter taking it apart.

{{< figure src="/llm-maths/figures/07-sparse-lab/v4-paper/deepseek-v4-figure3-csa.webp"
           alt="DeepSeek V4 paper Figure 3: Core architecture of Compressed Sparse Attention. Hidden states of KV tokens enter at the bottom, go through Token-Level Compressors into Compressed KV Entries. A Lightning Indexer (dashed box, right side) produces Index Scores from compressed indexer keys and indexer queries. A Top-k Selector picks Selected Compressed KV Entries. Sliding Window KV Entries enter from the left. All three feed into a Concatenation and then Shared Key-Value Multi-Query Attention."
           caption="**Figure 3 of the DeepSeek-V4 paper.** Core architecture of CSA. The KV cache of every $m=4$ tokens is compressed into one entry (the **Token-Level Compressor**, blue triangle, bottom). The query token's hidden state is projected to indexer queries (the **Lightning Indexer**, dashed box on the right) which score every compressed KV entry. A **Top-k Selector** picks the highest-scoring $k$ compressed entries. A small **Sliding Window** branch (left) supplies recent uncompressed entries for local detail. Everything feeds a **Shared Key-Value Multi-Query Attention** at the top — meaning one key/value vector is shared across all $n_h$ query heads."
           credit="Reproduced from DeepSeek-AI, DeepSeek-V4 Technical Report (2026), Fig. 3." >}}

## The Two Token-Level Compressors

[From the paper:

$$C^a = H \cdot W^{aKV}, \quad C^b = H \cdot W^{bKV}$$
$$Z^a = H \cdot W^{aZ}, \quad Z^b = H \cdot W^{bZ}$$

These are four linear projections of the hidden states $H \in \mathbb{R}^{n \times d}$. $C^a$ and $C^b$ are two parallel streams of KV entries; $Z^a$ and $Z^b$ are the compression weights for each stream.

Then for compressed index $i$:

$$[S^a_{mi:m(i+1)-1};\; S^b_{m(i-1):mi-1}] = \text{Softmax}_{\text{row}}([Z^a_{mi:m(i+1)-1} + B^a;\; Z^b_{m(i-1):mi-1} + B^b])$$

$$C^{\text{Comp}}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j + \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j$$

The compressed entry $C^{\text{Comp}}_i$ is a softmax-weighted blend of $2m$ raw entries — $m$ from the $a$ stream centered on block $i$, and $m$ from the $b$ stream centered on block $i-1$. The "overlapped" compression means adjacent compressed entries share their input window — useful for smooth content transitions.

Net effect: the sequence length shrinks from $n$ to $n/m$.]

[See the [token compression primer](../12-token-compression/) for the geometry of why the softmax-weighted average is the right operator here.]

## The Lightning Indexer Returns

[From the paper:

$$c^Q_t = h_t \cdot W^{DQ}$$
$$[q^{I,1}_t; q^{I,2}_t; \ldots; q^{I,n^I_h}_t] = q^I_t = c^Q_t \cdot W^{IUQ}$$

Each query token produces:
1. A *compressed* query latent $c^Q_t$ of dimension $d_c$ (DeepSeek-V4-Pro: $d_c = 1536$, V4-Flash: $d_c = 1024$).
2. From that latent, $n^I_h$ indexer query heads of dimension $c_I = 128$ each.

The index score is:

$$I_{t,s} = \sum_{h=1}^{n^I_h} w^{I,h}_t \cdot \text{ReLU}(q^{I,h}_t \cdot K^{\text{IComp}}_s)$$

A weighted sum of ReLU'd dot products. The ReLU is the *only* non-linearity in the scoring path — this is what makes the indexer cheap and FP4-quantizable.]

## The Top-k Selector

[From the paper:

$$\mathcal{C}^{\text{SprsComp}}_t = \{C^{\text{Comp}}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}$$

V4-Pro: $k = 1024$. V4-Flash: $k = 512$. Each query attends to that many compressed KV entries — which means each query effectively attends to $k \cdot m$ raw tokens.

V4-Pro with $k=1024$ and $m=4$: each query effectively reaches **4,096 underlying tokens**. Out of a 1M-token context, that is **0.4%**.]

[Forward-link to the [hard top-k primer](../13-topk-and-routing/) for the differentiable training of this discrete selector.]

## Shared Key-Value MQA

[From the paper:

$$[q_{t,1}; q_{t,2}; \ldots; q_{t,n_h}] = q_t = c^Q_t \cdot W^{UQ}$$

The same latent $c^Q_t$ that produced the indexer queries *also* produces the main attention queries. One latent, two outputs.

Then attention itself is Multi-Query Attention (Shazeer 2019): **one** key vector and **one** value vector serve all $n_h$ query heads.

$$o_{t,i} = \text{CoreAttn}(\text{query}=q_{t,i},\; \text{key}=\mathcal{C}^{\text{SprsComp}}_t,\; \text{value}=\mathcal{C}^{\text{SprsComp}}_t)$$

The compressed entries serve as both keys and values. Same vector for both. This is what "Shared Key-Value Multi-Query Attention" in the diagram means.]

## Grouped Output Projection

[Direct from paper. $n_h$ query heads (V4-Pro: 128; V4-Flash: 64) produce $n_h$ output vectors of dimension $c=512$. Naively projecting $c \cdot n_h = 65{,}536$ floats down to $d=7168$ is a 470M-param weight. Instead: group the $n_h$ outputs into $g$ groups (V4-Pro: 16; V4-Flash: 8), project each group through an intermediate $d_g=1024$, then sum the group outputs into the final $d$-dim residual update.]

## V4-Pro Hyperparameters In One Table

| Hyperparameter | V4-Pro | V4-Flash | What it controls |
|---|---|---|---|
| Compression rate $m$ | 4 | 4 | Tokens per compressed entry |
| Top-k | 1024 | 512 | Compressed entries attended per query |
| Indexer query heads $n^I_h$ | 64 | 64 | Indexer parallelism |
| Indexer head dim $c_I$ | 128 | 128 | Indexer expressivity |
| Query compression dim $d_c$ | 1536 | 1024 | Shared latent for indexer + main queries |
| Main query heads $n_h$ | 128 | 64 | Main attention parallelism |
| Main head dim $c$ | 512 | 512 | Main attention expressivity |
| Output groups $g$ | 16 | 8 | Grouped output projection |
| Sliding window $n_{\text{win}}$ | 128 | 128 | Local-only branch size |

[Values are from V4 paper section 4.2.1, *Model Setups*.]

## Why The Shared Latent Matters

[Crucial point. By computing indexer queries and main queries from the *same* $c^Q_t$, V4 makes a strong statement: the same "what is this query about" representation that runs the lightning indexer also runs the main attention.

This is the MLA absorption trick stretched one step further. In MLA, the latent $c_t$ folded K and V together. In CSA, the query latent $c^Q_t$ folds the indexer query and the main query together. Two birds with one projection.]

## The Sliding Window Branch

[Per the V4 paper, $n_{\text{win}} = 128$ recent uncompressed KV entries are concatenated alongside the top-k selected compressed entries. This is the same sliding-window survival pattern that recurs from Longformer through Mistral through NSA. The local-detail bandage that every method needs.]

## A Concrete Compute Bill

[Napkin math: V4-Pro at $T=1M$, single decode step.
- Indexer cost: $n^I_h \cdot c_I \cdot T/m = 64 \cdot 128 \cdot 250{,}000 \approx 2 \times 10^9$ FLOPs in FP4.
- Top-k: trivial.
- Main attention: $n_h \cdot c \cdot k = 128 \cdot 512 \cdot 1024 \approx 7 \times 10^7$ FLOPs.

Compare to V3.2 dense at the same T: $n_h \cdot c \cdot T = 6 \times 10^{10}$. That's ~900× more.

The lightning indexer is the bulk of the cost. The main attention is essentially free.]

## What CSA Did Not Do

[Setup for HCA. CSA still computes one index score per past compressed entry per query. At T=1M and m=4, that's 250K scores per query per layer per head. Even at FP4, this is the next bottleneck.

The next chapter shows what HCA does about it.]

## What To Remember

1. **CSA = compress-then-DSA.** Two stages. Compress every $m$ tokens into one entry; then run the lightning indexer + top-k over compressed entries.
2. **The compressor is softmax-weighted block pooling.** Two streams ($C^a$, $C^b$) overlapping by half a block.
3. **The query latent is shared.** $c^Q_t$ produces both indexer queries and main queries.
4. **MQA on the compressed side.** One K, one V per compressed entry, serves all $n_h$ heads.
5. **V4-Pro: m=4, top-k=1024 → effective reach 4096 raw tokens per query, ~0.4% of 1M context.**

**Continue to** → [Heavily Compressed Attention](../08-hca/) — the complement to CSA. Compress every $m'=128$ tokens into one. Don't bother with sparse selection. Run dense attention over the heavily compressed sequence. Includes Figure 4 of the V4 paper.
