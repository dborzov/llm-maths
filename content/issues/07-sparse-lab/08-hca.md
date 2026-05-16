---
title: "Heavily Compressed Attention (V4)"
description: "V4's complement mode. Compress every m'=128 tokens into one entry. Drop the sparse-selection machinery entirely. Run dense attention over the heavily compressed sequence. Why this works on roughly half the layers — and what it tells us about what attention actually does."
topics: [attention, deepseek, hca, compression, v4]
tags: [hca, deepseek-v4, heavily-compressed-attention, token-compression, dense-attention]
theme: teal
math: true
draft: false
date: 2026-05-16T10:10:00-04:00
issue: 7
weight: 80
techKind: mainline
techNode: hca
header: default.png
---

## Figure 4

[Open with the diagram.]

{{< figure src="/llm-maths/figures/07-sparse-lab/v4-paper/deepseek-v4-figure4-hca.webp"
           alt="DeepSeek V4 paper Figure 4: Core architecture of Heavily Compressed Attention. Hidden states of KV tokens go through a single Token-Level Compressor with rate m', producing Heavily Compressed KV Entries. These feed directly into Shared Key-Value Multi-Query Attention along with the query and a Sliding Window branch. No indexer. No top-k selector."
           caption="**Figure 4 of the DeepSeek-V4 paper.** Core architecture of HCA. Compared to CSA (previous chapter), HCA has *no lightning indexer*, *no top-k selector*, *one compressor stream* (instead of two), and a much heavier compression rate $m' = 128$ (vs. $m=4$ in CSA). What remains: token-level compressor → dense attention over compressed entries → sliding window for local detail. Half the diagram of CSA. The sequence length collapses to $T/128$ before attention even runs."
           credit="Reproduced from DeepSeek-AI, DeepSeek-V4 Technical Report (2026), Fig. 4." >}}

## The Heavy Compression

[From the paper, the HCA compressor is simpler than CSA's:

$$C = H \cdot W^{KV}$$
$$Z = H \cdot W^Z$$
$$S_{m'i:m'(i+1)-1} = \text{Softmax}_{\text{row}}(Z_{m'i:m'(i+1)-1} + B)$$
$$C^{\text{Comp}}_i = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j$$

One compressor stream (no $C^a$, $C^b$ overlap). Compression rate $m' = 128$. The sequence collapses to $T/128$.

For $T = 1\text{M}$: $T/m' = 7813$ entries. Dense attention is now $O((T/m')^2) = O(6 \times 10^7)$ scores per layer per head. Tractable.]

## No Indexer, No Top-k

[Why does HCA drop the sparse-selection machinery? Because at $m'=128$, the compressed sequence is short enough that dense attention is already affordable. There is no need to select; you can attend over all of it.

This is a key insight. Sparsity and compression are *substitutes*, not complements, beyond a certain compression ratio. CSA uses both because $m=4$ leaves a long sequence. HCA uses only compression because $m'=128$ makes the sequence short enough for dense.]

## The Same Query Plumbing

[From the paper, HCA reuses the latent-query infrastructure:

$$c^Q_t = h_t \cdot W^{DQ}$$
$$[q_{t,1}; \ldots; q_{t,n_h}] = q_t = c^Q_t \cdot W^{UQ}$$

Note: HCA does *not* share the latent with an indexer (there is no indexer). But the latent shape and the up-projection to query heads are identical to CSA's.]

```pyplot {id="csa-vs-hca-cost" caption="CSA AND HCA HAVE DIFFERENT COMPUTE CURVES. CSA: O(T/m × d_I) FOR INDEXER + O(k × d_h × n_h) FOR ATTENDED. HCA: O((T/m')² × d_h × n_h). AT m'=128, HCA IS CHEAPER UP TO T~256K, AND CSA WINS AT T~1M."}
import numpy as np
import matplotlib.pyplot as plt

T = np.logspace(np.log10(1024), np.log10(1_048_576 * 4), 200)
n_h = 128
c = 512
d_I = 128
n_I = 64
m = 4
m_prime = 128
k = 1024

# CSA cost per decode step: indexer scores + attended dense
csa_indexer = n_I * d_I * (T / m)
csa_attend = n_h * c * k
csa_total = csa_indexer + csa_attend

# HCA cost: dense attention over T/m' entries
hca_total = n_h * c * (T / m_prime)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.loglog(T / 1000, csa_total / 1e6, color='#FF007F', linewidth=2.5, label='CSA total')
ax.loglog(T / 1000, hca_total / 1e6, color='#FFD700', linewidth=2.5, label='HCA total')
ax.loglog(T / 1000, csa_indexer / 1e6, color='#FF007F', linewidth=1.2, linestyle='--', alpha=0.6, label='  CSA indexer alone')
ax.loglog(T / 1000, csa_attend * np.ones_like(T) / 1e6, color='#FF007F', linewidth=1.2, linestyle=':', alpha=0.6, label='  CSA attend alone')
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('attention MFLOPs per decode step')
ax.set_title('CSA scales with T (indexer-dominated); HCA scales linearly via compression', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.spines[['top','right']].set_visible(False)
```

## Why Use Both?

[Interleaving CSA and HCA. From the paper:
- V4-Flash: layers 1-2 use pure sliding window. Layers 3-43 alternate CSA and HCA.
- V4-Pro: layers 1-2 use HCA. Layers 3-61 alternate CSA and HCA.

The pattern: heavy compression where dense aggregation is good enough; sparse selection where fine-grained retrieval matters. The next chapter, [the hybrid pattern](../09-hybrid-pattern/), unpacks the interleaving logic in detail.]

## What HCA Throws Away

[Important caveat: heavy compression *destroys* token-level identity. A single compressed entry at $m'=128$ represents an entire paragraph. You can no longer retrieve a specific quoted phrase from inside that paragraph through this branch.

This is why HCA needs the sliding window for local detail (last 128 raw tokens stay uncompressed). And it is why HCA cannot be used on *every* layer — at least one layer needs fine-grained retrieval, which is CSA's job.

The interleaving is doing a real division of labor: CSA layers preserve token-level retrievability; HCA layers do paragraph-level aggregation.]

## A Concrete Compute Bill

[Same napkin math as the previous chapter, now for HCA.
- V4-Pro at $T=1M$, single decode step, single HCA layer:
  $n_h \cdot c \cdot (T/m') = 128 \cdot 512 \cdot 7813 \approx 5 \times 10^8$ FLOPs.

Compare to V3.2 dense at the same T: $\sim 6 \times 10^{10}$. HCA is ~120× cheaper per layer.

Compare to CSA (from previous chapter): $\sim 2 \times 10^9$ FLOPs. HCA is ~4× cheaper than CSA per layer.

This is why V4 spreads HCA across roughly half the layers: it is structurally cheaper.]

## What To Remember

1. **HCA = compress, then dense attention.** No sparse selection, no indexer.
2. **Compression rate $m' = 128$, vs. CSA's $m = 4$.** Sequence collapses to $T/128$.
3. **HCA destroys token-level identity by design.** It does paragraph-scale aggregation. Pair it with CSA (token-scale) and sliding window (local).
4. **Cheaper per layer than CSA.** Half the diagram, ~4× fewer FLOPs at 1M context.

**Continue to** → [The Hybrid Pattern](../09-hybrid-pattern/) — why interleave CSA and HCA layers, what each does for the model's information flow, and how V4's two variants (Pro and Flash) deploy the same recipe with different layer budgets.
