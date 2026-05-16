---
title: "The Hybrid Pattern (V4)"
description: "V4 interleaves CSA and HCA layers. V4-Flash uses sliding window for the first two layers, then alternates. V4-Pro uses HCA for the first two layers, then alternates. What does the interleaving buy you that pure CSA or pure HCA does not?"
topics: [attention, deepseek, hybrid, v4]
tags: [hybrid-attention, csa, hca, deepseek-v4, layer-mixing, sliding-window]
theme: cream
math: true
draft: false
date: 2026-05-16T10:20:00-04:00
issue: 7
weight: 90
techKind: mainline
techNode: hybrid-pattern
header: default.png
---

## The Architecture, Once More

[Show Figure 2 of the V4 paper — the overall stack. The "CSA / HCA" box is the layer that alternates.]

{{< figure src="/llm-maths/figures/07-sparse-lab/v4-paper/deepseek-v4-figure2-architecture.webp"
           alt="DeepSeek V4 paper Figure 2: overall architecture. Transformer blocks repeat L times. Each block has a CSA/HCA attention layer feeding through mHC (Manifold-Constrained Hyper-Connections) residual mixing, then a DeepSeekMoE feed-forward layer. MTP modules feed the prediction head. Embedding and input tokens at the bottom."
           caption="**Figure 2 of the DeepSeek-V4 paper.** Overall architecture. The transformer block uses Pre-Block Mixing → CSA/HCA → Post-Block Mixing → mHC residual update → Pre-Block Mixing → DeepSeekMoE → Post-Block Mixing → mHC residual update. The mHC (Manifold-Constrained Hyper-Connections) and Muon optimizer changes are out of scope for this attention-focused issue; the [V4 paper §2.2](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) covers them. What matters for us is the rhythm of the CSA/HCA layer assignment."
           credit="Reproduced from DeepSeek-AI, DeepSeek-V4 Technical Report (2026), Fig. 2." >}}

## The Layer Assignment

[Direct from V4 paper §4.2.1:

**V4-Flash** (43 layers):
- Layers 1-2: pure sliding window attention
- Layers 3-43: CSA and HCA in interleaved manner

**V4-Pro** (61 layers):
- Layers 1-2: HCA
- Layers 3-61: CSA and HCA in interleaved manner

Note the asymmetry. V4-Flash uses pure local attention for the first two layers; V4-Pro uses HCA. The paper does not fully justify this — likely Pro's larger MoE benefits from heavier compression earlier.]

```pyplot {id="v4-layer-assignment" caption="V4-PRO LAYER ASSIGNMENT, 61 LAYERS. PURPLE: HCA (HEAVY COMPRESSION). PINK: CSA (COMPRESSED SPARSE). THE EARLY-LAYER HCA BIAS IS A V4-PRO-SPECIFIC CHOICE."}
import numpy as np
import matplotlib.pyplot as plt

L = 61
layers = np.arange(1, L + 1)
# Stylized: layers 1-2 HCA, then alternating CSA-HCA-CSA-HCA-...
kind = np.array(['hca', 'hca'] + (['csa', 'hca'] * 30))[:L]

fig, ax = plt.subplots(figsize=(11, 2.5))
for i, k in enumerate(kind, start=1):
    color = '#FF007F' if k == 'csa' else '#00A8A8'
    ax.barh(0, 1, left=i - 1, color=color, edgecolor='white', linewidth=0.5)
ax.set_xlim(0, L)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('layer index')
ax.set_yticks([])
ax.set_title('V4-Pro layer assignment: HCA (teal) and CSA (pink) interleaved')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#FF007F', label='CSA'), Patch(color='#00A8A8', label='HCA')],
          loc='upper right', ncol=2, frameon=False)
ax.spines[['top','right','left']].set_visible(False)
```

## Why Interleave At All

[The architectural argument:

1. **CSA layers** preserve fine-grained token-level retrievability through the lightning indexer. They are expensive but they let the model "look up" specific tokens.

2. **HCA layers** do paragraph-scale aggregation. They are cheap but they lose token identity.

Both are needed. Information flow through a 61-layer stack benefits from alternation: aggregate (HCA), retrieve (CSA), aggregate, retrieve. Each CSA layer can pull specific tokens that the preceding HCA layer summarized.

This is structurally similar to U-Nets in vision — coarse-to-fine alternation. Or to mixture of experts: different layers, different specializations.]

## What The Ablations Say

[The V4 paper reports ablations (paraphrase, since I can't reproduce them exactly without re-running training):
- Pure CSA on all layers: small quality gain, much higher compute cost.
- Pure HCA on all layers: small compute saving, quality loss on long-context retrieval benchmarks.
- Interleaved: best of both, the V4 default.

The interleaving ratio (which layers are which) is mostly a hyperparameter. The paper picks a simple alternation; the choice could likely be optimized further per task.]

## The Efficiency Calculation

[Putting CSA + HCA together at 1M context, V4-Pro:
- 30 CSA layers × ~$2 \times 10^9$ FLOPs/step = $6 \times 10^{10}$
- 31 HCA layers × ~$5 \times 10^8$ FLOPs/step = $1.5 \times 10^{10}$
- Total attention per decode: ~$7.5 \times 10^{10}$ FLOPs

Compare to V3.2 dense attention at 1M: ~$3 \times 10^{12}$. **~40× cheaper attention per decode step.**

This is the napkin math behind Figure 1's "3.7× lower" per-token FLOPs claim. The 3.7× number is the *total* including MoE; the attention block alone is much cheaper. The MoE doesn't shrink in V4.]

## What V4 Did Not Try

[The paper is explicit that the architecture is "complex". The authors flag this in the conclusion. Likely future-iteration moves:
- Drop one of the two modes (probably HCA — it's cheaper but coarser, and maybe a tuned single mode does better).
- Replace the discrete interleaving with a per-token mode selector.
- Make $m$ and $m'$ learnable per layer.

For now: two modes, hand-set ratio, hand-set compression rates. It works.]

## What To Remember

1. **V4 interleaves CSA and HCA.** Roughly half each, starting after a 2-layer warmup.
2. **CSA = token-level retrieval. HCA = paragraph-level aggregation.** Different jobs.
3. **The combined cost is ~40× cheaper than V3.2 dense at 1M context.** Plus MoE savings on top.
4. **The interleaving ratio is a hyperparameter.** V4 picks the simplest schedule.

**Continue to** → [Decoupling Memory From Time](../10-decoupling/) — the boss capstone. Stack MLA + DSA + CSA + HCA. Memory per token becomes constant in $T$. Compute per token becomes constant in $T$. The unit cost of context becomes *linear*. The economics of every long-context product on the planet pivots on this one fact.
