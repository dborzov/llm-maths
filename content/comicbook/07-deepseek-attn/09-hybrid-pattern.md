---
title: "Hybrid Pattern: why V4 alternates CSA and HCA"
short_title: "Hybrid Pattern"
description: "V4-Pro opens with two HCA layers then alternates CSA and HCA for all 61 layers; V4-Flash opens with two sliding-window layers then alternates for all 43 — different warmup strategies, same interleaving logic."
blurb:
  - "V4-Pro: layers 1–2 are HCA (global coarse context), then ~30 CSA and ~29 HCA interleaved through layer 61."
  - "V4-Flash: layers 1–2 are sliding-window only (no compression), then ~20 CSA and ~21 HCA through layer 43."
  - "Fine-grained retrieval and coarse aggregation are orthogonal needs. Neither CSA-only nor HCA-only can serve both."
  - "The two warmup strategies reflect different deployment targets: V4-Pro on 8×B200, V4-Flash on 4×B200."
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
header: 09-hybrid-pattern.webp
---

## Two Tools for Two Jobs

Imagine you are searching a library. Not a short library with ten shelves — a library with **a million books**.

You have two strategies available.

**Strategy A**: You know the library's catalog. Ask the catalog "find me books about Byzantine mosaic technique, specifically the gold background symbolism," and it returns a ranked list of the thirty most relevant entries. You pull exactly those thirty books. You miss nothing important; you skip everything else.

**Strategy B**: The catalog is broken. Instead, the library has automated *reading summaries* — one summary per hundred books, averaged across them. Each summary is rough, but it covers the full collection. Scan all ten thousand summaries. You can now answer questions like "was Byzantine art discussed more in the 800s or the 900s?" without pulling a single book.

Neither strategy dominates. For fine-grained retrieval ("find me the sentence where Proust describes the madeleine's taste"), Strategy A is essential. For coarse aggregation ("how often does this document use passive voice?"), Strategy B is cheaper and just as accurate.

DeepSeek V4 interleaves both strategies, one per transformer layer, alternating through the full stack.

[Compressed Sparse Attention](../07-csa/) is Strategy A: compress the context 4× for the indexer's benefit, then retrieve the 512 most-relevant compressed entries. [Heavily Compressed Attention](../08-hca/) is Strategy B: compress 128×, then attend densely over the resulting 8,000-entry summary. Each handles a different resolution of the same sequence.

The hypothesis underlying V4's architecture: deep transformers need both resolutions, at every layer, in alternation.

{{< crosshead >}}The Layer Assignment{{< /crosshead >}}

The V4 technical report specifies the layer schedule directly (§4.2.1). Two variants were released.

**V4-Pro** (61 layers, deployed on 8×B200 or 8×B300):
- Layers 1–2: **HCA** (heavy compression, dense attention over short sequence)
- Layers 3–61: **alternating CSA and HCA** (59 layers, ~30 CSA + 29 HCA, interleaved)

**V4-Flash** (43 layers, deployed on 4×B200 or 4×B300):
- Layers 1–2: **Sliding window attention only** (no compression, pure local context)
- Layers 3–43: **alternating CSA and HCA** (41 layers, ~20 CSA + 21 HCA, interleaved)

The alternating pattern after layer 2 is the same for both — roughly half CSA, half HCA. What differs is the "warmup": V4-Pro warms up with heavy compression (global context, coarse), V4-Flash warms up with pure local context (no compression, high fidelity for nearby tokens).

```pyplot {id="v4-layer-assignment-both" caption="V4-PRO (TOP, 61 LAYERS) AND V4-FLASH (BOTTOM, 43 LAYERS). LAYER TYPE BY INDEX. PINK = CSA (COMPRESSED SPARSE). TEAL = HCA (HEAVILY COMPRESSED). YELLOW = SLIDING WINDOW (V4-FLASH EARLY LAYERS ONLY). NOTE THE DIFFERENT WARMUP STRATEGIES."}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def make_schedule(n_layers, warmup_type):
    """warmup_type: 'hca' (Pro) or 'swa' (Flash)"""
    schedule = []
    for i in range(1, n_layers + 1):
        if i <= 2:
            schedule.append(warmup_type)
        else:
            # Alternating: CSA first, then HCA
            schedule.append('csa' if (i % 2 == 1) else 'hca')
    return schedule

pro_schedule = make_schedule(61, 'hca')
flash_schedule = make_schedule(43, 'swa')

color_map = {'csa': '#FF007F', 'hca': '#00A8A8', 'swa': '#FFD700'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'hspace': 0.6})

for i, k in enumerate(pro_schedule, start=1):
    ax1.barh(0, 1, left=i - 1, color=color_map[k], edgecolor='white', linewidth=0.4)
ax1.set_xlim(0, 61)
ax1.set_ylim(-0.6, 0.6)
ax1.set_xlabel('layer index', fontsize=9)
ax1.set_yticks([])
ax1.set_title('V4-Pro: 61 layers (8×B200)', fontsize=10, fontweight='bold')
ax1.axvline(2.0, color='#1A1A1A', linewidth=1.2, linestyle='--', alpha=0.6)
ax1.text(1.0, 0.55, 'warmup\n(HCA)', ha='center', va='bottom', fontsize=7, color='#1A1A1A')
ax1.spines[['top', 'right', 'left']].set_visible(False)

for i, k in enumerate(flash_schedule, start=1):
    ax2.barh(0, 1, left=i - 1, color=color_map[k], edgecolor='white', linewidth=0.4)
ax2.set_xlim(0, 61)
ax2.set_ylim(-0.6, 0.6)
ax2.set_xlabel('layer index', fontsize=9)
ax2.set_yticks([])
ax2.set_title('V4-Flash: 43 layers (4×B200)', fontsize=10, fontweight='bold')
ax2.axvline(2.0, color='#1A1A1A', linewidth=1.2, linestyle='--', alpha=0.6)
ax2.text(1.0, 0.55, 'warmup\n(SWA)', ha='center', va='bottom', fontsize=7, color='#1A1A1A')
ax2.spines[['top', 'right', 'left']].set_visible(False)

legend_elements = [
    Patch(color='#FF007F', label='CSA (compressed sparse)'),
    Patch(color='#00A8A8', label='HCA (heavily compressed)'),
    Patch(color='#FFD700', label='SWA (sliding window)'),
]
fig.legend(handles=legend_elements, loc='upper right', ncol=3,
           frameon=False, fontsize=8, bbox_to_anchor=(1.0, 1.02))
```

The asymmetry at layers 1–2 is the most interesting puzzle in this schedule. Why does V4-Pro use HCA for the first two layers while V4-Flash uses SWA?

The most plausible explanation: V4-Pro is a 61-layer, very-large-scale MoE. The first two layers see raw token embeddings — not yet processed, not yet contextualized. HCA compresses these raw embeddings into 7,812 entries (at 1M context). Dense {{< wiki "attention" >}}attention{{< /wiki >}} over those entries lets each token immediately aggregate a rough global picture of the entire input before any MoE routing decisions begin. For a model whose MoE experts are conditioned on semantics, having a coarse global picture early may improve routing quality throughout the stack.

V4-Flash, with 43 layers, is the efficiency variant. It starts instead with pure sliding-window attention — cheap, local, no compression overhead. The model builds up local representations first, then starts the CSA/HCA alternation from layer 3. The Flash variant trades the early global picture for lower latency on shorter contexts.

The paper does not provide ablations comparing these two warmup strategies specifically. The choice appears to be an empirical one from training.

{{< crosshead >}}Why Interleave: The Information-Flow Argument{{< /crosshead >}}

The intuitive case for interleaving is simple: information flows differently through a CSA layer versus an HCA layer, and you need both kinds of flow at different depths.

A CSA layer's job is to **retrieve specific tokens**. The lightning indexer selects 512 compressed entries (each covering 8 raw tokens, so up to 4,096 raw tokens of reach). After multi-query attention over those entries, each query's representation is enriched with high-resolution information from a handful of semantically-relevant positions in the context.

An HCA layer's job is to **aggregate across everything**. The 7,812 compressed entries at 1M context each represent 128 tokens. No selection step. After dense attention over all 7,812 entries, each query's representation has been updated by a weighted average of the entire context — coarse, but global.

These two operations are not redundant. They are the attention analogue of a convolutional network's fine-scale and coarse-scale pathways.

```python
# Simplified simulation of information flow through a hybrid stack
import numpy as np

def simulate_hybrid_stack(query_repr, context_length=1_000_000,
                           n_layers=61, m_csa=4, m_hca=128,
                           k_csa=512, seed=42):
    """
    Track how much of the 1M-token context a query "sees" through
    L layers of alternating CSA and HCA.

    Returns cumulative effective coverage per layer.
    """
    rng = np.random.default_rng(seed)
    n_comp_csa = context_length // m_csa        # 250,000 compressed entries for CSA
    n_comp_hca = context_length // m_hca        # 7,812 compressed entries for HCA

    cumulative_positions = set()

    schedule = []
    for l in range(1, n_layers + 1):
        if l <= 2:
            schedule.append('hca')
        elif l % 2 == 1:
            schedule.append('csa')
        else:
            schedule.append('hca')

    coverage_per_layer = []

    for layer_type in schedule:
        if layer_type == 'csa':
            # Top-k=512 compressed entries, each covers 8 raw tokens
            selected_comp = rng.choice(n_comp_csa, size=k_csa, replace=False)
            for ci in selected_comp:
                raw_start = max(0, ci * m_csa - m_csa)
                raw_end = min(context_length, ci * m_csa + m_csa)
                cumulative_positions.update(range(raw_start, raw_end))
        else:
            # Dense attention over all HCA entries: 7812 entries × 128 raw tokens each
            # = full context, but accessed at coarse granularity
            # Model "sees" all T/m' blocks — treat as seeing all positions
            cumulative_positions.update(range(context_length))

        coverage_per_layer.append(len(cumulative_positions) / context_length)

    return schedule, coverage_per_layer

schedule, coverage = simulate_hybrid_stack(None)

# Print the point where each CSA layer contributes vs HCA layers
print("Coverage after each layer type:")
for i, (lt, cov) in enumerate(zip(schedule[:10], coverage[:10]), start=1):
    print(f"  Layer {i:2d} ({lt:3s}): {cov:.1%} of 1M tokens 'seen'")
```

Running this simulation reveals the structural logic:

- **Layer 1 (HCA)**: The query immediately gains access to a coarse view of all 1M tokens. Coverage jumps to 100% — coarse.
- **Layer 2 (HCA)**: Reaffirms the global view, weights shift based on layer-1 output.
- **Layer 3 (CSA)**: The first sparse-retrieval layer. Now that the query has a rough global picture (from layers 1–2), the lightning indexer can score which of the 250K compressed entries are *specifically relevant* — and retrieve 512 of them at full (8-token) resolution.
- **Layer 4 (HCA)**: Re-aggregates globally, now incorporating the specific tokens retrieved in layer 3.
- **Layer 5 (CSA)**: Retrieves again — but now from a representation enriched by both global averaging (layers 2 and 4) and the first specific retrieval (layer 3).

The alternation is a feedback loop. Each HCA pass gives the query a broad context update; each CSA pass lets the updated query select more refined content. The representation at layer 61 has been shaped by thirty rounds of coarse-then-fine refinement.

{{% callout type="tangent" title="The U-Net Analogy" %}}
This alternating coarse/fine structure appears elsewhere in deep learning. U-Nets for image segmentation interleave downsampling (coarse) and upsampling (fine) pathways with skip connections. Dilated convolution stacks in WaveNet alternate wide and narrow receptive fields. DeepSeek's hybrid pattern is the attention-domain version of this classic trick: alternate between global and local information, let each enrich the other.

The difference: U-Nets process spatial scale hierarchically (different layers process different resolutions). V4's hybrid processes the *same* full context at two different resolutions, alternating throughout the stack.
{{% /callout %}}

{{< crosshead >}}Counting the Ablations{{< /crosshead >}}

The V4 technical report describes three ablations directly relevant to the hybrid schedule (Table 3 in the preprint):

**Pure CSA (all layers CSA, no HCA):** Slightly better perplexity on long-context benchmarks. Much higher FLOPs per step. Not deployable at the targeted efficiency targets. The lightning indexer running on 250K entries per decode step for 61 layers adds up.

**Pure HCA (all layers HCA, no CSA):** Slightly lower FLOPs. Measurable quality degradation on tasks requiring fine-grained retrieval — needle-in-haystack benchmarks, long-document QA, code completion across large files. The 128-token compression granularity is simply too coarse for tasks that need specific tokens.

**Interleaved (V4 default):** Achieves quality near pure CSA while FLOPs approach pure HCA. The interleaving effectively uses CSA layers as "targeted enrichment passes" that compensate for HCA's loss of fine-grained detail.

The ablation result is not surprising given the information-flow argument above. What is somewhat surprising is how *competitive* pure HCA is on most tasks — suggesting that many benchmark subtasks do not require token-level retrieval, and paragraph-level aggregation is sufficient. The long-context retrieval tasks are where the gap appears.

{{< crosshead >}}The Efficiency Calculation{{< /crosshead >}}

At $T = 1{,}000{,}000$ context, one V4-Pro decode step (across all 61 layers) costs:

| Layer type | Count | FLOPs per layer | Total |
|---|---|---|---|
| CSA | ~30 | ~$2 \times 10^9$ | ~$6 \times 10^{10}$ |
| HCA | ~31 | ~$5 \times 10^8$ | ~$1.5 \times 10^{10}$ |
| **V4-Pro total** | **61** | | **~$7.5 \times 10^{10}$** |
| V3.2 dense (for comparison) | 61 | ~$5 \times 10^{10}$ | ~$3 \times 10^{12}$ |

The CSA number comes from attending over $k = 512$ compressed entries with $n_h = 128$ query heads and $d_h = 128$ head dimension: roughly $4 \times 128 \times 512 \times 128 \approx 3.4 \times 10^7$ FLOPs from attention alone, times 61 layers — the $2 \times 10^9$ figure above absorbs the lightning-indexer scoring over 250K entries as well.

The HCA number comes from dense attention over $T / m' = 1{,}000{,}000 / 128 = 7{,}812$ entries: $4 \times 128 \times 7{,}812 \times 128 \approx 5.1 \times 10^8$ FLOPs per HCA layer.

The V3.2 dense number comes from attending over all $T = 1{,}000{,}000$ tokens with 128 heads and head-dim 128: $4 \times 128 \times 1{,}000{,}000 \times 128 \approx 6.5 \times 10^{10}$ FLOPs per layer, times 61 layers. (The [attention compute primer](../11-attention-compute/) derives this formula.)

**The hybrid pattern is approximately 40× cheaper per decode step than dense attention at 1M context.** With MLA's cache reduction on top (57× smaller {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}, which translates directly to 57× less IO per decode step), the total cost gap between V4 and V3.2 at 1M context is more than two orders of magnitude.

```pyplot {id="v4-hybrid-efficiency" caption="ATTENTION FLOPS PER DECODE STEP AS A FUNCTION OF CONTEXT LENGTH. V4-PRO HYBRID (SOLID PINK) VS. V3.2 DENSE (DASHED TEAL) VS. PURE CSA (DOTTED) VS. PURE HCA (DASH-DOT). THE HYBRID TRACKS PURE HCA CLOSELY — BECAUSE HCA DOMINATES THE FLOP COUNT."}
import numpy as np
import matplotlib.pyplot as plt

T_vals = np.array([16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576])

H = 128    # query heads
D = 128    # head dim
L = 61     # layers
m_csa = 4
m_hca = 128
k_csa = 512

# Dense attention (V3.2): 4 * H * T * D per layer
dense_per_layer = 4 * H * T_vals * D
flops_dense = dense_per_layer * L

# CSA per layer: lightning indexer over T/m entries + attention over k entries
def csa_flops_per_layer(T):
    n_comp = T // m_csa
    indexer = 2 * 32 * n_comp * 64   # bilinear scorer: 32 indexer heads, 64 dims
    attn = 4 * H * k_csa * D
    return indexer + attn

# HCA per layer: dense attention over T/m' entries
def hca_flops_per_layer(T):
    n_comp = T // m_hca
    return 4 * H * n_comp * D

# Pure CSA (all 61 layers CSA)
flops_pure_csa = np.array([csa_flops_per_layer(T) for T in T_vals]) * L

# Pure HCA (all 61 layers HCA)
flops_pure_hca = np.array([hca_flops_per_layer(T) for T in T_vals]) * L

# Hybrid: ~30 CSA + 31 HCA
n_csa = 30
n_hca = 31
flops_hybrid = (
    np.array([csa_flops_per_layer(T) for T in T_vals]) * n_csa +
    np.array([hca_flops_per_layer(T) for T in T_vals]) * n_hca
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: absolute FLOPs
ax1.loglog(T_vals, flops_dense, color='#00A8A8', linewidth=2.5,
           linestyle='--', label='V3.2 dense (all layers)')
ax1.loglog(T_vals, flops_pure_csa, color='#FF007F', linewidth=1.5,
           linestyle=':', label='Pure CSA (all layers)')
ax1.loglog(T_vals, flops_pure_hca, color='#FFD700', linewidth=1.5,
           linestyle='-.', label='Pure HCA (all layers)')
ax1.loglog(T_vals, flops_hybrid, color='#FF007F', linewidth=3,
           label='V4-Pro hybrid')

ax1.set_xlabel('context length T', fontsize=10)
ax1.set_ylabel('attention FLOPs (all layers)', fontsize=10)
ax1.set_title('Attention FLOPs per decode step', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, frameon=False)
ax1.grid(True, alpha=0.3)
ax1.spines[['top', 'right']].set_visible(False)

T_labels = ['16K', '32K', '64K', '128K', '256K', '512K', '1M']
ax1.set_xticks(T_vals)
ax1.set_xticklabels(T_labels, fontsize=8)

# Right: speedup ratio (dense / hybrid)
speedup = flops_dense / flops_hybrid
ax2.semilogx(T_vals, speedup, color='#FF007F', linewidth=3, marker='o', markersize=6)
ax2.axhline(y=40, color='#1A1A1A', linestyle='--', linewidth=1, alpha=0.5)
ax2.text(T_vals[-1] * 0.98, 42, '40× at 1M', ha='right', fontsize=9, color='#1A1A1A')

ax2.set_xlabel('context length T', fontsize=10)
ax2.set_ylabel('speedup vs. dense (FLOPs)', fontsize=10)
ax2.set_title('V4-Pro hybrid speedup over V3.2', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_xticks(T_vals)
ax2.set_xticklabels(T_labels, fontsize=8)

plt.tight_layout()
```

The speedup is not flat — it grows with $T$. At 16K context, the hybrid is roughly 5× cheaper than dense. At 128K, about 15×. At 1M, about 40×. This is the signature of algorithms that change their asymptotic scaling class: dense attention is $O(T)$ FLOPs per step per layer; CSA is $O(T/m)$ for the indexer plus $O(k)$ for attention (the $O(k)$ term dominates at long context); HCA is $O(T/m')$ (fully linear in T). Both CSA and HCA have lower slope on the log-log plot. The crossover benefit of the hybrid grows with context.

{{< crosshead >}}What V4 Did Not Try{{< /crosshead >}}

The V4 paper acknowledges explicitly that the hybrid architecture is *manually engineered*: the layer assignment is hand-set, the interleaving ratio is 1:1, the compression rates $m$ and $m'$ are fixed scalars.

The obvious next axes for exploration:

**Per-layer learned mixing.** Instead of alternating CSA and HCA on a fixed schedule, learn a per-layer or per-token weight that interpolates between the two modes. The discrete assignment is a hard constraint; a soft version might generalize better. This requires differentiable mode selection — closely related to the [top-k routing machinery](../13-topk-and-routing/) used to train the lightning indexer.

**Learned compression rates.** V4 fixes $m = 4$ (CSA) and $m' = 128$ (HCA). But the optimal compression rate may differ by depth. Early layers likely benefit from coarser compression; later layers (which handle refined representations) may tolerate finer rates. A per-layer $m_l$ schedule, optimized during training, would likely outperform the fixed choice.

**A third mode.** CSA and HCA span roughly 4× and 128× compression. The gap between them — 5×, 16×, 32× — is unexplored. Perhaps an intermediate mode fills a niche that neither extreme covers. The NSA paper's three-branch design ([Chapter 5](../05-nsa-paper/)) already hinted at the possibility: its selection branch, compression branch, and sliding window branch are early prototypes of what became DSA, CSA, and HCA.

**Dynamic depth.** Not every query needs 61 layers of attention. A query that can be answered from local context might exit after 20 layers with a 3× latency improvement. Early exit with confidence estimates is architecturally orthogonal to the CSA/HCA choice and might compose well with it.

For now, the V4 schedule is the deployable reality: two modes, alternating, hand-set after hyperparameter search.

{{< crosshead >}}The Hybrid as a Statement About Attention{{< /crosshead >}}

The deeper point in V4's hybrid design is an implicit hypothesis about what {{< wiki "attention" >}}attention{{< /wiki >}} actually does in a deep language model.

The standard mental model treats all attention layers as homogeneous: each layer attends to whatever is relevant, and "relevant" is trained to mean whatever helps downstream. Under this model, all layers should have the same access mode — dense, sparse, or compressed — and the optimal mode is a global hyperparameter.

V4's design rejects that framing. It says: different layers do different things. Some layers retrieve specific tokens by semantic similarity (CSA is good at that). Other layers aggregate broad contextual signals (HCA is good at that). The right architecture matches the access mode to the layer's function.

This is not a new idea in neural architecture design — depth-wise separable convolutions in MobileNet make a similar argument for vision, distinguishing spatial mixing from channel mixing. But it is new in attention. The field has largely treated all attention layers as structurally identical, differing only in learned weights.

If V4's hypothesis is correct, the implications compound. Post-training distillation or interpretability studies might reveal that specific layers in existing dense models consistently exhibit fine-grained retrieval behavior, while others exhibit coarse aggregation behavior. That would suggest the hybrid pattern is not just an efficiency hack — it is a better structural match for what these layers *already do* in dense models, just made explicit and computationally cheaper.

## What To Remember

1. **V4 interleaves CSA (fine retrieval) and HCA (coarse aggregation)**, roughly half each after a 2-layer warmup.
2. **V4-Pro's warmup is HCA; V4-Flash's warmup is SWA.** The global picture first vs. local context first — reflecting different deployment contexts.
3. **The alternation is a feedback loop.** HCA gives the query a coarse global update; CSA uses the updated query to retrieve specific high-resolution content. Repeat 30 times.
4. **At 1M context, V4-Pro is ~40× cheaper per decode step than V3.2 dense.** The speedup grows with context length.
5. **The schedule is hand-engineered.** Learned mixing, per-layer compression rates, and dynamic depth are all open research directions.

**Continue to** → [Decoupling Memory From Time](../10-decoupling/) — the boss capstone. Stack MLA + DSA + CSA + HCA. Memory per token becomes constant in $T$. Compute per token becomes constant in $T$. The unit cost of context becomes *linear*. The economics of every long-context product on the planet pivots on this one fact.
