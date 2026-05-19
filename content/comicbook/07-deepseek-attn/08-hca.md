---
title: "HCA: compress 128× and skip the indexer"
short_title: "HCA"
description: "Heavily Compressed Attention compresses every 128 tokens into one vector, then runs dense attention over the resulting short sequence — no indexer, no top-k selector, just 128× compression making dense attention already cheap."
blurb:
  - "m'=128: at 1M context, the compressed sequence is just 7,812 entries. Dense attention over that costs less than DSA's indexer."
  - "HCA has no lightning indexer, no top-k selector, one compressor stream — half the diagram of CSA."
  - "CSA retrieves the specific clause. HCA remembers the plot. A real model needs both, at every layer, in alternation."
  - "Sparsity and compression are substitutes beyond a certain ratio: compress enough, and selection becomes unnecessary."
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
header: 08-hca.webp
---

## Figure 4

The previous chapter, [Compressed Sparse Attention](../07-csa/), gives you Figure 3 of the V4 paper: a two-stream architecture with a token compressor, a lightning indexer, a top-k selector, and dense attention over the selected tokens. It is a busy diagram. Lots of boxes, lots of arrows.

Figure 4 is the antidote.

{{< figure src="/llm-maths/figures/07-deepseek-attn/v4-paper/deepseek-v4-figure4-hca.webp"
           alt="DeepSeek V4 paper Figure 4: Core architecture of Heavily Compressed Attention. Hidden states of KV tokens go through a single Token-Level Compressor with rate m', producing Heavily Compressed KV Entries. These feed directly into Shared Key-Value Multi-Query Attention along with the query and a Sliding Window branch. No indexer. No top-k selector."
           caption="**Figure 4 of the DeepSeek-V4 paper.** Core architecture of HCA. Compared to CSA (previous chapter), HCA has *no lightning indexer*, *no top-k selector*, *one compressor stream* (instead of two), and a much heavier compression rate $m' = 128$ (vs. $m=4$ in CSA). What remains: token-level compressor → dense attention over compressed entries → sliding window for local detail. Half the diagram of CSA. The sequence length collapses to $T/128$ before attention even runs."
           credit="Reproduced from DeepSeek-AI, DeepSeek-V4 Technical Report (2026), Fig. 4." >}}

No indexer. No top-k selector. No sparse-selection machinery of any kind. Just one box labeled "Token-Level Compressor" feeding into dense {{< wiki "attention" >}}attention{{< /wiki >}}.

The compressed sequence is short enough that dense attention over it is *already cheap*. You do not need to select; you attend to all of it.

{{< crosshead >}}The Core Idea{{< /crosshead >}}

CSA is V4's workhorse for fine-grained retrieval. But not every layer needs fine-grained retrieval. Some layers just need to know: "what was the general theme of what happened 50,000 tokens ago?" The exact words don't matter. The topic, the emotional register, the rough factual content — that's enough.

HCA answers that question. Compress every 128 tokens into a single vector. Run dense attention over the resulting short sequence. Attend over *everything* at the coarser granularity.

Think of it as the difference between reading a novel in full versus reading the chapter summaries. You can't retrieve a specific quote from a chapter summary. But you can remember the plot. If someone asks "what was the character's motivation in chapter seven?", a chapter summary is often enough — and it is 128× cheaper to read.

HCA is the chapter-summary reader. CSA is the full-text searcher. A real model needs both.

{{% pullquote type="standard" %}}
Sparsity and compression are substitutes, not complements, beyond a certain compression ratio. CSA compresses by 4× and still needs sparse selection. HCA compresses by 128× and drops selection entirely — the sequence is short enough to attend densely.
{{% /pullquote %}}

## The Heavy Compression

HCA's compressor is simpler than CSA's. CSA has two overlapping streams ($C^a$ and $C^b$, stride-4 with overlap for boundary continuity). HCA has one stream, no overlap, stride = $m' = 128$.

The math, from the V4 paper:

$$C = H \cdot W^{KV}$$
$$Z = H \cdot W^Z$$
$$S_{m'i : m'(i+1)-1} = \text{Softmax}_{\text{row}}(Z_{m'i : m'(i+1)-1} + B)$$
$$C^{\text{Comp}}_i = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j$$

In Python:

```python
def hca_compress(H: np.ndarray, m_prime: int, W_kv, W_z, B):
    """
    H:       [T, d_model] -- all hidden states in this layer
    m_prime: compression rate (128 in V4)
    W_kv:    [d_model, d_kv] -- project hidden states to KV space
    W_z:     [d_model, d_kv] -- produce per-token softmax weights
    B:       [d_kv]          -- learned bias for weight calibration
    Returns: C_comp [T // m_prime, d_kv] -- compressed KV entries
    """
    T = H.shape[0]
    n_blocks = T // m_prime

    C = H @ W_kv   # [T, d_kv] -- project to KV space
    Z = H @ W_z    # [T, d_kv] -- compression weights

    C_comp = np.zeros((n_blocks, C.shape[-1]))
    for i in range(n_blocks):
        block = slice(i * m_prime, (i + 1) * m_prime)
        # Per-dim softmax over the block: each output dimension gets its own
        # weighted average of the 128 input tokens in the block.
        S_i = softmax(Z[block] + B, axis=0)   # [m_prime, d_kv]
        C_comp[i] = (S_i * C[block]).sum(0)    # [d_kv]
    return C_comp   # T/128 entries — that's the entire compressed history
```

{{< crosshead >}}Line-by-Line Intuition{{< /crosshead >}}

The `Z @ W_z + B` term is what makes this a *learned* compression rather than a mean pool. If every token contributed equally, you would just average the 128 hidden states. But some tokens are more informative than others — the subject of a sentence matters more than "the", the verb matters more than punctuation. The {{< wiki "softmax" >}}softmax{{< /wiki >}} over `Z` learns those weights during training. The bias `B` allows the network to shift the weight distribution per output dimension, giving the compressor fine-grained control over what it preserves.

Crucially, the softmax is applied *per dimension* (`axis=0` — softmax over the T direction, one weight per position per KV dimension). This means dimension 7 of the output might be a weighted average dominated by token 43 in the block (the noun), while dimension 37 might be dominated by token 91 (the main verb). The compressor is learning a different "representative token" per output dimension.

{{% callout type="definition" %}}
**Token-level compressor.** The learned weighted average $C^{\text{Comp}}_i = \sum_j S_{ij} \odot C_j$ with per-dimension {{< wiki "softmax" >}}softmax{{< /wiki >}} weights $S_{ij}$ is the same structure as the CSA compressor. The difference: CSA uses $m=4$ (stride-4 blocks with overlap), while HCA uses $m'=128$ (stride-128 blocks, no overlap). Same formula, 32× heavier compression.
{{% /callout %}}

## The Numbers That Matter

For V4-Pro at $T = 1\text{M}$ tokens, $m' = 128$:

$$\text{Compressed sequence length} = T / m' = 1{,}000{,}000 / 128 = 7{,}813 \text{ entries.}$$

Dense attention over 7,813 entries instead of 1,000,000. That is the compression.

**Napkin math for one HCA decode step** (one new query token attending over the compressed history):

- Attention FLOPs: $n_h \times d_{\text{kv}} \times (T/m') = 128 \times 512 \times 7813 \approx 5.1 \times 10^8$ FLOPs.

**Compare to V3.2 dense attention at the same T = 1M:**

- Attention FLOPs: $n_h \times d_{\text{kv}} \times T = 128 \times 512 \times 10^6 \approx 6.6 \times 10^{10}$ FLOPs.
- **HCA is ~130× cheaper than V3.2 dense attention per layer.**

**Compare to CSA at T = 1M** (from the [previous chapter](../07-csa/)):

- CSA total: indexer scoring (~$2 \times 10^9$) + dense attend over k=1024 (~$6.7 \times 10^7$) $\approx 2 \times 10^9$ FLOPs.
- **HCA is ~4× cheaper than CSA per layer.**

This is the cost structure that makes HCA worth including in half of V4's layers.

## No Indexer, No Top-k

{{% callout type="counterintuitive" %}}
HCA does not need a sparse selector because heavy compression already made the sequence short. This is the key insight: at $m'=128$, the answer to "which entries matter?" is "all of them" — because there are only 7,813 of them.
{{% /callout %}}

CSA uses a sparse indexer because its compressor runs at $m=4$. A 4× compression of a 1M-token sequence leaves 250,000 entries — far too many for dense decode attention. So CSA adds an indexer to select the top-k=1024 from those 250,000 entries.

HCA compresses 32× harder and gets to 7,813 entries directly. Dense attention over 7,813 entries at decode time is not expensive. There is nothing to select from. Every compressed entry gets attended to.

The result is a clean architectural separation:
- **CSA layers**: coarse compression, fine-grained selection, high retrievability of specific token content.
- **HCA layers**: heavy compression, no selection, dense global coverage at paragraph granularity.

## The Same Query Plumbing

Both CSA and HCA share the same query computation. From the V4 paper, HCA reuses the latent-query infrastructure from {{< wiki "mla" >}}MLA{{< /wiki >}}:

$$c^Q_t = h_t \cdot W^{DQ}$$
$$[q_{t,1}; \ldots; q_{t,n_h}] = q_t = c^Q_t \cdot W^{UQ}$$

There is no indexer query ($q_I$) in HCA. The latent $c^Q_t$ is computed and immediately up-projected to the $n_h$ full query heads. The query path is shorter than CSA's — one fewer projection.

The MLA absorption trick still applies: $W^{UK}$ can be absorbed into the query side, so the compressed KV cache stores the raw compressed latents and the projection is fused at decode time. The [Issue 5 ch.19 MLA](/comicbook/05-microgpt/19-mla/) article covers this absorption in detail; the same algebra applies to the compressed entries here.

```pyplot {id="csa-vs-hca-cost" caption="CSA AND HCA HAVE DIFFERENT COMPUTE CURVES. CSA: O(T/m × d_I) FOR INDEXER + O(k × d_h × n_h) FOR ATTENDED. HCA: O((T/m')² × d_h × n_h). AT m'=128, HCA IS CHEAPER UP TO T~256K, AND CSA WINS AT T~1M."}
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

The crossover point — where HCA becomes more expensive than CSA — is around T = 256K. Below 256K, HCA's dense attend over $T/128$ entries is cheaper than CSA's indexer over $T/4$ entries. Above 256K, the CSA indexer (which scales with $T/4$) is larger than HCA's attend (which scales with $T/128$) only when the indexer dominates the CSA total, which it does past T≈256K. At T = 1M, HCA is ~4× cheaper per layer.

This is why the two modes are *complementary*, not competing: CSA handles retrieval quality (fine-grained selection), HCA handles compute efficiency (cheap global coverage). Each is cheaper in the domain where the other is more expensive.

{{< crosshead >}}The Attention Sink in HCA{{< /crosshead >}}

HCA uses learnable sink logits, just as CSA does (detailed in the [attention sink primer](../17-attention-sink/)). With only 7,813 compressed entries, if a query has "nothing relevant" in the compressed history — perhaps it is a syntactic query that only needs local context — the {{< wiki "softmax" >}}softmax{{< /wiki >}} over 7,813 entries would still force the model to distribute attention mass somewhere. The result would be noisy garbage being mixed into the output.

The sink logit is a learnable scalar added to a fixed "null" position. When nothing in the compressed history is relevant, the query can concentrate its attention mass on the sink and effectively ignore the history. This is a small but critical safety valve. Without it, layers that want to attend narrowly (to the sliding window only) would be forced to carry ghost signal from the compressed history at every position.

## What HCA Throws Away

This is the honest part. $m' = 128$ is aggressive. Compressing 128 tokens into one vector means:

**What HCA can still answer:**
- "There was a kitchen scene around position 50K." (Topic is preserved.)
- "The protagonist's name was mentioned repeatedly in the first quarter." (High-frequency content survives weighted averaging.)
- "The emotional register of chapter 3 was anxious." (Aggregate semantic signal is preserved.)
- "A time stamp appeared around token 30K." (Structured tokens with distinctive statistics survive.)

**What HCA cannot answer:**
- "What exact words were spoken in the kitchen scene?" (Token-level identity is destroyed.)
- "Was the word 'always' used or 'sometimes'?" (Lexical precision is gone.)
- "Was the figure exactly 3.7 or 3.8?" (Numerical precision is gone.)
- "Which of two similar arguments appeared first?" (Positional ordering within a compressed block is gone.)

For the things HCA cannot answer, you need CSA's fine-grained retrieval. For everything else, HCA's paragraph-level summary is sufficient and much cheaper.

{{% pullquote type="counter-intuitive" %}}
Destroying token-level identity is a feature, not a bug. A layer that needs to summarize the emotional arc of a 100K-token document does not benefit from token-level precision — it benefits from attending over 781 paragraph-level summaries instead of 100K raw tokens.
{{% /pullquote %}}

This also explains why HCA cannot appear on *every* layer. At least some layers must preserve fine-grained retrievability. CSA's job is exactly that. In V4-Pro, the pattern alternates: one CSA layer (retrieve), one HCA layer (aggregate). Neither is sufficient alone.

## Why Use Both?

{{< crosshead >}}The Layer Assignment in V4{{< /crosshead >}}

V4-Pro has 61 transformer layers. The assignment:

| Layers | Attention mode |
|---|---|
| 1–2 | HCA (early layers process low-level features; global but coarse context is enough) |
| 3, 5, 7, … 61 (odd) | CSA (fine-grained retrieval) |
| 4, 6, 8, … 60 (even) | HCA (coarse aggregation) |

V4-Flash has 43 layers:

| Layers | Attention mode |
|---|---|
| 1–2 | Pure sliding window (smaller model, simpler early processing) |
| 3, 5, 7, … 43 (odd) | CSA |
| 4, 6, 8, … 42 (even) | HCA |

The [hybrid pattern](../09-hybrid-pattern/) chapter examines this interleaving in detail, including ablations on what happens when you use only CSA or only HCA on all layers. The summary: all-HCA loses on precision benchmarks (needle-in-haystack tasks). All-CSA is more expensive and, counterintuitively, slightly worse on document-level summarization tasks (where HCA's coarser view is actually better).

The three attention modes — HCA (coarse global), CSA (fine-grained retrieval), sliding window (local detail) — cover complementary ranges:

```pyplot {id="coverage-ranges" caption="THREE ATTENTION MODES IN V4: EACH COVERS A DIFFERENT RANGE OF THE CONTEXT. HCA=COARSE GLOBAL, CSA=FINE-GRAINED SELECTED, SLIDING WINDOW=LOCAL IMMEDIATE."}
fig, ax = plt.subplots(figsize=(9, 3.8))

# Illustrate coverage ranges for a T=1M context
T = 1_000_000
x = np.arange(T)

# Sliding window: last 4096 tokens (shown as fraction of T)
sw_start = T - 4096
ax.barh(0, 4096 / T, left=sw_start / T, color='#00A8A8', height=0.5, label='Sliding window (last 4096 raw tokens)', alpha=0.9)

# CSA: k=1024 selected entries from T/4=250K compressed entries, anywhere
ax.barh(1, 1.0, left=0, color='#FF007F', height=0.5, alpha=0.3, label='CSA: sparse selection from full context (k=1024 entries)')
for _ in range(12):
    pos = np.random.uniform(0, 0.95)
    ax.barh(1, 0.01, left=pos, color='#FF007F', height=0.5, alpha=0.7)

# HCA: all T/128=7813 compressed entries, uniform global coverage
ax.barh(2, 1.0, left=0, color='#FFD700', height=0.5, alpha=0.5, label='HCA: dense attend over T/128=7813 compressed entries')

ax.set_xlim(0, 1)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Sliding\nwindow', 'CSA\n(selected)', 'HCA\n(compressed)'])
ax.set_xlabel('fraction of 1M-token context')
ax.set_title('Three complementary coverage patterns in V4 at T=1M', fontsize=11)
ax.legend(loc='lower right', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
```

The plot shows the coverage intuition: sliding window is a narrow strip at the end (right edge). CSA is punctate — scattered selected tokens from anywhere in the context. HCA is a uniform smear across the entire context at coarse resolution.

## A Concrete Compute Bill

Same napkin math as the previous chapter, now for HCA. V4-Pro at $T = 1\text{M}$, single decode step, single HCA layer:

$$n_h \cdot d_{\text{kv}} \cdot (T/m') = 128 \times 512 \times 7813 \approx 5.1 \times 10^8 \text{ FLOPs.}$$

For the HBM bandwidth: compressed cache size per HCA layer = $7813 \times d_{\text{kv}} \times 2 \text{ bytes} = 7813 \times 512 \times 2 \approx 8 \text{ MB}$. At H800 HBM bandwidth of ~3.35 TB/s: reading the HCA cache takes $\approx 2.4 \mu\text{s}$. Compare to reading V3.2-Exp's raw cache at T=1M: $1{,}000{,}000 \times 576 \times 2 \text{ bytes} = 1.15 \text{ GB}$, which takes $\approx 340 \mu\text{s}$.

**HCA reduces cache read time per layer from 340 µs to 2.4 µs at T=1M.** That is a 140× reduction in attention memory bandwidth per HCA layer.

This is why V4 can serve 1M-context queries at all. The combination of HCA (for roughly half the layers) and CSA (for the other half) keeps the per-layer decode memory traffic manageable. The MLP layers, which don't change with context length, dominate the remaining compute budget.

{{% callout type="tangent" %}}
**Why not compress more aggressively?** If $m'=128$ is good, why not $m'=1024$? The answer is the granularity tradeoff. At $m'=1024$, each compressed entry represents roughly a page of text. The weighted average of a page of text loses almost all structural information — you end up with something like a document embedding. The model can no longer distinguish "this page had a question vs an answer vs a list." Ablations in the V4 tech report show quality plateauing around $m'=64$–$128$ and degrading sharply above $m'=256$.
{{% /callout %}}

## What To Remember

1. **HCA = compress, then dense attention.** No sparse selection, no indexer. One compressor stream at $m' = 128$.
2. **Compression rate $m' = 128$, vs. CSA's $m = 4$.** Sequence collapses to $T/128$. At T=1M that is 7,813 entries.
3. **HCA destroys token-level identity by design.** It does paragraph-scale aggregation. Pair it with CSA (token-scale retrieval) and sliding window (local context).
4. **~130× cheaper than dense attention per layer at T=1M.** ~4× cheaper than CSA per layer.
5. **Used on roughly half of V4's layers**, interleaved with CSA.
6. **Attention sink logits prevent ghost signal** when a query wants local-only context.

**Continue to** → [The Hybrid Pattern](../09-hybrid-pattern/) — why interleave CSA and HCA layers, what each does for the model's information flow, and how V4's two variants (Pro and Flash) deploy the same recipe with different layer budgets.
