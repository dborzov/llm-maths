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

At first glance this looks complex. There are two compressor streams, an indexer box, a top-k diamond, a concatenation, and a multi-query attention block at the top. But the diagram has a logic, and once you see it, the whole architecture is legible.

The logic: **DSA compressed nothing. CSA compresses first, then selects.** The sequence shrinks by 4× before the lightning indexer even fires. At 1M context, the indexer scores 250,000 compressed entries rather than 1,000,000 raw tokens. The same k-budget buys more coverage over a shorter sequence.

Everything else in the diagram is either something you already know from [Chapter 6](../06-lightning-indexer/) (the lightning indexer, the top-k selector, the sliding window) or a detail we will build up from scratch in this chapter.

{{< crosshead >}}The Two-Stream Compressor: Why 8 Tokens Per Entry, Not 4{{< /crosshead >}}

The first surprise in the V4 paper is that CSA's compression rate is $m = 4$ — stride 4 — but each compressed entry represents **8 tokens**, not 4. The vLLM engineering team makes this explicit in their implementation notes:

> *"c4a: compress the KV cache by roughly 1/4. One compressed token is a weighted sum of 8 uncompressed tokens, with a stride of 4."*

The two-stream design is how 8 tokens flow into one entry with stride 4. Let's build it from the math.

**Stream A** and **Stream B** are two parallel projections of the hidden states $H \in \mathbb{R}^{n \times d}$:

$$C^a = H \cdot W^{aKV}, \quad C^b = H \cdot W^{bKV}$$
$$Z^a = H \cdot W^{aZ}, \quad Z^b = H \cdot W^{bZ}$$

Here $C^a$ and $C^b$ are the KV projections for each stream, and $Z^a$, $Z^b$ are the unnormalized compression weights. Both streams cover the same tokens; what differs is *which block each covers*.

For compressed index $i$, the softmax weights draw from two windows:

$$[S^a_{mi:m(i+1)-1};\; S^b_{m(i-1):mi-1}] = \text{Softmax}_{\text{row}}([Z^a_{mi:m(i+1)-1} + B^a;\; Z^b_{m(i-1):mi-1} + B^b])$$

Stream A covers positions $[mi, m(i+1)-1]$ — the block aligned to index $i$. Stream B covers positions $[m(i-1), mi-1]$ — the block *before* index $i$.

The compressed entry $i$ is then:

$$C^{\text{Comp}}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j + \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j$$

A softmax-weighted blend of $2m = 8$ raw entries. In the vLLM implementation's terms: positions $[4i - 4, 4i + 3]$ all contribute to compressed entry $i$, with the RoPE anchor position set to $4i$. This is the "overlapped" compression — adjacent compressed entries share a half-block of input.

```python
# Two-stream CSA compressor (simplified, omitting RoPE and bias terms)
def csa_compress(H, W_aKV, W_bKV, W_aZ, W_bZ, m=4):
    """
    H:    [T, d_model]  — hidden states
    m:    compression stride (4 in V4)
    Returns: C_comp [T//m, d_kv] — one compressed entry per m raw tokens
    """
    T = H.shape[0]
    n_comp = T // m

    # Project to KV and compression-weight spaces (two streams each)
    Ca = H @ W_aKV   # [T, d_kv]
    Cb = H @ W_bKV   # [T, d_kv]
    Za = H @ W_aZ    # [T, d_kv]
    Zb = H @ W_bZ    # [T, d_kv]

    C_comp = []
    for i in range(n_comp):
        a_start, a_end = i * m, (i + 1) * m         # stream A: current block
        b_start, b_end = max(0, (i - 1) * m), i * m # stream B: previous block

        # Softmax over the concatenated 2m-token window
        Za_block = Za[a_start:a_end]  # [m, d_kv]
        Zb_block = Zb[b_start:b_end]  # [m, d_kv] (0 tokens at i=0)

        # Per-dim softmax across all 2m tokens
        Z_cat = np.concatenate([Za_block, Zb_block], axis=0)
        S_cat = np.exp(Z_cat) / np.exp(Z_cat).sum(axis=0, keepdims=True)
        Sa = S_cat[:m]
        Sb = S_cat[m:]

        # Weighted blend
        entry = (Sa * Ca[a_start:a_end]).sum(0) + (Sb * Cb[b_start:b_end]).sum(0)
        C_comp.append(entry)

    return np.stack(C_comp)  # [T//m, d_kv]

# Effect on sequence length:
T = 1_000_000
m = 4
print(f"Raw tokens:        {T:,}")
print(f"Compressed entries: {T // m:,}  (250K instead of 1M)")
print(f"Causality note: entry i is complete when position {m}i+{m-1} arrives.")
```

**Why the overlap matters:** Without it, each compressed entry covers tokens $[4i, 4i+3]$ only. Content that straddles a block boundary — a phrase whose first two words land in block $i-1$ and last two in block $i$ — gets split between two entries with no way to reconstruct the phrase from either. The overlapping two-stream design ensures each compressed entry has context from both sides of the boundary. Adjacent entries share 4 tokens of input, so boundary-spanning content is never fully invisible.

{{% callout type="tip" %}}
**Causality.** For the $j$-th compressed token (covering positions $[4j - 4, 4j + 3]$), a query at position $i$ can only attend to this entry if $i \ge 4j + 3$. In other words: the compressed entry is only available once all 8 of its source tokens have arrived. This is why the sliding window exists: for the current query position's *most recent* 128 tokens (which are still within the causality horizon of many compressed entries), the model falls back to uncompressed {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} entries.
{{% /callout %}}

{{< crosshead >}}The Lightning Indexer Returns{{< /crosshead >}}

Once the compressor has produced $n = T/m$ compressed entries, the [lightning indexer](../06-lightning-indexer/) runs over those entries rather than raw token latents. The indexer architecture is identical to DSA's — a bilinear scorer with a ReLU gate — but it operates on a 4× shorter sequence.

From the V4 paper, each query token computes:

$$c^Q_t = h_t \cdot W^{DQ}$$

This is the **shared query latent** — the same $c^Q_t$ feeds both the indexer and the main attention. One projection, two purposes.

From $c^Q_t$, indexer query heads are computed:

$$q^I_t = c^Q_t \cdot W^{IUQ}, \quad q^I_t \in \mathbb{R}^{n^I_h \times c_I}$$

And the indexer keys come from the *compressed* entries' representations (not from the raw latent cache):

$$K^{\text{IComp}}_s = C^{\text{Comp}}_s \cdot W^{IUK}$$

The index score is:

$$I_{t,s} = \sum_{h=1}^{n^I_h} w^{I,h}_t \cdot \text{ReLU}(q^{I,h}_t \cdot K^{\text{IComp}}_s)$$

Same bilinear ReLU-gated formula as DSA. But $s$ now indexes over compressed entries ($n = T/m$ total) rather than raw tokens ($T$ total).

{{% pullquote type="counter-intuitive" %}}
The shared query latent $c^Q_t$ is the key design insight of CSA. The same vector that tells the indexer what to look for also generates the main attention queries. The indexer and main attention are not two separate models—they are two read-heads on the same encoded query state.
{{% /pullquote %}}

## The Top-k Selector

The top-k selection chooses the best $k$ compressed entries:

$$\mathcal{C}^{\text{SprsComp}}_t = \{C^{\text{Comp}}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}$$

The production default value, per the V4 technical report and confirmed by the vLLM engineering team:

- **V4-Flash: $k = 512$** compressed entries.
- **V4-Pro: $k = 512$** compressed entries (same default; the larger model can afford the same $k$ because its compressed-entry dimension $d_{kv} = 512$ is larger, giving more expressivity per entry).

{{% marginnote %}}
The [vLLM blog post](https://vllm.ai/blog/deepseek-v4) reports $k = 512$ as the production default for c4a. The V4 paper's table lists different values depending on model variant.
{{% /marginnote %}}

What does $k = 512$ compressed entries mean in terms of raw tokens? Each compressed entry represents 8 raw tokens (two overlapping windows of 4). So attending to 512 compressed entries gives the model effectively **4,096 raw tokens' worth of content** from the compressed history.

At $T = 1\text{M}$:

$$\text{Fraction reached} = \frac{k \times 2m}{T} = \frac{512 \times 8}{1{,}000{,}000} = 0.41\%$$

Less than half a percent of the context. Yet quality on long-context benchmarks (RULER, LongBench V2, Needle-in-a-Haystack) is preserved — because the lightning indexer selects *the right* half a percent.

For the differentiable training of this discrete selector, see the [top-k routing primer](../13-topk-and-routing/).

```pyplot {id="csa-coverage" caption="CSA TOP-K COVERAGE AT DIFFERENT CONTEXT LENGTHS. k=512 COMPRESSED ENTRIES COVERS A CONSTANT 4096 RAW TOKENS — AN EVER-SMALLER FRACTION AS T GROWS, BUT THE INDEXER SELECTS THE RELEVANT FRACTION."}
T_vals = np.array([16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576])
k = 512
m = 4
window = 2 * m  # tokens per compressed entry

# Absolute raw tokens covered (constant)
raw_covered = k * window * np.ones_like(T_vals)

# Fraction of context covered
fraction_pct = 100 * raw_covered / T_vals

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.semilogx(T_vals / 1000, raw_covered / 1000, color='#FF007F', linewidth=2.5,
             marker='o', markersize=6)
ax1.set_xlabel('context length T (k tokens)')
ax1.set_ylabel('raw tokens reached (k)')
ax1.set_title(f'absolute coverage: constant {k * window // 1000}K tokens', fontsize=10)
ax1.set_ylim(0, 6)
ax1.grid(True, alpha=0.3)
ax1.spines[['top', 'right']].set_visible(False)

ax2.semilogx(T_vals / 1000, fraction_pct, color='#00A8A8', linewidth=2.5,
             marker='s', markersize=6)
ax2.set_xlabel('context length T (k tokens)')
ax2.set_ylabel('fraction of context reached (%)')
ax2.set_title('fraction coverage: shrinks as T grows', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The left panel shows that the absolute coverage is constant — 4,096 raw tokens always, regardless of how long the context grows. The right panel shows the fraction shrinking from 25% at 16K to 0.4% at 1M. The indexer's job becomes harder as context grows, but the top-k coverage budget stays fixed.

## Shared Key-Value Multi-Query Attention

After selection, the $k$ chosen compressed entries go to the main {{< wiki "attention" >}}attention{{< /wiki >}} block. The same latent $c^Q_t$ that fed the indexer now generates the main query heads:

$$q_t = c^Q_t \cdot W^{UQ}, \quad q_t \in \mathbb{R}^{n_h \times c}$$

(V4-Pro: $n_h = 128$ query heads, each of dimension $c = 512$.)

The main attention is **Multi-Query Attention** (Shazeer 2019): one key vector and one value vector per compressed entry, shared across all $n_h$ query heads. This is what "Shared Key-Value" means in Figure 3.

$$o_{t,i} = \text{CoreAttn}(\text{query}=q_{t,i},\; \text{key}=\mathcal{C}^{\text{SprsComp}}_t,\; \text{value}=\mathcal{C}^{\text{SprsComp}}_t)$$

The compressed entries serve as *both* keys and values — the same vector. This is a stronger form of MQA than usual: not just shared KV across heads, but also shared K and V with each other.

The **inverse RoPE** trick, introduced in V4 and explained in the vLLM engineering blog, is relevant here. Because K and V are shared (the same compressed vector serves both roles), the attention output carries absolute position information through the rotation matrices. V4 applies an inverse RoPE $R(-i)$ to the output to restore translation invariance:

$$R(-i) \cdot a_i = \sum_p \frac{\exp(q_i^\top R(j_p - i) k_{j_p})}{\sum_r \exp(q_i^\top R(j_r - i) k_{j_r})} R(j_p - i) k_{j_p}$$

The output now depends only on *relative* position $(j_p - i)$, not absolute position $j_p$. This is the same translation invariance property that standard attention has — inverse RoPE restores it when K and V are shared.

{{< crosshead >}}Grouped Output Projection{{< /crosshead >}}

With $n_h = 128$ query heads each producing a $c = 512$-dimensional output, the naive approach would project the concatenated $n_h \times c = 65{,}536$-dimensional output down to the residual stream dimension $d = 7168$. That's a single weight matrix of $65{,}536 \times 7{,}168 \approx 470\text{M}$ parameters. At this scale, it is the biggest single weight in the attention block.

V4 uses grouped projection: split the $n_h$ output heads into $g = 16$ groups (V4-Pro), project each group of $128/16 = 8$ heads through a shared $d_g = 1024$-dimensional intermediate, then sum the group outputs.

$$\text{out} = \sum_{g=1}^{G} \text{GroupProject}_g(q_{t, g \cdot (n_h/G) : (g+1) \cdot (n_h/G)})$$

This reduces the projection parameter count from 470M to roughly $G \times (n_h/G \times c) \times d_g + G \times d_g \times d \approx 80\text{M} + 14\text{M} = 94\text{M}$ parameters. A 5× parameter reduction for the output path at the cost of a two-stage projection.

## V4 Hyperparameters in One Table

| Hyperparameter | V4-Pro | V4-Flash | What it controls |
|---|---|---|---|
| Compression stride $m$ | 4 | 4 | Tokens per stride; each entry covers $2m = 8$ tokens |
| Top-k | 512 | 512 | Compressed entries attended per query |
| Effective raw token reach | 4,096 | 4,096 | $k \times 2m$ raw tokens per query |
| Indexer query heads $n^I_h$ | 64 | 64 | Indexer parallelism |
| Indexer head dim $c_I$ | 128 | 128 | Indexer expressivity |
| Query compression dim $d_c$ | 1,536 | 1,024 | Shared latent for indexer + main queries |
| Main query heads $n_h$ | 128 | 64 | Main attention heads |
| Main head dim $c$ | 512 | 512 | Main attention expressivity |
| Output groups $g$ | 16 | 8 | Grouped output projection |
| Sliding window $n_{\text{win}}$ | 128 | 128 | Local-only branch size |

{{% callout type="definition" %}}
**Why compression stride ≠ tokens per entry.** The "c4a" label (vLLM notation) refers to stride $m = 4$. But the two-stream overlap means each entry covers the window $[4j-4, 4j+3]$, which is $2m = 8$ tokens. The stride determines how many compressed entries exist ($T/m = T/4$); the window determines how many raw tokens each entry sees ($2m = 8$). This is distinct from the HCA compressor (c128a), which uses a single stream with no overlap — stride equals window size.
{{% /callout %}}

## The Sliding Window Branch

Per the V4 paper, the most recent $n_{\text{win}} = 128$ raw tokens are concatenated alongside the top-$k$ selected compressed entries before the final attention computation. This is the local-detail backstop.

Why? The causality constraint on compressed entries means that for the very recent past — positions within the last $m' = 128$ tokens from the current query position — no compressed entry is yet complete (they would need future tokens to be finalized). The sliding window provides uncompressed access to this recent window, ensuring the model always has precise access to what just happened.

The vLLM team notes this is also important for c128a layers (HCA): a query at position 100 cannot attend to any c128a compressed entry (the first one covers positions 0–127, but only positions 0–99 are causal). The sliding window is the only way to access recent context in HCA layers, making it an architectural necessity, not just an optimization.

## A Concrete Compute Bill

At $T = 1{,}000{,}000$, single CSA layer, single decode step — using the production defaults from vLLM:

| Operation | FLOPs |
|---|---|
| Indexer scoring: $n^I_h \times c_I \times (T/m) = 64 \times 128 \times 250{,}000$ | 2,048 MFLOPs |
| Top-k selection | ~trivial |
| Main attention: $n_h \times c \times k = 128 \times 512 \times 512$ | 33.6 MFLOPs |
| Sliding window: $n_h \times c \times n_\text{win} = 128 \times 512 \times 128$ | 8.4 MFLOPs |
| **Total CSA layer** | **≈ 2,090 MFLOPs** |

For comparison:

- **V3.2 dense at $T = 1M$:** $n_h \times D \times T = 128 \times 128 \times 10^6 \approx 16{,}384$ MFLOPs. **CSA is ~8× cheaper.**
- **V3.2 DSA at $T = 1M$:** $n^I_h \times c_I \times T + n_h \times k_\text{DSA} \times D = 32 \times 64 \times 10^6 + 128 \times 2048 \times 128 \approx 2048 + 33.6 \approx 2082$ MFLOPs. **CSA and DSA have nearly identical per-layer cost at 1M context**—but CSA covers 4× more raw history per selected entry.

The big picture: at 1M context, CSA's indexer (2,048 MFLOPs) dominates the per-layer cost by 60×. The main attention (33.6 MFLOPs) is essentially free once you know which tokens to attend to. The investment is in scoring; the payoff is in quality.

```pyplot {id="csa-flop-breakdown" caption="CSA FLOP BREAKDOWN PER LAYER AT 1M CONTEXT. THE INDEXER DOMINATES (98%). MAIN ATTENTION IS ALMOST FREE."}
operations = ['Indexer\nscoring\n(2048 MF)', 'Main\nattention\n(33.6 MF)', 'Sliding\nwindow\n(8.4 MF)']
flops = [2048, 33.6, 8.4]
colors = ['#FF007F', '#00A8A8', '#FFD700']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Bar chart
ax1.bar(operations, flops, color=colors, edgecolor='#1A1A1A', linewidth=1.5)
ax1.set_ylabel('MFLOPs')
ax1.set_title('CSA FLOPs per layer at T=1M (log scale)')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y', which='both')
ax1.spines[['top', 'right']].set_visible(False)

# Show how CSA cost evolves with T
T_vals = np.logspace(np.log10(16384), np.log10(1_048_576), 100)
m_csa, n_I, c_I_dim, n_h_csa, c_csa, k_csa, n_win = 4, 64, 128, 128, 512, 512, 128
indexer_flops = n_I * c_I_dim * (T_vals / m_csa) / 1e6
attend_flops = n_h_csa * c_csa * k_csa * np.ones_like(T_vals) / 1e6
sw_flops = n_h_csa * c_csa * n_win * np.ones_like(T_vals) / 1e6
dense_flops = n_h_csa * c_csa * T_vals / 1e6

ax2.loglog(T_vals / 1000, indexer_flops, color='#FF007F', linewidth=2.5, label='CSA indexer')
ax2.loglog(T_vals / 1000, attend_flops, color='#00A8A8', linewidth=2, linestyle='--', label='CSA attend (const)')
ax2.loglog(T_vals / 1000, dense_flops, color='#1A1A1A', linewidth=1.5, linestyle=':', alpha=0.7, label='dense baseline')
ax2.set_xlabel('context length T (k tokens)')
ax2.set_ylabel('MFLOPs per layer per decode step')
ax2.set_title('CSA cost grows with T (indexer), not T² (dense)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, which='both')
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

## The Exact Memory Arithmetic

The vLLM blog post provides the exact per-layer KV cache arithmetic for V4 at 1M context. For a **V4 CSA layer** (c4a) with BF16:

$$\text{shared-KV cache} = (n_\text{win} + T/m) \times 1024 \text{ bytes} = (128 + 250{,}000) \times 1024 \approx 250\text{ MiB}$$
$$\text{indexer cache} = (T/m) \times 256 \text{ bytes} = 250{,}000 \times 256 \approx 61\text{ MiB}$$
$$\text{Total CSA layer at 1M} \approx 311 \text{ MiB}$$

For a **V3.2 DSA layer** at the same context (for comparison):
$$\text{MLA cache} = T \times 1152 \text{ bytes} + T \times 256 \text{ bytes (indexer)} = 1{,}408 \text{ bytes/token} \times 1{,}048{,}576 \approx 1.38\text{ GiB}$$

**CSA layer is ~4.4× smaller** than a V3.2 layer at 1M context.

Across all 61 layers (30 CSA + 31 HCA), V4 totals approximately **9.62 GiB** at 1M context in BF16 — versus **83.9 GiB** for a V3.2-style 61-layer stack. That is **8.7× smaller**. With FP4 indexer cache and FP8 main KV cache (V4's production quantization), this shrinks by another ~2×, reaching roughly 4.8 GiB — fitting in the HBM of a single B200.

## What CSA Did Not Solve

CSA's indexer still scores $T/m$ entries per query per layer. At $T = 1M$ and $m = 4$: 250,000 scoring operations per query. Even in FP4, and even with the multi-stream CUDA kernel optimizations vLLM deployed (the indexer pipeline runs on a dedicated stream in parallel with KV compression and sliding-window insertion), the indexer is the dominant cost at 1M context.

The next chapter — [Heavily Compressed Attention](../08-hca/) — asks a simpler question: what if we skip the selection problem entirely? Compress by 128× instead of 4×, and attend densely over the 7,812 resulting entries. No indexer. No top-k. Just compress more aggressively.

It turns out that on roughly half the transformer layers, this is exactly what you want.

## What To Remember

1. **CSA = compress-then-DSA.** Compress $m=4$ tokens per entry (two overlapping streams of 4), then lightning-index over the $T/m$ compressed entries, keep top $k=512$.
2. **Each compressed entry covers 8 raw tokens**, not 4 — because the two-stream overlap covers $[4j-4, 4j+3]$.
3. **The query latent is shared.** $c^Q_t$ produces both indexer queries and main queries. One vector, two read-heads.
4. **Shared key-value MQA.** One K, one V per compressed entry; inverse RoPE applied to the output to restore translation invariance.
5. **At 1M context, V4 uses 9.62 GiB** total KV cache (vs 83.9 GiB for V3.2) — an 8.7× reduction, and 2× more with FP4/FP8 quantization.
6. **Sliding window (128 tokens) is architecturally necessary.** Causality prevents very recent tokens from appearing in compressed entries; the raw window provides access to them.

**Continue to** → [Heavily Compressed Attention](../08-hca/) — V4's complement mode. Compress every $m' = 128$ tokens into one. Attend densely. No indexer. No top-k. And why this is enough for half the layers.
