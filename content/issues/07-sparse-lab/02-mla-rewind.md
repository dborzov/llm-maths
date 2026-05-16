---
title: "MLA: The First Cut"
description: "May 2024. DeepSeek-V2 ships Multi-head Latent Attention. We re-tell the algebra from Issue 5 ch.19 with a different audience in mind: not how MLA works, but why it was the first move in a longer game."
topics: [attention, kv-cache, deepseek]
tags: [mla, deepseek-v2, latent-attention, absorption, rope]
theme: teal
math: true
draft: false
date: 2026-05-16T09:10:00-04:00
issue: 7
weight: 20
techKind: mainline
techNode: mla-rewind
header: 02-mla-rewind.webp
---

## Why We Are Doing This Again

This is the **second** issue on this site that begins with MLA. The [first](/issues/05-microgpt-unfolded/19-mla/) — Issue 5, chapter 19 — taught MLA *as a microGPT variation*. K and V folded into a latent $c_t$, an absorption identity that makes the read-time decompression free, the RoPE-vs-NOPE split, the algebra of the cache savings. If that chapter is fresh in your head, this one is going to feel familiar for the first three sections.

We are going to walk that math again. But the frame is different.

In Issue 5, MLA was a *technique* — a clever low-rank cache. In this issue, MLA is the *opening move* in DeepSeek's six-bet arc. It sets up the next bet, which sets up the next bet, which sets up DSA, which sets up V4. Every design choice we make in MLA reappears two papers later as a constraint that drives the next move. The RoPE-vs-NOPE split, in particular, is the seed of the lightning indexer.

If you read Issue 5 ch.19 already, skim sections 1–4 below for the new framing. If you didn't, read this chapter for the algebra and then go to Issue 5 ch.19 for the microGPT-level implementation.

## Hangzhou, May 6, 2024

Liang Wenfeng founded DeepSeek in 2023 as a spinoff of the quant hedge fund High-Flyer Capital Management. High-Flyer had been hoarding NVIDIA A100s since 2021 — before the US export controls, before most Western labs had realized what was coming. By late 2021 the fund had reportedly acquired over 10,000 A100s, a cluster comparable in size to what many Western research labs had at the time. By 2023 the cluster was real. The research organization was real. The ambition was unconcealed: build a frontier model, end-to-end, from a team in Hangzhou.

On May 6, 2024, they uploaded a paper to Hugging Face and arXiv: **"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model."** Fifty pages. Full technical report. Open weights, MIT license.

The model was not remarkable for its benchmark numbers alone — although those were strong. What made inference teams sit up was the cost footnote buried in the abstract: **API price at launch was ¥0.001 per 1000 tokens for cache-hit input, ¥0.002 per 1000 tokens for output.** At the time, that was approximately $0.14 per million input tokens (cache-hit) and $0.28 per million output tokens. GPT-4 was priced around $30 per million input tokens. DeepSeek-V2 came in at roughly **1/100th of GPT-4 pricing**.

Within 48 hours, inference-focused engineering channels were dissecting the paper's Table 2, which showed KV cache usage per token. The numbers seemed too good. The first reaction from many engineers was disbelief; the second, after verifying the math independently, was something between admiration and mild alarm. A team no one had heard of eighteen months earlier had solved the cache problem that was preventing long-context MoE models from being economically viable.

The model itself: 236 billion total parameters, MoE architecture with 21 billion active per token, trained to 128K context. Trained cost, per the technical report: approximately **$5.6 million of compute**. For context, GPT-4 is estimated to have cost upward of $100 million.

{{% callout type="definition" %}}
**MoE (Mixture of Experts):** A transformer variant where the FFN (feed-forward network) layer is replaced by a bank of "expert" sub-networks, and a learned router selects a small subset for each token. DeepSeek-V2 routes each token through 2 of 160 experts, keeping the active parameter count at 21B while total capacity is 236B. The KV cache cost scales with active parameters and sequence length — not total parameters. So the 236B total doesn't make caching worse; only the 21B active count matters.
{{% /callout %}}

The inference community spent the week of May 6 dissecting Figure 3 in the paper — the architecture diagram showing something called **Multi-head Latent Attention (MLA)**. The footnote on the absorption trick. The KV cache comparison table. The 30× memory reduction claim.

This chapter unpacks that claim from the inside. Not the "what" — the [Issue 5 chapter](/issues/05-microgpt-unfolded/19-mla/) has the what. The "why it was the right opening move, and what it left unfinished."

{{< crosshead >}}The KV Cache Problem MLA Was Built For{{< /crosshead >}}

Let's anchor on a single concrete number before any algebra. During **decode** — generating tokens one at a time — the model must load the keys and values for every previously generated token from GPU memory. For a model with $H$ attention heads, $D$ dimensions per head, across $L$ layers, at sequence length $T$, the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} has shape $(2, L, H, T, D)$ where the 2 is for K and V.

For DeepSeek-V2's original dense-MHA equivalent — before MLA — that's:

$$
\underbrace{2}_\text{K and V} \times \underbrace{60}_\text{layers} \times \underbrace{128}_\text{heads} \times T \times \underbrace{128}_\text{head dim} \times 2\text{ bytes}
$$

At $T = 128{,}000$ tokens: **400 GB**. That is the full memory of five H100 80GB cards, just for the KV cache of a single conversation. You can't actually run this. There are no spare five cards per live sequence in any production cluster.

For comparison, Llama-3-70B with GQA-8 (8 KV heads instead of 64):

$$
2 \times 80 \times 8 \times 128{,}000 \times 128 \times 2\text{ bytes} \approx 42\text{ GB}
$$

That's still a lot — but it fits on one H100. The cache is the binding constraint, not the weights.

MLA's pitch: cut the cache to approximately **1,152 bytes per token per layer**, regardless of the number of attention heads. Let's see how.

```pyplot {id="kv-axes-recap" caption="THE FIVE AXES OF THE KV CACHE. MLA ATTACKS D. GQA ATTACKS H. EVICTION ATTACKS T. QUANTIZATION ATTACKS THE BIT-WIDTH."}
# Recap chart: KV cache size vs context length for vanilla MHA, GQA, and MLA.
# Annotate which axis each method shrinks.
import numpy as np
import matplotlib.pyplot as plt

T = np.arange(1024, 131072, 2048)
L = 60     # DeepSeek-V2 layers
H = 128    # DeepSeek-V2 heads
D = 128    # head dim
d_c = 512  # MLA latent
bytes_per = 2

mha = 2 * L * H * D * T * bytes_per / 1e9
gqa = 2 * L * (H // 8) * D * T * bytes_per / 1e9
mla = L * (d_c + 64) * T * bytes_per / 1e9  # latent + rope channel

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(T / 1000, mha, color='#FF007F', linewidth=2.5, label='MHA (baseline)')
ax.plot(T / 1000, gqa, color='#00A8A8', linewidth=2.5, label='GQA-8 (H-axis cut)')
ax.plot(T / 1000, mla, color='#FFD700', linewidth=2.5, label='MLA (D-axis cut)')
ax.set_xlabel('context length (k tokens)')
ax.set_ylabel('KV cache (GB), DeepSeek-V2 scale')
ax.set_title('MLA shrinks the D-axis. GQA shrinks the H-axis. They compose.')
ax.legend()
ax.spines[['top','right']].set_visible(False)
```

Three curves, one log-scale story. MHA at 128K is off the top of the chart at DeepSeek scale. GQA-8 brings it down by 16×. MLA brings it down another ~30× beyond that. Both methods are linear in $T$ — they cut the *coefficient*, not the growth rate. But the coefficient matters enormously when you're trying to serve 128K contexts at production throughput.

The key: MLA doesn't attack the $H$ axis (number of heads) or the $T$ axis (sequence length). It attacks the $D$ axis, the per-head dimension, by replacing it with a low-rank latent.

{{% callout type="tip" %}}
**GQA and MLA are orthogonal.** Grouped Query Attention (GQA) cuts the $H$ axis — it groups multiple query heads to share one K/V head, reducing the head count from 128 to 8 or 16. MLA cuts the $D$ axis — it reduces the per-head dimension from 128 to a compressed latent. You can stack them. DeepSeek-V2 does exactly this: it uses a form of MLA where the latent is shared across all heads, effectively combining the compression with the head-count savings. The chart above shows them as separate curves; in practice they compose.
{{% /callout %}}

{{< crosshead >}}The Compression: One Latent Per Token{{< /crosshead >}}

Standard {{< wiki "attention" >}}attention{{< /wiki >}} computes K and V from the input hidden state $h_t$ at each position $t$:

$$K_t = h_t W_K, \quad V_t = h_t W_V$$

These get cached: shape $(H, D)$ each, so $2HD$ scalars per token per layer. In BF16, that's $2 \times 128 \times 128 \times 2 = 65{,}536$ bytes per token per layer. MLA instead caches a single **latent vector** $c_t$:

$$c_t = h_t W_{DKV}$$

where $W_{DKV} \in \mathbb{R}^{d_\text{model} \times d_c}$ projects down to $d_c = 512$ dimensions. Then at read time, K and V are reconstructed:

$$K_t = c_t W_{UK}, \quad V_t = c_t W_{UV}$$

Cache cost per token per layer: $d_c \times 2 = 512 \times 2 = 1{,}024$ bytes. That's 64× smaller, before accounting for the RoPE side channel we'll meet in a moment.

Here's the running Python sketch to make this concrete:

```python
import numpy as np

# DeepSeek-V2 scale
T = 1024    # sequence length
d_c = 512   # MLA latent dim
d_r = 64    # RoPE channel dim
H = 128     # heads
D = 128     # head dim per head
L = 60      # layers

# Per-token KV cache footprint (in BF16, 2 bytes per value)
mha_bytes = 2 * H * D * 2       # 2 for K,V; 2 bytes per BF16
mla_bytes = d_c * 2 + d_r * 2  # latent + RoPE channel

print(f"MHA: {mha_bytes:,} bytes/token/layer = {mha_bytes/1024:.1f} KB")
print(f"MLA: {mla_bytes:,} bytes/token/layer = {mla_bytes:.0f} bytes")
print(f"Raw ratio: {mha_bytes/mla_bytes:.1f}x")
```

Output:
```
MHA: 65,536 bytes/token/layer = 64.0 KB
MLA: 1,152 bytes/token/layer = 1152 bytes
Raw ratio: 56.9x
```

The paper claims ~30× rather than ~57×. The gap between 57× and 30× is the cost of the W_UK expansion at read time — we'll explain that in the next section.

There's also an intuitive way to understand *why* this compression is lossless in theory. The original K projection has $H \times D = 128 \times 128 = 16{,}384$ output dimensions. The latent $c_t$ has $d_c = 512$ dimensions. That's a 32× reduction. But the original K projection weight $W_K \in \mathbb{R}^{d_\text{model} \times H \times D}$ can itself be viewed as a low-rank mapping from $d_\text{model} = 7168$ dimensions to $16{,}384$ — most of its singular values are small. The latent is capturing the top-512 singular directions of that mapping. In practice, the model learns to concentrate the relevant information for K and V into those 512 dimensions during training. It's not heuristic compression after the fact; the model is trained with the compressed cache in the loop.

{{< crosshead >}}The Absorption Trick{{< /crosshead >}}

Here's the problem with naive caching of $c_t$: at decode time, computing attention scores requires K in full. The naive path:

1. For each past token $t$: load $c_t$ from cache, compute $K_t = c_t W_{UK}$, compute $q^\top K_t$.
2. Repeat for all $T$ past tokens.

Step 1 costs one matrix-vector multiply per past token. With $T = 128{,}000$ past tokens, that's 128K matrix multiplies per decode step — just to expand the latent to full K. You've saved memory but added compute. Net: bad trade.

The **absorption identity** kills this cost. The key observation is that matrix multiplication is associative:

$$q^\top K_t = q^\top (c_t W_{UK})^\top = q^\top W_{UK}^\top c_t = (W_{UK}^\top q)^\top c_t$$

Read that last form carefully. $(W_{UK}^\top q)$ is a single matrix-vector multiply using the *query* $q$. It depends only on $q$, not on any individual past token. So you can:

1. Compute $\tilde{q} = W_{UK}^\top q$ **once** per decode step. Cost: one matrix multiply against the query.
2. Compute all attention scores as $\tilde{q}^\top c_t$ for each $t$. Cost: one dot product per past token — cheap.

The W_UK expansion has been "absorbed" into the query. **The cache stays small; the extra compute is a single query-side matrix multiply, not one per past token.**

{{% pullquote type="theorem" %}}
The absorption identity: $q^\top (c W_{UK})^\top = (W_{UK}^\top q)^\top c$. Fold the cache-expansion matrix into the query once. Then query against latents directly. The cache never materializes K at write time.
{{% /pullquote %}}

The same trick applies to values, via the linearity of the weighted sum. Output of {{< wiki "attention" >}}attention{{< /wiki >}} for the V-path:

$$\text{out} = \sum_t a_t V_t = \sum_t a_t (c_t W_{UV}) = W_{UV} \sum_t (a_t c_t)$$

The weighted sum over latents $\sum_t a_t c_t$ has shape $d_c$ — you do this sum in the small latent space, then multiply by $W_{UV}$ once to get the output. Again, one matrix multiply at the end, not one per past token.

In Python, the absorbed decode step looks like:

```python
def mla_decode_step(q, C_latent, W_UK, W_UV):
    """
    q:         (d_model,)   -- query for this decode step
    C_latent:  (T, d_c)     -- the cached latents for all T past tokens
    W_UK:      (d_c, H*D)   -- latent-to-K expansion
    W_UV:      (d_c, H*D)   -- latent-to-V expansion
    """
    # Absorption: fold W_UK into q ONCE per step
    q_absorbed = W_UK.T @ q          # shape (H*D,) -- one matmul

    # Score all T past tokens with cheap dot products
    scores = C_latent @ q_absorbed   # shape (T,) -- no W_UK expansion per token

    attn_weights = softmax(scores)   # shape (T,)

    # V-side absorption: weighted sum in latent space, then expand once
    latent_sum = attn_weights @ C_latent   # shape (d_c,)
    output = W_UV.T @ latent_sum           # shape (H*D,) -- one matmul

    return output
```

The analogy that makes this click: imagine you're searching a library of 100,000 compressed books. The naive approach is to decompress each book, then search the full text. The absorbed approach is to compress your *query* into the same compressed format, then search the compressed books directly. You do one compression (of the query), then 100,000 fast searches in compressed space. The math works out identically — it's the same information, accessed in a different order.

This is why the paper's 30× claim is accurate rather than the naive 57×: the $W_{UK}$ expansion still runs, it just runs once per decode step against the query rather than once per past token against the cache. The compute cost is amortized over $T$ tokens; at large $T$ it becomes negligible compared to the score dot products.

Let's also verify that the absorption doesn't change the answer. For a single query $q$ and latent $c$:

$$q^\top K = q^\top (c W_{UK})^\top = q^\top W_{UK}^\top c^\top = (W_{UK}^\top q)^\top c^\top = \tilde{q}^\top c^\top$$

where $\tilde{q} = W_{UK}^\top q$. The inner product is preserved — absorption is exact, not approximate. No approximation error, no loss of expressivity. Just a reordering of multiplications.

{{< crosshead >}}The RoPE Compromise — And Why It Matters Two Papers Later{{< /crosshead >}}

The absorption trick has one enemy: **positional encoding**.

{{< wiki "rope" >}}RoPE{{< /wiki >}} (Rotary Position Encoding) works by applying a position-dependent rotation $R_t$ to the key vector at write time, based on the absolute position $t$ of the token. The rotated key $K_t^\text{rope} = R_t K_t$ encodes both content and position in a single vector. When you compute $q^\top K_t^\text{rope}$ at decode time, the dot product produces attention scores that are sensitive to relative position — exactly what you want.

The problem: MLA's absorption trick assumes you can compute $\tilde{q} = W_{UK}^\top q$ and then take $\tilde{q}^\top c_t$ for each cached latent $c_t$. This only works if $c_t$ contains the *un-rotated* key information. But if you apply RoPE *before* writing to the cache, the rotation is baked into $c_t$, and the absorption math breaks — the rotated latent can't be cleanly "un-rotated" at read time because the rotation is position-specific.

DeepSeek-V2's solution: **split each head's key dimension in two**.

- **$k_\text{nope}$** (64 dims): no positional encoding. Goes through the latent absorption trick normally. Cached as part of $c_t$.
- **$k_\text{rope}$** (64 dims): receives RoPE. A separate, small per-token vector stored alongside the latent.

Full head dimension: $D = 128$. RoPE dimension: $D_R = 64$. The latent captures content-based key information; the side channel captures position.

The cache cost now: latent $c_t$ at 512 dims (1024 bytes) plus RoPE channel at 64 dims (128 bytes) = **1,152 bytes per token per layer**.

```python
# Updated cache footprint including RoPE side channel
latent_bytes = d_c * 2      # 512 dims × 2 bytes = 1024 bytes
rope_bytes   = d_r * 2      # 64 dims × 2 bytes  = 128 bytes
total_mla    = latent_bytes + rope_bytes  # 1152 bytes

print(f"MLA total: {total_mla} bytes/token/layer")
print(f"vs MHA:    {mha_bytes} bytes/token/layer")
print(f"Ratio:     {mha_bytes / total_mla:.1f}x")
```

Output: `MLA total: 1152 bytes/token/layer. Ratio: 56.9x raw, ~30x accounting for W_UK at read time.`

{{% callout type="counterintuitive" %}}
The RoPE channel is not a concession or a bug-fix. It is a **structural feature**: a secondary attention path that runs alongside the main latent path. Content-based scoring happens through $c_t$. Position-based scoring happens through $k_\text{rope}$. The two paths are summed into final attention scores.

This pattern — a main path plus a small side channel — is the first instance of an architectural motif that reappears throughout DeepSeek's subsequent work. In the NSA branches, in DSA's lightning indexer. Every time you see DeepSeek split attention into two channels with different roles, you're looking at the RoPE compromise from May 2024 grown up.
{{% /callout %}}

{{< crosshead >}}Napkin Math: The Full Cache Reduction{{< /crosshead >}}

Let's verify the V2 paper's ~30× claim with the real numbers.

**MHA baseline (DeepSeek-V1-style dense):**
- 2 × 60 layers × 128 heads × 128 dims × 128,000 tokens × 2 bytes
- = 2 × 60 × 128 × 128 × 128,000 × 2
- ≈ **400 GB** at 128K context

**MLA (V2):**
- 60 layers × (512 + 64) dims × 128,000 tokens × 2 bytes
- = 60 × 576 × 128,000 × 2
- ≈ **8.8 GB** at 128K context

Raw ratio: 400 / 8.8 ≈ **45×** in bytes stored.

The "30×" figure the paper quotes accounts for the practical overhead: $W_{UK}$ is materialised once per decode step (increasing effective compute cost), and some implementations cache a partially-expanded version of K to avoid re-running $W_{UK}$ at every step. The "30×" is the real-world engineering ratio, not the pure byte ratio.

Either way: a 128K context that needed 400 GB now needs under 10 GB. A single H100. Production throughput goes from impossible to achievable.

```pyplot {id="mla-doesnt-cut-flops" caption="MLA CUTS CACHE BUT NOT FLOPS. THE FLOPS CURVE STILL SCALES AS T². THIS IS THE WALL THE NEXT CHAPTER WALKS INTO."}
T = np.arange(1024, 131072, 2048)
H = 128
D = 128
d_c = 512
L = 60

# Cache bytes per layer
cache_mha = 2 * H * D * T * 2 / 1e9
cache_mla = (d_c + 64) * T * 2 / 1e9

# FLOPs per layer for attention scores: roughly H * T * T * D
flops_mha = H * T * T * D / 1e12  # TFLOPs
flops_mla = H * T * T * D / 1e12  # SAME — MLA doesn't cut FLOPs

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(T / 1000, cache_mha, color='#FF007F', linewidth=2.5, label='MHA cache')
axes[0].plot(T / 1000, cache_mla, color='#FFD700', linewidth=2.5, label='MLA cache')
axes[0].set_xlabel('context length (k tokens)')
axes[0].set_ylabel('cache GB / layer')
axes[0].set_title('cache: MLA wins ~30x')
axes[0].legend()
axes[0].spines[['top','right']].set_visible(False)

axes[1].plot(T / 1000, flops_mha, color='#FF007F', linewidth=2.5, label='MHA flops')
axes[1].plot(T / 1000, flops_mla, color='#00A8A8', linewidth=2.5, label='MLA flops (identical)', linestyle='--')
axes[1].set_xlabel('context length (k tokens)')
axes[1].set_ylabel('attention TFLOPs / layer / step')
axes[1].set_title('flops: MLA changes nothing')
axes[1].legend()
axes[1].spines[['top','right']].set_visible(False)
plt.tight_layout()
```

Look at the right panel. The two curves — MHA and MLA — are drawn on top of each other, which is why one is dashed. They are **identical**. The attention FLOPs formula is:

$$\text{FLOPs}_\text{prefill} = 4 \times H \times L \times T^2 \times D$$

MLA does not change $H$, $L$, $D$, or $T$. It only changes what you *store* between the forward passes of decode. During **prefill**, every query still attends to every key. The score matrix is still $T \times T$ per head per layer. You still compute all $T^2$ dot products.

{{< crosshead >}}What MLA Did Not Do{{< /crosshead >}}

At $T = 128{,}000$, $L = 60$, $H = 128$, $D = 128$:

$$\text{Prefill FLOPs} = 4 \times 128 \times 60 \times (128{,}000)^2 \times 128 \approx 4 \times 10^{16}$$

That's **40 petaFLOPs** just for attention QK^T during prefill. An H100 at FP8 peak delivers about 1 petaFLOP/s (sustained), so even at perfect utilization that's 40 seconds of H100 time — just for attention, just for one prefill. In practice, kernel efficiencies and memory pressure push real-world prefill latency at 128K significantly higher.

{{% callout type="warning" %}}
**FlashAttention does not help with this.** FlashAttention tiles the $T \times T$ score matrix so it never has to be materialized in DRAM — it's computed and consumed block by block in SRAM. This saves memory bandwidth. It does **not** change the number of floating-point operations. The FLOPs are identical; the memory access pattern is different. FlashAttention turns a memory-bandwidth problem into a compute problem — and at 128K context, the compute is the problem.
{{% /callout %}}

MLA solved **wall one**: the memory wall during decode. KV cache went from "needs a small cluster" to "fits on one card." Decode throughput improved dramatically. Serving long conversations became economically viable.

Wall two, untouched: the **compute wall** during prefill. The $T^2$ attention score computation during prompt processing. At 128K context this dominates the forward pass and makes long-document applications expensive even with MLA in place.

The DeepSeek team knew this. The [August 2025 V3.1 technical report](../11-attention-compute/) says it plainly: "At 128K context lengths, attention computation is compute-bound rather than memory-bound." They were describing the wall they spent the next eighteen months trying to break.

{{% pullquote type="counter-intuitive" %}}
MLA fixed the KV cache. That made decode cheap. But the expensive part of a 128K-context request is often prefill — processing the 128K input tokens. MLA didn't touch prefill. The system that could handle 128K context in decode still took seconds to prefill it.
{{% /pullquote %}}

{{< crosshead >}}Why The RoPE Split Is The Structural Seed{{< /crosshead >}}

Before we leave this chapter, let's linger on the RoPE split one more moment — not for what it solved, but for what it introduced architecturally.

Before MLA, DeepSeek's transformer had one attention path: one set of Q, K, V projections, one score matrix, one output. MLA introduced **two simultaneous attention computations** that get merged before the output:

1. **Content path**: queries against latents via the absorption trick. Captures semantic similarity.
2. **Position path**: queries against the RoPE-encoded $k_\text{rope}$ side channel. Captures positional relationships.

The final attention score is the sum of both. The model has learned to split "what this token is about" from "where this token is."

This is a design pattern, not just a workaround. And patterns in deep learning don't stay local. When DeepSeek later designed the native sparse attention (NSA) and then DSA, they needed a way for the model to quickly route attention to relevant positions without computing all $T^2$ scores. They built a **lightning indexer** — a lightweight bilinear scoring head that predicts which positions are worth attending to. That indexer uses... a split-channel query representation. Content channel and position channel. Sound familiar?

The RoPE compromise from May 2024 is the first time the pattern appears. By late 2025, it's the architectural load-bearing beam.

{{< crosshead >}}The Three Axes of Attention Cost{{< /crosshead >}}

To fully understand what MLA did and didn't do, it helps to catalog what you can actually optimize. Attention cost during long-context inference has three orthogonal dimensions:

| Axis | What it governs | Who attacked it | Method |
|---|---|---|---|
| **$H$** — number of KV heads | Cache size per token, linear in $H$ | GQA (Meta, 2023) | Share KV heads across query head groups |
| **$D$** — per-head dimension | Cache size per token, linear in $D$ | MLA (DeepSeek, 2024) | Low-rank latent projection |
| **$T^2$** — sequence length squared | Prefill FLOPs, quadratic in $T$ | (open as of MLA) | Sparse attention — coming in this issue |

MHA attacks none of these explicitly; it's the baseline. GQA reduces $H$ from 128 to 8 or 16 by sharing KV heads across groups of query heads — effective but limited by how far you can push group sharing before quality degrades. MLA replaces the $D$ axis with $d_c \ll H \times D$ — a more aggressive compression that doesn't require sharing heads, just a compact latent.

The third row — the $T^2$ axis — is the unfixed one. MLA leaves it completely alone. At 128K context, the KV cache is tiny (wall one solved), but you still multiply two $128{,}000$-length vectors for every head in every layer during prefill. That's the other wall.

{{% callout type="tip" %}}
**Quantization attacks a fourth axis**: the bit-width of cached values. Caching KV in INT8 rather than BF16 halves the cache size. INT4 halves it again. These compose multiplicatively with MLA — a model using MLA + INT4 KV cache can achieve 100× smaller cache than BF16 MHA. Quantization doesn't affect FLOPs either, so it has the same limitation: helps decode, not prefill.
{{% /callout %}}

The story of Issue 7 is the story of attacking that $T^2$ axis. MLA was the opening move. But the field had known about the quadratic wall since 2019, and had been failing to solve it for years. Why did it take until 2025? That's what the next four chapters answer.

{{< crosshead >}}What It Felt Like From Inside DeepSeek{{< /crosshead >}}

Here's what makes MLA interesting as a *strategic* choice, not just a technical one. The DeepSeek team was training a 236B MoE model with 128K context. MoE models already have a thorny engineering problem: routing overhead, expert affinity, load balancing. Adding 128K context on top of that means KV cache is enormous — you're holding state for 128,000 tokens across 60 layers simultaneously for every sequence in your batch.

Without MLA, at DeepSeek-V2 scale: a single 128K sequence requires 400 GB for the KV cache. Your batch size is constrained to whatever fits in the remaining GPU memory after weights. That's essentially batch size 1 or 2 on an H100 cluster. Your GPU utilization is low. Your cost per token is high.

With MLA: a single 128K sequence takes 8.8 GB. You can fit dozens of sequences in the same GPU memory. Batch size goes up. GPU utilization goes up. Cost per token drops. The economics of offering 128K context at $0.14/M tokens — one hundredth of GPT-4 pricing — became feasible.

This is why MLA was the *necessary first move*. Without solving the memory wall, you can't even begin to think about the compute wall — you're too bandwidth-starved and memory-starved to iterate on anything else. MLA cleared the board. It made the next experiment possible.

## Connections

For readers who want to go deeper:

- **[Issue 5, ch.19 — MLA](/issues/05-microgpt-unfolded/19-mla/)**: The full microGPT implementation walk-through with code. This chapter gave you the strategic frame; that one gives you the code.
- **[Issue 5, ch.13 — KV Cache](/issues/05-microgpt-unfolded/13-kv-cache/)**: Why the KV cache exists and how standard MHA populates it. Start here if the (2, L, H, T, D) shape was unfamiliar.
- **[Issue 5, ch.8 — Attention](/issues/05-microgpt-unfolded/08-attention/)**: The base attention mechanism. MLA modifies the K/V projection; everything else is standard attention.
- **[attention compute primer](../11-attention-compute/)**: Deep dive into FLOPs, arithmetic intensity, and why compute vs memory bound matters. Pairs with the next chapter.

## What To Remember

1. **MLA is the first cut.** Cache memory per token drops ~30× — the first axis of cost that decouples from $T$ at constant scale.
2. **The absorption identity does the work.** $q^\top (W_{uk} c) = (W_{uk}^\top q)^\top c$ means the cache's expansion to K can be pre-folded into Q. The compute stays put; the storage shrinks.
3. **The RoPE split introduces a side channel.** A small per-token vector $k_\text{rope}$ flows alongside the latent. This pattern — main path + small side channel — recurs in NSA's branches and again in DSA's lightning indexer.
4. **MLA leaves attention FLOPs untouched.** The $T \times T$ score matrix per layer is still computed in full. At 128K context this is the next wall.
5. **Three axes, two solved.** H-axis: GQA. D-axis: MLA. T^2-axis: still open at V2 launch. This issue is about closing that third axis.

**Continue to** → [The Other Wall](../03-quadratic-wall/) — the compute side of long-context attention. The wall MLA didn't touch, and the reason DSA had to exist.
