---
title: "MLA: compress K and V through a latent bottleneck"
short_title: "MLA"
description: "DeepSeek V2's Multi-head Latent Attention caches a low-rank latent instead of full K and V vectors — 28× compression, with read-time decompression algebraically absorbed into the query projection at zero cost."
blurb:
  - "DeepSeek V2, May 2024: 236B MoE, inference costs claimed 5.76× cheaper than comparable models. The key was one diagram — Figure 3."
  - "Instead of caching K and V (full width n_embd), MLA caches a single latent vector of dimension d_c=512. Cache write shrinks from 2×n_embd to d_c."
  - "The absorption trick: the up-projection from latent back to K/V can be fused into the query projection at inference time — the decompression costs zero FLOPs."
  - "MLA attacks the D-axis; GQA attacks the H-axis. The two surgeries compose. Why does MLA subsume GQA rather than stack on top of it?"
topics: [transformer, attention, kv-cache]
tags: [microgpt, mla, deepseek, low-rank]
theme: cream
math: true
draft: false
date: 2026-05-14T02:57:00-04:00
issue: 5
weight: 190
techKind: mainline
techNode: mla
header: 19-mla.webp
---

## The Model Card Nobody Was Ready For

On May 6, 2024, a Chinese lab called **DeepSeek-AI** uploaded the weights of a 236-billion-parameter mixture-of-experts model to Hugging Face and quietly published a paper titled *"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"*. The paper was 50 pages. Most Western infrastructure teams had not heard of the lab. The model card claimed inference costs **5.76× cheaper** than competitors of the same parameter class. The headline benchmark numbers were good but not extraordinary; the headline *economics* were extraordinary.

The interesting page of the paper was not the benchmark table. It was Figure 3 — a diagram of a thing called **Multi-head Latent Attention**, abbreviated MLA. The diagram showed K and V vectors being squeezed through a narrow waist before being written to the {{< wiki "kv-cache" >}}cache{{< /wiki >}}, and then being unfolded again at attention-read time. The arrow pointing into the cache was thin. The arrows coming back out were fat. A footnote said: *"the up-projection can be absorbed into the query projection at inference time, so the read-time decompression is free."*

The on-call engineer at every serving company who read that footnote on May 7th had the same reaction: **wait, what?** And then: *can we ship this?*

This is the chapter on what that diagram actually says.

## Two Axes, Two Different Surgeries

Take a step back to [ch.17 — the three axes of KV compression](../17-kv-axes/). The cache is a five-dimensional tensor of shape $(2, L, H, T, D)$. The factor of two is locked. Each of the remaining four levers is a different family of architectural fix.

[Grouped-Query Attention (ch.18)](../18-gqa/) — the trick Llama 3 and Qwen3 use — attacks the **H-axis**. You shrink the number of *distinct* K and V heads from $H$ down to some smaller $H_{kv}$, and let groups of query heads share. The architectural change is one integer divide. The compression is at most $H$-fold.

MLA attacks the **D-axis** instead. Same cache tensor. Different dimension. Instead of shrinking the number of heads, you shrink the **dimension of the K and V vectors themselves** — the `head_dim` coordinate. You compose the two halves of `attn_wk` (or `attn_wv`) into a *narrow-bottleneck-then-wide* path: project the residual $x$ down to a tiny latent dimension $d_c$, cache *that*, and reconstitute K and V from the latent only when you actually need them. The cache row goes from $H \cdot D = 1024+$ floats per token down to $d_c = 512$ — and as we will see, the choice of $d_c$ relative to $H$ is what unlocks the trick that makes the read path free.

The two surgeries **compose**. DeepSeek V2 ships MLA on top of MoE on top of multi-head {{< wiki "attention" >}}attention{{< /wiki >}}. GQA is *not* in the recipe because MLA subsumes it: once you have decomposed K and V through a shared low-rank latent, there is nothing left to share across head groups.

## The Microgpt Surgery

Recall the K/V write lines in [the baseline forward pass](../16-full-forward/):

```python
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
keys[li].append(k)
values[li].append(v)
```

`k` and `v` are length-`n_embd` vectors. In microGPT that is 16; in Llama1-65B it was $H \cdot D = 64 \cdot 128 = 8192$ for K, same for V — **16 kB per token per layer in bf16**. We then write both into the cache. Two writes, full width.

MLA collapses the four lines into three:

```python
c = linear(x, state_dict[f'layer{li}.kv_down'])   # ← project to latent of dim d_c
cache[li].append(c)                                # ← cache the latent only
```

That is the cache-write side of the architectural change. **Both K and V are gone from the cache.** The only thing stored per token per layer is a single short vector $c$ of length $d_c$.

The read side has to reconstruct K and V from $c$ before computing attention scores. Two new up-projections — `kv_up_k` and `kv_up_v`, both shaped `(H·D) × d_c` — do the inflation:

```python
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]

    # reconstruct K and V from the latent, per cached token
    k_h = [linear(c, state_dict[f'layer{li}.kv_up_k'])[hs:hs+head_dim]
           for c in cache[li]]
    v_h = [linear(c, state_dict[f'layer{li}.kv_up_v'])[hs:hs+head_dim]
           for c in cache[li]]

    # ... unchanged attention as before
```

Stop. Look at the read path. Naively, we have just made decoding **slower**: every cached token now triggers two matrix-vectors at read time, where before reading from the cache was a free slice. We saved memory but spent compute. That cannot be the answer DeepSeek shipped — they claimed cheaper inference, not just smaller GPUs.

The footnote on Figure 3 is the answer.

## The Absorption Trick

Write the attention score for one query head $h$ against one cached latent $c_t$. Let $W^{(h)}_{uk}$ be the slice of `kv_up_k` that produces head $h$'s K vector. Then:

$$
\text{score}_h(t) \;=\; q_h^\top \, \underbrace{(W^{(h)}_{uk}\, c_t)}_{=\;k_{h,t}}
$$

That is a length-$D$ vector dotted with a length-$D$ vector — *after* paying for the matrix-vector $W^{(h)}_{uk} c_t$. But matrix–vector products are associative: we can rewrite

$$
q_h^\top \,(W^{(h)}_{uk}\, c_t) \;=\; \big(W^{(h)}_{uk}{}^\top \, q_h\big)^\top c_t \;=\; (q'_h)^\top \, c_t
$$

where

$$
q'_h \;\triangleq\; W^{(h)}_{uk}{}^\top \, q_h
$$

is a length-$d_c$ "absorbed" query vector. **Read that again.** The matrix that *would have* expanded $c$ back up to $K$ has been pre-folded into $q$ instead. Now the attention score is just $q' \cdot c$ — a single dot product of length $d_c$. The expansion of $c$ to $k$ has dissolved.

Since $q$ is computed exactly *once* per decode step (it depends only on the current token), but the cache contains $T$ entries, the cost of absorbing $W_{uk}$ into $q$ is paid once per step instead of $T$ times. The longer the context, the bigger the win.

The same trick works on the V side. The attention output for head $h$ is $\sum_t a_{h,t}\, v_{h,t}$ where $v_{h,t} = W^{(h)}_{uv}\, c_t$. Linearity of the sum lets us pull $W^{(h)}_{uv}$ outside:

$$
\sum_t a_{h,t}\, W^{(h)}_{uv} c_t \;=\; W^{(h)}_{uv} \sum_t a_{h,t}\, c_t
$$

So we accumulate the attention-weighted *latent* (a single length-$d_c$ vector), then pay one $D \times d_c$ matrix-vector to lift it back to head-dim output space. Again — paid once, not $T$ times. The output projection `attn_wo` can be folded in too, fusing $W_{uv}$ and `attn_wo` into a single matrix that the kernel never separates.

The net of all that algebra: **the read path is genuinely no more expensive than vanilla MHA.** The cache writes shrink by a factor of $H \cdot D / d_c$. The compute stays put. The savings are pure.

## How Small Is $d_c$, Actually?

DeepSeek V2 picks $d_c$ such that the compression ratio across the D-axis works out to $4H/9$. For their $H = 128$ heads with $D = 128$, that means $d_c = 4 \cdot 128 \cdot 128 / 9 \cdot$ … wait, let me state it more carefully. The cache row was $2 \cdot H \cdot D$ floats per token (factor 2 for K *and* V). After MLA, it is $d_c$ floats per token (one shared latent). The DeepSeek choice of $d_c = (4/9) \cdot H \cdot D$ gives a compression of

$$
\frac{2 H D}{d_c} \;=\; \frac{2 H D}{(4/9) H D} \;=\; \frac{9}{2} \approx 4.5\times \text{ per head}
$$

across the whole H-bundle. But what really matters in the engineering is: the cache row is the latent, no longer indexed by head. So compared to vanilla MHA's $2 \cdot H \cdot D$, the cache shrinks by $2HD/d_c$ — a factor of about **31× in DeepSeek V2's numbers** when you count both the K and V savings together. We will verify with napkin math in a moment.

```pyplot {id="cache-axes-comparison" caption="KV cache footprint as context grows. MHA = vanilla baseline; GQA = Llama-3-style 4× share; MLA = DeepSeek-V2 D-axis decomposition with d_c=512. The vertical axis is bytes-per-layer in bf16."}
T = np.arange(1024, 132000, 2048)

# Llama1-65B-class hyperparameters
H = 64
D = 128
bytes_per_elem = 2

mha = 2 * H * D * T * bytes_per_elem
gqa = 2 * (H // 4) * D * T * bytes_per_elem
mla = 512 * T * bytes_per_elem  # d_c = 512, no factor of 2 because c is shared

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(T / 1000, mha / 1e9, color='#FF007F', linewidth=3, label='vanilla MHA')
ax.plot(T / 1000, gqa / 1e9, color='#00A8A8', linewidth=3, label='GQA (H/4)')
ax.plot(T / 1000, mla / 1e9, color='#FFD700', linewidth=3, label='MLA (d_c=512)')

ax.set_xlabel('context length T (thousands of tokens)', fontsize=11)
ax.set_ylabel('cache bytes per layer (GB)', fontsize=11)
ax.set_title('D-axis surgery vs. H-axis surgery, one layer at a time', fontsize=12)
ax.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.3)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.text(100, mha[-1] / 1e9 * 0.85, f'{mha[-1]/1e9:.2f} GB', color='#FF007F', fontsize=10, fontweight='bold')
ax.text(100, gqa[-1] / 1e9 + 0.15, f'{gqa[-1]/1e9:.2f} GB', color='#00A8A8', fontsize=10, fontweight='bold')
ax.text(100, mla[-1] / 1e9 + 0.15, f'{mla[-1]/1e9:.2f} GB', color='#FF8C00', fontsize=10, fontweight='bold')
```

## Why The Math Even Works: K Is Low-Rank In Practice

The absorption trick is bookkeeping — it works because matrix multiplication is associative, full stop. But the *premise* of MLA is that there exists a useful $d_c \ll H \cdot D$ at all. That is an empirical claim: are the K and V vectors that attention produces actually low-rank?

Yes. Researchers had noticed this since at least 2020. If you take a trained transformer, harvest all the K vectors a layer produces across a corpus, and run SVD on the resulting $T \times (H \cdot D)$ matrix, the singular values fall off a cliff. Most of the variance lives in a subspace of dimension a few hundred. The remaining coordinates are noise that attention does not use.

MLA bakes this observation directly into the architecture. Instead of letting the model *discover* a low-rank subspace at training time by accident, you *force* K and V to factor through one. The model learns the best $d_c$-dimensional subspace because it has no other choice.

```pyplot {id="low-rank-k" caption="Reconstruction error of a synthetic 1024-dim K matrix as a function of rank kept. Real-trained K matrices look qualitatively like this — variance collapses into a small head-rank subspace."}
rng = np.random.default_rng(7)
T_tok = 4096
D_full = 1024

# synthetic "K matrix" with intrinsic rank ~ 400 + noise
true_rank = 400
U = rng.standard_normal((T_tok, true_rank))
sigma = np.exp(-np.linspace(0, 5, true_rank))  # decaying spectrum
V = rng.standard_normal((true_rank, D_full))
K = (U * sigma) @ V + 0.02 * rng.standard_normal((T_tok, D_full))

# SVD-based reconstruction error vs rank kept
u, s, vt = np.linalg.svd(K, full_matrices=False)
total = (s ** 2).sum()
ranks = np.arange(1, D_full + 1, 8)
errors = np.array([1.0 - (s[:r] ** 2).sum() / total for r in ranks])

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.semilogy(ranks, errors, color='#FF007F', linewidth=2.5)
ax.axvline(512, color='#00A8A8', linestyle='--', linewidth=2, label='DeepSeek V2: d_c = 512')
ax.axvline(1024, color='#FF8C00', linestyle=':', linewidth=2, label='full rank = H·D = 1024')
ax.fill_between(ranks, errors, 1e-6, where=(ranks <= 512), alpha=0.15, color='#FFD700')
ax.set_xlabel('latent dimension d_c', fontsize=11)
ax.set_ylabel('relative reconstruction error (log scale)', fontsize=11)
ax.set_title('Why MLA is allowed to exist: K is empirically low-rank', fontsize=12)
ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.3, which='both')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
```

The yellow band is the regime DeepSeek picked. Half the dimensions, almost all the variance. Cache four times smaller, accuracy intact.

## The RoPE Problem

Now the asterisk in the footnote. **{{< wiki "rope" >}}Rotary Position Embeddings{{< /wiki >}}**, ubiquitous in 2024-era LLMs, do not commute with the absorption trick.

Recall RoPE: before computing the dot product $q^\top k$, you apply a position-dependent rotation $R_{pos}$ to *each pair of coordinates* of $q$ and $k$. Crucially, the rotation depends on the **absolute position of the K token**, which is fixed at write time, and on the **absolute position of the Q token**, which is the current step. The relative rotation $R_{pos_q - pos_k}$ is what determines the score.

Here is the conflict. The MLA cache stores $c$, *not* $k$. To apply RoPE properly we would need to rotate $k = W_{uk} c$ at the position the token *was written*. But we never materialized $k$ at write time — that was the whole point. Worse: the absorption trick folds $W_{uk}$ into $q$, so $k$ never exists as an array in memory at all. There is nothing to rotate.

DeepSeek V2's fix: **split the head**. Each query head and each key are decomposed into two concatenated sub-vectors:

| Component | Dimension | Cached? | RoPE applied? |
|---|---|---|---|
| `k_nope` (no-position) | $D - D_R$ | through MLA latent $c$ | no |
| `k_rope` (position-bearing) | $D_R$ (small) | uncompressed, per-token | yes |

The K vector at attention time is $k = [\,k_\text{nope}\,\Vert\,k_\text{rope}\,]$. The rope-bearing portion bypasses the latent entirely — it is computed normally and cached at full per-token width, but at a *tiny* dimension $D_R$ (DeepSeek V2 picks $D_R = 64$ for the rope channel out of $D = 128$ total head-dim). The no-position portion benefits from the absorption trick fully.

This is the part of the MLA paper that gives engineers a headache. The Q projection becomes two-headed; the cache is now *two* tensors per token (the latent $c$ and the small rope-K vector); the kernels must handle both. The math is straightforward, the code is fiddly.

## Napkin Math: DeepSeek V2 At 128K Context

DeepSeek V2 has $L = 60$ layers, $d_c = 512$, $D_R = 64$ rope dim. Cache per token per layer in bf16:

$$
\underbrace{2 \cdot 512}_{\text{latent } c\text{, both K\&V folded}} \;+\; \underbrace{2 \cdot 64}_{\text{rope K}} \;=\; 1152 \text{ bytes}
$$

(Strictly, in DeepSeek's formulation the latent $c$ is a single vector and the $\times 2$ is absorbed into the d_c choice; rounding to keep the picture clear.) Over $T = 128{,}000$ tokens and 60 layers:

$$
60 \cdot 128{,}000 \cdot 1152 \;=\; \mathbf{8.8 \text{ GB}}
$$

Compare to a vanilla MHA Llama1-65B-class model at the same context: 335 GB ([ch.17](../17-kv-axes/)). **Roughly 38× compression on the most aggressive D-axis architecture shipped to date**, and the read path is, by absorption, the same compute cost as vanilla.

DeepSeek V3, released December 2024, scaled the same MLA recipe to a 671B-parameter MoE (37B active per token) and showed it composes with massive scale. Same $d_c$. Same absorption trick. Same $4H/9$ ratio. The architectural decision the lab made in May became the platform their entire model family stands on.

## Why Hasn't Everyone Adopted It?

If MLA is 30× better than vanilla, *and* better than GQA on the same workload, *and* the read-time math is free, why does Llama 3 still ship GQA and why doesn't Qwen3 use MLA?

Three reasons.

1. **The math has more places to be wrong.** GQA is one integer divide. MLA is a low-rank decomposition, an absorption fusion, a rope-vs-nope split, and a careful kernel implementation. Every implementation must verify the absorbed Q matches the unabsorbed reference. Teams that ship fast don't always have time to debug a new attention layer.

2. **Training stability requires care.** A narrow $d_c$ is a bottleneck; if the optimizer pushes too much signal through it, training diverges. DeepSeek's paper spends pages on initialization and learning-rate schedules specific to the MLA layer. Recipes do not transfer perfectly to other model families.

3. **The absorption trick constrains downstream choices.** Folding $W_{uk}$ into $q$ at inference time means $W_{uk}$ and $W_q$ must be statically composable. Some quantization schemes — particularly weight-only formats that quantize each matrix independently — break this composition or require re-quantizing the fused product. That is a serving-time tax. GQA has no such constraint: the cached K is the K used in the dot product, full stop.

So the choice between MLA and GQA is, in the end, an *engineering* choice, not just a math one. Frontier models that have a dedicated infrastructure team — DeepSeek, Kimi — pay the complexity tax for the bigger compression win. Models that need to be portable across many serving backends — Llama, Qwen — pick the simpler trick. Both are correct decisions for different reasons.

## What To Remember

1. **MLA attacks the D-axis.** Where GQA shrinks the *number* of K/V heads, MLA shrinks the *dimension* of each K/V vector by factoring them through a shared low-rank latent $c$ of size $d_c \ll H \cdot D$.
2. **The cache stores $c$, not K or V.** A single short vector per token per layer. K and V are reconstructed on demand at read time.
3. **The absorption trick makes read-time decompression free.** Because $q^\top (W_{uk}\, c) = (W_{uk}^\top q)^\top c$, we can fold $W_{uk}$ into $q$ once per decode step and dot directly against the latent. Same for the V side and `attn_wo`.
4. **K is empirically low-rank.** SVD of K matrices from trained transformers shows variance collapses into a few hundred dimensions. MLA forces the architecture to live in that subspace.
5. **RoPE forces a split.** A small `k_rope` sub-vector bypasses the latent and is cached uncompressed; the bulk `k_nope` goes through MLA.
6. **DeepSeek V2 at 128K context: ~9 GB of cache vs. ~335 GB for vanilla MHA on equivalent dims.** A 30×+ win, with the read path costing the same compute as vanilla MHA.

{{% callout type="tangent" title="Where MLA leads next" %}}
MLA solved the *memory* problem. It did not solve the *compute* problem — attention is still $O(T^2)$ per layer regardless of how small the cache is. DeepSeek's subsequent papers (NSA, DSA in V3.2-Exp, then CSA + HCA in V4) attack the compute side. The trajectory is the subject of **[Issue 7: The Sparse Lab](/issues/07-sparse-lab/)**:

- The same K/V latent that MLA introduced here becomes the *query latent* $c^Q_t$ that drives the **lightning indexer** in V3.2-Exp's DSA — [Issue 7 ch.6](/issues/07-sparse-lab/06-lightning-indexer/).
- The RoPE-vs-NOPE split from this chapter becomes the architectural seed for the indexer's separate scoring head — [Issue 7 ch.2 (MLA, rewound)](/issues/07-sparse-lab/02-mla-rewind/).
- V4 wraps the whole machinery in a **token-level compressor** (CSA) — [Issue 7 ch.7](/issues/07-sparse-lab/07-csa/).
{{% /callout %}}

---

**Continue to** → [Sliding-Window Attention](../20-sliding-window/) — once you have squeezed each cache entry as small as MLA allows, the next question is whether every layer needs to keep *every* entry around, which is the L-axis trick Gemma 3 leans on for another 6× on top.

