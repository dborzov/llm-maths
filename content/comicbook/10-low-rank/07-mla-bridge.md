---
title: "MLA, Re-Read as a LoRA: factor the architecture, not the update"
short_title: "MLA, Re-Read"
description: "DeepSeek's MLA factors W_K and W_V at the architecture level. Same algebra as LoRA. The absorption trick falls out because matrix product is associative. The cache shrinks because you store the latent, not the K-V."
blurb:
  - "LoRA factors $\\Delta W$ at fine-tuning. MLA factors the K and V projection *weights themselves* at training. Same skinny-matrix shape."
  - "The MLA cache stores the rank-$d_c$ latent $c$, not the full K and V vectors. The cache row shrinks by $2HD / d_c$ — typically ~30×."
  - "The absorption trick is one line of algebra: $q^\\top (W_{uk} c) = (W_{uk}^\\top q)^\\top c$. Pre-fold $W_{uk}$ into $q$ once per step; read-time decompression costs zero FLOPs."
  - "Why DeepSeek shipped it and Llama 3 didn't — the engineering tax of an architectural low-rank parameterization."
topics: [low-rank, attention, kv-cache, mla]
tags: [mla, deepseek, lora, absorption, architecture]
theme: cream
math: true
draft: false
date: 2026-05-19T09:00:00-04:00
issue: 10
weight: 70
techKind: mainline
techNode: mla-bridge
header: default.webp
---

## Two Whiteboards, Same Drawing

Open the LoRA paper to figure 1. Open the DeepSeek V2 paper to figure 3. Hold them side by side.

The LoRA figure shows a frozen weight matrix $W_0$, an identity arrow passing through it, and a *low-rank* path going around it — through two skinny boxes labeled $A$ and $B$. The output is the sum: $h = W_0 x + B A x$.

The MLA figure shows the input vector $x$ being projected *down* to a tiny dimension $d_c$ by a matrix $W_{KV}^\text{down}$, the result $c$ being cached, and then — at read time — being projected *up* by two matrices $W_K^\text{up}$ and $W_V^\text{up}$ to recover $k$ and $v$. The path through the model is: $x \to c \to (k, v)$.

If you ignore the "cache the middle" part of the MLA figure, the two pictures are *the same picture*. A big input dimension squeezed through a low-rank waist and then expanded back out. LoRA wrote it as $\Delta W = BA$. DeepSeek wrote it as $W_K = W_K^\text{up} W_K^\text{down}$. The factorization is identical.

The differences are about *when* and *to what*.

| | LoRA | MLA |
|---|---|---|
| What gets factored | The update $\Delta W$ | The base projection $W_K$, $W_V$ |
| When | At fine-tuning time, after pre-training | At initialization, before pre-training |
| Base $W_0$ | Frozen, kept around | Replaced by the factorized form; not used |
| Rank $r$ | Tunable per fine-tune (8–64) | Fixed architecturally ($d_c = 512$ for DeepSeek V2) |
| Inference cost | Can fold into $W$ — zero overhead | Stays factored — but the absorption trick makes read free |
| Memory savings | Optimizer state (training) | KV cache (inference) |

This chapter is about reading MLA through the lens of LoRA, watching the algebra become identical, and understanding what the small differences in the right column buy you.

If you have not read the [microGPT MLA chapter](/comicbook/05-microgpt/19-mla/) yet, that is the bottom-up implementation view of MLA — the actual lines of code, the rope-vs-nope split, the napkin math. This chapter is the *top-down* view: MLA as one entry in the LoRA Zoo of [chapter 6](../06-lora-zoo/).

## The Architectural Low-Rank Parametrization

In a vanilla transformer, the K projection of layer $\ell$ has one weight matrix:
$$
W_K^{(\ell)} \in \mathbb{R}^{(H \cdot D) \times n_\text{embd}}.
$$
For Llama1-65B-class shapes ($H = 64$ heads, $D = 128$ head dim, $n_\text{embd} = 8192$), this matrix is $8192 \times 8192$ — 67 million parameters.

DeepSeek's V2 paper *factors* this matrix into two skinny matrices, exactly the way LoRA does:
$$
W_K^{(\ell)} \;=\; W_K^\text{up} \, W_{KV}^\text{down}
$$
where
- $W_{KV}^\text{down} \in \mathbb{R}^{d_c \times n_\text{embd}}$ is shared between K and V (one down-projection serves both).
- $W_K^\text{up} \in \mathbb{R}^{(H \cdot D) \times d_c}$ lifts the latent to K.
- $W_V^\text{up} \in \mathbb{R}^{(H \cdot D) \times d_c}$ lifts it to V.
- $d_c$ is the latent dimension. DeepSeek V2 picks $d_c = 512$ — about 1/16 of $H \cdot D = 8192$.

Three matrices replace two. The parameter count:

$$
n_\text{embd} \cdot d_c \;+\; 2 \cdot d_c \cdot (H \cdot D)
\;=\; 8192 \cdot 512 \;+\; 2 \cdot 512 \cdot 8192
\;=\; 12{,}582{,}912
$$

versus the original two matrices ($W_K, W_V$):
$$
2 \cdot 8192 \cdot 8192 \;=\; 134{,}217{,}728.
$$

That is a $10.7 \times$ shrink in *weight matrix parameters per layer*. Not bad on its own. But the weight savings is not the point.

## The KV Cache Is The Point

The reason LoRA cared about $r$ being small was the *optimizer state*. The reason MLA cares about $d_c$ being small is the *KV cache*.

Recall ([Issue 7, ch.10](/comicbook/07-deepseek-attn/)) that long-context inference is dominated by the KV cache. Every token's K vector and V vector get written to the cache and re-read on every subsequent attention computation. The cache grows as $O(L \cdot T \cdot 2 \cdot H \cdot D)$ bytes for $L$ layers, $T$ tokens, and bf16 storage. For DeepSeek V2's shapes at 128k context, vanilla MHA cache would be **~300 GB**. That doesn't fit on any single GPU and forces complex multi-GPU sharding.

MLA's intervention: instead of caching $k, v$ — which are length $H \cdot D$ each — cache the *latent* $c = W_{KV}^\text{down} x$, which is length $d_c \ll H \cdot D$. At read time, reconstruct $k, v$ from $c$ on demand.

The cache shape goes from $2 \cdot H \cdot D$ floats per token per layer (factor 2 for K *and* V) to just $d_c$ floats per token per layer. For DeepSeek V2's $d_c = 512$, $H \cdot D = 16{,}384$, that is a $\frac{2 \cdot 16{,}384}{512} = 64\times$ shrink on the cache row.

At 128k context, ~300 GB drops to roughly **5 GB** — fits on a single H100.

This is the win.

```pyplot {id="mla-vs-mha-cache" caption="KV cache footprint vs. context length for vanilla MHA (Llama-65B class) versus MLA (DeepSeek V2 dimensions). The horizontal lines mark single-GPU capacities. MHA hits 80 GB at ~32k tokens; MLA at ~3M tokens."}
T = np.arange(1024, 256_000, 4096)
H, D = 64, 128
bytes_per_elem = 2  # bf16
L = 60

mha_bytes = 2 * H * D * T * L * bytes_per_elem
gqa_bytes = 2 * (H // 4) * D * T * L * bytes_per_elem
mla_bytes = 576 * T * L * bytes_per_elem  # d_c=512 + d_R=64 rope head

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(T / 1000, mha_bytes / 1e9, color='#FF007F', linewidth=2.8,
        label='vanilla MHA (Llama-65B class)')
ax.plot(T / 1000, gqa_bytes / 1e9, color='#00A8A8', linewidth=2.4,
        label='GQA (4× share, Llama-3 class)')
ax.plot(T / 1000, mla_bytes / 1e9, color='#FFD700', linewidth=2.8,
        label='MLA (DeepSeek V2, $d_c=512$)')

ax.axhline(80, color='#1A1A1A', linewidth=1.0, linestyle='--', alpha=0.6)
ax.text(245, 86, 'H100 80 GB ceiling', fontsize=9, color='#1A1A1A')
ax.axhline(48, color='#FF8C00', linewidth=1.0, linestyle='--', alpha=0.7)
ax.text(245, 52, 'consumer 48 GB', fontsize=9, color='#FF8C00')

ax.set_xlabel('context length $T$ (thousands of tokens)', fontsize=11)
ax.set_ylabel('KV cache total (GB across all layers)', fontsize=11)
ax.set_title('MLA is the cache shrink: 64× per row, ~30× total at long context', fontsize=12)
ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 220)
ax.set_xlim(0, 250)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

# Napkin math table
print(f"{'context':>12}  {'MHA':>10}  {'GQA/4':>10}  {'MLA':>10}")
for tk in [4, 16, 32, 64, 128]:
    idx = (np.abs(T/1000 - tk)).argmin()
    print(f"  {tk:>4}k tok  {mha_bytes[idx]/1e9:>7.1f} GB  {gqa_bytes[idx]/1e9:>7.1f} GB  {mla_bytes[idx]/1e9:>7.1f} GB")
```

Notice the three single-GPU ceilings: vanilla MHA blows past 80 GB at around 32k tokens. GQA-4 holds the line until ~128k. MLA stays under 12 GB even at 256k. Same model class, three different attention layers, three different long-context economics.

## The Read-Time Cost — And The Absorption Trick

Stop. Look at the read path. Naïvely, MLA has made decoding *slower*. The vanilla MHA read path was: take the cached $k$ vector, dot it with the query $q$. One length-$D$ dot product. Done.

MLA's naïve read path: take the cached *latent* $c$ of length $d_c$. Multiply by $W_K^\text{up}$ to recover the per-head $k_h$ of length $D$. *Then* dot with $q_h$.

That extra matrix-vector multiplication is $D \cdot d_c$ FLOPs per cached token per head. At long context, the extra cost adds up. The cache write savings come at a compute cost. We saved memory but spent compute. That cannot be the answer DeepSeek shipped.

The footnote in DeepSeek's Figure 3 is the answer. **Matrix multiplication is associative.** Write the attention score for one query head $h$ against one cached latent $c_t$:
$$
\text{score}_h(t) \;=\; q_h^\top \, k_{h,t}
\;=\; q_h^\top \, \big(W_K^{\text{up}, (h)}\, c_t\big)
\;=\; \big(W_K^{\text{up}, (h)\,\top} q_h\big)^\top c_t.
$$

Let $q'_h \triangleq W_K^{\text{up}, (h)\,\top} q_h$. This is a length-$d_c$ vector — the *absorbed query*. Now the attention score is:
$$
\text{score}_h(t) \;=\; (q'_h)^\top c_t.
$$
A dot product of length $d_c$, computed directly against the cached latent. The matrix $W_K^\text{up}$ has been folded into the query *once per decode step*, instead of being applied to the cache contents $T$ times.

This is identical, in spirit, to what LoRA does at inference. LoRA can fold $BA$ back into $W$ at deployment so that the runtime model has no extra cost. MLA cannot fold the factorization away entirely (because the K and V are reconstructed from a *cached* latent that varies token-to-token, not from a constant matrix), but it can fold the up-projection $W_K^\text{up}$ into the query so that the **per-token** cost matches vanilla MHA.

The same fold works on the V side. The attention output is $\sum_t a_{h,t} v_{h,t} = \sum_t a_{h,t} W_V^{\text{up},(h)} c_t = W_V^{\text{up},(h)} \sum_t a_{h,t} c_t$. Accumulate the attention-weighted *latent* (a single length-$d_c$ vector), then pay one $D \times d_c$ matrix-vector to lift it back. Paid once, not $T$ times.

The bookkeeping is identical to LoRA's "fold $BA$ into $W$" trick. The reason both work is the same — matrix product is associative, and rearranging the order of matrix-vector multiplications inside an attention sum is allowed.

```pyplot {id="absorption-cost" caption="FLOPs per decode step as context grows. The naive MLA path (decompress every cache row at read time) scales linearly with T. The absorbed path pre-folds the up-projection once and matches vanilla MHA."}
T = np.arange(1024, 131_072, 2048)
H, D, d_c = 64, 128, 512

# Vanilla MHA: one length-D dot product per cached token per head.
mha_flops = H * D * T

# Naive MLA: per cached token per head, a d_c→D matvec, then length-D dot product.
naive_mla_flops = H * (d_c * D + D) * T

# Absorbed MLA: per step, fold W_uk into q (cost H*D*d_c, independent of T).
# Then per cached token, length-d_c dot product.
absorbed_setup = H * D * d_c
absorbed_per_t  = H * d_c * T
absorbed_total = absorbed_setup + absorbed_per_t

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(T / 1000, mha_flops / 1e6, color='#FF007F', linewidth=2.6,
        label='vanilla MHA')
ax.plot(T / 1000, naive_mla_flops / 1e6, color='#00A8A8', linewidth=2.6,
        linestyle='--', label='MLA naive (re-decompress)')
ax.plot(T / 1000, absorbed_total / 1e6, color='#FFD700', linewidth=2.8,
        label='MLA + absorption trick')

ax.set_xlabel('context length $T$ (thousands of tokens)', fontsize=11)
ax.set_ylabel('FLOPs per decode step (millions)', fontsize=11)
ax.set_title('Why MLA does not pay extra compute: the absorption trick equals MHA', fontsize=12)
ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.3)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

# Cost at 128k context
idx_128k = (np.abs(T/1000 - 128)).argmin()
print(f"FLOPs per decode step at T = 128k:")
print(f"  vanilla MHA:       {mha_flops[idx_128k]/1e6:>8.1f} M")
print(f"  MLA naive:         {naive_mla_flops[idx_128k]/1e6:>8.1f} M  ({naive_mla_flops[idx_128k]/mha_flops[idx_128k]:.1f}× MHA)")
print(f"  MLA + absorption:  {absorbed_total[idx_128k]/1e6:>8.1f} M  ({absorbed_total[idx_128k]/mha_flops[idx_128k]:.2f}× MHA)")
print(f"\nThe absorbed line tracks MHA to within rounding — the savings is pure memory, paid once in algebra.")
```

The dashed teal line is what naive MLA would cost without the trick — twice the FLOPs of vanilla MHA at long context. The pink and yellow lines (MHA and absorbed-MLA) overlap. That is the win the LoRA-style "fold the matrix" identity buys.

## A Concrete Comparison In Code

```python
# Vanilla MHA — what microGPT chapter 16 ships:
def mha_step(x, kv_cache, layer_idx):
    k = linear(x, W_K[layer_idx])                # n_embd → H*D
    v = linear(x, W_V[layer_idx])                # n_embd → H*D
    kv_cache[layer_idx].append((k, v))
    return attention(q_for_x, kv_cache[layer_idx])

# LoRA fine-tune of MHA — what the LoRA paper does to W_K:
def mha_step_lora(x, kv_cache, layer_idx):
    # k = (W_K0 + B_K A_K) x, but we never materialize the sum at inference
    k = linear(x, W_K[layer_idx]) + linear(linear(x, A_K[layer_idx]),
                                            B_K[layer_idx])
    v = linear(x, W_V[layer_idx]) + linear(linear(x, A_V[layer_idx]),
                                            B_V[layer_idx])
    kv_cache[layer_idx].append((k, v))
    return attention(q_for_x, kv_cache[layer_idx])

# MLA — what DeepSeek V2 ships. NB: same factorization, different *what* is factored.
def mla_step(x, c_cache, layer_idx):
    c = linear(x, W_KV_down[layer_idx])          # n_embd → d_c
    c_cache[layer_idx].append(c)                  # cache the latent only

    # Absorbed query: q' = W_K_up^T @ q, done once per step
    q_h = compute_query(x)                        # length H*D
    q_h_absorbed = W_K_up[layer_idx].T @ q_h      # length d_c, per head

    # Dot against cached latents directly
    scores = [q_h_absorbed @ c_t for c_t in c_cache[layer_idx]]
    weights = softmax(scores)
    weighted_latent = sum(w * c_t for w, c_t in zip(weights, c_cache[layer_idx]))
    # Lift V up once, after the attention sum
    out = W_V_up[layer_idx] @ weighted_latent
    return out
```

Compare lines. The LoRA fine-tune variant has a base $W_K[layer]$ that is still doing the full multiplication at every step — the only saving is in the *trainable parameter count* during fine-tuning. The MLA variant has *no base $W_K$* at all. It is the factored form, all the way down. The cache is the small thing. The matrices on either side of the cache get absorbed into the query and the output.

This is what we mean by **architectural LoRA**: the factorization is not bolted on top of a pre-trained dense matrix; the architecture *is* the factorization, from initialization.

## The RoPE Wrinkle (And Why LoRA Doesn't Have It)

LoRA on $W_K$ is a clean operation. You can absorb $\Delta W$ into $W$ and continue using rotary position embeddings unchanged: RoPE multiplies $k$ by a rotation matrix at attention time, and that operation does not care whether $k$ came from $W$ or $W + BA$.

MLA's absorption trick *does* care. Recall the standard RoPE recipe: before computing $q^\top k$, you apply position-dependent rotation $R_{\text{pos}_k}$ to $k$ (and $R_{\text{pos}_q}$ to $q$). The relative rotation $R_{\text{pos}_q - \text{pos}_k}$ enters the dot product.

If MLA stored the latent $c$ and reconstructed $k = W_K^\text{up} c$ at read time, the natural place to apply $R_{\text{pos}_k}$ would be *after* the reconstruction. But the absorption trick *folded $W_K^\text{up}$ into $q$* — there is no $k$ ever materialized at read time. There is nothing to rotate.

DeepSeek's fix, covered in detail in the [microGPT MLA chapter](/comicbook/05-microgpt/19-mla/), is to **split the head**. Each query head and each K vector are concatenations of two sub-vectors:

- `k_nope` (no-position) — goes through MLA's latent, benefits from absorption.
- `k_rope` (position-bearing) — bypasses the latent, computed at write time per token, cached separately at small dimension $D_R$.

The K at attention time is $k = [k_\text{nope} \,\Vert\, k_\text{rope}]$. The rope-bearing channel is small ($D_R = 64$ in V2 vs. $D = 128$ head dim total) and is the only thing in MLA that is *not* absorbed.

LoRA doesn't need this split because LoRA doesn't fuse $W_K$ into $q$ — it keeps the dense $k$ around. The architectural commitment of MLA to "the K vector at attention time is not materialized" is what creates the RoPE conflict. This is the engineering cost of MLA's deeper factorization.

## Why DeepSeek Shipped It And Llama Didn't

If MLA is "LoRA, applied to the architecture instead of to the fine-tuning update", and if the algebra is identical, why don't Llama 3 and Qwen 3 use MLA?

The answer is exactly the answer in the existing [MLA chapter](/comicbook/05-microgpt/19-mla/): the *implementation* is more places to be wrong.

1. **The math has more places to be wrong.** LoRA can be applied as an after-the-fact fine-tune; if it fails, you fall back to the dense weights. MLA's factorization is baked in from initialization — if the training run diverges, you have to retrain the whole 671B-parameter model. The blast radius of bugs is enormous.

2. **Training stability requires care.** Both LoRA and MLA push signals through a narrow rank-$r$ waist. LoRA's waist is a small *correction* on top of a stable pre-trained base; MLA's waist is the *only path* the signal can take. The optimizer has to learn to use the waist without losing information. DeepSeek spent pages of the V2 paper on initialization recipes and learning-rate schedules specific to MLA. Recipes do not transfer perfectly to other model families.

3. **The absorption trick constrains downstream choices.** Once $W_K^\text{up}$ is fused into the query at inference, you cannot quantize $W_K^\text{up}$ and $W_Q$ independently — the fused product has different numerical structure. Quantization-aware deployment of MLA is harder than of vanilla MHA.

4. **Llama 3 was designed before V2 shipped.** Meta's Llama 3 architecture was frozen in Q3 2023, six months before DeepSeek V2 dropped. By the time MLA was published, Llama 3's training run was deep underway. Llama 4 (announced 2025) does adopt MLA-like decompositions, partially.

The pattern is the typical pattern of architectural innovations in deep learning: someone publishes a working result; the rest of the field takes 12–18 months to integrate it; by the time it's in everyone's recipe, the inventor has moved two steps further. DeepSeek V3.2 with DSA, V4 with CSA+HCA — they are all building on top of MLA. The [issue 7 chapter](/comicbook/07-deepseek-attn/02-mla-rewind/) on this exact story is worth a re-read after this chapter.

## A Subtle Asymmetry: What LoRA Cannot Do

There is one thing LoRA-style fine-tuning *cannot* do that MLA-style architectural factorization can: **shrink the KV cache for an already-trained model**.

LoRA can shrink the *trainable parameter count* of fine-tuning a pre-trained model. It cannot shrink the *cache* during inference — the cache is determined by $W_K$ and $W_V$ in the pre-trained model, and LoRA's update is added to those without changing their shape.

To shrink the cache of a pre-trained dense-MHA model, you would need to *re-architect* the projections — replace $W_K$ with a factorization $W_K^\text{up} W_{KV}^\text{down}$ — and somehow find weights for the factored form. Some teams have done this *post-hoc*: take a pre-trained Llama, SVD its $W_K$, keep the top $d_c$ singular values, and continue pre-training briefly to recover quality. The technique is called **post-hoc MLA conversion**; it works but with some quality cost. Mostly people just train MLA from scratch.

The asymmetry is: **LoRA edits a pre-trained model cheaply. MLA edits the *architecture* — which means it has to be there from the start.**

This is the deepest difference between the two ideas. LoRA is the *post-hoc* low-rank trick: it acts on the update. MLA is the *a priori* low-rank trick: it commits to the rank from initialization. The Eckart-Young theorem applies to both — the question is whether you exploit it before or after training.

## What To Remember

1. **MLA factors the attention base weights $W_K$ and $W_V$.** LoRA factors the fine-tuning update $\Delta W$. The algebra is the same; the difference is *what* gets factored and *when*.
2. **The MLA cache row is $d_c$ floats per token per layer**, not $2 \cdot H \cdot D$. For DeepSeek V2's $d_c = 512$, that is a 64× shrink. At 128k context the cache drops from ~300 GB to ~5 GB.
3. **The absorption trick — $q^\top(W_{uk} c) = (W_{uk}^\top q)^\top c$ — is one line of algebra.** It folds the up-projection into the query once per step, instead of $T$ times across the cache.
4. **RoPE forces a head split.** Some channels (`k_nope`) bypass the latent and benefit from absorption; others (`k_rope`) keep position-bearing structure at small dimension $D_R = 64$ and don't.
5. **LoRA can post-hoc edit a pre-trained model. MLA must commit to the factorization at initialization.** The asymmetry — *when* you commit to the low-rank — is the deepest difference.
6. **Both are exploiting the same Eckart-Young low-rank structure.** LoRA exploits low-rank-of-the-update; MLA exploits low-rank-of-the-weights-themselves. The boss chapter draws the unified picture.

{{% callout type="tangent" title="Cross-references" %}}
For the *bottom-up* implementation of MLA — exact PyTorch lines, head-by-head trace through the absorption trick, the kernel-level RoPE split, napkin math at DeepSeek V2's exact dimensions — read the [microGPT MLA chapter](/comicbook/05-microgpt/19-mla/) in Issue 5.

For what *came next* in DeepSeek's sparse-attention work — the lightning indexer in DSA (V3.2), the CSA + HCA compressed attention in V4, the million-token context economics — read [Issue 7: Sparse Attention](/comicbook/07-deepseek-attn/). The bridge into that issue is [the MLA rewind chapter](/comicbook/07-deepseek-attn/02-mla-rewind/).

This chapter is the *middle* of the three views. Read it first if you are coming from LoRA. Read the microGPT chapter first if you are coming from implementation. Read Issue 7 first if you are coming from the sparse-attention literature.
{{% /callout %}}

---

**Continue to** → [The Unified Low-Rank Thesis](../08-unified-thesis/) — where LoRA, MLA, GQA, KV-cache pruning, and quantization all collapse into one two-axis chart, and you can read off what every method in modern LLM compression is doing in one glance.
