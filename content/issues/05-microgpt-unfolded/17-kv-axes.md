---
title: "The Three Axes of KV Compression"
description: "The KV cache is a 5-dimensional tensor. You can only shrink it by attacking one of its axes. This chapter introduces the H/D/L framework that organizes every modern architecture-level compression trick — and rules out the one axis nobody has cracked."
topics: [transformer, inference, kv-cache]
tags: [microgpt, kv-cache, gqa, mla, sliding-window, ssm]
theme: cream
math: true
draft: false
date: 2026-05-14T02:57:00-04:00
issue: 5
weight: 170
techKind: mainline
techNode: kv-axes
header: default.webp
---

## A 335 GB Footnote

In May 2023, a team of researchers at Fudan University posted a paper called *"Llama1-65B Inference Memory Profiling"* to arXiv. Sixty-five billion parameters, in fp16, is 130 GB on disk. The headline number everyone quoted in the press releases. The paper's actual contribution was a table buried on page 4.

The table broke down how much GPU memory the model needed at *inference time*, as a function of context length. For a 4K-token chat, the weights dominated. For a 32K-token document, weights and cache were comparable. For a **128K-token codebase ingest**, the breakdown looked like this:

| Item | Memory |
|---|---|
| Model weights (fp16) | 130 GB |
| Activations (per-layer) | 1 GB |
| **KV cache (128K tokens)** | **335 GB** |

The cache was **2.5× bigger than the weights**. To run that single inference job you needed five $30,000 H100s just to *hold* the cache — before you had done any actual computation. The model that fit in 130 GB had become, in production, a 465 GB problem.

This is the moment the KV cache stopped being a footnote and started being the *thing*. Every major frontier-model architecture released since — Llama 3, GLM 4.5, Qwen3, DeepSeek V2 and V3, Gemma 3, GPT-OSS, Jamba, Kimi-Linear, Nemotron 3 Nano — has reshaped its attention block to make this number smaller.

This chapter is the **map** of how they did it.

## The Shape Of The Cache

Recall from [The KV Cache](../13-kv-cache/) that microGPT's cache is two Python lists per layer:

```python
keys   = [[] for _ in range(n_layer)]    # one list per layer
values = [[] for _ in range(n_layer)]    # one list per layer
```

Each entry `keys[li][t]` is a vector of length `n_embd` — the concatenation of `n_head` per-head slices each of length `head_dim`. In real implementations these lists become contiguous tensors, and the *full* cache is a single five-dimensional array of shape:

$$
\text{KV cache shape} = (\underbrace{2}_{\text{K and V}},\ \underbrace{L}_{\text{layers}},\ \underbrace{H}_{\text{heads}},\ \underbrace{T}_{\text{tokens}},\ \underbrace{D}_{\text{head\_dim}})
$$

Memory is the product of all five:

$$
\text{bytes} = 2 \cdot L \cdot H \cdot T \cdot D \cdot \text{(bytes per element)}
$$

Plug in Llama1-65B at 128K context, bf16: $L=80$, $H=64$, $T=128{,}000$, $D=128$, 2 bytes/element:

$$
2 \cdot 80 \cdot 64 \cdot 128{,}000 \cdot 128 \cdot 2 = 335 \text{ GB}
$$

That is the number. To shrink it, you have to grab one of the five dimensions and **make it smaller**. There are no other options. The factor of 2 ($K$ versus $V$) is locked by the math of attention. The remaining four — $L$, $H$, $T$, $D$ — are the four levers, and each one corresponds to a different family of architectural fix.

```pyplot {id="cache-shape" caption="The KV cache is a 5D tensor. Architecture-level compression attacks one of these dimensions. The fifth (the factor of 2 from K vs V) is locked by attention's math."}
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 4))
labels = ['2\n(K and V)', 'L\n(layers)', 'H\n(heads)', 'T\n(tokens)', 'D\n(head_dim)']
vals   = [2, 80, 64, 128000, 128]
display = ['×2', '×80', '×64', '×128K', '×128']
colors = ['#999999', '#FF007F', '#00A8A8', '#FFD700', '#FF8C00']
fixed  = [True, False, False, False, False]

x = list(range(len(labels)))
ax.bar(x, [1]*5, color=colors, edgecolor='#1A1A1A', linewidth=2, width=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticks([])
for i, d in enumerate(display):
    ax.text(i, 0.5, d, ha='center', va='center', fontsize=18, fontweight='bold', color='#1A1A1A')
for i, is_fixed in enumerate(fixed):
    note = 'locked' if is_fixed else 'targetable'
    ax.text(i, 1.06, note, ha='center', va='bottom', fontsize=9, fontstyle='italic', color='#555')

ax.set_title('KV cache shape (Llama1-65B @ 128K context) — 335 GB total', fontsize=12)
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(0, 1.2)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#1A1A1A')
ax.spines['bottom'].set_linewidth(2)
```

## The H-Axis: Share The Heads

Look at the inner loop of multi-head attention in microGPT:

```python
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
    ...
```

Every head has its own slice of `q`, its own slice of `k`, and its own slice of `v`. That is `n_head` *independent* sets of K/V vectors per token per layer. In Llama1-65B, $H=64$ means **64 copies** of K/V data per token per layer.

The first observation people made — going back to Noam Shazeer's [2019 MQA paper](../18-gqa/) — was: do the **K** and **V** projections really need to be that different across heads? What if many heads *shared* the same K/V?

If you collapse all `n_head` heads onto a single shared K/V — Multi-Query Attention (MQA) — the cache shrinks by a factor of $H$. That is extreme. If you instead split heads into `n_kv_head` groups and share K/V within each group — Grouped-Query Attention (GQA) — you trade off cache size for quality. Llama 3 picked `n_head / n_kv_head = 4`. Qwen3-235B-A22B picked 16.

The recipe modifies exactly one thing in microGPT: the slicing index for `k_h` and `v_h` becomes `h // group_size` instead of `h`. **One integer divide**. That is the entire architectural change.

Compression factor along the H-axis: **`n_head / n_kv_head`**, typically 4× to 16×.

→ [Chapter 18: Grouped-Query Attention](../18-gqa/) tells the full story.

## The D-Axis: Compress The Vectors Themselves

The second observation: what if a `head_dim`-long K vector contains *redundant* information you could project down before caching, and back up just before the attention dot product?

This is the DeepSeek V2 trick, called **Multi-head Latent Attention** (MLA). At cache-write time, K and V get projected through a low-rank matrix down to a much smaller "latent" representation $c$. At cache-read time, you project back up. The cache stores only $c$, not the full $K$ and $V$.

If $c$ has dimension $d_c$, the compression along the D-axis is $D / d_c$. DeepSeek V2 picks $d_c = 4 \cdot D / 9$ — chosen, the paper notes, so that the inflated K/V vectors at read time can be *fused* into the Q projection using a clever associativity trick, eliminating the read-time inflate-then-dot-product cost.

Mechanically, this replaces

```python
keys[li].append(k)        # full-width K vector
values[li].append(v)      # full-width V vector
```

with

```python
c = linear(x, state_dict[f'layer{li}.kv_down'])  # project DOWN to d_c
cache[li].append(c)
```

and pushes the *up*-projection into the read path of each attention step. The cache shrinks by the ratio $4H/9 \approx 28\times$ for DeepSeek-V2's hyperparameters.

→ [Chapter 19: Multi-head Latent Attention](../19-mla/) walks through the matrix algebra and the absorption trick that makes the read-time math free.

## The L-Axis: Don't Cache Every Layer The Same

The third observation: what if most layers do *not* need a full-context KV cache? Maybe only some of them genuinely use far-past tokens; the others are doing local pattern-matching that a much shorter window would handle.

Two flavors of architectural surgery exploit this.

**Sliding-window attention (SWA).** For a fraction of layers, replace the unbounded `for ki in keys[li]` loop with a bounded `for ki in keys[li][-W:]` — only the last $W$ tokens are visible. The cache for those layers stops growing once $T > W$. Gemma3 interleaves sliding-window with full-attention layers in a 5-to-1 ratio, achieving **6× cache compression**. GPT-OSS-120B uses a 1-to-1 interleave, getting 2×.

**State-space hybrids.** Replace some attention layers entirely with **state-space models** (Mamba, RWKV) that maintain a *fixed-size* recurrent state instead of a growing KV cache. Jamba uses an attention-to-Mamba ratio of 1:7 in its blocks, yielding **8× compression**. Kimi-Linear and Nemotron 3 Nano take softer ratios for 4× and 4.8× respectively.

Both tricks are L-axis interventions: they touch a *subset* of the $L$ layers rather than narrowing every layer's H or D. From microGPT's perspective, the change is per-layer dispatch:

```python
for li in range(n_layer):
    if attention_type[li] == 'full':
        x = full_attention(x, keys[li], values[li], ...)
    elif attention_type[li] == 'sliding':
        x = sliding_attention(x, keys[li][-W:], values[li][-W:], ...)
    elif attention_type[li] == 'mamba':
        x, state[li] = mamba_block(x, state[li], ...)
```

→ [Chapter 20: Sliding-Window Attention](../20-sliding-window/) and [Chapter 21: State-Space Hybrids](../21-ssm-hybrids/) walk through each.

## The T-Axis: The Unsolved One

The remaining lever is the **T-axis** — the token count. This is the one nobody has architecturally solved. Why?

Look back at the inner loop:

```python
attn_logits = [
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
    for t in range(len(k_h))
]
```

The attention mechanism *defines* itself as a function of *every past token*. You cannot drop a token from the cache architecturally without changing the model's mathematical contract. The model was trained with full-history attention; truncating it at inference time is a different model.

This is why every T-axis trick is a **pruning** method, not an architectural change: it tries to identify which past tokens *will not be attended to* and drop those, hoping the model agrees. H₂O, SnapKV, KVzip, KVzap — all are heuristic post-hoc selectors operating at inference time. None has been adopted in production engines like vLLM or SGLang. The accuracy–latency–faithfulness trade-off has not yet given a clean win.

Mark this distinction firmly:

- **H, D, L axes** → architectural. Bake into the model. Train once, save forever.
- **T axis** → algorithmic. Decide per-inference what to keep. Has never quite worked.

The architectural fixes are the ones that ship in production today.

## The Compression Table

A consolidated lookup of the public production numbers as of mid-2026:

| Model | Family | Axis | Compression | Mechanism |
|---|---|---|---|---|
| Llama 3 | Meta | H | **4×** | GQA, `n_kv_head = n_head / 4` |
| GLM 4.5 | Zhipu | H | **12×** | GQA |
| Qwen3-235B-A22B | Alibaba | H | **16×** | GQA |
| DeepSeek V2 / V3 | DeepSeek | D | **~28×** (= 4H/9 with $H=64$) | MLA latent decomposition |
| GPT-OSS-120B | OpenAI | L | **2×** | 50% sliding-window layers |
| Gemma 3 | Google | L | **6×** | 5:1 sliding:full interleave |
| Jamba | AI21 | L | **8×** | 1:7 attention:Mamba interleave |
| Kimi-Linear | Moonshot | L | **4×** | Linear-attention hybrid |
| Nemotron 3 Nano | NVIDIA | L | **4.8×** | Mamba-2 hybrid |

Notice nothing in the table touches the T-axis.

Notice also that **compositions** are perfectly legal: a model could ship MLA *and* sliding-window layers *and* GQA on top of the few remaining full-attention layers. DeepSeek V3 actually does exactly this. The 2026 frontier is no longer "which axis do we attack" — it is "how do we stack the attacks."

## What To Remember

1. **The KV cache is a 5D tensor of shape $(2, L, H, T, D)$.** The factor of 2 is locked. The other four are the four levers.
2. **H-axis** → Grouped-Query Attention. Share K/V across head groups. 4×–16× in production.
3. **D-axis** → Multi-head Latent Attention. Low-rank decompose K and V. ~28× in DeepSeek.
4. **L-axis** → Mix attention layer types. Sliding-window or SSMs for some layers. 2×–8×.
5. **T-axis** → Architecturally untouched. Every "T-axis" method is a runtime pruning heuristic, and none has shipped.
6. **Compositions are legal.** Frontier models stack two or three of these tricks together.

---

**Continue to** → [Grouped-Query Attention](../18-gqa/), the H-axis fix that became universal — Llama 3 picked 4×, Qwen3 picked 16×, and the architectural change is *one integer divide* inside the multi-head loop.
