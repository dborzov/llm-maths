---
title: "Sliding-Window Attention"
description: "Replace `for ki in keys[li]` with `for ki in keys[li][-W:]` in some layers. The cache stops growing past window size W in those layers; the full-attention layers keep the long-range memory. Gemma 3's 5:1 interleave is the canonical 6× recipe."
topics: [transformer, attention, kv-cache]
tags: [microgpt, sliding-window, gemma, gpt-oss, mistral]
theme: teal
math: true
draft: false
date: 2026-05-14T02:57:00-04:00
issue: 5
weight: 200
techKind: primer
techNode: sliding-window
header: default.webp
---

## The Mistral Magnet Link

On the evening of **September 27, 2023**, the team at Mistral AI did something that ten months earlier nobody would have done with a serious frontier model: they tweeted a **magnet link**. No press release. No safety statement. Just a torrent hash and the weights of *Mistral 7B*. The blog post that followed a day later was almost defiant in its brevity. Two paragraphs of benchmarks, then a section titled "Sliding Window Attention" that contained one diagram and three sentences.

The claim was simple. In every one of Mistral 7B's 32 layers, each query token would attend only to the previous **4,096** tokens — not the full sequence. Older entries in the KV cache were *unused*, and so could be discarded. This was the first time a major-lab production LLM had shipped with pure sliding-window attention. The Longformer paper that introduced the idea (Beltagy, Peters, Cohan at AI2, April 2020) and the earlier Sparse Transformers paper from OpenAI (Child et al., 2019) had been sitting on arXiv for three and four years respectively. Mistral's contribution was less invention than nerve: they trusted the trick at 7B scale, and shipped.

Two years later the trick had become a *layout choice*. **GPT-OSS-120B** (OpenAI, 2025) interleaves sliding-window and full-attention layers 1:1 — half the cache, same model. **Gemma 3** (Google DeepMind, 2025) pushes the ratio to 5:1 and gets **6× compression**. This chapter is about the line of microGPT those models edited.

## The Line, Edited

Pull the inner loop out of [ch.16](../16-full-forward/). The unbounded loops over `keys[li]` and `values[li]` are the cache reader:

```python
# baseline microGPT (chapter 9, 13)
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]            # ← all past tokens
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]          # ← all past tokens
    ...
```

That `for ki in keys[li]` over the *entire* cache is what makes the [T-axis grow without bound](../17-kv-axes/). Sliding-window attention is one slice operator:

```python
# sliding-window variant
W = 4096
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li][-W:]]       # ← last W only
    v_h = [vi[hs:hs+head_dim] for vi in values[li][-W:]]     # ← last W only
    ...
```

`[-W:]`. That's the entire architectural change. Once $T > W$, the model never reads `keys[li][0], …, keys[li][T-W-1]` again, so a sliding-window-aware cache implementation **evicts** them. The list stops growing.

> **The mask, mathematically.** Standard causal attention restricts each query to indices $\{0, 1, \ldots, t\}$. Sliding-window restricts it to $\{\max(0,\,t-W+1), \ldots, t\}$ — a width-$W$ band hugging the diagonal:
>
> $$
> \mathrm{Mask}(t, s) = \begin{cases} 0 & \text{if } t-W+1 \le s \le t \\ -\infty & \text{otherwise} \end{cases}
> $$

```pyplot {id="swa-mask" caption="Attention masks: dense causal (left) and sliding-window with W=8 (right). The sliding band hugs the diagonal — every query sees only its W most recent keys."}
T = 24
W = 8

mask_full = np.tril(np.ones((T, T)))
mask_swa  = mask_full * (np.arange(T)[:, None] - np.arange(T)[None, :] < W)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, m, title, color in [
    (axes[0], mask_full, f"causal (T={T})",     '#FF007F'),
    (axes[1], mask_swa,  f"sliding-window W={W}", '#00A8A8'),
]:
    ax.imshow(m, cmap='gray_r', aspect='equal', interpolation='nearest', alpha=0.85)
    ax.set_title(title, fontsize=12, color=color, fontweight='bold')
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_xticks([0, W, T-1])
    ax.set_yticks([0, W, T-1])
    for spine in ax.spines.values():
        spine.set_color('#1A1A1A'); spine.set_linewidth(1.5)
plt.tight_layout()
```

Two pictures, same model up to the mask. The right-hand band has a constant width — and that width is what the cache for that layer ever has to hold.

## Why Most Layers Don't Need Forever

The motivating intuition is empirical and a little embarrassing in its simplicity. **Most attention heads do local work.** They want the previous comma, the matching brace, the noun this adjective is modifying. Years of attention-visualization papers (starting with Vig's *BertViz* in 2019) showed that a large fraction of heads, in a large fraction of layers, learn patterns whose attention mass concentrates in the last few dozen tokens.

If a head only ever cares about the last 4,096 tokens, *making* it look beyond that wastes both the cache and the dot products. SWA is the architectural acknowledgement of that fact, applied uniformly to every head in a chosen layer.

The price is real: a single SWA layer with window $W$ truly cannot see token $t - W - 1$. But — and this is the trick worth the chapter — *the stack of layers* can.

## The Receptive Field Trick

Stack two SWA layers, each with window $W$. The query at position $t$ in layer 2 reads keys from positions $t-W+1$ through $t$ — but those keys were themselves computed from residual-stream vectors that, in layer 1, had read from positions $t-2W+2$ through $t$. Information leaks **backwards** in time through the residual stream, one window per layer.

$$
\text{effective receptive field} \approx L \cdot W
$$

For Mistral 7B with $L=32$ and $W=4096$, that's a **theoretical** receptive field of 131,072 tokens — enough that the model can, in principle, condition on a 128K-token document even though no individual layer ever sees more than 4K keys at once. (In practice the influence decays sharply with distance; the receptive field is *available*, not *uniformly weighted*.)

The Mistral paper visualized it as light propagating through a CNN: a stack of small kernels has an effective receptive field much larger than any single kernel. Transformers had been doing the same thing in disguise — every transformer is a wide-but-shallow stack — and SWA just made the analogy structural.

## Interleaving: The Modern Recipe

Pure SWA, the way Mistral shipped it in 2023, is not where the frontier landed. The 2025 generation does something stronger: **mix** SWA and full-attention layers. Full-attention layers preserve the direct long-range pathway; SWA layers compress the bulk of the cache.

| Model | Sliding : Full ratio | Window | L-axis compression |
|---|---|---|---|
| Mistral 7B (2023) | all sliding | 4096 | T-bound = W everywhere |
| GPT-OSS-120B (2025) | **1 : 1** | 4096 | **2×** |
| Gemma 3 (2025) | **5 : 1** | 4096 | **6×** |

**The napkin math.** A full-attention layer's cache at context $T$ is $T \cdot d_{kv}$ bytes per token. A sliding layer's cache is $\min(T,\,W) \cdot d_{kv}$. With $f$ fraction of layers being sliding, total cache size per token across all layers averages

$$
\bar{B}(T) \;=\; f \cdot \min(T, W) \;+\; (1-f) \cdot T
$$

For Gemma 3 ($f = 5/6$, $W = 4096$) at $T = 128\text{K}$:

$$
\bar{B} \;=\; \tfrac{5}{6} \cdot 4096 \;+\; \tfrac{1}{6} \cdot 128000 \;\approx\; 24{,}746 \quad\text{vs.}\quad 128{,}000 \text{ full}
$$

That's the 6× number, derived from one weighted average.

```pyplot {id="cache-vs-context" caption="Per-token KV-cache size (summed across all layers) vs. context length, for four cache regimes. All-sliding flatlines at W; interleaves stay linear but with much smaller slope. At T=128K, Gemma 3's 5:1 SWA is ~6× smaller than dense."}
T = np.linspace(0, 128_000, 400)
W = 4096

def layered(T, f, W):
    return f * np.minimum(T, W) + (1 - f) * T

curves = [
    ("dense (full attn)",         np.ones_like(T) * 0 + T,    '#FF007F'),
    ("GPT-OSS  (1:1, f=0.50)",    layered(T, 0.50, W),         '#FF8C00'),
    ("Gemma 3  (5:1, f=0.83)",    layered(T, 5/6, W),          '#FFD700'),
    ("Mistral  (all SWA, f=1.0)", layered(T, 1.0, W),          '#00A8A8'),
]

fig, ax = plt.subplots(figsize=(8.6, 4.6))
for label, y, c in curves:
    ax.plot(T / 1000, y / 1000, color=c, linewidth=2.6, label=label)

ax.axvline(128, color='#1A1A1A', linestyle=':', linewidth=1)
ax.text(128, 6, ' T=128K', va='bottom', fontsize=9, color='#1A1A1A')

ax.set_xlabel("context length T (×1000 tokens)")
ax.set_ylabel("KV cache size per-token (×1000, arbitrary units)")
ax.set_title("Cache growth: dense vs. sliding-window regimes")
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.set_xlim(0, 128)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#1A1A1A'); ax.spines['bottom'].set_color('#1A1A1A')
```

The pink line is the budget that ate Llama1-65B's GPUs in [ch.17](../17-kv-axes/). The yellow and teal lines are the ones that ship today.

## The Attention-Sink Footgun

Pure sliding-window has a wart that took the field a year to fully appreciate. In late 2023, Guangxuan Xiao and colleagues at MIT published *"Efficient Streaming Language Models with Attention Sinks"*. They observed that transformers, trained on causal data, learn to **dump attention mass onto the first few tokens** — usually the BOS — even when those tokens carry no semantic relevance. This is the *attention sink* phenomenon: a numerical pressure-release valve baked into the model.

A naive sliding window **destroys the sinks**. Once $t > W$, the BOS token rolls out of the window and the softmax has nowhere to dump its excess mass. Perplexity blows up.

The fix is to keep the first few tokens *pinned* in the cache, always visible, in addition to the rolling window. The cache becomes the union of two slices:

```python
SINK = 4
k_h = [ki[hs:hs+head_dim] for ki in (keys[li][:SINK] + keys[li][-W:])]
```

Production engines (vLLM's `StreamingLLM`, SGLang's sliding-window path) bake this in. The cost is `SINK` extra entries per layer — single-digit tokens — and the model's perplexity over a 1M-token stream stays sane.

## Eviction, Or: The Memory-Layout Wart

Sliding-window has a second cost that doesn't show up in equations: when the cache *shrinks* (because old entries roll out), you need a memory layout that supports it. Naive contiguous tensors don't. Append-eviction churns memory bandwidth.

This is where **paged attention** earns its keep. PagedAttention (vLLM, Kwon et al. 2023) stores the cache in fixed-size *blocks* — typically 16 tokens — chained through a block table. To evict the oldest $W'$ tokens, you free the head blocks of the chain and remap. **No memcopy.** That's why every production SWA implementation in 2025 is also a paged-attention implementation.

## Per-Head Windows, And Other Wrinkles

A handful of 2025 papers (most notably the GPT-OSS technical report) experiment with **heterogeneous windows**:

- Some heads in a layer get $W = 256$ (very local).
- Other heads in the same layer get $W = 4096$.
- A third group sees the full context.

The change to microGPT is one extra index lookup — `W = window_per_head[h]` inside the head loop — but the operational story is that *one layer* now mixes a sliding regime with a dense regime, blurring the L-axis distinction. The accounting in the table above stops being clean. The compression number still falls out of a weighted sum; it just has more terms.

## What To Remember

1. **Sliding-window attention is one slice operator** — `keys[li][-W:]` — applied to selected layers. The cache for those layers caps at $W$ entries.
2. **Stacks of SWA layers have receptive field $L \cdot W$.** Information propagates backward through the residual stream, one window per layer.
3. **Interleaving is the modern default.** Pure SWA (Mistral 7B) gave way to mixed SWA + full layers (GPT-OSS 1:1 → 2×, Gemma 3 5:1 → 6×). Full layers preserve direct long-range; SWA layers compress.
4. **Watch out for attention sinks.** Pin the first few BOS-adjacent tokens. Without them, naive SWA inflates perplexity.
5. **Eviction needs paged memory.** Sliding-window in production rides on paged-attention block tables — no memcopy on eviction.
6. **SWA is the L-axis [from ch.17](../17-kv-axes/).** [MLA](../19-mla/) attacks D, GQA attacks H, SWA attacks L. They compose.

---

**Continue to** → [State-Space Hybrids](../21-ssm-hybrids/) — if sliding-window caps a layer's cache at $W$, what if we capped it at *one*: a fixed-size recurrent state that never grows at all?

