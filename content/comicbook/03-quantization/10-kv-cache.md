---
title: "KV Cache: the new memory bill after weights got small"
short_title: "KV Cache"
description: "Llama-2-70B quantized to 4-bit weighs 35 GB; at 32K context its KV cache weighs 80 GB — and K and V need different quantization axes."
blurb:
  - "Per-token KV cache footprint for Llama-2-70B: 2.5 MB. At 32K context that's 80 GB — bigger than the quantized weights."
  - "KV cache grows linearly with context length. Quantized weights are fixed. The cache becomes the bottleneck."
  - "K outliers are per-channel (same feature dimensions persist across tokens). V outliers are per-token (same token positions persist across heads)."
  - "The asymmetry between K and V means symmetric weight-style methods fail: each tensor needs its own axis."
topics: [quantization, attention]
tags: [kv-cache, kivi, attention, long-context]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 100
techKind: mainline
techNode: kv-cache
header: 10-kv-cache.webp
---

## What Happens After You Win

Suppose you have done everything in this issue. Your 70B-parameter Llama is quantized to 4 bits with GPTQ, takes 35 GB of GPU memory, and runs cleanly on a single A100. You celebrate.

Then a user asks for a 32,000-token context window.

Suddenly your memory bill explodes again. Not from weights — those are still 35 GB. The new culprit is the **{{< wiki "kv-cache" >}}KV cache{{< /wiki >}}**: the running buffer of key and value tensors that the model carries from token to token during {{< wiki "attention" >}}self-attention{{< /wiki >}}.

For Llama 2 70B at 32K context, the KV cache is roughly **40 GB** — *bigger than the quantized weights*. For 128K context, it's 160 GB. The cache scales **linearly with context length**, while weights are fixed.

We have, in some sense, traded one tyrant for another.

## A Quick Refresher On What KV Cache Is

In standard transformer self-attention, each token computes three projections: query $q$, key $k$, value $v$. The attention output for the current token is:

$$
\text{attn} = \sum_{t'} \text{softmax}_{t'}(q^\top k_{t'}) \cdot v_{t'}
$$

To generate the next token, you need *all the previous $k$ and $v$ tensors*. You could recompute them — but that would be $O(n^2)$ in context length. Instead, you **cache** $k_{t}$ and $v_t$ as you go.

The cache shape per layer is: `[context_length, num_heads, head_dim] × 2 (K and V)`. For Llama-2-70B (80 layers, 64 heads, 128 head dim, FP16), the per-token footprint is:

$$
80 \text{ layers} \times 2 \text{ (K and V)} \times 64 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes} = 2{,}621{,}440 \text{ bytes} \approx 2.5 \text{ MB per token}
$$

Multiply by 32,768 tokens: **80 GB.** (You can shave this down with grouped-query attention, multi-query attention, or sliding windows; we'll set those aside for clarity.)

So **the KV cache eats everything.** It's the bottleneck for long-context inference. Naturally, people thought: can we quantize it?

## The Naive Move And Why It Breaks

Apply LLM.int8-style per-tensor INT8 to K and V. Run benchmarks. The model loses about 4–10% on most tasks. Not catastrophic, but unsatisfying.

The reason is exactly the geometry we set up in [The Geometry Of Weights](../05-geometry-of-weights/). Activations have outliers, K and V are activations, so K and V have outliers. The bulk values get crushed by max-based scaling, the bulk recovery is poor.

Standard fix: per-channel scales? Per-token scales? Both?

It turns out the right answer depends on which tensor — **and K and V need different answers**.

## The KIVI Insight

In June 2023, a paper from Yujun Lin's group at MIT (with collaborators) appeared on arXiv with the unassuming title **"KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"**. The crux was a careful empirical observation about *how the outliers in K differ from the outliers in V*.

**Keys have outlier *channels*** — specific feature dimensions that are consistently large across all tokens.

**Values have outlier *tokens*** — specific positions in the sequence whose entire $v$ vector is large.

This is a fundamental structural difference. It means:

- **Per-channel quantization** (one scale per feature dim) is right for K, because the outlier dimensions get their own scale and the bulk dimensions are unaffected.
- **Per-token quantization** (one scale per sequence position) is right for V, because outlier tokens get their own scale and the bulk tokens are unaffected.

```pyplot {id="kv-asymmetry" caption="Synthetic KV tensors showing the asymmetry: K has outlier channels (vertical stripes), V has outlier tokens (horizontal stripes)."}
np.random.seed(2)
seq_len = 64
d_head = 32

# K: outliers per CHANNEL (some dims are large across all tokens)
K = np.random.randn(seq_len, d_head) * 0.3
outlier_channels = [5, 18, 27]
K[:, outlier_channels] *= 12

# V: outliers per TOKEN (some positions are large across all dims)
V = np.random.randn(seq_len, d_head) * 0.3
outlier_tokens = [10, 33, 50]
V[outlier_tokens, :] *= 12

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, data, title in [(axes[0], K, "K — outlier CHANNELS (vertical stripes)"),
                         (axes[1], V, "V — outlier TOKENS (horizontal stripes)")]:
    im = ax.imshow(np.abs(data), aspect='auto', cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("feature dim")
    ax.set_ylabel("token position")
plt.tight_layout()
```

You can see why naive per-tensor scaling fails. The max is dominated by the outlier stripes; the bulk gets crushed. But per-*channel* scales on K give each column its own range. Per-*token* scales on V give each row its own range. The bulk in both cases is rescued.

Quantitatively, **KIVI is able to go all the way to 2-bit KV cache** (rather than 4-bit or 8-bit) with minimal accuracy loss — by exploiting this asymmetry rather than fighting it.

## Why The Asymmetry Exists

The empirical observation begs the theoretical question: *why* do K and V have different outlier geometries?

The honest answer is that nobody has a rigorous derivation. But there's a satisfying intuition:

- **K is a projection that selects "what to look up."** Specific feature dimensions encode "what kind of information this position is associated with" — and a small set of these features carry a lot of weight, structurally. Hence outlier *channels*.
- **V is the actual content being passed.** Some positions in a sequence carry hugely-important content (the first BOS token; positions that match the query pattern). Hence outlier *tokens*.

This roughly maps to the difference between *addressing* (K) and *content* (V) in associative memory. The structure of *how a query reads the cache* makes each tensor have a different natural axis of variation.

## What This Asymmetry Buys

KIVI reported these headline numbers on Llama-2-7B at 2-bit KV cache:

| Method | WikiText-2 perplexity |
|---|---|
| FP16 baseline | 5.47 |
| Per-tensor 2-bit | 13.84 (broken) |
| KIVI 2-bit | 5.55 |

A nearly-lossless 2-bit cache. Eight times smaller than FP16. For a 70B model with 32K context, this brings the cache from 40 GB to 5 GB.

The intellectual point isn't "use 2 bits"; it's "**match the quantization axis to the data's actual structure**". This is the same lesson we've heard repeatedly:

- Lloyd-Max says: match the levels to the distribution.
- LLM.int8 says: match the precision to whether you're an outlier.
- AWQ says: match the protection to the activation magnitude.
- KIVI says: match the scaling axis to the outlier orientation.

All of them are facets of one thing: pay attention to the actual geometry of the data.

## Sliding Windows And The Other Half Of The Story

KV-cache quantization is one of two threads attacking the long-context memory problem. The other is **sliding-window attention** (e.g., Mistral 7B, Longformer): just don't keep every token's cache, drop older ones outside a window.

The two are compatible — you can quantize *and* slide. Modern long-context engines (vLLM, SGLang, TGI) combine both, and the result is that 100K-token contexts on a single GPU are now routine.

There's also a third thread: **alternative attention shapes** that fundamentally reduce KV footprint per token. Multi-query attention (PaLM) shares K and V across heads, cutting cache by 8-64×. Grouped-query attention (Llama 2) is a middle ground. Multi-head latent attention (DeepSeek-V2) caches a *compressed* latent representation. Each of these is a different architectural answer to the same problem.

Quantization stays useful in all of these — it's a *constant-factor* saving on top of whatever architecture you have.

## The Big Picture

When weights were the problem, you got [LLM.int8](../06-outliers/), [GPTQ](../08-brain-surgery/), and the [family of methods](../09-method-family-tree/) we just toured. Once weights got small, the KV cache became the new dominant cost — and KIVI-style methods carved into it.

But there's an even larger move just over the horizon: **what if the hardware itself supported KV-cache-native low-precision arithmetic?** What if you could do attention math directly in {{< wiki "number-formats" >}}FP8{{< /wiki >}} or FP4 without the dequantize-then-compute dance? That's the territory of the next-generation silicon, and the closing chapter of this issue.

**Continue to** → [Inside K and V](../16-kv-distribution/) for the deep distribution-detective story behind the K-channel / V-token asymmetry — the empirical discovery that drives every entry in [The KV Method Family Tree](../15-kv-method-family/). Then onwards to the underlying machinery in [Calibration & Blocks](../11-calibration-and-blocks/) and the hardware finale.
