---
title: "KV Cache Anatomy: four axes compressed, one exploding"
short_title: "KV Cache Anatomy"
description: "The KV cache has five dimensions; the field compressed four of them with GQA, MLA, quantization, and hybrid attention — but the sequence-length axis grows without bound."
blurb:
  - "At 32k tokens, a single Llama-65B user session needs 84 GB — more than the model weights. Ten concurrent users need 840 GB."
  - "Grouped-query attention, multi-latent attention, INT8/INT4 quantization: each shaves one axis."
  - "The T dimension — sequence length — is the only one that cannot be fixed at training time."
  - "Four solutions, one stubborn frontier: why wasn't pruning the obvious answer?"
topics: [kv-cache, attention, architecture]
tags: [gqa, mla, hybrid-models, long-context, kv-cache]
theme: teal
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 20
techKind: mainline
techNode: kv-crisis
header: 02-kv-crisis.webp
---

## March 2023, Somewhere Between SFO and Heathrow

A principal engineer at a large cloud provider is doing napkin math at 36,000 feet.

Her company has just committed to shipping a 70B-parameter chat product with 32k-token context support. She has the serving numbers in front of her — the internal estimates her team ran last week before she boarded. And the number she keeps coming back to is not the cost of running attention. It is the cost of *remembering*.

For a single user conversation at 32k tokens, the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} for a Llama-65B-class model consumes **84 GB of GPU memory**. That is more than the weights themselves. And it scales linearly with the number of concurrent users: 10 users means 840 GB, before accounting for the model weights. Her four-card H100 server has 320 GB total. She can serve *three* concurrent long-context users per machine.

At the planned price point, the unit economics do not work.

This is not a research problem. It is a product-shipping problem, and it is hitting ML engineers in March 2023 at companies ranging from Google to startups with two-person inference teams. The KV cache — once a routine implementation detail in the [ch.13 KV Cache](/comicbook/05-microgpt/13-kv-cache/) of every transformer textbook — has become the dominant cost at scale.

The field's response, over the next 18 months, is to compress every axis it can find.

## The Anatomy of the Cache

The {{< wiki "attention" >}}multi-head attention{{< /wiki >}} mechanism in a transformer requires, at every decode step, the ability to attend over every previous position. This means the model must *store* something for every past token — specifically, two vectors per token, per layer, per head. One vector is the key, used to compute attention scores. The other is the value, used to compute the attended output.

Store those vectors across all layers and you have the KV cache. Its shape is:

$$
\text{cache} \in \mathbb{R}^{2 \times L \times H \times T \times D_h}
$$

Where:
- **2** — one block for keys, one for values
- **L** — number of transformer layers
- **H** — number of KV attention heads per layer
- **T** — number of tokens in the context (the sequence length)
- **D_h** — head dimension, equal to `n_embd / n_heads`{{% marginnote %}}In microGPT: `D_h = n_embd // n_head`. For Llama-65B: `D_h = 8192 // 64 = 128`.{{% /marginnote %}}

The total memory in bytes is:

$$
\text{Memory} = 2 \times L \times H \times T \times D_h \times \texttt{bytes\_per\_element}
$$

In {{< wiki "number-formats" >}}bfloat16{{< /wiki >}} — the standard for production inference — each element is 2 bytes. Plug in Llama-65B's numbers (L=80, H=64, D_h=128) and you reproduce the 335 GB figure from the cold open.

To make this concrete, here is the same formula applied to three models you might actually deploy:

| Model | Layers | KV Heads | Head Dim | Cache @ 32k tokens | Cache @ 128k tokens |
|---|---|---|---|---|---|
| Llama-65B | 80 | 64 | 128 | **84 GB** | 335 GB |
| Llama-3.1-8B | 32 | 8 | 128 | **2.7 GB** | 10.7 GB |
| {{< wiki "qwen3" >}}Qwen3-32B{{< /wiki >}} | 64 | 16 | 128 | **8.4 GB** | 33.7 GB |

Llama-65B at 128k context needs more memory for its cache than the entire GPU memory budget of a four-card H100 server. Llama-3.1-8B is more manageable — but at 128k context with 100 concurrent users, you need 1.07 TB just for the caches. The math keeps being unpleasant.

{{< crosshead >}}The Five-Axis Problem{{< /crosshead >}}

Every factor in that formula is a target. And between 2021 and 2025, the field attacked four of them with real success — mostly by changing the model architecture before training begins.

What follows is the story of that compression race, axis by axis. Think of it as an audit: we will tally what has already been solved, so we can see clearly what has not.

## The H-Axis: Multi-Query and GQA

The original **multi-head attention** design from [Inside K and V](/comicbook/03-quantization/16-kv-distribution/) uses a separate key and value head for every query head. If you have 64 query heads, you have 64 KV heads. Each head independently learns its own projection and stores its own cache.

**Multi-Query Attention (MQA)**, proposed by Noam Shazeer in 2019, collapses this entirely: all query heads share *one* pair of KV heads. The cache shrinks by a factor of H — from 64 KV heads down to 1. The accuracy penalty is real but manageable for smaller models.

**Grouped-Query Attention (GQA)**, published by Ainslie et al. in 2023, finds the middle ground: group Q heads into clusters, with each cluster sharing one KV head. Llama-3 uses 8 KV heads for 64 query heads — a factor of **8× compression** on the H-axis. Qwen3-32B uses 16 KV heads for 64 query heads — **4× compression**. These are not approximations. The weights are trained to use the shared KV structure from scratch, so there is no accuracy penalty at all if you design the model this way from day one.

The H-axis is solved. Move on.

## The D-Axis: MLA and Low-Rank Decomposition

**{{< wiki "deepseek" >}}DeepSeek V2{{< /wiki >}}** (2024) introduced **Multi-head Latent Attention (MLA)**: instead of caching full-rank key and value matrices, the model caches a *low-rank latent vector* and reconstructs the keys and values on the fly. The cached object has dimension `4H/9` instead of `2HD_h` — a compression factor of roughly **4.5× on the effective cache size** relative to standard MHA.

The trick is architectural: the attention projection weights are designed so that a single compressed latent encodes all the information needed to recover K and V at query time. You pay a small compute cost at each decode step to decompress, but you save enormously on memory bandwidth — which is the binding constraint on modern accelerators.

The D-axis can be substantially compressed. It requires a custom architecture; you cannot apply MLA to a standard Llama model post-hoc. But if you design for it, it works.

## The L-Axis: Hybrid Models

Not every layer of a transformer needs full attention. **Hybrid architectures** interleave full attention layers with cheaper alternatives — Mamba SSM blocks, sliding-window attention, or no-attention (MLP-only) layers. Each non-attention layer has no KV cache cost at all.

**Jamba** (AI21 Labs, 2024) interleaves attention and Mamba blocks in a roughly 1:7 ratio, reducing the attention-layer count from, say, 80 to ~11. **Gemma3** uses a 1:5 ratio of full attention to sliding-window attention. **GPT-OSS-120B** (2025) achieves 6× compression on the L-axis through aggressive hybridization.

The cost is architectural lock-in: you cannot hybridize a model after training any more than you can retrofit GQA. But for new model families being designed from scratch, the L-axis is a legitimate target.

## The T-Axis: ???

Now look at the formula again. We have made progress on H (GQA gives 4–8×), on D (MLA gives ~4.5×), and on L (hybrids give 6–8×). These gains can be stacked. A model designed with GQA + MLA + hybrid layers could, in principle, reduce its KV cache footprint by **100× or more** relative to a naive Llama-65B-class model.

But none of those tricks touch the T-axis.

T is the number of tokens in the context window. It grows linearly with the user's prompt length, with the document being analyzed, with the conversation history. No architectural choice at training time can predict what T will be for any given user. If the user sends a 128k-token legal brief, T is 128,000 — regardless of whether your model uses GQA or MLA or anything else.

{{% pullquote type="counter-intuitive" %}}
Every axis of the KV cache can be compressed at training time — except the one that grows with what the user sends you.
{{% /pullquote %}}

The T-axis is uniquely resistant to pre-training solutions because it encodes *content*. H, D, and L are structural properties of the model. T is semantic: it is a record of every token the model has processed in this conversation, in this order, with this meaning. Compressing T means deciding, at runtime, which tokens to forget.

That is a fundamentally different kind of problem.

{{< crosshead >}}Why T Is Hard{{< /crosshead >}}

To see why runtime compression of the T-axis is difficult, consider what you would have to know to do it well.

Imagine the model is in the middle of processing a 100k-token technical report. It has just generated the first word of its executive summary. The KV cache holds 100,000 key-value pairs across 80 layers and 64 heads. You have to decide: which of these 100,000 entries can you safely delete?

The obvious criterion — *which tokens did the model attend to most?* — immediately runs into a problem. Attention patterns shift constantly. A detail that received zero attention while processing the introduction might become essential when answering a question about the conclusion. Attention weights at one decode step are a bad predictor of what future decode steps will need.

You also cannot run a fresh evaluation at every decode step. At 100k tokens and 80 layers, a single forward pass costs tens of milliseconds. Token generation has to happen at hundreds of tokens per second. You have roughly *one millisecond* per generated token to decide what to keep.

The naive approaches all fail:
- **Keep recents:** discard old tokens, keep new ones. Works for pure conversation; catastrophic for document retrieval.
- **Keep by attention:** discard low-attention tokens. Brittle to shifting attention patterns.
- **Keep everything:** correct but defeats the point. The cache fills up and you OOM.

```pyplot {id="axes-compression" caption="KV CACHE COMPRESSION ACHIEVED ON EACH AXIS BY 2025. THE T-AXIS BAR IS EMPTY — THE GAP THIS ISSUE FILLS."}
import numpy as np
import matplotlib.pyplot as plt

axes = ['L-axis\n(layers)', 'H-axis\n(heads)', 'D-axis\n(head dim)', 'T-axis\n(tokens)']
compressions = [7.0, 6.0, 4.5, 1.0]  # approximate best-case per axis
methods = ['Hybrid (Jamba/Gemma3)', 'GQA (Llama-3)', 'MLA (DeepSeek V2)', 'Pruning (TBD)']
colors = ['#FFD700', '#00A8A8', '#FF007F', '#FF8C00']
alpha_vals = [0.9, 0.9, 0.9, 0.25]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(axes, compressions, color=colors, alpha=0.9,
              edgecolor='#1A1A1A', linewidth=1.5, width=0.55)
bars[3].set_alpha(0.25)

for bar, c, method in zip(bars, compressions, methods):
    label = f"{c:.1f}×\n{method}" if c > 1 else f"1×\n{method}"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            label, ha='center', va='bottom', fontsize=8.5, color='#1A1A1A')

ax.set_ylabel("compression factor (×)")
ax.set_title("KV cache compression by axis — three axes solved, one remains", fontweight='bold')
ax.set_ylim(0, 10)
ax.spines[['top', 'right']].set_visible(False)

ax.text(3, 5, "The Gap", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#FF8C00', alpha=0.6,
        rotation=0, style='italic')

print("Compression factors by axis (approximate best-case, 2025):")
for axis, c, method in zip(axes, compressions, methods):
    axis_clean = axis.replace('\n', ' ')
    print(f"  {axis_clean}: {c:.1f}× — {method}")
```

The bar chart tells the story in one picture. Three axes have clean, widely-deployed solutions. One axis — T — has a nearly empty bar. And it is the axis that drives memory cost to infinity as context windows grow.

{{% callout type="note" title="What counts as a T-axis solution?" %}}
For a KV pruning method to count as a real T-axis solution, it must satisfy four criteria:

1. **Fast**: the scoring overhead must be negligible compared to generation time
2. **Phase-agnostic**: must work during both prefill *and* decode
3. **Compatible**: must not break FlashAttention or PagedAttention
4. **Faithful**: accuracy on long-context benchmarks must stay within ~2% of the full-cache baseline

KVzip fails criteria 2 (decode incompatible) and arguably criterion 1 (2× prefill overhead). KVzap is designed to satisfy all four. We will see whether it does.
{{% /callout %}}

## The Fundamental Asymmetry

There is a deeper reason why the T-axis resisted training-time solutions when the others did not.

H, D, and L compression all work by changing the *structure* of what gets computed and stored. GQA trains heads to share KV projections; MLA trains a low-rank bottleneck; hybrid models train some layers to use no attention at all. These are architectural choices made before the model sees any user input. They are baked into the weights.

T compression requires reasoning about *content* — about the semantic meaning of individual tokens in the context of the current conversation. No amount of architectural cleverness at training time can anticipate which tokens in an arbitrary future document will be semantically redundant.

This is the fundamental asymmetry. And it implies that any effective T-axis compression method must do something qualitatively different from what worked on the other axes: it must make a judgment call, at runtime, about what the model will and will not need.

The question is whether that judgment call can be made cheaply enough to matter.

{{< crosshead >}}The Observation That Changed Everything{{< /crosshead >}}

In the summer of 2023 — a few months after our engineer's sleepless transatlantic flight — a team at Carnegie Mellon published an observation that reframed the entire problem.

They were studying {{< wiki "softmax" >}}softmax{{< /wiki >}} attention patterns in large transformers at scale. And they noticed something that, once you see it, you cannot unsee:

**Most tokens almost never get attended to.**

In a typical generation step, attention is not spread diffusely across the entire context. It is concentrated. A small fraction of tokens — maybe 5% to 20% — absorb the vast majority of the attention weight. The rest contribute almost nothing.

And crucially: these high-attention tokens are not random. The same positions accumulate attention again and again across layers and heads. The CMU team called them **Heavy Hitters**.

If 80% of tokens almost never receive meaningful attention across a long generation sequence — if the model is effectively ignoring most of its own past — then a question becomes unavoidable: what if you just *deleted the non-hitters*?

**Continue to** → **[The Heavy Hitter Oracle](../03-heavy-hitter/)** — the 2023 observation that shattered assumptions about how much of the KV cache you actually need, and the first algorithm to exploit it in production.
