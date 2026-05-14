---
title: "Inside K and V — A Distribution Detective Story"
description: "Three years of opening up attention's running buffer and finding strange shapes inside it. The outlier-channel pattern in K, the outlier-token pattern in V, the BOS-as-attention-sink discovery, and the timeline that connects them."
topics: [quantization, attention, statistics]
tags: [kv-cache, kivi, attention-sinks, streamingllm, massive-activations, distribution]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 160
techKind: mainline
techNode: kv-distribution
header: default.webp
---

## Three Microscopes

You can spend a year working with transformers without ever looking *inside* the KV cache. The cache is just a buffer; the model writes to it, the model reads from it, the wrapper logic feeds it back into attention. From outside the box, the bits that go in and out are floating-point numbers and that is that.

Then in **early 2023**, three different research groups, working independently, point microscopes at the same buffer and start reporting strange things. **Yujun Liu's group at MIT** notices that K vectors have a few channels with persistently giant values, while V vectors have a few sequence positions with persistently giant values. **Guangxuan Xiao's team at MIT** (overlapping people) notices that the *first token* of every sequence — the BOS token — receives a wildly disproportionate fraction of attention from later positions. **Mingjie Sun's team at CMU** notices that *some hidden states*, in the residual stream itself, develop magnitudes a thousand times larger than their neighbors.

Three observations. Three teams. They turn out to be the same observation.

This article is the story of how the field assembled the picture, and why the picture matters for every method in [The KV Method Family Tree](../15-kv-method-family/). It is a detective story with three witnesses, all unreliable in different ways.

## The Boring Picture: What You'd Naïvely Expect

Open up a trained LLM, run a forward pass, and pull out the K and V tensors at, say, the 16th transformer layer. Each is shaped `[batch, num_heads, seq_len, head_dim]`. Squash a single head and you get a 2D matrix: rows are sequence positions, columns are feature dimensions.

If transformers were well-behaved Gaussian objects — the way our loss-landscape intuitions kept telling us they were until 2022 — this matrix would look like white noise. Roughly zero mean, roughly the same variance everywhere, no structure.

```pyplot {id="naive-kv-expectation" caption="What you'd expect K and V to look like, if transformers were well-behaved. Spoiler: they don't."}
np.random.seed(0)
seq_len, d_head = 64, 32

K_imagined = np.random.randn(seq_len, d_head) * 0.5
V_imagined = np.random.randn(seq_len, d_head) * 0.5

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, data, title in [(axes[0], K_imagined, "K (imagined)"),
                         (axes[1], V_imagined, "V (imagined)")]:
    im = ax.imshow(np.abs(data), aspect='auto', cmap='magma', vmin=0, vmax=2.0)
    ax.set_title(title)
    ax.set_xlabel("feature dim")
    ax.set_ylabel("token position")
plt.tight_layout()
```

This was the working assumption of every quantization scheme prior to 2023. The naive INT8 KV cache implementation in early HuggingFace `transformers` used a per-tensor scale, and it *almost worked*: a few percent perplexity loss, easy to explain away as "well, you're throwing away half the bits."

It almost worked because most of the K and V cache *is* well-behaved Gaussian noise. The trouble is that the bits that aren't noise are doing all the work.

## Witness #1 — KIVI's K-Channel / V-Token Asymmetry

In **June 2023**, Liu, Wang, and collaborators at MIT post **KIVI** to arXiv. The paper does the simplest possible empirical study: histogram every K and V tensor across many layers and many inputs of a Llama-2 forward pass.

The histograms are not Gaussian. They are *bizarre*. K shows a fat tail concentrated in **specific feature dimensions** — the same dimension indices, every input, every layer, with magnitudes 10–50× the bulk. V shows a fat tail concentrated in **specific sequence positions** — different position indices for different inputs, but always a small number per sequence, again 10–50× the bulk.

Same tensor shape. Two completely different geometric distributions of the outliers. Plotted side by side as in [the KIVI panel from the previous chapter](../10-kv-cache/), the asymmetry is unmissable.

```pyplot {id="kivi-real-asymmetry" caption="The KIVI observation, plotted on simulated data tuned to the magnitudes the paper reports. K's outliers form vertical stripes; V's form horizontal stripes."}
np.random.seed(7)
seq_len, d_head = 96, 64

# K: outlier channels
K = np.random.randn(seq_len, d_head) * 0.25
out_chans = [4, 17, 31, 48]
for c in out_chans:
    K[:, c] = np.random.randn(seq_len) * 4 + np.random.choice([-1, 1]) * 6

# V: outlier tokens
V = np.random.randn(seq_len, d_head) * 0.25
out_tokens = [0, 12, 47, 80]  # note: position 0 is special — see witness #2
for t in out_tokens:
    V[t, :] = np.random.randn(d_head) * 4 + np.random.choice([-1, 1]) * 5

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

for ax, data, title in [(axes[0,0], K, "K — abs value heatmap"),
                         (axes[0,1], V, "V — abs value heatmap")]:
    ax.imshow(np.abs(data), aspect='auto', cmap='magma', vmin=0, vmax=8)
    ax.set_title(title)
    ax.set_xlabel("feature dim")
    ax.set_ylabel("token position")

axes[1,0].bar(range(d_head), np.abs(K).max(axis=0), color='#1A1A1A', width=0.9)
for c in out_chans:
    axes[1,0].bar(c, np.abs(K).max(axis=0)[c], color='#FF007F', width=1.5)
axes[1,0].set_title("K: max |val| per CHANNEL — a few are tens-of-times larger")
axes[1,0].set_xlabel("channel")
axes[1,0].set_ylabel("max |K|")
axes[1,0].spines[['top', 'right']].set_visible(False)

axes[1,1].bar(range(seq_len), np.abs(V).max(axis=1), color='#1A1A1A', width=0.9)
for t in out_tokens:
    axes[1,1].bar(t, np.abs(V).max(axis=1)[t], color='#00A8A8', width=1.5)
axes[1,1].set_title("V: max |val| per TOKEN — a few are tens-of-times larger")
axes[1,1].set_xlabel("token position")
axes[1,1].set_ylabel("max |V|")
axes[1,1].spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The bottom row makes the asymmetry quantitative. K's per-channel max has four spikes and a flat bulk. V's per-token max has four spikes and a flat bulk. The *axis* of the spikes is opposite for the two tensors.

The KIVI paper's table says it precisely. Take the per-element maximum across either axis:

| Tensor | Axis with persistent outliers | Axis with no special structure |
|---|---|---|
| K | Channel | Token |
| V | Token | Channel |

This is the empirical observation that drives everything in [the KV method family tree](../15-kv-method-family/). It is the load-bearing fact.

## Why Does The Asymmetry Exist?

KIVI doesn't *explain* the asymmetry, just notes it. The intuitive interpretation — repeated in dozens of follow-up papers — is associative-memory-flavoured:

- **K is the "address."** When attention computes $\text{softmax}(QK^\top)$, the dot product $Q_t \cdot K_{t'}$ is a *similarity* between the current query and the cached position. For the attention pattern to be sharp, *some channels* of K need to carry strong, distinguishing signals across many positions. Those high-signal channels are the outliers.
- **V is the "content."** $V_{t'}$ is the actual representation that gets retrieved when $K_{t'}$ matches the query. For attention to be a useful retrieval mechanism, *some positions* need to carry information-rich content (e.g. the BOS token, which encodes nothing-particular and acts as a default; topic-defining tokens; punctuation that anchors syntax). Those high-content positions are the outlier rows.

This is intuitive but not a proof. There is no rigorous derivation of *why* trained LLMs converge to this geometry. There are several plausible hypotheses; nobody has nailed it. But the empirical pattern is as robust as anything in the field — every model from GPT-2 to Llama-3 to Mixtral shows it, with the *same* structural asymmetry, even though the absolute outlier magnitudes vary.

## Witness #2 — StreamingLLM And The Attention Sink

In **September 2023**, three months after KIVI, the same MIT lab (Xiao et al.) publishes **StreamingLLM**, with a different microscope. They aren't looking at K or V values directly; they're looking at the **attention pattern** $\text{softmax}(QK^\top)$ itself.

What they find is, again, weird. Across most layers and most inputs, the **first few tokens of the sequence** receive a hugely disproportionate fraction of the attention mass — even when the *content* of those tokens is irrelevant to the current query. In a 4K-token document about quantum mechanics, the BOS token gets 30% of the attention from token 3,500. The model is not "looking back to remember the topic." It is *parking* attention there.

The Xiao paper coins the name **attention sink** for this phenomenon. The first few tokens — typically the BOS token plus the first one or two real tokens — act as catch basins for whatever attention isn't being spent on more semantically-relevant positions.

The mechanistic argument: softmax is a probability distribution. It must sum to 1. If a query has nothing particularly relevant to attend to in the cache, the softmax has to put its mass *somewhere*. The model learns, during training, to direct that excess mass to a fixed "junk drawer" position — the BOS token, which is always present. The result is that **specific token positions develop K-vector patterns that act as an attention sink**: their K vectors have high cosine similarity with *any* generic query.

```pyplot {id="attention-sink" caption="Schematic attention pattern with a sink. The first-position column (BOS) collects mass from almost every later query, even when content-relevance is elsewhere."}
np.random.seed(2)
seq_len = 32

# Build a synthetic attention matrix where:
# - position 0 is a sink: every later token sends a lot of attention there
# - there's also a "content peak" near the diagonal (local attention)
# - some sparse "long-range" links
attn = np.zeros((seq_len, seq_len))
for t in range(seq_len):
    # local diagonal
    for tp in range(max(0, t-3), t+1):
        attn[t, tp] += np.exp(-(t - tp))
    # sink at position 0
    if t > 1:
        attn[t, 0] += 2.5
    # sparse long-range
    if t > 8 and (t % 7 == 0):
        attn[t, t // 2] += 1.0
# normalize per row
attn = attn / attn.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(attn, aspect='auto', cmap='magma')
ax.set_xlabel("Key position (j)")
ax.set_ylabel("Query position (t)")
ax.set_title("Attention map with a sink at position 0\n(every row leaks ~30% mass into column 0)")
plt.colorbar(im, ax=ax, label="attention weight", shrink=0.7)
ax.axvline(0.5, color='#FF007F', linewidth=2, alpha=0.7)
ax.text(1, seq_len/2, "BOS sink", color='#FF007F', fontweight='bold', fontsize=10)
```

**Why does StreamingLLM matter for KV quantization?** Because it *connects* witnesses 1 and 2. Recall from KIVI: V has outlier tokens. *Which* tokens? Well, primarily the **attention-sink tokens** — the BOS and first few positions. They are the ones that get retrieved from again and again, so the V vectors at those positions are doing huge amounts of work and develop large magnitudes during training.

KIVI saw the symptom (per-token V outliers); StreamingLLM identified the cause (attention sinks). The two papers are reading the same phenomenon from opposite ends of the attention computation.

The practical consequence: **at least the first 4 tokens of the KV cache should be kept at higher precision** (or in FP16). Almost every subsequent KV-quantization method does this. KIVI keeps the first ~4 V tokens unquantized; KVQuant keeps a top-K outlier set in FP16; StreamingLLM literally proposes "always keep the sink tokens, drop everything else when memory is tight." All three converge on the same operational rule.

## Witness #3 — Sun Et Al. And Massive Activations

In **April 2024**, Mingjie Sun and collaborators at CMU publish **"Massive Activations in Large Language Models"** — a microscope pointed at the **residual stream** itself, layer by layer.

They find that across many transformer layers, a small number of *individual activation values* — specific (token, channel) entries in the residual stream — are 10,000 to 100,000 times larger than the median activation. Not "one channel is large across all tokens" (that's the K-channel observation). Not "one token is large across all channels" (that's the V-token observation). *Specific (token, channel) entries* with absolutely enormous magnitudes, persistent across many layers.

These are the **massive activations**. Their character:

1. They concentrate at **specific tokens**: typically the BOS token, the first newline, and one or two other "anchor" positions.
2. They concentrate at **specific channels**: a small handful of dimensions per layer, stable across inputs.
3. They appear **after a particular transformer layer** (often around layer 2-4 in Llama-2) and **persist** through the residual stream until late in the network, where they fade.
4. **Removing them breaks the model** — Sun et al. show that zeroing the top massive activation in Llama-2-7B's residual stream causes catastrophic perplexity collapse.

The Sun paper *unifies* the previous two witnesses. The K-channel outliers are the projections of massive activations through $W_K$. The V-token outliers are the projections of massive activations through $W_V$. The attention sink at the BOS token is the model **using** the massive activation at BOS as a destination for unwanted attention mass.

It's all one phenomenon. The residual stream develops a small set of "anchor" cells with enormous magnitude. Every downstream operation that touches those cells inherits an outlier somewhere in its tensor.

```pyplot {id="massive-activation-cascade" caption="Schematic: a single massive activation in the residual stream produces an outlier in K (when projected through W_K), an outlier in V (when projected through W_V), and an attention sink (when softmaxed against)."}
np.random.seed(11)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')

def box(x, y, w, h, label, color='#FFD700', big=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, color=color, ec='#1A1A1A', lw=2.5))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=11 if not big else 13, fontweight='bold')

def arrow(x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=2))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, ha='center', fontsize=8)

box(0.5, 2, 2.5, 1, "Residual\nMASSIVE\nactivation", color='#FF007F', big=True)
ax.text(1.75, 3.2, "(BOS token, channel 1415)", ha='center', fontsize=8, style='italic')

arrow(3.0, 2.5, 4.5, 4.2, "× W_K")
arrow(3.0, 2.5, 4.5, 2.5, "× W_V")
arrow(3.0, 2.5, 4.5, 0.8, "softmax(Q·K^T)")

box(4.5, 3.6, 3.0, 1.0, "K with outlier\nCHANNEL")
box(4.5, 2.0, 3.0, 1.0, "V with outlier\nTOKEN")
box(4.5, 0.4, 3.0, 1.0, "Attention\nSINK at BOS")

arrow(7.5, 4.1, 9.0, 4.1)
arrow(7.5, 2.5, 9.0, 2.5)
arrow(7.5, 0.9, 9.0, 0.9)

box(9.0, 3.6, 4.5, 1.0, "KIVI: per-CHANNEL K scale", color='#FFD700')
box(9.0, 2.0, 4.5, 1.0, "KIVI: per-TOKEN V scale",   color='#FFD700')
box(9.0, 0.4, 4.5, 1.0, "StreamingLLM: keep BOS K/V", color='#FFD700')
```

The diagram is the field's current best answer to "what is going on inside the KV cache?" One root cause (massive activations in the residual stream) → three downstream symptoms (K-channel outliers, V-token outliers, attention sinks) → three converging mitigation strategies (per-channel K, per-token V, BOS preservation).

## A Five-Year Timeline Of What We Learned

The discoveries above happened in a tight ~14-month window, but the lineage runs back further. Here is the abbreviated timeline.

{{< timeline name="kv-distribution-discoveries" >}}

The post-2024 work has mostly been *integration* of these insights into production methods (see [The KV Method Family Tree](../15-kv-method-family/)) plus the rotation-based methods that *bypass* the asymmetry rather than respecting it (see [Rotations](../17-rotations/)).

## What Hasn't Been Explained

Three open questions, in order of decreasing importance:

**1. Why does the model want an attention sink at all?** The hypothesis is "the softmax has to sum to 1, so excess mass needs somewhere to go." But this is descriptive, not explanatory. *Why* does the model not learn to spread the excess attention across many positions instead of pinning it to one? Why specifically the BOS? Some recent work (Cancedda 2024) argues that softmax sinks are an inevitable consequence of normalization without a "no-op" output, and proposes alternative attention mechanisms (softmax-1, softplus) that don't require sinks. These work but are not yet widely adopted.

**2. Why are massive activations stable across layers?** Once a token, channel pair becomes a massive activation, it stays massive through 20+ subsequent layers. The mechanism by which the residual stream *protects* a particular cell against being washed out by 20 layers of MLPs and attention — that's a small mystery. The current best guess is that the LayerNorm γ parameter for that channel becomes large during training, effectively reinforcing the channel.

**3. Are massive activations *necessary*?** Sun et al. show that ablating them breaks the model. But this could be because the model has *learned to depend on them*, not because they are inherent. If you trained a transformer with explicit anti-sink regularization, would it still emerge with massive activations? Two papers in 2025 say no — the model can be trained sink-free without quality loss. If true, this opens a path to training quantization-friendly LLMs that don't have the outlier asymmetry to begin with. It's early.

## What This Means For Quantization

A summary, returning to what motivated this article:

1. **The asymmetric quantization axis (per-channel K, per-token V) is empirically correct because of how massive activations propagate.** Don't fight it. KIVI was right to special-case the geometry.
2. **Keep the first ~4 tokens of K and V at full precision.** They are the attention sinks. Quantizing them is asking for trouble. Almost every modern method does this.
3. **Outliers are not noise.** They are a *load-bearing feature* the model uses to manage attention. Treat them as first-class, not as values to clip.
4. **Rotation methods sidestep the asymmetry but pay a kernel cost.** A Hadamard rotation flattens out the per-channel and per-token outliers, making everything quantize uniformly — at the cost of an in-line transform during attention. See [Rotations](../17-rotations/).
5. **The story is not done.** If the field figures out how to *train without* attention sinks, the entire KV-quantization rulebook changes.

**Continue to** → [Rotations](../17-rotations/) — the geometric trick that bypasses the K/V asymmetry by spreading outliers uniformly across the whole basis. Used by the most recent KV methods (TurboQuant) and the most recent weight methods (QuaRot, SpinQuant) alike.
