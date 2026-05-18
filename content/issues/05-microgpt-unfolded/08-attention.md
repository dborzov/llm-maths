---
title: "Attention: why divide by sqrt(head_dim)?"
short_title: "Attention"
description: "The six lines of microGPT that implement scaled dot-product attention — `attn_logits`, `attn_weights`, `head_out` — and why the scale factor is exactly `sqrt(head_dim)`."
blurb:
  - "Additive attention (Bahdanau) vs. dot-product (Luong): the team expected the MLP scorer to win. It didn't."
  - "Without the `/ head_dim**0.5` scale, softmax saturates into a one-hot at initialization. Training stops."
  - "Three variables: `attn_logits` (raw similarity, length T), `attn_weights` (probability distribution, length T), `head_out` (weighted average of values, length head_dim)."
  - "The description calls attention 'a content-addressable database.' What does that mean in terms of the six lines?"
topics: [transformer, attention]
tags: [microgpt, attention, scaled-dot-product]
theme: teal
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 80
techKind: mainline
techNode: attention
header: 08-attention.webp
---

## Mountain View, Spring 2017

In March 2017, the eight authors of *Attention Is All You Need* were in the middle of one of those polite, low-stakes arguments that decide research papers. Noam Shazeer had a draft of what we now call the **Transformer** running on a translation benchmark. It worked. The fight, internally, was about whether the attention scoring function should be the **additive attention** of Bahdanau et al. (2014) — a small feed-forward network that takes `q` and `k` and emits a scalar — or the much cheaper **dot product** that Luong et al. (2015) had proposed.

The team had a strong prior that additive attention would win. Dot products were *too simple*. A two-layer MLP score function had more parameters, more expressivity, more places for the model to put cleverness. Surely that would matter.

It didn't. When they ran the head-to-head ablation, scaled dot product matched or beat additive attention at every model size — *and* it was several times faster, because a dot product is a single fused matmul on hardware designed to make matmuls fast. The "scaled" part of the name came from a second discovery the team made along the way: plain dot product worked **worse** than additive attention until they divided the logits by $\sqrt{d_k}$. That one square root was the difference between a model that trained and a model whose softmax saturated into one-hot uselessness at initialization.

Today, every frontier LLM in production runs the algorithm they grumblingly settled on. The inner loop is six lines of microGPT. Let's read them.

## The Six Lines

Open [the cold open](../01-cold-open/) and find the inner-most block inside the per-head loop. The lines that matter, with everything else stripped out:

```python
attn_logits = [
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
    for t in range(len(k_h))
]
attn_weights = softmax(attn_logits)
head_out = [
    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
    for j in range(head_dim)
]
```

That is **all of attention**. In the textbook notation,

$$
\text{Attention}(Q, K, V) \;=\; \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

— but for our purposes, the six lines above *are* the formula. They are scaled-dot-product attention on one head, for one query token, against a {{< wiki "kv-cache" >}}cache{{< /wiki >}} of `T` past keys and values. Three things come out:

| Variable | What it is | Shape |
|---|---|---|
| `attn_logits` | raw similarity scores: how much does this query "want" each past token? | length `T` |
| `attn_weights` | softmax of the logits: a probability distribution over the past | length `T`, sums to 1 |
| `head_out` | weighted average of past values | length `head_dim` |

`q_h`, `k_h`, `v_h` arrive pre-sliced from [the QKV projections](../06-qkv-projections/) and [the multi-head split](../09-multi-head/) — `q_h` is this token's query for head `h`, `k_h[t]` is past token `t`'s key for head `h`, `v_h[t]` is past token `t`'s value for head `h`. Where they come from is somebody else's chapter. *What this chapter answers* is: now that we have them, how do they combine?

## Attention Is A Content-Addressable Database

Forget every neuroscience metaphor you've read. The cleanest way to think about a single attention head is as a one-shot **soft lookup table**.

- The `T` past tokens have each filed a `(key, value)` pair into a tiny database: `k_h[t]` is the index card, `v_h[t]` is the contents on the back of the card.
- The current token shows up at the desk holding a query `q_h`.
- The librarian compares `q_h` against every index card using the dot product. *Higher dot product = more aligned in vector space = more "this card is the one you want".*
- Instead of returning *one* card (a hard lookup), the librarian returns a **probability-weighted mix** of all the cards' contents. The {{< wiki "softmax" >}}softmax{{< /wiki >}} decides the mixing weights.

The mixing-rather-than-picking is what makes attention **differentiable**, which is what makes it trainable. Hard `argmax` lookup has no gradient. Softmax does.

That's it. That is the whole conceptual picture. The dot product is the similarity sensor. The softmax is the soft-argmax that picks. The value-weighted sum is the contents-of-the-card you walk out with.

## The Dot Product, Mechanically

Let's see one attention head fire. We'll construct a synthetic situation: 16 past tokens, each with a hand-picked key, plus one current query whose key-alignment we can read off by eye.

```pyplot {id="attn-weights-heatmap" caption="A single attention head over a 16-token cache. Top: dot-product logits before scaling. Middle: scaled and softmaxed. Bottom: which past token gets attended to. The query was deliberately pointed near token 4 and token 11."}
np.random.seed(7)
head_dim = 16
T = 16

# A synthetic cache of T keys, drawn iid Gaussian
k_h = np.random.randn(T, head_dim)

# Build the query to deliberately align with keys 4 and 11
q_h = 0.6 * k_h[4] + 0.4 * k_h[11] + 0.3 * np.random.randn(head_dim)

# The actual six lines of microGPT, vectorised
attn_logits_unscaled = k_h @ q_h
attn_logits = attn_logits_unscaled / np.sqrt(head_dim)

# Numerically stable softmax
e = np.exp(attn_logits - attn_logits.max())
attn_weights = e / e.sum()

fig, axes = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True)

axes[0].bar(range(T), attn_logits_unscaled, color='#FF8C00', edgecolor='#1A1A1A')
axes[0].set_ylabel('q · k\n(unscaled)')
axes[0].set_title('Step 1: dot products. Tokens 4 and 11 score highest.')
axes[0].axhline(0, color='#1A1A1A', linewidth=0.5)

axes[1].bar(range(T), attn_logits, color='#00A8A8', edgecolor='#1A1A1A')
axes[1].set_ylabel('attn_logits\n(scaled)')
axes[1].set_title('Step 2: divide by sqrt(head_dim). Same shape, smaller range.')
axes[1].axhline(0, color='#1A1A1A', linewidth=0.5)

axes[2].bar(range(T), attn_weights, color='#FF007F', edgecolor='#1A1A1A')
axes[2].set_ylabel('attn_weights\n(softmax)')
axes[2].set_xlabel('past token index t')
axes[2].set_title(f'Step 3: softmax. Mass concentrates on t=4 ({attn_weights[4]:.2f}) and t=11 ({attn_weights[11]:.2f}).')

plt.tight_layout()
```

Three plots, three lines of code. The bottom panel is the actionable output: a probability distribution over the past, peaked exactly where we built the query to look. The `head_out` produced by line three of the microGPT block is then a $0.4 v_4 + 0.3 v_{11} + \text{(small contributions from everyone else)}$ kind of vector.

Notice what the model has *not* done: it never compared `q_h` to itself in a special way, never used position information, never asked "what comes after token 4?" — none of that. The only operation is **dot product, scale, softmax, weighted sum**. Position information has to be smuggled in somewhere else (RoPE in modern models, `wpe` in microGPT — see [embeddings](../03-embeddings/)). Causality is handled by *what's even in the cache to begin with* (more on that below). Attention itself is a beautifully memoryless soft lookup.

## Why Divide By $\sqrt{d_k}$?

Here is the question that confused everyone in 2017, including the authors. Why exactly $\sqrt{d_k}$? Why not $d_k$? Why not nothing? The answer is a one-line Fermi estimate that you can do on the back of an envelope.

Suppose `q_h` and `k_h[t]` each have $d = \text{head\_dim}$ components, and assume — purely for the napkin — that each component is independent with mean 0 and unit variance. Then

$$
q_h \cdot k_h = \sum_{j=1}^{d} q_h[j] \cdot k_h[t][j]
$$

is a sum of $d$ independent products of unit-variance numbers. Each summand has variance $1$, the sum has variance $d$, and so the **standard deviation** of the raw dot product is $\sqrt{d}$.

For Llama-style models with $d = 128$, the typical unscaled dot product magnitude is $\sqrt{128} \approx 11$. For GPT-2's $d = 64$, it's $\sqrt{64} = 8$. Either way, **softmax of values in the range $\pm 10$ is approximately one-hot.** $e^{10}$ is twenty-two thousand; one logit one unit higher than the next swallows all the probability mass. At initialization, before training has shaped anything, the model would attend to exactly one past token, gradient information about every other past token would be near-zero, and learning would crawl.

Dividing by $\sqrt{d}$ rescales the dot product back to unit variance, regardless of head dimension. The softmax sees logits in the range $\pm 2$ or $\pm 3$, gives non-trivial weight to several past tokens, and gradients flow to all of them.

```pyplot {id="scale-vs-noscale" caption="Softmax of dot products as the head dimension grows. Without the sqrt(d) scale, the softmax saturates to one-hot at large d. With the scale, the distribution stays usable at every dimension."}
np.random.seed(0)
dims = [4, 16, 64, 256, 1024]
T = 32
fig, axes = plt.subplots(2, len(dims), figsize=(13, 4.5), sharey='row')

for col, d in enumerate(dims):
    q = np.random.randn(d)
    k = np.random.randn(T, d)
    raw = k @ q

    # unscaled
    e1 = np.exp(raw - raw.max()); w1 = e1 / e1.sum()
    # scaled
    rs = raw / np.sqrt(d)
    e2 = np.exp(rs - rs.max()); w2 = e2 / e2.sum()

    axes[0, col].bar(range(T), w1, color='#FF8C00', edgecolor='#1A1A1A', linewidth=0.4)
    axes[0, col].set_title(f'd={d}\nmax weight {w1.max():.2f}')
    axes[0, col].set_ylim(0, 1.05)

    axes[1, col].bar(range(T), w2, color='#00A8A8', edgecolor='#1A1A1A', linewidth=0.4)
    axes[1, col].set_title(f'max weight {w2.max():.2f}')
    axes[1, col].set_ylim(0, 1.05)

axes[0, 0].set_ylabel('UNSCALED\nsoftmax(q·k)')
axes[1, 0].set_ylabel('SCALED\nsoftmax(q·k / √d)')
for ax in axes.flat:
    ax.set_xticks([])
fig.suptitle('Why the √d scale matters: unscaled softmax saturates as d grows', y=1.02)
plt.tight_layout()
```

Read the top row left-to-right: at $d=4$, the unscaled softmax is broad and usable. By $d=64$, almost all the mass is on one token. By $d=1024$, the model is doing hard argmax — and getting zero gradient on every other key it could have learned from. The bottom row shows the scaled version stays well-behaved at every $d$. That little division by $\sqrt{d}$ is the difference between a transformer that trains and a transformer that doesn't.

## Where The Causal Mask Went

If you have read any other attention tutorial, you've probably seen a big lower-triangular matrix called the **causal mask**. The mask is the upper-triangular set to $-\infty$, so that when softmax is applied along each row, every "future" token in the row gets weight zero. In a training pipeline this matters enormously — training shoves an entire sequence through the model in one batched matmul, and without the mask token $t$ could trivially "cheat" by attending to token $t+5$.

In microGPT — and in production *decode* — **there is no mask, because there is nothing to mask**. The function `gpt(token_id, pos_id, keys, values)` runs *exactly once per token*. By the time the model is computing attention for token 17, `keys[li]` only contains the keys for tokens 0 through 17. The cache *is* the causal structure: future tokens aren't masked, they simply haven't been computed yet.

This is one of the small but cognitively-clarifying moves of the microGPT framing. The causal mask isn't a piece of attention. It's a piece of how training reshapes attention into a batched-rectangular form. **Inference is naturally causal.**

(The story flips back during *prefill*, where the prompt is processed all-at-once and a mask reappears. [Prefill vs Decode](../14-prefill-decode/) is the chapter that draws the distinction in full.)

## Napkin Math: How Many FLOPs Per Token?

Let's count operations for one head, one layer, one decode step at context length $T$:

- **Logits**: $T$ dot products of length `head_dim`. Cost: $T \cdot \text{head\_dim}$ multiplies + $T \cdot \text{head\_dim}$ adds $\approx 2 T \cdot \text{head\_dim}$ FLOPs.
- **Softmax**: $T$ exponentials + $T$ divides. Linear in $T$, no `head_dim` factor.
- **Value mix**: `head_dim` outputs, each a length-$T$ weighted sum. Cost: $T \cdot \text{head\_dim}$ multiplies + adds $\approx 2 T \cdot \text{head\_dim}$ FLOPs.

So **attention costs about $4 T \cdot \text{head\_dim}$ FLOPs per head per token**, plus a softmax that is negligible. Now scale to Llama-3-8B at the long-context regime: $T = 8192$, $\text{head\_dim} = 128$, $n_\text{head} = 32$, $n_\text{layer} = 32$.

$$
\underbrace{4 \cdot 8192 \cdot 128}_{\approx 4.2 \text{M FLOPs / head}} \;\times\; \underbrace{32}_{\text{heads}} \;\times\; \underbrace{32}_{\text{layers}} \;\approx\; 4.3 \text{ GFLOPs per generated token}
$$

Four gigaflops. Just for the attention. Per decoded token. The MLP block, which is also $O(n_\text{embd}^2)$ but with a 4× expansion, is roughly $4 \cdot 2 \cdot 4096 \cdot 4 \cdot 4096 \approx 0.5$ GFLOPs — *eight times cheaper than attention* at this context length. The longer the context, the more attention dominates. This is exactly why the entire post-2022 long-context literature is about not paying $O(T)$ per token. We see one of the answers in [The Three Axes of KV Compression](../17-kv-axes/).

A related sting: the attention computation **doesn't materialize a full $T \times T$ matrix** for any given decoder step. We only have *one* query (the new token) and $T$ keys/values. The whole logits array is length $T$, not $T \times T$. The $T \times T$ matrix only appears during training (and during prefill, sort of). When you read about [Flash-Attention](https://arxiv.org/abs/2205.14135) — the kernel that fuses the three steps above so that the intermediate logits never even hit GPU memory — keep in mind that the win at decode time isn't about avoiding a $T \times T$ matrix; it's about avoiding $T$ separate kernel launches. The win at training time is the real $T \times T$ memory savings. (Issue 4, [The Heatmap That Lied](../../../issues/04-heatmap-that-lied/), has more on this.)

## So Why Does This Work, Really?

Here is the part that, even in 2026, is still mildly miraculous.

The dot product `q · k` is a single scalar. The query is a vector of `head_dim` floats. The key is a vector of `head_dim` floats. **Nothing learned at train time lives inside the dot product itself** — it's a hardware operation. All the model learns is **which `Q`, `K`, `V` projection matrices to use** so that the right tokens end up close together in that scalar after the dot product.

You can think of it like this: the model is learning a *language of similarity*. It learns to project the residual stream into a space where "this is a noun and I'm looking for its determiner" comes out as a high dot product. It learns to project differently for "this is the closing parenthesis and I'm looking for the opening one". Every head, every layer, gets its own private projection geometry, and each one is free to use the same dot-product hardware in a different way.

That's the 2017 surprise. The compute is dumb; the projections are smart; and the field-defining insight was that this was *enough*. No MLP score function. No fancy attention variant. Just $QK^\top / \sqrt{d}$ and a softmax, billions of times.

## What To Remember

1. **Attention is six lines.** Dot product against every past key, divide by $\sqrt{d_k}$, softmax, weight-average the values. That's it.
2. **`attn_logits` is length $T$, `attn_weights` is a probability distribution of length $T$, `head_out` is length `head_dim`.** Memorize the three shapes; everything follows.
3. **The $\sqrt{d_k}$ is not arbitrary.** It's the standard deviation of a sum of $d$ unit-variance products. Without it, softmax saturates at the head dimensions used by real models.
4. **No mask at decode.** Causality is enforced by the cache being naturally past-only. Training masks; inference doesn't need to.
5. **Attention is the model's content-addressable memory.** The dot-product hardware is dumb; the `Q`, `K`, `V` projections (chapter 6) are where the model writes its dialect of "similarity".

---

**Continue to** → [Multi-Head Attention](../09-multi-head/) — one head is a single language of similarity; an LLM speaks dozens of them in parallel, and the trick is how cheaply they share the residual stream.

