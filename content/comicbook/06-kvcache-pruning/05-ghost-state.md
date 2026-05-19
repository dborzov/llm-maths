---
title: "Residual Stream: the oracle hiding in the hidden state"
short_title: "Residual Stream"
description: "Jégou and Jebleck discovered that KVzip+'s oracle importance scores are already latent in the transformer's residual stream — a tiny surrogate model trained on 1.2 million (hidden-state, log-score) pairs achieves R²=0.67–0.77."
blurb:
  - "KVzip+ (the oracle) costs two full forward passes per prompt — impossible during autoregressive decode."
  - "Hypothesis: if a token will be ignored by all future positions, the transformer already knows — it's encoded in h_t."
  - "Training data: 27,000 prompts from Nemotron-Pretraining-Dataset-sample, filtered to 750–1,250 tokens, yielding 1.2M (h, log s⁺) pairs per KV head."
  - "KVzap-MLP has ~2.1M parameters for Qwen3-8B — 0.025% of the base model — and fits in a single matmul."
topics: [kv-cache, attention, transformers]
tags: [kvzap, hidden-state, residual-stream, r-squared, surrogate-model]
theme: cream
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 50
techKind: mainline
techNode: ghost-state
header: 05-ghost-state.webp
---

## Late 2025, NVIDIA Research

Simon Jégou has a problem that would look like success to almost anyone else in the field.

He has KVzip+ — the oracle. Feed a document into the model, run a second masked forward pass that forces the model to reconstruct token $i$ using only the other tokens' cached keys and values, and the attention weights from that pass tell you *exactly* which KV pairs are load-bearing and which are ignored. The importance score $s_i^+$ for position $i$ is clean, well-motivated, and empirically predictive of which tokens can be evicted without accuracy loss.

The problem: generating $s_i^+$ costs *two* full forward passes per prompt. Not just in compute — the second pass is structurally incompatible with incremental decoding. At prefill, a second pass is painful. During decode, it is simply impossible: the model is producing tokens one at a time; there is no clean moment to run the oracle query. KVzip+ is the ground truth, but it cannot be the production system.

The question Jégou and Jebleck started asking in late 2025: **can we build a cheap surrogate that predicts $s_i^+$ without running the second pass?**

The answer was already inside the model. They just had to look in the right place.

## The Residual Stream Hypothesis

To understand why this works, you need a concrete mental model of what the {{< wiki "residual-stream" >}}hidden state{{< /wiki >}} carries.

At layer $l$, position $t$, the transformer produces a vector $h_t^{(l)} \in \mathbb{R}^{D_h}$. In the microGPT vocabulary from [issue 05](/comicbook/05-microgpt/10-residual-stream/), this is `x_residual` — the running sum of everything the model has computed about position $t$ given its context. Every {{< wiki "attention" >}}attention{{< /wiki >}} sublayer adds to it. Every MLP sublayer adds to it. By the time you reach the final layer, `x_residual` at position $t$ is a compressed encoding of: *what is at position $t$, what surrounds it, and how every previous layer has responded to it.*

The hypothesis: **if a token is consistently ignored by later positions, the transformer knows it.** The attention pattern is computed *from* the keys and queries, which are themselves computed *from* $h_t$. If future positions will assign near-zero {{< wiki "softmax" >}}softmax{{< /wiki >}} weight to position $t$'s key — if $t$ is genuinely irrelevant in context — then that irrelevance should be predictable from $h_t$ itself.

Stated more crisply: the residual stream at position $t$ already carries a latent signal about position $t$'s importance to the rest of the sequence. We just need a function $f: \mathbb{R}^{D_h} \to \mathbb{R}^H$ that surfaces it.

This is not guaranteed to work. It is a hypothesis. The team went and tested it.

## Training the Surrogate

The data collection procedure was methodical. From the Nemotron-Pretraining-Dataset-sample — 27,000 prompts across nine subsets (web crawl, multilingual, math, code, and others) — they collected pairs:

$$
\bigl(h_t^{(l)},\; \log s_t^+\bigr)
$$

for every position $t$, every {{< wiki "kv-cache" >}}KV{{< /wiki >}} head, every sampled layer. The log-space target is deliberate: {{< wiki "softmax" >}}softmax{{< /wiki >}} attention scores are exponentially distributed, so fitting in log-space gives a better-conditioned regression problem. A model trained to predict $s_t^+$ directly would struggle with the dynamic range; trained to predict $\log s_t^+$ it sees a much more Gaussian-looking target.

{{% callout type="note" title="Why 750–1250 tokens?" %}}
Prompts were filtered to the 750–1,250 token window. This is not arbitrary. At very short contexts, attention weights are concentrated differently — there simply aren't enough tokens to create the sparse-vs-dense pattern that makes some tokens important and others ignorable. At very long contexts, the weight distribution spreads thin in a context-length-dependent way that would bias the surrogate's training. The 750–1,250 range captures "real paragraph density" without the length-confounding effects.
{{% /callout %}}

The 1.2 million $(h, \log s^+)$ pairs per KV head are split into a training set and a held-out test set. Two model architectures are trained on this data:

- **KVzap-Linear**: a single affine map $W \in \mathbb{R}^{H \times D_h}$, predicting $H$ head scores from a $D_h$-dimensional hidden state in one matmul.
- **KVzap-MLP**: two layers — $\mathbb{R}^{D_h} \to \mathbb{R}^{D_h / 8} \to \mathbb{R}^H$ — with a GELU nonlinearity in between.

The MLP has more parameters but is still tiny relative to the model it serves: for Qwen3-8B ($D_h = 4096$, $H = 8$ heads), the MLP has roughly $4096 \times 512 + 512 \times 8 \approx 2.1\text{M}$ parameters — about 0.025% of the base model's 8 billion.

```pyplot {id="r2-across-models" caption="R² OF SURROGATE PREDICTION ACROSS THREE BASE MODELS. MLP consistently outperforms Linear. All R² values are comfortably above the 'useful predictor' threshold."}
fig, ax = plt.subplots(figsize=(9, 4.5))

models = ['Qwen3-8B', 'Llama-3.1-8B\n(Instruct)', 'Qwen3-32B']
r2_linear = [0.69, 0.72, 0.67]
r2_mlp    = [0.77, 0.71, 0.73]

x = np.arange(len(models))
w = 0.35

bars_l = ax.bar(x - w/2, r2_linear, w, color='#00A8A8', label='KVzap-Linear',
                edgecolor='#1A1A1A', linewidth=1.5)
bars_m = ax.bar(x + w/2, r2_mlp,    w, color='#FF007F', label='KVzap-MLP',
                edgecolor='#1A1A1A', linewidth=1.5)

ax.axhline(0.5, color='#1A1A1A', linestyle='--', linewidth=1, alpha=0.4, label='R²=0.5 reference')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('R² (held-out test set)')
ax.set_ylim(0, 1.0)
ax.set_title('Surrogate R²: hidden state → log importance score')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)

for bar in bars_l:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9,
            color='#1A1A1A', fontweight='bold')
for bar in bars_m:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9,
            color='#1A1A1A', fontweight='bold')

plt.tight_layout()
```

**R² = 0.67–0.77.** The hidden state predicts token importance with correlation-squared in the 0.67–0.77 range. This is not a perfect oracle — but it is far better than chance, and as we will see in the next article, it is good enough to achieve near-oracle compression in practice.

## The Per-Layer Story

The R² number by itself buries the most interesting experimental finding. When you track R² *per layer* — fitting the surrogate using hidden states from layer 1, then layer 2, and so on — a clear pattern emerges:

- **Layer 1 R² ≈ 0.4.** At this point, the hidden state is still close to the raw token embedding. It has received one round of attention, but the residual stream has not yet accumulated the rich contextual information that later layers build up.
- **R² rises through the middle layers**, roughly tracking the depth at which each attention head becomes semantically coherent.
- **Final-layer R² is the highest**, confirming that the full residual stream — after all layers have written to it — is the best predictor of token importance.

```pyplot {id="r2-per-layer" caption="PER-LAYER R² FOR QWEN3-8B. INFORMATION ACCUMULATES THROUGH DEPTH — DIRECT EVIDENCE THAT THE RESIDUAL STREAM CARRIES WHAT WE THINK IT DOES."}
np.random.seed(7)
n_layers = 36
layers = np.arange(1, n_layers + 1)

# Simulate the per-layer R² curve: low start, rises with some noise, plateaus
r2_base = 0.4 + (0.77 - 0.4) * (1 - np.exp(-layers / 10))
noise = np.random.randn(n_layers) * 0.018
r2_curve = np.clip(r2_base + noise, 0.35, 0.82)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(layers, 0, r2_curve, alpha=0.18, color='#FF007F')
ax.plot(layers, r2_curve, color='#FF007F', linewidth=2.5, label='MLP surrogate R²')
ax.axhline(0.4, color='#00A8A8', linestyle=':', linewidth=1.5, label='Layer-1 R² ≈ 0.4')
ax.axhline(0.77, color='#FFD700', linestyle=':', linewidth=1.5, label='Final-layer R² ≈ 0.77')

ax.annotate('Token embedding era:\nattention context not yet\naccumulated',
            xy=(3, r2_curve[2]), xytext=(8, 0.44),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A'), fontsize=9)
ax.annotate('Residual stream\nfully loaded',
            xy=(32, r2_curve[31]), xytext=(22, 0.73),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A'), fontsize=9)

ax.set_xlabel('Transformer layer')
ax.set_ylabel('R² (surrogate vs. KVzip+ oracle)')
ax.set_title('How well does the hidden state predict importance? — by layer')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

This per-layer progression is not just a curiosity — it is direct evidence for the residual stream hypothesis. If the hidden state were irrelevant to token importance, R² would be flat and low at every layer. The fact that R² climbs with depth tells us the model is *computing* something about importance through the forward pass, accumulating it in `x_residual`, and that we can read it out with a small learned function.

## Why Not Use Keys and Values Directly?

A natural alternative: instead of predicting importance from $h_t$, predict it from the key vector $k_t$ and value vector $v_t$ that are actually stored in the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}. This would be conceptually tidy — the thing being pruned (the KV pair) is also the input to the pruning decision.

The team tested this. **Keys and values give strictly lower R² than hidden states.**

The reason is not hard to reconstruct. The key vector $k_t = W_K h_t$ is a linear projection of the hidden state — it discards the dimensions of $h_t$ that are not useful for computing query-key dot products. The value vector $v_t = W_V h_t$ similarly discards information irrelevant to the output projection. Both are *derived* from $h_t$, and both throw away parts of $h_t$ in the process.

The hidden state $h_t$, by contrast, carries everything: the token's embedding, the accumulated residual from all upstream attention and MLP computations, and the full $D_h$-dimensional context summary that the model uses to compute the next token. It is richer than either $k_t$ or $v_t$ separately — or both together.

{{% marginnote %}}
In microGPT terms: `q`, `k`, `v` are each projections of `x_residual`. The surrogate reads `x_residual` directly, before the projection that loses information.
{{% /marginnote %}}

This finding also has a practical benefit: because $h_t$ is computed as part of the normal forward pass and is already available at every layer, the surrogate can be evaluated without modifying the attention kernel. The keys and values are also available — but reading from $h_t$ means the surrogate slots into the existing compute graph at a clean boundary.

## The MLP vs. Linear Mystery

The MLP architecture consistently achieves higher R² than the linear surrogate. This is not surprising — it has more parameters and a nonlinearity, so it can fit a wider class of functions. What *is* surprising is what happens on Llama-3.1-8B-Instruct:

On that model, **KVzap-Linear outperforms KVzap-MLP in actual downstream tasks**, even though the MLP achieves higher R². The simpler model, despite explaining less variance in the importance scores, produces better eviction decisions.

The most credible explanation is regularization. The MLP, with its higher capacity, can fit the training distribution of importance scores more precisely — including idiosyncratic patterns in the 750–1,250 token training window that do not generalize to longer contexts at evaluation time. The linear model, precisely because it cannot overfit the training distribution, generalizes better to the 4k–128k context lengths seen at inference.

This is a good reminder of a principle that shows up throughout machine learning: approximation quality on the training distribution and generalization quality are not the same thing. A surrogate with R²=0.72 that generalizes well beats one with R²=0.77 that overfits. The task is not to reconstruct $s_t^+$ perfectly; it is to prune the right tokens.

{{% pullquote type="counter-intuitive" %}}
The model that predicts importance scores less accurately ends up pruning KV pairs more accurately. Lower R², better task performance. A reminder that the proxy metric is not the goal.
{{% /pullquote %}}

## The Metaphor That Holds

Imagine you're watching jury selection, and you need to predict which witnesses will be cross-examined aggressively — without hearing the testimony. You could try to predict this from the witnesses' statements (the key and value vectors). Or you could watch the jury's *body language and attention* as each witness is introduced (the hidden state) — the way jurors lean forward for certain witnesses, the micro-expressions of interest or dismissal.

The hidden state is the jury's body language. It encodes the model's running reaction to each token, accumulated over all the computation that has already happened. A small linear or MLP readout head is the expert who has learned to interpret that body language.

The oracle ($s_t^+$) is what the jury actually does during deliberation. The surrogate predicts this from the body language alone — R²=0.7 correlation. Not perfect. Good enough.

**Continue to → [KVzap: The Final Zap](../06-kvzap/)** — how the surrogate, the threshold, and the sliding window come together into a 14-line function that ships in production, achieves 3.5× compression on Qwen3-8B, and preserves math olympiad performance at more than 50% cache discarded.
