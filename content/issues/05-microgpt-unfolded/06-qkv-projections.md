---
title: "Q, K, V: The Three Projections"
description: "Three linear maps applied to the same input vector. The model uses one copy of `x` to ask a question (`q`), another to advertise what it knows (`k`), and a third to carry the payload (`v`). The aha is that these are *learned* roles."
topics: [transformer, attention]
tags: [microgpt, attn_wq, attn_wk, attn_wv]
theme: teal
math: true
draft: false
date: 2026-05-14T02:48:52-04:00
issue: 5
weight: 60
techKind: mainline
techNode: qkv-projections
header: 06-qkv-projections.webp
---

## Montréal, September 2014

A 28-year-old PhD student named **Dzmitry Bahdanau** is in his third year at the Université de Montréal, working under Yoshua Bengio on neural machine translation. The state of the art is *seq2seq*: an encoder RNN reads a French sentence and squashes it into a single fixed-length vector, and a decoder RNN expands that vector into English. It works. Barely. Past about twenty words the translation goes sideways — the single bottleneck vector cannot remember everything the source sentence said.

Bahdanau's idea, scribbled with Bengio and Kyunghyun Cho over a few weeks, is so simple it sounds like cheating: **let the decoder peek back at the encoder's hidden states**. At every output step, compute a similarity between *what the decoder is trying to produce* and *each encoder state*, softmax those similarities into weights, and take a weighted average of the encoder states. The single bottleneck dies. The 30-word ceiling on translation quality dies with it. The paper introduces a word that did not exist in machine learning before: **attention**.

Three years later, in June 2017, eight researchers at Google Brain publish a paper whose title is a thesis statement: *Attention Is All You Need*. They strip every recurrent connection out of the architecture, leave only stacks of attention layers, and call the result the **Transformer**. Buried in section 3.2 is a refactoring of Bahdanau's "let the decoder look at the encoder" trick, renamed:

> **Q**uery, **K**ey, **V**alue.

These three letters — Q, K, V — are the entire reason microGPT works. And those three letters live, in our [cold-open listing](../01-cold-open/), in exactly these three [ch.4 linear](../04-linear/) calls:

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
```

Three matrix-vector products. **The same input `x` going into all three.** This article is about why three, and what each one means.

## The Library Card Catalog

Before the math, the metaphor. Imagine you walk into a public library in 1985, before the internet. You have a question: *"How does a microwave oven work?"* You go to the card catalog — a wooden cabinet of index cards, one per book. Each card has a **title** on its top line. You flip through the cards looking for titles that *match* your question. When you find a matching card, you copy the call number and walk to the stacks to fetch **the book itself**.

Three things happened, and they are exactly the three things Q, K, V are for:

| You | The library |
|---|---|
| The question in your head | **Query** `q` — what you are looking for |
| The title line on each card | **Key** `k` — what each entry advertises itself as |
| The actual book you fetch | **Value** `v` — the payload, the thing you take home |

This is the architecture of attention. You match a **query** against many **keys**, and once you match, you fetch the **value**. The query and keys must share a vocabulary — you cannot match an English question against a German title. The value lives separately; it is the *payload* retrieved once the match has been made.

Inside a transformer, the "library" is the set of tokens in the current context. The current token turns its residual vector into a query — *"what do I need to know to write the next word?"* — and every other token (and itself!) has long since turned its residual vector into a key (*"I am a noun, I am plural, I sit two words back from a verb"*) and a value (*"here is what I'd hand you if you called on me"*). The model dots the query against every key, softmaxes the scores, and weighted-averages the values. That is [ch.7 softmax](../07-softmax/) and [ch.8 attention](../08-attention/). *Here* in chapter 6, our concern is the three projections that **produce** `q`, `k`, and `v` from the same `x`.

## Three Readings Of The Same Vector

Look at the code again:

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
```

`x` is the **same vector** in all three lines. Length `n_embd` — sixteen floats in microGPT. The three weight matrices `attn_wq`, `attn_wk`, `attn_wv` are different. Each has shape `(n_embd, n_embd)` — sixteen by sixteen. Each is independently learned during training. The output is three vectors `q`, `k`, `v`, each of length `n_embd`.

In math, with subscripts that match the [ch.4 linear](../04-linear/) convention `y_i = sum_j W_ij · x_j`:

$$
q = W_Q\, x, \qquad k = W_K\, x, \qquad v = W_V\, x
$$

This is the **aha**: **the model never sees three different inputs.** It sees one vector `x` and produces three *readings* of it. The query reading asks "what am I hunting for"; the key reading announces "this is what I am, if anyone is hunting"; the value reading prepares "this is what I'd hand over." The three roles are not given — they are *learned*, jointly, by gradient descent over hundreds of billions of training tokens.

Why three and not one? Suppose we used the same projection: $W_Q = W_K = W_V = W$. Then the attention score between two tokens is $q_i \cdot k_j = (Wx_i) \cdot (Wx_j)$ — the model can *only* attend to tokens whose `Wx` looks like its own. A pronoun cannot ask "who is my antecedent?" because the question and the answer would have to look alike. A verb cannot ask "where is my subject?" because subjects and verbs would have to encode the same way. The same-projection model is forced to attend to copies of itself.

Separating $W_Q$ from $W_K$ breaks the symmetry: the model can learn "what I need" in one geometry and "what I have" in another. Separating $W_V$ from both frees the **payload** from the **matching key** — a card catalog entry can advertise "history book" on its title line and contain *A Brief History of Time* in the volume itself. **What you advertise yourself as is not what you carry.**

## A Toy Forward Pass, Heat-Stripped

Pictures help. Let us project a single 16-dimensional input through three randomly-initialized weight matrices and look at the resulting Q, K, V vectors as heat strips.

```pyplot {id="qkv-three-readings" caption="The same input x, projected three different ways. Different W matrices produce wildly different output vectors — three readings of one input."}
np.random.seed(7)
n_embd = 16

# One token's residual stream after rmsnorm
x = np.random.randn(n_embd) * 0.6

# Three independently-initialized projection matrices
W_q = np.random.randn(n_embd, n_embd) * (1.0 / np.sqrt(n_embd))
W_k = np.random.randn(n_embd, n_embd) * (1.0 / np.sqrt(n_embd))
W_v = np.random.randn(n_embd, n_embd) * (1.0 / np.sqrt(n_embd))

q = W_q @ x
k = W_k @ x
v = W_v @ x

fig, axes = plt.subplots(4, 1, figsize=(9, 4.5))
vmax = max(np.abs(x).max(), np.abs(q).max(), np.abs(k).max(), np.abs(v).max())
for ax, vec, name, color in [
    (axes[0], x, 'x  (residual)', '#1A1A1A'),
    (axes[1], q, 'q  = W_q · x', '#FF007F'),
    (axes[2], k, 'k  = W_k · x', '#00A8A8'),
    (axes[3], v, 'v  = W_v · x', '#FFD700'),
]:
    ax.imshow(vec.reshape(1, -1), aspect='auto', cmap='RdBu_r',
              vmin=-vmax, vmax=vmax)
    ax.set_yticks([0])
    ax.set_yticklabels([name], fontsize=10, fontweight='bold', color=color)
    ax.set_xticks(range(n_embd))
    ax.set_xticklabels([str(i) for i in range(n_embd)], fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1A1A1A')
        spine.set_linewidth(1.2)

axes[-1].set_xlabel('dimension index (0 .. n_embd-1)')
fig.suptitle('Three projections of one 16-dim input  (random init, untrained)',
             fontweight='bold')
plt.tight_layout()
```

Notice that **`q`, `k`, and `v` have no visible relationship to `x`** — different sign patterns, different magnitudes. Each is a linear function of `x`, but each *reads* a different combination of `x`'s entries. After two trillion tokens of pretraining, each row of $W_Q$ becomes a specific *question template*, each row of $W_K$ an *advertising template*, each row of $W_V$ a *payload template*.

In microGPT these vectors are also concatenations of `n_head = 4` per-head slices of length `head_dim = 4`. The slicing happens in [ch.9 multi-head](../09-multi-head/); the three *projections* don't care about head structure — they happen first.

## After Training: Q And K Get Cozy, V Goes Its Own Way

Learned QKV projections leave a fingerprint in the weights: **the rows of $W_Q$ and $W_K$ become correlated; the rows of $W_V$ remain decorrelated.**

The intuition: the attention score is $q \cdot k = x^\top W_Q^\top W_K\, x$. Whatever directions in `x`-space the model wants to attend along, *both* $W_Q$ and $W_K$ must learn to project along. They are partners in a single inner product and must agree on a coordinate system. $W_V$ has no such constraint — it can encode whatever payload it likes.

A small simulation. Pretend $W_Q$ and $W_K$ both align to a shared low-dimensional "topic" subspace during training, while $W_V$ stays random:

```pyplot {id="qk-correlation" caption="Top-left: QK Gram matrix (W_Q^T W_K) after simulated training — strong diagonal/block structure. Bottom-left: same for V — diffuse, near-orthogonal. Right: row-row correlation distributions."}
np.random.seed(2)
n_embd = 64

# Simulate "trained" Q and K: both pulled toward a shared low-rank subspace
shared_basis = np.random.randn(n_embd, 8)
W_q_trained = 0.3 * np.random.randn(n_embd, n_embd) + shared_basis @ np.random.randn(8, n_embd)
W_k_trained = 0.3 * np.random.randn(n_embd, n_embd) + shared_basis @ np.random.randn(8, n_embd)

# V has no such pressure — stays roughly random
W_v_trained = np.random.randn(n_embd, n_embd)

def row_corr(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A @ B.T).flatten()

qk = W_q_trained.T @ W_k_trained
vv = W_v_trained.T @ W_v_trained

fig = plt.figure(figsize=(11, 5))
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax2 = plt.subplot2grid((2, 3), (1, 0))
ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)

m = max(np.abs(qk).max(), np.abs(vv).max())
ax1.imshow(qk, cmap='RdBu_r', vmin=-m, vmax=m, aspect='auto')
ax1.set_title('W_Q$^T$ W_K  (after "training")', fontsize=10, color='#FF007F', fontweight='bold')
ax1.set_xticks([]); ax1.set_yticks([])

ax2.imshow(vv, cmap='RdBu_r', vmin=-m, vmax=m, aspect='auto')
ax2.set_title('W_V$^T$ W_V', fontsize=10, color='#FFD700', fontweight='bold')
ax2.set_xticks([]); ax2.set_yticks([])

qk_corrs = row_corr(W_q_trained, W_k_trained)
vv_corrs = row_corr(W_v_trained, W_v_trained)
# Drop the diagonal self-similarities for V
vv_corrs = vv_corrs[np.abs(vv_corrs - 1) > 1e-6]

bins = np.linspace(-1, 1, 60)
ax3.hist(qk_corrs, bins=bins, alpha=0.75, color='#FF007F',
         label='Q-row vs K-row correlations', edgecolor='#1A1A1A', linewidth=0.4)
ax3.hist(vv_corrs, bins=bins, alpha=0.6, color='#FFD700',
         label='V-row vs V-row correlations', edgecolor='#1A1A1A', linewidth=0.4)
ax3.axvline(0, color='#1A1A1A', linewidth=1)
ax3.set_xlabel('row-row cosine similarity')
ax3.set_ylabel('count')
ax3.set_title('Q and K learn a shared geometry. V does not.', fontweight='bold')
ax3.legend(loc='upper left')
ax3.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The pink histogram has a fat shoulder away from zero — Q rows and K rows have *learned* to correlate. The yellow histogram sits centered on zero — V rows are doing their own thing. This signature shows up in real pretrained checkpoints; mechanistic-interpretability papers lean on it.

> **The Q-K subspace** is the geometry of *what tokens look for in each other*.
> **The V subspace** is the geometry of *what tokens hand each other once they've matched*.
> Different jobs. Different shapes. Same `x`.

## Modern Wrinkles On The Three-Projection Recipe

**No biases.** The 2017 Transformer paper had biases — affine maps $Wx + b$. Modern frontier LLMs (Llama, Mistral, Gemma, Qwen, DeepSeek) drop them. Same justification as [ch.4 linear](../04-linear/): [ch.5 rmsnorm](../05-rmsnorm/) sits in front of every projection and its learnable scale absorbs what a bias would have done.

**QK-norm.** Past 30B parameters, training stability gets fragile. The patch: apply RMSNorm to `q` and `k` *after* the projections and *before* the dot product. Without it, Q and K magnitudes drift, attention logits blow up, and the softmax saturates. Gemma 2, DeepSeek, and several Llama variants ship with QK-norm; microGPT, for legibility, does not.

**Fused QKV.** On a real GPU, three back-to-back matmuls with the same `x` is wasteful — three kernel launches, three HBM reads of `x`. Modern kernels stack the three weight matrices into one tall `(3·n_embd, n_embd)` matrix, do a single matmul, then split the output. Same math, one kernel.

**GQA: K and V get smaller.** The biggest deviation from the vanilla recipe. In **Grouped Query Attention** — Llama 3, Mistral, Gemma 2, every frontier LLM post-2023 — the K and V projections produce vectors *shorter* than `n_embd`. Multiple query heads share each K/V head. The K/V matrices become `(n_kv_head · head_dim, n_embd)` with `n_kv_head < n_head`. Full story in [ch.18 gqa](../18-gqa/); headline now: **the three projections in modern LLMs are not the same shape anymore**.

## Napkin Math: Parameters In The QKV Block

For microGPT's toy model with `n_embd = 16`:

$$
3 \cdot n_\text{embd}^2 = 3 \cdot 16^2 = 768 \text{ params per layer for QKV.}
$$

Across both layers, **1536 parameters**. That's 21% of the toy model's 7264-parameter budget, just for asking, advertising, and packaging.

Now **Llama 3 8B**: `n_embd = 4096`, `n_head = 32`, `n_kv_head = 8` (GQA, 4:1 ratio), `head_dim = 128`.

If Llama 3 used *vanilla* MHA — three full `4096 × 4096` matrices — that would be

$$
3 \cdot 4096^2 = 50{,}331{,}648 \approx \mathbf{50.3\text{M params per layer}}
$$

Across 32 layers: **1.61 billion parameters** just for the QKV projections. Of an 8B-parameter model.

With GQA, the math is asymmetric:

$$
\underbrace{4096 \cdot 4096}_{W_Q} + \underbrace{4096 \cdot (8 \cdot 128)}_{W_K} + \underbrace{4096 \cdot (8 \cdot 128)}_{W_V}
= 16.8\text{M} + 4.2\text{M} + 4.2\text{M} = \mathbf{25.2\text{M params per layer}}
$$

Half the parameters per layer. Across 32 layers, **806M** instead of 1.61B — **800M parameters saved**, before counting the matching savings in KV cache memory at inference time (see [ch.13 kv-cache](../13-kv-cache/) and [ch.17 kv-axes](../17-kv-axes/)).

This is why GQA is the default in every frontier model from 2024 onward. Q has to express the *full diversity of attention patterns*; K and V only have to express the *retrievable content of each token*. The latter compresses; the former does not. The architecture follows the asymmetry.

## What To Remember

1. **Q, K, V are three *readings* of the same input**, not three different inputs. `q = W_Q x`, `k = W_K x`, `v = W_V x`. The three matrices are learned jointly during training.
2. **The library card catalog metaphor is exact.** Q is your search query, K is the title line on each card, V is the book you take home. Q and K must speak the same language; V doesn't.
3. **Why three matrices and not one**: untying $W_Q$ from $W_K$ lets the model ask questions different from the answers it offers; untying $W_V$ from both lets the payload be different from the matching key.
4. **Q and K learn correlated geometry; V doesn't.** Q and K are partners in an inner product, so they must agree on a coordinate system. V is free to encode whatever payload is useful.
5. **Modern wrinkles**: no biases (RMSNorm absorbs them), optional QK-norm for training stability, fused QKV for one-kernel matmuls, and — most consequentially — **GQA**, which shrinks K and V to a fraction of Q's width and saves nearly a billion parameters in an 8B model.
6. **Napkin math**: vanilla MHA Llama 3 8B would burn 1.6B params on QKV; with GQA it's 800M. Half the parameters, the same expressive ceiling on attention patterns.

---

**Continue to** → [Softmax, Numerically](../07-softmax/) — once we have q, k, v in hand, we dot the query against every key and need to turn those raw scores into a probability distribution without overflowing a 16-bit float.

