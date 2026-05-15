---
title: "Sixty Lines, One LLM"
description: "A complete, working transformer language model in 60 lines of plain Python. No torch. No numpy. No magic. Read it once, then we spend the rest of the issue unpacking it."
topics: [transformer, inference]
tags: [microgpt, gpt, pure-python, kv-cache]
theme: cream
math: true
draft: false
date: 2026-05-14T02:48:52-04:00
issue: 5
weight: 10
techKind: mainline
techNode: cold-open
header: 01-cold-open-05.webp
---

## The Engineer Who Could Not Sleep

In the summer of 2024, an engineer at a mid-sized AI startup was three days into a new on-call rotation. A production inference service had started returning gibberish for one specific user prompt — only that prompt — and only on one of the eight serving nodes. The stack was the usual modern thing: vLLM in front of CUDA Graphs in front of `flash-attn` v3 in front of a custom paged-attention allocator. By Thursday night they were five layers deep into NVCC error messages and had a vague hypothesis about a misaligned FP8 scale tensor.

They printed out their `model.py`. It was thirty-seven thousand lines.

That weekend, the same engineer sat down with a pen and a single sheet of A4 paper and wrote, by hand, the *minimal* inference loop of a transformer language model. They allowed themselves Python lists. They allowed themselves a `for` loop. They allowed themselves the `math` module. Everything else — every CUDA kernel, every batched tensor op, every cache-coherence trick — was forbidden.

When they were done, they typed the result into a file called `inference.py`. It worked. It was slow — about one character per second for a tiny model trained on first names — but it produced text. They went back to the bug on Monday morning and **had it pinned in forty minutes**. The misaligned FP8 scale wasn't the cause. The actual cause was a stray `position_id` getting reused across batch slots in the paged-attention free list. They could only see it because they finally knew, in their fingertips, *what `position_id` was supposed to be doing*.

This is the engineer's listing.

## The Whole Damn Model

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        # Multi-head Attention
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]

            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [relu(xi) for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, state_dict['lm_head'])
```

That is it. That is one decoder pass of a generative pre-trained transformer. **One token in, one logit vector out.** Everything you have ever heard about LLMs — context windows, attention heads, KV cache, residual stream, MLP blocks, embedding tables — is in there, in exactly one place each, with names short enough to fit on the page.

The two helpers that the listing leans on are even smaller:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(val - max_val) for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def relu(x_val):
    return max(0.0, x_val)
```

The hyperparameters of the toy model — the ones you would normally see as 32, 4096, 8192, 32 in a frontier model — are scaled down so each tensor is a list you could write out by hand:

```python
n_layer    = 2     # depth (number of transformer blocks)
n_embd     = 16    # width (embedding dimension)
block_size = 16    # max context length
n_head     = 4     # number of attention heads
head_dim   = 4     # = n_embd / n_head
```

That is the entire model. Two layers, sixteen-dimensional embeddings, four heads, sixteen-token context. We are going to call this listing **microGPT** for the rest of this project. When a later issue talks about "the `attn_wk` projection" or "the K-channel outliers" or "the residual stream after the MLP block" — *this* is the file those phrases point at.

## What This Code Is Actually Doing

Read the listing once more, but this time skim it like prose. There are exactly seven movements.

> **1. Look up the token.** Every input is a single integer `token_id`. We pull the row at that index from a table `wte` (the **w**ord-**t**oken **e**mbedding) and get back a vector of `n_embd` floats. This is the first time the model "knows" what character it just saw.

> **2. Look up the position.** Same trick, different table. `wpe` (**w**ord-**p**osition **e**mbedding) is indexed by *where* in the sequence we are, not *what* we are. We add the two vectors element-wise. The model now knows what *and* where.

> **3. Normalize.** We rescale the vector to unit-ish magnitude using `rmsnorm`. This is glue. It keeps the magnitudes from exploding as we cascade through layers. We will do it before every sub-block from here on.

> **4. Run `n_layer` transformer blocks.** Each block has two halves: an attention half and an MLP half. Both halves share the same shape: *normalize → do something → add the result back to the input*. The "do something" is what makes the layer interesting.

> **5. In the attention half:** Project `x` three different ways into a query `q`, a key `k`, and a value `v`. Stash `k` and `v` into the KV cache for this layer. Then, for each of the `n_head` heads, slice `q`, `k`, and `v` into shorter chunks; compute a similarity score between this token's query and every previous token's key; turn the scores into a probability distribution with `softmax`; use those probabilities to weight-average the values. Concatenate all the heads back into one vector. Multiply by an output projection `attn_wo`. Add to the residual.

> **6. In the MLP half:** Normalize again. Multiply by a "fattening" matrix `mlp_fc1` to expand to a wider hidden dimension. Apply a nonlinearity (`relu`). Multiply by a "skinnying" matrix `mlp_fc2` to come back to width `n_embd`. Add to the residual.

> **7. Project to vocabulary.** After all the blocks, one final linear `lm_head` takes the residual vector and returns one logit per vocabulary token. The caller turns logits into probabilities, samples one, and feeds that integer back into `gpt()` next round.

There is no extra cleverness. Every "trick" of modern LLM inference — paged {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}, continuous batching, speculative decoding, RoPE, GQA — is **a localized optimization of one specific line above**. By the end of this issue, you will be able to point at the line each of those tricks is replacing.

## Where The Cache Goes

Two parameters that look strange in the function signature deserve a flag now: `keys` and `values`. They come in as arguments, and they are **mutated** inside the loop:

```python
keys[li].append(k)
values[li].append(v)
```

They are not random helper objects. They are how the model **remembers**. `keys[li]` is the list of all key vectors that have ever been computed in layer `li`, going back to the very first token of the sequence. Same for `values[li]`. Every time we run a new token through `gpt()`, we *append* its new `k` and `v` to those lists and read from the whole accumulated history when computing attention.

This is the **KV cache**. It is the single most important data structure in production LLM serving — entire papers exist about its memory layout — and in microGPT it is literally `list.append()`. We will earn the right to those papers in [chapter 13](../13-kv-cache/).

The matching driver code, in the same file, shows how the cache is used:

```python
keys = [[] for _ in range(n_layer)]
values = [[] for _ in range(n_layer)]

# Prefill: stuff the prompt through, building up the cache
for p_token in prompt_tokens:
    _ = gpt(token_id, pos_id, keys, values)
    token_id = p_token
    pos_id += 1

# Decode: generate one new token at a time
for _ in range(remaining_context):
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax([l / temperature for l in logits])
    token_id = random.choices(range(vocab_size), weights=probs)[0]
    if token_id == BOS:
        break
    sample.append(uchars[token_id])
    pos_id += 1
```

Two loops. The first walks the prompt forward and throws the resulting logits away — we are only here to populate the cache. The second loop is the *real* generation: each step calls `gpt()` once, turns the returned logits into a probability distribution, samples one token, and repeats. The split between these two loops is **prefill vs decode**, and almost every paper about LLM serving performance is about doing one of them faster than the other. That story is [chapter 14](../14-prefill-decode/).

## The Names You Should Tattoo

Before we start unpacking, lock the **vocabulary**. These names are going to show up in every other chapter of this issue, and in every other issue of this project. They are the lingua franca:

| Name in microGPT | What it is | Shape (for the toy model) |
|---|---|---|
| `wte` | token embedding table | `vocab_size × 16` |
| `wpe` | positional embedding table | `block_size × 16` = `16 × 16` |
| `attn_wq` | per-layer query projection | `16 × 16` |
| `attn_wk` | per-layer key projection | `16 × 16` |
| `attn_wv` | per-layer value projection | `16 × 16` |
| `attn_wo` | per-layer attention output projection | `16 × 16` |
| `mlp_fc1` | MLP "fattening" projection | `(4·16) × 16` = `64 × 16` |
| `mlp_fc2` | MLP "skinnying" projection | `16 × 64` |
| `lm_head` | vocab projection | `vocab_size × 16` |
| `q`, `k`, `v` | query/key/value for the current token | length 16 each |
| `q_h`, `k_h`, `v_h` | per-head slice | length 4 each |
| `attn_logits` | raw similarity scores before softmax | length = #tokens seen so far |
| `attn_weights` | softmaxed scores | length = #tokens seen so far |
| `head_out` | per-head output | length 4 |
| `x_attn` | concatenation of all `head_out` | length 16 |
| `x_residual` | the residual stream copy held aside | length 16 |
| `keys[li]`, `values[li]` | the KV cache for layer `li` | list of length-16 vectors |

Memorize these. Every later chapter assumes you know what `attn_wk` is on sight.

## Napkin Math: How Many Numbers Are In There?

Let's count parameters in the toy. Two layers, each with four `16×16` attention projections and one `16×64` plus one `64×16` MLP pair:

$$
\text{params per layer} = 4 \cdot (16 \cdot 16) + 2 \cdot (16 \cdot 64) = 1024 + 2048 = 3072
$$

Add the embeddings and head — assume `vocab_size = 27` for a model trained on lowercase letters plus BOS:

$$
\text{embeddings + head} = 27 \cdot 16 + 16 \cdot 16 + 27 \cdot 16 = 432 + 256 + 432 = 1120
$$

$$
\text{total} \approx 2 \cdot 3072 + 1120 = \mathbf{7264 \text{ parameters}}
$$

Seven thousand floats. A frontier model is fifty *billion* to one *trillion* — the same structure, repeated wider and deeper. The fact that the per-line code does not change as you scale is the entire reason this listing is useful as a study aid.

```pyplot {id="param-share" caption="Where the 7,264 parameters live in microGPT. Attention dominates per-layer, MLP dominates totally."}
labels = ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2', 'embeddings\n+ lm_head']
sizes  = [2*16*16,   2*16*16,   2*16*16,   2*16*16,   2*16*64,   2*64*16,   27*16+16*16+27*16]
colors = ['#FF007F', '#FF007F', '#FF007F', '#FF007F', '#00A8A8', '#00A8A8', '#FFD700']

fig, ax = plt.subplots(figsize=(8, 4))
y = list(range(len(labels)))
ax.barh(y, sizes, color=colors, edgecolor='#1A1A1A', linewidth=1.5)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
for i, s in enumerate(sizes):
    ax.text(s + 30, i, f'{s}', va='center', fontsize=10)
ax.set_xlabel('parameter count')
ax.set_title('microGPT parameter budget (toy: n_layer=2, n_embd=16)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

The two MLP matrices, between them, hold **56%** of the model's parameters. The four attention matrices together hold **35%**. The embedding/head triple is the remaining 9%. This is, to a first approximation, true at every scale: in a 70B-parameter model, the MLPs are still the majority of the weights, the attention projections are the second largest chunk, and the input/output tables are a rounding error.

## What This Issue Will Do

The rest of this issue is a guided walk through that 60-line listing. Every chapter takes one operation — one *line*, often — and answers three questions:

1. **What does this line do?** The mechanics, in plain English with a worked numerical example.
2. **Why is it shaped exactly this way?** The historical or mathematical reason it looks like this and not something else.
3. **Where does this idea show up in modern LLMs?** The contemporary papers, kernels, and serving tricks that are local optimizations of this exact line.

The order is chosen so that by the time you reach an article, every concept it references has already been introduced. The [tech tree on the cover page](../) is the dependency graph; the chapter numbers respect a topological sort of that graph.

When you finish the last chapter — [The Full Forward Pass](../16-full-forward/) — we will reprint the listing one more time, this time annotated, every variable and every loop bound hyperlinked to the chapter that explained it. Think of it as the index to the rest of this entire project.

*microGPT is a pedagogical derivative of **{{< wiki "karpathy" >}}Andrej Karpathy{{< /wiki >}}**'s [nanoGPT](https://github.com/karpathy/nanoGPT) — the same spirit of stripping everything to the minimum, rewritten for plain-Python clarity.*

---

**Continue to** → [The State Dict, Demystified](../02-state-dict/) — before we run any code, we'd better know what is in that `state_dict` dictionary the listing keeps fishing tensors out of.
