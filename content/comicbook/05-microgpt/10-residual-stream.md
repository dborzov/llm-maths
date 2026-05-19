---
title: "Residual Stream: every block reads, then writes back"
short_title: "Residual Stream"
description: "Two `x = [a + b for ...]` lines per transformer block — the entire network is a sequence of read-modify-write operations on a single persistent vector."
blurb:
  - "Kaiming He's 2015 insight: a 30-layer network was *worse on the training set* than the 18-layer one. The fix was one plus sign."
  - "The pattern: save → normalize → compute → add back. Repeated twice per layer. The variable `x_residual` is reused on purpose."
  - "The 16-dimensional `x` vector in the toy model is the residual stream. Every sub-block adds a delta to it — nothing replaces it."
  - "Interpretability research calls this the 'residual stream.' Why does naming the vector matter for understanding what heads are doing?"
topics: [transformer, interpretability]
tags: [microgpt, residual-stream, residual-connection]
theme: teal
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 100
techKind: primer
techNode: residual-stream
header: 10-residual-stream.webp
---

## A Net That Refused To Get Deeper

In late 2014, a team at Microsoft Research Asia in Beijing — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun — were doing something that, on paper, should have just worked. They wanted to train a very deep convolutional network for ImageNet. Not eight layers (AlexNet, 2012). Not nineteen (VGG, 2014). They wanted **thirty**. And then sixty. And then a hundred.

It did not just work. It broke in a way that, before then, nobody had a clean name for. The 30-layer network was *worse* on the training set than the 18-layer one. Not on the *test* set — that would be overfitting, a familiar foe. On the **training** set. The bigger model could not even memorize the data the smaller one had already memorized. Adding capacity made the optimizer dumber.

He's team called this the **degradation problem**, and in their December 2015 paper "Deep Residual Learning for Image Recognition" they proposed the simplest fix anyone has ever published in a top-tier ML venue. Take the output of a block, and instead of passing it forward as-is, **add it to the block's own input**:

$$
y = F(x) + x
$$

That is it. A plus sign. They called the line carrying $x$ forward a *shortcut connection*, or *skip connection*, or *residual connection*. With it in place, training a 152-layer network became routine. They won ImageNet 2015 by a humiliating margin, and within eighteen months the same idea was in every architecture worth naming — including, when Vaswani's team published *Attention Is All You Need* in 2017, the transformer.

Open microGPT. Search for the plus-sign-in-a-zip pattern. You will find it twice per layer:

```python
x = [a + b for a, b in zip(x, x_residual)]
```

That is the resnet shortcut. That is the same five-character idea, hauled into 2026 and rebranded as the **residual stream**.

## The Shape Every Sub-Block Wears

Pull up the body of microGPT's per-layer loop and stare at it for a moment. Once you know what you're looking for, *every* sub-block has the same four-line skeleton:

```python
x_residual = x
x = rmsnorm(x)
... (do something interesting) ...
x = [a + b for a, b in zip(x, x_residual)]
```

Save. Normalize. Compute. Add back. Save, normalize, compute, add back. The "do something interesting" in the first half is multi-head {{< wiki "attention" >}}attention{{< /wiki >}}; in the second half it is the {{< wiki "mlp-block" >}}MLP{{< /wiki >}}. The wrapper is identical:

| Step                | Code                                              | What it does                          |
|---------------------|---------------------------------------------------|---------------------------------------|
| **save**            | `x_residual = x`                                  | take a snapshot of the current state  |
| **normalize**       | `x = rmsnorm(x)`                                  | scale to unit-ish magnitude (see [ch.5 rmsnorm](../05-rmsnorm/)) |
| **compute**         | attention or MLP                                  | produce a length-`n_embd` *contribution* |
| **add back**        | `x = [a + b for a, b in zip(x, x_residual)]`      | overwrite `x` with `x_residual + contribution` |

The variable name `x_residual` is **reused twice** inside one layer, and that is *not* a copy-paste bug. It is a small, deliberate piece of code-as-documentation: each time you see `x_residual = x` you should hear a voice in your head say *"snapshot taken; the next few lines are going to compute a delta against this."* The variable's job is exactly one transformer sub-block long.

> A reader who has never written down the pattern will look at this code and think "we keep overwriting `x`, so the original embedding is *lost* by layer 2." Wrong. Every overwrite is an **addition** to the original, mediated by a small additive contribution from a sub-block. The original embedding is *literally still in there*, summed against everything the layers have written on top of it.

## The Residual Stream As A Highway

Here is the metaphor that, once you have it, makes a lot of modern interpretability research click. Imagine the per-token vector `x` not as "the activation at layer $\ell$" but as a **lane on a highway** that runs the full depth of the network, from the embedding lookup to the `lm_head`:

```
   [ wte + wpe ]  ─► rmsnorm ─►  +══►  +══►  +══►  +══► ... ─► lm_head
                                 ▲     ▲     ▲     ▲
                                 │     │     │     │
                             attn_0  mlp_0  attn_1  mlp_1
```

The double-line is `x`, the residual stream. The single lines coming in from below are the **contributions** each sub-block writes. Nothing in the architecture forces any sub-block to write a big contribution — it can write nearly zero and the highway carries the previous value through almost untouched. Equivalently, nothing forces a sub-block to write a *small* contribution — it can completely overwhelm the previous value if it wants to. The model learns, during training, how loudly each sub-block should speak.

This framing is the central abstraction in Anthropic's [Transformer Circuits](https://transformer-circuits.pub/) line of work (Elhage et al., 2021). They were trying to reverse-engineer what attention heads actually do, and the move that unlocked everything was to stop thinking of `x` as "the input to layer $\ell$" and start thinking of it as a **shared memory bus** that every head reads from and writes to:

> "We think of the residual stream as the *medium of communication* between layers. Heads read information from the residual stream by applying their `wq` and `wk` projections, do their computation, and then write back via `wo`. The next head reads what the previous head wrote."

In microGPT terms, "reads" means a `linear(x, ...)` call early in a sub-block, and "writes" means the final `[a + b for a, b in zip(...)]` at the end of the sub-block. Read, think, write. Read, think, write.

## A Picture Of The Stream

Let's actually watch a residual stream evolve. We'll simulate a 16-dimensional `x` for a single token, walking through 12 sub-blocks (six attention + six MLP, alternating). Each sub-block writes a small additive contribution. The picture below tracks each of the 16 coordinates as a separate line going down through the network.

```pyplot {id="residual-evolution" caption="Sixteen coordinates of a residual stream as a token passes through twelve sub-blocks. Each sub-block writes a small additive perturbation; the original embedding's fingerprint persists all the way to the head."}
rng = np.random.default_rng(7)

n_embd = 16
n_sub  = 12

# Initial embedding: each coord starts at some random value
x = rng.normal(0, 1.0, size=n_embd)
history = [x.copy()]

# Each sub-block: small additive contribution, magnitude shrinks slightly with depth
for s in range(n_sub):
    contribution = rng.normal(0, 0.35, size=n_embd)
    x = x + contribution
    history.append(x.copy())

history = np.array(history)  # shape (n_sub + 1, n_embd)

fig, ax = plt.subplots(figsize=(9, 4.5))
for d in range(n_embd):
    ax.plot(range(n_sub + 1), history[:, d], color='#00A8A8', alpha=0.55, linewidth=1.4)

# highlight one coordinate to show how it drifts
ax.plot(range(n_sub + 1), history[:, 3], color='#FF007F', linewidth=2.4, label='coord 3 (highlighted)')

ax.axhline(0, color='#1A1A1A', linewidth=0.6, linestyle='--', alpha=0.5)
ax.set_xticks(range(n_sub + 1))
ax.set_xticklabels(['embed'] + [f'sb{i}' for i in range(n_sub)], rotation=45, fontsize=8)
ax.set_xlabel('sub-block index (attn / mlp interleaved)')
ax.set_ylabel('coordinate value')
ax.set_title('one residual stream, evolving through depth')
ax.legend(loc='upper left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

Two things to notice. **One:** every line wanders, but no line snaps. Each sub-block adds a small kick; the trajectory of any single coordinate is a near-continuous walk. **Two:** the *initial* value at `sb0` (the embedding) is correlated with the *final* value at `sb11` — the highlighted pink coordinate ends roughly where it started plus the cumulative sum of twelve small contributions. The embedding is not erased. It is **buried under contributions**, like a base coat of paint under twelve glazes.

## The Backward Pass Loves The Plus Sign

There is a second, equally important reason He et al. won ImageNet, and it has nothing to do with interpretability. It has to do with **gradients**.

When you backpropagate through $y = F(x) + x$, the chain rule gives:

$$
\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + 1
$$

The **plus one** is the magic. In a network without residual connections, the gradient of the loss with respect to early-layer parameters is a product of many Jacobians, and if any of them is small in some direction, that direction's gradient vanishes by the time it reaches the input. With residual connections, **every layer's Jacobian gets a +1 baseline**, so the gradient signal always has at least an identity path back to the embedding. Vanishing gradients still happen, but the rate is dramatically reduced.

We can show this with a numpy simulation. Pretend each sub-block's Jacobian is a random matrix with entries roughly of size $1/\sqrt{n}$ (a standard assumption for "the network is initialized sensibly"). We'll compute the norm of a unit gradient signal as it backpropagates through 60 sub-blocks, with and without residual connections.

```pyplot {id="gradient-norm-depth" caption="Gradient norm vs depth for a 60-sub-block synthetic network. Without residual connections (orange), the signal collapses inside ~20 layers. With residual connections (pink), the +1 path keeps the gradient alive all the way to the embedding."}
rng = np.random.default_rng(42)

n_embd = 64
n_sub  = 60
trials = 8

def jacobian():
    # Random sub-block Jacobian, scaled so a vanilla network is on the edge of stability
    return rng.normal(0, 1.0 / np.sqrt(n_embd), size=(n_embd, n_embd))

norms_plain    = np.zeros((trials, n_sub + 1))
norms_residual = np.zeros((trials, n_sub + 1))

for t in range(trials):
    g_plain    = rng.normal(0, 1.0, size=n_embd); g_plain /= np.linalg.norm(g_plain)
    g_residual = g_plain.copy()
    norms_plain[t, 0]    = np.linalg.norm(g_plain)
    norms_residual[t, 0] = np.linalg.norm(g_residual)
    for s in range(n_sub):
        J = jacobian()
        # Plain: gradient gets multiplied by J at each layer
        g_plain    = J @ g_plain
        # Residual: gradient gets multiplied by (J + I) at each layer
        g_residual = (J + np.eye(n_embd)) @ g_residual
        norms_plain[t, s+1]    = np.linalg.norm(g_plain)
        norms_residual[t, s+1] = np.linalg.norm(g_residual)

fig, ax = plt.subplots(figsize=(9, 4.5))
mean_plain    = norms_plain.mean(axis=0)
mean_residual = norms_residual.mean(axis=0)

ax.semilogy(range(n_sub + 1), mean_plain,    color='#FF8C00', linewidth=2.2,
            label='no residual: gradient = J · g')
ax.semilogy(range(n_sub + 1), mean_residual, color='#FF007F', linewidth=2.2,
            label='with residual: gradient = (J + I) · g')

for t in range(trials):
    ax.semilogy(range(n_sub + 1), norms_plain[t],    color='#FF8C00', alpha=0.15, linewidth=0.8)
    ax.semilogy(range(n_sub + 1), norms_residual[t], color='#FF007F', alpha=0.15, linewidth=0.8)

ax.axhline(1.0, color='#1A1A1A', linewidth=0.6, linestyle='--', alpha=0.5)
ax.set_xlabel('sub-block depth (backward direction)')
ax.set_ylabel('||gradient|| (log scale)')
ax.set_title('residual connections keep gradients alive through depth')
ax.legend(loc='lower left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

Without the residual, the gradient norm collapses by many orders of magnitude before it reaches the embedding — there is nothing for the optimizer to chew on at the bottom of the network, and the early layers fail to train. With the residual, the `+I` term keeps the gradient norm bounded near 1.0 across the whole depth. Same number of parameters; vastly easier optimization.

This is *why* you can stack 96 transformer layers in GPT-3 and still have it train. Not because the architecture is magical, but because every one of those 192 sub-blocks (96 attention + 96 MLP) has a `+ x_residual` line that gives gradients a fast lane home.

## A microGPT Quirk: The First Sub-Block Has No Snapshot

Reread the top of microGPT's forward pass:

```python
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
for li in range(n_layer):
    x_residual = x
    ...
```

Notice anything? **There is no `x_residual = ...` before the very first `rmsnorm(x)`.** That normalization happens *to* the embedding sum, not as part of a save-normalize-compute-add-back cycle. It is preparation: it gives the highway a sensible starting magnitude before the first sub-block reads from it. The per-layer pattern only kicks in once we enter the `for li in range(n_layer)` loop.

This is a stylistic choice — some transformers omit that pre-loop `rmsnorm` and instead put the first `rmsnorm` inside the first attention sub-block, where it then *also* sees the residual. Either works. microGPT picked the version that reads more cleanly: **embed, prep, then loop**.

## Napkin Math: Where Does The Stream Live?

For our toy with `n_embd = 16` and `n_layer = 2`, the residual stream is a single length-16 vector that gets *re-derived* sixteen times in a forward pass: once after the initial `rmsnorm`, then twice per layer (after attention, after MLP), times two layers, plus the post-loop value that `lm_head` reads. Each rederivation requires holding `x_residual` in memory for as long as the sub-block runs — that's 16 floats, or 64 bytes at fp32. Pocket change.

Now scale up. Llama 3 8B has `n_embd = 4096` and `n_layer = 32`. The residual stream is a length-4096 vector. There are `1 + 2·32 = 65` rederivations per token, each requiring an `x_residual` snapshot of $4096 \cdot 2 = 8192$ bytes at fp16. During *training* — when you have to keep every intermediate around for the backward pass — those snapshots become a significant chunk of activation memory. Activation checkpointing is, in part, the art of deciding which residual snapshots are worth keeping vs. recomputing.

For inference at one token at a time, it's still pocket change. The residual stream is a couple of kilobytes per token per layer. The KV cache that we'll meet next door in [ch.13](../13-kv-cache/) — that one will be measured in gigabytes.

## What To Remember

1. **Two `[a + b for a, b in zip(x, x_residual)]` lines per layer.** One after multi-head attention, one after MLP. Same pattern: save snapshot, normalize, compute contribution, add back.
2. **`x_residual` is reused twice on purpose.** It is not a variable; it is a *role*. Whatever value `x` had at the top of a sub-block, hold it aside so we can add to it later.
3. **The residual stream is the model's shared memory.** Every sub-block reads from it (via `linear(x, attn_wq)` etc., after a `rmsnorm`) and writes to it (via the final `+ x_residual` line). Interpretability research lives or dies on this framing.
4. **The plus sign rescues gradients.** Without residual connections, deep networks can't train — gradients collapse exponentially with depth. With them, every Jacobian gets a `+I` baseline. He et al. 2015's one-line trick is the reason 100-layer networks exist.
5. **Information flows from embedding to `lm_head` along the stream, unobstructed.** No sub-block can *block* a feature from reaching later layers — it can only add or subtract from what is already there.

---

**Continue to** → [The MLP Block](../11-mlp-block/) — now that you know the residual stream is the highway, the next question is what those big `mlp_fc1` and `mlp_fc2` matrices actually *write* into it.
