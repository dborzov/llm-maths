---
title: "RMSNorm, Not LayerNorm"
description: "Why every modern LLM dropped LayerNorm in favor of a simpler cousin. The arithmetic difference is a single subtraction — but it costs a third of the normalization budget and nobody could tell the difference downstream."
topics: [transformer, normalization]
tags: [microgpt, rmsnorm, layernorm]
theme: cream
math: true
draft: false
date: 2026-05-14T02:48:52-04:00
issue: 5
weight: 50
techKind: primer
techNode: rmsnorm
header: default.webp
---

## Edinburgh, 2019

In the autumn of 2019, two researchers at the University of Edinburgh — **Biao Zhang** and **Rico Sennrich** — were trying to make Transformer training cheaper. Not faster on a GPU; cheaper *in operations*. They had been staring at the LayerNorm block that sat in front of every sub-layer of every Transformer in production, and the equation kept bothering them. It computed a mean. It subtracted that mean. It computed a variance from the centered values. It rescaled. It multiplied by a learned gain. It added a learned bias. **Five passes over the same `n_embd`-long vector** for what was supposed to be a "lightweight" normalization.

Zhang and Sennrich tried the most embarrassing thing a researcher can try: they **deleted the parts they could not justify**. They threw out the mean subtraction. They threw out the bias. They kept exactly one thing — divide by the root-mean-square of the entries — and called the result **Root Mean Square Layer Normalization** (RMSNorm). They ran it across machine translation, language modeling, and image classification benchmarks. The numbers came back: **the same accuracy, 7–64% faster, depending on workload**.

The paper landed at NeurIPS 2019 and was, for three years, a curiosity. Then in 2022 Google's **T5** quietly used it. In early 2023 **Llama 1**'s release notes listed three modifications from the vanilla Transformer — *"we use the RMSNorm normalising function"* sat at the top of that list, between *pre-normalization* and *SwiGLU*. By the end of 2024, essentially every open-weight frontier model — Llama, Mistral, Gemma, Qwen, DeepSeek, Phi — had quietly switched. **The most boring possible normalization layer had eaten the world.**

This chapter is about the one line of microGPT that does this work:

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

Three lines. No mean. No bias. No learnable gain (in microGPT — real LLMs add one back). It is called from exactly three places in the [cold-open listing](../01-cold-open/): once after the embedding addition at the top, once before each attention sub-block, once before each MLP sub-block. Pull on this thread and a surprising amount of modern Transformer practice unspools.

## LayerNorm: The Thing Being Replaced

To see what RMSNorm is doing, hold the older operation next to it. The original Transformer (Vaswani et al., 2017) put **LayerNorm** (Ba, Kiros & Hinton, 2016) in front of every sub-block:

$$
\text{LayerNorm}(x)_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

where

$$
\mu = \frac{1}{d}\sum_j x_j, \qquad \sigma^2 = \frac{1}{d}\sum_j (x_j - \mu)^2.
$$

In plain English: **center, scale, then learn a per-feature multiply and add**. The `γ` (gamma) is a length-`d` learned gain. The `β` (beta) is a length-`d` learned bias. Both are stored in the state dict — that's 2·`n_embd` parameters per LayerNorm call, of which a 32-layer model has 2·32 + 1 = 65 instances. In Llama 3 8B's geometry that's 65 · 2 · 4096 ≈ **530K parameters** spent on normalization, if you used LayerNorm. Not the bulk of the model, but not nothing.

Now RMSNorm:

$$
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}}
$$

What disappeared? **The mean subtraction. The bias `β`.** And, in microGPT's didactic version, the gain `γ` too. (Real-world RMSNorm — the one in Llama and friends — keeps `γ`, written as `weight` in the state dict; it just drops `β`. We will come back to this in a minute.)

Side by side, counting the passes over the vector:

| Step | LayerNorm | RMSNorm |
|---|---|---|
| Compute mean μ | yes | **no** |
| Subtract mean | yes | **no** |
| Compute variance / RMS | yes (variance) | yes (mean of squares) |
| Divide | yes | yes |
| Multiply by γ | yes | yes (in real LLMs) |
| Add β | yes | **no** |

Two operations vanish on every call. That is the whole content of the paper.

## What microGPT Actually Computes

Re-read the three-line helper. Translate it into the math notation we just wrote.

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)     # ms = (1/d) Σ xᵢ²
    scale = (ms + 1e-5) ** -0.5                # scale = 1 / sqrt(ms + ε)
    return [xi * scale for xi in x]            # yᵢ = xᵢ · scale
```

For our toy with `n_embd = 16`, this is sixteen multiplies (squaring), fifteen adds (summing), one divide, one square root, and sixteen more multiplies. Roughly **`2d` FLOPs**, where `d = n_embd`. LayerNorm needs **`3d` FLOPs** for the same job (it makes an extra full pass to subtract the mean before computing variance). That `3d → 2d` is where the "30% faster" headline comes from — though once you account for kernel launch overhead, memory bandwidth, and the rest of the layer, real-world wall-clock speedup is more like 5–15%.

A worked example. Suppose `x = [3, -1, 4, 0]` (`d = 4`). Then:

$$
\text{ms} = \tfrac{1}{4}(9 + 1 + 16 + 0) = 6.5, \quad \text{scale} = \tfrac{1}{\sqrt{6.5}} \approx 0.392
$$

$$
\text{RMSNorm}(x) \approx [1.177,\ -0.392,\ 1.569,\ 0]
$$

Notice **the negative entries stay negative, the zero stays zero, the relative magnitudes are preserved**. RMSNorm is just a rescaling — it pulls the vector onto a sphere of fixed radius `√d` without rotating it. LayerNorm does the same thing *after first translating to the origin*. That translation is what RMSNorm skips.

## Why Skipping The Mean Works At All

This is the central puzzle. LayerNorm subtracts the mean because, in 2016, that felt obviously right — "normalize" had meant "zero mean, unit variance" since the era of feature scaling for SVMs. Why does the Transformer not care?

Two observations, layered on top of each other.

**One.** Inside a well-trained Transformer with **pre-norm** (more on that in a moment) and residual connections, the activations sitting on the residual stream are *already approximately mean-zero*. The architecture has a natural symmetry — every contribution to the residual comes from a `linear()` whose weights, at initialization and after Adam-style optimization, are symmetric around zero. So subtracting μ ≈ 0 is doing nothing useful. You are paying for an op that the optimizer has already arranged to be free.

**Two.** Even when μ ≠ 0, the *downstream* operation — the next `linear(x, W)` — is itself shift-invariant up to a learned bias. A constant DC offset added to `x` becomes a constant offset added to every `Wx + b`, and any frontier model's training process can absorb that offset into the bias of the next layer (or, since modern models have no biases, into the RMSNorm's learnable gain γ). The residual stream simply does not need its mean managed externally.

Let's *see* this. The picture below feeds the same random vector to LayerNorm and to RMSNorm, then to both again with a large added DC offset. LayerNorm produces identical outputs in both cases — it eats the offset. RMSNorm produces visibly different outputs. **The question is whether that difference matters downstream**, and a decade of empirical work says it does not.

```pyplot {id="rmsnorm-vs-layernorm-dc" caption="LayerNorm is shift-invariant: adding a DC offset to its input produces an identical output. RMSNorm is not: the offset rotates the output around the origin. Empirically, this difference is absorbed by the next learned layer."}
rng = np.random.default_rng(7)
d = 32
x = rng.standard_normal(d)
offset = 3.0
x_shifted = x + offset

def layernorm(v, eps=1e-5):
    mu = v.mean()
    sigma = v.std()
    return (v - mu) / (sigma + eps)

def rmsnorm_np(v, eps=1e-5):
    rms = np.sqrt((v * v).mean() + eps)
    return v / rms

fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)

ax = axes[0]
ax.plot(layernorm(x), color='#1A1A1A', linewidth=2, label='LayerNorm(x)', marker='o', markersize=4)
ax.plot(layernorm(x_shifted), color='#FF8C00', linewidth=2, label='LayerNorm(x + 3)', linestyle='--', marker='s', markersize=4)
ax.set_title('LayerNorm: the two curves overlap exactly (mean is subtracted)')
ax.set_ylabel('output')
ax.axhline(0, color='#1A1A1A', linewidth=0.5)
ax.legend(loc='upper right', fontsize=9)
ax.spines[['top','right']].set_visible(False)

ax = axes[1]
ax.plot(rmsnorm_np(x), color='#FF007F', linewidth=2, label='RMSNorm(x)', marker='o', markersize=4)
ax.plot(rmsnorm_np(x_shifted), color='#00A8A8', linewidth=2, label='RMSNorm(x + 3)', linestyle='--', marker='s', markersize=4)
ax.set_title('RMSNorm: the DC offset shifts the whole curve upward — but the *shape* is preserved')
ax.set_xlabel('feature index')
ax.set_ylabel('output')
ax.axhline(0, color='#1A1A1A', linewidth=0.5)
ax.legend(loc='upper right', fontsize=9)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
```

The top panel is the LayerNorm guarantee: *any* DC offset is erased, and the network downstream is shielded from it. The bottom panel shows that RMSNorm makes a different bet — *trust the next learned matrix to handle constant offsets on its own*. In a network of 32 layers each containing four learned `linear()` projections (see [ch.4 linear](../04-linear/)), there is plenty of slack to absorb a DC term. Zhang and Sennrich's experiments simply confirmed empirically what the architecture already promised.

## Pre-norm vs Post-norm

There is a second, equally important wrinkle hidden in microGPT — *where* the normalization sits. Look again at the attention half of the cold-open listing:

```python
x_residual = x
x = rmsnorm(x)            # <-- norm BEFORE the sub-block
q = linear(x, state_dict[f'layer{li}.attn_wq'])
# ... attention math ...
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
x = [a + b for a, b in zip(x, x_residual)]    # <-- residual add
```

The norm runs *before* the sub-block and the residual add comes *after*. This is **pre-norm**. The 2017 original Transformer ("Attention Is All You Need") did the opposite — sub-block first, then residual add, then norm — **post-norm**. The diagram in that paper still confuses graduate students every winter.

> **Post-norm:** `y = LayerNorm(x + Sublayer(x))` — the norm is the *last* thing each block does.
>
> **Pre-norm:** `y = x + Sublayer(LayerNorm(x))` — the norm is the *first* thing each block does, and the residual is left untouched as it streams straight through.

Why the switch? **Gradient flow.** In a post-norm Transformer, the gradient of the loss with respect to the residual at layer `ℓ` has to pass *through* a LayerNorm on the way to layer `ℓ-1`. LayerNorms divide by σ — and if σ varies even a little across the depth of the network, those divisions multiply into a tower of variance-rescaling that destabilizes training. Xiong et al. (2020), in a paper titled *"On Layer Normalization in the Transformer Architecture"*, proved that pre-norm Transformers have a much better-conditioned forward Jacobian and can be trained *without learning-rate warmup*, which post-norm models could not survive.

Practically: **pre-norm Transformers are stable up to hundreds of layers**, post-norm ones top out around 12-24 before training collapses without elaborate warmup schedules. Every frontier LLM since GPT-2 has been pre-norm. microGPT is pre-norm because that is the standard.

In the listing, count the normalizations per layer: one before attention, one before MLP. **Two RMSNorms per transformer block, plus one final norm after the embeddings.** For Llama 3 8B with `n_layer = 32`, that's

$$
\underbrace{2 \cdot 32}_{\text{per-layer norms}} + \underbrace{1}_{\text{post-embedding}} + \underbrace{1}_{\text{final pre-lm\_head norm}} = \mathbf{66 \text{ RMSNorms per token}}
$$

Each acting on a 4096-dimensional vector. Each costing roughly `2 · 4096 ≈ 8K FLOPs`. Total RMSNorm budget per token: about **530K FLOPs** — compared to the ~16 GFLOPs the linear layers eat. **Normalization is a rounding error in compute** *if you implement it as a fused kernel*. The reason anyone cares about the 30% saving is that on consumer GPUs and on mobile silicon, RMSNorm fits in fewer instruction slots and frees up I/O bandwidth, both of which matter for batch-1 inference.

## The Epsilon

The `+ 1e-5` in `(ms + 1e-5) ** -0.5` is the unsung hero of every normalization layer ever written. Without it, a vector of all zeros — or any vector whose squared mean rounds to zero in float16 — would cause a divide-by-zero and produce `inf` or `NaN`, and a single `NaN` anywhere on the residual stream poisons every subsequent token.

The value `1e-5` is a convention inherited from PyTorch's `nn.LayerNorm`. Llama uses `1e-5`. Mistral uses `1e-5`. GPT-2 used `1e-5`. Some training runs use `1e-6`. The exact choice is unimportant **as long as it's small relative to the typical RMS of the residual stream (around 1.0 in a well-trained model) and large enough that `ms + ε` is representable in your training dtype** (in fp16, `ε = 1e-8` is too small — it rounds to zero, and you've reintroduced the bug). For bf16, the range of safe epsilons is much wider, which is one quiet reason every frontier lab trains in bf16 these days.

## The Learnable Gain γ — The Bit microGPT Skips

The didactic version of `rmsnorm()` in microGPT throws out the learnable gain to keep the helper a three-liner. **Real LLMs keep it.** A faithful RMSNorm forward pass is:

```python
def rmsnorm(x, weight):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale * w for xi, w in zip(x, weight)]
```

`weight` is a length-`n_embd` vector stored in the state dict, typically under keys like `model.layers.0.attention_norm.weight`. It is initialized to all ones (so the layer starts as the identity scaling) and learned during training. It does the job that `γ` did in LayerNorm, with no `β` companion.

A Llama 3 8B state dict has **two RMSNorm weight vectors per layer** — one before attention, one before MLP — plus one final norm before `lm_head`. That's $2 \cdot 32 + 1 = 65$ vectors, each of length 4096, for a grand total of **266K parameters** spent on RMSNorm gains. Negligible against the 8B-parameter total, but still real.

What does the trained `γ` *do*? Each entry of `γ` says, in effect, *"the residual feature at this channel matters this much for the next sub-block — amplify or attenuate it."* In a model that has been trained on real data, you often see a small handful of `γ` entries that are *much* larger than their neighbors — the famous **outlier features** that [issue 03 chapter 06](../../03-sixteen-numbers/06-outliers/) is devoted to. RMSNorm's per-channel gain is one of the places outliers are born.

## QK-Norm: A 2024 Wrinkle

A modern stability hack worth mentioning, even though it is not in microGPT. When training extremely deep or extremely large Transformers, the dot products `Q · K` inside each attention head can grow unstably large early in training, pushing the softmax into a saturated regime where most of the gradient vanishes. The fix, proposed by Henry et al. (2020) and adopted by Google's PaLM-2 and several frontier labs around 2023–2024, is to **apply an RMSNorm to Q and K immediately after their projections, before the dot product**:

```python
q = rmsnorm(linear(x, attn_wq))   # NEW: an extra norm
k = rmsnorm(linear(x, attn_wk))   # NEW: an extra norm
# ... rest of attention as before ...
```

This is called **QK-Norm** (or sometimes QK-LayerNorm, since it predates the RMSNorm-eats-the-world era). It is essentially free at inference time and dramatically stabilizes training at scale. We will return to it in [chapter 6: Q, K, V projections](../06-qkv-projections/).

## What To Remember

1. **RMSNorm is LayerNorm minus the mean subtraction and the bias.** Same divide-by-RMS structure, fewer arithmetic ops, no β parameter.
2. **Modern LLMs keep the learnable gain γ but drop β.** microGPT's three-line helper skips γ too, for didactic clarity — real implementations restore it.
3. **The architecture absorbs DC offsets downstream.** Subtracting the mean is unnecessary because pre-norm + residual connections + Adam-trained linear layers leave the residual stream already approximately mean-zero, and the next `linear()` can absorb any residual offset into its (learned) output.
4. **Pre-norm beats post-norm for training stability.** Modern Transformers normalize *before* each sub-block, not after. Pre-norm Transformers train without warmup; post-norm ones collapse past ~24 layers.
5. **Two RMSNorms per layer.** Plus one after the embedding addition and one before `lm_head`. For Llama 3 8B that's **66 calls per token**, but each is so cheap (~8K FLOPs at 4096 width) that they collectively cost <0.01% of the per-token compute.
6. **The `+ 1e-5` is non-negotiable.** A vector of all-zeros or a fp16 underflow would otherwise produce `NaN`s that poison every downstream token.
7. **Read [ch.4 linear](../04-linear/) first.** RMSNorm exists to keep the inputs to `linear()` well-conditioned. Without it, the dot products inside attention would drift in scale across layers, and the softmax would saturate. **RMSNorm is the glue that lets all the other lines work.**

---

**Continue to** → [Q, K, V: The Three Projections](../06-qkv-projections/) — every transformer block opens with three identical `linear()` calls fed by the same RMSNormed residual; the magic is what we ask each of those three matrices to encode.

