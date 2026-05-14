---
title: "ReLU and Friends"
description: "`max(0, x)` works. So does GELU, SiLU, and the gated SwiGLU. The differences are smaller than the marketing suggests, and the reasons each one won its decade are mostly historical accident."
topics: [transformer, activations]
tags: [microgpt, relu, gelu, swiglu]
theme: teal
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 120
techKind: primer
techNode: activations
header: default.webp
---

## The Bend In The Wire

In the spring of 2010, Xavier Glorot, Antoine Bordes, and Yoshua Bengio submitted a paper to AISTATS with a title so dry it could have come from a 1960s electrical-engineering journal: **"Deep Sparse Rectifier Neural Networks"**. The contents were anything but. They had taken a function so simple that any undergraduate would discard it as a candidate activation — $f(x) = \max(0, x)$, a hinge, a kinked wire — and shown that it trained *deep* networks faster, to lower error, and with more interpretable hidden activations than the smooth sigmoids and tanhs that had defined the field for two decades.

For people who had grown up on Yann LeCun's 1998 tanh-based LeNet, this was heresy. **The function is not differentiable at zero.** It has unbounded output. Half of its derivative is identically zero, which any reasonable textbook would warn you means "the gradient will die." The textbooks were wrong. By 2012 a rectified-linear AlexNet had won ImageNet and the rest of the decade was a rout. Geoff Hinton would later call it "the activation function that ate the world," and would tell interviewers, half-joking, that it took him "an embarrassingly long time to be willing to try `max(0, x)`."

If you `grep` microGPT for activation functions, this is what you find — the entire menagerie, four lines long:

```python
def relu(x_val):
    return max(0.0, x_val)
```

That's it. One comparison. No exponentials. No lookup table. No fused kernel. In the [cold open's listing](../01-cold-open/), `relu` shows up exactly once, sandwiched between the two MLP projections:

```python
# MLP block
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # fatten:  16  → 64
x = [relu(xi) for xi in x]                        # bend
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # skinny:  64  → 16
```

That kink in the middle is the entire reason the [MLP block](../11-mlp-block/) is more than an expensive identity function. Without it, the two `linear` calls would collapse into one. With it, you have, in theory, a universal function approximator. This is a primer about that kink, the family of close cousins that have tried to replace it, and the surprising fact that for the *bulk* of LLM behaviour, **it almost doesn't matter which cousin you pick**.

## Why The Bend Is Mandatory

Strip the `relu` line out of the MLP block. What's left?

$$
x \mapsto W_2 (W_1 x) = (W_2 W_1) x
$$

Two matrix multiplies in sequence are *the same operation* as one matrix multiply with the product $W_2 W_1$. The "depth" you so carefully built is illusion. Worse, this collapses *all the way through the model*: every transformer block becomes equivalent to one giant linear map from input embedding to output logits. A model with 80 layers and 70 billion parameters degenerates, in expressive power, to a single $4096 \times 4096$ matrix.

This is the **linearity trap**, and it is the reason every neural network in history has had a nonlinearity wedged between its linear pieces. The shape of the nonlinearity is, in some sense, optional. *Having one at all* is not.

The minimum bar a candidate must clear:

1. **It must not be a linear function.** A bend somewhere.
2. **You must be able to differentiate it (almost everywhere) and the gradients shouldn't all be zero or all infinity.** Otherwise training stalls.
3. **It should be cheap.** This thing runs once per element of a 4096-wide hidden vector, per layer, per token. Two hundred billion times in a forward pass of a frontier model.

ReLU clears all three. So does every cousin we're about to meet.

## The Cast

Here are the five candidates that have, at one point or another, defined the state of the art:

| Name | Formula | Year on the throne |
|---|---|---|
| **Sigmoid** | $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ | 1986–2010 |
| **tanh** | $\tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$ | 1998–2010 |
| **ReLU** | $\max(0, x)$ | 2010–2018 |
| **GELU** | $x \cdot \Phi(x)$, $\Phi$ = standard normal CDF | 2018–2022 (GPT-2, BERT) |
| **SiLU / Swish** | $x \cdot \sigma(x)$ | 2017– (Llama, Mistral) |

Let's see them all on the same axes.

```pyplot {id="five-activations" caption="Five activation functions on the same axes. Notice how they all 'pass through' something like ReLU in shape, just with varying degrees of smoothness."}
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def tanh_f(x):  return np.tanh(x)
def relu(x):    return np.maximum(0.0, x)
def gelu(x):
    # x * Phi(x), Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    # erf via the Abramowitz & Stegun 7.1.26 polynomial approximation (numpy-only)
    sign = np.sign(x)
    ax = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    erf = sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * np.exp(-ax*ax))
    return x * 0.5 * (1.0 + erf)
def silu(x):    return x * sigmoid(x)

xs = np.linspace(-5, 5, 400)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, color='#1A1A1A', lw=0.8)
ax.axvline(0, color='#1A1A1A', lw=0.8)
ax.plot(xs, sigmoid(xs), color='#FFD700', lw=2.5, label='sigmoid')
ax.plot(xs, tanh_f(xs),  color='#FF8C00', lw=2.5, label='tanh')
ax.plot(xs, relu(xs),    color='#FF007F', lw=3,   label='ReLU')
ax.plot(xs, gelu(xs),    color='#00A8A8', lw=2.5, label='GELU')
ax.plot(xs, silu(xs),    color='#1A1A1A', lw=2.5, ls='--', label='SiLU / Swish')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('Five activation functions, $x \\in [-5, 5]$')
ax.legend(loc='upper left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

Two stories jump out of this plot.

**Story 1: sigmoid and tanh saturate.** Drift more than three units away from zero in either direction and they're stuck against a horizontal asymptote. The output stops changing. In a deep network this means the *gradient* coming back through them stops changing too — backprop multiplies derivatives, and a chain of `0.0001 × 0.0001 × ...` evaporates before it can update the early layers. This is the **vanishing gradient problem**, and it is the reason that, throughout the late 90s and early 2000s, networks deeper than five or six layers were notoriously hard to train.

**Story 2: ReLU, GELU, SiLU all look almost the same for $x > 1$.** They are linear-ish on the positive half-line. The differences are entirely in the *negative half* and the *transition region near zero*: ReLU clips hard to zero, GELU dips slightly negative and comes back up, SiLU follows the same trajectory more smoothly. None of them saturates. Their derivatives stay alive.

```pyplot {id="five-derivatives" caption="Derivatives of the same five functions. Sigmoid and tanh peak at the center then collapse to zero — the textbook 'vanishing gradient.' ReLU's derivative is the Heaviside step; GELU and SiLU smooth it out so backprop has a continuous signal everywhere."}
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def gelu(x):
    sign = np.sign(x)
    ax_ = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * ax_)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    erf = sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * np.exp(-ax_*ax_))
    return x * 0.5 * (1.0 + erf)

def d_sigmoid(x):
    s = sigmoid(x); return s * (1.0 - s)
def d_tanh(x):
    return 1.0 - np.tanh(x)**2
def d_relu(x):
    return (x > 0).astype(float)
def d_gelu(x, h=1e-3):
    return (gelu(x + h) - gelu(x - h)) / (2 * h)   # numerical, plenty for a plot
def d_silu(x):
    s = sigmoid(x); return s + x * s * (1.0 - s)

xs = np.linspace(-5, 5, 400)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, color='#1A1A1A', lw=0.8)
ax.axvline(0, color='#1A1A1A', lw=0.8)
ax.plot(xs, d_sigmoid(xs), color='#FFD700', lw=2.5, label="sigmoid'")
ax.plot(xs, d_tanh(xs),    color='#FF8C00', lw=2.5, label="tanh'")
ax.plot(xs, d_relu(xs),    color='#FF007F', lw=3,   label="ReLU' (Heaviside)")
ax.plot(xs, d_gelu(xs),    color='#00A8A8', lw=2.5, label="GELU'")
ax.plot(xs, d_silu(xs),    color='#1A1A1A', lw=2.5, ls='--', label="SiLU'")
ax.set_xlabel('x'); ax.set_ylabel("f'(x)")
ax.set_title('Derivatives — the actual signal that flows back during training')
ax.legend(loc='upper left', frameon=False)
ax.set_ylim(-0.2, 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

Look hard at the orange and yellow curves. Past $|x| > 3$ they are pinned at zero. That is the vanishing gradient, drawn from first principles. The pink, teal, and dashed curves all stay above zero everywhere — the gradient signal survives a hundred-layer chain.

ReLU's derivative is the cleanest of them all: it is the **Heaviside step**, $1$ for $x > 0$ and $0$ for $x < 0$. During backprop you multiply by 1 or you don't multiply at all. It's the cheapest possible gating mechanism, which is also why ReLU runs at memory-bandwidth speeds on every piece of silicon ever built.

## The Dead-Neuron Caveat

The Heaviside derivative has a dark side. If, during training, a particular hidden unit ends up with weights that push its pre-activation perpetually negative *for every input in the training set*, then ReLU outputs zero for that unit, the gradient through it is zero, and **it never updates again**. The unit is dead. Its weights never move. It is, for the rest of training, a paperweight.

This is the **dying ReLU problem**, and it's the reason all the modern "ReLU-flavored" variants — Leaky ReLU, GELU, SiLU — give the negative half-line a little wiggle room rather than zeroing it out. GELU does this by multiplying $x$ with the normal CDF $\Phi(x)$, which is small but nonzero for moderately negative inputs:

$$
\text{GELU}(x) = x \cdot \Phi(x), \qquad \Phi(x) = \frac{1}{2}\!\left[1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

SiLU (a.k.a. **Swish**, discovered by a neural architecture search at Google in 2017, then independently rediscovered and renamed by everyone for two years) does the same thing with the sigmoid playing the role of $\Phi$:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

These two functions are nearly identical in shape — both pass through the origin, both dip slightly negative around $x = -1$, both straighten to $f(x) \approx x$ for large positive $x$. The choice between them in practice is largely a matter of which one your framework's kernel library has optimized.

## SwiGLU: The Plot Twist

In 2020, Noam Shazeer published a one-trick-pony paper titled simply **"GLU Variants Improve Transformer"**. It was three pages. It contained no theory. It just methodically swapped the MLP's activation function for a family of **Gated Linear Units** and reported the perplexity numbers. One of them, **SwiGLU**, beat plain Swish by a fraction of a percent and has since become the default activation in Llama, Mistral, PaLM, Gemma, and essentially every frontier open-weight model trained after 2022.

SwiGLU is not a single function of one variable. It's a **structure**. The MLP block changes shape:

$$
\text{MLP}_{\text{SwiGLU}}(x) = \big(\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big) W_{\text{down}}
$$

Two projections from $x$ in parallel. One gets Swish'd; the other passes through linearly. They're multiplied **element-wise** ($\odot$). The result goes through a third projection back to the residual width. The state dict for a Llama MLP layer therefore has **three** matrices instead of microGPT's two: `mlp_fc_gate`, `mlp_fc_up`, `mlp_fc_down`. ([Chapter 11](../11-mlp-block/) does the parameter accounting in full.)

Why does this help? The intuition: SwiGLU lets the network **learn its own gate**. For each hidden unit, one half-projection decides "should this dimension be on for this input?" and multiplies the other half by that decision. ReLU's gate is hardcoded ($x > 0$); SwiGLU's gate is learned per-dimension and per-input. It is the same idea as the input/forget gates of an LSTM, ported to feedforward layers.

The cost: one extra matrix. To keep the parameter count constant against a vanilla GLU-free MLP, frontier models shrink the hidden width from $4 \cdot n_{\text{embd}}$ to about $\tfrac{8}{3} \cdot n_{\text{embd}}$. In Llama 3 8B, that's $4096 \to 14336$ instead of $4096 \to 16384$. The compute budget stays roughly fixed; the *shape* of what the MLP can express changes.

## Napkin Math: How Expensive Is Each Bend?

Per-element costs of computing the activation itself (ignoring the linear projections around it):

| Activation | Operations per element |
|---|---|
| ReLU | 1 comparison |
| Sigmoid / tanh | 1 exponential, 1 division |
| GELU | 1 exponential + small polynomial (erf approx), ~5 multiplies, 1 division |
| SiLU | 1 exponential, 1 division, 1 multiply |
| SwiGLU (per output element) | 1 SiLU + 1 multiply + the parameter cost of one extra matrix |

On a modern GPU, all of these are **memory-bound** for any reasonable hidden width — meaning the cost of `relu` vs `gelu` is dominated by the time to read the input vector from VRAM and write the output back, not the arithmetic. Switching from ReLU to GELU costs maybe 5% of MLP wall-clock time, which is itself ~30% of the layer, which is itself ~3% of the model. The user-visible difference: a fraction of a percent.

The *parameter* cost of SwiGLU, however, is real and load-bearing: 1.5× the MLP parameters at constant width, or equivalently a 2/3-width hidden if you hold parameters fixed. This is why moving to SwiGLU shows up as a budget line in model cards.

## Why The Differences Mostly Don't Matter

Here is the slightly demoralizing truth that took the field a decade to internalize: **the difference in final task performance between ReLU, GELU, SiLU, and even SwiGLU is, on most benchmarks, within the noise of a single training run**.

A handful of ablation studies on small-to-medium LLMs find SwiGLU ahead by 0.1-0.3 perplexity points on Wikipedia validation sets. Some find GELU ahead. Some find no difference at all. The signal is real but tiny, and it's almost entirely dominated by other architectural choices — pre-norm vs post-norm, RMSNorm vs LayerNorm, where exactly the residual sums happen, the learning-rate schedule.

The modern wisdom — which is **wisdom, not law** — is to pair **SwiGLU with [RMSNorm](../05-rmsnorm/) in a pre-norm configuration**. Llama and Mistral both do this. It is not because there is a proof that this triple is optimal. It is because someone trained a useful model with this recipe, the recipe was open-sourced, and everyone else copied it.

microGPT, being a teaching artifact, uses plain ReLU because plain ReLU is one line. **If you replaced `relu` with `gelu` or `silu` in the [cold open's listing](../01-cold-open/), the model would be roughly equally capable.** If you replaced the MLP block with a SwiGLU structure, you'd gain a tiny edge at the cost of a third state-dict entry per layer. The kink is mandatory. The exact shape of the kink is taste.

## What To Remember

1. **Without a nonlinearity, the entire transformer collapses into one matrix multiply.** The bend in the wire is the only reason "deep" means anything.
2. **ReLU = `max(0, x)`.** One comparison. Sparse output (about half the units are dead per input). The 2010 breakthrough that killed sigmoid/tanh.
3. **GELU and SiLU are smoothed cousins of ReLU.** They fix the dying-neuron problem by giving the negative half-line a small nonzero output. Used in GPT-2/BERT and Llama/Mistral respectively.
4. **SwiGLU is a structural change**, not a new function. It adds a third projection and uses element-wise gating: $(\text{Swish}(xW_g) \odot xW_u) W_d$. 1.5× the MLP parameters; modern frontier default.
5. **The differences between modern activations are real but small** — within the noise on most benchmarks. The choice is usually driven by ecosystem compatibility, not theoretical superiority.
6. **In microGPT, `relu` lives between [`mlp_fc1`](../11-mlp-block/) and [`mlp_fc2`](../11-mlp-block/)**, executed once per element. The single line `x = [relu(xi) for xi in x]` is, mechanically, the entire activation discussion compressed.

---

**Continue to** → [The KV Cache](../13-kv-cache/) — every token we've ever seen leaves a trail of keys and values in a growing pair of lists; let's finally pin down what that data structure looks like, why it exists, and what production serving stacks do to keep it from eating their VRAM.

