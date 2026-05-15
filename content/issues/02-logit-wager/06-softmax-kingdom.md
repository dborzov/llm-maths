---
title: "The Softmax Kingdom"
description: "In 1998, LeNet-5 classifies 10 handwritten digits. But sigmoid outputs don't sum to 1. The fix — softmax — is logistic regression generalized to K classes, and it still runs on Berkson's 1944 approximation."
topics: [machine-learning, deep-learning, probability]
tags: [softmax, cross-entropy, logits, pytorch, lm-head, sigmoid, numerical-stability]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 2
weight: 60
techKind: boss
techNode: softmax-kingdom
header: default.webp
---

## Bell Labs, 1998

In 1998, **Yann LeCun** and colleagues at AT&T Bell Labs shipped LeNet-5 — a convolutional neural network that classified handwritten digits from the MNIST dataset with 99.2% accuracy. It was the most impressive demonstration of neural network capability since Rosenblatt's perceptron, and it used a key ingredient that the perceptron had always lacked: the sigmoid function, the logistic function that [Joseph Berkson named](../03-berksons-gamble/) in 1944.

But LeNet-5 had ten output classes (digits 0–9), and here the sigmoid ran into a fatal flaw for multi-class problems.

Ten sigmoid outputs are ten independent probabilities. Each lives in $[0, 1]$. But if you ask "what is the probability this image is a 3?", and you ask "what is the probability this image is a 7?", and separately "what is the probability this image is a 0?", you get ten numbers that have no reason to sum to 1. They might sum to 2.7 on some input, or 0.4 on another. They are not a *probability distribution* over the ten classes. You cannot interpret them that way. You cannot sample from them. You cannot use them in a probabilistic framework.

The solution was already known from statistics. It is the natural multi-class generalization of logistic regression.

## Softmax: The Multi-Class Generalization

Given a vector of raw scores $z = [z_1, z_2, \ldots, z_K]$ — one per class — the **softmax** function produces a proper probability distribution:

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Every output is positive (because $e^{z_k} > 0$), and they sum to 1 (because we divide by the total). The function is **monotone in each $z_k$**: increasing $z_k$ increases $P(y=k)$ while decreasing all other probabilities proportionally.

*This IS logistic regression for $K > 2$.* When $K = 2$, softmax reduces to:

$$
P(y=1) = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{z_2 - z_1}} = \sigma(z_1 - z_2)
$$

Binary logistic regression is the $K=2$ special case of softmax, where the single score is the *difference* $z_1 - z_2$ between the two class logits.

## The Pre-Softmax Scores Are Called "Logits"

The raw scores $z_k$ before the softmax are called **logits** — Berkson's word from 1944.

The connection is direct. In a $K$-class softmax, the log-probability ratio between class $k$ and class $K$ (the reference class) is:

$$
\log \frac{P(y=k)}{P(y=K)} = z_k - z_K
$$

This is the log-odds of class $k$ versus class $K$ — a logit. The pre-softmax scores are the log-odds of each class against some implicit reference, exactly as Berkson's original log-odds formula described. PyTorch calls them "logits" because that is what they are.

```pyplot {id="softmax-examples" caption="Three different logit vectors and the probability distributions they produce via softmax. Notice how a large logit gap (right panel) concentrates nearly all probability on one class."}
def softmax(z):
    z = np.array(z, dtype=float)
    z = z - z.max()
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()

logit_sets = [
    np.array([1.0, 2.0, 3.0]),
    np.array([0.5, 0.5, 0.5]),
    np.array([5.0, 1.0, 0.2]),
]
labels = [['C0', 'C1', 'C2']] * 3
colors = ['#FF007F', '#00A8A8', '#FFD700']
titles = [
    "Logits: [1.0, 2.0, 3.0]",
    "Logits: [0.5, 0.5, 0.5]\n(uniform → uniform probs)",
    "Logits: [5.0, 1.0, 0.2]\n(dominant class wins big)",
]

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
x = np.arange(3)

for col, (zs, title) in enumerate(zip(logit_sets, titles)):
    probs = softmax(zs)

    # Top row: logits
    ax_top = axes[0, col]
    ax_top.bar(x, zs, color=colors, edgecolor='#1A1A1A', linewidth=2)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(['C0', 'C1', 'C2'])
    ax_top.set_title(title, fontsize=9)
    ax_top.set_ylabel("logit value" if col == 0 else "")
    for i, v in enumerate(zs):
        ax_top.text(i, v + 0.05 * (1 if v >= 0 else -1), f'{v:.1f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=9, fontweight='bold')
    ax_top.spines[['top', 'right']].set_visible(False)

    # Bottom row: probabilities
    ax_bot = axes[1, col]
    ax_bot.bar(x, probs, color=colors, edgecolor='#1A1A1A', linewidth=2)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(['C0', 'C1', 'C2'])
    ax_bot.set_ylim(0, 1.0)
    ax_bot.set_ylabel("probability" if col == 0 else "")
    for i, v in enumerate(probs):
        ax_bot.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_bot.spines[['top', 'right']].set_visible(False)

axes[0, 0].set_title("Logits (raw scores)", fontsize=9)
fig.text(0.04, 0.75, "Logits →", va='center', rotation='vertical', fontsize=10, fontweight='bold')
fig.text(0.04, 0.28, "Softmax →", va='center', rotation='vertical', fontsize=10, fontweight='bold')
plt.suptitle("Three logit vectors, three probability distributions", fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0.07, 0, 1, 0.97])

for col, zs in enumerate(logit_sets):
    probs = softmax(zs)
    print(f"Logits {zs} → probs {probs.round(3)} (sum={probs.sum():.6f})")
```

## Why PyTorch Wants Raw Logits

PyTorch's `CrossEntropyLoss` takes **logits**, not probabilities. This is not an API quirk — it is a numerical stability decision.

The cross-entropy loss for class $y$ is:

$$
\mathcal{L} = -\log P(y) = -\log \text{softmax}(z)_y = -z_y + \log \sum_j e^{z_j}
$$

If you compute this naively — apply softmax first, then take the log — you lose precision. Near the edges of the distribution, `softmax(z)[k]` can be very close to 0, and `log(very_small_number)` is numerically unstable. You get `-inf` where you should get a large but finite negative number.

The numerically stable version is the **log-sum-exp trick**:

$$
\log \sum_j e^{z_j} = \max(z) + \log \sum_j e^{z_j - \max(z)}
$$

Subtracting $\max(z)$ before exponentiating keeps all values in $(-\infty, 0]$, where `exp` is well-behaved. PyTorch computes this internally when you pass logits directly to `CrossEntropyLoss`. If you pass `softmax(logits)` and then `log`, you lose the numerical stability benefit and the gradient computation also becomes less clean.

## Why Logistic Beat Probit in Deep Learning

The [probit transform](../02-probit-transform/) uses the normal CDF; the logit uses the logistic. Statistically they are nearly equivalent. But for deep learning, the logit won on three grounds:

**1. Gradient elegance.** The sigmoid derivative is $\sigma'(z) = \sigma(z)(1-\sigma(z))$. With cross-entropy loss, the gradient through a sigmoid output neuron simplifies to $y - \hat{p}$ — just the prediction error, as we derived in [From Insects to ImageNet](../05-logistic-regression/). For the probit, the analogous gradient involves $\phi(\Phi^{-1}(p))$ — the standard normal PDF evaluated at the probit of the prediction. This is more expensive to compute and less numerically stable near $p \to 0$ or $p \to 1$.

**2. Numerical stability.** The log of the sigmoid — needed during training — simplifies as:

$$
\log \sigma(z) = -\log(1 + e^{-z}) = -\text{softplus}(-z)
$$

This is numerically stable for any $z$ via the log-sum-exp trick. The log of the normal CDF near the tails — $\log \Phi(z)$ for $z \ll 0$ — is notoriously problematic: the normal CDF goes to zero exponentially fast, and floating-point underflows to `+0` before `log` gets to run.

**3. Interpretability.** The logit has a direct meaning: log-odds. The coefficient $e^\beta$ is an odds ratio. This is a quantity that physicians, epidemiologists, and anyone who has read a medical statistics paper already understands. The probit coefficient has a meaning (a change in tolerance z-scores) but it is harder to communicate.

```pyplot {id="sigmoid-gradient-comparison" caption="Gradient magnitude through a sigmoid output vs a probit output (approximated numerically). They are similar in the center but the sigmoid gradient is computed analytically while the probit gradient requires numerical differentiation of the normal CDF — slower and less stable near the extremes."}
def _norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def _norm_cdf(x):
    _p_ = 0.2316419
    _b_ = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + _p_ * np.abs(x))
    poly = t * (_b_[0] + t*(_b_[1] + t*(_b_[2] + t*(_b_[3] + t*_b_[4]))))
    cdf_pos = 1.0 - _norm_pdf(np.abs(x)) * poly
    return np.where(x >= 0, cdf_pos, 1.0 - cdf_pos)

z = np.linspace(-4, 4, 500)
dz = 0.001

# Sigmoid gradient: analytic
sig = 1 / (1 + np.exp(-z))
grad_sig = sig * (1 - sig)

# Probit gradient: numerical derivative of Phi(z)
grad_probit = (_norm_cdf(z + dz) - _norm_cdf(z - dz)) / (2 * dz)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel 1: gradient curves
ax1.plot(z, grad_sig, color='#FF007F', linewidth=2.5, label="σ'(z) = σ(z)(1−σ(z))\n[sigmoid, analytic]")
ax1.plot(z, grad_probit, color='#00A8A8', linewidth=2, linestyle='--',
         label="Φ'(z) = φ(z)\n[probit, numerical]")
ax1.set_xlabel("z (input to output function)")
ax1.set_ylabel("gradient magnitude")
ax1.set_title("Gradient through output nonlinearity")
ax1.legend(fontsize=8)
ax1.spines[['top', 'right']].set_visible(False)

# Panel 2: ratio
ratio = grad_sig / (grad_probit + 1e-10)
ax2.plot(z, ratio, color='#FF8C00', linewidth=2)
ax2.axhline(1, color='#1A1A1A', linewidth=0.8, linestyle='--')
ax2.set_xlabel("z")
ax2.set_ylabel("σ'(z) / Φ'(z)")
ax2.set_title("Ratio of gradients\n(1.0 = identical gradient)")
ax2.set_ylim(0, 2.5)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
print("Peak sigmoid gradient:", grad_sig.max())
print("Peak probit gradient:", grad_probit.max())
```

## The 90-Year Chain

{{% pullquote %}}
When you call `model(x)` in PyTorch and get a tensor of logits, you are working with the exact mathematical object that Joseph Berkson named in 1944 at the Mayo Clinic, which itself was an approximation to the probit that Chester Bliss invented in 1934 to count dead insects in Connecticut.
{{% /pullquote %}}

{{< timeline name="logit-history" >}}

## Napkin Math: The Exponentials Per Second

GPT-2 has a vocabulary of 50,257 tokens. Every single forward pass ends with:

1. A matrix multiply: $768$-dimensional hidden state times a $768 \times 50{,}257$ weight matrix `lm_head`. That's $768 \times 50{,}257 \approx 38.6\text{M}$ multiply-accumulate operations per token.
2. A softmax over 50,257 values: 50,257 exponentials, a sum, and 50,257 divisions.

On an A100 GPU at roughly 12 million tokens per second:

$$
12 \times 10^6 \text{ tok/s} \times 50{,}257 \text{ exp/tok} \approx 6 \times 10^{11} \text{ exponential evaluations per second}
$$

Just for the output layer. Six hundred billion exponentials per second — all to evaluate $\sigma_\text{scaled}(z)$, which is Berkson's 1944 approximation of the normal CDF, which is Bliss's 1934 model of how insects die.

## What You Carry Away

The entire probability chain that modern deep learning rests on — from binary classifiers to language models with 100 billion parameters — reduces to three moves, all invented before computers:

1. **Model log-odds as a linear function of inputs.** (Berkson, 1944)
2. **Apply the logistic function to get a probability.** (Same source)
3. **Generalize to $K$ classes with softmax and minimize cross-entropy.** (Clear by the 1960s; standard in deep learning by the 1980s)

From Bliss's dead aphids to a 70B-parameter LLM's prediction of the next word: in the end, one elegant mathematical idea — model the log-odds as a linear function of your inputs — runs inside every classifier that has ever been trained.
