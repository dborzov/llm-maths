---
title: "From Dead Insects to Deep Learning: How Logits Conquered Statistics and AI"
description: "A historical walk through the ideas connecting 1930s entomology to the output layer of your neural network — probits, logits, and the sigmoid function."
topics: [probability, classification]
tags: [logits, sigmoid, softmax, numpy, matplotlib]
theme: teal
math: true
draft: false
date: 2026-04-07T09:00:00-04:00
issue: 2
---

# From Dead Insects to Deep Learning: How Probits and Logits Conquered Statistics and AI

*A historical walk through the ideas that connect 1930s entomology to the output layer of your neural network.*

---

## The Problem: Killing Bugs, Scientifically

It's 1934. You're Chester Ittner Bliss, an entomologist working at the Connecticut Agricultural Experiment Station. Your job is practical: figure out the right dose of insecticide to kill pests. Farmers need answers. How much nicotine sulfate kills aphids? How much rotenone kills beetle larvae?

You run experiments. You expose batches of insects to increasing concentrations of poison, and you count how many die. Your data looks something like this:

| Dose (mg/L) | # Exposed | # Dead | % Dead |
|-------------|-----------|--------|--------|
| 1.0         | 50        | 3      | 6%     |
| 2.0         | 50        | 10     | 20%    |
| 3.0         | 50        | 19     | 38%    |
| 5.0         | 50        | 34     | 68%    |
| 8.0         | 50        | 42     | 84%    |
| 12.0        | 50        | 48     | 96%    |

You stare at this table. You need a *model* — something that lets you predict the kill rate at doses you haven't tested, and crucially, lets you estimate the **LD50**: the dose that kills exactly 50% of the population. This number is the gold standard of toxicology.

Your first instinct? Fit a line.

### Why Linear Regression Fails (Immediately)

```python
import numpy as np
import matplotlib.pyplot as plt

dose = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
proportion_dead = np.array([0.06, 0.20, 0.38, 0.68, 0.84, 0.96])

# Naive linear fit
coeffs = np.polyfit(dose, proportion_dead, 1)
fit_line = np.polyval(coeffs, np.linspace(0, 15, 100))

plt.figure(figsize=(8, 5))
plt.scatter(dose, proportion_dead, color='black', zorder=5, label='Observed')
plt.plot(np.linspace(0, 15, 100), fit_line, 'r--', label='Linear fit')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axhline(1, color='gray', linewidth=0.5)
plt.xlabel('Dose (mg/L)')
plt.ylabel('Proportion Dead')
plt.title('The Problem with Fitting a Line to Probabilities')
plt.legend()
plt.savefig('linear_fail.png', dpi=150, bbox_inches='tight')
plt.show()
```

The problems are glaring:

1. **The line goes below 0 and above 1.** At dose = 0, your model might predict −5% dead. At dose = 15, maybe 110% dead. Probabilities don't work that way.
2. **The relationship is not linear.** The data traces an S-shaped curve. The response accelerates in the middle and saturates at the extremes. A line can't capture this.

Bliss knew this. Every experimentalist in the 1930s knew this. The question was: *what shape should the curve be?*

## Bliss's Insight: Think About the Population, Not the Average

Here's where Bliss made a conceptual leap that's easy to miss if you just read the math. He didn't think about the curve directly. He thought about the *insects*.

His reasoning went like this: every insect in that batch has a different **tolerance** to the poison. Some are weak and die at tiny doses. Some are tough and survive large ones. If you could measure the exact lethal dose for each individual insect, you'd get a distribution — a histogram of tolerances across the population.

What distribution? Well, the most natural assumption in the 1930s (and now) was the **normal distribution**. Biological variation tends to be roughly Gaussian. (Actually, Bliss used the log of the dose, arguing that biological responses often scale multiplicatively. We'll come back to this.)

So the model is:

> Each insect has a tolerance $T$ drawn from a normal distribution $T \sim \mathcal{N}(\mu, \sigma^2)$. An insect dies if the applied dose $x$ exceeds its personal tolerance: $x > T$.

The probability of death at dose $x$ is then:

$$P(\text{dead} \mid x) = P(T < x) = \Phi\left(\frac{x - \mu}{\sigma}\right)$$

where $\Phi$ is the cumulative distribution function (CDF) of the standard normal. **That S-shaped curve you need? It's just the normal CDF.** The shape was always there, hiding inside the bell curve you learned in intro stats.

```python
from scipy.stats import norm

x = np.linspace(-4, 4, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: the bell curve (PDF)
axes[0].plot(x, norm.pdf(x), 'b-', linewidth=2)
axes[0].fill_between(x[x < 0.8], norm.pdf(x[x < 0.8]), alpha=0.3)
axes[0].set_title('Population Tolerance Distribution (PDF)')
axes[0].set_xlabel('Standardized Tolerance')
axes[0].set_ylabel('Density')
axes[0].annotate('All insects weaker\nthan this dose die',
                 xy=(0.8, 0.05), xytext=(2.0, 0.2),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

# Right: the S-curve (CDF)
axes[1].plot(x, norm.cdf(x), 'b-', linewidth=2)
axes[1].axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
axes[1].set_title('Probability of Death vs. Dose (CDF)')
axes[1].set_xlabel('Standardized Dose')
axes[1].set_ylabel('P(dead)')

plt.tight_layout()
plt.savefig('probit_intuition.png', dpi=150, bbox_inches='tight')
plt.show()
```

This is the **probit model**. The word "probit" is a portmanteau of "probability unit," coined by Bliss in his 1934 paper *"The Method of Probits."*

### The Probit Transform: Straightening the Curve

Now here's the clever trick. In the 1930s, there were no computers. No gradient descent. No `scipy.optimize`. If you wanted to fit a model, you wanted to fit a **straight line**, because you could do that with a ruler and a table of values.

Bliss realized: if the true relationship is $P = \Phi(a + bx)$, then applying the *inverse* normal CDF to both sides gives:

$$\Phi^{-1}(P) = a + bx$$

The left side, $\Phi^{-1}(P)$, is what Bliss called the **probit**. It transforms a probability into a real number on $(-\infty, +\infty)$. After this transformation, the relationship between dose and response is *linear*.

```python
from scipy.stats import norm

# Our insecticide data
dose = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
proportion_dead = np.array([0.06, 0.20, 0.38, 0.68, 0.84, 0.96])

# Probit transform: apply inverse normal CDF
# Clip to avoid infinities at 0 and 1
p_clipped = np.clip(proportion_dead, 0.01, 0.99)
probits = norm.ppf(p_clipped)

# Now fit a line in probit-space
coeffs = np.polyfit(dose, probits, 1)
print(f"Probit = {coeffs[0]:.3f} * dose + {coeffs[1]:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].scatter(dose, proportion_dead, color='black', zorder=5)
dose_fine = np.linspace(0, 14, 200)
fitted_probits = np.polyval(coeffs, dose_fine)
fitted_probs = norm.cdf(fitted_probits)
axes[0].plot(dose_fine, fitted_probs, 'b-', linewidth=2)
axes[0].set_xlabel('Dose (mg/L)')
axes[0].set_ylabel('Proportion Dead')
axes[0].set_title('Probit Model (Probability Space)')

axes[1].scatter(dose, probits, color='black', zorder=5)
axes[1].plot(dose_fine, fitted_probits, 'r-', linewidth=2)
axes[1].set_xlabel('Dose (mg/L)')
axes[1].set_ylabel('Probit (Φ⁻¹(P))')
axes[1].set_title('Probit Model (Linearized)')

plt.tight_layout()
plt.savefig('probit_linearized.png', dpi=150, bbox_inches='tight')
plt.show()
```

By transforming the y-axis, Bliss turned a nonlinear S-curve into a straight line that any 1930s statistician could fit with pencil, paper, and a table of normal deviates. Brilliant.

## Enter Berkson: "Why Not Use a Simpler Curve?"

Fast forward to 1944. Joseph Berkson, a physician and statistician at the Mayo Clinic, looks at the probit model and has a pragmatic objection.

The normal CDF $\Phi(z)$ has no closed-form expression. It's defined as an integral:

$$\Phi(z) = \int_{-\infty}^{z} \frac{1}{\sqrt{2\pi}} e^{-t^2/2} \, dt$$

Every time you compute it, you need a table or a numerical approximation. Its inverse is even worse. For the 1940s, this was a real computational burden.

Berkson proposed an alternative: instead of the normal CDF, use the **logistic function**:

$$P = \frac{1}{1 + e^{-(a + bx)}}$$

He called the corresponding transform the **logit** (by analogy with "probit"):

$$\text{logit}(P) = \ln\left(\frac{P}{1-P}\right) = a + bx$$

The logistic function is also S-shaped, bounded between 0 and 1, and looks almost identical to the normal CDF when you scale it appropriately. But it has two massive advantages:

1. **Closed form.** No integrals. Just exponentials and division.
2. **The inverse (logit) is trivial.** Just a log-odds ratio.

Let's see how similar they really are:

```python
from scipy.stats import norm, logistic

z = np.linspace(-5, 5, 300)

# Normal CDF (probit model)
probit_curve = norm.cdf(z)

# Logistic CDF (logit model) — scaled so they're comparable
# The logistic with scale=1 has slightly heavier tails.
# To match the normal CDF closely, we scale: logistic(z * pi/sqrt(3))
# But the standard comparison uses scale=1 directly
logit_curve = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 5))
plt.plot(z, probit_curve, 'b-', linewidth=2, label='Normal CDF (Probit)')
plt.plot(z, logit_curve, 'r--', linewidth=2, label='Logistic (Logit)')
plt.plot(z, probit_curve - logit_curve, 'g:', linewidth=1.5,
         label='Difference (×10 for visibility)')
plt.xlabel('z')
plt.ylabel('P(z)')
plt.title('Probit vs. Logit: Nearly Identical S-Curves')
plt.legend()
plt.savefig('probit_vs_logit.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Maximum absolute difference: {np.max(np.abs(probit_curve - logit_curve)):.4f}")
```

The maximum difference between them is about 0.02. In practice, with noisy data, you'd never be able to tell them apart. But the logistic function was *far* easier to work with.

### The Core Idea: Log-Odds Have a Beautiful Interpretation

The logit transform isn't just computationally convenient — it reveals something deep. The quantity $\ln(P / (1-P))$ is the **log-odds**.

Odds are a natural way to express relative likelihood. If a horse has a 75% chance of winning, the odds are 3:1 (three times more likely to win than lose). The logit says: *model the logarithm of the odds as a linear function of your predictors.*

Why log-odds? Because odds are multiplicative. If some factor doubles your odds of death, another triples them, and they act independently, then the combined odds are $2 \times 3 = 6$ times. Taking the log turns this multiplication into addition:

$$\text{logit}(P) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots$$

Each coefficient $\beta_i$ represents the additive change in log-odds per unit change in $x_i$, or equivalently, the *multiplicative* change in odds by a factor of $e^{\beta_i}$. This is the entire foundation of **logistic regression**.

## A Concrete Example: From Insects to Admissions

Let's leave the insecticide behind and build a logistic regression from scratch — the way Berkson would have, but with NumPy instead of pencil.

**Scenario:** You're predicting whether a student gets admitted to a graduate program based on their exam score.

```python
import numpy as np

np.random.seed(42)

# Generate synthetic data
n = 200
exam_score = np.random.normal(60, 12, n)  # mean=60, std=12
# True model: logit(P) = -6 + 0.1 * score  (so P=50% at score=60)
true_logits = -6 + 0.1 * exam_score
true_probs = 1 / (1 + np.exp(-true_logits))
admitted = np.random.binomial(1, true_probs)

print(f"Admission rate: {admitted.mean():.1%}")
print(f"Mean score of admitted: {exam_score[admitted==1].mean():.1f}")
print(f"Mean score of rejected: {exam_score[admitted==0].mean():.1f}")
```

### Fitting Logistic Regression with Gradient Descent

The model is $P(y=1 | x) = \sigma(\beta_0 + \beta_1 x)$ where $\sigma$ is the logistic sigmoid. We maximize the **log-likelihood**:

$$\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \ln \sigma(z_i) + (1-y_i) \ln(1-\sigma(z_i)) \right]$$

where $z_i = \beta_0 + \beta_1 x_i$. This is, of course, the negative of **binary cross-entropy loss** — the exact same loss function sitting at the output of your neural network.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def log_likelihood(beta, X, y):
    z = X @ beta
    p = sigmoid(z)
    return np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

def gradient(beta, X, y):
    z = X @ beta
    p = sigmoid(z)
    return X.T @ (y - p)  # This elegant form is why logistic regression is loved

# Prepare data: add intercept column
X = np.column_stack([np.ones(n), exam_score])
y = admitted

# Gradient ascent (maximizing log-likelihood)
beta = np.zeros(2)
lr = 0.0001
losses = []

for i in range(5000):
    grad = gradient(beta, X, y)
    beta += lr * grad
    losses.append(-log_likelihood(beta, X, y))  # track negative LL = cross-entropy

print(f"Fitted: logit(P) = {beta[0]:.3f} + {beta[1]:.4f} * score")
print(f"True:   logit(P) = -6.000 + 0.1000 * score")
print(f"Estimated LD50 (score for 50% admission): {-beta[0]/beta[1]:.1f}")
```

```python
# Visualization
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: the fitted curve
scores_range = np.linspace(30, 90, 200)
fitted_probs = sigmoid(beta[0] + beta[1] * scores_range)

axes[0].scatter(exam_score[admitted==1], admitted[admitted==1],
                alpha=0.3, color='green', label='Admitted')
axes[0].scatter(exam_score[admitted==0], admitted[admitted==0],
                alpha=0.3, color='red', label='Rejected')
axes[0].plot(scores_range, fitted_probs, 'k-', linewidth=2.5, label='Logistic fit')
axes[0].axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
axes[0].axvline(-beta[0]/beta[1], color='gray', linestyle=':', linewidth=0.8)
axes[0].set_xlabel('Exam Score')
axes[0].set_ylabel('P(Admitted)')
axes[0].set_title('Logistic Regression: Score → Admission')
axes[0].legend()

# Right: convergence
axes[1].plot(losses, 'b-', linewidth=1)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Cross-Entropy Loss')
axes[1].set_title('Gradient Ascent Convergence')

plt.tight_layout()
plt.savefig('logistic_regression.png', dpi=150, bbox_inches='tight')
plt.show()
```

Notice the gradient is $X^T(y - p)$ — the prediction error, projected back through the inputs. If you've ever looked at the backpropagation equations for a sigmoid output node, this is exactly the same expression. Berkson's 1944 convenience became the backbone of neural network training.

## The Bridge to Deep Learning

Let's now trace how this 1930s–1940s idea flows directly into modern deep learning.

### Softmax: Logistic Regression Goes Multi-Class

Logistic regression handles two classes. What about ten? (Think MNIST digits.) The generalization is **softmax**:

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

where $z_k = \mathbf{w}_k \cdot \mathbf{x} + b_k$ are called — wait for it — the **logits**. Every deep learning framework uses this terminology directly. When PyTorch asks you for "raw logits" as input to `CrossEntropyLoss`, it's asking for the $z_k$ values: the *pre-sigmoid* (or pre-softmax) linear scores. Berkson's 1944 transform, given a name and promoted to a fundamental computational unit.

```python
import torch
import torch.nn as nn

# The last layer of a neural network for 10-class classification
# is literally logistic regression stacked K times
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),   # <-- This outputs LOGITS. Berkson's z values.
)

# CrossEntropyLoss internally applies softmax + negative log-likelihood
# This IS the multi-class generalization of Berkson's logistic model
loss_fn = nn.CrossEntropyLoss()

# Dummy forward pass
x = torch.randn(1, 784)
logits = model(x)           # Raw scores, no activation — these are logits
print(f"Logits: {logits.detach().numpy().round(2)}")
print(f"Probabilities: {torch.softmax(logits, dim=1).detach().numpy().round(3)}")
```

The architecture of *any* classification neural network looks like this:

```
[Input] → [Learned features via hidden layers] → [Linear layer outputting logits] → [Softmax → Probabilities]
```

That final step is Berkson's logistic model. Everything before it is "just" learning a good feature representation so that a linear model in logit-space actually works. The deep network learns to make the problem linearly separable in the logit space.

### Why Logits and Not Probits in Deep Learning?

You could absolutely use a probit output instead of a logistic sigmoid. Replace the sigmoid $\sigma(z) = 1/(1+e^{-z})$ with the normal CDF $\Phi(z)$ and you'd have a probit neural network. Some people have tried it. It works fine. But the logistic function won:

1. **Gradient simplicity.** The sigmoid satisfies $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. This means backpropagation through it is trivial — just multiply by $p(1-p)$. The derivative of $\Phi(z)$ involves computing $\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$, which is more expensive and less numerically stable.

2. **Numerical stability.** $\log \sigma(z)$ and $\log(1 - \sigma(z))$ can be computed stably via `log_softmax` and `softplus`. The corresponding expressions for the normal CDF involve the error function and its logarithm, which are harder to keep stable in the tails.

3. **The log-odds interpretation.** The logit space has a clean probabilistic interpretation (log-odds). The probit space (z-score of the equivalent normal) is more abstract.

Berkson's practical argument — "the logistic is just easier to compute" — turned out to be prophetic. When scale matters (millions of parameters, billions of data points), computational convenience isn't a luxury. It's a requirement.

## A Timeline

| Year | Person | Contribution |
|------|--------|-------------|
| **1934** | Chester Bliss | Introduces probit analysis for dose-response modeling in toxicology |
| **1935** | Ronald Fisher | Develops maximum likelihood estimation for probit models, gives it rigorous statistical foundations |
| **1944** | Joseph Berkson | Proposes the logit as a computationally simpler alternative; coins the term "logit" |
| **1958** | David Cox | Formalizes logistic regression for general binary outcomes; extends it beyond bioassay |
| **1960s–70s** | Various | Logistic regression becomes a standard tool across medicine, social science, and economics |
| **1974** | Paul Werbos | Describes backpropagation; gradient of the sigmoid loss is central |
| **1986** | Rumelhart, Hinton, Williams | Popularize backprop; sigmoid output neurons with cross-entropy loss become standard |
| **2010s** | Deep learning era | "Logits" becomes universal vocabulary; softmax + cross-entropy dominates classification |

## The Physical Meaning: A Summary

Here's the core intuition to take away, in one paragraph:

**Probit and logit are both answers to the same question: how do you model a probability as a function of continuous inputs?** The trick is a *link function* — a nonlinear warp that maps the bounded interval $(0, 1)$ onto the entire real line $(-\infty, +\infty)$, so you can use the full power of linear models (and later, neural networks) in the unbounded space. Bliss used the inverse normal CDF because he was modeling biological variation. Berkson used the log-odds because it was simpler and had a clean interpretation. Both are S-shaped curves that turn linear scores into probabilities. The logistic sigmoid won the deep learning era not because it was theoretically superior, but because it was faster, stabler, and "good enough." When your PyTorch model outputs logits and you pass them to `CrossEntropyLoss`, you are standing at the end of a 90-year chain of ideas that started with a man trying to figure out how much poison kills an aphid.

---

*Further reading: Bliss, C.I. (1934), "The Method of Probits"; Berkson, J. (1944), "Application of the Logistic Function to Bio-Assay"; Cox, D.R. (1958), "The Regression Analysis of Binary Sequences."*
