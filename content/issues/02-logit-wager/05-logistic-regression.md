---
title: "From Insects to ImageNet"
description: "The logistic regression gradient turns out to be the cleanest computation in all of machine learning — and it is identical to backpropagation through a sigmoid output neuron."
topics: [machine-learning, statistics, optimization]
tags: [logistic-regression, cross-entropy, gradient, sigmoid, backpropagation]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 2
weight: 50
techKind: mainline
techNode: logistic-regression
header: default.webp
---

## Harvard, 1974

In 1974, **Paul Werbos** submitted his PhD thesis at Harvard, describing a general algorithm for computing gradients through multi-layer networks by propagating errors backward through the computation graph. He called it backpropagation. Almost nobody read it.

The logistic regression model, meanwhile, had been well-established in medicine and statistics for over a decade — David Cox had published its definitive treatment in 1958. But nobody had tried to train it on a million data points. The question of whether Berkson's 1944 approximation could *scale* had not been asked yet, because computers fast enough to make it relevant didn't exist.

That question would be answered in the 1980s, when logistic regression turned out to be the piece of deep learning that everything else plugs into.

## The Model

Logistic regression models the probability of a binary outcome $y \in \{0, 1\}$ given features $x$:

$$
P(y = 1 \mid x; \beta) = \sigma(\beta^\top x) = \frac{1}{1 + e^{-\beta^\top x}}
$$

where $\sigma$ is the logistic (sigmoid) function from [The Simpler S-Curve](../03-berksons-gamble/).

The quantity $\beta^\top x$ is the **logit** in the sense of [The Language of Risk](../04-log-odds/): the log-odds of the outcome. The model says: the log-odds is a linear function of the features. The sigmoid maps that linear function back to a probability.

## Fitting: Maximum Likelihood

We do not fit logistic regression with least squares. Instead, we use **maximum likelihood estimation**: find the $\beta$ that makes the observed data most probable.

For a dataset of $n$ observations $(x_i, y_i)$, the likelihood is:

$$
L(\beta) = \prod_{i=1}^n P(y_i \mid x_i; \beta) = \prod_{i=1}^n \sigma(\beta^\top x_i)^{y_i} \cdot (1 - \sigma(\beta^\top x_i))^{1-y_i}
$$

Taking the log (which turns products to sums and doesn't change the argmax):

$$
\ell(\beta) = \sum_{i=1}^n \left[ y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i)) \right], \quad z_i = \beta^\top x_i
$$

This is the **negative binary cross-entropy loss** — or equivalently, the log-likelihood under the logistic model. Maximizing $\ell(\beta)$ is the same as minimizing the cross-entropy.

## The Gradient: Surprisingly Clean

Now comes the computation that made logistic regression the engine of modern machine learning. Differentiating the log-likelihood with respect to $\beta$:

$$
\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^n (y_i - \hat{p}_i) \, x_i = X^\top (y - \hat{p})
$$

where $\hat{p}_i = \sigma(\beta^\top x_i)$ is the predicted probability for observation $i$.

*That's it.* The gradient is just the **prediction error** $y_i - \hat{p}_i$ dotted back into the features $x_i$.

What makes this remarkable is the simplicity. The derivative of $\sigma(z)$ with respect to $z$ is $\sigma(z)(1 - \sigma(z))$. You might expect this to appear explicitly in the gradient. It does appear — but it cancels beautifully against the denominator of $\log \sigma(z)$, leaving nothing but the residual $y - \hat{p}$.

Intuitively: if the model predicted $\hat{p} = 0.9$ but the true label was $y = 0$, the gradient pushes $\beta$ down by $0.9 \cdot x$. The model was very confident about the wrong answer; the gradient is proportionally large. If the model predicted $\hat{p} = 0.5$ and the true label was $y = 0$, the gradient pushes $\beta$ down by only $0.5 \cdot x$ — less confident, smaller correction.

## From Scratch

Here is what this looks like in code. We generate synthetic student admissions data (exam score → admitted or not), and fit logistic regression by gradient ascent on the log-likelihood:

```python
import numpy as np

np.random.seed(42)

# Generate synthetic admissions data
# True model: logit(p) = -6 + 0.1 * exam_score
n = 200
exam_score = np.random.uniform(40, 90, n)
true_beta = np.array([-6.0, 0.1])
X = np.column_stack([np.ones(n), exam_score])
z_true = X @ true_beta
p_true = 1 / (1 + np.exp(-z_true))
y = (np.random.uniform(0, 1, n) < p_true).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def log_likelihood(beta, X, y):
    p = sigmoid(X @ beta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def gradient(beta, X, y):
    p = sigmoid(X @ beta)
    return X.T @ (y - p)

# Gradient ascent
beta = np.zeros(2)
lr = 0.001
history = []

for step in range(2000):
    ll = log_likelihood(beta, X, y)
    history.append(-ll / n)  # store cross-entropy loss
    beta += lr * gradient(beta, X, y)

print(f"True β:    intercept={true_beta[0]:.2f}, slope={true_beta[1]:.4f}")
print(f"Fitted β:  intercept={beta[0]:.2f}, slope={beta[1]:.4f}")
print(f"Final cross-entropy loss: {history[-1]:.4f}")
```

```pyplot {id="logistic-regression-loss" caption="Cross-entropy loss over gradient ascent iterations. The loss decreases monotonically — logistic regression has a convex log-likelihood, so gradient ascent is guaranteed to reach the global optimum."}
np.random.seed(42)
n = 200
exam_score = np.random.uniform(40, 90, n)
true_beta = np.array([-6.0, 0.1])
X = np.column_stack([np.ones(n), exam_score])
z_true = X @ true_beta
p_true = 1 / (1 + np.exp(-z_true))
y = (np.random.uniform(0, 1, n) < p_true).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def log_likelihood(beta, X, y):
    p = sigmoid(X @ beta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def gradient(beta, X, y):
    p = sigmoid(X @ beta)
    return X.T @ (y - p)

beta = np.zeros(2)
lr = 0.001
history = []
for step in range(2000):
    ll = log_likelihood(beta, X, y)
    history.append(-ll / n)
    beta += lr * gradient(beta, X, y)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(history, color='#FF007F', linewidth=2)
ax.set_xlabel("gradient ascent step")
ax.set_ylabel("cross-entropy loss")
ax.set_title("Logistic regression training: cross-entropy converges to global optimum")
ax.spines[['top', 'right']].set_visible(False)
ax.fill_between(range(len(history)), history, alpha=0.15, color='#FF007F')
print(f"Initial loss: {history[0]:.4f}")
print(f"Final loss: {history[-1]:.4f}")
print(f"Fitted intercept: {beta[0]:.3f}, slope: {beta[1]:.4f}")
```

```pyplot {id="logistic-regression-fit" caption="Scatter of admitted (pink) and rejected (teal) students, with the fitted logistic sigmoid in orange. The curve crosses 0.5 at the decision boundary — the score where an applicant is equally likely to be admitted or rejected."}
np.random.seed(42)
n = 200
exam_score = np.random.uniform(40, 90, n)
true_beta = np.array([-6.0, 0.1])
X = np.column_stack([np.ones(n), exam_score])
z_true = X @ true_beta
p_true = 1 / (1 + np.exp(-z_true))
y = (np.random.uniform(0, 1, n) < p_true).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def gradient(beta, X, y):
    p = sigmoid(X @ beta)
    return X.T @ (y - p)

beta = np.zeros(2)
lr = 0.001
for step in range(2000):
    beta += lr * gradient(beta, X, y)

score_fine = np.linspace(35, 95, 300)
X_fine = np.column_stack([np.ones(300), score_fine])
p_fine = sigmoid(X_fine @ beta)
boundary = -beta[0] / beta[1]

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(exam_score[y == 1], y[y == 1], color='#FF007F', alpha=0.5, s=40,
           label='admitted', zorder=3)
ax.scatter(exam_score[y == 0], y[y == 0], color='#00A8A8', alpha=0.5, s=40,
           label='rejected', zorder=3)
ax.plot(score_fine, p_fine, color='#FF8C00', linewidth=2.5, label='σ(β₀ + β₁·score)')
ax.axvline(boundary, color='#1A1A1A', linewidth=1.5, linestyle='--',
           label=f'decision boundary: {boundary:.1f}')
ax.axhline(0.5, color='#1A1A1A', linewidth=0.8, linestyle=':')
ax.set_xlabel("exam score")
ax.set_ylabel("P(admitted | score)")
ax.set_title("Logistic regression: fitted sigmoid")
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
print(f"Decision boundary (P=0.5): score = {boundary:.1f}")
```

## The Connection to Deep Learning

The gradient $\partial \ell / \partial \beta = X^\top(y - \hat{p})$ is not just clean — it is *exactly* the backpropagation computation through a sigmoid output neuron with cross-entropy loss.

In a neural network, the output layer computes $z = W \cdot h$ (where $h$ is the final hidden state and $W$ is the weight matrix), applies $\sigma(z)$ to get a probability, and computes the binary cross-entropy loss. The gradient with respect to $W$ is $(y - \hat{p}) \cdot h^\top$ — the same "error times input" formula, just written with a hidden representation $h$ instead of raw features $x$.

This is not a coincidence. **Logistic regression is the last layer of a neural network classifier.** Hidden layers transform the input into a better feature representation. The final linear layer plus sigmoid is still logistic regression — in feature space.

The insight that made deep learning practical was that you could stack transformations on top of each other, backpropagate the logistic regression gradient through each layer, and everything worked out. The elegance of the logistic gradient — the fact that it reduces to nothing more than the prediction error — propagates cleanly through the entire network.

## Napkin Math

In a 10-class MNIST classifier, the final layer is not one logistic regression but *ten simultaneous logistic regressions* — one per class. Each competes against the others. To turn them into a proper probability distribution over 10 classes, you need to normalize. That normalization is the softmax function, which is the subject of the final chapter.

- **MNIST final layer**: $784 \to 128 \to 64 \to 10$. The last layer has $64 \times 10 + 10 = 650$ parameters. Of those, 10 are biases.
- **GPT-2 final layer**: $768 \to 50{,}257$ (vocabulary size). That is $768 \times 50{,}257 \approx 38.6\text{M}$ parameters — just in the last layer, called `lm_head`.
- **Each forward pass**: one matrix multiply, one softmax. For GPT-2, that's 38.6M multiplications just to get the next-token distribution.

---

**Continue to** → [The Softmax Kingdom](../06-softmax-kingdom/) — the boss capstone where Berkson's 1944 approximation meets its final form: a 50,257-class probability distribution computed billions of times per second, and still named after a word Berkson coined in a Mayo Clinic memo.
