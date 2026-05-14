---
title: "Taylor & Hessians"
description: "A primer on second-order Taylor expansion of the loss, the Hessian as a sensitivity matrix, and why curvature tells you which weights are safe to wreck."
topics: [calculus, optimization]
tags: [taylor-series, hessian, calculus, sensitivity]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 70
techKind: primer
techNode: taylor-hess
header: default.png
---

## What If We Just Rounded?

Here is a thought experiment. You have a trained neural network. You're going to change one weight from $w$ to $w + \delta$. How much does your loss go up?

If $\delta$ is tiny, you'd guess: roughly proportional to $\delta$ times the gradient. That's true if $\delta$ is small enough that *higher-order terms in the Taylor expansion are negligible*. But "small enough" depends on the curvature of the loss in that direction. In a flat valley you can perturb a weight wildly with no consequence; on a steep ridge, a tiny perturbation explodes the loss.

The quantization problem is, in disguise, this question repeated for every weight in the network. When you round $w$ to its nearest 4-bit representable value, you're introducing a $\delta$. The damage you do to the loss is determined by *the second derivative of the loss in the direction of that perturbation*.

This article is a fast tour of how to compute, approximate, and *exploit* that second derivative. The payoff comes in the next article ([Brain Surgery Returns](../08-brain-surgery/)), where Hassibi and Stork's 1992 idea — pruning the *least sensitive* weight while compensating the others — becomes the workhorse algorithm of modern LLM quantization.

## The Second-Order Taylor Expansion

Let $L(\theta)$ be the loss as a function of the weights $\theta \in \mathbb{R}^n$. Expanding around the trained weights $\theta^*$:

$$
L(\theta^* + \delta) \approx L(\theta^*) + g^\top \delta + \frac{1}{2} \delta^\top H \delta + O(\|\delta\|^3)
$$

where $g = \nabla L(\theta^*)$ is the **gradient** and $H = \nabla^2 L(\theta^*)$ is the **Hessian** — the matrix of all second partial derivatives.

At a trained model's solution, the gradient $g \approx 0$ (training has converged). So to leading order:

$$
\Delta L \approx \frac{1}{2} \delta^\top H \delta
$$

This is the **quadratic approximation of the loss surface around the minimum**. It says the change in loss caused by perturbing the weights by $\delta$ is determined entirely by the **Hessian**. Some perturbations cost a lot (large eigenvalues of $H$, steep curvature). Others cost almost nothing (small eigenvalues, flat directions).

## What The Hessian Looks Like

For a network with $n$ parameters, $H$ is an $n \times n$ symmetric matrix:

$$
H_{ij} = \frac{\partial^2 L}{\partial w_i \, \partial w_j}
$$

The **diagonal entries** $H_{ii} = \partial^2 L / \partial w_i^2$ are the *sensitivity of the loss to perturbing $w_i$ alone*. The **off-diagonal entries** $H_{ij}$ encode how perturbing $w_i$ and $w_j$ together interacts.

Visualize a tiny 2-parameter loss landscape:

```pyplot {id="hessian-curvature" caption="A quadratic bowl with anisotropic curvature. The same perturbation magnitude costs different amounts of loss depending on direction."}
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

# Anisotropic quadratic: steep in y, gentle in x
H = np.array([[1.0, 0.0], [0.0, 8.0]])

L = 0.5 * (H[0,0]*X**2 + 2*H[0,1]*X*Y + H[1,1]*Y**2)

fig, ax = plt.subplots(figsize=(7, 6))
levels = [0.5, 1, 2, 4, 8, 16]
cs = ax.contour(X, Y, L, levels=levels, colors='#1A1A1A', linewidths=1)
ax.clabel(cs, inline=True, fontsize=9)

# Two perturbations of equal magnitude
for vec, color, label in [((1.4, 0), '#FF007F', 'gentle direction'),
                           ((0, 1.4), '#00A8A8', 'steep direction')]:
    ax.annotate('', xytext=(0, 0), xy=vec,
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.text(vec[0]*1.08, vec[1]*1.08, label, fontsize=10, color=color, fontweight='bold')

ax.scatter([0], [0], s=80, color='#1A1A1A', zorder=4)
ax.set_xlabel("weight 1")
ax.set_ylabel("weight 2")
ax.set_title("Two equal-magnitude perturbations, different costs (H diag = [1, 8])")
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
```

Same $\|\delta\| = 1.4$ in both cases. The "gentle" direction costs $\Delta L \approx 1$; the "steep" direction costs $\Delta L \approx 8$. The Hessian eigenvalues *are* the cost ratio.

For LLM quantization, this becomes a sentence to repeat:

> **Weights along flat directions of the loss landscape can absorb large quantization errors without hurting accuracy. Weights along steep directions cannot.**

## You Can't Just Compute The Full Hessian

For a 7B-parameter model, $H$ is a $7 \times 10^9$ by $7 \times 10^9$ matrix. Storing it would require $1.9 \times 10^{20}$ bytes — roughly the entire annual data storage produced by humanity. So you have to approximate.

The trick that makes modern quantization tractable is to **block-diagonalize**. We don't compute $H$ for the *whole network*; we compute it **one layer at a time**, restricting to the weights in that layer. This is sometimes called the **layer-wise** approximation.

Even within a layer, the Hessian of the *task loss* is intractable. But there's a clever substitute: the **Hessian of the layer's output reconstruction loss**, which has a closed form.

For a linear layer $Y = X W^\top$ with weights $W$ and input activations $X$ collected on a calibration set, the loss

$$
L_{\text{layer}}(W) = \frac{1}{2} \| Y_{\text{original}} - X W^\top \|_F^2
$$

has Hessian (with respect to a single row of $W$):

$$
H_{\text{layer}} = X^\top X
$$

This is the **calibration covariance**. It's a $d \times d$ matrix where $d$ is the input width of the layer — typically $\sim$4000 for LLMs, so $\sim$16 million entries. *That* you can fit on a GPU.

```pyplot {id="calibration-hessian" caption="The empirical layer Hessian X^T X for a synthetic activation matrix. Off-diagonal blocks reveal correlated features — they're the ones quantization has to be careful with."}
np.random.seed(1)
d = 64
n_samples = 1024
# Synthetic activations with two correlated feature clusters
A = np.random.randn(n_samples, d) * 0.5
# Inject correlation in dims 8-16
mix = np.random.randn(n_samples, 1) * 2
A[:, 8:16] += mix
# Inject a few "outlier dimensions"
A[:, [32, 48]] *= 30

H = A.T @ A / n_samples

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
im = ax.imshow(H, cmap='RdBu_r', vmax=H.diagonal().max()*0.5,
               vmin=-H.diagonal().max()*0.5)
ax.set_title("Layer Hessian H = XᵀX / n")
ax.set_xlabel("input feature j")
ax.set_ylabel("input feature i")
plt.colorbar(im, ax=ax, shrink=0.7)

ax = axes[1]
ax.bar(range(d), H.diagonal(), color='#FF007F', edgecolor='#1A1A1A', linewidth=0.3)
ax.set_yscale('log')
ax.set_title("Hessian diagonal (sensitivity per input feature)")
ax.set_xlabel("input feature index")
ax.set_ylabel("H[i,i] (log)")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Two important features visible here:

1. **The diagonal of $H$ is the per-feature sensitivity.** It tells you which input features the layer pays a high price for getting wrong. Feature dims 32 and 48 (the outliers) dominate the spectrum.
2. **The off-diagonal block at 8–16 is non-zero.** Correlated features must be quantized *together-aware* — if you naively quantize them independently, you may compound errors. This is exactly why GPTQ's *sequential* update strategy matters (next article).

## Cholesky And The Sequential Trick

For modern LLM quantization the key algorithmic operation isn't using $H$ directly — it's using $H^{-1}$. Specifically, given that we've decided to quantize weight $w_q$ to its nearest representable value (incurring some error $\epsilon_q$), the **optimal compensation** to apply to all the remaining weights is:

$$
\delta_{\text{rest}} = -\epsilon_q \cdot \frac{H^{-1}_{:, q}}{H^{-1}_{q,q}}
$$

(We derive this in [Brain Surgery Returns](../08-brain-surgery/); the point is to set up the math now.) The compensation is a single column of $H^{-1}$ divided by its diagonal entry.

You don't compute $H^{-1}$ by inverting $H$ directly. Instead you compute the **Cholesky decomposition** $H = LL^\top$, then $H^{-1} = L^{-\top} L^{-1}$. The columns of $L^{-1}$ can be processed sequentially, which means **you can compensate weights one at a time** in left-to-right order — no need to recompute anything.

This sequential structure is the operational core of GPTQ. It's why a method based on second-order analysis can run in *minutes* on a 70B model rather than days: the algorithm marches column-by-column, using cached Cholesky factors, never re-inverting.

## Why The Loss-Surface Geometry Matters

To close the primer, here's the intuition you should carry into the next article.

A trained model sits at a (local) minimum of the loss. Around that minimum, the loss is approximately a quadratic bowl whose shape is governed by the Hessian.

**Some directions are flat.** You can move weights along these directions with almost no cost. Quantization noise that happens to project onto flat directions is *free*. This is why models are so robust to many forms of weight perturbation — most of weight-space is flat.

**Some directions are steep.** Moving along them ruins the model fast. Quantization noise that projects onto steep directions is *catastrophic*. Outlier weights tend to sit in steep directions — that's why they matter so much per-parameter.

The Hessian is the geometry that tells you which is which. And — Hassibi and Stork realized in 1992 — if you can *compensate* a small perturbation in one steep direction by an opposing perturbation in another direction, you can mostly undo the damage. The next article tells that story.

**Continue to** → [Brain Surgery Returns](../08-brain-surgery/) — a 1992 pruning paper, alive again in 2022 as GPTQ.
