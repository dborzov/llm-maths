---
title: "SVD: the 1873 decomposition that became LoRA"
short_title: "SVD"
description: "Every real matrix factors as U Σ Vᵀ — three pieces with clean geometric meaning. Truncate the smallest σ values and you get the provably-best low-rank approximation. The Eckart-Young theorem is the floor LoRA and MLA stand on."
blurb:
  - "Beltrami (Italy, 1873) and Jordan (France, 1874) independently discovered SVD while working on bilinear forms."
  - "$W = U \\Sigma V^\\top$: an orthogonal rotation, an axis-aligned diagonal stretch, another orthogonal rotation. Three pieces, one matrix."
  - "Eckart-Young (1936): truncating the SVD to the top $r$ singular values gives the *provably* closest rank-$r$ matrix in Frobenius norm. Not a heuristic."
  - "Run on a 1024×1024 trained weight matrix, the top 64 σ-values typically hold 90%+ of the energy. That ratio is the entire reason this issue exists."
topics: [linear-algebra, low-rank]
tags: [svd, eckart-young, beltrami, jordan, history]
theme: teal
math: true
draft: false
date: 2026-05-19T09:00:00-04:00
issue: 10
weight: 20
techKind: primer
techNode: svd
header: default.webp
---

## Padua, 1873

In a quiet office in **Padua, Italy** in the year **1873**, a 38-year-old mathematician named **Eugenio Beltrami** is staring at a bilinear form. He is the chair of mathematics at the University of Padua, recently arrived from Pavia. He is interested in the geometric meaning of a peculiar two-variable function: $f(x, y) = x^\top W y$ where $W$ is some matrix and $x, y$ are vectors.

The question Beltrami asks is this: *can I always rotate the $x$ and $y$ coordinate systems independently so that the bilinear form becomes a sum of single-axis terms?* That is, can I find rotations $U$ and $V$ such that $x^\top W y$ becomes $(U^\top x)^\top \Sigma (V^\top y)$ for some diagonal $\Sigma$?

His answer, published in the *Giornale di Matematiche* in 1873 ({{< cite text="Beltrami, 1873" url="http://www.bibmath.net/dico/index.php?action=affiche&quoi=./b/beltrami.html" kind="paper" >}}), is *yes — always*. The next year, in 1874, the French mathematician **Camille Jordan** independently rediscovers the result while working on a slightly different problem ({{< cite text="Jordan, 1874" url="https://archive.org/details/JordanSurReductionFormeBilineaire" kind="paper" >}}). For the next sixty years the decomposition is a curiosity in the European bilinear-forms literature. The Americans will call it the **Singular Value Decomposition** when they pick it up in the 1930s.

The decomposition Beltrami discovered says that every real matrix $W \in \mathbb{R}^{m \times n}$ can be written as
$$
W \;=\; U \, \Sigma \, V^\top
$$
where $U$ is an $m \times m$ orthogonal matrix, $V$ is an $n \times n$ orthogonal matrix, and $\Sigma$ is a rectangular $m \times n$ diagonal matrix whose diagonal entries — the **singular values** — are non-negative and (by convention) sorted in decreasing order:
$$
\sigma_1 \;\ge\; \sigma_2 \;\ge\; \ldots \;\ge\; \sigma_{\min(m,n)} \;\ge\; 0.
$$
The number of *nonzero* singular values is the **rank** of $W$. Everything else in this issue is downstream of this one identity.

## What The Three Pieces Mean Geometrically

The hard thing about SVD when you first see it is that the algebra hides a beautiful geometric picture. Strip it back to two dimensions and look.

A 2×2 matrix $W$ is a linear map from the plane to the plane. The unit circle in input space — the set of all vectors of length 1 — gets mapped to *something* in output space. That something is always an **ellipse** (possibly degenerate). The two axes of that ellipse are exactly the two singular vectors of $W$, and the half-lengths of those axes are exactly the two singular values.

The SVD says: any linear map $W$ does three things, in order.

1. **$V^\top$ — rotate the input.** Spin the input space so that the singular-vector directions line up with the standard axes.
2. **$\Sigma$ — stretch along the standard axes.** Pull along the first axis by $\sigma_1$, the second by $\sigma_2$, and so on. This is the only step that changes lengths or determines the rank.
3. **$U$ — rotate the output.** Spin the result into its final orientation.

Three operations: rotate, stretch, rotate. Two rotations and one diagonal scaling. **Every linear map ever, broken into the only three things that matter.**

```pyplot {id="svd-geometry" caption="A 2×2 matrix W maps the unit circle (top-left) into an ellipse (bottom-right). The SVD factors that map into three steps: rotate by Vᵀ, stretch by Σ, rotate by U."}
W = np.array([[2.0, 0.6],
              [-0.5, 1.4]])
U, s, Vt = np.linalg.svd(W)
S = np.diag(s)

# Unit circle sampled at 200 points.
theta = np.linspace(0, 2*np.pi, 200)
circle = np.stack([np.cos(theta), np.sin(theta)])

step0 = circle                          # input
step1 = Vt @ step0                      # after V^T (rotation)
step2 = S @ step1                       # after Sigma (stretch)
step3 = U @ step2                       # after U (rotation) — equals W @ circle

# Singular vectors highlighted as red and gold arrows.
v1, v2 = Vt[0], Vt[1]
u1, u2 = U[:,0], U[:,1]
sig1, sig2 = s

fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
panels = [
    (axes[0], step0, "input: unit circle"),
    (axes[1], step1, "after $V^\\top$: rotated"),
    (axes[2], step2, "after $\\Sigma$: stretched"),
    (axes[3], step3, "after $U$ = $Wx$: ellipse"),
]
for ax, pts, title in panels:
    ax.plot(pts[0], pts[1], color='#FF007F', linewidth=2.4)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='#1A1A1A', linewidth=0.4)
    ax.axvline(0, color='#1A1A1A', linewidth=0.4)
    ax.set_title(title, fontsize=10)
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)

# Annotate the axis vectors in the input panel (Vᵀ rotates these to the axes)
axes[0].arrow(0, 0, v1[0], v1[1], head_width=0.13, color='#FFD700', linewidth=2)
axes[0].arrow(0, 0, v2[0], v2[1], head_width=0.13, color='#00A8A8', linewidth=2)
axes[3].arrow(0, 0, sig1*u1[0], sig1*u1[1], head_width=0.16, color='#FFD700', linewidth=2)
axes[3].arrow(0, 0, sig2*u2[0], sig2*u2[1], head_width=0.13, color='#00A8A8', linewidth=2)

print(f"W = U S Vᵀ with:")
print(f"  σ₁ = {s[0]:.3f}  (long axis of ellipse)")
print(f"  σ₂ = {s[1]:.3f}  (short axis of ellipse)")
print(f"  ratio σ₁/σ₂ = {s[0]/s[1]:.2f}")
```

The gold and teal arrows in the input panel are the **right singular vectors** $v_1, v_2$ — the input directions that the map treats specially. The gold and teal arrows in the output panel are the **left singular vectors** $u_1, u_2$, scaled by their respective singular values $\sigma_1, \sigma_2$. The matrix is saying: *send the gold input direction to the gold output direction, scaled by $\sigma_1$. Send the teal input direction to the teal output direction, scaled by $\sigma_2$. Anything else? Spread it out as a combination of those two.*

That is what SVD *is*. The rest is bookkeeping.

## The Outer-Product View (The One You Need For LoRA)

For LoRA and MLA the geometric ellipse picture is nice, but the load-bearing identity is **the outer-product form**. Rewrite $W = U \Sigma V^\top$ as a sum:
$$
W \;=\; \sum_{i=1}^{\min(m,n)} \sigma_i \, u_i \, v_i^\top
$$
Each term in the sum is a **rank-1 matrix** (an outer product of two vectors), scaled by a singular value. The full $W$ is a weighted sum of rank-1 matrices, with the weights decreasing in magnitude.

This expansion is everything you need to understand the rest of the issue. A matrix is a sum of rank-1 pieces, *ordered by importance*. Drop the smallest pieces — keep only the top $r$ — and you get an approximation $W_r$:
$$
W_r \;=\; \sum_{i=1}^{r} \sigma_i \, u_i \, v_i^\top
$$
The approximation is a rank-$r$ matrix. It uses $r (m + n)$ numbers to store (the $r$ left singular vectors, the $r$ right singular vectors; the singular values can be folded into either side). The full $W$ uses $m \cdot n$ numbers. If $r \ll \min(m, n)$ that is a savings of orders of magnitude.

But is $W_r$ the *best* rank-$r$ approximation to $W$, or just *a* rank-$r$ approximation? Could a smarter cobbling-together of other rank-1 pieces, perhaps not aligned with the singular vectors, do better?

## The 1936 Theorem That Closes The Question

In **March 1936**, a few months before the Berlin Olympics, a Yale physicist named **Carl Henry Eckart** and a Yale statistician named **Gale Young** published a four-page note in *Psychometrika* — a journal devoted to the mathematics of psychological measurement. The note ({{< cite text="Eckart & Young, 1936" url="https://link.springer.com/article/10.1007/BF02288367" kind="paper" >}}) had a title that gave the game away: *"The Approximation of One Matrix by Another of Lower Rank."*

They proved the following.

{{% callout type="theorem" title="Eckart-Young theorem (1936)" %}}
Let $W \in \mathbb{R}^{m \times n}$ have singular values $\sigma_1 \ge \sigma_2 \ge \ldots$. Then, for any $r \le \mathrm{rank}(W)$, the truncated SVD $W_r = \sum_{i=1}^{r} \sigma_i u_i v_i^\top$ minimizes
$$
\Vert W - X \Vert_F \quad \text{over all rank-}r\text{ matrices } X.
$$
The minimum value is $\sqrt{\sigma_{r+1}^2 + \ldots + \sigma_{\min(m,n)}^2}$ — the Frobenius norm of the tail you threw away.
{{% /callout %}}

In English: **the best rank-$r$ approximation to a matrix, measured by squared-error, is the one you get by keeping the top-$r$ singular vectors and zeroing out everything else.** It is not a heuristic. It is not "usually pretty good". It is the *provably optimal* answer to a well-defined optimization problem.

A few years later, **Lev Mirsky** generalized the theorem to all *unitarily invariant norms* (Frobenius, spectral, nuclear, Schatten-$p$, anything orthogonal-invariant) ({{< cite text="Mirsky, 1960" url="https://academic.oup.com/qjmath/article-abstract/11/1/50/1502161" kind="paper" >}}). The theorem is even more general than Eckart and Young originally let on: under any reasonable notion of "close", the truncated SVD is the right thing.

This is the result LoRA, MLA, and every modern matrix-factorization trick stands on.

## A Worked Example: Compressing An Image

The most visually obvious application of Eckart-Young is image compression. A grayscale photograph is a matrix — pixel values. Take the SVD, keep only the top $r$ singular values, reconstruct. Watch the photograph degrade smoothly as $r$ decreases. Watch how forgiving it is.

```pyplot {id="svd-image" caption="Rank-r SVD reconstructions of a 200×200 grayscale image. At r=20 (3% of full rank) the image is recognizable. At r=80 it is visually lossless. Real LLM weight matrices behave more like the latter."}
# Build a synthetic test image with structured low-rank-ish content.
n = 200
yy, xx = np.mgrid[0:n, 0:n] / n
img = (
    0.6 * np.exp(-((xx - 0.4)**2 + (yy - 0.55)**2) / 0.03) +
    0.4 * np.exp(-((xx - 0.7)**2 + (yy - 0.35)**2) / 0.015) +
    0.25 * (np.sin(yy * 18) * np.cos(xx * 8)) +
    0.1 * np.random.RandomState(2).randn(n, n)
)
img = np.clip(img, 0, 1)

U, s, Vt = np.linalg.svd(img, full_matrices=False)

ranks = [5, 20, 80, 200]
fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
total_energy = (s**2).sum()
for ax, r in zip(axes.flat, ranks):
    recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
    err_frac = 1.0 - (s[:r]**2).sum() / total_energy
    bytes_saved = 1.0 - r * (n + n) / (n * n)
    ax.imshow(recon, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"$r = {r}$  ·  err = {err_frac:.1%}  ·  storage saved: {bytes_saved:.0%}",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# Print the singular-value tail
print("First 12 singular values:")
for i, sv in enumerate(s[:12]):
    print(f"  σ_{i+1:2d} = {sv:7.3f}  ({sv/s[0]:.3f} of σ₁)")
```

Notice two things in the printed table. First, the singular values decay *fast* — the first one is much larger than the rest. Second, at $r=20$ (out of 200) the reconstruction error is already small enough that the eye barely cares. At $r=80$ the reconstruction is visually identical to the original *and uses 20% of the storage*.

The image-compression demo is a toy. The *interesting* version is the same calculation on a 4096×4096 weight matrix from a trained transformer. We will run that experiment in the [next primer](../03-low-rank-hypothesis/). The conclusion is the same shape but the spectrum is even more dramatic.

## The Algorithms (Briefly)

How do you actually *compute* the SVD? There is a long story here and we will skip it.

The short answer: for a matrix $W \in \mathbb{R}^{m \times n}$ with $m \ge n$, the canonical algorithm is the **Golub-Kahan-Reinsch bidiagonalization** ({{< cite text="Golub & Kahan, 1965" url="https://doi.org/10.1137/0702016" kind="paper" >}}; {{< cite text="Golub & Reinsch, 1970" url="https://link.springer.com/article/10.1007/BF02163027" kind="paper" >}}), which runs in $O(m n^2)$ time and is what `numpy.linalg.svd` calls under the hood.

For the LoRA use case you usually do not actually compute a full SVD of any large matrix. You *parameterize* a low-rank matrix as the product $BA$ of two random skinny matrices and let gradient descent optimize them. The Eckart-Young theorem is the *promise* that this is enough: there exists some pair $(B^*, A^*)$ whose product is within $\Vert W - W_r \Vert_F$ of the true best, and the optimizer's job is to find it.

For the MLA use case there is an even more interesting wrinkle: DeepSeek does not start from a pre-trained model and project. They *initialize the architecture* with the low-rank factorization baked in, and train from scratch. The model finds its own best $r$-dimensional subspace as part of training. The Eckart-Young theorem in this case is more like a *permission slip* — it says, "the architecture you've imposed is allowed to express anything a rank-$r$ matrix can express, and we know that's a lot of useful structure".

## The Three Norms That Will Recur

We will see three different norms used to score low-rank approximations across this issue. Worth naming them once.

| Norm | Definition | Used for |
|---|---|---|
| **Frobenius** $\Vert W \Vert_F$ | $\sqrt{\sum_{ij} W_{ij}^2} = \sqrt{\sum_i \sigma_i^2}$ | the "default" — Eckart-Young, LoRA reconstruction error |
| **Spectral** $\Vert W \Vert_2$ | $\sigma_1(W)$ — largest singular value | worst-case input direction; bounds activation magnitude |
| **Nuclear** $\Vert W \Vert_*$ | $\sum_i \sigma_i$ — sum of singular values | rank-promoting convex surrogate; convex optimization |

The Eckart-Young theorem (extended by Mirsky) says the truncated SVD is optimal under *all three* of these. Pick the norm that matches your problem and the answer is still: keep the top $r$ singular values.

## Why This Matters For LLMs

We have spent a primer establishing that every matrix factors as $U \Sigma V^\top$ and that throwing away small singular values is provably the best way to approximate it with a low-rank matrix.

For LLMs the question of the next primer is *whether real weight matrices have small enough singular values for this to be useful*. The answer — spoiler — is overwhelmingly yes, and we will see exactly how dramatic the cliff is on a real transformer's weights.

For LoRA the question of [chapter 5](../05-lora/) is *whether you can replace the fine-tuning update with a rank-$r$ matrix*. The answer there relies not only on the weight matrix being low-rank, but on the *delta* between the pre-trained weight and the fine-tuned weight being even lower-rank. That is the [intrinsic dimension](../04-intrinsic-dimension/) story.

For MLA the question of [chapter 7](../07-mla-bridge/) is *whether you can parameterize the K and V projections themselves as the product of two skinny matrices, from scratch, and have training discover a good subspace*. The Eckart-Young theorem is the permission slip; the rest is engineering.

## What To Remember

1. **Every real matrix factors as $W = U \Sigma V^\top$** — two orthogonal rotations and one diagonal stretch. The diagonal entries (singular values) are non-negative and sorted.
2. **A matrix is a sum of rank-1 outer products**, ordered by singular value: $W = \sum_i \sigma_i u_i v_i^\top$.
3. **Eckart-Young (1936): the truncated SVD is the best rank-$r$ approximation in Frobenius norm.** Not a heuristic — provably optimal. Mirsky (1960) generalized to all unitarily invariant norms.
4. **Storage: a rank-$r$ approximation of an $m \times n$ matrix needs $r(m+n)$ scalars** instead of $mn$. For $m=n=4096$ and $r=16$, that is $128\times$ shrink.
5. **LoRA and MLA both rely on this theorem.** LoRA factors the *update* during fine-tuning; MLA factors the *weight matrices themselves* in the architecture. Same algebra, different surface.

{{% callout type="tangent" title="A historical curiosity" %}}
The phrase "Eckart-Young theorem" leaves out at least two prior contributors. **Erhard Schmidt** (yes, of Gram-Schmidt) proved the same result for integral operators in 1907 ({{< cite text="Schmidt, 1907" url="https://link.springer.com/article/10.1007/BF01449770" kind="paper" >}}). **Lev Mirsky** extended it to all unitarily invariant norms in 1960. Some authors call it the **Schmidt-Eckart-Young-Mirsky** theorem to give credit to all four. We will keep saying "Eckart-Young" because that is the convention in the deep-learning literature, but the result is older and more general than the 1936 citation suggests.
{{% /callout %}}

---

**Continue to** → [The Low-Rank Hypothesis](../03-low-rank-hypothesis/) — why real transformer weight matrices have rapidly-decaying singular values in the first place, and what fails to explain it.
