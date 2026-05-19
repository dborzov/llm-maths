---
title: "SVD (Singular Value Decomposition)"
slug: "svd"
description: "Singular Value Decomposition: W = U Σ Vᵀ. The Eckart-Young theorem (1936) proves the truncated SVD is the best rank-r approximation. Foundation of LoRA, MLA, and every low-rank compression in modern LLMs."
category: "transformer"
also_known_as:
  - "Singular Value Decomposition"
  - "Eckart-Young theorem"
  - "Eckart-Young-Mirsky theorem"
  - "matrix factorization"
  - "low-rank approximation"
  - "rank-r truncation"
  - "U Sigma V"
  - "singular values"
  - "singular vectors"
  - "left singular vectors"
  - "right singular vectors"
  - "principal component analysis"
  - "stable rank"
  - "effective rank"
  - "Frobenius norm"
  - "spectral norm"
  - "nuclear norm"
source_of_truth: "/comicbook/10-low-rank/02-svd/"
source_of_truth_title: "ch.2 SVD & Eckart-Young (Issue 10)"
related: ["low-rank", "lora", "attention", "transformer-weights"]
draft: false
---

Every real matrix factors as $W = U \Sigma V^\top$ — two orthogonal rotations and one diagonal stretch. The diagonal entries (the **singular values**) are non-negative and sorted descending.

| Symbol | What it is |
|---|---|
| $W$ | The matrix being decomposed, shape $m \times n$ |
| $U$ | Orthogonal $m \times m$ matrix — left singular vectors |
| $\Sigma$ | Diagonal rectangular — singular values $\sigma_1 \ge \sigma_2 \ge \ldots \ge 0$ |
| $V$ | Orthogonal $n \times n$ matrix — right singular vectors |
| $W_r$ | Truncated SVD: $\sum_{i=1}^{r} \sigma_i u_i v_i^\top$ — best rank-$r$ approximation |
| stable rank | $\Vert W \Vert_F^2 / \Vert W \Vert_2^2$ — numerical measure of effective rank |

**Eckart-Young theorem (1936):** the truncated SVD $W_r$ minimizes $\Vert W - X \Vert_F$ over all rank-$r$ matrices $X$. Mirsky (1960) extended to all unitarily invariant norms.

The theorem is the floor under [LoRA](/comicbook/10-low-rank/05-lora/), [MLA](/comicbook/05-microgpt/19-mla/), and every low-rank-direction compression method. See [Low-Rank Magic — ch.2](/comicbook/10-low-rank/02-svd/) for the full derivation with worked image-compression and weight-matrix examples.
