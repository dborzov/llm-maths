---
title: "Low-Rank Magic: the same trick under fine-tuning and inference"
short_title: "Low-Rank Magic"
description: "LoRA (2021) shrunk fine-tuning by replacing weight updates with a product of two skinny matrices. DeepSeek MLA (2024) did the same thing to attention projections. Both ideas are a 1936 theorem in disguise."
blurb:
  - "June 2021: LoRA fine-tunes GPT-3 175B with 10,000× fewer trainable parameters. The Hugging Face PEFT library is built around it within a year."
  - "May 2024: DeepSeek V2 ships MLA — KV projections routed through a tiny shared latent. Inference cache shrinks ~30×."
  - "Two papers, three years apart, two ends of the LLM pipeline. The whiteboard math is identical: $W \\approx BA$ with $A$ and $B$ skinny."
  - "The factorization was proven optimal by Eckart and Young at Yale in 1936, while the transistor was still a decade away."
topics: [linear-algebra, low-rank, fine-tuning, attention]
tags: [svd, lora, mla, deepseek, microsoft, history]
theme: cream
math: true
draft: false
date: 2026-05-19T09:00:00-04:00
issue: 10
weight: 10
techKind: mainline
techNode: cold-open
header: default.webp
---

## A Monday Morning In Redmond

It is the morning of **June 17, 2021**, and a researcher named **Edward Hu** is finishing a paper he and seven coauthors have been arguing about for six months. The argument is about a single sentence in the introduction. The sentence reads, more or less: *fine-tuning a 175-billion-parameter language model on a downstream task can be done by training only ten million parameters, with no loss in quality.*

The number — ten million out of 175 billion — looks like a typo. Hu's team has triple-checked it on every benchmark they can find. They have ablated every component. They have re-run RoBERTa, DeBERTa, GPT-2, GPT-3. Every time, the answer comes back the same: **the matrix of changes you make to a pre-trained model during fine-tuning is empirically of very {{< wiki "low-rank" >}}low rank{{< /wiki >}}**. If you constrain it to *be* low-rank from the start — by parameterizing the update as the product of two skinny matrices — you save almost everything. Memory, time, disk space.

Hu pushes the paper to arXiv. The title is *"{{< wiki "lora" >}}LoRA: Low-Rank Adaptation of Large Language Models{{< /wiki >}}"* ({{< cite text="Hu et al., 2021" url="https://arxiv.org/abs/2106.09685" kind="paper" >}}). Within six months, the Hugging Face PEFT library — which will become the standard fine-tuning toolkit for every open-source LLM project in the world — is built around it.

The math in the paper is roughly ten lines long.

## Three Years Later, In Hangzhou

Skip ahead to **May 6, 2024**. A Chinese lab named **DeepSeek-AI** uploads a 50-page technical report and the weights of a 236-billion-parameter mixture-of-experts model. Buried in section 2.1 — neither the abstract nor the headline benchmark table — is a diagram of a thing they call **Multi-head Latent Attention**, MLA for short.

The diagram shows the K and V vectors of every attention layer being squeezed through a narrow waist before being written to the KV cache, then being un-squeezed at read time. The headline economic claim of the paper is that inference is **5.76× cheaper** than competitors. The math behind the squeezing is roughly ten lines long.

If you put the two whiteboards side by side — Hu's LoRA factorization on the left, DeepSeek's MLA on the right — they say the same thing. **A weight matrix $W$ of shape $d_\text{out} \times d_\text{in}$ can be replaced by the product of two skinny matrices $B$ (shape $d_\text{out} \times r$) and $A$ (shape $r \times d_\text{in}$), where $r$ is much smaller than either side of $W$.** LoRA does this to the *update* $\Delta W$ applied during fine-tuning. MLA does this to the *base weights* of K and V, from initialization. The algebra is the same. The savings are real in both cases. The papers do not cite each other; the two teams are working three years and ten thousand kilometers apart, on two different parts of the LLM pipeline.

So the question is not *why does this work twice*. The question is: **why does it work even once?** What is so special about the weight matrices of a trained transformer that lets you replace them — or the updates to them — with a product of skinny matrices and pay nothing in quality?

The answer lives, as so many answers in this field do, in a forgotten paper from before the war.

## Yale, 1936

In **1936**, a year before Alan Turing's *On Computable Numbers* and a decade before the first electronic computer, two researchers at **Yale University** published a four-page note in the journal *Psychometrika*. Their names were **Carl Eckart** (a physicist) and **Gale Young** (a statistician). The paper had a forgettable title: *"The Approximation of One Matrix by Another of Lower Rank"* ({{< cite text="Eckart & Young, 1936" url="https://link.springer.com/article/10.1007/BF02288367" kind="paper" >}}).

The problem they cared about was psychometric. Suppose you have a matrix $W$ — let's say it's a table where row $i$ is person $i$ and column $j$ is their score on test $j$. The table is noisy: actual test scores wobble. You believe the *true* underlying structure has only $r$ explanatory factors (maybe verbal ability, spatial reasoning, and arithmetic skill — say $r=3$). What is the best way to summarize the noisy matrix $W$ with a clean rank-$r$ matrix?

Their answer used a piece of machinery that Eugenio Beltrami and Camille Jordan had developed in 1873 — the **{{< wiki "svd" >}}Singular Value Decomposition{{< /wiki >}}**. Every real matrix $W$ can be written as
$$
W \;=\; U \, \Sigma \, V^\top
$$
where $U$ and $V$ are orthogonal matrices (rotations / reflections) and $\Sigma$ is diagonal, with non-negative entries $\sigma_1 \ge \sigma_2 \ge \ldots \ge \sigma_{\min(m,n)} \ge 0$ called the **singular values**.

Eckart and Young proved that the best rank-$r$ approximation of $W$ — measured by Frobenius norm $\Vert W - W_r \Vert_F$ — is what you get by keeping only the top $r$ singular values and zeroing out the rest:
$$
W_r \;=\; \sum_{i=1}^{r} \sigma_i \, u_i \, v_i^\top \;=\; U_r \Sigma_r V_r^\top
$$
This is *not* a heuristic. It is the optimal answer to the problem *"give me the closest rank-$r$ matrix to $W$"*. Truncate the SVD, you get the best low-rank approximation. Period.

Eckart and Young wrote four pages, drew no pictures, and went back to their other work. The result became a workhorse in psychometrics, then in chemistry (PCA on spectra), then in image processing (JPEG2000), then in recommender systems (the Netflix Prize, 2009), then — and this is the only step we care about — in **deep learning**, the moment somebody noticed that the weight matrices of trained neural networks have rapidly-decaying singular values.

## The Empirical Fact That Makes Everything Work

You can verify the empirical fact in five lines of NumPy. Take any pre-trained transformer. Pick a layer. Extract a weight matrix — let's say the query projection $W_Q$ in some middle layer. Run SVD on it. Plot the singular values.

```pyplot {id="cold-open-spectrum" caption="Singular-value spectra of three synthetic matrices. Real LLM weight matrices look like the pink curve: a sharp head, a long heavy tail of near-zero values. The rank-r truncation throws away the tail and keeps the variance."}
np.random.seed(11)
n = 1024

# A random Gaussian matrix — singular values are roughly flat (Marchenko-Pastur).
M_random = np.random.randn(n, n) * 0.05
_, s_random, _ = np.linalg.svd(M_random)

# A "trained" matrix: a low-rank core (rank ~ 80) + small noise.
true_rank = 80
U_true = np.random.randn(n, true_rank)
V_true = np.random.randn(true_rank, n)
sigma_true = np.exp(-np.linspace(0, 4, true_rank))
M_trained = (U_true * sigma_true) @ V_true + 0.005 * np.random.randn(n, n)
_, s_trained, _ = np.linalg.svd(M_trained)

# A "data" matrix: very heavy tail, perhaps stock returns or word counts.
M_data = np.random.randn(n, n) * 0.01
M_data[0] += 5.0   # one direction dominates
M_data[1] += 2.5
_, s_data, _ = np.linalg.svd(M_data)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.semilogy(s_trained / s_trained[0], color='#FF007F', linewidth=2.8,
            label='trained transformer weights (low-rank + noise)')
ax.semilogy(s_data / s_data[0], color='#FFD700', linewidth=2.2,
            label='heavy-tailed data matrix')
ax.semilogy(s_random / s_random[0], color='#00A8A8', linewidth=2.2,
            label='random Gaussian (no structure)')
ax.axvspan(0, 80, alpha=0.12, color='#FF8C00',
           label='the rank LoRA / MLA actually uses')
ax.set_xlabel('singular-value index $i$', fontsize=11)
ax.set_ylabel('$\\sigma_i / \\sigma_1$ (log scale)', fontsize=11)
ax.set_title('Why the trick works: real weight matrices have a few dominant directions', fontsize=12)
ax.legend(loc='lower left', fontsize=9.5, frameon=True, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0, 600)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Napkin: how many singular values to capture 90% of the Frobenius energy?
def coverage(s, frac):
    total = (s**2).sum()
    cum = (s**2).cumsum()
    return int((cum / total < frac).sum()) + 1

print(f"rank needed to capture 90% Frobenius energy:")
print(f"  trained-like matrix:  {coverage(s_trained, 0.90):4d} / {n}")
print(f"  heavy-tailed data:    {coverage(s_data, 0.90):4d} / {n}")
print(f"  random Gaussian:      {coverage(s_random, 0.90):4d} / {n}")
```

Stare at the printed table. A random Gaussian matrix needs almost its full dimension to capture 90% of its Frobenius energy. A real trained weight matrix needs less than a tenth. **That is the entire game.** The matrix is *in principle* full-rank — every singular value is nonzero — but *in practice* almost all of the variance lives in a tiny low-rank head.

When LoRA replaces a fine-tuning update $\Delta W \in \mathbb{R}^{d \times d}$ with $BA$ where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$, it is *projecting the update into the orange band of this plot*. The truncation is lossy in theory and almost-lossless in practice, because the update wasn't using the tail anyway.

When MLA replaces the $H \cdot D$-dimensional K vector with a length-$d_c$ latent — which is then re-expanded by a weight matrix at read time — it is doing the same thing, to the same plot, on a different matrix. The K matrix the attention layer produces is empirically low-rank. MLA bakes that fact into the architecture.

The cold-open question of this issue is: *if these are the same trick, why did we need two papers to discover it?*

The answer turns out to involve the historical accident of how the deep-learning research community formed (in California, between 2012 and 2018, mostly ignoring the linear-algebra literature from before 2000), the practical engineering question of *which matrix gets factored* (the weight update, the weight, the activation), and a piece of 2020 Facebook AI research that you have probably never heard of — but which proves, mathematically, that fine-tuning is allowed to be cheap.

## The Three Threads This Issue Pulls

The first thread is **theoretical**: the SVD itself, Eckart-Young, and the geometry of why low-rank approximation is the best you can do under squared-error loss. That is the next chapter, [SVD & Eckart-Young](../02-svd/). Read it if linear algebra feels rusty.

The second thread is **empirical**: how do we know real weight matrices are low-rank? We trace the observation back through three pre-LoRA papers — Sainath et al. (2013) on speech recognition, Denil et al. (2013) on "predicting parameters in deep learning", and crucially, [Aghajanyan et al. (2020)](../04-intrinsic-dimension/) on intrinsic dimension. The 2020 paper proves, with a remarkably clean experiment, that fine-tuning happens in a sub-200-dimensional subspace.

The third thread is **applied**: how do you actually do this in code? The mainline articles walk through [LoRA](../05-lora/) (the training-time trick) and [MLA, re-read as a LoRA](../07-mla-bridge/) (the same trick applied to the architecture itself). The bridge chapter is where the two stories converge.

The capstone chapter, [The Unified Low-Rank Thesis](../08-unified-thesis/), zooms out one final level. LoRA, MLA, GQA, KV-cache pruning, even quantization — all of them are exploiting different *kinds* of low-rank-ness in different parts of the transformer. By the end of the chapter, the entire compression literature collapses into a single two-axis chart and you can read off what every method is doing in one glance.

---

**Continue to** → [SVD & Eckart-Young](../02-svd/) — the 1936 theorem, the matrix decomposition that proves everything else in this issue is allowed to exist, and the running example we will keep on the desk for the next 30 pages.
