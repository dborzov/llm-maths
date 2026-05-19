---
title: "Intrinsic Dimension: fine-tuning happens in a sub-200-D subspace"
short_title: "Intrinsic Dimension"
description: "Aghajanyan, Zettlemoyer, Gupta (Facebook, 2020) proved BERT-large can be fine-tuned to MRPC accuracy using a single 200-dimensional update — out of 110 million parameters. The paper that made LoRA inevitable."
blurb:
  - "Facebook AI, October 2020: train a random projection P from $\\mathbb{R}^d$ into a network's full parameter space. Find the smallest $d$ that still solves the task."
  - "BERT-large on MRPC: $d_{90} = 207$. The update lives in a subspace of dimension 207 out of 110 million parameters."
  - "Larger pre-trained models have *smaller* intrinsic dimensions. The bigger the base model, the cheaper fine-tuning gets."
  - "The paper does not propose LoRA — but its experimental result *demands* LoRA. Microsoft proposes it eight months later."
topics: [low-rank, fine-tuning, optimization]
tags: [aghajanyan, intrinsic-dimension, facebook, lora-prequel]
theme: teal
math: true
draft: false
date: 2026-05-19T09:00:00-04:00
issue: 10
weight: 40
techKind: primer
techNode: intrinsic-dimension
header: default.webp
---

## Menlo Park, October 2020

In **October 2020**, a researcher at **Facebook AI Research** named **Armen Aghajanyan** posts a paper to arXiv called *"Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"* ({{< cite text="Aghajanyan, Zettlemoyer & Gupta, 2020" url="https://arxiv.org/abs/2012.13255" kind="paper" >}}). It is co-authored with **Luke Zettlemoyer**, who is at the time a giant of NLP, and **Sonal Gupta**.

The paper has a strange structure for the NLP literature of 2020. It does not propose a new model. It does not beat a benchmark. It runs an *experiment* — borrowed almost verbatim from a 2018 paper on convnets by **Li, Farkhoor, Jiang and Yosinski** ({{< cite text="Li et al., 2018" url="https://arxiv.org/abs/1804.08838" kind="paper" >}}) — and reaches a conclusion that, on a quick read, sounds absurd:

> Fine-tuning BERT-large from a pre-trained checkpoint to within 90% of its full-fine-tuning accuracy on the MRPC sentence-pair task **can be done by training a single 207-dimensional vector**. The other 110-million-minus-207 parameters of BERT-large are determined by composition with a fixed random projection matrix.

If you parse the sentence carefully you realize what it is saying. BERT-large has 110 million parameters. Fine-tuning it normally means optimizing all 110 million. Aghajanyan's experiment instead parameterizes the *update* — the difference between the pre-trained weights and the fine-tuned weights — as a tiny vector $\theta \in \mathbb{R}^d$ projected up to the full parameter space through a *frozen random matrix* $P$ of shape $110M \times d$:
$$
W_{\text{fine-tuned}} \;=\; W_{\text{pre-trained}} \;+\; P \, \theta
$$
He sweeps $d$ from 10 to 50,000. He records the smallest $d$ that still solves the task. He calls it the **intrinsic dimension** of the fine-tuning problem.

For BERT-large on MRPC, that number is **207**.

For a task with 110 million degrees of freedom in principle, the actual amount of *new information* needed to adapt it to a downstream task is **less than three hundred numbers**. The other parameters re-arrange themselves automatically through the random projection.

Eight months later, [LoRA](../05-lora/) appears at Microsoft. It is not Aghajanyan's paper — LoRA has structure where the Aghajanyan experiment has randomness — but the *empirical premise* of LoRA is exactly what Aghajanyan documented.

This primer takes that experiment apart, runs a tiny version of it, and explains why the result is more important than it looked at the time.

## The Random-Projection Experiment

The setup is elegant. Let $W_\text{pre}$ be the full parameter vector of a pre-trained model — for BERT-large, a vector in $\mathbb{R}^{110M}$. Pick an *intrinsic dimension* $d$. Sample a random matrix $P \in \mathbb{R}^{110M \times d}$ with iid Gaussian entries — once, frozen. Now define a new model with trainable parameters $\theta \in \mathbb{R}^d$ as:
$$
W(\theta) \;=\; W_\text{pre} \;+\; P\, \theta
$$
You only optimize $\theta$, not $W$. The random matrix $P$ stays frozen for the entire training run. The model's expressive power is constrained to live in a $d$-dimensional affine subspace of parameter space, centered on the pre-trained weights and oriented by the random projection.

Run fine-tuning. Measure final accuracy. Sweep $d$.

The **intrinsic dimension** $d_{90}$ is defined as *the smallest $d$ for which the constrained model reaches 90% of the unconstrained-fine-tuning accuracy*. (You can pick any threshold; 90% is the convention.)

The remarkable findings of Aghajanyan's paper:

| Model | Parameters | $d_{90}$ on MRPC |
|---|---|---|
| BERT-base | 110M | 1,608 |
| BERT-large | 340M | 207 |
| RoBERTa-base | 125M | 896 |
| RoBERTa-large | 355M | 207 |
| BART-large | 400M | 250 |

Two patterns jump off the table.

First, **the intrinsic dimensions are tiny**. Hundreds, not millions. Whatever fine-tuning is doing, it is not exploring most of the parameter space.

Second, **bigger pre-trained models have *smaller* intrinsic dimensions**. BERT-large at 340M parameters has $d_{90} = 207$. BERT-base at 110M parameters needs $d_{90} = 1608$. The trend continues — the bigger the model, the less you need to change to adapt it.

This is the opposite of what you might naively expect. Bigger models have more parameters, more capacity, more "stuff to adjust". You would think adapting them takes more work. The actual finding is reversed: a bigger pre-trained model is *more* easily fine-tuned, because pre-training has already done so much of the work that whatever's left for the downstream task lives in a tiny subspace.

## A Toy Version You Can Run In NumPy

Let us replicate the spirit of the experiment on a much smaller object. Take a randomly-initialized "model" — a single 256×256 weight matrix. Pre-train it (in our toy, "pre-training" is solving a regression with synthetic labels). Then fine-tune it to a new downstream task. Sweep the dimension of the random-projection subspace and find the smallest $d$ that gets us close.

```pyplot {id="intrinsic-toy" caption="Toy intrinsic-dimension experiment. A 256×256 'model' is pre-trained on Task A, then fine-tuned to Task B by adding a $P\\theta$ correction where $P$ is a frozen random projection and $\\theta$ has $d$ entries. Loss curve plotted vs $d$."}
np.random.seed(0)
n = 64    # input/output dim
m = 256   # number of training examples for each task

# Task A: regression with weights W_A
X = np.random.randn(m, n)
W_A = np.random.randn(n, n) * 0.2
y_A = X @ W_A

# Pre-train: solve W exactly for Task A via least squares.
W_pre, *_ = np.linalg.lstsq(X, y_A, rcond=None)

# Task B: a related but distinct linear regression.
# We construct W_B as W_A plus a low-rank perturbation, so the *true* delta
# lives in a low-dimensional subspace — mirroring the empirical claim.
true_rank = 4
B_true = np.random.randn(n, true_rank) * 0.1
A_true = np.random.randn(true_rank, n) * 0.1
W_B = W_A + B_true @ A_true
y_B = X @ W_B

# Baseline: full fine-tuning (solve directly).
W_full, *_ = np.linalg.lstsq(X, y_B, rcond=None)
loss_full = np.mean((X @ W_full - y_B)**2)
loss_pre = np.mean((X @ W_pre - y_B)**2)
print(f"loss at pre-trained weights (no fine-tuning):  {loss_pre:.5f}")
print(f"loss at full fine-tuning:                       {loss_full:.5f}")

# Constrained fine-tuning: W(θ) = W_pre + reshape(P @ θ, (n, n))
# where P is a random projection of size n*n × d.
def constrained_finetune(d, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n*n, d))
    # Minimize ||X(W_pre + reshape(Pθ)) - y_B||²
    # Define linear map θ → X @ reshape(Pθ, (n,n))
    M = np.zeros((m * n, d))
    for col in range(d):
        delta = P[:, col].reshape(n, n)
        M[:, col] = (X @ delta).ravel()
    target = (y_B - X @ W_pre).ravel()
    theta, *_ = np.linalg.lstsq(M, target, rcond=None)
    delta = (P @ theta).reshape(n, n)
    W_constrained = W_pre + delta
    return np.mean((X @ W_constrained - y_B)**2)

dims = [1, 2, 4, 8, 16, 32, 64, 128, 256]
losses = [constrained_finetune(d) for d in dims]

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.semilogy(dims, losses, color='#FF007F', linewidth=2.6,
            marker='o', markersize=8, label='constrained fine-tuning loss')
ax.axhline(loss_pre, color='#00A8A8', linewidth=2, linestyle='--',
           label=f'pre-trained loss = {loss_pre:.4f}')
ax.axhline(loss_full, color='#FFD700', linewidth=2, linestyle='--',
           label=f'full fine-tuning loss = {loss_full:.4f}')
ax.set_xlabel('intrinsic dimension $d$', fontsize=11)
ax.set_ylabel('loss on Task B (log scale)', fontsize=11)
ax.set_title("Random-projection fine-tuning: the curve flattens fast", fontsize=12)
ax.set_xscale('log')
ax.legend(loc='upper right', fontsize=9.5, frameon=True, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.3, which='both')
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

# Print the intrinsic dimension at 90% accuracy
gap = loss_pre - loss_full
threshold = loss_full + 0.10 * gap
print(f"\n90% recovery threshold: {threshold:.5f}")
for d, ell in zip(dims, losses):
    mark = "*" if ell <= threshold else " "
    print(f"  d = {d:>3}  loss = {ell:.5f}  {mark}")
```

Stare at the pink curve. The model with $d=4$ — *four* trainable scalars composed with a frozen $256 \times 256 \times 4$ random tensor — already covers most of the gap between the pre-trained and the fully-fine-tuned models. By $d=16$ we are essentially at the bottom.

The true update in this toy is rank-4 by construction. The experiment finds that $d \approx 4$ is enough — *even though the optimizer doesn't know that and $P$ is a random projection unrelated to the true subspace*. This is the Aghajanyan claim, played at toy scale.

## Why Does The Random Projection Work?

Here is the part that should bother you. The matrix $P$ is *random*. The "true" low-rank subspace that captures the fine-tuning update is some specific low-dimensional subspace of parameter space — call it the "useful" subspace. A random projection is essentially never going to align *exactly* with the useful subspace. Yet the experiment works. Why?

The mathematical answer comes from the **Johnson-Lindenstrauss lemma** ({{< cite text="Johnson & Lindenstrauss, 1984" url="https://www.semanticscholar.org/paper/Extensions-of-Lipschitz-mappings-into-a-Hilbert-Johnson-Lindenstrauss/76d63d34d8434c12e8e1d3bb33afb9311fff58b6" kind="paper" >}}). Roughly: a random projection of dimension $d$ preserves the distances of $N$ points to within a small distortion, *provided $d > c \log(N) / \varepsilon^2$* for some constants $c, \varepsilon$.

In our setting, the "$N$ points" are the relevant trajectories that gradient descent could take through parameter space. The Johnson-Lindenstrauss bound says we can preserve them all with a random projection of dimension $d$ that grows only *logarithmically* in the number of trajectories. For a fine-tuning task with a small number of "meaningful" gradient directions, a random projection of dimension $d \approx 200$ has a high probability of containing a direction close enough to each of the meaningful ones.

This is, in fact, *why* Aghajanyan's experiment works. The random matrix is not magically aligned with the useful subspace; it is just *high-dimensional enough* that some linear combination of its columns lies close to the useful subspace. The "intrinsic dimension" is therefore an upper bound on the true low-rank dimensionality of fine-tuning — the actual structure could be even smaller.

LoRA exploits this further: instead of using a random fixed $P$, it uses *learned* $A$ and $B$ matrices. The same dimensionality, *learned* alignment, even better results.

## The Larger-Model Paradox, Resolved

The most counter-intuitive finding of the paper — that larger models have smaller intrinsic dimensions — deserves its own explanation.

The intuition Aghajanyan offers in the paper, supported by their measurements across model families, goes like this. Pre-training learns a very rich set of *features*. Larger models learn richer features faster. Fine-tuning for a downstream task is not learning new features — it is *re-weighting* the existing features for the new objective.

For a small model, the pre-trained feature set is incomplete, and fine-tuning has to learn some new features. For a large model, the pre-trained feature set is overcomplete, and fine-tuning only has to do re-weighting. Re-weighting is intrinsically low-dimensional — you can express it as a small matrix acting on the feature gallery.

This *predicts* that:

1. As model size grows, intrinsic dimension *decreases*. (Confirmed across BERT, RoBERTa, BART.)
2. As pre-training data and compute grow, intrinsic dimension decreases *even faster*. (Confirmed in the RoBERTa-vs-BERT comparison.)
3. Tasks closer to the pre-training distribution should have lower intrinsic dimensions than tasks further from it. (Roughly confirmed across the GLUE benchmark in the paper.)

The implication for LLM economics is striking. **The bigger your pre-trained model, the cheaper LoRA gets**, in both rank and total optimizer state. This is the engineering reason QLoRA can fine-tune a 70B model on a single 24GB consumer GPU: at 70B parameters the intrinsic dimension is probably in the dozens, and the optimizer state for a rank-16 LoRA is comically small relative to the base model.

## Connections To The Rest Of The Issue

The Aghajanyan paper does not propose LoRA. It does not even *imply* LoRA in any architectural sense. What it does is *prove the empirical premise* on which LoRA rests — that fine-tuning genuinely happens in a low-dimensional subspace, not just appears to.

The chain of reasoning is:

1. **[Eckart-Young (1936)](../02-svd/):** the best rank-$r$ approximation of any matrix is the truncated SVD.
2. **[Low-rank hypothesis (this issue, ch.3)](../03-low-rank-hypothesis/):** trained weight matrices empirically have small effective rank.
3. **Intrinsic dimension (this chapter):** fine-tuning *updates* are even more low-rank than the base weights — measurably and reliably.
4. **[LoRA (next chapter)](../05-lora/):** therefore, you can parameterize the fine-tuning update as $\Delta W = BA$ with rank $r \ll d$ and pay almost no quality cost.

The chain is tight. The 2020 paper sat in the literature for eight months. Within those eight months, the Aghajanyan result was something every fine-tuning researcher had seen at a conference but nobody had productionalized. Microsoft's LoRA paper is, at one level, exactly that productionalization. They take Aghajanyan's *low-dimensional subspace* observation and replace the random projection with a learned rank-$r$ factorization. Same empirical claim, dramatically better implementation.

## What To Remember

1. **Aghajanyan et al. (Facebook, 2020) measured the *intrinsic dimension* of fine-tuning** by parameterizing the update as $W = W_\text{pre} + P\theta$, $P$ random and frozen, $\theta \in \mathbb{R}^d$ optimized. Sweep $d$ to find the smallest dimension that recovers 90% of full-fine-tuning accuracy.
2. **BERT-large on MRPC: $d_{90} = 207$.** Two hundred parameters out of 110 million. The update space is laughably low-dimensional.
3. **Larger pre-trained models have *smaller* intrinsic dimensions.** A 340M model needs fewer trainable parameters than a 110M model. The bigger the base, the cheaper fine-tuning.
4. **The mechanism is Johnson-Lindenstrauss.** A random projection of dimension $d$ preserves the geometry of $\sim e^d$ trajectories; if the "useful" subspace has dimension $\ll d$, random projection contains enough of it to fine-tune through.
5. **This is the empirical bedrock LoRA stands on.** Without Aghajanyan's measurement, LoRA is a guess; with it, LoRA is the inevitable consequence.

{{% callout type="tangent" title="The convnet precursor" %}}
The random-projection experiment was not Aghajanyan's invention — it was lifted directly from a 2018 paper by **Li, Farkhoor, Jiang and Yosinski** of Uber AI Labs ({{< cite text="Li et al., 2018" url="https://arxiv.org/abs/1804.08838" kind="paper" >}}). The 2018 paper measured intrinsic dimensions for *image-classification convnets* and found surprisingly small numbers (CIFAR-10 fully-convolutional: $d \approx 2900$). Aghajanyan's contribution was to apply the experiment to *pre-trained* language models — where the gap between pre-training and fine-tuning is the entire story — and to discover the larger-is-cheaper scaling law. The 2018 paper has 200 citations as of 2026; the 2020 paper has 5,000. The application matters.
{{% /callout %}}

---

**Continue to** → [LoRA](../05-lora/) — Microsoft, June 2021. The same insight, productionalized. The skinny-matrix factorization that ate fine-tuning, plus the implementation detail (Adam's optimizer state) that was actually the largest win.
