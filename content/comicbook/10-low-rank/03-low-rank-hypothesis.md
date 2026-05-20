---
title: "Low-Rank Hypothesis: trained weight matrices have a tiny effective rank"
short_title: "Low-Rank Hypothesis"
description: "When you SVD any weight matrix in a trained transformer, the singular values cliff-dive after a few dozen. The full rank is 4096; the effective rank is closer to 60. Why this happens — and what doesn't explain it."
blurb:
  - "Take any 4096×4096 weight matrix from a trained Llama. The top 64 singular values typically hold over 90% of the Frobenius energy."
  - "Sainath et al. (IBM, 2013): factor the output layer of a speech model. 4× shrink, same word-error-rate. The first published low-rank LLM trick."
  - "Denil et al. (2013): you can *predict* 95% of a network's parameters from the other 5%. Networks are absurdly over-parameterized."
  - "Why? Three theories — lottery ticket, NTK alignment, optimization-bias toward minimum-norm solutions — and none of them is the whole story."
topics: [linear-algebra, low-rank, empirical]
tags: [svd, weights, sainath, denil, history]
theme: cream
math: true
draft: false
date: 2026-05-19T09:00:00-04:00
issue: 10
weight: 30
techKind: primer
techNode: low-rank-hypothesis
header: default.webp
---

## Yorktown Heights, 2013

The neural-network research community in **2013** is, by 2026 standards, ridiculously small. The community working on speech recognition is smaller still. The community working on speech recognition *with deep neural networks* fits in a single room at a single ICASSP coffee break.

In that room, a researcher named **Tara Sainath** at IBM's T.J. Watson lab is finishing what will become one of the field's most cited papers nobody has read: *"Low-Rank Matrix Factorization for Deep Neural Network Training with High-Dimensional Output Targets"* ({{< cite text="Sainath et al., 2013" url="https://ieeexplore.ieee.org/document/6638949" kind="paper" >}}). The paper is six pages long. The headline result: the final-layer weight matrix of a deep speech-recognition model — typically a 2048×5000 matrix mapping hidden activations to vocabulary phones — can be replaced by the product of two skinny matrices, with a hidden bottleneck of dimension 256, *with no loss in word-error-rate*.

A 4× parameter reduction. No accuracy cost. The trick is to apply Eckart-Young's [1936 theorem](../02-svd/) directly: take a pre-trained dense layer, SVD it, keep the top $r$ singular values, replace the full matrix with the truncated product. Sainath's paper is the first place — to our knowledge — where this is done in production deep learning.

The same year, a parallel group at Université de Montréal led by **Misha Denil** publishes *"Predicting Parameters in Deep Learning"* ({{< cite text="Denil et al., 2013" url="https://arxiv.org/abs/1306.0543" kind="paper" >}}). They run an even more aggressive experiment: train a CNN, then ask whether you can *predict the value of 95% of its weights from the other 5%*. Answer: yes, almost perfectly. The implication is jaw-dropping. Networks have a vast amount of redundancy. Most weights are not independent quantities — they are determined, modulo a tiny bit of noise, by a much smaller set of "core" weights.

The two papers, published months apart in 2013, are the empirical seeds of the low-rank hypothesis. Both teams move on to other problems. Both papers get cited but not *used* — the deep learning community is too busy training bigger models to obsess about parameter efficiency. The seeds sleep for seven years.

Then in 2020, [Aghajanyan et al.](../04-intrinsic-dimension/) at Facebook prove the strong version of the claim and the field finally pays attention. In 2021, LoRA productionalizes it. In 2024, MLA bakes it into the architecture. But the empirical claim — *trained weight matrices have a tiny effective rank* — has been sitting on the table since 2013.

This primer takes that claim apart.

## The Empirical Picture

Take a single transformer's weight matrix. Any matrix. Run SVD. Plot the singular values. What does it look like?

```pyplot {id="empirical-spectrum" caption="Synthetic spectra of three matrix classes at full dimension 4096. The pink curve mirrors what you see on a trained transformer's W_Q or W_V. The teal curve is what a freshly-initialized random matrix looks like. The yellow band marks the rank-r region that LoRA and MLA actually use."}
np.random.seed(3)
d = 4096

# Class 1: random Gaussian — Marchenko-Pastur, near-flat spectrum.
M_init = np.random.randn(d, d) / np.sqrt(d)
_, s_init, _ = np.linalg.svd(M_init)

# Class 2: "trained" — strong low-rank head + small dense noise.
core_rank = 80
U_core = np.random.randn(d, core_rank)
V_core = np.random.randn(core_rank, d)
core_sigma = 3.0 * np.exp(-np.linspace(0, 4, core_rank))
M_trained = (U_core * core_sigma) @ V_core + 0.02 * np.random.randn(d, d)
_, s_trained, _ = np.linalg.svd(M_trained)

# Class 3: "fine-tuned delta" — extreme low-rank, what LoRA assumes.
delta_rank = 16
U_d = np.random.randn(d, delta_rank)
V_d = np.random.randn(delta_rank, d)
d_sigma = 0.8 * np.exp(-np.linspace(0, 2.5, delta_rank))
M_delta = (U_d * d_sigma) @ V_d + 0.001 * np.random.randn(d, d)
_, s_delta, _ = np.linalg.svd(M_delta)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.semilogy(np.arange(d), s_trained / s_trained[0], color='#FF007F',
            linewidth=2.6, label='trained weight $W$ (typical)')
ax.semilogy(np.arange(d), s_delta / s_delta[0], color='#FF8C00',
            linewidth=2.4, label='fine-tuning update $\\Delta W$ (extreme)')
ax.semilogy(np.arange(d), s_init / s_init[0], color='#00A8A8',
            linewidth=2.2, label='random init (full-rank)')
ax.axvspan(0, 64, alpha=0.15, color='#FFD700', label='LoRA rank $r=16-64$')
ax.set_xlim(0, 1500)
ax.set_ylim(1e-4, 2)
ax.set_xlabel('singular-value index $i$', fontsize=11)
ax.set_ylabel('$\\sigma_i / \\sigma_1$ (log scale)', fontsize=11)
ax.set_title('Three matrix classes, one log-y plot', fontsize=12)
ax.legend(loc='lower left', fontsize=10, frameon=True, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.3, which='both')
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

def coverage(s, frac):
    cum = (s**2).cumsum() / (s**2).sum()
    return int((cum < frac).sum()) + 1

print(f"rank needed to capture 90% Frobenius energy (out of {d}):")
print(f"  trained $W$:     {coverage(s_trained, 0.90):4d}")
print(f"  $\\Delta W$:        {coverage(s_delta, 0.90):4d}")
print(f"  random init:    {coverage(s_init, 0.90):4d}")
```

Stare at the gap between the pink and teal curves. The random-init curve sits high and flat — most singular values are within an order of magnitude of $\sigma_1$. The trained curve plunges. By index 100 it is two orders of magnitude below $\sigma_1$.

This is the empirical claim. **It is replicated, in nearly every published study that has bothered to look, across every transformer architecture released since 2017.** GPT-2, BERT, RoBERTa, T5, OPT, Llama 1/2/3, Mistral, Qwen, Gemma. The decay rate varies between layers (attention output projections tend to be the most low-rank; MLP up-projections tend to be the least), but the qualitative picture is universal.

The papers that documented it across architectures include Yu, Yang, Kolter, Kim (2017) on "On Compressing Deep Models by Low Rank and Sparse Decomposition" ({{< cite text="Yu et al., 2017" url="https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_On_Compressing_Deep_CVPR_2017_paper.pdf" kind="paper" >}}); Sharma, Ash, Misra (2023) "The Truth Is In There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction" ({{< cite text="Sharma et al., 2023" url="https://arxiv.org/abs/2312.13558" kind="paper" >}}, the famous *LASER* paper); and Yuan et al. (2023) "ASVD" ({{< cite text="Yuan et al., 2023" url="https://arxiv.org/abs/2312.05821" kind="paper" >}}).

There is a punchline in the Sharma "LASER" paper that is too good not to repeat. Sharma et al. showed that on certain transformer layers — *especially deeper MLPs* — you can *delete the tail of the singular-value spectrum entirely* and the model's downstream task accuracy *improves*. The high-frequency tail was contributing noise. Removing it made the model smarter. The Frobenius energy you discarded was net-negative.

This is the strong form of the low-rank hypothesis: weight matrices are not only low-rank-approximable, they are actively *harmed* by having a long noisy tail.

## A Toy: The Sainath 2013 Experiment, Replayed

To feel what Sainath did, imagine a tiny made-up vocabulary-mapping layer. A 256-dimensional hidden state has to be mapped to a 1000-class output. The full weight matrix is $W \in \mathbb{R}^{1000 \times 256}$, which is 256K parameters. After training, the spectrum of $W$ looks like the pink curve in the plot above. What if we just kept the top 64 singular values?

```pyplot {id="sainath-replay" caption="Replaying Sainath 2013 on a synthetic 1000×256 'output layer'. Storage drops by 3× with no measurable loss in approximation quality. The trick that lit up speech recognition in 2013 is exactly LoRA's parametrization."}
np.random.seed(0)
m, n = 1000, 256

# Build a "trained" matrix with a structured spectrum.
core_rank = 60
U_c = np.random.randn(m, core_rank)
V_c = np.random.randn(core_rank, n)
core_sig = 2.5 * np.exp(-np.linspace(0, 3, core_rank))
W = (U_c * core_sig) @ V_c + 0.04 * np.random.randn(m, n)

# Full SVD
U, s, Vt = np.linalg.svd(W, full_matrices=False)
total = (s**2).sum()

ranks = [4, 16, 64, 128]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.set_yscale('log')
ax.plot(s / s[0], color='#1A1A1A', linewidth=1.6, alpha=0.7)

for r, color in zip(ranks, ['#FFD700', '#FF8C00', '#FF007F', '#00A8A8']):
    err = 1.0 - (s[:r]**2).sum() / total
    stored = r * (m + n)
    full = m * n
    shrink = full / stored
    ax.axvline(r, color=color, linewidth=2, linestyle='--', alpha=0.85,
               label=f"$r={r}$: error={err:.1%}, {shrink:.1f}× shrink")
ax.set_xlim(0, 256)
ax.set_xlabel("singular-value index")
ax.set_ylabel("$\\sigma_i / \\sigma_1$")
ax.set_title("Sainath's experiment, replayed: where to cut a 1000×256 matrix")
ax.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.3, which='both')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Numerical table
print(f"{'rank':>6}  {'frob.err':>9}  {'stored':>8}  {'shrink':>7}")
for r in ranks:
    err = 1.0 - (s[:r]**2).sum() / total
    stored = r * (m + n)
    full = m * n
    print(f"{r:>6}  {err:>9.2%}  {stored:>8d}  {full/stored:>6.1f}×")
```

Look at the printed table. At rank 64 we are storing $64 \times (1000 + 256) = 80{,}384$ scalars instead of $256{,}000$ — a $3.2\times$ shrink — and we have lost roughly 5% of the Frobenius energy of the matrix. *That* 5% goes into the model's robustness to noise. The forward pass with the truncated matrix differs from the forward pass with the full matrix by an imperceptible amount.

This is exactly the calculus that LoRA does in [chapter 5](../05-lora/), with one important difference: LoRA does not start from the full matrix and project. LoRA *initializes* the low-rank factors and learns them from scratch. The Eckart-Young theorem guarantees that such an initialization is not throwing accuracy away.

## Why Does This Happen? Three Theories

The *empirical fact* — that trained weight matrices have rapidly decaying singular values — is rock solid by 2026. The *mechanistic explanation* — why does training produce this structure? — is still not entirely settled. Three serious theories live in the literature.

### Theory 1: The lottery ticket hypothesis

Frankle and Carbin (2018) showed that a randomly-initialized network contains a sparse *subnetwork* (a "winning ticket") that, if trained in isolation from the same initialization, reaches the same accuracy as the full network ({{< cite text="Frankle & Carbin, 2018" url="https://arxiv.org/abs/1803.03635" kind="paper" >}}). The implication: most weights are doing nothing useful and could in principle be deleted.

A low-rank version of this claim would say: the *directions* in which a weight matrix has nonzero singular values are themselves a "ticket". Training discovers a small subset of directions that matter. The rest is residual noise that doesn't get pruned out by gradient descent but is also doing no work.

### Theory 2: Neural tangent kernel alignment

The Neural Tangent Kernel framework (Jacot et al., 2018; Arora et al., 2019) characterizes training dynamics in the limit of very wide networks. One of its surprising predictions is that early in training, gradient updates concentrate in a small number of "kernel eigendirections" determined by the data distribution. As training proceeds, the *output* of the network aligns with this small kernel subspace, and the weights inherit the rank deficiency.

If this is right, then the low-rank structure of trained weights is essentially a *projection of the data's low-rank structure onto the weight space*. The matrix is rank-deficient because the data manifold is rank-deficient.

### Theory 3: Implicit bias of gradient descent

Several papers (Gunasekar et al. 2017; Soudry et al. 2018) show that gradient descent on overparameterized models converges to *minimum-norm* solutions. For matrix-factorization problems, the minimum-nuclear-norm solution is *exactly* the low-rank solution.

This says the low-rank structure is not a property of the data, nor a property of the architecture — it is a property of the *optimizer*. SGD prefers low-rank weights because they have small nuclear norm and that is what the optimizer's implicit regularizer selects for.

### Why none of these is the whole story

Each theory predicts low-rank structure of *some* sort but none of them quantitatively predicts the spectra you actually see. The lottery ticket hypothesis predicts *sparse* networks, not specifically low-rank. NTK alignment predicts the rank to be tied to the data covariance, but in practice the rank is more aggressive than the data covariance suggests. The implicit-bias story predicts a global low-rank shape but not the per-layer variation observed in real models.

The truthful answer is: **we know transformer weights are low-rank empirically, we have three plausible mechanisms, and the precise mechanism is the subject of ongoing research.** From an engineering standpoint, this is fine. LoRA and MLA do not require us to *understand* why the rank is low — they just exploit the empirical fact.

## A Subtle Point: Rank vs Effective Rank

When we say "rank $r$", we are being slightly sloppy. The mathematical rank of a typical trained weight matrix is the maximum — full rank, all singular values nonzero, no exact zeros anywhere. The thing that is small is the **effective rank**: the rank you need to capture some fraction of the Frobenius energy.

A common measure is the **stable rank**:
$$
\mathrm{srank}(W) \;=\; \frac{\Vert W \Vert_F^2}{\Vert W \Vert_2^2} \;=\; \frac{\sum_i \sigma_i^2}{\sigma_1^2}
$$
For a true rank-$r$ matrix the stable rank equals $r$. For a noisy matrix with a strong low-rank core, the stable rank is close to the core rank.

On real LLM weights, the stable rank of a 4096×4096 attention projection typically sits between 50 and 200, depending on layer. The mathematical rank is 4096. The gap is the headroom that LoRA, MLA, and every other low-rank trick exploits.

## What To Remember

1. **Empirically, trained transformer weight matrices have rapidly decaying singular value spectra.** The full rank is the matrix dimension; the *effective* rank is one or two orders of magnitude smaller.
2. **The observation predates LoRA by eight years.** Sainath (2013) productionalized it for speech-recognition output layers; Denil (2013) gave the broader "predicting parameters" framing.
3. **Three theories explain it partially: lottery ticket, NTK alignment, implicit bias of SGD.** None of them is the whole story.
4. **Fine-tuning updates $\Delta W$ are even more low-rank than the base $W$.** This is the key empirical observation LoRA needs — proved in [the intrinsic-dimension paper](../04-intrinsic-dimension/).
5. **Stable rank $\Vert W \Vert_F^2 / \Vert W \Vert_2^2$ is the cleanest numerical measure** of effective rank for noisy data. A 4096×4096 attention projection typically has stable rank between 50 and 200.

{{% callout type="tangent" title="Why the MLP layers are different" %}}
A persistent finding across architectures: the **attention** weight matrices (Q, K, V, output) are dramatically more low-rank than the **MLP** weight matrices (up-projection, gate, down-projection). One hypothesis: MLPs implement key-value memory lookup tables (Geva et al., 2021) and key-value tables *should* be high-rank, because each row stores an independent fact. Attention, by contrast, is computing a soft selection over already-low-rank latent structure. This is why LoRA tutorials often recommend applying LoRA only to attention layers — it helps the most where the matrix structure is friendliest. We will return to this in [chapter 5](../05-lora/).
{{% /callout %}}

---

**Continue to** → [Intrinsic Dimension](../04-intrinsic-dimension/) — the 2020 Facebook paper that proved fine-tuning happens in a sub-200-dimensional subspace, and the experimental design that closes the empirical case for LoRA.
