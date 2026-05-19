---
title: "LoRA: ten million trainable params on a 175-billion model"
short_title: "LoRA"
description: "Microsoft, June 2021. Freeze the base model. Add a rank-r residual decomposed as BA. Trainable parameters drop 10,000×, optimizer memory drops with them, accuracy holds. The fine-tuning recipe that ate the open-source world."
blurb:
  - "Edward Hu et al. (Microsoft, 2021): freeze $W$, learn $\\Delta W = BA$ with $A \\in \\mathbb{R}^{r \\times d}$, $B \\in \\mathbb{R}^{d \\times r}$, $r \\ll d$."
  - "GPT-3 175B: 175,000M total params → 4.7M trainable. A 37,000× reduction. Roughly 3× less GPU memory at train time."
  - "The dominant savings isn't the trainable weights — it's the *Adam optimizer state*, which is 12 bytes per trainable parameter in fp32."
  - "At inference: fold $BA$ back into $W$. The runtime model has zero extra params. LoRA is free-at-inference fine-tuning."
topics: [low-rank, fine-tuning, training]
tags: [lora, hu, microsoft, adam, optimizer-state]
theme: cream
math: true
draft: false
date: 2026-05-19T09:00:00-04:00
issue: 10
weight: 50
techKind: mainline
techNode: lora
header: default.webp
---

## Redmond, June 17, 2021

It is the afternoon before the **LoRA** paper goes onto arXiv. Edward Hu, the first author, is sitting in a video call with his coauthors **Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen** — Microsoft Research and Microsoft's Azure AI team, distributed across Redmond, Beijing, and Bellevue. They are arguing about the abstract.

The pre-print includes a number that makes them nervous. The number is **0.0027%**.

That is the fraction of the 175-billion parameters of GPT-3 that LoRA trains during fine-tuning. The other 99.9973% stay frozen at their pre-trained values. The accuracy on every benchmark they have measured matches or beats full fine-tuning.

The team is nervous because the prior literature on parameter-efficient fine-tuning had been a wasteland. **Adapter layers** (Houlsby et al., 2019) added small bottleneck modules — they worked but slowed inference. **Prefix tuning** (Li & Liang, 2021) prepended learnable tokens — works for some tasks, breaks on others. **BitFit** (Ben-Zaken et al., 2021) tuned only the bias terms — too restrictive. Every paper had its asterisk.

LoRA is conspicuously light on asterisks. It works on GPT-3 175B (zero of the seven authors have access to GPT-3's weights — they fine-tune RoBERTa and DeBERTa themselves and then collaborate with the OpenAI team for the GPT-3 ablations). It works across NLU and NLG. It doesn't add any inference latency.

The team pushes the paper anyway ({{< cite text="Hu et al., 2021" url="https://arxiv.org/abs/2106.09685" kind="paper" >}}). Eighteen months later, every open-source fine-tuning recipe on GitHub is built around it. The Hugging Face PEFT library is essentially a LoRA wrapper. Apple ships LoRA-quantized adapters for on-device personalization in iOS 18. The technique is, by 2024, the *default* mode of LLM fine-tuning.

What is the actual math?

## The Reparameterization

The traditional fine-tuning recipe is: take a pre-trained model with weight matrices $W_0$, run more gradient descent, end up with new weight matrices $W = W_0 + \Delta W$ where $\Delta W$ is the update accumulated during fine-tuning. You optimize all entries of $\Delta W$ directly.

LoRA's reparameterization is one substitution. Replace $\Delta W$ with the product of two skinny matrices:
$$
\Delta W \;=\; B A
\qquad \text{where} \qquad
A \in \mathbb{R}^{r \times d_\text{in}},\quad
B \in \mathbb{R}^{d_\text{out} \times r},
\quad r \ll \min(d_\text{in}, d_\text{out})
$$
The fine-tuned model's effective weight matrix is now
$$
W \;=\; W_0 \;+\; B A.
$$
You only optimize $A$ and $B$. The base $W_0$ stays frozen. At forward time:
$$
y \;=\; W x \;=\; W_0 x \;+\; B (A x).
$$
The right-hand side computes $A x$ first (a length-$r$ vector), then $B$ times that (back to length $d_\text{out}$). Total compute per token: $r (d_\text{in} + d_\text{out})$ floating-point operations on top of the frozen base.

That is the entire architectural change. Two lines of code in PyTorch.

```python
# LoRA in PyTorch — for one weight matrix.
A = nn.Parameter(torch.randn(r, d_in) * 0.01)   # Gaussian init
B = nn.Parameter(torch.zeros(d_out, r))         # zero init — Δ W = 0 at start

def lora_forward(x, W_frozen):
    return x @ W_frozen.T + (x @ A.T) @ B.T
```

Two things worth noticing in those lines.

First, $B$ starts at *zero*. This means at step 0, $\Delta W = BA = 0$, so the LoRA-fine-tuned model is identical to the pre-trained model on every input. Training begins from a controlled starting point — no random initialization shock.

Second, $A$ starts at *small Gaussian noise*. This gives the gradient signal somewhere to flow into both $A$ and $B$ in the first few steps. (If both started at zero, $\partial \mathcal{L} / \partial B = 0$ and the optimizer would never move.)

## How Much Memory This Actually Saves

The official LoRA paper headline is "10,000× fewer trainable parameters". The number is technically true. It is *not* the most important number.

The most important number is the **optimizer state**.

Modern LLM training uses **Adam** (or AdamW), which stores, per trainable parameter:

- The parameter itself, in fp32 (for gradient stability).
- The first moment (running mean of gradients), in fp32.
- The second moment (running variance of gradients), in fp32.

That's **12 bytes per trainable parameter** in Adam's natural state. (Some implementations save more.) Plus the weight itself, plus its gradient.

For Llama-7B fine-tuning, the full breakdown:

| Component | Bytes per param | 7B params | Total |
|---|---|---|---|
| fp16 weights | 2 | 7B | 14 GB |
| fp16 gradients | 2 | 7B | 14 GB |
| fp32 weights (Adam) | 4 | 7B | 28 GB |
| fp32 first moment | 4 | 7B | 28 GB |
| fp32 second moment | 4 | 7B | 28 GB |
| **Total optimizer + grad** | **14** | **7B** | **98 GB** |
| Activations (batch 4, seq 2k) | — | — | ~20 GB |
| **Grand total** | — | — | **~118 GB** |

That requires multiple H100s networked together. One graduate student cannot fit Llama-7B fine-tuning on a single GPU.

Now switch to LoRA with rank $r = 8$ on all attention projections. The trainable parameters drop from 7B to roughly **3 million**.

| Component | Bytes per param | Params trained | Total |
|---|---|---|---|
| fp16 base weights (frozen) | 2 | 7B | 14 GB |
| fp16 LoRA weights | 2 | 3M | 6 MB |
| fp32 LoRA + Adam state | 12 | 3M | 36 MB |
| **Total optimizer + grad** | — | — | **~42 MB** |
| Activations (batch 4, seq 2k) | — | — | ~20 GB |
| **Grand total** | — | — | **~34 GB** |

That fits on a single 40GB A100. The base-model weights are still 14 GB and unavoidable, but the optimizer state — the part that scaled with model size — collapses to almost nothing.

```pyplot {id="lora-memory" caption="Where the GPU memory goes during Llama-7B fine-tuning. Top: full fine-tuning; the optimizer state dominates. Bottom: LoRA with r=8; the optimizer state vanishes and the base weights re-emerge as the bottleneck. The black band is fixed; the colored bands shrink."}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

categories = ['base weights\n(fp16, frozen-ish)',
              'gradients\n(fp16)',
              'fp32 weights\n(Adam shadow)',
              'Adam m\n(first moment)',
              'Adam v\n(second moment)',
              'LoRA params\n+ Adam state',
              'activations\n(batch 4, seq 2k)']
colors = ['#1A1A1A', '#00A8A8', '#FFD700', '#FF8C00', '#FF007F', '#FF007F', '#00A8A8']

# Full fine-tuning (GB)
full = [14, 14, 28, 28, 28, 0, 20]
# LoRA (GB) — base is unchanged, gradients vanish for the base, optimizer state vanishes.
lora = [14, 0.006, 0, 0, 0, 0.04, 20]

x = np.arange(len(categories))
axes[0].bar(x, full, color=colors, edgecolor='#1A1A1A', linewidth=1.0)
axes[0].set_title("Full fine-tuning Llama-7B\n(total: ~118 GB)", fontsize=11)
axes[0].set_ylabel("memory (GB)")
axes[0].set_xticks(x); axes[0].set_xticklabels(categories, rotation=35,
                                                ha='right', fontsize=8.5)
for i, v in enumerate(full):
    if v > 0:
        axes[0].text(i, v + 1, f"{v} GB", ha='center', fontsize=8.5)

axes[1].bar(x, lora, color=colors, edgecolor='#1A1A1A', linewidth=1.0)
axes[1].set_title("LoRA r=8 fine-tuning Llama-7B\n(total: ~34 GB)", fontsize=11)
axes[1].set_xticks(x); axes[1].set_xticklabels(categories, rotation=35,
                                                ha='right', fontsize=8.5)
for i, v in enumerate(lora):
    if v >= 0.001:
        if v > 1:
            label = f"{v:.0f} GB"
        elif v >= 0.1:
            label = f"{v:.1f} GB"
        else:
            label = f"{v*1000:.0f} MB"
        axes[1].text(i, max(v, 0.5) + 1, label, ha='center', fontsize=8.5)

for ax in axes:
    ax.set_ylim(0, 36)
    for spine in ['top','right']: ax.spines[spine].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

print("memory budgets:")
print(f"  full fine-tune total: {sum(full):.0f} GB")
print(f"  LoRA r=8 total:      {sum(lora):.1f} GB")
print(f"  shrink:               {sum(full)/sum(lora):.1f}×")
```

The full-FT bar chart on the left is dominated by the three big yellow/orange/pink columns of Adam optimizer state. On the right, those columns are *gone*. The remaining bottleneck is the frozen base weights — which you cannot shrink with LoRA alone (you can with **QLoRA** — see [the next primer](../06-lora-zoo/)).

This is the headline that mattered. LoRA's contribution to fine-tuning was not "fewer trainable parameters" — it was "the Adam optimizer state collapses to nothing".

## Why It Works: Aghajanyan's Theorem, Made Practical

LoRA's empirical premise is exactly the [intrinsic-dimension result](../04-intrinsic-dimension/) of Aghajanyan et al. (2020). Fine-tuning updates are low-rank. The Eckart-Young theorem says that any rank-$r$ matrix can be written as $BA$ with $A$ and $B$ skinny. Therefore, if the *true* low-rank $\Delta W$ lives in some $r$-dimensional subspace, an over-parameterized $BA$ factorization that the optimizer is free to fill in is *capable* of finding that subspace.

The remarkable engineering surprise of LoRA was that the optimizer *does* find it, reliably, even with $r$ much smaller than the value Aghajanyan measured. Aghajanyan needed $d \approx 207$ for BERT-large; LoRA achieves comparable quality with $r = 8$ on most tasks. The gap is because LoRA's $A$ and $B$ are *learned* — they can adapt to align with the useful subspace — whereas Aghajanyan's $P$ is random and has to be high-dimensional to contain the subspace by chance.

The other interesting effect, documented in the LoRA paper itself: LoRA's effective rank as you increase $r$ saturates. Past a certain point, increasing $r$ does not improve accuracy. The optimizer settles into the same low-dimensional subspace regardless. Hu et al. report that going from $r = 8$ to $r = 64$ on GPT-3 produces no measurable improvement on most benchmarks. The fine-tuning task is intrinsically low-dimensional and LoRA finds it.

```pyplot {id="lora-rank-saturation" caption="LoRA accuracy saturates well before $r$ reaches the matrix dimension. The cost in trainable parameters grows linearly; the accuracy gain flatlines past $r \\approx 16$. Synthetic curves matching the qualitative pattern in Hu et al. 2021, Fig. 6."}
ranks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])

# Synthetic accuracy curve: rapid rise then saturation.
# Modeled as a smooth approach to a ceiling determined by the task's intrinsic dim.
intrinsic_dim_eff = 12
acc_full = 0.91
acc_pre  = 0.62
acc = acc_pre + (acc_full - acc_pre) * (1 - np.exp(-ranks / intrinsic_dim_eff))

# Trainable parameter count for one 4096×4096 weight matrix
d = 4096
trainable = 2 * d * ranks
full_count = d * d

fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
ax1.set_xscale('log')

l1, = ax1.plot(ranks, acc * 100, color='#FF007F', linewidth=2.8, marker='o',
               markersize=8, label='downstream accuracy (synthetic)')
ax1.axhline(acc_full*100, color='#00A8A8', linestyle='--', linewidth=1.5)
ax1.text(0.7, acc_full*100 + 0.5, 'full fine-tuning ceiling', fontsize=9, color='#00A8A8')
ax1.axhline(acc_pre*100, color='#1A1A1A', linestyle=':', linewidth=1.0, alpha=0.5)
ax1.text(0.7, acc_pre*100 - 1.7, 'pre-trained baseline', fontsize=9, color='#1A1A1A')

ax1.set_xlabel('LoRA rank $r$', fontsize=11)
ax1.set_ylabel('downstream accuracy (%)', color='#FF007F', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#FF007F')
ax1.set_ylim(55, 95)

ax2 = ax1.twinx()
l2, = ax2.plot(ranks, 100 * trainable / full_count, color='#FFD700', linewidth=2.4,
               marker='s', markersize=7, label='trainable params (% of $d^2$)')
ax2.set_ylabel('trainable params (% of full matrix)', color='#FF8C00', fontsize=11)
ax2.tick_params(axis='y', labelcolor='#FF8C00')
ax2.set_yscale('log')
ax2.set_ylim(0.02, 200)

ax1.legend(handles=[l1, l2], loc='lower right', fontsize=9.5,
           frameon=True, edgecolor='#1A1A1A')
ax1.grid(True, alpha=0.3, which='major')
ax1.set_title('LoRA rank: accuracy saturates, cost keeps climbing', fontsize=12)
for spine in ['top']: ax1.spines[spine].set_visible(False)

print(f"{'rank':>6}  {'accuracy':>9}  {'trainable':>11}  {'% of full':>10}")
for r, a, t in zip(ranks, acc, trainable):
    print(f"  {r:>4}  {100*a:>7.2f}%  {t:>9d}  {100*t/full_count:>8.2f}%")
```

The yellow curve grows linearly with $r$; the pink curve saturates after about $r = 16$. The sweet spot is the knee — accept the small accuracy gap above $r = 16$ and pay 16× fewer trainable parameters than a full-rank update would need. The LoRA defaults across the open-source community settle on $r = 8$ or $r = 16$ for exactly this reason.

## The Inference-Time Trick

When training is done you have a base $W_0$ plus a learned LoRA pair $(B, A)$. At deployment, you can either:

1. **Keep them separate**: serve the model as $W_0 + BA$ via two consecutive matrix multiplies. This adds $r(d_\text{in} + d_\text{out})$ multiplies per token, which for $r = 8$ and $d = 4096$ is about 65k multiplies on top of the 16M multiplies that the base $W_0$ already does. A 0.4% overhead — measurable but small.

2. **Fold them back in**: compute $W_\text{merged} = W_0 + BA$ once, store $W_\text{merged}$, throw away $A$ and $B$. The runtime model is now exactly the shape and cost of the base model. Zero overhead.

Option (2) is what production deployments use when serving a *single* fine-tuned model. Option (1) is what production deployments use when serving *multiple* fine-tuned models on the same hardware — you keep the base $W_0$ once in GPU memory and load different LoRA pairs $(B_i, A_i)$ for different customers. Apple's "personal LoRA adapter" pattern in iOS 18 is exactly this: the base model is in firmware, each user has their own ~50 MB adapter that gets loaded into the same compute.

This is a property neither full fine-tuning nor MLA has. **You can swap LoRA adapters at runtime without restarting the model.** It is a deployment superpower we will return to in the [boss chapter](../08-unified-thesis/).

## Which Matrices To Apply LoRA To

The LoRA paper benchmarks four placements:

- LoRA on $W_Q$ only.
- LoRA on $W_K$ only.
- LoRA on $W_V$ only.
- LoRA on $W_Q, W_V$ (the modal choice in production).
- LoRA on $W_Q, W_K, W_V, W_O$ (all attention projections).

Their conclusion: **applying LoRA to $W_Q$ and $W_V$ recovers most of the benefit** of applying it to all four attention matrices. The Q and V projections are most affected by fine-tuning; K and O matter less. The MLP layers — which are higher-rank empirically (see [ch.3](../03-low-rank-hypothesis/), the MLP-vs-attention digression) — benefit less from LoRA and are often left frozen entirely in production recipes.

This finding has interesting structural implications. The matrix that LoRA targets *first* — the attention $W_Q$ — is the same matrix MLA targets *first* in its absorption trick (see [ch.7](../07-mla-bridge/)). Both methods independently identified that $W_Q$ is the most "composable" matrix in the transformer for low-rank surgery. It is not an accident.

## The Hyperparameter That Matters: Alpha

Buried in the LoRA paper, treated as an implementation detail in the appendix, is a hyperparameter the paper calls $\alpha$. The full LoRA forward pass is actually:
$$
W x \;=\; W_0 x \;+\; \frac{\alpha}{r} B (A x).
$$
The $\alpha/r$ scaling exists for a numerical reason: as you change $r$, you want the *effective magnitude* of the LoRA contribution to stay constant. With $\alpha/r$ scaling, doubling $r$ does not double the update magnitude.

In practice $\alpha$ is set to either $r$ (so the scaling factor is 1, the simplest case) or to $2r$ (slight up-weighting). The most common community convention is $\alpha = r$.

We mention this only because nearly every published LoRA reproduction tunes $\alpha$ as part of the experiment, and it matters. A LoRA model with the wrong $\alpha$ will train slowly or diverge. We will not dwell on this further — the [LoRA Zoo](../06-lora-zoo/) primer surveys the variations.

## The Number That Sticks

The number that comes out of LoRA, and that you should remember from this chapter, is **128×**.

For a Llama-style transformer with $d = 4096$ and rank $r = 16$, the LoRA factorization replaces a $4096 \times 4096 = 16{,}777{,}216$ parameter matrix with two skinny matrices totaling $16 \times 4096 + 4096 \times 16 = 131{,}072$ parameters. That is a shrink ratio of $16M / 131k = 128\times$.

Per layer. Across the whole model. For optimizer state, gradients, and the trainable-parameter count.

For a 32-layer Llama-7B with $r = 16$ LoRA on $W_Q$ and $W_V$:
- Full fine-tune: $2 \times 32 \times 16{,}777{,}216 = 1.07 \times 10^9$ trainable params just on attention.
- LoRA r=16: $2 \times 32 \times 131{,}072 = 8.4 \times 10^6$ trainable params.

That is a **128× reduction in trainable parameter count**, and (by the optimizer-state argument above) a *much larger* reduction in the actual memory footprint of training. The original LoRA paper claims a 3× total memory reduction on GPT-3 175B; in practice, on smaller models, the win is bigger because activations become a larger share of the budget.

## What To Remember

1. **LoRA replaces the fine-tuning update $\Delta W$ with the product $BA$ of two skinny matrices.** $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$, $r$ typically 8–64. The base $W_0$ stays frozen.
2. **The dominant memory savings is the Adam optimizer state**, not the trainable parameters themselves. 12 bytes per trainable parameter in fp32 — collapses by 100×–10,000× depending on $r$.
3. **At inference, you fold $BA$ back into $W_0$ — zero overhead.** Or you keep them separate and swap adapters per request.
4. **It works because fine-tuning updates are empirically low-rank** ([Aghajanyan, 2020](../04-intrinsic-dimension/)). LoRA is the *productionalization* of the intrinsic-dimension finding.
5. **Apply LoRA to $W_Q$ and $W_V$** — that recovers most of the benefit of applying it everywhere. The MLP layers are too high-rank to benefit much.
6. **$\alpha / r$ scaling matters.** A LoRA forward pass is $W_0 x + (\alpha / r) B(Ax)$. Set $\alpha = r$ as a default; tune if accuracy is off.

{{% callout type="tangent" title="Why the deep-learning field needed this paper" %}}
The 2020 Aghajanyan paper had already *proved* fine-tuning is low-rank. So why did LoRA — published eight months later — get all the attention?

Three reasons. First, **LoRA has structure**: $\Delta W = BA$ is a concrete parameterization, not a random projection. Anyone could implement it in PyTorch in five minutes. Second, **LoRA has the optimizer-state win**: random-projection fine-tuning still computes gradients with respect to the full $W$ implicitly, so the Adam state of size 12*|W| bytes is still around. LoRA literally drops it. Third, **LoRA composed with inference**: you can fold $BA$ back into $W$ and pay zero runtime cost. Random-projection fine-tuning cannot.

The Aghajanyan paper measured *why* LoRA would work. The LoRA paper made it work.
{{% /callout %}}

---

**Continue to** → [The LoRA Zoo](../06-lora-zoo/) — a field guide to QLoRA, DoRA, AdaLoRA, VeRA, PiSSA and a dozen other variations, each fixing one of the original LoRA's small remaining inefficiencies.
