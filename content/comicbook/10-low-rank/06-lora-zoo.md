---
title: "LoRA Zoo: QLoRA, DoRA, AdaLoRA and the family"
short_title: "LoRA Zoo"
description: "After LoRA, the variants. QLoRA quantizes the base. DoRA splits magnitude from direction. AdaLoRA reallocates rank between layers. VeRA shares the random factors. A field guide to the half-dozen tricks that mattered."
blurb:
  - "QLoRA (Dettmers et al., 2023): NF4-quantize the frozen base. Fine-tune Llama-65B on a single 48 GB GPU."
  - "DoRA (Liu et al., 2024): split each LoRA into a magnitude scalar and a unit-direction LoRA. Higher rank-for-rank quality."
  - "AdaLoRA (Zhang et al., 2023): use SVD parametrization and prune small singular values during training. Spend rank where it helps."
  - "Eight variants, one taxonomy. Read the table; pick the one that matches your bottleneck."
topics: [low-rank, fine-tuning, quantization]
tags: [lora, qlora, dora, adalora, peft, dettmers]
theme: teal
math: true
draft: false
date: 2026-05-19T09:00:00-04:00
issue: 10
weight: 60
techKind: primer
techNode: lora-zoo
header: default.webp
---

## What LoRA Got Wrong (Or At Least, Did Not Solve)

LoRA shipped in June 2021 and was instantly the new default. By the end of 2022 it was clear that the technique had some specific weaknesses. Three of them attracted enough attention to spawn entire follow-up papers.

1. **The base weights still live in fp16.** LoRA collapses the optimizer state to nothing, but the base model is still 14 GB for Llama-7B and 130 GB for Llama-65B. You cannot fine-tune Llama-65B on one consumer GPU with vanilla LoRA — the *frozen* weights don't fit.
2. **All matrices get the same rank.** LoRA assigns the same $r$ to $W_Q$, $W_V$, every layer. But some layers need more rank than others — the early attention layers in BERT are higher-rank than the deeper ones, the ratio of attention-to-MLP importance varies between tasks. A uniform $r$ wastes parameters on layers that don't need them and starves layers that do.
3. **The two factors $A$ and $B$ are independent.** This is a feature for expressiveness but a bug for stability. The LoRA update can change the *magnitude* of $W_0$'s columns in ways that disturb the activations through unrelated layers. Many papers report training instabilities when $r$ is large.

Each of those three weaknesses spawned a specific variant. We will walk through them, then briefly survey the rest of the zoo.

## QLoRA — Quantize The Frozen Base

In **May 2023**, **Tim Dettmers** (the same researcher who shipped `bitsandbytes` in [issue 3](/comicbook/03-quantization/01-cold-open/)) publishes **QLoRA** ({{< cite text="Dettmers et al., 2023" url="https://arxiv.org/abs/2305.14314" kind="paper" >}}). The paper has one move: notice that the *base weights are frozen* during LoRA training, so there is no need to keep them in fp16. Quantize them down to **NF4** — the 4-bit normal-float format from the same Dettmers et al. line of work (see [Lloyd-Max](/comicbook/03-quantization/03-lloyd-max/) for the NF4 derivation).

The base model now takes 25% of its original memory. For Llama-65B, that means **17 GB instead of 130 GB**. The LoRA update is still trained in fp16 with fp32 Adam state. The forward pass dequantizes a row of $W_0$ on the fly when it is needed, multiplies, dequantizes the next row, and so on. The dequantization happens inside a fused CUDA kernel — the dequantized values never hit HBM, so the memory savings actually materialize.

Three engineering tricks make QLoRA possible:

1. **NF4 quantization** of the base weights. (Already proven in the original 4-bit-fine-tuning literature.)
2. **Double quantization**: the per-block quantization constants themselves are quantized again. Saves another 0.3 bits per weight on average.
3. **Paged optimizers**: the Adam state pages between CPU and GPU memory when the GPU runs low. Slows things down a hair but makes the impossible possible.

The headline result of the QLoRA paper: **fine-tune Llama-65B on a single 48 GB GPU**. Or, more vividly, fine-tune Llama-7B on a 16 GB consumer card. Or fine-tune Llama-13B on a 24 GB RTX 3090. QLoRA put fine-tuning of frontier-scale models into the hands of hobbyists in a single weekend's release.

```pyplot {id="qlora-memory" caption="LoRA vs QLoRA memory footprint across model scales. The LoRA training cost is dominated by the frozen base in fp16; QLoRA cuts that 4×. The Llama-65B bar at 48 GB is the consumer-GPU win."}
sizes = [7, 13, 33, 65]
labels = [f"Llama-{s}B" for s in sizes]

# Memory components (in GB) — fp16 base + LoRA Adam state + activations (rough)
base_fp16 = [s * 2 for s in sizes]
base_nf4  = [s * 0.5 for s in sizes]
lora_extra = [0.5, 0.7, 1.2, 1.8]  # LoRA adam state + activations + grad scaling
acts = [8, 10, 14, 18]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(sizes))
w = 0.36

# LoRA bar (left): base fp16 + LoRA state + activations
ax.bar(x - w/2, base_fp16, w, color='#1A1A1A', edgecolor='#1A1A1A',
       label='base weights')
ax.bar(x - w/2, lora_extra, w, bottom=base_fp16, color='#FF007F',
       edgecolor='#1A1A1A', label='LoRA params + Adam state')
ax.bar(x - w/2, acts, w, bottom=np.array(base_fp16)+np.array(lora_extra),
       color='#FFD700', edgecolor='#1A1A1A', label='activations')

# QLoRA bar (right): base NF4 + LoRA state + activations
ax.bar(x + w/2, base_nf4, w, color='#1A1A1A', edgecolor='#1A1A1A',
       hatch='///')
ax.bar(x + w/2, lora_extra, w, bottom=base_nf4, color='#FF007F',
       edgecolor='#1A1A1A', hatch='///')
ax.bar(x + w/2, acts, w, bottom=np.array(base_nf4)+np.array(lora_extra),
       color='#FFD700', edgecolor='#1A1A1A', hatch='///')

# Consumer GPU markers
ax.axhline(24, color='#00A8A8', linestyle='--', linewidth=1.5,
           label='RTX 3090 (24 GB)')
ax.axhline(48, color='#FF8C00', linestyle='--', linewidth=1.5,
           label='A6000 (48 GB)')

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('memory (GB)')
ax.set_title('LoRA (left bar) vs QLoRA (hatched right bar) — fine-tune memory budget')
ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='#1A1A1A')
ax.grid(axis='y', alpha=0.3)
for spine in ['top','right']: ax.spines[spine].set_visible(False)

print(f"{'model':>10}  {'LoRA':>7}  {'QLoRA':>7}  {'on RTX 3090?':>12}")
for s, l16, l4, le, a in zip(sizes, base_fp16, base_nf4, lora_extra, acts):
    lora_total = l16 + le + a
    qlora_total = l4 + le + a
    fits = "yes" if qlora_total <= 24 else ("48GB" if qlora_total <= 48 else "no")
    print(f"  Llama-{s:2d}B  {lora_total:>4.0f} GB  {qlora_total:>4.0f} GB  {fits:>12}")
```

The picture is brutal. LoRA Llama-65B at ~140 GB — multi-GPU. QLoRA Llama-65B at 38 GB — fits on a 48 GB consumer card. Same accuracy on virtually every benchmark Dettmers measured.

## DoRA — Decompose Magnitude From Direction

In **February 2024**, a team led by **Shih-Yang Liu** at NVIDIA publishes **DoRA: Weight-Decomposed Low-Rank Adaptation** ({{< cite text="Liu et al., 2024" url="https://arxiv.org/abs/2402.09353" kind="paper" >}}). The paper makes an observation that, in retrospect, is obvious: when LoRA updates a weight matrix, it is *simultaneously* changing the magnitudes of the column vectors and their directions. Maybe those should be handled separately.

DoRA decomposes the weight matrix into a *magnitude vector* $\mathbf{m} \in \mathbb{R}^{d_\text{out}}$ (one scalar per output dimension) and a *direction matrix* $V \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ whose columns are unit vectors:
$$
W \;=\; \mathbf{m} \odot \frac{V}{\Vert V \Vert_c}
$$
where $\Vert V \Vert_c$ denotes column-wise norms. DoRA then trains:
- $\mathbf{m}$ directly (it's a small vector, $d_\text{out}$ parameters).
- $V$ via a LoRA-style low-rank update: $V = V_0 + BA$.

The split lets the optimizer change magnitude (a cheap, low-dimensional adjustment) without entangling it with directional changes (the higher-dimensional, structurally richer LoRA part). The reported result: DoRA at rank $r$ matches the accuracy of LoRA at rank $2r$, with only a small parameter-count increase from the $\mathbf{m}$ vector.

The picture in the paper showing the magnitude-vs-direction trajectory of LoRA's updates is striking: vanilla LoRA wiggles uselessly in magnitude even when only direction matters. DoRA pins magnitude and lets direction move.

## AdaLoRA — Re-Allocate Rank Per Layer

In **March 2023**, a team led by **Qingru Zhang** at Georgia Tech and Microsoft publishes **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning** ({{< cite text="Zhang et al., 2023" url="https://arxiv.org/abs/2303.10512" kind="paper" >}}). Their move: instead of fixing the rank $r$ uniformly across all LoRA-adapted matrices, learn it.

AdaLoRA parameterizes each adapter as a proper SVD:
$$
\Delta W \;=\; P \,\Lambda\, Q
$$
where $P \in \mathbb{R}^{d_\text{out} \times r}$ has orthonormal columns, $Q \in \mathbb{R}^{r \times d_\text{in}}$ has orthonormal rows, and $\Lambda$ is a diagonal matrix of "singular values". During training, AdaLoRA:

1. Periodically scores the importance of each diagonal entry of $\Lambda$ across all layers using a gradient-based saliency metric.
2. Prunes the smallest entries globally.
3. The result is that high-importance layers retain high rank; low-importance layers shrink to rank 2 or 3.

The headline result: with a *total* parameter budget equivalent to LoRA-$r=8$ across all layers, AdaLoRA can spend rank-32 on the layers that need it and rank-1 on the layers that don't, beating uniform LoRA by ~1.5 points on GLUE.

This is a *direct* application of the Eckart-Young theorem from [chapter 2](../02-svd/): AdaLoRA's parameterization is literally the SVD outer-product expansion, and pruning the smallest $\lambda_i$ is exactly Eckart-Young's "drop the tail" recipe applied per-layer.

## The Smaller Variations

Beyond the big three (QLoRA, DoRA, AdaLoRA), the published variations are too numerous to cover in depth. A taxonomy:

### Sharing parameters between adapters

- **VeRA** ({{< cite text="Kopiczko et al., 2024" url="https://arxiv.org/abs/2310.11454" kind="paper" >}}): the $A$ and $B$ matrices are *shared across layers* and frozen randomly. Only small scaling vectors are learned per-layer. Parameter count drops another 10×.
- **Tied-LoRA** (Renduchintala et al., 2023): $A$ and $B$ are tied across layers but in a learned, not random, way.

### Better initialization

- **PiSSA** ({{< cite text="Meng et al., 2024" url="https://arxiv.org/abs/2404.02948" kind="paper" >}}): initialize $A, B$ from the *top* singular vectors of $W_0$, not from random / zero. Training converges faster.
- **MiLoRA**: initialize $A, B$ from the *bottom* singular vectors of $W_0$. The intuition is that the top singular directions are the "essential" pre-trained knowledge that should be preserved; learn updates orthogonal to them.

### Combining with sparsity

- **LoSparse** (Li et al., 2023): $\Delta W = BA + S$ where $S$ is a sparse matrix. Captures the parts of $\Delta W$ that don't fit a clean low-rank.
- **HiRA** (Huang et al., 2024): hierarchical low-rank — chunks of $W$ get their own rank-$r$ adapters.

### Across modalities

- **DyLoRA** (Valipour et al., 2022): train multiple ranks simultaneously, sample randomly at each step. Inference time picks the best $r$ per task.

The PEFT library at Hugging Face supports a dozen of these. The most common production stack as of 2026 is **QLoRA + DoRA**, with the base in NF4 and DoRA's magnitude-direction split applied on top.

## A Decision Tree For Picking The Variant

| If your problem is… | Use |
|---|---|
| The base model doesn't fit in GPU memory at all | **QLoRA** — quantize base to NF4 |
| You want the best per-parameter quality | **DoRA** — magnitude/direction split |
| Some layers matter much more than others (RLHF, instruction tuning) | **AdaLoRA** — global rank reallocation |
| You're training many adapters and want them tiny | **VeRA** — share factors across layers |
| Fast convergence matters more than final accuracy | **PiSSA** — top-singular-vector init |
| You want to mix with other PEFT methods | Just use HuggingFace's PEFT library — it composes |

## Connections Back To The Theory

Every entry in the zoo is some perturbation of the LoRA factorization. Crucially, none of them break the *Eckart-Young justification* — they all assume the update is low-rank in some axis. They differ on:

- Which matrix gets factored (the update $\Delta W$, or the base $W_0$ itself in the PiSSA init?)
- How rank is allocated (uniform, learned, hierarchical)
- What is shared vs. separate (per-layer, cross-layer, magnitude-vs-direction)
- What numerical format the base sits in (fp16, NF4, FP8)

Notice that **MLA**, the subject of [the next chapter](../07-mla-bridge/), can be viewed as a particular point in this design space — one where the factorization is applied to the *attention projections themselves* (not their updates), the rank is *fixed at the start of training*, and the factorization is *learned together with the rest of the model from scratch*. It is, in this sense, the LoRA Zoo's most aggressive entry, with the LoRA philosophy taken from "edit the model after training" all the way to "edit the architecture before training".

The boss chapter, [The Unified Low-Rank Thesis](../08-unified-thesis/), will draw the entire chart.

## What To Remember

1. **QLoRA quantizes the frozen base to NF4** — fine-tune Llama-65B on a single 48 GB GPU. The base shrinks 4×; LoRA updates stay fp16.
2. **DoRA splits magnitude from direction** — magnitude is a small vector trained directly; direction is the LoRA-style low-rank update. Rank-for-rank, DoRA beats vanilla LoRA.
3. **AdaLoRA parameterizes as $P\Lambda Q$ and prunes small $\lambda_i$ globally during training**. The Eckart-Young theorem applied per-layer with a learned budget.
4. **The PEFT library at Hugging Face implements all of these** — the common production stack is QLoRA + DoRA.
5. **All zoo entries are perturbations of the same factorization assumption.** Eckart-Young is doing all the work; the variants differ on engineering details (rank allocation, init, sharing, quantization of the base).

{{% callout type="tangent" title="What about full fine-tuning?" %}}
For very small models (under 1B parameters) or very large datasets (over 1M examples), full fine-tuning still wins on accuracy by 1–3 points. The LoRA zoo dominates the *cost/quality frontier* in the LLM regime — bigger models, smaller datasets, more diverse downstream tasks. The crossover point as of 2026 sits around 1B parameters: under that, full FT; over it, QLoRA + DoRA. The frontier moves leftward each year as quantization gets better.
{{% /callout %}}

---

**Continue to** → [MLA, Re-Read As A LoRA](../07-mla-bridge/) — the LoRA factorization, applied not to the *update* but to the *base attention weights themselves*, baked into the architecture from initialization. DeepSeek's contribution to the same conversation, with the same algebra.
