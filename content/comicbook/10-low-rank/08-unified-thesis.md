---
title: "Low-Rank Thesis: one chart for every compression trick in modern LLMs"
short_title: "Low-Rank Thesis"
description: "LoRA, MLA, GQA, KV pruning, quantization — all five collapse onto a single two-axis chart: when do you exploit low rank, and which kind. The boss view of the entire compression landscape."
blurb:
  - "Eckart-Young (1936) proves the best rank-$r$ approximation; trained transformers (2017–2026) provide the empirical low-rank that makes the theorem useful."
  - "Two-axis taxonomy: *when* you exploit low rank (pre-train / training / fine-tune / inference) × *which* low-rank (magnitude / direction / support / projection)."
  - "Every modern compression trick lives in exactly one cell of the 4×4 grid. LoRA = fine-tune × direction. MLA = pre-train × direction. Quantization = inference × magnitude."
  - "The big open cell — pre-train × support — is the frontier. Sparse expert routing in MoE is the first move into it."
topics: [low-rank, compression, taxonomy, thesis]
tags: [unified, taxonomy, compression, lora, mla, gqa]
theme: teal
math: true
draft: false
date: 2026-05-19T09:00:00-04:00
issue: 10
weight: 80
techKind: boss
techNode: unified-thesis
header: default.webp
---

## The Claim

We have spent six chapters following a single mathematical idea — the rank-$r$ approximation of a matrix — through three decades of papers, three orders of magnitude of model scale, and two different parts of the LLM pipeline. The thread is clean enough that we can stop and write down the thesis directly:

{{% pullquote type="technical" %}}
**The Low-Rank Thesis.** Every meaningful compression technique deployed in modern large language models can be classified as exploiting one of four kinds of low-rank-ness, applied at one of four stages of the model lifecycle. The 4×4 grid that results contains every published method, and the empty cells of the grid point at the next set of research questions.
{{% /pullquote %}}

This chapter draws the grid, fills in the cells, and points at what is missing.

## The Two Axes

### Axis 1: *When* You Exploit Low Rank

Modern LLMs go through four major stages between conception and deployment.

| Stage | When | What it means |
|---|---|---|
| **Pre-train** | Architecture-time | The factorization is *baked into the model* before any data is seen. Initialization-time commitment. |
| **Train** | Optimization-time | The model is dense in shape, but the optimizer's trajectory or gradient computation exploits low rank. |
| **Fine-tune** | Post-pre-train-time | The pre-trained weights are kept; only a low-rank *update* is learned for a downstream task. |
| **Inference** | Deployment-time | The weights are already final; low-rank tricks shrink the inference-time memory or compute footprint. |

These are not strict categories — some techniques span two cells — but they capture the dominant *moment* when the low-rank assumption gets enforced.

### Axis 2: *Which Kind* Of Low Rank

The matrices we want to shrink are low-rank in different senses depending on which method you pick.

| Kind | Symbol | What is "small" |
|---|---|---|
| **Direction** | $\Vert W \Vert_F$ in low-rank basis | Most singular vectors are useless. SVD truncation drops them. |
| **Magnitude** | range of $\{W_{ij}\}$ | Most entries are small. Bits-per-entry can shrink (quantization). |
| **Support** | $\#\{W_{ij} \ne 0\}$ | Most entries are *zero*. Pruning drops them. |
| **Identity** | $W_{ij} \approx W_{kl}$ for some structure | Many entries are repeats or share structure. Weight tying. |

The four kinds are not mutually exclusive — a weight matrix might be low-rank in direction *and* low magnitude *and* somewhat sparse — but the *primary axis of attack* of each compression method tends to be one of them.

## The 4×4 Grid

Now we fill in the grid. Each cell holds one or more methods we have met across the LLM Maths series.

```pyplot {id="unified-grid" caption="The Low-Rank Thesis as a 4×4 grid. Rows are the four kinds of low-rank-ness; columns are the four stages of the model lifecycle. Every published compression method lives in exactly one cell. Pink cells: discussed in this issue. Teal cells: discussed elsewhere on the site."}
import matplotlib.patches as mpatches

stages = ['Pre-train\n(architecture)', 'Train\n(optimizer)', 'Fine-tune\n(adapter)', 'Inference\n(deployment)']
kinds  = ['Direction\n(SVD)', 'Magnitude\n(quantization)', 'Support\n(sparsity)', 'Identity\n(tying)']

# Each cell: list of (method_name, color_class) tuples
cells = [
    # Direction row
    [('MLA',         'pink'),  ('factorized-Adam', 'gray'), ('LoRA',     'pink'), ('SVD weight fold',   'gray')],
    # Magnitude row
    [('FP8 init',    'teal'),  ('mixed precision', 'teal'), ('QLoRA',    'pink'), ('GPTQ, AWQ, NF4',    'teal')],
    # Support row
    [('MoE routing', 'gray'),  ('lottery ticket',  'gray'), ('sparse PEFT', 'gray'), ('GQA, KV pruning, attention sinks', 'teal')],
    # Identity row
    [('weight tying','gray'),  ('cyclic schedule', 'gray'), ('shared LoRA (VeRA)', 'pink'), ('weight reuse', 'gray')],
]
color_map = {'pink': '#FF007F', 'teal': '#00A8A8', 'gray': '#D8D5C8'}

fig, ax = plt.subplots(figsize=(11, 6.4))

cell_w = 1.0
cell_h = 1.0
for i, row in enumerate(cells):
    for j, (name, c) in enumerate(row):
        rect = mpatches.Rectangle((j*cell_w, (3-i)*cell_h), cell_w, cell_h,
                                   facecolor=color_map[c], edgecolor='#1A1A1A',
                                   linewidth=1.4, alpha=0.85)
        ax.add_patch(rect)
        ax.text(j*cell_w + cell_w/2, (3-i)*cell_h + cell_h/2, name,
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                color='#1A1A1A')

# Axis labels
for j, s in enumerate(stages):
    ax.text(j*cell_w + cell_w/2, 4*cell_h + 0.15, s, ha='center', va='bottom',
            fontsize=10, fontweight='bold')
for i, k in enumerate(kinds):
    ax.text(-0.12, (3-i)*cell_h + cell_h/2, k, ha='right', va='center',
            fontsize=10, fontweight='bold')

ax.set_xlim(-1.4, 4.2)
ax.set_ylim(-0.4, 4.6)
ax.set_aspect('equal')
ax.set_axis_off()

# Legend
legend_handles = [
    mpatches.Patch(facecolor='#FF007F', edgecolor='#1A1A1A', label='Discussed in this issue'),
    mpatches.Patch(facecolor='#00A8A8', edgecolor='#1A1A1A', label='Discussed in another issue'),
    mpatches.Patch(facecolor='#D8D5C8', edgecolor='#1A1A1A', label='Frontier / not yet definitive'),
]
ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
          ncol=3, fontsize=9.5, frameon=True, edgecolor='#1A1A1A')

print("the 16 cells (rows = kind, columns = stage):")
for i, row in enumerate(cells):
    print(f"\n{kinds[i].replace(chr(10), ' '):>30}:")
    for j, (name, c) in enumerate(row):
        print(f"  {stages[j].replace(chr(10), ' '):>26} -> {name}")
```

The chart is the boss-level deliverable of this issue. Walk through it row by row.

### Row 1: Direction (SVD low-rank)

This is the row this issue is about. Every cell here exploits the fact that trained weight matrices have *rapidly decaying singular value spectra* — the [low-rank hypothesis](../03-low-rank-hypothesis/).

- **Pre-train × Direction = MLA.** Bake the factorization into the architecture. The K and V projections are *born* as $W_K^\text{up} W_{KV}^\text{down}$, trained that way from initialization. We covered this in [chapter 7](../07-mla-bridge/) and at length in the [microGPT MLA chapter](/comicbook/05-microgpt/19-mla/).

- **Train × Direction = factorized optimizers.** GaLore (Zhao et al., 2024) projects the gradient onto its top singular subspace at each step and runs Adam on the projected gradient. The optimizer state shrinks because it lives in the low-rank gradient subspace. This was big in 2024; we have not covered it on the site in depth.

- **Fine-tune × Direction = LoRA.** [Chapter 5](../05-lora/). The factorization is on the *update* during fine-tuning; the optimizer state for the update collapses to nothing.

- **Inference × Direction = SVD weight fold.** Post-hoc: take a pre-trained model, SVD its weights, keep the top $r$ singular values. The LASER paper (Sharma et al., 2023) is the modern instance. It is the dual of MLA — same factorization but applied to a model that was already trained dense.

### Row 2: Magnitude (quantization)

This is the row that [Issue 3](/comicbook/03-quantization/) is about. The exploit is that weight values have a heavy-tailed distribution; the dynamic range can be summarized with fewer bits.

- **Pre-train × Magnitude = FP8 native init.** Some 2025 frontier models initialize weights in FP8 and train in FP8. The architecture commits to low magnitude precision.

- **Train × Magnitude = mixed precision.** fp16 forward, fp32 accumulator for gradients. Standard since 2017.

- **Fine-tune × Magnitude = QLoRA.** [LoRA Zoo chapter](../06-lora-zoo/). Quantize the frozen base; LoRA stays in fp16.

- **Inference × Magnitude = GPTQ, AWQ, NF4.** The classics. [Issue 3 ch.8](/comicbook/03-quantization/08-brain-surgery/) and ch.9.

### Row 3: Support (sparsity)

The exploit here is that many entries of the relevant matrix are zero (or near-zero). This is the row sparse attention lives in.

- **Pre-train × Support = mixture-of-experts.** MoE chooses a few experts per token. From the per-token view, the active subset of the model is sparse. DeepSeek V2 has $256$ routed experts; only $6$ active per token.

- **Train × Support = lottery ticket pruning.** Iterative magnitude pruning during training. Practical results have been mixed; the strong form of the lottery ticket hypothesis is still empirically contested.

- **Fine-tune × Support = sparse PEFT methods.** Some LoRA variants prune to extreme sparsity, e.g. LoSparse.

- **Inference × Support = GQA, sparse attention, KV cache pruning, attention sinks.** This is the [Issue 7](/comicbook/07-deepseek-attn/) territory — the DSA lightning indexer, CSA token compression, attention sinks, all of these sparsify the attention computation at inference time.

### Row 4: Identity (parameter tying)

The exploit is repetition — entries of the matrix or the model are *the same* as other entries.

- **Pre-train × Identity = weight tying.** Tied input/output embeddings in original GPT-2, factorized embeddings in ALBERT. Half the embedding parameters are shared by definition.

- **Train × Identity = curriculum / cyclic learning-rate schedules.** Weak example — at training time, you're not really tying weights; you might be revisiting the same data in patterns that imply low effective complexity. The cell is mostly empty.

- **Fine-tune × Identity = VeRA, Tied-LoRA.** The LoRA factors $A$ and $B$ are *shared across layers*. [Chapter 6](../06-lora-zoo/).

- **Inference × Identity = explicit weight reuse.** Tied embeddings used at inference; ALBERT-style cross-layer sharing. Not yet common in frontier LLMs.

## What The Grid Says

Stare at the grid for a minute. Several observations.

### 1. The Same Math Recurs

Every cell in the **Direction** row is a direct application of the same Eckart-Young theorem. The cells differ only in *when* you apply it (architecture, optimizer, adapter, post-hoc) and *what* gets factored (the weight, the gradient, the update, the deployed matrix). When LoRA and MLA "rediscovered" the same trick three years apart, they were not rediscovering different math — they were rediscovering different *applications* of the same math, in different stages of the lifecycle.

This is the empirical answer to the cold-open question: *why do we keep finding the same trick?* Because the trick is one theorem and the LLM pipeline has many places to insert it.

### 2. The Cells Are Composable

LoRA composes with QLoRA. MLA composes with GQA. SVD-fold composes with GPTQ. Every cell on the chart is, in principle, multipliable by every other cell, because each one attacks a different axis (a different *kind* of low-rank-ness at a different *stage*).

In practice, certain combinations dominate:

- **QLoRA + DoRA** (column 3): magnitude + direction at fine-tune. The hobbyist's stack as of 2026.
- **MLA + MoE** (column 1): direction + support at pre-train. DeepSeek V2/V3's recipe.
- **GPTQ + GQA + KV quantization** (column 4): magnitude + support at inference. The vLLM stack ([Issue 8](/comicbook/08-vLLM/)).

Reading the grid: when you pick a new compression strategy, you are picking a *path* through the cells. The cells are not alternatives; they are independent dimensions you can stack.

### 3. Some Cells Are Empty

Look at "Pre-train × Support" or "Train × Identity". The cells contain weak entries. These are the frontier.

**Pre-train × Support** would be an architecture that is genuinely *sparse* by design — not just MoE (which is dense per-expert), but a model where most weights are *exactly zero* at initialization and stay that way. The closest published work as of 2026 is Sparsified Language Models (Sun et al., 2025), where structured sparsity patterns are baked into the attention matrix. The cell is the next big open problem.

**Train × Identity** would be an optimizer that *recognizes* when groups of parameters should be tied and ties them mid-training. The cell is essentially empty in the LLM literature — no production model does this. There is some related work in graph neural networks, but the LLM pipeline does not yet exploit this dimension.

### 4. Inference Is The Most Crowded Column

Look at column 4 (inference). It contains GPTQ, AWQ, NF4 (magnitude); GQA, KV pruning, sparse attention (support); SVD weight fold (direction); weight reuse (identity). Every kind of low-rank-ness has a representative.

This is why inference cost has fallen by roughly **10× per year between 2022 and 2026**. The four axes compose. Each axis is exploited by a different research community (LLM.int8 from Dettmers, GQA from Meta, MLA from DeepSeek, weight tying from ALBERT) and they all stack onto the same deployed model.

Training cost has fallen more slowly — the corresponding columns (Pre-train, Train) are sparser on the grid. The frontier of cost reduction in LLMs in 2026 is on the *left* side of the chart: making *training* itself low-rank, the way *inference* already is.

## A Single-Sentence Summary

The thesis, stripped to the bone:

{{% pullquote type="counter-intuitive" %}}
**Every "compression" trick in modern LLMs is an exploitation of low-rank-ness in some matrix in the pipeline. The methods differ on *which* matrix and *when* in the lifecycle the rank is exploited, not on the underlying mathematics.**
{{% /pullquote %}}

The mathematics is Eckart-Young, 1936. The empirical fact that makes it useful is that trained transformer weight matrices have rapidly-decaying singular value spectra. The engineering question — for any new method you propose — is which cell of the 4×4 grid you are in, and whether you are dominating the existing entries of that cell or moving into an empty one.

## Reading Onward

The unified grid points at where to read next on this site.

| If you want to go deeper on… | Read |
|---|---|
| The quantization row (magnitude) | [Issue 3 — Quantization](/comicbook/03-quantization/) |
| The microGPT primer view of attention | [Issue 5 — microGPT](/comicbook/05-microgpt/) |
| MLA at the implementation level | [Issue 5 ch.19 — MLA](/comicbook/05-microgpt/19-mla/) |
| MLA's downstream — DSA, CSA, HCA, sparse attention | [Issue 7 — Sparse Attention](/comicbook/07-deepseek-attn/) |
| KV cache pruning (the inference × support cell) | [Issue 6 — KV-cache pruning](/comicbook/06-kvcache-pruning/) |
| KV cache quantization at deployment | [Issue 9 — KV-cache quant in vLLM](/comicbook/09-kvcache-quant-in-vllm/) |

The arc of LLM Maths Comics, viewed from this thesis, is *the arc of mapping the grid*. Each issue tracks one row or one column. The 4×4 chart of this chapter is the closest the site has yet come to a single picture of the whole landscape.

## What To Remember

1. **One theorem (Eckart-Young) and one empirical fact (trained weights are low-rank) underlie every meaningful compression technique in modern LLMs.**
2. **The methods are organized along two axes:** *when* the low-rank is exploited (pre-train, train, fine-tune, inference) and *which kind* of low-rank (direction, magnitude, support, identity).
3. **LoRA and MLA are the same idea in different cells:** fine-tune × direction and pre-train × direction respectively. The algebra is identical; the *stage* of the lifecycle differs.
4. **The cells are composable.** QLoRA stacks two cells (fine-tune × magnitude + fine-tune × direction). MLA + GQA + KV quant stacks three. Production LLMs end up combining 4–6 cells.
5. **Some cells are empty. They are the frontier.** Pre-train × Support and Train × Identity are the next open problems. The first move into them in 2026 is structured-sparsity initialization and weight-tying-aware optimizers.

{{% callout type="tangent" title="The Burkean punchline" %}}
The James Burke move here is the realization that **two innovations the LLM community celebrates as breakthrough new ideas — LoRA in 2021 and MLA in 2024 — are the same 1936 theorem, applied at different moments in the model lifecycle**. The theorem was proven by Carl Eckart and Gale Young at Yale, in a four-page paper in *Psychometrika*, the year of the Berlin Olympics. The same theorem was used to compress speech in the 1970s, JPEG2000 images in the 1990s, Netflix Prize matrix completion in the 2000s. Every two decades a different research community rediscovers it, applies it to a new substrate, and declares it new. The lesson is not that LoRA and MLA aren't novel — both are genuinely beautiful engineering. The lesson is that *the math is the cheap part*. The expensive part is finding the right place to drop it.
{{% /callout %}}

---

*This is the end of Issue 10. The next issue continues the arc — see the [comic book index](/comicbook/) for what's published.*
