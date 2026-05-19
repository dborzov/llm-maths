---
title: "LoRA (Low-Rank Adaptation)"
slug: "lora"
description: "LoRA fine-tunes a model by replacing the weight update ΔW with the product BA of two skinny matrices. Trainable parameters drop 100×-10,000×, optimizer memory drops with them. Variants: QLoRA, DoRA, AdaLoRA, VeRA, PiSSA."
category: "transformer"
also_known_as:
  - "LoRA"
  - "Low-Rank Adaptation"
  - "parameter-efficient fine-tuning"
  - "PEFT"
  - "QLoRA"
  - "DoRA"
  - "AdaLoRA"
  - "VeRA"
  - "PiSSA"
  - "MiLoRA"
  - "Tied-LoRA"
  - "LoSparse"
  - "rank-r adapter"
  - "low-rank adapter"
  - "LoRA rank"
  - "alpha scaling"
source_of_truth: "/comicbook/10-low-rank/05-lora/"
source_of_truth_title: "ch.5 LoRA (Issue 10)"
related: ["svd", "low-rank", "attention", "transformer-weights"]
draft: false
---

Replace the fine-tuning update $\Delta W$ with the product of two skinny matrices:
$$
W \;=\; W_0 \;+\; B A
\quad\text{where}\quad
A \in \mathbb{R}^{r \times d_\text{in}},\quad
B \in \mathbb{R}^{d_\text{out} \times r},\quad
r \ll d.
$$

| Component | Standard initialization | Typical rank $r$ |
|---|---|---|
| $A$ | Gaussian noise (small scale) | 4, 8, 16, 32, 64 |
| $B$ | Zero — so $\Delta W = 0$ at step 0 | (same as $A$) |
| $\alpha$ (scaling) | usually $\alpha = r$ — gives unit scaling | tunable |

**Forward pass:** $y = W_0 x + (\alpha/r) B (A x)$ — costs $r(d_\text{in} + d_\text{out})$ extra multiplies. At inference, optionally fold $BA$ back into $W$: zero overhead.

**Dominant savings: optimizer state.** Adam needs 12 bytes/param in fp32 (param + 1st moment + 2nd moment); LoRA shrinks the optimizer state by the parameter-count ratio, not just the trainable params.

**Variants (see [LoRA Zoo](/comicbook/10-low-rank/06-lora-zoo/)):**

- **QLoRA** ({{< cite text="Dettmers 2023" url="https://arxiv.org/abs/2305.14314" kind="paper" >}}) — quantize the frozen base to NF4. Llama-65B fine-tunes on a single 48 GB GPU.
- **DoRA** ({{< cite text="Liu 2024" url="https://arxiv.org/abs/2402.09353" kind="paper" >}}) — split each LoRA into magnitude + direction; rank-for-rank quality win.
- **AdaLoRA** ({{< cite text="Zhang 2023" url="https://arxiv.org/abs/2303.10512" kind="paper" >}}) — parameterize as $P\Lambda Q$ (SVD form), prune small $\lambda_i$ during training.
- **VeRA, PiSSA, MiLoRA** — parameter sharing, smarter initialization.

LoRA exploits the same low-rank mathematics ([Eckart-Young](/comicbook/10-low-rank/02-svd/)) that [MLA](/comicbook/05-microgpt/19-mla/) bakes into the architecture. The two methods are sister applications of one theorem — see [The Unified Low-Rank Thesis](/comicbook/10-low-rank/08-unified-thesis/) for the family tree.
