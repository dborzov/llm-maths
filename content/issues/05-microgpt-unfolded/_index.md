---
title: "MicroGPT, Unfolded"
description: "A complete LLM inference engine in 60 lines of plain Python — and a line-by-line tour of every concept inside it. The canonical reference implementation for this whole project."
issue: 5
layout: issue-cover
theme: cream
math: false
header: 05-microgpt-unfolded-cover.webp
date: 2026-05-14T02:48:42-04:00
---

## The Mystery

Open any major LLM inference engine — vLLM, SGLang, TensorRT-LLM, llama.cpp — and you will be staring down hundreds of thousands of lines of code. Custom CUDA kernels. Paged attention. Continuous batching. Speculative decoding. Tensor and pipeline parallelism. Buried in there somewhere is the actual neural network, but you will not find it by reading top-down.

So here is the inversion. **Strip every optimization away.** No batching. No broadcasting. No `torch`. No `numpy`. No fused kernels. No autograd. Just a Python script that walks a list of integers through a list of floating-point numbers and prints another integer at the end.

What is left fits in **sixty lines**. It is a complete, working, autoregressive transformer language model. You can step through every line in a debugger. You can `print()` any intermediate value. Every variable has a name short enough to fit on a sticky note.

We call this listing **microGPT**, and it is going to be the *Rosetta stone* for the rest of this project. When a later issue says "the K-channel outliers in `attn_wk` are why naive 4-bit quantization breaks", that sentence has a precise meaning, pointing at a specific line in a specific file. **MicroGPT is the file.**

This issue is the line-by-line tour.

## How To Read This Issue

Start with the [cold open](01-cold-open/) — it prints the full sixty-line listing and explains, paragraph by paragraph, what the code is doing at a *human* level (no math yet).

After that the **tech tree** below is the table of contents. The arrows point UP from prerequisite to user. Read bottom-up if you want strict prerequisite order; read top-down if you want to dive straight into the inference loop and trace dependencies as you hit them.

The capstone — [The Full Forward Pass](16-full-forward/) — re-prints the same listing one more time, but now with every variable, every loop bound, every arithmetic operation annotated with a link to the article that explains it. Think of it as the **index** to the whole project.

Above the capstone, a second wing — the **architecture variants** chapters ([ch.17–21](17-kv-axes/)) — covers the four families of KV-cache-shrinking modifications (GQA, MLA, sliding-window, SSM-hybrids) that became mainstream in 2023–2025. Each one is presented as a *minimal change to the microGPT listing*, so you can see precisely which line gets edited and why.

{{< techtree name="issue05" >}}

## What You Will Walk Away With

By the end of this issue you should be able to point at any line of microGPT and answer, with confidence and a sketch on a napkin:

- What the **state dict** actually contains and how each tensor is shaped. `wte`, `wpe`, `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`, `mlp_fc1`, `mlp_fc2`, `lm_head` — every key, its dimensions, and the operation that consumes it.
- Why the **token embedding** (`wte`) is just a lookup table, why the **positional embedding** (`wpe`) is added rather than concatenated, and what happens if you swap learned positions for sinusoidal or rotary.
- How the **`linear()` helper** — three lines of Python — generalizes to every weight matrix in the model.
- Why modern LLMs use **RMSNorm** instead of LayerNorm, and the single arithmetic difference between them.
- What the **three projections `Q`, `K`, `V`** geometrically *are*, and why splitting one tensor `x` into three lets attention be both a database lookup and a differentiable operation.
- The mechanical, byte-by-byte difference between **`attn_logits`**, **`attn_weights`**, and **`head_out`** — and the role of the **`/ head_dim**0.5`** scale factor.
- Why there are exactly **`n_head` parallel** attention computations and what the **`attn_wo`** output projection is doing afterwards.
- Why every block has **two `x_residual` additions**, and what the residual stream looks like as a horizontal "highway" running through the network.
- What an **MLP block** actually computes (it's an `fc1 → activation → fc2` sandwich) and why the hidden dimension is conventionally 4× the embedding dimension.
- The line-by-line mechanics of the **KV cache**: how `keys[li].append(k)` turns a quadratic computation into a linear one, and exactly which tensors are kept versus recomputed.
- The difference between the **prefill phase** and the **decode phase** of inference — and why almost every modern serving optimization (paged attention, chunked prefill, continuous batching) is a refinement of these two loops.
- What the **LM head** is (a single linear layer with a tied-or-untied vocabulary matrix) and how **temperature**, **top-k**, and **top-p** sampling reshape its output distribution.

---

*Every later issue in this project — quantization, long context, attention variants, anything else — points back to a specific line of microGPT. Read this issue once and the rest of the project becomes a set of footnotes on a listing you already understand.*

**Continue to** → [Sixty Lines, One LLM](01-cold-open/) for the first read-through, no math, no jargon, just the code and the story.
