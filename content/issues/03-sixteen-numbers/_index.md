---
title: "Sixteen Numbers Walk Into A GPU"
description: "An issue on how LLM weights got 4-bit, why FP4 only has 16 possible values, and why the models somehow stay smart anyway."
issue: 3
layout: issue-cover
theme: cream
math: false
header: default.png
date: 2026-05-13T09:00:00-04:00
---

## The Mystery

Autumn 2022. Meta releases **OPT-175B**: a model with 175 billion parameters. Stored in the standard 16-bit float format of the day, that's **350 gigabytes of weights**. To run inference at a reasonable speed you need 8 × A100 80GB GPUs, networked together. The hardware cost alone is north of $80,000. Most researchers can't touch it.

Six months later, a hobbyist with a single $1,500 consumer GPU is generating text from it. Six months after *that*, people are running 70-billion-parameter models on **phones**. By 2025, NVIDIA's flagship silicon supports a number format called **FP4** — a "floating point" number with exactly **2⁴ = 16** distinct possible values — and frontier labs are *training* in it. The loss curves barely flinch.

What did we do to the numbers?

The answer turns out to involve **a 1992 paper on brain surgery**, **a 1957 Bell Labs memo about analog telephones**, a quirk in the statistical distribution of weights that nobody predicted, and a quiet revolution in what we even *mean* by "a number". This issue walks the dependency tree of the mathematical ideas that made it possible.

## How To Read This Issue

Start with the [cold open](01-cold-open/) — it sets the stakes and introduces the cast. After that, the **tech tree** below is the table of contents. Each node is an article. Arrows show which articles build on which: if you follow them upward, you walk from the deepest mathematical foundations to the modern algorithms that exploit them.

If you already know IEEE 754 and rate-distortion theory, skip the primers and stay on the mainline (pink nodes). If you want the full James Burke treatment — connections between Bell Labs and your iPhone, between 1990s pruning theory and DeepSeek — read everything in numerical order.

{{< techtree name="issue03" >}}

## What You Will Walk Away With

By the end of this issue you should be able to answer, with confidence and napkin math:

- Why FP16 is *not* simply "half of FP32" but a fundamentally different number system.
- What the **Lloyd-Max** bargain is, and why nobody actually uses uniform quantization for anything that matters.
- Why "just round the weights to 4 bits" *breaks* an LLM — and what the **1%** of activations doing the breaking actually looks like.
- How a 1992 paper on **pruning neural networks** silently became the workhorse algorithm of LLM quantization in 2022.
- Why **K** and **V** in your transformer's KV cache need *different* quantization schemes.
- What the **OCP microscaling** standard is, why **NVIDIA Blackwell** bet the farm on it, and what it means that "FP4" is now a number format your GPU has hardware support for.

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
