---
title: "The Great Compression"
description: "October 2022. OPT-175B weighs 350 gigabytes. Six months later, hobbyists run it on $1,500 consumer cards. What did we do to the numbers?"
topics: [quantization]
tags: [opt-175b, dettmers, history]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 10
techKind: mainline
techNode: cold-open
header: default.webp
---

## A 350-Gigabyte Brain

In the early evening of **May 3, 2022**, Meta AI published a model called **OPT-175B**. The name is a description: 175 billion parameters, trained from scratch, weights released under a non-commercial license so that academics could finally poke at a frontier-scale language model. It was the first time anything close to GPT-3's size had been openly released, and the press release made a point of saying so.

But somewhere in Meta's data center, an honest engineer was looking at the file system. Each of the 175 billion parameters was a single 16-bit floating-point number — two bytes — and 175 billion × 2 bytes is:

$$
175 \times 10^9 \times 2 \text{ bytes} = 3.5 \times 10^{11} \text{ bytes} \approx 350 \text{ GB}
$$

A pile of weights big enough to fill seven Blu-ray discs. Or — more painfully — far bigger than any single GPU could hold. The biggest accelerator on the consumer market in 2022 was NVIDIA's **A100 80GB**. To run OPT-175B at all, you needed **eight** of them, networked across NVLink, on a server worth roughly **$80,000 before the cooling bill**.

Most "academic" labs could not afford this. Most individual researchers definitely could not. The dirty joke at the time was that Meta had open-sourced a model that nobody could actually run.

Six months later, the joke was over.

By December 2022 a German graduate student named **Tim Dettmers** had a [GitHub repo](https://github.com/TimDettmers/bitsandbytes) where, with a single command-line flag, OPT-175B would run on **one** A100 — or on a hobbyist's RTX 3090. By 2023 the entire stack had shifted underneath us. By 2024 people were running 70-billion-parameter models on **phones**. By 2025, NVIDIA's flagship silicon was *training* models in a number format with exactly **sixteen** possible values.

This is the story of how that happened, told as a tour of the mathematical ideas that made it possible.

## The First Question: What Is "A Number"?

The standard line you'll hear in any LLM tutorial is: "weights are stored as fp16." Six characters in, three unspoken assumptions:

1. That "a number" naturally takes 16 bits.
2. That those 16 bits should be split into a sign, exponent, and mantissa — IEEE 754's geometry.
3. That floating-point is the right idea in the first place.

Every single one of those assumptions cracks open in the next decade. By the end of this issue, you'll have seen them all be replaced — not because the replacements are "better" in some universal sense, but because of a specific, beautiful argument about *what a neural network actually needs from its numbers*. Spoiler: not very much.

The first crack is the easiest to see: **a 16-bit float is not just "half a 32-bit float"**. It's a totally different geometric object. We pull that idea apart in [Numbers In Boxes](../02-numbers-in-boxes/), the issue's first primer.

## The Second Question: How Do You Throw Away Information On Purpose?

Once you accept that "a number" is a design choice, you face the harder question: given a real-valued weight matrix, *which* discrete number system do you use? Why round to multiples of 0.01 instead of multiples of 0.005? Why use 16 evenly-spaced levels instead of 16 unevenly-spaced ones?

The mathematical name for this question is **quantization**, and it has an *answer* — a beautiful one, from 1957, written by an engineer named **Stuart Lloyd** working on optimizing analog telephone signal levels at Bell Labs. He proved it, declined to publish, then watched a less elegant version of the same theorem get published by Joel Max in 1960. The eventual paper, when it appeared in 1982, became foundational for compressing speech, music, JPEG images — and now, your LLM weights. We tell its story in [The Lloyd-Max Bargain](../03-lloyd-max/), and follow it forward into [the rate-distortion bridge](../04-rate-distortion/) that connects all lossy compression schemes.

## The Third Question: Why Does Naive Quantization Wreck An LLM?

If quantization theory is so well-developed, why does *just rounding* the weights of a 175B model destroy its accuracy?

The answer, when Tim Dettmers found it, was so weird that he spent months convinced he'd made a measurement error.

It turns out that **roughly 0.1% of the activations in a large transformer have values 50× to 100× larger than the rest** — giant, persistent spikes concentrated in a small handful of feature dimensions. These outliers are *systematic* (the same dimensions, batch after batch) and they are *essential* (zeroing them out destroys the model). And — critically — every quantization scheme that works on Imagenet-era CNNs **assumes outliers don't exist**.

The whole field of LLM quantization, post-2022, is in some sense about dealing with this 1%. The story is in [The 1% That Ruins Everything](../06-outliers/), with the [Geometry Of Weights](../05-geometry-of-weights/) primer underneath as the statistical scaffolding.

## The Fourth Question: How Do You Know *Which* Weights Are Safe To Crush?

Suppose you accept that some weights are more important than others. How do you tell which? You could rely on intuition. Or — and this is the great unlock of 2022 — you could ask **the second derivative of the loss function** for each weight.

The mathematical idea (the **Hessian**: the matrix of second-order partial derivatives) has been around since calculus. The specific *algorithmic* application — pruning the least-important weight and *compensating* by adjusting all the others — was published by Babak Hassibi and David Stork in **1992** as the **Optimal Brain Surgeon**. It got modestly cited and then mostly forgotten. Until 2022, when **Elias Frantar and Dan Alistarh** at IST Austria realized that what Hassibi proposed for *pruning* was almost exactly what you wanted for *quantization*. They called it **GPTQ**, and it shipped overnight.

We take the math apart in [Taylor & Hessians](../07-taylor-and-hessians/) (the primer), then watch Hassibi's 30-year-old algorithm light up the entire LLM stack in [Brain Surgery Returns](../08-brain-surgery/).

## The Fifth Question: All Right, So Which Method Do I Actually Use?

By 2024 the menu had six entries — LLM.int8, GPTQ, AWQ, SmoothQuant, QLoRA/NF4, HQQ — each with its own trick, its own trade-off, its own preferred hardware target. [The Method Family Tree](../09-method-family-tree/) lays them side by side, so you can see which trick each one is making, where they agree, and where they actively disagree.

## The Sixth Question: Why Did The KV Cache Become The New Problem?

Here is the irony: once you make weights small, **inference is no longer bottlenecked by weights**. It's bottlenecked by the **KV cache** — the per-token memory of activations the model needs to hand back to itself for self-attention. For long-context inference, the KV cache can be *bigger than the weights themselves*.

But you cannot quantize K and V the same way you quantize weights, for a wonderful, geometric reason that we unpack in [KV Cache Tyranny](../10-kv-cache/), built on [Calibration & Blocks](../11-calibration-and-blocks/).

## The Last Question: What Comes Next?

By 2025, the conversation has moved from software to silicon. NVIDIA's **Hopper** architecture introduced native FP8 support. **Blackwell** went all the way to FP4. The Open Compute Project's **microscaling** specification (the "MX" formats) is becoming an industry standard. The number format itself is now a hardware feature.

The closing chapter, [Hardware Horizon](../12-hardware-horizon/), connects all the prior threads to where the field is right now — and tells the story through five turning points from 2018 to 2025.

---

That's the map. Twelve articles, one issue, one mystery. To begin: ask yourself what it actually means that the cleanest modern way to represent a neural-network weight uses *exactly sixteen distinct numbers*.

**Continue to** → [Numbers In Boxes](../02-numbers-in-boxes/) for the first primer: a tour of how floating-point numbers actually work, and why FP16 is *not* just "smaller FP32".
