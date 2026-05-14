---
title: "The Geometry Of Weights"
description: "What does a trained LLM actually look like, numerically? A primer on the empirical distribution of weights and activations — Gaussian here, heavy-tailed there, and the surprise that broke quantization."
topics: [quantization, statistics]
tags: [distributions, outliers, empirical, statistics]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 50
techKind: primer
techNode: shape-weights
header: default.webp
---

## What Are You Actually Quantizing?

You can't compress something you don't understand. Before you go pick a number format, you should know what the numbers *look like*.

So: open up a trained LLM, pull out a weight matrix, and plot it. What do you see?

The answer turns out to depend a lot on **which weights** and **which layer** you're looking at. The whole story of LLM quantization is the story of someone — usually [Tim Dettmers](https://timdettmers.com) — discovering that some part of the answer is *not what you'd guess*.

## The Boring Part: Weights Are Mostly Gaussian-ish

Pull a random row of a feedforward matrix out of a trained 7B model. Histogram it. You'll see something that looks like this:

```pyplot {id="weight-hist" caption="Synthetic stand-in for an FFN weight row in a trained LLM. Approximately Gaussian, centered on zero, with mildly heavy tails."}
np.random.seed(42)
# Mix of Gaussian + slight heavy tail (Student t with df=5)
core = np.random.randn(20000) * 0.07
tails = np.random.standard_t(df=5, size=2000) * 0.04
weights = np.concatenate([core, tails])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(weights, bins=120, color='#FF007F', alpha=0.75, edgecolor='#1A1A1A',
        linewidth=0.3, density=True)
xs = np.linspace(-0.5, 0.5, 200)
mu, sigma = weights.mean(), weights.std()
ax.plot(xs, np.exp(-(xs-mu)**2/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi)),
        color='#1A1A1A', linewidth=2, linestyle='--', label='Gaussian fit')
ax.set_xlabel("weight value")
ax.set_ylabel("density")
ax.set_title("Synthetic LLM feedforward weight distribution (per-row)")
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
print(f"mean: {mu:.4f}, std: {sigma:.4f}")
print(f"kurtosis (Gaussian = 3): {((weights - mu)**4).mean() / sigma**4:.2f}")
```

Notice three things:

1. **Roughly symmetric around zero.** This is by design — most weight initializers (Xavier, Kaiming) initialize with zero mean, and training doesn't change that drastically for most weight matrices.
2. **Approximately bell-shaped.** Most trained weights are well-modeled by a Gaussian.
3. **The tails are slightly heavier than Gaussian.** The kurtosis is greater than 3 (the Gaussian value). Real LLM weight rows often have kurtosis 4–8.

This empirical fact — that *normalized per-row weights are approximately Gaussian* — is the **entire load-bearing assumption** of NF4 from [QLoRA](https://arxiv.org/abs/2305.14314). Recall from [The Lloyd-Max Bargain](../03-lloyd-max/) that the optimal placement of levels depends on the distribution. NF4 picks its 16 levels to be the **conditional centroids of a standard normal**. That choice is information-theoretically optimal *if* the assumption holds. And on most LLM weight rows, it does.

## The Concentration Trick: Per-Row Scaling

There's a subtlety. Different rows of a weight matrix have wildly different scales. Some rows might have standard deviation 0.005, others 0.5. If you fit *one* Gaussian to the whole matrix, the tails are dominated by the high-variance rows and the centre by the low-variance rows.

So in practice you **normalize per row** (or per column, depending on the algorithm). Each row gets its own scale factor. After normalization, the rows look like independent samples from approximately the same standard-shaped distribution. And *now* the Lloyd-Max picture from the previous chapter applies cleanly.

This trick — per-row or per-column scales — is the bedrock of every quantization method we'll see. It's how [GPTQ](../08-brain-surgery/), [AWQ](../09-method-family-tree/), and [NF4](../09-method-family-tree/) all "use the same 16 numbers" effectively: they're sharing those 16 numbers across many tiny groups of weights that happen to all look like scaled versions of the same Gaussian.

## The Interesting Part: Activations Are Not Like Weights

Now do the same exercise for **activations** — the values that flow between layers when you run a forward pass. Don't take a single token; aggregate across a batch.

```pyplot {id="activation-hist" caption="Synthetic stand-in for a transformer's hidden state activations. The bulk is small, but a few feature dimensions have giant outlier values."}
np.random.seed(7)
# Most dimensions: small Gaussian-ish activations
batch, dim = 200, 768
mostly_quiet = np.random.randn(batch, dim) * 0.5
# A handful of dimensions with large persistent magnitudes (the "outlier dimensions")
outlier_dims = np.random.choice(dim, 6, replace=False)
mostly_quiet[:, outlier_dims] *= 80   # 100x amplification
activations = mostly_quiet.flatten()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.hist(activations, bins=80, color='#00A8A8', alpha=0.75, edgecolor='#1A1A1A', linewidth=0.3)
ax.set_yscale('log')
ax.set_xlabel("activation value")
ax.set_ylabel("count (log)")
ax.set_title("Activations: all dimensions pooled")
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
max_per_dim = np.abs(mostly_quiet).max(axis=0)
ax.bar(range(dim), max_per_dim, color='#1A1A1A', width=1.0)
ax.bar(outlier_dims, max_per_dim[outlier_dims], color='#FF007F', width=1.5)
ax.set_xlabel("hidden dimension index")
ax.set_ylabel("max |activation| across batch")
ax.set_title("Per-dimension maximum — outliers are systematic, not random")
ax.spines[['top', 'right']].set_visible(False)
```

The left plot shows what you see if you pool every value: a thin spike near zero, with a long, sparse tail of huge values reaching ±40. The right plot shows the secret: the huge values are **concentrated in a tiny number of specific feature dimensions**. Not randomly distributed. The *same dimensions* light up for every input.

This is the **outlier dimension phenomenon**, and it is the single most important empirical observation in the entire field of LLM quantization. Without an explanation for it, every standard quantization method that worked on CNNs *fails catastrophically* on a 6.7B-parameter transformer. It is the discovery that made [LLM.int8()](../06-outliers/) necessary.

We chase this story properly in [The 1% That Ruins Everything](../06-outliers/). The relevant empirical fact for this primer is: **activations contain a small number of giant, persistent, systematic outliers, while weights mostly don't**. This asymmetry is *why* we end up with separate quantization treatments for weights and activations.

## Why "Per-Row" Vs "Per-Token" Matters: The K/V Asymmetry

Here is one more empirical curiosity that pays off later. When you look at the **K** (keys) and **V** (values) tensors of a transformer's attention layer:

- **K** has a few systematically-large *channels* (dimensions) — an outlier-channel pattern, similar to FFN activations.
- **V** has a few systematically-large *tokens* — outlier rows that vary by sequence position.

This is a real, well-documented asymmetry. The 2023 paper KIVI exploits it: they quantize K **per-channel** (one scale per feature dimension), and V **per-token** (one scale per sequence position). Same data, two different scaling axes, because the *outlier geometry* is different.

This is the load-bearing observation for [KV Cache Tyranny](../10-kv-cache/). Quantizing K and V the same way leaves accuracy on the table because the outliers don't share a structure.

## The Heavy-Tail Aside: Why Not Just Clip?

A naïve reaction to outliers is: "fine, clip the top 0.1% and quantize the rest." But this *destroys the model*. The outlier features aren't noise — they encode something the model genuinely uses. Empirically, ablating the outlier-channel directions in a 6.7B-parameter model causes catastrophic accuracy collapse. They are *load-bearing*.

The intellectual move that unlocked the field was: **stop trying to suppress outliers; build a quantization scheme that handles them with a different mechanism.**

This is the core insight of LLM.int8 (split weights into "regular" and "outlier" blocks, quantize the former, leave the latter in higher precision), of SmoothQuant (mathematically *migrate* outliers from activations to weights, where they're easier to handle), of AWQ (use the activation magnitudes to decide which weights to be most careful about), and of mixed-precision schemes in general.

Every one of those methods makes more sense once you internalize the empirical picture: **weights are Gaussian-ish, activations have giant systematic spikes, and the spikes matter.**

## Napkin Estimates To Remember

For a transformer with $d = 4096$ hidden dimensions:

- Roughly **6–10 outlier dimensions** appear above ~6.7B parameters. Below that scale, they often don't show up at all (a result Dettmers found that *itself* shocked the field — outlier-driven behavior is a scale-emergent phenomenon).
- Outlier *values* are 20–100× the median activation magnitude. The exact ratio depends on layer and architecture.
- Outlier dimensions are **stable across inputs**: the *same* dimension indices contain outliers across thousands of different prompts.
- The fraction of *parameters* that interact strongly with outliers is small (often <1%), but their gradient-of-loss is huge — they matter much more than their count suggests.

## What This Means For The Rest Of The Issue

When you read about a quantization method, ask yourself which assumption about the distribution it is making:

- "Quantize uniformly" → assumes weights are roughly uniform (rarely true).
- "Quantize with FP4" → assumes log-distributed weights (close for some layers).
- "Quantize with NF4" → assumes weights are Gaussian after normalization (mostly true).
- "Handle outliers separately" → admits the Gaussian assumption breaks on activations.
- "Use second-order weighting" → admits not all errors are equally bad (see [Taylor & Hessians](../07-taylor-and-hessians/)).

The shape of the data drives the shape of the algorithm.

**Continue to** → [The 1% That Ruins Everything](../06-outliers/) — the discovery of the outlier dimensions, and the first method that solved them.
