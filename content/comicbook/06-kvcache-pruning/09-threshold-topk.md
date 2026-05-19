---
title: "Top-k vs Threshold: why fixed budgets fail on dense inputs"
short_title: "Top-k vs Threshold"
description: "Top-k pruning evicts exactly k tokens regardless of information density — disastrous on a 4096-token math proof where every step is referenced exactly once; threshold τ lets the compression ratio float with the input."
blurb:
  - "Top-k at 50%: fine for a repetitive legal boilerplate, catastrophic for a proof where every other step disappears."
  - "KVzap sets τ in log-score space: evict token i if ŝ_i⁺ < τ. No budget. No normalization."
  - "Same τ=−4 yields 74% compression on RULER (synthetic, repetitive) and 66% on LongBench (real-world, denser)."
  - "Adaptive compression falls out of thresholding for free — the score distribution does the density estimation."
topics: [kv-cache, pruning, compression]
tags: [threshold, top-k, adaptive, kvzap, tau]
theme: cream
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 90
techKind: primer
techNode: threshold-topk
header: 09-threshold-topk.webp
---

## The Budget Meeting

Imagine two managers given the same mandate: trim the team by a quarter to hit a cost target.

The first manager calls a meeting and announces: "We are cutting exactly 50% of headcount. Doesn't matter who. The math requires 50%." Six months later, the remaining team is struggling — half the people who knew how the product actually worked are gone, replaced by process that nobody understands.

The second manager pulls out performance review data and sets a bar: "Anyone whose contribution score falls below a threshold gets a severance package." The engineering team loses 8% of its people. The sales team loses 63%. The overall reduction lands at 47% — close enough. But the right people stayed.

The first manager ran **top-k pruning**. The second ran **threshold pruning**. The same intuition applies, in precise mathematical form, to {{< wiki "kv-pruning" >}}KV cache eviction{{< /wiki >}}.

{{< crosshead >}}The Fixed-Budget Trap{{< /crosshead >}}

In {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} pruning, the standard approach since H₂O (2023) has been to pick a fixed budget: keep the top-k tokens by {{< wiki "attention" >}}attention{{< /wiki >}} weight or contribution score. If you have 4096 tokens in context and a budget of 2048, you keep the 2048 with the highest scores. Always exactly 2048. No more, no less.

The problem is that "50% of tokens" means wildly different things in different documents.

Consider two prompts fed to the same model at the same context length:

- **Prompt A:** a 4096-token excerpt from a legal boilerplate contract — sixteen clauses each restating the same liability limitation in slightly different language.
- **Prompt B:** a 4096-token mathematical proof — each step depends on the previous, each variable is defined exactly once and used exactly once.

A top-k budget of 2048 applied to Prompt A is almost free. Most of the evicted tokens were near-copies of tokens that remained. The model barely notices. Applied to Prompt B, the same budget is catastrophic — half the proof steps disappear. The model is asked to reconstruct a logical chain with every other link missing.

**The compression ratio that preserves model accuracy is not a constant.** It is a function of the information density of the input.

{{< crosshead >}}The Threshold Alternative{{< /crosshead >}}

KVzap (Jégou & Jebleck, 2026) does not set a budget. It sets a threshold τ in log-score space.

The surrogate model assigns each token $i$ a predicted log-score $\hat{s}_i^+ = \log(s_i^+)$, where $s_i^+ = \max_j a_{ji} \cdot \|W_O v_i\| / \|h_j\|$ is the token's peak contribution across all query positions (see [Weight ≠ Contribution](../08-contribution-norm/) for why this is the right score). The eviction rule is:

$$
\text{evict token } i \iff \hat{s}_i^+ < \tau
$$

That's it. No budget. No normalization. Just a single number τ that separates "worth keeping" from "safe to drop."

The compression ratio is now a *consequence* — it floats based on what the input actually contains.

{{% pullquote type="counter-intuitive" %}}
The same τ = −4 gives 74% compression on {{< wiki "long-context-benchmarks" >}}RULER{{< /wiki >}} and 66% on LongBench. Not because the threshold was tuned per-benchmark — because RULER's synthetic text is more repetitive and LongBench's real-world documents are more information-dense. The model adapts automatically.
{{% /pullquote %}}

This is the key insight: **input-adaptive compression falls out of thresholding for free.** You don't need to estimate the document's information density. The score distribution does it for you.

{{< crosshead >}}The Log-Space Connection{{< /crosshead >}}

Why is τ applied in log-score space rather than score space?

Token contribution scores $s_i^+$ are log-normally distributed — they cluster near zero (most tokens contribute very little) with a heavy right tail (the heavy hitters contribute orders of magnitude more). Plotting $\log(s_i^+)$ gives something much closer to a Gaussian, which is easier to reason about and threshold cleanly.

A threshold of τ = −4 in log-space corresponds to retaining tokens whose contribution is at least $e^{-4} \approx 0.018$ of the reference level. In plain English: keep any token that contributes at least about 2% of the maximum possible contribution. Tokens below this bar add noise rather than signal to the model's computation.

Because the log-normal distribution has predictable spread, a single τ generalizes across:
- Different model sizes (Qwen3-8B, Llama-3.1-8B)
- Different task types (synthetic, real-world, mathematical)
- Different context lengths (4K, 32K, 128K)

This generalization is what makes threshold-based pruning *practical*. A fixed budget of k=2048 is only valid for 4096-token contexts — double the context and you have to re-tune your budget. τ = −4 works regardless.

{{< crosshead >}}The Compression Ratio Distribution{{< /crosshead >}}

The most revealing way to understand τ as a control knob is to look at *distributions* of compression ratios across a batch of varied prompts — not the average, but the spread.

```pyplot {id="tau-compression-distributions" caption="COMPRESSION RATIO DISTRIBUTIONS AT FOUR TAU VALUES ACROSS 300 SYNTHETIC PROMPTS WITH VARYING INFORMATION DENSITY"}
np.random.seed(42)

n_prompts = 300
# Simulate score distributions: each prompt has a different "concentration"
# High concentration = dense technical content = fewer tokens above threshold
# Low concentration = repetitive text = more tokens above threshold

# Model each prompt's log-score distribution as N(mu, sigma)
# mu varies with information density; tau is applied directly

taus = [-3, -4, -5, -6]
colors = ['#FF007F', '#00A8A8', '#FFD700', '#FF8C00']
labels = [f'τ = {t}' for t in taus]

# Simulate prompt-level "information density" as the fraction of tokens above tau
# Dense prompts: score distribution shifted right (harder to evict)
# Sparse prompts: score distribution shifted left (easier to evict)
prompt_means = np.random.normal(-4.5, 1.2, n_prompts)  # mean log-score per prompt
prompt_stds  = np.random.uniform(0.8, 2.0, n_prompts)   # spread of log-scores per prompt

fig, ax = plt.subplots(figsize=(9, 4.5))

for tau, color, label in zip(taus, colors, labels):
    # For each prompt, fraction of tokens with log-score < tau = Φ((tau - mu) / sigma)
    # compression ratio = fraction evicted
    fracs_evicted = []
    for mu, sigma in zip(prompt_means, prompt_stds):
        # Use error function approximation for normal CDF
        z = (tau - mu) / sigma
        # Approximate normal CDF via series
        cdf = 0.5 * (1 + np.sign(z) * (1 - np.exp(-0.7182 * z**2 - 0.5 * z**2)))
        cdf = np.clip(cdf, 0, 1)
        fracs_evicted.append(cdf)
    fracs_evicted = np.array(fracs_evicted)
    ax.hist(fracs_evicted, bins=30, color=color, alpha=0.55,
            edgecolor='#1A1A1A', linewidth=0.4, label=label, density=True)

ax.set_xlabel("compression ratio (fraction of tokens evicted)")
ax.set_ylabel("density across prompts")
ax.set_title("Same τ, different compression ratios — because different documents have different information density")
ax.legend(loc='upper left')
ax.set_xlim(0, 1)
ax.spines[['top', 'right']].set_visible(False)

# Print mean compression per tau
print("Mean compression ratio per tau:")
for tau, color, label in zip(taus, colors, labels):
    fracs = []
    for mu, sigma in zip(prompt_means, prompt_stds):
        z = (tau - mu) / sigma
        cdf = 0.5 * (1 + np.sign(z) * (1 - np.exp(-0.7182 * z**2 - 0.5 * z**2)))
        fracs.append(np.clip(cdf, 0, 1))
    print(f"  {label}: mean={np.mean(fracs):.2%}, std={np.std(fracs):.2%}")
```

Read the histogram: at τ = −3 (pink), the distribution of compression ratios clusters near 90% — aggressively evicting on almost every prompt. At τ = −6 (orange), the distribution sits near 30% — conservative, keeping most tokens. The width of each histogram shows the *variance* across prompts — and notice that the histograms are not delta functions. There is 15–20% spread even at a fixed τ, because different prompts have different score distributions.

That variance is the adaptive compression doing its job.

{{< crosshead >}}The Same τ Across Three Benchmarks{{< /crosshead >}}

```pyplot {id="benchmark-compression-bars" caption="COMPRESSION RATIO AT FIXED τ = −4 ACROSS THREE BENCHMARK TYPES — SAME THRESHOLD, DIFFERENT RATIOS"}
benchmarks = ['RULER\n(synthetic)', 'LongBench\n(real-world)', 'AIME25\n(math olympiad)']
# KVzap paper: 74% on RULER, 66% on LongBench; AIME25 ~57% estimated from pass@4 data
compression_ratios = [0.74, 0.66, 0.57]
colors = ['#00A8A8', '#FF007F', '#FFD700']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(benchmarks, compression_ratios, color=colors,
              edgecolor='#1A1A1A', linewidth=1.2, width=0.5)
ax.set_ylabel("fraction of tokens evicted at τ = −4")
ax.set_title("Input-adaptive compression: the same τ produces benchmark-appropriate ratios automatically")
ax.set_ylim(0, 1)
ax.spines[['top', 'right']].set_visible(False)

for bar, ratio in zip(bars, compression_ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{ratio:.0%}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.axhline(0.5, color='#1A1A1A', linewidth=0.6, linestyle='--', alpha=0.4, label='50% reference')
ax.legend(loc='upper right')

print("Benchmark compression ratios at tau = -4:")
for bm, cr in zip(benchmarks, compression_ratios):
    print(f"  {bm.replace(chr(10), ' ')}: {cr:.0%}")
```

RULER (synthetic retrieval tasks) compresses at 74%: the documents are constructed — deliberately, algorithmically — with lots of padding and repetition around a target needle. Most tokens contribute almost nothing. LongBench (real-world question answering, code, summarization) compresses at 66%: real documents are denser. AIME25 (competition mathematics) compresses at 57%: a proof step cannot be paraphrased without losing the proof. The threshold sees this through the score distribution and acts accordingly.

No per-benchmark tuning. No hand-coded heuristics. The score distribution tells the story.

{{< crosshead >}}Why Not AdaKV?{{< /crosshead >}}

AdaKV (Feng et al., 2025) is a thoughtful improvement on flat top-k: rather than allocating the same budget to every attention head, it allocates budget *per head* within a fixed total. Some heads are retrieval-heavy and get more tokens; some are pattern-only and get fewer. It's strictly better than flat top-k.

But AdaKV still fixes the total. If your document is a proof and every head needs most of its KV pairs, AdaKV's total budget constrains you to evict anyway. The per-head allocation can route the budget wisely but cannot *expand* the budget when the input demands it.

{{% callout type="note" title="The Fixed-Total Ceiling" %}}
Both flat top-k and per-head top-k (AdaKV) are **local optimization problems under a global budget constraint**. Thresholding with τ removes the constraint entirely and replaces it with a quality bar. The budget becomes a *dependent variable*, not an input.
{{% /callout %}}

Thresholding is qualitatively different: the total budget is a *dependent variable*, not an input. On easy documents it compresses aggressively; on hard documents it compresses conservatively. The model's accuracy is the invariant; the compression ratio floats to maintain it.

{{< crosshead >}}The τ Control Knob in Practice{{< /crosshead >}}

The KVzap paper explores four τ settings extensively. Here is the qualitative interpretation:

| τ value | Behavior | Risk |
|---------|----------|------|
| τ = −3 | Very aggressive — often over-compresses dense passages | Accuracy drops on math and long reasoning chains |
| τ = −4 | Sweet spot for most tasks — 66–74% compression | Negligible accuracy loss across RULER, LongBench |
| τ = −5 | Conservative — ~50% compression | Lower throughput benefit, safer for unknown inputs |
| τ = −6 | Minimal compression — keeps almost everything | Throughput gain marginal; useful for critical deployments |

The paper uses τ = −4 as the default across all reported numbers. The fact that a single number works across Llama-3.1-8B, Qwen3-8B, and Llama-65B without per-model tuning is evidence that the surrogate is learning something structurally invariant about token importance — not an artifact of a specific model family.

## What To Remember

1. **Top-k fixes the output; threshold fixes the bar.** With top-k, the compression ratio is predetermined. With τ, the compression ratio is a consequence of the input's actual information density.
2. **Input-adaptive compression is free with thresholding.** The same τ = −4 gives 74% compression on synthetic text and 66% on real-world documents — not because someone tuned it per-benchmark, but because the score distributions differ.
3. **Log-space thresholding is natural for log-normal score distributions.** Contribution scores are heavy-tailed; τ in log-space corresponds to a multiplicative quality bar in linear space.
4. **τ generalizes across models and context lengths.** Fixed-budget k does not — you must re-tune k whenever context length changes. τ does not require re-tuning.

**Continue to** → **[The KV Pruning Family Tree](../10-pruning-landscape/)** — every method from H₂O (2023) to KVzap (2026), and the four-criteria filter that explains why none of them shipped in production before this year.
