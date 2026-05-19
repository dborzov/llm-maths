---
title: "Attention Sparsity: why softmax buries the losers"
short_title: "Attention Sparsity"
description: "Softmax's exponential amplification means that a logit advantage of just 3 units gives one token 74% of the total probability mass — sparsity is structural, not accidental, which is why evicting 80–90% of the KV cache is mathematically safe."
blurb:
  - "A logit 3 units above its neighbors captures 74% of softmax mass. A 6-unit advantage: 98%."
  - "For a 512-token context, maybe 8–10 tokens carry weights of 0.15–0.31; the rest are 0.0001 or less."
  - "Is this sparsity a quirk of the task or structural? The argument is surprisingly elementary."
  - "If sparsity were accidental, pruning the bottom 90% would occasionally catastrophically fail — but it doesn't."
topics: [attention, theory]
tags: [softmax, attention-sparsity, kv-pruning, temperature]
theme: cream
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 70
techKind: primer
techNode: attention-sparsity
header: 07-attention-sparsity.webp
---

## The Plot That Started an Argument

Pull up any {{< wiki "attention" >}}attention head{{< /wiki >}} in a trained transformer and print its weight vector for a single query token against a 512-token context. Not the logits — the {{< wiki "softmax" >}}softmax{{< /wiki >}} probabilities. The first thing you see is strange: most numbers are microscopic. Weights of 0.0003, 0.0001, 0.0000. A handful — maybe eight or ten out of five hundred — carry 0.15, 0.22, 0.31. The entire probability mass is piled into a small fraction of the tokens.

Is this a coincidence? Is it a quirk of the particular task, the particular model, the particular input? Or is it **structural** — something baked into the softmax function itself that means attention distributions *have* to look like this?

The answer matters enormously. If sparsity is structural, then throwing away the bottom 90% of the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} is mathematically safe, and building a pruning system around it makes sense. If sparsity is accidental, you will drop something important sooner or later.

Spoiler: it is structural, and the argument is surprisingly elementary.

## The Softmax Operator, Stripped Down

The {{< wiki "softmax" >}}softmax{{< /wiki >}} function takes a vector of real-valued *logits* and returns a proper probability distribution:

$$
\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

The key property hiding in that formula is **exponential amplification of differences**. If token $i$ has a logit that is 3 units higher than its neighbors, its probability is $e^3 \approx 20\times$ higher. A 6-unit advantage gives $e^6 \approx 403\times$. Softmax doesn't just pick a winner; it *buries* the losers.

To feel this viscerally, consider a vector where six entries are all 1 and a seventh entry slides upward from 1 to 10:

```pyplot {id="softmax-winner" caption="SOFTMAX AMPLIFICATION: AS ONE LOGIT RISES, IT RAPIDLY CAPTURES ALL PROBABILITY MASS."}
xs = np.linspace(1, 10, 200)
results = []
for x_winner in xs:
    logits = np.array([1.0, 1.0, 1.0, x_winner, 1.0, 1.0, 1.0])
    logits -= logits.max()  # numerical stability
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum()
    results.append(probs[3])

fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.plot(xs, results, color='#FF007F', linewidth=2.5)
ax.fill_between(xs, results, alpha=0.15, color='#FF007F')
ax.axhline(1/7, color='#00A8A8', linewidth=1.2, linestyle='--', label='uniform (1/7)')
ax.axvline(4, color='#FFD700', linewidth=1.0, linestyle=':', alpha=0.8)
ax.annotate('3 units above neighbors\n→ winner gets 74% of mass',
            xy=(4, 0.74), xytext=(5.5, 0.55),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A'),
            fontsize=9, color='#1A1A1A')
ax.set_xlabel('winning logit value  (neighbors fixed at 1)')
ax.set_ylabel('probability assigned to winner')
ax.set_title('softmax([1,1,1,x,1,1,1])  —  exponential winner-take-all')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
```

At $x=4$ (three units above its neighbors) the winner already holds 74% of the probability. At $x=7$, it holds 99%. The other six entries, despite being *identical* to each other, share a rounding error between them.

{{% pullquote type="counter-intuitive" %}}
Softmax doesn't reward the best token — it *obliterates* the competition. A 3-point logit gap produces a 20× probability gap. A 6-point gap produces a 400× gap. The long tail is not noise; it is the function doing its job.
{{% /pullquote %}}

## Why Logit Gaps Are Real: The Scaling Story

So softmax amplifies differences exponentially. The next question is whether real attention logits — which are dot products of query and key vectors — actually have meaningful gaps between them, or whether most logits end up nearly equal.

The raw dot product of a query $\mathbf{q}$ and key $\mathbf{k}$, each of dimension $D$, is:

$$
\text{logit}_{i} = \mathbf{q} \cdot \mathbf{k}_i = \sum_{d=1}^{D} q_d k_{d,i}
$$

If $\mathbf{q}$ and the keys are random unit vectors, each product $q_d k_{d,i}$ is a mean-zero random variable with variance $\sim 1/D$ (for normalized vectors). The dot product is a sum of $D$ such terms, so by the central limit theorem it has variance $\sim 1$. Good: the logits are all order-1 numbers, and they *do* vary.

But without the $1/\sqrt{D}$ scaling factor, the story changes. If the vectors are not constrained to unit norm — just initialized with variance 1 per component — each term $q_d k_{d,i}$ has variance 1, and the dot product has variance $D$. With $D=128$, the typical logit magnitude is $\sqrt{128} \approx 11$. The differences between logits become enormous relative to what softmax can distinguish, and softmax collapses to near-uniform: every token looks equally relevant.

<details>
<summary>The variance calculation in full</summary>

Let $q_d, k_{d,i} \sim \mathcal{N}(0, 1)$ independently. Then $q_d k_{d,i}$ has mean zero and variance $\mathbb{E}[q_d^2]\,\mathbb{E}[k_{d,i}^2] = 1$. The dot product $\mathbf{q}\cdot\mathbf{k}_i = \sum_{d=1}^D q_d k_{d,i}$ has variance $D$ (sum of $D$ independent unit-variance terms). The standard deviation is $\sqrt{D}$.

After dividing by $\sqrt{D}$: the scaled logit $\mathbf{q}\cdot\mathbf{k}_i / \sqrt{D}$ has variance 1 regardless of $D$.

The $1/\sqrt{D}$ factor in the attention formula is not for aesthetic symmetry — it is the only scaling that keeps the softmax input distribution stationary as model width increases.

</details>

## Temperature: The Dial That Everything Turns

There is a cleaner way to think about this. Softmax with a temperature parameter $\tau$ is:

$$
\text{softmax}_\tau(\mathbf{x})_i = \frac{e^{x_i / \tau}}{\sum_j e^{x_j / \tau}}
$$

**High temperature** ($\tau \gg 1$): divides all logits by a large number, making them nearly equal, making the output nearly uniform. Every token gets attended to equally. The head is "confused."

**Low temperature** ($\tau \ll 1$): amplifies differences, making the output spike at the maximum. Almost all probability mass goes to the single highest-scoring token.

The $1/\sqrt{D}$ scaling in attention is implicitly setting the temperature. When $D$ is large but unscaled, the logits are large, which corresponds to a *low effective temperature* — already in winner-take-all territory. The $\sqrt{D}$ divisor brings the temperature back up toward a sensible operating range.

But here is the twist: trained transformers, in practice, do **not** end up in the uniform-attention regime. They are specifically trained to route attention selectively. The model learns weight matrices $W_Q$ and $W_K$ such that certain query-key pairs produce systematically large dot products, while most pairs remain small. The network has learned to operate in the cold regime for most heads, because cold heads carry information; warm heads carry noise.

## The Simulated Long Tail

All of this is theory. Let us make it concrete with numbers.

Generate $T=512$ random keys in $D=128$-dimensional space. Generate a single random query. Compute the attention weights with $1/\sqrt{D}$ scaling and sort them from largest to smallest. What fraction of the total probability mass do the top 10 tokens hold?

```pyplot {id="simulated-sparsity" caption="SIMULATED ATTENTION WEIGHTS, T=512 TOKENS, D=128. THE LONG TAIL IS STRUCTURAL — EVEN WITH RANDOM VECTORS."}
np.random.seed(42)
D = 128
T = 512

# Random query and keys — unit Gaussian, then scaled dot product
q = np.random.randn(D)
K = np.random.randn(T, D)

logits = (K @ q) / np.sqrt(D)
logits -= logits.max()  # numerical stability
weights = np.exp(logits)
weights /= weights.sum()

sorted_w = np.sort(weights)[::-1]
cumsum = np.cumsum(sorted_w)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: sorted weight bar chart
ax1.bar(np.arange(len(sorted_w)), sorted_w,
        color=np.where(np.arange(len(sorted_w)) < 10, '#FF007F', '#FFD700'),
        edgecolor='#1A1A1A', linewidth=0.2)
ax1.axvline(9.5, color='#00A8A8', linewidth=1.5, linestyle='--')
top10 = sorted_w[:10].sum()
ax1.text(12, sorted_w[0] * 0.7,
         f'top 10 tokens\n= {top10:.0%} of mass', fontsize=9, color='#FF007F')
ax1.set_xlabel('token rank (0=highest)')
ax1.set_ylabel('attention weight')
ax1.set_title(f'T={T} random tokens, D={D}')
ax1.spines[['top', 'right']].set_visible(False)

# Right: cumulative mass
ax2.plot(np.arange(len(cumsum)), cumsum, color='#FF007F', linewidth=2.0)
ax2.axhline(0.80, color='#00A8A8', linewidth=1.2, linestyle='--', label='80% mass')
ax2.axhline(0.95, color='#FF8C00', linewidth=1.2, linestyle='--', label='95% mass')
threshold_80 = np.searchsorted(cumsum, 0.80)
threshold_95 = np.searchsorted(cumsum, 0.95)
ax2.axvline(threshold_80, color='#00A8A8', linewidth=0.8, alpha=0.6)
ax2.axvline(threshold_95, color='#FF8C00', linewidth=0.8, alpha=0.6)
ax2.text(threshold_80 + 3, 0.3, f'{threshold_80} tokens\n→ 80%', fontsize=8, color='#00A8A8')
ax2.text(threshold_95 + 3, 0.15, f'{threshold_95} tokens\n→ 95%', fontsize=8, color='#FF8C00')
ax2.set_xlabel('number of top-k tokens kept')
ax2.set_ylabel('cumulative probability mass')
ax2.set_title('cumulative attention mass by rank')
ax2.legend(loc='lower right')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

print(f"D={D}, T={T} random vectors:")
print(f"  Top 1 token:  {sorted_w[0]:.3f} ({sorted_w[0]:.1%})")
print(f"  Top 5 tokens: {sorted_w[:5].sum():.3f} ({sorted_w[:5].sum():.1%})")
print(f"  Top 10 tokens: {sorted_w[:10].sum():.3f} ({sorted_w[:10].sum():.1%})")
print(f"  Tokens to reach 80% mass: {threshold_80}")
print(f"  Tokens to reach 95% mass: {threshold_95}")
print(f"  Tokens below 0.001:  {(sorted_w < 0.001).sum()}")
```

With completely *random* vectors — no training, no signal, just Gaussian noise — a handful of tokens captures the majority of the probability mass. This is not a property of the model having learned to be selective. It is a property of softmax plus high dimension. The long tail is there from the start; training only makes it more pronounced.

{{< crosshead >}}Why Sparsity Makes Pruning Safe{{< /crosshead >}}

Here is the argument that makes this matter for {{< wiki "kv-pruning" >}}KV cache pruning{{< /wiki >}}.

The output of a single attention head at position $j$ is a weighted sum of value vectors:

$$
\text{head\_out}_j = \sum_{i \leq j} a_{ji}\, \mathbf{v}_i
$$

where $a_{ji}$ is the attention weight from query $j$ to key $i$. Now suppose we remove token $i$ from the KV cache — we discard both its key and its value. The error introduced in $\text{head\_out}_j$ is exactly the dropped term:

$$
\Delta_j^{(i)} = a_{ji}\, \mathbf{v}_i
$$

The magnitude of this error is bounded:

$$
\|\Delta_j^{(i)}\| \leq a_{ji} \cdot \|\mathbf{v}_i\|
$$

If $a_{ji}$ is near zero for all recent queries $j$, then the error from dropping token $i$ is near zero for all those positions — bounded by $\epsilon \cdot \|\mathbf{v}_i\|$ where $\epsilon$ is the maximum attention weight the token ever receives. The key insight is that softmax's long tail means **most tokens are in the low-$a_{ji}$ regime for most queries**. They are attending but barely. Their contribution to the residual stream update is negligible.

{{% callout type="theorem" title="The Bounded Eviction Error" %}}
If token $i$ satisfies $\max_{j \geq i} a_{ji} \leq \varepsilon$ for all future query positions $j$, then evicting it from the KV cache introduces per-position error at most $\varepsilon \|\mathbf{v}_i\|$.

For a typical value vector with $\|\mathbf{v}_i\| \approx 1$ and $\varepsilon = 0.001$, the error is of order $10^{-3}$ — well below the floating-point noise floor of fp16 computation.
{{% /callout %}}

This is the linearity argument: the attention output is linear in the attention weights, so small weights imply small contribution, and small contribution implies safe eviction.

## The Attention Sink: The Exception That Proves the Rule

There is a well-known complication worth naming before moving on. Certain tokens — most famously the beginning-of-sequence (BOS) token — receive disproportionately *high* attention across essentially all heads and all layers, regardless of the current query. These are the **attention sinks** first documented by Xiao et al. in 2023.

Why does this happen? The BOS token has been present through the entire context. Every key vector in the sequence has "seen" it, and the model has learned to route "I don't have a specific answer for this query" attention to a fixed anchor. It is the conversational equivalent of saying "um" — a place to put attention when nothing specific fires.

The practical upshot for pruning: always preserve the first few tokens, regardless of their measured attention weight under the current query. They are structural sinks. See [Inside K and V](../../03-quantization/16-kv-distribution/) for the distribution-level analysis of why this happens in the geometry of trained key vectors.

## What To Remember

1. **Softmax exponentially amplifies logit differences.** A 3-unit gap between logits produces a ~20× probability gap. This is the mathematical root of attention sparsity.
2. **The $1/\sqrt{D}$ scaling factor keeps the softmax in a sensible regime** — without it, large $D$ collapses the distribution toward uniform, but scaling restores variance to ~1 regardless of head dimension.
3. **Sparsity is structural, not learned.** Even random query-key vectors in high dimension produce heavily peaked attention distributions. Training makes it more extreme, not less.
4. **Sparse attention = safe pruning.** The error from evicting token $i$ is bounded by $a_{ji} \cdot \|\mathbf{v}_i\|$. Tokens with consistently near-zero weights contribute negligible error when evicted.
5. **Attention sinks break the rule.** Always keep BOS and other systematic sinks regardless of measured weights.

**Continue to** → **[The Heavy Hitter Oracle](../03-heavy-hitter/)** — now that we know attention is sparse by construction, the H₂O paper builds a concrete eviction policy around it, and finds that the 20% of tokens that matter are surprisingly predictable from early context.
