---
title: "Contribution Norm: why attention weight misleads"
short_title: "Contribution Norm"
description: "A token's actual contribution to the residual stream is a_{ji} × ‖W_O v_i‖ — both the attention weight and the projected value norm matter, and conflating them with raw attention weight causes systematic mispricing of token importance."
blurb:
  - "The residual stream update: h_j gets added a_{ji} × W_O v_i for each past token i."
  - "High attention weight + tiny value norm = near-zero residual change. The token barely mattered."
  - "Low attention weight + large value norm = token may still dominate the update. It should not be evicted."
  - "The fix: rank by a_{ji} × ‖W_O v_i‖, not by a_{ji} alone — this is the KVzip+ contribution norm."
topics: [attention, theory]
tags: [kv-pruning, residual-stream, contribution-norm, kvzip]
theme: teal
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 80
techKind: primer
techNode: contribution-norm
header: 08-contribution-norm.webp
---

## The Manager Who Fills the Room

Picture a post-mortem meeting after a failed product launch. The CTO is mentioned in every slide. Her name comes up thirty times. Her quarterly review is full of citations. By the "attention" metric — how often she is referenced — she is the most important person in the room.

Now look at the git log. She committed three lines of config. The quietly exhausted staff engineer in the back row committed the authentication module, the payment gateway, and the bulk of the test suite. His name appears once in the slides. By the attention metric he barely exists.

This is exactly the problem with using raw {{< wiki "attention" >}}attention weights{{< /wiki >}} as a proxy for token importance in {{< wiki "kv-pruning" >}}KV cache pruning{{< /wiki >}}. The attention weight $a_{ji}$ measures how much query position $j$ "looks at" key position $i$. But what we want to know is: how much does token $i$ actually *change* the model's output?

Those two things can diverge dramatically. The mathematics of why — and what to do about it — is the subject of this primer.

## The Residual Stream Update

Start with the equation that governs what a single attention head actually does to the {{< wiki "residual-stream" >}}residual stream{{< /wiki >}}. When the model processes position $j$, the hidden state is updated by:

$$
\mathbf{h}_j^{\text{out}} = \mathbf{h}_j + \sum_{i \leq j} a_{ji}\, W_O\, \mathbf{v}_i
$$

Walk through each piece:

- $\mathbf{h}_j$ is the current hidden state at position $j$ before this head — a vector of dimension $D_{\text{model}}$.
- $\mathbf{v}_i$ is the value vector for token $i$ (stored in the KV cache), dimension $D_{\text{head}}$.
- $W_O$ is the output projection matrix, shape $D_{\text{model}} \times D_{\text{head}}$. It maps value space back into residual stream space.
- $a_{ji}$ is the scalar attention weight from query $j$ to key $i$.
- The sum over $i \leq j$ accumulates contributions from every context token.

Token $i$'s contribution to position $j$'s hidden state is the *vector*:

$$
\mathbf{c}_{ji} = a_{ji}\, W_O\, \mathbf{v}_i
$$

The magnitude of this contribution — how much it actually moves the residual stream — is:

$$
\|\mathbf{c}_{ji}\| = a_{ji} \cdot \|W_O\, \mathbf{v}_i\|
$$

This is the quantity we care about. Notice that it depends on *two* things: the attention weight $a_{ji}$ and the norm $\|W_O\,\mathbf{v}_i\|$ of the projected value vector. High weight alone is not enough.

{{< crosshead >}}The Two Pathological Cases{{< /crosshead >}}

Consider what can go wrong if you rank tokens purely by attention weight:

| Case | $a_{ji}$ | $\|W_O\,\mathbf{v}_i\|$ | $\|\mathbf{c}_{ji}\|$ | What happens if you keep this token? |
|------|----------|--------------------------|----------------------|---------------------------------------|
| **High-weight, low-value** | 0.35 | 0.004 | 0.0014 | Heavy attention, microscopic effect. The token is "talked about" but says nothing. |
| **Low-weight, high-value** | 0.003 | 8.2 | 0.025 | Rarely attended, but when the head fires it dominates. Evicting causes large error. |

Case A is the CTO. Case B is the staff engineer. Ranking by $a_{ji}$ alone keeps the CTO and evicts the staff engineer, which is precisely backwards.

{{% pullquote type="counter-intuitive" %}}
A token that is heavily attended but has a near-zero value projection contributes nothing to the residual stream. A token that is barely glanced at but has a large value projection can dominate the output. Attention weight alone is a dangerously misleading importance score.
{{% /pullquote %}}

## The Physical Analogy: Work = Force × Displacement

There is a physics picture that makes this immediate.

**Work** = force × displacement. No displacement, no work. A very strong force applied to a wall does zero work — the wall doesn't move.

In attention:
- **Force** = attention weight $a_{ji}$. How hard is query $j$ pulling on token $i$?
- **Displacement** = $\|W_O\,\mathbf{v}_i\|$. How far does token $i$ actually move the {{< wiki "residual-stream" >}}residual stream{{< /wiki >}} when it fires?
- **Work** = $a_{ji} \cdot \|W_O\,\mathbf{v}_i\|$. How much does including token $i$ actually change the model state?

High force at zero displacement is zero work. That is Case A — the token is heavily attended but its value projection is tiny, so it contributes nothing. A small force can still do substantial work if the displacement is large. That is Case B — low attention weight, but a large value projection means each unit of weight moves the state significantly.

The contribution norm $a_{ji} \cdot \|W_O\,\mathbf{v}_i\|$ is the *work done by token $i$ on position $j$*. It is the right quantity to minimize over when choosing what to evict.

## Normalizing by the Receiving State

The {{< wiki "kv-pruning" >}}KVzip+{{< /wiki >}} paper adds one more twist: divide the contribution magnitude by the norm of the hidden state that receives it.

$$
s_{ji}^{+} = \frac{a_{ji} \cdot \|W_O\,\mathbf{v}_i\|}{\|\mathbf{h}_j\|}
$$

Why? Because the same absolute perturbation matters differently depending on the scale of the state being perturbed.

Returning to the physics: imagine you are adding 1 gram to a scale. If the scale is measuring feathers (total load: 2 grams), you have changed the reading by 50%. If the scale is measuring boulders (total load: 200 kilograms), you have changed the reading by 0.0005%. The *relative* perturbation is what determines whether the addition is detectable and meaningful.

Dividing by $\|\mathbf{h}_j\|$ converts the absolute contribution to a *relative* one — a signal-to-noise ratio for the contribution. This is the normalization that makes the score comparable across positions with different-magnitude hidden states.

The **KVzip+ importance score** for token $i$ is the maximum relative contribution it ever makes, taken over all positions $j$ in the prompt:

$$
s_i^{+} = \max_{j \in \text{prompt}} \frac{a_{ji} \cdot \|W_O\,\mathbf{v}_i\|}{\|\mathbf{h}_j\|}
$$

Taking the max rather than the mean or sum reflects a conservative philosophy: if token $i$ ever matters a lot to even one position, it is important and should be kept. The maximum is the worst-case contribution; a token whose max contribution is tiny is safe to evict.

## The Log-Space Trick

{{< wiki "kv-pruning" >}}KVzip{{< /wiki >}} does not directly predict $s_i^{+}$. It predicts $\log(s_i^{+})$.

The reason is distributional. Look at the shape of real $s^{+}$ scores across tokens in a context. A handful are large (0.5, 1.2, 0.8). Most are small (0.003, 0.0001, 0.00004). The distribution has a heavy upper tail and a vast lower tail squashed against zero. It is approximately **log-normal**.

Predicting a log-normal target in linear space is a numerically poor regression problem. The linear surrogate model is trying to fit values ranging over four or five orders of magnitude — it will either overfit the large values (with large absolute errors on the small values) or average them into mud.

In log-space, the same scores become approximately Gaussian (that is what log-normal means). Now a linear predictor can fit them with additive errors in log-space, which correspond to *multiplicative* errors in score-space — exactly the right error structure for a quantity that spans many orders of magnitude.

```pyplot {id="log-normal-scores" caption="SYNTHETIC s+ SCORES: LINEAR SCALE OBSCURES STRUCTURE, LOG SCALE REVEALS IT."}
np.random.seed(7)
n_tokens = 256

# Simulate realistic s+ scores: most tokens have tiny contributions,
# a few have large ones. Log-normal is a reasonable model.
log_scores = np.random.normal(loc=-4.0, scale=1.8, size=n_tokens)
s_scores = np.exp(log_scores)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Linear scale
ax1.hist(s_scores, bins=40, color='#00A8A8', edgecolor='#1A1A1A', linewidth=0.4)
ax1.set_xlabel('s+ score (linear scale)')
ax1.set_ylabel('count')
ax1.set_title('linear scale — most tokens piled at zero')
ax1.spines[['top', 'right']].set_visible(False)

# Log scale
ax2.hist(log_scores, bins=30, color='#FF007F', edgecolor='#1A1A1A', linewidth=0.4)
ax2.set_xlabel('log(s+) score')
ax2.set_ylabel('count')
ax2.set_title('log scale — approximately Gaussian, easy to regress')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

print(f"s+ score statistics (n={n_tokens} tokens):")
print(f"  mean={s_scores.mean():.4f}, median={np.median(s_scores):.4f}")
print(f"  max={s_scores.max():.4f}, min={s_scores.min():.6f}")
print(f"  range: {s_scores.max() / (s_scores.min() + 1e-12):.0f}x span")
print(f"log(s+) statistics:")
print(f"  mean={log_scores.mean():.2f}, std={log_scores.std():.2f}")
```

## Attention Weight vs. KVzip+ Score: The Rankings Diverge

Let us make the two pathological cases concrete with synthetic data. Generate 20 tokens with random attention weights, random value projection norms, and random hidden state norms. Rank them two ways: by raw attention weight, and by $s_i^{+}$.

```pyplot {id="ranking-divergence" caption="RAW ATTENTION WEIGHT vs. KVZIP+ SCORE: RANKINGS DIVERGE. DISAGREEMENTS HIGHLIGHTED IN PINK."}
np.random.seed(13)
n = 20

# Simulate tokens with varying attention weights and value norms
a = np.random.exponential(scale=0.05, size=n)   # typical sparse attention
a /= a.sum()                                      # normalize to probability
v_norm = np.random.lognormal(mean=0.3, sigma=1.2, size=n)  # heavy-tailed value norms
h_norm = np.random.lognormal(mean=1.5, sigma=0.5, size=n)  # hidden state norms

s_plus = (a * v_norm) / h_norm

# Rankings (0 = most important)
rank_attn = np.argsort(np.argsort(-a))
rank_s = np.argsort(np.argsort(-s_plus))

# Disagreements: top-5 by one metric but not by the other
top5_attn = set(np.argsort(-a)[:5])
top5_s = set(np.argsort(-s_plus)[:5])
disagreements = top5_attn.symmetric_difference(top5_s)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
token_ids = np.arange(n)

colors_attn = ['#FF007F' if i in disagreements else '#FFD700' for i in token_ids]
colors_s    = ['#FF007F' if i in disagreements else '#00A8A8' for i in token_ids]

ax1.barh(token_ids, a[np.argsort(-a)], color=[colors_attn[i] for i in np.argsort(-a)],
         edgecolor='#1A1A1A', linewidth=0.4)
ax1.set_xlabel('attention weight a_ji')
ax1.set_title('ranked by raw attention weight')
ax1.set_yticks(token_ids)
ax1.set_yticklabels([f'tok {np.argsort(-a)[i]}' for i in token_ids], fontsize=7)
ax1.spines[['top', 'right']].set_visible(False)

ax2.barh(token_ids, s_plus[np.argsort(-s_plus)],
         color=[colors_s[i] for i in np.argsort(-s_plus)],
         edgecolor='#1A1A1A', linewidth=0.4)
ax2.set_xlabel('KVzip+ score s+')
ax2.set_title('ranked by KVzip+ score')
ax2.set_yticks(token_ids)
ax2.set_yticklabels([f'tok {np.argsort(-s_plus)[i]}' for i in token_ids], fontsize=7)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

print("Top 5 by attention weight:", sorted(np.argsort(-a)[:5].tolist()))
print("Top 5 by s+ score:        ", sorted(np.argsort(-s_plus)[:5].tolist()))
print("Tokens in top-5 by one metric but not the other:", sorted(disagreements))
```

The pink bars mark tokens that appear in the top-5 by one metric but not the other. These are the tokens a naive pruner would get wrong — either evicting something important (the staff engineer) or preserving something useless (the CTO).

## Napkin Math: How Much Does This Matter?

In a typical LLaMA-class model with $D_{\text{head}} = 128$, a token's value vector is mapped through $W_O$ to produce a 4096-dimensional residual stream update. The norm $\|W_O\,\mathbf{v}_i\|$ varies substantially across tokens: empirically, a factor of 10–100× between the quietest and loudest tokens in a context is common.

At the same time, $\|\mathbf{h}_j\|$ itself varies across positions — early positions with shorter context have smaller residual stream norms than late positions that have accumulated many layer updates. The per-position normalization captures this and prevents the score from being artifactually biased toward early or late tokens.

Rough order-of-magnitude: if raw attention weights range from 0.0001 to 0.3, and value norms range from 0.01 to 5.0, then $s_i^{+}$ scores range from $0.0001 \times 0.01 / 2 = 5 \times 10^{-7}$ to $0.3 \times 5.0 / 0.5 = 3.0$ — six orders of magnitude. The log scale covers this gracefully. Linear scale does not.

{{% callout type="tip" title="Implementing KVzip+ Efficiently" %}}
In practice, $\|W_O\,\mathbf{v}_i\|$ can be computed once per token during the prefill pass and cached alongside the KV cache entry — it costs one extra scalar per token. The hidden state norm $\|\mathbf{h}_j\|$ at each query position is also available during prefill. The max over $j$ is maintained as a running maximum. Total overhead: one float32 per token in the cache.
{{% /callout %}}

## What To Remember

1. **The residual stream update** is $\mathbf{h}_j^{\text{out}} = \mathbf{h}_j + \sum_i a_{ji}\,W_O\,\mathbf{v}_i$. Token $i$'s contribution magnitude is $a_{ji} \cdot \|W_O\,\mathbf{v}_i\|$ — a product of weight and value projection norm.
2. **Two pathological cases** expose raw attention weight as a flawed importance score: high-weight tokens with tiny value projections (lots of attention, no contribution) and low-weight tokens with large value projections (barely attended, but dominant when they fire).
3. **Work = force × displacement.** The contribution norm is the mechanical analogue of work. High force at zero displacement — high attention weight with a tiny value projection — does zero work.
4. **The KVzip+ score** $s_i^{+} = \max_{j} a_{ji} \cdot \|W_O\,\mathbf{v}_i\| / \|\mathbf{h}_j\|$ normalizes by the receiving hidden state magnitude to make contributions comparable across positions.
5. **Predict in log-space.** The $s^{+}$ scores follow a log-normal distribution spanning many orders of magnitude. Regression in log-space is better conditioned and gives lower prediction error.

**Continue to** → **[Ghost in the Hidden State](../05-ghost-state/)** — having defined the right importance score, KVzip now faces a harder question: can we *predict* $s_i^{+}$ for tokens we have not yet processed, using only the hidden state vectors we already have? The answer turns out to be yes, and the reason is startling.
