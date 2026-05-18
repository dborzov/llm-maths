---
title: "H₂O: 20% of tokens absorb 80% of attention"
short_title: "H₂O"
description: "UT Austin's H₂O paper (NeurIPS 2023) measured that roughly 20% of tokens absorb 80% of softmax attention mass — and built the first principled eviction policy around that observation."
blurb:
  - "The heatmap finding: most query rows are nearly zero except for 3–4 spikes at the start, the end, and a handful of recurring positions."
  - "The name: Heavy Hitter Oracle — borrowed from data-streaming theory, where a heavy hitter is any element appearing far more than its share."
  - "H₂O's score for token i: the running sum of all attention weights ever placed on it across all queries."
  - "What it couldn't do: work inside FlashAttention or during prefill — the two things production requires."
topics: [kv-cache, attention, pruning]
tags: [h2o, heavy-hitters, kv-pruning, zhang-2023, sparse-attention]
theme: cream
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 30
techKind: mainline
techNode: heavy-hitter
header: 03-heavy-hitter.webp
---

## Twenty Percent of Your Tokens Own Eighty Percent of the Attention

Mid-2023. Multiple research groups are independently staring at {{< wiki "attention" >}}attention weight{{< /wiki >}} heatmaps — the grids that show which positions each query position looks at. The setup is mundane: run a decent LLM on a 1,000–4,000 token context, dump `attn_weights` at each layer, and plot.

What they see is not mundane.

The heatmaps are almost blank. For most query positions, the row is nearly zero everywhere — except for three or four spikes: the very beginning of the sequence, the most recent few tokens, and a handful of tokens scattered across the middle that keep appearing in spike after spike, row after row. The same positions, lighting up across every layer, every head, every query.

This was the observation that the **H₂O** paper — *"H₂O: Heavy-Hitter Oracle for Efficient Generative Inference"* by Zhenyu Zhang, Ying Sheng, Tianyi Zhou, and colleagues at **UT Austin** — would formalize in late 2023. Their empirical measurement: roughly **20% of tokens receive roughly 80% of the total {{< wiki "attention" >}}attention{{< /wiki >}} mass**. The other 80% of tokens are nearly invisible to the model.

The team named these high-attention tokens *heavy hitters* — borrowing from the data-streaming literature, where a heavy hitter is any element that appears far more than its fair share of the time. The water metaphor came as a bonus: H₂O. Heavy-Hitter Oracle. A name that makes you think of something essential, everywhere, and inexorable.

```pyplot {id="attention-long-tail" caption="SIMULATED ATTENTION MASS VS. TOKEN RANK — 80% OF MASS CONCENTRATES IN TOP 20% OF TOKENS"}
np.random.seed(7)
n_tokens = 512

# Simulate softmax concentration in high dimension
# Real LLM attention weights have this Zipfian / power-law-like shape
# We model it: draw from exponential, sort descending, normalize
raw = np.random.exponential(scale=1.0, size=n_tokens)
raw[:20] *= 8.0     # a handful of "heavy hitter" tokens get large boosts
raw[:4]  *= 3.0     # attention sinks (BOS etc.) get extra boost
raw = np.sort(raw)[::-1]
weights = raw / raw.sum()

cumulative = np.cumsum(weights)
fractions  = np.arange(1, n_tokens + 1) / n_tokens

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: individual weights (log scale shows the long tail clearly)
ax = axes[0]
ax.bar(np.arange(n_tokens), weights, color='#FF007F', alpha=0.85, linewidth=0)
ax.set_yscale('log')
ax.set_xlabel("Token rank (by attention weight, descending)", fontsize=10)
ax.set_ylabel("Attention weight (log scale)", fontsize=10)
ax.set_title("Per-token attention weight — 512-token context", fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

# Right: cumulative mass curve — the 80/20 line
ax2 = axes[1]
ax2.plot(fractions * 100, cumulative * 100, color='#00A8A8', linewidth=2.5)
ax2.axvline(20, color='#FF8C00', linewidth=1.5, linestyle='--', label='20% of tokens')
ax2.axhline(80, color='#FFD700', linewidth=1.5, linestyle='--', label='80% of mass')
ax2.scatter([fractions[np.searchsorted(cumulative, 0.80)] * 100], [80],
            color='#FF007F', s=80, zorder=5)
ax2.set_xlabel("Fraction of tokens kept (%, ranked by weight)", fontsize=10)
ax2.set_ylabel("Cumulative attention mass (%)", fontsize=10)
ax2.set_title("Cumulative attention mass — the 80/20 rule", fontsize=10)
ax2.legend(fontsize=9)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
```

The plot above captures the essential structure: a few tokens absorb most of the mass, and the rest trail off into near-zero. Keep the top 20% of tokens by attention weight and you retain ~80% of the total probability mass. Discard the bottom 80% and you lose almost nothing — at least in terms of what the model is actually reading.

## Why the Math Backs This Up

The attention mechanism computes a weighted sum of value vectors:

$$
\text{head\_out}_j \;=\; \sum_{i=1}^{T} a_{ji} \cdot v_i
$$

where $a_{ji}$ is the {{< wiki "softmax" >}}softmax{{< /wiki >}}-normalized {{< wiki "attention" >}}attention weight{{< /wiki >}} that query $j$ places on key $i$, and $v_i$ is the value vector for token $i$.

Now suppose token $i$ has received near-zero attention from every query so far: $a_{ji} \approx 0$ for all $j \leq t$. What does evicting token $i$ from the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} cost? Precisely the error:

$$
\varepsilon_{j} \;=\; a_{ji} \cdot v_i \;\approx\; 0
$$

When weights are exactly zero the removal is **exact** — not an approximation. When they're *near* zero the error is bounded by $|a_{ji}| \cdot \|v_i\|$. Since $\|v_i\|$ is typically on the order of one (values live on the residual stream scale), a weight of $0.001$ introduces an error on the order of a thousandth of one value vector. Negligible.

{{% callout type="theorem" title="H₂O's Eviction Guarantee" %}}
If token $i$'s cumulative attention score $c_i = \sum_{j \leq t} a_{ji}$ falls below a threshold, and all future queries similarly attend minimally to $i$, then evicting token $i$ from the KV cache introduces error bounded by $\max_j a_{ji} \cdot \|v_i\|_2$. For tokens with $c_i \approx 0$, this is tight.
{{% /callout %}}

The **consistency** finding was the empirical kicker. Heavy hitters are not arbitrary — a token that gets lit up by query 100 also tends to get lit up by queries 200 and 300. The cumulative attention score $c_i$ is a stable predictor of future importance. This is why a *running* count works at all: you do not need a crystal ball. History is enough.

{{< crosshead >}}The Eviction Policy, Spelled Out{{< /crosshead >}}

H₂O maintains a **budget**: at most $K$ KV pairs in memory at any time. As decoding proceeds:

1. After computing attention at each step, update running cumulative scores: $c_i \mathrel{+}= a_{\text{new},i}$ for all $i$ in the current cache.
2. If the cache exceeds $K$ entries, evict the entry with the smallest $c_i$.
3. The evicted token's key and value are gone — no recovery.

Two slots are always reserved for **attention sinks**: the first few tokens of every sequence (BOS, system prompt opening) that empirically receive near-universal attention regardless of content. Without this exception, H₂O's eviction logic immediately discards them and accuracy collapses.

{{% marginnote %}}
Attention sinks were characterized independently in Xiao et al. 2023 (*StreamingLLM*), published roughly concurrently with H₂O. Both papers noticed that the first few tokens act as a "sink" absorbing excess softmax probability mass.
{{% /marginnote %}}

The whole scheme is elegant. Napkin math on savings: a Llama-3 70B model has $L = 80$ layers, $H = 64$ heads, $D = 128$ dimensions per head. At 16-bit precision a KV cache for $T = 8{,}192$ tokens weighs $2 \times 80 \times 64 \times 8192 \times 128 \times 2\,\text{bytes} \approx 21\,\text{GB}$. H₂O at 20% retention: $4\,\text{GB}$. The saving is real and large.

```pyplot {id="h2o-eviction-sim" caption="SIMULATED H₂O EVICTION — CUMULATIVE SCORES OVER TIME. ONLY TOP-K SURVIVE."}
np.random.seed(42)
T = 200     # sequence length
K = 40      # keep top-K budget

# Simulate cumulative attention scores over time
# A few tokens are "heavy hitters" with persistent high attention
n_heavy = 8
n_sink  = 2
scores  = np.zeros(T)

# Simulate step-by-step decoding
history = np.zeros((T, T))  # history[t, i] = cumulative score of token i at step t
cumulative = np.zeros(T)

for step in range(T):
    # Generate attention weights for this decoding step over all prior tokens
    weights = np.random.exponential(0.5, size=(step + 1))
    weights[:n_sink] *= 10.0   # sinks always attended
    if step > 20:
        weights[n_sink:n_sink + n_heavy] *= 5.0  # heavy hitters emerge
    weights = weights / weights.sum()
    cumulative[:step + 1] += weights
    history[step, :step + 1] = cumulative[:step + 1]

final_scores = cumulative

# Determine which tokens survive H2O eviction
rank = np.argsort(final_scores)[::-1]
survivors = set(rank[:K])
evicted   = set(rank[K:])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: final cumulative scores, coloured by survival
colors = ['#00A8A8' if i in survivors else '#FF8C00' for i in range(T)]
ax = axes[0]
ax.bar(np.arange(T), final_scores, color=colors, linewidth=0, alpha=0.9)
ax.set_xlabel("Token position", fontsize=10)
ax.set_ylabel("Cumulative attention score", fontsize=10)
ax.set_title(f"Final scores — teal=kept ({K}), orange=evicted ({T-K})", fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

# Right: score trajectories for a sample of tokens
ax2 = axes[1]
for i in range(n_sink):
    ax2.plot(history[:, i], color='#FFD700', linewidth=1.5, alpha=0.9)
for i in range(n_sink, n_sink + n_heavy):
    ax2.plot(history[:, i], color='#FF007F', linewidth=1.2, alpha=0.7)
for i in range(n_sink + n_heavy, n_sink + n_heavy + 15):
    ax2.plot(history[:, i], color='#1A1A1A', linewidth=0.6, alpha=0.35)
ax2.set_xlabel("Decoding step", fontsize=10)
ax2.set_ylabel("Cumulative attention score", fontsize=10)
ax2.set_title("Score trajectories — yellow=sinks, pink=heavy hitters, dark=evicted", fontsize=10)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
```

The right panel shows why consistency matters. Attention sinks (yellow) climb steeply from step one — the model attends to them immediately. Heavy hitters (pink) emerge after a few dozen steps and accumulate steadily. The rest (dark) barely move. By the time eviction kicks in, the rankings are nearly stable.

## Where H₂O Falls Short

H₂O was the right idea at the right moment. The 2023 NeurIPS version reported near-lossless performance on GPT-NeoX and LLaMA at 20% KV retention. For the research community that was remarkable. For production inference engineers reading the paper, three problems were immediately visible.

{{< crosshead >}}Problem One: Fixed Global Budget{{< /crosshead >}}

H₂O keeps a global top-$K$ across all heads. But {{< wiki "attention" >}}attention heads{{< /wiki >}} differ dramatically in how sparse they are. Some heads in a Llama model attend almost exclusively to the previous three tokens — a moving window of 3 would suffice. Other heads spread attention broadly across hundreds of positions. A global budget wastes slots on the already-sparse heads and starves the dense ones.

The right metric would be *per-head* sparsity — and the right budget would be *adaptive*, not fixed. H₂O doesn't do this.

{{< crosshead >}}Problem Two: Retroactive Scoring{{< /crosshead >}}

The cumulative score $c_i$ for token $i$ is computed *after* attention runs — after the model has already paid the memory cost of keeping token $i$ around. This is fine during decoding when you're generating one token at a time. But it means:

- You cannot score tokens during *prefill* (the bulk processing of the input prompt) without a second pass.
- A token that scored low at position 100 might become a heavy hitter at position 5,000 — by which point it's already been evicted.

H₂O is retroactive. The ideal system would be *predictive*: score a token before committing to keeping it in the cache.

{{< crosshead >}}Problem Three: The FlashAttention Problem{{< /crosshead >}}

Tracking a running cumulative sum of attention weights requires reading those weights — which means materializing the $T \times T$ attention matrix in memory. **FlashAttention** (the dominant production attention kernel since 2022) deliberately avoids materializing this matrix. The entire point of FlashAttention is to recompute attention in tiles without ever storing the full matrix.

Bolting H₂O's accumulation logic onto FlashAttention would require kernel modifications that break the carefully tuned SRAM/DRAM tiling. The production teams at NVIDIA, Meta, and Mistral who maintain these kernels had no appetite for this. H₂O was never integrated into any major inference engine.

{{% pullquote type="counter-intuitive" %}}
H₂O's insight was right: attention is sparse and heavy hitters are persistent. Its method was elegant. And yet it shipped in exactly zero production inference engines. The idea was ahead of the implementation.
{{% /pullquote %}}

## What H₂O Leaves Open

The H₂O result crystallizes the fundamental question for {{< wiki "kv-pruning" >}}KV cache pruning{{< /wiki >}}: we know *that* some tokens are heavy hitters. We know *which* tokens they are — after they've accumulated attention. We don't know which tokens they *will be* before attention runs.

The dream is a scoring function that operates *before* attention: look at a token's {{< wiki "residual-stream" >}}hidden state{{< /wiki >}} as it emerges from the previous layer, and predict whether it will be a heavy hitter for the remaining context. If you could do that, you could prune at prefill time, during decoding, and without touching the attention kernel.

That dream required someone to first *measure* token importance precisely — to build a ground truth signal that a predictor could learn from. The next attempt came in 2025, with an idea so simple it seems obvious in retrospect: make the model copy-paste the document, and watch what it looks at.

**Continue to** → **[The Pretext Trick](../04-copy-paste/)** — the 2025 method that tried to measure token importance with a copy-and-paste trick, and why it almost worked.
