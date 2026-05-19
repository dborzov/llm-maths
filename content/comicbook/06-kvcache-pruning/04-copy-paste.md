---
title: "KVzip: measuring importance by making the model copy"
short_title: "KVzip"
description: "NVIDIA's KVzip (2025) scores token importance by making the model repeat the document verbatim — tokens the model has to look up to reproduce receive high scores — achieving 4× compression with near-zero accuracy loss."
blurb:
  - "The pretext task: extend the prompt with 'Repeat the previous context exactly' and collect attention weights during that repetition."
  - "Score formula: for each original token i, the max attention weight any repetition position j ever places on i."
  - "4× compression on LongBench and passkey retrieval with accuracy losses that barely clear measurement noise."
  - "The fatal flaw: scoring requires a second forward pass — 2× prefill overhead, unusable during decode."
topics: [kv-cache, attention, pruning]
tags: [kvzip, copy-paste, pretext-task, contribution-norm, devoto-2025, kv-pruning]
theme: teal
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 40
techKind: mainline
techNode: copy-paste
header: 04-copy-paste.webp
---

## The Graveyard of Good Ideas

By 2025, the field had produced more than twenty published methods for {{< wiki "kv-pruning" >}}KV cache pruning{{< /wiki >}}. The survey papers were long. The citation graphs were dense. And the number of these methods that shipped in a production inference engine was: zero.

Not a few. Zero.

The problem was structural. Every method that scored tokens *after* attention ran hit the same wall: modifying the attention kernel conflicts with FlashAttention. Every method that tried to *predict* importance without a signal to learn from was guessing. The [H₂O paper's](../03-heavy-hitter/) cumulative-attention approach was elegant and empirically validated — and still undeployed after two years.

The question that kept circulating in inference team email threads was sharp: *what if, instead of approximating importance, you just measured it directly?*

Minsoo Kim, Hana Kwon, and colleagues at **NVIDIA** had an answer in 2025. It is one of those ideas that, in retrospect, seems obvious. It was not.

## The Copy-and-Paste Pretext Task

The insight came from *self-supervised learning*, specifically the concept of a **pretext task** — a task you don't care about intrinsically, but whose successful completion reveals something useful about the data.

KVzip's pretext task is almost childishly simple. Given a user prompt, extend it like this:

```
user: <original prompt>
Repeat the previous context exactly.
assistant: <original prompt repeated verbatim>
```

Then run the model on this extended input and collect the {{< wiki "attention" >}}attention weights{{< /wiki >}} during the repetition phase. The reasoning: to faithfully copy every word of a document, the model needs to *look up* words it cannot reconstruct from context. "The" is predictable — you barely need to read it. "42" is not predictable — you have to look it up. "Paris" is not predictable. The name of the obscure theorem in paragraph six is not predictable.

The attention pattern during repetition therefore reveals, in principle, which tokens carry information the model cannot reproduce without attending to the source. High attention during repetition → high importance.

{{% marginnote %}}
The pretext-task idea has deep roots: BERT's masked language modelling (Devlin et al., 2019) is a pretext task where predicting masked tokens teaches representations. KVzip applies the same intuition to *token importance measurement* rather than representation learning.
{{% /marginnote %}}

The KVzip scoring formula formalizes this. For each token $i$ in the original prompt, its importance score is:

$$
s_i \;=\; \max_{j \in \langle\text{repetition}\rangle} a_{ji}
$$

where $a_{ji}$ is the attention weight that repetition position $j$ places on original position $i$. The score is the *maximum* (not the mean) because a single high-attention lookup is enough to establish that the token matters. If no repetition position ever strongly attends to token $i$, the score stays low and the token is a candidate for eviction.

{{< crosshead >}}What 4× Compression Actually Means{{< /crosshead >}}

The KVzip results were striking. On standard benchmarks — LongBench, passkey retrieval, multi-document QA — the method achieved **4× compression** (retaining 25% of the KV cache) with near-zero accuracy degradation. That means: for a context that would generate a {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} of 20 GB, the evicted cache weighs 5 GB, and the model barely notices the missing 15 GB.

To make this concrete: a 128k-token context on Llama-3 70B generates a KV cache of roughly 160 GB. At 4× compression that becomes 40 GB — the difference between requiring six H100s and fitting on two.

This is not a benchmark artifact. The pretext task is providing *genuine signal*. A token that the model never looks up during repetition really is one that the model can reconstruct or ignore. The attention pattern during the copy task is a direct measurement of what the model needs, not an approximation of it.

```pyplot {id="kvzip-attention-heatmap" caption="SIMULATED COPY-AND-PASTE ATTENTION — REPETITION QUERIES ATTENDING TO ORIGINAL TOKENS. HOT SPOTS = HIGH-IMPORTANCE TOKENS."}
np.random.seed(3)

n_orig = 40   # original prompt tokens
n_rep  = 40   # repetition tokens (same length, different positions)

# Simulate attention from repetition tokens → original tokens
# "Important" tokens get looked up more; common/predictable tokens don't
importance = np.zeros(n_orig)
# Sinks
importance[:2] = 0.9
# Scattered high-importance tokens (facts, names, numbers)
important_idx = [5, 8, 14, 17, 23, 28, 33, 37]
importance[important_idx] = np.random.uniform(0.6, 1.0, len(important_idx))
# Moderate
importance[10:13] = 0.3
# Rest near zero
importance[importance == 0] = np.random.uniform(0.0, 0.08, (importance == 0).sum())

# Build attention matrix: each repetition token mostly attends to its "matching" position
# plus the important tokens
attn = np.zeros((n_rep, n_orig))
for j in range(n_rep):
    # Local lookup (parallel position in original)
    attn[j, j] = 0.25 + np.random.uniform(0, 0.1)
    # Importance-weighted lookups
    attn[j] += importance * np.random.uniform(0.1, 0.4, n_orig)
    # Add noise
    attn[j] += np.random.uniform(0, 0.02, n_orig)
    # Softmax-normalize
    attn[j] = np.exp(attn[j] * 5)
    attn[j] /= attn[j].sum()

kvzip_scores = attn.max(axis=0)   # KVzip: max over repetition rows

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
im = ax.imshow(attn, aspect='auto', cmap='magma', vmin=0, vmax=attn.max())
ax.set_xlabel("Original token position", fontsize=10)
ax.set_ylabel("Repetition token position", fontsize=10)
ax.set_title("Attention: repetition → original (brighter = more attended)", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.03)

ax2 = axes[1]
colors = ['#FF007F' if s > 0.15 else '#00A8A8' for s in kvzip_scores]
ax2.bar(np.arange(n_orig), kvzip_scores, color=colors, linewidth=0)
ax2.axhline(0.15, color='#FFD700', linewidth=1.5, linestyle='--', label='eviction threshold')
ax2.set_xlabel("Original token position", fontsize=10)
ax2.set_ylabel("KVzip score $s_i$ (max attention received)", fontsize=10)
ax2.set_title("KVzip scores — pink=keep, teal=evict", fontsize=10)
ax2.legend(fontsize=9)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
```

The left panel shows the attention heatmap during repetition: vertical stripes mark tokens that every repetition position looks at — these are the candidates for keeping. The right panel converts this into per-token scores; the pink bars survive, the teal bars get evicted.

## The Footnote Problem: Why Attention Weight Isn't Enough

The 2025 follow-on work from Devoto, Berto, Marin, and colleagues at the same NVIDIA team identified a subtle flaw in the basic KVzip score. The problem is this: a token can receive *substantial* attention and still contribute almost nothing to the model's computations.

To see why, recall how the {{< wiki "attention" >}}attention{{< /wiki >}} output folds into the {{< wiki "residual-stream" >}}residual stream{{< /wiki >}}. The full update for position $j$ is:

$$
h_j^{\text{out}} \;=\; h_j \;+\; \sum_i a_{ji} \, W_O v_i
$$

where $W_O$ is the output projection matrix of the attention head. Token $i$'s contribution to position $j$'s {{< wiki "residual-stream" >}}hidden state{{< /wiki >}} is not just $a_{ji}$ — it is $a_{ji} \cdot W_O v_i$. Its *magnitude* is:

$$
\|a_{ji} \, W_O v_i\|_2 \;=\; a_{ji} \cdot \|W_O v_i\|_2
$$

Here is the problem. Suppose token $i$ has $\|W_O v_i\|_2 \approx 0$. This happens when the value vector $v_i$ nearly lies in the null space of the output projection $W_O$ — a geometrically possible and empirically observed situation. Then *no matter how high* $a_{ji}$ is, token $i$'s contribution to the residual is negligible. It is being looked at, but it has nothing to say.

{{% pullquote type="counter-intuitive" %}}
A token that receives 30% of an attention head's probability mass can contribute less to the output than a token that receives 1%, if the first token's value vector is nearly orthogonal to the output projection. Attention weight and attention contribution are not the same thing.
{{% /pullquote %}}

The metaphor: a **widely-cited footnote**. Every reader glances at the footnote marker. Almost no one's understanding changes as a result. The footnote receives massive "attention" and delivers almost zero information. KVzip's raw $s_i$ cannot distinguish the footnote from a load-bearing claim.

{{< crosshead >}}The KVzip+ Score{{< /crosshead >}}

The fix, which the team called KVzip+, normalizes by both the output-projected value norm and the receiving hidden state norm:

$$
s_i^+ \;=\; \max_j \; a_{ji} \;\cdot\; \frac{\|W_O v_i\|_2}{\|h_j\|_2}
$$

The $\|h_j\|_2$ denominator accounts for the fact that a large contribution to a large hidden state is proportionally less important than the same contribution to a small hidden state. This is the score that, empirically, best predicts which tokens can be evicted without accuracy loss.

The deep dive on exactly why this normalization works — and how it connects to the geometry of the residual stream — is in [Weight ≠ Contribution](../08-contribution-norm/). For now: the raw attention weight $a_{ji}$ measures *interest*. The KVzip+ score $s_i^+$ measures *actual impact on computation*.

```pyplot {id="kvzip-vs-kvzipplus" caption="ATTENTION WEIGHT VS. KVZIP+ SCORE — CASES WHERE THEY DIVERGE REVEAL TOKENS THAT ARE WATCHED BUT NOT LOAD-BEARING."}
np.random.seed(17)
n = 60

# Simulate: some tokens have high attention but small ||W_O v||
# others have modest attention but large ||W_O v||
raw_attention = np.random.exponential(0.5, size=n)
raw_attention[:8] *= 6.0      # high attention tokens
raw_attention = raw_attention / raw_attention.max()

# ||W_O v_i|| — independent of attention; some tokens have small norms
wo_v_norm = np.random.exponential(0.8, size=n)
wo_v_norm[:4] *= 0.05           # the "footnotes": high attention, near-zero output norm
wo_v_norm[10:14] *= 3.5         # the "quiet workhorses": low attention, high output norm
wo_v_norm = np.clip(wo_v_norm, 0, None)

h_norm = np.ones(n) * 2.5 + np.random.randn(n) * 0.3  # hidden state norms ~ constant

kvzip_score   = raw_attention                            # raw: just max attention
kvzipplus_score = raw_attention * wo_v_norm / h_norm    # normalized

# Normalize both for comparability
kvzip_score    = kvzip_score / kvzip_score.max()
kvzipplus_score = kvzipplus_score / kvzipplus_score.max()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.arange(n)
width = 0.4

ax = axes[0]
ax.bar(x - width/2, kvzip_score,    width=width, color='#FF007F', alpha=0.85, label='KVzip  (attention only)',  linewidth=0)
ax.bar(x + width/2, kvzipplus_score, width=width, color='#00A8A8', alpha=0.85, label='KVzip+ (× ‖W_O v‖ / ‖h‖)', linewidth=0)
ax.set_xlabel("Token index", fontsize=10)
ax.set_ylabel("Score (normalized)", fontsize=10)
ax.set_title("KVzip vs KVzip+ — diverge for 'footnote' tokens 0–3", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(-1, 25)   # zoom to the interesting region
ax.spines[['top', 'right']].set_visible(False)

ax2 = axes[1]
ax2.scatter(kvzip_score, kvzipplus_score, c='#FF8C00', s=40, alpha=0.85, linewidths=0)
# Highlight the footnote tokens
ax2.scatter(kvzip_score[:4], kvzipplus_score[:4],
            c='#FF007F', s=90, zorder=5, label='"Footnote" tokens:\nhigh attention, tiny ‖W_O v‖')
# Highlight the workhorse tokens
ax2.scatter(kvzip_score[10:14], kvzipplus_score[10:14],
            c='#00A8A8', s=90, zorder=5, label='"Workhorse" tokens:\nmodest attention, large ‖W_O v‖')
ax2.plot([0, 1], [0, 1], color='#1A1A1A', linewidth=1, linestyle=':', alpha=0.5)
ax2.set_xlabel("KVzip score (attention only)", fontsize=10)
ax2.set_ylabel("KVzip+ score (contribution-normalized)", fontsize=10)
ax2.set_title("Score correlation — divergence reveals the footnote problem", fontsize=10)
ax2.legend(fontsize=8.5, loc='upper left')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
```

The right panel is the key picture. Points below the diagonal are "footnote" tokens: KVzip would keep them (high raw attention) but KVzip+ correctly identifies them as low-contribution. Points above the diagonal are "workhorse" tokens: KVzip might evict them (modest attention) but KVzip+ keeps them because $\|W_O v_i\|$ is large. The correction matters — especially at aggressive compression ratios where the margin for error is thin.

For more on the background to this sparsity pattern, see [Softmax's Long Tail](../07-attention-sparsity/). The full derivation of the normalization is in [Weight ≠ Contribution](../08-contribution-norm/).

## Two Fatal Flaws

KVzip+, when computed honestly, produces beautiful importance scores. In controlled experiments, it matches or exceeds H₂O at every compression ratio while avoiding the kernel-compatibility problem (scoring happens at prefill, not inside the attention kernel). It shipped in the KVzip research code and reproduced faithfully.

It did not ship in production. Two reasons.

**Flaw one: 2× prefill cost.** The extended prompt — original context, plus the repeat instruction, plus the repeated context again — is roughly twice the length of the original input. For a 128k-token document, the model must process approximately 256k tokens just to score the original 128k. Prefill time and memory scale linearly with sequence length. At 128k context, prefill is already the expensive step. Doubling it is not a rounding error — it doubles the entire time-to-first-token cost. NVIDIA's own inference benchmarks at the time showed that systems were already memory-bandwidth-bound at 128k; KVzip's doubled prefill made the situation worse, not better.

{{% callout type="warning" title="The 2× Prefill Wall" %}}
For long-context use cases — RAG over a 100k-word document, legal document QA, code-base reasoning — the entire value proposition of KV cache pruning is faster inference. KVzip's 2× prefill overhead *directly contradicts this*. You cannot claim to speed up inference with a method that doubles the most expensive step.
{{% /callout %}}

**Flaw two: scoring ends at prefill.** The KVzip pretext task requires a complete, finished context to repeat. You cannot run the copy-and-paste task on tokens that haven't been generated yet. For a model doing **chain-of-thought reasoning** — generating 16,000 to 32,000 reasoning tokens before outputting an answer — the generated tokens are themselves filling the KV cache. KVzip cannot score any of them. The entire approach is blind during decoding, which is precisely where long-context {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} pressure is worst.

{{% marginnote %}}
Chain-of-thought reasoning with long scratchpads became the dominant inference pattern in 2025–26, driven by o1-style models. This made KVzip's decode-time blindness more damaging, not less, as the field moved forward.
{{% /marginnote %}}

{{< crosshead >}}The Shape of the Problem{{< /crosshead >}}

Step back and see what we have. KVzip+ produces, empirically, the right signal. The scores $s_i^+$ genuinely identify which tokens can be evicted without accuracy loss. They're not approximating importance — they're measuring it, directly, from the model's own attention behavior.

The problem is not the *signal*. The problem is the *cost of acquiring the signal*.

Is there something the model computes *anyway*, in the normal course of a forward pass, that is *predictive* of the KVzip+ score? Some feature of the {{< wiki "residual-stream" >}}hidden state{{< /wiki >}} $h_i$ — available at layer $l$ without any additional forward pass — that tells you whether token $i$ will score high or low?

If such a feature exists, you could train a tiny predictor to map $h_i \to \hat{s}_i^+$, run it cheaply at each layer, and prune based on the prediction rather than the measurement. You would never need to run the pretext task at all.

That is exactly what the KVzap paper attempts. And the answer to whether such a feature exists turns out to be: yes, surprisingly cleanly.

**Continue to** → **[Ghost in the Hidden State](../05-ghost-state/)** — the discovery that a model's hidden states already know which tokens are important, before attention even runs.
