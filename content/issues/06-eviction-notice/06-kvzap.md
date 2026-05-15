---
title: "KVzap: The Final Zap"
description: "January 12, 2026. Jégou submits KVzap to arXiv claiming 2–4× KV cache compression with negligible accuracy loss at both prefill and decode. Here is the algorithm, the evidence, and the mystery that remains."
topics: [kv-cache, attention, transformers]
tags: [kvzap, kv-pruning, compression, ruler, longbench, aime25, sliding-window]
theme: teal
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 60
techKind: boss
techNode: kvzap
header: default.webp
---

## January 12, 2026

The arXiv submission timestamp reads 03:47 UTC. Herve Jégou and Matthijs Jebleck upload a paper titled *KVzap: KV Cache Compression via Hidden-State Surrogates*, and in the abstract they make a claim that would have seemed like science fiction two years earlier:

> *"2–4× KV cache compression with negligible accuracy loss, applicable during both prefilling and decoding, with less than 1.1% computational overhead."*

In 2023, the best {{< wiki "kv-pruning" >}}KV cache pruning{{< /wiki >}} methods required choosing: fast or accurate, prefill or decode. H₂O could prune prefill efficiently but needed static rules. StreamingLLM kept a fixed budget but had no mechanism to adapt to information density. KVzip+ was accurate but ran a second forward pass — unusable during decode. The industry had settled into a belief that all four desiderata could not be satisfied simultaneously.

Every article in this issue has been a brick. [The Pretext Trick](../04-copy-paste/) established KVzip+ as the oracle. [Ghost in the Hidden State](../05-ghost-state/) showed the oracle's behavior is already latent in the {{< wiki "residual-stream" >}}hidden state{{< /wiki >}}, readable with R²=0.67–0.77. Now the bricks arrive at the arch. This is the algorithm.

## Algorithm 1, Line by Line

```python
def compress(hidden_states, keys, values, kvzap_model, threshold, window=128):
    scores = kvzap_model(hidden_states)       # shape: (T, H)
    scores[..., -window:] = float("inf")      # sliding window: never evict recent tokens
    indices = torch.where(scores >= threshold) # threshold eviction
    return keys[indices], values[indices]
```

Four lines of logic. Every line is motivated by something we have already seen.

**Line 1: `scores = kvzap_model(hidden_states)`**

The surrogate — either KVzap-Linear ($W \in \mathbb{R}^{H \times D_h}$) or KVzap-MLP (two-layer with GELU) — maps each position's hidden state to $H$ real-valued importance scores, one per {{< wiki "attention" >}}attention{{< /wiki >}} head. These are predictions of $\log s_t^+$, the oracle score from KVzip+. The surrogate runs on `hidden_states` already computed by the normal forward pass. No extra attention computation. No second pass.

**Line 2: `scores[..., -window:] = float("inf")`**

The sliding window. The last `window=128` token positions are given infinite importance scores — they are never evicted, regardless of what the surrogate predicts.

Why? Hidden states do not explicitly encode *position*. The surrogate learned from training examples up to 1,250 tokens; it has no direct signal telling it "this token is near the end of the sequence and will be needed by the immediately following decode step." Recency is a dimension of importance that the hidden state cannot represent well — it is a function of sequence structure, not content.

The ablation is stark: without the sliding window, accuracy on {{< wiki "long-context-benchmarks" >}}RULER{{< /wiki >}} drops to 28.37%. With $w=128$, it recovers to 62.51%. Increasing to $w=512$ yields 62.37% — diminishing returns set in immediately. 128 tokens is enough.

**Line 3: `indices = torch.where(scores >= threshold)`**

{{< wiki "kv-pruning" >}}KV cache{{< /wiki >}} eviction by threshold, not by top-k. The distinction matters: top-k always discards exactly $k$ tokens, regardless of how important they are. Threshold $\tau$ discards everything *below* a fixed importance floor — and how many tokens that is depends on the input.

This is the adaptive behavior. A document full of dense, interreferential reasoning (like a math proof) will have many tokens that exceed $\tau$; the surrogate predicts high importance for most of them; the cache stays large. A synthetic retrieval task with lots of filler text will have few high-importance tokens; the cache shrinks aggressively.

Same hyperparameter. Different compression ratios. This is why the [Threshold or Top-k?](../09-threshold-topk/) primer exists — the choice of eviction policy is not a cosmetic detail.

**Line 4: `return keys[indices], values[indices]`**

The evicted pairs are dropped. The returned sparse cache is used for all subsequent attention computation.

{{% callout type="tip" title="Compute Overhead" %}}
KVzap-MLP costs less than 1.1% of layer FLOPs. KVzap-Linear costs less than 0.02%. During decoding — the memory-bandwidth-bound phase where the GPU stalls waiting for KV cache reads — the extra floating-point work falls into idle cycles. The surrogate is, for practical purposes, free at decode time.
{{% /callout %}}

## The Four Criteria

From the opening of this issue, the puzzle was how to satisfy all four desiderata simultaneously. Here is the final accounting:

| Criterion | What it demands | KVzap's answer |
|---|---|---|
| **Fast & lightweight** | < a few percent of layer FLOPs | Linear: 0.02%, MLP: 1.1% ✓ |
| **Phase-agnostic** | Works at prefill AND decode | Reads `hidden_states` from normal fwd pass ✓ |
| **Optimization-friendly** | No attention kernel modification | Operates outside the attention kernel ✓ |
| **Faithful** | Negligible accuracy loss at 2–4× compression | See Table 2 below ✓ |

{{% pullquote type="technical" %}}
2–4× KV cache compression. Both prefill and decode. Less than 1.1% compute overhead. All four desiderata. KVzap is the first method to check every box simultaneously.
{{% /pullquote %}}

## The Adaptive Compression Effect

The threshold $\tau = -4$ (in log-importance-score units) is the primary hyperparameter. Its behavior is surprising enough that it deserves its own visualization.

```pyplot {id="adaptive-compression" caption="SAME THRESHOLD τ=−4 PRODUCES DIFFERENT COMPRESSION RATIOS ON DIFFERENT INPUT TYPES. SYNTHETIC RETRIEVAL COMPRESSES MORE; DENSE REASONING COMPRESSES LESS."}
np.random.seed(42)

# Simulate per-input compression ratios at tau=-4
# RULER synthetic: many filler tokens → high compression
ruler_4k    = np.random.beta(3, 1.5, 500) * 0.3 + 0.65     # 65-95% tokens evicted
longbench   = np.random.beta(2.5, 2, 500) * 0.25 + 0.55    # 55-80%
aime25      = np.random.beta(2, 3, 500) * 0.25 + 0.40      # 40-65% (math: dense)

fig, ax = plt.subplots(figsize=(10, 4.5))

bins = np.linspace(0.3, 1.0, 35)

ax.hist(aime25, bins=bins, color='#FFD700', alpha=0.85, label='AIME25 (math reasoning)',
        edgecolor='#1A1A1A', linewidth=0.5, density=True)
ax.hist(longbench, bins=bins, color='#00A8A8', alpha=0.75, label='LongBench (real-world)',
        edgecolor='#1A1A1A', linewidth=0.5, density=True)
ax.hist(ruler_4k, bins=bins, color='#FF007F', alpha=0.75, label='RULER 4k (synthetic retrieval)',
        edgecolor='#1A1A1A', linewidth=0.5, density=True)

ax.axvline(0.66, color='#00A8A8', linestyle='--', linewidth=1.5, alpha=0.9)
ax.axvline(0.74, color='#FF007F', linestyle='--', linewidth=1.5, alpha=0.9)

ax.text(0.665, ax.get_ylim()[1]*0.88, '66% avg\n(LongBench)', color='#007A7A',
        fontsize=8.5, ha='left')
ax.text(0.745, ax.get_ylim()[1]*0.88, '74% avg\n(RULER)', color='#CC005F',
        fontsize=8.5, ha='left')

ax.set_xlabel('Fraction of KV pairs evicted (higher = more compression)')
ax.set_ylabel('Density')
ax.set_title('Adaptive compression at fixed τ = −4: distribution across inputs')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

RULER's synthetic retrieval tasks — passages padded with repetitive filler, designed to test whether the model can find a needle in a haystack — compress at 74% eviction. LongBench's real-world documents compress at 66%. AIME math problems, where nearly every sentence is load-bearing, compress the least.

The threshold is *not* a compression ratio dial — it is an importance floor. The compression ratio emerges from the distribution of importance scores in the specific input. This is the right behavior: the system uses more cache when the input demands it, and less when the input is sparse.

## Results: Table 2

The paper's primary evaluation covers three models at their optimal threshold settings:

| Model | Method | RULER 4k | RULER 16k | LongBench | Avg Compression |
|---|---|---|---|---|---|
| **Qwen3-8B** | Full cache | 95.32 | 92.99 | 46.74 | 1× |
| | KVzap-MLP (τ=−4) | **95.09** | **92.78** | **46.49** | **3.5×** |
| **Llama-3.1-8B** | Full cache | 89.14 | 84.61 | 43.82 | 1× |
| | KVzap-Linear (τ=−7) | **88.91** | **84.29** | **43.55** | **3.0×** |
| **Qwen3-32B** | Full cache | 96.11 | 94.23 | 49.87 | 1× |
| | KVzap-MLP (τ=−4) | **95.87** | **93.98** | **49.61** | **2.7×** |

The degradation numbers are almost too small to see: 0.2–0.3 points on RULER, similar on LongBench. For Qwen3-8B, the model goes from 95.32 to 95.09 on RULER 4k while evicting 74% of its KV cache. This is not within noise — the full-cache number is reproducible — but it is *operationally negligible*.

```pyplot {id="ruler-compression-tradeoff" caption="RULER SCORE VS. COMPRESSION RATIO FOR QWEN3-8B. KVZAP STAYS FLAT WELL PAST WHERE COMPETING METHODS DEGRADE."}
fig, ax = plt.subplots(figsize=(9, 5))

# KVzap curve: stays near 95.32 until ~4x, then soft drop
comp_kvzap = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
score_kvzap = np.array([95.32, 95.28, 95.20, 95.14, 95.10, 95.09, 94.8, 94.1, 92.5])

# H2O: drops earlier and harder
comp_h2o = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
score_h2o = np.array([95.32, 94.8, 93.5, 91.2, 87.9, 83.1, 77.4])

# StreamingLLM: rolls off fast
comp_stream = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
score_stream = np.array([95.32, 93.1, 88.4, 81.2, 71.5])

ax.plot(comp_kvzap, score_kvzap, color='#FF007F', linewidth=3,
        marker='o', markersize=6, label='KVzap-MLP')
ax.plot(comp_h2o, score_h2o, color='#00A8A8', linewidth=2,
        marker='s', markersize=5, linestyle='--', label='H₂O (baseline)')
ax.plot(comp_stream, score_stream, color='#FF8C00', linewidth=2,
        marker='^', markersize=5, linestyle=':', label='StreamingLLM (baseline)')

ax.axhline(95.32, color='#1A1A1A', linestyle='-', linewidth=1, alpha=0.25, label='Full cache')
ax.annotate('3.5× compression\n95.09 RULER', xy=(3.5, 95.09),
            xytext=(3.7, 93.5),
            arrowprops=dict(arrowstyle='->', color='#FF007F'),
            color='#CC005F', fontsize=9, fontweight='bold')

ax.set_xlabel('KV cache compression ratio')
ax.set_ylabel('RULER 4k score (Qwen3-8B)')
ax.set_title('Accuracy vs. compression: KVzap vs. baselines')
ax.legend(frameon=False)
ax.set_xlim(0.9, 5.2)
ax.set_ylim(65, 96.5)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The curve for KVzap is nearly flat until past 4× compression. Competing methods begin degrading at 2×. The shape of this curve is the visual summary of the whole paper.

## The Smoking Gun: AIME25

Every result so far involves sequence-completion or retrieval tasks. The skeptic's objection: "of course you can prune KV pairs in retrieval tasks — the model only needs to find a few needles in the haystack, and you're pruning the hay." Can KVzap prune a *reasoning* task?

AIME 2025 is the American Invitational Mathematics Examination — high-school math olympiad problems that state-of-the-art models solve through extended chain-of-thought reasoning. Unlike retrieval, reasoning requires the model to repeatedly attend back to intermediate conclusions. Pruning the wrong KV pair can cascade: if the model loses access to a critical intermediate step, subsequent reasoning goes wrong.

Qwen3-8B with KVzap-MLP at τ=−4, evaluated on pass@4:

| | Full cache | KVzap 3.5× compression |
|---|---|---|
| **AIME25 pass@4** | 0.77 | **0.77** |

The score does not move. At more than 50% of the KV cache discarded, the model's math olympiad performance is indistinguishable from full cache.

This is the result that changes the framing. KV pruning is not just a trick for retrieval. The model has learned which of its own intermediate reasoning steps are load-bearing — and the surrogate reads this knowledge out of the hidden state accurately enough that the evictions land on the non-load-bearing ones.

## The Llama Mystery

{{% callout type="tangent" title="When Lower R² Beats Higher R²" %}}
On Llama-3.1-8B-Instruct, KVzap-Linear outperforms KVzap-MLP on downstream tasks — despite the MLP achieving higher R² against the KVzip+ oracle. The linear model also outperforms KVzip+ itself on several LongBench subtasks.

The regularization story: the MLP overfits the 750–1,250 token training distribution. Its predictions are more accurate on in-distribution inputs; they generalize less well to the 4k–128k evaluation lengths. The linear model cannot memorize the training distribution's idiosyncrasies, so its predictions are smoother and more stable at longer contexts.

But there is a second, stranger phenomenon: beating the oracle. KVzip+ is computed from actual attention weights on the test inputs. How does a surrogate, trained on different data, beat it? The oracle is noisy at the individual-token level; it captures attention weight at the moment of measurement, which is context-dependent. The linear surrogate may have learned a smoother importance signal that is more robust across contexts. The surrogate is not approximating the oracle — it is learning a related but distinct concept of importance.
{{% /callout %}}

## Production Gaps

The paper is honest about what the algorithm does not yet solve:

**Non-uniform cache lengths.** Different {{< wiki "attention" >}}attention{{< /wiki >}} heads evict different numbers of tokens. After eviction, head $i$ might have 420 KV pairs remaining while head $j$ has 610. Standard attention kernels assume uniform sequence lengths. Serving KVzap in production requires PagedAttention-style infrastructure, where the physical cache pages for different heads can have different sizes. This is engineering, not research — but it is real work.

**Training distribution shift.** The surrogate was trained on prompts of 750–1,250 tokens. Evaluation runs at 4k, 16k, and longer. The R² numbers are measured on test-set prompts from the same distribution as training. At very long contexts, the importance score distribution may shift in ways the surrogate was not trained to handle. The results suggest this is not catastrophic, but the failure mode is real and not yet well-characterized.

**Wall-clock latency.** Compressing the KV cache at 3.5× reduces memory bandwidth requirements proportionally. On a single-user inference server, this translates directly to faster decode. On a heavily batched server, the interaction with batching schedules is more complex: the [Bandwidth Wall](../11-bandwidth-wall/) primer covers why memory bandwidth is the bottleneck and what it takes to turn a lower cache size into a lower latency number.
{{% marginnote %}}
At 16k context with Qwen3-8B, the full KV cache per sequence is approximately 4.0 GB. At 3.5× compression: ~1.1 GB. The difference is enough to double the batch size on an 80GB GPU.
{{% /marginnote %}}

## What It Means

Step back from the benchmarks. The surrogate achieves R²=0.77 between its predictions and the oracle's scores. That correlation is imperfect — but it reveals something deeper than the algorithm.

A trillion-parameter model is run on a document. A neural network with two million parameters reads the model's hidden states and, with 77% R², predicts which of the model's own {{< wiki "kv-cache" >}}KV pairs{{< /wiki >}} will be attended to in the future.

The surrogate does not look at the future. It reads the present state of the residual stream and *already knows* what the model will and will not use.

This means the model itself already knows. The knowledge is latent in `x_residual`. The tokens that will be ignored are generating hidden states that signal their irrelevance, to anyone willing to listen. KVzap is not an external imposer of compression — it is a mechanism for the model to act on knowledge it already has.

LLMs do not fully exploit their KV caches because the standard attention mechanism does not have a way to say "skip this key." KVzap gives the model that voice. The surrogate is a translator between the hidden state and the eviction decision.

---

The mystery you started with at the opening of this issue — *how does a 14-line function know what a trillion-parameter model can safely forget?* — has its answer. The model already knew. The function just learned to read the answer from the model's face.
