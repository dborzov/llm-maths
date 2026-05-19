---
title: "Sparse Attention: six years of papers, zero products"
short_title: "Sparse Attention"
description: "Every sparse-attention proposal from Longformer (2020) to DuoAttention (2024) failed at least one of five production criteria — fast, phase-agnostic, optimization-friendly, faithful, trainable from scratch."
blurb:
  - "Five criteria for production deployment. Every method from 2020 to 2024 passes at most four."
  - "Fixed patterns (Longformer, BigBird) are fast but not faithful. Learned patterns are faithful but not optimization-friendly."
  - "Trainable from scratch: if sparsity needs dense pretraining first, you pay twice for a $5M frontier model run."
  - "Twelve methods. Six hundred pages of appendices. One engineer's verdict after a week of reading: none of them."
topics: [attention, sparse-attention, history]
tags: [longformer, bigbird, reformer, routing-transformer, linformer, performer, streamingllm, h2o, snapkv, duoattention]
theme: teal
math: true
draft: false
date: 2026-05-16T09:30:00-04:00
issue: 7
weight: 40
techKind: mainline
techNode: sparse-detour
header: 04-sparse-detour.webp
---

## A Stack of Papers and Zero Products

Imagine a scene: late 2023, a GPU cluster humming in a San Francisco research lab. A senior engineer is flipping through a printed stack of papers — Longformer, BigBird, Reformer, Linformer, Performer, Routing Transformer, H₂O, SnapKV. Twelve methods. Six hundred pages of appendices. Twelve arxiv numbers.

She has one question: which of these can I actually ship?

After a week of reading and a weekend of implementation experiments, she concludes: **none of them.** Not a single one. Not in a frontier model at production scale. The methods are technically ingenious. The ablations are real. The FLOPs arithmetic checks out. But every single method fails at least one test that a production inference stack requires — and often fails it in a way that is fundamental, not fixable with an engineering sprint.

This article is the forensic reconstruction of why. It is not a survey. It is a failure analysis.

{{< crosshead >}}The Five Production Criteria{{< /crosshead >}}

In [Issue 6's KVzap chapter](/comicbook/06-kvcache-pruning/06-kvzap/), we established four criteria that any production inference component must pass. Every {{< wiki "attention" >}}attention{{< /wiki >}} mechanism in a frontier model gets tested against this list whether the paper authors knew it or not:

1. **Fast** — the mechanism adds less than 1% to total inference latency. Not 10%. Not "comparable to." Less than one percent.
2. **Phase-agnostic** — the mechanism works identically in the prefill phase (processing a user prompt) and the decode phase (generating token by token). A mechanism that requires knowing the sequence length in advance is phase-broken.
3. **Optimization-friendly** — composes cleanly with the rest of the production stack: FlashAttention, tensor parallelism, PagedAttention. If a method requires a custom kernel that conflicts with FlashAttention, it is not optimization-friendly.
4. **Faithful** — preserves output quality on benchmarks to at least 99% of the dense baseline. Not 95%. Not "comparable with some headroom."

For sparse {{< wiki "attention" >}}attention{{< /wiki >}} specifically, there is a fifth criterion that applies at the architectural level:

5. **Trainable from scratch** — you do not pay extra compute to introduce sparsity after pretraining. A method requiring full-dense pretraining followed by fine-tuning to learn the sparse pattern costs twice. If you're going to train a $5M frontier model, the sparse pattern has to be part of the training from day one.

{{% pullquote type="theorem" %}}
Every sparse attention method from 2020 to 2024 passes at most four of these five criteria. Not one method passes all five. That is the whole story.
{{% /pullquote %}}

{{< crosshead >}}The Running Example{{< /crosshead >}}

Before the tour, let us fix a concrete setup to anchor all the failures against. Suppose we have a sequence of $T = 8192$ tokens. We want to compute {{< wiki "attention" >}}attention{{< /wiki >}} for every query token against all past keys, but we want to do so *sparsely* — attending to at most $k = 256$ tokens per query instead of all $T$.

Here is what *ideal* sparse attention would look like if we had a perfect scorer:

```python
# Ideal sparse attention: a fast learned scorer that picks the right tokens
def ideal_sparse_attend(q, k, v, scorer, top_k=256):
    """
    q: (d_head,)       — one query vector
    k: (T, d_head)     — all past keys
    v: (T, d_head)     — all past values
    scorer: fn(q, k) → (T,)  — cheap importance score for each past token
    """
    T = k.shape[0]

    # Step 1: Score ALL past tokens cheaply
    raw_scores = scorer(q, k)          # O(T × d_score) — ideally d_score << d_head

    # Step 2: Keep only the top-k
    top_idx = np.argsort(raw_scores)[-top_k:]   # O(T log T) argpartition

    # Step 3: Full attention over top-k tokens only
    k_selected = k[top_idx]            # (k, d_head)
    v_selected = v[top_idx]            # (k, d_head)
    scores_full = q @ k_selected.T / np.sqrt(q.shape[-1])  # (k,)
    weights = np.exp(scores_full - scores_full.max())
    weights /= weights.sum()           # softmax over k, not T
    return weights @ v_selected        # (d_head,)

# Napkin math: savings at T=8192, k=256, d_head=128
T, k_kept, d = 8192, 256, 128
flops_dense   = T * d                # 1,048,576 — full attention row
flops_sparse  = k_kept * d          #    32,768  — attend over k_kept tokens
flops_scorer  = T * 32              #   262,144  — scorer at d_score=32
total_sparse  = flops_scorer + flops_sparse
print(f"Dense attention row:  {flops_dense:>9,} FLOPs")
print(f"Scorer overhead:      {flops_scorer:>9,} FLOPs")
print(f"Sparse attention row: {flops_sparse:>9,} FLOPs")
print(f"Total sparse:         {total_sparse:>9,} FLOPs")
print(f"Ratio: {total_sparse/flops_dense:.2f}× dense")
# → Total sparse: 294,912 FLOPs → 0.28× dense
# i.e., we'd use 28% of dense compute with a perfect scorer
```

The trick is: `scorer()` must be *much* cheaper than full attention. And `scorer()` must produce differentiable gradients so we can train the scoring weights along with everything else. Every method below is an attempt to build this scorer — and each fails in a different way.

{{< crosshead >}}A Tour by Failure Mode{{< /crosshead >}}

## Longformer (2020) — fixed-pattern sparsity

**Iz Beltagy, Matthew Peters, Arman Cohan at AllenAI.** The paper that restarted the field after the Transformer's O(n²) problem was named. The idea is surgical: instead of attending everywhere, each token attends to:
- A **sliding window** of size $w$ — that is, $w/2$ tokens to the left and $w/2$ to the right.
- A small set of **global tokens** at user-specified positions — positions that attend to *everything* and that *everything* attends to.

The global tokens are the trick. In a document classification task, the `[CLS]` token is a perfect global token: you want the model's final answer to have attended to every part of the document, so you make CLS global. In question answering, each question token is global: the model needs the question to inform reading of the entire passage.

The sliding window maps cleanly to a GPU kernel. No scatters. No variable-length structures. Longformer ships with a custom CUDA kernel that is genuinely fast. **Criteria 1, 2, and 3 pass.**

The problem is Criterion 4. For tasks where you *cannot* pre-specify which positions should be global, quality collapses. The paper says this quietly:

> "For tasks that do not have a natural global token (e.g., language modeling), we add task-agnostic global tokens at the beginning of the document."

Translation: for generic language modeling — which is *all* of pretraining — you have to pick global positions by hand, before you know what the model will care about. The first four tokens are global not because they're important but because someone had to put global tokens somewhere. This is not learning. It is a prior.

At 16K context, the quality gap between Longformer and full attention on long-range retrieval tasks (finding a specific fact buried in a document) is substantial. The model simply cannot attend to arbitrary past positions it hasn't pre-designated as global.

**Criteria failed: 4 (Faithful), 5 (Trainable from scratch for general LM).**

## BigBird (2020) — sliding + global + random

**Zaheer, Gurudath, Prateek et al. at Google.** BigBird adds a third sparsity pattern to Longformer's two: **random attention**. Every query attends to a random subset of past tokens in addition to its window and any global positions.

The motivation is theoretical. Bigbird's authors prove that a combination of sliding-window, global, and random patterns can approximate the expressive power of full attention for a certain class of graph problems. The random component is what provides the approximation guarantee — without it, you can't route information across arbitrary token pairs.

The theoretical guarantee is correct in expectation over random graph realizations. In practice, you are running one specific random pattern per forward pass, and the variance is high.

But the deeper problem is the kernel. The random pattern requires **scatter/gather operations**: for each query at position $t$, you gather keys from random positions $\{r_1, r_2, ..., r_m\}$ scattered across the sequence. Those positions are not adjacent in memory. Each gather is a separate memory access to a potentially uncached cache line. On a GPU, coalescing memory access is critical — a 32-thread warp needs all 32 threads to access consecutive addresses to hit full bandwidth. Random positions destroy coalescing.

The BigBird paper includes a custom CUDA kernel implementation. Read the kernel carefully and you will find that it is a research implementation — correct, but heavily optimized for the paper's benchmark scenarios, not for a production tokenizer that processes batches of variable-length sequences from arbitrary users. No frontier model shipped on it.

**Criteria failed: 3 (Optimization-friendly).**

## Reformer (2020) — LSH bucketing

**Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya at Google Brain.** The Reformer's insight is elegant: if a query $q$ and key $k$ have high dot product (high attention weight), they must be similar vectors. Locality-Sensitive Hashing (LSH) is designed to find similar vectors efficiently. So hash queries and keys into buckets; attend only within each bucket.

The hash used is angular LSH — project $q$ and $k$ onto random hyperplanes and record which side each falls on. After enough projections, similar vectors hash to the same bucket with high probability. The bucket assignment provides the "who attends to whom" map, and the map is O(T log T) to compute instead of O(T²).

Here is where it breaks down. To attend only within each bucket, you need to bring same-bucket items *adjacent* in memory — because GPUs process data in fixed-size tiles, and you need all the keys for a given query to be contiguous. That means sorting by bucket assignment.

```python
# LSH chunked sort — the FlashAttention-killer
buckets = lsh_hash(keys, n_buckets=64)   # hash each key → bucket index
sort_idx = buckets.argsort()             # sort by bucket: NON-DIFFERENTIABLE
keys_sorted = keys[sort_idx]             # gather to sorted order: BREAKS STRIDED ACCESS

# Problem 1: argsort is non-differentiable.
# Gradients do not flow through bucket assignments.
# The training signal for "learn to hash similar queries together" is severed.

# Problem 2: FlashAttention expects fixed-size tiles.
# Bucket sizes are variable. Bucket 3 might have 600 tokens; bucket 17, 12.
# A fixed tile size of 128 means bucket 3 takes 5 tiles, bucket 17 takes 1 tile.
# The kernel grid has uneven load: some SMs run 5× harder than others.
# Peak GPU utilization collapses.

# Problem 3: The sort itself is O(T log T) on CPU or O(T log T) on GPU.
# At T=128K tokens, this is measurable latency. Not under 1%.
print("LSH sort overhead at T=128K:")
T = 128_000
sort_flops_approx = T * np.log2(T)  # rough O(T log T)
print(f"  argsort FLOPs (rough): {sort_flops_approx:,.0f}")
# For comparison, one attention head at T=128K, d=128:
attn_flops = T * 128
print(f"  attention head FLOPs:  {attn_flops:,.0f}")
print(f"  sort overhead ratio:   {sort_flops_approx/attn_flops:.2f}×")
```

Multiple hash repetitions are needed to get quality back up (if query $q$ and key $k$ happen to fall in different buckets due to hash variance, you miss their interaction). With 4–8 repetitions, the sorting overhead grows proportionally and the FLOPs savings from sparsity are eaten by the sort.

**Criteria failed: 1 (Fast — sort overhead), 3 (Optimization-friendly — FlashAttention incompatible).**

## Linformer (2020) — low-rank projection

**Sinong Wang, Belinda Li, Madian Khabsa, Han Fang, Hao Ma at Facebook AI.** Instead of selecting which tokens to attend to, why not compress the key and value matrices to a smaller size? Project:

$$K' = E \cdot K \in \mathbb{R}^{m \times d}, \quad V' = F \cdot V \in \mathbb{R}^{m \times d}$$

where $E, F \in \mathbb{R}^{m \times n}$ are learned projection matrices and $m \ll n$ (the sequence length). The attention computation becomes $O(n \cdot m)$ instead of $O(n^2)$.

The math works. The benchmarks on fixed-length tasks look fine. And then you hit inference.

The projection matrices $E$ and $F$ have dimension $m \times n$ — they depend on $n$, the sequence length during training. If you train with $n = 512$ and then want to run inference at $n = 2048$, your projection matrices don't exist. They were sized for 512. You cannot extrapolate. A new $n$ means retraining.

This is not an engineering fix. The low-rank approximation is a **parameter tied to a sequence length**. The whole value of a modern LLM is handling variable-length user inputs. A model that can only operate at its training sequence length is useless in production.

{{% callout type="warning" %}}
**The phase-agnostic check catches Linformer immediately.** At decode time, the sequence grows one token per step. The projection matrix $E$ would need to be re-learned for every possible prefix length. In practice, Linformer is just not usable for autoregressive generation — which is the primary use case for frontier LLMs.
{{% /callout %}}

The quality gap on long-context retrieval tasks (LongBench) between Linformer and full {{< wiki "attention" >}}attention{{< /wiki >}} is also substantial — 3–5 points on retrieval-oriented benchmarks — because the low-rank projection provably loses information that is necessary for precise retrieval.

**Criteria failed: 2 (Phase-agnostic), 4 (Faithful at extrapolated lengths).**

## Performer (2020) — kernelized attention

**Krzysztof Choromanski, Valerii Likhosherstov, David Dohan et al. at Google Brain.** The most mathematically ambitious entry in this tour. The Performer introduces FAVOR+ (Fast Attention Via positive Orthogonal Random features), which rewrites the {{< wiki "softmax" >}}softmax{{< /wiki >}} attention kernel as a linear dot product:

$$\text{Attention}(Q, K, V) \approx \phi(Q) \cdot (\phi(K)^T V)$$

where $\phi$ is a random feature map designed to approximate the exponential function. This makes attention $O(T \cdot d)$ instead of $O(T^2)$. The theoretical derivation is correct. The random feature map provably converges to the true softmax in expectation as the number of random features increases.

The problem is practical: the approximation introduces noise. On real language model benchmarks — perplexity on C4, LAMBADA accuracy, HumanEval — there is a persistent gap of 1–3 perplexity points versus full attention that nobody could close through 2022. Research groups tried increasing the number of random features (larger $\phi$ → more FLOPs, erasing the savings), tried alternative feature maps, tried hybrid training schedules. None of it closed the gap to 99% quality retention.

The quality gap is not a bug in the implementation. It is fundamental: softmax attention is not well-approximated by a kernel with bounded random features on the token distributions that LLMs actually see. The distributions of $q \cdot k$ values in a real transformer at 13B+ parameters are heavy-tailed and structured in ways that the random feature map cannot capture without a prohibitively large feature bank.

**Criteria failed: 4 (Faithful).**

## Routing Transformer (2021) — k-means clusters

**Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier at Google.** The most intellectually honest entry in this list — the authors knew their method was data-dependent and made the data-dependence explicit. Cluster the keys using k-means. Each query attends only to tokens in its cluster. The cluster assignments change every forward pass as the model learns.

This is genuinely trainable and data-dependent — the clusters adapt to the content being processed. Quality on long-document tasks is competitive with full attention.

The cost has two components:

**Component 1: k-means at every layer.** Running k-means with $k=128$ cluster centers over $T=128K$ keys, repeated at every transformer layer (say, 80 layers for a V3-class model), is a large fraction of total forward pass compute.

```python
# Napkin math: routing transformer k-means cost vs. attention cost
T = 128_000    # sequence length
d = 128        # head dimension
L = 80         # transformer layers
k = 128        # clusters
iters = 5      # k-means iterations

# K-means cost per layer per head: T * k * d per iteration
kmeans_flops_per_layer = iters * T * k * d
total_kmeans_flops = kmeans_flops_per_layer * L
print(f"k-means FLOPs (all layers, all iters): {total_kmeans_flops:,.0f}")

# Full attention cost (for comparison): T^2 * d per layer per head
attention_flops_per_layer = T * T * d
total_attention_flops = attention_flops_per_layer * L
print(f"Full attention FLOPs (all layers):     {total_attention_flops:,.0f}")
print(f"k-means overhead as fraction of dense attn: {total_kmeans_flops/total_attention_flops:.1%}")
# → k-means overhead is 0.4% per head but scales per head per layer
# With 128 heads: 0.4% × 128 = 51% — you've wiped out your savings
```

**Component 2: Cluster boundary gradients.** The cluster assignment is an argmin — which cluster center is closest to this key? Argmin is not differentiable. The training signal for "learn cluster centers that put similar keys together" cannot flow back through the assignment step cleanly. Straight-through estimators exist, but they introduce bias.

**Criteria failed: 1 (Fast — k-means overhead), 3 (Optimization-friendly — non-differentiable assignment).**

```pyplot {id="sparse-attention-fail-matrix" caption="THE CRITERIA MATRIX OVER 2020-2024 SPARSE ATTENTION METHODS. NSA AND DSA ARE THE FIRST METHODS TO TICK ALL FIVE."}
methods = ["Longformer", "BigBird", "Reformer", "Linformer", "Performer",
           "Routing-T", "Sliding+Sinks", "H₂O", "SnapKV", "DuoAttn",
           "NSA (2025)", "DSA (2025)"]
criteria = ["Fast", "Phase-agnostic", "Kernel-friendly", "Faithful", "Trainable-from-scratch"]

# 1 = pass, 0 = fail
data = np.array([
    [1,1,1,0,1],  # Longformer
    [1,1,0,1,1],  # BigBird
    [0,1,0,1,1],  # Reformer
    [1,1,1,0,1],  # Linformer
    [1,1,1,0,1],  # Performer
    [0,1,0,1,1],  # Routing
    [1,1,1,0,0],  # SW+sinks
    [1,0,0,1,0],  # H2O
    [0,0,1,1,0],  # SnapKV
    [1,1,1,1,0],  # DuoAttn
    [1,1,1,1,1],  # NSA
    [1,1,1,1,1],  # DSA
])

fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(criteria)))
ax.set_xticklabels(criteria, rotation=20, ha='right')
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
for i in range(len(methods)):
    for j in range(len(criteria)):
        sym = "✓" if data[i,j] else "✗"
        ax.text(j, i, sym, ha='center', va='center', fontsize=12, color='#1A1A1A')
ax.set_title("the criteria matrix: every prior method failed at least one")
plt.tight_layout()
```

{{< crosshead >}}The Engineering Retreats{{< /crosshead >}}

The methods above were pure research. Starting in 2022, some teams retreated to simpler patterns that actually shipped — but at the cost of fundamental capability.

## Sliding Window + Sinks (Mistral, StreamingLLM, 2023)

**Mistral-7B** (September 2023) shipped with a window size $W = 4096$: each token attends only to the 4096 most recent tokens. No global tokens. No learned selection. Just a hard cutoff.

It worked. Mistral-7B was faster than Llama-2-7B on short to medium contexts. The inference kernel was simple. Production engineers could maintain it.

**StreamingLLM** (Guangxuan Xiao et al., MIT, October 2023) refined this with an important discovery: without the first 4 tokens being globally attended, sliding-window quality collapses. These early tokens act as **attention sinks** — the model dumps excess attention probability into them. Drop the sinks, and the attention distribution becomes unstable. Keep the first 4 tokens as global attendees, and sliding window is stable across arbitrarily long streaming contexts.

The quality problem is structural. Both approaches fail on any task where the answer requires attending to a token more than $W$ positions ago. On needle-in-a-haystack benchmarks at context lengths $> W$, accuracy drops to random-chance levels. The mechanism *cannot* look beyond the window — there is no mechanism for it. This is not a quality degradation. It is a hard binary failure on long-range retrieval.

**Criteria failed: 4 (Faithful) on long-range tasks.**

## H₂O (2023) — Heavy Hitters Oracle

**Zhenyu Zhang et al. at UT Austin.** Score every token by its cumulative {{< wiki "attention" >}}attention{{< /wiki >}} weight across all decode steps so far. When the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} hits a budget, evict the lowest-scoring tokens.

The elegant insight: tokens that receive less cumulative attention are less important. Evict them. Keep the "heavy hitters" — tokens that get most of the attention.

This is data-dependent and adapts to the actual content being generated. Quality on many tasks is quite good. But notice: H₂O is **eviction**, not sparse attention. Once a token is evicted from the KV cache, it is gone. You cannot retrieve it on a later decode step when a different part of the generated text might need it. The policy is irreversible.

Additionally, H₂O evicts one token at a time based on a score. PagedAttention manages the KV cache in variable-length memory pages (blocks). Token-level eviction conflicts with block-level memory management — you cannot evict a single token without fragmenting a page. H₂O requires a non-standard cache manager that undermines the very PagedAttention infrastructure that makes production vLLM scale.

{{% callout type="definition" %}}
**The eviction vs. sparsity distinction:** Eviction means removing past KV entries from the cache entirely. The token is gone — it cannot be attended to in future steps. Sparse attention means computing a smaller set of attention weights within a single forward pass — the tokens are still in the cache, just not attended to this pass. H₂O and SnapKV are eviction methods. Longformer through DuoAttention are sparse attention methods. The distinction matters because sparse attention is fully reversible: a later decode step can attend to a different subset. Eviction is not.
{{% /callout %}}

**Criteria failed: 2 (Phase-agnostic — decode only), 3 (Optimization-friendly — PagedAttention conflict).**

## SnapKV (2024) — Cluster KV Pairs

**Yuhong Li et al.** Cluster the KV cache entries during prefill; retain only representative entries from each cluster. Better than H₂O at preserving long-range information, because the clustering is designed to keep one representative from each region of the context rather than just the tokens with the highest raw attention scores.

The clustering step is expensive and must run during prefill — before the model starts generating. This makes SnapKV incompatible with streaming prefill (processing long documents chunk by chunk) because you need the full prefill KV cache before you can cluster it. And like H₂O, once you've committed to the clustered representation after prefill, you're locked in for the entire decode. If a decode step later needs a token that was dropped in clustering, you cannot recover it.

**Criteria failed: 1 (Fast — clustering overhead in prefill), 2 (Phase-agnostic — prefill-only clustering step).**

## DuoAttention (2024) — Per-Head Heterogeneity

**Guangxuan Xiao et al. at MIT.** The cleanest pre-DSA approach. The core insight is empirical: when you measure which attention heads in a large model actually use long-range information (in the Longformer sense), you find that most heads don't. A minority of heads — call them **retrieval heads** — genuinely need full attention. The rest — **streaming heads** — perform equivalently with a sliding window.

DuoAttention finds the retrieval/streaming partition through offline importance scoring (run the model on representative tasks, measure which heads use long-range attention, label them). At inference, retrieval heads get full KV cache; streaming heads get only a window. Memory use drops. Latency drops. Quality, on average, is competitive.

The problem is the word "fixed." The partition is determined offline and baked into the model configuration. At inference, a head labeled as "streaming" uses streaming regardless of what the user is asking about. If the user is doing a long-document summarization task and a particular streaming head happens to be critical for tracking a specific entity across 50K tokens — too bad. The partition doesn't adapt.

More critically: the retrieval/streaming partition was determined by measuring the model *after* dense pretraining. Running DuoAttention requires either (a) dense pretraining followed by the partitioning procedure, or (b) guessing the partition before training and hoping it's right. Path (a) defeats the trainable-from-scratch criterion. Path (b) produces a worse model because the partition was guessed, not learned.

**Criteria failed: 5 (Trainable from scratch — the partition requires post-training).**

{{< crosshead >}}The Pattern of Failure{{< /crosshead >}}

Step back and look at the table. The failures cluster into two families:

**Family 1: Fixed patterns.** Longformer, BigBird, sliding window + sinks. These methods pass the kernel tests (hardware-friendly, fast) but fail on quality or faithfulness because the pattern is pre-specified, not learned. The model attends where the designer said it should attend, not where the content requires it to attend.

**Family 2: Learned patterns.** Reformer, Routing Transformer, H₂O. These methods are data-dependent and produce better quality — but the mechanism for making selection decisions is either non-differentiable (sort by bucket, argmin cluster assignment) or irreversible (eviction), which breaks kernel compatibility, training stability, or both.

{{% pullquote type="counter-intuitive" %}}
The two families fail in opposite directions. You can have a fast, kernel-friendly sparse pattern that doesn't learn anything useful. Or you can have a learned, quality-preserving selection that can't be implemented efficiently. You can't have both — until NSA.
{{% /pullquote %}}

The exact tension is: **differentiability versus discreteness**. Selecting which tokens to attend to is inherently discrete — you either attend to token 7,432 or you don't. Discrete decisions don't have gradients. Every method that tried to make selection differentiable (Routing Transformer's soft cluster assignment, FAVOR+'s continuous approximation, Linformer's projection) introduced approximation error that degraded quality. Every method that used hard selection (Reformer's bucket sort, H₂O's eviction) introduced non-differentiable operations that broke the training pipeline.

```pyplot {id="fixed-vs-learned-tradeoff" caption="THE FUNDAMENTAL TRADEOFF: FIXED PATTERNS ARE FAST AND DIFFERENTIABLE BUT DON'T ADAPT. LEARNED PATTERNS ADAPT BUT BREAK KERNELS. THE GOAL IS THE TOP-RIGHT CORNER."}
# Position each method in (kernel-friendliness, data-dependence) space
methods_2d = {
    'Longformer':     (0.9, 0.15, '#FF007F'),
    'BigBird':        (0.4, 0.25, '#FF007F'),
    'Sliding+Sinks':  (0.95, 0.05, '#FF007F'),
    'Reformer':       (0.2, 0.65, '#00A8A8'),
    'Routing-T':      (0.15, 0.75, '#00A8A8'),
    'H₂O':            (0.35, 0.70, '#00A8A8'),
    'Linformer':      (0.8, 0.30, '#FFD700'),
    'Performer':      (0.75, 0.35, '#FFD700'),
    'DuoAttn':        (0.85, 0.55, '#FF8C00'),
    'SnapKV':         (0.5, 0.60, '#FF8C00'),
    'NSA / DSA':      (0.95, 0.95, '#1A1A1A'),
}

fig, ax = plt.subplots(figsize=(8, 6))
for name, (x, y, c) in methods_2d.items():
    ax.scatter(x, y, s=120, color=c, edgecolors='#1A1A1A', linewidths=1.5, zorder=3)
    offset_y = 0.03 if name != 'NSA / DSA' else -0.05
    offset_x = 0.01
    ax.text(x + offset_x, y + offset_y, name, fontsize=8.5,
            ha='left', va='bottom', color='#1A1A1A')

# Shade the "success zone"
ax.fill_betweenx([0.85, 1.0], [0.85, 0.85], [1.0, 1.0],
                 color='#00A8A8', alpha=0.15, label='success zone')

ax.set_xlabel("Kernel-friendliness (FlashAttn compat + no scatter/gather)", fontsize=10)
ax.set_ylabel("Data-dependence (learns what to attend to)", fontsize=10)
ax.set_title("The sparse attention design space: 2020–2025", fontsize=11)
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)

# Legend for color families using proxy artists (no import needed)
legend_elements = [
    plt.Rectangle((0,0), 1, 1, facecolor='#FF007F', label='Fixed-pattern family'),
    plt.Rectangle((0,0), 1, 1, facecolor='#00A8A8', label='Learned (kernel-breaking) family'),
    plt.Rectangle((0,0), 1, 1, facecolor='#FFD700', label='Approximation family'),
    plt.Rectangle((0,0), 1, 1, facecolor='#FF8C00', label='Hybrid/eviction'),
    plt.Rectangle((0,0), 1, 1, facecolor='#1A1A1A', label='NSA/DSA (2025)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

{{< crosshead >}}What Was Missing{{< /crosshead >}}

The gap is now clear. Twelve methods, six years, zero frontier deployments. The missing piece is not clever mathematics — nearly every method above contains clever mathematics. The missing piece is a **differentiable, fine-grained, learned scorer** that satisfies all three of these simultaneously:

1. **Evaluates every past token** in $O(T \cdot d_{\text{score}})$ time where $d_{\text{score}} \ll d_{\text{head}}$.
2. **Produces a differentiable signal** for training — gradients must flow back through the scorer to update the scoring weights.
3. **Maps to efficient GPU kernels** — the selection must be block-aligned, contiguous in memory, and composable with FlashAttention.

No method before 2025 achieved all three. Reformer achieved approximate versions of (1) and (2) but failed (3). Routing Transformer achieved (2) approximately and struggled with (1) and (3). Fixed-pattern methods achieved (3) trivially but provided nothing for (1) or (2).

{{% callout type="tip" %}}
Note that the [attention compute primer](../11-attention-compute/) has the full FLOP accounting that makes the "fast enough" criterion precise. And the [top-k routing primer](../13-topk-and-routing/) explains the straight-through estimator that eventually solves the differentiability problem. Read those if you want the mathematical tools before seeing how NSA uses them.
{{% /callout %}}

**DeepSeek's NSA paper, arriving on February 16, 2025, proposes exactly this.** A three-branch architecture where Branch 2 — the selection branch — is a learned, block-level scorer with a straight-through estimator for gradients and block sizes aligned to GPU SM tiles. It is not conceptually exotic. In hindsight, it looks obvious. But the path through six years of failed attempts is what made it possible to see.

## What To Remember

1. **Sparse attention was tried.** Twelve credible methods between 2020 and 2024, from AllenAI, Google, Google Brain, Facebook AI, MIT, UT Austin.
2. **All of them failed at least one production criterion.** No frontier foundation model shipped on any of them by 2024.
3. **The pattern of failure was data-dependence vs. kernel friendliness.** Methods that picked patterns by hand (Longformer, sliding window) shipped but lost quality. Methods that learned (Reformer, Routing) had quality but no kernels.
4. **The eviction/sparsity distinction matters.** H₂O and SnapKV are eviction methods and belong to [Issue 6](/comicbook/06-kvcache-pruning/06-kvzap/). They fail on different criteria than sparse attention methods.
5. **The missing piece was a trainable, fast, fine-grained scorer** that produces block-aligned, kernel-compatible selections with differentiable gradients.

**Continue to** → [The NSA Blueprint](../05-nsa-paper/) — February 2025. DeepSeek's research preprint proposes the three-branch architecture that, eight months later, became the production lightning indexer.
