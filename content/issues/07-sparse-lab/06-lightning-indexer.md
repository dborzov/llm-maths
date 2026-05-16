---
title: "Lightning Strikes Twice"
description: "September 29, 2025. V3.2-Exp ships with DeepSeek Sparse Attention. The 'lightning indexer' is a tiny bilinear scorer that ranks every past token in low-rank space, and a top-k selector that picks the survivors. The 50% API price cut happens at midnight Beijing time."
topics: [attention, sparse-attention, deepseek, dsa]
tags: [dsa, lightning-indexer, deepseek-v3.2-exp, top-k, mla, integration]
theme: teal
math: true
draft: false
date: 2026-05-16T09:50:00-04:00
issue: 7
weight: 60
techKind: mainline
techNode: lightning-indexer
header: 06-lightning-indexer.webp
---

## The Drop, Re-Examined

It is 03:17 UTC on September 29, 2025. The Hugging Face repository `deepseek-ai/DeepSeek-V3.2-Exp` flickers into existence. Five paragraphs of model card. The third paragraph starts with a sentence that will cost OpenAI, Anthropic, and Google an uncomfortable Tuesday morning:

> *"We introduce DeepSeek Sparse Attention (DSA), a production sparse attention mechanism that integrates directly with Multi-head Latent Attention without increasing the per-token cache footprint."*

The vLLM patch is already merged. Not in a fork — upstream, in main, reviewed and committed. This is not an accident. The team had been coordinating with the vLLM maintainers for weeks. By noon UTC, every major inference team has the model running. By evening, engineers at other labs are not deploying the model; they are *dissecting the vLLM patch*, because what it contains is 13 lines of PyTorch and a 200-line attention forward-pass modification that represents six years of failed ideas suddenly, infuriatingly, working.

The price cut announcement comes at midnight Beijing time: **50% off the V3.2 API for all context lengths above 32K.** This is not a promotional discount. It is the result of a compute bill that genuinely halved at 128K context.

To understand why the patch is so short, you need to understand what it is actually doing.

{{< crosshead >}}What DSA Actually Is{{< /crosshead >}}

{{< wiki "attention" >}}Attention{{< /wiki >}} over a 128K-token context costs $O(T^2)$ FLOPs for prefill and $O(T)$ memory bandwidth for every decode step. The [quadratic wall](../03-quadratic-wall/) and the [attention compute primer](../11-attention-compute/) work through the exact numbers. The core problem is this: every new token queries against every past token, and at 128K tokens that is 128,000 comparisons — per head, per layer.

DSA asks: what if you only attend to the 2,048 tokens that *actually matter* for this query? Pick them fast with a cheap scorer. Then run full {{< wiki "mla" >}}MLA{{< /wiki >}} attention only over those 2,048. The scorer is the lightning indexer. The rest is standard flash attention on a short sequence.

The reason this works now, when Reformer, BigBird, and Longformer couldn't crack it in production, is MLA's {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}. Every prior sparse attention scheme needed to store *something extra* per token to support fast selection. MLA already stores something: $c_t$, the compressed latent of dimension $d_c = 512$. DSA derives its scorer keys from $c_t$. Zero new cache storage.

{{% pullquote type="counter-intuitive" %}}
The thing that blocked sparse attention in production for six years was not the algorithm. It was the absence of a cheap, already-cached key to score against. MLA provided that key without knowing it would.
{{% /pullquote %}}

## DSA in One Diagram

Here is the structure, in plain terms. The V3.2-Exp decode step for a single layer looks like:

1. **Compute the query latent** $c^Q_t = h_t W^{DQ}$ — same as MLA always did.
2. **Project to indexer query heads** via a small learned matrix $W^{IUQ}$.
3. **Score every past token** using its cached latent $c_s$ (already in cache) projected through $W^{IUK}$. Bilinear dot product, ReLU-gated.
4. **Hard top-$k$**: keep the 2,048 tokens with the highest scores.
5. **Full MLA attention** only over those 2,048 tokens' cached latents $c_s$ and RoPE keys $k^R_s$.

Steps 1–4 are the lightning indexer. Step 5 is what MLA always did — just over a shorter list.

This is also the reason V4's [Compressed Sparse Attention](../07-csa/) will be a natural extension: CSA plugs a token compressor *before* the indexer, reducing the number of entries that need to be scored. But that's the next chapter. V3.2-Exp's DSA runs the indexer over raw cached latents, no compression.

## The Lightning Indexer, Line By Line

Here is the function. This is faithful to the V3.2-Exp technical report, with types spelled out for clarity:

```python
def lightning_indexer(
    h_t: np.ndarray,          # [d_model] hidden state of current query token
    C_latent: np.ndarray,     # [T, d_c] cached MLA latents — already in cache!
    W_DQ: np.ndarray,         # [d_model, d_c] down-project to latent query
    W_IUQ: np.ndarray,        # [d_c, n_I_h * c_I] up-project to indexer queries
    W_IUK: np.ndarray,        # [d_c, n_I_h * c_I] indexer key projection
    head_weights: np.ndarray, # [n_I_h] learned per-head weights (softmax-normed)
    k_index: int = 2048,
    n_I_h: int = 32,
    c_I: int = 64,
) -> np.ndarray:              # returns [k_index] indices into cache

    T = C_latent.shape[0]

    # Step 1: compute query latent — identical to MLA's main path
    c_Q = h_t @ W_DQ            # [d_c]

    # Step 2: project to indexer query heads
    q_I = c_Q @ W_IUQ           # [n_I_h * c_I]
    q_I = q_I.reshape(n_I_h, c_I)   # [n_I_h, c_I]

    # Step 3: indexer keys from cached latents — no extra cache needed
    k_I = C_latent @ W_IUK      # [T, n_I_h * c_I]
    k_I = k_I.reshape(T, n_I_h, c_I)  # [T, n_I_h, c_I]

    # Step 4: bilinear score — ReLU-gated, weighted sum across indexer heads
    raw_scores = np.einsum('hd,thd->th', q_I, k_I)  # [T, n_I_h]
    raw_scores = np.maximum(0, raw_scores)            # ReLU: the ONLY nonlinearity
    index_scores = (raw_scores * head_weights).sum(-1)  # [T] — scalar per token

    # Step 5: hard top-k
    top_indices = np.argsort(index_scores)[-k_index:]   # [k_index]
    return top_indices
```

{{< crosshead >}}Five Design Choices Worth Understanding{{< /crosshead >}}

**Why ReLU and not softmax for the scores?** This is the most important design choice, and it is counterintuitive. With {{< wiki "softmax" >}}softmax{{< /wiki >}}, every query is *forced* to find something relevant — the probabilities must sum to 1, so even if nothing in the 128K context is useful for this query, the softmax will distribute attention mass anyway. ReLU lets a token score zero. A query can genuinely abstain from retrieving anything from most of the cache. In long documents, most positions really don't need content from most other positions. ReLU models that reality.

{{% callout type="tip" %}}
The ReLU score is why the training recipe needs a soft top-k warm-up phase. During early training, most scores are near zero (ReLU kills negative values) and the top-k selection gradient is sparse. Temperature annealing from 5 → 0.1 lets the indexer first learn *which tokens are nonzero*, then learn *fine-grained ranking* among them.
{{% /callout %}}

**Why bilinear (separate $W^{IUQ}$, $W^{IUK}$) and not a shared projection?** A shared projection would mean that "what a query looks for" and "what a key offers" live in the same semantic space. The bilinear formulation lets the model learn asymmetric relationships. Query head 7 might look for "has a verb in the recent past" while the same dimension in key space means "is a declarative sentence." This asymmetry is exactly what MLA's separate up-projections $W^{UQ}$ and $W^{UK}$ exploit — the indexer uses the same trick at smaller scale.

**Why $n_{I,h} = 32$ heads and not 128?** The main MLA has $H = 128$ heads. The indexer uses 32. This is not a coincidence — it is 4× cheaper. The indexer's job is ranking, not detailed content retrieval. You need enough heads to capture diverse retrieval signals (syntactic, semantic, positional), but you do not need 128 of them. 32 is sufficient and the ablations in the tech report confirm the plateau.

**Why $c_I = 64$ dimensions per head?** The latent dimension $d_c = 512$. Each indexer head gets $64$ dims. $32 \times 64 = 2048$ total indexer dimensions. This is the $d_I$ in the compute plot below — the dimension that determines the O(T) cost of the scoring pass.

**Why can $W^{IUK}$ be absorbed offline?** The indexer key for past token $s$ is $k_I[s] = c_s W^{IUK}$. Since $c_s$ is already in the cache and $W^{IUK}$ is a fixed model weight, this product can be precomputed once and stored. MLA uses the same absorption trick for the main content keys: $k^C_t = c_t W^{UK}$, so the KV cache stores $c_t$ and $W^{UK}$ is multiplied into the query side at decode time. For DSA in practice, V3.2-Exp computes $k_I$ from $c_t$ on the fly (fast, because $d_c = 512 \ll T = 128000$), but the absorption option exists and is used in optimized inference kernels.

```pyplot {id="lightning-indexer-cost" caption="THE LIGHTNING INDEXER COSTS O(T × d_I) WHERE d_I IS THE INDEXER DIMENSION (~128). FULL ATTENTION COSTS O(T × d_h × H). FOR d_I=128, d_h=128, H=128: THE INDEXER IS ~128× CHEAPER THAN THE FULL SCORE PATH."}
T = np.logspace(np.log10(1024), np.log10(1_048_576), 200)
H = 128
d_h = 128
d_I = 128
# Full attention scores: H heads × T queries × T keys × d_h dim
# But for decoding (one new query): H × T × d_h per step
full_decode = H * T * d_h
indexer_decode = d_I * T  # one indexer "head" of dim d_I
# Top-k selection then attends to only k tokens
k = 2048
sparse_attend = H * k * d_h * np.ones_like(T)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.loglog(T / 1000, full_decode / 1e9, color='#FF007F', linewidth=2.5, label='dense attention (full T)')
ax.loglog(T / 1000, indexer_decode / 1e9, color='#FFD700', linewidth=2.5, label='lightning indexer score')
ax.loglog(T / 1000, sparse_attend / 1e9, color='#00A8A8', linewidth=2.5, label='attend over top-k=2048')
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('GFLOPs per decode step')
ax.set_title('lightning indexer + sparse attend << full dense for T > 16K', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, which='both')
ax.spines[['top','right']].set_visible(False)
```

Read that plot carefully. The dense attention line (pink) climbs linearly in T — because decode queries one new token against all T past tokens. The indexer (yellow) also climbs linearly in T but at a much lower slope: 128 dimensions instead of $H \times D = 128 \times 128 = 16384$ dimensions. The top-k attend (teal) is flat — 2048 tokens regardless of context length. At T = 128K the indexer is ~128× cheaper than full scoring, and the attend pass over k=2048 is ~62× cheaper than full dense attention. Combined: the decode attention cost drops by roughly 60× in FLOPs at 128K context.

## The MLA Integration

The [MLA rewind](../02-mla-rewind/) article and [Issue 5 ch.19 MLA](/issues/05-microgpt-unfolded/19-mla/) cover MLA's cache layout in detail. The critical fact for DSA is:

**MLA stores, per token per layer:** $c_t \in \mathbb{R}^{d_c}$ (the compressed latent, $d_c = 512$) plus $k^R_t \in \mathbb{R}^{d_R}$ (the {{< wiki "rope" >}}RoPE{{< /wiki >}} channel, $d_R = 64$). That's $512 + 64 = 576$ numbers per token per layer, versus the naive approach of storing $H \times D = 128 \times 128 = 16384$ numbers. This is MLA's cache compression trick.

DSA uses $c_t$ for the indexer keys. Nothing additional. The indexer projection $W^{IUK}$ maps $c_t \in \mathbb{R}^{d_c}$ to $k_I \in \mathbb{R}^{n_{I,h} \times c_I}$, but $c_t$ is the same thing that was already in the cache for the main MLA decode. Zero extra bytes per token.

This is why the vLLM patch is 200 lines, not 2,000. The storage layout doesn't change. The existing CUDA kernels for reading the cache don't change. The FlashAttention call for the actual attend step doesn't change. What changes is a preprocessing step before the attention call: score the cache, produce indices, slice.

{{% callout type="definition" %}}
**The absorption trick revisited.** MLA absorbs $W^{UK}$ into the query side, avoiding materializing explicit key vectors in the cache. DSA can do the same for $W^{IUK}$: instead of storing $k_I = c_t W^{IUK}$, you store only $c_t$ and perform the projection at inference time. Both tricks exploit the same algebraic identity: $q^\top k = q^\top (c W^{UK}) = (q W^{UK\top})^\top c$. The expensive projection is moved to the side where it can be amortized or fused.
{{% /callout %}}

The RoPE component of the cache ($k^R_t$) is used by the main MLA attend pass after the indexer selects the top-k tokens. The indexer itself does not use RoPE keys — it operates entirely in the latent (NOPE) channel. This separation is deliberate: positional information is not what you use for semantic selection. You want to know "does this token's *content* match what I'm looking for?", not "is this token close to me positionally?".

## How Training Worked

DeepSeek did not train V3.2-Exp from scratch. The training recipe is a classic continue-training pattern with a careful annealing schedule:

**Phase 0: Starting point.** The V3.1 checkpoint. Fully converged on 128K context, dense MLA, ~10T tokens of training. The indexer heads ($W^{IUQ}$, $W^{IUK}$, head weights) are initialized randomly.

**Phase 1 — Soft top-$k$ warm-up (~50B tokens).** The hard `argsort(-k)` selection is replaced by a temperature-weighted {{< wiki "softmax" >}}softmax{{< /wiki >}} over scores. At temperature $\tau = 5$, nearly all tokens get nonzero attention weight, and the gradient flows through the indexer from the full LM loss. Temperature anneals from 5 → 0.1 over ~50B tokens. At $\tau = 0.1$ the softmax is nearly hard: only the top few tokens have weight above 0.01. The indexer has learned to concentrate on the right tokens without ever being forced to commit to a hard set.

**Phase 2 — Hard top-$k$ with STE (~150B tokens).** The `argsort(-k)` becomes genuinely hard. Gradients back through the selection use the Straight-Through Estimator (the [top-k routing primer](../13-topk-and-routing/) covers STE in detail). The LM loss now drives the indexer to select the exact right tokens.

**Auxiliary loss throughout.** A small auxiliary term (weight 0.01) directly penalizes the indexer when it fails to select a token that dense MLA would have attended to with high weight. This accelerates convergence in Phase 1 and prevents quality regression in Phase 2.

**Total continue-training:** ~200B tokens. For context, the original V3 pretraining was ~10T tokens. This is **2% of V3's compute**, or roughly $1M of H800 GPU-hours, versus V3's ~$5.6M.

{{% pullquote type="standard" %}}
200 billion tokens to retrofit sparse attention onto a fully trained model. That is 2% of the cost of building the model in the first place. The algebra of making DSA work in MLA makes the retrofit cheap.
{{% /pullquote %}}

## The Top-k Choice

V3.2-Exp picks $k = 2048$ for a 128K context. That is 1.56% of the cache. Why 2048?

The tech report's ablation sweeps $k$ from 128 to 16384 on RULER (the long-context quality benchmark). Here is what the quality curve looks like:

| k | RULER score drop vs dense |
|---|---|
| 128 | −8% |
| 256 | −4% |
| 512 | −2% |
| 1024 | −0.5% (the knee) |
| 2048 | −0% (plateau) |
| 4096 | −0% (identical to 2048) |

The knee is at k = 1024. Above 1024, additional tokens contribute essentially nothing to output quality. V3.2-Exp picks 2048 as a **2× safety margin above the knee**. This handles:

1. **Distribution shift.** Deployment queries may be harder than the ablation benchmark. The safety margin absorbs this.
2. **Kernel alignment.** FlashAttention kernels prefer $k$ to be a multiple of 128 (block tile size). 2048 = 16 × 128 — perfectly aligned.
3. **Batch diversity.** In a batched inference server, some queries need more top-k than others. A single $k = 2048$ serves the demanding queries without running two $k$ values per batch.

```pyplot {id="topk-sweep" caption="QUALITY ON RULER VS K (TOKENS ATTENDED PER QUERY) FOR DSA. THE KNEE IS AROUND k=1024. V3.2-EXP PICKS 2048 AS A SAFETY MARGIN."}
# Stylized sweep: x = k (log scale), y = quality (e.g., RULER score). Show:
# - Steep climb from k=128 to k=1024
# - Plateau from k=1024 to k=8192
# Plus a dashed line at the dense baseline.
k = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
# Stylized quality curve
quality = 0.92 / (1 + np.exp(-(np.log2(k) - 9) * 2)) + 0.05

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.semilogx(k, quality, color='#FF007F', linewidth=2.5, marker='o', markersize=7)
ax.axhline(0.94, color='#1A1A1A', linewidth=1, linestyle='--', label='dense MLA baseline')
ax.axvline(2048, color='#00A8A8', linewidth=1.5, linestyle=':', label='V3.2-Exp choice (k=2048)')
ax.set_xlabel('k (tokens attended per query)')
ax.set_ylabel('stylized RULER score')
ax.set_title('DSA quality vs top-k: knee at ~1024, plateau above')
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)
```

That flat plateau above k = 1024 is the empirical signature of a fact we have known theoretically since Bahdanau attention: most of the attention weight in a real language model concentrates on a small number of tokens. The question was always whether a learned scorer could *predict* which tokens in advance, before running full attention. The answer in V3.2-Exp is yes.

{{< crosshead >}}What the Inference Engineers Found{{< /crosshead >}}

On the morning of September 29, the teams pulling apart the vLLM patch made a list. Here is what they found, item by item:

- The indexer is **13 lines of PyTorch**.
- It uses **the existing MLA latent cache** with zero new storage axes.
- The kernel is **FlashAttention-compatible**: the selected indices are block-aligned, so the attend step runs in the same CUDA kernel with a mask.
- The vLLM integration required a **modified attention forward pass, ~200 lines** — mostly bookkeeping around the index selection and kernel dispatch.
- **Total latency overhead of the indexer:** <2ms per decode step at 128K context, benchmarked in the tech report. The scoring pass over 128K latents at 32 heads × 64 dims is fast because it is a matrix-vector multiply: $C_\text{latent} \in \mathbb{R}^{128000 \times 512}$, $W^{IUK} \in \mathbb{R}^{512 \times 2048}$, result is $\mathbb{R}^{128000 \times 2048}$ — all in FP8. On an H800 that is a batched GEMM of 128K × 512 × 2048, comfortably under 2ms.

{{% callout type="counterintuitive" %}}
The indexer is slower at very short contexts (say T < 8K) than dense attention, because the overhead of the scoring pass exceeds what it saves in the attend pass. At T = 8K, full dense attention is already fast enough that DSA's bookkeeping overhead dominates. V3.2-Exp uses a context-length threshold: DSA only activates for T > 32K.
{{% /callout %}}

## The Price Cut

The 50% API price cut is not marketing. Here is the napkin math.

**V3.1 at T = 128K, one decode step:**

- Attention FLOPs (decode): $4 \times H \times L \times T \times D = 4 \times 128 \times 61 \times 128000 \times 128 \approx 4.0 \times 10^{12}$ FLOPs.
- But decode is memory-bandwidth-limited, not compute-limited. KV cache size for one sequence: $d_c + d_R \text{ per token per layer} = (512 + 64) \times 61 \times 128000 \approx 4.4 \text{ GB}$.
- HBM bandwidth (H800): ~3.35 TB/s. Time to read the cache: $4.4 \text{ GB} / 3.35 \text{ TB/s} \approx 1.3 \text{ ms per decode step}$.
- At a typical batch size, attention is roughly **50% of total inference compute** at 128K context (the MLP layers account for the other 50% and are context-independent).

**V3.2-Exp at T = 128K, one decode step with DSA:**

- Indexer scoring: $32 \times 64 \times 128000 = 2.6 \times 10^8$ FLOPs — negligible.
- Top-k attend over k=2048: reads $576 \times 2048 = 1.2 \text{ MB}$ of KV cache instead of 4.4 GB. Cache read time: $1.2 \text{ MB} / 3.35 \text{ TB/s} < 1 \mu\text{s}$.
- Total memory traffic for attention: **1/3500 of the V3.1 attention memory traffic**.

The attention memory wall disappears. The remaining inference cost is the MLP forward pass, which doesn't change. Since attention was ~50% of total cost at 128K, eliminating the attention memory wall cuts total inference cost by roughly 50%. The price cut is the price cut.

{{% callout type="warning" %}}
This math is at T=128K, the maximum context. At T=32K (a more typical enterprise query), DSA is not activated (threshold behavior) and the savings are smaller. The 50% headline is accurate for maximum-context queries; typical savings at shorter contexts are 20–35%. This matters when comparing inference providers who report averages across all context lengths.
{{% /callout %}}

## What V3.2-Exp Did Not Solve

DSA cut the *computation* cost of attending over a long context. It did not cut the *storage* cost of the context.

At T = 128K, the MLA latent cache for V3.2-Exp is still:
$$576 \text{ numbers} \times 61 \text{ layers} \times 128000 \text{ tokens} \times 2 \text{ bytes} \approx 8.9 \text{ GB per sequence.}$$

For a provider running thousands of concurrent sequences with 128K context, that is 8.9 TB of GPU RAM for KV caches alone. DSA does not compress that number — it just selects which 2,048 of those 128,000 entries to *read* per decode step. The entries are all still there.

To go to 1M context, you cannot simply extend DSA. The cache scales linearly with T. At 1M tokens you would need 70 GB of HBM per sequence — which does not fit on an H800 node per sequence. The next move is to compress the cached sequence itself, turning T entries into T/m entries. That is [Compressed Sparse Attention](../07-csa/) and [Heavily Compressed Attention](../08-hca/).

## What To Remember

1. **The lightning indexer is a tiny bilinear scorer.** Low-rank QK, ReLU-gated, 32 heads of 64 dims each. One score per past token per query.
2. **Top-k = 2048 at 128K context.** Roughly 1.5% of the cache. The knee is at k=1024; 2048 is the safety margin.
3. **It bolts onto MLA.** The indexer keys are derived from the same latent $c_t$ already in the cache. Zero new storage bytes per token.
4. **Continue-training, not from scratch.** ~200B tokens, ~2% of V3's original training compute, ~$1M.
5. **The price cut is real.** At 128K context, attention was ~50% of inference cost. DSA eliminates the attention memory bottleneck. Half the bill disappears.
6. **The cache itself still scales with T.** DSA fixes the *compute wall*. The *storage wall* requires token compression — the subject of the next two chapters.

**Continue to** → [Compressed Sparse Attention](../07-csa/) — V4's first new attention mode. Compress every 4 tokens into one entry *before* DSA runs. The indexer queries inherit MLA's absorption trick. The official paper figure included.
