---
title: "KV Pruning Landscape: 20+ methods, zero shipped before 2026"
short_title: "KV Pruning Landscape"
description: "H₂O, StreamingLLM, SnapKV, DuoAttention, KVzip, and 15+ more KV pruning methods were all declined by vLLM, SGLang, and TensorRT-LLM operators — a four-criteria filter (fast, phase-agnostic, optimization-friendly, faithful) explains every rejection."
blurb:
  - "The NVIDIA/kvpress leaderboard lists 20+ methods. The Awesome-KV-Cache-Compression list has dozens more. Merged into production: zero."
  - "H₂O fails optimization-friendly (requires a custom kernel) and phase-agnostic (decode-only running sum)."
  - "KVzip fails fast (2× prefill overhead) and phase-agnostic (second forward pass impossible during decode)."
  - "KVzap, January 2026, is the first method to pass all four — here is why every predecessor fell short of at least one."
topics: [kv-cache, pruning, survey]
tags: [h2o, streamingllm, snapkv, duoattention, kvzip, kvzap, landscape]
theme: teal
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 100
techKind: primer
techNode: pruning-landscape
header: 10-pruning-landscape.webp
---

## Late 2025, GitHub

The `NVIDIA/kvpress` repository has a leaderboard. It lists more than twenty {{< wiki "kv-pruning" >}}KV cache pruning{{< /wiki >}} methods — H₂O, StreamingLLM, SnapKV, DuoAttention, Compactor, AdaKV, KVzip, Expected Attention, and a dozen more. Each one comes with an arXiv paper, ablation tables, and a claimed compression ratio. The `Awesome-KV-Cache-Compression` GitHub list maintained by the community has dozens more. The academic output is impressive.

And yet: zero of them are integrated in vLLM, zero in SGLang, zero in TensorRT-LLM — the three inference engines that actually serve LLMs in production. The operators who run real multi-tenant inference clusters have evaluated these methods repeatedly and declined to merge any of them.

Why?

The answer is not "because the research is bad." The research is, in many cases, excellent. The answer is that *production inference has requirements that academic benchmarks don't measure* — and every method before 2026 fails at least one of them.

{{< crosshead >}}The Four-Criteria Filter{{< /crosshead >}}

KVzap (Jégou & Jebleck, 2026) introduces a four-criteria checklist as a retrospective diagnostic. It is worth stating upfront because it is the organizing logic of the entire family tree:

| Criterion | What it means |
|-----------|---------------|
| **Fast** | Overhead at inference time < ~2% of total compute. Not "fast on a single A100 with one prompt." Fast on a saturated multi-tenant cluster. |
| **Phase-agnostic** | Works during both prefill (parallel processing of all input tokens) and decode (autoregressive generation of each output token). Methods that only work during prefill are incomplete. |
| **Optimization-friendly** | No custom CUDA kernels required. Can be integrated into existing attention implementations (FlashAttention 2/3, PagedAttention) without patching the kernel. |
| **Faithful** | Actual accuracy preservation at the claimed compression ratio, measured on standard benchmarks — not cherry-picked tasks where the method happens to work well. |

A method that fails *fast* is too slow to deploy. A method that fails *phase-agnostic* cannot be used for streaming generation. A method that fails *optimization-friendly* requires every inference engine to maintain a custom fork forever. A method that fails *faithful* doesn't actually work.

KVzap, published January 2026, is the first method to pass all four. Let us walk through every major milestone to understand why it took three years.

{{< crosshead >}}H₂O (2023) — The Pioneer{{< /crosshead >}}

**Zhang et al., UT Austin, NeurIPS 2023.** The paper that started the conversation.

H₂O — "Heavy Hitter Oracle" — observes that {{< wiki "attention" >}}attention{{< /wiki >}} weight is sparse: across a long sequence, 80% of the {{< wiki "softmax" >}}softmax{{< /wiki >}} mass goes to roughly 20% of the tokens. It calls these tokens "heavy hitters." The key idea: track a running sum of accumulated attention weights across all past queries, and evict the tokens with the lowest running totals when the cache exceeds a budget.

The running sum approach is elegant — it requires no separate model, no precomputed scores, no knowledge of future queries. The score for token $i$ is:

$$
\text{score}_\text{H2O}(i) = \sum_{j=1}^{T} a_{ji}
$$

where $a_{ji}$ is the attention weight token $j$ places on token $i$. High score = got attended to a lot = is probably important. Evict the stragglers.

**What H₂O passes:** Faithful — it genuinely preserves quality on many benchmarks at 20–40% cache retention.

**What H₂O fails:**
- *Optimization-friendly* — requires a modified attention kernel that tracks running sums and performs eviction during attention computation. Cannot be dropped into FlashAttention without rewriting the kernel.
- *Phase-agnostic* — the running-sum score is a decode-only mechanism; during prefill there are no "past queries" to accumulate over.
- *Fast* — the modified kernel adds measurable overhead, and budget management during decode adds memory traffic.

H₂O is the most-cited paper in this family tree. It is also the method most frequently described as "works in a demo, impossible to ship."

{{< crosshead >}}StreamingLLM (2023) — The Speedrunner{{< /crosshead >}}

**Xiao et al., MIT, ICLR 2024.** StreamingLLM is not really a pruning method — it is a windowing method. But it is the most widely deployed "KV cache compression" technique in practice (it ships in llama.cpp's `--cache-type` option), so it belongs in the family tree.

The observation: even with a short fixed window, language model perplexity stays low *if you always keep the first few tokens*. These early tokens — often the BOS token and the system prompt — accumulate massive attention weight because every subsequent token attends to them. They are called **attention sinks**.

StreamingLLM keeps the last $w$ tokens in context plus a small fixed-size sink budget (typically 4 tokens). This fits trivially in existing attention implementations with no kernel changes. The overhead is zero.

**What StreamingLLM passes:** Fast, phase-agnostic, optimization-friendly.

**What StreamingLLM fails:** Faithful — *it is not actually pruning*. It discards all long-range context beyond the window. For retrieval tasks where the answer appears at token position 1000 and you have a window of 512, StreamingLLM has no answer. It is excellent for infinite-context streaming (chatbots, long generation) but degrades gracefully to zero on needle-in-haystack and document QA tasks.

StreamingLLM is the method inference engines *have* shipped — but it solves a different problem than KV pruning. It trades faithfulness for speed, explicitly and correctly.

{{% callout type="tangent" title="The Sliding Window Connection" %}}
The sliding window with attention sinks in StreamingLLM is the same mechanism KVzap uses as a baseline: always keep the last $w = 128$ tokens regardless of their scores. The KVzap paper shows that setting $w = 0$ (no sliding window) drops accuracy to 28.37% on {{< wiki "long-context-benchmarks" >}}RULER{{< /wiki >}} — the sinks matter enormously. The innovation is combining the window with a faithful, adaptive eviction policy for the tokens outside the window.
{{% /callout %}}

{{< crosshead >}}SnapKV (2024) — The Clusterer{{< /crosshead >}}

**Li et al., 2024.** SnapKV observes that attention patterns cluster: different query tokens tend to attend to the same small sets of key positions. Rather than keeping individual KVs, it clusters the key vectors and keeps one representative KV pair per cluster.

The intuition is good. SnapKV achieves impressive compression on many long-document tasks and is genuinely faithful on summarization and question answering.

**What SnapKV passes:** Faithful on prefill-dominated tasks.

**What SnapKV fails:**
- *Phase-agnostic* — clustering happens over the full key matrix after prefill. During decode, new query positions emerge that may need different representatives. There is no mechanism to re-cluster or evict during generation.
- *Optimization-friendly* — clustering adds O(T log T) overhead at prefill and requires a non-standard attention path.

SnapKV is a legitimate method for a specific use case: read a long document, ask a question, get an answer. It fails on multi-turn conversations and long generation tasks.

{{< crosshead >}}DuoAttention (2024) — The Static Classifier{{< /crosshead >}}

**Xiao et al., MIT, 2024.** DuoAttention is the most architecturally elegant method in this family tree. It divides an LLM's attention heads into two permanent categories:

- **Retrieval heads**: heads that consistently attend to specific past tokens (measured across a calibration dataset). These keep their full KV cache.
- **Streaming heads**: heads that primarily attend to recent tokens and attention sinks. These use a sliding window.

The classification happens offline, once per model, and never changes at inference time. Retrieval heads pay full KV cost; streaming heads are essentially free.

**What DuoAttention passes:** Optimization-friendly (the head classification is baked in; no runtime changes), fast (zero inference-time classification overhead).

**What DuoAttention fails:**
- *Phase-agnostic* — the streaming head classification assumes decoding behavior. During prefill, both head types process all tokens in parallel; the savings only appear during decode.
- *Faithful on all tasks* — a head classified as "streaming" for document QA may behave as a retrieval head on a reasoning task. The static classification cannot adapt. On benchmarks like AIME25 where mathematical reasoning requires different attention patterns than text retrieval, DuoAttention streaming heads miss critical dependencies.

DuoAttention represents the state of the art in *static* approaches. The fundamental limit is that "retrieval head" is not a property of a head in isolation — it is a property of a head *on a particular input type*.

{{< crosshead >}}Expected Attention (2025) — The Theorist{{< /crosshead >}}

**Devoto, Jeblick, Jégou, 2025.** Expected Attention is the theoretical predecessor to KVzip+. It provides rigorous grounding for *why* accumulating attention weights is the right metric — and, critically, why raw accumulated attention is insufficient.

The key insight: attention weight $a_{ji}$ tells you how much query $j$ cares about key $i$, but not how much key $i$ actually *changes* the output. The output change is proportional to both the attention weight and the value vector's magnitude projected through the output matrix:

$$
s_i^{\text{EA}} = \sum_j a_{ji} \cdot \|W_O v_i\|
$$

This is strictly better than H₂O's score: it accounts for the contribution of the value projection, not just the attention weight.

**What Expected Attention passes:** Faithful (significantly better than H₂O on dense tasks), theoretically grounded.

**What Expected Attention fails:**
- *Fast during decode* — computing $\|W_O v_i\|$ requires a matrix multiplication per token per query step during generation. At long contexts, this overhead is proportional to generation length.
- The normalization by $\|h_j\|$ that makes scores comparable across queries (the KVzip+ insight) is not yet present.

Expected Attention is where the critical ideas crystallize, but it cannot be deployed without the overhead fix.

{{< crosshead >}}KVzip (2025) — The Oracle{{< /crosshead >}}

**Kim et al., 2025.** KVzip takes a different approach entirely. Rather than scoring tokens using attention weights during inference, it scores them using a *pretext task* during prefill: it runs a masked prediction task (similar to BERT's MLM objective) where each token must predict itself from context, and the prediction difficulty proxies for the token's importance to the downstream task.

The pretext-task score is extraordinarily faithful. Tokens that are hard to predict from context tend to carry unique information; tokens that are easy to predict are redundant. The score correlates strongly with contribution magnitude.

**What KVzip passes:** Faithful — best-in-class on most benchmarks among pre-KVzap methods. The scores are almost optimal.

**What KVzip fails:**
- *Fast* — running the pretext task doubles prefill cost. For a 128K-token document, this means 2× longer time-to-first-token. Unacceptable in production.
- *Phase-agnostic* — the pretext scoring happens entirely during prefill. No mechanism for decode-time scoring of newly generated tokens.

KVzip is the method that definitively proves the scores *can* be computed faithfully. The challenge KVzap inherits is: can we get KVzip's score quality at a fraction of KVzip's cost?

{{< crosshead >}}The Criteria Matrix{{< /crosshead >}}

| Method | Year | Fast | Phase-agnostic | Optimization-friendly | Faithful |
|--------|------|------|----------------|----------------------|----------|
| H₂O | 2023 | ✗ | ✗ | ✗ | ✓ |
| StreamingLLM | 2023 | ✓ | ✓ | ✓ | ✗ |
| SnapKV | 2024 | ✗ | ✗ | ✗ | ✓ |
| DuoAttention | 2024 | ✓ | ✗ | ✓ | ✗ |
| Expected Attention | 2025 | ✗ | ✗ | ✓ | ✓ |
| KVzip | 2025 | ✗ | ✗ | ✓ | ✓ |
| **KVzap** | **2026** | **✓** | **✓** | **✓** | **✓** |

No method before KVzap passes all four. KVzap is the first.

{{< crosshead >}}KVzap (2026) — The Surrogate{{< /crosshead >}}

**Jégou & Jebleck, NVIDIA, January 2026.** KVzap's insight is to separate the *problem of scoring* from the *mechanism of scoring*.

H₂O and Expected Attention score tokens using *attention computation itself* — which means the scoring is entangled with the forward pass and cannot be made cheap. KVzip scores tokens using a separate pretext task — which means the scoring is correct but expensive.

KVzap scores tokens using a *surrogate model* trained on the {{< wiki "residual-stream" >}}hidden states{{< /wiki >}} $h_t$ — the residual stream after each attention layer. The key empirical observation: $h_t$ carries enough information to predict whether token $i$ will have a high KVzip+ score ($R^2 = 0.67$–$0.77$ across benchmarks). The surrogate is a tiny two-layer MLP:

$$
\hat{s}_i^+ = \text{MLP}_\text{surrogate}(h_i)
$$

This is fast — one cheap matrix multiplication per token, computed from activations that the forward pass already produces. It adds less than 1.1% overhead to total inference compute. And because hidden states are produced at every layer for every token during both prefill and decode, the surrogate can score newly generated tokens during generation — making it genuinely phase-agnostic.

The sliding window (always keep the last 128 tokens) provides a safety net for very recent tokens whose hidden states haven't stabilized yet. Together with threshold τ applied to the surrogate's log-score predictions, this gives KVzap its full eviction policy.

```pyplot {id="accuracy-vs-compression" caption="ACCURACY VS COMPRESSION RATIO — SYNTHETIC CURVES ILLUSTRATING QUALITATIVE BEHAVIOR OF EACH METHOD FAMILY"}
np.random.seed(0)

# Synthetic curves: accuracy (0-1) vs fraction of KV cache retained (0-1)
# These illustrate qualitative behavior, not exact paper numbers
ratios = np.linspace(0.05, 1.0, 100)

def sigmoid_curve(ratios, mid, steepness, peak=1.0):
    return peak / (1 + np.exp(-steepness * (ratios - mid)))

# StreamingLLM: good at low retention (recent tokens ok), but flat ceiling
streamingllm = np.minimum(sigmoid_curve(ratios, 0.15, 15), 0.72)

# H2O: moderate, degrades at low retention
h2o = sigmoid_curve(ratios, 0.35, 8, peak=0.92)

# SnapKV: better than H2O, but prefill only (shown as prefill performance)
snapkv = sigmoid_curve(ratios, 0.25, 10, peak=0.95)

# KVzip: near-optimal faithfulness, high compute cost (shown as achievable quality)
kvzip = sigmoid_curve(ratios, 0.15, 12, peak=0.98)

# KVzap: near-KVzip faithfulness at low compute cost
kvzap = sigmoid_curve(ratios, 0.18, 11, peak=0.97)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(ratios, streamingllm, color='#FF8C00', linewidth=1.8, linestyle='--', label='StreamingLLM (2023)')
ax.plot(ratios, h2o,         color='#FFD700', linewidth=1.8, label='H₂O (2023)')
ax.plot(ratios, snapkv,      color='#00A8A8', linewidth=1.8, label='SnapKV (2024)')
ax.plot(ratios, kvzip,       color='#FF007F', linewidth=1.8, linestyle=':', label='KVzip (2025)')
ax.plot(ratios, kvzap,       color='#1A1A1A', linewidth=2.5, label='KVzap (2026)')

ax.set_xlabel("fraction of KV cache retained")
ax.set_ylabel("relative accuracy (normalized)")
ax.set_title("Accuracy vs. compression — qualitative family tree (synthetic curves)")
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.spines[['top', 'right']].set_visible(False)

# Add annotation
ax.annotate('KVzap: KVzip-quality\nscores at <1.1% overhead',
            xy=(0.3, kvzap[int(0.3*99)]),
            xytext=(0.15, 0.55),
            fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1))

print("Qualitative accuracy at 30% KV retention (index 25):")
for name, curve in [('StreamingLLM', streamingllm), ('H2O', h2o),
                     ('SnapKV', snapkv), ('KVzip', kvzip), ('KVzap', kvzap)]:
    idx = np.searchsorted(ratios, 0.30)
    print(f"  {name}: {curve[idx]:.3f}")
```

The KVzap curve tracks near KVzip's ceiling (the theoretical best achievable) across the full range of compression ratios, while operating with inference-compatible overhead.

{{< crosshead >}}What the KVpress Leaderboard Says{{< /crosshead >}}

The NVIDIA/kvpress repository maintains a living benchmark on the RULER 4k task for {{< wiki "qwen3" >}}Qwen3-8B{{< /wiki >}} and Llama-3.1-8B. As of early 2026, KVzap leads at matched compression ratios — specifically outperforming AdaKV, SnapKV, and Expected Attention at compression ratios above 50%.

The leaderboard is significant not just for the rankings but for what it signals: this is the first time a production-viable method (KVzap, with its < 1.1% overhead and phase-agnostic operation) leads a fair comparison. Previous leaderboard leaders were research artifacts — impressive on the eval, impractical to ship.

{{< crosshead >}}The Road Not Taken (Yet){{< /crosshead >}}

The family tree has branches that don't connect to KVzap but point toward future directions.

**AdaKV (Feng et al., 2025)** — per-head budget allocation within a fixed total. Still budget-constrained, but a meaningful improvement over flat top-k. Compatible with threshold-based ideas in principle.

**Compactor (Chari & Durme, 2025)** — approximate leverage scores for pruning. Stronger theoretical grounding than H₂O, slower than KVzap. A candidate for offline/batch settings where latency is not the constraint.

**End-to-end objectives** — methods like DMS (Łańcucki et al., 2025) that learn pruning policies jointly with model weights during training, rather than applying post-hoc heuristics to a frozen model. These sidestep the four-criteria problem by baking the pruning into the model itself. If a future model is trained with KV compression as a first-class objective, post-hoc methods like KVzap become unnecessary.

**Architectural integration** — sparse attention mechanisms (see {{< wiki "deepseek" >}}DeepSeek V3{{< /wiki >}} MLA, Mistral sliding-window variants) that reduce KV cache *structurally* rather than through post-hoc eviction. These change the denominator rather than shrinking the numerator.

The long-term question is whether post-hoc pruning methods survive the shift to architectures designed from scratch for memory efficiency. KVzap may be the apex of the post-hoc era — the method that finally solved the four-criteria problem just as the field is beginning to rethink whether post-hoc is the right frame at all.

## What To Remember

1. **Twenty methods in the literature, none in production (before 2026).** The four-criteria filter — fast, phase-agnostic, optimization-friendly, faithful — explains every rejection. Each method in the family tree fails at least one criterion.
2. **H₂O set the frame; StreamingLLM found the attention sinks; Expected Attention found the right score; KVzip proved the scores work; KVzap made the scores cheap.** The family tree is a relay, not a competition.
3. **KVzap is the first method to pass all four criteria** — which is why it is the first method with a realistic path into vLLM and SGLang.
4. **Post-hoc pruning may have a successor**: end-to-end training with compression objectives, or architectures designed for sparse KV from the start. KVzap is the current apex, not the final word.

**Continue to** → **[KVzap: The Final Zap](../06-kvzap/)** — every method in this family tree compared directly on the KVpress Leaderboard, and the winner takes production.
