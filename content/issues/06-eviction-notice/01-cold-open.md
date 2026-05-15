---
title: "Memory Full"
description: "January 2026. Simon Jégou stares at a rejection email and a benchmark he can't ship. The KV cache is too big to fit, too important to delete — and a 14-line Python function is about to change that."
topics: [kv-cache, inference, pruning]
tags: [kvzap, nvidia, kv-cache, long-context, inference-efficiency]
theme: cream
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 10
techKind: mainline
techNode: cold-open
header: default.webp
---

## Santa Clara, January 2026

**Simon Jégou** is staring at two tabs on his monitor.

The left tab is a benchmark chart. It shows KVzip — his team's own compression method, published a few months earlier — running at 4× compression on a Llama-3.1 70B model, with accuracy losses so small they barely clear the measurement noise. For a {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} pruning result, it is genuinely spectacular. The kind of result that gets cited.

The right tab is an email. Three emails, actually — one from the vLLM maintainers, one from the SGLang team, one from NVIDIA's own TRT-LLM group. All three say the same thing, with slightly different phrasing:

> *2× prefill overhead. Decode incompatible. We cannot integrate.*

Jégou is an inference research engineer at **NVIDIA Santa Clara**, and those three engines — vLLM, SGLang, TRT-LLM — run the inference stack for roughly 90% of deployed LLM services. A compression method that none of them can use is, from an engineering standpoint, a compression method that does not exist.

He opens a third tab. A terminal. He starts typing.

## The Number That Started It All

Before we follow Jégou to what he types next, we need to understand what is actually happening to the servers his team is trying to fix.

A {{< wiki "attention" >}}multi-head attention{{< /wiki >}} layer does not operate on the current token in isolation. At every inference step, it needs to look back at every previous token it has ever seen — and to do that, it needs to remember two vectors for each of those tokens, for each layer, for each attention head. Those vectors are called **keys** and **values**, and the memory structure that holds them is the KV cache.

The shape of the cache is `(2, L, H, T, D)`:

- **2**: one slice for keys, one for values
- **L**: one slice per transformer layer
- **H**: one slice per attention head
- **T**: one slot per token in the current context
- **D**: the dimension of each key or value vector

The first four dimensions are fixed by the model architecture. The fifth — **T**, the number of tokens — grows without bound as the conversation or document gets longer.

Now run the napkin math on a real model. **Llama-65B** has 80 layers, 64 attention heads, and a head dimension of 128. Each value is stored in bfloat16 — two bytes. At a context length of 128,000 tokens:

$$
\text{KV memory} = 2 \times 80 \times 64 \times 128{,}000 \times 128 \times 2 \text{ bytes}
$$

That comes to **335 gigabytes**. For the KV cache alone, before you've loaded a single weight.

A four-card H100 server has **320 GB** of GPU memory total.

The cache does not fit.

```pyplot {id="kv-cache-memory" caption="KV CACHE MEMORY VS CONTEXT LENGTH FOR THREE PRODUCTION MODELS. THE H100 SERVER LINE IS TOTAL GPU MEMORY FOR A 4×H100 NODE."}
import numpy as np
import matplotlib.pyplot as plt

# Model specs: (name, n_layers, n_kv_heads, head_dim, color)
models = [
    ("Llama-65B",      80, 64, 128, '#FF007F'),
    ("Llama-3.1-8B",   32,  8, 128, '#00A8A8'),
    ("Qwen3-32B",      64, 16, 128, '#FFD700'),
]

context_lengths = np.logspace(np.log10(1000), np.log10(200_000), 300)
bytes_per_value = 2  # bfloat16

fig, ax = plt.subplots(figsize=(9.5, 5))

for name, L, H, D, color in models:
    memory_gb = (2 * L * H * context_lengths * D * bytes_per_value) / 1e9
    ax.plot(context_lengths / 1000, memory_gb, color=color, linewidth=2.5, label=name)

# Four H100 server line
ax.axhline(320, color='#FF8C00', linewidth=1.8, linestyle='--', label='4×H100 total GPU memory (320 GB)')

ax.fill_between([1, 200], [320, 320], [1e4, 1e4],
                color='#FF8C00', alpha=0.08)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("context length (k tokens)")
ax.set_ylabel("KV cache memory (GB)")
ax.set_title("KV cache memory alone can exceed an entire server's GPU memory", fontweight='bold')
ax.legend(framealpha=0.9)
ax.spines[['top', 'right']].set_visible(False)

# annotate the crossing
ax.annotate("Llama-65B crosses\nserver limit at ~90k tokens",
            xy=(90, 320), xytext=(30, 800),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A'),
            fontsize=9, color='#1A1A1A')

print("KV cache size at 128k context (bfloat16):")
for name, L, H, D, _ in models:
    gb = (2 * L * H * 128_000 * D * 2) / 1e9
    print(f"  {name}: {gb:.1f} GB")
```

At 128k context, Llama-65B needs 335 GB just for the cache. Even Llama-3.1-8B — a comparatively modest model — needs 42 GB, leaving almost no room for the weights, the activations, or anything else the serving engine needs to function.

This is not a hypothetical stress test. Real users routinely send 100k-token prompts to legal document analysis services, genomics pipelines, code review tools. Long context is not an edge case. It is the product.

{{< crosshead >}}The KVzip Problem{{< /crosshead >}}

KVzip was Jégou's first answer. The method works by running a **copy-and-paste pretext task** during prefill: it pastes a short question-and-answer pair at the end of the prompt and measures how much each cached token contributes to recovering the answer. Tokens that the model ignores contribute little; tokens the model attends to heavily contribute a lot. Score every token, keep the top fraction, discard the rest.

The results are excellent. 4× compression, negligible accuracy loss. But the scoring mechanism has two fatal engineering properties.

First: it doubles prefill time. You have to run the model *twice* — once to fill the cache, again to run the scoring pass. On a 128k-token document, that means 128k tokens of extra compute before the user sees a single response character.

Second: the scoring happens at prefill time, but the model also needs to prune during *decode* — as new tokens arrive and old ones age out of a fixed-size memory budget. KVzip's scoring pass cannot run token-by-token during decode without stalling generation entirely.

Jégou has a method that works mathematically. He does not have a method that can ship.

## Fourteen Lines

A few weeks after the rejection emails, Jégou writes a function. It is fourteen lines long in the version he eventually publishes, and it has an almost offensively simple name: `compress()`.

The function does not run a second forward pass. It does not look at attention weights. It does not examine what the model is attending to. It looks at a single tensor — the {{< wiki "residual-stream" >}}hidden state{{< /wiki >}} `h` at the current layer — and it runs that tensor through a tiny surrogate model that was trained once, offline, to predict something called a **KVzip+ score**.

That surrogate model is either a linear layer or a two-layer MLP. Smaller than a single attention head. It costs almost nothing to run.

And yet: when you use its predictions to decide which KV vectors to keep, the accuracy on {{< wiki "long-context-benchmarks" >}}RULER{{< /wiki >}}, LongBench, and AIME25 barely moves. You are throwing away 70% of the model's working memory — the entire accumulated context of the conversation — based on the judgment of a model small enough to fit in a footnote. And the big model, the one doing the actual work, barely notices.

The question this raises is not primarily an engineering one. It is a question about what the KV cache actually *contains*.

{{% pullquote type="counter-intuitive" %}}
70% of the KV cache can be discarded at runtime — and a surrogate model smaller than one attention head knows which 70%.
{{% /pullquote %}}

If a 14-line function can throw away 70% of the cache and the model keeps working, then 70% of the cache was never contributing very much to begin with. The KV cache is supposed to be the model's memory — its complete record of everything it has ever processed in this context window. How can most of that memory be garbage?

That question has an answer. And the answer starts earlier than KVzip, in a different year, with a different group of researchers staring at a different anomalous result.

## What You Need To Know Going In

Two pieces of context will make the rest of this issue legible.

**First:** {{< wiki "kv-pruning" >}}KV cache pruning{{< /wiki >}} is not the same as weight quantization. Quantization compresses the values stored in the cache — squeezing each float16 into a fewer bits. Pruning removes entire rows from the cache, discarding the key and value vectors for specific tokens entirely. The two approaches are complementary and are often combined in production systems, but they operate on different axes of the problem.

**Second:** the KV cache has five dimensions: layers, heads, tokens, and the key/value dimension. The field has made serious progress on compressing four of those five axes at *training time* — choosing architectures that are inherently cheaper to cache. The T-axis, the token axis, is the one that has resisted training-time solutions, because it depends on what the user actually sends. You cannot know at training time which tokens will matter for an arbitrary future prompt. You have to decide at runtime, with the actual content in front of you.

KVzap is a runtime solution. To understand why it works — why a surrogate model can predict token importance well enough to delete 70% of the cache — we need to understand the shape of the problem it is solving, and why the other four axes fell first.

**Continue to** → **[The Cache That Ate the Server](../02-kv-crisis/)** — the anatomy of the KV cache, the race to shrink each of its five dimensions, and the one dimension that nobody cracked until 2023.
