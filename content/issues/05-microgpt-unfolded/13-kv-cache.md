---
title: "The KV Cache"
description: "`keys[li].append(k)`. The most important data structure in production LLM serving is one line of Python. We turn it inside out and see why paged attention, KV quantization, and continuous batching are all optimizations of this one append."
topics: [transformer, inference]
tags: [microgpt, kv-cache, paged-attention]
theme: cream
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 130
techKind: mainline
techNode: kv-cache
header: 13-kv-cache.webp
---

## The Token That Cost A Thousand Tokens

In July 2020, two weeks after the GPT-3 API beta opened, a developer at a small Brooklyn startup ran a back-of-envelope calculation that should have been embarrassing for OpenAI but somehow wasn't. The pricing model at the time billed by **tokens generated**, not by **compute spent**. The developer noticed something off: their per-call latency scaled with the *prompt* length, not the *output* length. A 4-token completion off a 200-token prompt took roughly as long as a 4-token completion off a 20-token prompt times ten. The output was four tokens either way. What was the server doing for the other nine-tenths of the wall-clock?

Around the same time, an engineer working on the serving layer at OpenAI was reportedly heard muttering some version of: "if we re-run the prompt for every new token, we are doing $O(N^2)$ work to produce $N$ tokens." It was not a research insight. It was an *invoicing* insight. The fix was already old — it appears in Shazeer's *Fast Transformer Decoding* (2019) — but until people started paying $0.02 per thousand tokens for a 175B model, nobody outside research had counted the FLOPs.

The fix is the **KV cache**, and in microGPT it is *two lines of Python*:

```python
keys[li].append(k)
values[li].append(v)
```

That is the whole mechanism. The most important data structure in modern LLM serving — the thing that turns a $30K/H100 from a science demo into a profit-positive API — is `list.append`. The rest of this chapter is an excuse for those two lines.

## What Attention Demands

Recall the {{< wiki "attention" >}}attention{{< /wiki >}} math from [ch.8 attention](../08-attention/). To produce the output for token at position $t$, the model needs:

- The **query** $q_t$ at position $t$.
- The **keys** $k_0, k_1, \ldots, k_t$ — all of them, going back to the beginning of the sequence.
- The **values** $v_0, v_1, \ldots, v_t$ — same.

It then computes $t+1$ dot products $q_t \cdot k_i$, runs softmax, and uses the weights to average the $v_i$.

The key insight: $k_i$ and $v_i$ depend **only on the input at position $i$**. Once you have run the model on token $i$, the K and V vectors at every layer for that position are *fixed forever*. They will never change. The transformer is causal — future tokens cannot retroactively alter past keys.

So if we have already computed $k_0, \ldots, k_{t-1}$ on previous calls, why on earth would we recompute them when generating token $t$?

We wouldn't. We **save them**. That is the cache.

## The Mechanism, In Full

Look at the attention sub-block of microGPT. Stripped to essentials:

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
keys[li].append(k)
values[li].append(v)

# ... attention over keys[li] and values[li] follows ...
```

Three projections produce $q$, $k$, $v$ for the **current** token only. The `append` lines stuff $k$ and $v$ into a layer-local growing list. The subsequent inner loop — see [ch.9 multi-head](../09-multi-head/) — slices across `keys[li]` and `values[li]` for the dot-product-and-softmax.

The driver makes the lifecycle explicit:

```python
keys   = [[] for _ in range(n_layer)]
values = [[] for _ in range(n_layer)]

# Prefill: walk the prompt through, building the cache
for p_token in prompt_tokens:
    _ = gpt(token_id, pos_id, keys, values)
    token_id = p_token
    pos_id  += 1

# Decode: one token per call, reading + extending the cache
for _ in range(remaining):
    logits = gpt(token_id, pos_id, keys, values)
    ...
```

Two phases. Prefill **builds** the cache; decode **uses and grows** it. That split deserves its own chapter — see [ch.14 prefill vs decode](../14-prefill-decode/) — but the cache itself does not know the difference. It is just `n_layer` Python lists growing by one entry each time `gpt()` is called.

Two lines. **That is the cache.**

## The Without-Cache Counterfactual

Imagine, in a parallel universe, that we never added those `append` lines. To produce token $N+1$ we'd have to:

1. Run the model on token $0$ → get $k_0^{(li)}, v_0^{(li)}$ at every layer.
2. Run the model on token $1$ → get $k_1^{(li)}, v_1^{(li)}$ at every layer.
3. ...
4. Run the model on token $N$ → get $k_N^{(li)}, v_N^{(li)}$ at every layer.
5. *Now* compute attention at position $N$ using all of $k_0, \ldots, k_N$.

That's $N+1$ forward passes per new token, each itself an $O(N)$ attention computation. To generate a sequence of $T$ tokens: $O(T^3)$ total work. With cache: each new token costs $O(T)$ for its single attention step against the accumulated history; total $O(T^2)$.

A factor of $T$ savings. At $T = 1000$, that is **1000×**. At $T = 100{,}000$ — the kind of context modern frontier models advertise — it is **a hundred thousand times** the work.

```pyplot {id="with-without-cache" caption="FLOPs per new decode step. Without the cache, every token re-runs the model on the entire prefix. With the cache, only one new K/V projection happens; attention itself is linear in cache size."}
T = np.arange(1, 4096)

# Without cache: each new token re-runs the model on all N previous tokens
# Per-token FLOPs grow linearly with N (work for one decode step)
without_cache = T

# With cache: only project K, V for the new token (constant) + attention dot
# products against T cached keys (linear in T, but with a small constant).
# Project costs are constant per step; attention costs scale linearly with cache.
with_cache = 1 + 0.05 * T  # constant projection + tiny per-cached-key cost

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(T, without_cache, color='#FF007F', linewidth=2.5, label='without cache (recompute everything)')
ax.plot(T, with_cache,    color='#00A8A8', linewidth=2.5, label='with cache (append + dot)')
ax.fill_between(T, with_cache, without_cache, color='#FFD700', alpha=0.25, label='savings')

ax.set_xlabel('cache size T (tokens)')
ax.set_ylabel('relative FLOPs per decode step')
ax.set_title('Decode step cost: with vs without KV cache')
ax.legend(loc='upper left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 4096)
ax.set_ylim(0, 4500)
```

Note something important: even *with* cache, per-step cost is not constant — it scales **linearly with $T$** because each new query still has to dot-product against $T$ cached keys. The win is that you stopped re-running the **MLP** blocks and the **K/V projections** for the prefix. Those are the giant matrices. The remaining cost is just `q · k` and `attn · v` for cached entries.

For decode workloads on a 70B-parameter model, that distinction is everything. MLPs are roughly 60% of FLOPs per forward pass. Skipping $N$ MLP runs to produce one token is the difference between "interactive chat" and "go get coffee."

## Where The Cache Lives In Memory

Each `keys[li]` entry is a vector. In microGPT, of length `n_embd = 16`. In Llama 3 8B, of length `n_kv_head · head_dim = 8 · 128 = 1024` (with GQA — see [ch.18 GQA](../18-gqa/) for why $n_{\text{kv\_head}}$ differs from $n_{\text{head}}$).

In real implementations these Python lists become **contiguous tensors**. The full cache is a 5D array — described in detail in [ch.17 KV axes](../17-kv-axes/) — of shape

$$
(\underbrace{2}_{K,V}, \ \underbrace{L}_{\text{layers}}, \ \underbrace{H}_{\text{heads}}, \ \underbrace{T}_{\text{tokens}}, \ \underbrace{D}_{\text{head\_dim}})
$$

with total byte count

$$
\text{bytes} = 2 \cdot L \cdot H \cdot T \cdot D \cdot b
$$

where $b$ is bytes per element (2 for bf16, 1 for int8, 0.5 for int4).

## Napkin Math: How Big Does It Get?

Llama 3 8B, bf16, with its production settings:

- $L = 32$ layers
- $H = 8$ KV heads (GQA: 32 query heads grouped into 8 KV heads)
- $D = 128$ head dimension
- $b = 2$ bytes per element

Per token, the cache costs

$$
2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072 \text{ bytes} \approx 128 \text{ KB / token}
$$

So a 128K-token context demands

$$
128{,}000 \cdot 128 \text{ KB} \approx 16.4 \text{ GB}
$$

of cache, in addition to the 16 GB of weights. On a 24 GB consumer GPU you simply cannot run a 128K-token Llama 3 8B inference — the *cache* alone exceeds 80% of VRAM. On an 80 GB H100 it fits, but barely, and you have no headroom for batching.

Now imagine vanilla Multi-Head Attention with $H = 32$ heads (no GQA):

$$
2 \cdot 32 \cdot 32 \cdot 128{,}000 \cdot 128 \cdot 2 \approx 67 \text{ GB}
$$

GQA saved **50 gigabytes** of GPU memory. That is the single architectural decision that made long-context Llama inference economically viable on commodity hardware. With DeepSeek-V2's MLA — which collapses $H \cdot D$ into a tiny latent dimension — the same context fits in **under 2 GB** of cache. See [ch.19 MLA](../19-mla/) for that mathematical trick.

```pyplot {id="cache-vs-context" caption="KV cache memory vs context length for three architectures. Vanilla MHA crosses the 80GB H100 line at ~150K tokens. GQA buys you a 4× headroom. MLA pushes the ceiling another order of magnitude out."}
T = np.linspace(1024, 200000, 200)
L, D, b = 32, 128, 2

# Vanilla MHA: H = 32 KV heads
mha = 2 * L * 32 * T * D * b / 1e9          # GB

# GQA: H = 8 KV heads (Llama 3 setup)
gqa = 2 * L * 8 * T * D * b / 1e9            # GB

# MLA-ish: replace H*D with a latent of dim ~4*D/9 (per the DeepSeek paper, with H=32 in source model)
# total per-layer cache element count: ~4*32*128/9 ~ 1820 elements per token per layer per K-or-V
mla = 2 * L * (4 * 32 * D / 9) * T * b / 1e9 # GB

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(T/1000, mha, color='#FF007F', linewidth=2.5, label='MHA (H=32)')
ax.plot(T/1000, gqa, color='#FF8C00', linewidth=2.5, label='GQA (H=8) — Llama 3')
ax.plot(T/1000, mla, color='#00A8A8', linewidth=2.5, label='MLA (latent) — DeepSeek V2')
ax.axhline(80, color='#1A1A1A', linewidth=1.5, linestyle='--', alpha=0.7)
ax.text(5, 82, '80 GB H100 ceiling', fontsize=9, color='#1A1A1A')
ax.axhline(24, color='#1A1A1A', linewidth=1.5, linestyle=':', alpha=0.5)
ax.text(5, 25.5, '24 GB consumer GPU', fontsize=9, color='#555')

ax.set_xlabel('context length T  (thousands of tokens)')
ax.set_ylabel('KV cache size  (GB, bf16)')
ax.set_title('KV cache memory vs context length — L=32, D=128')
ax.legend(loc='upper left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 200)
ax.set_ylim(0, 100)
```

## The Bytes-Per-Token Trick

The cache shape is fixed by the architecture, but the **element type is not**. Real serving stacks store the cache in whatever {{< wiki "number-formats" >}}precision{{< /wiki >}} they can get away with:

| Element type | Bytes/elem | Llama-3-8B at 128K | Notes |
|---|---|---|---|
| fp32 | 4 | 32.8 GB | Never used in practice |
| bf16 / fp16 | 2 | 16.4 GB | Default |
| int8 | 1 | 8.2 GB | Drop-in with light calibration |
| int4 (KIVI, KVQuant) | 0.5 | 4.1 GB | Per-channel quant, channel-aware groups |
| int2 (KIVI-2) | 0.25 | 2.1 GB | Asymmetric per-channel quant, modest quality hit |

Issue 03 took the entire 16-chapter arc to explain why naïve 4-bit quantization of attention weights fails (outlier channels) but works well on KV-cache tensors (different statistics). The mechanism is identical to the one for weights — just applied to a different tensor. The cache *is* a tensor.

## Two Lines, A Hundred Papers

Looking at `keys[li].append(k)` it is hard to believe entire research subfields live inside it. They do.

- **Paged attention (vLLM, 2023).** Stop allocating one contiguous tensor per sequence. Chop the cache into fixed-size *pages* (typically 16 tokens each) and let an OS-style page table track which pages belong to which conversation. Pages can be shared across batched sequences that share a prefix, doubling effective batch size in chat workloads. The Python equivalent: replace `list.append` with a page-allocator. Same data; different memory layout.

- **KV quantization (KIVI, KVQuant, 2023–2024).** Compress the cache to 4-bit or 2-bit per element. Per-channel scales, channel-grouped quantization to defend against outlier dimensions. See [issue 03's KV-cache deep-dive](../../03-sixteen-numbers/) for why the K and V tensors have very different statistical profiles and need different quantization schemes.

- **Continuous batching (Orca, 2022).** When one sequence in a batch finishes, immediately replace it with a new one *without waiting for the batch to drain*. Practically, this requires the cache layout to be per-sequence and decoupled — which paged attention enables.

- **Architectural compression.** Shrink the cache by changing the model itself. Either share K/V across head groups ([GQA, ch.18](../18-gqa/)), project K/V into a low-rank latent ([MLA, ch.19](../19-mla/)), use sliding-window attention for some layers ([ch.20](../20-sliding-window/)), or replace some attention layers with state-space models ([ch.21](../21-ssm-hybrids/)). The whole organizing framework is in [ch.17 KV axes](../17-kv-axes/).

Every one of those four families is, at the implementation level, **a replacement for the two-line append**. The cache mechanism is invariant. What changes is where the bytes live, how many of them there are, and what precision they are stored in.

## What To Remember

1. **The KV cache is `list.append` per layer.** Two lines of Python. Everything sophisticated about it is an optimization of where those lists live in memory.
2. **Without the cache, decoding token $N$ costs $O(N)$ work; with the cache, $O(N)$ for attention plus $O(1)$ for K/V projection.** Generating $T$ tokens drops from $O(T^3)$ to $O(T^2)$.
3. **Cache memory is $2 \cdot L \cdot H \cdot T \cdot D \cdot b$.** For Llama 3 8B at 128K bf16: 16.4 GB. For vanilla MHA same model: 67 GB. GQA saved 50 GB.
4. **At long contexts, the cache exceeds the weights.** This is why the entire 2023–2026 frontier-architecture race has been about shrinking specific axes of this five-dimensional tensor.
5. **`keys[li]` is fixed once written.** Past keys never change. That immutability is what makes caching mathematically valid and architecturally interesting.

---

**Continue to** → [Prefill vs Decode](../14-prefill-decode/) — the cache is built in one phase and consumed in another, and those two phases have such different compute profiles that modern serving stacks run them on entirely different hardware.

