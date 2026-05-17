---
title: "Reusing The Prologue"
description: "Every conversation in a deployment starts with the same system prompt. Hashing prefix blocks and re-using their physical memory turns a full prefill into a cache hit — the single highest-leverage optimization in modern LLM serving, and an almost embarrassingly simple one in hindsight."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T14:30:00-04:00
issue: 8
weight: 120
techKind: mainline
techNode: prefix-caching
header: 12-prefix-caching.webp
---

## A 1,500-Token Letter Of Introduction

**January 2024**, somewhere in a deployment engineer's terminal. The system prompt for a production assistant currently reads, in its entirety:

> *You are Aria, a helpful and knowledgeable customer support agent for Acme Corp. You are friendly, professional, and concise. You always respond in the customer's language. You have access to the following tools: search\_kb(query), create\_ticket(...), escalate(...). Here are sixteen worked examples of ideal responses…*

It is **1,500 tokens long**. Every single request to the assistant starts with these 1,500 tokens — followed, eventually, by something like *"hi, my package hasn't arrived,"* a 50-token user question.

Now do the napkin math on what happens server-side.

A naïve serving stack treats every request as fresh. The prompt is $1{,}550$ tokens. Prefill cost scales linearly in tokens (each token must do a forward pass through the model, doing $O(\text{tokens} \times \text{params})$ FLOPs). So this request does $1{,}550$ tokens of prefill, of which $1{,}500$ are the *same byte-for-byte sequence the model just prefilled for the previous user, and the user before that, ten thousand times today*.

The {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} vectors for those 1,500 prefix tokens are deterministic functions of those tokens. The model's weights have not changed. The token IDs have not changed. The position embeddings have not changed. The KV vectors must, by elementary causality, be identical. We are recomputing them anyway, on every request, because nothing remembered the answer.

This is the kind of bug that, once you see it, you cannot un-see. The fix saves **31×** of prefill compute on this workload — collapsing 1,550 tokens of work down to 50 — and roughly the same factor on user-visible time-to-first-token. It is **the highest-leverage single optimization in the entire modern serving stack**. And it works, end-to-end, because of the page-table abstraction we built in [Borrowing from 1965](../10-paged-attention/).

## The Empirical Premise

The reason this optimization works at all is an empirical observation about production traffic: **most prompts are mostly prefix**.

Walk through the categories of prompt content a real serving cluster sees:

- **System prompts.** 200 to 5,000 tokens. The same string repeats across hundreds of thousands of requests per day. Shared coverage: $100\%$.
- **Few-shot examples.** 500 to 10,000 tokens of "here are 16 examples of how to answer." Shared across all requests in a deployment. Coverage: $100\%$.
- **Tool / function schemas.** A few thousand tokens describing every callable. Coverage: $100\%$.
- **Conversation history.** Each turn extends the prefix of the *next* turn in the same conversation by exactly the user's message and the model's reply. Coverage: ~$100\%$ within a conversation; ~$0\%$ across users.
- **RAG context.** Retrieved chunks. Highly variable; some hit rate from documents that get retrieved often, none from the long tail.
- **The user's actual new turn.** The only part that is uniquely fresh.

When you measure real chat workloads in production logs, you find that **60% to 90% of all tokens in incoming prompts are byte-identical to tokens the model has previously prefilled in the last few minutes**. The cache hit rate is enormous because users are not unique snowflakes — they share infrastructure, system prompts, and conversation patterns.

```pyplot {id="prefix-coverage" caption="Distribution of prefix-cache coverage across 20,000 synthetic production requests. Three workload mixes; the chatbot and RAG distributions concentrate above 70% coverage. The naive 'every prompt is unique' assumption is the outlier on the left."}
np.random.seed(42)

n_samples = 20_000

# Chatbot: same system prompt + few-shot, plus short user question
chat = np.clip(np.random.beta(8, 1.4, n_samples), 0, 1)

# RAG: shared instructions + variable retrieved context + short query
rag = np.clip(np.random.beta(3.5, 1.6, n_samples), 0, 1)

# Code completion / API: less shared; mix of stable headers and fresh code
code = np.clip(np.random.beta(1.6, 2.5, n_samples), 0, 1)

fig, ax = plt.subplots(figsize=(10, 4.6))
bins = np.linspace(0, 1, 41)
ax.hist(chat, bins=bins, alpha=0.65, color='#FF007F',
        label=f'chatbot  (mean = {chat.mean():.0%})',
        edgecolor='#1A1A1A', linewidth=0.3)
ax.hist(rag, bins=bins, alpha=0.55, color='#FFD700',
        label=f'RAG       (mean = {rag.mean():.0%})',
        edgecolor='#1A1A1A', linewidth=0.3)
ax.hist(code, bins=bins, alpha=0.55, color='#00A8A8',
        label=f'codegen   (mean = {code.mean():.0%})',
        edgecolor='#1A1A1A', linewidth=0.3)
ax.axvline(0.0, color='#1A1A1A', linewidth=0.6, linestyle='--')
ax.text(0.02, ax.get_ylim()[1]*0.9, "naive\nassumption",
        fontsize=8, color='#1A1A1A')

ax.set_xlabel("Fraction of prompt tokens that match a cached prefix")
ax.set_ylabel("Number of requests")
ax.set_title("Real workloads spend most of their prompt in shared prefix")
ax.legend(loc='upper left')
ax.spines[['top','right']].set_visible(False)

for name, arr in [("chatbot", chat), ("RAG", rag), ("codegen", code)]:
    pct_high = (arr > 0.5).mean()
    print(f"{name:8s}: mean={arr.mean():.0%}, p50={np.median(arr):.0%}, >50% cached: {pct_high:.0%}")
print()
print("→ Naive 'every prompt is unique' (the leftmost bin) is empirically false.")
print("→ The cache hit rate is enormous because users share infrastructure.")
```

The picture is sobering. If you are designing a serving system and you are not exploiting prefix sharing, you are doing roughly $3\times$ more prefill compute than physics requires. The naive bar at $0\%$ on the left is what a serving stack looks like with prefix caching disabled. Every production deployment now lives somewhere in the right two thirds of this plot.

{{% marginnote %}}Anthropic and OpenAI both advertise prefix caching as a billing line: cached input tokens are charged at 10× lower rate, because they cost the provider 10× to 30× less to serve. The business model and the GPU memory layout are the same fact, observed from different sides.{{% /marginnote %}}

## Equal Prefix, Equal KV

The premise that makes this work is almost suspiciously clean. {{< wiki "attention" >}}Attention{{< /wiki >}} is causal: the KV vector at position $t$ is a function only of tokens $0, 1, \ldots, t$. The model weights are fixed. Position encodings are deterministic. Therefore:

$$
\text{KV}_t(x_{0:T}) = \text{KV}_t(y_{0:T'}) \quad \text{whenever} \quad x_{0:t} = y_{0:t}.
$$

If two requests share the first $k$ tokens, their KV vectors at positions $0, \ldots, k$ are *bitwise identical*. Not approximately — identical. So if request A computed them last minute, request B can just *point at the same physical memory*. The block-table abstraction from [the previous chapter](../10-paged-attention/) was built exactly for this — a block table is per-request, but two block tables can hold the same physical block ID and the KV pool will treat both lookups equivalently.

{{% callout type="theorem" %}}
**The prefix-sharing theorem.** For a causal transformer with fixed weights, two sequences sharing a length-$k$ prefix produce bit-identical KV vectors for all positions $0, \ldots, k$. Therefore they can share the *physical* KV blocks for those positions with zero loss of correctness.

This is not an approximation. There is no quality degradation, no calibration step, no quantization error. The math says two equal things are equal, and the implementation says two equal things use one allocation.
{{% /callout %}}

## Hash The Block, Chain The Hash

The remaining question is operational: given an incoming request's first $k$ tokens, *how do we look up* whether the KV for some prefix of those tokens is already resident? Doing a linear scan over every block in the pool is $O(N \cdot k)$. Way too slow.

The answer is the standard one: hash the contents, hash-table the lookup. But there is a subtlety. We are hashing *KV blocks*, each of which represents 16 tokens — and a block's KV depends on **all preceding tokens**, not just its own 16. So we need a hash scheme where:

- Two blocks with the same hash imply the same KV.
- A block's hash incorporates everything before it.

The solution is a **chained hash**:

$$
h_j \;=\; H\!\left( h_{j-1} \, \Vert \, \text{token\_ids}[16j : 16(j+1)] \right), \qquad h_{-1} := 0.
$$

Block $j$'s hash is the hash of the previous block's hash concatenated with the 16 tokens in this block. Two sequences whose first $16(j+1)$ tokens agree will produce the same $h_j$; any mismatch anywhere in those tokens propagates downstream and breaks the chain.

```pyplot {id="hash-chain" caption="Two requests share the first 32 tokens (block 0 and block 1 hash equal). They diverge at token 33; block 2's hash differs and so does every block after. The chain is exactly the prefix-match boundary."}
import hashlib

def block_hash(prev, toks):
    s = hashlib.blake2b(digest_size=8)
    s.update(prev.to_bytes(8, 'big', signed=False) if isinstance(prev, int) else prev)
    s.update(bytes(toks))
    return int.from_bytes(s.digest(), 'big')

# Two requests' token IDs (length 64), differing only at position 33
np.random.seed(99)
seq_a = np.random.randint(0, 100, 64)
seq_b = seq_a.copy()
seq_b[33] = (seq_a[33] + 1) % 100   # tiny mutation, position 33

def hashes(seq, block=16):
    hs, prev = [], 0
    for j in range(len(seq) // block):
        h = block_hash(prev, seq[j*block:(j+1)*block].tolist())
        hs.append(h)
        prev = h
    return hs

ha = hashes(seq_a)
hb = hashes(seq_b)

fig, ax = plt.subplots(figsize=(10.5, 3.8))
labels = [f"block {j}\ntokens {j*16}-{j*16+15}" for j in range(4)]
xs = np.arange(4)

for i, (h, color, label, y) in enumerate([
    (ha, '#FF007F', 'request A', 0.65),
    (hb, '#00A8A8', 'request B', 0.0),
]):
    for j, hv in enumerate(h):
        match = (hv == ha[j])
        face = color if (i == 0 or match) else '#FDF5E6'
        edge = color
        ax.add_patch(plt.Rectangle((j, y), 0.92, 0.45,
                                    facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(j+0.46, y+0.22, f"{hv:0>16x}"[:8] + "…",
                ha='center', va='center',
                fontsize=8, color='#1A1A1A', family='monospace')
    ax.text(-0.3, y+0.22, label, ha='right', va='center', fontsize=10, fontweight='bold')

ax.set_xticks(xs+0.46)
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks([])
ax.set_xlim(-1.6, 4.1)
ax.set_ylim(-0.25, 1.35)
ax.set_title("Hash chain: A and B agree on blocks 0-1, diverge at block 2 onward",
             fontsize=10, loc='left')
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)

shared = sum(1 for a, b in zip(ha, hb) if a == b)
print(f"Blocks where A's hash == B's hash: {shared} of {len(ha)}")
print(f"  → A and B can share KV for the first {shared * 16} tokens")
print(f"  → For position {shared*16} onward, KV must be computed fresh")
print(f"  → One token changed at position 33 invalidates blocks 2 and 3.")
```

The hash table holds one entry per *unique block prefix* the system has seen recently: `block_hash → physical_block_id`. On an incoming request, walk the new prompt 16 tokens at a time, compute each chained hash, probe the table. Each hit lets us point the new block table at an existing physical block and bump its refcount. **Stop on the first miss** — the moment one block diverges, every downstream hash is poisoned, so the rest of the prompt has to be prefilled fresh.

## What This Buys, Quantitatively

The payoff curve as a function of hit rate is the most important plot in this chapter. Time-to-first-token (TTFT) is roughly proportional to prefill tokens computed; prefill tokens computed is $(1 - h) \cdot T$ where $h$ is the prefix hit fraction. So:

```pyplot {id="ttft-vs-hitrate" caption="TTFT and prefill compute as a function of prefix-cache hit rate. The relationship is linear (in token count) but the user-visible experience drops off a cliff: at 90% hit rate you serve users 10× faster while spending 10× less on every prefill."}
T_full = 1550
hit_rate = np.linspace(0, 1, 101)
prefill_tokens = (1 - hit_rate) * T_full
ttft_ms = prefill_tokens * 0.18   # roughly 0.18 ms / token on a heavy model

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(hit_rate*100, prefill_tokens, color='#FF007F', linewidth=2.6)
ax1.fill_between(hit_rate*100, 0, prefill_tokens, color='#FF007F', alpha=0.18)
ax1.set_xlabel("Prefix-cache hit rate (%)")
ax1.set_ylabel("Tokens to prefill")
ax1.set_title("Prefill work collapses linearly in hit rate")
ax1.spines[['top','right']].set_visible(False)
ax1.axhline(50, color='#1A1A1A', linewidth=0.8, linestyle='--')
ax1.text(2, 100, "fresh user turn (50 tokens)",
         fontsize=9, color='#1A1A1A')

ax2.plot(hit_rate*100, ttft_ms, color='#00A8A8', linewidth=2.6)
ax2.fill_between(hit_rate*100, 0, ttft_ms, color='#00A8A8', alpha=0.18)
ax2.axhline(300, color='#FF8C00', linewidth=1.2, linestyle='--')
ax2.text(2, 320, "300 ms 'feels instant' threshold",
         fontsize=9, color='#FF8C00')
ax2.set_xlabel("Prefix-cache hit rate (%)")
ax2.set_ylabel("TTFT (ms)")
ax2.set_title("User-visible latency: from 280 ms to 9 ms")
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()

for h in [0, 0.5, 0.9, 0.97]:
    pt = (1-h) * T_full
    t  = pt * 0.18
    print(f"hit rate = {h:>4.0%}  →  prefill {pt:>4.0f} tokens  →  TTFT {t:>5.1f} ms")
print()
print("The chatbot from the opener (1,500 prefix + 50 user) at 97% hit rate sees")
print("TTFT collapse from ~280 ms to ~9 ms. Same model. Same hardware.")
```

The numbers on the right column are the ones a product manager cares about. The user *feels* the difference between $280$ ms and $9$ ms more sharply than they feel almost anything else about a chatbot's behaviour — it is the difference between waiting for a response and the response appearing *as you stop typing*. Prefix caching is the cheapest path to that experience, by far.

{{% pullquote type="counter-intuitive" %}}
The single most user-visible latency optimization in a modern LLM serving stack is not a new kernel, a new dtype, or a new model architecture. It is a hashmap.
{{% /pullquote %}}

## The Tricky Bits

Three things make a real implementation harder than the sketch.

**1. Eviction policy.** Prefix-cached blocks must be *kept around after the request finishes* — that's the whole point. They sit in the free queue tail (from [The Block Manager](../11-block-manager/)) with refcount $0$, waiting for the next request whose hash matches. When the pool truly runs out, the LRU tail is evicted. This is correct by construction: the most recently used prefix blocks survive longest, which is exactly what you want for system prompts and few-shot examples that get hit constantly.

**2. Hash collisions.** vLLM uses **xxHash** (64-bit), not SHA-256. Cryptographic strength is overkill — we are not protecting against adversaries, we are protecting against the birthday paradox. With $N$ in-pool blocks, the collision probability per insert is bounded by

$$
P(\text{collision}) \;\lesssim\; \frac{N}{2^{64}}.
$$

For $N = 200{,}000$ blocks (a fully populated 64 GB pool), that's $\sim 10^{-14}$ per insert. {{% marginnote %}}If you are running a 1,000-GPU fleet at full saturation for a year, you might see one collision. The mitigation is to *additionally* compare the actual token-ID tuples on a hit before trusting it; vLLM does this. The expected cost is $O(1)$ comparisons per hit because $10^{-14}$ collisions are not, in fact, a thing that happens.{{% /marginnote %}}

**3. Tokenization sensitivity.** This one is treacherous. A single byte change at position zero — say, the prompt starts with `"Hi, "` versus `"Hi,"` (one extra space) — produces different token IDs, which produces different $h_0$, which poisons every downstream hash and forces a full fresh prefill. Real systems pre-canonicalize whitespace and BOS-token handling at the API boundary, so that *logically equivalent* prompts produce byte-identical token sequences. If you ever wondered why production tokenizers are picky about whitespace, this is why.

## Beyond Server-Side Caching

Once you have prefix-cached the KV, the next question is whether you can be even smarter inside the attention kernel itself. Two flavors stand out.

**Hydragen.** {{< cite text="Juravsky et al., 2024" url="https://arxiv.org/abs/2402.05099" kind="paper" >}} observed that when many requests in a batch share a prefix, the attention computation for the shared prefix tokens is *the same K and V being read against $B$ different queries*. So instead of reading those K and V tokens $B$ times (once per batch element), you read them *once* and dispatch all $B$ queries against them. The math comes out identical (online softmax composes across the prefix and the per-request suffix); the bandwidth saving is up to $B \times$ on the prefix portion. Hydragen is the kernel-level cousin of prefix caching: same insight, different layer of the stack.

**Cascade attention.** Generalizes Hydragen to *tree-structured* prefix sharing. If five requests share a 1,500-token system prompt and three of those five share an additional 500-token few-shot block on top, the attention kernel can read the 1,500-token block once for all five, the 500-token block once for three, and the user-specific suffixes per request. Each level of the tree is one batched attention pass.

**SGLang's RadixAttention.** {{< cite text="Zheng et al., 2023" url="https://arxiv.org/abs/2312.07104" kind="paper" >}} took the same idea and built it into an entire serving framework, storing prefixes in a radix tree rather than a flat hashmap. The radix tree lets the cache see partial-block prefixes and arbitrary branching patterns. The headline result was 5× throughput on agent workloads where many requests share long structured prefixes — exactly the workload that becomes most of production in the agent era.

**TRT-LLM** ships an equivalent feature under a different name. Every serious serving stack now has prefix caching as a first-class citizen. They all converge on the same architecture: paged KV cache, chained hashes, refcount-and-LRU. Convergence under independent invention is, as always, evidence that the abstraction is the right one.

## What To Remember

1. **Equal prefix implies equal KV.** Causal attention makes prefix positions independent of future tokens, so they cache. There is no quality penalty — it is bitwise identical math.
2. **Hash 16 tokens at a time, chain through the previous block's hash.** Same hash implies same prefix back to position $0$. Stop on first miss. The hashmap *is* the cache, and the cache *is* the LRU tail of the free queue.
3. **In production, this is the optimization with the largest user-visible effect.** TTFT collapses from "a noticeable pause" to "instant." Throughput rises by the cache-hit factor. The GPU stops re-doing work that was done a thousand times this minute.

**Continue to → [Slicing the Prefill](../13-chunked-prefill/)** — even when there is no prefix to reuse, the prefill itself can be chopped up so it stops monopolizing the GPU while every other user waits for their next decode token.
