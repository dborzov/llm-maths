---
title: "FlashAttention: never write the score matrix to HBM"
short_title: "FlashAttention"
description: "Naive attention at T=8192 writes and re-reads 686 GB of intermediate scores per forward pass across Llama-3-70B's heads — FlashAttention tiles the computation in SRAM and runs online softmax so that matrix never crosses the HBM bus."
blurb:
  - "Tri Dao's 2022 insight: the FLOPs are nearly free; the bytes are murder — and the score matrix doesn't have to leave SRAM."
  - "At T=128K, a single FP16 score matrix is 32 GB per head — the naive pipeline is architecturally impossible, not just slow."
  - "Tiled SRAM + online softmax gives the same exact output as the naive formula, 2–4× faster, with no accuracy change."
  - "FlashAttention is the skeleton every modern attention variant — paged, sliding-window, GQA, MLA — is layered on top of."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T11:30:00-04:00
issue: 8
weight: 60
techKind: primer
techNode: flash-attention
header: 06-flash-attention.webp
---

## Stanford, May 2022

The thesis is overdue, again.

**Tri Dao** is a Stanford PhD student in **Christopher Ré**'s lab, the same group that has spent five years asking whether sequence-modelling architectures *have* to look like attention. By spring 2022 the lab has produced state-space models, structured matrices, mixer architectures — a parade of attention alternatives, each almost-as-good but never quite better. The pattern is familiar across the field. The whole community has been trying to *replace* attention because of one ugly fact: attention's compute and memory both scale as $T^2$, which means doubling the context quadruples the cost, which means context length is permanently stuck around 2048.

Dao stops trying to replace it. He asks a simpler question, the one nobody is asking because everyone assumes the answer is obvious. The question is:

> Is attention really $O(T^2)$?

The *math* is. Sure. Compute the score matrix $S = QK^\top$, that's $T \times T$ entries, that's quadratic. Softmax it, multiply by $V$ — all quadratic. Asymptotically you cannot escape it. But asymptotic FLOPs are not what makes a kernel slow on real silicon. What makes a kernel slow is **bytes moved across the HBM bus**. And, Dao notices, the $T \times T$ score matrix is being *written to HBM* and then *read back from HBM* before being multiplied by $V$. That round-trip is the actual cost. The FLOPs are nearly free; the bytes are murder.

So what if you never wrote the score matrix to HBM at all?

The paper that comes out a few weeks later — {{< cite text="Dao et al., 2022" url="https://arxiv.org/abs/2205.14135" kind="paper" >}} — is titled, with engineer's bluntness, *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."* The result is the same {{< wiki "attention" >}}attention{{< /wiki >}} you have always known — the same softmax, the same $QK^\top V$ — computed 2–4× faster, with no accuracy change, by rearranging the order of operations so that the score matrix never leaves SRAM. In the [roofline picture](../04-roofline/), Dao is performing **Move 3**: load the byte once, use it many times.

This primer is the autopsy of that rearrangement. {{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}} is the single most important inference kernel of the past decade not because of its FLOP count — that did not change — but because it is the algorithmic skeleton on which every modern attention variant (paged, sliding-window, MLA, NSA, GQA) is layered. Master it once and you have mastered 90% of attention engineering.

## The Naive Pipeline And Why It Hurts

Standard attention, as written in a clean tutorial, looks like this:

$$
S \;=\; \frac{Q K^\top}{\sqrt{d}}, \qquad P \;=\; \text{softmax}(S), \qquad O \;=\; P V.
$$

If you implement that *literally* on a GPU, you get the following four-act tragedy:

1. **Load** $Q$ and $K$ tiles into SRAM, compute $S = QK^\top$, **write $S$ to HBM**. Bytes written: $T^2 \cdot 2$ (FP16).
2. **Read $S$ back from HBM**, compute row-wise {{< wiki "softmax" >}}softmax{{< /wiki >}} $P$, **write $P$ to HBM**.
3. **Read $P$ back**, load $V$ tiles, compute $O = PV$, **write $O$**.
4. Send $O$ down the residual stream.

Two full $T \times T$ matrices crossing the HBM bus twice each. Let's do the napkin math at modern context lengths.

For $T = 8192$ in FP16, the score matrix is $8192^2 \cdot 2 = 134$ MB **per head**. Llama-3-70B has $64$ heads, $80$ layers — so a single forward pass writes and re-reads $134 \cdot 64 \cdot 80 = 686$ GB of intermediate score data. At H100 HBM bandwidth of $3.35$ TB/s, that's $\sim 200$ ms of pure bus traffic, *not counting the multiplies.* For $T = 32{,}768$ — the long-context regime LLMs were already pushing into — multiply by $16$. The whole picture turns into a thirty-second-per-prompt nightmare *just to move scores around.*

It gets worse: the score matrix doesn't fit in HBM at all for very long contexts. At $T = 128\text{K}$, a single $T \times T$ FP16 score matrix is $32$ GB *per head*. The model literally cannot store its own attention scores.

So the naive pipeline is more than slow. It is **architecturally impossible** at long context. Something had to give.

```pyplot {id="naive-vs-flash-hbm" caption="HBM bytes moved by the attention kernel vs sequence length. Naive attention's score-matrix round-trips give a quadratic parabola; FlashAttention's tile-and-reuse gives a line. By T = 32K the gap is over 1000x."}
import numpy as np
import matplotlib.pyplot as plt

T = np.array([128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
D = 128  # head dim

# Naive: write and read T*T score + T*T softmax + load QKV. Two FP16 bytes each.
naive_bytes = (2 * T**2 * 2)   # write+read S, plus write+read P (same matrix conceptually)

# FlashAttention: load Q,K,V each exactly once. ~3 * T * D * 2.
flash_bytes = 3 * T * D * 2

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.loglog(T, naive_bytes / 1e9, color='#FF007F', linewidth=2.6, marker='o',
          markersize=8, markerfacecolor='#FFD700', markeredgecolor='#1A1A1A',
          markeredgewidth=1.2, label='naive (writes T x T to HBM)')
ax.loglog(T, flash_bytes / 1e9, color='#00A8A8', linewidth=2.6, marker='s',
          markersize=8, markerfacecolor='#FFD700', markeredgecolor='#1A1A1A',
          markeredgewidth=1.2, label='FlashAttention (Q, K, V once)')

# Crossover annotation
for Ti in [1024, 8192, 32768]:
    naive_i = (2 * Ti**2 * 2) / 1e9
    flash_i = (3 * Ti * D * 2) / 1e9
    ratio = naive_i / flash_i
    ax.annotate(f"{ratio:,.0f}x", (Ti, naive_i), xytext=(-8, 8),
                textcoords='offset points', fontsize=10, fontweight='bold',
                color='#1A1A1A')

ax.set_xlabel("sequence length T")
ax.set_ylabel("HBM bytes moved per head per layer  (GB)")
ax.set_title("Naive attention pays for the score matrix; Flash pays for Q, K, V")
ax.legend(loc='upper left', framealpha=1, edgecolor='#1A1A1A')
ax.grid(True, which='both', alpha=0.15)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Two curves, $O(T^2)$ and $O(T)$. Until they diverge in the picture, you might think the difference is marginal. By $T = 32\text{K}$ — context length the field passed through in 2023 — naive attention is moving $1{,}500\times$ more bytes than Flash. At $T = 128\text{K}$, it's $6{,}000\times$. The gap is the whole reason your favourite chatbot can read a 200K-token codebase before answering a question.

## The Insight That Makes The Tiling Legal

The reason the naive pipeline writes the score matrix to HBM is **softmax**. Softmax over a row needs to know the row's maximum and the row's sum:

$$
\text{softmax}(s_j) \;=\; \frac{e^{s_j - \max_i s_i}}{\sum_i e^{s_i - \max_i s_i}}.
$$

You cannot finish a row's softmax until you have seen every score in that row. Therefore — the naive argument goes — you must compute every score first, write them down, then go back and softmax. Quadratic memory, no choice.

This is wrong, and the wrongness is the entire breakthrough.

The trick is **online softmax**, an idea older than FlashAttention — it shows up in numerical-analysis literature in the 1970s as a way to compute logsumexp without overflow — and one that Dao's contribution is to weaponize for IO efficiency. The observation: you can maintain a *running* maximum and a *running* sum, and as new scores arrive, you can correct your previous estimates with a multiplicative rescaling.

Let $m$ be the running max and $\ell$ be the running normalizer. When a new block of scores arrives with its own block-max $m_b$ and block-sum $\ell_b$, the update is:

$$
m_\text{new} \;=\; \max(m_\text{old},\, m_b),
$$

$$
\ell_\text{new} \;=\; e^{m_\text{old} - m_\text{new}}\, \ell_\text{old} \;+\; e^{m_b - m_\text{new}}\, \ell_b.
$$

Read those two lines carefully. The first updates the running max — trivial. The second rescales the *previous* normalizer by $e^{m_\text{old} - m_\text{new}}$, which is the correction factor for the fact that we are now subtracting a different max in the exponent. It is the same trick you use when you compute logsumexp incrementally: subtract whichever max you currently know, and when you learn a bigger max, multiply your accumulated sum by $e^{\text{old}-\text{new}}$ to bring it onto the new scale.

The output accumulator $O$ gets the same treatment. After processing block $b$ with partial output $O_b = P_b V_b$, the update is:

$$
O_\text{new} \;=\; e^{m_\text{old} - m_\text{new}}\, O_\text{old} \;+\; e^{m_b - m_\text{new}}\, O_b.
$$

Three running scalars per query row — $m$, $\ell$, and the vector $O$ — and you can stream through the keys and values in arbitrary block order, never materializing the score matrix anywhere except inside the SRAM tile currently being processed. When you have walked every key-value tile, divide the final $O$ by the final $\ell$ and you have *exact* attention output, to floating-point precision, byte-for-byte identical to the naive computation.

{{% callout type="theorem" %}}**The online softmax is exact.** Despite the streaming, no approximation is made. The rescaling factor $e^{m_\text{old} - m_\text{new}}$ exactly converts a partial logsumexp computed under one max into a partial logsumexp under another. FlashAttention is *not* an approximation to attention. It is *attention*, computed in a different order with a different memory access pattern.{{% /callout %}}

That theorem is the legality clause for the entire kernel. Everything that follows is just the orchestration.

## The Loop Skeleton

Once online softmax is in your toolbox, the actual algorithm is two nested loops over tiles. Here is the FA-2 skeleton written as Python-pseudocode, deliberately stripped of the GPU-specific stuff for clarity:

```python
# Q is (T, d), K is (T, d), V is (T, d). T can be huge; SRAM cannot hold the whole thing.
# Block sizes Bq for queries, Bk for keys. Chosen so that 4 * Bq * Bk * 2 bytes fits in SRAM.

for q_tile in range(0, T, Bq):                       # outer loop: rows of Q
    Q_tile = load_from_hbm(Q, q_tile, Bq)            # one slab of queries -> SRAM
    O_tile = zeros((Bq, d))                          # running output
    m      = full((Bq,), -inf)                       # running max (per query row)
    ell    = zeros((Bq,))                            # running normalizer (per query row)

    for k_tile in range(0, T, Bk):                   # inner loop: rows of K and V
        K_tile = load_from_hbm(K, k_tile, Bk)        # -> SRAM
        V_tile = load_from_hbm(V, k_tile, Bk)        # -> SRAM

        # Compute partial scores INSIDE SRAM, never write to HBM.
        S_tile = (Q_tile @ K_tile.T) / sqrt(d)       # (Bq, Bk), lives in SRAM
        m_b    = S_tile.max(axis=1)                  # (Bq,)
        P_tile = exp(S_tile - m_b[:, None])          # softmax numerator, SRAM
        ell_b  = P_tile.sum(axis=1)                  # (Bq,)

        # Online softmax merge.
        m_new   = maximum(m, m_b)
        alpha   = exp(m   - m_new)                   # correction for old accumulator
        beta    = exp(m_b - m_new)                   # correction for new block
        ell     = alpha * ell + beta * ell_b
        O_tile  = (alpha[:, None] * O_tile
                   + beta[:, None] * (P_tile @ V_tile))
        m       = m_new

    O_tile = O_tile / ell[:, None]                   # final normalisation
    write_to_hbm(O, q_tile, O_tile)                  # one write per Q tile
```

The whole kernel is forty lines of arithmetic. Read it once with the [memory hierarchy](../03-memory-hierarchy/) in mind:

- **Outer loop** parks one tile of $Q$ in SRAM. Loaded from HBM exactly once per outer iteration.
- **Inner loop** streams $K$ and $V$ tiles through SRAM. Each loaded once per outer iteration.
- The score tile $S_\text{tile}$ is $B_q \times B_k$, lives entirely in SRAM, and **never gets written to HBM**.
- $O$ is written *once* at the end of each outer iteration — never as an intermediate.

Three loads, one store, per outer iteration. HBM traffic for the full kernel: $O(T \cdot d)$, not $O(T^2)$. That is the entire story.

```pyplot {id="flash-tiling" caption="FlashAttention's tile structure. The outer loop walks Q-tiles down the page. For each Q-tile, the inner loop sweeps every KV-tile (pink trail). Every yellow cell in the implicit T x T score matrix is touched in SRAM but never written to HBM. PagedAttention modifies the inner loop's indexing into a block-table lookup."}
import numpy as np
import matplotlib.pyplot as plt

T = 12
Bq, Bk = 3, 3
n_q, n_k = T // Bq, T // Bk

fig, ax = plt.subplots(figsize=(7.5, 6))

# The implicit T x T score matrix
for i in range(T):
    for j in range(T):
        ax.add_patch(plt.Rectangle((j, T-1-i), 1, 1,
                                     facecolor='#FFD700', alpha=0.20,
                                     edgecolor='#1A1A1A', linewidth=0.3))

# Tile borders
for qi in range(n_q + 1):
    ax.axhline(qi * Bq, color='#1A1A1A', linewidth=1.6)
for ki in range(n_k + 1):
    ax.axvline(ki * Bk, color='#1A1A1A', linewidth=1.6)

# Outer loop: a specific Q-tile highlighted (say, the middle one)
qi = 1
for j in range(T):
    for i in range(qi * Bq, (qi + 1) * Bq):
        ax.add_patch(plt.Rectangle((j, T-1-i), 1, 1,
                                     facecolor='#FF007F', alpha=0.45,
                                     edgecolor='#1A1A1A', linewidth=0.3))

# Highlight one KV tile being processed inside the inner loop
ki = 2
for i in range(qi * Bq, (qi + 1) * Bq):
    for j in range(ki * Bk, (ki + 1) * Bk):
        ax.add_patch(plt.Rectangle((j, T-1-i), 1, 1,
                                     facecolor='#00A8A8', alpha=0.85,
                                     edgecolor='#1A1A1A', linewidth=0.6))

# Arrows for inner-loop sweep
for ki in range(n_k - 1):
    x = (ki + 0.5) * Bk
    y = T - (qi + 0.5) * Bq
    dx = Bk
    ax.annotate("", xy=(x + dx, y), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='#FF007F',
                                lw=1.8, alpha=0.7))

ax.set_xlim(0, T); ax.set_ylim(0, T)
ax.set_aspect('equal')
ax.set_xlabel("keys (T)")
ax.set_ylabel("queries (T)")
ax.set_title("FlashAttention tiling: outer Q-tile (pink row) sweeps KV-tiles (teal cell)")
ax.set_xticks([]); ax.set_yticks([])

# Legend
from matplotlib.patches import Patch
legend = [
    Patch(facecolor='#FFD700', alpha=0.4, edgecolor='#1A1A1A',
          label='implicit score matrix (in SRAM only)'),
    Patch(facecolor='#FF007F', alpha=0.45, edgecolor='#1A1A1A',
          label='active Q tile this outer iter'),
    Patch(facecolor='#00A8A8', alpha=0.85, edgecolor='#1A1A1A',
          label='active KV tile this inner iter'),
]
ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=3, framealpha=1, edgecolor='#1A1A1A', fontsize=8.5)
plt.tight_layout()
```

Two things to note about the picture for what comes next in the issue.

First, the **outer loop indexes contiguous rows of $Q$**. Queries are dense and live in a flat tensor — easy.

Second, the **inner loop indexes contiguous rows of $K$ and $V$**. This is true for FlashAttention as written. It is precisely this assumption that **PagedAttention** ([→ ch.10](../10-paged-attention/)) breaks. When the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} is paged, the rows of $K$ and $V$ for a single sequence are scattered across non-contiguous physical pages. The kernel's inner loop has to walk a *block table* — a level of indirection — to find each KV tile. Everything else stays identical: same tiling, same online softmax, same accumulation. *The entire PagedAttention contribution, kernel-wise, is changing how the inner loop computes its load addresses.*

## Three Generations Of The Same Idea

The original FlashAttention (2022) was an **A100** kernel. It worked, it was 2–4× faster than naive on long context, and within six months every serious inference engine had integrated it. Then Tri Dao did something the rest of the field rarely manages: he kept improving the same kernel, against newer hardware, for three more silicon generations. Each version is, mechanically, a different way to spend SRAM and tensor cores. The math underneath does not change.

**FlashAttention-2** ({{< cite text="Dao, 2023" url="https://arxiv.org/abs/2307.08691" kind="paper" >}}): the main contribution is swapping the outer/inner loops to put queries on the outside. In the original, the outer loop was over $K$, which forced *all* warps to cooperatively rescale the running statistics every inner iteration — a serialization bottleneck. Putting $Q$ outside lets each warp own a row of $Q$ independently and rescale on its own. The result: **~2× faster** than FA-1 on A100, and a much cleaner story for occupancy on Hopper.

**FlashAttention-3** ({{< cite text="Shah et al., 2024" url="https://arxiv.org/abs/2407.08608" kind="paper" >}}): the Hopper-native version. Uses the **TMA** (Tensor Memory Accelerator) to issue async global-memory loads, **WGMMA** instructions to drive the warp-group matrix engine, and **warp specialization** — different warps in the same block specialize as "producer" (loaders) and "consumer" (compute) — to overlap data movement with computation. Adds an **FP8** path that nearly doubles compute throughput on H100. The kernel still computes the exact same attention output; it just keeps the H100 silicon busy in a way FA-2 could not.

**FlashAttention-4** (2025, Blackwell era): uses **block-FP4** tensor cores, the new `tcgen05` instructions that fuse load + compute + accumulate into a single warp-group operation, and async chained TMAs to keep the pipeline full at much shorter tile sizes. Each generation widens the gap between fused attention and any plausible naive implementation, often by 2–3× per generation. By 2025, no production inference stack runs anything but FlashAttention or a paged descendant of it.

The pattern across all four generations is the same: **the algorithm is the same; the silicon-aware orchestration changes**. This is why the question "what is the IO complexity of attention?" became a *recurring* research question rather than a one-shot answer. Every new tensor core architecture invites a new kernel.

{{% pullquote type="profound" %}}FlashAttention did not change the FLOPs of attention. It changed the *order* of memory accesses. The lesson, two years later, is that on modern hardware **what order you do the math in matters more than how much math you do**.{{% /pullquote %}}

## Foreshadowing The Decode Special Case

One last piece, because the rest of the issue depends on it.

The tiling story above is *prefill*: many queries, many keys, both sides tiled. In **decode**, you have **exactly one query** (the new token) and **many keys** (the full {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} so far). The outer Q-loop has only one iteration.

That degenerate Q-loop is what makes decode attention sit at $I \approx 1$ in the [roofline](../04-roofline/). You sweep the entire $K, V$ cache through SRAM, do one row's worth of FLOPs, and emit one output vector. The bytes are huge; the math is one query. Flash still helps — you don't write the (now $1 \times T$) score row to HBM — but the underlying intensity is what it is. Decode is bandwidth-bound by physics, not by inefficiency.

This is also the reason **PagedAttention** is a *decode-time* optimization more than a prefill one. In prefill the KV cache is freshly computed and contiguous in memory; you don't need a block table. In decode you've accumulated KVs over hundreds of steps, possibly belonging to many concurrent users sharing memory pages — and that's where the block-table indirection earns its keep.

Hold that thought. We will return to it when we [tear apart the KV cache's allocator](../09-kv-fragmentation/) and discover that the operating-systems community already solved this problem in 1965.

## What To Remember

1. **The score matrix never has to land in HBM.** That single trick — online softmax over SRAM tiles — is the entire FlashAttention contribution.
2. **HBM traffic drops from $O(T^2)$ to $O(T)$.** Naive attention is unworkable at long contexts not because of FLOPs but because of bytes.
3. **Every modern attention variant** — paged, sliding-window, MLA, NSA, sparse — is a wrinkle on the same tiled inner loop. Master Flash and you've mastered 90% of attention engineering.

**Continue to → [Two Phases, Two Personalities](../07-prefill-vs-decode/)** — the GPU primers are done. Now the engine takes over, and the first thing it has to confront is that prefill and decode are *not the same kernel.*
