---
title: "V4 Cache Engineering: three pools, one allocator"
short_title: "V4 Cache Engineering"
description: "V4's three attention modes produce KV entries of three different sizes and lifetimes — the vLLM team unified them using a 256-token logical block divisible by both compression strides, with three separate physical pools behind one address space."
blurb:
  - "Raw sliding-window entries: 1 KV pair per token. CSA compressed entries: 1 pair per 4 tokens. HCA compressed entries: 1 pair per 128 tokens."
  - "256 logical positions per block: divisible by 4 (64 CSA entries per block) and by 128 (2 HCA entries per block)."
  - "The allocator never learns that pages have different physical weights — it hands out one block from each pool simultaneously."
  - "Three kernel fusions: the vLLM April 2026 engineering blog documents exactly how compressor, indexer, and attention fuse into single CUDA passes."
topics: [attention, cache, systems, primer, vllm]
tags: [kv-cache, heterogeneous-cache, vllm, kernel-fusion, paged-attention, multi-stream, csa, hca]
theme: cream
math: true
draft: false
date: 2026-05-16T12:20:00-04:00
issue: 7
weight: 190
techKind: primer
techNode: cache-engineering
header: 19-cache-engineering.webp
---

## Three Caches, One Budget

Picture the GPU memory manager's problem the moment it boots a V4 inference server.

It has to allocate space for the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} — the growing buffer of past-token projections that lets the model skip re-computing attention over everything it has already read. In prior models (GPT, Llama, Mistral, even DeepSeek V2 with its MLA), this was a solved problem. One format, one size per slot, one pool. Paged attention could chop the buffer into equal-sized blocks and hand them out like pages from an OS virtual memory allocator.

V4 breaks that assumption. At the layer level, V4 has three fundamentally different attention modes, each producing cache entries of a different type and a different size:

| Attention mode | Chapter | Cache entry | Covers how many raw tokens? |
|---|---|---|---|
| Sliding-window branch (CSA/HCA) | — | 1 raw KV pair | 1 |
| {{< wiki "attention" >}}CSA{{< /wiki >}} compressed entry | [Ch. 7](../07-csa/) | 1 compressed KV pair | 4 (stride $m=4$) |
| HCA compressed entry | [Ch. 8](../08-hca/) | 1 compressed KV pair | 128 (stride $m'=128$) |

Three different sizes. Three different update rates. Three different lifetimes — the sliding-window buffer is evicted as the window slides; the compressed entries persist for the full context. A single uniform pool cannot serve all three without either wasting memory (padding the small entries up to the size of the large ones) or breaking the block-allocator abstraction.

This is the heterogeneity problem. The vLLM team's April 2026 engineering blog describes exactly how they solved it.

{{% callout type="tangent" title="Source" %}}
This article draws heavily on the vLLM Engineering Blog post **"Serving DeepSeek V4 Efficiently"** (April 2026) by the vLLM team at UC Berkeley / LMSys, building on DeepSeek-AI's V4 Technical Report (2026). Architecture details are attributed to DeepSeek-AI; the engineering solutions — pool design, kernel fusions, multi-stream parallelism — are the vLLM team's work.
{{% /callout %}}

## The Unified Block Design

The insight is to pick a **logical block size** that is divisible by both compression strides, then run three separate physical pools — each pool using a different physical page size for its entries, but all presenting the same number of logical positions per block to the allocator.

The vLLM team chose **256 native token positions** as the logical block unit.

- 256 is divisible by 4. So one logical block of 256 raw-token positions corresponds to exactly $256 / 4 = 64$ CSA compressed entries.
- 256 is divisible by 128. So the same logical block corresponds to exactly $256 / 128 = 2$ HCA compressed entries.

This gives three pools:

| Pool | Slots per block | What each slot holds | Memory per slot |
|---|---|---|---|
| **Raw** (sliding window) | 256 | 1 raw KV pair | $2 \times d_{kv}$ scalars |
| **Medium** (CSA, c4a) | 64 | 1 compressed entry covering 4 raw tokens | $2 \times d_{kv}$ scalars |
| **Small** (HCA, c128a) | 2 | 1 compressed entry covering 128 raw tokens | $2 \times d_{kv}$ scalars |

The **allocator sees only logical blocks.** When a sequence needs one more logical block of context, the allocator hands it one block from each pool simultaneously — one 256-slot raw page, one 64-slot medium page, one 2-slot small page. All three are "block N of sequence S." The physical bytes differ; the logical identity is the same.

{{% pullquote type="technical" %}}
Choose a block size divisible by every compression stride. Three pools, one logical address space. The allocator never learns that pages have different weights.
{{% /pullquote %}}

The elegance is that **paged attention, as a concept, survives unchanged.** Every existing mechanism — block tables, free-list allocators, KV cache sharing across beam hypotheses — works without modification. Only the physical backing store is split into three pools instead of one.

{{< crosshead >}}Napkin math: how much memory does each pool type use?{{< /crosshead >}}

Suppose $d_{kv} = 512$ (head dim × number of KV heads, representative for a large model), FP8 format (1 byte/scalar), and a context window of 128k tokens — so $128{,}000 / 256 = 500$ logical blocks.

- **Raw pool:** $500 \times 256 \times 512 \times 2 = 131{,}072{,}000$ bytes ≈ 125 MB
- **Medium pool (c4a):** $500 \times 64 \times 512 \times 2 = 32{,}768{,}000$ bytes ≈ 31 MB
- **Small pool (c128a):** $500 \times 2 \times 512 \times 2 = 1{,}024{,}000$ bytes ≈ 1 MB

The raw sliding-window pool dominates by a factor of 4×. The HCA pool is negligible. This is exactly what you would expect: high compression ratio means you store far fewer entries, even if each entry is the same size as an uncompressed pair.

## Compressor State

The three pools handle *storing* compressed entries. But how do you *produce* them?

Each compressor — CSA's stride-4 compressor and HCA's stride-128 compressor — is a small learned neural network that takes a window of raw hidden states and emits one compressed KV pair. The window is the **compressor state**: a sliding buffer of the most recent raw tokens not yet folded into a compressed entry.

- **CSA compressor state:** 4 raw token positions. The compressor fires every 4 tokens, producing one c4a entry and clearing the state.
- **HCA compressor state:** 128 raw token positions. The compressor fires every 128 tokens, producing one c128a entry and clearing the state.

### Prefill vs. Decode

During **prefill**, the model processes the entire prompt in one forward pass. The compressor state fills rapidly. Every 4 tokens, the CSA compressor fires and deposits a c4a entry into the medium pool. Every 128 tokens, the HCA compressor fires and deposits a c128a entry into the small pool. If the prompt is 4,096 tokens long, prefill produces $4096 / 4 = 1024$ c4a entries and $4096 / 128 = 32$ c128a entries — in addition to the rolling sliding-window buffer in the raw pool.

During **decode**, only one new raw token arrives per step. The compressor state grows by one slot per step. It is *not yet full*, so no compressed entry is produced — until step 4 (for CSA) or step 128 (for HCA). The asymmetry matters: in heavy decode workloads, HCA compression happens so infrequently (once every 128 steps) that the medium and small pools are nearly static, and almost all cache traffic goes through the raw pool.

The vLLM implementation tracks the state buffer separately from the three pools, treating it as a small fixed-size tensor associated with each running sequence — not as paged memory, since its lifetime is sub-block and it never needs to survive eviction.

## Python: The Heterogeneous Cache

The structure above maps cleanly to code. Below is a stripped-to-the-bone illustration of the three-pool cache, with `append()` for writing new raw tokens and `lookup()` for retrieving entries by logical position.

```python
class HeterogeneousKVCache:
    """Manages three pools: raw (sliding window), c4a (CSA), c128a (HCA)."""

    def __init__(self, context_len, d_kv, dtype=np.float8):
        block_size = 256
        n_blocks = context_len // block_size

        self.raw_pool   = np.zeros((n_blocks, block_size,       d_kv), dtype=dtype)
        self.c4a_pool   = np.zeros((n_blocks, block_size // 4,  d_kv), dtype=dtype)
        self.c128a_pool = np.zeros((n_blocks, block_size // 128, d_kv), dtype=dtype)

        self.csa_state  = np.zeros((4,   d_kv))
        self.hca_state  = np.zeros((128, d_kv))

        self._raw_cursor  = 0
        self._csa_cursor  = 0
        self._hca_cursor  = 0

    def append(self, raw_kv, compress_csa, compress_hca):
        """
        raw_kv:       shape (d_kv,) — the new token's raw KV pair
        compress_csa: callable (state) → (d_kv,) compressed entry
        compress_hca: callable (state) → (d_kv,) compressed entry
        """
        pos = self._raw_cursor
        block, slot = divmod(pos, 256)
        self.raw_pool[block, slot] = raw_kv

        self.csa_state[pos % 4] = raw_kv
        self.hca_state[pos % 128] = raw_kv

        if (pos + 1) % 4 == 0:
            c4a_entry = compress_csa(self.csa_state)
            c4a_pos = pos // 4
            c4a_block, c4a_slot = divmod(c4a_pos, 64)
            self.c4a_pool[c4a_block, c4a_slot] = c4a_entry
            self._csa_cursor += 1

        if (pos + 1) % 128 == 0:
            c128a_entry = compress_hca(self.hca_state)
            c128a_pos = pos // 128
            c128a_block, c128a_slot = divmod(c128a_pos, 2)
            self.c128a_pool[c128a_block, c128a_slot] = c128a_entry
            self._hca_cursor += 1

        self._raw_cursor += 1

    def lookup_raw(self, token_pos):
        block, slot = divmod(token_pos, 256)
        return self.raw_pool[block, slot]

    def lookup_c4a(self, compressed_pos):
        block, slot = divmod(compressed_pos, 64)
        return self.c4a_pool[block, slot]

    def lookup_c128a(self, compressed_pos):
        block, slot = divmod(compressed_pos, 2)
        return self.c128a_pool[block, slot]
```

A few things to notice.

The three pools are independent arrays, but they all index off the same logical position `pos`. The divisor changes — `256`, `64`, `2` — but the block-and-slot arithmetic is identical in form. This is why the single block size of 256 was chosen: the math simplifies to clean integer divisions everywhere.

The compressor state buffers (`csa_state`, `hca_state`) are ring buffers of size 4 and 128 respectively. They are separate from the pools because they hold *partial* blocks — raw tokens that have been received but not yet compressed. They live and die with the sequence; they are not paged.

The `compress_csa` and `compress_hca` callables abstract over the actual learned compressor networks. In production they are fused CUDA kernels; here they are just functions.

## Kernel Fusions

Producing and storing compressed entries involves four sequential operations on each raw hidden state before it can be inserted into the cache: the compressor network itself, an RMSNorm over its output, RoPE rotation of the resulting keys, and the cache write. Each of these, run as a separate CUDA kernel, incurs kernel-launch overhead and a round-trip through GPU memory — the intermediate tensor written by one kernel must be read back by the next.

The vLLM team identified three high-value fusion points and measured the speedups empirically.

{{< crosshead >}}Fusion 1: Compressor + RMSNorm + RoPE + KV Insert — 1.4–3× speedup{{< /crosshead >}}

The compressor output must be normalised (RMSNorm) and position-encoded ({{< wiki "rope" >}}RoPE{{< /wiki >}}) before it can be written into the medium or small pool. Fusing all four steps into a single kernel keeps the intermediate activations in registers, never writing them to HBM. The speedup (1.4–3×) varies with batch size: larger batches amortise launch overhead better, so the relative gain from launch-overhead elimination shrinks, but the memory-bandwidth saving persists.

{{< crosshead >}}Fusion 2: Inverse RoPE + FP8 Quantize — 2–3× speedup{{< /crosshead >}}

The compressed KV entries are stored in FP8 to save memory (see [Mixed-Precision KV Cache](../15-mixed-precision-kv/)). But RoPE encodes absolute position, which is incompatible with a cache that will be reused at different future positions. The solution is to *un-rotate* the keys before storing them — apply the inverse RoPE transform — and then quantise to FP8. Without fusion, this is two kernels: one to correct position, one to quantise. Fused, the position-corrected floating-point intermediate never lands in HBM; it goes straight from the correction arithmetic into the FP8 packing logic in the same register file. 2–3× speedup.

{{< crosshead >}}Fusion 3: Q-norm + KV RoPE + K Insert (prefill) — 10–20× speedup{{< /crosshead >}}

This is the dramatic one. During prefill, query normalisation, key/value RoPE rotation, and the cache write all happen to the same set of Q/K/V tensors in tight succession. Pre-fusion, these are three kernels; fused, they are one. The 10–20× figure reflects that prefill is heavily memory-bound — the Q, K, V tensors are large, and eliminating two full round-trips through HBM halves or better the total bytes moved. The gain is largest at long prompt lengths, exactly the regime where V4 is most likely to be deployed.

{{% callout type="tip" title="Fusion rule of thumb" %}}
Fuse whenever the output of one operation is consumed *only* by the next operation in the same sequence. Every unfused edge is a write + read of the full intermediate tensor. On a GPU with 3.35 TB/s HBM bandwidth, even a 1 GB intermediate costs ~300 µs per kernel boundary.
{{% /callout %}}

## Multi-Stream Parallelism

Even with all three fusions applied, there is a structural inefficiency in the naive sequential execution order.

After the raw KV pairs have been written to the raw pool, the query computation (computing attention scores over the sliding-window and selected compressed entries) can begin *immediately*. It does not need to wait for the compressor to finish running on those same raw KVs — the compressor's output goes into the medium and small pools, which feed *future* decode steps, not the *current* step's attention.

In other words: **the compressor stream and the attention stream are data-independent once the raw KV write is committed.**

vLLM exploits this with two CUDA streams. The compressor (Fusion 1 and Fusion 2, above) runs on stream A. The attention computation runs on stream B. Both streams are dispatched to the GPU simultaneously, and the CUDA scheduler overlaps them on the available SM partitions.

The measured latency reduction is **5–6%** at typical decode batch sizes. That sounds modest, but consider: every single decode step, for every sequence in the batch, benefits from this overlap. At high throughput — hundreds of sequences, thousands of steps — 5–6% compounds into a meaningful reduction in time-to-first-token and in sustained tokens-per-second throughput.

```pyplot {id="multistream-overlap" caption="SINGLE DECODE STEP: SEQUENTIAL VS. MULTI-STREAM. THE 5–6% SAVING COMES FROM OVERLAPPING THE COMPRESSOR AND ATTENTION STREAMS."}
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

STAGE_COLORS = {
    'Raw KV write': '#FFD700',
    'Compressor (Fusion 1+2)': '#FF007F',
    'Attention (Fusion 3 + attn compute)': '#00A8A8',
}

sequential_stages = [
    ('Raw KV write',                    0.0,  0.8),
    ('Compressor (Fusion 1+2)',          0.8,  2.2),
    ('Attention (Fusion 3 + attn compute)', 2.2, 5.0),
]

parallel_stages = [
    ('Raw KV write',                    0.0,  0.8),
    ('Compressor (Fusion 1+2)',          0.8,  2.2),
    ('Attention (Fusion 3 + attn compute)', 0.8, 3.65),
]

for ax, stages, label in zip(axes,
                              [sequential_stages, parallel_stages],
                              ['Sequential (single stream)', 'Multi-stream (overlap)']):
    for stage, start, end in stages:
        ax.barh(0, end - start, left=start, height=0.5,
                color=STAGE_COLORS[stage], edgecolor='#1A1A1A', linewidth=0.8)
        mid = (start + end) / 2
        ax.text(mid, 0, stage, ha='center', va='center',
                fontsize=8, color='#1A1A1A', fontweight='bold')
    total = max(end for _, _, end in stages)
    ax.axvline(total, color='#FF8C00', linewidth=1.5, linestyle='--')
    ax.text(total + 0.05, 0.3, f'{total:.2f} ms', color='#FF8C00', fontsize=9)
    ax.set_yticks([])
    ax.set_title(label, loc='left', fontsize=10, fontweight='bold')
    ax.spines[['top', 'right', 'left']].set_visible(False)

axes[-1].set_xlabel('time (ms, illustrative)')
saving_pct = (5.0 - 3.65) / 5.0 * 100
axes[0].set_xlim(0, 5.5)

print(f'Sequential total: 5.00 ms')
print(f'Multi-stream total: 3.65 ms  (illustrative; actual saving ~5-6%)')
print(f'Illustrated saving: {saving_pct:.1f}%')

plt.tight_layout()
```

The plot shows one decode step. The raw KV write must finish first — both streams depend on it. After that, the compressor and the attention computation diverge onto separate streams and run in parallel. The step completes when the longer of the two finishes (attention, in the typical case). The compressor work, which previously sat on the critical path, now lives *off* the critical path.

{{% marginnote %}}
At batch size 1 the overlap saving is smaller because the GPU is underutilised even without overlap — there are not enough concurrent warps to keep both streams busy. At batch size 32+ the saving is consistently 5–6%.
{{% /marginnote %}}

## Putting It Together: One Decode Step, End to End

Here is the full picture of a single V4 decode step under the vLLM implementation, with all the pieces connected:

1. **New token arrives.** Its hidden state is projected to raw Q, K, V.
2. **Raw KV write (Fusion 3, partial).** K and V are RoPE-rotated and inserted into the raw pool. This commits the raw KV — both streams can now start.
3. **Stream A: Compressor.** The new raw hidden state is appended to `csa_state` and `hca_state`. If `csa_state` is full (every 4 steps), Fusion 1 fires: compress → RMSNorm → RoPE → write c4a entry. Fusion 2 immediately applies inverse RoPE → FP8 quantise → final c4a store. Identically for HCA every 128 steps.
4. **Stream B: Attention.** Fusion 3 (Q-norm + KV RoPE + K insert, for the query side) fires. The attention score computation runs over: the sliding-window raw entries from the raw pool, the top-k selected c4a entries from the medium pool (for CSA layers), and all c128a entries from the small pool (for HCA layers). Output is the weighted value sum.
5. **Synchronise.** Both streams join at a CUDA event before the output projection layer.
6. **Output projection and residual.** Standard MLP + residual stream update.

Steps 3 and 4 overlap in wall-clock time. Everything else is sequential.

## Connections

- **CSA chapter** — The compressor that produces c4a entries, and the top-k selection over the medium pool: [Compressed Sparse Attention (Ch. 7)](../07-csa/)
- **HCA chapter** — The compressor that produces c128a entries, and dense attention over the small pool: [Heavily Compressed Attention (Ch. 8)](../08-hca/)
- **Mixed-Precision KV Cache** — Why Fusion 2 must quantise to FP8 *after* the inverse RoPE step, and the broader three-precision scheme V4 uses: [Mixed-Precision KV Cache (Ch. 15)](../15-mixed-precision-kv/)
- **Paged attention** — The original vLLM paper (Kwon et al., 2023) introduced the block allocator abstraction that the three-pool design extends.

**Continue to** → [Empirical Sparsity (Ch. 16)](../16-empirical-sparsity/), where we ask: does V4's sparse attention actually behave sparsely on real data, or is the sparsity pattern so dense that the indexer is bottlenecked anyway?
