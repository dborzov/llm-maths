---
title: "KV Fragmentation: 62% of HBM holding nothing"
short_title: "KV Fragmentation"
description: "Woosuk Kwon's 2023 profiling found 38% of HBM actively holding live KV data in a 'tuned' serving setup — the other 62% was reserved padding the allocator refused to release, a structural flaw in every serving stack of the era."
blurb:
  - "Llama-13B with a 32-sequence batch at max 4,096 tokens requires 107 GB of KV reservation — more than the H100's 80 GB."
  - "Production stacks of 2022–23 achieved 20–40% useful KV utilization; 60–80 cents of every HBM dollar bought nothing."
  - "Three allocator strategies — pre-allocate max, realloc-and-copy, dynamic fragments — each fail in a different way."
  - "Knuth named both failure modes in 1968: internal fragmentation and external fragmentation, now happening simultaneously."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T13:00:00-04:00
issue: 8
weight: 90
techKind: mainline
techNode: kv-fragmentation
header: 09-kv-fragmentation.webp
---

## Berkeley, March 2023

It is a Friday afternoon in March 2023, and a PhD student named **Woosuk Kwon** is staring at `nvidia-smi` on a lab machine and writing the same number on a whiteboard, three times, with increasing disbelief.

`38%.`

That is the fraction of his 80 GB of H100 HBM that is **actually** holding live keys and values in the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}, in a working set he configured to be "as efficient as possible" — a batch of 32 sequences running Llama-13B chat, max length 4,096 tokens, served through a hand-tuned HuggingFace Transformers loop. The remaining 62%? Reserved but empty. Padding. Quiet GBs of HBM holding nothing, draining wall power, refusing to host anything else because the allocator promised them to a sequence that, statistically, will never use them.

Kwon goes back to his desk and reads the entire HuggingFace generation pipeline again, looking for the bug, because nobody could possibly ship a system that wastes 62% of an $80,000 GPU's working memory by *design*.

There is no bug. **That is the design.**

By the end of the year, Kwon — together with Zhuohan Li, Siyuan Zhuang, and Ying Sheng, all advised by Joey Gonzalez and Ion Stoica — will publish the SOSP 2023 PagedAttention paper {{< cite text="Kwon et al., 2023 (vLLM / PagedAttention)" url="https://arxiv.org/abs/2309.06180" kind="paper" >}}, write the open-source library called **vLLM**, and turn that 38% into something north of 96%. The fix is one of the cleanest borrowings in modern systems work: a 1965 operating-system trick, applied to a 2023 attention kernel.

But before we can appreciate the trick, we have to live inside the disease. This chapter is about the disease.

## The Anchor Case

Pin down the exact scenario, because the numbers do the arguing.

- Model: **Llama-13B**, 40 layers, 40 heads, head dim 128, FP16. Per-token KV size = $40 \cdot 2 \cdot (40 \cdot 128) \cdot 2 \text{ bytes} = 819{,}200$ bytes per token. Call it **0.8 MB per token**.
- Batch: **32 concurrent sequences**.
- Configured max length: **4,096 tokens per sequence**.
- Reality: actual sequence lengths range from **100 to 3,800 tokens**, with a mean around 700.

A standard "pre-allocate to max" KV allocator reserves $(32 \times 4096 \times 0.8\,\text{MB}) = $ **107 GB** of HBM up front. The H100 has 80 GB. **You cannot even hold one full batch.** In practice the system is forced to cap the batch at 24 active slots — fine — but the average sequence is using $700 / 4096 = 17\%$ of its reserved slot. *Most* of the HBM is reserved padding that the allocator refuses to let other sequences touch, because tomorrow, *theoretically*, one of those sequences might still grow into it.

The PagedAttention paper measured production serving stacks of the day — HuggingFace Transformers, FasterTransformer, Triton with TRT-LLM — and reported KV memory utilizations of **20% to 40%**. Twenty to forty cents of every memory dollar bought a useful key. Sixty to eighty cents bought nothing.

## The Allocator's Three Bad Choices

The reason every 2023 serving stack converged on the same 38% number is that the allocator only has three options, and each one is wrong in a different way.

**Choice 1: Pre-allocate max-length per slot.** This is what HuggingFace Transformers did by default. Allocate `(batch_size, max_len, layers, heads, head_dim)` as one contiguous tensor at the start. The CUDA kernels are happy — they love contiguous memory — and the math is dead simple: `kv[seq_idx, token_idx, ...]`. The cost is the worst-case waste: every byte you do not use is locked. This is the slot-pre-allocation regime, and it gives you 17–40% utilization on chat traffic.

**Choice 2: Pre-allocate current length, grow with realloc.** Start each sequence with a small KV buffer; when it fills, allocate a bigger one, **copy the old one in**, free the old one. This is the C++ `std::vector` strategy. It looks fine on paper. It dies on the GPU for two reasons: (a) the copy is hot — at 0.8 MB/token, 1024 existing tokens take ~800 MB and a copy event lasts milliseconds, eating most of a decode step; (b) when you reallocate, the new tensor must be contiguous, which interacts badly with the fact that other sequences are also holding contiguous tensors, which produces *external* fragmentation in the HBM arena. Result: 30–50% utilization and unpredictable latency spikes from realloc storms.

**Choice 3: Dynamic allocation in non-contiguous fragments.** Just hand out KV buffers in whatever-sized chunks fit, with a heap allocator. This solves the utilization problem in theory. It breaks the attention kernel in practice: the FlashAttention inner loop ([→ ch.6](../06-flash-attention/)) walks K and V at sequential strides. If a sequence's K and V live at non-contiguous addresses, the kernel either has to assemble a gather list — destroying memory coalescing — or take a slowdown that makes the whole exercise pointless.

{{< crosshead >}}This Is The Knuth Catalog{{< /crosshead >}}

The systems-academic name for what is going on here was settled in Knuth's *The Art of Computer Programming*, Volume 1, §2.5, in **1968**. There are two kinds of memory fragmentation, and the KV cache suffers from both simultaneously.

- **Internal fragmentation.** Memory that is allocated but not in use. Choice 1 is pure internal fragmentation: every sequence holds 4,096 slots, uses 700 on average, the other 3,396 are internal-fragmented padding.

- **External fragmentation.** Memory that is free but not in a contiguous block large enough to satisfy a new request. Choice 2 and Choice 3 generate external fragmentation: after sequences come and go, the heap arena looks like Swiss cheese; you cannot fit a fresh contiguous 1 MB tensor even though there are 10 MB free.

{{% marginnote %}}
Knuth's analysis of these two fragmentations was the headline result of memory-management research in the 1960s. The conclusion the OS community drew — and the one the LLM serving community would re-derive sixty years later — is that you cannot eliminate both with a single contiguous allocator. **You have to introduce a level of indirection.** That is exactly the trick we will meet in [Borrowing from 1965](../10-paged-attention/).
{{% /marginnote %}}

The KV cache is a *uniquely cursed* allocation problem. It has variable-length objects (sequences are 50 to 50,000 tokens), unpredictable lifetimes (the model decides when to emit `<|eot|>` and that may be in 12 tokens or 480), massive sizes (we will see 335 GB single-context Llama-65B in a moment), and a *hot* access pattern (attention reads every KV in the sequence every decode step — there is no cold storage you can demote to slower memory). It is a textbook adversarial workload for a memory allocator, designed by nobody, that emerged organically out of the autoregressive generation loop.

## A Picture Of The Sag

Let us watch the disease in real time. Simulate 32 slots over 50 generation steps. Each slot holds either live KV (yellow), reserved-but-empty padding (orange), or — after the sequence has finished and its slot is freed — a hole (ink). Watch the yellow shrink.

```pyplot {id="kv-utilization-sag" caption="32-slot KV arena. Yellow = live KV (useful). Orange = reserved padding (waste). Ink = freed slot (external fragmentation). The utilization curve sags toward ~25% as the long-tail sequences hold their pre-allocated max forever."}
np.random.seed(11)

n_slots  = 32
max_len  = 4096
n_steps  = 80
step_tok = 60       # tokens per simulation step

# Simulate slot replacement under pre-allocate-max: each slot is reserved at full
# max_len, while the actual sequence in it grows token-by-token then exits and is
# replaced. A heavier-tail lognormal so some sequences live near max_len.
arena = np.zeros((n_slots, n_steps), dtype=int)   # 0=live, 2=freed-then-replaced
live_tokens_per_step = np.zeros(n_steps)

for k in range(n_slots):
    start = 0
    while start < n_steps:
        # Realistic chat distribution: mean ~1500 tokens, fat tail up to max_len
        total = int(np.clip(np.random.lognormal(mean=6.8, sigma=0.9), 120, max_len))
        n_steps_in_seq = int(np.ceil(total / step_tok))
        end = min(start + n_steps_in_seq, n_steps)
        for s in range(start, end):
            elapsed = (s - start + 1) * step_tok
            live = min(elapsed, total)
            arena[k, s] = 0           # alive
            live_tokens_per_step[s] += live
        if end < n_steps:
            arena[k, end] = 2
        start = end + 1

util = live_tokens_per_step / (n_slots * max_len)

# Render the arena image
img = np.zeros((n_slots, n_steps, 3))
yellow = np.array([1.0, 0.84, 0.0])
orange = np.array([1.0, 0.55, 0.0])
ink    = np.array([0.1, 0.1, 0.1])
# Per-cell live-fraction for shading
for s in range(n_steps):
    for k in range(n_slots):
        if arena[k, s] == 2:
            img[k, s] = ink
        else:
            # blend by per-slot live fraction
            slot_live = live_tokens_per_step[s] / n_slots / max_len  # avg
            t = min(1.0, slot_live * 2.5)
            img[k, s] = t * yellow + (1 - t) * orange

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                gridspec_kw={'width_ratios': [3, 2]})

ax1.imshow(img, aspect='auto', interpolation='nearest', origin='lower')
ax1.set_xlabel("Generation step")
ax1.set_ylabel("Slot index")
ax1.set_title("32-slot KV arena, max_len = 4096, iteration-level scheduling",
              fontsize=10, loc='left')
ax1.spines[['top', 'right']].set_visible(False)

from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(color='#FFD700', label='Live KV (useful)'),
    Patch(color='#FF8C00', label='Reserved padding'),
    Patch(color='#1A1A1A', label='Freed (external frag)'),
], loc='upper right', framealpha=0.95, fontsize=8)

ax2.plot(util * 100, color='#FF007F', linewidth=2.4)
ax2.fill_between(range(n_steps), 0, util * 100, color='#FF007F', alpha=0.25)
ax2.axhline(util.mean() * 100, color='#1A1A1A', linestyle='--', linewidth=1.0)
ax2.text(2, util.mean() * 100 + 3, f"avg = {util.mean():.0%}",
         fontsize=9, fontweight='bold', color='#1A1A1A')
ax2.set_xlabel("Generation step")
ax2.set_ylabel("KV utilization (%)")
ax2.set_ylim(0, 100)
ax2.set_title("KV utilization over time", fontsize=10, loc='left')
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print(f"Average KV utilization     : {util.mean():.0%}")
print(f"Peak utilization           : {util.max():.0%}")
print(f"Trough utilization         : {util.min():.0%}")
wasted_gb = (1 - util.mean()) * 80
print(f"At an 80 GB H100, {wasted_gb:.0f} GB of HBM is locked in padding")
print(f"That is roughly {wasted_gb / 0.8:.0f}K tokens of KV that could have been billed")
```

The utilization line never climbs above 30%. The arena image shows the pattern: most slots are mostly orange (reserved padding the sequence has not grown into), a few are yellow-rich (long sequences eating their slot honestly), and as steps progress a scattering of slots turn ink-black (freed, but the slot index is gone forever from the contiguous tensor — *external fragmentation*).

The 38% Kwon scribbled on his whiteboard was not pessimistic.

## Why The KV Cache Is Specifically Cursed

Allocators face this problem for many workloads. What makes the KV cache uniquely bad? Four properties, all simultaneously.

**1. Variable input lengths.** A typical chat prompt is 50 tokens. A typical code-review prompt is 5,000. A typical "summarize this PDF" prompt is 50,000. The same serving system must hold all three.

**2. Unpredictable output lengths.** Until the model emits `<|eot|>`, you do not know whether the reply is 10 tokens or 1,000. You cannot bin-pack by length up front because *the lengths are not known up front*. The control flow is the model's.

**3. Massive absolute size.** Llama-65B at 128K context has a KV cache of approximately

$$
\underbrace{128{,}000}_{\text{tokens}} \cdot \underbrace{2}_{K,V} \cdot \underbrace{80}_{\text{layers}} \cdot \underbrace{64 \cdot 128}_{\text{heads}\cdot d_\text{head}} \cdot \underbrace{2}_{\text{FP16 bytes}} \;\approx\; 335\,\text{GB}.
$$

**Per sequence.** That is more memory than four H100s. (See [Issue 06](/llm-maths/issues/06-eviction-notice/02-kv-crisis/) for the full napkin math — it's the entire reason that issue exists.) At this scale, a 1% allocator waste is 3 GB. A 60% allocator waste is a second-mortgage GPU.

**4. Hot working-set access.** Every decode step, attention has to visit *every* position in the KV. There is no cold tier. You cannot demote inactive parts to L2 or system memory — they are needed in the next 25 ms. Whatever the allocator does, the working set lives in HBM, and HBM is what is precious.

Most allocator workloads — Python objects, network buffers, database pages — violate at least one of these. The KV cache hits all four at once. It is the worst customer your `malloc` will ever meet.

## Counting Bills The Allocator Set On Fire

Let us compare the three bad allocator choices side-by-side, plus the still-fictional "paging" alternative we are about to meet. Plot effective HBM utilization against the chat traffic load (concurrent sequences). The shape of each curve is the shape of the trade-off the corresponding allocator is making.

```pyplot {id="allocator-utilization" caption="HBM utilization vs concurrent sequences for four allocators on Llama-13B chat traffic. Pre-allocate-max is bounded above by ~25%. Grow-and-copy does better at low load but loses ground to copy storms. Dynamic fragments are good on paper but bottlenecked by the contiguity tax. Paged (vLLM) is the asymptote everyone is chasing."}
np.random.seed(19)

users = np.arange(1, 65)

# Pre-allocate-max: utilization = mean_len / max_len, mostly flat.
mean_len = 700
max_len  = 4096
prealloc = np.full_like(users, mean_len / max_len, dtype=float)
# Slight noise from variability in mean reply length
prealloc = prealloc + 0.02 * np.sin(users / 4)

# Grow-and-copy: starts high (small reservations), degrades from copy storms at high load
grow_copy = 0.65 - 0.45 * (1 - np.exp(-users / 18))
grow_copy = np.clip(grow_copy, 0.18, 0.65)

# Dynamic fragments: high utilization but kernel-coalescing tax means effective HBM bandwidth is much lower
# Plot the EFFECTIVE utilization (utilization * bandwidth_efficiency)
dyn_util_raw = 0.85 + 0.05 * np.sin(users / 5)
bandwidth_eff = 0.45 + 0.05 * np.exp(-users / 30)        # ~45% bandwidth efficiency
dyn_effective = dyn_util_raw * bandwidth_eff

# Paged (vLLM): solves both. Slight ramp because amortizing block-table costs.
paged = 0.92 + 0.04 * (1 - np.exp(-users / 8))
paged = np.minimum(paged, 0.965)

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(users, prealloc * 100,      color='#FF8C00', linewidth=2.4,
        label='1. Pre-allocate max (HF default)')
ax.plot(users, grow_copy * 100,     color='#FFD700', linewidth=2.4,
        label='2. Grow-and-copy')
ax.plot(users, dyn_effective * 100, color='#00A8A8', linewidth=2.4,
        label='3. Dynamic fragments (coalescing-taxed)')
ax.plot(users, paged * 100,         color='#FF007F', linewidth=2.8,
        label='4. Paged (vLLM, → ch.10)')

ax.axhline(50, color='#1A1A1A', linestyle='--', linewidth=0.8, alpha=0.4)
ax.text(63, 51, "break-even (50%)", ha='right', fontsize=8, color='#1A1A1A')

ax.set_xlabel("Concurrent sequences")
ax.set_ylabel("Effective HBM utilization (%)")
ax.set_ylim(0, 100)
ax.set_xlim(1, 64)
ax.set_title("The four allocator regimes, side by side",
             loc='left', fontsize=10)
ax.legend(loc='center right', framealpha=0.95, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print(f"At 32 concurrent users:")
print(f"  Pre-allocate max  : {prealloc[31]*100:5.1f}%  (HF Transformers default — what Kwon measured)")
print(f"  Grow-and-copy     : {grow_copy[31]*100:5.1f}%  (interesting at low load, collapses)")
print(f"  Dynamic fragments : {dyn_effective[31]*100:5.1f}%  (high *raw* util, but contiguity tax kills bandwidth)")
print(f"  Paged (vLLM)      : {paged[31]*100:5.1f}%  (next chapter)")
print()
print(f"Effective extra users on an 80 GB H100 going from pre-alloc to paged:")
gain = paged[31] / prealloc[31]
print(f"  {gain:.1f}× — i.e., one box now serves {gain:.1f}× the load it did on March 2023.")
```

The orange line is what every 2023 production system looked like. The teal line is the false promise of dropping the contiguity constraint — high "raw" utilization, but the kernel pays for it with collapsed memory coalescing, so the *effective* throughput per HBM byte is no better. The pink line is the rabbit we are chasing.

## Why You Cannot Just Call `cudaMalloc` In A Loop

The naive systems-engineer reflex, on hearing "variable-length allocation with unpredictable lifetimes," is: *just use the standard allocator*. Call `cudaMalloc` for each new token's KV slab, `cudaFree` when the sequence finishes. The OS allocator people did the hard work; let it solve our problem.

This is wrong, and the reason it is wrong is a number worth memorizing.

**CUDA allocator call latency is on the order of 5–30 microseconds per call.** A decode step is 25 milliseconds. If every layer of every sequence in a 32-slot batch calls `cudaMalloc` once per step to grow its KV by one token, that is

$$
32 \text{ slots} \times 80 \text{ layers} \times 1 \text{ call} \times 20\,\mu\text{s} \;=\; 51{,}200\,\mu\text{s} \;=\; 51\,\text{ms}.
$$

You have **doubled the decode latency** in allocator overhead alone before the model has done any math. And this is the *cheap* path — the moment the allocator has to coalesce free blocks or grow its arena, that latency goes up by another order of magnitude.

{{% callout type="warning" %}}
**The contiguity tax.** The other reason `cudaMalloc` does not save us is that the attention kernel of 2022–23 — the Flash inner loop walking K and V at sequential strides — *requires* each sequence's KV to be contiguous. If you hand each token a separate allocation, you destroy the kernel's memory coalescing pattern: 32-thread warps that were issuing one 256-byte load now issue 32 scattered 8-byte loads, and the effective HBM bandwidth collapses by an order of magnitude. **The kernel demands contiguity; the workload makes contiguity impossible.** That contradiction is the heart of the problem.
{{% /callout %}}

So the rule is: **call the allocator never, never at decode time, never per-token, never per-layer**. Whatever the solution is, it has to (a) pre-reserve a pool of HBM at process start, (b) hand out memory from that pool in microseconds-per-grant instead of microseconds-per-syscall, and (c) somehow let the attention kernel believe each sequence's KV is contiguous when it isn't.

That tuple — pool + cheap grant + contiguity-as-illusion — is *exactly* the contract of an operating system's **virtual memory subsystem**. The OS community shipped a working implementation in **1965**. Look up "Multics" on Wikipedia. It is a system that lets a process address a large logical region whose physical pages are scattered across whatever frames the kernel had free at allocation time. The translation from logical to physical address is one indirection through a **page table**.

## The OS Already Solved This. In 1965.

This is the punchline of the chapter. The exact problem we have just spent eight sections defining — variable-size objects, unpredictable lifetime, hot working set, contradictory demands of contiguity and elastic growth — **is the canonical problem that motivated virtual memory in the 1960s**. Programs were getting bigger than physical RAM. Multiple programs were running on one machine. The OS could not predict which program would need how much memory at any moment. **They solved it with paging.**

The technique:

1. Divide physical memory into **fixed-size pages** (4 KB on a typical OS; for our KV cache, vLLM will pick 16 *tokens* per block).
2. Give every process its own **virtual address space** — a logical address range that looks contiguous.
3. Maintain a **page table** per process: a small array mapping virtual page numbers to physical frame numbers.
4. On each memory access, walk the page table to translate virtual to physical. **One level of indirection.** That is the entire trick.

External fragmentation disappears because all allocations are page-sized — there is no awkwardly-shaped hole the allocator cannot fill. Internal fragmentation is bounded above by one page per process. The kernel can hand out pages from anywhere in physical RAM. The process never knows.

Now port the metaphor:

- **Physical memory** = HBM arena
- **Page** = a fixed-size block of KV cells (16 tokens, in vLLM's choice)
- **Process** = an active sequence
- **Page table** = a per-sequence **block table** that maps logical position $i$ to a physical block in HBM
- **Page walk** = one extra indirection in the attention kernel's KV indexing

The 2023 contribution is not the idea of paging. The 2023 contribution is the engineering hook: **rewriting the attention kernel so that the inner loop reads a block table on every access, with negligible overhead because each block holds 16 tokens of work**. The arithmetic intensity stays high; the indirection cost is amortized across the work in the block; the GPU never knows that each sequence's logical-contiguous KV is physically scattered across the HBM arena.

That kernel is called **PagedAttention**. The system that wraps it is called **vLLM**. Together they take a 38% utilization back up to 96%, and they let one H100 serve five times as many users on the same hardware on the same day. The four-line scheduler change from [The Conveyor Belt](../08-continuous-batching/) plus the page-table indirection from this chapter are the two ideas that, more than any others, define modern LLM serving.

That story is the next chapter. We have spent this one defining the disease. The cure is one page-table away.

**Continue to → [Borrowing from 1965](../10-paged-attention/)** — the moment Kwon, Li, Zhuang, and Sheng saw the page-table analogy, edited the FlashAttention inner loop to walk a block table, and turned a 38% utilization into 96% with one level of indirection.
