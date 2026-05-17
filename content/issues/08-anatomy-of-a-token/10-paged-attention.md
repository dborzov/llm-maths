---
title: "Borrowing From 1965"
description: "PagedAttention is virtual memory for KV caches: a per-request block table maps logical token positions to non-contiguous physical KV blocks, eliminating fragmentation. The same abstraction enables prefix sharing, copy-on-write branching for beam search, and tiered KV offload — all for free."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T13:30:00-04:00
issue: 8
weight: 100
techKind: mainline
techNode: paged-attention
header: 10-paged-attention.webp
---

## Cambridge, Massachusetts, October 1965

Picture **Fernando Corbató** in a fluorescent-lit basement at MIT. He is forty years old, slim, with rolled-up shirt sleeves and a perpetual look of someone who has been arguing with the same machine for ninety hours. The machine is a GE-645 mainframe — a metal cabinet the size of a station wagon — and the operating system he and his team are writing for it will be called **Multics**. Multics is the most ambitious software project in computing history to date. It also, almost as a side effect, contains the idea that thirty floors of every modern data center will eventually be built on top of: **virtual memory**.

The problem Corbató is staring at is this. Multics is *time-shared*: a dozen users sit at teletypes around campus, and the OS gives each of them the illusion that they own the whole machine. Each user's program wants a long, contiguous chunk of address space — say, addresses $0$ through $2^{18}$. But the actual physical RAM is a fixed 256 KB, and you cannot give twelve programs a contiguous 256 KB each. You also cannot ask each programmer to manually pack their code into whatever gaps happen to be free that afternoon. The data layout cannot be the programmer's problem; the OS has to lie.

So the OS lies. Each process gets a **page table** — a small per-process array that maps logical pages (chunks of the program's pretend address space) to physical frames (actual locations in RAM). When the program reads logical address $0$x$\text{1A40}$, the hardware walks the page table, finds that logical page $1$ currently lives at physical frame $7$, and silently rewrites the read. Programs see neat, contiguous, growing-on-demand address spaces. The OS sees a fragmented confetti of frames it can recycle, share, or swap to disk as it pleases.

This abstraction is so foundational it is, in 2026, *invisible*. Every operating system on Earth uses it. The hardware MMU enforces it without us thinking. Computer science undergraduates learn it in sophomore year and then forget it, because it is the air computers breathe.

In **June 2023**, in Sutardja Dai Hall at UC Berkeley, a graduate student named **Woosuk Kwon** is sitting in front of a monitor watching an early LLM serving system bleed memory. He is profiling the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} allocator. The patterns scrolling past him — variable-length sequences, unpredictable growth, requests arriving and leaving asynchronously, the GPU's contiguous allocator forced to over-reserve to the worst case — look uncannily like Corbató's 1965 confetti.

The conceptual jump that follows is one of the cleanest acts of analogy in modern systems: *Kwon and his collaborators realize that the problem is not novel. It is the same problem Multics solved, on a different substrate, in a different decade. The solution is on a bookshelf.* The paper that drops three months later, {{< cite text="Kwon et al., 2023 (SOSP)" url="https://arxiv.org/abs/2309.06180" kind="paper" >}}, will be titled *"Efficient Memory Management for Large Language Model Serving with PagedAttention."* It will spawn vLLM, then SGLang, then TRT-LLM's KV reuse, then every serving stack in production today. It is borrowed wisdom — but borrowing wisdom is the highest form of engineering.

## Why The Naive Allocator Burns

Before we walk into the fix, let us put numbers on the fire. The problem set up in [The KV Cache Is a Heap](../09-kv-fragmentation/) is this: each request grows its KV cache one token at a time, but the GPU's contiguous CUDA allocator has no idea how long any request will be. It has two bad choices, and only two:

- **Allocate to the max.** Reserve `max_seq_len` worth of KV memory per request up front. For Llama-70B at 32K context with 80 layers and 8 KV-heads, this is roughly *5 GB per request*. With 512 in-flight requests, you'd need 2.5 TB of HBM you don't have.
- **Allocate to the current length.** Realloc-and-copy every time the sequence grows. With decode steps issuing one token per request per millisecond across 512 requests, you'd be memcpy'ing tens of gigabytes per second on the critical path. The kernel would never run.

Real systems in 2022 — FasterTransformer, TGI, the first wave — picked a third option: **pre-allocate to a generous estimate** (say 2,048 tokens) and pad the tail. This left 60–80% of allocated KV memory permanently unused, which directly reduces the number of concurrent users the box can serve. {{% marginnote %}}The Kwon paper measures this on real workloads at 20%–40% useful KV utilization. The remaining 60%–80% is fragmentation tax.{{% /marginnote %}} The HBM that should be serving requests is sitting empty, "in case the sequence grows," for sequences that will never grow.

```pyplot {id="fragmentation-vs-paged" caption="LEFT: naive allocator pads each sequence's KV region to the worst-case length, wasting most HBM. RIGHT: paged allocator hands out fixed-size blocks on demand; the same HBM serves many more concurrent requests."}
np.random.seed(7)

# Simulate 12 in-flight requests with very different lengths
n_req = 12
max_len = 2048
true_lengths = np.array([180, 520, 90, 1340, 240, 760, 60, 1980, 410, 130, 880, 300])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# LEFT: naive — every request reserves max_len
for i, t in enumerate(true_lengths):
    ax1.barh(i, max_len, color='#1A1A1A', alpha=0.18, edgecolor='#1A1A1A', linewidth=0.6)
    ax1.barh(i, t,       color='#FF007F', edgecolor='#1A1A1A', linewidth=0.6)
ax1.set_xlim(0, max_len)
ax1.set_yticks(range(n_req))
ax1.set_yticklabels([f"req {i}" for i in range(n_req)], fontsize=8)
ax1.set_xlabel("KV cache positions (tokens)")
ax1.set_title(
    f"NAIVE: useful = {true_lengths.sum()}/{n_req*max_len} = {true_lengths.sum()/(n_req*max_len):.0%}",
    fontsize=10, loc='left')
ax1.spines[['top','right']].set_visible(False)

# RIGHT: paged — 16-token blocks, only ceil(t/16) blocks reserved per request
block = 16
total_blocks = (n_req * max_len) // block
for i, t in enumerate(true_lengths):
    n_used = int(np.ceil(t / block))
    for b in range(n_used):
        ax2.barh(i, block, left=b*block, color='#00A8A8',
                 edgecolor='#1A1A1A', linewidth=0.4)
ax2.set_xlim(0, max_len)
ax2.set_yticks(range(n_req))
ax2.set_yticklabels([f"req {i}" for i in range(n_req)], fontsize=8)
ax2.set_xlabel("KV blocks allocated (16 tokens each)")
useful = sum(int(np.ceil(t / block)) * block for t in true_lengths)
ax2.set_title(f"PAGED: useful = {true_lengths.sum()}/{useful} = {true_lengths.sum()/useful:.0%}",
              fontsize=10, loc='left')
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()

print("Naive allocator wastes 60-80% of HBM on padding.")
print(f"  Useful tokens stored:    {true_lengths.sum()}")
print(f"  Memory committed:        {n_req * max_len} token-slots")
print(f"  Utilization:             {true_lengths.sum()/(n_req*max_len):.0%}")
print()
print("Paged allocator wastes at most one block per request (15 token-slots).")
print(f"  Useful tokens stored:    {true_lengths.sum()}")
print(f"  Memory committed:        {useful} token-slots")
print(f"  Utilization:             {true_lengths.sum()/useful:.0%}")
```

The picture on the right is what we're about to build. Each coloured bar is a **block**. Each block is a fixed-size 16-token slab. Sequences accrue blocks one at a time, and when a sequence finishes, its blocks return to a free pool to be picked up by the next request that needs them. The total "wasted" memory is at most one partially-filled block per active request — *15 token-slots*, not 1,500.

## The Analogy, Laid Out

Before we get to the data structure, here is the dictionary. Every entry on the left was understood by 1969. Every entry on the right is built in 2023, on the same idea, in a domain that did not exist when Multics shipped.

| OS virtual memory (1965) | LLM serving (2023) |
|---|---|
| Process | In-flight request (sequence) |
| Logical address | Token position in sequence |
| Physical page (4 KB) | KV block (16 tokens × K-and-V × heads × dim × dtype) |
| Page table | **Block table** (per-request) |
| Free frame pool | Free block pool |
| Demand paging | Allocate-on-append (per decode step) |
| Copy-on-write (`fork`) | Beam-search / parallel sampling branching |
| Shared library page | Prefix-cached shared blocks |
| Swap to disk | KV offload to CPU / SSD / peer-GPU |
| MMU page-walk | Block-table indirection inside the attention kernel |

Stare at this table until it stops looking surprising. Every row is the same architectural move at a different scale. Multics' twelve concurrent users mapped to 256 KB of RAM became vLLM's five hundred concurrent users mapped to 80 GB of HBM. The block size grew, the access pattern grew, the dtype changed — but the *abstraction* is identical.

{{< crosshead >}}Block Anatomy{{< /crosshead >}}

A block is the unit of paging. vLLM's default is 16 tokens per block, which is not arbitrary — it's a tile size that fits one warp's worth of work cleanly inside a {{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}} inner loop, and it amortizes the per-block indirection cost across enough useful arithmetic that the indirection is essentially free.

Let us size one block for Llama-70B running in FP8.

$$
\text{block bytes} \;=\; \underbrace{80}_{\text{layers}} \times \underbrace{8}_{\text{KV-heads}} \times \underbrace{16}_{\text{tokens}} \times \underbrace{128}_{\text{head dim}} \times \underbrace{2}_{\text{K and V}} \times \underbrace{1}_{\text{FP8}} \approx 327 \text{ KB.}
$$

{{% marginnote %}}That's roughly one Linux *huge page* (2 MB / 8 ≈ 256 KB rounded). Not a coincidence — both numbers come from the same engineering pressure: large enough to amortize TLB / indirection overhead, small enough to keep internal fragmentation bounded.{{% /marginnote %}}

For a 64 GB KV pool on a single H100 you get roughly **200,000 blocks**, or three million tokens of capacity. That's not "context length per user"; that's *total tokens across all users you can keep resident simultaneously*. If your average user is 4K tokens, you can serve ~750 concurrent users out of one GPU's KV pool. That number was about 250 under the old contiguous allocator. The 3× rises directly out of paging.

## The Block Table Is The Entire Mechanism

For each in-flight request, vLLM maintains a tiny array of physical block IDs:

```python
# Request 42 has produced 67 tokens so far → ceil(67/16) = 5 blocks
block_table[42] = [1812, 47, 9, 1810, 1811]  # 5 physical block ids

# To attend to logical position t=53:
logical_block = 53 // 16          # → 3
offset_in_block = 53 % 16          # → 5
phys_id = block_table[42][3]      # → 1810
# KV is at kv_pool[1810][offset 5]
```

That is the whole API. Three integer operations: divide, modulo, dereference. The logical positions look contiguous to the model; the physical blocks are scattered all over a 64 GB pool, in whatever order the free queue handed them out.

```pyplot {id="block-table-indirection" caption="The block table for one request: logical token positions 0-31 mapped through a 2-entry per-request array to non-contiguous physical blocks in the global KV pool."}
np.random.seed(3)

# Global pool: 32 blocks. Free, in-use by other requests, or owned by us.
n_blocks = 32
states = np.zeros(n_blocks, dtype=int)
others = np.random.choice(n_blocks, 14, replace=False)
states[others] = 2  # in use by other requests
my_blocks = [7, 22]  # the two physical blocks our request owns
for b in my_blocks:
    states[b] = 1

fig, ax = plt.subplots(figsize=(10.5, 4.4))

# Top: logical view (the model thinks it sees contiguous tokens 0..31)
logical_y = 1.6
for i in range(32):
    color = '#FFD700' if i < 16 else '#FF8C00'
    ax.add_patch(plt.Rectangle((i, logical_y), 1, 0.6, facecolor=color,
                                edgecolor='#1A1A1A', linewidth=0.4))
ax.text(-0.4, logical_y+0.3, "Logical\ntokens 0-31",
        ha='right', va='center', fontsize=9, fontweight='bold')
ax.text(8,  logical_y+0.85, "block 0", ha='center', fontsize=8, color='#1A1A1A')
ax.text(24, logical_y+0.85, "block 1", ha='center', fontsize=8, color='#1A1A1A')

# Middle: block table (just two ints)
table_y = 0.6
ax.add_patch(plt.Rectangle((6, table_y), 4, 0.6, facecolor='#FDF5E6',
                            edgecolor='#1A1A1A', linewidth=1.5))
ax.text(8, table_y+0.3, f"block_table = [{my_blocks[0]}, {my_blocks[1]}]",
        ha='center', va='center', fontsize=10, fontweight='bold', family='monospace')
ax.text(-0.4, table_y+0.3, "Block table\n(2 ints)",
        ha='right', va='center', fontsize=9, fontweight='bold')

# Bottom: physical pool of 32 blocks
phys_y = -0.6
for i, s in enumerate(states):
    if s == 0:
        c = '#FDF5E6'  # free
    elif s == 1:
        c = '#FFD700' if i == my_blocks[0] else '#FF8C00'
    else:
        c = '#1A1A1A'
    ax.add_patch(plt.Rectangle((i, phys_y), 1, 0.5, facecolor=c,
                                edgecolor='#1A1A1A', linewidth=0.4))
ax.text(-0.4, phys_y+0.25, "Physical\nKV pool",
        ha='right', va='center', fontsize=9, fontweight='bold')

# Arrows
for tgt_log, tgt_phys, c in [(8, my_blocks[0], '#FF007F'),
                              (24, my_blocks[1], '#FF007F')]:
    ax.annotate('', xy=(tgt_phys+0.5, phys_y+0.55),
                xytext=(tgt_log, table_y),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.6))

ax.set_xlim(-3, 33)
ax.set_ylim(-1, 2.6)
ax.set_aspect('equal')
ax.axis('off')

print("Logical view (model sees this):  tokens 0-31 contiguous")
print("Physical reality:                two scattered blocks at phys ids 7 and 22")
print("Translation cost:                one divmod + one array lookup per tile")
```

When the model wants to compute attention from query token 53 onto every cached key, the attention kernel walks the block table once per 16-token tile and reads the right physical block. The compute layer sees a tidy logical sequence. The memory layer sees confetti. Nobody is unhappy.

{{% callout type="note" %}}
**The page-table walk happens on the GPU.** The block table for each request lives in HBM alongside the KV pool, and the attention kernel reads it with a normal load. There is no host round-trip, no kernel re-launch, no driver-mediated address translation. The "MMU" is just three lines of CUDA in the inner loop.
{{% /callout %}}

## What Changes Inside The Attention Kernel

The {{< wiki "attention" >}}attention{{< /wiki >}} kernel we want to modify is the FlashAttention skeleton from [Attention in SRAM](../06-flash-attention/): an outer loop over query tiles, an inner loop over KV tiles, online softmax accumulating into per-query running statistics. The whole point of FlashAttention is that the KV is streamed through SRAM in tiles and never lives in HBM as one giant materialized score matrix.

**PagedFlash** changes exactly one thing about that loop: the inner-loop tile load.

```
for q_tile in q_tiles:
    m, l, o = init_running_stats()
    for kv_tile_idx in range(0, ctx_len, BLOCK_SIZE):
        logical_block = kv_tile_idx // BLOCK_SIZE
        phys_id = block_table[logical_block]       # <-- THIS LINE IS NEW
        K_tile = kv_pool[phys_id].K_view()
        V_tile = kv_pool[phys_id].V_view()
        m, l, o = online_softmax_step(q_tile, K_tile, V_tile, m, l, o)
    write(o)
```

One extra integer load per 16 tokens of context. Two thousand FLOPs of attention math per loaded block. The indirection cost is below noise — the lookup pipelines with the previous tile's compute, the block table sits in L2 (it's tiny), the page-table walk is invisible.

The benefit? Look back at the fragmentation pyplot. Sixty to eighty percent of HBM that used to be padding is now usable. Effective batch size — the number of users you can pack into your GPU at once — jumps **2× to 4×**. Throughput follows in lockstep, because decode is bandwidth-bound and bandwidth scales with how many useful KV bytes per request you can keep resident.

{{% pullquote type="profound" %}}
PagedAttention is virtual memory for KV caches. Every architectural win it produces — prefix sharing, copy-on-write branching, tiered offload — is a corollary of one 1965 idea applied to a workload that didn't exist sixty years ago.
{{% /pullquote %}}

## Three Things Paging Enables For Free

The paper's title only advertises "memory management." The real surprise is everything that falls out of the abstraction once you have it. Pages turned out to be the right unit for *every* form of KV sharing, not just packing.

**1. Prefix caching.** Two requests start with the same 1,500-token system prompt. In contiguous-allocator land they each compute the same KV vectors for the same tokens, store them in separate physical regions, and pay full prefill twice. With paging, the two requests' block tables can simply *point to the same physical blocks* for the shared prefix. Hash the block contents, look up, share. We unpack this in [Reusing the Prologue](../12-prefix-caching/) — it is the single highest-leverage optimization in production serving.

**2. Copy-on-write branching.** Beam search and parallel sampling generate $k$ candidate continuations from the same prompt. Naively, you'd duplicate the KV cache $k$ times. With paging, the $k$ candidates share blocks while they agree, and the moment one of them writes a token its parent doesn't have, the block manager copies just that one block and decrements the parent's refcount. This is literally Unix `fork()`'s copy-on-write semantics; the implementation is twenty lines.

**3. Tiered storage.** A "physical" block doesn't have to live in HBM. Cold blocks — say, the KV for a conversation whose user has paused — can be swapped out to CPU memory, NVMe, or even another GPU over NVLink. When the conversation resumes, the block is paged back in. This is *exactly* swap-to-disk on a 1970s mainframe, played at GB/s on a 2025 RDMA fabric. It is what makes the KV-cache disaggregation we'll meet in [Two Houses, Divided](../17-disagg-pd/) possible at all.

None of these were Kwon's original goal. The original goal was just defragmentation. The other three fell out of the abstraction in the same way that, in 1965, nobody was specifically trying to design `fork()` or shared libraries when they wrote the page-table code — those features just *became possible* once pages existed.

## The Cost Side Of The Ledger

It would be dishonest not to name what paging gives up.

- **Slightly more indirection.** Every KV tile load goes through one extra integer dereference. Measurable in microbenchmarks; invisible in end-to-end runs.
- **Block boundaries impose alignment.** Sequence lengths get rounded up to the nearest 16. For a 1-token request this is 16× wasteful in absolute terms (one block, fifteen empty slots); for any real request it's a sub-1% overhead.
- **Implementation complexity.** The contiguous allocator was twenty lines. The block manager — which we examine next in [The Block Manager](../11-block-manager/) — is two thousand. The page table, the free queue, the LRU, the reference counts, the hash index, the copy-on-write paths all sit on the per-iteration critical path and have to be measured in nanoseconds.

The trade-off is so lopsided it isn't really a trade-off. You give up nothing the user can feel; you get back the GPU's actual capacity.

## What To Remember

1. **PagedAttention is virtual memory for KV caches.** Logical token positions are translated through a per-request block table to non-contiguous physical blocks in a shared pool. The fragmentation problem disappears the same way Multics made it disappear in 1965.
2. **The block table is the entire mechanism.** One integer lookup per 16-token tile in the attention kernel. Everything downstream — prefix caching, CoW branching, KV offload, disaggregation — is built on this one indirection.
3. **The free wins are the real story.** Sharing prefixes, sharing beam-search forks, sharing across memory tiers were not the goal. They fall out of the abstraction. This is how good systems abstractions work: solve the problem you have, get five problems you didn't know you were going to have solved for you.

**Continue to → [The Block Manager](../11-block-manager/)** — the data structure that turns the page-table abstraction into running code, in nanoseconds per request per step.
