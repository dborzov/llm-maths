---
title: "The Block Manager"
description: "The block manager is the data structure that implements paging for KV caches: a free-block pool, per-request block tables with reference counting, and an O(1) LRU eviction policy. Understanding it is the key to understanding how prefix caching, copy-on-write, and KV offload work at the implementation level."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T14:00:00-04:00
issue: 8
weight: 110
techKind: primer
techNode: block-manager
header: 11-block-manager.webp
---

## A Single Decode Step, Slowed Down

It is **March 2024**. A vLLM instance is serving a production chat endpoint. At any given instant there are roughly **512 in-flight requests** living in its scheduler. Every 25 milliseconds the engine wakes up to compute one decode step — one new token for each active request. In the time it takes you to blink, the following has to happen, deterministically, with no allocator hiccup:

- About **20** of those 512 requests' KV caches overflow their current last block and need a fresh one allocated.
- About **5** requests finish their final token and need *all* their blocks returned to the free pool.
- All 512 block tables must be readable by the attention kernel before it launches — meaning the data structure is consistent, not mid-mutation.

The budget per request, per step, for all this bookkeeping is roughly **2 microseconds**. If you spend more, the GPU starts idling waiting for the scheduler, and the box's tokens-per-second falls off a cliff. The {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} allocator can never be the bottleneck — but it lives on the critical path of every single forward pass, so it can easily *become* the bottleneck if you are not careful with your data structures.

The thing doing that work is the **block manager**. It is the unsexy plumbing that makes everything in the previous chapter ([Borrowing from 1965](../10-paged-attention/)) real. This primer is a tour of it, written against the actual vLLM source ({{< cite text="vLLM source" url="https://github.com/vllm-project/vllm" kind="repo" >}}) — specifically `vllm.v1.core.kv_cache_manager` and `vllm.v1.core.block_pool` as of late 2025.

## Three Data Structures, And That's It

The whole system, conceptually, is three things.

**1. BlockPool.** One enormous contiguous KV tensor on each GPU, conceptually indexed `[0..N)`. This is the slab. Block $i$ occupies a fixed-size region — for Llama-70B FP8 with 16-token blocks, that's about 327 KB per block, so a 64 GB KV pool gives you $N \approx 200{,}000$ blocks. The slab is allocated *once* at startup. Nothing in steady state ever calls `cudaMalloc`.

**2. FreeKVCacheBlockQueue.** A doubly-linked list of physical block IDs that are currently free (or evictable). Push to head and pop from head are both $O(1)$. The tail of the queue is where the **LRU** lives: blocks that have been free the longest are the ones evicted first when prefix caching wants to recycle them. We will see why the tail matters in a moment.

**3. KVCacheManager.** A per-request orchestrator. It owns the **block table** for each request (an integer array; see [previous chapter](../10-paged-attention/)), the request's hashes (for prefix caching), and a thin layer of Python that calls into the BlockPool to acquire and return blocks.

That's the system. A slab, a free list, an orchestrator. Three nouns. Everything else — prefix caching, copy-on-write, eviction, offload — is operations on these three nouns.

{{% callout type="tip" %}}
The block pool is *not* a `dict[int, Block]`. It's a `numpy.ndarray` of fixed-size structs (refcount, hash, prev_pointer, next_pointer), accessed by index. Python's hashmap overhead would devour the per-step budget; the array is cache-line friendly and `__slots__`-tight.
{{% /callout %}}

## The Operations Table

Every operation the block manager performs in steady state is one of these five, each with a tight big-O.

| Operation | When called | Complexity | What happens |
|---|---|---|---|
| `allocate_initial(seq_len)` | At prefill | $O(\lceil T / 16 \rceil)$ | Pop $\lceil T/16 \rceil$ blocks from free queue head; write block table |
| `append_token()` | Every decode step | $O(1)$ amortized | Maybe pop one block from free queue (only on block boundary, i.e. 1 in 16 steps) |
| `free(request_id)` | At request completion | $O(\lceil T / 16 \rceil)$ | Decrement refcount on each block; zero-refcount blocks rejoin free queue tail |
| `cow_branch(parent, n)` | Beam search / parallel sampling | $O(1)$ up front, $O(\text{copy})$ on first divergent write | Refcount bump on parent's blocks; copy on demand |
| `lookup_prefix(hash)` | Prefix cache hit path | $O(1)$ average (hashmap) | Dict probe; refcount bump if hit |

Notice what's not on this list: there is no `malloc` and there is no `memcpy`. The hot path is integer arithmetic on a pre-allocated slab. The only memcpy in the whole system is a *block copy* triggered lazily during copy-on-write, and that copy happens on the GPU in parallel with other work.

## Why The Free Queue Is Doubly-Linked

This deserves a paragraph of its own, because it is the kind of detail that looks like a CS-101 textbook flourish until you see why it matters in production.

A naive free list is a stack: push to head, pop from head. $O(1)$. Done.

vLLM uses a *doubly-linked* list, which costs four bytes per block extra (a back-pointer), so that the same data structure can serve two purposes:

- **Allocator:** pop from head when somebody needs a new block. Same as a stack.
- **LRU evictor:** if prefix caching is on, recently-freed blocks are kept *in the queue* with their KV contents intact, in case a future request wants to read them. When the pool truly runs out, the tail (least recently freed → least recently used) is the eviction victim.

The doubly-linked structure makes one more operation $O(1)$: **detach a specific block from the middle of the list**. When a prefix-cache hit promotes a free-but-not-evicted block back to in-use, the block manager pulls it out of the middle of the LRU queue and links its neighbors. With a singly-linked list this would be $O(\text{queue length})$ — disastrous on the per-iteration budget.

```pyplot {id="blockpool-snapshot" caption="Block pool snapshot across three decode steps. Free blocks (cream) sit in the LRU queue tail; the head is where new allocations come from. Prefix-shared blocks (yellow) are pinned by refcount > 1."}
np.random.seed(11)
N = 48

def render_state(ax, states, title):
    cols = 12
    rows = N // cols
    for i in range(N):
        r, c = i // cols, i % cols
        state = states[i]
        if state == 'free':
            color, edge = '#FDF5E6', '#1A1A1A'
        elif state == 'in_use':
            color, edge = '#FF007F', '#1A1A1A'
        elif state == 'shared':
            color, edge = '#FFD700', '#1A1A1A'
        else:
            color, edge = '#00A8A8', '#1A1A1A'
        ax.add_patch(plt.Rectangle((c, rows - r - 1), 0.92, 0.92,
                                    facecolor=color, edgecolor=edge, linewidth=0.5))
        ax.text(c+0.46, rows - r - 0.54, str(i), ha='center', va='center',
                fontsize=7, color='#1A1A1A')
    ax.set_xlim(-0.2, cols+0.2)
    ax.set_ylim(-0.2, rows+0.2)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, loc='left', fontweight='bold')
    ax.axis('off')

# Step 0: baseline. ~24 in-use, 8 shared (prefix), rest free.
states0 = ['free']*N
for i in [0,1,2,3,4,5,6,7]: states0[i] = 'shared'  # system prompt blocks
for i in [12,13,14,15,16,17,18,19,20,21,22,23,
          24,25,26,27,28,29,30,31,32,33,34]: states0[i] = 'in_use'

# Step 1: req 3 finishes (frees 36,37), reqs 5,8 grow (alloc 40,41).
states1 = list(states0)
states1[40] = 'in_use'
states1[41] = 'in_use'

# Step 2: more churn. Prefix-cache hit: block 36 (freed earlier elsewhere) gets re-claimed.
states2 = list(states1)
states2[36] = 'shared'   # promoted back via prefix-cache hit
states2[42] = 'in_use'
states2[43] = 'in_use'

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
render_state(axes[0], states0, "STEP 0 (start)")
render_state(axes[1], states1, "STEP 1 (free + alloc)")
render_state(axes[2], states2, "STEP 2 (prefix hit re-claim)")

# Legend
legend_elems = [
    plt.Rectangle((0,0),1,1, facecolor='#FDF5E6', edgecolor='#1A1A1A', label='free (LRU queue)'),
    plt.Rectangle((0,0),1,1, facecolor='#FF007F', edgecolor='#1A1A1A', label='in use by one request'),
    plt.Rectangle((0,0),1,1, facecolor='#FFD700', edgecolor='#1A1A1A', label='shared (refcount > 1)'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()

free0 = sum(1 for s in states0 if s == 'free')
free1 = sum(1 for s in states1 if s == 'free')
free2 = sum(1 for s in states2 if s == 'free')
print(f"step 0: {free0} free, {sum(1 for s in states0 if s=='shared')} shared, {sum(1 for s in states0 if s=='in_use')} in-use")
print(f"step 1: {free1} free, {sum(1 for s in states1 if s=='shared')} shared, {sum(1 for s in states1 if s=='in_use')} in-use")
print(f"step 2: {free2} free, {sum(1 for s in states2 if s=='shared')} shared, {sum(1 for s in states2 if s=='in_use')} in-use")
print()
print("The pool never grows. Blocks rotate through three states.")
```

## Reference Counting And LRU, In Two Sentences

Every physical block carries a small integer **refcount**: the number of in-flight requests whose block tables currently point at it. The block is *in use* iff its refcount is $\geq 1$; it is *evictable* iff its refcount is exactly $0$.

When a request finishes, the manager walks its block table and decrements each block's refcount. Blocks that drop to zero don't immediately rejoin the free queue head — they go to the **tail**, with their KV contents still intact, so a future prefix-cache lookup can resurrect them. Only when the head of the free queue is empty does the allocator start evicting from the tail. This is exactly the **LRU cache** discipline, implemented in $O(1)$ per operation by the doubly-linked list.

The whole prefix-caching scheme of [Reusing the Prologue](../12-prefix-caching/) is built on top of this. A "prefix cache" is just the LRU-tail blocks that happen to still match the hash of an incoming request's prefix. There is no separate cache data structure. The free queue *is* the prefix cache.

{{< crosshead >}}Copy-on-Write, In Practice{{< /crosshead >}}

Beam search with $k=4$ on a 200-token prompt. The four candidate sequences share the first 200 tokens — that's $\lceil 200/16 \rceil = 13$ blocks. Naively we'd duplicate those blocks four times = 52 blocks. With CoW we allocate 13 blocks for the prompt, set refcount = 4, and the four candidates' block tables all point at them. Total blocks consumed so far: 13.

Now the four candidates start generating. Candidate A writes token 201. That write lands in *block 12* (logical block of position 201 = 12), offset 9. But block 12's refcount is 4 — it's shared. The block manager intercepts the write:

1. Allocate a fresh physical block $P$.
2. Copy block 12's existing contents into $P$ (one 327 KB GPU memcpy — fast, but real).
3. Update candidate A's block table at slot 12 to point to $P$.
4. Decrement the original block 12's refcount to 3.
5. Apply candidate A's write to $P$.

Now A has its own block 12; B, C, D still share the original. Each candidate pays the copy cost exactly once at the moment of divergence. Until divergence the four candidates are physically identical and computationally free.

This is *literally* Unix `fork()`. If you have ever wondered why `fork()` is so cheap despite duplicating an entire process — same trick.

## Why This Is The Critical Path

It is tempting to think of the block manager as bookkeeping that runs "between" the heavy GPU kernels. It isn't between them. It runs *every iteration*, *before* every kernel launch, on the CPU, on the engine's main loop. If the engine spends 5 ms per step on block manager work, that's 5 ms the GPU is sitting idle.

Let us put numbers on it. Suppose the manager does one `dict[hash]` lookup per request per step. Python dict access is ~100 ns. With 512 requests that's 51 μs per step — fine. But if the dict lookup happens to be a **miss** that triggers a Python object allocation (`Block(...)` constructor call), that's 1–2 μs. Now 512 misses = 1 ms. Now you've eaten 4% of the decode-step budget on one line of Python.

This is why the vLLM V1 codebase has aggressive optimizations that look weird in isolation:

- `__slots__` on `Block` to avoid Python's per-object dict overhead.
- Object pools that recycle `Block` instances rather than allocating fresh.
- Caching hash values in the block struct rather than recomputing.
- Vectorized batch operations over arrays of block IDs (NumPy, not loops).

```pyplot {id="overhead-per-step" caption="Estimated Python overhead per decode step as the number of concurrent requests grows. Naive dict + per-block object allocation crosses the 25 ms decode budget at ~600 requests; the optimized hot loop stays linear and safe out to a few thousand."}
n_reqs = np.arange(0, 2001, 50)

# Microsecond cost per request, per step
naive_us = 8.0      # dict-with-allocations + per-block Python object churn
optimized_us = 1.2  # __slots__ + object pool + ndarray-backed pool

naive_ms = n_reqs * naive_us / 1000.0
opt_ms   = n_reqs * optimized_us / 1000.0

fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(n_reqs, naive_ms, color='#FF007F', linewidth=2.4, label='Naive Python (dict + per-block obj)')
ax.plot(n_reqs, opt_ms,   color='#00A8A8', linewidth=2.4, label='Optimized (__slots__ + obj pool)')
ax.axhline(25, color='#1A1A1A', linewidth=1.2, linestyle='--')
ax.text(2000, 25.6, "25 ms decode-step budget", ha='right', fontsize=9, color='#1A1A1A')
ax.fill_between(n_reqs, 25, np.maximum(naive_ms, opt_ms),
                where=(naive_ms > 25), color='#FF007F', alpha=0.10)
ax.set_xlabel("Concurrent in-flight requests")
ax.set_ylabel("Block manager overhead per step (ms)")
ax.set_title("Why __slots__ and object pools matter")
ax.legend(loc='upper left')
ax.spines[['top','right']].set_visible(False)

cross_naive = n_reqs[naive_ms > 25][0] if any(naive_ms > 25) else None
print(f"Naive hot loop crosses the 25 ms budget at ~{cross_naive} requests.")
print(f"Optimized version stays at {opt_ms[-1]:.1f} ms even at 2000 requests.")
print()
print("Every microsecond per request, per step, taxes throughput linearly.")
print("This is why the V1 codebase looks like C in a Python costume.")
```

The plot is the answer to "why does the block manager source look so micro-optimized." The answer is: the difference between 8 μs and 1.2 μs per request per step is the difference between serving 600 concurrent users and serving 3,000.

## What To Remember

1. **The free queue is a doubly-linked list, and that matters.** $O(1)$ allocation (head), $O(1)$ eviction (tail), and crucially $O(1)$ middle-detach for prefix-cache promotions. Reference counting plus LRU live in this one structure.
2. **Copy-on-write is lazy and `fork()`-shaped.** Beam-search candidates share prompt blocks until one of them writes; the write triggers a single 327 KB block copy and a refcount split. Until divergence, branching is free.
3. **The block manager is on the per-iteration critical path.** Every microsecond of Python overhead taxes every decode step linearly with concurrent users. The seemingly paranoid optimizations in the vLLM hot loop are not paranoia — they are the difference between serving hundreds and serving thousands of users on the same box.

**Continue to → [Reusing the Prologue](../12-prefix-caching/)** — the highest-leverage payoff the block manager unlocks, and the reason every production deployment cares about hash-block-by-block.
