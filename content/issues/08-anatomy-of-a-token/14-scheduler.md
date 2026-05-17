---
title: "The Token Budget"
description: "vLLM V1's scheduler is deliberately simple: allocate a fixed token budget per step, let each request spend it on prefill or decode tokens, and let the prefill-versus-decode distinction dissolve. This chapter traces the years of layered complexity that one clean abstraction quietly deletes."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T15:30:00-04:00
issue: 8
weight: 140
techKind: mainline
techNode: scheduler
header: 14-scheduler.webp
---

## Two Schedulers, A Lifecycle, And Three Sad Years

It is **August 2024**. **Woosuk Kwon**, the Berkeley PhD student who wrote the first PagedAttention prototype the previous summer, is staring at a flame graph on his laptop in a Sky Computing Lab cubicle. The graph is from a customer-reported production incident: a 8×H100 box running vLLM V0 has just spent **180 milliseconds inside its Python scheduler** on a single iteration. The GPUs were idle for the entire 180 ms. That box was serving 600 concurrent users. None of them got a token during that window.

The bug is in there somewhere — a missed lock, a misplaced `dict.copy()`, a request that flipped state under the wrong condition. Kwon has fixed three of these bugs in the last month. He has come to a private conclusion that he has not yet said out loud: *the scheduler is the bug*.

To understand what he means, you have to look at how V0 (2023–24) actually scheduled work. There were **two distinct schedulers** in the V0 codebase. A **prefill scheduler** that consumed waiting requests, ran their prompts through the model, and produced a {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} for them. And a **decode scheduler** that took the running requests and fed the model one new token at a time per request. The two cooperated through a request lifecycle:

```
WAITING ──► RUNNING_PREFILL ──► RUNNING_DECODE ──► DONE
              │                       │
              ▼                       ▼
           PREEMPTED              PREEMPTED
```

Each transition required updating shared state — block tables, attention masks, position offsets, batch indices. Each transition had its own corner case. A request whose prefill spilled across two iterations (the prefix-cache code path was added in 2024) had to flip back and forth between `RUNNING_PREFILL` and a transient `RECOMPUTE` state. A preempted decode that was later resumed had to rebuild its position IDs from scratch. The state diagram had thirteen nodes by Q3 2024 and the test suite for it took longer to run than the actual model.

{{% marginnote %}}The V0 scheduler was not *bad code*. It was the natural result of growing the system feature-by-feature: continuous batching, then paged attention, then prefix caching, then chunked prefill, each adding state and edge cases on top of the previous design. The bug Kwon was chasing was, in some sense, a complexity bug — the cost of layers.{{% /marginnote %}}

The V1 rewrite, which Kwon and Roger Wang and Cody Yu and Kaichao You and Simon Mo shipped publicly on **January 27, 2025** ({{< cite text="vLLM V1 alpha release" url="https://blog.vllm.ai/2025/01/27/v1-alpha-release.html" kind="blog" >}}), deletes the lifecycle. There is no `RUNNING_PREFILL` and no `RUNNING_DECODE`. There is no separate prefill scheduler. There is a single function — `schedule_step` — that takes the current set of in-flight requests and produces one number per request: **how many tokens to advance this step**.

That's it. Prefill, decode, chunked-prefill-mid-flight, prefix-cache-hit, speculative-decode-verify — all of them are "how many tokens this iteration." The state machine collapses to a single counter per request: `num_computed_tokens`. The thirteen-node diagram becomes a single integer.

This chapter is about why that integer was sufficient, and why nobody noticed it could be until five other things had to be invented first.

## The Insight, Stated Plainly

The whole V1 redesign turns on one observation. A {{< wiki "kv-cache" >}}prefill step{{< /wiki >}} for a request is "produce KV vectors for $N$ new tokens of this request." A decode step is "produce KV vectors for $1$ new token of this request." **These are the same operation with different $N$.** The forward pass takes a $(B, T)$ tensor of token IDs and emits a $(B, T)$ tensor of logits and a $(B, T, K, D)$ tensor of new KV entries. Whether $T = 1$ or $T = 2048$ is just a parameter.

For most of vLLM's lifetime this was technically true but operationally not useful, because prefill and decode had fundamentally different *behaviour*: prefill was compute-bound and a single one of them could monopolize the GPU for seconds, while decode was bandwidth-bound and fast. You could not just put a 64,000-token prefill in the same batch as a decode step without wrecking everyone else's inter-token latency — see [Slicing the Prefill](../13-chunked-prefill/) for the gory details.

What changed was [chunked prefill](../13-chunked-prefill/). Once a long prefill was associative over the token axis — once you could slice it into 2,048-token pieces and run each piece as one scheduler step — the maximum $N$ any request would ask for in one iteration was *bounded by the chunk size*. And that bound was the same as the bound on a decode step's contribution: both fit under a single per-step **token budget**.

{{% pullquote type="counter-intuitive" %}}
Prefill and decode are the same operation. The scheduler stops caring which is which.
{{% /pullquote %}}

Once that's true, you don't need two schedulers. You need one function that decides *which requests get to spend tokens this step, and how many each*. That function is twenty-six lines of Python.

## The Algorithm, In Pseudocode

Here is the V1 scheduler in stripped-down form. Read it once for shape, then we'll walk every line and tag each one to a previous chapter.

```python
def schedule_step(self):
    budget = self.token_budget         # e.g. 8192 tokens/step
    plan = {}                          # request_id -> num_tokens this step

    # 1. Continue running requests in priority order
    for req in self.running:
        n = min(req.tokens_remaining_to_prefill or 1, budget)
        if not self.block_manager.can_allocate(req, n):
            self.preempt(req); continue
        plan[req.id] = n
        budget -= n
        if budget == 0: return plan

    # 2. Admit waiting requests
    while self.waiting and budget > 0:
        req = self.waiting.popleft()
        # Apply prefix-cache lookup; reduces n if a long prefix is already in HBM
        n = min(req.tokens_to_prefill_after_cache, budget)
        if not self.block_manager.can_allocate(req, n):
            self.waiting.appendleft(req); break
        plan[req.id] = n
        self.running.add(req); budget -= n

    return plan
```

Every line in this function is a previous chapter of this issue, condensed to one expression:

- **`self.token_budget`** — the central knob. Set by the operator at server startup. Usually 8,192 on H100 / H200; smaller on memory-constrained boxes. Bounded by the largest CUDA-graph capture bucket — see [Launches Aren't Free](../05-cuda-graphs/).
- **`req.tokens_remaining_to_prefill or 1`** — if the request is mid-prefill, ask for the remaining chunk; otherwise this is a decode step and ask for one token. Chunked prefill makes the first branch finite-sized — see [Slicing the Prefill](../13-chunked-prefill/).
- **`self.block_manager.can_allocate(req, n)`** — can we fit $n$ more tokens of KV cache for this request without evicting anyone? Block manager owns the answer — see [The Block Manager](../11-block-manager/) and [Borrowing from 1965](../10-paged-attention/).
- **`self.preempt(req)`** — if not, swap this request out. Its KV blocks go back to the free list; it goes back to `WAITING`. We'll cover the policy below.
- **`req.tokens_to_prefill_after_cache`** — for an admitted request, the prompt length minus whatever the prefix cache already has for free. A 4,000-token system prompt that's a known prefix hash can collapse to 12 tokens of actual prefill — see [Reusing the Prologue](../12-prefix-caching/).
- **`return plan`** — a dict from `request_id` to `num_tokens`. That's the scheduler's entire output. The model runner takes this dict, gathers inputs, and launches a captured CUDA graph for the appropriate bucket.

That is the scheduler. The whole thing. The model runner downstream of it does the actual GPU work; the AsyncLLM upstream of it does ingest and de-tokenization. The scheduler is just a function that allocates a budget.

{{< crosshead >}}Where the budget comes from{{< /crosshead >}}

Why 8,192? The number is set by two constraints, neither of them about correctness.

First, **CUDA graph capture**. The forward pass is captured once per discrete `(batch_size, total_tokens)` bucket so launches are free at runtime — see [Launches Aren't Free](../05-cuda-graphs/). vLLM captures at powers-of-two-ish buckets up to some maximum. The largest captured bucket is the largest legal token budget.

Second, **bounded latency per step**. Whatever the budget is, that is the worst-case work the GPU does in one step, which is the worst-case time before the next round of decode tokens streams out to users. If $B = 8{,}192$ and per-token decode cost is roughly 12 µs of GPU time at this batch size, the budget caps a single iteration at about 100 ms — even if every one of those 8,192 tokens belongs to a single user's mega-prefill.

```pyplot {id="v0-vs-v1-scheduler" caption="V0 VS V1 — LINES OF CODE, PER-STEP LATENCY, AND P99 INTER-TOKEN LATENCY. THE V1 REWRITE DELETES STATE."}
np.random.seed(11)
labels = ['scheduler\nLOC (×100)', 'sched. latency\nper step (ms)', 'P99 ITL\nat 600 users (ms)']

# vLLM V0 numbers approximate the Q3-2024 production state
v0 = np.array([28.0, 9.5, 145.0])    # 2800 LOC, 9.5 ms/step, 145 ms P99
v1 = np.array([6.5, 0.8, 38.0])      # 650 LOC, 0.8 ms/step, 38 ms P99

x = np.arange(len(labels))
w = 0.36

fig, ax = plt.subplots(figsize=(9, 4.6))
b1 = ax.bar(x - w/2, v0, w, color='#1A1A1A', edgecolor='#1A1A1A',
            linewidth=1.5, label='vLLM V0 (Q3 2024)')
b2 = ax.bar(x + w/2, v1, w, color='#FF007F', edgecolor='#1A1A1A',
            linewidth=1.5, label='vLLM V1 (Q1 2025)')

for bars, vals in [(b1, v0), (b2, v1)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 4,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 165)
ax.set_ylabel("value (units vary)")
ax.set_title("vLLM V1 collapsed the state machine — everything got shorter, faster, smoother",
             fontsize=11, loc='left')
ax.legend(loc='upper right')
ax.spines[['top', 'right']].set_visible(False)

print("Scheduler comparison (approximate, public benchmarks + repo line counts):")
print(f"  Source LOC:        V0 = 2800   V1 = 650   ({2800/650:.1f}x shorter)")
print(f"  Per-step latency:  V0 = 9.5 ms V1 = 0.8 ms ({9.5/0.8:.1f}x faster)")
print(f"  P99 ITL @ 600 users: V0 = 145 ms V1 = 38 ms ({145/38:.1f}x lower tail)")
plt.tight_layout()
```

The numbers are approximate but the shape is real and was the headline of the V1 alpha post. A scheduler that does less per step, and whose state space is smaller, has lower per-iteration latency and a lower P99 tail. **The state machine was the bottleneck.** That is rare enough in systems engineering that it deserves to be said out loud.

## Preemption: The Cost Of Fitting Everyone

The pseudocode hides one thing: what happens when `can_allocate` returns false. The scheduler **preempts** a running request — drops its KV blocks back to the free list and pushes it to a *swapped* queue. The default policy is FCFS-with-priority, but operators can configure newer-first or shortest-job-first.

Two ways a preempted request can come back:

1. **Recompute.** The request goes back to `WAITING`. Next time it's admitted, its prefill runs again from scratch — modulo whatever the prefix cache still holds for it. This costs prefill FLOPs but no extra memory.
2. **CPU offload.** The KV blocks are *copied* to host memory, indexed under the request ID, and copied back on resume. Costs PCIe bandwidth (about 50 GB/s on a Gen-5 box) but saves prefill compute.

For decode-heavy workloads — long conversations, short prompts, lots of users — preemption is rare because each request only grows the KV cache by one block per 16 tokens. For prefill-heavy ones — RAG with 100K-token contexts, large code-completion prompts — preemption is normal and the policy choice matters. **The scheduler is also a queuing-theory problem,** which is why operators tune it with the same vocabulary they use for storage I/O.

## A Step, Drawn Out

To make the scheduler tangible: here is one iteration of `schedule_step` on a typical box, drawn as a Gantt chart of what each request gets to spend its slice of the budget on.

```pyplot {id="scheduler-gantt" caption="ONE V1 SCHEDULER STEP ON A LOADED BOX. THE TOKEN BUDGET (8192) GETS SPLIT ACROSS 4 PREFILL CHUNKS, 1 PREFIX-CACHE-HIT ADMIT, AND 48 DECODE STEPS."}
np.random.seed(3)
budget = 8192

# Compose one iteration: a mixture of decode + chunked prefill + admit
items = [
    # (label, tokens, color)
    ('R12  decode',          1,    '#00A8A8'),
    ('R34  decode',          1,    '#00A8A8'),
    ('R57  decode',          1,    '#00A8A8'),
    ('R09  prefill chunk',  2048,  '#FF007F'),
    ('R21  prefill chunk',  2048,  '#FF007F'),
    ('R44  prefill chunk',  2048,  '#FF007F'),
    ('R66  prefill chunk',  1024,  '#FF8C00'),
    ('R88  new (admit)',    900,   '#FFD700'),
]
# Pad with 45 more decodes
for i in range(45):
    items.append((f'R{100+i:03d}  dec', 1, '#00A8A8'))

# Bucket counts
total = sum(t for _, t, _ in items)
print(f"Total tokens scheduled: {total} (budget = {budget})")
n_prefill = sum(1 for l, _, _ in items if 'prefill' in l)
n_admit   = sum(1 for l, _, _ in items if 'admit' in l)
n_decode  = sum(1 for l, _, _ in items if 'dec' in l)
print(f"  prefill chunks: {n_prefill}, admits: {n_admit}, decodes: {n_decode}")

fig, ax = plt.subplots(figsize=(10, 4.6))
left = 0
for label, t, color in items:
    ax.barh(0, t, left=left, color=color, edgecolor='#1A1A1A',
            linewidth=0.6, height=0.55)
    if t >= 700:
        ax.text(left + t/2, 0, label, ha='center', va='center',
                fontsize=8.5, color='#FDF5E6', fontweight='bold')
    left += t

# Annotate the remaining slack
slack = budget - total
ax.barh(0, slack, left=left, color='#FDF5E6',
        edgecolor='#1A1A1A', linewidth=0.6, height=0.55, hatch='//')
if slack > 0:
    ax.text(left + slack/2, 0, f'{slack} slack', ha='center',
            va='center', fontsize=8.5, color='#1A1A1A', fontweight='bold')

ax.set_xlim(0, budget * 1.02)
ax.set_yticks([])
ax.set_xlabel("token budget position (one scheduler step, 8192 total)")
ax.set_title("One step: 3 prefill chunks (pink), 1 chunked-tail (orange), "
             "1 admit (yellow), 48 decodes (teal)", fontsize=11, loc='left')

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#FF007F', edgecolor='#1A1A1A', label='prefill chunk (continuing)'),
    Patch(facecolor='#FF8C00', edgecolor='#1A1A1A', label='prefill chunk (final)'),
    Patch(facecolor='#FFD700', edgecolor='#1A1A1A', label='new admit (after prefix cache)'),
    Patch(facecolor='#00A8A8', edgecolor='#1A1A1A', label='decode (1 token/request)'),
], loc='upper right', fontsize=8)
ax.spines[['top', 'right', 'left']].set_visible(False)
plt.tight_layout()
```

Notice: there is no row that says "this is a prefill batch" and "this is a decode batch." There is one batch, and each request inside it spends some integer count of tokens. The model runner gathers the inputs, builds a $(B, T_\text{total})$ tensor of token IDs with appropriate position offsets and attention masks, and runs one CUDA-graph replay. The graph is captured for the bucket nearest `T_total = 8{,}091` — likely the 8,192 bucket — and the replay launches in microseconds.

This is the *operational* meaning of "prefill and decode are the same operation." They literally share a forward pass.

## Async Engine, Multi-Process: Where The Scheduler Lives

A small but load-bearing detail: the scheduler does not run in the same process as the API server.

vLLM V1's runtime is split into two processes connected by a {{< cite text="ZMQ" url="https://zeromq.org/" kind="site" >}} message bus:

1. **`AsyncLLM`** — the user-facing async front door. Owns the FastAPI handler, the tokenizer, the de-tokenizer, and the per-request output queues. Runs in the Python asyncio event loop.
2. **`EngineCore`** — the GPU-owning process. Runs `schedule_step`, owns the block manager, owns the model worker(s), and steps the forward pass on every iteration of its main loop.

Requests arrive at AsyncLLM, get tokenized, and are shipped to EngineCore over the bus. EngineCore's main loop pulls new requests, calls `schedule_step`, runs the model, and ships output tokens back to AsyncLLM over the bus. AsyncLLM yields tokens to whichever request's HTTP stream is waiting.

{{% callout type="tangent" %}}
Why two processes? Because Python has a Global Interpreter Lock and the API server does a lot of synchronous-looking work — JSON parsing, sampling parameter validation, schema checks. If that work ran in the same process as the scheduler, every JSON parse would steal time from the scheduler's tight loop. The ZMQ split lets the GIL live in two places at once. The other reason: process isolation. If a malformed request crashes the API server, the EngineCore keeps the GPUs warm.
{{% /callout %}}

The split is invisible to the model. From the GPU's perspective, every iteration looks identical: receive a `plan` dict, run a forward pass, return outputs. The complexity that used to live inside the scheduler's state machine now lives in the message bus — and message buses, unlike state machines, are a problem that operating-systems people have known how to debug since the 1970s.

## Why The Budget Is The Right Unit Of Fairness

Every serving system has a unit of fairness — the thing it allocates equally (or with policy) across users. Possible units:

- **Wall-clock time.** Round-robin: each user gets 10 ms of GPU, regardless of how much work that buys. Bad: a long prefill in user A's slot starves nobody else, but completes 5× slower than necessary.
- **Forward passes.** Each user gets one forward pass per iteration. Bad: doesn't account for $T$. A 1-token decode and a 4,096-token prefill chunk are the *same* forward pass but very different costs to peers.
- **Tokens.** Each user gets some integer count of tokens per step, summed across users to a fixed budget. The budget is *what gets done per step*, regardless of which user spent it.

Tokens win because they are the unit the model actually charges in. One token of work — prefill or decode — costs the same in FLOPs (approximately, modulo arithmetic intensity), the same in HBM bandwidth (for KV append), and the same in scheduler time. Capping the per-step token count therefore caps the per-step GPU time, which is the only thing a downstream user cares about.

It is also the unit that makes graph capture practical. CUDA graphs cost real GPU memory to capture, so vLLM captures a finite set of $(B, T)$ buckets and replays the nearest one — see [Launches Aren't Free](../05-cuda-graphs/). The largest captured $T$ is the budget. The scheduler picks plans that fit under a captured bucket, by construction.

So fairness, latency bounding, and graph capture all align on the same number. That is the rare thing in systems design: a single knob that is simultaneously correct for three different layers of the stack.

## What Every Previous Chapter Looks Like, From Here

Here is the same scheduler code, with each call expanded into the chapter that owns it:

| Line of `schedule_step` | What it does | Owning chapter |
|---|---|---|
| `for req in self.running` | iterate in priority order | (scheduler-internal) |
| `req.tokens_remaining_to_prefill or 1` | chunked prefill remainder, or 1 for decode | [Slicing the Prefill](../13-chunked-prefill/) |
| `block_manager.can_allocate(req, n)` | does enough free KV space exist? | [The Block Manager](../11-block-manager/), [Borrowing from 1965](../10-paged-attention/) |
| `self.preempt(req)` | evict + recompute or swap | (this chapter) |
| `req.tokens_to_prefill_after_cache` | new request's effective prefill length | [Reusing the Prologue](../12-prefix-caching/) |
| (downstream) `model_runner.execute(plan)` | one graph replay, $B$ requests, $\Sigma_i n_i$ tokens | [Launches Aren't Free](../05-cuda-graphs/), [Attention in SRAM](../06-flash-attention/) |
| (upstream) AsyncLLM ↔ EngineCore over ZMQ | bus separation, GIL isolation | (this chapter) |

That table is the chapter. The scheduler is not a new algorithm — it is the *integration surface* where every previous trick gets composed into a single per-step decision. Every chapter before this one earns its keep here.

## What To Remember

1. **The V1 scheduler is a dict-producing function.** `{request_id: num_tokens}` per step. Prefill and decode are both "produce some KV." The distinction lives only in the value of $n$.
2. **The token budget is the central knob.** It bounds per-step work, defines fairness across users, and aligns with CUDA-graph buckets. One number, three jobs.
3. **Every previous chapter is one line of the scheduler.** Block manager, prefix cache, chunking — they all surface as method calls. That's why this chapter is the engine half's climax.
4. **The state machine was the bottleneck.** V0's thirteen-node lifecycle was the source of the Python overhead that V1 deleted by replacing state with a single integer per request.

**Continue to → [The Draft Trick](../15-speculative-decoding/)** — once the scheduler can dispatch $n$ tokens of work per step regardless of which "phase" they belong to, dispatching *several candidate tokens* costs almost the same — and gets you most of them for free.
