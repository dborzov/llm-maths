---
title: "Slicing The Prefill"
description: "A 100K-token prefill can monopolize the GPU for seconds, wrecking inter-token latency for every other user in the batch. Chunked prefill slices long prompts into token-budget-sized pieces that interleave with decode steps, giving every user bounded and predictable time-to-first-token."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T15:00:00-04:00
issue: 8
weight: 130
techKind: mainline
techNode: chunked-prefill
header: 13-chunked-prefill.webp
---

## The 9 PM Pager

Imagine you are on call for an LLM serving cluster. It is **September 2023**. The product runs a chat endpoint, and on Tuesday at 9 PM the monitoring dashboard turns red. Five hundred users are mid-conversation, expecting tokens every $30$ ms. For five seconds, every conversation in the entire cluster *freezes*. No tokens. No streaming. Then everything resumes. The pager goes off; you open the trace and find this.

User A sent a request thirty seconds ago to summarize a $64{,}000$-token document. The request is now in prefill. Prefill is compute-bound and scales linearly in input length, and at $64$K tokens through a 70B model it takes **roughly five seconds** of GPU time. The scheduler, doing what it was designed to do, picked the next batch by maximizing GPU occupancy. User A's giant prefill landed in that batch alongside 32 other users' decode steps. The whole batch runs at the speed of its slowest member, which is the prefill, which is 5 seconds. Every decode-step user in the batch — Users B through Z — is silently being held back from their next token for *five entire seconds*.

This is a tail-latency catastrophe. The throughput dashboard looks fine; the p99 latency dashboard is on fire. Five hundred individual user experiences just got broken by one inconsiderate big prompt.

The fix, which Microsoft Research India would crystallize a few months later in the {{< cite text="SARATHI paper (Agrawal et al., 2023; ASPLOS 2024)" url="https://arxiv.org/abs/2308.16369" kind="paper" >}} and which vLLM and SGLang would adopt within a release cycle, is conceptually one sentence: **chop the prefill into bite-sized pieces and interleave them with everyone else's decode steps.** It works because of an algebraic property of attention that most people who use the kernel daily never explicitly notice. Let us pull it out.

## The Naive Picture, Quantified

To feel how bad the naive scheduler is, hold the workload constant — one big prefill plus a bunch of in-flight decodes — and watch the per-user inter-token latency (ITL) as the prefill length grows.

```pyplot {id="naive-vs-chunked-tail" caption="p99 inter-token latency for decode-mode users as one big prefill arrives. Without chunking (pink) the tail explodes hockey-stick-style. With chunked prefill (teal) every step caps at the chunk budget, so tail latency stays flat."}
prefill_lens = np.arange(0, 65001, 1000)

# Naive: one prefill step is single-batch-blocking. ITL ~ prefill_time + decode.
us_per_prefill_token = 75.0   # roughly: 0.075 ms/token through a 70B FP8 model
decode_step_us = 25_000        # 25 ms per decode step
naive_tail_us = decode_step_us + prefill_lens * us_per_prefill_token

# Chunked: prefill split into chunks of `chunk_size`. Max single-step cost
# is bounded by chunk_size * us_per_prefill_token + decode overhead.
chunk_size = 2048
chunked_tail_us = decode_step_us + chunk_size * us_per_prefill_token + 0*prefill_lens

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(prefill_lens, naive_tail_us / 1000, color='#FF007F', linewidth=2.6,
        label='naive scheduler')
ax.plot(prefill_lens, chunked_tail_us / 1000, color='#00A8A8', linewidth=2.6,
        label=f'chunked prefill ({chunk_size}-token chunks)')

ax.axhline(150, color='#1A1A1A', linestyle='--', linewidth=1.0)
ax.text(2000, 165, "150 ms 'streaming feels smooth' threshold",
        fontsize=9, color='#1A1A1A')

ax.set_xlabel("Length of co-batched prefill (tokens)")
ax.set_ylabel("p99 inter-token latency for everyone else (ms)")
ax.set_title("One inconsiderate prefill ruins everyone's stream — until you chunk it")
ax.legend(loc='upper left')
ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, 5500)

for L in [1024, 8192, 32768, 64000]:
    nt = (decode_step_us + L * us_per_prefill_token) / 1000
    ct = (decode_step_us + chunk_size * us_per_prefill_token) / 1000
    print(f"prefill {L:>6d} tokens:  naive p99 ITL ≈ {nt:>7.1f} ms   chunked ≈ {ct:>6.1f} ms")
print()
print("The chunked column is constant. That is the whole point.")
```

The picture is unambiguous. Without chunking, p99 ITL grows linearly with the largest prefill that happens to be in the batch — meaning your service-level objective is at the mercy of your most demanding user. With chunking it is bounded by the chunk size, which is a configurable knob.

What makes this work is not a clever batching trick. It is an algebraic property of the prefill operation itself.

## The Associativity That Saves Us

A standard prefill computes the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} for tokens $0, 1, \ldots, T-1$ by running a single forward pass with $T$ tokens of context. The output is, layer by layer, the K and V vectors at every position.

The crucial observation: that forward pass *can be split along the token axis*. We can run a prefill over tokens $[0, T_1)$, write its KV vectors into the cache, then run a second prefill over tokens $[T_1, T_2)$ where the new tokens are the queries and the *already-computed* KV blocks from the first chunk are the keys and values. The attention output is mathematically identical to what a single $[0, T_2)$ prefill would have produced.

$$
\text{prefill}([0, T_2)) \;\equiv\; \text{prefill}([T_1, T_2) \mid \text{KV}([0, T_1)))
$$

This is just the fact that causal attention is left-associative over the sequence axis. Tokens only attend backwards. The K and V for token $T_1$ never depended on whether tokens after it were going to be revealed in the same kernel launch or a different one. The chunked computation is bit-identical to the single-shot computation up to floating-point reassociation, which for FP16/BF16 attention scores is in the noise.

{{% callout type="theorem" %}}
**Associativity of prefill over the token axis.** For a causal transformer, computing the KV cache for tokens $[0, T)$ as one $T$-token prefill, or as a sequence of chunks $[0, T_1), [T_1, T_2), \ldots, [T_{k-1}, T)$ where each chunk attends to all previously-cached KV, produces mathematically identical KV vectors.

This is a *free* property — no algorithmic change, no calibration, no quality loss. It just falls out of causality.
{{% /callout %}}

The implication: if your attention kernel can accept *"the queries are a chunk; the keys and values are this chunk plus everything previously cached,"* you can split a prefill anywhere you like. And — happily — the {{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}}-plus-paged kernel from [PagedAttention](../10-paged-attention/) already does exactly that. Each forward pass takes a query tensor of shape $[\text{n\_query\_tokens}, \ldots]$ and a block table pointing at all previously written KV blocks. Whether `n_query_tokens` is 1 (decode), 16 (a tiny chunk), 2,048 (a normal prefill chunk), or 65,000 (a full prefill) is a runtime parameter, not a kernel decision.

## What A Mixed Step Looks Like

Once chunking is on the table, every scheduler step becomes a *mix* of work: some requests contribute one query token (those are decodes), some contribute a chunk of $C$ query tokens (those are prefills mid-flight). The attention kernel processes them as one fused batch. The step has a single new quantity to budget:

$$
\text{token\_budget}(\text{step}) \;=\; \underbrace{C}_{\substack{\text{prefill}\\\text{chunk}}} \;+\; \underbrace{n_{\text{decoding}}}_{\substack{\text{one token}\\\text{per active request}}}
$$

Pick $C$ such that the total step cost — measured in HBM-bound work for the decodes and in compute-bound work for the chunk — fits in the target step duration. vLLM exposes this as the configuration knob `long_prefill_token_threshold`, typically $2048$ to $8192$. Below the threshold a prefill goes through whole; above, it gets sliced into threshold-sized pieces.

```pyplot {id="chunked-timeline" caption="Wall-clock comparison: naive scheduler blocks Users B through F while User A's 64K prefill runs. Chunked scheduler interleaves 32 chunks of A with B-F's decodes; every user keeps streaming."}
np.random.seed(1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)

T_max = 5000  # ms of wall clock to draw

# === Naive scheduler ===
# A's 64K prefill = one big 5000 ms block.
# B-F each do decode steps, but they're frozen during A's prefill.
labels = ['A: 64K prefill', 'B: decode', 'C: decode', 'D: decode', 'E: decode', 'F: decode']
colors_a = ['#FF007F'] + ['#FFD700']*5

# A: one giant prefill spanning 0-5000 ms
ax1.barh(0, 5000, left=0, height=0.6, color='#FF007F', edgecolor='#1A1A1A')
ax1.text(2500, 0, "A: 64K prefill  (5000 ms — everything else waits)",
         ha='center', va='center', fontsize=9, color='#1A1A1A', fontweight='bold')

# B-F: their decode steps cannot run while A's prefill is active.
# They resume after A completes.
for i in range(1, 6):
    # one quick step at t≈0, then a 5000 ms gap, then resumed decodes
    ax1.barh(i, 30, left=0, height=0.5, color='#FFD700', edgecolor='#1A1A1A')
    # the gap (visual emphasis)
    ax1.barh(i, 4970, left=30, height=0.5, color='#FDF5E6',
             edgecolor='#FF007F', linewidth=0.6, hatch='///')
    # one resumed step (just for visual)
    ax1.barh(i, 25, left=5000, height=0.5, color='#FFD700', edgecolor='#1A1A1A')

ax1.set_yticks(range(6))
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlim(0, T_max + 200)
ax1.set_title("NAIVE: one big prefill blocks every decode for 5 seconds", fontsize=10, loc='left', fontweight='bold')
ax1.spines[['top','right']].set_visible(False)

# === Chunked scheduler ===
# A's 64K prefill = 32 chunks of 2K. Each chunk ~150 ms.
# B-F decode steps run inside each chunk.
chunk_count = 32
chunk_ms = 150

# Each step has one chunk from A plus 5 decode tokens from B-F.
for step in range(chunk_count):
    t0 = step * chunk_ms
    # A's chunk
    ax2.barh(0, chunk_ms - 10, left=t0, height=0.5,
             color='#FF007F', edgecolor='#1A1A1A', linewidth=0.4)
    # B-F decodes (interleaved in same step)
    for i in range(1, 6):
        ax2.barh(i, chunk_ms - 10, left=t0, height=0.5,
                 color='#FFD700', edgecolor='#1A1A1A', linewidth=0.4)

# Last A chunk completes at 32*150 = 4800 ms
ax2.text(2400, 0.5, "A's 32 prefill chunks of 2K each (4800 ms total)",
         ha='center', va='bottom', fontsize=8, color='#1A1A1A', style='italic')

ax2.set_yticks(range(6))
ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlim(0, T_max + 200)
ax2.set_xlabel("Wall-clock time (ms)")
ax2.set_title("CHUNKED: A's 64K split into 32 chunks of 2K; every user keeps streaming", fontsize=10, loc='left', fontweight='bold')
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()

print("Naive scheduler:")
print(f"  A's prefill takes 5000 ms, blocks 5 other users for 5000 ms each.")
print(f"  p99 ITL for B-F: ~5000 ms.")
print()
print("Chunked scheduler:")
print(f"  A's prefill takes 32 * 150 = 4800 ms (slightly more total wall time).")
print(f"  B-F see an ITL of ≤ 150 ms throughout. Smooth stream.")
print()
print("The trade is: A's TTFT slightly worse, everyone else's ITL dramatically better.")
```

Read the two timelines as the same physical resources (one GPU, one engine) spent two different ways. The naive scheduler gives User A their answer slightly sooner but at the cost of stalling everyone else. The chunked scheduler costs User A a few hundred milliseconds of extra wall-clock and saves all the other users from a five-second freeze. In every production deployment that has measured this trade, the second choice wins by miles.

{{% pullquote type="technical" %}}
The scheduler's job isn't to minimize any one user's latency. It is to keep p99 across the whole cluster inside the SLO. Chunked prefill is what makes that p99 bound by your token budget instead of by your worst user's prompt.
{{% /pullquote %}}

## Picking The Chunk Size

The chunk size $C$ trades two things against each other.

**Smaller $C$** $\Rightarrow$ finer-grained interleaving, lower tail ITL, but more kernel launches per unit of prefill (each chunk pays a fixed kernel-launch and Python-scheduler overhead), and lower arithmetic intensity per step.

**Larger $C$** $\Rightarrow$ fewer kernel launches, higher arithmetic intensity (closer to the roofline ridge point — see [roofline](../04-roofline/)), better prefill throughput, but a worse worst-case ITL for co-batched decodes.

The right value depends on hardware and target ITL. On an H100 with a target ITL of $100$ ms, $C \approx 2{,}048$ is a sweet spot. On an H200 (twice the HBM bandwidth, slightly more compute), $C \approx 4{,}096$ is fine. If you are running on a smaller GPU with a tighter ITL budget, $C$ might drop to $512$.

{{% marginnote %}}A useful mental model: $C$ is the answer to "how many query tokens does it take to saturate the attention kernel's arithmetic intensity?" Below that, you're underutilizing the GPU per chunk. Above that, you're paying tail-latency tax for no throughput gain.{{% /marginnote %}}

One more wrinkle: the relevant per-step cost isn't really $C$ alone. It is $C$ *times the current context length being attended to*, because each query token attends to every cached KV token. At $C = 2{,}048$ and a context of $64{,}000$, the attention math is $2{,}048 \times 64{,}000 \approx 131$ M score computations per layer per head — and that grows as the prefill progresses. Some implementations adapt $C$ downward as the context grows to keep the per-step cost flat. vLLM's default just picks a conservative $C$ that's safe across the whole sequence.

## When To Disable Chunking

Chunking is not free, and there are workloads where it is a small loss. Three cases.

**1. Short prompts.** A request whose prompt is below the chunk threshold is one chunk regardless — chunking adds nothing but a tiny bit of bookkeeping overhead. vLLM correctly skips chunking when the remaining prefill tokens fit in one chunk.

**2. Already-cached prefixes.** If the request hits a prefix cache (see [Reusing the Prologue](../12-prefix-caching/)) and only a 30-token tail needs fresh prefill, the tail is below threshold and runs as a single chunk. The two optimizations compose cleanly: prefix caching shrinks the prefill, chunking shrinks any prefill that's still too big.

**3. Latency-critical single-request workloads.** If you are running an isolated job — one user, one prompt, batch size 1, optimizing for TTFT — there is nobody else in the batch to share the GPU with, so chunking just adds per-chunk overhead with no benefit. The right call here is no chunking, max arithmetic intensity, finish as fast as possible. vLLM lets you set the threshold high enough to disable it.

The general rule: enable chunking when you are serving multi-tenant workloads and care about p99 ITL. Disable it when you are running a benchmark or a single-user job and care only about that one user's wall-clock.

## The Quiet Unification

Once chunked prefill exists, something subtle happens to the architecture of the scheduler. A "prefill step" and a "decode step" used to be distinct things, run by distinct code paths, scheduled by separate policies. With chunked prefill, every step is just *a token budget allocated across active requests* — some requests contribute one token (decoding), others contribute $C$ tokens (mid-flight prefill). The kernel doesn't care; the scheduler doesn't care; the block manager doesn't care. The whole prefill-vs-decode distinction softens into "how many tokens does this request want this step?"

This is precisely the abstraction the [V1 unified scheduler](../14-scheduler/) is built on. The scheduler's data structure is a dictionary `{request_id: num_tokens}`, and the prefill/decode dichotomy quietly stops being a thing. Chunked prefill is the *operational reason* that unification works — without it, prefills are still discrete monolithic events that the scheduler has to special-case.

## What To Remember

1. **Prefill is associative over the token axis.** A $T$-token prefill is equivalent to a sequence of chunked prefills, each attending to the growing KV cache. Causality buys you this for free; FlashAttention-plus-paged already supports it as a runtime parameter.
2. **One long request can starve everyone else.** Without chunking, p99 inter-token latency is bounded by your largest co-batched prefill. Chunked prefill caps it at your chunk-size budget instead, regardless of how long any one user's prompt is.
3. **The chunk size is a knob, not a constant.** Smaller chunks give finer interleaving at the cost of kernel-launch overhead and per-step arithmetic intensity. Tune it for your hardware and your SLO; the right value is a few thousand tokens on current-generation accelerators.

**Continue to → [The Token Budget](../14-scheduler/)** — once chunking exists, the prefill/decode boundary stops being interesting and the scheduler treats the entire workload as one stream of tokens to dispense.
