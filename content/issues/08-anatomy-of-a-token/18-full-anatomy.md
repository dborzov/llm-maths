---
title: "Full Anatomy: 6.1 seconds, every layer labelled"
short_title: "Full Anatomy"
description: "The cold-open trace walked through a second time — same packet, same H200 box, same 280 ms to first character — but now every layer is named, every chapter cashed in, and every latency budget itemized from TCP arrival to streamed token."
blurb:
  - "280 ms to first character; 6.1 s for a complete four-paragraph answer; 793 users sharing the same box the whole time."
  - "The annotated timeline shows where each millisecond goes: tokenizer, scheduler, block manager, prefill kernel, decode loop, sampler, HTTP stream."
  - "Every chapter from the issue appears as a labelled node in the final picture — the boss collects all seventeen debts."
  - "The architecture that delivers this was non-existent five years ago; most of it didn't exist three years ago."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T17:30:00-04:00
issue: 8
weight: 180
techKind: boss
techNode: full-anatomy
header: 18-full-anatomy.webp
---

## 4:12 PM, Some Tuesday

Anya is on a bus. Her phone is unlocked, balanced on her knee. She has been reading a Wikipedia article about Diocletian's reforms and a question has occurred to her, the kind of question that arrives whole and demands an answer right now. She thumbs over to the ChatGPT app, taps the input box, and types:

> *Why did the Roman Empire fall?*

She hits send at exactly **4:12:00.000 PM**, local time. The TCP packet leaves her phone, hops across two cell towers, terminates at an HTTPS load balancer in some hyperscaler's us-east region, and arrives at the door of an 8×H200 server. She is user **#347** on this box at this moment. Eight hundred others are doing essentially the same thing, on this same box.

At **4:12:06.100 PM**, six point one seconds later, the streamed reply has finished. Anya has read a fluent four-paragraph answer covering monetary debasement, the Antonine plague, the bifurcation under Diocletian, and the long migration period. The first character of that reply arrived at her phone at **4:12:00.280 PM** — two hundred and eighty milliseconds after she tapped send.

This chapter is about those 6.1 seconds.

In [the cold open](../01-cold-open/) we walked through them at the level of *names*, with every component listed and almost nothing explained. The intervening seventeen chapters have explained everything. Now we walk through Anya's request one more time, fully annotated. Every link in every chapter should light up. Every primer should pay off. The whole tech tree — every yellow, pink, and cream node from the cover — collapses, at the end, into a single picture: **the life of one token, from packet to phoneme, with everything labelled**.

This is the boss.

## The Annotated Timeline

Each step below gives a wall-clock budget, the component doing the work, the chapter that explains it, the physical action, and one sentence on why this step exists in the architecture at all.

### t = 0.000 ms — TCP arrives at the edge

The HTTPS connection has been alive for an hour (Anya's app uses HTTP/2 with a long-lived stream). The new request is a `POST /v1/chat/completions` with a small JSON body. The TLS termination happens at the edge load balancer; the cleartext request hops over the data-center backbone and lands at vLLM's OpenAI-compatible HTTP server inside the engine container.

**Why this step exists.** Out of scope for the inference stack — but worth naming. Everything below is what happens *inside the box* once the HTTP layer has done its job.

### t = 0.3 ms — Tokenizer + AsyncLLM ingest

The HTTP server parses the JSON, finds the message text, and runs the **BPE tokenizer** ({{< wiki "embeddings" >}}tokenization{{< /wiki >}} is byte-pair encoding). Forty-seven bytes of UTF-8 become **12 token ids**. The first eight of those — the system prompt header, identical for every chat on this deployment — match a hash key the engine has seen ten million times today.

`AsyncLLM` wraps the request in a `Request` object and ships it across a ZMQ socket to the **EngineCore** process. Two processes, not one, because the GIL would otherwise serialize everything in CPython. The IPC takes maybe 0.2 ms round-trip.

**Why this step exists.** The HTTP server is a frontend that needs to talk to many concurrent users; the engine is a backend that needs to drive the GPU at near-peak utilization with zero stalls. Splitting them across processes lets each do its job without stepping on the other.

### t = 0.7 ms — Engine ingests the request

The `Request` object enters the `waiting` queue. The engine notes its arrival, records the timestamp for SLA bookkeeping, and proceeds to the next scheduler step.

### t = 1.0 ms — Scheduler step ([ch.14](../14-scheduler/))

The V1 unified scheduler examines its state at the start of step number 4,201,973. Currently running: **53 in-flight requests**. Their next-step token totals already sum to roughly 270 (most are decoding one token apiece; a few are mid-chunk on chunked prefills). The token budget for this step is **8,192 tokens**.

The scheduler walks the `waiting` queue. It finds Anya. It calls the **block manager** ([ch.11](../11-block-manager/)) to ask: how many of Anya's 12 prompt tokens are already cached?

The block manager hashes Anya's prompt prefix-by-prefix against the **prefix cache** ([ch.12](../12-prefix-caching/)). The first 8 tokens match a system-prompt entry that has been resident in HBM since 6 AM. *Hit.* Anya's request inherits a one-block prefix from that cache entry; her refcount on that block goes from 9,847 to 9,848.

The remaining **4 tokens** need real prefill compute. The scheduler tags her request with `(anya, num_tokens=4)` and ships her into this step's batch.

The step plan emitted to the model runner looks roughly like this:

```python
# Schedule emitted at t = 1.0 ms
step_plan = {
    "req_028":  4,    # Anya — 4 fresh prefill tokens
    "req_011":  1,    # decode, mid-stream
    "req_012":  1,    # decode
    "req_017":  512,  # chunked prefill — chunk 3 of 7
    "req_023":  1,    # decode
    # ... 50 more
}
sum(step_plan.values())  # ≈ 1,200 tokens, well under the 8,192 budget
```

This is **the V1 insight** in its purest form. Prefill and decode are the same operation, parameterized by `num_tokens`. The scheduler stops caring about which is which; it just balances a token budget. We unpacked this in [The Token Budget](../14-scheduler/) — re-read it now if it didn't click the first time, because the rest of the timeline assumes it.

**Why this step exists.** Old serving stacks scheduled at the request level: a whole prefill, then a whole decode loop. The V1 scheduler operates at the token level, which is the level at which the GPU actually computes. The gap between scheduler abstraction and hardware reality is what continuous batching ([ch.8](../08-continuous-batching/)) and chunked prefill ([ch.13](../13-chunked-prefill/)) work so hard to close.

### t = 1.3 ms — Model runner assembles inputs

For each of the **4 tensor-parallel ranks** ([ch.16](../16-tp-pp/)) holding a slice of the Llama-3-70B weights, the runner builds the input tensor. It looks up the bucket size from the CUDA-graph table ([ch.5](../05-cuda-graphs/)) — the captured graphs handle batches of size 1, 2, 4, 8, 16, ..., 256 — and pads up to the next size. The total token count for this step is ~1,200, so the runner picks the **2048-token graph**.

Block tables for every active request are packed into a single contiguous tensor. Anya's block table has one entry (the prefix-cache block) plus one entry pointing to her freshly allocated **prefill block** in the paged KV pool. The pool's free queue handed out block 184,033 in 8 microseconds.

**Why this step exists.** CUDA graphs require static shapes; the inference workload has dynamic shapes. The bucketing trick is the compromise — pre-capture a small set of static shapes, pad up to the next one, eat a few percent of FLOPs as the price of avoiding millions of per-step kernel-launch overheads.

### t = 1.5 ms — Forward pass starts (CUDA graph replay)

A single driver call submits the entire captured CUDA graph: **80 layers × ~12 kernels per layer = ~960 kernels** queued onto the GPU stream in one launch. CPU overhead for the whole forward pass: roughly **200 microseconds**, not 2 milliseconds. The chapter on this is [The Conductor's Wand](../05-cuda-graphs/).

Inside each layer, on each TP rank, the kernel sequence is:

1. {{< wiki "normalization" >}}RMSNorm{{< /wiki >}} on the {{< wiki "residual-stream" >}}residual stream{{< /wiki >}}.
2. Column-parallel QKV {{< wiki "transformer-weights" >}}matmul{{< /wiki >}} ([ch.16](../16-tp-pp/)) — each rank computes its slice of Q, K, V.
3. {{< wiki "rope" >}}RoPE{{< /wiki >}} positional rotation on Q and K.
4. **PagedFlashAttention** ([ch.6](../06-flash-attention/), [ch.10](../10-paged-attention/)) — the fused attention kernel reads K and V through Anya's block table; the score matrix never lands in HBM; the {{< wiki "softmax" >}}softmax{{< /wiki >}} is computed online.
5. Row-parallel output projection; **all-reduce** across the TP group (one NCCL call, ~30 µs on NVLink).
6. RMSNorm.
7. {{< wiki "mlp-block" >}}MLP block{{< /wiki >}}: column-parallel up-projection, SiLU {{< wiki "activations" >}}activation{{< /wiki >}}, row-parallel down-projection; all-reduce.

Anya's 4 prefill tokens flow through this same kernel sequence as everyone else's decode tokens. They sit in the same batch dimension, indexed by the same block tables. The kernel cannot tell which tokens are prefill and which are decode — only the scheduler knew.

```pyplot {id="signature-swimlane" caption="The signature illustration: Anya's request as it traverses the inference stack from t=0 to t=6.1s. Each swimlane is a component; each badge is the chapter where it was explained. Time is on the y-axis (log scale). This is the whole issue in one picture."}
np.random.seed(0)

# (start_ms, end_ms, lane, label, chapter, color)
events = [
    (0.0,    0.3,  0,  "TCP + HTTPS",          "edge",   '#1A1A1A'),
    (0.3,    0.7,  1,  "tokenize + AsyncLLM",  "ch.1",   '#FF8C00'),
    (0.7,    1.0,  2,  "engine waiting queue", "ch.14",  '#FF8C00'),
    (1.0,    1.3,  3,  "scheduler + block mgr","ch.14/11/12",  '#FFD700'),
    (1.3,    1.5,  4,  "runner + CUDA graph",  "ch.5",   '#FFD700'),
    (1.5,    270,  5,  "forward pass (prefill)","ch.6/10/16", '#FF007F'),
    (270,    275,  6,  "logits + sample",      "ch.7",   '#FF007F'),
    (275,    280,  7,  "stream first token",   "ch.1",   '#FF8C00'),
    (280,    320,  8,  "decode step 1+spec",   "ch.8/15",'#00A8A8'),
    (320,    6000, 9,  "decode loop (×80)",    "ch.8/15",'#00A8A8'),
    (6000,   6080, 10, "<|eot|> + cleanup",    "ch.11",  '#FF8C00'),
    (6080,   6100, 11, "final HTTP chunk",     "edge",   '#1A1A1A'),
]

lane_labels = [
    "edge / HTTP",
    "tokenize",
    "engine queue",
    "scheduler",
    "model runner",
    "PREFILL  forward",
    "sampler",
    "stream  → phone",
    "DECODE  step 1",
    "decode loop",
    "EOT  cleanup",
    "final chunk",
]

fig, ax = plt.subplots(figsize=(11, 8.5))
for (start, end, lane, label, chapter, colour) in events:
    s = max(start, 0.05)  # log-safe
    ax.barh(lane, end - start, left=s, height=0.65,
            color=colour, edgecolor='#1A1A1A', linewidth=1.0)
    mid = np.sqrt(s * end) if end > 0.1 else (s + end) / 2
    txt = f"{label}   [{chapter}]"
    ax.text(mid, lane, txt, ha='center', va='center',
            fontsize=8.2, color='#FDF5E6' if colour != '#FFD700' else '#1A1A1A',
            fontweight='bold')

# Mark key wall-clock moments
markers = [
    (0.3,   "tokenized"),
    (1.5,   "fwd starts"),
    (280,   "TTFT 280 ms"),
    (6100,  "EOT 6.1 s"),
]
for t, label in markers:
    ax.axvline(t, color='#1A1A1A', linewidth=0.5, alpha=0.5, linestyle=':')
    ax.text(t, len(lane_labels) - 0.3, label, rotation=90,
            fontsize=8, ha='right', va='top', color='#1A1A1A')

ax.set_yticks(range(len(lane_labels)))
ax.set_yticklabels(lane_labels, fontsize=9)
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlim(0.05, 12000)
ax.set_xlabel("wall clock from t=0  (ms, log scale)")
ax.set_title("Anya's request  —  the full anatomy of a token",
             fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.15)
plt.tight_layout()

# Sanity print: cumulative latency budget
print("Wall-clock landmarks for Anya's request:")
print(f"  t = 0.3   ms   tokenized")
print(f"  t = 1.5   ms   forward pass begins")
print(f"  t = 280   ms   TTFT — first reply token reaches phone")
print(f"  t = 320   ms   first decode step done (3 tokens via spec)")
print(f"  t = 6100  ms   <|eot|> — 247 tokens emitted, request done")
print(f"  total response: {247} tokens in {6.1:.1f} s = {247/6.1:.0f} tok/s")
```

The diagram above is the signature visualization. Print it, pin it to a wall, and the rest of this chapter is just narrating it.

### t ≈ 6 ms — Optional: layer-1 KV streams to a decode peer

In a deployment that runs disaggregated prefill/decode ([ch.17](../17-disagg-pd/)), Anya's prefill is happening on a prefill-class machine, not the same machine that will run her decode loop. The moment layer 1's K and V are computed, the engine fires a NIXL one-sided RDMA write to ship those blocks to the decode peer. By the time layer 80's compute finishes at t = 270 ms, the transfer is essentially done. The handoff is invisible to the user.

If the deployment is co-located instead, this step doesn't exist — the KV cache stays in HBM on the same GPU that will decode. In Anya's case, this particular cluster is *co-located* because the model is "only" Llama-70B and the workload mix on this box leans short-prompt. The router decided on co-location based on the prompt length.

### t = 270 ms — Prefill forward pass completes

Eighty layers of TP-parallel matmul plus PagedFlashAttention plus all-reduce plus MLP, for a batch of ~1,200 tokens, runs at near-peak FP8 throughput. The prefill portion of Anya's 4 tokens cost the GPU maybe 50 microseconds of pure compute, sliced into the broader step. Most of the wall-clock 270 ms was *waiting for the step to finish for all the other users*, plus warmup of the captured CUDA-graph buckets.

### t = 275 ms — Logits + sampling

The logits for Anya's last prefill position — a $50{,}000$-element {{< wiki "logit" >}}logit vector{{< /wiki >}} over the vocabulary — arrive at EngineCore. Top-p sampling with $p = 0.9$ picks the first reply token. It's almost always `"The"`. The id is detokenized.

### t = 280 ms — First token streams to Anya's phone

The text fragment goes back through AsyncLLM → HTTP/2 SSE chunk → load balancer → cell tower → Anya's phone. Her UI renders the `T`. **TTFT = 280 ms.**

Anya does not consciously notice the latency. Anything under ~500 ms feels instant. Anything over 1 s feels like the model is "thinking". The whole industry has organized itself around defending the 500 ms ceiling.

### t = 280..320 ms — First decode step with speculative decoding

Anya's request joins the **decode set** ([ch.7](../07-prefill-vs-decode/), [ch.8](../08-continuous-batching/)). The scheduler emits `{anya: 1, others: ...}` for the next step. But before the target model runs, the **draft model** ([ch.15](../15-speculative-decoding/)) produces 4 candidate tokens autoregressively. The target Llama-70B then verifies all 4 in a *single memory pass* — because decode is bandwidth-bound, verifying 1 token vs 4 candidates costs effectively the same.

Acceptance for this step: 3 out of 4 candidates match. The fourth diverges; the target emits its own choice for that position. Net tokens produced this step: **3**. Average ITL since spec decode landed in production has dropped from ~45 ms to ~28 ms — a 1.6× speedup, for the cost of running a 1B draft model alongside the 70B target.

### t = 320..6000 ms — The stream

For the next 5.7 seconds, Anya's request lives in the decode set. Eighty more steps; ~247 total tokens. Each step is one CUDA-graph replay, one all-reduce per layer, one PagedFlashAttention call. Each step takes ~28 ms on average (some steps emit 1 token, some emit 4 from spec decode).

Three notable things happen during the stream:

- **t ≈ 800 ms.** Anya's KV cache crosses a block boundary. Her 16th decoded token fills the second block; the block manager's free queue hands out **block 184,621** in a few microseconds. The block table for her request now has 3 entries.
- **t ≈ 2.4 s.** The free queue runs low (some other user just allocated a long-prefill burst). The scheduler **preempts** a long-idle decode request — one whose user closed the browser tab 90 seconds ago but whose connection hadn't yet timed out — and returns its blocks to the pool. Anya doesn't see this; she just keeps streaming.
- **t ≈ 4.1 s.** A new user, #1,094, lands on this box with a 32K-token RAG prompt. In an old co-located stack this would have spiked Anya's ITL for hundreds of milliseconds. With **chunked prefill** ([ch.13](../13-chunked-prefill/)), the long prefill is sliced into 1,024-token chunks that interleave with Anya's decode steps. Her ITL bumps from 28 ms to 32 ms for about ten steps and then recovers. Nothing she would notice.

### t = 6.1 s — `<|eot|>` sampled, request closes

The sampler emits the end-of-turn token. The engine marks the request done. The block manager decrements refcounts on every block in Anya's chain. Most of them drop to zero and return to the free queue. One — the system-prompt prefix block — stays warm at refcount 9,847 (some other user upgraded refcount to 9,848 and it's still in flight).

The final SSE chunk flushes. The HTTP/2 stream stays open; her app maintains the connection for the next turn. The next time Anya types something, the system-prompt prefix block will still be there.

Anya puts her phone in her pocket and gets off the bus. Total elapsed: 6.1 seconds.

## Counting The Wins

Now the *celebratory* part. Each optimization in this issue's tech tree shows up on Anya's request as a multiplicative gain. Let's count them.

```pyplot {id="wins-counted" caption="Every optimization in this issue, measured as a multiplicative gain on Anya's request. Bars are log-scaled. Stack them all and a 2022 vLLM-equivalent setup would have been roughly 200× slower or smaller-batched for the same workload."}
optimizations = [
    ("Paged attention",        850/120, "batch size: 120 → 850"),
    ("Prefix caching",         375,     "prefill cost: 1500/4 tokens"),
    ("Chunked prefill",        12,      "P99 ITL: blocked → 32 ms"),
    ("V1 scheduler",           3.5,     "head-of-line reduction"),
    ("Speculative decoding",   1.6,     "ITL: 45 → 28 ms"),
    ("CUDA graphs",            10,      "CPU overhead: 2 ms → 200 µs"),
    ("FlashAttention",         5,       "attention SRAM-resident"),
    ("Continuous batching",    8,       "vs static batching"),
    ("Disagg P/D (when on)",   2.2,     "decode ITL SLA recovery"),
    ("FP8 quantization",       2,       "model bytes halved"),
]
labels = [o[0] for o in optimizations]
gains  = np.array([o[1] for o in optimizations])
notes  = [o[2] for o in optimizations]

# Sort by gain
order = np.argsort(gains)
labels = [labels[i] for i in order]
gains = gains[order]
notes = [notes[i] for i in order]

colors_cycle = ['#FF007F', '#00A8A8', '#FFD700', '#FF8C00'] * 3
colors = colors_cycle[:len(labels)]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(labels, gains, color=colors, edgecolor='#1A1A1A', linewidth=1.0)
for i, (g, n) in enumerate(zip(gains, notes)):
    ax.text(g*1.05, i, f"  {g:.1f}×   ({n})",
            va='center', fontsize=8.5, family='monospace')
ax.set_xscale('log')
ax.set_xlim(1, 2000)
ax.set_xlabel("multiplicative win  (log scale)")
ax.set_title("Counting the wins  —  every optimization, measured", fontsize=11)
ax.spines[['top', 'right']].set_visible(False)

# The cumulative product (a crude upper bound — these are not strictly multiplicative,
# but it gives a sense of how much architecture has moved in three years).
cumulative = np.prod(gains)
ax.text(0.98, 0.02,
        f"Naïve composition of all gains:\n  ≈ {cumulative:,.0f}× total\n"
        f"(They don't strictly multiply — many overlap.\nReal end-to-end gain vs 2022: ~50–200×.)",
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#FDF5E6', edgecolor='#1A1A1A'))
plt.tight_layout()

print("Wins ranked by raw factor:")
for lbl, g, n in zip(labels[::-1], gains[::-1], notes[::-1]):
    print(f"  {lbl:25s}  {g:6.2f}x   ({n})")
print(f"\nCumulative naive product: {cumulative:,.0f}x")
print("(The real gain stacks more like 50–200x because optimizations overlap.)")
```

The wins, said in prose:

- **Without paged attention** ([ch.10](../10-paged-attention/)), the KV cache for variable-length sequences would have to be pre-allocated at the worst case, wasting most of HBM. The same box that fits 850 concurrent users today would have fit roughly **120**. Anya would have been queued.
- **Without prefix caching** ([ch.12](../12-prefix-caching/)), Anya's prefill would have included the full 1,500-token system prompt instead of just her 4 fresh tokens. That's a **375× saving on prefill compute** *for this turn alone*. Multiplied across millions of daily users, the savings are absurd.
- **Without continuous batching** ([ch.8](../08-continuous-batching/)) and the **V1 scheduler** ([ch.14](../14-scheduler/)), head-of-line blocking from a 64K-prefill arriving mid-stream would have injected *seconds* of dead air into Anya's ITL. Her conversation would feel broken.
- **Without chunked prefill** ([ch.13](../13-chunked-prefill/)), the same.
- **Without speculative decoding** ([ch.15](../15-speculative-decoding/)), Anya's ITL would have been ~45 ms instead of ~28 ms. Her stream would feel halting instead of fluent.
- **Without CUDA graphs** ([ch.5](../05-cuda-graphs/)), the per-step CPU overhead of ~2 ms × 200 steps would have added **400 ms** of pure overhead to her response.
- **Without FlashAttention** ([ch.6](../06-flash-attention/)), the attention kernel would materialize a $4096 \times 4096$ score matrix in HBM every prefill step. Memory traffic would dominate; long-context inference at any reasonable throughput would not exist.
- **Without disagg P/D** ([ch.17](../17-disagg-pd/)) on a long-prompt workload, prefill compute would compete for Anya's decode bandwidth, and the ITL SLA across the fleet would collapse under load.

And every one of these depends on the GPU primer chapters in the foundation: that decode is bandwidth-bound ([ch.4](../04-roofline/), [ch.7](../07-prefill-vs-decode/)), that HBM is two thousand times slower than registers ([ch.3](../03-memory-hierarchy/)), that a streaming multiprocessor needs occupancy to mask latency ([ch.2](../02-gpu-anatomy/)).

{{% pullquote type="profound" %}}
Modern LLM inference is an operating system. Paged virtual memory, a preemptive scheduler, processes, IPC, RDMA networking, multi-tier caches — every CS-101 abstraction reappears, with new names and new constants, in the service of getting one token to one phone in two hundred and eighty milliseconds.
{{% /pullquote %}}

Pin that quote somewhere. It is the thesis of the issue. Everything we built was *forced moves* from a single starting fact: at decode time, the GPU loads bytes faster than it multiplies them. The roofline ([ch.4](../04-roofline/)) said so in 2017; the entire 2022–2026 stack of optimizations is the field's accumulated response.

## Three Years Of Recent History

Five of these techniques didn't exist three years ago. The history is recent and traceable, and worth holding in one place.

- **2022.** Orca introduces continuous batching ({{< cite text="Yu et al., 2022 (OSDI)" url="https://www.usenix.org/conference/osdi22/presentation/yu" kind="paper" >}}). FlashAttention v1 demonstrates SRAM-resident attention ({{< cite text="Dao et al., 2022" url="https://arxiv.org/abs/2205.14135" kind="paper" >}}).
- **2023.** vLLM ships PagedAttention ({{< cite text="Kwon et al., 2023 (SOSP)" url="https://arxiv.org/abs/2309.06180" kind="paper" >}}). Microsoft's Splitwise prototypes disagg P/D ({{< cite text="Patel et al., 2023" url="https://arxiv.org/abs/2311.18677" kind="paper" >}}). Speculative decoding goes mainstream ({{< cite text="Leviathan et al., 2023" url="https://arxiv.org/abs/2211.17192" kind="paper" >}}).
- **2024.** DistServe formalizes disagg ({{< cite text="Zhong et al., 2024 (OSDI)" url="https://arxiv.org/abs/2401.09670" kind="paper" >}}). Mooncake adds the KV cache fabric ({{< cite text="Qin et al., 2024 (Mooncake)" url="https://arxiv.org/abs/2407.00079" kind="paper" >}}). Chunked prefill becomes standard. Prefix caching ships in every major engine.
- **2025.** vLLM V1 unifies the scheduler around a single token budget. NIXL standardizes the KV transport ({{< cite text="NVIDIA NIXL announcement" url="https://developer.nvidia.com/blog/nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/" kind="doc" >}}). EAGLE-style spec decode goes from research to production.
- **2026.** The architectural pattern is stable enough that this issue exists.

Three years. From "static batching with stop-the-world prefill" to "multi-tier KV fabrics with one-sided RDMA, paged virtual memory, and a token-budget scheduler". The pace was uncomfortable to live through and is breathtaking to look back on.

{{< crosshead >}}Where The Story Goes Next{{< /crosshead >}}

By mid-2026 the inference stack has roughly the shape it will keep for a while. The next chapter of the story is being written along three threads, each of which pulls on the same primitives we just spent eighteen chapters mapping.

**Multi-tier KV cache fabrics.** Mooncake started it; LMCache and a half-dozen open-source projects are turning the KV cache into a fully first-class distributed storage system, with replication, eviction, geo-distribution. The boundary between "inference engine" and "vector database" is dissolving.

**Disaggregated experts.** {{< wiki "hyperparameters" >}}Mixture-of-experts{{< /wiki >}} models route each token to a handful of expert MLPs. For inference, those experts can live on their own cluster, hit by all-to-all over the same RDMA fabric that already moves KV cache. The architectural pattern is identical to disagg P/D, generalized to a third role.

**Sparse long-context attention.** Anya's 4K prompt is tiny by 2026 standards. Long-context inference at 1M+ tokens requires attention complexity below $\mathcal{O}(N^2)$, and the sparse-attention work in [Issue 07: The Sparse Lab](/llm-maths/issues/07-sparse-lab/) is where that story is being written. The serving primitives we built — paged KV, FlashAttention, scheduler token budgets — generalize to it almost unchanged.

**Quantization all the way down.** FP8 weights are standard; FP4 weights and KV are coming. [Issue 03: Sixteen Numbers](/llm-maths/issues/03-sixteen-numbers/) is the deep dive on what is mathematically possible and where the floor actually is. Every halving of bytes-per-weight is a halving of decode bandwidth pressure, which is the most precious resource in the entire stack.

**On-device inference.** The same architectural pattern — paged KV, FlashAttention, scheduler — is being compressed onto consumer phones and laptops. The constants change (one chip, one user, much smaller models); the primitives are the same.

Every one of these threads pulls on what this issue mapped. The vocabulary has settled. The primitives are stable. **What changes from here on is what we build with them, not what they are.**

## The Last Thought

Anya's request was 4:12:00 PM to 4:12:06 PM. Six point one seconds. From her perspective, she asked a question and got an answer. From the box's perspective, eight hundred users were served in parallel by a sliced-up 600-billion-parameter mixture-of-experts model running on a chassis full of silicon that didn't exist three years ago, mediated by a serving stack that didn't exist five years ago, expressing a forty-year-old idea (virtual memory + page tables) in a brand-new domain.

The reason she got her answer in six seconds is that thousands of engineers, across a dozen labs, spent those three years staring at the roofline plot and asking: where is the bandwidth going? Each chapter of this issue is one of their answers. Stitched together, they are the operating system of an idea.

The next time you tap send on a chat app and a fluent paragraph streams back, you will know what just happened. **You will know what the GPU was doing, which kernel was launched, which block was allocated, which RDMA write was overlapped, which token was speculated, which prefix was cached.** You will know it not as a recipe but as a system — a system whose pieces fit together because the people who built them were each solving a piece of the same problem, one chapter at a time.

That is what this issue was for. Every chapter pointed at a piece of the stack. Every piece pointed back at the same roofline plot. Every roofline plot pointed back at one mute fact about silicon: the bytes are slow and the math is fast.

And the rest, as the saying goes, is engineering.

*This is the boss. The issue is done. If you want to follow specific threads onward, the [issue index](../) points to the issues that pick up each one — quantization in [Sixteen Numbers](/llm-maths/issues/03-sixteen-numbers/), sparse attention in [The Sparse Lab](/llm-maths/issues/07-sparse-lab/), the transformer math in [microGPT Unfolded](/llm-maths/issues/05-microgpt-unfolded/). But for this story — the life of one token, from packet to phoneme — we are at the end.*
