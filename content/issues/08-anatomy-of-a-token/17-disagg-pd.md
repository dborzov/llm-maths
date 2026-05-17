---
title: "Two Houses, Divided"
description: "Prefill needs high compute throughput; decode needs high memory bandwidth. Running them on the same GPU is the worst of both worlds. Disaggregated prefill/decode splits the job across two purpose-built machine types connected by an RDMA KV-cache fabric — DistServe, Mooncake, and NIXL are the field's answers."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T17:00:00-04:00
issue: 8
weight: 170
techKind: mainline
techNode: disagg-pd
header: 17-disagg-pd.webp
---

## Redmond, Late 2023

A site-reliability engineer at Microsoft is staring at a Grafana dashboard that should be boring.

The dashboard shows the **inter-token latency** — ITL, the gap between successive streamed tokens — for one of the new Azure OpenAI inference clusters. For most of the day the line is flat: a respectable 28 ms, indistinguishable from background noise. Users see fluent paragraphs streaming at sixty words per second. Then, sporadically, every few minutes, the line jumps. 28 ms → 600 ms → 1.4 s → back to 28 ms. A whole second of dead air, then the stream catches up. Customers complain. The on-call rotation files tickets. The dashboards show **no** corresponding spike in GPU utilization, **no** spike in HBM pressure, **no** scheduler starvation.

For weeks the team chases ghosts. Eventually somebody correlates the latency spikes against the request log. The pattern is unambiguous: every spike coincides with the arrival of a **long-prompt request** — a 32K-token RAG query, a 50K-token code-review prompt, a transcript summarization. The long prompts arrive once every minute or two. When one lands in the batch, *everybody else's decode tokens are delayed by hundreds of milliseconds*.

This is **head-of-line blocking**, and in 2023 it was the dirty secret of every production LLM serving stack. We will spend this chapter understanding why it was such a hard problem, why the field eventually agreed it had to be solved by **physically splitting prefill and decode onto different machines**, and what the bytes-on-the-wire details of that split actually look like.

The papers that established the playbook came in three waves: Microsoft's **Splitwise** ({{< cite text="Patel et al., 2023" url="https://arxiv.org/abs/2311.18677" kind="paper" >}}) was the first to name the problem and prototype a fix; PKU + UC San Diego's **DistServe** ({{< cite text="Zhong et al., 2024 (OSDI)" url="https://arxiv.org/abs/2401.09670" kind="paper" >}}) gave it a clean theoretical framing and SLA-aware scheduling; Moonshot's **Mooncake** ({{< cite text="Qin et al., 2024 (Mooncake)" url="https://arxiv.org/abs/2407.00079" kind="paper" >}}) ran it in production behind Kimi at the scale of a billion-dollar consumer chatbot and added a third architectural tier nobody had seen before. NVIDIA's **NIXL** ({{< cite text="NVIDIA NIXL announcement" url="https://developer.nvidia.com/blog/nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/" kind="doc" >}}) in 2025 turned the whole thing into a standardized transport layer that the rest of the ecosystem could build on.

By 2026 the disaggregated topology is the default for any deployment serving more than a few hundred concurrent users. This chapter explains why.

## Two Workloads In One Tube

To see the head-of-line problem clearly, you have to remember what prefill and decode actually *do* at the hardware level — and the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} is at the center of both.

We already drew this picture in [Two Phases, Two Personalities](../07-prefill-vs-decode/). Here is the one-paragraph recap.

A **prefill** is the forward pass over the prompt. If the prompt has $N$ tokens, prefill computes $N$ {{< wiki "attention" >}}attention{{< /wiki >}} positions in parallel — every token attends to every earlier token — and the dominant cost is an $\mathcal{O}(N^2)$ matmul against the KV cache being constructed. The GPU's tensor cores get to do real work. The kernel is **compute-bound**: the {{< wiki "transformer-weights" >}}weight matrices{{< /wiki >}} are reused across all $N$ positions, so the arithmetic intensity is high. On an H100 a long prefill happily sits near the roofline ceiling, burning 500 TFLOP/s on FP8 matmuls.

A **decode step** is the forward pass for *one* new token. The model still has to load every weight from HBM once. But it only does $1 \times d_\text{model}$ worth of matmul per layer instead of $N \times d_\text{model}$. The arithmetic intensity collapses by a factor of $N$. The kernel is **bandwidth-bound**: the GPU spends almost all its time waiting for bytes to arrive from HBM. The tensor cores are largely idle. The same H100 that hit 500 TFLOP/s during prefill is now using maybe 5% of its compute. What it is *fully* using is its 3.35 TB/s HBM3 bandwidth.

Put these two facts side by side and you can already feel the trouble.

```pyplot {id="prefill-decode-roofline" caption="Prefill lives near the compute roof; decode lives on the bandwidth slope. On the same GPU, the two workloads occupy completely different regions of the roofline plot — and have totally different bottlenecks."}
np.random.seed(0)

# H100 SXM5 ballpark: 989 TFLOP/s FP16, 3.35 TB/s HBM. FP8 doubles compute.
peak_flops = 989e12       # FP16
peak_bw    = 3.35e12      # bytes/s
ridge_pt   = peak_flops / peak_bw  # ~295 FLOP/byte

# Arithmetic intensity (FLOP/byte) for prefill at various prompt lengths, and decode.
# Per-layer matmul: weight bytes ~ 2 * d^2; FLOPs ~ 2 * N * d^2.
# Intensity ~ N (in FP16).
prompt_N = np.array([1, 4, 16, 64, 256, 1024, 4096])
decode_intensity = 1.0
prefill_intensity = prompt_N.astype(float)

def attainable(intensity):
    # min(peak_flops, peak_bw * intensity)
    return np.minimum(peak_flops, peak_bw * intensity) / 1e12  # TFLOP/s

fig, ax = plt.subplots(figsize=(9, 5))
xs = np.logspace(-1, 4, 400)
ys = np.minimum(peak_flops, peak_bw * xs) / 1e12
ax.plot(xs, ys, color='#1A1A1A', linewidth=2)
ax.fill_between(xs, 0, ys, color='#FFD700', alpha=0.15)

# Prefill points
ax.scatter(prefill_intensity, attainable(prefill_intensity),
           s=80, color='#FF007F', edgecolor='#1A1A1A', linewidth=1.0,
           zorder=4, label='prefill (N tokens)')
for N, x in zip(prompt_N, prefill_intensity):
    ax.annotate(f"N={N}", (x, attainable(x)), xytext=(6, -10),
                textcoords='offset points', fontsize=8)

# Decode point
ax.scatter([decode_intensity], [attainable(decode_intensity)],
           s=140, color='#00A8A8', edgecolor='#1A1A1A', linewidth=1.2,
           zorder=4, marker='D', label='decode (1 token)')
ax.annotate("decode", (decode_intensity, attainable(decode_intensity)),
            xytext=(6, 10), textcoords='offset points',
            fontsize=10, fontweight='bold')

ax.axvline(ridge_pt, color='#FF8C00', linestyle='--', linewidth=1.2, alpha=0.8)
ax.text(ridge_pt*1.05, 3, f"ridge pt\n≈ {ridge_pt:.0f} FLOP/byte",
        fontsize=8, color='#FF8C00')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("arithmetic intensity (FLOP / HBM byte)")
ax.set_ylabel("attainable throughput (TFLOP/s)")
ax.set_title("H100 roofline — same chip, two workloads, two regimes")
ax.legend(loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print(f"H100 ridge point: {ridge_pt:.0f} FLOP/byte")
print(f"Prefill at N=4096 reaches {attainable(4096):.0f} TFLOP/s ({attainable(4096)/peak_flops*100*1e12:.0f}% of peak)")
print(f"Decode (N=1) reaches    {attainable(1):.1f} TFLOP/s   ({attainable(1)/peak_flops*100*1e12:.2f}% of peak)")
print(f"Decode is bottlenecked at peak HBM bandwidth = {peak_bw/1e12:.2f} TB/s")
```

Stop and re-read the printed numbers. **The same chip delivers 700 TFLOP/s on a long prefill and 3 TFLOP/s on a decode step.** Two hundred-fold gap. The decode kernel is not doing anything wrong; it is just stuck on the bandwidth slope, fundamentally unable to use the tensor cores because there is nothing to feed them.

{{< crosshead >}}The Co-Located Trap{{< /crosshead >}}

Now imagine these two workloads sharing a GPU. This is the **co-located** topology — what every serving system did from 2022 through 2024 — and it has been the default for so long that calling it a "trap" feels uncharitable.

The trap is this. When the scheduler steps the engine, it builds a single batched forward pass over *whatever happens to be in flight*. Some requests are in prefill (a few long sequences worth of tokens), most are in decode (one token each). Continuous batching ([ch.8](../08-continuous-batching/)) and the V1 scheduler ([ch.14](../14-scheduler/)) made this mixing possible at the level of the token budget. But the *kernel* that runs is still one kernel, on one GPU, with one set of warps, hitting one HBM bus. And the kernel's wall-clock time is dominated by whatever is most expensive in the batch.

When a 32K-token prefill lands in the batch, the kernel takes hundreds of milliseconds. Every decoding user in the same batch waits *that whole time* between successive tokens. From their perspective the stream just stalled.

{{% marginnote %}}**Chunked prefill** ([ch.13](../13-chunked-prefill/)) softens this by slicing a long prefill into 512- or 1024-token chunks that each fit in a normal step. It's a real fix at the scheduler level — but it caps prefill throughput and still spends the GPU's compute capacity at moments when *no decoder* in the batch can benefit from it.{{% /marginnote %}}

The naïve fix is "just don't batch them together." The vLLM V1 scheduler will happily emit prefill-only steps and decode-only steps. This solves the head-of-line problem *within a batch*. It does not solve the more fundamental issue, which is that **a single GPU running a 50/50 mix of prefill and decode is using the wrong hardware for both**:

- On the decode steps, the tensor cores idle.
- On the prefill steps, the HBM bandwidth idles.
- Worst of all, the **ITL SLA** for decode users degrades any time a prefill step has to run, because the prefill step blocks the GPU for hundreds of milliseconds during which no decode token is produced.

You bought one of the most expensive chips in the world to run *both* of its dimensions at full tilt. Most of the time, you are running one dimension at full tilt and the other near zero. The averages look fine. The tail latencies are catastrophic.

## The Idea: One Workload, One Machine

The disaggregated solution is, in retrospect, almost embarrassingly obvious. **Prefill and decode want different machines. Give them different machines.**

A **prefill machine** maximizes compute-per-dollar:
- Lots of tensor cores. FP8/FP4 throughput is the headline.
- Modest HBM (just enough to hold the model and a moderate working set of KV during the forward pass).
- The hardware sweet spot in 2026 is something like H100, B200, MI300A — compute-rich, bandwidth-respectable.

A **decode machine** maximizes bandwidth-per-dollar:
- Lots of HBM, ideally the latest generation (HBM3e capacity → more concurrent users' KV cache fits).
- Wide NVLink between local GPUs (decode batches benefit from fast TP all-reduce since the matmuls are small).
- Compute is almost wasted; you would happily trade FP16 TFLOP/s for more bytes-per-second.
- The hardware sweet spot is H200 (141 GB HBM3e at 4.8 TB/s), MI300X (192 GB at 5.3 TB/s).

The two clusters run the same model weights. They differ only in *which slice of the forward pass they're optimized for*. Requests enter the prefill cluster, get their prompts processed there, then **hand off** to the decode cluster, which carries them through the long tail of token generation.

{{% callout type="theorem" %}}
**The Disaggregation Theorem (informal).** If prefill and decode have different roofline regions, then for a fixed dollar budget the per-token serving cost is minimized by *physically separating them onto different hardware SKUs*, sized in the ratio at which the workload produces prefill-tokens vs decode-tokens.
{{% /callout %}}

There is a beautiful corollary nobody quite says out loud. The optimal **ratio** of prefill machines to decode machines depends entirely on the **workload mix**. If your users send long prompts and want short answers (think code review, RAG), you need lots of prefill capacity. If they send short prompts and want long answers (think chat, agents, reasoning), you need lots of decode. The disaggregated topology lets you *scale these two dimensions independently*. The co-located topology forced you to buy them in fixed pairs.

This is why every major lab has converged on disagg by 2026. It is not just faster. It is **separately tunable**, in a way the co-located stack fundamentally cannot be.

## What Has To Move

You cannot just hand off a request from prefill to decode. The whole point of prefill was to build up the KV cache — the contextual K and V tensors at every layer, for every token in the prompt. The decode step needs that KV cache to attend over. So when prefill is done, **the KV cache for the prompt has to travel from the prefill machine's HBM to the decode machine's HBM, before the first decode step can run**.

How big is that?

### Napkin Math: One Llama-70B Prompt

Llama-3-70B in FP8 has:
- $L = 80$ transformer layers.
- $h_\text{kv} = 8$ KV heads (grouped-query attention — only the K/V projections are sharded over 8 heads, not 64).
- $d_\text{head} = 128$ per head.
- 2 tensors (K and V) per layer.
- 1 byte per element (FP8).

Per token, the KV cache takes:

$$
\text{bytes}_\text{tok} = L \cdot h_\text{kv} \cdot d_\text{head} \cdot 2 \cdot 1 \;=\; 80 \cdot 8 \cdot 128 \cdot 2 \;=\; 163{,}840 \text{ bytes} \;\approx\; 160 \text{ KB}.
$$

For a 4K-token prompt:

$$
\text{bytes}_\text{prompt} = 4096 \cdot 160 \text{ KB} \;\approx\; 640 \text{ MB}.
$$

For a 32K-token prompt: **~5 GB**. For a 128K prompt: **~20 GB**. The KV cache scales linearly in tokens, and for long prompts it can be larger than the model itself in compressed form.

Now how fast can we move that across a network?

A modern data-center RDMA NIC moves 200 Gb/s = 25 GB/s sustained. NVIDIA NVLink moves more — 900 GB/s on the latest gen — but only within a single chassis. For inter-chassis transfer, RDMA over InfiniBand (or its Ethernet cousin RoCE) is the workhorse.

So for our 4K-token Llama-70B prompt: **640 MB / 25 GB/s ≈ 25 ms** on the wire.

Twenty-five milliseconds is **a lot**. The user's TTFT was already going to be ~280 ms for a long prefill; adding 25 ms of synchronous transfer at the end is a 10% regression. For long prompts the transfer time scales linearly: 32K tokens → 200 ms of transfer, which is intolerable on the critical path.

This is the engineering crux of the whole topology. **If the KV transfer is synchronous, disagg is a non-starter.** The hand-off has to be hidden.

## Hiding The Transfer

The trick is hilariously simple to state and devilishly intricate to implement. Prefill is a layer-by-layer forward pass: layer 1 computes its K and V, then layer 2, then layer 3, all the way to layer 80. By the time the model is computing **layer 12**, the K and V tensors for layers 1 through 11 are already done and just sitting in HBM, waiting.

So: **ship them while the model keeps computing.**

```pyplot {id="layer-streaming-gantt" caption="Layer-by-layer KV streaming during prefill. Each layer's KV is shipped to the decode cluster as soon as it is produced, overlapped with the next layer's matmul. By the time the last layer finishes, the transfer is essentially done."}
np.random.seed(0)

# 80 layers, each ~ T_layer ms of compute, each producing B MB of KV.
n_layers = 80
T_layer_ms = 1.2          # compute time per prefill layer at this prompt length
B_MB = 640 / n_layers     # 8 MB per layer for our 4K Llama-70B example
BW_MBps = 25_000          # 25 GB/s RDMA
T_xfer_ms = B_MB / BW_MBps * 1000

fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True,
                         gridspec_kw={'height_ratios': [3, 3]})

# Lane 1: compute on prefill GPU.
for ell in range(n_layers):
    start = ell * T_layer_ms
    axes[0].barh(0, T_layer_ms, left=start, height=0.7,
                 color='#FF007F', edgecolor='#1A1A1A', linewidth=0.3)
axes[0].set_yticks([0])
axes[0].set_yticklabels(["prefill GPU\ncompute"], fontsize=9)
axes[0].set_title("Layer-by-layer KV streaming  —  4K token Llama-70B prompt", fontsize=10)
axes[0].set_xlim(-0.5, n_layers * T_layer_ms + 5)
axes[0].spines[['top', 'right']].set_visible(False)

# Lane 2: RDMA stream, each layer ℓ shipped as soon as compute(ℓ) completes.
for ell in range(n_layers):
    ship_start = (ell + 1) * T_layer_ms  # KV(ℓ) available after layer ℓ finishes
    # If transfer is faster than compute, transfers queue tightly.
    actual_start = max(ship_start, ell * T_xfer_ms)
    axes[1].barh(0, T_xfer_ms, left=actual_start, height=0.7,
                 color='#00A8A8', edgecolor='#1A1A1A', linewidth=0.3)
axes[1].set_yticks([0])
axes[1].set_yticklabels(["RDMA →\ndecode HBM"], fontsize=9)
axes[1].set_xlabel("time (ms)")
axes[1].spines[['top', 'right']].set_visible(False)

compute_done = n_layers * T_layer_ms
xfer_done = compute_done + T_xfer_ms  # last layer ships only after it's computed
axes[0].axvline(compute_done, color='#FF8C00', linewidth=1, linestyle='--')
axes[1].axvline(xfer_done, color='#FF8C00', linewidth=1, linestyle='--')
axes[1].text(xfer_done + 0.5, 0, f"transfer done\nt={xfer_done:.1f} ms",
             fontsize=8, color='#FF8C00', va='center')
plt.tight_layout()

print(f"per-layer KV bytes:   {B_MB:.2f} MB")
print(f"per-layer compute:    {T_layer_ms:.2f} ms")
print(f"per-layer xfer time:  {T_xfer_ms:.3f} ms  (transfer is {T_layer_ms/T_xfer_ms:.0f}x faster than compute)")
print(f"naïve sequential:     compute then ship = {compute_done + n_layers*T_xfer_ms:.1f} ms")
print(f"overlapped streaming: {xfer_done:.1f} ms")
print(f"saved by overlap:     {(compute_done + n_layers*T_xfer_ms) - xfer_done:.1f} ms ({(1 - xfer_done/(compute_done + n_layers*T_xfer_ms))*100:.0f}%)")
```

The print-out is the whole point. **For Llama-70B at 4K tokens, the per-layer KV transfer is faster than the per-layer compute, by something like an order of magnitude.** That means the network is never the bottleneck during layer-by-layer streaming; the prefill GPU's matmul is. The transfer effectively *hides itself* behind compute, and the wall-clock latency of the hand-off goes from ~25 ms (synchronous) to ~25 *microseconds* (the time to ship the last layer after the last matmul).

This is the secret sauce. Disagg works because **modern RDMA fabrics are fast enough that, with layer-aligned streaming, the KV transfer disappears into the prefill's own latency budget**.

There is a subtle prerequisite: the KV cache has to already be organized in transferable chunks. This is where [Borrowing From 1965](../10-paged-attention/) pays off. Paged KV stores the cache as fixed-size blocks (16 or 32 tokens) at addressable physical positions. The unit of transfer is a block, not the whole tensor — so the streaming can be arbitrarily fine-grained, and the receiving side can place blocks at whatever physical address its own block manager hands out.

{{% callout type="tip" %}}
**Why paged KV is the foundation.** Without PagedAttention, every disaggregated transfer would have to ship one contiguous tensor per layer, in a fixed shape, to a pre-reserved decode address. With paging, the transfer becomes a *list of (layer, block_id) → bytes* messages that the receiver places freely. The block manager on the decode side allocates from its free queue *while the transfer is in flight*. This is how the prefill cluster and decode cluster can have completely independent memory topologies and still share a request.
{{% /callout %}}

## NIXL: Making The Transfer Boring

Until 2024 every serving stack rolled its own KV-transfer code. DistServe used NCCL. Mooncake used a custom RDMA library. vLLM had its own transport layer. The interfaces were all different, and the abstractions leaked: you had to know what kind of fabric was underneath (RDMA? NVLink? PCIe?) to write the transfer correctly.

NVIDIA's **NIXL** (NVIDIA Inference Xfer Library, 2025) is the standardization move. It exposes a single interface:

```python
# Pseudocode for the NIXL API
xfer = nixl.create_transfer(
    src=nixl.MemoryRegion(prefill_kv_blocks, dev="cuda:3"),
    dst=nixl.MemoryRegion(decode_kv_blocks,  dev="remote:cuda:1"),
    transport=nixl.AUTO,    # picks RDMA / NVLink / PCIe based on topology
)
xfer.submit()               # non-blocking, one-sided
xfer.wait()                 # only if you need synchronization
```

Under the hood NIXL picks the right transport — NVLink if the source and destination are in the same chassis, RDMA-over-InfiniBand if not, GPUDirect Storage if one side is NVMe — and uses **one-sided RDMA writes** so the prefill GPU can push bytes into the decode GPU's HBM without involving the decode CPU at all. The decode side doesn't even know the transfer happened until its block manager checks the destination address.

The point of NIXL is not novelty — every piece of it was being done by hand somewhere — but **uniformity**. The whole inference ecosystem now writes against the same transfer abstraction, which means transferable KV blocks are a first-class type in the serving framework, like tensors or batches. By mid-2026 vLLM, SGLang, and TensorRT-LLM all use NIXL or a NIXL-compatible wrapper.

This is the standard arc for an emerging systems abstraction: research papers in 2023, ad-hoc implementations in 2024, vendor library in 2025, default-everywhere in 2026.

## Mooncake: A Third Tier

DistServe and Splitwise had two tiers: prefill and decode. Moonshot — the Chinese AI lab behind Kimi, a long-context assistant serving hundreds of millions of users — looked at the system in 2024 and added a **third** ({{< cite text="Qin et al., 2024 (Mooncake)" url="https://arxiv.org/abs/2407.00079" kind="paper" >}}).

The insight: **the KV cache is a storage hierarchy too**.

Hot KV blocks (currently serving a live request) live on the decode GPU's HBM. Warm KV blocks (recently used by a finished request, likely to be hit again as a prefix-cache lookup) live in **host DRAM** on a shared KV-cache server. Cold KV blocks (rarely re-hit but cheap to keep) spill to **NVMe SSD**. The whole thing is a multi-tier cache, with the prefill workers feeding it new entries and the decode workers reading prefixes from it.

```pyplot {id="mooncake-tiers" caption="Mooncake's KV cache as a storage hierarchy. Each tier is roughly an order of magnitude slower and an order of magnitude larger than the one above it. The disaggregated cluster shares this fabric — prefix-cache hits anywhere in the hierarchy save prefill compute somewhere."}
tiers = [
    ('GPU HBM\n(active decode)', 100, 5_000, '#FF007F'),
    ('Remote GPU HBM\n(disagg peer)', 1_500, 50_000, '#FF8C00'),
    ('Host DRAM\n(KV cache server)', 50_000, 500_000, '#FFD700'),
    ('NVMe SSD\n(cold storage)', 2_000_000, 5_000_000, '#00A8A8'),
]

fig, ax = plt.subplots(figsize=(9.5, 4.5))
for i, (label, capacity_mb, latency_ns, colour) in enumerate(tiers):
    width = np.log10(capacity_mb)
    ax.barh(i, width, height=0.7, color=colour,
            edgecolor='#1A1A1A', linewidth=1.0)
    ax.text(width + 0.1, i,
            f"  {capacity_mb/1000:.1f} GB-ish per node, "
            f"≈ {latency_ns/1000:.1f} µs / KV-block read",
            va='center', fontsize=8.5, family='monospace')

ax.set_yticks(range(len(tiers)))
ax.set_yticklabels([t[0] for t in tiers], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("log₁₀(capacity per node, MB)")
ax.set_title("Mooncake KV hierarchy  —  the cache is now part of the storage stack",
             fontsize=10)
ax.set_xlim(1, 9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print("Each step down the hierarchy costs ~10x latency, gains ~10x capacity.")
print("A prefix-cache hit anywhere in this hierarchy saves the full prefill compute cost.")
```

Two operational consequences fall out of this design.

First, **prefix-cache-aware routing** becomes a global problem rather than a per-machine one. The scheduler asks "which prefill machine, across the whole cluster, has the longest cached prefix for this request?" and routes accordingly. A shared 50K-token system prompt that has been hit a million times today might be resident in DRAM on the KV cache server; whichever prefill worker handles the next request reads it from there in ~50 µs and avoids ten seconds of recomputation. We discussed this in [Reusing The Prologue](../12-prefix-caching/) — Mooncake extends it from per-worker to per-cluster.

Second, **decode workers become stateless**. They borrow KV blocks from the fabric for the duration of a request and return them at the end. If a decode worker dies, the request can be migrated to another decode worker — the KV cache is fetched again from the fabric, possibly from a different physical machine, and the stream continues. This is a level of operational sanity that the co-located stack simply could not offer.

## The Distributed Scheduler

A disaggregated cluster needs two coordinating schedulers, and a control plane on top.

- The **prefill scheduler** decides which prefill machine handles each new request. Its objective: maximize prefix-cache reuse (route to the worker that has the longest matching cached prefix) and balance compute load.
- The **decode scheduler** decides which decode machine receives each finished prefill. Its objective: balance HBM occupancy (the decoder with the most free KV blocks wins) and respect ITL SLAs (don't dump a long-decode request onto a worker that is already saturating its bandwidth).
- The **control plane** matches the two: when prefill is done on worker $P_i$, it queries the decode scheduler for a target $D_j$, registers a NIXL transfer from $P_i$'s KV blocks to $D_j$'s pre-allocated blocks, and hands the request over.

The protocol is more elaborate than a vanilla load balancer. The papers describe variants — DistServe's bipartite optimization, Mooncake's two-level dispatcher — but the structure is always the same: **two schedulers with different objectives, talking through a fabric**.

{{% marginnote %}}If this is starting to sound like a distributed database, it is. Cache-aware routing is the same problem as cache-aware sharding in DBMS literature. The disaggregated LLM stack is rediscovering, with very fresh terminology, ideas from twenty-year-old systems papers. The patterns are remarkably stable.{{% /marginnote %}}

## When Disagg Is Not Worth It

Disagg is a sophisticated machine and it pays for itself only at scale. Three regimes where it isn't worth the operational complexity:

- **Small models.** A 7B-class model has small KV cache, low prefill cost, and barely fills one GPU. The co-located fleet is fine.
- **Short prompts.** If the workload is all short (system prompt + chat turn ≤ 1K tokens), prefill is cheap, the head-of-line problem is mild, and chunked prefill plus continuous batching solve it within a single fleet.
- **Bursty, low-QPS workloads.** Disagg shines when you can keep both the prefill and decode fleets continuously utilized. If your traffic is bursty, you end up with one fleet idle while the other is saturated — exactly the failure mode you were trying to avoid.

The 2026 industry consensus is to **run both topologies side by side** and route based on workload profile. Long-prompt or long-context requests go to the disagg fleet; short or interactive requests stay on co-located. The router uses the prompt length and a rough cost-model to pick.

This is a recurring pattern in the inference stack: you do not replace the old thing with the new thing, you stack them and route. Static batching, continuous batching, paged KV, prefix caching, chunked prefill, speculative decoding, disagg — they are all *additions* to the toolbox, and a mature deployment uses all of them at once.

## The Bargain

Disagg is the architectural acknowledgement that **prefill and decode are different jobs, deserving different machines**. The cost is a network hop between them; the benefit is that each side gets to operate at the roofline ceiling of its own regime, the SLAs decouple, and you can scale the two dimensions independently as the workload mix shifts.

The reason this took until 2024 to become standard is that it required three pieces to land in the same year: paged KV (the transferable unit), fast enough RDMA fabrics (the medium), and a workload at production scale where the head-of-line problem actually hurt (the motive). When all three arrived, the field pivoted within twelve months. By the time NIXL shipped in 2025, every major serving stack supported a disaggregated mode.

And once you have the fabric, more ideas suggest themselves. Disaggregated *expert routing* for MoE models — the experts live on their own cluster, requests fly to them. KV cache *migration* under load — a decoder rebalances by handing a live stream to a peer mid-request. The 2026 inference stack is, increasingly, a distributed operating system whose primary data structure is the KV cache.

Which brings us back to Anya. Twelve tokens, 4:12 PM, an H200 box, a 280-millisecond TTFT. Time to walk her request through the whole stack — one last time, with everything named.

**Continue to → [The Full Anatomy of a Token](../18-full-anatomy/)**
