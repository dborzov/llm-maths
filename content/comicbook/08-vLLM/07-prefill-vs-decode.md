---
title: "Prefill vs Decode: two workloads on one GPU"
short_title: "Prefill vs Decode"
description: "Prefill runs at thousands of FLOP/byte and saturates the tensor cores; decode runs at ~1 FLOP/byte and turns a $40,000 GPU into an oversized HBM controller — and those opposite bottlenecks drive every design decision in this issue."
blurb:
  - "Prefill arithmetic intensity scales linearly with token count: at T=4096 through Llama-70B it is well above the H100 ridge point."
  - "Decode intensity is ~1 FLOP/byte regardless of model size — the lower bound on per-token latency is 29 ms, set by HBM bandwidth alone."
  - "That 29 ms floor is thermodynamic: no kernel tuning moves it, only fewer bytes or fewer loads."
  - "Every optimization in this issue — quantization, speculative decoding, batching, disaggregation — is a variation on three options."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T12:00:00-04:00
issue: 8
weight: 70
techKind: mainline
techNode: prefill-vs-decode
header: 07-prefill-vs-decode.webp
---

## Two Tracks On One Trace

It is a Tuesday afternoon in spring 2026, and an engineer at a serving company is staring at an NVIDIA Nsight trace. One H100, tensor-parallel rank 0. One Llama-70B. One user. The user has pasted in a 4,096-token prompt — about ten pages of code review — and asked for a summary.

The trace is split clean in half. On the left, a single dense slab: **600 ms of green tensor-core activity**, the GEMM kernels packed nose-to-tail, occupancy pegged near 100%, HBM traffic visible but unremarkable. On the right, the slab dissolves into a picket fence: **a hundred narrow vertical bars**, each roughly **25 ms wide**, with **gaps between them** that the engineer cannot, on first inspection, explain. The tensor cores in the picket-fence region are reading at 78% as warm — but the FLOP/s counter shows the chip is at less than 4% of peak. The HBM is the busy lane, not the tensor cores. Same GPU. Same model. Same forward pass code. Two completely different workloads.

This is the picture that organizes everything in this issue. The slab is **prefill**. The picket fence is **decode**. They share a GPU, they share a {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}, and they share approximately nothing else.

{{% marginnote %}}
microGPT [Prefill vs Decode](/llm-maths/comicbook/05-microgpt/14-prefill-decode/) covers the **transformer-internals** version of this split — what tokens, masks, and KV writes look like at each step. This chapter is the **systems** re-framing: the same split, but viewed as two workloads with opposite arithmetic intensity fighting over one GPU.
{{% /marginnote %}}

## The Compute Slab

Prefill is the model reading the prompt. Given $T$ input tokens, it runs **one forward pass** that processes all $T$ positions simultaneously. Every layer takes a $(T \times D)$ activation tensor, slams it through {{< wiki "attention" >}}attention{{< /wiki >}} and the MLP, and emits another $(T \times D)$ tensor. At the end of all $L$ layers, the model writes one full pair of keys and values per position per layer — building the KV cache from scratch, in one shot, all at once.

That "all at once" is what makes prefill compute-bound. Take any linear layer in the MLP: weight $W \in \mathbb{R}^{D \times 4D}$. Loading $W$ from HBM costs $D \cdot 4D \cdot 2$ bytes in FP16 — call it $B$. The matmul against the $T$-token activation $X \in \mathbb{R}^{T \times D}$ takes $2 \cdot T \cdot D \cdot 4D$ FLOPs. The arithmetic intensity is

$$
I_\text{prefill} = \frac{2 \cdot T \cdot D \cdot 4D}{D \cdot 4D \cdot 2 + T \cdot D \cdot 2} \;\approx\; T \quad\text{(when } T \gg 1\text{)}.
$$

For Llama-70B with $D = 8192$ and $T = 4096$, that is **thousands of FLOPs per HBM byte** — well above the H100 ridge point of $\approx 93$ FLOP/byte (see [The Roofline](../04-roofline/)). The tensor cores are doing useful work on every cycle. HBM is just *delivering* the weights; the bottleneck is silicon, not bus.

## The Memory Picket Fence

Decode is the opposite animal. After prefill, the model has the KV cache populated for the full prompt and one logit distribution over the vocabulary. It samples a token. Then it does something **strange and wasteful** by every classical-HPC standard: it runs another full forward pass — through all $L$ layers, every {{< wiki "transformer-weights" >}}weight matrix{{< /wiki >}} — *for that single new token*.

The arithmetic of decode is brutal. The same MLP layer that processed 4,096 tokens at once during prefill now processes **one**:

$$
I_\text{decode} = \frac{2 \cdot 1 \cdot D \cdot 4D}{D \cdot 4D \cdot 2 + 1 \cdot D \cdot 2} \;\approx\; 1 \quad\text{FLOP/byte}.
$$

One floating-point operation per byte loaded. The H100 ridge is at ~93. We are **two orders of magnitude below it**. The tensor cores are idle. The chip is, functionally, an oversized HBM controller. Every decode step has to drag the entire 140 GB of Llama-70B weights across the memory bus to produce a single token. The bandwidth is 4.8 TB/s. The model is 140 GB. A back-of-the-envelope **lower bound** on per-token decode latency is

$$
t_\text{decode}^\text{min} \;\approx\; \frac{140\,\text{GB}}{4.8\,\text{TB/s}} \;\approx\; 29\,\text{ms per token}.
$$

This is not an implementation detail. **It is a thermodynamic floor.** No amount of kernel tuning can move it. The only way to get below 29 ms/token is to load fewer bytes (quantization, pruning, MoE), load them less often (speculative decoding, batching), or use them more times once loaded (FlashAttention-style tile reuse). Three options. The entire field is variations on those three.

Pause on that for a second. Prefill saturates the chip's most expensive resource (the tensor cores, the thing NVIDIA charges $40,000 for). Decode underutilizes it by a factor of twenty-something, *by construction*. A decode-heavy serving cluster is a cluster of GPUs whose tensor cores are mostly off. The cost-per-token math of the entire LLM serving industry — the reason inference is expensive, the reason providers throttle, the reason a million-dollar box can serve only a few hundred chat users — flows directly out of that 1 FLOP/byte number. It is the **single most important number** in this issue, and we will hit it from a different angle in every chapter from here on.

{{< crosshead >}}Two Metrics In Tension{{< /crosshead >}}

The systems community has names for the two regimes that anyone running a serving cluster has to keep on a Grafana board:

- **TTFT** — *time to first token*. Dominated by prefill. The user pastes their prompt, hits send, and stares at a spinner. TTFT is what they feel during that stare.
- **TPOT** (also called **ITL**, *inter-token latency*) — *time per output token*. Dominated by decode. Once the first token lands, every subsequent character of the streaming response arrives one TPOT apart.

These two metrics live on **opposite axes of the same plot**, and every serving decision is a Pareto point on the trade-off between them.

```pyplot {id="profile-trace" caption="Profile trace: 600 ms dense prefill (one wide bar) followed by 100 decode steps (~25 ms each). HBM utilization spikes during decode; tensor-core utilization spikes during prefill. Same GPU, two regimes."}
np.random.seed(11)

prefill_ms   = 600.0
decode_steps = 60         # fewer steps so the plot stays readable
tpot_ms      = 25.0
gap_ms       = 1.0        # CPU-side scheduling gap between decode steps

# Compute total span: prefill, then a string of (tpot + gap) intervals.
ts = []
t = 0.0
ts.append(("prefill", t, prefill_ms))
t += prefill_ms
for _ in range(decode_steps):
    ts.append(("decode", t, tpot_ms))
    t += tpot_ms + gap_ms
total = t

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2]})

# Top: Gantt-style row.
for kind, start, dur in ts:
    color = '#FF8C00' if kind == "prefill" else '#00A8A8'
    ax1.barh(0, dur, left=start, height=0.6,
             color=color, edgecolor='#1A1A1A', linewidth=0.8)
ax1.text(prefill_ms / 2, 0.45, "PREFILL\n600 ms compute-bound",
         ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1A1A1A')
ax1.text(prefill_ms + (total - prefill_ms) / 2, 0.45,
         "DECODE\n60 × ~25 ms bandwidth-bound",
         ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1A1A1A')
ax1.set_ylim(-0.6, 1.1)
ax1.set_yticks([])
ax1.set_title("Llama-70B, H100 TP=4, one user, 4K-token prompt → 60 output tokens",
              loc='left', fontsize=10)
ax1.spines[['top', 'right', 'left']].set_visible(False)

# Bottom: stylized utilization curves.
t_axis = np.linspace(0, total, 2000)
tc_util = np.zeros_like(t_axis)
hbm_util = np.zeros_like(t_axis)
for kind, start, dur in ts:
    mask = (t_axis >= start) & (t_axis < start + dur)
    if kind == "prefill":
        tc_util[mask] = 0.92 + 0.05*np.sin(20*t_axis[mask])
        hbm_util[mask] = 0.55 + 0.05*np.sin(15*t_axis[mask])
    else:
        tc_util[mask] = 0.04 + 0.02*np.random.randn(mask.sum())
        hbm_util[mask] = 0.78 + 0.05*np.sin(40*t_axis[mask])

ax2.fill_between(t_axis, 0, tc_util, color='#FF007F', alpha=0.55,
                 label='Tensor-core util')
ax2.fill_between(t_axis, 0, hbm_util, color='#FFD700', alpha=0.55,
                 label='HBM util')
ax2.set_xlabel("Wall-clock time (ms)")
ax2.set_ylabel("Utilization")
ax2.set_ylim(0, 1.05)
ax2.legend(loc='center right', framealpha=0.95)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print(f"prefill duration  : {prefill_ms:.0f} ms (one slab)")
print(f"decode per token  : {tpot_ms:.0f} ms (× {decode_steps} steps)")
print(f"total wall-clock  : {total:.0f} ms")
print(f"prefill share     : {prefill_ms / total:.0%}")
```

The two utilizations are **anti-correlated**. During the 600 ms slab, tensor cores light up and HBM looks unremarkable. During the picket-fence section, the cores idle and HBM hits 78%. The trace is two different jobs taped together.

## The Latency Budget, Itemized

Let us pin down some numbers, because the orders of magnitude are how the rest of this issue gets its shape. **Llama-70B**, tensor-parallel across four H100s (TP=4), FP16 weights — the standard 2025 deployment. Numbers below are within ~20% of what Nsight reports.

| Phase | Operation | Per-token cost | At $T = 4096$ |
|---|---|---|---|
| **Prefill** | Attention (FlashAttention-2) | $2 T D + 2 T^2 D$ FLOP/layer | $\sim 90$ ms across 80 layers |
| **Prefill** | MLP (gated SwiGLU, $4D$) | $24 T D^2$ FLOP/layer | $\sim 230$ ms across 80 layers |
| **Prefill** | All-reduce (TP=4) | $T \cdot D \cdot 2$ bytes/layer | $\sim 20$ ms across 80 layers |
| | **TTFT** | | **$\sim 340$ ms** |
| **Decode** | Attention over $T+k$ KV | dominated by KV-cache reads | $\sim 8$ ms/token |
| **Decode** | MLP | dominated by weight reads | $\sim 14$ ms/token |
| **Decode** | All-reduce | small | $\sim 2$ ms/token |
| | **TPOT** | | **$\sim 24$ ms/token** |

For a short reply (200 output tokens):

$$
\text{end-to-end} \;=\; \underbrace{340\,\text{ms}}_{\text{prefill}} + \underbrace{200 \times 24\,\text{ms}}_{\text{decode}} \;=\; 5.1\,\text{seconds}.
$$

Prefill is 7% of the wall-clock. The user mostly waits on decode.

But flip the regime — long-context query, 32K input, 200 output tokens — and the picture inverts. Prefill scales as $T \cdot D$ in the MLP and as $T^2 \cdot D$ in the attention term. At $T = 32{,}768$:

$$
\text{end-to-end} \;=\; \underbrace{\sim 3.2\,\text{s}}_{\text{prefill}} + \underbrace{200 \times 26\,\text{ms}}_{\text{decode}} \;=\; 8.4\,\text{seconds},
$$

prefill is now **38%** of wall-clock and TTFT, not TPOT, is what the user feels. The same model, the same code path; only $T$ moved.

{{% callout type="theorem" %}}
**The Asymmetry Law.** Prefill cost is $O(T^2 D)$ in attention and $O(T D^2)$ in MLP. Decode cost is $O(T D)$ in attention and $O(D^2)$ in MLP, *per token*. As $T$ grows, prefill cost rises super-linearly while decode cost per token rises linearly. **There is always a context length above which prefill dominates and below which decode dominates.** For 70B-class models, the crossover sits around $T \approx 10\text{K}$ for typical short replies.
{{% /callout %}}

The flip is one of the most important inflection points in serving-economics math. A code-completion product (200-token prompts, 20-token completions) is overwhelmingly decode-bound; its dollars-per-token are mostly the HBM bus on a decode-friendly GPU. A long-document RAG product (32K-token prompts, 100-token answers) is overwhelmingly prefill-bound; its dollars-per-token are mostly the tensor cores on a prefill-friendly GPU. **They are different businesses.** The same model serves both, but the bill of materials, the GPU choice, and the SLA shape are different.

## The Head-of-Line Problem

Here is where the systems story gets ugly, and where the rest of the issue earns its keep.

The GPU is a single shared resource. If User A submits a 32K-token query (a 3-second prefill) and User B's decode is mid-stream, **User B's next token cannot be produced until User A's prefill finishes**. User B has been waiting on a 25 ms cadence; suddenly there is a 3-second stall in the middle of their reply. From their phone screen, it looks like the model froze.

This is **head-of-line blocking**, and it is the disease that organizes the next half of this issue. There are exactly three angles of attack — every modern serving system uses some combination of them:

1. **Run prefill and decode on different machines.** Send prefills to a "prefill cluster" optimized for throughput, decodes to a "decode cluster" optimized for latency. Ship the KV cache across the network when prefill is done. This is the [disaggregated prefill/decode](../17-disagg-pd/) architecture (DistServe, Mooncake, NIXL).

2. **Slice prefill into pieces small enough to interleave with decode.** A 3-second prefill becomes thirty 100 ms chunks; each chunk piggy-backs on a regular decode step. The user mid-stream sees a 25 ms cadence stretching to 125 ms instead of stalling for 3 seconds. This is [chunked prefill](../13-chunked-prefill/).

3. **Merge prefill and decode into one workload and let the scheduler arbitrate.** Every iteration, allocate a token budget across active requests — some tokens go to a prefill-in-progress, others to decode steps. The split stops being a phase distinction and becomes a *number*. This is the [vLLM V1 unified scheduler](../14-scheduler/), and it is the surprise twist at the end of this issue.

The point worth holding in mind right now: **all three optimizations exist because of the arithmetic intensity asymmetry**. The two workloads have different shapes, different bottlenecks, and different latency budgets, but they share one piece of silicon. Every system in modern LLM serving is a way of arbitrating that share.

## Cranking The Batch: The Pareto Curve

Decode is bandwidth-bound, which means you can *batch* it almost for free. The expensive part of decode is loading the weights from HBM. If you load the weights once and use them against eight users' decode tokens simultaneously, the per-token cost falls by close to 8×. **Batching is the free lunch of decode.**

But batching has a cost on the prefill side. Prefill is already compute-bound at $T = 4096$; adding more requests doesn't speed it up — they queue. The bigger the batch, the longer any individual user waits for their prefill slot. **Throughput goes up; TTFT goes up.**

Plot it.

```pyplot {id="ttft-vs-throughput" caption="As batch size grows, throughput (tokens/sec/GPU) rises while TTFT (ms) also rises. The SLA line — TTFT < 300 ms, the typical chat product requirement — cuts the curve and tells you the largest batch size you are allowed to run."}
np.random.seed(3)

batch = np.arange(1, 33)

# Decode throughput: roughly batch / (TPOT_floor * (1 + epsilon * batch))
# Below the saturating ridge, doubling batch ~doubles throughput.
tpot_floor = 24.0
tpot       = tpot_floor * (1 + 0.04 * (batch - 1))
throughput = batch * (1000.0 / tpot)   # tokens / sec / GPU (decode steady state)

# TTFT: queue-time roughly proportional to (batch - 1) * prefill_chunk
ttft_floor = 120.0
ttft       = ttft_floor + 18.0 * (batch - 1) + 0.4 * (batch - 1)**2

fig, ax1 = plt.subplots(figsize=(9, 4.6))
ax1.set_xlabel("Concurrent requests in batch")
ax1.set_ylabel("Throughput (tokens/sec/GPU)", color='#FF007F')
l1, = ax1.plot(batch, throughput, color='#FF007F', linewidth=2.4,
               marker='o', markersize=4, label='Decode throughput')
ax1.tick_params(axis='y', labelcolor='#FF007F')
ax1.spines[['top']].set_visible(False)

ax2 = ax1.twinx()
ax2.set_ylabel("TTFT (ms)", color='#00A8A8')
l2, = ax2.plot(batch, ttft, color='#00A8A8', linewidth=2.4,
               marker='s', markersize=4, label='TTFT (p50)')
ax2.tick_params(axis='y', labelcolor='#00A8A8')
ax2.spines[['top']].set_visible(False)

# SLA line at TTFT = 300 ms
ax2.axhline(300, color='#1A1A1A', linewidth=1.4, linestyle='--')
ax2.text(1.4, 310, "TTFT SLA ≤ 300 ms", fontsize=9, fontweight='bold',
         color='#1A1A1A')

# Where does TTFT cross 300 ms?
crossing = np.where(ttft > 300)[0]
if len(crossing):
    bx = batch[crossing[0]]
    ax1.axvline(bx, color='#FFD700', linewidth=2, alpha=0.6)
    ax1.text(bx + 0.4, throughput.max() * 0.55,
             f"Largest legal\nbatch = {bx-1}",
             fontsize=9, fontweight='bold', color='#1A1A1A')

ax1.legend(handles=[l1, l2], loc='upper left', framealpha=0.95)
ax1.set_title("The Pareto frontier: decode throughput vs prefill TTFT",
              loc='left', fontsize=10)
plt.tight_layout()

print(f"At batch=1  : throughput = {throughput[0]:5.1f} tok/s,  TTFT = {ttft[0]:5.0f} ms")
print(f"At batch=8  : throughput = {throughput[7]:5.1f} tok/s,  TTFT = {ttft[7]:5.0f} ms")
print(f"At batch=16 : throughput = {throughput[15]:5.1f} tok/s,  TTFT = {ttft[15]:5.0f} ms")
print(f"At batch=32 : throughput = {throughput[31]:5.1f} tok/s,  TTFT = {ttft[31]:5.0f} ms")
print(f"\nA TTFT SLA of 300 ms caps the batch around {bx-1}.")
print(f"Beyond that, you are trading user-felt latency for raw throughput.")
```

The two curves bend in opposite directions. **There is no batch size that maximizes both.** You either accept a long TTFT to extract throughput out of the decode-bound regime, or you keep TTFT tight and throw money at GPUs.

The horizontal dashed line is your **service-level agreement** — a typical chat product promises sub-300 ms TTFT, and the SLA cuts the throughput curve at a specific batch size. That cut is the largest legal batch you can run. Make TTFT looser (a documentation summarization product, no real-time pressure) and the legal batch grows. Make TTFT tighter (a code-completion product where every keystroke matters) and the legal batch shrinks toward 1.

The entire economics of serving — dollars per token, GPUs per QPS — sits on this plot.

## What This Sets Up

We now know enough to predict the shape of every chapter to come.

- **The KV cache has two birthrates.** Prefill creates $T$ KV entries at once; decode adds one per step. The allocator has to handle both. The result is a fragmented heap — the subject of [The KV Cache Is a Heap](../09-kv-fragmentation/) — that PagedAttention finally tames by borrowing a 1965 operating-system trick.

- **The batch has two clocks.** Prefill takes 600 ms; decode takes 25 ms. Pinning a request to a 600 ms-aligned schedule is murder for inter-token latency. The fix — *iteration-level scheduling*, where the batch is rebuilt every forward pass — is the [Conveyor Belt](../08-continuous-batching/), and it is the work of one OSDI 2022 paper.

- **The two phases live on opposite hardware.** Eventually someone is going to notice that you can buy throughput-optimized GPUs (lots of FLOPs, less bandwidth) for prefill and bandwidth-optimized GPUs (slower compute, fast HBM) for decode, run them in separate physical clusters, and ship the KV cache between them over a 400 Gb/s fabric. That happens in 2024–25 and gets us [Disaggregated Prefill/Decode](../17-disagg-pd/).

- **The two phases stop being phases.** The story has a twist. By 2025, the vLLM V1 scheduler observes that prefill is just "decode with more tokens this step" and merges them: every iteration, allocate a token budget across requests, where each request says "I want $k$ tokens of work" and the scheduler picks. The dichotomy dies. The [scheduler chapter](../14-scheduler/) is where the asymmetry gets quietly reabsorbed into a single, generalized scheduling primitive.

**The prefill/decode split is the field's organizing principle for three years (2022 to 2025).** It is the right mental model for understanding what every optimization in this issue is trying to do. It will also stop being the right model by the time we get to chapter 14, and the merging is its own kind of payoff. Hold the tension. The two regimes are categorically different at the hardware level — but at the software level, eventually, they collapse into one schedule with one budget. Both things are true at once.

**Continue to → [The Conveyor Belt](../08-continuous-batching/)** — the 2022 paper that made it possible to serve more than one user at a time without throwing away most of the GPU.
