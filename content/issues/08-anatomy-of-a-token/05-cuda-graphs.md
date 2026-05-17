---
title: "Launches Aren't Free"
description: "A decode-step forward pass launches hundreds of tiny CUDA kernels. The Python scheduling overhead can eat the entire latency budget before the GPU starts. CUDA graphs capture the launch sequence and replay it as a single GPU command, dropping CPU-side overhead from milliseconds to microseconds."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T11:00:00-04:00
issue: 8
weight: 50
techKind: primer
techNode: cuda-graphs
header: 05-cuda-graphs.webp
---

## A Profile You Have Seen Before

A Discord screenshot, autumn 2023. A graduate student in Zurich pastes a Nsight Systems timeline of a freshly-pulled Hugging Face decode loop running Llama-2-7B on an A100. They typed two messages.

> "Why is my GPU at 8%? It says it's running."
> "Help."

The picture they posted is the same picture every infrastructure engineer has stared at, half-grinning, at some point in their career. The top half of the screen is the GPU timeline. It is a sad meadow of pastel-coloured rectangles, each about $20$ microseconds wide, separated by **wide white gaps** of forty to sixty microseconds. The kernels are running. The GPU is mostly *not* running. Below, in a parallel lane, sits the CPU timeline — and there the CPU is *thrashing*, executing PyTorch Python code, hitting the driver, launching the next kernel. The CPU is the saturated thing. The H100 is the customer at a restaurant where the waiter is still tying their shoes.

Total decode-step wall time: $60$ ms. Sum of GPU kernel times: $4.7$ ms. **Eight percent of utilization.** A $25{,}000 chip pretending to be a $300 chip.

This is the bug **CUDA graphs** were invented to fix. The fix is conceptually trivial — *run the same launch sequence with O(1) driver overhead* — and the work to make it actually function inside a real serving engine took the field three years and produced an architecture, vLLM V1's piecewise compilation, that is one of the cleanest pieces of systems engineering in the modern stack.

This primer walks through both: why the launch overhead exists at all, and how vLLM's piecewise graph carves it back down to nearly zero.

## What A Kernel Launch Actually Costs

Trace the path of a single Python line — `torch.matmul(x, w)` — and watch how it becomes a GPU instruction.

1. Python interpreter resolves the dispatch (PyTorch overload, autograd, autocast, device check). ~$1$–$2$ µs of Python.
2. C++ binding constructs the kernel arguments, picks a CUTLASS configuration, hits a few hashmaps. ~$1$–$3$ µs.
3. `cuLaunchKernel` enters the CUDA driver. Driver checks contexts, copies arguments to a command-buffer slot, pushes onto a stream queue. ~$2$–$4$ µs.
4. The GPU's command processor eventually pulls the launch packet off the stream and dispatches it to the SMs. ~$1$–$2$ µs latency.

Total: **roughly $5$–$10$ µs per kernel launch**, of which the GPU sees the last microsecond or two. Most of that time is CPU-side. None of it is "real" computation.

Now multiply. A 32-layer Llama-2 decode step launches, conservatively:

- `rmsnorm`, `qkv_proj`, `rope`, `attn`, `o_proj`, `add_residual`, `rmsnorm`, `gate_proj`, `up_proj`, `silu`, `down_proj`, `add_residual` — **12 kernels per block**, times 32 blocks = $384$ launches.
- Plus the embedding lookup, the final norm, the LM head, the sampler — call it $\sim 400$ launches per decode step.

{{% marginnote %}}Real numbers from a 2024 Hopper benchmark: HF-naive Llama-2-7B decode launches $\sim 440$ kernels per step at ~$7$ µs Python overhead each, for ~$3$ ms of pure scheduling cost per token — on a model whose useful kernel work fits in $\sim 4$ ms. The CPU is more than 40% of decode wall time.{{% /marginnote %}}

At $7$ µs per launch and $400$ launches: $2.8$ ms of CPU-side overhead **per decoded token**, before the GPU has multiplied anything. If the actual matmul work for one decode step is also a few milliseconds (because we live at $I = 1$ in the [roofline diagram](../04-roofline/)), the CPU is contributing **half of decode wall time** as pure overhead.

Look at where the budget goes for that single decode step at small batch:

```pyplot {id="decode-time-breakdown" caption="Decode step time budget at batch size 1, Llama-2-7B on H100. The CPU launch overhead (pink) eats more than half the wall clock. Each layer's actual compute (teal) is a polite tenant. The white slivers are the gaps between kernels — GPU idle, waiting for the next launch."}
import numpy as np
import matplotlib.pyplot as plt

# 32 layers. Per-layer: ~12 kernels.
# Each kernel: ~7us CPU launch + ~10us GPU run, with ~30us idle gap between them.
n_layers = 32
kernels_per_layer = 12

per_layer = {
    "kernel launch (CPU)": kernels_per_layer * 7,   # us
    "GPU compute":         kernels_per_layer * 10,  # us
    "stream gap (GPU idle)": kernels_per_layer * 30,  # us
}
total_per_layer = sum(per_layer.values())
total = total_per_layer * n_layers

fig, ax = plt.subplots(figsize=(10, 3.4))
left = 0
colors = {
    "kernel launch (CPU)": '#FF007F',
    "GPU compute":         '#00A8A8',
    "stream gap (GPU idle)": '#FFD700',
}
for label, us in per_layer.items():
    w = us * n_layers / 1000   # ms
    ax.barh(0, w, left=left, color=colors[label],
            edgecolor='#1A1A1A', linewidth=1.2, label=label)
    ax.text(left + w/2, 0, f"{w:.2f} ms",
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='#1A1A1A')
    left += w

ax.set_yticks([])
ax.set_xlabel("decode step wall time (ms)")
ax.set_title("Where the 5 ms goes at batch=1: half is the launch loop")
ax.legend(loc='lower right', framealpha=1, edgecolor='#1A1A1A')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.set_xlim(0, left * 1.05)

print(f"per-layer total: {total_per_layer} us")
print(f"step total:      {total} us = {total/1000:.2f} ms")
print(f"pure CPU launch fraction: {per_layer['kernel launch (CPU)'] / total_per_layer:.0%}")
print(f"GPU idle (white) fraction: {per_layer['stream gap (GPU idle)'] / total_per_layer:.0%}")
print(f"useful GPU work fraction: {per_layer['GPU compute'] / total_per_layer:.0%}")
```

The teal "GPU compute" slice in the picture is the work the silicon was bought to do. The pink "kernel launch" slice is the CPU shouting instructions at it. The yellow "stream gap" is the GPU staring at the ceiling between instructions. Eight percent utilization is not a measurement bug. It is *exactly* what this picture predicts.

The cure looks like a single sentence in the CUDA programmer's guide:

> If the launch sequence is the same every step, capture it once and replay it.

That sentence is CUDA graphs, in full.

## Capture Once, Replay Forever

A **CUDA graph** is a recorded directed-acyclic-graph of kernel launches — node = kernel, edge = dependency — together with the arguments and stream-positions each kernel will use. You build it once by running the model in a special "capture" mode and asking the driver to record everything that happens. The output is a frozen object: a `cudaGraphExec_t`. From that point on, instead of launching $400$ kernels through the Python-C++-driver gauntlet, you make **one** driver call — `cuGraphLaunch` — and the entire 400-kernel sequence flies onto the stream at near-hardware speed.

There are three things to understand about the capture, and they are exactly the three constraints that make CUDA graphs annoying to actually use.

1. **Shape-frozen.** Every tensor allocation, every stride, every pointer that goes into a captured kernel is baked into the graph. If at replay time the batch size differs by one row, the graph is *wrong*. You must either re-capture or pad.
2. **Pointer-stable.** The captured graph holds raw device pointers. If your serving engine reallocates intermediate buffers — and PyTorch does, eagerly, all the time — replaying the graph reads garbage. You need to either pin allocations or use a memory pool that won't move them.
3. **Control-flow-free.** A graph is a static DAG. There are no `if`s, no `while`s, no Python branches. If the network has a path that depends on tensor *values* (like "skip attention when KV cache is empty"), you cannot capture both branches as one graph.

Constraint 1 is the killer. A decode-time serving engine sees a *parade* of different batch sizes — 13 users, then 27 users, then 8 users, then a request finished so now 7, then a new prefill arrived so the metadata changes. Capturing a graph for every possible batch size is impossible (and would burn gigabytes of memory storing them). Capturing zero of them means you keep your launch overhead.

The compromise that the field converged on is the **bucket-and-pad** pattern. Pick a small set of canonical batch sizes — say, $\{1, 2, 4, 8, 16, 32, 64, 128, 256, 512\}$. Capture *one graph per bucket*. At runtime, if the in-flight batch is $53$, round it *up* to $64$, fill the extra $11$ slots with dummy padding tokens whose outputs are discarded, and replay the size-$64$ graph.

The cost is wasted FLOPs and HBM bandwidth on $11$ ghost tokens. The savings are thousands of microseconds of launch overhead. In the [roofline framing](../04-roofline/), this is **free**: padding spends compute and bandwidth that were already going to sit idle, and buys back CPU time that was the actual bottleneck.

```pyplot {id="eager-vs-graph-itl" caption="Inter-token latency (ITL) vs batch size, eager vs CUDA-graph replay. At small batch the gap is huge because launch overhead is most of the budget. At large batch the GPU work dominates and the curves meet. The break-even hides above batch 200."}
import numpy as np
import matplotlib.pyplot as plt

batch = np.arange(1, 256)
launches = 400
launch_us = 7      # per kernel, CPU
gap_us = 30        # GPU idle between kernels at batch 1

# Eager: launch overhead is constant; useful GPU work grows linearly with batch.
eager = launches * launch_us + 10e3 + batch * 80    # ms-scale work
# Graph: a single O(1) driver call replaces the per-kernel cost.
graph = 50 + launches * 1.5 + batch * 80            # near-zero launch, same compute

# Convert from us to ms
eager = eager / 1000
graph = graph / 1000

fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.plot(batch, eager, color='#FF007F', linewidth=2.5, label='eager (per-kernel launch)')
ax.plot(batch, graph, color='#00A8A8', linewidth=2.5, label='CUDA graph replay')
ax.fill_between(batch, graph, eager, color='#FFD700', alpha=0.25,
                label='launch-overhead savings')

# Annotate gap at three batch sizes
for b in [1, 16, 128]:
    e, g = eager[b-1], graph[b-1]
    ax.annotate("", xy=(b, g), xytext=(b, e),
                arrowprops=dict(arrowstyle='<->', color='#1A1A1A', lw=1.2))
    ax.text(b + 2, (e+g)/2, f"{(e-g):.1f} ms",
            fontsize=9, fontweight='bold')

ax.set_xlabel("decode batch size")
ax.set_ylabel("inter-token latency  (ms / token)")
ax.set_title("CUDA graphs close the latency gap — most at low batch")
ax.legend(loc='upper left', framealpha=1, edgecolor='#1A1A1A')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Two things in that plot. First, eager mode is *strictly worse* at every batch size; CUDA graphs are a free lunch as long as you can capture them. Second, the gap *narrows* as batch grows — because at $B = 256$, the kernels are big enough that they spend most of their time computing rather than launching. The CPU loop catches up. Past some break-even — for Llama-7B on an H100, around $B \approx 200$–$300$ — eager mode becomes acceptable.

This is the reason CUDA graphs are *primarily* a decode optimization. Prefill is already compute-bound, kernel launches are amortized over thousands of useful FLOPs per call, and the eager loop's overhead is a rounding error. Decode is where the launches are the work.

## Piecewise Compilation: The vLLM V1 Move

There is a final wrinkle, and it is the wrinkle that makes CUDA graphs usable in a real serving system.

Naive CUDA-graph capture assumes the *whole forward pass* is shape-frozen. It is not. The {{< wiki "attention" >}}attention{{< /wiki >}} kernel reads from a **block table** ([→ ch.10](../10-paged-attention/)) whose size depends on each request's current KV-cache length. The kernel arguments — list of block ids per request, sequence lengths, page strides — change at every step. You cannot capture them.

Piecewise compilation is the architectural insight that **the transformer body is a long stable suffix attached to a small dynamic prelude**, and the right thing to do is capture only the suffix.

The design lands in {{< cite text="vLLM V1 design notes" url="https://blog.vllm.ai/2025/01/27/v1-alpha-release.html" kind="blog" >}}, primarily authored by **Kaichao You** and the vLLM team. The structure is something like:

```
[ EAGER PRELUDE ]
  - schedule tokens for this step
  - compute attention metadata (which blocks, which lengths)
  - prepare positions, masks, KV indices
  - dispatch into the captured graph for this batch bucket
       ↓
[ CAPTURED SUFFIX ]  (one per batch bucket: 1, 2, 4, 8, ..., 512)
  - for layer in range(L):
        rmsnorm
        qkv projection
        rope
        ATTENTION_CALLABLE(metadata_from_prelude)   ← branches out, then back in
        output projection
        residual add
        rmsnorm
        gated MLP
        residual add
  - final norm
  - LM head
       ↓
[ EAGER POSTLUDE ]
  - sampling, log-prob extraction
  - structured-output masking
  - update KV cache pointers
```

Two clever things happen here. The first is the boundary: attention metadata is *computed eagerly* in the prelude, while attention's *kernel call* is invoked from inside the graph. This is the literal *piecewise* in "piecewise compilation." Most of the graph is rigid; attention is a registered callable that the graph branches into, receiving its dynamic arguments through pre-allocated metadata tensors. Inside the captured graph, the *control flow* is fixed even though the underlying attention kernel reads variable-length data.

The second is the bucketing. vLLM V1 captures graphs at a curated set of batch sizes — typically $1, 2, 4, 8, 16, \dots, 512$ — and pads the live batch up to the nearest bucket at every step.

Here is what the **padding cost** looks like in practice. Suppose the scheduler has $53$ active decodes, which is between buckets $32$ and $64$.

- **Live batch:** $53$ tokens worth of useful work.
- **Padded batch:** $64$ slots — $11$ ghost tokens whose outputs are computed and discarded.
- **Wasted FLOPs:** $11/64 \approx 17\%$.
- **Wasted HBM bandwidth:** same fraction.
- **Saved CPU time:** $\sim 2.8$ ms of launch overhead per step that *would have happened in eager mode*.

In the [roofline frame](../04-roofline/), we live at intensity $I \approx B$ in the decode regime. The marginal cost of $11$ extra ghost tokens is precisely $11/64$ of the bandwidth-limited budget — i.e., a tiny fraction of an already-cheap millisecond. The marginal saving is the *entire* launch loop. The trade is overwhelmingly in your favor at small batches; it becomes barely worth it past $B \approx 256$, which is also where vLLM stops bucketing.

{{% callout type="tip" %}}**A heuristic for graph design.** Capture the **largest piece of the model whose shapes are determined by the batch bucket alone**, and keep the dynamic-shape operations (attention, sampling) outside. The captured piece's job is to reduce the launch count from "one per layer × per kernel" to "one for the whole stable middle." vLLM's piecewise design is the cleanest worked example, but the principle is broader — see how TensorRT-LLM, SGLang, and TGI all converged on the same partitioning.{{% /callout %}}

## What Stays Eager (And Why)

For completeness, here is the catalogue of operations vLLM does *not* capture into the graph, and why each one resists capture.

- **Attention.** Block-table size, per-request KV length, optional sliding-window mask — all dynamic.
- **Sampling.** Top-k, top-p, temperature, structured-output masks — the masks themselves vary per request and per step.
- **Log-prob extraction.** Variable-rank index gather based on which tokens the caller asked for.
- **KV cache writeback.** Indices depend on each request's *next free* block, which the block manager computes eagerly.
- **Scheduler.** The whole loop that decides which requests to run this step lives in Python; the graph is just one part of one step.

Everything else — every linear, every norm, every activation, every residual add — is fair game and gets captured.

## When CUDA Graphs Don't Help (Or Hurt)

A short list of failure modes that anybody touching the graph layer will eventually meet, written so you can recognize them.

1. **Tiny model, large batch.** If kernel launches were never the bottleneck — e.g., you're running a 1B model at batch 256 — capturing buys you nothing and costs you the capture-time and the bucketing memory.
2. **Frequent re-capture.** If your engine reshapes things often (LoRA hot-swap, dynamic quantization, weight streaming), you may end up re-capturing graphs at runtime, which is *more* expensive than running eager.
3. **Memory pool drift.** PyTorch's caching allocator sometimes hands out a pointer that didn't exist at capture time. Symptom: an inscrutable `CUDA_ERROR_INVALID_GRAPH_ARGS`. The fix is to use a persistent allocator (vLLM does this with its own pool) and to pin every intermediate tensor.
4. **Profile-then-replay drift.** A graph captured under one driver minor version sometimes mis-replays after a driver upgrade. Re-capture on engine start.
5. **Mixed prefill + decode in one step.** Chunked prefill ([→ ch.13](../13-chunked-prefill/)) is the case where a captured-suffix graph and an eager-prefill prelude have to be glued together in one step. vLLM V1's scheduler treats this carefully; older systems had bugs here for months.

If you ever read a bug report in the inference-engine world that says "the model collapsed after a driver upgrade and we don't know why," there is a 30% chance the answer involves CUDA graphs. They are the most powerful and the most brittle optimization in the entire stack.

## What To Remember

1. **At decode time the CPU can be the bottleneck**, not the GPU. ~$7$ µs per launch × hundreds of layers × thousands of tokens/sec is hundreds of milliseconds of pure overhead.
2. **CUDA graphs replay a captured launch sequence as a single driver call** with O(1) overhead. vLLM applies this *piecewise* — the stable transformer suffix is captured, the dynamic prelude (attention metadata, sampling) stays eager.
3. **Padding to the nearest captured batch-size bucket is nearly always worth it.** Wasted FLOPs are cheap when you live in the bandwidth-bound regime. You are buying CPU time with bandwidth time you weren't using anyway.

**Continue to → [Attention in SRAM](../06-flash-attention/)** — the last GPU primer before the engine takes over. The single most important inference kernel of the decade.
