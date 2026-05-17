---
title: "Inside the Silicon"
description: "A GPU is a throughput machine: 132 streaming multiprocessors, thousands of simultaneous threads, and a memory hierarchy that spans four orders of magnitude. Understanding the hardware is the foundation for every inference optimization that follows."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T09:30:00-04:00
issue: 8
weight: 20
techKind: primer
techNode: gpu-anatomy
header: 02-gpu-anatomy.webp
---

## Mountain View, October 1999

A man in a black leather jacket walks onto a stage at the Computer Game Developers Conference in Santa Clara. **Jensen Huang** has just turned 36. The company he started in a Denny's booth six years ago is named after the Latin word for "envy", and it has thirty days of cash left in the bank. He is about to announce a chip he has been calling the **GeForce 256**, a name dreamed up by his marketing director, with one new piece of jargon stapled on the front: **"the world's first GPU."**

What "GPU" meant in October 1999 was very specific: a single die that combined a triangle setup engine, a rasterizer, two pixel pipelines, and a transform-and-lighting unit. **Twenty-three million transistors.** Roughly 480 megaflops of geometry throughput, which Jensen described in his keynote as "ten million polygons per second", because polygons sold games.

Twenty-five years later, the same company's H200 is on the operating table in Ohio. It has **80 billion transistors**, **989 teraflops** of dense FP16 throughput on its tensor cores, **141 GB of HBM3e**, and a 700-watt thermal envelope. The 2024 chip is roughly **two million times** the polygon engine of 1999.

But "two million times faster than 1999" is a misleading headline. What actually changed, structurally, is the story of this article. The H200 is not a *faster* GeForce 256. It is a profoundly different kind of computer — a **throughput machine**, optimized along axes the CPU world deliberately refused to optimize. To understand why every inference optimization in this issue looks the way it looks, you have to understand the shape of the machine they are optimizing for.

## CPUs Versus GPUs: A Tale Of Two Floor Plans

The processor on your laptop is descended from the 1971 Intel 4004, which had four-thousand transistors and ran one instruction at a time. Fifty-five years of CPU design have been about making *one thread of execution* run as fast as possible: out-of-order execution, branch prediction, deep pipelines, vast caches, speculative loads. A modern AMD EPYC 9654 has **96 cores**; with two-way SMT it can run **192 hardware threads** in parallel. Each of those threads is exceptionally smart and individually fast.

The H200 makes the opposite bet. It has **132 streaming multiprocessors** (SMs), each of which runs **2,048 threads** concurrently. That is **270,336 threads** in flight at once.

```pyplot {id="cpu-vs-gpu-parallelism" caption="Concurrent hardware threads: a 96-core EPYC versus a 132-SM H200. Same chip area. Four orders of magnitude more parallelism."}
np.random.seed(42)

# CPU grid: 96 cores in 12x8 layout
cpu_rows, cpu_cols = 8, 12
# GPU grid: 270k threads is unphotographable; show 132 SMs each as a cluster of 2048
gpu_sms_rows, gpu_sms_cols = 11, 12

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

# Left: CPU — 96 fat cores
core_w, core_h = 0.8, 0.55
for r in range(cpu_rows):
    for c in range(cpu_cols):
        ax1.add_patch(plt.Rectangle((c, r), core_w, core_h,
                                     facecolor='#00A8A8', edgecolor='#1A1A1A',
                                     linewidth=0.8))
ax1.set_xlim(-0.5, cpu_cols + 0.2)
ax1.set_ylim(-0.5, cpu_rows + 0.2)
ax1.set_aspect('equal')
ax1.set_title("96-core EPYC 9654\n192 hardware threads",
              fontsize=11, fontweight='bold', loc='left')
ax1.text(cpu_cols/2, -0.4, "Each square: 1 SMT core\n(~30M transistors, ~500K/threadslot)",
         ha='center', va='top', fontsize=8, color='#1A1A1A')
ax1.axis('off')

# Right: GPU — 132 SMs, each a tiny dense cluster
sm_w, sm_h = 0.95, 0.65
for r in range(gpu_sms_rows):
    for c in range(gpu_sms_cols):
        idx = r * gpu_sms_cols + c
        if idx >= 132:
            break
        # SM itself
        ax2.add_patch(plt.Rectangle((c, r), sm_w, sm_h,
                                     facecolor='#FF007F', edgecolor='#1A1A1A',
                                     linewidth=0.4))
        # Sparkle dots inside SM to convey "lots of threads"
        ys = r + 0.15 + 0.10 * np.arange(4)
        xs = c + 0.1 + 0.18 * np.arange(5)
        XX, YY = np.meshgrid(xs, ys)
        ax2.scatter(XX, YY, s=2, color='#FFD700', alpha=0.85, zorder=3)

ax2.set_xlim(-0.5, gpu_sms_cols + 0.5)
ax2.set_ylim(-0.5, gpu_sms_rows + 0.2)
ax2.set_aspect('equal')
ax2.set_title("132-SM H200\n270,336 concurrent threads",
              fontsize=11, fontweight='bold', loc='left')
ax2.text(gpu_sms_cols/2, -0.4,
         "Each pink rectangle: 1 SM = 2,048 threads in 64 warps\n"
         "(~600M transistors per SM, ~290K/threadslot)",
         ha='center', va='top', fontsize=8, color='#1A1A1A')
ax2.axis('off')

print(f"Concurrent thread ratio (GPU / CPU): {270336 / 192:.0f}x")
print(f"Transistor budget per concurrent thread:")
print(f"  EPYC 9654: ~12B trans / 192 threads = ~62M trans/thread")
print(f"  H200:      ~80B trans / 270k threads = ~296K trans/thread")
print(f"  The GPU spends ~200x less silicon per thread.")
```

That ratio — **1,408×** more concurrent threads — comes at a price the chip designer knowingly pays. Each H200 thread is *stupider* than a CPU thread by every metric you might care to name. No branch predictor. No out-of-order execution. No private cache. When two threads in the same warp disagree about which branch to take, the GPU serializes them — runs the `if` and the `else` consecutively, masking out whichever threads are on the wrong side. A single ill-placed `if` can halve throughput. The whole machine is a bet that **for the workloads you care about, you don't need a smart thread; you need a million dumb threads**.

The workload they had in mind in 1999 was video games. Pixels are embarrassingly parallel: every pixel in a frame can be shaded independently. It turned out — and this is the joke at the heart of the modern AI boom — that **neural networks have the same shape as pixels.** Every element of a matrix multiplication can be computed independently. Every position in a sequence can be processed in parallel during prefill. The hardware that was built to shade *Quake III* in 1999 is the same hardware that is now serving Anya's tokens in Ohio, scaled up by six orders of magnitude.

## A Short Hardware History, Ten Punch Cards Deep

The GPU did not become a numerical accelerator on day one. It took a decade.

| Year | Chip | The new idea |
|---|---|---|
| **1999** | GeForce 256 | Fixed-function transform & lighting. 23M transistors. The word "GPU" gets invented. |
| **2001** | GeForce 3 | Programmable *vertex* and *pixel* shaders. Tiny assembly-language programs run per-pixel. |
| **2006** | G80 (GeForce 8800) | **CUDA**. Unified shader cores become general SIMT processors. ({{< cite text="Nickolls et al., Scalable Parallel Programming with CUDA, 2008" url="https://dl.acm.org/doi/10.1145/1365490.1365500" kind="paper" >}}) |
| **2012** | Kepler GK110 (Tesla K20) | AlexNet trains on two of these. The deep learning era begins. |
| **2017** | Volta V100 | **Tensor cores**: a 4×4×4 mixed-precision matmul in one clock. ({{< cite text="NVIDIA Volta whitepaper, 2017" url="https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf" kind="doc" >}}) |
| **2020** | Ampere A100 | TF32, sparsity, 40→80 GB HBM. The dominant ChatGPT training chip. |
| **2022** | Hopper H100 | FP8, transformer engine, DPX instructions, 80 GB HBM3. |
| **2024** | H200 | Same chip, 141 GB HBM3e, 4.8 TB/s bandwidth — pure memory upgrade for the inference era. |
| **2024** | Blackwell B200 | Dual-die, FP4, 192 GB HBM3e, 8 TB/s. ({{< cite text="NVIDIA Blackwell whitepaper, 2024" url="https://resources.nvidia.com/en-us-blackwell-architecture" kind="doc" >}}) |

Two arrows run through the table. **The first** is the slow conversion of programmable graphics shaders into general-purpose numerical units — fixed pipeline → programmable shaders → unified shaders → CUDA cores → tensor cores. Each generation the silicon got a little more "do what I say" and a little less "do what I was wired for". **The second** is the steady marriage to deep learning. After 2012's AlexNet result, NVIDIA stopped pretending these chips were graphics processors first. The 2017 Volta tensor core is unambiguously a matrix multiplier with a vestigial fragment shader stuck to the side.

By 2024 the vestigial fragment shader is gone. The B200 has no graphics pipeline. It is a matrix multiplier with a memory controller.

## The SM, Up Close

The atomic unit of GPU compute is the **streaming multiprocessor**. Everything else — the kernel launch, the warp scheduler, the global memory bus — exists to keep SMs fed. An H100 / H200 SM contains:

- **Four sub-partitions**, each with its own warp scheduler, register file, and dispatch unit.
- Per sub-partition: **16 FP32 ALUs**, **16 INT32 ALUs**, **8 FP64 ALUs**, and one full-width **tensor core**.
- A shared **register file** of 65,536 32-bit registers (256 KB).
- **228 KB** of combined L1 / shared memory.
- A texture unit (mostly vestigial for AI workloads).

```pyplot {id="sm-anatomy" caption="One H100 streaming multiprocessor. Four sub-partitions, each its own scheduler. The tensor cores do most of the AI work; the CUDA cores do everything else."}
fig, ax = plt.subplots(figsize=(10, 6))

# Outer SM box
ax.add_patch(plt.Rectangle((0.2, 0.3), 11.6, 7.0,
                            facecolor='#FDF5E6', edgecolor='#1A1A1A',
                            linewidth=2.5))
ax.text(6, 7.5, "STREAMING MULTIPROCESSOR (SM)",
        ha='center', va='bottom', fontweight='bold', fontsize=12)

# Four sub-partitions
sub_positions = [(0.5, 4.2), (3.4, 4.2), (6.3, 4.2), (9.2, 4.2)]
for i, (x, y) in enumerate(sub_positions):
    # Sub-partition box
    ax.add_patch(plt.Rectangle((x, y), 2.4, 2.9,
                                facecolor='#FFD700', alpha=0.55,
                                edgecolor='#1A1A1A', linewidth=1.5))
    ax.text(x + 1.2, y + 2.7, f"sub-partition {i}",
            ha='center', va='top', fontsize=9, fontweight='bold')
    # FP32 lanes
    ax.add_patch(plt.Rectangle((x + 0.15, y + 1.7), 2.1, 0.4,
                                facecolor='#FF007F', edgecolor='#1A1A1A'))
    ax.text(x + 1.2, y + 1.9, "16 FP32 lanes",
            ha='center', va='center', fontsize=8, color='#FDF5E6', fontweight='bold')
    # INT32 lanes
    ax.add_patch(plt.Rectangle((x + 0.15, y + 1.2), 2.1, 0.4,
                                facecolor='#00A8A8', edgecolor='#1A1A1A'))
    ax.text(x + 1.2, y + 1.4, "16 INT32 lanes",
            ha='center', va='center', fontsize=8, color='#FDF5E6', fontweight='bold')
    # Tensor core
    ax.add_patch(plt.Rectangle((x + 0.15, y + 0.5), 2.1, 0.6,
                                facecolor='#FF8C00', edgecolor='#1A1A1A',
                                linewidth=1.5))
    ax.text(x + 1.2, y + 0.8, "TENSOR CORE",
            ha='center', va='center', fontsize=9, fontweight='bold', color='#1A1A1A')
    # Register file
    ax.add_patch(plt.Rectangle((x + 0.15, y + 0.1), 2.1, 0.3,
                                facecolor='#1A1A1A', edgecolor='#1A1A1A'))
    ax.text(x + 1.2, y + 0.25, "16K registers (64 KB)",
            ha='center', va='center', fontsize=7, color='#FDF5E6')

# Shared memory / L1 (spans the full bottom)
ax.add_patch(plt.Rectangle((0.5, 2.5), 11.0, 1.2,
                            facecolor='#FF007F', alpha=0.3,
                            edgecolor='#1A1A1A', linewidth=1.5))
ax.text(6, 3.1, "Shared Memory + L1 Cache  —  228 KB combined  —  ~19 TB/s",
        ha='center', va='center', fontsize=10, fontweight='bold')

# Warp scheduler header
ax.add_patch(plt.Rectangle((0.5, 1.5), 11.0, 0.7,
                            facecolor='#00A8A8', alpha=0.4,
                            edgecolor='#1A1A1A', linewidth=1.5))
ax.text(6, 1.85, "4 × Warp Scheduler  +  4 × Dispatch Unit  (issues 1 warp/cycle each)",
        ha='center', va='center', fontsize=9, fontweight='bold')

# Caption at bottom
ax.text(6, 0.7,
        "Total: 128 FP32 ALUs · 128 INT32 ALUs · 4 tensor cores · 65,536 registers · 228 KB SRAM\n"
        "Concurrent threads: 2,048  ·  Concurrent warps: 64  ·  Peak FP16 tensor: 989 TFLOP/s",
        ha='center', va='center', fontsize=9, style='italic')

ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

print("SM math:")
print(f"  4 sub-partitions × 16 FP32 lanes = 64 FP32 lanes per SM")
print(f"  But because each lane runs at 2 FMAs/clock and the SM runs at 1.83 GHz:")
print(f"  FP32 peak per SM = 64 × 2 × 1.83 = 234 GFLOP/s")
print(f"  Across 132 SMs: 132 × 234 = 30.9 TFLOP/s of FP32 muscle on the H100.")
print(f"  Add tensor cores: 132 × 989/132 = 989 TFLOP/s of FP16.")
print(f"  Ratio: tensor cores deliver ~32x more arithmetic per second.")
```

The **tensor core** is the protagonist of the modern era. A single H100 tensor core executes a $16 \times 8 \times 16$ FP16-input/FP32-accumulate matmul in one clock — that's 4,096 multiply-accumulates per cycle per tensor core, times four tensor cores per SM, times 132 SMs, times the 1.83 GHz clock = the **989 TFLOP/s** quoted on the H100 spec sheet.

Compare that to the plain CUDA core path: 16 FP32 lanes × 2 FMAs × 4 sub-partitions × 132 SMs × 1.83 GHz = **31 TFLOP/s** of pure FP32. Tensor cores are **~32× faster** than the scalar pipeline. This is the single most important number on the chip. If your kernel is not using tensor cores, you are leaving 97% of the silicon idle.

{{% callout type="note" %}}
**The corollary you will live with for the rest of this issue:** any operation that *can* be expressed as a matrix multiply (matmul, FFN, attention's QK and PV steps) belongs on the tensor cores. Any operation that can't (softmax, layer-norm, sampling, the `if`-statements in beam search) is forced onto the scalar lanes and is, per joule and per clock, *thirty times more expensive*. This is why FlashAttention's central trick — fusing softmax with the matmul — matters so much. See [Attention in SRAM](../06-flash-attention/).
{{% /callout %}}

## Warps, Blocks, Grids: The CUDA Execution Hierarchy

Threads on the GPU do not run individually. They run in lockstep groups of **32**, called **warps**. A warp is the atomic unit of *scheduling*: the warp scheduler picks one ready warp per clock and dispatches it. All 32 threads in a warp execute the same instruction simultaneously, with per-thread masks for branches they disagree on. This execution model is called **SIMT** — Single Instruction, Multiple Threads — and it is the reason `if`-statements are expensive.

Above the warp sits the **block** (also called a *cooperative thread array* or CTA). A block can be 1 to 1024 threads (1 to 32 warps), is assigned to a single SM for its lifetime, and shares two things across all its threads: the SM's **shared memory** and a `__syncthreads()` barrier.

Above the block sits the **grid**, which is the whole kernel launch. A grid can contain millions of blocks, scheduled across all 132 SMs as SM capacity becomes available.

Take a worked example: add two billion-element float vectors, `C = A + B`. The CUDA pattern:

```python
# pseudo-CUDA
def vector_add(A, B, C, N):
    i = blockIdx.x * blockDim.x + threadIdx.x  # global thread index
    if i < N:
        C[i] = A[i] + B[i]

# Launch:
threads_per_block = 256
blocks = (N + threads_per_block - 1) // threads_per_block  # ~3.9M blocks
vector_add[blocks, threads_per_block](A, B, C, N)
```

Each thread handles one element. The grid is 3.9M blocks; only 132 of them can run at a time; the SM warp scheduler picks them off the queue as previous blocks complete. From the programmer's perspective, the GPU is a giant `for i in range(N):` loop; from the hardware's perspective it is millions of dumb threads grinding through a queue. **Both views are correct simultaneously**, which is the cognitive trick of CUDA programming.

{{% marginnote %}}This kind of pure element-wise work is bandwidth-bound — see the next chapter, [The Pyramid of Speed](../03-memory-hierarchy/). The kernel reads 8 GB, writes 4 GB, does 1 GFLOP. The HBM bus is the bottleneck by a factor of ~500.{{% /marginnote %}}

{{< crosshead >}}The Hidden Cost: Kernel Launch{{< /crosshead >}}

Every time you start a kernel, the CPU has to package up its arguments, validate them, walk a few software queues, and finally hand a launch record to the GPU's frontend. The fixed cost of all this is roughly **3–5 microseconds per launch**, even for a kernel that does no work.

This sounds tiny. It is not. At decode time, vLLM runs roughly **1,200 kernels per forward pass** for an 80-layer model (RMSNorm, QKV proj, RoPE, attention, output proj, MLP up, GELU, MLP down, residual add, etc., per layer, plus per-step ops). At 4 µs each, that's **4.8 ms of pure CPU overhead** for a forward pass that, in pure GPU time, takes 2 ms. The CPU is the bottleneck. This is the entire reason **CUDA graphs** exist — pre-record the launch sequence once, replay it in a single hand-off. See [Launches Aren't Free](../05-cuda-graphs/).

## Napkin Math For The H200

Let's collect everything onto a single mental scaffold:

$$
\begin{aligned}
\text{SMs} \;&=\; 132 \\
\text{threads / SM} \;&=\; 2{,}048 \\
\text{warps / SM} \;&=\; 64 \\
\text{registers / SM} \;&=\; 65{,}536 \times 4\text{B} \;=\; 256\text{ KB} \\
\text{SRAM / SM} \;&=\; 228\text{ KB} \\
\text{Total register file} \;&=\; 132 \times 256\text{ KB} \;\approx\; 34\text{ MB} \\
\text{Total SRAM} \;&=\; 132 \times 228\text{ KB} \;\approx\; 30\text{ MB} \\
\text{HBM3e} \;&=\; 141\text{ GB} \;@\; 4.8\text{ TB/s} \\
\text{Tensor-core FP16} \;&=\; 989\text{ TFLOP/s}
\end{aligned}
$$

Now stare at the registers number. **132 SMs × 65,536 registers × 4 bytes = ~34 megabytes** of register file across the whole chip. The H200's register file is *larger than the L1 cache of an EPYC CPU*. The chip is, mechanically, a very wide register file with some arithmetic units stuck to its sides.

Two more pieces of arithmetic that will haunt the rest of this issue:

- **One FMA on a tensor core** = ~1 clock @ 1.83 GHz = ~0.55 nanoseconds.
- **One byte loaded from HBM** = ~600 clocks of round-trip latency = ~330 nanoseconds.

The ratio is roughly **600 to 1**. In the time it takes a single byte to crawl from HBM to a register, the tensor core could have run six hundred multiply-accumulates — except it has nothing to multiply, because the byte hasn't arrived yet. This number is the entire reason the next chapter exists. See [The Pyramid of Speed](../03-memory-hierarchy/).

## What To Remember

1. **A GPU is a SIMT processor.** Thousands of threads in lockstep warps of 32, hundreds of streaming multiprocessors, hundreds of thousands of in-flight threads. It does not make individual operations faster; it makes *many* operations happen at once. Every optimization in this issue is downstream of that.

2. **The real arithmetic muscle is the tensor core, not the scalar CUDA core.** ~32× faster on FP16 matmuls. Modern attention kernels, modern matmul kernels, every load-bearing op in {{< wiki "attention" >}}attention{{< /wiki >}} and the {{< wiki "mlp-block" >}}MLP block{{< /wiki >}} lives on the tensor cores. Everything else — {{< wiki "softmax" >}}softmax{{< /wiki >}}, {{< wiki "normalization" >}}layer norms{{< /wiki >}}, sampling — sits awkwardly on the scalar lanes.

3. **Throughput, not latency.** Wide registers, deep pipelines, no branch prediction, no out-of-order execution. The GPU is a bet that for workloads shaped like graphics or matmul, you'd rather have a million dumb threads than one hundred and ninety-two smart ones. That bet, made in 1999 for video games, pays the salary of every inference engineer in 2026.

**Continue to → [The Pyramid of Speed](../03-memory-hierarchy/)** — the SMs are hungry, and the rest of the issue is the story of feeding them. Four orders of magnitude of bandwidth separate the fastest storage on the chip from the slowest, and *every* inference optimization is a trade along that pyramid.
