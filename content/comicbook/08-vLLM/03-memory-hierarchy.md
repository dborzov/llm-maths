---
title: "Memory Hierarchy: four orders of magnitude on one chip"
short_title: "Memory Hierarchy"
description: "Registers run at 20,000 GB/s; HBM runs at 4.8 TB/s; NVLink at 900 GB/s; PCIe at 64 GB/s — four orders of magnitude of bandwidth across four rungs, and every inference optimization is a deliberate trade between them."
blurb:
  - "Seymour Cray's 1976 line still holds: memory bandwidth is the only thing that matters."
  - "The H200's 4.8 TB/s HBM3e alone exceeds the combined bandwidth of every Cray supercomputer ever shipped."
  - "Each rung of the pyramid is roughly 10× faster and 100× smaller than the one below it."
  - "FlashAttention, quantization, and paging are all just different ways to spend less time on the slow rungs."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T10:00:00-04:00
issue: 8
weight: 30
techKind: primer
techNode: memory-hierarchy
header: 03-memory-hierarchy.webp
---

## Chippewa Falls, 1976

A bearded man in a flannel shirt is pacing the floor of a converted ski-resort warehouse in northern Wisconsin. **Seymour Cray** is fifty years old, the most famous computer architect alive, and he is about to ship the **Cray-1** — the world's first commercial supercomputer, a horseshoe-shaped tower of 200,000 hand-wired chips, cooled by a Freon-filled fluorocarbon bath, sold for **$8.8 million** to Los Alamos. The Cray-1 will run at 80 megaflops, a number that will hold the world record for four years.

Cray has a line that he repeats to anyone who asks how he designs computers. He repeats it at NSA briefings, at COMDEX, in the *New York Times*. The line is:

> *"You can't fake what you don't have. Memory bandwidth is the only thing that matters."*

The Cray-1's defining feature was not its 80-MHz clock or its vector instructions. It was its **memory subsystem**: a custom-built array of bipolar SRAM, 1 megabyte total, running at 320 MB/s of bandwidth — more than every other commercial computer of 1976 combined. Cray spent more silicon, more money, and more cooling on his memory than on his ALUs. The arithmetic was the easy part. *Feeding the arithmetic* was the hard part.

Fifty years later, the H200 sitting in Ohio has **141 gigabytes** of HBM3e attached to it, running at **4.8 terabytes per second**. The H200's memory bandwidth alone, on a single chip, exceeds the *combined* memory bandwidth of every Cray supercomputer ever shipped — Cray-1, Cray X-MP, Cray-2, Y-MP, T90, C90 — combined.

And Seymour Cray's line is *more true now than it was then*. This article is the story of why.

## The Pyramid

Every computer ever built has had a memory hierarchy because of a brutal physical trade-off: **fast memory is small, big memory is slow**, and you cannot have both. The H200 has four major rungs on its on-chip ladder, each ~10× larger and ~10× slower than the one above. Four rungs. Four orders of magnitude.

```pyplot {id="bandwidth-pyramid" caption="The H200 memory pyramid, log-scale bandwidth. Each rung is ~10x slower and ~100x larger than the one above. Four orders of magnitude across four levels."}
levels = [
    ("Registers",       34,        20_000, "#FF007F"),  # MB, GB/s
    ("Shared / L1",     30,        19_000, "#FF8C00"),
    ("L2 cache",        50,         6_000, "#FFD700"),
    ("HBM3e",       141_000,        4_800, "#00A8A8"),
    ("NVLink",     "(off-chip)",      900, "#1A1A1A"),
    ("PCIe Gen5",  "(host)",           64, "#1A1A1A"),
]

fig, ax = plt.subplots(figsize=(10, 5.5))
bandwidths = [l[2] for l in levels]
names = [l[0] for l in levels]
colors = [l[3] for l in levels]

y_pos = np.arange(len(levels))[::-1]
bars = ax.barh(y_pos, bandwidths, color=colors,
               edgecolor='#1A1A1A', linewidth=1.5, height=0.7)

ax.set_xscale('log')
ax.set_xlim(10, 100_000)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=11, fontweight='bold')
ax.set_xlabel("bandwidth, GB/s (log scale)")
ax.set_title("H200 memory hierarchy — four orders of magnitude on one chip",
             loc='left', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Annotate bar tips with bandwidth + capacity
caps = [l[1] for l in levels]
for i, (bw, cap) in enumerate(zip(bandwidths, caps)):
    y = y_pos[i]
    if isinstance(cap, str):
        label = f"  {bw:,} GB/s  ({cap})"
    elif cap >= 1000:
        label = f"  {bw:,} GB/s  ({cap/1000:.0f} GB total)"
    else:
        label = f"  {bw:,} GB/s  ({cap} MB total)"
    ax.text(bw * 1.05, y, label, va='center', fontsize=9)

print(f"Register-to-HBM bandwidth ratio: {20_000 / 4_800:.1f}x")
print(f"Register-to-PCIe ratio:          {20_000 / 64:.0f}x")
print(f"Total span (registers / PCIe):   {20_000 / 64:.0f}x")
print()
print("Capacity grows in the opposite direction:")
print(f"  Registers ({34} MB) → HBM ({141_000/1000:.0f} GB) is {141_000/34:.0f}x more capacity")
print(f"  But you pay {20_000 / 4_800:.1f}x in bandwidth for that capacity.")
```

The table form of the same data, in case bars are not your love language:

| Level | Per SM | Total chip | Bandwidth | Latency | Scope |
|---|---|---|---|---|---|
| **Registers** | 256 KB | ~34 MB | ~20 TB/s | ~1 cycle | per-thread |
| **Shared / L1 (SRAM)** | 228 KB | ~30 MB | ~19 TB/s | ~30 cycles | per-block |
| **L2 cache** | — | 50 MB | ~6 TB/s | ~200 cycles | chip-wide |
| **HBM3e** | — | 141 GB | 4.8 TB/s | ~600 cycles | chip-wide |
| **NVLink** (peer GPU) | — | — | 900 GB/s | µs | chassis |
| **PCIe Gen5** (host) | — | — | 64 GB/s | µs | system |

Two arrows go in opposite directions. As you climb *up* the pyramid, you gain bandwidth and lose latency. As you climb *down*, you gain capacity. **No level offers both.** This is the physics of the problem, and no clever software trick is going to repeal it.

## Why HBM Exists

For most of computing history, "memory" meant DRAM chips soldered to a motherboard, connected by traces, talking to the CPU through a standardized parallel bus (DDR). The bus width was a function of how many pins you could afford to solder. Even on a server motherboard, the upper bound was ~512 bits wide times a clock around 3 GHz, which put a hard ceiling on bandwidth around 200 GB/s.

In **2015**, a coalition of AMD, SK Hynix, and JEDEC shipped the first **High-Bandwidth Memory** (HBM) on AMD's *Fiji* GPU. The trick was geometric. Instead of soldering DRAM dies to a motherboard, you **stack** them — four to eight DRAM dies, one on top of another, drilled through with **through-silicon vias (TSVs)** to give every layer direct vertical access to the bus. The whole stack sits on a tiny silicon **interposer** *next to* the GPU die, inside the same package. The bus is suddenly **1024 bits wide** because it doesn't have to cross a motherboard — it crosses two millimeters of silicon ({{< cite text="JEDEC HBM3 standard, 2022" url="https://www.jedec.org/standards-documents/docs/jesd238" kind="doc" >}}).

```
   +--------+--------+--------+--------+
   |  HBM   |  HBM   |  HBM   |  HBM   |    <- six stacks (HBM3e), 8 dies each
   |  HBM   |  HBM   |  HBM   |  HBM   |
   |  HBM   |  HBM   |  HBM   |  HBM   |
   +--------+--------+--------+--------+
        |        |        |        |       <- 1024-bit bus per stack
   +--------------------------------+
   |       SILICON INTERPOSER       |       <- ~2mm
   +--------------------------------+
   |          GPU DIE (H200)        |       <- 80B transistors
   +--------------------------------+
```

By 2024's HBM3e the trick has reached **24 GB per stack**, **1.2 TB/s per stack**, six stacks per H200 — giving the 141 GB at 4.8 TB/s headline number. **The bandwidth is the geometry**. You cannot make a board trace into a 1024-bit bus; you can only get there by stacking silicon. Every commercial AI chip from 2024 onward — H200, Blackwell, MI300X, TPU v5p, Groq LPU, Cerebras WSE — uses HBM. Nobody has found a better idea.

{{% marginnote %}}HBM is expensive. A single H200 has roughly *$2,500* of HBM3e on it — about a third of the bill of materials. The 2024 HBM shortage is the binding constraint on global AI capacity, not GPU dies.{{% /marginnote %}}

## Why SRAM Is Tiny

Stare at the pyramid again. **Registers + SRAM = 64 MB on a chip with 141 GB of HBM.** A factor of 2,200 in capacity. Why does the H200 not just put a few gigabytes of SRAM on the die and skip HBM entirely?

The answer is one of the cleanest area trade-offs in semiconductors. **DRAM is 1 transistor + 1 capacitor per bit**. SRAM is **6 transistors per bit**. Per bit of storage, SRAM costs roughly **6× more silicon area** *and* requires more leakage-prone fast logic transistors instead of slow dense storage transistors.

The H200 die is approximately **814 mm²**. If we replaced its 80 GB of HBM with on-die SRAM at 6-transistor SRAM density (~0.03 mm² per Mbit on a leading-edge process), 80 GB of SRAM would require **~20,000 mm² of die area**, which is larger than a dinner plate, which does not fit in a stepper, which is therefore not a chip you can manufacture.

So you stack. Capacity goes off-die into HBM where DRAM density gives you the bytes-per-millimeter to fit; speed stays on-die as a tiny on-chip SRAM for the hot working set. The pyramid is forced by lithography.

## The Cost Of A Single Load

Now the napkin math that justifies the rest of the issue. We want to know: in the time it takes to load one byte from HBM, how much arithmetic could the GPU have done?

- HBM3e round-trip latency: **~330 ns** (≈ 600 clocks at the H200's 1.83 GHz).
- One FMA on a tensor core: **~0.55 ns** (≈ 1 clock).

Naïvely the ratio is 600 FMAs per byte. But this latency number is for a *single* outstanding load — and the GPU is allowed to have thousands of loads in flight simultaneously. The right number to compare is **bandwidth versus arithmetic**: how many bytes can arrive per second, and how many operations can finish per second?

$$
\begin{aligned}
\text{HBM bandwidth} \;&=\; 4.8 \text{ TB/s} \;=\; 4.8 \times 10^{12} \text{ bytes/s} \\
\text{Tensor FP16 throughput} \;&=\; 989 \text{ TFLOP/s} \;=\; 989 \times 10^{12} \text{ FLOPs/s} \\
\text{Ridge ratio} \;&=\; \frac{989}{4.8} \;\approx\; \textbf{206 FLOPs per byte}
\end{aligned}
$$

This is the famous **ridge point** of the H200 roofline ({{< cite text="Williams, Waterman, Patterson — Roofline, 2009" url="https://dl.acm.org/doi/10.1145/1498765.1498785" kind="paper" >}}; see also [The Roofline](../04-roofline/)). For your kernel to use the chip fully, every byte it reads from HBM must trigger at least 206 floating-point operations. If you read a byte and do less than that, you are bandwidth-bound and the tensor cores sit idle waiting.

```pyplot {id="bandwidth-pressure" caption="Two regimes of LLM inference. Prefill keeps tensor cores busy: arithmetic intensity > ridge point. Decode does not: ~2 FLOPs per byte, 100x below the ridge."}
np.random.seed(2)

# Two regimes: prefill (compute-bound), decode (bandwidth-bound)
ridge = 206  # FLOPs per byte on H200

# Synthetic kernels along the roofline
intensities = np.logspace(-1, 4, 200)  # FLOPs / byte
peak_compute = 989  # TFLOP/s
peak_bw = 4.8  # TB/s, so 4.8 PFLOP/s if you could do 1 FLOP/byte

attainable = np.minimum(peak_compute, intensities * peak_bw)

# Mark some real LLM ops
ops = [
    ("decode attention\n(B=1, seq=4k)",     1.0,    'bw'),
    ("decode FFN\n(B=1)",                   2.0,    'bw'),
    ("prefill FFN\n(B=16, seq=2k)",       400.0,    'comp'),
    ("prefill attention\n(B=16, seq=4k)", 600.0,    'comp'),
    ("vector add",                          0.125,  'bw'),
    ("softmax",                             1.0,    'bw'),
]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(intensities, attainable, color='#1A1A1A', linewidth=2.5, label='roofline')
ax.fill_between(intensities, 0, attainable, where=(intensities < ridge),
                color='#FF007F', alpha=0.18, label='bandwidth-bound zone')
ax.fill_between(intensities, 0, attainable, where=(intensities >= ridge),
                color='#FFD700', alpha=0.35, label='compute-bound zone')
ax.axvline(ridge, color='#1A1A1A', linewidth=1.2, linestyle='--')
ax.text(ridge*1.1, 100, f"ridge: {ridge} FLOPs/byte",
        rotation=90, va='bottom', fontsize=9)

for label, x, kind in ops:
    y = min(peak_compute, x * peak_bw)
    color = '#FF007F' if kind == 'bw' else '#FF8C00'
    ax.scatter([x], [y], s=90, color=color, edgecolor='#1A1A1A',
               linewidth=1.4, zorder=4)
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=(8, -5), fontsize=8.5, fontweight='bold')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.05, 5000)
ax.set_ylim(0.3, 2000)
ax.set_xlabel("arithmetic intensity (FLOPs / byte read)")
ax.set_ylabel("attainable TFLOP/s")
ax.set_title("H200 roofline: decode lives on the left of the ridge, prefill on the right",
             loc='left', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(alpha=0.3, linestyle=':')
ax.spines[['top', 'right']].set_visible(False)

print(f"H200 ridge point: {ridge} FLOPs/byte")
print()
print("Where real ops sit:")
for label, x, kind in ops:
    short = label.replace('\n', ' ')
    state = "BW-bound" if x < ridge else "compute-bound"
    print(f"  {short:35s} {x:7.2f} FLOPs/byte  ({state})")
print()
print("Decode is ~100x below the ridge.")
print("This single fact is the entire reason the rest of this issue exists.")
```

Prefill — where you push 2,048 fresh tokens through the model in one shot — has an arithmetic intensity in the high hundreds of FLOPs per byte. Tensor cores stay busy; you are **compute-bound**, and the H200's 989 TFLOP/s is the limit.

Decode — where you push one token per step through the model — has an arithmetic intensity of roughly **1–2 FLOPs per byte**. You are **bandwidth-bound** by a factor of ~100. The 989 TFLOP/s is irrelevant; the binding constraint is the 4.8 TB/s.

{{% callout type="warning" %}}
**This is the entire shape of modern LLM inference.** Every optimization in the rest of this issue — FlashAttention, KV cache paging, continuous batching, speculative decoding, prefill/decode disaggregation — is a variant of the same move: *do more work per byte you load from HBM*, or equivalently, *amortize the HBM load over more arithmetic*. Once you internalize the ridge-point math, you can predict the existence of every chapter in the tech tree.
{{% /callout %}}

## A Worked Example: Loading Llama-3-70B

Consider what it costs to do *one decode step* on the 70B-parameter Llama-3 sitting in Anya's box from [The Cold Open](../01-cold-open/). In FP8, the model weights are 70 GB. Every parameter must be read from HBM once during the forward pass. Then they're multiplied by a single hidden vector (4096 dims, 4 KB), producing 1 output token.

$$
\begin{aligned}
\text{Bytes loaded} \;&=\; 70 \text{ GB} \;\approx\; 7.0 \times 10^{10}\text{ B} \\
\text{FLOPs needed} \;&=\; 2 \;\times\; 70 \times 10^9 \;=\; 1.4 \times 10^{11}\text{ FLOPs (2 per parameter for the matmul)} \\
\text{Arithmetic intensity} \;&=\; \frac{1.4 \times 10^{11}}{7.0 \times 10^{10}} \;=\; \textbf{2 FLOPs / byte}
\end{aligned}
$$

Two. Two FLOPs per byte. The ridge point is 206. **You are using ~1% of the tensor cores' arithmetic capacity** during a decode step on a single user. The chip is mostly waiting on memory.

How long does the decode step take?

$$
t \;\approx\; \frac{70\text{ GB}}{4.8\text{ TB/s}} \;\approx\; 14.6 \text{ ms per decode step at batch=1}
$$

At one user, the H200 spits out about **70 tokens per second** of Llama-3-70B. Now stack 64 users in a batch: the weights are still loaded once (the same 70 GB), but each load now feeds 64 matmuls instead of one. Arithmetic intensity jumps to **128 FLOPs / byte** — much closer to the ridge — and *each* user still gets 70 tokens per second, while the GPU is now processing **64×70 ≈ 4,500 tokens per second of aggregate throughput**.

This is the entire economic logic of LLM serving. **At batch=1 you are wasting the chip; at batch=64 you are saturating it.** Continuous batching, prefix caching, paged attention — every system technique in this issue exists to push effective batch size up without making any one user wait. See [Two Phases, Two Personalities](../07-prefill-vs-decode/).

## Every Important Kernel Is A Tiling Story

Once you accept the pyramid, a single design pattern starts repeating across every important GPU kernel: **bring a tile from HBM, do as much work on it as you can while it's in SRAM, write the result back**. The pattern has a name in numerical computing — **blocked algorithms** — and it dates back to the LINPACK papers of the 1970s.

For LLM inference the canonical example is **FlashAttention** ({{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}}; {{< cite text="Dao et al., FlashAttention, 2022" url="https://arxiv.org/abs/2205.14135" kind="paper" >}}). Vanilla attention computes the score matrix $S = QK^\top$, applies softmax, then multiplies by $V$ — and along the way it writes $S$ (which can be 4k × 4k × 4 bytes = **64 MB** per head) to HBM, reads it back to apply softmax, writes the softmaxed version back, reads it again to multiply by $V$. The score matrix gets read and written *three times*. The arithmetic intensity collapses.

FlashAttention's trick is to **never materialize $S$ in HBM at all**. It loads small tiles of $Q$, $K$, $V$ into SRAM, computes the partial attention output for that tile, and uses an "online softmax" trick to update a running maximum and running sum so that you don't need the full row of $S$ in one place. The result: ~5× lower wall-clock time on long sequences and roughly **zero HBM writes of intermediates**. See [Attention in SRAM](../06-flash-attention/).

Every other important inference kernel — **paged attention** (tiles the K/V across non-contiguous blocks), **fused-MLP** (tiles activations through SRAM between the two MLP matmuls), **fused-rotary** (combines {{< wiki "rope" >}}RoPE{{< /wiki >}} into the QKV projection) — is a variant of the same move. **Keep the working set in SRAM. Read HBM once. Write HBM once.** Anything else is throwing performance away.

{{% pullquote type="technical" %}}
Every important GPU kernel is, at heart, a tiling story. Bring a tile down from HBM. Pin it in SRAM. Do all the arithmetic you can while it's there. Write the result back. The whole stack — Flash, Paged, fused — is variations on this one move.
{{% /pullquote %}}

## Off-Chip: NVLink And PCIe

The two bottom rungs of the pyramid live *off* the chip. **NVLink** is NVIDIA's proprietary point-to-point GPU-to-GPU interconnect — in the H200 generation, 18 lanes per GPU at 50 GB/s each, totaling **900 GB/s of peer bandwidth**. The eight GPUs in Anya's chassis sit on a fully-connected NVSwitch fabric: any GPU can reach any other at the full 900 GB/s. This is what makes tensor parallelism affordable: every transformer block runs an all-reduce after the attention and MLP, and the all-reduce is small (~16 MB) and fast (~17 µs at 900 GB/s). See [Two Slices of the Same Brain](../16-tp-pp/).

**PCIe Gen5** is the link to the host CPU at **64 GB/s** — 75× slower than HBM. This is why model weights are loaded onto GPUs at startup and never moved: re-uploading 70 GB of weights would take a full second over PCIe, an eternity in inference time. Anything that crosses PCIe in the hot path — kernel launches, scheduling decisions, sampling outputs — is by definition a CPU-side overhead the GPU's frontend has to tolerate.

## What To Remember

1. **Four orders of magnitude on one chip.** Register bandwidth (~20 TB/s) is roughly 4× faster than HBM (4.8 TB/s), and HBM is in turn ~75× faster than PCIe. Capacity flows in the opposite direction. Every architectural decision in inference is a placement choice along this pyramid.

2. **Bandwidth, not capacity, is the binding constraint** for the operations that dominate LLM decode. Capacity gates *whether* a request fits on the box. Bandwidth gates *how fast* every token after the first one arrives. The H200's 989 TFLOP/s is mostly aspirational at batch=1.

3. **Every important kernel is a tiling story.** FlashAttention, paged attention, fused-MLP — they are all instances of "keep the tile in SRAM as long as possible." The reason this pattern dominates is that the chip's ridge point is 206 FLOPs per byte and most natural ways of writing the math are ~10 FLOPs per byte. Tiling is how you close the gap. Seymour Cray would have nodded.

**Continue to → [The Roofline](../04-roofline/)** — the napkin-math diagram that compresses everything in this chapter onto two axes. Once you have the roofline, you can predict the performance of any LLM kernel without writing a single line of CUDA.
