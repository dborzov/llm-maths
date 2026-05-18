---
title: "Roofline: the one number that predicts everything"
short_title: "Roofline"
description: "Arithmetic intensity — FLOPs divided by bytes moved — tells you in one ratio whether a kernel is starved for compute or starved for bandwidth; decode attention sits at roughly 1 FLOP/byte, catastrophically left of the H100 ridge point at ~93."
blurb:
  - "Sam Williams drew two lines on a Berkeley whiteboard in 2008. Reviewers rejected the paper three times for being too simple."
  - "SAXPY: 0.17 FLOP/byte. Decode attention: ~1 FLOP/byte. Dense 4K×4K matmul: ~680 FLOP/byte."
  - "The H100 ridge point is ~93 FLOP/byte — decode is two orders of magnitude below it, every step, by construction."
  - "Every inference optimization — FlashAttention, speculative decoding, batching — is an attempt to move the dot rightward on this plot."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T10:30:00-04:00
issue: 8
weight: 40
techKind: primer
techNode: roofline
header: 04-roofline.webp
---

## Berkeley, 2008

It is a Tuesday morning in Soda Hall and **Sam Williams** is trying to explain a Cray X1E to a room of graduate students who keep arguing about cache misses.

Williams is a postdoc at the **Lawrence Berkeley National Lab** working with **Andrew Waterman** and the kind of advisor — **David Patterson** — whose name is on the textbook the graduate students learned the word *cache* from. Their problem, the one that has eaten the previous two years of their lives, is mundane and infuriating: scientific kernels run on multicore CPUs at a *fraction* of peak FLOP/s, and nobody can predict, from the code alone, *which fraction*. A 7-point stencil on a 3D grid gets 4% of peak on a Cray; a dense matrix multiply gets 92%. Same compiler, same machine, two orders of magnitude apart. Why?

The room argues for an hour. Someone blames cache associativity. Someone blames the prefetcher. Someone blames the C compiler. Williams listens, walks to the whiteboard, and draws **two lines**.

One horizontal — peak compute, in FLOP/s. One diagonal, rising at 45° on a log-log plot — peak memory bandwidth multiplied by a quantity he calls **arithmetic intensity**. He plots stencil at one spot on the x-axis and matmul at another. The model says, without any simulation or measurement, that stencil sits *under the diagonal* and matmul sits *under the ceiling*. Both kernels are correctly performing as fast as they possibly can. Neither is "slow." They are bottlenecked by **different physical resources**.

That whiteboard sketch is the **roofline model** {{< cite text="Williams, Waterman, Patterson (2008)" url="https://dl.acm.org/doi/10.1145/1498765.1498785" kind="paper" >}}. Eighteen years later, every senior engineer reasoning about an LLM inference kernel still draws the same two lines on the same whiteboard, with the dots in different places. Decode attention, it turns out, sits in exactly the spot Williams's stencil sat — and for exactly the same reason.

{{% marginnote %}}The original paper was rejected three times before publication. Reviewers thought the model was *too simple* to be useful. That, of course, was the point.{{% /marginnote %}}

## The One Number That Decides Everything

Strip the model down to its skeleton and what is left is a single ratio.

For any kernel, define its **arithmetic intensity** as

$$
I \;=\; \frac{\text{FLOPs performed}}{\text{bytes moved from main memory}}.
$$

That is it. Useful arithmetic over the cost of feeding it. A kernel with high $I$ does a lot of math per byte it loads; a kernel with low $I$ spends most of its time waiting for bytes. The number has units of FLOP/byte, and it can vary by **six orders of magnitude** across the kinds of code an LLM serving box runs in a single second.

A few worked examples to calibrate the ruler:

- **SAXPY** ($y \leftarrow \alpha x + y$). For every $y_i$: load $x_i$ (4 B), load $y_i$ (4 B), store $y_i$ (4 B) — 12 bytes for 2 FLOPs. $I \approx 0.17$. Famously, hopelessly, memory-bound.
- **Dense matmul** $C = AB$, with all three matrices $N \times N$. Total work $2N^3$ FLOPs. Total bytes if you stream everything once: $3N^2 \cdot 4 = 12 N^2$. Intensity $I \approx N/6$. *Intensity grows with the matrix.* At $N = 4096$ you get $I \approx 680$ — comfortably compute-bound.
- **Naive softmax attention**, one query, $T$ keys. Load $T$ keys and $T$ values (2 × $T \cdot D \cdot 2$ B in FP16). FLOPs: $\sim 4TD$. Intensity $I \approx 1$. As bad as it gets.

Three kernels, three regimes. The first is starved. The second is fed. The third is, in 2025 LLM-serving terms, the entire reason your decode step is slow.

```pyplot {id="intensity-examples" caption="Arithmetic intensity of five reference kernels. SAXPY and decode attention live in the same memory-bound basement; large matmul lives in the compute-bound penthouse. Two orders of magnitude separate them."}
import numpy as np
import matplotlib.pyplot as plt

# Worked-example intensities (FLOPs per byte moved from HBM).
kernels = [
    ("SAXPY",                       0.17, '#FF8C00'),
    ("decode attention (T=4k)",     1.0,  '#FF007F'),
    ("decode MLP (one token)",      1.0,  '#FF007F'),
    ("4K x 4K matmul",              680,  '#00A8A8'),
    ("prefill (B=64, T=4k, D=4k)",  2000, '#00A8A8'),
]

fig, ax = plt.subplots(figsize=(9, 3.4))
y = np.arange(len(kernels))
for i, (name, I, col) in enumerate(kernels):
    ax.barh(i, np.log10(I), color=col, edgecolor='#1A1A1A', linewidth=1.2)
    ax.text(np.log10(I) + 0.05, i, f"  I = {I}", va='center', fontsize=10,
            fontfamily='monospace')

ax.set_yticks(y)
ax.set_yticklabels([k[0] for k in kernels])
ax.set_xlabel("log10( FLOPs / byte )")
ax.set_xlim(-1, 4.2)
ax.axvline(np.log10(93), color='#1A1A1A', linewidth=1.5, linestyle='--')
ax.text(np.log10(93) + 0.05, -0.7, "H100 ridge\n(I = 93)",
        color='#1A1A1A', fontsize=9, fontweight='bold')
ax.set_title("Arithmetic intensity spans four orders of magnitude")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Pause on that picture. The decode kernels sit *two orders of magnitude* to the left of large matmul. That gap is not a quirk of the hardware. It is structural: a 4K×4K matmul reuses every loaded byte $N$ times because of the cubic-vs-quadratic asymmetry; decode loads a 70 GB weight tensor and multiplies it by a *single token*. Each byte is used once and discarded.

## Building The Diagram

To turn intensity into a *prediction*, we need two more numbers — the hardware's peak compute and peak bandwidth.

For an **H100 SXM5**:

- Peak compute on tensor cores, FP16/BF16 with FP32 accumulate: $312$ TFLOP/s. (Without sparsity. The "989 TFLOP/s" number in marketing slides includes 2:4 structured sparsity, which inference workloads almost never hit.)
- Peak HBM3 bandwidth: $3.35$ TB/s.

A kernel with intensity $I$ FLOP/byte that is bandwidth-bound runs at $B \cdot I$ FLOP/s, where $B$ is bandwidth. As $I$ grows, that diagonal line eventually hits the horizontal compute ceiling. The crossover — the **ridge point** — is at

$$
I_\text{ridge} \;=\; \frac{C}{B} \;=\; \frac{312 \times 10^{12}}{3.35 \times 10^{12}} \;\approx\; 93 \;\;\text{FLOP/byte}.
$$

{{% marginnote %}}H200 has the same compute but higher bandwidth (4.8 TB/s), so its ridge sits at $I \approx 65$. B200 FP8 with 8 TB/s HBM3e and ~2 PFLOP/s lands at $I \approx 250$. Each generation pushes the ridge *up* (more bytes per FLOP available) — but the bandwidth crisis at decode time gets worse, not better, because absolute bandwidth isn't growing as fast as compute.{{% /marginnote %}}

That single number — **93** — is the dividing line of the entire issue. To its right, you are compute-bound, and the only way to go faster is to do less math. To its left, you are bandwidth-bound, and the only way to go faster is to move fewer bytes (or get more work out of each byte you move).

Now draw the diagram and drop our kernels onto it.

```pyplot {id="h100-roofline" caption="The H100 roofline. Peak compute is the horizontal ceiling at 312 TFLOP/s; peak bandwidth is the rising diagonal. The ridge point at I=93 is the boundary. Three labelled dots: a 4K x 4K matmul (compute-bound), prefill (compute-bound), decode (catastrophically memory-bound)."}
import numpy as np
import matplotlib.pyplot as plt

# H100 SXM5 peaks
C = 312e12      # FLOP/s (tensor-core, FP16, no sparsity)
B = 3.35e12     # B/s   (HBM3)
ridge = C / B

I = np.logspace(-1, 4, 400)
roof = np.minimum(B * I, np.full_like(I, C))

fig, ax = plt.subplots(figsize=(9.5, 5.3))
ax.loglog(I, roof / 1e12, color='#1A1A1A', linewidth=2.5)
ax.fill_between(I, roof / 1e12, 1e-3, color='#FFD700', alpha=0.18)

# The ridge point
ax.axvline(ridge, color='#1A1A1A', linewidth=0.6, linestyle=':')
ax.text(ridge * 1.1, 4, f"ridge\nI = {ridge:.0f}", fontsize=9,
        fontweight='bold', color='#1A1A1A')

# Three reference kernels: (name, I, achieved FLOP/s, colour)
kernels = [
    ("4K x 4K matmul",                 680,  290e12, '#00A8A8'),
    ("prefill (B=64, T=4k, D=4k)",     2000, 295e12, '#00A8A8'),
    ("decode attention (T=4k)",        1.0,  B*1.0,  '#FF007F'),
    ("decode MLP (single token)",      1.0,  B*1.0,  '#FF007F'),
    ("spec-decode verify (K=4)",       4.0,  B*4.0,  '#FF8C00'),
]
for name, I_k, FLOPs, col in kernels:
    ax.scatter([I_k], [FLOPs / 1e12], s=130, color=col,
               edgecolor='#1A1A1A', linewidth=1.4, zorder=4)
    dy = 1.4 if 'spec' in name or 'decode' in name else 0.55
    ax.annotate(name, (I_k, FLOPs / 1e12), xytext=(8, -4),
                textcoords='offset points', fontsize=9,
                color='#1A1A1A', fontweight='bold')

ax.set_xlim(0.1, 1e4)
ax.set_ylim(1e-3, 1e3)
ax.set_xlabel("arithmetic intensity I  (FLOP / byte)")
ax.set_ylabel("achieved throughput (TFLOP/s)")
ax.set_title("H100 roofline: where the LLM kernels actually live")
ax.grid(True, which='both', alpha=0.15)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Read the picture. Three things should jump out, and each one is a small earthquake.

**First**, the two prefill-class kernels sit pinned against the horizontal ceiling, right of the ridge. They are running as fast as the silicon can physically multiply. Adding bandwidth to an H100 would not help them; they already have all the bytes they can chew.

**Second**, decode attention and decode MLP sit at $I = 1$ — at a throughput of $3.35$ TFLOP/s, which is **2.1% of peak compute**. They are not running slowly because of bad kernels. They are running slowly because they have a 93-fold mismatch with the ridge, and physics will not let them go faster on this hardware as written.

**Third**, that little orange dot labelled *spec-decode verify (K=4)* sits at $I = 4$ — already a four-fold throughput improvement just by raising the FLOPs-per-byte ratio. Foreshadow that. We will come back to it in [The Draft Trick](../15-speculative-decoding/).

{{% callout type="note" %}}**Why decode intensity is exactly 1.** A decode step on a 70B-parameter model in BF16 loads $\sim 140$ GB of weights from HBM, multiplied against a *single* hidden vector. The FLOPs are roughly $2 \cdot 70 \cdot 10^9$ per token (two FLOPs per parameter — multiply-accumulate). Bytes: $140 \cdot 10^9$. The ratio is one. Decode-time arithmetic intensity is, almost by definition of "decode," equal to one. The only way out is to **change what "one token" means** — either batch many users together, or verify many candidate tokens in parallel.{{% /callout %}}

## The Three Kernels, Walked

Let's pin three concrete kernels to the diagram with the napkin math behind each.

### Kernel 1 — 4K × 4K matmul

Three matrices, each $4096 \times 4096$ in FP16. Bytes: $3 \cdot 4096^2 \cdot 2 = 96$ MB. FLOPs: $2 \cdot 4096^3 = 1.4 \cdot 10^{11}$. Intensity $I = 1.4 \cdot 10^{11} / 9.6 \cdot 10^7 = 1430$ FLOP/byte. Comfortably right of ridge. The H100 happily delivers ~290 TFLOP/s on this kernel (93% of peak). The bandwidth is barely visible in the timeline; the kernel is a pure tensor-core feast.

### Kernel 2 — Prefill

For a Llama-style 8B model with hidden dim $D = 4096$, batch size $B = 64$, prompt length $T = 4096$ — the typical workload after a server warms up. Per layer, each MLP weight matrix is $\sim 16 D^2 \cdot 2 = 0.5$ GB loaded once. Total token-FLOPs through that weight: $B \cdot T \cdot 16 D^2 \cdot 2 \approx 1.4$ TFLOP per layer, per weight. Intensity per weight matrix: $B \cdot T \approx 250{,}000$ FLOP/byte. **Way** past ridge. {{< wiki "attention" >}}Attention{{< /wiki >}} during prefill is similar (the $T \times T$ score matrix gets reused across $D$ heads). Prefill is the kernel the H100 was designed for; it's why your $700$/hour cloud GPU bill is even *thinkable*.

### Kernel 3 — Decode

Same model, but now we are generating one token at a time, for one user. Each layer must reload its weights from HBM — they don't fit in SRAM. The FFN weights of an 8B model occupy $\sim 6$ GB; we multiply them against a vector of length $D = 4096$. FLOPs $\approx 6 \cdot 10^9 \cdot 2 = 1.2 \cdot 10^{10}$. Bytes loaded $\approx 1.2 \cdot 10^{10}$ (each weight read once at FP16/INT8). Intensity = 1. The GPU spends 98% of its silicon idle.

That last sentence is the **defining pathology of LLM inference**. Every other phenomenon in this issue — continuous batching, paged attention, speculative decoding, disaggregated prefill, the bizarre 600B model on one node — is downstream of "decode lives at $I = 1$."

## The Spec-Decode Lever

There is a fourth dot on the roofline that the picture spent only one line on, and it deserves its own moment.

If you take the decode kernel and instead of asking for *one* output token per HBM weight load you ask for $K$ output tokens (perhaps from $K$ different speculative candidates, or from $K$ batched users, or from $K$ chunked-prefill positions), the FLOPs scale linearly in $K$ while the bytes don't. Intensity goes from $1$ to $K$.

Watch what that does to decode throughput as a function of batch size. We can plot the predicted FLOP/s on a single 70B-class decode step as the batch grows from one user to many.

```pyplot {id="decode-intensity-vs-batch" caption="Predicted decode throughput vs effective batch size on an H100. At B=1 you are pinned to bandwidth at 1 FLOP/byte. At B=93 you finally touch the ridge. Below the ridge, every extra batched/speculated token is essentially free."}
import numpy as np
import matplotlib.pyplot as plt

C = 312e12          # FLOP/s peak
B_hbm = 3.35e12     # B/s peak
ridge = C / B_hbm   # = 93

# Effective intensity == effective batch size (for decode, one FLOP/byte per token)
batch = np.arange(1, 257)
I = batch.astype(float)
throughput = np.minimum(B_hbm * I, C) / 1e12   # TFLOP/s

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(batch, throughput, color='#FF007F', linewidth=2.8)
ax.axhline(C/1e12, color='#1A1A1A', linewidth=1.2, linestyle='--')
ax.text(180, C/1e12 - 18, "peak compute 312 TFLOP/s",
        fontsize=9, color='#1A1A1A')

# Annotations
for b, lab in [(1, 'B = 1\nlonely user'),
               (8, 'B = 8'),
               (32, 'B = 32'),
               (93, 'B = 93\nridge'),
               (192, 'B = 192\ncompute-bound')]:
    t = min(B_hbm * b, C) / 1e12
    ax.scatter([b], [t], s=80, color='#FFD700',
               edgecolor='#1A1A1A', linewidth=1.2, zorder=3)
    ax.annotate(lab, (b, t), xytext=(6, -14),
                textcoords='offset points', fontsize=9, fontweight='bold')

ax.fill_between(batch, 0, throughput, color='#FFD700', alpha=0.18)
ax.set_xlabel("effective batch size  (= concurrent decodes × spec-K)")
ax.set_ylabel("predicted decode throughput (TFLOP/s)")
ax.set_title("The decode escalator: every batched token is free until B = 93")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The plot says something almost unreasonable: from $B = 1$ to $B = 93$, the cost of decoding $B$ tokens is **the same** as the cost of decoding one. The bytes are already being moved; the math is sitting idle. *Add tokens*. Stack them. Speculate them. Batch them across users. Until you hit the ridge, every additional token is on the house.

This is the single most important economic fact about LLM serving in 2026. The reason providers can charge per-token prices that round to fractions of a cent is not because the math is cheap — it is because the math was already paid for, and the *first* customer was bankrolling it.

## The Forced Moves

When you are pinned below the ridge, the laws of physics permit exactly three escape routes. Every optimization in the rest of this issue is one of them.

{{< crosshead >}}Move 1 — reduce bytes{{< /crosshead >}}

If you can't change $I$'s denominator's units, change its size. Fewer bytes loaded means more $I$ for the same FLOPs.

- **Quantization** — store weights and activations in INT8, INT4, FP8, FP4. A 70B model in INT4 is 35 GB instead of 140 GB. That is a 4× intensity multiplier for free. (Treated in depth in Issue 03.)
- **{{< wiki "kv-pruning" >}}KV pruning{{< /wiki >}}** and **MLA-style {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} compression** — store fewer KV slots, or compress them. Cuts the attention byte budget directly. (Issue 06; Issue 05 ch.19.)
- **Pruning the model itself** — sparsity, expert pruning, layer skipping. Same lever.

{{< crosshead >}}Move 2 — raise FLOPs per byte{{< /crosshead >}}

Keep the bytes, do more work with them.

- **Batching.** $N$ users decoding in lockstep means the same weight load services $N$ tokens. Intensity $\to N$. The simple version of this is **static batching**; the modern version is **continuous batching** (Orca, 2022 → [ch.8](../08-continuous-batching/)).
- **Speculative decoding.** A small *draft* model produces $K$ candidate tokens; the *target* model verifies all $K$ in a single forward pass. Intensity goes from $1$ to $K$ on the verify step — the same byte budget covers more output. ([→ ch.15](../15-speculative-decoding/).)
- **Multi-token prediction.** Architectural variants (Medusa, EAGLE, MTP) that emit several tokens per forward pass at the cost of extra parameters. Same lever.

{{< crosshead >}}Move 3 — move the byte once, use it many times{{< /crosshead >}}

If you have to load a byte from HBM, at least use it for everything you possibly can before it falls out of SRAM.

- **{{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}}.** Don't write the $T \times T$ score matrix to HBM. Tile $K$ and $V$ into SRAM, run online softmax, accumulate $O$ in place. The bytes from HBM are loaded *once* and reused across the softmax, the matmul, and the output. ([→ ch.6](../06-flash-attention/).)
- **Fused MLP, fused rotary, fused softmax.** Same idea, smaller scale. Don't trip out to HBM between operations that share data.
- **Prefix caching.** Don't re-load the system prompt's KVs at all if you computed them last request. ([→ ch.12](../12-prefix-caching/).)

Three moves. That is the entire menu. Every chapter that follows from here is a different combination of them, with different scaffolding around them — schedulers, allocators, parallelism patterns, network fabrics. The roofline is the floor plan.

{{% pullquote type="counter-intuitive" %}}Pinning a kernel against the bandwidth diagonal is not a bug. It is the *natural state* of decode. The job of the inference stack is not to make decode compute-bound — it is to extract every drop of useful work out of bytes that are going to be loaded anyway.{{% /pullquote %}}

## Why The Map Survives

Williams's two lines were drawn before tensor cores, before Hopper, before LLMs, before the word "attention" meant anything in a kernel-engineering meeting. And yet the diagram has not aged a day, because the underlying constraint — *finite bandwidth times finite intensity is your throughput ceiling* — is not a quirk of any particular processor. It is a statement about the geometry of computation.

Every silicon generation moves the ridge a little. Compute scales faster than bandwidth, so the ridge drifts to the right; what was "compute-bound" on an A100 may be "memory-bound" on a B200. But the *shape* of the diagram is invariant. The strategy for living under it is invariant. The questions you ask about a new kernel — *what is $I$? where does it land? which of the three moves do I have?* — are exactly the questions Sam Williams asked his graduate students in 2008.

Hold this picture in your head as you read the rest of the issue. Every time vLLM does something that looks clever, draw the diagram and watch the kernel's dot slide right.

## What To Remember

1. **Arithmetic intensity is destiny.** One number — FLOPs per HBM byte — predicts whether your kernel is compute-bound or bandwidth-bound.
2. **Prefill is right of the ridge; decode is far, far left.** Their performance profiles are not slightly different; they are categorically different.
3. **There are three levers to move toward the ridge** — reduce bytes, raise FLOPs-per-byte, or amortize a loaded byte over many operations. The rest of this issue is variations on those three themes.

**Continue to → [Launches Aren't Free](../05-cuda-graphs/)** — even when the GPU is fetching bytes, the CPU enqueueing the kernels can still wreck your latency budget.
