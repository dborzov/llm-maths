---
title: "GPU Registers: why head_dim 256 broke the fix"
description: "Tile-size optimization in FA3 FP8: why head_dim 64 and 128 absorbed the new accumulator register, why 256 spilled to local memory, and what three PRs over six weeks did about it."
blurb:
  - "Every warp in a fused attention kernel owns a fixed slice of the register file."
  - "Add one accumulator register: head_dim 64 and 128 absorb it. Head_dim 256 spills."
  - "Spilling to local memory costs ~60% prefill throughput at the worst tile size."
  - "Three PRs over six weeks traded tile sizes until the regression closed."
topics: [kernel-engineering, primer, performance]
tags: [flash-attention-3, tile-size, register-pressure, occupancy, head-dim, fp8]
theme: cream
math: true
draft: false
date: 2026-05-18T09:00:00-04:00
issue: 9
weight: 70
techKind: primer
techNode: tile-sizes
header: default.webp
---

## Why This Primer Exists

The [two-level accumulation fix](../06-two-level-fix/) recovers FP8 KV-cache accuracy but introduces a prefill regression at `head_dim = 256`. The team's response is **tile-size optimization** — picking different `(Bq, Bk)` block sizes per `head_dim` so that the extra register from two-level accumulation doesn't push the kernel over the register-file ceiling.

This primer explains what "tile size" is in a fused attention kernel, why the choice has a sharp accuracy-performance interaction with the register budget, what specifically happens at `head_dim = 256` that makes it the hard case, and what the optimization PRs ([`flash-attention#91`](https://github.com/vllm-project/flash-attention/pull/91), [`flash-attention#96`](https://github.com/vllm-project/flash-attention/pull/96), [`flash-attention#125`](https://github.com/vllm-project/flash-attention/pull/125)) actually did.

If you have read [Issue 8's FlashAttention primer](/issues/08-anatomy-of-a-token/06-flash-attention/), you already know that fused attention has two nested loops — an outer Q-tile loop and an inner KV-tile loop — and that the tile sizes determine how much SRAM and how many registers each warp owns. This primer goes deeper into the register-budget side of the trade-off, which is where the FP8 fix lives.

## What "Tile Size" Actually Decides

A {{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}}-3 kernel processes the attention matmul in tiles. Two block sizes parameterize the kernel: `Bq` (the number of query rows per tile) and `Bk` (the number of key/value rows per tile). For each pair of `(Bq, Bk)` tiles, the kernel:

1. Loads `Q_tile` (shape `[Bq, D]`) into SRAM.
2. For each KV tile, loads `K_tile` and `V_tile` (shape `[Bk, D]`) into SRAM.
3. Computes the partial score matrix `S_tile = Q_tile @ K_tile.T` (shape `[Bq, Bk]`) directly in registers.
4. Runs the online softmax bookkeeping and accumulates into an output tile `O_tile` (shape `[Bq, D]`).

The choice of `(Bq, Bk)` determines four things at once:

- **SRAM footprint.** `Q_tile`, `K_tile`, `V_tile`, `S_tile`, and `O_tile` all have to fit in SRAM. The total SRAM cost is roughly $(B_q + 2 B_k) \cdot D + B_q \cdot B_k$ elements per warp group.
- **Register footprint.** The output accumulator `O_tile` lives in registers throughout the inner loop. Its size is $B_q \cdot D$ elements. The score matrix `S_tile` also occupies registers during the active inner iteration.
- **Arithmetic intensity.** Larger tiles mean more reuse of each loaded tile and higher FLOP/byte ratio. Smaller tiles mean less reuse but better parallelism across warp groups.
- **Number of concurrent warp groups.** A smaller per-warp footprint lets more warp groups run concurrently on the same SM, improving latency hiding.

All four are in tension. There is no globally-best tile size; the right choice depends on `head_dim`, the hardware, and what fits in the register file.

## The Register File Ceiling

A Hopper SM has a register file of roughly **256 KB per SM**, split across all the warp groups running on that SM. Each warp can use up to 256 32-bit registers (a hard CUDA limit). A warp group is four warps, so a warp group has up to ~1024 registers across its threads. If a kernel asks for more registers than the warp group can hold, the compiler **spills** the excess to *local memory* — DRAM-backed per-thread storage that has the latency of HBM, not registers.

A spill is a disaster for a tensor-core kernel. The whole point of using tensor cores is to keep the operands in registers so the multiplier can issue continuously. If the operands have to be fetched from local memory on every cycle, the tensor core sits idle waiting for DRAM, and the kernel's effective throughput drops by an order of magnitude.

The output tile `O_tile` of size $B_q \cdot D$ is the dominant register cost in the FA3 inner loop. For `head_dim = 128` and `Bq = 64`, that's $64 \times 128 = 8192$ elements per warp group, or 8192 registers if stored in FP16, 16384 in FP32. For `head_dim = 256`, the same `Bq = 64` doubles this to 16384 in FP16, 32768 in FP32.

The two-level accumulation fix adds *one more* output-shaped accumulator — `acc_O_true` — also of size $B_q \cdot D$ in true FP32. For `head_dim = 64`, this is small enough to fit. For `head_dim = 128`, fits with some slack. For `head_dim = 256`, this is the straw that breaks the register file: the warp group runs out of room, the compiler spills, and the inner loop is no longer running at tensor-core speed.

```pyplot {id="register-pressure-by-headdim" caption="Register cost of FA3 FP8 inner loop, before and after two-level accumulation, by head_dim. The 256-register-per-warp ceiling is the hard line. Above it, the compiler spills to local memory."}
import numpy as np
import matplotlib.pyplot as plt

head_dims = [64, 128, 256]
Bq = 64

# Approximate per-warp register cost (per element in O_tile, half-float = 1 register; FP32 = 2 registers)
def cost(D, two_level):
    base = (Bq * D) // 4   # working registers, rough
    accum = (Bq * D) // 4
    second_accum = (Bq * D) // 2 if two_level else 0   # FP32 second accum
    overhead = 32           # softmax stats, addresses, etc.
    return base + accum + second_accum + overhead

cost_before = [cost(D, two_level=False) for D in head_dims]
cost_after  = [cost(D, two_level=True)  for D in head_dims]

x = np.arange(len(head_dims))
w = 0.36

fig, ax = plt.subplots(figsize=(8.5, 4.6))
bars_b = ax.bar(x - w/2, cost_before, w, color='#00A8A8',
                edgecolor='#1A1A1A', linewidth=1.4, label='before two-level fix')
bars_a = ax.bar(x + w/2, cost_after,  w, color='#FF007F',
                edgecolor='#1A1A1A', linewidth=1.4, label='after two-level fix')

ax.axhline(256, color='#1A1A1A', linewidth=2.0, linestyle='--')
ax.text(2.45, 264, 'per-warp ceiling (256 regs)',
        ha='right', fontsize=9, style='italic')

for bars in [bars_b, bars_a]:
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 8,
                f'{b.get_height():.0f}', ha='center', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([f'head_dim={d}' for d in head_dims])
ax.set_ylabel('approximate per-warp registers used')
ax.set_title('Where two-level accumulation pushes the register budget over the ceiling',
             fontsize=11, fontweight='bold')
ax.set_ylim(0, 360)
ax.legend(framealpha=1, edgecolor='#1A1A1A', loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

You can see what makes `head_dim = 256` painful. The pre-fix kernel was already pushing the limit; the second accumulator tips it over. The compiler can either (a) reduce `Bq` to fit, which hurts the kernel's arithmetic intensity, or (b) accept the spill, which hurts everything. Pre-tuning, the FA3 kernel was choosing (b) on `head_dim = 256` and paying ~1.6× more cycles per prefill step.

## What The Tile-Size PRs Did

The optimization story has three PRs, each addressing a different facet.

### `flash-attention#91`: Decode-Specific Tile Selection

The first PR targets the *decode* path, which is bandwidth-bound and has a different optimal tile shape than prefill. Decode has `Bq = 1` (only one query token per step), so the output tile `O_tile` is trivially small and register pressure is not the binding constraint — *bandwidth* is.

For decode, the optimal `Bk` is determined by the SRAM footprint of `K_tile` and `V_tile` and the overhead of loading them. Larger `Bk` means fewer trips to HBM per Q-tile but more SRAM per tile. The pre-fix FA3 kernel used a one-size-fits-all `Bk = 64` for both prefill and decode. PR #91 introduces decode-specific tile configurations that increase `Bk` to 128 or 256 depending on `head_dim`, fitting the larger KV tile in SRAM (since `Bq` is just 1) and reducing the HBM traffic of the inner loop's KV-tile loads.

The result: the slope of `ITL = slope × T + intercept` from [Chapter 2](../02-bandwidth-bargain/) drops further on decode-heavy workloads. The Llama-3.1-8B slope ratio improvement from 63% (pre-fix) to 54% (post-fix) of BF16 is in large part due to this PR.

### `flash-attention#96`: Asynchronous Prefetch

The second PR addresses a related decode issue: even with larger `Bk`, the inner loop pipeline can stall on KV-tile loads. PR #96 adds asynchronous prefetching of the next KV tile while the current tile is being consumed, using Hopper's TMA (Tensor Memory Accelerator) to issue the load in the background. This overlaps data movement with compute and further reduces the per-token attention cost.

This is the same pattern FA-3 itself introduced (see [Issue 8's FlashAttention primer](/issues/08-anatomy-of-a-token/06-flash-attention/)) — warp specialization plus async TMA — applied specifically to the decode tile pattern. The decode slope improvement compounds with PR #91's contribution.

### `flash-attention#125`: Reduce Spills On `head_dim = 256`

The third PR is the one that directly addresses the post-two-level prefill regression on `head_dim = 256`. The idea is to *reduce* `Bq` from 64 to 32 for `head_dim = 256` prefill, which halves the size of both `O_tile` and `acc_O_true` and brings the register budget back under the ceiling. The cost is that fewer query rows are processed per outer iteration, reducing the kernel's parallelism slightly. The benefit is that the inner loop runs at full tensor-core speed instead of spilling.

Empirically, the smaller `Bq` recovers most of the prefill regression on Gemma-4-E2B but does not fully close the gap to pre-fix performance. The team's headline number for `head_dim = 256` after PR #125 is a TTFT quadratic coefficient ratio of ~1.3 to BF16 (down from 1.6 immediately after the two-level fix). Not zero, but tolerable for most workloads.

The PR also surfaces the broader engineering question of whether prefill should use a *separate kernel variant* for `head_dim = 256` that avoids the two-level accumulation entirely. This is, as of the issue's publication, an open conversation in the FA3 PR thread — the trade-off is whether the prefill speedup justifies the accuracy risk on long prefills where the contraction dimension is still in the regime where the accumulator drifts.

## A Simpler Model: Roofline With Spills

To make the tile-size cost concrete, here is a Roofline-style picture of the FA3 FP8 kernel's effective throughput as a function of register pressure. [Issue 8's roofline primer](/issues/08-anatomy-of-a-token/04-roofline/) introduces the model; this is its application.

```pyplot {id="roofline-with-spills" caption="Roofline-style view of effective tensor-core throughput as register pressure increases. Below the ceiling, the kernel runs at peak. Above the ceiling, the compiler spills to local memory and throughput drops by an order of magnitude."}
import numpy as np
import matplotlib.pyplot as plt

regs = np.linspace(50, 350, 400)
ceiling = 256

# Effective throughput: peak below ceiling, then exponential drop
throughput = np.where(
    regs <= ceiling,
    1.0,                                        # full speed
    np.exp(-(regs - ceiling) / 30) * 0.85 + 0.1  # spills, ~10x slower asymptotically
)

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(regs, throughput, color='#00A8A8', linewidth=2.6)
ax.fill_between(regs, throughput, 0,
                where=regs <= ceiling, color='#00A8A8', alpha=0.15)
ax.fill_between(regs, throughput, 0,
                where=regs > ceiling, color='#FF007F', alpha=0.15)
ax.axvline(ceiling, color='#1A1A1A', linewidth=2, linestyle='--')

# Mark our three head_dim points
points = {
    'head_dim=64 (after fix)':  (130, 1.0),
    'head_dim=128 (after fix)': (180, 1.0),
    'head_dim=256 (after fix, before PR#125)': (310, 0.18),
    'head_dim=256 (after PR#125, Bq=32)':       (200, 1.0),
}
for label, (x, y) in points.items():
    ax.plot(x, y, marker='o', markersize=10, color='#FFD700',
            markeredgecolor='#1A1A1A', markeredgewidth=1.5)
    ax.annotate(label, xy=(x, y), xytext=(x+10, y - 0.15 if 'spilled' in label.lower() else y + 0.08),
                fontsize=8.5)

ax.set_xlabel('per-warp register usage')
ax.set_ylabel('effective tensor-core throughput (peak = 1.0)')
ax.set_title('Crossing the register ceiling = order-of-magnitude throughput drop',
             fontsize=11, fontweight='bold')
ax.set_xlim(50, 350)
ax.set_ylim(0, 1.15)
ax.text(ceiling - 5, 0.05, 'register file ceiling',
        ha='right', fontsize=9, style='italic')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The picture is a step function with a cliff at the ceiling. Below the cliff, the kernel is at peak. Above the cliff, throughput collapses. PR #125's job was to slide the `head_dim = 256` configuration back below the cliff by reducing `Bq`. The trade-off of smaller `Bq` is that the kernel processes fewer query rows per outer iteration — a tax on parallelism — but the tax is small compared to the cost of a spill.

## Query Quantization Fusion: A Different Angle

A related optimization the team shipped at the same time is **query quantization fusion**, in [`vllm#24914`](https://github.com/vllm-project/vllm/pull/24914). The pre-fix design quantized Q (the query tensor) from BF16 to FP8 *inside* the attention backend, as a fixed per-token preamble. This cost a small amount of latency on every decode step, regardless of context length — appearing as part of the *intercept* in the `ITL = slope × T + intercept` model.

The fusion PR moves the Q quantization out of the attention backend into a plain `torch.compile`-able PyTorch operation. Once it lives in plain PyTorch, `torch.compile` can fuse it with the surrounding operations — the Q projection from the residual stream, the position embedding application, the reshape into multi-head layout — eliminating the dedicated kernel launch and the redundant memory round-trip.

The result: the *intercept* term of the linear ITL model drops by a constant. For Llama-3.1-8B, the intercept went from ~6.7 ms to ~6.58 ms. Small but free.

The broader pattern is that **vLLM's compilation infrastructure is incrementally absorbing operations that used to be hand-tuned kernel boundaries**. Each operation pulled into the compiled graph allows fusion with its neighbors, eliminating launch overhead and intermediate memory traffic. The Q quantization fusion is one example; vLLM's V1 engine ([Issue 8 ch.14](/issues/08-anatomy-of-a-token/14-scheduler/)) is the broader story.

## Per-Head Scales: An Architectural Refinement

A third related change deserves a mention because it shows up in the FA3 path alongside the tile-size work: **per-head FP8 scales**, in [`vllm#30833`](https://github.com/vllm-project/vllm/pull/30833) and [`vllm#30141`](https://github.com/vllm-project/vllm/pull/30141).

The pre-existing FA3 FP8 API supported a single per-tensor scale for the entire attention operation. The newer API supports an *array of scales*, with one scale per KV head. The motivation: when activation distributions vary substantially across heads (which they sometimes do, particularly in models with grouped-query attention), a single per-tensor scale forces all heads to compromise on a shared scale, whereas per-head scales let each head pick the scale that fits its actual distribution.

The wiring work involved:

- Generalizing the static quantization support in vLLM to handle arbitrary scale tensor shapes (#30833).
- Expanding the `reshape_and_cache_flash` kernel — the operation that writes new K and V vectors into the paged KV cache — to apply per-head scales during the write (#30141).

The accuracy benefit is small for most models (most heads have similar activation distributions) but non-trivial for some. The configuration is opt-in: by default, vLLM still uses per-tensor scales, but users with finer accuracy requirements can enable per-head scales via the configuration or via `LLM-Compressor`.

This is the **calibration story** in miniature, applied to the granularity of the scale tensor itself: more granularity costs more bookkeeping but recovers more accuracy. The trade-off discussion in [Chapter 10 — Scale Equals One](../10-calibrate/) generalizes this.

## What To Remember

1. **Tile size is a multi-way trade-off** among SRAM footprint, register footprint, arithmetic intensity, and warp-group occupancy. The right tile depends on `head_dim`, hardware, and what fits.
2. **The register file has a hard ceiling** (~256 registers/warp on Hopper). Going over it spills to local memory, which collapses tensor-core throughput by an order of magnitude.
3. **Two-level accumulation adds one output-shaped accumulator**. Free at `head_dim = 64` and `head_dim = 128`, painful at `head_dim = 256`.
4. **PR #91 / #96 / #125** are the three optimization PRs that recovered (most of) the prefill regression and improved the decode slope: decode-specific tile selection, async TMA prefetch, smaller `Bq` for `head_dim = 256`.
5. **Query quantization fusion** and **per-head scales** are the two other engineering wins shipped alongside the tile-size work — both small individually, both contributing to the overall production-ready FP8 KV path.

**Continue to** → [Slope vs Intercept](../08-itl-slope-model/) — the analytical model used to characterize every benchmark in this issue, and the per-model break-even arithmetic.
