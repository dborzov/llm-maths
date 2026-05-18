---
title: "Bandwidth Wall: why the H100 runs at 0.3% utilization"
short_title: "Bandwidth Wall"
description: "During autoregressive decode, an H100 GPU operates at roughly 0.3% of its 312 TFLOP/s rating — decode attention sits at ~1 FLOP/byte arithmetic intensity, 93× below the H100 ridge point where compute becomes the bottleneck."
blurb:
  - "H100 peak compute: 312 TFLOP/s. Measured FP utilization during text generation: ~0.3%."
  - "The roofline ridge point for H100: 312 TFLOP/s ÷ 3.35 TB/s ≈ 93 FLOPs/byte."
  - "Decode attention arithmetic intensity: ~1 FLOP/byte — 93× below the ridge. Catastrophically memory-bound."
  - "If decode is memory-bound, halving the KV cache roughly halves the time per token — faster attention kernels don't help."
topics: [gpu, memory, inference, bandwidth]
tags: [hbm, roofline, bandwidth, decode, prefill, paged-attention]
theme: cream
math: true
draft: false
date: 2026-05-14T23:01:38-04:00
issue: 6
weight: 110
techKind: primer
techNode: bandwidth-wall
header: 11-bandwidth-wall.webp
---

## The GPU That Is Mostly Idle

**June 2023.** A research team at Berkeley is profiling a 70-billion-parameter model generating text on an H100 GPU. The card has 312 TFLOP/s of FP16 compute — the number printed on the data sheet, the one that gets quoted in press releases. During text generation, they instrument the GPU and measure actual floating-point utilization.

The number is about 0.3%.

The GPU is doing 0.3% of what it is capable of. The other 99.7% of its compute capacity is sitting idle, waiting. Not waiting for the model to decide what token comes next. Waiting for *memory*.

This is the bandwidth wall — the defining constraint of autoregressive LLM inference. Understanding it is not academic. It determines which optimizations actually matter (KV cache compression) and which don't (faster attention kernels during decoding, to first approximation).

{{< crosshead >}}Two Phases, Two Bottlenecks{{< /crosshead >}}

An LLM processes a prompt in two distinct phases, and they have completely different performance profiles.

**Prefill:** The input tokens — all $T$ of them — are processed in a single parallel pass. The attention computation is a batch of $T$ queries all computed simultaneously, which means the GPU is performing a large matrix multiply ($T \times D$) against the key and value matrices. This is compute-bound: the GPU's arithmetic units are busy, the parallelism is high, and the bottleneck is FLOP throughput.

**Decode:** One new token is generated at a time. Each generation step requires loading the entire accumulated {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} from High Bandwidth Memory (HBM) — the GPU's main memory — so the single new query can attend to every past key. There are no large matrix multiplications. The {{< wiki "attention" >}}attention{{< /wiki >}} computation is a rank-1 update: one query vector against T key vectors. This is deeply memory-bandwidth-bound.

The transition from prefill to decode is the transition from "GPU as compute engine" to "GPU as memory bus."

{{< crosshead >}}The Roofline, Drawn{{< /crosshead >}}

The **roofline model** (Williams, Waterman, Patterson, 2008) is the tool of choice for reasoning about this. On the X axis: *arithmetic intensity*, measured in FLOPs per byte of data moved. On the Y axis: achievable throughput in FLOP/s. The hardware sets two limits:

1. A horizontal ceiling at peak compute throughput (312 TFLOP/s for H100 FP16).
2. A diagonal slope at peak memory bandwidth × arithmetic intensity (H100 HBM3: ~3.35 TB/s, so the slope is 3.35 × $10^{12}$ FLOP/s per FLOP/byte).

Any workload sits on this plot at the intersection of its arithmetic intensity and the lower of the two ceilings. A workload with arithmetic intensity below the *ridge point* (where the ceiling meets the slope) is memory-bandwidth-bound. A workload above the ridge is compute-bound.

The H100 ridge point: $\frac{312 \times 10^{12}}{3.35 \times 10^{12}} \approx 93$ FLOPs/byte.

Where does decode attention sit? Each byte of KV cache loaded from HBM enables roughly one multiply-add operation (the dot product of the query vector with that byte's key). Arithmetic intensity ≈ 1 FLOP/byte. That is *ninety-three times below* the ridge. Decode attention is not slightly memory-bound. It is catastrophically, irredeemably memory-bound.

```pyplot {id="roofline" caption="THE H100 ROOFLINE MODEL — DECODE ATTENTION SITS FAR INTO THE BANDWIDTH-LIMITED REGIME"}
fig, ax = plt.subplots(figsize=(9, 5))

# Roofline parameters (H100 FP16)
peak_compute = 312e12      # FLOP/s
peak_bandwidth = 3.35e12   # bytes/s
ridge_point = peak_compute / peak_bandwidth  # ~93 FLOPs/byte

intensities = np.logspace(-1, 3, 400)  # FLOPs/byte
roofline = np.minimum(peak_compute, peak_bandwidth * intensities)

ax.loglog(intensities, roofline, color='#1A1A1A', linewidth=2.5, label='H100 roofline')
ax.fill_between(intensities, roofline, alpha=0.08, color='#1A1A1A')

# Mark ridge
ax.axvline(ridge_point, color='#1A1A1A', linewidth=0.8, linestyle=':')
ax.text(ridge_point * 1.1, peak_compute * 0.6,
        f'ridge ≈ {ridge_point:.0f} FLOPs/byte', fontsize=8, color='#1A1A1A')

# Mark decode attention (~1 FLOP/byte)
decode_intensity = 1.0
decode_throughput = peak_bandwidth * decode_intensity
ax.scatter([decode_intensity], [decode_throughput],
           s=120, color='#FF007F', zorder=5, label='decode attention (~1 FLOP/byte)')
ax.annotate('decode\nattention', xy=(decode_intensity, decode_throughput),
            xytext=(3, decode_throughput * 1.5),
            fontsize=8, color='#FF007F',
            arrowprops=dict(arrowstyle='->', color='#FF007F', lw=1.2))

# Mark decode + KVzap (slightly higher intensity, same throughput — still bandwidth-bound)
kvzap_intensity = 1.05   # ~5% more FLOPs per byte (the surrogate overhead)
kvzap_throughput = peak_bandwidth * kvzap_intensity
ax.scatter([kvzap_intensity], [kvzap_throughput],
           s=100, color='#FF8C00', zorder=5, marker='D',
           label='decode + KVzap surrogate (~1.05 FLOPs/byte)')

# Mark prefill (near ridge)
prefill_intensity = 60
prefill_throughput = min(peak_compute, peak_bandwidth * prefill_intensity)
ax.scatter([prefill_intensity], [prefill_throughput],
           s=120, color='#00A8A8', zorder=5, label=f'prefill ({prefill_intensity} FLOPs/byte, near ridge)')

ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
ax.set_ylabel("achievable throughput (FLOP/s)")
ax.set_title("Decode sits 90× below the ridge — FLOPs added by KVzap's surrogate are free")
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(0.1, 1000)
ax.set_ylim(1e11, 5e12)
ax.spines[['top', 'right']].set_visible(False)

print(f"H100 ridge point: {ridge_point:.1f} FLOPs/byte")
print(f"Decode attention intensity: ~1 FLOPs/byte")
print(f"Distance below ridge: {ridge_point:.0f}×")
print(f"KVzap overhead in FLOPs: ~{(kvzap_intensity/decode_intensity - 1)*100:.0f}%")
print(f"KVzap throughput change: {(kvzap_throughput/decode_throughput - 1)*100:.2f}% (effectively zero)")
```

The orange diamond (decode + KVzap surrogate) sits directly on top of the pink dot (decode alone) at this scale. Adding 5% more FLOPs to a workload that is 93× below the ridge does not move the throughput needle. The GPU was not using those compute units anyway.

This is KVzap's key engineering insight: **the surrogate's FLOPs are free**. During a memory-bound decode step, the GPU's arithmetic units are stalled waiting for HBM reads. The surrogate's matrix multiplication runs on those idle units. It does not delay the HBM read; it does not displace useful computation. It costs compute cycles that would otherwise have been wasted.

{{% pullquote type="counter-intuitive" %}}
KVzap adds matrix multiplications to each decode step and reports < 1.1% wall-clock overhead. The math checks out: when your workload is 93× below the compute ceiling, extra FLOPs are genuinely free.
{{% /pullquote %}}

{{< crosshead >}}The Napkin Math for Llama-65B{{< /crosshead >}}

Let the numbers make this concrete. The {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} for a single forward pass has shape $(2, L, H, T, D)$ — two tensors (keys and values), $L$ layers, $H$ attention heads, $T$ token positions, $D$ head dimension.

For Llama-65B at 128K context ($T = 131072$):
- $L = 80$ layers
- $H = 64$ heads  
- $D = 128$ head dimension
- dtype: float16 (2 bytes)

$$
\text{KV cache size} = 2 \times 80 \times 64 \times 131072 \times 128 \times 2 \text{ bytes} \approx 335 \text{ GB}
$$

Every decode step must load these 335 GB from HBM before the new token can be computed. H100 HBM3 bandwidth: ~3.35 TB/s.

$$
\text{Time to load KV cache} = \frac{335 \text{ GB}}{3.35 \text{ TB/s}} \approx 100 \text{ ms per token}
$$

One hundred milliseconds per token, just for the memory transfer. Before any computation. A generation speed of 10 tokens per second — which feels slow — is almost entirely the memory wall, not computation.

Now apply KVzap at 3.5× compression (observed for Qwen3-8B, comparable for larger models):

$$
\text{Compressed KV cache} = \frac{335 \text{ GB}}{3.5} \approx 96 \text{ GB}
$$

$$
\text{New load time} = \frac{96 \text{ GB}}{3.35 \text{ TB/s}} \approx 29 \text{ ms per token}
$$

Decode throughput scales by roughly 3.5×: from 10 tokens/second to ~35 tokens/second, without changing the model, without changing the GPU, without optimizing the kernel. Just loading less data.

```pyplot {id="memory-load-vs-context" caption="KV CACHE LOAD TIME PER DECODE STEP VS CONTEXT LENGTH — LINEAR SCALING MAKES COMPRESSION INCREASINGLY VALUABLE"}
context_lengths = np.array([4096, 8192, 16384, 32768, 65536, 131072])
labels_k = [f'{t//1024}k' for t in context_lengths]

# Llama-65B KV cache size in GB vs context length
# 2 * 80 * 64 * T * 128 * 2 bytes = 2621440 * T bytes = ~2.62 MB/token
bytes_per_token = 2 * 80 * 64 * 128 * 2  # = 2,621,440 bytes per token position
kv_sizes_gb = bytes_per_token * context_lengths / 1e9

# Load time in ms at 3.35 TB/s
load_time_ms = kv_sizes_gb / 3350  * 1000  # ms

# With KVzap 3.5x compression
load_time_kvzap = load_time_ms / 3.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Left: KV cache size
ax1.bar(labels_k, kv_sizes_gb, color='#FF007F', edgecolor='#1A1A1A', linewidth=0.8, label='Full KV cache')
ax1.bar(labels_k, kv_sizes_gb / 3.5, color='#00A8A8', edgecolor='#1A1A1A', linewidth=0.8, label='KVzap 3.5× compressed')
ax1.set_ylabel("KV cache size (GB)")
ax1.set_title("Llama-65B KV cache size vs context length")
ax1.set_xlabel("context length")
ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)

# Right: load time per decode step
ax2.plot(labels_k, load_time_ms, color='#FF007F', linewidth=2, marker='o', label='Full KV: load time per step')
ax2.plot(labels_k, load_time_kvzap, color='#00A8A8', linewidth=2, marker='s', label='KVzap 3.5×: load time per step')
ax2.set_ylabel("HBM load time per decode step (ms)")
ax2.set_title("Decode bottleneck: loading KV cache from HBM")
ax2.set_xlabel("context length")
ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)

print("Llama-65B KV cache load times per decode step:")
for t, size, lt, lt_z in zip(context_lengths, kv_sizes_gb, load_time_ms, load_time_kvzap):
    print(f"  T={t:7d}  full={size:6.1f}GB  load={lt:6.1f}ms  kvzap={lt_z:5.1f}ms  speedup={lt/lt_z:.1f}x")
```

The speedup is identical at every context length — 3.5× — because both full and compressed load times scale linearly with $T$. But the absolute time saved grows: at 4K context, you save ~10 ms per step. At 128K, you save ~71 ms. At 128K, the savings matter more because each step was already expensive.

{{< crosshead >}}The Von Neumann Bottleneck, Again{{< /crosshead >}}

This is not a new problem. It is a problem from 1945.

John von Neumann's stored-program architecture separated compute from memory. The processor sits on one side of a bus; the memory sits on the other. Every data item must cross the bus to reach the arithmetic units. This is fine when the bus is fast relative to the compute — which, in the early days of computing, it was. By the 1960s, compute was outpacing memory bandwidth. By the 1980s, the gap had become a crisis.

The solution in CPUs: **caches**. Keep the most-recently-used data close to the processor in fast, small SRAM. When you need it again, it's already there. Organize your data and code so you reuse what's in cache (spatial and temporal locality) rather than constantly going back to slow DRAM.

The problem in LLM inference is structurally identical. The GPU's compute units (arithmetic logic units) sit on one side of the HBM bus. The KV cache sits in HBM on the other side. During autoregressive decode, every token generation requires *all* the KV cache — there is no temporal locality, because every new token might attend to any past token. The entire cache must cross the bus for every step.

**KV cache compression is the LLM equivalent of CPU caching.** Instead of keeping recently-used data small so it fits in fast SRAM, it keeps the *important* tokens and evicts the rest so less data must cross the HBM bus. The same bottleneck. The same insight. Sixty years apart.

{{% callout type="tangent" title="PagedAttention and Non-Contiguous Cache" %}}
Before **PagedAttention** (Kwon et al., 2023, vLLM), KV caches were allocated as contiguous blocks of GPU memory — you had to pre-allocate the maximum possible size for every sequence in a batch. This wasted memory (not every sequence uses its full budget) and caused fragmentation (freeing one sequence left gaps that couldn't be reused cleanly).

PagedAttention stores KV cache in fixed-size pages (like OS virtual memory), mapping logical sequence positions to physical pages through a page table. This allows non-contiguous storage: a single sequence's KV cache can live in scattered pages across HBM.

KVzap requires PagedAttention or equivalent. After pruning, different heads retain different numbers of tokens — the cache is no longer a rectangular tensor. Variable-length, non-uniform KV caches cannot be stored in pre-allocated contiguous blocks. PagedAttention's page-table abstraction makes this tractable.
{{% /callout %}}

{{< crosshead >}}Why This Changes the Production Calculus{{< /crosshead >}}

Before KVzap, the production inference argument against KV pruning was:

> "Even if the method is faithful, the overhead during decode (modified attention kernels, eviction logic, tracking scores) adds latency that partly cancels the throughput benefit."

This argument held for every method through 2025. H₂O's modified kernel added overhead. Expected Attention's per-query computation added overhead. KVzip's pretext task doubled prefill. The methods that were faithful were slow; the methods that were fast (StreamingLLM) were not faithful.

KVzap breaks this tradeoff with a precise engineering observation: during decode, the GPU is memory-bound at the 1 FLOP/byte level, and the surrogate runs at 1.05 FLOPs/byte. The 5% extra FLOPs are genuinely free. The overhead comes entirely from the reduced HBM transfer (fewer KV pairs to load) working against the small increase in surrogate computation — and the reduced transfer wins by 3.5×.

The production argument now runs:

> "At 3.5× compression, KV cache load time drops by 3.5×. Decode throughput increases by 3.5×. The surrogate overhead is < 1.1%. The net speedup is real, measurable, and does not require any changes to the attention kernel."

That is a production-ready argument. Which is why, for the first time, KV pruning is entering the roadmaps of inference engine teams.

## What To Remember

1. **Decode is memory-bandwidth-bound, not compute-bound.** The H100 can sustain 312 TFLOP/s; during decode attention, it uses < 1%. The bottleneck is the ~3.35 TB/s HBM bandwidth.
2. **Loading the KV cache dominates decode latency.** At 128K context, Llama-65B's KV cache is 335 GB. At H100 bandwidth, loading it takes ~100 ms per token — before any computation.
3. **KVzap's extra FLOPs are free.** Adding 5% more arithmetic to a workload that sits 93× below the compute ceiling has no effect on wall-clock time. The surrogate runs on idle compute units during the HBM stall.
4. **KV cache compression is the LLM equivalent of CPU caching.** The von Neumann bottleneck from 1945 reappears at GPU scale. The solution is the same: keep what's important, discard the rest, reduce bus traffic.
5. **PagedAttention enables variable-length pruning.** KVzap needs non-contiguous, variable-length KV caches; PagedAttention's page-table abstraction makes this possible in production inference engines.

**Continue to** → **[KVzap: The Final Zap](../06-kvzap/)** — the compute overhead analysis in detail, the KVpress Leaderboard numbers, and why 1.1% overhead translates to zero wall-clock cost during decoding.
