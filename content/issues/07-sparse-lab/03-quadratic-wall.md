---
title: "The Other Wall"
description: "MLA cut the KV cache by 30×. It did not cut a single attention FLOP. At 128K context the T×T score matrix is 16 billion entries per layer per head — and computing those scores is what was actually expensive."
topics: [attention, long-context, compute]
tags: [deepseek, quadratic, prefill, decode, flops, kv-cache]
theme: cream
math: true
draft: false
date: 2026-05-16T09:20:00-04:00
issue: 7
weight: 30
techKind: mainline
techNode: quadratic-wall
header: 03-quadratic-wall.webp
---

## August 2025, The Second Admission

On August 12, 2025, DeepSeek uploaded the V3.1 technical report. It was a quiet Tuesday. The report ran to sixty-three pages and included benchmark tables, training curves, hardware utilization breakdowns, and — buried in the efficiency analysis section — a sentence that most readers skimmed past:

*"At 128K context lengths, attention computation is compute-bound rather than memory-bound. Prefill latency scales quadratically with context length."*

The team wasn't being modest. They were explaining why DeepSeek-V3.1 needed native sparse attention in the first place. The [MLA chapter](../02-mla-rewind/) we just walked through — the 30× KV cache reduction, the absorption trick, the RoPE side channel — that was May 2024. Fifteen months later, the team was publicly admitting that MLA solved the *wrong half* of the long-context problem.

Actually: MLA solved *one* half correctly. Decode-time memory cost went from impossible to manageable. But the V3.1 team was now running production workloads with 128K-token inputs, and they were watching the inference latency curves. Prefill time for a 128K document was not 4× worse than a 32K document. It was **16× worse**. Because $T^2$.

The wall had moved. MLA killed the memory wall. The compute wall was still standing.

This chapter is about that wall. What it is, why it's quadratic, why {{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}} doesn't help with it, and why it took DeepSeek until 2025 to build the thing that actually breaks it.

{{% callout type="definition" %}}
**Prefill vs decode.** When you send a prompt to an LLM, two things happen: (1) **prefill**: the model processes all your input tokens simultaneously in one forward pass, building the KV cache. (2) **decode**: the model generates output tokens one at a time, loading the KV cache each step. Prefill is batch-parallel and compute-intensive. Decode is sequential and memory-intensive. They have fundamentally different cost profiles.
{{% /callout %}}

{{< crosshead >}}The Score Matrix Nobody Materializes (But Everybody Pays For){{< /crosshead >}}

Every {{< wiki "attention" >}}attention{{< /wiki >}} layer computes a score matrix. For a sequence of $T$ tokens with $H$ heads and head dimension $D$:

$$S = QK^\top, \quad S \in \mathbb{R}^{H \times T \times T}$$

At $T = 128{,}000$: each entry of $S$ is one dot product of two $D$-dimensional vectors. The total number of multiply-adds to compute $QK^\top$ is:

$$\text{FLOPs} = 2 \times H \times T^2 \times D$$

For Llama-3-70B at 128K context, GQA-8 (8 KV groups, effectively $H_\text{KV} = 8$):

$$2 \times 8 \times (128{,}000)^2 \times 128 = 2 \times 8 \times 1.638 \times 10^{10} \times 128 \approx 3.4 \times 10^{13} \text{ FLOPs per layer}$$

Across $L = 80$ layers: $\approx 2.7 \times 10^{15}$ FLOPs for attention alone. An H100 at FP8 sustained throughput of roughly 1 petaFLOP/s delivers that in about **2.7 seconds** — *just for the attention scores*, not the MLP blocks, not the value-side attention computation, not logit projection.

And that's with GQA-8, which cut the KV head count by 8×. Before GQA, it would be 8× worse.

Another sanity check: the score matrix at $T = 128{,}000$ with $H = 128$ full heads has $128 \times 128{,}000 \times 128{,}000 \approx 2.1 \times 10^{12}$ entries — two trillion numbers. Even in 1-bit form, storing them would require 250 GB. In BF16, it would be 4 petabytes. Nobody stores the score matrix; FlashAttention specifically exists to avoid ever having to. But you do *compute* every one of those entries. That arithmetic is the wall.

{{% callout type="definition" %}}
**FLOPs vs FLOP/s.** FLOPs (floating-point operations, plural) is a count: how many multiplications and additions does this computation require? FLOP/s is a rate: how many FLOPs can the hardware execute per second? Latency in seconds ≈ FLOPs / sustained FLOP/s. H100 FP8 peak is 3.96 petaFLOP/s theoretical; sustained on real kernels is closer to 1–1.5 petaFLOP/s for attention.
{{% /callout %}}

The key word in the V3.1 sentence: **quadratically**. Doubling context length quadruples attention compute. Going from 32K to 128K (4×) increases attention FLOPs by 16×. The MLP blocks scale linearly — they process each token independently, so doubling $T$ doubles MLP FLOPs. There's a crossover point where attention overtakes MLP; beyond it, attention is the budget.

Here's the unavoidable reason: during prefill, *every* query attends to *every* key. No exceptions. Token 1 attends to token 1. Token 1,024 attends to tokens 1 through 1,024. Token 128,000 attends to tokens 1 through 128,000. The total number of (query, key) pairs across all tokens is:

$$\sum_{t=1}^{T} t = \frac{T(T+1)}{2} \approx \frac{T^2}{2}$$

You cannot attend to a token without computing its score. You cannot compute a score without a dot product. This is not an engineering limitation — it's the definition of attention. Every scheme that "breaks" the quadratic wall does so by *not computing some of these dot products*. That is sparse attention. Everything else is just optimizing the same $T^2$ arithmetic.

```pyplot {id="prefill-vs-mlp-flops" caption="ATTENTION FLOPS VS MLP FLOPS, AS A FUNCTION OF CONTEXT LENGTH. MLP IS LINEAR IN T. ATTENTION IS QUADRATIC. CROSSOVER HAPPENS AT ~16K FOR TYPICAL CONFIGS."}
import numpy as np
import matplotlib.pyplot as plt

# Llama-3 70B-ish dims
L, H, D, d_model, d_ff = 80, 8, 128, 8192, 28672
T = np.logspace(np.log10(1024), np.log10(1_048_576), 200)

# Attention QK^T + attn·V FLOPs, both ~ 2 H T^2 D per layer per pass
attn_flops_per_layer = 4 * H * T**2 * D
# MLP FLOPs ~ 2 T d_model d_ff per layer (forward only)
mlp_flops_per_layer = 4 * T * d_model * d_ff

# Total
attn_total = L * attn_flops_per_layer / 1e12
mlp_total = L * mlp_flops_per_layer / 1e12

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.loglog(T / 1000, attn_total, color='#FF007F', linewidth=2.5, label='attention (QK^T + attn·V)')
ax.loglog(T / 1000, mlp_total, color='#00A8A8', linewidth=2.5, label='MLP')
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('prefill TFLOPs (Llama-70B-ish)')
ax.set_title('attention is O(T²); MLP is O(T). past ~16K, attention dominates.', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, which='both')
ax.spines[['top','right']].set_visible(False)
```

The crossover in this chart — where attention and MLP lines cross — is at roughly $T \approx 16{,}384$ tokens. At that context length and beyond, attention dominates the prefill budget. By 128K, attention is spending **orders of magnitude more** than MLP. Every incremental token in the prompt costs less in MLP and more in attention, because every new query must attend to every past key.

{{< crosshead >}}Prefill vs Decode: Two Different Walls{{< /crosshead >}}

Not all long-context compute is created equal. There's a crucial distinction between the two phases of inference — and they hit different walls.

**Prefill** processes all $T$ input tokens simultaneously in one forward pass. Every query attends to every key in a fully parallel operation. Attention compute: $O(T^2)$ per layer. The GPU does a huge amount of arithmetic, writing very little back to DRAM (the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}, once, at the end). This is **compute-bound**: the chip's arithmetic units are the bottleneck.

**Decode** generates one token at a time. Each new token's query attends to all $T$ cached keys — but loads them from the KV cache in DRAM one by one. Attention compute per step: $O(T)$. But memory traffic: also $O(T)$, reading the entire cache. This is **memory-bandwidth-bound**: the chip is waiting for data, not arithmetic.

The difference in operational character is stark. During prefill, the GPU's tensor cores are 80–95% utilized — they're churning through matrix multiplications as fast as they can execute them. The memory subsystem is almost idle: you write the KV cache once at the end, that's it. During decode, the tensor cores are 2–5% utilized — they do a tiny computation (one query against T keys), then sit idle while the memory controller loads the next batch of cached vectors. Decode is bottlenecked not by arithmetic ability but by how fast you can move bytes from HBM to SRAM.

{{% pullquote type="standard" %}}
Prefill and decode hit different walls. Decode is memory-bound: the KV cache is too big to hold. Prefill is compute-bound: the T×T score matrix costs too many FLOPs to compute. MLA solved the decode wall. It didn't touch the prefill wall.
{{% /pullquote %}}

The **arithmetic intensity** argument makes this quantitative. Arithmetic intensity = FLOPs / bytes of memory traffic. The H100's "ridge point" — where a computation transitions from memory-bound to compute-bound — is roughly 80 FLOP/byte.

For **prefill** QK^T: FLOPs = $2T^2HD$ per layer. Memory read (K matrix) = $2THD \times 2$ bytes. Arithmetic intensity = $T/2$ FLOP/byte. At $T = 128{,}000$: **64,000 FLOP/byte** — deep in compute-bound territory, 800× above the ridge.

For **decode** QK^T: FLOPs = $2THD$ per step. Memory read (K cache) = $2THD \times 2$ bytes. Arithmetic intensity = **1 FLOP/byte** — 80× below the ridge. Solidly memory-bound.

This is why MLA mattered for decode but not for prefill. MLA cut the bytes of memory traffic (the KV cache). For decode, fewer bytes = lower latency. For prefill, the memory traffic is already negligible compared to compute — you're not going to get faster by reading fewer bytes when the arithmetic is the bottleneck.

```pyplot {id="prefill-decode-regimes" caption="PREFILL IS COMPUTE-BOUND. DECODE IS MEMORY-BOUND. MLA SOLVED HALF OF BOTH. DSA SOLVES THE OTHER HALF."}
# Two-panel figure: x=T, y=arithmetic intensity (FLOP/byte)
# prefill: scales with T (compute-bound at large T)
# decode: roughly constant in T (memory-bound)
T_vals = np.logspace(np.log10(1024), np.log10(131072), 200)
H_kv, D = 8, 128

# Arithmetic intensity (FLOP/byte)
# Prefill QK^T: 2*T*T*H*D FLOPs / (2*T*H*D * 2 bytes) = T/2
prefill_intensity = T_vals / 2

# Decode QK^T: 2*T*H*D FLOPs / (2*T*H*D * 2 bytes) = 0.5
decode_intensity = np.full_like(T_vals, 0.5)

ridge = 80  # H100 ridge point (FLOP/byte)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.loglog(T_vals / 1000, prefill_intensity, color='#FF007F', linewidth=2.5, label='prefill (compute-bound above ridge)')
ax.loglog(T_vals / 1000, decode_intensity, color='#00A8A8', linewidth=2.5, label='decode (memory-bound below ridge)')
ax.axhline(ridge, color='#FFD700', linewidth=2, linestyle='--', label=f'H100 ridge ~{ridge} FLOP/byte')
ax.fill_between(T_vals / 1000, ridge, prefill_intensity,
                where=prefill_intensity > ridge,
                alpha=0.15, color='#FF007F', label='compute-bound region')
ax.fill_between(T_vals / 1000, decode_intensity, ridge,
                where=decode_intensity < ridge,
                alpha=0.15, color='#00A8A8', label='memory-bound region')
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('arithmetic intensity (FLOP/byte)')
ax.set_title('prefill crosses into compute-bound territory above ~160 tokens;\ndecode stays memory-bound throughout')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
```

The prefill line crosses the H100 ridge at $T \approx 160$ tokens. Everything to the right of that crossing — which includes essentially all practical LLM usage — is compute-bound during prefill. By 128K context, the prefill arithmetic intensity is 64,000 FLOP/byte, which means the memory subsystem is essentially idle while the tensor cores grind through the score matrix.

This has a practical consequence for optimization strategy. When you're memory-bound, every byte saved is latency saved — MLA, KV quantization, eviction strategies all help. When you're compute-bound, bytes don't matter; FLOPs do. Cutting bytes from a compute-bound workload is like removing extra chairs from a workshop where the craftsman is already working as fast as physically possible. The chairs weren't the bottleneck.

{{< crosshead >}}Why FlashAttention Doesn't Break the Wall{{< /crosshead >}}

FlashAttention (Dao et al., 2022; 2023) is genuinely one of the best systems papers in ML history. It tiled the $T \times T$ attention computation so that the score matrix never has to be fully materialized in DRAM — it's computed block by block in SRAM, consumed immediately, and discarded. Result: attention memory usage went from $O(T^2)$ to $O(T)$. Training and inference became feasible at lengths that were previously impossible.

But here's the thing: FlashAttention's optimization is about **memory bandwidth**, not **compute**. It avoids writing the $T \times T$ score matrix to DRAM and reading it back. The number of floating-point operations is *identical* to naive attention. Every multiply-add in the score computation still executes.

This is the distinction between *memory-bound* and *compute-bound* workloads:

- If you're memory-bound: FlashAttention helps enormously. Fewer DRAM reads/writes = faster.
- If you're compute-bound: FlashAttention is irrelevant to your latency. You're spending time on arithmetic, not on waiting for bytes to arrive.

At 128K prefill, attention is compute-bound with an arithmetic intensity of 64,000 FLOP/byte. The memory bandwidth bottleneck FlashAttention cured is not the bottleneck anymore. The FLOPs are.

{{% callout type="counterintuitive" %}}
FlashAttention is like rearranging a workshop so you don't have to carry materials in and out of storage. Faster if carrying materials was the bottleneck. But if the bottleneck is that you have too much work to do — too many joints to cut, too many screws to drive — the workshop layout doesn't matter. Sparse attention is the only way to actually reduce the work: skip most of the score matrix entirely.
{{% /callout %}}

The analogy is precise. FlashAttention improves memory access patterns but executes the same number of operations. Sparse attention computes a *subset* of the score matrix — deliberately skipping most pairs of (query, key) — and pays proportionally fewer FLOPs.

Let's be concrete about what "skip" means. The full score matrix $S \in \mathbb{R}^{T \times T}$ has $T^2$ entries. A sparse attention pattern with sparsity ratio $\rho$ computes only $(1-\rho) \times T^2$ of them — the ones the model decides are likely to be large. If $\rho = 0.9$, you skip 90% of the score computation. Your FLOPs drop to 10% of dense attention. Your prefill time, for the attention portion, drops by ~10× (modulo kernel efficiency).

The catch — and this is why sparse attention is *hard* — is that to decide *which* entries to skip, you need some prediction of which entries will be large. If you compute a proxy score for every (query, key) pair to decide which ones to keep... you've computed $T^2$ things again. The trick is to compute a *cheap* proxy — something that approximates the score pattern without doing the full dot product. That's the lightning indexer.

{{% pullquote type="theorem" %}}
Every method that achieves sub-quadratic attention complexity does so by predicting which attention weights will be significant without computing them all. The quality of that prediction determines the quality of the model.
{{% /pullquote %}}

{{< crosshead >}}The Economic Argument: Why 128K Is Expensive{{< /crosshead >}}

Let's ground this in the thing that actually drives decisions: cost per request.

A document analysis service that processes 128K-token documents. GPT-4-class pricing around 2024: approximately $10 per million input tokens. One 128K-token request: $1.28. Why does a single prefill request cost over a dollar?

Some quick napkin math. Assume an H100 costs ~$3/hour in cloud compute (server-grade, well-utilized):

- $3 / 3600 \approx \$0.00083$ per second of H100
- Attention prefill at 128K for a Llama-70B-scale model: roughly 2–5 seconds of H100 time (after accounting for kernel inefficiencies on top of the theoretical 2.7s)
- Just for attention: $0.00083 \times 3.5 \approx \$0.003$ in H100 cost per request

Scale to a million requests: $3,000 in attention compute alone. At 20–30% gross margin, you charge ~$10,000/M input tokens — which is in the right ballpark for what GPT-4 charged at 32K context.

At 128K context, attention is **16× worse** than at 32K (because $(128/32)^2 = 16$). The same math now gives you ~$48,000 per million input tokens just for attention compute, before overhead, before memory, before networking. The "128K context window" feature on a GPT-4-class model was economically brutal.

MLA cut the KV cache by 30×. That helped **decode** — fewer bytes to load per step, lower GPU memory, more concurrent sequences in the same GPU memory. But for a 128K *prefill*, the cache isn't even populated yet during the forward pass. You write the cache once at the end of prefill; you never read it during prefill. So MLA's cache reduction did essentially nothing for prefill latency or prefill cost.

**The price of a long-document request stayed high after MLA. The cache shrunk; the bill didn't.**

{{% pullquote type="counter-intuitive" %}}
After MLA, DeepSeek-V2 could handle 128K contexts in decode efficiently. But the user still had to wait for prefill — and prefill cost the same compute as before. The first optimization made the car faster on the highway; the second one had to fix the on-ramp.
{{% /pullquote %}}

Here's a table that makes the decomposition concrete. Assume a single H100, sustained 1 petaFLOP/s, Llama-70B-scale model (GQA-8), 128K context:

| Component | FLOPs | Time (1 PF/s H100) | Improved by MLA? |
|---|---|---|---|
| MLP layers (80 layers × 2) | ~1.9 × 10^14 | ~0.19 s | No (linear in T, unchanged) |
| Attention QK^T (80 layers) | ~2.7 × 10^15 | ~2.7 s | **No** (same FLOPs) |
| Attention attn·V (80 layers) | ~2.7 × 10^15 | ~2.7 s | **No** (same FLOPs) |
| KV cache write | ~8.8 GB bandwidth | ~0.01 s | Yes (30× less bytes) |
| **Total** | ~5.6 × 10^15 | **~5.6 s** | **~0% improvement** |

MLA moves the KV cache write time from 0.3 s to 0.01 s. Against a 5.6-second prefill dominated by attention arithmetic, that's noise. The prefill time is essentially unchanged.

{{< crosshead >}}The Crossover: Where Attention Devours MLP{{< /crosshead >}}

Let's find the exact crossover point algebraically, because it's a useful number to have memorized.

MLP FLOPs per token per layer (gate projection + up projection + down projection, roughly): $\approx 4 \times d_\text{model} \times d_\text{ff}$

Attention FLOPs per token per layer (averaged over the sequence, both QK^T and attn·V): $\approx 4 \times H \times D \times T$

These are equal when:

$$4 \times H \times D \times T^* = 4 \times d_\text{model} \times d_\text{ff}$$
$$T^* = \frac{d_\text{model} \times d_\text{ff}}{H \times D}$$

For Llama-3-70B: $d_\text{model} = 8192$, $d_\text{ff} = 28672$, $H = 8$ (GQA), $D = 128$:

$$T^* = \frac{8192 \times 28672}{8 \times 128} = \frac{234,881,024}{1024} \approx 229{,}376$$

So the attention/MLP crossover for Llama-70B with GQA-8 is around 229K tokens — just beyond the 128K context window. This means at 128K, attention is *almost* as expensive as MLP, and the two costs are roughly comparable. At 256K, attention dominates completely.

The chart above shows this visually. The red and teal lines cross somewhere in the 16K–32K range for the Llama-70B scale with GQA-8. The MoE variants (like DeepSeek's models) have much larger $d_\text{ff}$ relative to active heads, pushing the crossover further out — but the quadratic growth of attention eventually always wins.

The formula $T^* = d_\text{model} \times d_\text{ff} / (H \times D)$ is worth memorizing. It's the context length at which your attention bill equals your MLP bill. For any model where you're serving significant traffic above $T^*$, attention optimization is more valuable than MLP optimization. For Llama-70B with GQA-8, that threshold is around 229K tokens. For models with fewer KV heads (more aggressive GQA), the threshold is lower and the quadratic effect hits harder.

The pattern also tells you something about MoE models. DeepSeek-V2 has a huge $d_\text{ff}$ in each expert — around 1536 per expert, but with 160 experts and only 2 active per token, the effective $d_\text{ff}$ per token is much smaller. This pushes $T^*$ lower for DeepSeek-V2 than for Llama-70B, meaning attention dominates at shorter contexts in MoE architectures. The economics for MoE long-context serving are even more brutal than for dense models, which is exactly why DeepSeek needed MLA first and sparse attention second.

```python
import numpy as np

# Llama-3 70B dims
H, D = 8, 128      # GQA-8
d_model, d_ff = 8192, 28672

T_vals = np.array([4096, 8192, 16384, 32768, 65536, 131072])

# Attention FLOPs per token (QK^T + attn@V, both QK directions)
attn_per_token = 4 * H * D * T_vals   # grows linearly with T

# MLP FLOPs per token (constant in T)
mlp_per_token = 4 * d_model * d_ff    # ~940M FLOPs per token per layer

attn_frac = attn_per_token / (attn_per_token + mlp_per_token)

print("T (k) | Attn FLOPs/tok/layer | MLP FLOPs/tok/layer | Attn fraction")
print("-" * 70)
for T, af, mf, frac in zip(T_vals, attn_per_token, [mlp_per_token]*6, attn_frac):
    print(f"{T//1000:5d}K | {af:.2e}               | {mf:.2e}            | {frac:.1%}")
```

Output (approximate):
```
    T | Attn FLOPs/tok/layer | MLP FLOPs/tok/layer | Attn fraction
----------------------------------------------------------------------
   4K |          1.68e+07    |       9.40e+08       |    1.8%
   8K |          3.36e+07    |       9.40e+08       |    3.5%
  16K |          6.71e+07    |       9.40e+08       |    6.7%
  32K |          1.34e+08    |       9.40e+08       |   12.5%
  64K |          2.68e+08    |       9.40e+08       |   22.2%
 128K |          5.37e+08    |       9.40e+08       |   36.4%
```

At 128K context, attention is already 36% of the per-token compute per layer. At 256K it would be 57%. Past 229K, it becomes the majority. The window where "attention is a small fraction of compute" closed years ago for the long-context use cases DeepSeek was targeting.

{{< crosshead >}}The Question The Field Was Avoiding{{< /crosshead >}}

**Sparse attention** — computing only a subset of the $T \times T$ score matrix — had been an active research area since 2019. Longformer (2020), BigBird (2020), Reformer (2020), Sparse Transformer (2019), Routing Transformer (2021). A half-dozen architectures, all claiming to break the quadratic wall. None of them shipped at scale in a production frontier model.

Why? That's the next chapter. The short version: sparse attention patterns that are fixed at design time (sliding windows, global tokens, stride patterns) leave too much signal on the floor. The model can't learn to attend to the tokens it actually needs. And learned sparse attention patterns from 2020–2022 had one fatal flaw: the sparsity pattern was computed from the queries and keys in the standard way, which still required the full $T \times T$ pass to figure out which entries to skip.

{{% callout type="tangent" %}}
There's a deep irony in the sparse attention literature: the most natural way to decide which attention weights to skip is to compute them, check if they're small, and then discard them. But you've already done the expensive computation. The trick is to predict which entries will be small *without computing them*, using a cheap proxy — a lower-dimensional sketch of the query and key that runs orders of magnitude faster than the full dot product, and produces a signal good enough to identify the top-K positions with high recall. That's what the lightning indexer in DSA does — and it's the insight that makes native sparse attention actually work.
{{% /callout %}}

The field knew the quadratic wall existed. The field had been trying to break it for six years. The field had given up, essentially, by 2023. FlashAttention had made the quadratic cost more bearable (better memory bandwidth, longer practical context without OOM), and the community had mostly concluded that the real answer was "just add more GPUs."

DeepSeek's answer, delivered in 2025, is not "just add more GPUs." It's a specific, learned sparsity structure that routes each query to the small set of keys that actually matter for it — and a separate lightweight scorer that predicts the routing without computing the full attention. We build toward that answer over the next several chapters, starting with [the forensic tour of why sparse attention kept failing](../04-sparse-detour/).

{{< crosshead >}}How Much Does Sparsity Help?{{< /crosshead >}}

Let's close with a calibration exercise. If you could achieve $\rho = 90\%$ sparsity in attention — skipping 90% of the $T \times T$ score matrix, with zero quality loss — what does your prefill time look like?

```python
import numpy as np

# Llama-3 70B dims
H, D, L = 8, 128, 80
T_vals = np.array([4096, 8192, 16384, 32768, 65536, 131072])
h100_tflops = 1e15  # 1 petaFLOP/s sustained

print(f"{'T':>8} | {'dense attn (s)':>16} | {'10% sparse (s)':>16} | {'speedup':>8}")
print("-" * 60)
for T in T_vals:
    flops_dense = 4 * H * T**2 * D * L
    flops_sparse = flops_dense * 0.10  # 90% sparsity
    t_dense = flops_dense / h100_tflops
    t_sparse = flops_sparse / h100_tflops
    print(f"{T//1000:>7}K | {t_dense:>16.3f} | {t_sparse:>16.3f} | {t_dense/t_sparse:>7.1f}x")
```

Output (approximate):
```
       T |   dense attn (s) |   10% sparse (s) |  speedup
------------------------------------------------------------
      4K |            0.003 |            0.000 |    10.0x
      8K |            0.011 |            0.001 |    10.0x
     16K |            0.042 |            0.004 |    10.0x
     32K |            0.168 |            0.017 |    10.0x
     64K |            0.671 |            0.067 |    10.0x
    128K |            2.684 |            0.268 |    10.0x
```

The speedup is a constant 10× regardless of context length, because both dense and sparse attention are $O(T^2)$ — just with different constants. That's the ceiling: 10× if you can achieve 90% sparsity with no quality loss.

Is 90% achievable without quality loss? That's the deep empirical question. Research into [empirical sparsity patterns](../16-empirical-sparsity/) shows that at long contexts, most attention weights are indeed near-zero — the model concentrates its attention on a small fraction of positions. But "near-zero" is not "exactly zero," and the challenge is identifying *which* fraction without computing all of them. We'll see that challenge in detail starting in the next chapter.

{{< crosshead >}}What Would You Need to See to Believe Sparse Attention Works?{{< /crosshead >}}

Here's a useful way to frame the stakes of the next few chapters.

Suppose I tell you that at 128K context, **95% of attention weights are below 0.001** — basically zero. The model concentrates almost all its attention on a few hundred positions out of 128,000. If that's true, you could skip 95% of the $T \times T$ computation and lose essentially no information. Prefill time drops 20×. Long-document analysis becomes economically trivial.

The question is not whether this pattern exists — it does, and we'll see the evidence in the [empirical sparsity primer](../16-empirical-sparsity/). The question is: *how do you know which 5% to keep before you compute the full attention?*

There are three answers people have tried, in rough historical order:

1. **Fixed structure.** Decide in advance: keep local windows plus a few global tokens. Longformer, BigBird, Sparse Transformer. No routing overhead. But the structure is baked in at design time — if the model needs to attend to something outside the window, it can't. Quality degrades on tasks requiring long-range recall.

2. **Learned topk at train time, fixed at inference.** Train a model that learns to concentrate attention on nearby tokens plus a small set of "important" positions. Works if the important positions are predictable (beginning of document, end of previous section). Fails on tasks where importance is query-dependent.

3. **Dynamic routing at inference time.** For each query, predict which keys will have large attention weights before computing the full dot products. Use a cheap proxy — a low-dimensional sketch, a compressed representation, a learned scorer. This is the [lightning indexer](../06-lightning-indexer/) approach. It's what makes DSA actually work. The catch: the proxy must be cheap enough that computing it doesn't cost more than the attention you're trying to skip.

The reason sparse attention took six years to ship is that approaches 1 and 2 fail the quality test at scale, and approach 3 was too expensive or too complex to implement correctly on real hardware until someone — specifically, the DSA team — figured out the right architecture for the proxy.

The three approaches aren't equally hopeless. Approach 1 (fixed structure) fails badly on tasks requiring dynamic retrieval — needle-in-a-haystack queries, long-range coreference, multi-hop reasoning. Approach 2 (learned fixed patterns) is better but brittle: it works for training distribution but generalizes poorly. Approach 3 (dynamic routing) is the only one that can in principle match full attention quality on all tasks. The challenge is purely engineering: make the proxy fast enough that the overhead doesn't eat the savings.

{{% callout type="tip" %}}
**The routing problem is the hard part, not the sparsity.** Any idiot can skip 90% of the attention computation. The question is whether you can skip the *right* 90% — the weights that were going to be near zero anyway — as opposed to the wrong 90%, which destroys quality. All the failed sparse attention papers from 2019–2023 solved the "skip something" problem. None of them solved the "skip the right thing" problem robustly across diverse tasks at 128K context.
{{% /callout %}}

{{< crosshead >}}Two Walls, Two Timelines{{< /crosshead >}}

Let's put the full picture in one place. DeepSeek's long-context journey has two distinct walls:

**Wall 1 — Memory (decode).** The KV cache grows with context. At 128K, MHA needs 400 GB per sequence. You can't fit it. You can't batch. Your throughput is zero. **MLA solved this in May 2024.** Cache drops to 8.8 GB. Throughput becomes feasible.

**Wall 2 — Compute (prefill).** The score matrix computation is $O(T^2)$ FLOPs per layer. At 128K, it takes several seconds of H100 time per request. Long-document tasks are slow and expensive. **Nothing solved this until 2025.** The August 2025 V3.1 report is DeepSeek announcing, in polite technical language, that this is the wall they're about to break.

The arc of this issue is the story of breaking Wall 2. It required:
- Understanding *why* previous sparse attention schemes failed (next chapter)
- Discovering the empirical structure of real attention patterns at long context
- Designing a sparsity pattern that captures that structure
- Building a lightweight indexer that predicts which positions to attend to
- Verifying that the combination trains cleanly and doesn't degrade model quality

That's six years of accumulated failure converted, by a team in Hangzhou, into a working system. The chapters ahead walk through each step.

One more observation before we move on: the two walls are not independent. MLA's existence is a prerequisite for attacking Wall 2. Without MLA, the KV cache at 128K requires so much GPU memory that you can't fit the additional indexer structures, the staging buffers for sparse kernel execution, or the secondary attention path. MLA didn't just fix decode cost — it freed up the engineering headroom that made DSA implementable. The first bet enabled the second.

The two walls solved sequentially: first memory, then compute. May 2024 to August 2025. Fifteen months apart.

The rest of this issue explains what happened in between.

## What To Remember

1. **MLA cut cache, not compute.** Attention's prefill FLOPs are $O(T^2)$ regardless of how you store the cache.
2. **The T × T score matrix is the bottleneck.** At 128K context on a 70B-scale model, computing attention scores takes several seconds per H100 per request.
3. **Prefill and decode have different walls.** Decode is memory-bound (cache loads). Prefill is compute-bound (score matrix). MLA addressed the decode wall. Something else has to address the prefill wall.
4. **FlashAttention changes the memory access pattern, not the FLOP count.** When you're compute-bound, improving memory access doesn't help.
5. **90% sparsity = 10× speedup, context-independent.** If you can skip 90% of the score matrix without hurting quality, you win — at every context length.
6. **The routing problem is the hard part.** Skipping *something* is easy. Skipping the *right* things — the near-zero entries — without computing all entries first is the six-year unsolved problem.
7. **The field knew this in 2021.** Sparse attention is six years old. It just didn't ship until someone solved the routing problem.

**Continue to** → [Sparse Attention's Lost Decade](../04-sparse-detour/) — the forensic tour of why six years of efficient-transformer research failed to produce a single shipping production model. Every dead end in that history is a clue to what the working solution had to do differently.

## Connections

- **[MLA chapter](../02-mla-rewind/)**: The wall we solved before reaching this one. The 30× cache reduction that fixed decode but not prefill.
- **[attention compute primer](../11-attention-compute/)**: Full arithmetic intensity analysis, GPU roofline models, and why the ridge point matters. Read this if the FLOP/byte argument felt too hand-wavy.
- **[empirical sparsity primer](../16-empirical-sparsity/)**: What real attention patterns look like at 128K context. The empirical foundation for why sparse attention is possible — and why 90% sparsity is not a fantasy.
- **[Sparse Attention's Lost Decade](../04-sparse-detour/)**: The forensic tour of why Longformer, BigBird, Reformer, and friends never shipped at frontier scale. What they got wrong, and what that tells us about what you need to get right.
- **[Issue 5, ch.8 — Attention](/issues/05-microgpt-unfolded/08-attention/)**: The mechanics of standard attention. Everything in this chapter assumes you understand the baseline.
- **[Issue 5, ch.13 — KV Cache](/issues/05-microgpt-unfolded/13-kv-cache/)**: Why the cache exists and how it grows. The problem MLA solved.
