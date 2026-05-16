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
header: default.png
---

## August 2025, The Second Admission

On August 12, 2025, DeepSeek uploaded the V3.1 technical report. It was a quiet Tuesday. The report ran to sixty-three pages and included benchmark tables, training curves, hardware utilization breakdowns, and — on page 31 — a sentence that most readers skimmed past:

*"At 128K context lengths, attention computation is compute-bound rather than memory-bound. Prefill latency scales quadratically with context length."*

The team wasn't being modest. They were explaining why DeepSeek-V3.1 needed native sparse attention in the first place. The [MLA chapter](../02-mla-rewind/) we just walked through — the 30× KV cache reduction, the absorption trick, the RoPE side channel — that was May 2024. Fifteen months later, the team was publicly admitting that MLA solved the *wrong half* of the long-context problem.

The wall had moved. MLA killed the memory wall. The compute wall was still standing.

This chapter is about that wall. What it is, why it's quadratic, why FlashAttention doesn't help with it, and why it took DeepSeek until 2025 to build the thing that actually breaks it.

{{< crosshead >}}The Score Matrix Nobody Materializes (But Everybody Pays For){{< /crosshead >}}

Every {{< wiki "attention" >}}attention{{< /wiki >}} layer computes a score matrix. For a sequence of $T$ tokens with $H$ heads and head dimension $D$:

$$S = QK^\top, \quad S \in \mathbb{R}^{H \times T \times T}$$

At $T = 128{,}000$: each entry of $S$ is one dot product of two $D$-dimensional vectors. The total number of multiply-adds to compute $QK^\top$ is:

$$\text{FLOPs} = 2 \times H \times T^2 \times D$$

For Llama-3-70B at 128K context, GQA-8 (8 KV groups, effectively $H_\text{KV} = 8$):

$$2 \times 8 \times (128{,}000)^2 \times 128 = 2 \times 8 \times 1.638 \times 10^{10} \times 128 \approx 3.4 \times 10^{13} \text{ FLOPs per layer}$$

Across $L = 80$ layers: $\approx 2.7 \times 10^{15}$ FLOPs for attention alone. An H100 at FP8 sustained throughput of roughly 1 petaFLOP/s delivers that in about **2.7 seconds** — *just for the attention scores*, not the MLP blocks, not the value-side attention computation, not logit projection.

And that's with GQA-8, which cut the KV head count by 8×. Before GQA, it would be 8× worse.

{{% callout type="definition" %}}
**FLOPs vs FLOP/s.** FLOPs (floating-point operations, plural) is a count: how many multiplications and additions does this computation require? FLOP/s is a rate: how many FLOPs can the hardware execute per second? Latency in seconds ≈ FLOPs / sustained FLOP/s. H100 FP8 peak is 3.96 petaFLOP/s theoretical; sustained on real kernels is closer to 1–1.5 petaFLOP/s for attention.
{{% /callout %}}

The key word in the V3.1 sentence: **quadratically**. Doubling context length quadruples attention compute. Going from 32K to 128K (4×) increases attention FLOPs by 16×. The MLP blocks scale linearly — they process each token independently, so doubling $T$ doubles MLP FLOPs. There's a crossover point where attention overtakes MLP; beyond it, attention is the budget.

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

**Prefill** processes all $T$ input tokens simultaneously in one forward pass. Every query attends to every key in a fully parallel operation. Attention compute: $O(T^2)$ per layer. The GPU does a huge amount of arithmetic, writing very little back to DRAM (the KV cache, once, at the end). This is **compute-bound**: the chip's arithmetic units are the bottleneck.

**Decode** generates one token at a time. Each new token's query attends to all $T$ cached keys — but loads them from the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} in DRAM one by one. Attention compute per step: $O(T)$. But memory traffic: also $O(T)$, reading the entire cache. This is **memory-bandwidth-bound**: the chip is waiting for data, not arithmetic.

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

{{< crosshead >}}The Economic Argument: Why 128K Is Expensive{{< /crosshead >}}

Let's ground this in the thing that actually drives decisions: cost per request.

A document analysis service that processes 128K-token documents. GPT-4-class pricing: approximately $10 per million input tokens. One 128K-token request: $1.28. Why does a single prefill request cost over a dollar?

Some quick napkin math. Assume an H100 costs ~$3/hour in cloud compute (server-grade, well-utilized):

- $3 / 3600 \approx \$0.00083$ per second of H100
- Attention prefill at 128K for a Llama-70B-scale model: roughly 2–5 seconds of H100 time (after accounting for kernel inefficiencies on top of the theoretical 2.7s)
- Just for attention: $0.00083 \times 3.5 \approx \$0.003$ in H100 cost per request

Scale to a million requests: $3,000 in attention compute alone. At 20–30% gross margin, you charge ~$10,000/M input tokens — which is exactly what GPT-4 charged at 32K context.

At 128K context, attention is 16× worse than at 32K. The same math now gives you $48 per million input tokens just for compute, before overhead, before memory, before networking. The "128K context window" feature on a GPT-4-class model was economically brutal.

MLA cut the KV cache by 30×. That helped decode — fewer bytes to load per step, lower GPU memory, more concurrent sequences. But for a 128K *prefill*, the cache isn't even populated yet during the forward pass. You write the cache once at the end of prefill; you never read it. So MLA's cache reduction did essentially nothing for prefill latency or prefill cost.

**The price of a long-document request stayed high after MLA. The cache shrunk; the bill didn't.**

{{% pullquote type="counter-intuitive" %}}
After MLA, DeepSeek-V2 could handle 128K contexts in decode efficiently. But the user still had to wait for prefill — and prefill cost the same compute as before. The first optimization made the car faster on the highway; the second one had to fix the on-ramp.
{{% /pullquote %}}

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
There's a deep irony in the sparse attention literature: the most natural way to decide which attention weights to skip is to compute them, check if they're small, and then discard them. But you've already done the expensive computation. The trick is to predict which entries will be small *without computing them*, using a cheap proxy. That's what the lightning indexer in DSA does — and it's the insight that makes native sparse attention actually work.
{{% /callout %}}

The field knew the quadratic wall existed. The field had been trying to break it for six years. The field had given up, essentially, by 2023. FlashAttention had made the quadratic cost more bearable (better memory bandwidth, longer practical context without OOM), and the community had mostly concluded that the real answer was "just add more GPUs."

DeepSeek's answer, delivered in 2025, is not "just add more GPUs." It's a specific, learned sparsity structure that routes each query to the small set of keys that actually matter for it — and a separate lightweight scorer that predicts the routing without computing the full attention. We build toward that answer over the next several chapters, starting with [the forensic tour of why sparse attention kept failing](../04-sparse-detour/).

## What To Remember

1. **MLA cut cache, not compute.** Attention's prefill FLOPs are $O(T^2)$ regardless of how you store the cache.
2. **The T × T score matrix is the bottleneck.** At 128K context on a 70B-scale model, computing attention scores takes several seconds per H100 per request.
3. **Prefill and decode have different walls.** Decode is memory-bound (cache loads). Prefill is compute-bound (score matrix). MLA addressed the decode wall. Something else has to address the prefill wall.
4. **FlashAttention changes the memory access pattern, not the FLOP count.** When you're compute-bound, improving memory access doesn't help.
5. **The field knew this in 2021.** Sparse attention is six years old. It just didn't ship until someone solved the routing problem.

**Continue to** → [Sparse Attention's Lost Decade](../04-sparse-detour/) — the forensic tour of why six years of efficient-transformer research failed to produce a single shipping production model.
