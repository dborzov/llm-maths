---
title: "Attention FLOPs: the exact bill, line by line"
short_title: "Attention FLOPs"
description: "The full FLOP count for attention is 4HLTD per layer in prefill — quadratic in T — and FlashAttention eliminates materialization of the score matrix but does not reduce the arithmetic."
blurb:
  - "QK^T alone costs 2HT²D FLOPs. Attention·V adds another 2HT²D. Total per layer: 4HT²D."
  - "Prefill is O(T²); decode is O(T) per step. The same formula, different regime, two very different cost curves."
  - "FlashAttention is a memory trick: it avoids storing the T×T score matrix. It does not skip computing any of its entries."
  - "Softmax is O(HT²) scalar operations — 20–30× cheaper per op than a GEMM — and safely negligible in the FLOP budget."
topics: [attention, flops, hardware, primer]
tags: [attention-flops, prefill, decode, arithmetic-intensity, flash-attention]
theme: teal
math: true
draft: false
date: 2026-05-16T11:00:00-04:00
issue: 7
weight: 110
techKind: primer
techNode: attention-compute
header: 11-attention-compute.webp
---

## The T² Monster in the Room

Every time someone says "transformers scale to long context," they are quietly ignoring a monster. The monster lives in [Issue 5 ch.8 attention](/comicbook/05-microgpt/08-attention/), specifically in the two lines of code that dominate the entire model's compute at long context:

```python
attn_logits = np.einsum('thd,shd->ths', q, k) / np.sqrt(D)  # [T, H, T]
head_out    = np.einsum('ths,shd->thd', attn_weights, v)     # [T, H, D]
```

Both of those `einsum` calls are $O(T^2)$. Double the context length; quadruple the FLOPs. At T = 128K, this is not a mild overhead — it is the dominant term in the entire forward pass. This primer puts exact numbers on it.

{{< crosshead >}}The Three Operations{{< /crosshead >}}

The full microGPT {{< wiki "attention" >}}attention{{< /wiki >}} function from [Issue 5 ch.8](/comicbook/05-microgpt/08-attention/):

```python
def attention(q, k, v):  # q, k, v: [T, H, D] tensors
    # Step 1: score matrix
    attn_logits = np.einsum('thd,shd->ths', q, k) / np.sqrt(D)  # [T, H, T]
    # Step 2: normalize
    attn_weights = softmax(attn_logits, axis=-1)                  # [T, H, T]
    # Step 3: aggregate values
    head_out = np.einsum('ths,shd->thd', attn_weights, v)         # [T, H, D]
    return head_out
```

Let's count FLOPs precisely for a sequence of $T$ tokens, $H$ attention heads, head dimension $D$:

**Step 1 — $QK^\top$:** For each of the $T$ query tokens and each of the $H$ heads, we compute a dot product with each of the $T$ key vectors of dimension $D$. That is $T \times H \times T$ dot products of dimension $D$. Each dot product takes $2D$ FLOPs (multiply-add). Total: $2 H T^2 D$ FLOPs.

**Step 2 — Softmax:** Computing $\exp$ and normalization over $T$ entries, $H$ heads, $T$ query positions. $O(H T^2)$ operations, but these are scalar exp and division rather than multiply-accumulates. On modern hardware this is ~20–30× cheaper per operation than a GEMM. We will treat it as negligible compared to Steps 1 and 3 for the purpose of this primer.

**Step 3 — Attention @ V:** Symmetric with Step 1. For each of the $T$ query tokens and each of the $H$ heads, compute a weighted sum over $T$ value vectors of dimension $D$. Total: $2 H T^2 D$ FLOPs.

**Total per layer:** $4 H T^2 D$ FLOPs.

**Total for the whole model** ($L$ layers): $4 H L T^2 D$ FLOPs.

{{% pullquote type="theorem" %}}
$$\text{Attention FLOPs (prefill)} = 4 H L T^2 D$$
$$\text{Attention FLOPs (decode, one step)} = 4 H L T D$$
Decode is linear in $T$; prefill is quadratic. Both scale with the number of heads $H$, layers $L$, and head dimension $D$.
{{% /pullquote %}}

## A Running Toy Example

Let's trace the exact FLOPs through a small toy transformer so the formula is not abstract:

```python
T, H, D, L = 1024, 4, 64, 6   # toy transformer

# Step 1: QK^T
# For each of H heads: T×D matrix @ D×T matrix = T×T score matrix
# Per head: 2 × T × D × T = 2 T^2 D FLOPs (multiply-add)
step1_flops = 2 * H * T**2 * D
print(f"QK^T FLOPs:     {step1_flops:>15,.0f}")   # 536,870,912

# Step 3: attn @ V (symmetric)
step3_flops = 2 * H * T**2 * D
print(f"attn@V FLOPs:   {step3_flops:>15,.0f}")   # 536,870,912

total_per_layer = step1_flops + step3_flops
print(f"Total / layer:  {total_per_layer:>15,.0f}") # 1,073,741,824 ≈ 1 GFLOPs

# Full model (L layers)
total_model = total_per_layer * L
print(f"Total / model:  {total_model:>15,.0f}")    # ~6.4 GFLOPs

# Formula check
formula = 4 * H * L * T**2 * D
print(f"Formula check:  {formula:>15,.0f}")        # matches
```

At T=1024, H=4, D=64, L=6: about 6.4 GFLOPs for attention across the whole forward pass. That is manageable on a laptop GPU.

Now double the context to T=2048. FLOPs quadruple: ~25.6 GFLOPs. Double again to T=4096: ~102 GFLOPs. The $T^2$ term is merciless.

{{< crosshead >}}Napkin Math: Llama-3-70B at 128K{{< /crosshead >}}

Llama-3-70B uses Grouped Query Attention (GQA) with 8 KV groups: effectively $H_{\text{KV}} = 8$ for the purpose of KV cache computation, but the query-side attention has 64 heads total. For the FLOPs formula with GQA, the number of independent QK^T products per token is $H_{\text{KV}} = 8$ (each KV group is shared by 8 query heads). Parameters: $H = 8$, $L = 80$, $D = 128$, $T = 128{,}000$.

**Prefill FLOPs (all T tokens processed simultaneously):**
$$4 \times 8 \times 80 \times (128{,}000)^2 \times 128 = 4.3 \times 10^{14} \text{ FLOPs} = 430 \text{ TFLOPs.}$$

An H100 at FP16 peaks at ~1.98 PFLOP/s (theoretical). With a realistic 60% utilization: ~1.2 PFLOP/s effective. Attention prefill alone takes roughly $430 / 1{,}200 \approx 0.36$ seconds per H100 — and that's just attention, not the MLP blocks.

**Decode FLOPs (one new token at step T = 128K):**
$$4 \times 8 \times 80 \times 128{,}000 \times 128 \approx 3.4 \times 10^{10} \text{ FLOPs} = 34 \text{ GFLOPs per step.}$$

At H100 peak compute: $34 \text{ GFLOPs} / 1{,}200{,}000 \text{ GFLOPs/s} \approx 28 \mu\text{s}$. Trivially fast in compute time.

But that is not the actual decode latency. The bottleneck is not compute — it is **memory bandwidth**.

{{< crosshead >}}The Real Decode Bottleneck: Memory Bandwidth{{< /crosshead >}}

Every decode step must read the entire {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} from HBM (GPU memory). For Llama-3-70B with GQA-8 at T = 128K:

$$\text{KV cache size} = 2 \times H_{\text{KV}} \times L \times T \times D \times 2 \text{ bytes} = 2 \times 8 \times 80 \times 128{,}000 \times 128 \times 2 \approx 42 \text{ GB.}$$

(The leading 2 is for keys and values; the trailing 2 is bytes per BF16 number.)

H100 HBM bandwidth: ~3.35 TB/s. Time to read 42 GB:

$$42 \text{ GB} / 3.35 \text{ TB/s} \approx 12.5 \text{ ms per decode step.}$$

The compute says 28 µs. The memory bus says 12.5 ms. Memory bandwidth is the binding constraint by a factor of $12.5 / 0.028 \approx \mathbf{450\times}$.

Every token you generate at 128K context costs 12.5 ms just for reading the KV cache. That's **80 tokens per second maximum throughput** for a single sequence — limited not by the GPU's compute engine, but by how fast its memory can stream.

{{% pullquote type="counter-intuitive" %}}
At 128K context, generating one token requires reading 42 GB from GPU memory. The GPU's compute units are idle ~99.7% of the time, waiting for data. Sparse attention and KV compression work by reducing this memory read — not by making the compute faster.
{{% /pullquote %}}

## Arithmetic Intensity: The Formal Framework

**Arithmetic intensity** (AI) is the ratio of compute operations to memory traffic: how many FLOPs you perform per byte you load from HBM. It tells you whether an operation is compute-bound (AI >> H100 "ridge point") or memory-bound (AI << ridge point).

H100's ridge point is approximately 80 FLOPs/byte: if AI > 80, you're compute-bound; if AI < 80, you're spending more time waiting for data than computing.

**Prefill arithmetic intensity:**

For a single attention layer, prefill QK^T:
- FLOPs: $2 H T^2 D$ (the QK^T part).
- Memory traffic: Read $Q \in \mathbb{R}^{T \times H \times D}$ from SRAM (fast, already there), read $K \in \mathbb{R}^{T \times H \times D}$ from HBM: roughly $2 H T D \times 2$ bytes.
- Arithmetic intensity: $\frac{2 H T^2 D}{4 H T D} = \frac{T}{2}$.

At $T = 2048$: $\text{AI} = 1024 \gg 80$. Compute-bound. \\
At $T = 128{,}000$: $\text{AI} = 64{,}000 \gg 80$. Deeply compute-bound.

**Decode arithmetic intensity:**

For one new query attending to T cached keys:
- FLOPs: $2 H \times 1 \times T \times D = 2 H T D$.
- Memory: Read the entire $K$ cache: $H T D \times 2$ bytes.
- Arithmetic intensity: $\frac{2 H T D}{2 H T D} = 1$ FLOPs/byte.

1 < 80 — always memory-bound, regardless of T. Decode is *fundamentally* memory-bandwidth-limited.

```pyplot {id="attention-arithmetic-intensity" caption="ARITHMETIC INTENSITY (FLOP/BYTE) FOR PREFILL AND DECODE ACROSS CONTEXT LENGTHS. PREFILL CROSSES THE H100 'RIDGE POINT' (~80 FLOP/BYTE) AT MEDIUM T. DECODE NEVER DOES."}
T = np.logspace(np.log10(1024), np.log10(1_048_576), 200)
H = 8
L = 80
D = 128

# Prefill: compute scales as T^2, memory traffic as T (cache writes only)
# For QK^T: write T x D for queries, read T x D for keys. Compute T^2 D per head per layer.
prefill_flops = 4 * H * L * T**2 * D
prefill_bytes = 2 * H * L * T * D * 2  # K and V written once each, bf16
prefill_AI = prefill_flops / prefill_bytes

# Decode: compute T D per step per head per layer, memory T D per step
decode_flops = 4 * H * L * T * D
decode_bytes = 2 * H * L * T * D * 2  # read whole cache per step
decode_AI = decode_flops / decode_bytes

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.semilogx(T / 1000, prefill_AI, color='#FF007F', linewidth=2.5, label='prefill (compute-bound regime)')
ax.semilogx(T / 1000, decode_AI, color='#00A8A8', linewidth=2.5, label='decode (memory-bound regime)')
ax.axhline(80, color='#FF8C00', linewidth=1.5, linestyle='--', label="H100 ridge (~80 FLOP/byte)")
ax.set_xlabel('context length T (k tokens)')
ax.set_ylabel('arithmetic intensity (FLOPs / byte)')
ax.set_title('Prefill crosses the ridge at medium T. Decode stays below.', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)
```

The prefill AI line (pink) starts at ~512 at T=1K and climbs linearly. It is always well above the ridge — prefill is always compute-bound. The decode AI line (teal) is flat at exactly 1 FLOPs/byte, regardless of T. Decode is always memory-bound.

{{% callout type="tip" %}}
**Why does decode AI = 1 regardless of T?** Both the FLOPs (2HTD) and the memory traffic (2HTD bytes, roughly) scale linearly with T. They cancel in the ratio. This is why longer context does not help decode throughput — the ratio stays fixed and stays below the ridge.
{{% /callout %}}

## Why FlashAttention Is Memory, Not Compute

{{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}} is the most widely deployed optimization in transformer inference. It is worth being precise about what it does and does not do.

{{< crosshead >}}What FlashAttention Does{{< /crosshead >}}

The naive attention implementation:

1. Computes the full $T \times T$ score matrix $S = QK^\top / \sqrt{D}$ and **writes it to HBM**. Cost: $T^2 \times H \times 2$ bytes per layer.
2. Reads $S$ back from HBM to compute {{< wiki "softmax" >}}softmax{{< /wiki >}}($S$). Another $T^2 \times H \times 2$ bytes.
3. Writes the softmax'd weights to HBM. Another $T^2 \times H \times 2$ bytes.
4. Reads them back to compute `attn_weights @ V`. Another $T^2 \times H \times 2$ bytes.

Total HBM traffic for the intermediate: $4 \times T^2 \times H \times 2$ bytes. At $T = 128K$, $H = 8$: $4 \times (128{,}000)^2 \times 8 \times 2 = 1.05 \times 10^{14}$ bytes = **105 TB** per layer. That is absurd — the model's total weight file is smaller than this intermediate.

FlashAttention eliminates this by processing attention in *tiles*. It computes a tile of the score matrix in SRAM (fast, on-chip memory), applies the softmax incrementally using a running normalization trick (Milakov and Gimelshein 2018), and accumulates the output directly. The full $T \times T$ score matrix is **never materialized in HBM**.

FLOPs are **identical**: still $4 H L T^2 D$. What changes is HBM traffic: instead of reading and writing the intermediate $T \times T$ matrices, you only read inputs ($Q, K, V$) once and write the output once.

**Memory savings at $T = 128K$, $H = 8$:**
$$\text{Saved} = 4 \times (128{,}000)^2 \times 8 \times 2 \approx 105 \text{ TB per layer.}$$

That is the HBM traffic FlashAttention avoids. In practice this is what enables efficient prefill at long context — without it, the GPU spends most of its time streaming a huge intermediate matrix to and from HBM.

{{% callout type="warning" %}}
FlashAttention does not help decode. At decode time, there is no large intermediate matrix — you are computing a single row of the score matrix at a time (one new query). The bottleneck is reading the K and V cache from HBM, which FlashAttention does not change. FlashAttention is a prefill optimization.
{{% /callout %}}

## What Sparse Attention Actually Cuts

This is the key distinction in Issue 7's mainline story.

**FlashAttention**: Same $T^2$ FLOPs, better memory bandwidth. You still compute all $T^2$ scores — you just don't write the intermediate to HBM.

**DSA / CSA (sparse attention)**: Different FLOPs. You do not compute all $T^2$ scores. The lightning indexer computes $O(T \times d_I)$ scores (linear in T). Top-k attend runs over only $k$ entries ($O(k \times D)$, constant in T).

Let's put numbers on both:

```python
T = 128_000
H = 128
D = 128   # head dimension
L = 61    # V3.2-Exp layers

# Dense attention FLOPs (prefill, full model)
dense_flops = 4 * H * L * T**2 * D
print(f"Dense prefill FLOPs:   {dense_flops:.2e}")    # ~5.4e16

# FlashAttention: same FLOPs
flash_flops = dense_flops
print(f"Flash prefill FLOPs:   {flash_flops:.2e}")    # same

# DSA indexer FLOPs (per layer, linear in T)
d_I = 2048    # n_I_h * c_I = 32 * 64
indexer_flops_per_layer = 2 * T * d_I
print(f"DSA indexer / layer:   {indexer_flops_per_layer:.2e}")  # ~5.2e8

# DSA attend FLOPs (per layer, k=2048 selected tokens)
k = 2048
dsa_attend_per_layer = 2 * H * k * D
print(f"DSA attend / layer:    {dsa_attend_per_layer:.2e}")  # ~6.7e7

# DSA total (all layers, prefill approximation -- per query)
dsa_total = L * (indexer_flops_per_layer + dsa_attend_per_layer)
print(f"DSA total (per query): {dsa_total:.2e}")    # ~3.2e10

# Ratio
print(f"Dense/DSA ratio:       {dense_flops/dsa_total:.0f}×")  # ~1700×
```

At prefill (computing one query's attention across all T past tokens), DSA is roughly 1700× fewer FLOPs than dense attention per query per layer. The catch is that prefill must be run for all T queries simultaneously — but DSA is designed primarily to help *decode*, where the quadratic scaling would compound with each new generated token.

For decode, the comparison is:

| Operation | FLOPs per step | Scales with T? |
|---|---|---|
| Dense decode attention | $4 H L T D$ | Yes (linear) |
| DSA indexer score | $2 L T d_I$ | Yes (linear, but $d_I \ll H D$) |
| DSA top-k attend | $2 H L k D$ | **No** (constant in T!) |
| FlashAttention decode | $4 H L T D$ (same as dense) | Yes |

The top-k attend being constant in T is the structural advantage. At T=1M, dense decode attention scales to 128× more FLOPs than at T=8K. DSA's attend step stays fixed at k=2048 regardless.

{{% pullquote type="theorem" %}}
FlashAttention changes the constant factor on $O(T^2)$. Sparse attention changes the exponent. For long-context inference, the exponent is what matters.
{{% /pullquote %}}

## The Full Picture: Three Regimes

```pyplot {id="three-regimes" caption="THREE REGIMES OF ATTENTION AT DIFFERENT CONTEXT LENGTHS. SHORT CONTEXT: DENSE IS FINE. MEDIUM: FLASHATTENTION HANDLES PREFILL. LONG: ONLY SPARSE ATTENTION (DSA/CSA) KEEPS DECODE TRACTABLE."}
T = np.logspace(np.log10(512), np.log10(1_048_576), 200)
H, D, L = 128, 128, 61
k, d_I = 2048, 2048

# Dense decode: linear in T
dense_decode = 4 * H * L * T * D / 1e12   # TFLOPs

# DSA decode: constant (indexer linear, attend constant — attend dominates)
dsa_decode = L * (2 * T * d_I + 2 * H * k * D) / 1e12

# Dense prefill (per-token cost amortized over T): 4 H L T D per token
# So total / T = 4 H L D — same as decode! But total scales T^2.
dense_prefill_total = 4 * H * L * T**2 * D / 1e15  # PFLOPs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.loglog(T / 1000, dense_decode, color='#FF007F', linewidth=2.5, label='dense decode (linear T)')
ax1.loglog(T / 1000, dsa_decode,   color='#00A8A8', linewidth=2.5, label='DSA decode (linear indexer + const attend)')
ax1.axvline(32, color='#FF8C00', linewidth=1.5, linestyle=':', alpha=0.7, label='DSA activation threshold (32K)')
ax1.set_xlabel('context length T (k tokens)')
ax1.set_ylabel('TFLOPs per decode step')
ax1.set_title('Decode: DSA diverges from dense at 32K+', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, which='both')
ax1.spines[['top','right']].set_visible(False)

ax2.loglog(T / 1000, dense_prefill_total, color='#FF007F', linewidth=2.5, label='dense prefill (quadratic T)')
ax2.loglog(T / 1000, dense_prefill_total * 0.95, color='#FFD700', linewidth=2.5, linestyle='--',
           label='FlashAttention prefill (same FLOPs, less HBM)')
ax2.set_xlabel('context length T (k tokens)')
ax2.set_ylabel('PFLOPs total prefill')
ax2.set_title('Prefill: Flash saves HBM, not FLOPs. Still quadratic.', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, which='both')
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
```

The left panel shows decode FLOPs. Dense decode (pink) scales linearly with T — every new token has to query against a longer cache. DSA decode (teal) has a slowly-growing indexer component but the attend step is flat at k=2048. The divergence above 32K is why DSA is worth the overhead.

The right panel shows prefill FLOPs. The FlashAttention line is almost on top of the dense line (dotted because FlashAttention saves HBM traffic, not FLOPs). Both are quadratic. FlashAttention does not help here.

{{% callout type="tangent" %}}
**What about prefill with sparse attention?** DSA during prefill is more complex. For training and prefill you must compute all queries' scores simultaneously, and the indexer must be differentiable. This is why training used a soft top-k (temperature-scaled softmax) rather than hard selection. Production V3.2-Exp uses DSA at decode time; prefill may use a mixture of dense and sparse depending on the context length and batch configuration.
{{% /callout %}}

## What To Remember

1. **Attention is $4 H L T^2 D$ FLOPs for prefill.** The $T^2$ term dominates at long context.
2. **Decode is memory-bandwidth-limited.** Arithmetic intensity ≈ 1 FLOPs/byte at decode. Always below the H100 ridge. Compute units wait for HBM.
3. **FlashAttention removes the intermediate T × T matrix from HBM.** Same FLOPs, ~105 TB/layer fewer HBM writes at T=128K. Primarily a prefill optimization.
4. **Sparse attention (DSA, CSA) changes the algorithmic complexity.** From $O(T^2)$ to $O(T \cdot d_I + k)$ per decode step. The indexer is linear; the attend step is constant. This is what makes 1M-context inference possible.
5. **The key numbers for Llama-3-70B at T=128K:** Prefill = 430 TFLOPs. Decode compute = 34 GFLOPs. Decode memory = 42 GB. Memory dominates by 450×.

**Continue to** → [The Other Wall](../03-quadratic-wall/), where these FLOP counts power the mainline argument about why DSA had to exist.
