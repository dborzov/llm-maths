---
title: "Context Window: the quadratic bill behind the marketing number"
short_title: "Context Window"
description: "A million-token context window is two bills paid before the model produces a word: a linear KV-cache memory bill and a quadratic attention compute bill."
blurb:
  - "Llama-3.1 70B: 320 KB of KV cache per token — 320 GB for a 1M-token prompt."
  - "Attention FLOP count scales as O(n²), so doubling context quadruples compute."
  - "February 2024: Google announces 1M tokens; engineers who read FlashAttention mutter into their coffee."
  - "What is the gap between 'the model accepts these tokens' and 'the model can usefully attend across them'?"
topics: [transformers, long-context, attention]
tags: [kv-cache, attention, rope, flops, napkin-math]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 20
techKind: primer
techNode: context-window
header: 02-context-window.webp
---

## A Marketing Number And A Hardware Number

In **February 2024**, Google announces that **Gemini 1.5 Pro** has a {{< wiki "hyperparameters" >}}context window{{< /wiki >}} of **one million tokens**. Six weeks later they raise the headline to **10 million tokens** in a developer preview, accompanied by a video where the model finds a specific frame inside a 44-minute Buster Keaton silent film. The reception across AI Twitter is rapturous. Several VC accounts pronounce that "RAG is dead" — why bother with retrieval if you can just *paste the whole codebase*.

A small minority of engineers, the ones who have read the FlashAttention paper, mutter into their coffee. They know that the marketing number ("**1M token context window**") and the *hardware* number — what it costs in silicon to actually carry one million tokens of cached state through one decoder step — are two different things by a wide margin. They know that {{< wiki "attention" >}}attention{{< /wiki >}} is **quadratic** in sequence length, that the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} is **linear** in sequence length but with a large constant, and that the difference between "the model accepts these tokens without crashing" and "the model can usefully attend across them" can be one or two orders of magnitude.

This primer is the napkin math behind their grumbling.

## What "A Token" Becomes Inside The Model

Before any arithmetic, the picture. A modern decoder-only transformer receives a sequence of token ids, {{< wiki "embeddings" >}}embeds{{< /wiki >}} each one as a vector in $\mathbb{R}^{d_{\text{model}}}$, and then runs that sequence through $L$ identical decoder blocks. Each block does two things to the sequence: **self-attention** (every token attends to every previous token) and a **feed-forward MLP** (each token transforms in place). At the end you take the last vector, project it through the output head, and sample a next token.

For numbers throughout this primer we will anchor on a single concrete model: **Llama-3.1 70B**, because its architecture is public and representative of mid-2024 frontier scale.

| Quantity                       | Llama-3.1 70B |
|--------------------------------|--------------:|
| Parameters                     | 70 B         |
| Decoder blocks $L$             | 80           |
| Hidden width $d_{\text{model}}$| 8192         |
| Attention heads (Q)            | 64           |
| KV heads (using GQA)           | 8            |
| Head dimension                 | 128          |
| Vocabulary size                | 128 K        |
| Stated context window          | 128 K tokens  |

When the prompt is one million tokens long, every single one of those tokens is going to leave a *footprint* in the model's working memory. That footprint is the **KV cache** — the stored keys and values from every attention layer, kept around so future tokens can attend backward to them. Its size is the first thing we want to measure.

## Bill #1 — The KV Cache (Linear In Length, Punishing Constant)

In a vanilla multi-head attention block, each token $t$ produces, per layer, a key vector $K_t$ and a value vector $V_t$ that must be kept for every later token to attend to. In the [microGPT reference implementation](../../05-microgpt/13-kv-cache/), these accumulate as `keys[li]` and `values[li]` — one growing list per layer `li`. Both have to be kept around for the rest of the sequence. So the **per-token** KV cache footprint, in bytes, is:

$$
\text{bytes per token} \;=\; \underbrace{2}_{K,\,V} \;\times\; n_{\text{kv\_head}} \;\times\; \text{head\_dim} \;\times\; L \;\times\; \text{(bytes per value)}
$$

where `n_kv_head` and `head_dim` are the microGPT architecture constants (see [ch.18 Grouped-Query Attention](../../05-microgpt/18-gqa/)).{{% marginnote %}}Full-precision (FP16) `keys` and `values` occupy `2 × n_kv_head × head_dim` bytes per token per layer. KV quantization — dropping to INT8 or INT4 — is covered in depth in [Issue 03, The KV Method Family](../../03-quantization/15-kv-method-family/).{{% /marginnote %}}

For Llama-3.1 70B at {{< wiki "number-formats" >}}FP16{{< /wiki >}} (2 bytes per value), and using **Grouped-Query Attention** (GQA) with 8 KV heads of width 128:

$$
2 \times 8 \times 128 \times 80 \times 2 \;\text{bytes} \;=\; 327{,}680 \;\text{bytes} \;\approx\; 320 \;\text{KB per token}
$$

{{% pullquote type="technical" %}}
**320 KB per cached token.** A single token in a 70B model leaves a footprint the size of a small JPEG photograph in working RAM. At 1M tokens that's 320 GB — four times the weight of the model itself.
{{% /pullquote %}}

This number is the first **napkin fact** of long-context inference. Stare at it for a moment.

Now scale up:

```pyplot {id="kv-cache-vs-ctx" caption="KV-cache footprint vs context length for Llama-3.1 70B (FP16, GQA). At 1M tokens the cache is 320 GB — four times the size of the model weights themselves, and well past the memory capacity of a single H100 SXM5 (80 GB) or even four of them. The dotted horizontal line is model weights at FP16."}
# Llama-3.1 70B at FP16 with GQA
L         = 80
n_kv      = 8
d_head    = 128
bytes_val = 2  # FP16

bytes_per_token = 2 * n_kv * d_head * L * bytes_val
print(f"Bytes per cached token: {bytes_per_token:,} ({bytes_per_token/1024:.1f} KB)")

ctx = np.array([1, 4, 16, 64, 128, 256, 512, 1024, 2048]) * 1024  # tokens
gb  = ctx * bytes_per_token / 1e9

weights_gb = 70e9 * 2 / 1e9          # 70B params * 2 bytes
h100_gb    = 80                      # one H100 SXM5
node_gb    = 80 * 8                  # 8 × H100 SXM5 node

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(ctx/1024, gb, marker='o', color='#FF007F', linewidth=2.2,
        markeredgecolor='#1A1A1A', label='KV cache (FP16, GQA)')
ax.axhline(weights_gb, color='#1A1A1A', linewidth=0.7, linestyle=':',
           label=f'Model weights ({weights_gb:.0f} GB)')
ax.axhline(h100_gb, color='#00A8A8', linewidth=1.0, linestyle='--',
           label=f'1× H100 (80 GB)')
ax.axhline(node_gb, color='#FFD700', linewidth=1.0, linestyle='--',
           label=f'8× H100 node (640 GB)')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('context length (K tokens)')
ax.set_ylabel('memory (GB)')
ax.set_title('Llama-3.1 70B: the KV cache eats your GPU before the weights do')
ax.legend(loc='upper left', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, which='both', alpha=0.15)

for c in [128*1024, 1024*1024]:
    g = c * bytes_per_token / 1e9
    print(f"  ctx={c//1024}K tokens → KV cache {g:.1f} GB (={g/weights_gb:.1f}× weight footprint)")
```

Read the print-out. At the **128K** window Llama-3.1 advertises, the KV cache is **40 GB** — half a single H100. Already a serious fraction of available silicon. At the imagined **1M tokens**, the cache balloons to **320 GB** — *four times the weights themselves*, and beyond the capacity of any single GPU on the market. You need at least four H100 80GBs to hold the cache alone, plus another two to hold the model. That's a $300K-plus server for a single inference session at the headline window.

This is why "1M context" is — for many production deployments — a marketing claim built on infrastructure most users will never touch.

{{< crosshead >}}Mitigations (Names, Briefly){{< /crosshead >}}

The KV-cache bill has driven most of the architectural ferment in transformers since 2022:

- **Grouped-Query Attention (GQA)** — shrinks `n_kv_head` relative to query heads (the [`n_kv_head` / `group_size` extension](../../05-microgpt/18-gqa/) in microGPT). Llama-3 uses 8 KV heads sharing 64 query heads, an **8× shrink** vs. multi-head attention. *Already applied in our number.*
- **Multi-head Latent Attention (MLA)** — DeepSeek-V2's trick: cache a low-rank latent `c` (`kv_down` projection, see [ch.19 Multi-head Latent Attention](../../05-microgpt/19-mla/)) instead of full K and V. Saves another **4–8×**.
- **KV-cache quantization** — drop the cache from FP16 to FP8 or INT4. Issue 03's [KV Method Family Tree](../../03-quantization/15-kv-method-family/) is dedicated to this.
- **Sliding-window or sparse attention** — discard old `keys[li]` / `values[li]` beyond a fixed window `W` (the [ch.20 Sliding-Window extension](../../05-microgpt/20-sliding-window/)). Reduces cache to constant size but *also* changes what the model can attend to.

Each is a story. The high-level point for *this* primer is that **the headline "1M tokens" almost always assumes one or more of these tricks is on**, and the trick has its own accuracy footprint — which is what the rest of this issue is, in part, measuring.

## Bill #2 — Attention Compute (Quadratic In Length)

The compute bill is *worse*. To produce the attention output for token $t$, the model multiplies the query vector $Q_t$ against **every previous key** $K_1, \ldots, K_t$. That's a dot product of cost $\mathcal{O}(d_{\text{head}})$ done $t$ times, so producing one token of output costs $\mathcal{O}(t \cdot d)$. Summing over the whole sequence:

$$
\text{Attention FLOPs} \;\approx\; \sum_{t=1}^{n} 2 \cdot t \cdot d_{\text{model}}
\;\approx\; n^2 \cdot d_{\text{model}}
$$

The constant changes a bit depending on whether you count the K-projection and the value combine, but the asymptotic is **$n^2$** and that's the part that matters. Let's count it for Llama-3.1 70B across reasonable sequence lengths:

```pyplot {id="attention-flops-vs-ctx" caption="Attention FLOPs (per forward pass) for Llama-3.1 70B as context grows. The quadratic shape kicks in at long context: at 1M tokens, attention alone is ~13 EFLOPs — roughly 13 seconds on an H100 even at the theoretical peak, before any MLP work. The bottom curve is the MLP FLOPs for comparison (linear in n)."}
L          = 80
d_model    = 8192
d_head     = 128
n_q_heads  = 64
d_ff       = 4 * d_model           # typical FFN inner dim

# Total attention FLOPs across all blocks for a sequence of length n.
# Score matmul: n * n * d_model.  Value combine: n * n * d_model.  Per block.
def attn_flops(n): return 2 * L * 2 * n * n * d_model       # ~4 L n^2 d
def mlp_flops(n):  return 2 * L * 2 * n * d_model * d_ff    # MLP is linear in n
def total_flops(n): return attn_flops(n) + mlp_flops(n)

H100_FP16_TFLOPS = 1_000  # ~1 PFLOP/s at FP16/sparse FP8 advertised peak

ns = np.logspace(np.log10(1024), np.log10(2*1024*1024), 40, dtype=int)
af = np.array([attn_flops(n) for n in ns])
mf = np.array([mlp_flops(n)  for n in ns])

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.loglog(ns, af / 1e15, color='#FF007F', linewidth=2.2,
          label='Attention  (∝ n²)')
ax.loglog(ns, mf / 1e15, color='#00A8A8', linewidth=2.2,
          label='MLP        (∝ n)')
ax.loglog(ns, (af+mf) / 1e15, color='#1A1A1A', linewidth=1.0,
          linestyle=':', label='total')
ax.set_xlabel('context length (tokens)')
ax.set_ylabel('forward-pass FLOPs (PFLOPs)')
ax.set_title('Llama-3.1 70B: attention crosses MLP cost around ~64K tokens')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left')
ax.grid(True, which='both', alpha=0.15)

for n in [8_192, 128_000, 1_000_000]:
    fa, fm = attn_flops(n), mlp_flops(n)
    h100_sec_attn = fa / (H100_FP16_TFLOPS * 1e12)
    print(f"  ctx={n:>8,d}: attn={fa/1e15:7.1f} PFLOPs, "
          f"mlp={fm/1e15:7.1f} PFLOPs, "
          f"attn/mlp ratio={fa/fm:5.2f}× → "
          f"attn ≈ {h100_sec_attn:6.2f} s on 1× H100 at peak")
```

Three landmarks from the print-out:

- At **8K tokens** (the original GPT-3 era), attention compute is a *fraction* of MLP compute. The model is basically a stack of feed-forward layers with a small attention sauce on top.
- At **128K tokens** (the Llama-3.1 advertised window), attention is already on par with MLP cost. The two halves of the transformer block now consume roughly equal silicon.
- At **1M tokens**, attention is roughly **8× the MLP cost**, and the total forward pass demands roughly **15 EFLOPs** of compute — about **15 seconds on a single H100 at theoretical peak**, and well over a minute at realistic throughput. Real models batch this work and amortize it, but the underlying quadratic is what's setting the price.

This is the deep reason the field is so obsessed with **sub-quadratic** attention variants. The recent literature on Mamba, Hyena, RWKV, Linear Attention, and friends is a several-year-long search for an architecture that pays $\mathcal{O}(n \log n)$ or $\mathcal{O}(n)$ in attention compute while preserving the qualitative behaviour of {{< wiki "softmax" >}}softmax{{< /wiki >}} attention. So far, every winner is a compromise: cheaper *and* a bit worse at long-context recall in some specific way.

## The "Position" Channel: How A Transformer Even Knows Token 999,999 Is Later Than Token 1

Here is a subtle point that always trips up first-time readers. The attention operation itself is **permutation-invariant**: if you re-order the tokens, the output re-orders too, but no token can tell where it sits relative to the others. The model has to be *told* about position separately, via a **positional encoding**.

The 2017 *Attention Is All You Need* paper used a fixed sinusoidal encoding added to token embeddings. Modern open models almost universally use **Rotary Position Embeddings** (RoPE, Su et al. 2021). The trick: instead of *adding* a position vector, you *rotate* the query and key vectors by an angle that depends on position. The rotation lives inside the dot product — when you compute $Q_t \cdot K_s$, the rotations interact in a way that encodes the *relative* offset $t - s$. Beautiful, simple, and crucial for what comes next.

Crucial because RoPE has a **wavelength range**. The slowest-rotating dimensions cycle every ~10,000 positions; the fastest cycle every ~1 position. Extend the model past the longest wavelength it was trained on and the rotations alias — positions in different "octaves" become indistinguishable. **This is why every long-context paper after 2023 includes a section on "RoPE extension"** — techniques like NTK-aware scaling, YaRN, and ABF that re-stretch the wavelengths so the model can keep distinguishing positions out to 1M and beyond. None of those tricks are free; each is a small regression on recall quality in exchange for window length.

We will not derive RoPE here. The summary fact you need to carry forward: **a long context window is also a stretched position-encoding regime**, and stretched positions are part of why models that scored 99% at 32K *don't* score 99% at 1M.

## Putting It Together: The Capacity-vs-Usable Gap

Now we can answer the question the cold open promised. Why is **"1M context window"** and **"1M *usable* context window"** routinely off by a factor of 5–10×?

The marketing number is set by **what the architecture accepts without crashing** — what fits inside the KV cache, what RoPE doesn't alias, what the inference stack will accept as a valid request. The usable number is set by **a stack of separate degradations**, every one of which is empirical:

1. **Quadratic attention dilution.** At long context, each query attends to *more* keys. Softmax normalises across all of them. As $n$ grows, the attention probability spread out across $n$ targets gets noisier — the "right" key is competing with $n - 1$ distractors for a finite probability budget. We will see the dramatic shape of this curve in [The U-Curve](../07-lost-in-the-middle/).

2. **Lost-in-the-middle.** Empirically, models recall best from the **start** and the **end** of the prompt and worst from the middle. {{< wiki "liu-2023" >}}Liu et al. (2023){{< /wiki >}} documented this. We unpack it in detail in the next primer cluster.

3. **RoPE extension artefacts.** The further you extend RoPE past training, the more positions start aliasing. This produces *qualitatively new* failure modes — the model conflates positions, mixes up co-references, gets the order of events backwards.

4. **KV-cache quantization noise.** Most long-context production stacks store K and V in 4-bit, not 16-bit. Each bit reduction is a small loss of fidelity per attention probe, and those losses compound across 80 layers.

5. **Training distribution mismatch.** Most models are trained on a length distribution that *averages* around 4K–16K tokens, with very few examples at the long tail. Even when the architecture can extend, the model has barely seen what to *do* with long contexts.

Each of these effects deserves a chapter in some other book. For *this* issue, what matters is the **aggregate**: the gap between the advertised window and the empirically usable window, as measured by benchmarks like MRCR v2 and GraphWalks, is large, predictable, and the central technical fact this whole issue revolves around.

## A Tiny Sanity Check You Can Run In Your Head

{{% callout type="tip" title="The 3-Second Lab-Number Check" %}}
When a lab quotes you a context-window number, run this fast:

- **KV cache footprint:** `2 × n_kv_head × head_dim × L` bytes per token. For 70B-scale GQA at FP16 that's ~300 KB/token, or ~300 GB/million tokens.
- **Attention compute:** $4 L n^2 d$. For 70B-scale at $n = 10^6$ that's ~10 EFLOPs — several seconds per forward step at H100 peak.
- **RoPE extension:** did they train natively at that length, or are they interpolating? Interpolated RoPE degrades quality at long positions.

If any of these checks look fishy, the "M-token context window" is probably a marketing ceiling, not a usable floor.
{{% /callout %}}

When a lab quotes you a number — say, "we built a 1M-token context window" — you can do the napkin math in three seconds:

- **KV cache footprint:** roughly $L \cdot d_{\text{model}} \cdot \text{bytes-per-value}$ bytes per token. For 70B-scale models with GQA at FP16, this is **300 KB per token**, or roughly **300 GB per million tokens**. Triple-check: are they shipping the model + cache on a single 8×H100 node (640 GB total)? If not, they're using cache quantization or MLA or some other trick.
- **Attention compute:** $4 L \cdot n^2 \cdot d_{\text{model}}$. For 70B-scale at $n = 10^6$, this is on the order of **$10^{16}$ FLOPs**, or ~10 seconds on a single H100 at peak. So 1M-token inference is *seconds-to-minutes* per forward step. This is fine for batch summarization, painful for interactive chat.
- **Position encoding:** are they using RoPE? If so, how far past training was the model extended? An 8B-token-trained model claiming 1M context is *certainly* doing aggressive RoPE extension. A 2T-token-trained model with native 1M training is doing far less.

These three back-of-envelopes will tell you, within a factor of two, what the underlying capacity *should* feel like. Then a benchmark like [MRCR v2 or GraphWalks](../09-latent-structure/) tells you what it *actually* feels like.

## What To Remember

1. **A "1M token context window" is two numbers.** The marketing number (what the API accepts) and the hardware number (what fits in memory + how long attention takes). The two are often an order of magnitude apart in usable behaviour.
2. **The KV cache is the dominant memory cost** at long context. Per-token footprint for a 70B-scale model is ~300 KB — so 1M tokens is **300 GB**, four times the size of the weights themselves. GQA, MLA, and KV quantization exist to compress this.
3. **Attention compute is quadratic.** It catches up to the MLP cost somewhere between 32K and 128K tokens, and dominates everything beyond that. Sub-quadratic alternatives are an active area, all with accuracy trade-offs.
4. **Position encoding is a separate channel** with its own scaling rules. RoPE-extension tricks let you stretch the window post-training, but the further you stretch, the more positions begin to alias.
5. **The "usable" window is a stack of degradations** — dilution, lost-in-the-middle, RoPE artefacts, cache quantization noise, training mismatch — that compound in non-obvious ways. Every benchmark in this issue is, in some sense, a different lens on a different layer of this stack.

**Continue to** → [Needle in a Haystack](../03-niah-mechanics/) — the actual mechanics of NIAH, why the visualization was so persuasive, and the December 2023 Anthropic blog post that should have unwound the whole benchmark a year earlier than it did.
