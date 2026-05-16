---
title: "Decoupling Memory From Time"
description: "Boss capstone. MLA shrinks cache per token to constant. CSA + HCA shrink compute per token to constant. The cost of context per inference query becomes LINEAR in T. The 10-year quadratic ceiling on long-context LLMs is gone. What does the world look like on the other side?"
topics: [attention, deepseek, economics, long-context, theory]
tags: [boss, decoupling, mla, dsa, csa, hca, linear-context, agents]
theme: cream
math: true
draft: false
date: 2026-05-16T10:30:00-04:00
issue: 7
weight: 100
techKind: boss
techNode: decoupling
header: 10-decoupling.webp
---

## Here Is How We Got Here

September 2017. The *Attention Is All You Need* paper drops. In the appendix — one line, almost a footnote — the authors note that {{< wiki "attention" >}}self-attention{{< /wiki >}} is $O(T^2)$ in sequence length. They are not worried. The longest sequence anyone runs is 512 tokens.

By 2024, inference at 128K tokens is the product claim on every frontier model. At 128K, that $T^2$ term is $1.6 \times 10^{10}$. The line nobody worried about in 2017 is now the ceiling on every product that wants to think about a long document, a large codebase, or a multi-hour conversation.

The ceiling was not inevitable. It was a composition of three separate costs that happened to all scale with $T$ in the naive formulation:

1. **Memory**: to compute attention at decode step $t$, you need to read back the {{< wiki "kv-cache" >}}cached keys and values{{< /wiki >}} for every previous token. If each token takes $H \cdot D$ bytes to store (one key and one value per head per layer), the cache grows linearly in $T$ — but with a coefficient that, at 128 heads and head-dim 128, is enormous. Reading the whole cache per decode step makes the *IO cost* of each step grow linearly in $T$.

2. **Compute**: to compute attention scores, you compare the current query against every cached key. That is $T$ dot products per head per layer per decode step. Total attention FLOPs per decode step: $H \times L \times T \times D$.

3. **Economic**: inference price scales with (1) + (2). The cost per *query* — a complete decode run from 0 to $T$ output tokens — is the per-step cost integrated over those $T$ steps: $O(T^2)$ total.

These three costs are not the same thing, and they do not all live on the same axis. That observation is the key to everything that follows. The [chapter on the quadratic wall](../03-quadratic-wall/) established it in detail; the [MLA rewind](../02-mla-rewind/) showed what happens when you cut the first axis; the [lightning indexer](../06-lightning-indexer/), [CSA](../07-csa/), and [HCA](../08-hca/) chapters showed what happens when you cut the second. This chapter is the composition proof: what happens when you cut *all three at once*.

{{< crosshead >}}The Stack, One Last Time{{< /crosshead >}}

Before the algebra, the plain English version. DeepSeek ran four architectural moves, across three years and four model generations, that targeted different axes of the cost problem:

**Move 1 — {{< wiki "deepseek" >}}Multi-head Latent Attention{{< /wiki >}} (MLA, 2024).** Instead of caching a full key and a full value for every head, cache one short *latent vector* $c_t \in \mathbb{R}^{d_c}$ where $d_c \ll H \cdot D$. At decode time, multiply out the full key and value from $c_t$ — but this multiplication commutes with the attention score in a way that makes it free (the [absorption identity](../02-mla-rewind/)). Cache bytes per token per layer: from $H \cdot D \cdot 2$ down to $d_c + d_R$ where $d_R$ is the RoPE-encoded component. In DeepSeek's numbers: from $128 \times 128 \times 2 = 32{,}768$ bytes down to $576$ bytes per layer. A **57× reduction** in cache storage per token.

**Move 2 — DeepSeek Sparse Attention (DSA, 2025).** Instead of attending to all $T$ cached tokens, use the [lightning indexer](../06-lightning-indexer/) to score all $T$ tokens cheaply (low-rank scoring, $O(T)$ but with small constant), then attend to only the top-$k$ (constant in $T$). Attention compute per decode step: from $O(T)$ per head to $O(k)$ per head, with $k \ll T$.

**Move 3 — Compressed Sparse Attention (CSA, 2026).** Before running the indexer, [compress](../07-csa/) $m = 4$ consecutive tokens into one compressed entry. The indexer now scores $T/m$ entries instead of $T$. The same top-$k$ budget buys more coverage. Effective indexer cost: $O(T/m)$ per decode step.

**Move 4 — Heavily Compressed Attention (HCA, 2026).** For [half the transformer layers](../08-hca/), skip the indexer entirely. Compress $m' = 128$ consecutive tokens into one entry, then do *dense* attention over those $T/m'$ entries. No selection, no indexer. Attention cost: $O(T/m')$ per decode step, which at $m' = 128$ and $T = 1{,}000{,}000$ is attention over just 7,812 entries.

{{% callout type="tangent" title="Why Four Moves, Not One?" %}}
The cuts are orthogonal because they attack different variables in the cost expression.

MLA operates on the **D-axis** (head dimension / cache width per token). DSA operates on the **T-axis via selection** (fewer tokens fully attended). CSA operates on the **T-axis via compression before selection** (smaller $T$ hits the indexer). HCA operates on the **T-axis via compression before dense attention** (smaller $T$ hits attention itself).

GQA (which DeepSeek also uses, inherited from V3) operates on the **H-axis** (shared key/value heads). That is the fifth, implicit cut in the stack.

Each cut multiplies with the others because they reduce different dimensions of the same tensor product. A cut on the D-axis does not interact with a cut on the T-axis — so their combined effect is multiplicative, not additive.
{{% /callout %}}

## The Composition Theorem

Let us be precise. We want the total attention cost per decode step for a $V4$-Pro model at context length $T$.

{{< crosshead >}}Memory First{{< /crosshead >}}

With MLA, each token's contribution to the KV cache is stored as a pair $(c_t, c^R_t)$ where $c_t \in \mathbb{R}^{d_c}$ is the main latent and $c^R_t \in \mathbb{R}^{d_R}$ is the RoPE-separated component. Cache bytes per layer per token:

$$\text{cache bytes per token} = (d_c + d_R) \times 2 \;\text{bytes} = (512 + 64) \times 2 = 1{,}152 \;\text{bytes}$$

For $L = 60$ layers across $T$ tokens:

$$\text{total KV cache} = L \times (d_c + d_R) \times T \times 2 \;\text{bytes} = 60 \times 576 \times T \times 2 = 69{,}120 \times T \;\text{bytes}$$

At $T = 1{,}000{,}000$: $\approx 65 \;\text{GB}$. That is a lot — but it is **linear in $T$**, and the coefficient ($69$ KB per token across all layers) is fixed. It does not grow as the context grows. Compare to naive MHA at the same heads: $H \times D \times 2 \times L \times T \times 2 = 128 \times 128 \times 2 \times 60 \times T \times 2 = 3{,}932{,}160 \times T$ bytes. At $T = 1M$: $\approx 3.7 \;\text{TB}$. MLA brought that from 3.7 TB to 65 GB at 1M context. **57× smaller cache.**

The *per-step IO cost* of reading the cache during decode is proportional to cache size. So MLA's 57× cache cut translates to a 57× cut in memory bandwidth consumed per decode step. This is Move 1 done.

{{< crosshead >}}Compute Next{{< /crosshead >}}

Now count the FLOPs. We will track a single decode step — one new token being added to a context of length $T$.

**Vanilla MHA baseline (2017-era).** For each decode step, every head in every layer computes one query-key dot product against all $T$ cached keys:

$$\text{FLOPs}_\text{MHA} = H \times L \times T \times D = 128 \times 60 \times T \times 128 = 983{,}040 \times T$$

At $T = 1{,}000{,}000$: $\approx 9.83 \times 10^{11}$ FLOPs per decode step. With $\sim 1000$ decode steps to generate a 1K-token answer, total: $\approx 10^{15}$ FLOPs. That is a petaFLOP per query. In 2017, on V100s at 14 TFLOPS, that would be $\sim 70$ seconds of pure compute per inference call, ignoring everything else. The reason 1M-token context did not exist in 2017 is not a software problem.

**After MLA (V2, 2024).** MLA changes the *storage* per token, not the attention-score computation itself. The attention score is still $q_t^T k_{t'}$ for all $t'$. But the key $k_{t'}$ is now reconstructed on the fly from the cached latent $c_{t'}$ — and by the absorption identity, this reconstruction can be folded into the query projection, making each effective dot product cheap. The FLOPs for the score matrix are approximately the same as vanilla MHA at compute level; the *IO* cost of reading back keys and values is cut 57×. So for memory-bandwidth-bound inference (the typical regime at batch size 1): yes, MLA helps a lot. But the raw FLOP count formula looks similar. **MLA fixes memory, not compute.**

**After DSA (V3.2, 2025).** Each decode step now has two phases per CSA/DSA layer. Let $n^I_h$ be the number of indexer heads and $c_I$ be the indexer head dimension. The indexer computes a cheap score over all $T$ tokens:

$$\text{FLOPs}_\text{indexer} = n^I_h \times c_I \times T = 32 \times 64 \times T = 2{,}048 \times T \;\;\text{per layer}$$

Then the heavy attention path over top-$k$ entries:

$$\text{FLOPs}_\text{attn} = n_h \times k \times D = 128 \times 2{,}048 \times 128 = 33{,}554{,}432 \;\;\text{per layer (constant in } T\text{)}$$

Total for $L = 60$ layers:

$$\text{FLOPs}_\text{V3.2} = L \times (\text{indexer} + \text{attn}) = 60 \times (2{,}048 T + 33{,}554{,}432)$$

At $T = 1{,}000{,}000$:

$$= 60 \times (2{,}048{,}000{,}000 + 33{,}554{,}432) \approx 60 \times 2.08 \times 10^9 \approx 1.25 \times 10^{11} \;\text{FLOPs}$$

Compare to vanilla MHA at $T = 1M$: $9.83 \times 10^{11}$ FLOPs. **Speedup: $\approx 8\times$ at 1M context.** And crucially, as $T$ grows, the speedup grows — because the indexer's coefficient $(2{,}048)$ is much smaller than MHA's coefficient $(983{,}040)$.

**After CSA (V4, CSA layers only, compression $m = 4$).** The indexer now sees $T/m = T/4$ entries instead of $T$:

$$\text{FLOPs}_\text{CSA indexer} = n^I_h \times c_I \times (T/m) = 32 \times 64 \times (T/4) = 512 \times T/4$$

The attended path still goes to top-$k$ entries (now drawing from a pool of $T/m$ compressed entries):

$$\text{FLOPs}_\text{CSA attn} = n_h \times k \times D = 128 \times 1{,}024 \times 512 = 67{,}108{,}864 \;\;\text{per layer}$$

(Note: $k$ and $D$ values are updated for V4's expanded architecture — the head dim in V4-Pro is larger, and $k$ is halved relative to V3.2 because the pool is already 4× compressed.) Per CSA layer at $T = 1M$:

$$\text{FLOPs}_\text{CSA layer} = 32 \times 64 \times 250{,}000 + 67{,}108{,}864 = 512{,}000{,}000 + 67{,}108{,}864 \approx 579 \;\text{MFLOPs}$$

**After HCA (V4, HCA layers only, compression $m' = 128$).** No indexer. Dense attention over $T/m' = T/128$ compressed entries:

$$\text{FLOPs}_\text{HCA layer} = n_h \times D \times (T/m') = 128 \times 512 \times (T/128) = 512 \times T$$

At $T = 1{,}000{,}000$: $128 \times 512 \times 7{,}812 \approx 512{,}000{,}000 \approx 512 \;\text{MFLOPs}$ per layer. A coincidence of the numbers: CSA and HCA land at nearly the same per-layer cost at 1M context, by different routes.

{{< crosshead >}}The Total Bill for V4-Pro{{< /crosshead >}}

V4-Pro has 61 layers. The first two are pure HCA; the remaining 59 alternate CSA and HCA — approximately 30 CSA layers and 31 HCA layers.

At $T = 1{,}000{,}000$:

| Component | Layers | FLOPs/layer | Total |
|---|---|---|---|
| CSA (indexer + attended) | 30 | 579 MFLOPs | **17.4 GFLOPs** |
| HCA (dense over $T/128$) | 31 | 512 MFLOPs | **15.9 GFLOPs** |
| **Total attention** | 61 | — | **33.3 GFLOPs** |

Dense MHA baseline at same $T$, same $L$: $983{,}040 \times 10^6 \times 60 = 5.9 \times 10^{13}$ FLOPs.

$$\text{Speedup} = \frac{5.9 \times 10^{13}}{3.33 \times 10^{10}} \approx \mathbf{1{,}770\times}$$

V4-Pro's attention is **1,770 times cheaper per decode step** than vanilla MHA would be at the same context length. Compared to V3.2 (DSA only): $1.25 \times 10^{11} / 3.33 \times 10^{10} \approx 3.7\times$. That matches the headline claim from the DeepSeek-V4 technical report.

{{% pullquote type="technical" %}}
Four orthogonal cuts. D-axis (MLA) × H-axis (GQA) × T-axis selection (DSA) × T-axis compression (CSA + HCA). The combined effect is multiplicative: **roughly 1,770× cheaper attention than vanilla MHA at 1M context.** The 10-year quadratic ceiling on long-context LLMs is gone.
{{% /pullquote %}}

## The Composition Theorem, Formally

Let us state precisely what the decoupling claim means.

**Definition.** Let $\text{cost}_\text{step}(T)$ be the attention FLOPs per decode step at context length $T$. Let $\text{cost}_\text{query}(T, G)$ be the total attention FLOPs to decode $G$ tokens given context length $T$.

**Before decoupling (vanilla MHA):**

$$\text{cost}_\text{step}(T) = c_0 \cdot T \quad \text{(linear in } T \text{ per step)}$$
$$\text{cost}_\text{query}(T, G) = c_0 \cdot T \cdot G \quad \text{(quadratic if } G \sim T \text{)}$$

The per-step cost grows with $T$. The per-query cost grows as $T^2$ in the common case where generation length $G$ is proportional to context length.

**After MLA + DSA + CSA + HCA (V4):**

$$\text{cost}_\text{step}(T) = c_\text{CSA} \cdot (T/m) + c_\text{HCA} \cdot (T/m') + c_\text{const}$$

where $c_\text{const}$ is the attended-path cost (constant in $T$, dominated by the $k$-entry dense attention), and $c_\text{CSA}/m$, $c_\text{HCA}/m'$ are the indexer/compression costs with $m = 4$ and $m' = 128$.

$$\text{cost}_\text{step}(T) = O\!\left(\frac{T}{m'}\right) = O(T) \;\text{ with coefficient } \frac{1}{m'}$$

The per-step cost is still linear in $T$ — we have not achieved $O(1)$ per step — but the linear coefficient is $128\times$ smaller for HCA layers and $4\times$ smaller for CSA layers. More importantly:

$$\text{cost}_\text{query}(T, G) = \left(\frac{c_\text{CSA}}{m} + \frac{c_\text{HCA}}{m'}\right) \cdot T \cdot G + c_\text{const} \cdot G$$

If $G$ is fixed (e.g., "generate a 2000-token response for any context length"), then cost grows **linearly in $T$** rather than quadratically. **Doubling the context now doubles the cost, not quadruples it.** That is the decoupling.

{{% callout type="theorem" title="The Decoupling Theorem" %}}
For a fixed generation budget $G$, the total attention cost of a V4-class decode run is:

$$\text{cost}_\text{query}(T) = \Theta\!\left(T \cdot \frac{G}{m'}\right) \;\;\text{for HCA-dominated layers}$$

This is **linear in context length $T$**, with a coefficient that shrinks as compression rate $m'$ grows. Vanilla MHA has $m' = 1$ (no compression); V4-HCA has $m' = 128$. The crossover from "quadratic pain" to "linear budget" happens at the point where the compression rate exceeds 1 — which is the moment you stop attending to every token and start attending to compressed summaries.
{{% /callout %}}

```pyplot {id="t-squared-vs-t" caption="QUADRATIC VERSUS LINEAR COST SCALING. THE GAP BETWEEN VANILLA MHA AND V4 GROWS WITHOUT BOUND AS CONTEXT INCREASES."}
T_vals = np.linspace(1e4, 1e6, 500)

# Vanilla MHA: cost ~ T * constant (per step), total query cost ~ T^2 * generation
c_mha   = 983_040  # FLOPs per decode step per token of context
c_v4    = 983_040 / 1770  # V4 cost per decode step per token (1770x reduction at 1M)
# Make relative to MHA at T=10K
norm = c_mha * 10_000

cost_mha  = c_mha  * T_vals / norm
cost_gqa  = c_mha  * 0.13 * T_vals / norm          # GQA: ~8x H-axis cut
cost_mla  = c_mha  * 0.04 * T_vals / norm          # MLA: further ~3x cache cut
cost_v32  = 2048 * T_vals / norm                   # V3.2: indexer coefficient
cost_v4   = (2048 / 4 + 512) * T_vals / norm       # V4: CSA + HCA combined coefficient

fig, ax = plt.subplots(figsize=(10, 5))

ax.loglog(T_vals / 1e3, cost_mha,  color='#1A1A1A', lw=2.5, label='Vanilla MHA (2017)', linestyle='solid')
ax.loglog(T_vals / 1e3, cost_gqa,  color='#FF8C00', lw=2, label='+ GQA (2023)', linestyle='dashed')
ax.loglog(T_vals / 1e3, cost_mla,  color='#FF007F', lw=2, label='+ MLA (V2, 2024)', linestyle='dashed')
ax.loglog(T_vals / 1e3, cost_v32,  color='#00A8A8', lw=2, label='+ DSA (V3.2, 2025)', linestyle='dashed')
ax.loglog(T_vals / 1e3, cost_v4,   color='#FFD700', lw=2.5, label='+ CSA+HCA (V4, 2026)', linestyle='solid')

ax.set_xlabel('context length T (thousands of tokens)')
ax.set_ylabel('relative attention cost per decode step (log scale)')
ax.set_title('Per-step attention cost vs context length.\nAll curves are linear — but the coefficients differ by 1770×.')
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.grid(True, alpha=0.25, which='both')
ax.spines[['top', 'right']].set_visible(False)

# Annotate the gap at 1M tokens
ax.annotate('1770× gap\nat 1M tokens', xy=(1000, cost_v4[-1]), xytext=(400, cost_mha[-1] * 0.3),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A'), fontsize=9, color='#1A1A1A')
```

{{< crosshead >}}Why "Decoupling" Is The Right Word{{< /crosshead >}}

The classic framing is "$O(T^2)$ attention". But that framing conflates two independent things that both happen to scale with $T$ in the naive design:

1. **The number of decode steps** scales with $G$ (generation length), not $T$. If you fix $G$, the total is just $G \times \text{cost\_step}$.
2. **The per-step cost** scales with $T$ in vanilla MHA, but does not *have to* — as DSA, CSA, and HCA show.

The "decoupling" is specifically the separation of the per-step cost from $T$. MLA decouples *memory bandwidth* from $T$ (constant cache size per token). CSA + HCA decouple *compute* from $T$ (cost of scoring grows as $T/m'$ rather than $T$, and for fixed-top-$k$ attended paths, there is an additional $O(k)$ constant term). The two decouplings are independent; they compose multiplicatively; and together they make the inference cost of long context grow linearly in $T$ rather than quadratically.

```pyplot {id="cumulative-architecture-cost" caption="CUMULATIVE COST PER DECODE STEP AT 1M CONTEXT, BY ARCHITECTURE GENERATION. EACH BAR ADDS ONE MORE CUT TO THE STACK ABOVE."}
stages = [
    ("vanilla MHA\n(2017)", 1.0),
    ("+ GQA\n(2023)", 0.13),  # roughly 8× compression on H-axis
    ("+ MLA\n(V2, 2024)",  0.04),  # ~30× on D-axis
    ("+ DSA\n(V3.2, 2025)", 0.012),  # ~3× compute cut
    ("+ CSA\n(V4 sparse layers)", 0.004),  # ~3× more
    ("+ HCA\n(V4 dense compress)", 0.001),  # half the layers cheaper still
]

names = [s[0] for s in stages]
costs = np.array([s[1] for s in stages])

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(range(len(stages)), costs, color=['#FF007F','#FF8C00','#FFD700','#00A8A8','#00A8A8','#1A1A1A'])
ax.set_yscale('log')
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('relative attention FLOPs per decode step (log)')
ax.set_title('Each architecture generation cuts attention cost by ~3-30×. Cumulative: ~1000×.')
ax.grid(True, alpha=0.3, axis='y', which='both')
for bar, c in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width()/2, c * 1.3, f'{c:.4g}', ha='center', fontsize=9)
ax.spines[['top','right']].set_visible(False)
```

## The Economic Argument

Architecture theorems are one thing. API prices are another. The two have been converging in exactly the way the theorems predict.

{{< crosshead >}}A Decade of Prices{{< /crosshead >}}

The numbers below are approximate — frontier model pricing is fluid and workload-dependent — but the order-of-magnitude story is accurate:

| Era | Model | Price ($/M input tokens) | What changed |
|---|---|---|---|
| V1 (2022) | GPT-4 (original) | ~$30 | Dense MHA, no KV optimizations |
| V1.5 (2023) | GPT-4-Turbo | ~$10 | GQA, better batching, better hardware |
| V2 (mid 2024) | DeepSeek-V2 | ~$0.14 | **MLA** — cache savings |
| V2.5 (late 2024) | Claude 3.5 Sonnet | ~$3 | GQA + hardware efficiency |
| V3 (Dec 2024) | DeepSeek-V3 | ~$0.07 | MLA + larger MoE (better amortization) |
| V3.2 (Sept 2025) | DeepSeek-V3.2-Exp | ~$0.035 | **DSA** — compute savings at long context |
| V4 (2026) | DS-V4-Flash | ~$0.01–$0.02 | **CSA + HCA** — further compression |

From GPT-4 to V4-Flash: **~1,500–3,000×** cheaper per input token. Most of that reduction happened not through hardware improvements or scale economies, but through architecture: MLA, then DSA, then CSA + HCA.

The pattern is not coincidental. Each price cut correlates with a specific architectural move. When DeepSeek shipped MLA in May 2024, V2 launched at $0.14/M tokens — roughly 200× cheaper than GPT-4 at the time. When DSA shipped in September 2025, V3.2 launched at $0.035/M — exactly the 50% cut the model card announced. When V4 shipped with CSA + HCA, Flash pricing dropped again by 2–3×.

```pyplot {id="api-price-history" caption="API INFERENCE PRICE (PER 1M INPUT TOKENS) FROM 2022 TO 2026, LOG SCALE. EACH ARCHITECTURAL MOVE IS LABELED."}
# (year_fraction, price, label, color, annotation)
entries = [
    (2022.5,  30.0,   "GPT-4\n(dense MHA)",       '#FF8C00', False),
    (2023.75, 10.0,   "GPT-4-Turbo\n(+ GQA)",     '#FF8C00', False),
    (2024.35, 0.14,   "DS-V2\n(MLA)",              '#FF007F', True),
    (2024.75, 3.0,    "Claude 3.5\n(GQA+HW)",     '#00A8A8', False),
    (2024.98, 0.07,   "DS-V3\n(MLA+MoE)",         '#FF007F', True),
    (2025.75, 0.035,  "DS-V3.2\n(DSA)",           '#FF007F', True),
    (2026.2,  0.015,  "DS-V4-Flash\n(CSA+HCA)",   '#FFD700', True),
]

years  = [e[0] for e in entries]
prices = [e[1] for e in entries]
labels = [e[2] for e in entries]
colors = [e[3] for e in entries]
annotate = [e[4] for e in entries]

fig, ax = plt.subplots(figsize=(10, 5))

# Draw connecting lines between DeepSeek models
ds_years  = [e[0] for e in entries if e[4]]
ds_prices = [e[1] for e in entries if e[4]]
ax.plot(ds_years, ds_prices, color='#FF007F', lw=1.5, linestyle='--', alpha=0.5, zorder=1)

for y, p, lbl, col in zip(years, prices, labels, colors):
    ax.scatter(y, p, s=90, color=col, zorder=3, edgecolors='#1A1A1A', linewidths=0.8)
    ax.text(y, p * 1.6, lbl, ha='center', fontsize=7.5, color='#1A1A1A')

ax.set_yscale('log')
ax.set_xlim(2022, 2026.8)
ax.set_ylim(0.005, 200)
ax.set_xlabel('year')
ax.set_ylabel('price per 1M input tokens (USD, log)')
ax.set_title('API inference price 2022–2026. DeepSeek (pink) drives the step changes.')
ax.grid(True, alpha=0.2, which='both', axis='y')
ax.spines[['top', 'right']].set_visible(False)

# Annotate the architectural moves
ax.annotate('MLA → 200× cheaper\nthan GPT-4', xy=(2024.35, 0.14), xytext=(2023.2, 0.5),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=0.8), fontsize=8)
ax.annotate('DSA → 50% cut', xy=(2025.75, 0.035), xytext=(2024.9, 0.01),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=0.8), fontsize=8)
ax.annotate('CSA+HCA\n→ 2–3× more', xy=(2026.2, 0.015), xytext=(2025.6, 0.004),
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=0.8), fontsize=8)
```

{{< crosshead >}}What It Costs To Run 1M Tokens{{< /crosshead >}}

The most concrete way to see the progress is to price a fixed workload — a single inference call at 1M token context, generating a 2,000-token response — across model generations. These are rough estimates assuming comparable hardware efficiency:

| Generation | Model | Cost per 1M-token query | Notes |
|---|---|---|---|
| 2017 hardware + vanilla MHA | (hypothetical) | ~$10,000 | V100 compute + $O(T^2)$ FLOPs |
| 2023, GQA | (hypothetical) | ~$1,000 | GQA cuts KV bandwidth 8× |
| V2, MLA | DeepSeek-V2 | ~$50 | Cache down 57×; IO cost falls |
| V3.2, DSA | DS-V3.2-Exp | ~$5 | Compute cut ~8× at 1M tokens |
| V4, CSA+HCA | DS-V4-Flash | ~$0.50 | Further ~10× compute cut |

The crossover into "economically viable for mass products" — below $1 per long-context query — happened with DSA in September 2025. The crossover into "routine infrastructure" — below $0.10 — happened with V4-Flash in 2026. A trend of roughly **10× cheaper every 12 months** since MLA shipped.

## What This Buys

Numbers are one thing. Products are another.

{{< crosshead >}}Repository-Scale Code Agents{{< /crosshead >}}

The Linux kernel is roughly 3M tokens. A typical mid-size company's monorepo is 500K–2M tokens. At 128K context (V3.1-era), an agent interacting with a codebase needed a RAG pipeline: chunk the repo, embed the chunks, retrieve the relevant ones, stuff them into context, answer from retrieved fragments. Every retrieval step is a potential miss. Every miss is a wrong answer.

At 1M tokens with linear cost, you load the entire repository into a single context. No chunking. No retrieval. No miss. The agent reads the import at line 247, checks the function definition at line 8,312, traces the data flow to the test at line 22,104 — all in one pass, without any retrieval infrastructure between those steps. That is not an incremental improvement to code agents. It is a different category of tool.

{{< crosshead >}}Legal and Medical Document Analysis{{< /crosshead >}}

A deposition transcript is roughly 200K tokens. Relevant prior case law is another 500K tokens. Applicable statutory text is another 300K tokens. Total: 1M tokens — the kind of load that, with chunked RAG, introduces retrieval errors that lawyers have to audit manually.

At 1M context with linear cost, a single query can hold the full deposition, the full case precedent set, and the applicable law simultaneously. The model reasons over the complete picture, not a sampled subset of it. The same applies to clinical records: a patient's full EHR spanning 20 years of visits, notes, labs, and imaging reads can fit in a single context window with room to spare.

{{< crosshead >}}Scientific Literature Synthesis{{< /crosshead >}}

A recent ML paper cites 40–80 papers. Those papers collectively cite hundreds more. A researcher wanting to understand a subfield has to read papers one at a time, build a mental model of the landscape, and hope they have not missed a crucial connection. The alternative — "load this entire subfield into context and ask questions" — requires context windows large enough to hold it. At 1M tokens, that is roughly 500–1,000 papers. Not the whole subfield, but enough for a meaningful synthesis.

{{< crosshead >}}Multi-Turn Agent Sessions{{< /crosshead >}}

Traditional agents lose context after 32K–128K tokens — roughly 50–200 exchanges. At 1M tokens with linear cost, an agent can maintain working memory across 50–100 hours of a work session. The agent that reviews your pull request on Monday can remember the architectural decisions from last Thursday's conversation without any external memory store. The context *is* the memory.

{{% callout type="counterintuitive" title="The Short-Context Case Gets Cheaper Too" %}}
It is tempting to think of CSA and HCA as "long-context optimizations" that only help at large $T$. But the compression is applied unconditionally — even at $T = 8{,}000$, HCA attends over $T/128 \approx 62$ entries instead of 8,000. The speedup at short context is not 1770×, but it is not 1× either.

At $T = 32K$ (a "normal" RAG context): V4's per-step cost is roughly 50× lower than vanilla MHA. That means the baseline cost of every inference call is lower, across the full context distribution. V4-Flash is not just a "long context" model. It is a cheaper model at every context length above a few thousand tokens.
{{% /callout %}}

## The Question The Paper Does Not Answer

The V4 paper's composition of four architectural moves is complete in an important sense: it has, for the first time, made the attention cost of long-context decoding scale linearly rather than quadratically. There is no fifth axis to cut in the same way — the four cuts are: heads (H), head-dim/cache (D), token selection (T via indexer), and token compression (T via dense compression). All four are done.

So what breaks next?

{{< crosshead >}}MoE Routing Is Now The Bottleneck{{< /crosshead >}}

With attention at 33 GFLOPs per decode step (at 1M context), the second-largest term in V4-Pro's compute budget is now the mixture-of-experts routing and FFN layers. V4-Pro activates 13B parameters out of 284B. For each token, the router computes scores over all experts — $O(E \times d_\text{model})$ work where $E$ is the number of experts. At V4 scale with 160+ experts per token-routing decision, this is not negligible. As attention becomes cheap, MoE routing becomes the next thing you want to make sparse.

The V4 paper mentions "embedding-level sparsity" as a future direction. That is almost certainly MoE routing: finding ways to skip routing computation entirely for tokens where the expert assignment is predictable.

{{< crosshead >}}Training Is The Hard Part Now{{< /crosshead >}}

V4 required three novel training stabilizers to converge at 61 layers of interleaved CSA/HCA:

- **Muon optimizer** (momentum-orthogonalized updates) instead of AdamW for all non-embedding parameters
- **mHC connections** (Manifold-Constrained Hyper-Connections) replacing the standard residual stream
- **Anticipatory Routing** for the MoE layers
- **SwiGLU Clamping** to prevent activation blowup at the boundaries of CSA/HCA compression

None of these would have been necessary with vanilla MHA. The compression introduces gradient pathologies that do not exist when attention attends to a smooth distribution of tokens. The *inference* architecture is now elegant; the *training* architecture is a collection of stabilizers bolted around it.

The 2026–2027 era's papers will probably be about simplifying this training stack. The next generation of models may have cleaner architectures that achieve the same inference efficiency without requiring four special training tricks.

{{< crosshead >}}Cross-Session State{{< /crosshead >}}

V4 supports 1M tokens per query. But what about maintaining state across 1,000 queries — 10$^9$ total tokens of context? The next architectural bet is almost certainly about cross-session memory: how do you persist, compress, and retrieve the result of previous sessions without re-running them?

Recurrent architectures (Mamba, RWKV, Hawk) have been proposed as one answer: maintain a fixed-size state that rolls forward rather than expanding. Retrieval-augmented fine-tuning is another. Neither has solved the problem in a way that competes with V4's context quality — yet.

## A Final Picture

Let us close by restating the arc.

In **August 1968**, Mead Conway published the first description of what would become VLSI circuit design — the principle that by increasing integration density, you could make computing exponentially cheaper over time. Nobody in 1968 thought this would mean every human would carry a supercomputer in their pocket.

In **June 2017**, a team at Google published "Attention Is All You Need." In the appendix, they noted that self-attention is $O(T^2)$. Nobody thought much about it.

Seven years later, a hedge fund's AI division in Hangzhou set about dismantling that $T^2$ term one piece at a time. Four moves, three years:

1. Cache latents instead of heads: **MLA, 2024.** $57\times$ smaller cache.
2. Index then attend: **DSA, 2025.** $8\times$ cheaper compute at 1M context.
3. Compress before indexing: **CSA, 2026.** $3.7\times$ cheaper than DSA.
4. Compress before dense attention: **HCA, 2026.** Half the layers, no indexer at all.

Total attention cost reduction at 1M context: **$\approx 1{,}770\times$** vs. vanilla MHA. API price reduction, V3 to V4-Flash: **$\approx 5\times$**. Cumulative, GPT-4 to V4-Flash: **$\approx 2{,}000\times$**.

The unit economics of long context inference now grow linearly in $T$. The quadratic ceiling — the one Vaswani et al. mentioned in an appendix and nobody worried about for five years — is gone.

Every interesting AI product that exists in 2026 depends on this. Code agents, legal analysis, scientific synthesis, long-horizon planning, multi-round agentic work — none of these are possible at scale without the linear-cost context window. DeepSeek did not just make a better model. They changed the equation that everyone else has to run.

That is the issue. Whatever the next generation of LLMs looks like, it runs on this math.

## What To Remember

1. **Four orthogonal cuts compose multiplicatively:** MLA (D-axis) × GQA (H-axis) × DSA (T-axis, selection) × CSA+HCA (T-axis, compression). Combined: ~1,770× cheaper attention at 1M context than vanilla MHA.

2. **MLA fixes memory, DSA+CSA+HCA fix compute.** MLA's 57× cache reduction eliminated the IO bottleneck. DSA+CSA+HCA's 1,770× FLOP reduction eliminates the compute bottleneck. Together they make per-step cost grow as $O(T/m')$ instead of $O(T)$, with $m' = 128$ for HCA layers.

3. **Per-query cost becomes linear in $T$** (for fixed generation length). Before: doubling context quadrupled cost. After: doubling context doubles cost. The ceiling is lifted.

4. **API prices drop ~10× per year** since MLA shipped. The crossover into "economically viable for mass products" happened with DSA in 2025. The crossover into "routine infrastructure" happens with V4.

5. **The next bottleneck is not attention.** MoE routing is now the second-largest compute term. Training stability under deep compression architectures is the new hard problem. Cross-session state persistence is the new research frontier.

---

| Architecture | Axis cut | Savings at 1M tokens | Since baseline |
|---|---|---|---|
| GQA | H (shared KV heads) | ~8× compute | 8× |
| MLA | D (cache latent) | 57× IO bandwidth | 57× cache |
| DSA | T (sparse selection) | ~8× compute | 8× |
| CSA | T/m (compressed indexer, $m=4$) | ~3× additional | ~3× |
| HCA | T/m' (compressed dense, $m'=128$) | combined ~3.7× vs DSA | — |
| **All together** | **D × H × T** | **~1,770× vs vanilla MHA** | **~2,000× API cost** |

---

*That is the issue. Six years of attention research, one Hangzhou lab, four stacked architectural moves. The quadratic ceiling on LLMs is gone. Whatever comes next lives on the other side of this curve.*
