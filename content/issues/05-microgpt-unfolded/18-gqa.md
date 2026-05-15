---
title: "Grouped-Query Attention"
description: "The H-axis fix that became universal. Share K/V tensors across head groups; pick a group size; ship it. Llama 3 picked 4×. Qwen3-235B picked 16×. The architectural change is one integer divide inside microGPT's multi-head loop."
topics: [transformer, attention, kv-cache]
tags: [microgpt, gqa, mqa, llama3, qwen3, glm]
theme: teal
math: true
draft: false
date: 2026-05-14T02:57:00-04:00
issue: 5
weight: 180
techKind: mainline
techNode: gqa
header: 18-gqa.webp
---

## One Write-Head Is All You Need

In **November 2019**, Noam Shazeer — same Shazeer who had been on the original *Attention Is All You Need* author list two years earlier — posted a four-page note to arXiv called *"Fast Transformer Decoding: One Write-Head Is All You Need."* No co-authors. No big experiments. Just a section of math, a table of decoding speeds, and a slightly grumpy tone.

The grumpiness was earned. Shazeer had been trying to run a transformer at production latency on Google's TPU pods and kept hitting the same wall: **memory bandwidth**, not compute. Every decode step had to *re-read* the entire KV cache from HBM into the chip's registers, and the cache was *gigantic*. On a 6-billion-parameter model, the cache reads alone took eleven times longer than the actual {{< wiki "attention" >}}attention{{< /wiki >}} math.

His proposal was almost rude in its simplicity. The K and V projections are the only operations whose outputs you have to *keep around* across decode steps. Q is computed fresh every step from the current token. So: **keep only one K and one V per layer, shared across all heads**. Each head still gets its own query — its own *question* — but they all ask that question against the *same* shared database.

Shazeer called this **Multi-Query Attention (MQA)** — a 5–10× decode speedup at a cost of a few hundredths of a BLEU point. The paper was ignored for three years, then Google's PaLM team picked it up, then Meta's Llama 2 team shipped a refined version called **Grouped-Query Attention**, then everyone else. By 2026, *every* frontier-scale model on the [chapter 17 leaderboard](../17-kv-axes/) uses some form of Shazeer's trick.

This chapter is about what changed in the [microGPT listing](../01-cold-open/) to make that happen. The answer is **one integer divide**.

## The Diagnosis

From [chapter 17](../17-kv-axes/), the KV cache shape is $(2, L, H, T, D)$ with memory

$$
\text{bytes} = 2 \cdot L \cdot H \cdot T \cdot D \cdot \text{(bytes per element)}.
$$

Every term in that product is fixed by either model architecture or user prompt — except $H$, the head count. $L$ is "how deep is the network." $T$ is "how long is the input." $D$ is locked once you pick `n_embd / n_head`. But $H$ is "how many *parallel* sets of K and V did the architect allocate, one per head?"

Look at the [chapter 9 multi-head](../09-multi-head/) inner loop. Each head computes its own attention pattern — head 0 might attend to the previous token, head 1 to the start-of-sentence marker, head 2 to a long-distance coreference. The plurality of heads gives multi-head attention its expressive power.

Shazeer's load-bearing question: **does each head really need its own private copy of every past token's K and V vector?** Or could many heads share the same "database" and only differ in *how they query* it?

## The MQA Extreme

Shazeer's first answer was: collapse it all. **One** shared K, **one** shared V, $H$ different Qs. In microGPT terms, the projection matrices change shape:

| Matrix | Vanilla MHA shape | MQA shape |
|---|---|---|
| `attn_wq` | `n_embd × n_embd` | `n_embd × n_embd` (unchanged) |
| `attn_wk` | `n_embd × n_embd` | `head_dim × n_embd` |
| `attn_wv` | `n_embd × n_embd` | `head_dim × n_embd` |
| `attn_wo` | `n_embd × n_embd` | `n_embd × n_embd` (unchanged) |

The K and V projections shrink from full-width to one-head-width. The cache stores **one** K vector and **one** V vector per layer per token — not $H$ of them.

Compression along the H-axis: $H \times$. For Llama1-65B's $H = 64$, that is a 64× shrink of the cache. The 335 GB problem becomes a 5 GB problem.

Google deployed this in **PaLM** (April 2022). The paper reports a 5× decode speedup at the same model size. They also report it cost them about half a point on benchmark averages.

## The MQA Quality Cliff

Why does MQA lose quality at all?

In vanilla MHA, head $h$ has its own private $(K_h, V_h)$ — its own private "view" of what every past token *was* and *contained*. Head 0 might encode "this token's grammatical role." Head 1 might encode "its semantic topic." They store these views in *different subspaces* of the key/value space.

MQA forces all heads to share a single $(K, V)$. The model now has only **one** view per token. Heads can still ask different questions — Q is still per-head — but they all query the *same database*. If head 0 wants "grammatical role" and head 1 wants "semantic topic," they look it up in the same set of vectors. The richness has been clipped. There is also an *optimization-time* effect: all the head gradients pile onto the single shared K and V, which makes the K/V projections train differently — and empirically worse.

So MQA is **almost** the right idea. The compression factor is enormous, but the quality cliff is also real.

## GQA: The Compromise

Joshua Ainslie and collaborators at Google published *"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"* in **May 2023**. The headline contribution sounds obvious in hindsight: **don't collapse all $H$ heads into 1; collapse them into $G$ groups, with $G$ somewhere between 1 and $H$.**

Define two new {{< wiki "hyperparameters" >}}hyperparameters{{< /wiki >}}:

- `n_kv_head` = the number of K/V groups. Call it $G$.
- `group_size = n_head // n_kv_head` = how many query heads share each K/V.

Then:

- $G = H$ → vanilla MHA. Every head has its own K/V.
- $G = 1$ → MQA. All heads share one K/V.
- $G \in \{H/2, H/4, H/8, H/16\}$ → the *tunable middle* where almost everyone lives.

Ainslie et al. showed empirically that going from $G=1$ to $G=8$ closed almost all of the MQA quality gap while keeping most of the cache-compression win. The curve is **strongly concave**: most of the quality comes back with the first few groups; the marginal benefit beyond $G=H/4$ is small.

The paper's harder contribution was the *up-casting* recipe. If you already have a trained MHA model and want a GQA model, you do **not** retrain from scratch. You **mean-pool** the K and V projection rows within each group, getting a `(n_kv_head * head_dim) × n_embd` matrix, then briefly fine-tune (5% of pretraining compute). Llama 2 was the first major model to ship this recipe, in **July 2023**.

## The microGPT Diff

Here is the change. We start from the [chapter 9 multi-head](../09-multi-head/) listing. The vanilla code:

```python
# vanilla MHA — every head h slices the SAME hs from k and v
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
    ...
```

The GQA replacement:

```python
# GQA — query heads use their own slice; K/V heads use the GROUP slice
group_size = n_head // n_kv_head

for h in range(n_head):
    hs   = h * head_dim                          # Q index — unchanged
    kvhs = (h // group_size) * head_dim          # K/V index — quantized to the group
    q_h = q[hs:hs+head_dim]
    k_h = [ki[kvhs:kvhs+head_dim] for ki in keys[li]]
    v_h = [vi[kvhs:kvhs+head_dim] for vi in values[li]]
    ...
```

**That is the whole architectural change.** One extra line (`kvhs = ...`). One extra integer divide (`h // group_size`). The Q projection is identical; the inner-loop math (`attn_logits`, `attn_weights`, `head_out`) is identical. The only thing that changed is *which slice of K and V each query head reads from*.

The supporting state-dict changes are equally local:

```python
# vanilla MHA: K and V are full-width
state_dict[f'layer{li}.attn_wk']  # shape: (n_embd, n_embd)
state_dict[f'layer{li}.attn_wv']  # shape: (n_embd, n_embd)

# GQA: K and V are narrowed to n_kv_head heads
state_dict[f'layer{li}.attn_wk']  # shape: (n_kv_head * head_dim, n_embd)
state_dict[f'layer{li}.attn_wv']  # shape: (n_kv_head * head_dim, n_embd)
```

And the cache itself shrinks at the source: `keys[li].append(k)` now appends a vector of length `n_kv_head * head_dim` instead of `n_embd`. The append is still `list.append`. Nothing else moves.

Concretely, if microGPT's toy had `n_head=4` and we set `n_kv_head=2`, then `group_size=2`. Heads 0 and 1 share K/V slice 0; heads 2 and 3 share K/V slice 1. The cache per token per layer drops from 16 floats (= `n_embd`) to 8 floats (= `n_kv_head * head_dim`). 2× compression along the H-axis. Same toy. One integer divide.

## Visualizing The Mapping

What does the head-to-KV-head mapping actually look like for production configurations? Llama 3 8B picked $H=32$, $G=8$ (4× compression). Qwen3-235B picked $H=64$, $G=4$ (16×). GPT-OSS uses smaller models with $G=H/4$ as well.

```pyplot {id="head-to-kvhead" caption="Head-to-KV-head mapping for three production configurations. Each colored bar is a query head; the y-axis tells you which K/V group it reads from. The 'staircase' is the integer divide h // group_size."}
configs = [
    ('MHA (vanilla)\nn_head=32, n_kv_head=32', 32, 32, '#FF8C00'),
    ('Llama 3 8B\nn_head=32, n_kv_head=8\n(4x)', 32, 8, '#FF007F'),
    ('Qwen3-235B\nn_head=64, n_kv_head=4\n(16x)', 64, 4, '#00A8A8'),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, (label, H, G, color) in zip(axes, configs):
    group_size = H // G
    heads = list(range(H))
    kv_idx = [h // group_size for h in heads]
    ax.bar(heads, kv_idx, color=color, edgecolor='#1A1A1A', linewidth=0.6, width=0.85)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('query head index h')
    ax.set_ylabel('K/V group index = h // group_size')
    ax.set_xlim(-0.7, H - 0.3)
    ax.set_ylim(0, max(G, 2))
    ax.set_yticks(list(range(0, G + 1, max(1, G // 4))))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('GQA head-sharing pattern — same code, different (H, G)', fontsize=12, y=1.02)
fig.tight_layout()
```

In vanilla MHA (left), the mapping is the identity. In Llama 3 (middle), every group of 4 query heads shares one K/V. In Qwen3 (right), every group of 16 query heads shares one K/V. The architectural difference between the three models is exactly the slope of that staircase.

## Napkin Math: How Much Cache Did We Actually Save?

Let's compute the cache size for Llama 3 8B at 128K context, in bf16 (2 bytes/element):

- $L = 32$ layers, $T = 128{,}000$ tokens, $D = 128$ per-head dimension.
- **Vanilla MHA** ($H = 32$): $2 \cdot 32 \cdot 32 \cdot 128{,}000 \cdot 128 \cdot 2 = 67$ GB.
- **GQA with $G = 8$** (4×): $67 / 4 \approx 17$ GB.
- **MQA** ($G = 1$): $67 / 32 \approx 2.1$ GB.

For Qwen3-235B at 128K context, $L=94$, $H=64$, $G=4$, $D=128$:

- Vanilla MHA equivalent: $2 \cdot 94 \cdot 64 \cdot 128{,}000 \cdot 128 \cdot 2 \approx 393$ GB.
- With GQA at $G=4$ (16×): $\approx 25$ GB.

The absolute numbers will shift once you account for fp8 cache quantization. The point is **the ratio**. Qwen3 pushed the compression all the way to 16×, accepting more quality risk to fit a 128K-token cache on a *single* H100.

```pyplot {id="cache-vs-context" caption="KV cache size vs context length for four head-sharing regimes on Llama 3 8B's L=32, D=128. The y-axis is log-scaled — every step down is another order of magnitude."}
L, D, bytes_per_elem = 32, 128, 2
T = np.linspace(1024, 256 * 1024, 200)

regimes = [
    ('Vanilla MHA (H=32)',  32, '#FF8C00'),
    ('GQA 4x (Llama 3, G=8)', 8, '#FF007F'),
    ('GQA 16x (Qwen3-like, G=2)', 2, '#00A8A8'),
    ('MQA (G=1)',             1, '#FFD700'),
]

fig, ax = plt.subplots(figsize=(9, 4.5))
for label, G, color in regimes:
    cache_gb = 2 * L * G * T * D * bytes_per_elem / 1e9
    ax.plot(T / 1024, cache_gb, color=color, linewidth=2.5, label=label)

ax.axhline(80, color='#1A1A1A', linestyle=':', linewidth=1.2)
ax.text(8, 90, 'one H100 (80 GB)', fontsize=9, color='#1A1A1A')

ax.set_xlabel('context length T (thousands of tokens)')
ax.set_ylabel('KV cache size (GB, bf16)')
ax.set_yscale('log')
ax.set_title('GQA shrinks the cache by exactly the H-axis compression factor')
ax.legend(loc='lower right', frameon=False)
ax.grid(True, which='both', alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

The curves are **parallel** in log-space — GQA does not change the *slope* of cache-vs-context, it multiplies the line by a constant of $H/G$. MHA crosses the H100 line at 32K context; GQA-4× at 128K; GQA-16× and MQA do not cross it within any sane window. **The choice of $G$ literally determines what context length fits on one GPU.**

## The Production Lineup, As Of 2026

| Model | $H$ | $G$ | Ratio $H/G$ | Year |
|---|---|---|---|---|
| Llama 1, GPT-3 | 64–96 | = $H$ | 1× (MHA) | 2020–2022 |
| PaLM | 48 | 1 | 48× (MQA) | 2022 |
| Llama 2 70B | 64 | 8 | 8× | 2023 |
| **Llama 3 8B / 70B** | **32 / 64** | **8 / 8** | **4× / 8×** | 2024 |
| Mistral Large | 96 | 8 | 12× | 2024 |
| **GLM 4.5** | **96** | **8** | **12×** | 2025 |
| **Qwen3-235B-A22B** | **64** | **4** | **16×** | 2025 |
| GPT-OSS-120B | 64 | 8 | 8× | 2025 |

Notice that the *number of K/V groups* has converged on a small constant: almost everyone picks $G \in \{4, 8\}$, regardless of model size. Bigger models grow $H$, not $G$ — which is why the compression ratio drifts upward as scale grows. The compression factor is, in effect, a *cultural* choice that the field made around 2023 and has barely revisited.

## What GQA Does *Not* Solve

GQA shrinks the H-axis. It is silent on the other axes:

- **Cache still grows linearly in $T$.** A 1M-token context with GQA-4× still produces a 130 GB cache on Llama 3 8B. GQA only delays the cliff.
- **Cache still grows linearly in $L$.** Sliding-window and SSM hybrids ([ch.20](../20-sliding-window/), [ch.21](../21-ssm-hybrids/)) attack this axis.
- **The $D$ dimension is untouched.** Each cached vector is still `head_dim` long. The DeepSeek [MLA trick](../19-mla/) attacks *this* axis.

GQA is a **localized fix** to one of [the three axes from chapter 17](../17-kv-axes/). It composes cleanly with the others — DeepSeek V3 stacks GQA-style head sharing on top of MLA on top of a sparse routing scheme.

## What To Remember

1. **GQA changes one number and one slice.** `n_kv_head` is the number of K/V groups; the slice index becomes `(h // group_size) * head_dim`. Everything else in microGPT is unchanged.
2. **The compression ratio along the H-axis is exactly $H / G$.** Llama 3 → 4×, GLM 4.5 → 12×, Qwen3 → 16×. The choice of $G$ is mostly cultural, not theoretical.
3. **MQA is GQA with $G=1$.** It is the extreme of the same family. It compresses maximally but pays a measurable quality cost. Nobody at frontier scale chooses it anymore.
4. **MHA is GQA with $G=H$.** Vanilla multi-head attention is the *other* extreme. Almost no new production model ships pure MHA.
5. **Q heads stay independent.** GQA shrinks the K/V *database*, not the *number of questions*. The model still asks $H$ distinct queries per token.
6. **Up-casting works.** You can convert a trained MHA model to GQA by mean-pooling the K/V rows in each group and briefly fine-tuning. Llama 2-Chat shipped this way.
7. **GQA only fixes one axis.** $T$, $L$, and $D$ are still wide open — and the next chapters fix those.

---

**Continue to** → [Multi-head Latent Attention](../19-mla/) — GQA shares K/V across heads; MLA does something stranger, *compressing* K/V into a lower-dimensional latent that gets *re-inflated* at attention time, fusing the inflation into the Q matrix for free.

