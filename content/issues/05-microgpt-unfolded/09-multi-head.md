---
title: "Multi-Head Attention"
description: "Single-head attention with a `for h in range(n_head)` wrapper. Why we slice the same Q, K, V into shorter chunks rather than projecting them separately, and what `attn_wo` is putting back together."
topics: [transformer, attention]
tags: [microgpt, multi-head, attn_wo]
theme: cream
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 90
techKind: mainline
techNode: multi-head
header: default.webp
---

## The Ablation That Wasn't Supposed To Matter

Spring 2017, somewhere on the second floor of Building 1965 at Google Mountain View. **Ashish Vaswani** and a handful of co-authors are racing to finish a NeurIPS submission with the working title "Attention Is All You Need." The architecture is mostly done. The training curves are good. There is exactly one parameter left to ablate: how many parallel attention "heads" should the layer have? The team's intuition is that this is one of those knobs that looks scary on paper but doesn't actually matter much. They run `h = 1` as a sanity check on the way to settling for `h = 8`.

The `h = 1` model is **a full BLEU point worse on English-to-German** than `h = 8`. A whole BLEU. For comparison: that gap is larger than the gap between their full Transformer and the best LSTM-based system from the previous year. One number, in the head-count column, is doing more than the entire architecture switch.

The team writes this up in the famous Table 3 of the paper, almost as an afterthought. They observe, in the deadpan voice that all NeurIPS papers use, that "single-head attention is 0.9 BLEU worse than the best setting." They do not, in 2017, claim to know why. They just know that **slicing the same Q, K, V into eight short chunks and running attention on each chunk in parallel** is dramatically better than running attention once on the whole vector.

That ablation, never advertised, is what made the rest of the decade possible. By 2026, every frontier LLM you can name has somewhere between 32 and 128 heads, and a small library of papers exists explaining what those heads have learned to do. This chapter is about [the `for h in range(n_head)` loop in microGPT](../01-cold-open/) — the one we glossed past in [ch.8 attention](../08-attention/) — and why splitting one big attention into many small ones is the cheapest free-lunch in deep learning.

## The Loop, Read Slowly

Here is the multi-head block of microGPT, lifted verbatim from the cold-open listing. The variables `q`, `k`, `v` were computed one chapter ago in [ch.6 QKV projections](../06-qkv-projections/); the attention math inside the loop is exactly what we built in [ch.8 attention](../08-attention/), just over shorter vectors.

```python
x_attn = []
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
    attn_logits = [
        sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
        for t in range(len(k_h))
    ]
    attn_weights = softmax(attn_logits)
    head_out = [
        sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
        for j in range(head_dim)
    ]
    x_attn.extend(head_out)

x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
```

Walk through it in your head for the toy model where `n_head = 4` and `head_dim = 4`:

- Iteration `h = 0` takes `q[0:4]`, the first four numbers of the query vector. It pairs them with `k[0:4]` and `v[0:4]` from every cached token. It runs attention. The output `head_out` is a length-4 vector. Extend it onto `x_attn`.
- Iteration `h = 1` takes `q[4:8]`, the *next* four numbers. Different chunk of the same `q`. Different chunks of the same `k` and `v`. Independent attention computation. Another length-4 output appended to `x_attn`.
- And so on for `h = 2` and `h = 3`.

After the loop, `x_attn` is a length-16 list — the concatenation of four length-4 head outputs in order. We then **mix the heads together** with a final linear projection: `linear(x_attn, attn_wo)`, where `attn_wo` is the `16 × 16` output-projection matrix from the [state dict](../02-state-dict/).

> **The slicing trick.** Each head looks at a SLICE of Q, K, V, not at separate Q, K, V tensors. We computed *one* `q`, *one* `k`, *one* `v` in [ch.6](../06-qkv-projections/). The heads then divvy them up by index range. This is the most important sentence on this page.

Stated as a formula, with $H$ heads and head dimension $d_h = n_\text{embd}/H$:

$$
\text{head}_h(x) = \text{softmax}\!\left( \frac{q_h \, K_h^\top}{\sqrt{d_h}} \right) V_h, \qquad q_h = q_{[h \, d_h : (h+1) \, d_h]}
$$

$$
\text{MHA}(x) = \big[\text{head}_0(x) \; \| \; \text{head}_1(x) \; \| \; \cdots \; \| \; \text{head}_{H-1}(x)\big] \, W_O
$$

Where $\|$ is concatenation along the feature axis, and $W_O$ is `attn_wo`. Read the formula and the microGPT loop side by side until you see they are the same thing.

## Why Heads At All

The 2017 paper's ablation said multi-head helped, but did not say *why*. By 2021, a small army of interpretability researchers had pried open trained transformers and started giving the heads job descriptions. A representative inventory, mostly from Anthropic's *A Mathematical Framework for Transformer Circuits* and assorted Chris Olah essays:

- **Positional heads.** A head whose attention pattern is essentially "look at the token four positions ago." Useful for syntax, dependency parsing, local n-gram features.
- **Induction heads.** A head that implements the rule "if the current token is X, and X has appeared before followed by Y, attend strongly to Y." This is how transformers do in-context learning. Removing all induction heads catastrophically degrades few-shot performance.
- **Copy heads.** A head that, when the previous token is `"`, attends to whatever was inside the most recent quotation. Used for verbatim quoting.
- **Coreference heads.** A head that resolves "she" to the most recent female-coded proper noun. Genuinely linguistically aware behavior, learned by gradient descent without supervision.
- **Attention sinks.** A head that, regardless of context, dumps most of its weight on the BOS token. We'll come back to this in the wrinkles section.

The intuition: a single attention head can compute **one** weighted-average of the value stream per token. If your model needs both "average over the syntactic dependents" and "average over the coreferent pronouns" to compute its next-token prediction, a single head is forced to choose, or to muddle the two together into one weighted average. Multi-head attention lets the model maintain **several parallel weighted averages** that get re-combined downstream.

```pyplot {id="head-patterns" caption="Four heads, four jobs. Synthetic but representative attention patterns from a single layer of a trained transformer. Pink = strong attention. Each row is a query token; each column is a key token. Lower-triangular because of causal masking."}
np.random.seed(7)
T = 12  # sequence length
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'it', 'purred', 'at', 'the', 'sun']

def causal(mat):
    mask = np.tril(np.ones_like(mat))
    mat = mat * mask
    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-9)
    return mat

# Head 0: positional — attends to the token 2 back
A0 = np.zeros((T, T))
for i in range(T):
    for j in range(T):
        A0[i, j] = np.exp(-((i - j - 2)**2) / 0.5)
A0 = causal(A0)

# Head 1: BOS / attention-sink — dumps most weight on token 0
A1 = np.zeros((T, T))
for i in range(T):
    A1[i, 0] = 5.0
    A1[i, max(0, i-1)] += 0.5
    A1[i, i] += 0.3
A1 = causal(A1)

# Head 2: coreference — "it" attends to "cat"
A2 = np.random.rand(T, T) * 0.1
A2[7, 1] = 4.0       # 'it' -> 'cat'
A2[10, 4] = 3.0      # second 'the' -> first 'the'
for i in range(T):
    A2[i, i] += 0.2
A2 = causal(A2)

# Head 3: bag-of-context — diffuse, averages everything seen so far
A3 = np.ones((T, T))
A3 = causal(A3)

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
titles = ['Head 0: positional (-2)', 'Head 1: attention sink (BOS)',
          'Head 2: coreference', 'Head 3: diffuse average']
for ax, A, title in zip(axes, [A0, A1, A2, A3], titles):
    ax.imshow(A, cmap='RdPu', aspect='equal', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticklabels(tokens, rotation=60, fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xlabel('key (attended to)', fontsize=8)
    if ax is axes[0]:
        ax.set_ylabel('query (the token doing the attending)', fontsize=8)
plt.tight_layout()
```

The four heatmaps above are synthetic, but the *patterns* are real — every published mech-interp paper on small transformers has examples that look like at least three of these four. The point is: **none of these four jobs is reducible to any of the others**. Head 0 cannot do Head 2's job, because the coreference distance "cat → it" is six tokens here and could be three or sixteen in a different sentence. Head 3 cannot do Head 0's job, because diffuse averaging blurs out the positional precision. The model gets all four behaviors for the price of one matrix multiplication — that's the win.

## Slice vs Separate: Why Slicing Wins The GPU

There is a sleight of hand in microGPT worth surfacing. We compute `q`, `k`, `v` as **single full-width** vectors via a single linear projection per role:

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])   # shape: n_embd
k = linear(x, state_dict[f'layer{li}.attn_wk'])   # shape: n_embd
v = linear(x, state_dict[f'layer{li}.attn_wv'])   # shape: n_embd
```

`attn_wq` is `n_embd × n_embd` (16 × 16 in the toy, 4096 × 4096 in Llama 3 8B). The heads then *slice* the result. **You could just as well have stored `n_head` separate projections** — `attn_wq_0` of shape `head_dim × n_embd`, `attn_wq_1` of the same, and so on — and computed each head's `q_h` directly. Mathematically these are identical:

$$
q = W_Q \, x, \quad q_h = q_{[h d_h:(h+1)d_h]} \quad \iff \quad q_h = W_{Q,h} \, x, \quad W_{Q,h} = (W_Q)_{[h d_h:(h+1)d_h, :]}
$$

The matrix is literally the same numbers. The slicing operation is a no-op view, not a copy. So why does every implementation in 2026 use the single-big-matrix form? Because **one big matmul is faster than `n_head` small matmuls** on every accelerator ever built. GPUs love big regular tile shapes. Doing one `(B × n_embd) × (n_embd × n_embd)` GEMM keeps the tensor cores saturated; doing 32 separate `(B × n_embd) × (n_embd × head_dim)` GEMMs leaves cycles on the floor. The slicing is, in this sense, *purely cosmetic* — it's the conceptual story you tell about a matmul whose actual implementation doesn't care.

## What `attn_wo` Is Doing

After the loop, `x_attn` is the concatenation of all `head_out` vectors. In microGPT for the toy model:

```text
x_attn = [head_0[0], head_0[1], head_0[2], head_0[3],   # head 0 owns these slots
          head_1[0], head_1[1], head_1[2], head_1[3],   # head 1 owns these
          head_2[0], head_2[1], head_2[2], head_2[3],   # head 2 owns these
          head_3[0], head_3[1], head_3[2], head_3[3]]   # head 3 owns these
```

Each head wrote its output into a private 4-slot territory. `attn_wo`'s job is the linear `n_embd × n_embd` projection that **mixes those territories back together**. Without `attn_wo`, head 0's information would never reach the slots that head 2's information was written into. The output projection is the only place where the heads talk to each other.

```pyplot {id="attn-wo-mixing" caption="The attn_wo output projection viewed as a head-mixing operation. Each row is an output channel; each column is an input channel. The vertical bands mark which 4-column block belongs to which head. Cells with large absolute value mean 'this output channel reads from that head's territory'."}
np.random.seed(11)
n_embd = 16
head_dim = 4
n_head = n_embd // head_dim

# Synthesize a plausible attn_wo: roughly Gaussian, plus structured biases
# so head 0 and head 2 mix tightly into the first half of output channels,
# while heads 1 and 3 dominate the second half.
W = np.random.randn(n_embd, n_embd) * 0.18
for h in range(n_head):
    rows = slice(0, 8) if h % 2 == 0 else slice(8, 16)
    cols = slice(h * head_dim, (h + 1) * head_dim)
    W[rows, cols] += np.random.randn(8, head_dim) * 0.35

fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(np.abs(W), cmap='magma', aspect='equal')

# vertical bands separating heads
for h in range(1, n_head):
    ax.axvline(h * head_dim - 0.5, color='#FFD700', linewidth=2)

ax.set_xticks(range(n_embd))
ax.set_yticks(range(n_embd))
ax.set_xlabel('input channel (= concat of head outputs)')
ax.set_ylabel('output channel (= residual stream slot)')
ax.set_title('|attn_wo| as a head-mixing matrix\n(yellow lines separate the n_head=4 territories)')

# annotate which columns belong to which head
for h in range(n_head):
    ax.text(h * head_dim + (head_dim - 1) / 2, -1.2, f'head {h}',
            ha='center', color='#FF007F', fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax, label='|weight|', fraction=0.046)
plt.tight_layout()
```

The vertical yellow bands are not in the matrix — they are visual reminders that *columns within a band were written by the same head*. The bright cells tell you which output channels read most strongly from which head. In a real trained transformer, this matrix is dense, but if you cluster its columns by which rows they activate, the clusters often line up suspiciously well with the heads. Anthropic's circuits work uses precisely this view: `attn_wo` is the place where a single "feature" of the residual stream gets composed out of contributions from multiple heads.

## Napkin Math: How Much Of An LLM Is This Loop?

Take **Llama 3 8B** as the reference point. From its config: `n_embd = 4096`, `n_head = 32`, `head_dim = 128`, `n_layer = 32`. Plug into the multi-head block:

- `attn_wq`, `attn_wk`, `attn_wv` each: $4096 \times 4096 = 16.8\text{M}$ params.
- `attn_wo`: another $4096 \times 4096 = 16.8\text{M}$ params.
- Per layer: $4 \times 16.8\text{M} = 67.1\text{M}$ params just in attention.
- Across 32 layers: $\approx 2.15\text{B}$ params — about **27% of the 8B total** is the attention projections.
- Of which, `attn_wo` alone accounts for $32 \times 16.8\text{M} \approx 537\text{M}$ params. **Half a gigabyte** of weights, in FP16, exists solely to do head-mixing.

Now zoom out further. **Llama 3 70B** uses `n_embd = 8192`, `n_head = 64`, `head_dim = 128`, `n_layer = 80`. The same calculation:

- Per-layer attention: $4 \times 8192^2 = 268\text{M}$ params.
- All layers: $\approx 21.5\text{B}$ params — **31% of the 70B model**.
- `attn_wo` alone: $80 \times 67\text{M} \approx 5.4\text{B}$ params.

This is why the modern architectural pressure is so intense to **shrink the K and V matrices** without touching `attn_wq` or `attn_wo`. The output projection is doing work no other layer can replace, and the query projection feeds it. Keys and values, however, only exist to be matmul'd against each other inside the per-head loop — and they sit in the KV cache, growing with sequence length. Hence the next two architectures we'll see.

## Three Modern Wrinkles On The Same Loop

**MHA → MQA → GQA.** Vanilla *Multi-Head Attention* gives every head its own `(q, k, v)` triple — `n_head` queries, `n_head` keys, `n_head` values. Noam Shazeer's 2019 *Multi-Query Attention* paper observed that the keys and values are doing *less* work than the queries, and proposed sharing **one** K and **one** V across all heads while keeping `n_head` separate Qs. Llama 2 70B and Mistral 7B both adopted the middle-ground *Grouped-Query Attention* (Ainslie et al., 2023) where you partition the heads into `n_kv_head` groups, each group sharing a K/V pair. Llama 3 8B uses 32 query heads and 8 KV heads — a 4:1 grouping. The microGPT loop barely changes; only the slicing of `keys[li]` and `values[li]` rebinds. The full story is [ch.18 GQA](../18-gqa/).

**Flash-Attention.** Tri Dao's *FlashAttention* (2022) does not change the math at all. It changes the **memory access pattern**: instead of materializing the full `(T × T)` `attn_logits` matrix in HBM, it tiles the computation so that each tile of `q_h`, `k_h`, `v_h` is loaded into SRAM, the per-head `softmax` and `head_out` are computed on-chip in one fused kernel, and only the final `head_out` ever leaves the chip. On an A100, this is a ~2–4× wall-clock speedup with bit-exact equivalence to the loop above. From microGPT's perspective: the loop body is replaced by one call to a CUDA kernel. The math is identical.

**Attention sinks.** A 2023 paper by Xiao et al. (*Efficient Streaming Language Models with Attention Sinks*) made an empirical observation that anyone who had stared at attention heatmaps already half-knew: in a trained LLM, the very first token of the sequence — often `<BOS>` — receives **disproportionately large** attention weight from many heads, regardless of content. The current best guess is that softmax forces probabilities to sum to 1, so any head that wants to "say nothing useful this step" needs *somewhere* to dump its weight, and the BOS token is the most convenient pressure-release valve. The implication for the [KV cache](../13-kv-cache/) is that you cannot evict the first few tokens even from a 2M-token window — they are load-bearing not for their content but for their *role as sinks*. Look at Head 1 in our heatmap above for the picture.

## What To Remember

1. **Multi-head attention is one `for` loop wrapping single-head attention.** Each iteration takes a length-`head_dim` slice of `q`, `k`, `v` and runs the [ch.8](../08-attention/) math. The heads are independent until `attn_wo`.
2. **Slicing is the trick.** We don't store `n_head` separate Q/K/V projections; we store one big projection per role and let array indexing carve out each head's territory. This is what makes the GPU happy.
3. **`attn_wo` is the only place heads interact.** Without it, the heads would each be writing to their own non-overlapping region of the residual stream and never sharing information. The output projection is what makes the per-head specialization useful.
4. **Heads learn specialized jobs.** Positional, induction, coreference, copy, sink. Mech-interp has a whole zoology. The point of running 32 of them in parallel is that the model can compute many different weighted averages of the value stream at once.
5. **A lot of LLM weights live here.** In Llama 3 8B, multi-head attention is 27% of the parameter count, and `attn_wo` alone is half a gigabyte in FP16. This is why every modern compression scheme worries about Q/K/V/O projections first.

---

**Continue to** → [The Residual Stream](../10-residual-stream/) — the heads write their output back into a shared 4096-dim "bus" via that `x = [a + b for a, b in zip(x, x_residual)]` line, and the geometry of that bus is the secret skeleton of the whole network.

