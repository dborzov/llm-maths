---
title: "The MLP Block"
description: "Fatten with `mlp_fc1`, squash with `relu`, skinny back down with `mlp_fc2`. Why the hidden dimension is conventionally 4×, and what 56% of every modern LLM's parameter budget is actually doing."
topics: [transformer, mlp]
tags: [microgpt, mlp_fc1, mlp_fc2, ffn]
theme: cream
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 110
techKind: primer
techNode: mlp-block
header: 11-mlp-block.webp
---

## A Throwaway Sentence In June 2017

On June 12, 2017, eight researchers at Google posted a preprint to arXiv with the unassuming title *Attention Is All You Need*. The paper is rightly famous for inventing the Transformer architecture — multi-head attention, positional encodings, the encoder-decoder split. But midway through Section 3.3, sandwiched between two paragraphs about attention, there is a sentence so casual it almost reads like a footnote:

> *"In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network … The dimensionality of input and output is $d_{\text{model}} = 512$, and the inner-layer has dimensionality $d_{ff} = 2048$."*

Two numbers, 512 and 2048. A ratio of **4**. No ablation. No justification. No table comparing 2× to 4× to 8×. The authors picked it, it worked, they moved on.

That ratio went on to define the shape of every large language model for the next decade. GPT-2: 4×. GPT-3: 4×. T5: 4×. Bloom, Falcon, Pythia, OPT, MPT: 4×. The number was so accepted that a 2021 paper measuring scaling laws didn't even list `d_ff / d_model` as a hyperparameter — it just *assumed* the conventional value, the way nobody puts $\pi$ on a list of free constants.

This chapter is about the six lines of microGPT that implement that 4×-wide block, what each row of those two matrices is doing, why the ratio became sacred, and what the modern frontier has done about it.

## The Six Lines

Here is the MLP half of the transformer block, lifted verbatim from the [ch.1 listing](../01-cold-open/):

```python
# MLP block
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [relu(xi) for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]
```

The shape is the same as the attention half: **normalize → do something → add the result back into the {{< wiki "residual-stream" >}}residual stream{{< /wiki >}}**. What's different is what happens in the middle. Two `linear()` calls separated by a per-element `relu`. The matrices have asymmetric shapes:

| Tensor | Shape `(out, in)` | microGPT toy | Llama 3 8B |
|---|---|---|---|
| `mlp_fc1` | `(4·n_embd, n_embd)` | `64 × 16` | `14336 × 4096` |
| `mlp_fc2` | `(n_embd, 4·n_embd)` | `16 × 64` | `4096 × 14336` |

That `4·n_embd` middle dimension is the **MLP hidden dimension**, sometimes called `d_ff` (feed-forward) or `intermediate_size`. The block widens the vector by 4×, does something nonlinear, then narrows it back. Fatten, squash, skinny.

> Llama 3 8B is technically *not* exactly 4×. It uses 14336 / 4096 ≈ **3.5×**. Meta picked 14336 because it factors nicely (`14336 = 2^11 · 7`) for tensor-parallel sharding and SwiGLU compensates for the missing 0.5× with a third matrix per block. The vanilla architecture in microGPT keeps the original convention.

## What `mlp_fc1` Is Really Doing

The mechanical reading of `linear(x, mlp_fc1)` from [ch.4](../04-linear/) is "for each of 64 rows of the weight matrix, dot it with the 16-element input." But what *are* those rows?

Think of each row as a **feature detector**. It is 16 numbers — the same dimensionality as the residual stream — and the dot product asks: "how much does the current residual look like *this* pattern?" If the residual aligns with row 7, the seventh entry of the fattened vector is large and positive. If it aligns with the *opposite* of row 7, the entry is large and negative.

Then comes `relu`. The function `max(0, x)` is the cheapest nonlinearity in the deep-learning catalog: keep positive entries, zero the rest. After ReLU, **roughly half** of the 64 hidden units are dead — they were pointing the wrong way. Only the ones whose detector matched the residual *with the correct sign* survive into the next stage.

`mlp_fc2` then reads those 64 firings and produces 16 numbers — one per output dimension. Each column of `mlp_fc2` is a "**value**" vector: when detector $i$ fires, it adds the $i$-th column of `mlp_fc2` (weighted by the firing strength) into the output. This pattern has a name. **Geva et al. (2021)** showed that the rows of `mlp_fc1` and the columns of `mlp_fc2` together form a **key-value memory**: `mlp_fc1` is the "keys" (what patterns to detect), `mlp_fc2` is the "values" (what to write back if you see them). The MLP block is, structurally, a **soft dictionary lookup** with `4·n_embd` slots.

In a real model, those slots have been observed to encode astonishingly specific things: a single hidden unit in GPT-2 fires for sentences about the verb "to be" in past tense; another fires for tokens inside Wikipedia infoboxes; another for any token that completes a phrasal verb. **Knowledge** lives here. Editing what a model "knows" — the [ROME](https://rome.baulab.info/) and MEMIT papers — is, mechanically, editing rows of `mlp_fc1` and columns of `mlp_fc2`.

## A Worked Example, End To End

Let's run a single token through the MLP block in our toy `n_embd = 16` model and watch the vector morph.

```pyplot {id="mlp-stages" caption="One vector's journey through the MLP block. Top to bottom: residual in (16-d), fattened by mlp_fc1 (64-d), after ReLU (about half clamped to zero), skinnied back (16-d). Pink positive, teal negative, white zero."}
np.random.seed(7)
n_embd = 16
hidden = 4 * n_embd

# Synthetic input residual: structured but mixed-sign
x = np.tanh(np.random.randn(n_embd) * 1.2)

# Pretend trained weights
W1 = np.random.randn(hidden, n_embd) * 0.5
W2 = np.random.randn(n_embd, hidden) * 0.3

# Forward pass
h_pre  = W1 @ x
h_post = np.maximum(0.0, h_pre)
y      = W2 @ h_post

# Plot as four heat strips stacked vertically
fig, axes = plt.subplots(4, 1, figsize=(10, 5.5),
                         gridspec_kw={'height_ratios': [1, 1, 1, 1]})

def strip(ax, vec, label, vmax):
    ax.imshow(vec[None, :], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
              aspect='auto')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(label, rotation=0, ha='right', va='center',
                  fontsize=11, labelpad=40)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1A1A1A'); spine.set_linewidth(2)

vmax_h = max(abs(h_pre).max(), abs(h_post).max())
vmax_x = max(abs(x).max(),     abs(y).max())

strip(axes[0], x,      'x  in\n(16)',                vmax_x)
strip(axes[1], h_pre,  'fc1(x)\nfattened (64)',      vmax_h)
strip(axes[2], h_post, 'relu\n(64, half = 0)',       vmax_h)
strip(axes[3], y,      'fc2\nskinnied (16)',         vmax_x)

# Count how many hidden units survived ReLU
n_alive = (h_post > 0).sum()
axes[2].text(hidden + 0.5, 0, f'{n_alive}/{hidden} alive',
             va='center', fontsize=10, color='#1A1A1A')

plt.suptitle('A vector through the MLP block (toy: n_embd=16, hidden=64)',
             fontsize=12, y=0.97)
plt.tight_layout()
```

Notice that the **fattened** strip has more visible structure than the input — the 64 detectors have spread the information into a wider, sparser representation. After ReLU, about half of those 64 cells are zero (the cool blue ones got clamped). Then `mlp_fc2` reads the surviving 32-ish firings and writes a fresh 16-d vector back into the residual stream.

Mathematically the whole block is:

$$
\text{MLP}(x) = W_2 \, \text{ReLU}(W_1 \, x)
$$

with $W_1 \in \mathbb{R}^{4d \times d}$ and $W_2 \in \mathbb{R}^{d \times 4d}$. **No bias terms** — see [ch.4](../04-linear/) for why modern LLMs drop them.

## Why 4×? Nobody Really Knows

There is no derivation. There is no theorem that says "the optimal MLP fattening ratio is four." The number was picked by Vaswani et al. in 2017, presumably after some informal sweeps inside Google Brain, and it resisted nearly a decade of attempts at improvement. Three partial explanations from the literature:

1. **Parameter balance.** With a 4× MLP hidden dim and no GQA, the MLP holds exactly **2× as many parameters** as the four attention matrices combined — a split Hoffmann et al. (Chinchilla, 2022) found sat near a broad efficiency optimum.
2. **Expressiveness.** Eldan & Shamir (2016) showed 2-layer ReLU networks need their hidden layer wider than the input to be universal approximators for "non-trivially curved" functions. 4× has comfortable headroom.
3. **Hardware.** A `4096 × 16384` matmul saturates tensor cores, fits shared-memory tiles, and divides cleanly by 8 (FP8) and 16 (tensor-parallel sharding).

Whatever the real reason, the number stuck.

## The MLP Is Half The Model

This is the punchline that everyone hits the first time they count parameters in a transformer. For a vanilla architecture with no GQA, **per layer**:

$$
\text{attn params} = 4 \cdot n_{\text{embd}}^2 \qquad \text{MLP params} = 2 \cdot n_{\text{embd}} \cdot (4 \cdot n_{\text{embd}}) = 8 \cdot n_{\text{embd}}^2
$$

The MLP holds **twice** as many parameters as attention, per layer. With GQA shrinking the K and V projections to roughly a quarter of their original size (see [ch.9](../09-multi-head/)), the attention share gets even smaller and the MLP share climbs further.

Napkin math for Llama 3 8B per layer:

- `mlp_fc1`: $4096 \times 14336 = 58{,}720{,}256$ parameters.
- `mlp_fc2`: $14336 \times 4096 = 58{,}720{,}256$ parameters.
- Total MLP: $\approx 117\,\text{M}$ parameters per layer.
- Across 32 layers: $\approx \mathbf{3.75\,B}$ parameters — **roughly half of the 8 B total**.

That is what 60% of your GPU's HBM is storing when Llama 3 8B sits idle: the rows and columns of those two matrices, in every one of the 32 layers, encoding what the model knows.

```pyplot {id="mlp-vs-attn-budget" caption="Parameter budget for one transformer layer. Vanilla: MLP is 2x attention. Llama-style with GQA + SwiGLU: MLP eats even more of the layer."}
n_embd = 4096
kv_groups_full = 1     # vanilla
kv_groups_gqa  = 4     # Llama 3: 8 KV heads, 32 Q heads -> 4:1 ratio

# Vanilla (microGPT-style): 4 attn matrices of n_embd^2, 2 MLP matrices of n_embd*4n_embd
attn_vanilla = 4 * n_embd * n_embd
mlp_vanilla  = 2 * n_embd * (4 * n_embd)

# Llama 3 8B-style: GQA shrinks Wk/Wv, SwiGLU has 3 MLP matrices each of (n_embd, ~3.5*n_embd)
mlp_hidden_l3 = 14336
attn_llama = 2 * n_embd * n_embd + 2 * n_embd * (n_embd // kv_groups_gqa)
mlp_llama  = 3 * n_embd * mlp_hidden_l3   # SwiGLU: W_gate, W_up, W_down

fig, ax = plt.subplots(figsize=(8.5, 4))
groups   = ['Vanilla\n(microGPT)', 'Llama 3 8B\n(GQA + SwiGLU)']
attn_M   = [attn_vanilla/1e6, attn_llama/1e6]
mlp_M    = [mlp_vanilla/1e6,  mlp_llama/1e6]

xs = np.arange(len(groups))
ax.bar(xs - 0.18, attn_M, width=0.36, label='attention',
       color='#FF007F', edgecolor='#1A1A1A', linewidth=1.5)
ax.bar(xs + 0.18, mlp_M,  width=0.36, label='MLP',
       color='#00A8A8', edgecolor='#1A1A1A', linewidth=1.5)

for x_, v in zip(xs - 0.18, attn_M):
    ax.text(x_, v + 4, f'{v:.1f} M', ha='center', fontsize=10)
for x_, v in zip(xs + 0.18, mlp_M):
    ax.text(x_, v + 4, f'{v:.1f} M', ha='center', fontsize=10)

ax.set_xticks(xs)
ax.set_xticklabels(groups)
ax.set_ylabel('parameters per layer (millions)')
ax.set_title('MLP dominates the per-layer parameter budget — and the gap is growing')
ax.legend(loc='upper left', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, max(mlp_M) * 1.22)

ratio_vanilla = mlp_vanilla / attn_vanilla
ratio_llama   = mlp_llama   / attn_llama
print(f"Vanilla:  MLP / attn = {ratio_vanilla:.2f}x")
print(f"Llama 3:  MLP / attn = {ratio_llama:.2f}x  (GQA shrinks attn, SwiGLU grows MLP)")
```

## Modern Wrinkles

The basic `W_2 \cdot \text{ReLU}(W_1 x)` block has, in eight years, been refined in two main directions. We'll cover them in depth in their own chapters — here is the pocket version.

### SwiGLU: Three Matrices Instead Of Two

Noam Shazeer's 2020 paper *GLU Variants Improve Transformer* replaced the simple `ReLU(W_1 x)` with a **gated** form:

$$
\text{SwiGLU}(x) = W_{\text{down}} \, \big(\,\text{Swish}(W_{\text{gate}}\,x) \;\odot\; (W_{\text{up}}\,x)\,\big)
$$

Three matrices. Two of them (`W_gate` and `W_up`) read the residual and produce hidden-dim vectors; one is passed through Swish (a smooth cousin of ReLU), one stays linear, and they are multiplied element-wise. `W_down` reads the gated product and writes back into the residual. To keep parameter count comparable, Llama and friends shrink the hidden dim from 4× to roughly 2.67× — Llama 3's 14336 / 4096 ≈ 3.5× lands in between because Meta tuned for divisibility rather than minimal-params parity.

Same fattening-and-skinnying skeleton. Just gated. The empirical wins are small but consistent across benchmarks. We'll dissect the math in [ch.12 (Activations)](../12-activations/).

### Mixture of Experts: K Replicated MLPs Plus A Router

If each token only needs a fraction of those detectors firing, why not store **K copies** of the MLP block and route each token to just two of them?

This is the **MoE** trick (Shazeer 2017; Mixtral 2023; DeepSeek-V3 2024). The `mlp_fc1` and `mlp_fc2` keys get replaced by `expert{e}.mlp_fc1` and `expert{e}.mlp_fc2`, plus a tiny `router` matrix that picks two experts per token. Mixtral 8x7B has K=8 experts per layer but activates 2 — 47 B params on disk, ~13 B active per token. The skeleton is unchanged; the trick is purely *which* `(W_1, W_2)` pair gets used.

## What To Remember

1. **The MLP block is six lines: copy residual, normalize, fatten via `mlp_fc1`, ReLU, skinny via `mlp_fc2`, add back to residual.** Same skeleton as the [attention half](../08-attention/), different middle.
2. **The 4× hidden dimension was picked by Vaswani et al. in 2017 with no published justification and never seriously revisited.** Modern variants (Llama, Mistral) use ~3.5× combined with SwiGLU.
3. **`mlp_fc1` rows are detectors; `mlp_fc2` columns are values.** Together they form a key-value memory of `4·n_embd` slots. Most of what a model "knows" lives in these two matrices.
4. **MLP holds ~2× the parameters of attention per layer in a vanilla transformer, and even more under GQA + SwiGLU.** In Llama 3 8B, ~3.75 B of the 8 B total parameters live in MLP blocks.
5. **No biases.** Just like every other [`linear()` call in microGPT](../04-linear/) — RMSNorm's learnable scale absorbs anything a bias would have done.
6. **SwiGLU and MoE are both refinements of these two matrices**, not redesigns of the block. The fattening-and-skinnying shape has survived every architectural revolution since 2017.

---

**Continue to** → [ReLU and Friends](../12-activations/) — we squashed the fattened vector with a `max(0, x)` and shrugged; the next chapter is about what the field tried instead, why Swish won, and the gating trick (SwiGLU) that turned two matrices into three.

