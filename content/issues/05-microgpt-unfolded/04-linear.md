---
title: "Linear, in Pure Python"
description: "The three-line `linear()` helper is the same operation every weight matrix in the model performs. Master this and `attn_wq`, `mlp_fc1`, and `lm_head` are all the same thing in different costumes."
topics: [transformer, linear-algebra]
tags: [microgpt, linear, matmul]
theme: teal
math: true
draft: false
date: 2026-05-14T02:48:52-04:00
issue: 5
weight: 40
techKind: primer
techNode: linear
header: 04-linear.webp
---

## Albuquerque, 1979

A 39-year-old mathematics professor at the University of New Mexico is teaching a numerical-analysis course and is *tired*. His name is **Cleve Moler**. The course requires students to write Fortran programs that call out to a library called **LINPACK** — a giant pile of routines, written by Moler and three colleagues earlier in the decade, for solving systems of linear equations on the era's supercomputers. The students are bright. The Fortran is not. By the time they have finished wrestling with `COMMON` blocks and column-major array layouts, the term is over and they have learned almost nothing about *linear algebra*.

So Moler does something out of character for a Fortran wizard. Over a single Christmas break, he writes — in Fortran, for portability — a small **interactive calculator** that lets a student type `A * B` at a prompt and get back the matrix product. No declarations. No `DIMENSION` statements. Just *matrix language*. He calls it **MATLAB**, short for "matrix laboratory", and gives it away to anyone who asks. Within five years it is on every applied-math desk in America, and within fifteen it has spun out into a company. The kernel of the whole thing — the operation that students typed most often, the one Moler had spent the previous decade hand-tuning in LINPACK — is matrix-vector multiplication.

You are about to spend the rest of this issue running that exact operation, by hand, on a list of sixteen floats. Forty-seven years and several Nobel prizes' worth of numerical-analysis labour are quietly waiting inside one three-line Python helper:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

This is the workhorse. Every weight matrix in microGPT — every single one of them — talks to its input through this function. There are no others. Master `linear()` and you have mastered, mechanically, **a hundred percent of the parameters** in the model.

## What The Three Lines Actually Do

Read the helper out loud.

> *For every row `wo` in `w`, zip that row with the input `x`, multiply elementwise, sum.*

That is a **dot product**, and we are doing one per row. Concretely: `w` is a list-of-lists with shape `(out_dim, in_dim)`. Each inner list `wo` has length `in_dim` — it is "one row of `w`", one neuron's worth of weights. Zipping it with `x` (which also has length `in_dim`) pairs each weight with its matching input feature. The `sum` collapses those `in_dim` products into a single scalar. We do that once per row, so the output is a list of length `out_dim`.

In math notation this is the matrix-vector product

$$
y_i = \sum_{j=0}^{\text{in\_dim} - 1} w_{ij} \, x_j \qquad \text{for } i = 0, \ldots, \text{out\_dim} - 1
$$

or, if you would rather see it as numpy:

```python
y = W @ x          # W.shape == (out_dim, in_dim); x.shape == (in_dim,); y.shape == (out_dim,)
```

Same answer. Same FLOPs. Same shape. The pure-Python version is just *slower and more explicit*. That is the trade we made when we wrote microGPT: every operation has to be readable as a loop, even if a real GPU would dispatch the same work as a single `cuBLAS` kernel call in nanoseconds.

A worked toy. Suppose `in_dim = 3`, `out_dim = 2`, and:

$$
W = \begin{bmatrix} 1 & 0 & -2 \\ 3 & 1 & 0 \end{bmatrix}, \qquad x = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}
$$

Then `linear(x, w)` walks two rows:

- Row 0: `1·4 + 0·5 + (-2)·6 = 4 + 0 - 12 = -8`
- Row 1: `3·4 + 1·5 + 0·6 = 12 + 5 + 0 = 17`

Output: `[-8, 17]`. Two dot products, six multiplies, four adds. That's it.

## "Wait, Is It x @ W Or x @ W.T?" — The Day Everyone Loses

There is a flavor of bug that costs every deep-learning engineer roughly one calendar day, exactly once, the first time they hit it. It looks like this. You wrote `torch.nn.Linear(in_features=16, out_features=64)` and you check `layer.weight.shape` and it says `torch.Size([64, 16])` — `(out, in)`, the same convention microGPT uses. Good.

Then you check what the `forward` method *actually does* in `torch/nn/functional.py`:

```python
def linear(input, weight, bias=None):
    # roughly: input @ weight.T + bias
    ...
```

Wait — `weight.T`? Why are we transposing? Because PyTorch defines `nn.Linear` so that you can pass it a **batched** input of shape `(batch, in)` and get back `(batch, out)`. To make the shapes line up with the row-major convention, the matmul is written as `x @ W.T`. The weight is stored as `(out, in)` so a *single row* is what one output neuron reads — which is great for caching — but the public-facing operation transposes it on the fly. The transpose is free; it's a stride-flip on a metadata field, not a copy.

microGPT keeps `(out, in)` storage and skips the transpose by writing the loop **one row at a time** — which is exactly what `for wo in w` does. No batched matmul, no `.T`, no confusion. The price: one Python `for` loop per output element. The Llama serving stack pays a few CUDA cycles for the transpose and gets, in exchange, a fused kernel that finishes ten billion multiplies in a millisecond. *Either layout is fine; mixing them in the same codebase is what burns a day.*

## How Many Times Per Token Does `linear()` Get Called?

Open the [cold-open listing](../01-cold-open/) again and count every place the word `linear` appears in the forward pass:

| Per layer | Calls | What they project |
|---|---|---|
| `attn_wq`, `attn_wk`, `attn_wv` | 3 | residual → Q, K, V |
| `attn_wo` | 1 | concatenated heads → residual |
| `mlp_fc1`, `mlp_fc2` | 2 | residual → MLP hidden → residual |

Six per layer. Plus a single `lm_head` at the very end. So:

$$
\text{calls per token} = 6 \cdot n_\text{layer} + 1
$$

For microGPT's toy `n_layer = 2`, that's **13 `linear()` calls per token**. For **Llama 3 8B**, with `n_layer = 32`, that's $6 \cdot 32 + 1 = \textbf{193}$ matrix-vector products *per token of output*. (Llama 3 also uses a gated MLP — SwiGLU — that has *three* matrices per MLP block instead of two, so the real count is 7·32 + 1 = 225. We'll keep it simple here and stick to the [ch.1 listing](../01-cold-open/) convention.)

Every one of those calls is the same three-line helper. Different `w`, different shape, identical Python code.

## Napkin Math: How Many FLOPs Per Token?

Each `linear(x, w)` call where `w` has shape `(out_dim, in_dim)` does `out_dim * in_dim` multiplies and roughly the same number of adds. By the deep-learning community's accounting convention, that's

$$
\text{FLOPs}(\texttt{linear}) \approx 2 \cdot \text{out\_dim} \cdot \text{in\_dim}
$$

For microGPT's toy model (`n_embd = 16`, 4× MLP fattening so the hidden dim is 64):

- Each attention projection: `2 · 16 · 16 = 512` FLOPs.
- Each MLP projection: `2 · 16 · 64 = 2048` FLOPs.
- Per layer: `4 · 512 + 2 · 2048 = 2048 + 4096 = 6144`.
- Two layers + a `lm_head` of `2 · 16 · 27 ≈ 864`: **about 13 kFLOPs per token**.

A modern CPU does this faster than you can blink.

Now scale to **Llama 3 8B**: `n_embd = 4096`, `n_layer = 32`, MLP hidden ≈ 14336 (it's not exactly 4×; Meta picked a number divisible by friendly factors). The four attention projections are each `4096 × 4096`, the two MLP matrices are roughly `14336 × 4096`. Per layer:

$$
4 \cdot (2 \cdot 4096^2) + 2 \cdot (2 \cdot 14336 \cdot 4096) \approx 134\text{M} + 235\text{M} = \mathbf{369 \text{ MFLOPs}}
$$

Multiply by 32 layers, add the `lm_head` projection to a 128K vocab (≈1 GFLOP on its own), and you land near

$$
32 \cdot 369\text{M} + 1\text{G} \approx \mathbf{12.8 \text{ GFLOPs per token}}
$$

(The actual figure depends on which weights you count; published estimates for Llama 3 8B forward-only inference cluster around **16 GFLOPs per generated token** once you include the residual adds, RMSNorms, and softmax kernels. Close enough.) Generate 100 tokens of response: about a **trillion floating-point operations**. On an H100 doing 1000 TFLOPs/s in FP16, that's a millisecond of math, *if you could feed the silicon weights fast enough* — which, spoiler, you cannot. The bottleneck is memory bandwidth, not compute, which is why [issue 03](../../03-sixteen-numbers/) exists at all.

## Where The FLOPs Actually Live

A look at the per-call cost in a single Llama-3-8B layer makes the asymmetry visible.

```pyplot {id="flops-per-linear-llama3" caption="FLOPs per linear() call inside one transformer layer of Llama 3 8B. The two MLP projections dominate because the hidden dimension is ~3.5x wider than the residual."}
n_embd = 4096
mlp_hidden = 14336   # Llama 3 8B value

calls = [
    ('attn_wq',  n_embd,    n_embd),
    ('attn_wk',  n_embd,    n_embd),
    ('attn_wv',  n_embd,    n_embd),
    ('attn_wo',  n_embd,    n_embd),
    ('mlp_fc1',  mlp_hidden, n_embd),
    ('mlp_fc2',  n_embd,     mlp_hidden),
]

names  = [c[0] for c in calls]
flops  = [2 * c[1] * c[2] / 1e6 for c in calls]   # MFLOPs per call
colors = ['#FF007F'] * 4 + ['#00A8A8'] * 2

fig, ax = plt.subplots(figsize=(8.5, 4))
ys = list(range(len(names)))
ax.barh(ys, flops, color=colors, edgecolor='#1A1A1A', linewidth=1.5)
ax.set_yticks(ys)
ax.set_yticklabels(names)
ax.invert_yaxis()
for i, f in enumerate(flops):
    ax.text(f + 2, i, f'{f:.1f} MFLOPs', va='center', fontsize=10)
ax.set_xlabel('MFLOPs per linear() call (one token, one layer)')
ax.set_title("Llama 3 8B: where the per-layer FLOPs go (pink = attention, teal = MLP)")
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(0, max(flops) * 1.25)
```

Each MLP projection is *almost twice* as expensive as all four attention projections combined. This holds at essentially every modern scale — the MLP block is where the parameters *and* the FLOPs live. When you read papers on **mixture-of-experts**, the entire motivation is to skip most of that pink-and-teal bar by routing tokens to only a few of K replicated MLP blocks. When you read papers on **flash-attention**, they speed up something that is not even in this chart — the per-head `Q·K` and `softmax(...) · V` computations, which scale with sequence length rather than with `n_embd²`.

## "Linear" Is A Lie We Tell Children

A pedagogical aside, then we move on.

A true *linear* map satisfies $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$. Multiplying by a matrix does that. **Adding a bias does not** — `Wx + b` is an *affine* map, not a linear one. Math people grit their teeth every time a deep-learning framework calls the affine layer `Linear`. PyTorch, JAX, Keras, all guilty. The convention is so entrenched it would take a generation to fix.

microGPT sidesteps the controversy by **omitting biases entirely**. Modern LLMs do too — Llama, Mistral, Gemma, and most of the rest of the 2023–2025 frontier ship with `bias=False` on every linear layer. The argument is empirical: with **RMSNorm** sitting in front of every linear (see [ch.5](../05-rmsnorm/)), the bias term has nothing to do that the norm's learnable scale isn't already doing. The model wastes a few thousand parameters per layer if you include biases, and the training-vs-inference shape of the layer becomes slightly fiddlier. Why bother?

So `linear()` in microGPT is, strictly speaking, *actually* linear. The textbook would approve.

## The Same Matrix, Three Costumes

The reason the title of this issue is "microGPT *Unfolded*" rather than "microGPT *Explained*" is that the same operation keeps showing up wearing different clothes. Once you see `linear()` for what it is, the entire forward pass simplifies.

| Weight name | Shape `(out, in)` | What `linear` is "really" doing |
|---|---|---|
| `attn_wq` | `(n_embd, n_embd)` | Asking "what am I looking for?" |
| `attn_wk` | `(n_embd, n_embd)` | Stamping "this is what I am" |
| `attn_wv` | `(n_embd, n_embd)` | Packaging "this is what I'll send if you call on me" |
| `attn_wo` | `(n_embd, n_embd)` | Mixing the heads' outputs back into the residual |
| `mlp_fc1` | `(4·n_embd, n_embd)` | "Fattening" — projecting into a wider workspace |
| `mlp_fc2` | `(n_embd, 4·n_embd)` | "Skinnying" — collapsing back to the residual width |
| `lm_head` | `(vocab_size, n_embd)` | Scoring every vocabulary token against the residual |

Seven costumes, one actor. The *only* thing that distinguishes `attn_wq` from `mlp_fc1` from `lm_head` is what *training* taught the entries of `w` to be. The function call site is identical. If you understand `linear()` and you understand what its `w` was trained to encode, you understand the whole forward pass.

(Some models also "tie" `lm_head` to `wte` — they reuse the same `(vocab_size, n_embd)` weight matrix for both directions, saving a chunk of parameters. See [ch.3 embeddings](../03-embeddings/) for the tying discussion. microGPT keeps them separate to keep the listing legible.)

## A One-Paragraph Foreshadow For Issue 03

Every entry of every `w` in the table above is, in a freshly trained model, a 16-bit or 32-bit float. There are billions of them. They are the bulk of the model's disk footprint and the bulk of the bytes the GPU has to read from HBM on every forward pass. **Quantization** is the art of replacing each weight matrix with a much smaller representation — 8-bit integers, 4-bit codes, even 2-bit lookup tables — such that `linear(x, w_quantized)` returns nearly the same answer as `linear(x, w_full)`. *Every word* of [issue 03 — Sixteen Numbers](../../03-sixteen-numbers/) is a story about how to shrink the `w` argument to this function without breaking what the model does. The function itself never changes. The numbers do.

## What To Remember

1. **`linear(x, w)` is matrix-vector multiplication, written as one dot product per row of `w`.** `wo` is "one row" — one neuron's weights. The output has length `out_dim = len(w)`.
2. **`w.shape == (out_dim, in_dim)`** in microGPT. PyTorch's `nn.Linear` stores it the same way but exposes it through `x @ W.T`. Mixing the conventions costs a day, once.
3. **Six `linear()` calls per layer, plus one `lm_head`.** For Llama 3 8B's 32 layers that's ~200 calls per token, ~16 GFLOPs of arithmetic, more than a trillion FLOPs per 100 tokens of response.
4. **MLP projections cost ~2× the attention projections** in modern architectures, because the hidden dimension is wider than `n_embd`. This is why MoE and SwiGLU papers all aim at the MLP block.
5. **No biases in microGPT** — and, in 2026, in most frontier LLMs. RMSNorm's learnable scale absorbs whatever a bias would have done.
6. **Every weight matrix in the model is the same operation in a different costume.** `attn_wq`, `mlp_fc1`, and `lm_head` are mechanically identical; only the contents of `w` differ. This is why [quantizing one weight matrix](../../03-sixteen-numbers/) is the same problem as quantizing all of them.

---

**Continue to** → [RMSNorm, Not LayerNorm](../05-rmsnorm/) — every `linear()` call in microGPT is preceded by a normalisation step, and the modern choice is the one with fewer moving parts than the original.

