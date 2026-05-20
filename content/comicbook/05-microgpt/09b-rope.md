---
title: "RoPE: position lives in the angle, not in the vector"
short_title: "RoPE"
description: "Replacing microGPT's `wpe` lookup with Rotary Position Embedding — derived from one observation at a time: position belongs in the geometry of `q` and `k`, not as an additive offset to the embedding."
blurb:
  - "wpe[16] is an `IndexError`. RoPE has no table at all — position is a rotation angle, defined for every integer for free."
  - "Two failed attempts (reserve dims, add a position vector) and the criterion they cough up: the score must depend only on m − n."
  - "One trig identity (`cos(m − n) = cos m cos n + sin m sin n`) is the entire mathematical core of RoPE."
  - "Eleven lines of Python replace a `block_size × n_embd` learned table. The KV cache stores already-rotated keys; no future re-rotation."
topics: [transformer, attention, positional-encoding]
tags: [microgpt, rope, rotary, position, attention]
theme: cream
math: true
draft: false
date: 2026-05-19T10:00:00-04:00
issue: 5
weight: 95
techKind: mainline
techNode: rope
header: default.webp
---

## The Line That Crashes

Open the microGPT [cold-open listing](../01-cold-open/) and look at its first two lines, the way it used to be:

```python
# the old, pre-RoPE microGPT
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

Two lookups. `wte[token_id]` is the model's dictionary of what a token *means* — see [embeddings](../03-embeddings/). `wpe[pos_id]` is, by contrast, the model's dictionary of what it *feels like to be at a position*: a `block_size × n_embd` grid of floats where each row is "the vector you add when you're at position $k$." Sixteen rows in microGPT, because `block_size = 16` in the toy. Sixteen rows, trained, frozen — and then the table just *ends*:

```python
>>> wpe[16]
IndexError: list index out of range
```

That `IndexError` is the entire drama in one line. The model has no concept of "position 16" because no row of `wpe` corresponds to it. Past `block_size`, there is nothing to look up. Every "long-context" trick the field has invented since 2021 — from 1M-token Gemini to the YaRN scaling we will see in [issue 4](/comicbook/04-long-context-bench/) — exists to keep this line from raising.

So `wpe` is broken at the edges. The natural questions are: **why is it here at all, and what would a better mechanism look like?** Both questions have to be answered before we can fix anything — we can't replace a thing we don't understand. This article answers both, and the answer to the second is {{< wiki "rope" >}}RoPE{{< /wiki >}}.

```pyplot {id="rope-01-wpe-cliff" caption="THE WPE EXTRAPOLATION CLIFF. TWENTY-THREE TRAINED ROWS, THEN NOTHING. THE MODEL HAS NO ROW TO LOOK UP FOR POSITION 23."}
np.random.seed(0)
block_size, n_embd = 16, 24
extended = 28

trained = np.random.normal(0, 0.1, size=(block_size, n_embd))
full = np.full((extended, n_embd), np.nan)
full[:block_size] = trained

fig, ax = plt.subplots(figsize=(7.5, 4.4))
cmap = plt.get_cmap('RdBu_r').copy()
cmap.set_bad(color='#dddddd')
im = ax.imshow(full, aspect='auto', cmap=cmap, vmin=-0.3, vmax=0.3)
ax.axhline(block_size - 0.5, color='#1A1A1A', linewidth=1.6)
ax.text(n_embd / 2, (block_size + extended) / 2,
        'no data — IndexError',
        ha='center', va='center', fontsize=12, color='#555555', style='italic')
ax.set_xlabel('embedding dimension (0 .. n_embd-1)')
ax.set_ylabel('position id')
ax.set_title("learned position table wpe[pos_id] — trained for positions 0..15")
fig.colorbar(im, ax=ax, shrink=0.7, label='value')
```

## Why Position Has To Be Injected Somewhere

Strip `wpe` out for a moment. Leave only token embeddings, and watch what attention sees.

```python
import math, random
random.seed(0)
d = 4
wq = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]
wk = [[random.gauss(0, 0.3) for _ in range(d)] for _ in range(d)]

def linear(x, w): return [sum(wi*xi for wi,xi in zip(row,x)) for row in w]
def dot(a, b):    return sum(ai*bi for ai,bi in zip(a,b))
def score(a, b):  return dot(linear(a, wq), linear(b, wk))   # one q·k

# Pretend these come out of wte for tokens 'd', 'o', 'g'.
d_, o_, g_ = [0.2, -0.5, 0.1, 0.4], [0.6, 0.3, -0.2, 0.1], [0.9, -0.1, 0.0, 0.2]

# Two sequences, same tokens in different orders.
seq1 = [d_, o_, g_]   # "dog"
seq2 = [g_, o_, d_]   # "god"

print(sorted(round(score(a, b), 4) for a in seq1 for b in seq1))
print(sorted(round(score(a, b), 4) for a in seq2 for b in seq2))
```

```
[-0.0784, -0.0124, -0.0091, 0.0066, 0.0721, 0.1084, 0.1263, 0.1911, 0.2109]
[-0.0784, -0.0124, -0.0091, 0.0066, 0.0721, 0.1084, 0.1263, 0.1911, 0.2109]
```

Identical. The *multiset* of attention scores between tokens is the same for `"dog"` and `"god"` — because attention is built from `q · k` between embeddings, and the embedding for `'d'` is the same whether `'d'` is at the start or at the end. Without a position signal, {{< wiki "attention" >}}attention{{< /wiki >}} literally cannot tell `"dog"` from `"god"`. Fine for a bag of words; catastrophic for language.

That is what `wpe` is *for*: to break the permutation symmetry. The bag-of-tokens view gets enriched with "and tokens at different positions look different to the network" — which is exactly what `wpe[pos]` accomplishes, by adding a different vector to the embedding at each position.

So the real question is not *whether* to inject position, but *how*. The next two sections walk through the two most obvious ways someone trying to invent this from scratch would try. Both are honest dead ends — and the precise way they fail is what lets us, a few sections later, write down exactly what the right answer has to look like, and then build it.

## Attempt 1: Reserve Some Embedding Dimensions For Position

If the issue is that the embedding currently carries zero position info, the most obvious fix is to carve out a few of the `n_embd` channels and fill them with a function of `pos_id`. With `n_embd = 24`:

- Dimensions 0..19 — token content (a 20-dim slice of `wte[tok]`).
- Dimensions 20..23 — position only, filled with some function of `pos_id`.

For the position slots we have to put *something* in. The simplest option, the raw integer:

```python
x[20:24] = [pos_id, 0, 0, 0]
```

Magnitudes blow up. Token-embedding entries live in the ±0.3 range; slamming a `22` into one channel makes that channel's contribution to every downstream dot product orders of magnitude larger than anything the token side puts in. The model's only response is to drive the weights on that channel to zero, throwing away the signal we put in. **Lesson 1:** the position values need to be bounded — on the same scale as the embedding.

Fine, bound them. Pick a fixed vector function, values in a reasonable range — say `pos_vec(p) = [p/N, sin(p), cos(p), p²/N²]`. Now the four slots stay $\mathcal{O}(1)$.

Does this work? Inside the first attention layer, the score `q·k` for query at position $m$ and key at position $n$ decomposes, when content and position live in disjoint dimension blocks, into:

```
<Wq · x_m, Wk · x_n> = <Wq · content_m, Wk · content_n>     ← what we want
                      + <Wq · pos_vec(m), Wk · pos_vec(n)>   ← the position term
```

Plot the second term — the "position-only" contribution — as a function of $(m, n)$:

```pyplot {id="rope-02-attempt1-absolute" caption="ATTEMPT 1: THE POSITION-ONLY DOT PRODUCT OVER (M, N). BANDS ARE NOT DIAGONAL — THE SCORE DEPENDS ON THE ABSOLUTE PAIR, NOT JUST THE GAP M − N."}
import math
np.random.seed(0)
pos_dim = 4

def pos_vec(p):
    return np.array([p / 20.0, math.sin(p * 0.3), math.cos(p * 0.3), (p**2) / 400.0])

rng = np.random.default_rng(1)
Wq = rng.normal(0, 0.5, size=(pos_dim, pos_dim))
Wk = rng.normal(0, 0.5, size=(pos_dim, pos_dim))

P = 24
scores = np.zeros((P, P))
for m in range(P):
    for n in range(P):
        q = Wq @ pos_vec(m)
        k = Wk @ pos_vec(n)
        scores[m, n] = float(np.dot(q, k))

fig, ax = plt.subplots(figsize=(6.0, 5.0))
m_abs = np.max(np.abs(scores))
im = ax.imshow(scores, origin='lower', cmap='RdBu_r', vmin=-m_abs, vmax=m_abs)
ax.set_xlabel('key position n')
ax.set_ylabel('query position m')
ax.set_title('attempt 1: q_pos · k_pos\nbands are not diagonal — score depends on absolute m, n')
fig.colorbar(im, ax=ax, shrink=0.85, label='dot-product score')
```

The bands are not diagonal. The position-term contribution to `score(m, n)` depends on the absolute pair $(m, n)$, not just the gap $m - n$. So even with position perfectly isolated and bounded, the score for a query-key pair *three slots apart* changes depending on where in the sequence that pair lives. $\text{score}(0, 3) \neq \text{score}(10, 13) \neq \text{score}(20, 23)$.

This is the deep failure. Think about what language actually cares about: the pair "the cat" plays the same syntactic role at the start of a document and on page 30. The *relationship* — query three slots from key — is what carries grammatical meaning; the absolute slot numbers are accidents of where in the document we happen to be reading. Any position mechanism that lets the absolute slot numbers leak into attention scores is forcing the model to spend capacity learning that the same relationship at different absolute positions should mean the same thing.

So we now have a precise constraint, born from this attempt's failure. **Lesson 2 — the criterion:**

{{% pullquote type="aha" %}}
The attention score between a query at position $m$ and a key at position $n$ must depend only on the tokens and on $m - n$. Nothing about the absolute values of $m$ or $n$ may leak into the score.
{{% /pullquote %}}

Plus a smaller lesson: **Lesson 3** — those four reserved channels are dead weight for token content. 4 of 24 dims (≈17%) tax in every layer.

We now have the yardstick. Every subsequent attempt is judged against it.

## Attempt 2: Add A Position Vector To The Whole Embedding

Lesson 3 (don't waste dims) is easy to address — don't isolate. Add a full-dimensional position vector to the token embedding, so every channel carries a little of both:

```python
x = wte[tok_id] + position_vector(pos_id)
```

This is exactly what GPT-2 (and the old microGPT) did, with `position_vector = wpe[pos_id]` — a learned table. It is also exactly what the original 2017 Transformer paper did, with a fixed function:

```python
def sinusoidal_pe(pos, d):
    out = []
    for i in range(d // 2):
        theta = 10000.0 ** (-2 * i / d)
        out.append(math.sin(pos * theta))
        out.append(math.cos(pos * theta))
    return out
```

Plot `sinusoidal_pe(p, 24)` for $p \in 0..63$:

```pyplot {id="rope-03-sinusoidal-pe" caption="SINUSOIDAL POSITION ENCODING. LEFT DIMENSIONS WIGGLE FAST (LARGE θ); RIGHT DIMENSIONS BARELY MOVE OVER 64 POSITIONS."}
import math
d, P = 24, 64

pe = np.zeros((P, d))
for p in range(P):
    for i in range(d // 2):
        theta = 10000.0 ** (-2 * i / d)
        pe[p, 2 * i] = math.sin(p * theta)
        pe[p, 2 * i + 1] = math.cos(p * theta)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
im = ax.imshow(pe, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xlabel('embedding dimension (0 .. d-1)')
ax.set_ylabel('position id')
ax.set_title('sinusoidal_pe(pos, d=24)\nleft dims = high frequency, right dims = low frequency')
fig.colorbar(im, ax=ax, shrink=0.8, label='value')
```

Left dimensions wiggle fast (small $i \Rightarrow$ large $\theta_i$); right dimensions barely move over 64 positions (large $i \Rightarrow$ tiny $\theta_i$). Bounded, parameter-free, defined for every integer $p$. So sinusoidal handles Lessons 1 and 3 cleanly, and the learned variant handles Lesson 3 (but not Lesson 1's spirit — it can drift to whatever magnitudes training pushes it to).

{{% callout type="note" %}}
**A note on what "learned" actually means here.** In GPT-2, `wpe` is just an `nn.Embedding(block_size, n_embd)` — random small Gaussians at init, then *trained end-to-end alongside every other matrix in the network*. Every batch, the cross-entropy loss backpropagates a gradient through `wpe[pos_id]` for every position visited in that batch, and the optimizer updates the row. There is no separate pretraining for `wpe`, and there is no separate pretraining for `wte` either: this is sometimes a surprise to readers carrying the older word2vec / GloVe mental model, where token embeddings were pretrained on a standalone objective and then *frozen* as inputs to a downstream LSTM. From "Attention Is All You Need" (2017) onward, both `wte` and `wpe` sit on the same gradient flow as every other learned matrix.

**Why did GPT-2 pick learned over sinusoidal?** Mostly historical contingency. Vaswani et al. tried both and reported "nearly identical results" on translation; GPT-1 (2018) went with learned for the marginal flexibility — the model carves its own representation rather than being constrained to fixed geometric frequencies — and GPT-2 inherited that choice. The `IndexError` cliff was not anyone's top problem in 2019, because GPT-2 was pretrained at a fixed `block_size = 1024` and used at the same length. The cliff only started to hurt years later, when practitioners wanted to extend the context window of an already-trained model post-hoc, and that pressure is a big part of what drove the field toward RoPE.
{{% /callout %}}

Now apply Lesson 2 — the criterion — to the additive scheme. Expand the attention dot product:

$$
\langle W_Q (\text{content}_m + \text{PE}(m)),\; W_K (\text{content}_n + \text{PE}(n)) \rangle
$$

$$
= \underbrace{\langle W_Q\text{content}_m, W_K\text{content}_n\rangle}_{\text{content↔content (wanted)}}
+ \underbrace{\langle W_Q\text{content}_m, W_K\text{PE}(n)\rangle}_{\text{content↔position}}
+ \underbrace{\langle W_Q\text{PE}(m), W_K\text{content}_n\rangle}_{\text{position↔content}}
+ \underbrace{\langle W_Q\text{PE}(m), W_K\text{PE}(n)\rangle}_{\text{position↔position}}
$$

Four cross terms. Only the first is what attention is conceptually asking for. The other three are bookkeeping the model must learn to denoise. And the *fourth* term — purely position — has the same problem we just diagnosed in Attempt 1: it is a dot product of two vectors indexed by $m$ and $n$, and there is no general reason for such a thing to depend only on $m - n$. For both the learned `wpe[m]·wpe[n]` flavour and the sinusoidal `PE(m)·PE(n)` flavour, the value depends on the absolute pair $(m, n)$. **Criterion violated.**

On top of that, the learned-`wpe` flavour still has the `IndexError` cliff: `wpe[16]` doesn't exist. The sinusoidal flavour at least returns *a* number for any $p$, but the actual extrapolation behavior of attention scores past the training range is no better than chance.

So additive injection fixes the cosmetic complaints of Attempt 1, but inherits its core failure (criterion not met) *and* introduces three additional cross-terms that aren't position information at all. **Lesson 4:** additive injection of position *into the embedding* cannot satisfy the criterion. The position information has to be injected somewhere else.

## The Reframe — Inject Position Inside Attention, On q And k

Where else *is* there? Trace the data flow: an embedding goes in, gets projected by [`attn_wq` and `attn_wk`](../06-qkv-projections/) into a query and a key, those two get dot-producted to make the attention score. Three places to inject:

- **(a)** on the embedding (additive — just ruled out);
- **(b)** on the projection matrices `attn_wq`, `attn_wk`;
- **(c)** on `q` and `k` themselves after projection but before the dot product.

(b) is unappealing: `attn_wq` and `attn_wk` are the model's only learned token-comparison machinery, and making them position-dependent means a separate $(W_Q, W_K)$ per position — back to the parameter cliff at `block_size`.

(c) is the only door left. We want a transformation $T(\cdot, p)$ such that

$$
\langle T(q, m), T(k, n)\rangle \;\;\text{depends only on}\;\; m - n \;\;(\text{and on}\; q, k).
$$

This is a much sharper question than we've ever asked, because we've ruled everything else out. Is there even any such transformation? The next section says yes, and constructs the simplest one.

## When Does A Dot Product Depend Only On The Angular Difference?

Look at one of the most familiar trig identities:

$$
\cos(m - n) \;=\; \cos m \cdot \cos n + \sin m \cdot \sin n
$$

The left-hand side depends only on $m - n$. The right-hand side is a sum of products, where each product factors into a function of $m$ only times a function of $n$ only. The right-hand side knows $m$ and $n$ individually — yet they are equal, so the whole thing only ever reveals $m - n$.

The structure of the right-hand side is a 2D inner product. Define

$$
u(p) = \big[\cos p,\;\sin p\big].
$$

Then $\langle u(m), u(n)\rangle = \cos m \cos n + \sin m \sin n = \cos(m - n)$.

$u(p)$ is a unit vector on the circle, pointed at angle $p$. The dot product of two such vectors is determined by the angle *between them*:

```pyplot {id="rope-04-unit-circle" caption="THREE (M, N) PAIRS WITH THE SAME GAP M − N = 1.2. THE VECTORS POINT IN DIFFERENT ABSOLUTE DIRECTIONS, BUT THE ANGLE BETWEEN THEM IS IDENTICAL, SO THE DOT PRODUCT IS IDENTICAL."}
import math
pairs = [(0.4, -0.8), (2.0, 0.8), (-1.5, -2.7)]  # all with m - n = 1.2

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
for ax, (m, n) in zip(axes, pairs):
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#cccccc', linewidth=1)
    ax.axhline(0, color='#eeeeee', linewidth=0.8)
    ax.axvline(0, color='#eeeeee', linewidth=0.8)
    ax.annotate('', xy=(math.cos(m), math.sin(m)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#FF007F', lw=2))
    ax.annotate('', xy=(math.cos(n), math.sin(n)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#00A8A8', lw=2))
    ax.text(math.cos(m) * 1.18, math.sin(m) * 1.18, f'u(m={m:.1f})',
            color='#FF007F', ha='center', fontsize=9)
    ax.text(math.cos(n) * 1.18, math.sin(n) * 1.18, f'u(n={n:.1f})',
            color='#00A8A8', ha='center', fontsize=9)
    arc_theta = np.linspace(min(m, n), max(m, n), 60)
    ax.plot(0.3 * np.cos(arc_theta), 0.3 * np.sin(arc_theta), color='#1A1A1A', lw=1.2)
    ax.text(0.62 * math.cos((m + n) / 2), 0.62 * math.sin((m + n) / 2),
            'm − n = 1.2', fontsize=9, ha='center')
    ax.text(0, -1.55, f'<u(m), u(n)> = cos(m−n) = {math.cos(m - n):.3f}',
            ha='center', fontsize=9, color='#444444')
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.65, 1.4)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle('same gap m − n ⇒ same dot product', y=1.02)
```

So an inner product of unit-circle vectors at angles $m$, $n$ is provably a function of $m - n$ only. The dot product literally cannot reveal the individual angles, no matter how you interrogate it. The criterion is satisfied — but only for *unit vectors at fixed angles*, which is not what `q` and `k` are. The next section closes that gap.

## Apply It To Attention: Rotate q And k

`q` and `k` are arbitrary vectors, not unit-circle vectors. But we don't have to *replace* them with anything — we can *rotate* them. Rotation by an angle $\theta$ is a linear map (the matrix $R(\theta)$) with the same algebraic property the trig identity gives us:

$$
R(m)^\top \, R(n) \;=\; R(n - m).
$$

So if we rotate `q` by $m$ and `k` by $n$, their inner product is

$$
\langle R(m)\,q,\; R(n)\,k\rangle \;=\; q^\top R(m)^\top R(n)\,k \;=\; q^\top R(n - m)\,k.
$$

The two rotations collapse into a single rotation by $(n - m)$. The inner product depends only on the relative offset. No additive cross-terms. No parameters spent. Position is no longer a thing you add to the embedding — it is an *orientation* you give to `q` and `k`.

Demonstrate in 2D:

```python
import math

def rotate(x, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [x[0]*c - x[1]*s, x[0]*s + x[1]*c]

q = [1.0, 0.3]
k = [0.4, 0.7]

for m, n in [(0, 0), (2, 0), (5, 3), (47, 45)]:
    qm = rotate(q, m * 1.0)
    kn = rotate(k, n * 1.0)
    s = qm[0]*kn[0] + qm[1]*kn[1]
    print(f"m={m:2d}, n={n:2d}, m-n={m-n:+d}: score = {s:+.6f}")
```

```
m= 0, n= 0, m-n=+0: score = +0.610000
m= 2, n= 0, m-n=+2: score = +0.273543
m= 5, n= 3, m-n=+2: score = +0.273543
m=47, n=45, m-n=+2: score = +0.273543
```

Every row with $m - n = 2$ produces the exact same score, including the row where the absolute positions sit far past `block_size = 16`. Sweep $(m, n)$ over a full grid to see the same property as a heatmap:

```pyplot {id="rope-05-2d-score-bands" caption="2D RoPE: ROTATED-q DOTTED WITH ROTATED-k IS CONSTANT ALONG EVERY DIAGONAL M − N = CONST. THE 2D MAP (M, N) → SCORE HAS COLLAPSED ONTO A 1D FUNCTION OF M − N."}
import math
q = np.array([1.0, 0.3])
k = np.array([0.4, 0.7])
theta = 1.0
P = 30

def rotate(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c])

scores = np.zeros((P, P))
for m in range(P):
    for n in range(P):
        scores[m, n] = float(np.dot(rotate(q, m*theta), rotate(k, n*theta)))

fig, ax = plt.subplots(figsize=(6.0, 5.0))
m_abs = np.max(np.abs(scores))
im = ax.imshow(scores, origin='lower', cmap='RdBu_r', vmin=-m_abs, vmax=m_abs)
ax.set_xlabel('key position n')
ax.set_ylabel('query position m')
ax.set_title('rotated-q · rotated-k\nconstant along diagonals ⇒ depends only on m − n')
fig.colorbar(im, ax=ax, shrink=0.85, label='dot-product score')
```

Diagonal bands. The score is constant along every diagonal $m - n = \text{const}$. The 2D map $(m, n) \mapsto \text{score}$ has collapsed onto a 1D function of $m - n$, exactly as the criterion demanded. This is RoPE in 2D — about ten lines of code, no parameters, criterion fully satisfied.

There is one immediate problem with the 2D-only version, which the next section fixes.

## One Frequency Aliases — Use Many

In the demo above the angular speed was $\theta = 1$ radian per position. That means $R(0) = R(2\pi)$ — positions $0$ and $2\pi \approx 6.28$ produce *identical* rotations. Past one full revolution, the score function repeats.

```pyplot {id="rope-06-multiscale-clock" caption="LEFT: SINGLE-FREQUENCY SCORE WRAPS EVERY 2π ≈ 6.28 POSITIONS. RIGHT: WITH HEAD_DIM=6 (THREE PAIRS), EACH PAIR ROTATES AT ITS OWN SPEED — A MULTI-HAND CLOCK THAT IS UNIQUE ACROSS THOUSANDS OF POSITIONS."}
import math
head_dim = 6
P = np.arange(0, 60)
thetas = [10000.0 ** (-2 * i / head_dim) for i in range(head_dim // 2)]
angles = np.array([[(p * t) % (2 * np.pi) for p in P] for t in thetas])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

q = np.array([1.0, 0.3])
k = np.array([0.4, 0.7])

def rotate(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c])

gaps = np.arange(0, 60)
single_scores = np.array([float(np.dot(rotate(q, g * 1.0), k)) for g in gaps])
axes[0].plot(gaps, single_scores, color='#FF007F', linewidth=2)
for k_wrap in range(1, 10):
    axes[0].axvline(k_wrap * 2 * np.pi, color='#bbbbbb', lw=0.7, ls=':')
axes[0].set_title('single frequency θ=1\nscore wraps every 2π ≈ 6.28 positions')
axes[0].set_xlabel('gap m − n')
axes[0].set_ylabel('attention score')
axes[0].spines[['top', 'right']].set_visible(False)

colors = ['#FF007F', '#00A8A8', '#FFD700']
periods = [2 * np.pi / t for t in thetas]
for i, (a, t, period, color) in enumerate(zip(angles, thetas, periods, colors)):
    axes[1].plot(P, a, color=color, linewidth=2,
                 label=f'pair {i}: θ={t:.4f}, period ≈ {period:.0f}')
axes[1].set_title('multi-scale clock: head_dim=6 ⇒ 3 frequency pairs')
axes[1].set_xlabel('position')
axes[1].set_ylabel('rotation angle (mod 2π)')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].spines[['top', 'right']].set_visible(False)
fig.tight_layout()
```

Left panel: the dot product $\langle R(g \cdot 1)\,q, k\rangle$ as a function of the gap $g$. It oscillates and aliases every $\approx 6.28$ positions — position 0 looks like position 6 looks like position 13. A 2D-only scheme can only distinguish positions inside one revolution.

The fix is borrowed straight from sinusoidal encoding: don't use one frequency, use many. Pair up consecutive dimensions of the `head_dim`-dimensional `q` (and the same pairs of `k`) and give each pair its own angular speed $\theta_i$. With `head_dim = 6`, three pairs, and the standard schedule

```python
theta_i = 10000.0 ** (-2 * i / head_dim)
```

gives:

| pair `i` | $\theta_i$ | period (positions per full turn) |
|----------|-----------|----------------------------------|
| 0 | 1.0000 | ~6.3 |
| 1 | 0.0464 | ~135 |
| 2 | 0.0022 | ~2920 |

Right panel above: angle of each pair vs position. Pair 0 is the second hand — full circle every six positions, exquisite local resolution but wraps fast. Pair 2 is the hour hand — over 60 positions it barely moves, encoding the coarse "where in the document am I" signal. Pair 1 in between. The joint reading of all three hands is unique across thousands of positions, even though each hand alone wraps.

Each pair contributes its own diagonal-banded score (each depends only on $m - n$), and the sum across pairs also depends only on $m - n$. The criterion survives addition.

## The Eleven-Line Helper

That entire construction is the eleven lines of `rope()`:

```python
def rope(x, pos):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = 10000.0 ** (-2 * i / d)
        c, s = math.cos(pos * theta), math.sin(pos * theta)
        x0, x1 = x[2 * i], x[2 * i + 1]
        out[2 * i]     = x0 * c - x1 * s
        out[2 * i + 1] = x0 * s + x1 * c
    return out
```

Pair $i$ is rotated by angle $\text{pos} \cdot \theta_i$. That is it.

Verify, on a 6-dimensional `q` and `k`:

```python
>>> def dot(a, b): return sum(x*y for x, y in zip(a, b))
>>> q = [1.0, 0.5, -0.3, 0.7, 0.2, -0.1]
>>> k = [0.4, -0.2, 0.6, 0.1, -0.5, 0.3]
>>> for m, n in [(2, 0), (5, 3), (10, 8), (50, 48)]:
...     print(f"m={m:2d}, n={n:2d}: score = {dot(rope(q, m), rope(k, n)):.6f}")
m= 2, n= 0: score = -0.769759
m= 5, n= 3: score = -0.769759
m=10, n= 8: score = -0.769759
m=50, n=48: score = -0.769759
```

Same gap → same score, including at position 50, far past the trained `block_size = 16`. **The forward pass is *defined* there.** Whether the trained model *generalizes* there is a separate empirical question — but the architecture no longer has a hard wall.

## Where It Plugs Into microGPT

RoPE displaces the additive `wpe` step. Where `wpe[pos]` used to be added to `wte[tok]` before the transformer, now there is no position addition at the input — position is injected *inside* attention, after `q` and `k` are projected, before their dot product:

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])

q_rot, k_rot = [], []
for h in range(n_head):
    hs = h * head_dim
    q_rot.extend(rope(q[hs:hs+head_dim], pos_id))
    k_rot.extend(rope(k[hs:hs+head_dim], pos_id))

keys[li].append(k_rot)     # KV cache stores the *rotated* key
values[li].append(v)       # v is not rotated
```

Three details worth flagging:

- **Per-head application.** RoPE rotates `head_dim`-sized chunks, not `n_embd`-sized vectors. Each [head](../09-multi-head/) gets its own copy of the same $\theta_i$ ladder.
- **Only `q` and `k` are rotated.** Values carry token *content*, not the comparison logic — and RoPE only needs to act where comparisons happen.
- **The {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} stores the already-rotated key.** The cached `k_rot` was rotated by *its own* position at the time it was computed; it never needs re-rotation. New queries at later positions are rotated by their own positions and dot-producted against these cached, pre-rotated keys, and the math automatically produces the right relative-position score.

That last bullet matters at scale. In a 1M-context serving stack, you do not want to revisit a million cached vectors and re-multiply them by anything; you want a stream-only append cache. RoPE is exactly that — the rotation is paid once, at write time, then forgotten.

## Scoreboard

Tally the four lessons from the failed attempts against the two candidates that have actually shipped — `wpe` (old microGPT, GPT-2/3, BERT) and RoPE (Llama 1/2/3, Mistral, Qwen, DeepSeek, GPT-NeoX, …):

| | old `wpe` | RoPE |
|---|---|---|
| Position-related parameters | `block_size × n_embd` | 0 |
| Forward pass defined at `pos ≥ block_size`? | no (`IndexError`) | yes |
| Cross-terms in attention's `q·k` | 4 (one wanted, three junk) | 1 |
| Criterion: score depends only on $m − n$? | no — leaks absolute $(m, n)$ | yes — exact |
| KV-cache-friendly? | yes (additively baked in) | yes (geometrically baked in) |

## The Hero Behind The Math

The math is one trig identity old, but it sat unused in machine learning for ten years. In late 2020, a researcher at Zhuiyi Technology in Shenzhen named **Su Jianlin** was tinkering with positional encodings on his blog `kexue.fm`. He had been irritated by the same four cross-terms we derived above — he kept writing them out and asking why the position information needed three of them to be junk. By the time he posted the RoPE preprint in April 2021, he had also noticed that the trick has a clean Euler-formula derivation in the complex plane:

$$
(q \, e^{i m \theta}) \cdot \overline{(k \, e^{i n \theta})} \;=\; q k\,e^{i(m-n)\theta}.
$$

Multiply complex numbers, conjugate one, watch the absolute angles vanish. The 2D rotation matrix is just $e^{i\theta}$ written in real coordinates. The eleven-line helper is just Su's complex-plane move, rendered as `math.cos`/`math.sin`.

For two years RoPE sat in a paper most Western labs hadn't read. Then in early 2023 Meta's Llama team picked it up; by the end of 2023 every open-weights model worth running had switched. The deeper reason is that RoPE turned out to be the natural vector to *stretch* — extending the context window of an already-trained model means evaluating the same rotation at a slightly different angle, which is what NTK-aware scaling, position interpolation, and YaRN all do. We will come back to that story in [issue 4 ch.2](/comicbook/04-long-context-bench/02-context-window/) and [ch.11](/comicbook/04-long-context-bench/11-context-rot-fix/).

## What To Remember

1. **Position lives in the geometry of `q` and `k`, not as an additive offset to the embedding.** RoPE removes `wpe` from the input and rotates `q` and `k` instead, per head, per pair of dimensions.
2. **The criterion: score must depend only on $m - n$.** Attempt 1 fails it by leaking absolute positions; Attempt 2 (additive PE, learned or sinusoidal) also fails it through the position↔position cross-term.
3. **The trig identity $\cos(m - n) = \cos m \cos n + \sin m \sin n$ is the whole core.** Rotation matrices have the same algebraic property: $R(m)^\top R(n) = R(n - m)$. The dot product of rotated `q` and rotated `k` automatically depends only on the gap.
4. **Use many frequencies, not one.** Pair up consecutive dimensions of the `head_dim` vector and give each pair its own angular speed $\theta_i = 10000^{-2i/\text{head\_dim}}$. A single frequency aliases every $2\pi$; a multi-frequency clock is unique across the trained range.
5. **The KV cache stores already-rotated keys.** The rotation is paid once at write time. New queries are rotated by their own position and dot-producted against the cached vectors — the math automatically gives the right relative-position score.
6. **`block_size` is no longer a hard wall.** RoPE is defined for every integer position. Long-context tricks (NTK scaling, YaRN, position interpolation) all operate by rescaling the angle, not by inventing new vectors.

---

**Continue to** → [The Residual Stream](../10-residual-stream/) — now that position has been moved out of the embedding entirely, we can look at what the residual highway *actually* carries from layer to layer, and why every modern interpretability paper starts from the picture of two highways with sub-blocks reading and writing to them.
