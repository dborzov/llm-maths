---
title: "Tokens & Positions"
description: "Two lookup tables, added together. The `wte` table tells the model what token it saw; the `wpe` table tells it where. Why addition (and not concatenation) is the right move."
topics: [transformer, embeddings]
tags: [microgpt, wte, wpe, positional]
theme: cream
math: true
draft: false
date: 2026-05-14T02:48:52-04:00
issue: 5
weight: 30
techKind: primer
techNode: embeddings
header: 03-embeddings.webp
---

## Prague, September 2013

A Czech graduate student named **Tomáš Mikolov** is in a Google office in Mountain View, doing the kind of thing graduate students do when they think nobody is watching: typing arithmetic into a Python REPL. The model on his disk is a freshly trained **word2vec** — a shallow neural network that has, over the course of three days on a beefy machine, eaten a billion words of Google News and produced, for each word in its 692,000-word vocabulary, a list of three hundred floats.

Mikolov types:

```python
v["king"] - v["man"] + v["woman"]
# → nearest neighbour in the vocabulary: "queen"
```

He tries another one. `v["paris"] - v["france"] + v["italy"]` lands closest to `rome`. `v["walking"] - v["walk"] + v["swim"]` lands closest to `swimming`. He keeps going for an hour. The vectors aren't just *clustering* synonyms — they have *internal directions*. The vector that points from a country to its capital is the same vector that points from another country to *its* capital. Grammatical tense lives along a consistent axis. Gender is a direction.

Mikolov's lab notebook entry for that day is one line. *"The arithmetic works."*

The paper he writes about it — *"Linguistic Regularities in Continuous Space Word Representations"* — will be cited fifty thousand times. But the bigger consequence is invisible in 2013: every transformer language model built in the next decade and a half will begin its forward pass with a **lookup into a table of vectors just like Mikolov's**. The cold-open chapter [showed you the line](../01-cold-open/); this primer is about what that line means.

```python
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

Three lines. Two lookups. One addition. There is more going on here than there appears.

## The Big Lookup Table

The first object on the stage is `wte`, the **w**ord-**t**oken **e**mbedding. In microGPT it is exactly what its shape advertises:

$$
\texttt{wte} \in \mathbb{R}^{\,\texttt{vocab\_size} \,\times\, \texttt{n\_embd}}
$$

A matrix. `vocab_size` rows, `n_embd` columns. Each *row* is the vector that represents one token in the model's vocabulary. For the toy model trained on lowercase names — `vocab_size = 27` (26 letters plus a beginning-of-string marker) and `n_embd = 16` — that means `wte` is a 27×16 grid. Twenty-seven rows of sixteen floats each.

The lookup itself is the most boring instruction in the entire forward pass:

```python
tok_emb = state_dict['wte'][token_id]
```

If `token_id` is `5`, you get row 5. If it is `12`, you get row 12. That's it. No matrix multiply. No attention. No softmax. The complexity that follows in the next fifteen chapters of this issue all comes *after* this look-up; the look-up itself is what a CPU calls "indirect addressing" and what your great-uncle would call "looking it up in the book." A learned token embedding is a row in a book of vectors.

Some people prefer to picture this lookup as a **one-hot matrix multiply**: if $e_t$ is the one-hot vector with a 1 in position `token_id` and zeros elsewhere, then $\texttt{tok\_emb} = e_t^\top \texttt{wte}$. That is *mathematically* identical and shows up in textbooks because it makes the gradient look natural. But mechanically — and certainly inside microGPT — nobody multiplies by a one-hot vector. You just index.

### Where did the rows come from?

In `wte`, every row was learned. During training, the gradient that wanted token *k*'s representation to drift in some direction got accumulated specifically into row *k* of `wte`. The result, in any well-trained model, is the property Mikolov noticed in 2013: rows for semantically related tokens end up nearby in $\mathbb{R}^{n\_embd}$. Rows for `cat` and `dog` are closer to each other than either is to `Beethoven`. Rows for `the` and `a` cluster together. Rows for the same-token-in-different-casing (`Cat` and `cat` in a BPE-tokenized model) are close enough that a single linear projection can move between them.

That's the entire purpose of this table. **`wte` is the model's dictionary of what tokens mean.** Every other operation in the forward pass massages those meanings.

## The Other Lookup Table

The next line of microGPT does the same trick with a different table.

```python
pos_emb = state_dict['wpe'][pos_id]
```

`wpe` — the **w**ord-**p**osition **e**mbedding — has shape

$$
\texttt{wpe} \in \mathbb{R}^{\,\texttt{block\_size} \,\times\, \texttt{n\_embd}}.
$$

`block_size` rows, `n_embd` columns. In the toy model, `block_size = 16`, so `wpe` is 16×16. A square of 256 floats. Where `wte` is indexed by *what* token you saw, `wpe` is indexed by *where* in the sequence it sits. Position 0 gets row 0. Position 7 gets row 7. The first token of every sequence gets the same row of `wpe`; the second token of every sequence gets the same *different* row; and so on.

This split — separating *what* from *where* — is the entire reason transformers need positional embeddings at all. Attention, the operation we'll meet in [chapters 6 through 9](../06-qkv-projections/), is fundamentally a **set** operation: scramble the order of the input tokens and the output is scrambled identically. Without a position signal, "the dog bit the man" and "the man bit the dog" look like the same multiset to the attention layer. `wpe` is the patch on that hole.

## The Puzzle Of The Plus Sign

Now look at the third line, the one that *combines* the two lookups:

```python
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

It adds them. Element-wise. The vector that goes into the first transformer block is the **sum** of the token vector and the position vector. The first time you see this you should be a little upset.

Why addition? The token embedding is *what* the token is. The position embedding is *where* it is. Those are different *kinds* of information. Smashing them together with a `+` sign feels like adding a person's height to their age and calling the result a person. The obvious instinct, the one any sensible engineer would write down on a napkin, is:

**concatenate** them. Put `tok_emb` first, then `pos_emb` after, and let downstream layers sort it out.

```python
# the "obvious" alternative — NOT what microGPT does
x = tok_emb + pos_emb        # microGPT: 16 dims in, 16 dims out
x = tok_emb ++ pos_emb       # concat:   16 dims in, 32 dims out
```

This is so obvious that it has to be wrong, and the reason it is wrong is worth dwelling on for a paragraph or three.

### The bookkeeping cost

If you concatenate, your residual stream is now `2 · n_embd` wide. Every single matrix downstream — `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`, `mlp_fc1`, `mlp_fc2`, `lm_head` — has to be twice as tall on the input side. For Llama 3 8B with `n_embd = 4096`, that's a doubling of the model's parameter count just to keep the bookkeeping consistent. The cost is **gigantic**, and the marginal benefit (the model now "knows" cleanly which dimensions are position and which are content) is something it can almost certainly *learn* to extract anyway.

### Addition is free decomposition

Here is the deeper reason. The "what" and "where" components do not actually have to live in *disjoint* subspaces; they only have to live in subspaces the model can *project* onto independently when it wants to.

Concretely: suppose during training, the gradients consistently push `wte` rows to occupy the first eight dimensions and `wpe` rows to occupy the last eight (we say "suppose" — in practice the split is rotated and rotated again by every linear). Then the sum

$$
x = \texttt{tok\_emb} + \texttt{pos\_emb}
$$

is, for all practical purposes, a concatenation in disguise: the first eight slots are pure token information, the last eight pure position. Any downstream linear can recover either piece by zeroing out the columns it doesn't want. Addition gives you concatenation **whenever the model chooses to organize its dimensions that way** — at zero parameter cost.

But addition gives you something stronger too: the model can *mix* the two signals when mixing is useful. If, for token `the` in position 3, the model wants to behave slightly differently than for token `the` in position 8, it has a way to do that — the two `x` vectors are genuinely different in `n_embd` slots, and attention can be sensitive to that difference. With strict concatenation, the position bits sit in their own ghetto and the model needs an explicit cross-multiplication to make use of them.

The empirical bet, made by the original 2017 *Attention Is All You Need* paper and confirmed by every transformer trained since, is that **the model can learn whichever decomposition it wants — fully separate, fully entangled, or anywhere in between — from data**. Addition is the most parsimonious operation that gives it that freedom. So we add, and we save the parameters.

## A Picture To Sit With

Let's actually look at what a `wpe` matrix tends to look like. We'll initialize a small one randomly and then visualize the "first-few-dimensions" cross-section as horizontal strips — one strip per position, colored by value.

```pyplot {id="wpe-heatmap" caption="A learned wpe matrix, after random initialization and a sketch of training. Each row is a position; each column is one of the 16 embedding dimensions. Real trained wpe matrices look smoother than this in the lowest dimensions and noisier in the highest — they end up encoding 'distance from start' along a few principal axes."}
np.random.seed(42)
block_size = 16
n_embd = 16

# Random init like nn.Embedding's default (normal, small std).
wpe = np.random.randn(block_size, n_embd) * 0.02

# Sketch what learned wpe matrices tend to develop:
# a smooth gradient along the first few dimensions (capturing 'position index'),
# noise in the rest.
positions = np.arange(block_size).reshape(-1, 1)
wpe[:, 0] += np.linspace(-1, 1, block_size)        # linear ramp
wpe[:, 1] += np.sin(positions.flatten() * 0.6)     # low-freq sine
wpe[:, 2] += np.cos(positions.flatten() * 0.6)     # low-freq cosine
wpe[:, 3] += np.sin(positions.flatten() * 1.3)     # higher-freq sine

fig, ax = plt.subplots(figsize=(8.5, 4.5))
im = ax.imshow(wpe, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xlabel("embedding dimension (0 .. n_embd-1)")
ax.set_ylabel("position (0 .. block_size-1)")
ax.set_title("wpe: one row per position, one column per embedding dimension")
ax.set_xticks(range(n_embd))
ax.set_yticks(range(block_size))
for spine in ax.spines.values():
    spine.set_edgecolor('#1A1A1A')
    spine.set_linewidth(1.5)
plt.colorbar(im, ax=ax, label='value')
```

The eye-grabbing pattern in the first few columns is the model's *position channel*: a smooth gradient that means "I am near the start" at the top and "I am near the end" at the bottom. The remaining columns are noise — capacity the model has reserved but not yet committed to any particular use of position. In a well-trained `wpe`, more of the columns end up doing useful work, but the *low-frequency* structure in the first few dimensions is universal.

## Napkin Math: How Big Are These Tables?

In the toy:

$$
|\texttt{wte}| = 27 \times 16 = 432 \text{ floats}
$$

$$
|\texttt{wpe}| = 16 \times 16 = 256 \text{ floats}
$$

Both tables fit in a screenshot. Cute.

For Llama 3 8B, the story is very different. Llama 3 uses a **128,000-token** BPE vocabulary, an embedding dimension of **4,096**, and a (RoPE-extended) context window we can pretend is **8,192** for accounting purposes. So:

$$
|\texttt{wte}| = 128{,}000 \times 4{,}096 = 524{,}288{,}000 \text{ floats} \approx \mathbf{1.0 \text{ GB at fp16}}
$$

A full *gigabyte* of weights, just for the token-lookup table, just to know what each token "means" before the model does anything else. That's about **6.5%** of Llama 3 8B's 16 GB fp16 footprint — sunk into a table that performs a single integer index. (Many production models *tie* `wte` to the final `lm_head` to halve this cost; we look at that trick in [chapter 15](../15-sampling/).)

For `wpe`, well — Llama 3 doesn't have a learned `wpe`. We'll get to why in a second. But if it *did*, with `block_size = 8192`:

$$
|\texttt{wpe}| = 8{,}192 \times 4{,}096 = 33{,}554{,}432 \text{ floats} \approx \mathbf{64 \text{ MB at fp16}}
$$

Small compared to `wte`, but big enough that you would want a good reason to keep it around.

## The Positional-Embedding Zoo

Llama 3 doesn't carry a `wpe` table because the field, since 2017, has held a slow-motion bake-off between four families of how to encode *where*. Each one swaps out the single microGPT line `pos_emb = state_dict['wpe'][pos_id]` for a different mechanism.

| Year | Name | What it does | Used by |
|---|---|---|---|
| 2017 | **Sinusoidal** | Hand-designed, fixed: $\text{PE}_{pos, 2i} = \sin(pos / 10000^{2i/d})$, with cosines on odd dims. No learned parameters. | Original *Attention Is All You Need*, T5 (in part). |
| 2018 | **Learned absolute** | Exactly what microGPT does. A `block_size × n_embd` table; rows learned during training. | GPT-2, GPT-3, original BERT. |
| 2021 | **RoPE** (Rotary Position Embedding) | Don't add a vector; instead *rotate* the query and key vectors of attention by a position-dependent angle. Position information is injected at the attention layer, not the input. | Llama 1/2/3, GPT-NeoX, GPT-4 (reportedly), most modern open-weights models. |
| 2021 | **ALiBi** (Attention with Linear Biases) | No position vectors at all. Instead, add a position-dependent *bias* to attention logits that linearly penalizes distant tokens. | BLOOM, some MosaicML models. |

The progression is, in spirit, a steady move *away* from learned position tables. Sinusoidal said "don't bother learning, the right answer is hand-derivable." Learned absolute said "no, the model can do better than your hand-derivation." RoPE said "the right place to inject position is into the dot-product mechanism that *uses* it, not into the input embedding." ALiBi said "you don't even need a vector — just bias the attention logits."

The technical reasons each newer scheme beats the older one are subtle and mostly about **length generalization**: how well does the model do when, at inference time, you give it a sequence *longer* than anything it saw in training? Learned absolute `wpe` falls off a cliff at `pos_id = block_size + 1` — there *is* no row to look up. Sinusoidal and RoPE extrapolate gracefully because their position signal is generated, not stored. We will mostly skirt this story in this issue, because the microGPT listing uses learned `wpe` and we are honor-bound to explain that one first. But every time you read about a "long-context model" — 128k context, a million tokens, whatever — you are reading about a small variation on this table.

## What To Remember

1. **`wte` is a `vocab_size × n_embd` lookup table.** One row per vocabulary token. `state_dict['wte'][token_id]` is *literally* an array index. The complexity of an LLM does not live here.
2. **`wpe` is a `block_size × n_embd` lookup table.** One row per position in the context window. Indexed by `pos_id`, not by anything to do with content.
3. **Addition, not concatenation.** The model can recover any decomposition of `what` and `where` it likes, at zero extra parameter cost, because addition leaves the residual stream's width alone. Concatenation would double every downstream weight matrix.
4. **`wte` in Llama 3 is ~1 GB.** A nontrivial fraction of model weight even at frontier scale. Tied embeddings (`lm_head = wte^T`) cut this in half.
5. **Learned `wpe` is the historical default; modern frontier models don't use it.** RoPE has effectively won the positional-embedding bake-off for autoregressive decoders. The microGPT line `pos_emb = state_dict['wpe'][pos_id]` is the **clearest** way to teach the concept — but in production, it has been replaced by an in-attention rotation.

The two lookups give us a single `n_embd`-dim vector `x` representing "token *t* at position *p*". From here on, every transformer block in the model will take that vector, transform it, and put a transformed version back. The first thing that happens to it — in microGPT, in GPT-2, in Llama 3 — is a linear projection. So that's where we go next.

---

**Continue to** → [Linear, in Pure Python](../04-linear/) — six characters of NumPy expand into the four-line list comprehension that does eighty percent of the FLOPs in every LLM on Earth.

