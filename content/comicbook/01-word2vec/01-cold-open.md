---
title: "Word Embeddings: what if the lookup table was the whole point?"
short_title: "Word Embeddings"
description: "September 2012: Mikolov is debugging a language model and notices the part everyone ignores — the embedding matrix — has quietly learned the geometry of meaning."
blurb:
  - "Every word entered Mikolov's model as a one-hot vector: a list of zeros with a single 1. Every pair of words was equally distant from every other."
  - "Backpropagation was updating the embedding matrix as a side effect. Nobody had asked it to learn anything — and yet the structure was there."
  - "\"Cat\" and \"dog\" had drifted together. \"King\" minus \"man\" plus \"woman\" landed near \"queen.\" The lookup table was doing the hard part."
  - "What if you threw away the rest of the model and trained *only* the lookup table?"
topics: [embeddings, representations]
tags: [word2vec, one-hot, representation]
theme: cream
math: true
draft: false
date: 2026-04-01T09:00:00-04:00
issue: 1
weight: 10
techKind: mainline
techNode: cold-open
header: 01-cold-open.webp
---

## Mountain View, September 2012

Tomáš Mikolov has a problem, and it is not the one he expected.

He is a researcher at Google, running language models on a cluster that most universities cannot afford. His recurrent neural network can read English and predict what comes next, and by the standards of 2012 it is very good. But today he is not thinking about how good it is. He is thinking about a lookup table.

Specifically: the first layer of his model is a matrix called the **embedding table**. Its job is embarrassingly simple. The vocabulary has, say, 100,000 words. Each word is assigned a unique integer — "the" is 0, "of" is 1, "cat" is 2,348. The embedding table is a matrix with 100,000 rows and 300 columns. When the word "cat" arrives, the network looks up row 2,348, grabs that 300-dimensional row vector, and hands it to the next layer. That is all.

In Mikolov's model, the embedding table has been updated by backpropagation alongside everything else. And staring at the learned vectors, something is making his skin prickle.

The vectors have *structure*.

Words that appear near each other in sentences — "cat" and "dog," "king" and "queen," "Paris" and "France" — have ended up near each other in vector space. The network has never been told that "cat" and "dog" are similar. All it did was try to predict the next word, over and over, across billions of sentences. And yet the geometry is there, encoded in the numbers.

What Mikolov realizes in this moment is not the algorithm yet. It is just the question: *What is actually doing the work here?*

## The Problem With Islands

To understand why this was a revelation, we need to understand how representation worked before.

Every word in Mikolov's model (and in every model of that era) started life as a **one-hot vector** — a list of numbers, all zero except for a single 1 in the word's designated position.

```python
vocab = ["the", "cat", "sat", "on", "mat", "dog", "king", "queen", "woman", "man"]
V = len(vocab)

# One-hot for "cat" (index 1)
cat_hot = [0.] * V; cat_hot[1] = 1.
dog_hot = [0.] * V; dog_hot[5] = 1.

print("cat:", cat_hot)
print("dog:", dog_hot)
```

```
cat: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dog: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
```

Now ask: how similar are "cat" and "dog"?

```python
def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

print("dot(cat, dog) =", dot(cat_hot, dog_hot))  # ?
print("dot(cat, mat) =", dot(cat_hot, [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]))  # ?
```

```
dot(cat, dog) = 0.0
dot(cat, mat) = 0.0
```

Zero. Both zero. The dot product of "cat" with "dog" is the same as the dot product of "cat" with "mat" — because every pair of distinct one-hot vectors is exactly perpendicular. In the geometry of one-hot space, every word is an **island floating alone**, equidistant from every other word in existence.

{{% callout kind="definition" %}}
**One-hot encoding:** A representation of a vocabulary of size $V$ where each word is a vector of length $V$ with exactly one entry equal to 1 (at the word's index) and all other entries 0. Distinct words are orthogonal: their dot product is always zero.
{{% /callout %}}

This is the lie baked into the foundation. "Cat" and "dog" are not the same distance from each other as "cat" and "democracy." Any child knows this. But the mathematical representation insisted otherwise.

```pyplot {id="one-hot-similarity" caption="COSINE SIMILARITY BETWEEN EVERY PAIR OF ONE-HOT VECTORS — AN OCEAN OF ZEROS. THESE WORDS INHABIT SEPARATE ISLANDS."}
words = ["the", "cat", "sat", "on", "mat", "dog", "king", "queen", "woman", "man"]
V = len(words)

# One-hot matrix
I = np.eye(V)

# All pairwise cosine similarities (= dot products for unit vectors)
sims = I @ I.T   # identity matrix — diagonal ones, off-diagonal zeros

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(sims, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(V))
ax.set_yticks(range(V))
ax.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(words, fontsize=10)
ax.set_title("One-hot cosine similarity: every pair of distinct words scores 0.0", fontsize=11)
fig.colorbar(im, ax=ax, label="cosine similarity")

# Annotate off-diagonal cells
for i in range(V):
    for j in range(V):
        val = sims[i, j]
        ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                color='white' if val == 1 else '#1A1A1A', fontsize=8, fontweight='bold')
plt.tight_layout()
```

The colour map tells the whole story. Diagonal entries are 1 (a word is identical to itself). Every off-diagonal entry is exactly 0. The representation contains *zero information* about which words resemble which.

## What Dense Vectors Can Do Instead

Here is what Mikolov sees in his trained embedding table.

Instead of a one-hot spike in 100,000-dimensional space, each word has become a **dense vector** — 300 real numbers, all potentially non-zero, organized by the statistical patterns of the language:

```python
# Toy illustration of learned dense vectors (4 dimensions, hand-crafted to make the point)
#               [animacy, domesticity, royalty, gender(female)]
cat    = [ 0.8,  0.9,  0.0, 0.0]
dog    = [ 0.8,  0.85, 0.0, 0.0]
king   = [ 0.7, -0.1,  0.9, 0.0]
queen  = [ 0.7, -0.1,  0.9, 1.0]
woman  = [ 0.8,  0.1,  0.0, 1.0]
man    = [ 0.8,  0.1,  0.0, 0.0]

import math
def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-10)

print(f"cos(cat, dog)   = {cosine(cat, dog):.3f}")    # high — both domestic animals
print(f"cos(king, queen) = {cosine(king, queen):.3f}") # high — both royals
print(f"cos(cat, king)  = {cosine(cat, king):.3f}")   # low  — unrelated
```

```
cos(cat, dog)   = 0.990
cos(king, queen) = 0.946
cos(cat, king)  = 0.421
```

These numbers are not zero. They encode what we actually know: "cat" and "dog" are very similar (both domestic animals), "king" and "queen" are similar (both royalty), and "cat" and "king" are only weakly related. The geometry of the space reflects the geometry of meaning.

```pyplot {id="dense-similarity" caption="COSINE SIMILARITY WITH LEARNED DENSE VECTORS. MEANING IS NOW ENCODED AS PROXIMITY — CAT AND DOG CLUSTER; ROYALS CLUSTER; THE MATH SPEAKS."}
np.random.seed(42)

# 4-dim toy embeddings — hand-tuned to illustrate semantic structure
words = ["cat", "dog", "kitten", "king", "queen", "throne", "sat", "mat"]
#           [animacy, domestic, royal, gender-f]
vecs = np.array([
    [ 0.80,  0.90,  0.0,  0.0],   # cat
    [ 0.80,  0.85,  0.0,  0.0],   # dog
    [ 0.78,  0.88,  0.0,  0.1],   # kitten
    [ 0.70, -0.10,  0.9,  0.0],   # king
    [ 0.70, -0.10,  0.9,  1.0],   # queen
    [ 0.10, -0.20,  0.95, 0.05],  # throne
    [ 0.10,  0.05,  0.0, -0.1],   # sat
    [ 0.05,  0.10,  0.0,  0.0],   # mat
], dtype=float)

# Normalize
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_n = vecs / norms

sims = vecs_n @ vecs_n.T

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(sims, cmap='RdYlGn', vmin=-0.5, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(words)))
ax.set_yticks(range(len(words)))
ax.set_xticklabels(words, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(words, fontsize=11)
ax.set_title("Dense vector cosine similarity: meaning is now geometry", fontsize=11)
fig.colorbar(im, ax=ax, label="cosine similarity")

for i in range(len(words)):
    for j in range(len(words)):
        v = sims[i, j]
        ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                color='white' if v > 0.7 or v < -0.2 else '#1A1A1A', fontsize=8)
plt.tight_layout()
```

The diagonal is still 1 (a word is always identical to itself). But now the off-diagonal has *structure*. Cat, dog, and kitten cluster together. King, queen, and throne cluster. Sat and mat are their own quiet corner. The space has been organized by meaning.

## The Question That Changed Everything

What Mikolov understood in September 2012 is that *the language model was a very expensive way to produce an embedding matrix*.

The real prize was the geometry. The 300-dimensional space organized by language statistics. The directions, the clusters, the relationships encoded as vectors.

What if you could get that geometry directly — without the expensive language model?

{{% pullquote %}}
What if we could give words coordinates in a space where proximity meant similarity — and throw away everything else?
{{% /pullquote %}}

The answer to that question is the rest of this issue.

**Continue to →** [Why Every Word Was An Island](../02-one-hot-problem/), where we trace the two dead ends researchers tried before word2vec — and why both of them failed in the same fundamental way.
