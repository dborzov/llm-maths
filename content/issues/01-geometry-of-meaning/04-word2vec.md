---
title: "Embarrassingly Simple"
description: "January 2013. Mikolov submits a four-page paper to ICLR: a model with no hidden layers that beats everything, trains in hours, and secretly encodes the geometry of human language."
topics: [embeddings, word2vec]
tags: [word2vec, skip-gram, negative-sampling, training]
theme: teal
math: true
draft: false
date: 2026-04-01T09:18:00-04:00
issue: 1
weight: 40
techKind: mainline
techNode: word2vec
header: word2vec.webp
---

## ICLR 2013, Workshop Track

In January 2013, Tomáš Mikolov submitted a four-page paper to the International Conference on Learning Representations. The title was "Efficient Estimation of Word Representations in Vector Space." The model it described had no hidden layers. The architecture was so simple it barely counted as a neural network.

It beat everything.

Within six months, researchers around the world had replaced their expensive embedding pipelines with Mikolov's model. The code was released as `word2vec` — a command-line program that could train on 100 billion words overnight and produce 300-dimensional embeddings that captured analogy, cluster, and direction better than anything that had come before.

The genius was not in what the model *did*. It was in what it *removed*.

## The Architecture: Almost Nothing

Mikolov's model, in the **skip-gram** variant, has a single idea: given a word in a sentence, predict the words that appear near it.

The model structure:
- **Input:** a one-hot vector for the center word
- **Embedding matrix $W_{in}$:** vocab_size × d — multiplying by the one-hot input is just a lookup
- **Output matrix $W_{out}$:** vocab_size × d — one row per word
- **Prediction:** the dot product between the center word's input embedding and each candidate word's output embedding

No hidden layer. No nonlinearity. Two matrices and a dot product.

```python
import numpy as np

vocab_size = 10_000
d = 300  # embedding dimension

# The entire model
W_in  = np.random.randn(vocab_size, d) * 0.01   # "center word" embeddings
W_out = np.random.randn(vocab_size, d) * 0.01   # "context word" embeddings

# Forward pass for center word "cat" (index 42), context word "sat" (index 87)
center_idx  = 42
context_idx = 87

v_cat = W_in[center_idx]    # shape: (300,)
u_sat = W_out[context_idx]  # shape: (300,)

score = np.dot(v_cat, u_sat)   # one number
print(f"Score(cat→sat) before training: {score:.4f}")
```

The score is a single dot product. Training will push this score *up* for genuine (center, context) pairs and *down* for random pairs. That is the entire model.

{{% callout kind="note" %}}
Two separate matrices $W_{in}$ and $W_{out}$ — one for words acting as center words, one for words acting as context words. After training, most practitioners use only $W_{in}$ as their final embeddings. The $W_{out}$ matrix is discarded or averaged in.
{{% /callout %}}

## The Training Objective: Too Expensive

The natural training objective is: given center word $w_c$, maximize the probability of the true context words. Using softmax over the full vocabulary:

$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{w_o} \cdot \mathbf{v}_{w_c})}{\displaystyle\sum_{w=1}^{V} \exp(\mathbf{u}_w \cdot \mathbf{v}_{w_c})}$$

The denominator is the problem. Summing over all $V$ words requires $V$ dot products per training step. At $V = 100{,}000$ and 10 billion training steps, that's $10^{15}$ dot products just for the denominator — weeks of compute.

The full softmax is dead on arrival.

## Negative Sampling: The Radical Simplification

Mikolov's second key insight: don't compute the probability over the full vocabulary. Instead, **ask a simpler question**: *Is this (center, context) pair real, or fake?*

For each real pair like (cat, sat), sample $k$ random "negative" words from the vocabulary — say (cat, democracy), (cat, refrigerator), (cat, parliamentary). Train a binary classifier: the real pair should score high (sigmoid → 1), the negative pairs should score low (sigmoid → 0).

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def neg_sample_loss(v_center, u_context, u_negatives):
    """
    Negative sampling loss for one (center, context) pair.
    v_center:    shape (d,) — center word input embedding
    u_context:   shape (d,) — true context word output embedding
    u_negatives: shape (k, d) — k negative sample output embeddings
    """
    # Positive: want dot product HIGH
    pos_loss = -np.log(sigmoid(np.dot(u_context, v_center)) + 1e-10)

    # Negatives: want dot products LOW
    neg_losses = -np.sum(np.log(sigmoid(-u_negatives @ v_center) + 1e-10))

    return pos_loss + neg_losses

# One training step
np.random.seed(0)
d = 4  # toy dimension
v_cat  = np.array([0.3, -0.1, 0.4, 0.2])
u_sat  = np.array([0.2,  0.1, 0.3, 0.1])   # true context
u_negs = np.random.randn(5, d) * 0.3        # 5 random negative samples

loss = neg_sample_loss(v_cat, u_sat, u_negs)
print(f"Negative sampling loss: {loss:.4f}")
```

This loss involves only $k+1$ dot products per training step (1 positive + $k$ negative), where $k$ is typically 5–15. Compare to $V = 100{,}000$ for full softmax. A 10,000× speedup.

{{% marginnote %}}
Negative samples are drawn with probability proportional to the word's corpus frequency raised to the 3/4 power: $p(w) \propto f(w)^{0.75}$. This power smooths the distribution — common words are less over-represented, rare words are less ignored.
{{% /marginnote %}}

## A Training Run You Can Watch

Let us train from scratch on a tiny corpus and watch the geometry converge:

```python
np.random.seed(42)

sentences = [
    ["the","cat","sat","on","the","mat"],
    ["the","dog","sat","on","the","rug"],
    ["a","cat","chased","a","dog"],
    ["the","dog","chased","the","cat"],
    ["a","king","ruled","the","land"],
    ["a","queen","ruled","the","land"],
    ["the","king","and","queen","sat"],
]

vocab = sorted(set(w for s in sentences for w in s))
w2i  = {w: i for i, w in enumerate(vocab)}
V, d = len(vocab), 10

W_in  = np.random.randn(V, d) * 0.1
W_out = np.random.randn(V, d) * 0.1

# Negative sampling distribution (freq^0.75)
freq = np.zeros(V)
for s in sentences:
    for w in s: freq[w2i[w]] += 1
freq = freq**0.75
freq /= freq.sum()

def sig(x): return 1/(1+np.exp(-np.clip(x,-20,20)))

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-10)

lr = 0.05; window = 2; k_neg = 5; epochs = 500

for epoch in range(epochs):
    for sent in sentences:
        idxs = [w2i[w] for w in sent]
        for i, ci in enumerate(idxs):
            for j in range(max(0,i-window), min(len(idxs),i+window+1)):
                if i == j: continue
                oi = idxs[j]

                # Positive update
                s_pos = sig(W_in[ci] @ W_out[oi])
                grad_p = (s_pos - 1) * lr
                W_in[ci]  -= grad_p * W_out[oi]
                W_out[oi] -= grad_p * W_in[ci]

                # Negative updates
                neg_idx = np.random.choice(V, size=k_neg, p=freq)
                for ni in neg_idx:
                    s_neg = sig(W_in[ci] @ W_out[ni])
                    grad_n = s_neg * lr
                    W_in[ci]  -= grad_n * W_out[ni]
                    W_out[ni] -= grad_n * W_in[ci]

    if (epoch+1) in [1, 50, 100, 200, 500]:
        cd = cos_sim(W_in[w2i["cat"]],   W_in[w2i["dog"]])
        ck = cos_sim(W_in[w2i["cat"]],   W_in[w2i["king"]])
        kq = cos_sim(W_in[w2i["king"]],  W_in[w2i["queen"]])
        print(f"Epoch {epoch+1:3d}: sim(cat,dog)={cd:+.3f}  sim(cat,king)={ck:+.3f}  sim(king,queen)={kq:+.3f}")
```

```
Epoch   1: sim(cat,dog)=+0.042  sim(cat,king)=-0.029  sim(king,queen)=+0.081
Epoch  50: sim(cat,dog)=+0.312  sim(cat,king)=-0.103  sim(king,queen)=+0.214
Epoch 100: sim(cat,dog)=+0.538  sim(cat,king)=-0.188  sim(king,queen)=+0.419
Epoch 200: sim(cat,dog)=+0.721  sim(cat,king)=-0.241  sim(king,queen)=+0.683
Epoch 500: sim(cat,dog)=+0.849  sim(cat,king)=-0.198  sim(king,queen)=+0.891
```

Three things happening simultaneously, without being told:
- `cat` and `dog` get pushed toward each other (same contexts: sat, chased, on, mat/rug)
- `cat` and `king` get pushed apart (nearly disjoint context sets)
- `king` and `queen` converge (almost identical contexts: ruled, land, and, sat)

```pyplot {id="training-convergence" caption="COSINE SIMILARITIES DURING TRAINING. CAT/DOG AND KING/QUEEN CONVERGE; CAT/KING DIVERGES. THE GEOMETRY OF THE CORPUS BECOMES THE GEOMETRY OF THE SPACE."}
np.random.seed(42)

sentences = [
    ["the","cat","sat","on","the","mat"],
    ["the","dog","sat","on","the","rug"],
    ["a","cat","chased","a","dog"],
    ["the","dog","chased","the","cat"],
    ["a","king","ruled","the","land"],
    ["a","queen","ruled","the","land"],
    ["the","king","and","queen","sat"],
]
vocab = sorted(set(w for s in sentences for w in s))
w2i = {w:i for i,w in enumerate(vocab)}
V, d = len(vocab), 10

W_in  = np.random.randn(V, d)*0.1
W_out = np.random.randn(V, d)*0.1

freq = np.zeros(V)
for s in sentences:
    for w in s: freq[w2i[w]] += 1
freq = freq**0.75; freq /= freq.sum()

def sig(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
def csim(a,b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10)

history = {'epoch':[], 'cat_dog':[], 'king_queen':[], 'cat_king':[]}
lr=0.05; window=2; k_neg=5; epochs=600

for epoch in range(epochs):
    for sent in sentences:
        idxs = [w2i[w] for w in sent]
        for i,ci in enumerate(idxs):
            for j in range(max(0,i-window), min(len(idxs),i+window+1)):
                if i==j: continue
                oi = idxs[j]
                s_p = sig(W_in[ci]@W_out[oi])
                g_p = (s_p-1)*lr
                W_in[ci]-=g_p*W_out[oi]; W_out[oi]-=g_p*W_in[ci]
                for ni in np.random.choice(V,size=k_neg,p=freq):
                    s_n = sig(W_in[ci]@W_out[ni])
                    g_n = s_n*lr
                    W_in[ci]-=g_n*W_out[ni]; W_out[ni]-=g_n*W_in[ci]
    if (epoch+1) % 10 == 0 or epoch==0:
        history['epoch'].append(epoch+1)
        history['cat_dog'].append(csim(W_in[w2i['cat']], W_in[w2i['dog']]))
        history['king_queen'].append(csim(W_in[w2i['king']], W_in[w2i['queen']]))
        history['cat_king'].append(csim(W_in[w2i['cat']], W_in[w2i['king']]))

epochs_arr = np.array(history['epoch'])
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs_arr, history['cat_dog'],    color='#FF007F', lw=2.5, label='sim(cat, dog)  — similar contexts')
ax.plot(epochs_arr, history['king_queen'], color='#FFD700', lw=2.5, label='sim(king, queen) — similar contexts',
        markeredgecolor='#1A1A1A')
ax.plot(epochs_arr, history['cat_king'],   color='#00A8A8', lw=2.5, linestyle='--',
        label='sim(cat, king) — different contexts')
ax.axhline(0, color='#1A1A1A', lw=0.6, linestyle=':')
ax.set_xlabel("training epoch", fontsize=11)
ax.set_ylabel("cosine similarity", fontsize=11)
ax.set_title("Geometry self-organizes from context statistics alone", fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(-0.5, 1.05)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
```

## Why the Geometry Emerges

Here is the deep reason the geometry works.

Two words $w_1$ and $w_2$ end up with similar input embeddings if and only if they need high dot products with the *same set of context words*. And the only way to have high dot products with the same set of context words is to point in similar directions.

The training objective forces: **same neighborhood in text → similar direction in space**.

Since neighborhoods in text reflect meaning — cats and dogs both live near "pet," "fur," "chase," "feed" — the vector geometry ends up reflecting meaning. No human ever told the model that cats and dogs are similar. The geometry is purely a consequence of the statistics of language.

{{% callout kind="tangent" %}}
**Why does Word2Vec learn two matrices?** The $W_{in}$ vectors encode "how this word behaves as a center word." The $W_{out}$ vectors encode "how this word behaves as a context word." After training, the two matrices are related but not identical. Most applications use only $W_{in}$, but averaging $\frac{1}{2}(W_{in} + W_{out})$ sometimes gives slightly better embeddings.
{{% /callout %}}

## Napkin Math: The Original Word2Vec Run

Mikolov's 2013 paper trained on the **Google News** corpus:

- **Corpus size:** ~100 billion tokens
- **Vocabulary:** 1 million words
- **Embedding dimension:** 300
- **Parameters:** $2 \times 1{,}000{,}000 \times 300 = 600\text{M}$ (two matrices)
- **Storage:** $600\text{M} \times 4 \text{ bytes} = 2.4\text{ GB}$ for the full model

Training time on a single machine: **roughly 3–6 hours** with the negative sampling objective. The full softmax would have taken **months**.

The embedding table after training is a $1{,}000{,}000 \times 300$ matrix — about 1.2 GB at float32. Finding the nearest neighbor to any query word (across 1 million candidates) is a single matrix-vector multiply:

```python
# Nearest neighbor in a vocabulary of 1M words
# (Illustrative — not actually allocating 1M×300)
vocab_size, d = 100_000, 300

embeddings = np.random.randn(vocab_size, d).astype(np.float32)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_normed = embeddings / norms

query = embeddings_normed[42]  # any word

import time
t0 = time.time()
sims = embeddings_normed @ query   # one matrix-vector multiply
top5 = np.argsort(sims)[-5:][::-1]
print(f"Nearest neighbors: indices {top5}")
print(f"Computed in {(time.time()-t0)*1000:.1f} ms")
```

Exact nearest-neighbor search in a 300-dimensional space of 100,000 words, in milliseconds.

```pyplot {id="negative-sampling-illustration" caption="NEGATIVE SAMPLING: EACH TRAINING STEP DRAWS ONE POSITIVE PAIR AND K=5 NEGATIVES. THIS IS 10,000× CHEAPER THAN FULL SOFTMAX AT V=100k."}
np.random.seed(7)

# Illustrate the training objective: positive pair vs negatives
vocab = ["cat","sat","on","mat","dog","king","queen","the","a","land","ruled","and","chased","rug"]
V_small = len(vocab)

# Toy 2D embeddings for visualization
angles_in = np.linspace(0, 2*np.pi, V_small, endpoint=False) + np.random.randn(V_small)*0.3
W2 = np.stack([np.cos(angles_in), np.sin(angles_in)], axis=1) * 0.8

center_word_idx = 0   # "cat"
context_word_idx = 1  # "sat"
neg_idxs = [4, 6, 7, 8, 10]  # dog, queen, the, a, ruled

fig, ax = plt.subplots(figsize=(7, 7))

# Draw all vectors lightly
for i in range(V_small):
    color = '#1A1A1A'
    alpha = 0.2
    ax.annotate('', xy=W2[i], xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0, alpha=alpha))
    ax.text(W2[i][0]*1.1, W2[i][1]*1.1, vocab[i], fontsize=8, color='#888888', ha='center')

# Center word (pink)
ax.annotate('', xy=W2[center_word_idx]*1.05, xytext=[0,0],
            arrowprops=dict(arrowstyle='->', color='#FF007F', lw=2.5))
ax.text(W2[center_word_idx][0]*1.2, W2[center_word_idx][1]*1.2,
        f'"{vocab[center_word_idx]}" (center)', fontsize=10, color='#FF007F', fontweight='bold', ha='center')

# True context word (yellow, wants HIGH dot product)
ax.annotate('', xy=W2[context_word_idx]*1.05, xytext=[0,0],
            arrowprops=dict(arrowstyle='->', color='#FFD700', lw=2.5))
ax.text(W2[context_word_idx][0]*1.2, W2[context_word_idx][1]*1.2+0.05,
        f'"{vocab[context_word_idx]}" (true context)\n→ score UP', fontsize=9, color='#996600', ha='center')

# Negatives (teal, want LOW dot product)
for ni in neg_idxs:
    ax.annotate('', xy=W2[ni]*0.95, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='#00A8A8', lw=1.8, alpha=0.7))
    ax.text(W2[ni][0]*1.15, W2[ni][1]*1.15, f'"{vocab[ni]}"\n→ score DOWN',
            fontsize=8, color='#006666', ha='center', alpha=0.8)

ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
ax.set_aspect('equal')
ax.axhline(0, color='#1A1A1A', lw=0.4); ax.axvline(0, color='#1A1A1A', lw=0.4)
ax.set_title('Negative sampling: push center toward true context, away from random words', fontsize=10)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
```

## The Two-Sentence Summary

Word2Vec is a model with **two matrices and a dot product**. Training pushes the dot product up for real (center, context) pairs and down for random pairs. After training, the input matrix rows are your word vectors.

The simplicity is the point. No hidden layer means no expensive intermediate representation. The only thing the model *can* learn is how to arrange vectors so that dot products predict co-occurrence. And arranging vectors to predict co-occurrence turns out to encode the geometry of meaning.

**Continue to →** [The Stranger Country](../05-high-dimensional/), a primer on why 300-dimensional space has exactly the right geometry for this to work at scale.
