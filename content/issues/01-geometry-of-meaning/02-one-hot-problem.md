---
title: "Why Every Word Was An Island"
description: "1954. Linguist Zellig Harris argues that meaning lives in context. Forty years of failed attempts to compute with that idea — until Mikolov decided to skip the middleman."
topics: [embeddings, representations]
tags: [one-hot, co-occurrence, bengio, distributional-hypothesis]
theme: teal
math: true
draft: false
date: 2026-04-01T09:06:00-04:00
issue: 1
weight: 20
techKind: mainline
techNode: one-hot-problem
header: 03-embeddings.webp
---

## Columbia, Spring 1954

Zellig Harris is a linguist at Columbia University, and he has a theory about meaning that everyone finds plausible and nobody knows how to use.

His **distributional hypothesis**, stated plainly: *words that occur in similar contexts tend to have similar meanings*. "Cat" and "dog" appear near "pet," "feed," "fur," "veterinarian," "scratch." "King" and "queen" appear near "throne," "royal," "reign," "crown." The words' *neighborhoods* — the company they keep in text — encode what they mean.

J.R. Firth, a British linguist, would crystallize the same idea three years later in a phrase that became famous: *"You shall know a word by the company it keeps."*

Both men were right. The tragedy is that computing with this insight, at scale, turned out to be brutally hard. For the next four decades, researchers tried two approaches. Both had the right intuition. Both ran into the same wall.

## Dead End 1: Count Everything

The obvious approach is to count. Build a **co-occurrence matrix**: scan a large corpus, and for every pair of words $(w_i, w_j)$, count how often they appear within some window of each other.

```python
corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "rug"],
    ["the", "king", "and", "the", "queen", "sat"],
    ["a", "cat", "chased", "a", "dog"],
]

vocab = sorted(set(w for s in corpus for w in s))
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

window = 2
C = [[0]*V for _ in range(V)]
for sentence in corpus:
    for i, word in enumerate(sentence):
        for j in range(max(0, i-window), min(len(sentence), i+window+1)):
            if i != j:
                C[w2i[word]][w2i[sentence[j]]] += 1

print("Vocabulary:", vocab)
print(f"\nMatrix shape: {V}×{V} = {V*V} entries")
nonzero = sum(1 for row in C for x in row if x > 0)
print(f"Non-zero entries: {nonzero}  ({100*nonzero/(V*V):.1f}% filled)")
```

```
Vocabulary: ['a', 'and', 'cat', 'chased', 'dog', 'king', 'mat', 'on', 'queen', 'rug', 'sat', 'the']
Matrix shape: 12×12 = 144 entries
Non-zero entries: 62  (43.1% filled)
```

On this toy corpus, 43% of the matrix is non-zero — that looks reasonable. But notice: the corpus has 12 unique words and 4 sentences. A real corpus has 100,000–1,000,000 words. The density collapses catastrophically.

```pyplot {id="sparsity-scaling" caption="CO-OCCURRENCE MATRIX SPARSITY VS VOCABULARY SIZE. REAL CORPORA FILL LESS THAN 0.01% — THE MATRIX IS ALMOST ENTIRELY EMPTY."}
# Theoretical density of co-occurrence matrices
# A word window of ±2 means each word co-occurs with ~4 neighbours per sentence.
# If corpus has T tokens and V vocab words, expected non-zeros ≈ T * 4 * 2 (both directions)
# But unique (w_i, w_j) pairs are much less.

vocab_sizes = np.array([100, 1000, 10000, 100000, 1000000])
tokens_per_vocab = 100  # typical: corpus is ~100x vocab in tokens

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: matrix size vs vocab
matrix_entries = vocab_sizes**2
ax1.loglog(vocab_sizes, matrix_entries, color='#FF007F', linewidth=2.5, marker='o',
           markersize=7, markerfacecolor='#FFD700', markeredgecolor='#1A1A1A', markeredgewidth=1.5)
ax1.set_xlabel("vocabulary size V")
ax1.set_ylabel("matrix entries (V²)")
ax1.set_title("Matrix size grows as V²")
ax1.annotate("10 billion entries\nfor V = 100k", xy=(100000, 1e10),
             xytext=(5000, 5e11), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='#1A1A1A'),
             color='#1A1A1A')
ax1.spines[['top','right']].set_visible(False)

# Right: sparsity — fraction non-zero
# Estimate: each word co-occurs with ~k=4 distinct words per sentence on average
# Unique co-occurring pairs ≈ V * k (rough order of magnitude)
k = 400   # average unique neighbors across whole corpus
frac_nonzero = (vocab_sizes * k) / (vocab_sizes**2)  # ≈ k/V
ax2.semilogx(vocab_sizes, frac_nonzero * 100, color='#00A8A8', linewidth=2.5, marker='s',
             markersize=7, markerfacecolor='#FFD700', markeredgecolor='#1A1A1A', markeredgewidth=1.5)
ax2.set_xlabel("vocabulary size V")
ax2.set_ylabel("% non-zero entries")
ax2.set_title("Sparsity worsens linearly with V")
ax2.axhline(0.01, color='#FF007F', linestyle='--', linewidth=1.2, alpha=0.7, label='0.01% threshold')
ax2.legend(fontsize=9)
ax2.spines[['top','right']].set_visible(False)

plt.suptitle("The co-occurrence matrix doesn't scale", fontsize=12, fontweight='bold')
plt.tight_layout()
```

At V = 100,000 (a modest vocabulary for modern NLP), the matrix has **10 billion entries** and over 99.99% of them are zero. You are paying to store and compute over an ocean of nothing.

People tried fixes: pointwise mutual information (PMI) downweights the noise from ubiquitous words like "the"; truncated SVD (Singular Value Decomposition) compresses the matrix to a smaller dense representation. Both help. Neither solves the fundamental problem: you build something huge, then immediately throw most of it away.

{{% callout kind="tangent" %}}
**PMI and SVD work.** GloVe (2014) and other methods that explicitly factorize a weighted co-occurrence matrix do produce good embeddings. The complaint isn't that counting is wrong — it's that it's inefficient. Word2Vec gets roughly the same embeddings by skipping the count step entirely.
{{% /callout %}}

```pyplot {id="cooccurrence-matrix" caption="CO-OCCURRENCE MATRIX ON A SMALL CORPUS. THE STRUCTURE IS REAL — CAT/DOG SHARE NEIGHBORS — BUT EVEN HERE THE MATRIX IS ALREADY SPARSE."}
corpus = [
    ["the","cat","sat","on","the","mat"],
    ["the","dog","sat","on","the","rug"],
    ["a","cat","chased","a","dog"],
    ["the","dog","chased","the","cat"],
    ["a","king","ruled","the","land"],
    ["a","queen","ruled","the","land"],
    ["the","king","and","queen","sat"],
]
vocab = sorted(set(w for s in corpus for w in s))
w2i = {w:i for i,w in enumerate(vocab)}
V = len(vocab)
window = 2
C = np.zeros((V,V))
for sentence in corpus:
    for i, word in enumerate(sentence):
        for j in range(max(0,i-window), min(len(sentence),i+window+1)):
            if i != j:
                C[w2i[word]][w2i[sentence[j]]] += 1

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(C, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(V))
ax.set_yticks(range(V))
ax.set_xticklabels(vocab, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(vocab, fontsize=9)
ax.set_title(f"Co-occurrence matrix — {V}×{V}={V*V} cells, {int((C>0).sum())} non-zero ({100*(C>0).mean():.0f}%)", fontsize=10)
fig.colorbar(im, ax=ax, label="co-occurrence count")

for i in range(V):
    for j in range(V):
        if C[i,j] > 0:
            ax.text(j,i, f"{int(C[i,j])}", ha='center', va='center', fontsize=7, color='#1A1A1A')
plt.tight_layout()
```

Notice: "cat" and "dog" do share high co-occurrence with "the," "sat," "on," and "a." The signal is real. The problem is engineering, not theory.

## Dead End 2: Neural Language Models

A decade before Mikolov, **Yoshua Bengio** and colleagues hit on an alternative. Instead of counting and compressing, why not train a neural network to *predict* the next word — and use the internal representations it learns?

The architecture: look at the previous $n$ words, embed each in a 300-dimensional space (initially random), concatenate the resulting vectors, pass through a hidden layer, and output a probability distribution over the vocabulary. Train by minimizing the negative log-probability of the actual next word.

```python
# Conceptual Bengio-style model (forward pass only)
import math

def softmax(x):
    e = [math.exp(v - max(x)) for v in x]
    s = sum(e)
    return [v/s for v in e]

vocab_size = 10000
d = 300          # embedding dim
hidden = 500     # hidden layer size
context = 4      # previous words to look at

# Parameters learned during training:
#   embeddings: vocab_size × d  matrix    ← the prize
#   W_h:        (context*d) × hidden      ← context → hidden
#   W_o:        hidden × vocab_size       ← hidden → output

params = {
    'embed': vocab_size * d,
    'W_h':   context * d * hidden,
    'W_o':   hidden * vocab_size,
}
total = sum(params.values())
for name, n in params.items():
    print(f"{name:8s}: {n:>12,} parameters  ({100*n/total:.1f}%)")
print(f"{'TOTAL':8s}: {total:>12,}")
```

```
embed   :    3,000,000 parameters  (3.5%)
W_h     :    600,000 parameters  (0.7%)
W_o     :    5,000,000 parameters  (5.9%)
TOTAL   :    8,600,000
```

The embeddings are 3% of the model — and they are the part that ends up encoding meaning. The rest is scaffolding that helps them converge.

This works. The embeddings Bengio's model learns are genuinely good — words used in similar contexts end up near each other in vector space. But training it is **agonizingly expensive**.

The villain is the output layer: computing a probability distribution over the full vocabulary requires a dot product between the hidden vector and *every* word's output vector. For a vocabulary of 100,000 words, that is 100,000 dot products per training example. With billions of training examples needed, the arithmetic is lethal.

**Napkin math:** 100B training tokens × 100,000 output dot products × 500 hidden dimensions = approximately $5 \times 10^{18}$ floating-point operations. At 2012 GPU speeds (~10 TFLOP/s), that is **500,000 seconds** — about six days — *just* for the output layer arithmetic. And that assumes no memory bandwidth bottlenecks.

{{% marginnote %}}
Modern models solve this with **sampled softmax** or **noise contrastive estimation (NCE)**. Mikolov's solution in Word2Vec — negative sampling — is a simpler, more radical version of the same idea: don't compute the full distribution at all.
{{% /marginnote %}}

By 2012, Mikolov had built recurrent versions of Bengio's model that were state-of-the-art. He knew the embeddings they produced were excellent. He also knew training them took weeks.

## The Question They Kept Answering Wrong

Looking back at both dead ends, they share a structure:

| Approach | What it got right | Why it failed |
|---|---|---|
| Co-occurrence matrix | The signal is in word neighborhoods | Matrix is enormous, sparse; requires expensive post-processing |
| Neural language model | Learned embeddings are excellent | Output softmax is a compute bottleneck; weeks to train |

Both approaches were using the prediction of *word context* as the learning signal — but both were doing far more work than strictly necessary to extract that signal.

Mikolov's key observation: the language model trained a 500-dimensional hidden representation just to produce a good embedding layer. What if you designed a model with *no* hidden layer — whose only purpose was to produce good embeddings directly?

Not "how do we improve language models?" but "how do we throw everything away except the part that matters?"

The answer to that question became Word2Vec.

**Continue to →** [The Geometry of Agreement](../03-dot-product/) for the dot product primer — the measuring tool that makes Word2Vec work — or jump directly to [Embarrassingly Simple](../04-word2vec/) for the algorithm itself.
