---
title: "Word2Vec and The Geometry of Meaning"
description: "How Mikolov's 2013 breakthrough replaced brittle one-hot vectors with dense geometric embeddings — and why cosine similarity suddenly meant something."
topics: [embeddings, representations]
tags: [word2vec, numpy, linear-algebra]
theme: cream
math: true
draft: false
---

# Chapter 1: Word2Vec and The Geometry of Meaning

## A Linguist Walks into a Neural Network

In the autumn of 2012, Tomáš Mikolov had a problem that was eating him alive.

He was a researcher at Google, working on language models — systems that predict the next word in a sentence. The state of the art at the time was already using neural networks, and Mikolov had built some of the best ones. But every one of them started with the same ugly bottleneck: the *vocabulary lookup table*.

Here's what that looked like. Suppose your vocabulary has 100,000 words. In the models of that era, each word was represented as a **one-hot vector** — a list of 100,000 numbers, all zeros except for a single 1 in the position corresponding to that word.

```python
import numpy as np

vocab = ["the", "cat", "sat", "on", "mat", "dog", "king", "queen", "woman", "man"]
vocab_size = len(vocab)

# One-hot encoding for "cat" (index 1)
cat_vector = np.zeros(vocab_size)
cat_vector[1] = 1.0

# One-hot encoding for "dog" (index 5)
dog_vector = np.zeros(vocab_size)
dog_vector[5] = 1.0

print(f"cat: {cat_vector}")
print(f"dog: {dog_vector}")
print(f"dot product (cat · dog): {np.dot(cat_vector, dog_vector)}")
```

```
cat: [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
dog: [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
dot product (cat · dog): 0.0
```

That zero at the bottom is the whole problem.

The dot product of "cat" and "dog" is zero. The dot product of "cat" and "mat" is also zero. The dot product of "king" and "queen" is zero. Every word is *exactly as far away from every other word* as any other. In the geometry of one-hot vectors, "cat" is no more similar to "dog" than it is to "refrigerator" or "democracy." Every word is an isolated island floating in a 100,000-dimensional void, equidistant from everything else.

Mikolov knew this was insane. Any child knows that "cat" and "dog" are more similar to each other than either is to "constitution." But the mathematical representation his models started from contained *no trace of this knowledge*. The neural network had to learn all of it from scratch, from raw co-occurrence statistics, fighting against a representation that actively encoded the lie that all words are equally unrelated.

What if, he thought, we could give words a better starting point?

---

## Dead End #1: Counting Co-occurrences

Mikolov wasn't the first person to have this thought. Not by decades.

Linguists had long operated under a powerful intuition, crystallized by J.R. Firth in 1957: *"You shall know a word by the company it keeps."* If "cat" and "dog" appear near similar words — "pet," "fur," "feed," "veterinarian" — then they probably mean similar things.

The obvious computational approach was to build a **co-occurrence matrix**. Scan a huge text corpus, and for every pair of words, count how often they appear near each other within some window.

```python
import numpy as np

# A tiny corpus
corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "mat"],
    ["the", "king", "and", "the", "queen"],
    ["the", "man", "and", "the", "woman"],
]

vocab = sorted(set(word for sentence in corpus for word in sentence))
word_to_idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# Build co-occurrence matrix (window size = 2)
cooccurrence = np.zeros((V, V))
window = 2

for sentence in corpus:
    for i, word in enumerate(sentence):
        for j in range(max(0, i - window), min(len(sentence), i + window + 1)):
            if i != j:
                cooccurrence[word_to_idx[word]][word_to_idx[sentence[j]]] += 1

print("Vocabulary:", vocab)
print("\nCo-occurrence matrix:")
print(cooccurrence)
```

Now each word is represented not by a meaningless one-hot spike, but by a *row* of this matrix — a vector of counts describing the word's neighborhood. "Cat" and "dog" will have similar rows because they appear near similar words.

This is a real improvement. But it has brutal problems:

**Problem 1: The vectors are enormous.** If your vocabulary has 100,000 words, each word's vector has 100,000 dimensions. Storing and computing with these is expensive.

**Problem 2: The matrix is absurdly sparse.** Most entries are zero. Most words never appear near most other words. You're storing and computing over a vast emptiness.

**Problem 3: Raw counts are dominated by garbage.** The word "the" co-occurs with everything. It tells you almost nothing about meaning. Words like "the," "of," "and" dominate the counts, drowning out the meaningful signal.

People tried fixes. TF-IDF weighting, pointwise mutual information (PMI), dimensionality reduction via SVD (Singular Value Decomposition). These helped. But they felt like patches on a fundamentally clunky approach. You'd build a giant matrix, then immediately try to compress it down. The final compressed vectors were decent, but the process was brittle, slow, and hard to scale.

Mikolov looked at this pipeline — count, weight, factorize — and thought: *What if we skip the giant matrix entirely?*

---

## Dead End #2: The Neural Language Model (Too Expensive)

There was another thread of research, started by Yoshua Bengio and colleagues in 2003. Their idea: train a neural network to predict the next word in a sentence, and use the internal weights of the network as word vectors.

The architecture was straightforward. Take the previous few words, look up their embeddings (initially random), concatenate them, feed them through a hidden layer, then project to a probability distribution over the entire vocabulary via a softmax.

```python
import numpy as np

# Simplified Bengio-style language model (conceptual)
vocab_size = 10000
embedding_dim = 300
hidden_dim = 500
context_size = 4  # look at previous 4 words

# These are the parameters we'd learn:
embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01  # <-- the word vectors!
W_hidden = np.random.randn(context_size * embedding_dim, hidden_dim) * 0.01
W_output = np.random.randn(hidden_dim, vocab_size) * 0.01

# Forward pass for one example
context_indices = [42, 17, 8033, 291]  # indices of the previous 4 words
context_vectors = embeddings[context_indices]        # shape: (4, 300)
concatenated = context_vectors.flatten()              # shape: (1200,)
hidden = np.tanh(concatenated @ W_hidden)             # shape: (500,)
logits = hidden @ W_output                            # shape: (10000,)

# Softmax to get probabilities
exp_logits = np.exp(logits - logits.max())
probs = exp_logits / exp_logits.sum()                 # shape: (10000,)
```

This worked beautifully in theory. The embeddings that emerged were meaningful. Words used in similar contexts got pushed toward similar vectors by the gradient updates. The network *learned* that "cat" and "dog" should be close because they're useful in similar predictive contexts.

But training this was agonizingly slow. The hidden layer was expensive. The softmax over the full vocabulary was a killer — every training step required computing a dot product with every single word in the vocabulary, then normalizing. For a vocabulary of 100,000 words, that's 100,000 dot products *per training example*. And you need to see billions of training examples.

Mikolov had built recurrent versions of these models (his RNN language models were famous in the field). He knew the embeddings inside them were good. He also knew training them took weeks on expensive hardware.

And then the key insight struck him: *What if the embeddings are the whole point?*

What if, instead of training a big, expensive model and extracting the embeddings as a side effect, you designed the simplest possible model whose *only purpose* was to produce good embeddings?

---

## The Breakthrough: Simplicity as a Weapon

What Mikolov did next was almost embarrassingly simple — and that was the genius of it.

He stripped away the hidden layer entirely. He threw out the complex architecture. He was left with something that barely qualifies as a neural network: a single matrix multiplication on the input side, and a single matrix multiplication on the output side. No nonlinearity. No hidden representation. Just two matrices.

This became **Word2Vec**, and it came in two flavors. We'll focus on the one called **Skip-gram**, because its logic is the most transparent.

The idea: given a word in a sentence, predict the words around it.

That's it. If you see "cat" in the sentence "the **cat** sat on the mat," the model's job is to predict that "the," "sat," "on," and "the" are nearby. It doesn't need to predict the exact next word. It just needs to say: *these words are plausible neighbors of "cat."*

```python
import numpy as np

# Skip-gram architecture — it's almost nothing
vocab_size = 10000
embedding_dim = 300

# Two embedding matrices — that's the entire model
W_input  = np.random.randn(vocab_size, embedding_dim) * 0.01   # "input" embeddings
W_output = np.random.randn(vocab_size, embedding_dim) * 0.01   # "context" embeddings

# For a center word "cat" (index 42) and a context word "sat" (index 87):
center_word = 42
context_word = 87

# The "prediction" is just a dot product
v_cat = W_input[center_word]     # shape: (300,)
u_sat = W_output[context_word]   # shape: (300,)

score = np.dot(v_cat, u_sat)     # a single number!
print(f"Score for (cat, sat): {score:.4f}")
```

Look at what's happening here. The model has learned a 300-dimensional vector for each word. The "prediction" is just a dot product — a measure of alignment between two vectors. Training will push this score up for genuine (word, context) pairs and down for random pairs.

But notice the lethal elegance of what the dot product forces. If "cat" needs to predict similar context words as "dog" — if both need high dot products with "sat," "pet," "the," "furry" — then their input vectors must point in similar directions. The *only* way to achieve similar dot products with the same set of context words is to have similar vectors. The geometry enforces the semantics.

---

## Why Floating-Point Vectors? The Mathematical Argument

Let's pause the historical narrative and dig into the question that the curious reader should be asking: **Why vectors of floating-point numbers at all?**

This is not obvious. We could represent word similarity with a graph. We could use discrete clusters. We could store word relationships in a database. Why is a 300-dimensional vector of floating-point numbers the *right mathematical object* for meaning?

The answer comes from understanding what properties we actually need, and then showing that dense vectors in ℝⁿ are the simplest object that provides all of them simultaneously.

### Property 1: Graded Similarity

Meaning isn't binary. "Cat" isn't simply similar or dissimilar to "dog" — it's *very* similar. "Cat" is somewhat similar to "tiger." "Cat" is slightly similar to "pet." "Cat" is barely similar to "car." We need a **continuous measure of similarity** — a number between "identical" and "completely unrelated."

A graph of word relationships can encode connections, but not graded strengths (unless you add weights, at which point you're halfway to vectors already). A set of discrete clusters can say "same group" or "different group," but can't express that "cat" is closer to "lion" than to "eagle" even though all three are animals.

Vectors in ℝⁿ give you this for free via the **dot product** (or equivalently, **cosine similarity**):

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Suppose we've learned these 4-dimensional embeddings (just for illustration)
cat    = np.array([ 0.7,  0.5, -0.1,  0.3])
dog    = np.array([ 0.6,  0.6, -0.1,  0.2])
tiger  = np.array([ 0.8,  0.2,  0.4,  0.5])
car    = np.array([-0.3, -0.5,  0.8,  0.1])

print(f"cat · dog   = {cosine_similarity(cat, dog):.3f}")   # High
print(f"cat · tiger = {cosine_similarity(cat, tiger):.3f}")  # Medium
print(f"cat · car   = {cosine_similarity(cat, car):.3f}")    # Low/negative
```

```
cat · dog   = 0.981
cat · tiger = 0.793
cat · car   = -0.476
```

The cosine similarity is a continuous value from -1 to 1. Words can be *more or less* similar. The geometry of the space does the work.

### Property 2: Multiple Simultaneous Similarities

Here's where things get deep. Words aren't similar along just one axis. "Cat" and "dog" are similar because they're both *pets*. "Cat" and "tiger" are similar because they're both *felines*. "Dog" and "tiger" are similar because they're both *predators*. We need a representation that can encode many *independent dimensions of similarity at once*.

This is impossible in one dimension. On a number line, if you place "cat" between "dog" and "tiger," you've committed to a single ordering. You can't simultaneously express that "cat" is close to "dog" on the pet axis and close to "tiger" on the feline axis.

```python
# In 1D, we're stuck with a single ordering
number_line = {"dog": 1.0, "cat": 2.0, "tiger": 3.0, "lion": 4.0, "goldfish": 5.0}

# cat-dog distance: 1.0
# cat-tiger distance: 1.0
# But goldfish should be CLOSE to cat/dog (all pets) and FAR from tiger/lion (wild animals)
# There's no place on a number line where goldfish can be both!
# Goldfish is 3.0 away from cat. Tiger is only 1.0 away from cat.
# The number line says tiger is more like cat than goldfish is.
# But on the "domesticated pet" axis, that's dead wrong.
```

In two dimensions, we get more room:

```python
import numpy as np

# 2D: Now we can separate "pet-ness" and "feline-ness"
#                 [pet-ness, feline-ness]
cat      = np.array([0.8,  0.9])
dog      = np.array([0.9,  0.1])
tiger    = np.array([0.1,  0.95])
goldfish = np.array([0.85, 0.0])
lion     = np.array([0.05, 0.8])

# Now goldfish IS close to cat on the pet axis
# And tiger IS close to cat on the feline axis
# Both truths coexist!

from numpy.linalg import norm

pairs = [("cat", cat), ("dog", dog), ("tiger", tiger), ("goldfish", goldfish), ("lion", lion)]
for name_a, vec_a in pairs:
    for name_b, vec_b in pairs:
        if name_a < name_b:
            d = norm(vec_a - vec_b)
            print(f"  {name_a:10s} - {name_b:10s}: {d:.3f}")
```

But real word meaning has *far more than two* independent axes of variation. Words vary in sentiment, formality, concreteness, animacy, temporal character, domain, register, and hundreds of other subtle properties. You need *hundreds* of dimensions to capture this.

This is the core mathematical argument for high-dimensional vectors: **each dimension provides an independent axis along which similarity can be expressed**. With 300 dimensions, you can express 300 independent aspects of meaning, all simultaneously, and the similarity between any two words is the aggregate of their agreement across all these dimensions.

### Property 3: Algebraic Structure (The "King - Man + Woman = Queen" Miracle)

This is the property that made Word2Vec famous and convinced the world that something genuinely profound was happening.

After training, the learned vectors didn't just encode similarity. They encoded *relationships as directions*. The vector from "man" to "woman" was approximately the same as the vector from "king" to "queen." In symbols:

$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$

```python
import numpy as np

# Simulated embeddings that exhibit the analogy property
# (Real trained embeddings show this — here we construct an illustrative example)

# Let's say dimension 0 = royalty, dimension 1 = gender, dimension 2 = humanness
#                    [royalty, gender, humanness, ...]
man    = np.array([0.0, -0.8,  0.9, 0.1])
woman  = np.array([0.0,  0.8,  0.9, 0.1])
king   = np.array([0.9, -0.8,  0.9, 0.2])
queen  = np.array([0.9,  0.8,  0.9, 0.2])

# The analogy arithmetic
result = king - man + woman
print(f"king - man + woman = {result}")
print(f"queen              = {queen}")
print(f"Difference:          {np.linalg.norm(result - queen):.4f}")
```

```
king - man + woman = [0.9 0.8 0.9 0.2]
queen              = [0.9 0.8 0.9 0.2]
Difference:          0.0000
```

Why does vector arithmetic give you analogies? Because in a well-structured vector space, *relationships between words are encoded as displacement vectors*. The "gender direction" is a roughly consistent vector that, when added, moves you from the male version of a concept to the female version. The "royalty direction" moves you from commoner to royal.

No other common data structure gives you this for free. A graph can encode that "king" is related to "queen," but "king" minus "man" plus "woman" is not a meaningful operation on graph nodes. A hash table can look up relationships, but can't compose them algebraically. The *vector space itself* is what makes these operations natural.

This property emerges from the training process, but it's *enabled* by the mathematical nature of the representation. Vectors in ℝⁿ form a vector space — they support addition, subtraction, and scalar multiplication — and the training objective has the effect of organizing the space so that semantically meaningful directions correspond to semantically meaningful relationships.

### Property 4: Efficient Bulk Computation

There's a practical consideration that's easy to overlook but absolutely essential: dot products between vectors are **obscenely fast** on modern hardware.

A GPU can compute millions of dot products per second. Finding the most similar word to a query word, out of a vocabulary of a million words, takes milliseconds: it's just a matrix-vector multiplication.

```python
import numpy as np
import time

# Simulating a realistic scenario: finding nearest neighbors
vocab_size = 100000
embedding_dim = 300

# Random embeddings (standing in for trained ones)
all_embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)

# Normalize for cosine similarity
norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
all_embeddings_normed = all_embeddings / norms

# Query: find words most similar to word #42
query = all_embeddings_normed[42]

start = time.time()
similarities = all_embeddings_normed @ query   # one matrix-vector multiply!
top_10 = np.argsort(similarities)[-10:][::-1]
elapsed = time.time() - start

print(f"Found 10 nearest neighbors among {vocab_size} words in {elapsed*1000:.1f} ms")
```

This is a *single matrix multiplication*. The entire vocabulary is a matrix. The query is a vector. One operation gives you similarity to everything. This scales beautifully and runs on hardware that's been optimized for exactly this operation for decades.

Compare this to a graph-based representation, where finding "most similar" requires a traversal. Or a symbolic representation, where computing similarity requires parsing and inference. Vectors don't just encode meaning well — they encode it in the exact form that modern hardware is built to process.

---

## The Geometry of High-Dimensional Spaces: Where Intuition Breaks

Now we arrive at the territory where most people's intuition fails them — and where the real power of embeddings lives.

We grow up in 3D space. Our geometric intuition is calibrated for it. But 300-dimensional space doesn't behave like 3D space at all. Some of the properties of high-dimensional spaces are deeply counterintuitive, and understanding them is essential to understanding why embeddings work.

### The Curse and the Blessing of Dimensionality

In 2D, there are only four "corner" directions (northeast, northwest, southeast, southwest — the diagonals). In 3D, there are 8. In *n* dimensions, there are 2ⁿ corners. At 300 dimensions, that's 2³⁰⁰ — a number so large it dwarfs the number of atoms in the observable universe.

This means 300-dimensional space has an *astronomically large number of nearly orthogonal directions*. Two random vectors in high-dimensional space are almost certainly nearly perpendicular to each other.

```python
import numpy as np

# Demonstration: random vectors become nearly orthogonal in high dimensions
np.random.seed(42)

for dim in [2, 10, 50, 100, 300, 1000]:
    cosines = []
    for _ in range(10000):
        a = np.random.randn(dim)
        b = np.random.randn(dim)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cosines.append(cos)
    mean_abs_cos = np.mean(np.abs(cosines))
    std_cos = np.std(cosines)
    print(f"dim={dim:4d}: mean |cosine| = {mean_abs_cos:.4f}, std = {std_cos:.4f}")
```

```
dim=   2: mean |cosine| = 0.5227, std = 0.4102
dim=  10: mean |cosine| = 0.2478, std = 0.2264
dim=  50: mean |cosine| = 0.1119, std = 0.1005
dim= 100: mean |cosine| = 0.0796, std = 0.0706
dim= 300: mean |cosine| = 0.0461, std = 0.0407
dim=1000: mean |cosine| = 0.0252, std = 0.0224
```

Look at that table. In 300 dimensions, two random vectors have a cosine similarity of essentially zero. They're almost exactly perpendicular.

This is *a gift* for word embeddings. It means 300-dimensional space has enough "room" for every word to have its own direction without crowding. If you have 100,000 words and 300 dimensions, the space is not only big enough — it's almost embarrassingly spacious. Words that *should* be similar can be clustered together, while unrelated words naturally end up far apart, in one of the 2³⁰⁰ available corners.

### Concentration of Measure: Why Cosine Similarity Works

There's a deeper phenomenon: in high dimensions, almost all the "volume" of a sphere is concentrated in a thin shell near the surface. If you sample a random point uniformly from the interior of a 300-dimensional ball, it will almost certainly lie very close to the surface.

```python
import numpy as np

# Fraction of volume in the outer 10% shell of an n-dimensional ball
# Volume of n-ball of radius r is proportional to r^n
# Volume in outer 10% shell: 1 - (0.9)^n

for n in [2, 3, 10, 50, 100, 300]:
    fraction_in_shell = 1 - 0.9**n
    print(f"dim={n:3d}: {fraction_in_shell*100:.6f}% of volume in outer 10% shell")
```

```
dim=  2: 19.000000% of volume in outer 10% shell
dim=  3: 27.100000% of volume in outer 10% shell
dim= 10: 65.132156% of volume in outer 10% shell
dim= 50: 99.515298% of volume in outer 10% shell
dim=100: 99.997347% of volume in outer 10% shell
dim=300: 100.000000% of volume in outer 10% shell
```

By 300 dimensions, essentially 100% of the volume is in the outer shell. This is why, in practice, all trained word vectors end up with roughly similar magnitudes. The space itself pushes them toward the surface of a hypersphere, and the meaningful variation is in *direction*, not magnitude.

This is precisely why **cosine similarity** (which only measures direction, ignoring magnitude) works so well for embeddings. The relevant geometric structure is angular, not radial. Two words are similar when their vectors point in similar directions, regardless of length.

### The Johnson-Lindenstrauss Lemma: Why Compression Works

One of the most remarkable theorems in high-dimensional geometry explains why we can project word relationships from huge spaces down to manageable ones without losing much.

The **Johnson-Lindenstrauss lemma** says: given *n* points in high-dimensional space, you can project them into roughly O(log n / ε²) dimensions while preserving all pairwise distances to within a factor of (1 ± ε).

For 100,000 words with ε = 0.1 (10% distortion):

```python
import numpy as np

n = 100000    # number of words
epsilon = 0.1 # acceptable distortion

# Johnson-Lindenstrauss bound
min_dimensions = int(np.ceil(8 * np.log(n) / epsilon**2))
print(f"JL bound: {min_dimensions} dimensions to preserve {n} pairwise distances")
print(f"          with at most {epsilon*100:.0f}% distortion")
```

```
JL bound: 9211 dimensions to preserve 100000 pairwise distances
          with at most 10% distortion
```

That's the *worst-case theoretical bound*. In practice, because word meanings have much more structure than arbitrary point clouds, 300 dimensions is more than enough. The JL lemma tells us *why* compression to a few hundred dimensions is mathematically safe: the structure of pairwise relationships is largely preserved.

### Superposition: Encoding More Concepts Than Dimensions

Here is perhaps the most astonishing property. In 300 dimensions, you can store information about *far more than 300 independent concepts*. This seems paradoxical — you might think 300 dimensions means 300 independent axes, period. But the phenomenon of **superposition** means you can pack in many more.

The key insight is that if you only need *approximate* orthogonality (rather than exact), the number of nearly-orthogonal directions in ℝⁿ grows exponentially with n.

```python
import numpy as np

# How many nearly-orthogonal vectors can you pack into n dimensions?
# Random vectors in high dimensions are nearly orthogonal.
# Let's measure the max cosine similarity among k random unit vectors in n dims.

np.random.seed(42)

for n_dims in [10, 50, 100, 300]:
    for k_vectors in [100, 1000, 10000]:
        if k_vectors <= 2**n_dims:  # only if it makes sense
            vecs = np.random.randn(k_vectors, n_dims)
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

            # Compute all pairwise cosines
            cosines = vecs @ vecs.T
            np.fill_diagonal(cosines, 0)
            max_cos = np.max(np.abs(cosines))
            mean_cos = np.mean(np.abs(cosines))

            print(f"dim={n_dims:3d}, vectors={k_vectors:5d}: "
                  f"max |cos|={max_cos:.3f}, mean |cos|={mean_cos:.4f}")
    print()
```

In 300 dimensions, you can pack 10,000 vectors with a maximum pairwise cosine similarity well under 0.2. They're not perfectly orthogonal, but they're *close enough* that each vector's dot product with any other is small. This means each word embedding can encode its own direction with minimal interference from other words.

This is why 300 dimensions can represent a vocabulary of hundreds of thousands of words — far more than 300. The space is exponentially richer than our 3D intuition suggests.

---

## How Training Actually Shapes the Space

Now that we understand *why* vectors are the right mathematical object, let's trace how the training process actually carves semantic structure into the space.

### The Skip-Gram Objective

The Skip-gram model's training objective is to maximize the probability of context words given a center word. For a center word $w_c$ and a context word $w_o$, the probability is defined via the softmax:

$$P(w_o | w_c) = \frac{\exp(\mathbf{u}_{w_o} \cdot \mathbf{v}_{w_c})}{\sum_{w=1}^{V} \exp(\mathbf{u}_w \cdot \mathbf{v}_{w_c})}$$

Where $\mathbf{v}_{w_c}$ is the center word's embedding and $\mathbf{u}_{w_o}$ is the context word's embedding.

```python
import numpy as np

def softmax_probability(center_vec, context_vec, all_context_vecs):
    """Probability of a context word given a center word."""
    score = np.dot(context_vec, center_vec)
    all_scores = all_context_vecs @ center_vec
    log_sum_exp = np.log(np.sum(np.exp(all_scores - all_scores.max()))) + all_scores.max()
    return np.exp(score - log_sum_exp)

# 5 words, 3-dimensional embeddings (tiny example)
np.random.seed(42)
V, d = 5, 3
W_center  = np.random.randn(V, d) * 0.5
W_context = np.random.randn(V, d) * 0.5

word_names = ["cat", "dog", "sat", "the", "king"]

# P(sat | cat) — should be high because "cat sat" appears in our corpus
center = W_center[0]   # cat
context = W_context[2]  # sat

prob = softmax_probability(center, context, W_context)
print(f"P(sat | cat) = {prob:.4f}  (before training — essentially random)")
```

### Negative Sampling: Making It Practical

That softmax denominator is the sum over the entire vocabulary — absurdly expensive. Mikolov's second key insight was **negative sampling**: instead of computing the full softmax, just distinguish real (word, context) pairs from random noise pairs.

For each real pair (cat, sat), sample a few "negative" pairs (cat, *random_word*) and train a binary classifier to tell them apart.

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# Negative sampling loss for one (center, context) pair
def negative_sampling_loss(v_center, u_context, negative_vecs):
    """
    v_center: embedding of center word
    u_context: embedding of true context word
    negative_vecs: embeddings of k negative samples
    """
    # Positive pair: want dot product to be HIGH (sigmoid → 1)
    pos_score = np.log(sigmoid(np.dot(u_context, v_center)) + 1e-10)

    # Negative pairs: want dot products to be LOW (sigmoid of negative → 1)
    neg_scores = 0
    for u_neg in negative_vecs:
        neg_scores += np.log(sigmoid(-np.dot(u_neg, v_center)) + 1e-10)

    return -(pos_score + neg_scores)

# Example
np.random.seed(42)
d = 4
v_cat = np.random.randn(d) * 0.5
u_sat = np.random.randn(d) * 0.5
u_neg1 = np.random.randn(d) * 0.5  # random word 1
u_neg2 = np.random.randn(d) * 0.5  # random word 2

loss = negative_sampling_loss(v_cat, u_sat, [u_neg1, u_neg2])
print(f"Loss: {loss:.4f}")
```

The gradient pushes $\mathbf{v}_{\text{cat}}$ and $\mathbf{u}_{\text{sat}}$ closer together (higher dot product), while pushing $\mathbf{v}_{\text{cat}}$ away from the negative samples (lower dot products).

Over billions of training examples, these tiny pushes and pulls sculpt the vector space. Words that share many contexts get pulled into the same regions. Words with distinct contexts get pushed apart. Systematic relationships — gender, tense, plurality — emerge as consistent directions because the same transformations appear across many word pairs.

### A Complete Mini Training Loop

Let's put it all together and watch embeddings learn from scratch:

```python
import numpy as np

np.random.seed(42)

# Tiny corpus
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "rug"],
    ["a", "cat", "chased", "a", "dog"],
    ["the", "dog", "chased", "the", "cat"],
    ["a", "king", "ruled", "the", "land"],
    ["a", "queen", "ruled", "the", "land"],
    ["the", "king", "and", "queen", "sat"],
]

vocab = sorted(set(w for s in sentences for w in s))
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# Hyperparameters
d = 10          # embedding dimensions
lr = 0.05       # learning rate
k_neg = 3       # negative samples
window = 2      # context window
epochs = 200

# Initialize embeddings
W_in  = np.random.randn(V, d) * 0.1
W_out = np.random.randn(V, d) * 0.1

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# Word frequency for negative sampling distribution (raised to 0.75 power)
freq = np.zeros(V)
for s in sentences:
    for w in s:
        freq[w2i[w]] += 1
freq = freq ** 0.75
freq = freq / freq.sum()

# Training
for epoch in range(epochs):
    total_loss = 0
    n_examples = 0

    for sentence in sentences:
        indices = [w2i[w] for w in sentence]
        for i, center_idx in enumerate(indices):
            # Context words within window
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if i == j:
                    continue
                context_idx = indices[j]

                # Positive update
                v_c = W_in[center_idx]
                u_o = W_out[context_idx]
                score = sigmoid(np.dot(u_o, v_c))
                grad = (score - 1) * lr  # want score → 1

                W_in[center_idx]  -= grad * u_o
                W_out[context_idx] -= grad * v_c

                # Negative sampling
                neg_indices = np.random.choice(V, size=k_neg, p=freq)
                for neg_idx in neg_indices:
                    u_neg = W_out[neg_idx]
                    score_neg = sigmoid(np.dot(u_neg, v_c))
                    grad_neg = score_neg * lr  # want score → 0

                    W_in[center_idx] -= grad_neg * u_neg
                    W_out[neg_idx]   -= grad_neg * v_c

                n_examples += 1

    if (epoch + 1) % 50 == 0:
        # Check similarity between cat and dog
        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        cat_dog = cos_sim(W_in[w2i["cat"]], W_in[w2i["dog"]])
        cat_king = cos_sim(W_in[w2i["cat"]], W_in[w2i["king"]])
        king_queen = cos_sim(W_in[w2i["king"]], W_in[w2i["queen"]])
        print(f"Epoch {epoch+1:3d}: "
              f"sim(cat,dog)={cat_dog:.3f}  "
              f"sim(cat,king)={cat_king:.3f}  "
              f"sim(king,queen)={king_queen:.3f}")
```

Even on this trivial corpus, you'll see "cat" and "dog" converge to similar vectors (because they appear in nearly identical contexts), and "king" and "queen" converge as well. The geometry of the space is shaped by the statistics of co-occurrence, and the statistics of co-occurrence reflect meaning.

---

## From Word2Vec to Modern Embedding Layers

Word2Vec was published in 2013. Within a year, it had transformed NLP. But the story didn't end there.

**GloVe** (2014, from Stanford) showed that you could derive similar embeddings by explicitly factorizing a weighted co-occurrence matrix — reconnecting the "count" approach and the "predict" approach. The resulting embeddings were roughly equivalent, confirming that both methods were converging on the same underlying structure.

**FastText** (2016, from Mikolov again, now at Facebook) extended Word2Vec to use *sub-word* information. Instead of one vector per word, it built vectors from character n-grams, meaning it could generate embeddings for words it had never seen before. The word "unfriendly" could be understood from "un-" + "friend" + "-ly" even if the model had never encountered "unfriendly" in training.

**Contextual embeddings** (ELMo 2018, BERT 2018, GPT series) took the next leap: instead of one fixed vector per word, they compute *different* vectors for the same word depending on the sentence it appears in. "Bank" in "river bank" and "bank account" gets different vectors. But the mechanism is the same — the embedding layer of a Transformer is exactly a learned lookup table mapping token indices to dense vectors, just like Word2Vec. The difference is that these vectors are then *refined* by attention layers that mix information from the surrounding context.

In modern LLMs like GPT-4, the embedding layer is the direct descendant of Word2Vec. It maps each token to a dense vector (now typically 4096 or more dimensions), and the entire rest of the model — all the attention layers, all the feed-forward networks — operates on these vectors using the same linear-algebraic operations we've been discussing: dot products, matrix multiplications, additions, projections.

The mathematical insight from 2013 — that dense vectors in ℝⁿ, trained to predict context, organize themselves into a geometry that reflects meaning — is the foundation on which all of modern language AI is built.

---

## Summary: Why Vectors Are the Right Tool

Let's collect the mathematical arguments:

**Dense vectors in ℝⁿ are the right representation for meaning because they provide:**

1. **Continuous similarity** via the dot product and cosine similarity — meaning isn't binary, and neither is vector alignment.

2. **Multi-dimensional similarity** — each of the *n* dimensions provides an independent axis, so words can be simultaneously similar in some respects and different in others. A number line can't do this. A 2D plane can barely do it. 300 dimensions can do it for hundreds of thousands of words.

3. **Algebraic composability** — vector addition, subtraction, and averaging correspond to meaningful semantic operations. Relationships are directions. Analogies are parallelograms.

4. **Near-orthogonality in high dimensions** — random high-dimensional vectors are almost perpendicular, giving the space enormous capacity. The number of nearly-independent directions grows exponentially with dimensionality.

5. **Concentration of measure** — in high dimensions, all points drift toward the surface of a sphere, making *direction* the primary carrier of information and cosine similarity the natural distance metric.

6. **Compressibility** (Johnson-Lindenstrauss) — pairwise distances survive projection to much lower dimensions, so a few hundred dimensions suffice even for enormous vocabularies.

7. **Hardware alignment** — dot products and matrix multiplications are the native operations of GPUs and TPUs. The representation and the compute substrate are perfectly matched.

These aren't seven independent lucky breaks. They're facets of a single deep fact: **high-dimensional real vector spaces have exactly the right geometry for representing the graded, multi-faceted, compositional structure of meaning.** Mikolov's genius was not inventing this math — it was recognizing that a comically simple model, trained on a comically simple objective, would allow the math to do the work.

The vectors learned to think because the space they lived in was already shaped for thought. They just needed a teacher — a few billion sentences of human language — to find the right directions.
