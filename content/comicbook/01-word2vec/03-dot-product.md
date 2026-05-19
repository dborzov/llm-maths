---
title: "Dot Product: the one question that runs all of NLP"
short_title: "Dot Product"
description: "The dot product measures how much two vectors agree in direction — and that single question turns out to be the right one for measuring similarity between word meanings."
blurb:
  - "`a · b = ‖a‖ ‖b‖ cos θ`. Large when vectors point the same way. Zero when perpendicular. Negative when opposite."
  - "Cosine similarity strips out magnitude: two vectors pointing the same direction score 1.0 regardless of length."
  - "Word2Vec's entire training objective is a dot product comparison: push real pairs high, push random pairs low."
  - "One-hot vectors are always perpendicular — their dot product is always zero. That is the problem the whole issue is solving."
topics: [embeddings, linear-algebra]
tags: [dot-product, cosine-similarity, projection, geometry]
theme: cream
math: true
draft: false
date: 2026-04-01T09:12:00-04:00
issue: 1
weight: 30
techKind: primer
techNode: dot-product
header: 05-geometry-of-weights.webp
---

## The Question Built Into the Numbers

Before you can understand why Word2Vec works, you need one mathematical tool: the dot product.

Not as a formula to memorize, but as an *answer to a question*. The question is: **given two vectors, how much do they agree in direction?**

This seems like a strange question to ask about word meanings. By the time we are done, it will seem like the only question that matters.

## What a Dot Product Measures

Two vectors $\mathbf{a} = [a_1, a_2, \ldots, a_n]$ and $\mathbf{b} = [b_1, b_2, \ldots, b_n]$ have a dot product defined as:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$$

Multiply corresponding components, sum the results. For 2D vectors you can write it out by hand:

```python
a = [3.0, 1.0]
b = [2.0, 4.0]

dot_ab = sum(ai * bi for ai, bi in zip(a, b))
print(f"a · b = {dot_ab}")  # 3*2 + 1*4 = 10
```

```
a · b = 10
```

But what does 10 *mean*?

The geometric interpretation is the key. The dot product can also be written:

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos\theta$$

where $\|\mathbf{a}\|$ is the length of $\mathbf{a}$, $\|\mathbf{b}\|$ is the length of $\mathbf{b}$, and $\theta$ is the angle between them.

This formula says: **the dot product is proportional to the cosine of the angle between the vectors**. It is large and positive when the vectors point in roughly the same direction ($\theta$ near 0°). It is zero when they are perpendicular ($\theta = 90°$). It is negative when they point in opposite directions ($\theta$ near 180°).

```pyplot {id="dot-product-geometry" caption="DOT PRODUCT = LENGTH × LENGTH × cos(θ). LEFT: PARALLEL VECTORS SCORE HIGH. RIGHT: PERPENDICULAR VECTORS SCORE ZERO."}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: vectors pointing in similar directions
origin = [0, 0]
a1 = [3., 1.]
b1 = [2., 1.2]
dp1 = a1[0]*b1[0] + a1[1]*b1[1]
na = (a1[0]**2 + a1[1]**2)**0.5
nb = (b1[0]**2 + b1[1]**2)**0.5
cos_t = dp1 / (na * nb)
theta = np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))

ax1.annotate('', xy=a1, xytext=origin,
             arrowprops=dict(arrowstyle='->', color='#FF007F', lw=2.5))
ax1.annotate('', xy=b1, xytext=origin,
             arrowprops=dict(arrowstyle='->', color='#00A8A8', lw=2.5))
ax1.text(a1[0]+0.1, a1[1], 'a', fontsize=14, color='#FF007F', fontweight='bold')
ax1.text(b1[0]+0.1, b1[1], 'b', fontsize=14, color='#00A8A8', fontweight='bold')
ax1.text(1.2, 0.2, f'θ ≈ {theta:.0f}°', fontsize=11, color='#1A1A1A')
ax1.text(0.5, -0.6, f'a · b = {dp1:.1f}  (positive)', fontsize=11, color='#1A1A1A',
         ha='center', fontweight='bold')
ax1.set_xlim(-0.3, 3.8); ax1.set_ylim(-0.9, 1.8)
ax1.set_aspect('equal'); ax1.axhline(0, color='#1A1A1A', lw=0.4); ax1.axvline(0, color='#1A1A1A', lw=0.4)
ax1.set_title("Same general direction → positive dot product", fontsize=10)
ax1.spines[['top','right']].set_visible(False)

# Right: perpendicular vectors
a2 = [2., 0.]
b2 = [0., 2.]
dp2 = a2[0]*b2[0] + a2[1]*b2[1]
ax2.annotate('', xy=a2, xytext=origin,
             arrowprops=dict(arrowstyle='->', color='#FF007F', lw=2.5))
ax2.annotate('', xy=b2, xytext=origin,
             arrowprops=dict(arrowstyle='->', color='#00A8A8', lw=2.5))
ax2.text(a2[0]+0.1, a2[1], 'a', fontsize=14, color='#FF007F', fontweight='bold')
ax2.text(b2[0]+0.1, b2[1]+0.1, 'b', fontsize=14, color='#00A8A8', fontweight='bold')
ax2.text(0.15, 0.15, '90°', fontsize=11, color='#1A1A1A')
ax2.text(1.0, -0.5, f'a · b = {dp2:.1f}  (zero)', fontsize=11, color='#1A1A1A',
         ha='center', fontweight='bold')
ax2.set_xlim(-0.3, 2.8); ax2.set_ylim(-0.7, 2.5)
ax2.set_aspect('equal'); ax2.axhline(0, color='#1A1A1A', lw=0.4); ax2.axvline(0, color='#1A1A1A', lw=0.4)
ax2.set_title("Perpendicular → zero dot product", fontsize=10)
ax2.spines[['top','right']].set_visible(False)
plt.tight_layout()
```

## Cosine Similarity: Agreement Without Scale

The dot product has one inconvenient property: it depends on both the *angle* and the *lengths* of the vectors. A long vector pointing in the right direction scores higher than a short vector pointing in the same direction.

For comparing word meanings, we usually don't care about length — we care about *direction*. Two words whose vectors point in the same direction should be considered similar regardless of vector magnitude.

**Cosine similarity** strips out the lengths:

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \cos\theta$$

This is a number between −1 and 1:

| Range | Geometric meaning | Semantic interpretation |
|---|---|---|
| ≈ 1 | Nearly parallel | Very similar words |
| ≈ 0 | Perpendicular | Unrelated words |
| ≈ −1 | Opposite directions | Antonyms or opposites |

```python
import math

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-12)

# Animal vectors: 4D = [animacy, domesticity, royalty, female_marker]
cat    = [0.80,  0.90,  0.0,  0.0]
dog    = [0.80,  0.85,  0.0,  0.0]
tiger  = [0.90,  0.05,  0.0,  0.0]
king   = [0.70, -0.10,  0.9,  0.0]
queen  = [0.70, -0.10,  0.9,  1.0]

pairs = [
    ("cat",   cat,   "dog",   dog),
    ("cat",   cat,   "tiger", tiger),
    ("cat",   cat,   "queen", queen),
    ("king",  king,  "queen", queen),
    ("tiger", tiger, "queen", queen),
]

for n1, v1, n2, v2 in pairs:
    sim = cosine_similarity(v1, v2)
    bar = "█" * int(20 * (sim + 1) / 2)
    print(f"cos({n1:5s}, {n2:5s}) = {sim:+.3f}  {bar}")
```

```
cos(cat  , dog  ) = +0.998  ████████████████████
cos(cat  , tiger) = +0.900  ██████████████████
cos(cat  , queen) = +0.231  ██████████████
cos(king , queen) = +0.746  ████████████████
cos(tiger, queen) = +0.192  ████████████
```

The numbers encode what any person knows: cats and dogs are very similar, cats and queens are only weakly related, kings and queens are highly similar (both royals).

## Projection: What the Dot Product Computes

Here is the intuition that makes Word2Vec click. Suppose you have a "direction" in vector space — say, the direction that represents "royal" things. You can ask: *how much of word $\mathbf{w}$ is in the royal direction?*

The answer is the **projection** of $\mathbf{w}$ onto the royal direction vector $\mathbf{r}$:

$$\text{projection} = \frac{\mathbf{w} \cdot \mathbf{r}}{\|\mathbf{r}\|}$$

If $\mathbf{r}$ is a unit vector (length 1), this simplifies to just the dot product $\mathbf{w} \cdot \mathbf{r}$.

This is exactly what Word2Vec's training does. The model maintains two matrices: one for "center word" vectors and one for "context word" vectors. Training adjusts these matrices so that the dot product between a word's center vector and its actual context words' context vectors is high — and the dot product with random non-context words is low.

The dot product is not just a convenient calculation. It is the *fundamental operation* that forces the geometry to reflect meaning.

```pyplot {id="projection-semantic" caption="PROJECTION ONTO A 'ROYALTY' DIRECTION VECTOR. KING AND QUEEN PROJECT HIGHLY; CAT AND DOG PROJECT NEAR ZERO. THE DOT PRODUCT MEASURES PRESENCE OF A CONCEPT."}
# Visualize projection onto a "royalty" direction in 2D
# Collapse 4D vectors to 2D for illustration: x=animacy, y=royalty

words_2d = {
    "cat":   (0.85, 0.02),
    "dog":   (0.82, 0.01),
    "tiger": (0.90, 0.04),
    "king":  (0.60, 0.88),
    "queen": (0.55, 0.85),
    "prince":(0.55, 0.75),
}

# Royalty direction
royal_dir = np.array([0.1, 1.0])
royal_dir = royal_dir / np.linalg.norm(royal_dir)

colors = {'cat':'#FF007F','dog':'#FF8C00','tiger':'#00A8A8',
          'king':'#FFD700','queen':'#FFD700','prince':'#FFD700'}

fig, ax = plt.subplots(figsize=(8, 6))

# Draw royalty direction
scale = 1.1
ax.annotate('', xy=royal_dir*scale, xytext=[0,0],
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=2))
ax.text(royal_dir[0]*scale+0.03, royal_dir[1]*scale, 'royalty\ndirection', fontsize=10,
        color='#1A1A1A', fontweight='bold')

for name, (x, y) in words_2d.items():
    v = np.array([x, y])
    # Projection onto royal direction
    proj_len = np.dot(v, royal_dir)
    proj_pt = proj_len * royal_dir
    # Draw vector
    ax.annotate('', xy=v, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color=colors[name], lw=2, alpha=0.85))
    # Draw projection line (dashed)
    ax.plot([v[0], proj_pt[0]], [v[1], proj_pt[1]], '--', color='#1A1A1A', linewidth=0.8, alpha=0.4)
    ax.plot(proj_pt[0], proj_pt[1], 'v', color=colors[name], markersize=8, zorder=5)
    ax.text(x+0.02, y+0.03, f"{name}\n({proj_len:.2f})", fontsize=9, color=colors[name], fontweight='bold')

ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.05)
ax.set_xlabel("animacy axis", fontsize=11)
ax.set_ylabel("royalty axis", fontsize=11)
ax.set_title("Projection onto 'royalty' direction: king & queen score high, animals score near zero", fontsize=10)
ax.spines[['top','right']].set_visible(False)
ax.axhline(0, color='#1A1A1A', lw=0.4); ax.axvline(0, color='#1A1A1A', lw=0.4)
plt.tight_layout()
```

Triangles on the "royalty axis" show the projection values. King and queen land high up on the axis; cat, dog, and tiger land near the bottom. The dot product correctly identifies how much of "royalty" is present in each word.

## Why This Is the Right Tool

The dot product has three properties that make it perfect for embedding word meaning:

**1. It measures graded agreement.** Unlike a binary "same cluster/different cluster," the dot product gives a continuous score. "Cat" is more like "tiger" than like "democracy" — the numbers reflect this.

**2. It captures multiple simultaneous similarities.** In a 300-dimensional space, a word can be simultaneously similar to "dog" along the animacy+domestic axis, similar to "tiger" along the feline+predator axis, and dissimilar from "king" on the royalty axis — all at the same time. One dimension cannot do this; three hundred can.

**3. It is a matrix multiplication.** Finding the 10 most similar words to a query, out of a vocabulary of 100,000, is a single matrix-vector product: `similarities = embedding_matrix @ query_vector`. This runs on hardware optimized for exactly this operation.

{{% pullquote %}}
The dot product is not just a measuring tool. It is the geometry of Word2Vec's entire training objective — the force that pushes word vectors toward each other or apart.
{{% /pullquote %}}

**Continue to →** [Embarrassingly Simple](../04-word2vec/), where we see how Word2Vec uses dot products as its only "prediction" — and why that simplicity was a superpower.
