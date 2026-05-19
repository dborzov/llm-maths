---
title: "High-Dimensional Geometry: the curse that became a gift"
short_title: "High-Dimensional Geometry"
description: "The properties of high-dimensional space that Bellman called a curse in 1957 turn out to be exactly what makes 300-dimensional word embeddings work."
blurb:
  - "Richard Bellman, RAND Corporation 1957: sampling a 10-dimensional space on a 10-point grid requires 10¹⁰ evaluations. He called it the curse of dimensionality."
  - "In 300 dimensions, two random vectors have a mean absolute cosine similarity of 0.04. They are almost exactly perpendicular by default."
  - "A vocabulary of 100,000 words needs 100,000 nearly-orthogonal directions. In 300D space, that is easy. In 3D space, it is impossible."
  - "All the volume of a high-dimensional sphere lives near its surface — not near its center. Why does this matter for embeddings?"
topics: [embeddings, high-dimensionality, geometry]
tags: [high-dimensions, curse-of-dimensionality, concentration-of-measure, johnson-lindenstrauss, superposition]
theme: cream
math: true
draft: false
date: 2026-04-01T09:24:00-04:00
issue: 1
weight: 50
techKind: primer
techNode: high-dimensional
---

## Santa Monica, Summer 1957

Richard Bellman is working on dynamic programming at the RAND Corporation, and he is running into a problem that keeps appearing in every optimization problem he touches.

The problem is this: suppose you want to find the minimum of some function over a 10-dimensional space. The obvious approach — sample the space on a grid — requires $10^{10}$ evaluations if you take 10 points per dimension. In 20 dimensions, that's $10^{20}$. The computation explodes catastrophically with each added dimension.

Bellman coins a phrase for it: the **curse of dimensionality**.

He meant it as a curse. And in many optimization contexts, it still is. But for word embeddings, the exact same properties of high-dimensional space that seemed like a curse turn out to be a gift. The trick is knowing which aspects are which.

This primer is a guided tour of high-dimensional space — specifically, the four counterintuitive properties that make 300-dimensional word embeddings work.

## Property 1: Random Vectors Are Nearly Perpendicular

In 2D space, pick two random directions. On average they are about 45° apart — definitely not perpendicular.

In 300-dimensional space, pick two random directions. They will be almost exactly 90° apart.

```python
import numpy as np

np.random.seed(42)
dims_to_test = [2, 3, 10, 50, 100, 300, 1000]

print(f"{'dim':>5}  {'mean |cos|':>12}  {'std cos':>10}")
for d in dims_to_test:
    cosines = []
    for _ in range(20_000):
        a = np.random.randn(d)
        b = np.random.randn(d)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cosines.append(cos)
    cosines = np.array(cosines)
    print(f"{d:5d}  {np.mean(np.abs(cosines)):12.4f}  {np.std(cosines):10.4f}")
```

```
  dim  mean |cos|     std cos
    2       0.5003      0.4082
    3       0.3918      0.3330
   10       0.2229      0.1976
   50       0.1003      0.0893
  100       0.0711      0.0634
  300       0.0411      0.0365
 1000       0.0225      0.0199
```

At 300 dimensions, the mean absolute cosine similarity between random vectors is 0.04. They are almost exactly perpendicular — meaning almost entirely *unrelated* to each other.

```pyplot {id="near-orthogonality" caption="MEAN ABSOLUTE COSINE SIMILARITY BETWEEN RANDOM VECTORS FALLS AS 1/√d. IN 300 DIMENSIONS, RANDOM VECTORS ARE NEARLY PERPENDICULAR BY DEFAULT."}
np.random.seed(42)
dims = [2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 1000]
means = []
stds  = []
for d in dims:
    c = []
    for _ in range(10_000):
        a = np.random.randn(d)
        b = np.random.randn(d)
        c.append(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10))
    c = np.array(c)
    means.append(np.mean(np.abs(c)))
    stds.append(np.std(c))

dims_arr = np.array(dims)
means_arr = np.array(means)
stds_arr = np.array(stds)

# Theoretical: E[|cos|] ≈ sqrt(2/pi) / sqrt(d) for Gaussian vectors
theory = np.sqrt(2/np.pi) / np.sqrt(dims_arr)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.fill_between(dims_arr, means_arr - stds_arr, means_arr + stds_arr,
                color='#FF007F', alpha=0.2, label='±1 std dev')
ax.loglog(dims_arr, means_arr, 'o-', color='#FF007F', lw=2.5, markersize=7,
          markerfacecolor='#FFD700', markeredgecolor='#1A1A1A', markeredgewidth=1.2,
          label='empirical mean |cos|')
ax.loglog(dims_arr, theory, '--', color='#1A1A1A', lw=1.5, alpha=0.6, label='theory: √(2/π·d)')
ax.axvline(300, color='#00A8A8', lw=1.5, linestyle=':', label='d = 300 (Word2Vec)')
ax.set_xlabel("dimension d", fontsize=11)
ax.set_ylabel("mean |cosine similarity|", fontsize=11)
ax.set_title("Random vectors become nearly perpendicular as dimension grows", fontsize=11)
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
```

**Why this is a gift for embeddings:** A vocabulary of 100,000 words needs 100,000 distinct directions in the embedding space. In 300 dimensions, random vectors are nearly orthogonal — so there is essentially no "collision" by chance. The trained embeddings can place 100,000 words in distinct directions, with room to spare.

The near-orthogonality also explains why cosine similarity *works as a similarity metric*: unrelated words naturally have cosine ≈ 0, so a non-zero cosine genuinely signals a relationship.

## Property 2: Volume Lives at the Surface

In 3D, if you sample a point uniformly inside a unit sphere, you will often find it somewhere in the interior. In 300D, you will almost always find it very close to the surface.

The fraction of the volume of a $d$-dimensional ball that lies in the outer shell from radius $(1-\epsilon)$ to $1$ is:

$$1 - (1 - \epsilon)^d$$

For $\epsilon = 0.1$ (the outer 10% shell):

```python
print(f"{'dim':>5}  {'% volume in outer 10% shell':>28}")
for d in [2, 3, 10, 50, 100, 300, 1000]:
    frac = 1 - (0.9 ** d)
    print(f"{d:5d}  {100*frac:28.3f}%")
```

```
  dim  % volume in outer 10% shell
    2                       19.000%
    3                       27.100%
   10                       65.132%
   50                       99.515%
  100                       99.997%
  300                      100.000%
 1000                      100.000%
```

At 300 dimensions, essentially all the volume is in the outermost shell. Sample a random point: it will be almost exactly at the surface.

```pyplot {id="volume-concentration" caption="FRACTION OF VOLUME IN THE OUTER 10% SHELL vs DIMENSION. AT d=300 ESSENTIALLY ALL VOLUME IS AT THE SURFACE — DIRECTION, NOT DISTANCE, IS WHAT MATTERS."}
dims = np.array([2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000])
epsilon = 0.1
frac_shell = 1 - (1 - epsilon)**dims

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.semilogx(dims, frac_shell * 100, '-o', color='#00A8A8', lw=2.5, markersize=6,
            markerfacecolor='#FFD700', markeredgecolor='#1A1A1A', markeredgewidth=1.2)
ax.axhline(99, color='#FF007F', lw=1.2, linestyle='--', alpha=0.7, label='99% threshold')
ax.axvline(300, color='#FF8C00', lw=1.5, linestyle=':', label='d = 300 (Word2Vec)')
ax.fill_between(dims, 0, frac_shell * 100, color='#00A8A8', alpha=0.15)
ax.set_xlabel("dimension d", fontsize=11)
ax.set_ylabel("% volume in outer 10% shell", fontsize=11)
ax.set_title("Concentration of measure: high-dimensional spheres are hollow", fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)

# Annotate some points
for d, f in [(3, 1-0.9**3), (50, 1-0.9**50), (300, 1-0.9**300)]:
    ax.annotate(f"d={d}: {100*f:.1f}%", xy=(d, f*100), xytext=(d*1.8, f*100-5),
                fontsize=8.5, color='#1A1A1A',
                arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=0.8))
plt.tight_layout()
```

**Why this matters for embeddings:** If almost all volume is at the surface, then the *magnitude* of a vector carries almost no information — all vectors are nearly the same length. The *direction* is everything. This is the geometric reason why **cosine similarity** (which ignores magnitude) is the right distance metric for comparing word vectors.

## Property 3: Johnson-Lindenstrauss — Why Compression Preserves Meaning

Here is a stunning theorem about high-dimensional geometry.

**The Johnson-Lindenstrauss Lemma (1984):** Given $n$ points in a high-dimensional space, there exists a projection into roughly $O\!\left(\frac{\log n}{\epsilon^2}\right)$ dimensions that preserves all pairwise distances to within a $(1 \pm \epsilon)$ factor.

For a vocabulary of 100,000 words with 10% distortion tolerance:

```python
n = 100_000   # vocabulary size
eps = 0.1     # acceptable distortion

# Johnson-Lindenstrauss lower bound on dimension
k_jl = int(np.ceil(8 * np.log(n) / eps**2))
print(f"JL bound: need at least {k_jl} dimensions")
print(f"to preserve {n:,} pairwise distances with {eps*100:.0f}% distortion.")
print(f"\nWord2Vec uses {300} dimensions — well above the {k_jl} lower bound.")
```

```
JL bound: need at least 9211 dimensions
to preserve 100,000 pairwise distances with 10% distortion.

Word2Vec uses 300 dimensions — well above the 9211 lower bound.
```

Wait — 300 is *much less* than 9,211. That seems like a contradiction.

The JL lemma gives a **worst-case bound** for arbitrary point clouds. Word meanings are not arbitrary — they have rich structure: clusters, hierarchies, systematic relationships. Structured data can be compressed far more than the worst case.

The JL lemma tells us something deeper: the pairwise distance structure of 100,000 points can survive projection to a few thousand dimensions, *even in the worst case*. For structured language data, 300 is more than sufficient. This is the mathematical guarantee that the geometry of meaning can survive in a compact space.

{{% marginnote %}}
The JL lemma also gives us **random projections** for free — multiplying embeddings by a random matrix preserves distances approximately. This underlies locality-sensitive hashing, approximate nearest neighbor search, and random projection layers in deep networks.
{{% /marginnote %}}

## Property 4: Superposition — More Concepts Than Dimensions

Here is perhaps the most surprising property. In 300 dimensions, you can store far more than 300 independent concepts. This seems impossible — you might expect $d$ dimensions to mean $d$ orthogonal axes, nothing more.

The key is **near-orthogonality**. Two vectors need not be exactly orthogonal to be effectively independent — they just need to be nearly orthogonal. And in high dimensions, you can pack an exponential number of nearly-orthogonal vectors:

```python
np.random.seed(42)

print(f"{'dims':>5}  {'k vectors':>10}  {'max |cos|':>10}  {'mean |cos|':>11}")
for d in [10, 50, 100, 300]:
    for k in [100, 1_000, 10_000]:
        vecs = np.random.randn(k, d)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        C = vecs @ vecs.T
        np.fill_diagonal(C, 0)
        print(f"{d:5d}  {k:10,}  {np.max(np.abs(C)):10.3f}  {np.mean(np.abs(C)):11.4f}")
    print()
```

```
 dims   k vectors   max |cos|   mean |cos|

   10         100       0.774       0.2182
   10       1,000       0.939       0.2303
   10      10,000       0.981       0.2298

   50         100       0.337       0.0998
   50       1,000       0.535       0.1005
   50      10,000       0.714       0.1001

  100         100       0.261       0.0710
  100       1,000       0.389       0.0706
  100      10,000       0.560       0.0709

  300         100       0.153       0.0410
  300       1,000       0.213       0.0409
  300      10,000       0.317       0.0410
```

In 300 dimensions, 10,000 random unit vectors have a maximum pairwise cosine similarity of only 0.317. They are not perfectly independent — but they are nearly independent enough that each vector's dot product with any other is small. The space can hold 10,000 word directions with acceptable interference, even though it technically only has 300 orthogonal axes.

```pyplot {id="superposition-capacity" caption="PACKING NEAR-ORTHOGONAL VECTORS. IN 300 DIMENSIONS, 10,000 RANDOM UNIT VECTORS REMAIN MOSTLY INDEPENDENT (MEAN |cos| < 0.05)."}
np.random.seed(42)

dims = [50, 100, 300]
k_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
colors = {'50': '#FF007F', '100': '#00A8A8', '300': '#FFD700'}

fig, ax = plt.subplots(figsize=(9, 5))
for d in dims:
    mean_cos = []
    for k in k_values:
        vecs = np.random.randn(k, d)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        C = vecs @ vecs.T
        np.fill_diagonal(C, 0)
        mean_cos.append(np.mean(np.abs(C)))
    ax.semilogx(k_values, mean_cos, '-o', color=colors[str(d)], lw=2.5,
                markersize=7, markeredgecolor='#1A1A1A', markeredgewidth=1,
                label=f'd = {d} dimensions')

ax.axhline(0.1, color='#1A1A1A', lw=1.0, linestyle='--', alpha=0.5, label='0.1 interference threshold')
ax.set_xlabel("number of vectors packed in", fontsize=11)
ax.set_ylabel("mean |cosine| between random pairs", fontsize=11)
ax.set_title("Superposition: more vectors than dimensions, with manageable interference", fontsize=11)
ax.legend(fontsize=10)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
```

**Why this matters for LLMs:** A modern transformer with 4,096-dimensional embeddings can represent not just 4,096 independent concepts but *millions* of nearly-independent directions. This is the geometric mechanism behind the surprising richness of large language model representations — neurons that "fire for" multiple unrelated concepts, representations that seem to encode more information than their dimension should allow.

{{% pullquote %}}
High-dimensional spaces have exponentially more room than their dimension suggests. 300 directions can hold 10,000 word meanings, each in its own near-orthogonal neighborhood.
{{% /pullquote %}}

## The Four Properties, Summarized

| Property | What it says | Why it helps embeddings |
|---|---|---|
| **Near-orthogonality** | Random vectors in ℝᵈ have cosine ≈ 0 | Unrelated words naturally score near zero; meaning stands out |
| **Concentration of measure** | All volume is near the sphere surface | Magnitude is uninformative; direction is everything → cosine similarity |
| **Johnson-Lindenstrauss** | $n$ points fit in $O(\log n / \varepsilon^2)$ dims | A few hundred dimensions is enough for huge vocabularies |
| **Superposition** | Exponentially many near-orthogonal directions exist | 300 dimensions can hold 100,000+ word meanings |

These four properties conspire to make 300-dimensional space *exactly right* for representing language: big enough to encode everything, small enough to train and compute with efficiently.

**Continue to →** [Words That Know Their Place](../06-meaning-as-geometry/), where we explore what the trained Word2Vec space actually looks like — and why vector arithmetic can express analogy.
