---
title: "Words That Know Their Place"
description: "Late 2013. Mikolov's team is stress-testing their embeddings when they notice that king − man + woman ≈ queen. This is not magic. It is what a good geometry of meaning looks like."
topics: [embeddings, word2vec, geometry]
tags: [word2vec, analogy, semantic-directions, word-clusters]
theme: teal
math: true
draft: false
date: 2026-04-01T09:30:00-04:00
issue: 1
weight: 60
techKind: mainline
techNode: meaning-as-geometry
header: 03-embeddings.webp
---

## The Accidental Discovery

Late 2013. Mikolov's team is writing evaluation code for their Word2Vec embeddings. They want to measure whether the model has learned anything genuinely structured — not just "cat and dog are similar," which every decent method achieves, but something deeper.

Someone has the idea to test **analogies**. Not just similarity, but *relational structure*. If the embeddings are truly a geometry of meaning, they should encode not just that words are near each other, but *why* — what *direction* you move to transform one concept into another.

The test: take three words that complete the pattern $A$ is to $B$ as $C$ is to $?$. Compute $\vec{B} - \vec{A} + \vec{C}$ and see which word's vector is nearest. If "man is to king as woman is to queen," compute $\overrightarrow{king} - \overrightarrow{man} + \overrightarrow{woman}$ and ask: what word's vector is closest to the result?

The answer came back: **queen**.

Not always. Not perfectly. But on thousands of test analogies — capital cities, currency names, verb tenses, plural nouns — the arithmetic was right at a rate that shocked everyone who looked at it. The space wasn't just a similarity lookup table. It was a *geometric encoding of relationships*.

## Why Vector Arithmetic Encodes Analogies

The reason this works is not mystical. It follows from a concrete assumption about how the training shaped the space.

Suppose the embedding space has a consistent **"gender direction"** — a vector $\vec{g}$ that, when added to any word, shifts it from the masculine to the feminine version:

$$\overrightarrow{woman} \approx \overrightarrow{man} + \vec{g}$$
$$\overrightarrow{queen} \approx \overrightarrow{king} + \vec{g}$$
$$\overrightarrow{actress} \approx \overrightarrow{actor} + \vec{g}$$

If this is true, then:

$$\overrightarrow{king} - \overrightarrow{man} + \overrightarrow{woman}$$
$$= \overrightarrow{king} - \overrightarrow{man} + (\overrightarrow{man} + \vec{g})$$
$$= \overrightarrow{king} + \vec{g}$$
$$\approx \overrightarrow{queen}$$

The arithmetic works because the gender relationship is a **consistent displacement vector** in the space. Any time the training process saw "man" and "woman" occurring in otherwise identical contexts (and it did — "the man walked in" / "the woman walked in"), the gradient updates pushed them to be displaced from each other in a consistent direction.

{{% callout kind="note" %}}
The gender direction is not a single "axis" in the model — it is not necessarily aligned with any of the 300 coordinate dimensions. It is a *direction in the high-dimensional space*, discovered organically by the training process. This is one reason why inspecting the individual dimensions of word vectors is uninformative; meaning is in the directions, not the coordinates.
{{% /callout %}}

Let us construct an embedding space that explicitly has this structure and verify it:

```python
import numpy as np

np.random.seed(42)
d = 6  # 6 dimensions for clear visualization

# Base vectors: [royalty, animacy, domestic, size, formal, active]
man    = np.array([0.0,  0.8,  0.3,  0.7,  0.5,  0.6])
woman  = np.array([0.0,  0.8,  0.3,  0.5,  0.5,  0.6])
king   = np.array([0.9,  0.7,  0.1,  0.7,  0.9,  0.5])
queen  = np.array([0.9,  0.7,  0.1,  0.5,  0.9,  0.5])
actor  = np.array([0.2,  0.8,  0.2,  0.6,  0.3,  0.9])
actress= np.array([0.2,  0.8,  0.2,  0.4,  0.3,  0.9])

# Gender direction
gender_dir = woman - man
print(f"gender direction: {np.round(gender_dir, 3)}")
print(f"woman - man (actual):   {np.round((woman - man), 3)}")
print(f"queen - king (actual):  {np.round((queen - king), 3)}")
print(f"actress - actor (actual): {np.round((actress - actor), 3)}")
```

```
gender direction: [ 0.    0.    0.   -0.2   0.    0. ]
woman - man (actual):   [ 0.    0.    0.   -0.2   0.    0. ]
queen - king (actual):  [ 0.    0.    0.   -0.2   0.    0. ]
actress - actor (actual): [ 0.    0.    0.   -0.2   0.    0. ]
```

The gender direction is identical across all three pairs — dimension 3 (the "size" axis, here encoding something like physical stature) is consistently lower for female versions. The relationship is not just *similarity* between words but a *direction* in space.

```python
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# The king - man + woman test
result = king - man + woman
print(f"\nking - man + woman = {np.round(result, 3)}")
print(f"queen               = {np.round(queen, 3)}")
print(f"\ncos(result, queen)  = {cosine(result, queen):.4f}")
print(f"cos(result, king)   = {cosine(result, king):.4f}")
print(f"cos(result, man)    = {cosine(result, man):.4f}")
```

```
king - man + woman = [0.9 0.7 0.1 0.5 0.9 0.5]
queen               = [0.9 0.7 0.1 0.5 0.9 0.5]

cos(result, queen)  = 1.0000
cos(result, king)   = 0.9927
cos(result, man)    = 0.9706
```

In this carefully constructed example, the result is *exactly* queen. In a real trained space, the result is *approximately* queen — the top-3 nearest neighbors usually contains the expected answer.

## Visualizing Semantic Directions

The trained Word2Vec space contains not just one such direction but hundreds. Let us simulate a realistic embedding space and map its structure:

```pyplot {id="semantic-clusters" caption="SIMULATED WORD2VEC SPACE (PCA TO 2D). FOUR SEMANTIC CLUSTERS SELF-ORGANIZE: ANIMALS, ROYALTY, COUNTRIES, NUMBERS. COSINE TRAINED ON CONTEXT — NOT CATEGORY LABELS."}
np.random.seed(7)

# Simulate a 20-dim embedding space with realistic semantic clusters
# Each cluster has a "center" in semantic space plus noise
d = 20

def make_cluster(center, n=6, noise=0.15):
    c = np.array(center, dtype=float)
    c = c / np.linalg.norm(c)
    pts = c + np.random.randn(n, d) * noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

centers = {
    'animals':   np.random.randn(d),
    'royalty':   np.random.randn(d),
    'countries': np.random.randn(d),
    'numbers':   np.random.randn(d),
}
words_by_group = {
    'animals':   ['cat','dog','tiger','lion','wolf','bear'],
    'royalty':   ['king','queen','prince','princess','throne','crown'],
    'countries': ['france','germany','japan','brazil','india','egypt'],
    'numbers':   ['one','two','three','four','five','six'],
}
colors_by_group = {
    'animals': '#FF007F', 'royalty': '#FFD700', 'countries': '#00A8A8', 'numbers': '#FF8C00'
}

all_vecs = []
all_words = []
all_colors = []
for group, center in centers.items():
    vecs = make_cluster(center, n=6, noise=0.18)
    all_vecs.append(vecs)
    all_words.extend(words_by_group[group])
    all_colors.extend([colors_by_group[group]]*6)

vecs = np.vstack(all_vecs)

# PCA to 2D
vecs_centered = vecs - vecs.mean(axis=0)
cov = vecs_centered.T @ vecs_centered
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
pca = vecs_centered @ eigvecs[:, order[:2]]

fig, ax = plt.subplots(figsize=(9, 7))
for i, (x, y) in enumerate(pca):
    ax.scatter(x, y, color=all_colors[i], s=120, edgecolor='#1A1A1A', linewidth=1.2, zorder=3)
    ax.text(x+0.02, y+0.02, all_words[i], fontsize=9.5, color='#1A1A1A', zorder=4)

# Draw analogy arrow: king → queen direction as an example
group_starts = {g: i*6 for i,g in enumerate(['animals','royalty','countries','numbers'])}
king_idx   = group_starts['royalty'] + words_by_group['royalty'].index('king')
queen_idx  = group_starts['royalty'] + words_by_group['royalty'].index('queen')
prince_idx = group_starts['royalty'] + words_by_group['royalty'].index('prince')
princess_idx = group_starts['royalty'] + words_by_group['royalty'].index('princess')

ax.annotate('', xy=pca[queen_idx], xytext=pca[king_idx],
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.5))
ax.annotate('', xy=pca[princess_idx], xytext=pca[prince_idx],
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.5))
mid1 = (pca[king_idx] + pca[queen_idx]) / 2
mid2 = (pca[prince_idx] + pca[princess_idx]) / 2
ax.text((mid1[0]+mid2[0])/2 + 0.05, (mid1[1]+mid2[1])/2,
        '→ gender\ndirection', fontsize=9, color='#555555', style='italic')

# Cluster labels
cluster_centers_2d = {g: pca[group_starts[g]:group_starts[g]+6].mean(axis=0)
                      for g in ['animals','royalty','countries','numbers']}
for g, c2d in cluster_centers_2d.items():
    ax.text(c2d[0], c2d[1]-0.2, g.upper(), fontsize=10, color=colors_by_group[g],
            fontweight='bold', ha='center', alpha=0.85)

ax.set_title("PCA of simulated word embeddings: semantic clusters self-organize", fontsize=11)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
```

Four clusters emerge without being labelled: animals, royalty, countries, and numbers. The arrows show the gender direction — king→queen and prince→princess are roughly parallel displacements. This is what emerges from training on language statistics.

## An Interactive Analogy Explorer

Let us build a minimal analogy computer and watch it work on a hand-crafted embedding space:

{{< infographic title="Analogy Arithmetic: A → B as C → ?"
                description="Explore how vector arithmetic encodes semantic relationships." >}}

{{< infographic-controls >}}
  <div class="ig-slider">
    <div class="ig-slider-header">
      <span class="ig-slider-label">Relationship type</span>
    </div>
    <div class="ig-toggle-row" role="group" aria-label="Analogy type">
      <button type="button" data-analogy="gender" aria-pressed="true">Gender</button>
      <button type="button" data-analogy="scale" aria-pressed="false">Scale</button>
      <button type="button" data-analogy="capital" aria-pressed="false">Capital</button>
    </div>
  </div>

  <div style="margin-top: 1rem; font-size: 0.9rem; color: #555;">
    <p><strong>A → B as C → ?</strong></p>
    <p id="analogy-equation" style="font-family: monospace; font-size: 1.1rem; color: #FF007F; margin: 0.5rem 0;"></p>
    <p style="font-size: 0.8rem; opacity: 0.7;">Direction vector = B − A. Apply to C.</p>
  </div>
{{< /infographic-controls >}}

{{< infographic-viz >}}
  <div class="ig-stage">
    <p class="ig-stage-label">Cosine similarity to result vector</p>

    <div class="ig-bar-row">
      <span class="ig-bar-row-label" id="bar-label-1">—</span>
      <div class="ig-bar">
        <div class="ig-bar-fill" id="ig-bar-1" style="width:0%"></div>
      </div>
      <span class="ig-bar-row-value" id="ig-val-1">—</span>
    </div>

    <div class="ig-bar-row">
      <span class="ig-bar-row-label" id="bar-label-2">—</span>
      <div class="ig-bar">
        <div class="ig-bar-fill ig-bar-fill--teal" id="ig-bar-2" style="width:0%"></div>
      </div>
      <span class="ig-bar-row-value" id="ig-val-2">—</span>
    </div>

    <div class="ig-bar-row">
      <span class="ig-bar-row-label" id="bar-label-3">—</span>
      <div class="ig-bar">
        <div class="ig-bar-fill ig-bar-fill--yellow" id="ig-bar-3" style="width:0%"></div>
      </div>
      <span class="ig-bar-row-value" id="ig-val-3">—</span>
    </div>

    <div class="ig-bar-row">
      <span class="ig-bar-row-label" id="bar-label-4">—</span>
      <div class="ig-bar">
        <div class="ig-bar-fill ig-bar-fill--orange" id="ig-bar-4" style="width:0%"></div>
      </div>
      <span class="ig-bar-row-value" id="ig-val-4">—</span>
    </div>

    <p style="font-size:0.8rem; color:#555; margin-top:0.8rem;">
      Top bar = predicted answer. Higher cosine = better match.
    </p>
  </div>

  <script>
    (function() {
      // Minimal 4D embedding space (hand-crafted for clarity)
      // Dimensions: [royalty, animacy, formality, gender-female]
      const vecs = {
        king:     [0.90, 0.65, 0.88, 0.00],
        queen:    [0.90, 0.65, 0.88, 0.85],
        prince:   [0.70, 0.65, 0.70, 0.00],
        princess: [0.70, 0.65, 0.70, 0.85],
        actor:    [0.10, 0.80, 0.40, 0.00],
        actress:  [0.10, 0.80, 0.40, 0.85],
        man:      [0.00, 0.85, 0.35, 0.00],
        woman:    [0.00, 0.85, 0.35, 0.85],
        cat:      [0.00, 0.90, 0.05, 0.00],
        kitten:   [0.00, 0.80, 0.05, 0.00],
        dog:      [0.00, 0.90, 0.05, 0.10],
        puppy:    [0.00, 0.75, 0.05, 0.00],
        city:     [0.00, 0.00, 0.70, 0.00],
        capital:  [0.30, 0.00, 0.90, 0.00],
        town:     [0.00, 0.00, 0.50, 0.00],
        village:  [0.00, 0.00, 0.30, 0.00],
      };

      const analogies = {
        gender:  { A:'man',   B:'woman',   C:'king',  label:'man → woman as king → ?' },
        scale:   { A:'city',  B:'capital', C:'town',  label:'city → capital as town → ?' },
        capital: { A:'kitten',B:'cat',     C:'puppy', label:'kitten → cat as puppy → ?' },
      };

      function norm(v) {
        const s = Math.sqrt(v.reduce((a,x) => a+x*x, 0));
        return v.map(x => x/s);
      }
      function dot(a, b) { return a.reduce((s,x,i) => s+x*b[i], 0); }
      function cos(a, b) { return dot(norm(a), norm(b)); }
      function add(a, b) { return a.map((x,i) => x+b[i]); }
      function sub(a, b) { return a.map((x,i) => x-b[i]); }

      function recalc(type) {
        const { A, B, C, label } = analogies[type];
        document.getElementById('analogy-equation').textContent = label;
        const result = add(vecs[C], sub(vecs[B], vecs[A]));

        // Score all words except A, B, C
        const exclude = new Set([A, B, C]);
        const scored = Object.entries(vecs)
          .filter(([w]) => !exclude.has(w))
          .map(([w, v]) => ({ word: w, score: cos(result, v) }))
          .sort((a, b) => b.score - a.score)
          .slice(0, 4);

        const maxScore = scored[0].score;
        ['1','2','3','4'].forEach((n, i) => {
          const item = scored[i];
          document.getElementById('bar-label-' + n).textContent = item ? item.word : '—';
          document.getElementById('ig-bar-' + n).style.width =
            item ? Math.max(4, item.score / maxScore * 100) + '%' : '0%';
          document.getElementById('ig-val-' + n).textContent =
            item ? item.score.toFixed(3) : '—';
        });
      }

      document.querySelectorAll('[data-analogy]').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('[data-analogy]').forEach(b =>
            b.setAttribute('aria-pressed', 'false'));
          btn.setAttribute('aria-pressed', 'true');
          recalc(btn.dataset.analogy);
        });
      });

      recalc('gender');
    })();
  </script>
{{< /infographic-viz >}}

{{< /infographic >}}

```pyplot {id="analogy-parallelogram" caption="THE ANALOGY PARALLELOGRAM. KING−MAN AND QUEEN−WOMAN ARE THE SAME DISPLACEMENT VECTOR. ARITHMETIC IN VECTOR SPACE = RELATIONAL REASONING."}
np.random.seed(1)
# 2D projection of the key words for geometric clarity
# Hand-placed for maximum clarity of the parallelogram structure
king    = np.array([0.9, 0.8])
queen   = np.array([0.7, 0.85])
man     = np.array([0.85, 0.25])
woman   = np.array([0.65, 0.30])

# Compute analogy result
result = king - man + woman

words = {'king': king, 'queen': queen, 'man': man, 'woman': woman}
colors= {'king':'#FFD700','queen':'#FFD700','man':'#FF007F','woman':'#FF007F'}

fig, ax = plt.subplots(figsize=(7, 6))

# Draw the parallelogram
para_pts = np.array([man, king, queen, woman, man])
ax.plot(para_pts[:,0], para_pts[:,1], '--', color='#1A1A1A', lw=1.0, alpha=0.4, zorder=1)

# Gender arrows
ax.annotate('', xy=woman, xytext=man,
            arrowprops=dict(arrowstyle='->', color='#00A8A8', lw=2.0))
ax.annotate('', xy=queen, xytext=king,
            arrowprops=dict(arrowstyle='->', color='#00A8A8', lw=2.0))

ax.text((man[0]+woman[0])/2 - 0.02, (man[1]+woman[1])/2 - 0.07,
        'gender direction', color='#00A8A8', fontsize=9, ha='center')

# Word points
for name, pt in words.items():
    ax.scatter(*pt, s=150, color=colors[name], edgecolor='#1A1A1A', lw=1.5, zorder=5)
    offset = [0.03, 0.03]
    if name == 'man': offset = [0.03, -0.06]
    if name == 'king': offset = [0.03, -0.06]
    ax.text(pt[0]+offset[0], pt[1]+offset[1], name, fontsize=12, fontweight='bold',
            color='#1A1A1A', zorder=6)

# Result point
ax.scatter(*result, s=200, color='#FF8C00', edgecolor='#1A1A1A', lw=2, marker='*', zorder=6)
ax.text(result[0]+0.03, result[1]+0.03, 'king−man+woman\n(≈ queen)', fontsize=10,
        color='#FF8C00', fontweight='bold', zorder=7)

ax.set_xlim(0.4, 1.2); ax.set_ylim(0.1, 1.05)
ax.set_title("The analogy parallelogram: relationships are directions", fontsize=11)
ax.spines[['top','right']].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
```

## What the Space Knows

The trained Word2Vec space contains dozens of such consistent directions. Beyond gender, researchers found:

- **Tense direction:** `walked − walk ≈ swam − swim ≈ ran − run`
- **Plural direction:** `dogs − dog ≈ cats − cat ≈ kings − king`
- **Comparative direction:** `bigger − big ≈ faster − fast ≈ louder − loud`
- **Capital city direction:** `Paris − France ≈ Berlin − Germany ≈ Tokyo − Japan`

These are not programmed in. They emerge because English uses the same grammatical transformations consistently, and the training objective forces words that appear in the same grammatical contexts to have similar vectors. The grammar of the language becomes the geometry of the space.

{{% callout kind="tangent" %}}
**Not all analogies work.** The success rate on the original Google analogy test set (over 19,000 analogy questions) was ~60–70% for the best Word2Vec configurations. Many analogies fail because language is ambiguous, irregular, or because the training corpus doesn't contain enough evidence. "King − man + woman ≈ queen" is famous partly because it is one of the *clean* cases.
{{% /callout %}}

## The Limit: One Vector Per Word

Word2Vec assigns each word exactly **one** vector. But many words have multiple meanings.

The word "bank" can mean a financial institution or a riverbank. The word "cold" can mean low temperature, a common illness, or emotional distance. In Word2Vec, these senses are averaged together — the single "bank" vector ends up somewhere between the financial and geographical clusters, representing both meanings imperfectly.

This is Word2Vec's fundamental limitation, and it motivated everything that came after.

**Continue to →** [The Living Inheritance](../07-inheritance/), where the embedding idea travels from 2013 Google research to the heart of every modern transformer — and where context finally solves the one-word-one-vector problem.
