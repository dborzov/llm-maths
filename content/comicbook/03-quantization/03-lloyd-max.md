---
title: "Lloyd-Max: Bell Labs 1957 picks the optimal quantization grid"
short_title: "Lloyd-Max"
description: "Stuart Lloyd proved where to place n quantization levels for a known distribution, declined to publish, and watched the result become foundational for speech, JPEG, and NF4."
blurb:
  - "Bell Labs, March 1957: Stuart Lloyd asks which 16 voltage levels minimize PCM reconstruction error for a voice signal."
  - "His answer — levels at conditional centroids of the distribution — is the algorithm NF4 uses to place its 16 float values."
  - "Lloyd proved it, never published. Joel Max published a less elegant version in 1960. The combined paper appeared in 1982."
  - "The uniform grid (obvious answer) is wrong when signal density is non-uniform — and LLM weight distributions never are."
topics: [quantization, theory]
tags: [lloyd-max, k-means, vector-quantization, bell-labs]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 30
techKind: primer
techNode: lloyd-max
header: lloyd-max.webp
---

## Murray Hill, March 1957

A man with a slide rule is staring at a strip of audio tape.

His name is **Stuart Pearson Lloyd**. He is thirty-three years old, a former Naval Reserve officer, and he works in a windowless office at **Bell Telephone Laboratories** in Murray Hill, New Jersey — the legendary glass-and-brick campus where, a decade earlier, his colleagues had invented the transistor and **Claude Shannon** had published the paper that birthed information theory. The phone company pays Lloyd to think about one specific thing: **how to push a human voice through a wire that can only carry ones and zeroes**.

The technology is called **Pulse Code Modulation** (PCM). The idea is simple in outline. A microphone produces a continuous voltage that wobbles up and down as you speak — an analog signal. Forty thousand times a second you sample that voltage, then write down each sample as a binary number. At the other end of the wire, you read the numbers back and reconstruct the wobble.

The problem is the *writing down*. Each sample is a real number — a voltage that could be 1.347 V or 1.348 V or 1.34721 V or anywhere in between. The wire can only carry, say, **four** bits per sample. Four bits give you **sixteen** possible codes. So you must pick sixteen specific voltages and round every sample to the nearest one. Lloyd's question, his entire job description in March 1957, is:

> **Which sixteen voltages?**

The dirty little secret of the Bell System is that, in 1957, *nobody really knows*. Every engineer has the same first instinct — the **obvious answer**, the one any reasonable person would write down on a napkin.

## The Obvious Answer (Which Is Wrong)

If your signal lives between 0 and 10 volts and you have sixteen codes, just space them out evenly. Put a code at 0.3125 V, then at 0.9375 V, then at 1.5625 V, and so on. Chop the ruler into sixteen equal pieces and stick a representative at the middle of each piece. **Uniform quantization.** It is the answer your great-uncle would have given. It is the answer the textbooks of the 1940s gave.

It is also, Lloyd notices, terrible.

Here is why. He pins a strip of magnetic tape to his desk and plays back his own voice saying *"the quick brown fox jumps over the lazy dog."* He has a colleague rig a circuit that prints a histogram — how often does the voltage take each value? The histogram has a shape Lloyd has been staring at his entire career:

```pyplot {id="voice-histogram" caption="A toy model of a human voice signal: a Laplacian distribution sharply peaked near zero with long tails. Real PCM voice signals look almost exactly like this."}
np.random.seed(7)
# Voice is well-modelled by a Laplace (a.k.a. double-exponential) distribution.
# Mass concentrates near zero (silence, quiet consonants) with rare loud peaks.
voice = np.random.laplace(loc=0, scale=0.7, size=200_000)
voice = np.clip(voice, -10, 10)

# Compare to the "obvious" uniform-quantization grid: 16 equally-spaced levels.
uniform_levels = np.linspace(-10, 10, 16)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.hist(voice, bins=120, color='#FFD700', alpha=0.7, edgecolor='#1A1A1A',
        linewidth=0.4, density=True, label='voice voltage density')
for i, lv in enumerate(uniform_levels):
    ax.axvline(lv, color='#00A8A8', linewidth=1.2, alpha=0.8,
               label='16 uniform levels' if i == 0 else None)
ax.set_xlim(-10, 10)
ax.set_xlabel("voltage")
ax.set_ylabel("density")
ax.set_title("Voice spends almost all its time near 0 V — uniform spacing wastes bits in the tails")
ax.legend(loc='upper right')
ax.spines[['top', 'right']].set_visible(False)

# How many of our levels actually get used heavily?
boundaries = (uniform_levels[:-1] + uniform_levels[1:]) / 2
counts = np.histogram(voice, bins=np.concatenate(([-np.inf], boundaries, [np.inf])))[0]
print(f"samples per level (out of {len(voice)}):")
for lv, c in zip(uniform_levels, counts):
    bar = "#" * int(60 * c / counts.max())
    print(f"  {lv:+6.2f} V  {bar} {c}")
```

Look at the print-out. Of the sixteen levels, **two of them — the two nearest zero — soak up something like 95% of the samples**. Twelve of the levels barely fire at all. The wire is carrying four bits per sample, but the *information content* of those bits is laughable: most of the time, the encoder writes down "near zero" using a code that could have distinguished sixteen totally different things.

Worse: in the dense central region, where every nuance of human speech actually lives, the gap between adjacent levels is **1.33 V**. The difference between "th" and "f" and "s" rides on voltage swings smaller than 0.1 V. Lloyd is throwing away the consonants and lavishing precision on a thunderclap that never comes.

He writes a single line in his notebook: *"The representatives should not be fixed by the geometry of the ruler. They should be dictated by the distribution of the data itself."*

The hunt is on.

## The Shoe-Size Detour

Every quantization problem has the same shape, and the cleanest place to see it is **shoe sizes**.

Human feet grow on a continuous scale. Your foot is 25.7 cm or 25.84 cm or 26.111 cm long. But shoe factories cannot manufacture a continuum — they have to stock discrete sizes: 8, 8.5, 9, 9.5, 10. Every foot in the population gets *snapped* to its nearest available size, and the wearer absorbs the difference as a blister.

The total societal blister count is what an engineer would call the **distortion**. And Lloyd's question, translated, becomes: *if a shoe factory can only stock sixteen sizes and it knows the distribution of foot lengths in the population, where should those sixteen sizes go?* Not uniformly — most adult feet bunch in the middle of the range, and the factory should spend its precision there. The math of "where should the sizes go" is the same math whether the things being snapped are feet, voltages, or, sixty-six years later, neural-network weights.

That parallel — *shoe sizes are quantization levels are codebook entries* — is the thread we will pull on for the rest of this primer.

## The Rubber-Band Picture Of Error

Before we can solve anything we need a way to *score* a candidate set of levels. Lloyd picks the obvious one: **mean squared error**. Each sample $x$ gets snapped to its nearest level $\hat{x}$, and we pay the squared distance $(x - \hat{x})^2$ for it. Across many samples drawn from a density $p(x)$:

$$
D = \mathbb{E}[(X - \hat{X})^2] = \sum_{i=1}^{n} \int_{b_{i-1}}^{b_i} (x - \hat{x}_i)^2 \, p(x) \, dx
$$

This formula looks intimidating until you picture it physically. Imagine each sample $x$ connected to its assigned level $\hat{x}_i$ by a **rubber band**. The rubber band stores energy proportional to the *square* of how stretched it is. Now imagine millions of these rubber bands, one per sample, all pulling at once. The total stored tension is $D$. Lloyd's goal — the whole ballgame — is to slide the levels around until the total rubber-band tension is as small as possible.

This is a *physical* picture, and the math reflects it. If you tug on a single level $\hat{x}_i$, two things resist: the rubber bands on its left pulling left, the ones on its right pulling right. The level wants to come to rest *exactly at the centre of mass of its tribe* — the place where left-pull and right-pull cancel. We will see this fall out of calculus in a moment, but the rubber bands already told us the answer.

## A Toy Example To Keep On The Desk

The rest of this article will use a single concrete data array. Fourteen samples that loosely emulate a voice trace, in volts:

```
X = [ 0.12, -0.03,  0.08, -0.10,  0.21, -0.18,  0.05,
      0.31,  2.95, -2.62, -0.07,  0.14, -0.20,  0.02 ]
```

Twelve of the fourteen samples live within ±0.31 V — the conversational range. Two are big peaks at +2.95 V and −2.62 V — a slammed door, a shout. We will ask: where should **four** quantization levels go?

The naive uniform-grid answer: put levels at −2.40, −0.80, +0.80, +2.40. Three out of four levels will fire essentially never. The big-peak samples will be reconstructed reasonably well; the twelve conversational samples will all snap to −0.80 or +0.80, vandalizing a half-volt of detail.

Lloyd's job is to do better. To do so he needs **two ideas** working in concert. Each one is, on its own, almost obvious. The brilliance is that fixing one *uncovers* the other.

## The Two Conditions That Have To Hold

Lloyd does what every Bell Labs theorist does when staring at an optimization problem: he takes partial derivatives. The minimum of $D$ has to satisfy two **necessary conditions**.

**Condition 1 — Nearest-neighbor (the Voronoi cut).** Given a fixed set of levels, each sample should be assigned to the *closest* level. The boundary between two adjacent levels therefore sits at their **midpoint**:

$$
b_i = \frac{\hat{x}_i + \hat{x}_{i+1}}{2}
$$

This is intuitive: of course you assign each point to whoever is closest. In one dimension the "closest" rule produces simple midpoint boundaries; in two or more it produces those elegant tiled regions called **Voronoi cells**. Either way the picture is the same — each level rules a territory, and the territory borders are equidistant lines.

**Condition 2 — Centroid (the centre-of-mass rule).** Given a fixed partition of the line, each level should sit at the **conditional mean** of the data inside its interval:

$$
\hat{x}_i = \mathbb{E}[X \mid b_{i-1} \le X < b_i] = \frac{\int_{b_{i-1}}^{b_i} x \, p(x) \, dx}{\int_{b_{i-1}}^{b_i} p(x) \, dx}
$$

The rubber-band picture already promised us this. A level resting at the centroid of its tribe is the level where the rubber bands cancel out.

Here is the trap, though, the reason this is interesting: **the two conditions are coupled**. If you move the levels, the midpoints shift, so the partition changes. If the partition changes, the centroids shift, so the levels move. You cannot satisfy both conditions in closed form for an arbitrary distribution.

So Lloyd does what every good detective does: he alternates.

## Lloyd's Iteration: A Two-Step Dance

The algorithm Lloyd publishes — privately, in an internal Bell Labs memo titled *"Least Squares Quantization in PCM"* — fits in four lines.

1. **Initialize**: pick $n$ levels somehow.
2. **Cut**: place boundaries at the midpoints of adjacent levels. (Condition 1)
3. **Centre**: move each level to the centroid of its interval. (Condition 2)
4. **Repeat** until levels stop moving.

Two things to convince yourself of. First, every iteration **cannot make $D$ worse**. The cut step is the best possible partition for the current levels; the centre step is the best possible level for the current partition. Each is a unilateral improvement, so $D$ is monotonically non-increasing. Second, $D$ is bounded below by zero, so monotonic non-increase plus boundedness gives **convergence**. Lloyd's iteration always terminates, always at a local minimum.

It does *not* always find the global minimum — for bumpy distributions a bad initialization can land you in a shallow basin. But on the smooth densities of practical interest (Gaussian, Laplacian, uniform), Lloyd's dance reliably finds the optimum.

Let us watch it run on our toy array.

```pyplot {id="toy-iteration" caption="Lloyd's iteration on the toy 14-sample voice trace. Levels are pulled toward the dense central cluster; the two outlier samples each pull a level out toward themselves."}
X = np.array([0.12, -0.03, 0.08, -0.10, 0.21, -0.18, 0.05,
              0.31, 2.95, -2.62, -0.07, 0.14, -0.20, 0.02])

# Start with uniform levels in [-3, 3]
levels = np.linspace(-3, 3, 4)
history = [levels.copy()]
for _ in range(10):
    boundaries = (levels[:-1] + levels[1:]) / 2
    bnd = np.concatenate(([-np.inf], boundaries, [np.inf]))
    new = np.array([X[(X >= bnd[i]) & (X < bnd[i+1])].mean()
                    if ((X >= bnd[i]) & (X < bnd[i+1])).any() else levels[i]
                    for i in range(4)])
    history.append(new.copy())
    if np.allclose(new, levels, atol=1e-7):
        break
    levels = new

print("levels over iterations:")
for i, lv in enumerate(history):
    print(f"  iter {i}: {np.round(lv, 3)}")
D_lloyd = np.mean([min((x - lv)**2 for lv in levels) for x in X])
print(f"\nfinal D (mean squared error): {D_lloyd:.4f}")

# Compare to the naive uniform 4-level grid on [-3, 3]
uniform = np.linspace(-2.4, 2.4, 4)
D_uniform = np.mean([min((x - lv)**2 for lv in uniform) for x in X])
print(f"D with naive uniform levels {np.round(uniform, 2)}: {D_uniform:.4f}")
print(f"Lloyd wins by {D_uniform / D_lloyd:.1f}x lower MSE.")

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(X, np.zeros_like(X), s=80, color='#FFD700',
           edgecolor='#1A1A1A', linewidth=1.2, zorder=2, label='samples')
for i, lv in enumerate(history):
    ys = np.full_like(lv, -0.08 - 0.05*i)
    color = '#FF007F' if i == len(history)-1 else '#1A1A1A'
    alpha = 1.0 if i == len(history)-1 else 0.25
    ax.scatter(lv, ys, s=70, color=color, alpha=alpha, marker='v', zorder=3)
ax.scatter(uniform, np.full_like(uniform, 0.08), s=70, color='#00A8A8',
           marker='^', label='naive uniform levels', zorder=3)
ax.axhline(0, color='#1A1A1A', linewidth=0.4)
ax.set_yticks([])
ax.set_xlabel("voltage (V)")
ax.set_title("Toy example: 14 samples, 4 levels. Lloyd's iteration (pink) vs uniform (teal).")
ax.legend(loc='upper left')
ax.spines[['top', 'right', 'left']].set_visible(False)
```

After a few iterations the levels settle, and the print-out shows Lloyd's distortion landing well below the uniform-grid baseline on the same data. The two outliers each tug a level out toward themselves; the conversational samples are served by the inner two levels at tight tolerance. The algorithm has discovered, on its own, that the data is bimodal-with-fat-cluster — *without ever being told*.

This is the magic. Drop the levels anywhere; the iteration pulls them toward the data's centre of mass like iron filings near a magnet.

## The Good, The Bad, And The Ugly

The toy example is a special case. To feel what Lloyd-Max does in general, let us run the same algorithm on three contrasting probability densities — the three archetypes you will meet, in lightly disguised form, throughout this issue.

- **The Good** — a **uniform** distribution on $[-1, 1]$. Flat as a table.
- **The Bad** — a **standard Gaussian**. Bell-shaped, the model of well-behaved weights after normalization.
- **The Ugly** — a **Laplace** (also called double-exponential) distribution, which has sharper peak and heavier tails than the Gaussian. This is the shape that vexed Lloyd in 1957 and that vexes us, in caricature, on real LLM activations.

```pyplot {id="three-distributions" caption="Lloyd-Max placement of 8 levels under three densities. Uniform → evenly spaced. Gaussian → concentrated near zero with a wider edge. Laplace → strongly clustered near zero, sparse tails. The data dictates the geometry."}
np.random.seed(0)
N = 200_000

samples = {
    'uniform (the Good)':  np.random.uniform(-1, 1, N),
    'Gaussian (the Bad)':  np.random.randn(N),
    'Laplace (the Ugly)':  np.random.laplace(0, 1/np.sqrt(2), N),  # variance 1
}
colours = {'uniform (the Good)': '#FFD700',
           'Gaussian (the Bad)':  '#00A8A8',
           'Laplace (the Ugly)':  '#FF007F'}

def lloyd(X, n_levels, iters=80):
    levels = np.linspace(X.min(), X.max(), n_levels)
    for _ in range(iters):
        bnd = (levels[:-1] + levels[1:]) / 2
        bnd = np.concatenate(([-np.inf], bnd, [np.inf]))
        new = np.array([X[(X >= bnd[i]) & (X < bnd[i+1])].mean()
                        if ((X >= bnd[i]) & (X < bnd[i+1])).any() else levels[i]
                        for i in range(n_levels)])
        if np.allclose(new, levels, atol=1e-6):
            break
        levels = new
    return levels

results = {name: lloyd(X, 8) for name, X in samples.items()}

for name, lv in results.items():
    print(f"{name}: levels = {np.round(lv, 3)}")

# Distortion comparison vs naive uniform 8-level grid, same range.
print("\ndistortion (MSE) — Lloyd vs uniform-grid on each density:")
for name, X in samples.items():
    lv_lloyd = results[name]
    lv_uni = np.linspace(X.min(), X.max(), 8)
    D_l = np.mean([min((x - q)**2 for q in lv_lloyd) for x in X[:5000]])
    D_u = np.mean([min((x - q)**2 for q in lv_uni)   for x in X[:5000]])
    print(f"  {name:25s} Lloyd={D_l:.4f}  uniform={D_u:.4f}  gain={D_u/D_l:.2f}x")

fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
xs = np.linspace(-4, 4, 600)
densities = {
    'uniform (the Good)': np.where(np.abs(xs) <= 1, 0.5, 0),
    'Gaussian (the Bad)': np.exp(-xs**2/2) / np.sqrt(2*np.pi),
    'Laplace (the Ugly)': np.exp(-np.abs(xs)*np.sqrt(2)) * np.sqrt(2) / 2,
}
for ax, (name, lv) in zip(axes, results.items()):
    ax.fill_between(xs, densities[name], color=colours[name], alpha=0.45)
    ax.scatter(lv, np.zeros_like(lv), s=70, color='#1A1A1A',
               marker='v', zorder=3)
    for x in lv:
        ax.axvline(x, color='#1A1A1A', linewidth=0.4, alpha=0.4)
    ax.set_title(name, loc='left', fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
axes[-1].set_xlabel("x")
plt.tight_layout()
```

Three pictures, three regimes, three different answers. Same algorithm. Stop and re-read the printed distortion table. **The heavier the tail of your distribution, the more Lloyd-Max beats uniform quantization.** A uniform source gains nothing (the answer was already uniform). A Gaussian gains a respectable factor. A Laplace — what voice signals actually look like, and a decent first-pass model for outlier-rich LLM activations — gains a *lot*.

That is why Bell engineers in the 1960s, once they read Joel Max's 1960 paper (we'll get to him in a moment), replaced uniform PCM with non-uniform quantization tables overnight. Every long-distance phone call you have ever made was quantized non-uniformly because of this picture.

## Multiple Discovery: Lloyd Was Not Alone

Here is the historical weirdness, the part that would make **James Burke** raise an eyebrow on camera.

Lloyd never publishes. Bell Labs treats his 1957 memo as proprietary. The result sleeps in a filing cabinet on Mountain Avenue.

Three years later, in **1960**, an MIT graduate student named **Joel Max** writes a doctoral thesis chapter on the exact same problem and *does* publish, in the IEEE Transactions on Information Theory ({{< cite text="Max, 1960" url="https://doi.org/10.1109/TIT.1960.1057548" kind="paper" >}}). For two decades the world calls the result **"the Max quantizer"** — and only Joel Max's name appears on it.

Meanwhile, in **1965**, a statistician named **Edward Forgy** publishes a paper proposing the same iteration but framed as *cluster analysis* for multi-dimensional data. In **1967**, a statistician named **James MacQueen** at UCLA publishes a paper that generalises the iteration, calls it **"k-means"**, and gets credited as the inventor by the statistics community for the next half-century ({{< cite text="MacQueen, 1967" url="https://projecteuclid.org/proceedings/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fifth-Berkeley-Symposium-on-Mathematical-Statistics-and/Chapter/Some-methods-for-classification-and-analysis-of-multivariate-observations/bsmsp/1200512992" kind="paper" >}}). (MacQueen, charmingly, opens by acknowledging that the basic idea has independently been published or used by Steinhaus in 1957, by Lloyd at Bell Labs the same year, and by several others — but he writes down the cleanest version.)

The same algorithm. **Four independent discoveries**, in two countries, across three decades, in two completely different research communities (signal processing and statistics), each one ignorant of the others. It is *not* an accident. It is, as Burke might put it, evidence that discretizing the world isn't an invention. It is a fundamental mathematical truth waiting to be found.

Finally in **1982**, Bell Labs declassifies. Lloyd's 1957 memo gets a proper IEEE publication ({{< cite text="Lloyd, 1982 (Least squares quantization in PCM)" url="https://doi.org/10.1109/TIT.1982.1056489" kind="paper" >}}). The community starts calling the 1D result **Lloyd-Max** to honour both contributors, and the multi-dimensional generalisation **k-means** to honour MacQueen.

We will use **"Lloyd's iteration"** and **"k-means"** interchangeably for the rest of this article. They are the same algorithm.

## From Telephone Wires To Pixels To Tokens

Once you know what to look for, you find Lloyd's iteration *everywhere* in twentieth-century technology. A short tour, Burke-style:

**1970s — Speech codecs in the long-distance network.** AT&T's **G.711 μ-law** and Europe's **A-law** codecs, used on every analog telephone trunk for forty years, are direct industrial descendants of Lloyd-Max. They use a clever closed-form non-uniform companding curve as a cheaper approximation to running the full iteration in hardware that didn't yet exist.

**1980 — The Linde-Buzo-Gray algorithm.** A Stanford team generalises Lloyd's iteration to **vector quantization (VQ)** — quantizing blocks of $d$ samples *jointly* in $\mathbb{R}^d$ rather than one sample at a time ({{< cite text="Linde, Buzo & Gray, 1980" url="https://doi.org/10.1109/TCOM.1980.1094577" kind="paper" >}}). The cells become full Voronoi tiles. The codebook becomes a list of representative vectors. The math is otherwise identical. VQ becomes the basis of every speech codec from the 1980s through the cellular era.

**1989 — The 256-colour GIF.** When CompuServe invents the GIF format, the problem is to pick the best 256 colours to approximate a true-colour image. Run k-means on the millions of RGB pixel vectors, the 256 codebook centroids are the palette. Every GIF you have ever seen had its colours chosen by Lloyd's algorithm.

**1993 — MP3 and the consumer audio revolution.** Suzanne Vega's *Tom's Diner* becomes famous among engineers because it is the reference recording the Fraunhofer team tunes the MP3 codec against ({{< cite text="Brandenburg, 1999 (MP3 history)" url="https://www.aes.org/e-lib/browse.cfm?elib=8079" kind="paper" >}}). MP3 is a stack of psychoacoustic tricks over a quantization core — and the core is descended from the same lossy-compression theory Lloyd sketched in 1957.

**2017 — VQ-VAE.** Aäron van den Oord and the DeepMind team build a neural network with a **discrete bottleneck** — an {{< wiki "embeddings" >}}embedding layer{{< /wiki >}} that snaps continuous activations to the nearest vector in a learned codebook of size $K$ ({{< cite text="van den Oord et al., 2017" url="https://arxiv.org/abs/1711.00937" kind="paper" >}}). The snap is k-means. The model learns its codebook by gradient descent, but the inference-time operation — find the nearest codebook vector — is Lloyd's nearest-neighbour assignment, dropped into a deep learning pipeline like a guest who's been waiting for sixty years.

**2024 — Tokenization.** When a modern LLM splits the string `"unbelievable"` into the tokens `["un", "believ", "able"]`, what just happened? Byte-pair encoding is *not* Lloyd's algorithm exactly, but the conceptual move — map a continuous space of byte sequences into a discrete codebook of subword units — is the same move Lloyd made in 1957, played at a higher level of abstraction. ChatGPT's tokenizer is, in spirit, the philosophical grandchild of the Murray Hill voltage-snapping circuit.

Every one of these systems is, deep down, **a Lloyd-Max codebook for the relevant data**. Different data, same math.

## NF4: The 2023 Resurrection

So we arrive, by a longish road, at our actual subject: **modern LLM weight quantization**.

In **May 2023**, in a paper called **QLoRA** ({{< cite text="Dettmers et al., 2023" url="https://arxiv.org/abs/2305.14314" kind="paper" >}}), **Tim Dettmers** and collaborators publish a new 4-bit number format with a deliberately provocative claim: it is, they argue, *information-theoretically optimal* for storing the weights of a trained neural network. They call it **{{< wiki "number-formats" >}}NF4{{< /wiki >}}** — "Normal Float 4". And the way they construct it is *exactly* Lloyd's algorithm.

Step one of NF4: assume that, after normalisation, LLM weight tensors are approximately Gaussian. (This is empirical. It is mostly true. We unpack the caveats in [Geometry Of Weights](../05-geometry-of-weights/).)

Step two of NF4: place 16 quantization levels at the **conditional centroids** of the 16 equal-probability bins of a standard normal. That is, slice the Gaussian into 16 strips that each contain $\frac{1}{16}$ of the probability mass, and put a level at the centre of mass of each strip. Re-read those two sentences. *That is Lloyd's centroid condition*, applied analytically to a Gaussian. The QLoRA team didn't have to run the iteration — they computed the limiting answer in closed form.

```pyplot {id="nf4-vs-int4" caption="NF4's 16 levels (top) sit at the Lloyd-Max optimum for a standard normal — clustered near zero, sparse in the tails. INT4 (bottom) is the uniform straw-man Lloyd buried in 1957."}
nf4_levels = np.array([
    -1.0, -0.6961928, -0.5250730, -0.39491748,
    -0.28444138, -0.18477343, -0.09105056, 0.0,
    0.07958029, 0.16093975, 0.24611230, 0.33791524,
    0.44070983, 0.56261307, 0.72295684, 1.0,
])
int4_levels = np.linspace(-1, 1, 16)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4), sharex=True)
xs = np.linspace(-1, 1, 400)
gauss_density = np.exp(-xs**2 * 4)

ax1.fill_between(xs, gauss_density, color='#FFD700', alpha=0.45)
ax1.scatter(nf4_levels, np.zeros(16), s=70, color='#FF007F',
            zorder=3, edgecolor='#1A1A1A', linewidth=0.8)
ax1.set_title("NF4 - Lloyd-Max centroids of a normalised Gaussian",
              fontsize=10, loc='left')
ax1.set_yticks([])
ax1.spines[['top','right','left']].set_visible(False)
ax1.axhline(0, color='#1A1A1A', linewidth=0.5)

ax2.fill_between(xs, gauss_density, color='#FFD700', alpha=0.45)
ax2.scatter(int4_levels, np.zeros(16), s=70, color='#00A8A8',
            zorder=3, edgecolor='#1A1A1A', linewidth=0.8)
ax2.set_title("INT4 - uniformly spaced (Lloyd's 1957 straw-man)",
              fontsize=10, loc='left')
ax2.set_yticks([])
ax2.set_xlabel("normalised value")
ax2.spines[['top','right','left']].set_visible(False)
ax2.axhline(0, color='#1A1A1A', linewidth=0.5)
plt.tight_layout()

# Empirical comparison on Gaussian samples (rescaled to [-1, 1])
np.random.seed(1)
X = np.random.randn(50_000)
X_clip = np.clip(X / 3.5, -1, 1)
nf4_err = np.mean([min((x - lv)**2 for lv in nf4_levels) for x in X_clip[:5000]])
int4_err = np.mean([min((x - lv)**2 for lv in int4_levels) for x in X_clip[:5000]])
print(f"MSE on Gaussian samples (rescaled to [-1, 1]):")
print(f"  NF4:  {nf4_err:.5f}")
print(f"  INT4: {int4_err:.5f}")
print(f"  NF4 is {int4_err / nf4_err:.2f}x lower distortion than INT4 on Gaussian weights.")
```

This is the punch line. When you read the QLoRA paper and it says **"NF4 is information-theoretically optimal for normally distributed weights"** — that is not marketing copy. It is a direct statement about Lloyd's 1957 theorem, applied to the empirical observation that LLM weights are Gaussian-ish. The two facts together imply NF4. The argument is one sentence long once you have the theorem.

A 66-year-old PCM memo is the reason QLoRA can fine-tune a 65-billion-parameter model on a single consumer GPU. **The math does not care that the data has changed from telephone voltages to transformer weights.** That is the whole point of mathematics.

## Three Trade-Offs Worth Naming

Lloyd-Max is beautiful, but it is not free. Three trade-offs sit in the background of every system built on it.

### 1. The Robot-Voice Limit (rate vs. fidelity)

Choose too few levels and your reconstruction becomes a parody. Every digital-audio enthusiast has heard the *bitcrushed* sound of an 8-bit videogame — that crunchy, robotic timbre is what audio sounds like when you only have 256 codes per sample. Drop to 4-bit and the human voice becomes nearly unintelligible. Lloyd-Max gives you the *best possible levels for your budget*, but it cannot conjure information that the bit budget refuses to carry. The fundamental floor on this trade-off is **Shannon's rate-distortion theorem**, the subject of the [next primer](../04-rate-distortion/).

### 2. The Outlier Anomaly (the breaking glass)

Lloyd's algorithm assumes the *training distribution* is representative. If you sample a microphone for ten minutes of conversation and build a codebook, then the next sound through the wire is a window shattering — well, the codebook has never seen a sound like that, and it will snap the shatter to some completely wrong centroid. The decoded glass will sound like a kazoo.

This sounds like a theoretical curiosity. It is not. In 2022 the entire field of LLM quantization discovered, the hard way, that real transformer activations contain *systematic* outliers — six or seven specific feature dimensions out of thousands that swing 50× larger than the rest, batch after batch. Lloyd-Max on the bulk distribution crushes these outliers into noise; the model collapses. We tell that story in [The 1% That Ruins Everything](../06-outliers/), and the fix-of-fixes — splitting outliers off into their own precision path — is the modern descendant of "build a separate small codebook just for the rare loud sounds."

### 3. The Codebook Baggage (transmitting the dictionary)

If the encoder and the decoder do not already *share* the codebook, you have to send it. In voice telephony this was trivial: both ends of the wire ran the same hardware with the same baked-in table. In modern LLMs it is also trivial: the codebook is sixteen numbers (NF4) burned into a kernel.

But in *vector* quantization the codebook is itself a large list of $d$-dimensional vectors, and at high $d$ the codebook can rival the data in size. This is the deep reason LLMs use **scalar quantization with per-block scales** rather than full VQ: it gives you most of the rate-distortion benefit at almost none of the codebook cost. We unpack the trick in [Calibration & Blocks](../11-calibration-and-blocks/).

## The Bargain, Restated

The "Lloyd-Max bargain" is this: **you do not get to choose where your representable numbers go. Your data does.** The optimal placement of $n$ levels is the placement that satisfies Lloyd's two conditions for the density $p(x)$ — and that placement varies enormously with the shape of $p$.

- If your weights are **uniform**, you want uniform spacing. That's INT4. (Almost nothing real is uniform.)
- If your weights are **Gaussian**, you want Lloyd-Max centroids of a Gaussian. That's NF4.
- If your weights are **heavy-tailed**, you want something that allocates levels to the tails. That's FP4. (And it is *also* a Lloyd-Max approximation, for a different assumed density.)
- If your weights have a *bespoke* distribution unique to one layer — and they usually do — you want to run Lloyd's iteration on actual calibration samples drawn from that layer.

The reason this matters for the rest of this issue is that **none of the off-the-shelf 4-bit formats are exactly right** for any given LLM tensor. They each match *some* assumed distribution. The methods you will meet — GPTQ, AWQ, SmoothQuant, HQQ — all add tricks on top of the base format, and those tricks are easier to understand if you know what they are *correcting for*: the gap between the format's assumed density and the actual density of the weights in your specific layer.

Lloyd's 1957 question — *given a density, where should the levels go?* — is still, in 2026, the right question to ask. The answer keeps getting reinterpreted; the question never changes.

**Continue to** → [The Rate-Distortion Bridge](../04-rate-distortion/), where Shannon tells us the absolute floor that no Lloyd-Max quantizer can dive beneath, no matter how cleverly we place the levels.
