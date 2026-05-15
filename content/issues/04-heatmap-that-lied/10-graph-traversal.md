---
title: "BFS as Reasoning"
description: "Why a breadth-first search through a small graph of hex-hash node ids is the right architectural stress test for a long-context transformer. The textbook 1959 algorithm, reframed as the cleanest measurement of in-context reasoning ever proposed."
topics: [algorithms, long-context, reasoning]
tags: [bfs, graph-traversal, graphwalks, complexity, multi-hop]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 100
techKind: primer
techNode: graph-traversal
header: default.webp
---

## A 1959 Algorithm Comes For Long Context

In **1959**, **Edward F. Moore** publishes a paper titled *"The shortest path through a maze"* in the *Proceedings of the International Symposium on the Theory of Switching*. The setup is wartime-pragmatic: route a telephone call through a network of switching stations with the fewest hops. Moore's algorithm — what we now call **breadth-first search**, BFS — has been a CS-101 standard for sixty-six years. It is the first non-trivial graph algorithm any student learns. It runs in $O(V + E)$ on a graph with $V$ nodes and $E$ edges. It is *embarrassingly* well understood.

Why is a 1959 algorithm at the center of the 2025 long-context benchmark wars?

Because it has one property that is *exactly* what the field needed: **its depth cannot be parallelised across {{< wiki "attention" >}}attention{{< /wiki >}} layers**. Each level of the BFS frontier *must* be computed after the previous level. No amount of clever batching changes this fact. And as we showed in [Scan vs Think](../04-retrieval-vs-reasoning/), this is the precise capability that single-pass transformer attention does not have. **BFS in context is the cleanest possible architectural stress test for in-context reasoning.**

This primer walks through what BFS is, why {{< wiki "long-context-benchmarks" >}}GraphWalks{{< /wiki >}} built its benchmark around it, and the napkin math that says BFS depth $k$ in a graph of $n$ nodes requires *at least* $k$ dependent attention hops to complete.

## BFS, From Scratch

Start with the basic operation. You have a directed graph: a set of nodes and a set of directed edges. From node $X$, you want to know "which nodes are reachable in *exactly* $k$ hops?" The algorithm:

```
frontier_0 = {start_node}
visited    = {start_node}
for depth in 1, 2, ..., k:
    frontier_d = {}
    for node in frontier_{d-1}:
        for child in graph.children(node):
            if child not in visited:
                frontier_d.add(child)
                visited.add(child)
return frontier_k
```

Four observations.

1. **Frontier $d$ depends on frontier $d-1$.** You cannot compute the depth-3 frontier without first having the depth-2 frontier. This is *sequential* in $d$ — the steps are causally ordered.
2. **Within a single depth, the lookups are independent.** Each node in frontier $d-1$ contributes its children to frontier $d$ in parallel. So you can parallelise *width* but not *depth*.
3. **`visited` is a state variable.** The algorithm requires keeping track of what's been seen to avoid revisiting. This is a *running* piece of state that must persist across iterations.
4. **The output frontier grows with branching factor.** If every node has $b$ children, the depth-$k$ frontier has up to $b^k$ entries. Even moderate $b$ and $k$ produces a frontier of dozens of nodes.

These four properties together make BFS *the* canonical multi-hop benchmark. Property (1) demands depth. Property (2) demands width. Property (3) demands working memory. Property (4) demands the model handle a frontier whose size is more than one item.

## A Worked Example

Let's walk BFS through a small hex-hash graph the way GraphWalks does. Nine nodes, each labelled with a four-character hex prefix. Directed edges. Start at node `a3f9`:

```pyplot {id="bfs-worked" caption="A worked BFS on a small directed graph of hex-hash nodes. Start at a3f9 (the orange seed). The depth-1 frontier (teal) is its direct children. The depth-2 frontier (pink) is the children of those children, minus anything already visited. Each level depends on the previous; no level can be computed in parallel with the next."}
import matplotlib.patches as mp
nodes = {
    "a3f9": (0.0, 0.0),
    "5b8d": (-1.2, -0.8),
    "7e4a": (0.0, -1.0),
    "d8b6": (1.2, -0.8),
    "9c4f": (-1.6, -2.0),
    "e2d4": (-0.6, -2.0),
    "1f8c": (0.0, -2.4),
    "b3c2": (1.6, -2.0),
    "f5a1": (1.0, -3.0),
}
edges = [
    ("a3f9", "5b8d"), ("a3f9", "7e4a"), ("a3f9", "d8b6"),
    ("5b8d", "9c4f"), ("5b8d", "e2d4"),
    ("7e4a", "1f8c"),
    ("d8b6", "1f8c"), ("d8b6", "b3c2"),
    ("b3c2", "f5a1"), ("e2d4", "f5a1"),
]

# Compute BFS frontiers
def bfs_levels(start, edges, max_depth=3):
    children = {n: [] for n in nodes}
    for u, v in edges:
        children[u].append(v)
    visited = {start}
    levels = [[start]]
    for _ in range(max_depth):
        frontier = []
        for n in levels[-1]:
            for c in children[n]:
                if c not in visited:
                    frontier.append(c)
                    visited.add(c)
        if not frontier:
            break
        levels.append(frontier)
    return levels

levels = bfs_levels("a3f9", edges, 3)

depth_color = {0: '#FF8C00', 1: '#00A8A8', 2: '#FF007F', 3: '#FFD700'}
node_depth = {n: d for d, lv in enumerate(levels) for n in lv}

fig, ax = plt.subplots(figsize=(8.5, 5.2))
# Draw edges
for u, v in edges:
    x0, y0 = nodes[u]; x1, y1 = nodes[v]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color='#1A1A1A', lw=1.2))
# Draw nodes
for label, (x, y) in nodes.items():
    d = node_depth.get(label, None)
    color = depth_color.get(d, '#cccccc') if d is not None else '#cccccc'
    ax.add_patch(mp.Circle((x, y), 0.22, color=color, ec='#1A1A1A', lw=1.5,
                           zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            fontweight='bold', zorder=4)

ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-3.6, 0.7)
ax.set_aspect('equal')
ax.set_yticks([])
ax.set_xticks([])
ax.set_title("BFS from a3f9 - depth 0 (orange), 1 (teal), 2 (pink)")
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

# Legend
legend_items = [(d, c, f"depth {d}") for d, c in depth_color.items() if d <= 2]
for i, (d, c, lab) in enumerate(legend_items):
    ax.add_patch(mp.Circle((-2.2 + i*1.2, 0.5), 0.10, color=c,
                           ec='#1A1A1A', lw=1.2))
    ax.text(-2.05 + i*1.2, 0.5, lab, fontsize=9, va='center')

print("BFS frontiers:")
for d, frontier in enumerate(levels):
    print(f"  depth {d}: {frontier}")
print()
print(f"GraphWalks BFS-2 from a3f9 asks: 'list all nodes at depth exactly 2'")
print(f"Correct answer: {levels[2] if len(levels) > 2 else '<none>'}")
```

Stare at that small graph for a moment. To answer the GraphWalks question *"what are all nodes at depth exactly 2 from `a3f9`?"*, you must:

1. **Find `a3f9` in the prompt** and read its children. That's `5b8d`, `7e4a`, `d8b6`. (One scan.)
2. **For each child, find that child in the prompt** and read *its* children. That's three separate lookups. (Three more scans.)
3. **Take the union**, **deduplicate**, **filter out anything already visited at depth 0 or 1**. (One aggregation.)

For depth $k$ the number of independent lookups is at least $k$ — and *each one depends on the answer to the previous*. The structural argument for why BFS cannot be solved in a single transformer forward pass is **exactly** that property.

## The Frontier-Growth Napkin Math

How many lookups does a depth-$k$ BFS *actually* require? It depends on the graph's branching factor $b$ and on the amount of pruning the model performs. The worst case is exponential.

```pyplot {id="frontier-growth" caption="BFS frontier size as a function of depth for varying branching factor b. At b=2 the depth-4 frontier is ~16 nodes; at b=4 it is ~256. Each node in the frontier is one independent lookup the model has to perform. GraphWalks instances are configured so the total work fits in the context window, but the *number of dependent hops* grows linearly with depth k."}
depths = np.arange(0, 7)
branching = [1.5, 2, 3, 4]
colors = ['#FFD700', '#FF007F', '#00A8A8', '#FF8C00']

fig, ax = plt.subplots(figsize=(8.5, 4.4))
for b, c in zip(branching, colors):
    frontier_size = b ** depths
    ax.plot(depths, frontier_size, marker='o', color=c, linewidth=2,
            markeredgecolor='#1A1A1A',
            label=f'b = {b}')
ax.set_xlabel('BFS depth k')
ax.set_ylabel('frontier size (= unique lookups required at depth k)')
ax.set_yscale('log')
ax.set_title('BFS frontier grows exponentially in depth; deep BFS exhausts lookup budget fast')
ax.legend(loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, which='both', alpha=0.15)

print("BFS at depth k with branching factor b = 3:")
for k in range(6):
    print(f"  depth {k}: ~{3**k:>5,d} nodes in the frontier")
print()
print("GraphWalks BFS-2 in a graph of 1000 nodes at b≈3 = ~9 lookups, all dependent.")
print("GraphWalks BFS-4 = ~81 lookups, with depth-4 dependency chain.")
print("Forward-pass attention with 80 layers can chain at most ~80 dependent hops.")
print("Real benchmarks pick (b, k) to make this nontrivial but achievable.")
```

GraphWalks's depth parameters are chosen carefully. The published depths are usually $k \in \{1, 2, 3, 4\}$, and the graph branching factor is around 2 or 3. So the *total work* per query is on the order of $b^k$ — manageable for the in-context lookup count, but the *dependency chain depth* is $k$, which scales linearly. **The hard part is the depth, not the width.**

This is the architectural reason GraphWalks BFS-1 is essentially trivial for frontier models — it's a single-step retrieval — while GraphWalks BFS-3 and BFS-4 separate them cleanly even at 32K context. The depth is the discriminator.

## Why The Hex Hashes Matter

A subtle design choice: GraphWalks node ids are random hex hashes like `a3f9c1e0…`, not strings like `Alice`, `Bob`, `Charlie`. Why?

Because **hex hashes have no semantic prior in the model's weights.** The model cannot reach into its pretraining knowledge and guess that `a3f9c1e0` is more likely to point to `5b8d2c7e` than to `7e4a1f3c`. The only information the model can use is the adjacency list *in the prompt*.

This is the same defensive design choice we saw in [Needle in a Haystack](../03-niah-mechanics/) and [Leakage and Drift](../15-contamination-drift/): use synthetic, semantically-neutral content so that *parametric* knowledge cannot short-circuit the test. A benchmark with semantically-meaningful node names (`Alice → Bob`, `Bob → Charlie`) would be solvable by reading half the adjacency list and guessing the rest from social-network priors the model already knows. The hex hashes force the model to *actually* attend to the in-prompt structure.

In the {{< wiki "long-context-concepts" >}}LSQ framework{{< /wiki >}} from [Vodrahalli's Chisel](../09-latent-structure/), this design choice is "make the latent structure *only* extractable from the prompt." Hex hashes are how GraphWalks operationalises that.

## How Models Solve GraphWalks: Chain-of-Thought As Hop Multiplier

Now the practical question: how does a real frontier model actually *answer* a GraphWalks BFS-4 query?

The short answer, post-2024, is **chain-of-thought**. The model generates intermediate tokens that explicitly enumerate the BFS frontiers:

```
Reasoning: starting from a3f9c1e0...
  depth 1: a3f9 → 5b8d, 7e4a, d8b6  (three children)
  depth 2: from 5b8d → 9c4f, e2d4
            from 7e4a → 1f8c
            from d8b6 → 1f8c, b3c2
            unique: {9c4f, e2d4, 1f8c, b3c2}
  depth 3: from 9c4f → ...
            ...
Answer: {<all nodes at depth k>}
```

This is the **hop-multiplier** trick we introduced in [Scan vs Think](../04-retrieval-vs-reasoning/). Each CoT line in the model's output is *itself* a forward pass, and each forward pass adds $L$ more available attention hops. By writing out the frontier explicitly at each step, the model effectively *externalises* its state and converts BFS-depth into BFS-width-times-CoT-length. A depth-4 BFS with branching factor 3 becomes maybe 30 CoT lines of explicit enumeration, each one a forward pass within a forward pass.

The cost is in tokens (and therefore latency and dollars). Anthropic's published numbers suggest Claude Opus 4.6 spends typically 8-15K tokens of CoT on a depth-4 GraphWalks BFS query at 128K input context. That's *a lot* of inference compute per query. **The reasoning-model paradigm is what made GraphWalks tractable**; it would have been nearly impossible for a direct-answer 2024 model.

## What GraphWalks Does *Not* Measure

A fair primer should also say what BFS-style benchmarks are *not* sensitive to. Three honest limits:

1. **Real reasoning is messier than BFS.** Real code refactoring, legal cross-referencing, multi-document research — the tasks GraphWalks was designed to *proxy* — involve graph traversals that are *probabilistic*, *partially-observed*, and *interleaved with content analysis*. GraphWalks abstracts all of this into a deterministic graph problem. The proxy is useful but not the whole story.

2. **GraphWalks rewards models that have learned BFS as an explicit *procedure*.** A model that has been trained on code containing BFS implementations, or has been RL-tuned on GraphWalks-like training data, will outperform a model with equal underlying capability but no procedural BFS recipe. This is a known gap. The mitigation is to vary the *encoding* of the graph (sometimes ASCII adjacency lists, sometimes embedded narratives, sometimes tabular), as more recent variants do.

3. **The graph branching factor and depth are *parameters*, not measurements.** A model that scores 88% on BFS-2 may score 30% on BFS-4. Reporting "GraphWalks: 88%" without specifying the depth is meaningless. This is again the *length-conditional, depth-conditional* reporting discipline emerging in 2026.

In the same way the U-curve from [The U-Curve](../07-lost-in-the-middle/) means a single NIAH number is suspicious, a single GraphWalks number is also suspicious. The benchmark is informative when reported with its parameter grid; less so when collapsed to a headline.

## The Deeper Pedagogical Point

If we squint, the entire arc from NIAH to MRCR to GraphWalks is a story about **what kind of computation we are asking a model to perform on its context window**.

- **NIAH**: lookup. $O(1)$ in dependency depth.
- **MRCR**: lookup + counter. $O(1)$ in dependency depth (the counter is updated by a linear scan).
- **GraphWalks BFS-$k$**: $k$ dependent lookups. $O(k)$ in dependency depth.
- **OOLONG aggregation**: $O(1)$ in dependency depth, but $O(n)$ in working-state size.
- **Real agentic workflows**: $O(t)$ in both, for $t$ tool calls over hours.

Each step on this gradient is a *qualitatively* harder computation for a transformer to perform in-context. The 2024 benchmark wave was the field discovering that *most* of NIAH's "long-context understanding" claim was on the $O(1)$-depth side. The 2025 reframe was a deliberate move to $O(k)$-depth tasks. The 2026 question is *what shape of computation* models will face in production — and how to measure the relevant axes of it.

Of which BFS is just the simplest, cleanest, most teachable example.

## What To Remember

1. **BFS is the textbook algorithm for "find all nodes at depth $k$ from a starting node"** in a graph. It has been a CS-101 standard since Moore's 1959 paper. Its critical structural property: the depth-$k$ frontier *cannot be computed* before depth-$(k-1)$.
2. **GraphWalks builds its benchmark around BFS** because depth-$k$ BFS is the cleanest possible test for *in-context, non-parallelisable reasoning depth*. Each level depends on the previous; no clever single-pass scan substitutes.
3. **Hex-hash node ids have no semantic prior** in the model's weights. The only information the model can use to perform the BFS is the adjacency list *in the prompt*. This is a deliberate defence against parametric short-circuiting.
4. **Modern frontier models solve GraphWalks via chain-of-thought**, externalising each BFS frontier as explicit reasoning tokens. This converts the architectural depth requirement into a token-budget requirement. Reasoning models spend typically 8–15K CoT tokens on deep GraphWalks queries.
5. **BFS depth $k$ requires $\geq k$ dependent attention hops** in a single forward pass, or $\geq k$ chain-of-thought iterations otherwise. This is the precise architectural reason GraphWalks separates models that look identical on NIAH and MRCR.
6. **Reporting discipline**: a GraphWalks score without specifying $(k, \text{context length})$ is meaningless. The benchmark is *deliberately* depth-parametrised; the parameter is the signal.

**Continue to** → [The 4× Jump](../11-context-rot-fix/) — the Anthropic story from Sonnet 4.5 in September 2025 (MRCR v2 8-needle 1M at 18.5%) to Opus 4.6 in February 2026 (76% on the same eval). What "fixing context rot" actually involved, and why Opus 4.7 immediately regressed on the longest variants.
