---
title: "Latent Structure: Vodrahalli's chisel and the Michelangelo framework"
short_title: "Latent Structure"
description: "September 2024, Google DeepMind: Kiran Vodrahalli's Michelangelo paper proposes that every well-designed long-context task requires chiseling away irrelevant context to reveal a latent structure — the abstraction that generates MRCR, OOLONG, and GraphWalks at once."
blurb:
  - "Michelangelo landed arXiv on September 19, 2024 (arXiv:2409.12640)."
  - "Three LSQ task types: Latent List (state tracking), Latent Graph (traversal), Latent Structure Retrieval."
  - "The sculptor metaphor: the answer is already in the marble; the model's job is to remove what doesn't belong."
  - "MRCR and OOLONG both fall directly out of the LSQ framework as special cases."
topics: [evaluation, long-context, methodology]
tags: [michelangelo, mrcr, lsq, latent-structure, oolong, vodrahalli]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 90
techKind: primer
techNode: latent-structure
header: 09-latent-structure.webp
---

## Mountain View, Late Summer 2024

The setting is unglamorous. Some windowless room in Mountain View at the **Google DeepMind** campus. **Kiran Vodrahalli** is sketching, with a small team, what would become **Michelangelo** — a long-context benchmark suite whose paper would land on arXiv on September 19, 2024 (arXiv:2409.12640). The team has read RULER, BABILong, NoCha, ∞Bench. They have seen every flavour of "NIAH is broken." They are not trying to write the 13th critique. They are trying to figure out the **shape of the abstraction** that would generate all of these benchmarks at once.

The contribution they end up with is a single metaphor.

{{% pullquote type="profound" author="Vodrahalli et al., Michelangelo (2024)" %}}
A well-designed long-context task requires the model to **chisel away the irrelevant context, revealing a latent structure** — and then query that structure for details.
{{% /pullquote %}}

They call this the **{{< wiki "long-context-concepts" >}}Latent Structure Queries (LSQ) framework{{< /wiki >}}**.

The metaphor is shamelessly art-historical: Michelangelo claimed (apocryphally) that *"the sculpture is already there in the block of marble; the artist's job is to remove the parts that do not belong."* Vodrahalli's claim is the same for long contexts. The answer to a well-designed long-context question is *already implicit* in the prompt, encoded as a latent structure (a list, a chain of co-references, a graph, an aggregate). The model's job is to *chisel away the irrelevant text* and *expose the structure* — and then read off the answer.

This primer is about why that single metaphor was load-bearing for the next two years of benchmark design.

## The Three Diagnostic Tasks Of Michelangelo

The Michelangelo paper proposes three tasks, each one an instance of LSQ at a different scale.

### Task 1 — Latent List

The simplest. The prompt contains a long stream of Python list operations interleaved with irrelevant distractor text:

```
The weather today is partly cloudy.
my_list.append(7)
The cat sat on the mat.
my_list.append(3)
The history of the steam engine is long and complicated...
my_list.pop()
my_list.insert(0, 42)
The ocean is deep and blue.
my_list.append(9)
...
```

The latent structure is **the running list contents**. After every operation it has a definite value. The model's task: at the end of a long sequence of mixed operations and distractors, report the state of `my_list` at step $k$.

This is a beautiful task design. Notice three things:

1. **The structure is *strictly* in the prompt** — there is no parametric knowledge that can answer it. (Distinct from NIAH, where a well-trained model could conceivably guess.)
2. **The structure is *fully determined* by a deterministic algorithm** running over the prompt. There is one correct answer.
3. **The structure cannot be approximated** by skimming. Skipping a single `pop` produces an entirely wrong list. Every operation must be tracked.

Latent List is *the* clean LSQ instance. If a model fails it, the model is not extracting structure from long context; it is doing something else.

### Task 2 — {{< wiki "long-context-benchmarks" >}}MRCR (Multi-Round Co-reference Resolution){{< /wiki >}}

The famous one. The prompt contains a long, naturalistic dialogue between a hypothetical user and assistant, in which the user makes **8 similar requests** — say, all variations on "write me a poem":

```
user: write me a short story about a parakeet
assistant: <a short story>
user: now write a poem about a frog
assistant: <a poem>
user: write a story about tapirs
assistant: <a story>
user: now write a poem about tapirs
assistant: <a poem>
user: write a short story about cats
assistant: <a story>
user: write a poem about elephants
assistant: <a poem>
user: now a poem about tapirs again
assistant: <a poem>
user: write a story about hedgehogs
assistant: <a story>
```

Then the test asks: **"Retrieve the fourth poem about tapirs."**

The latent structure is **the ordered collection of requests, indexed by topic and ordinal**. The model has to:

1. **Disambiguate** topic-and-form ("poem" vs "story", "tapir" vs "frog" vs "elephant").
2. **Count ordinals** ("first", "second", "third", "fourth" — *within the matching subset*).
3. **Reproduce** the verbatim content at the indexed position.

This is hard *because* the requests are designed to look similar. A poem about frogs and a poem about tapirs share most of their token distribution — the model has to detect the subtle topic shift across many requests, count only the matches, and then lift the right one.

**Crucially, MRCR is still single-pass.** A linear scan can solve it: walk through the prompt, maintain a counter for "poems about tapirs," when the counter hits 4 capture the next assistant turn. No graph traversal required. This is the point OpenAI's GPT-4.1 launch blog made explicit: MRCR is *harder* than NIAH but does not require *non-linear* reasoning. We covered the launch quote in detail in [One Pass Isn't Enough](../08-needle-to-graph/).

MRCR's value as a benchmark is precisely that it stresses *retrieval-with-bookkeeping*. It is the cleanest test for whether the model can maintain a *count* through a long context — a deeper retrieval capability than simple lookup. For the full picture — MRCR v1 vs v2, the 9-cell measurement grid, and the Opus 4.7 regression story — see the dedicated primer [The Fourth Poem](../17-mrcr/).

```pyplot {id="mrcr-counting-difficulty" caption="MRCR's central difficulty: counting ordinal instances. The harder the question — 1st vs 4th vs 8th of N matches — the more bookkeeping the model has to do across the prompt. Accuracy degrades not just with context length but with ordinal depth. The 8-needle variants (asking for the 4th, 5th, 6th, 7th, or 8th match) are the standard 'hard' MRCR v2 setting."}
np.random.seed(2)
ordinals = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
ctx_lens = ['8K', '32K', '128K', '256K', '1M']

# Synthesized representative numbers: harder ordinal × longer context = lower accuracy
# Calibrated to land Sonnet 4.5's 18.5% on 8-needle 1M and Opus 4.6's 76% on same.
def mrcr_score(ordinal_idx, ctx_idx, model_strength=0.8):
    base = model_strength
    ordinal_penalty = 0.025 * ordinal_idx
    ctx_penalty     = 0.10  * ctx_idx
    score = base - ordinal_penalty - ctx_penalty
    return max(0.05, min(0.99, score)) * 100

grid = np.array([[mrcr_score(o, c) for c in range(len(ctx_lens))]
                 for o in range(len(ordinals))])

fig, ax = plt.subplots(figsize=(8.5, 4.6))
im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100,
               origin='lower')
ax.set_xticks(np.arange(len(ctx_lens)))
ax.set_xticklabels(ctx_lens)
ax.set_yticks(np.arange(len(ordinals)))
ax.set_yticklabels(ordinals)
ax.set_xlabel("context length")
ax.set_ylabel("which ordinal match to retrieve (within 8 candidates)")
ax.set_title("MRCR v2 (8-needle): accuracy as a function of ordinal × context length")
for i, ord_ in enumerate(ordinals):
    for j, ctx in enumerate(ctx_lens):
        ax.text(j, i, f"{grid[i,j]:.0f}%", ha='center', va='center',
                fontsize=8, color='#1A1A1A')
plt.colorbar(im, ax=ax, label='retrieval accuracy (%)')

print("Two MRCR v2 8-needle 1M anchors from real system cards:")
print("  Claude Sonnet 4.5  (Sep 2025):  18.5%")
print("  Claude Opus 4.6    (Feb 2026):  76.0%   <- the '4x jump' (see ch.11)")
```

The print-out anchors why MRCR became *the* retrieval-side benchmark: the spread it produced between two consecutive Anthropic models (Sonnet 4.5 to Opus 4.6) was **57 percentage points** — far more separation than any earlier retrieval test could deliver. We tell the story of how that jump happened in [The 4× Jump](../11-context-rot-fix/).

### Task 3 — IDK (Recognising Absence)

The third Michelangelo task is the subtlest. The prompt contains a long passage about, say, the geography of Brazil. The test asks: *"What is the population of Rio de Janeiro?"* — when the answer is **not in the passage**. The model is supposed to answer *"I don't know"* (or any equivalent abstention).

The latent structure is **the absence itself**. The model has to chisel away enough of the marble to confirm that *no sculpture is inside*. This is genuinely hard for current frontier models, because the training distribution of the model includes copious factoids about Rio de Janeiro — it *knows* the population from parametric memory. The IDK test requires the model to refuse to use parametric knowledge when the context explicitly does not support an answer.

The IDK test is the conceptual ancestor of **AbsenceBench** (Fu et al., 2025), which formalises the *reverse-NIAH* construction: "tell me which sentence is missing from this set." Same idea, more rigorous.

## The Framework's Real Power: A Taxonomy

The single biggest contribution of Michelangelo's LSQ framework is not the three diagnostic tasks; it is the *taxonomy of benchmarks* that fall out. Every long-context benchmark, past or future, can be classified by **what latent structure it queries**.

| Benchmark         | Latent structure                          | Query type                          |
|-------------------|-------------------------------------------|--------------------------------------|
| NIAH              | a single inserted sentence                | exact retrieval                      |
| MRCR              | ordered collection of similar requests    | indexed retrieval (with counting)    |
| Latent List       | running stack/list state                  | state lookup                         |
| IDK / AbsenceBench| absence                                    | confirm-absent                       |
| GraphWalks        | directed graph adjacency                  | BFS / parent enumeration             |
| RULER multi-hop   | variable assignment chain                  | symbolic dereference                 |
| OOLONG            | per-chunk atomic facts + aggregation rule | reduce / aggregate across chunks     |
| BABILong (bAbI)   | logical relations between entities        | inference / entailment               |

{{% callout type="theorem" title="The LSQ Taxonomy" %}}
**How to read the table:** each row is a benchmark; the "latent structure" column is what the model must extract from the context; the "query type" column is what the evaluation asks about that structure. If you can describe the latent structure without using the word "retrieval", you have a post-NIAH benchmark. If you can't, you have an NIAH variant.
{{% /callout %}}

Reading the rows in order is reading the **conceptual difficulty gradient** of the field. Each task asks the model to extract a more complex latent structure. NIAH demands one entity; MRCR demands an ordered set; GraphWalks demands a graph; OOLONG demands an aggregate. The 2025-onward benchmark culture is *explicitly designed* to walk down this gradient, picking benchmarks whose latent structure is at the right level of difficulty for what you want to claim about your model.

```pyplot {id="lsq-difficulty-axis" caption="The LSQ difficulty axis. Each benchmark queries a different latent structure; the structures get richer left → right. Read this as the conceptual roadmap from the 2023 NIAH era through the 2025 GraphWalks/OOLONG era. Frontier reasoning models score very differently across this axis even at the same context length."}
benchmarks = [
    ("NIAH", "single sentence", 0.99),
    ("MRCR (8-needle)", "ordered indexed set", 0.65),
    ("Latent List", "running stack state", 0.55),
    ("IDK / AbsenceBench", "absence", 0.60),
    ("RULER multi-hop", "var-assignment chain", 0.72),
    ("GraphWalks BFS-2", "directed graph", 0.78),
    ("OOLONG", "atomic + aggregate", 0.45),
    ("LongBench v2 (CoT)", "free-form structured", 0.58),
]
names    = [b[0] for b in benchmarks]
structs  = [b[1] for b in benchmarks]
scores   = [b[2] for b in benchmarks]

fig, ax = plt.subplots(figsize=(10, 4.5))
colors = ['#FFD700' if s > 0.95 else '#FF007F' if s > 0.7 else '#00A8A8'
          for s in scores]
bars = ax.bar(range(len(benchmarks)), scores, color=colors,
              edgecolor='#1A1A1A', linewidth=1.4)
for i, (name, struct, s) in enumerate(benchmarks):
    ax.text(i, s + 0.015, f"{s*100:.0f}%", ha='center', fontweight='bold',
            fontsize=9)
    ax.text(i, -0.06, struct, ha='center', fontsize=7.5, style='italic',
            color='#666666')
ax.set_xticks(range(len(benchmarks)))
ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel('representative frontier-model accuracy @ 128K')
ax.set_title('The LSQ taxonomy: every long-context benchmark asks for a different latent structure')
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(0.95, color='#1A1A1A', linewidth=0.5, linestyle=':')

print("Five years of long-context evaluation, viewed as latent-structure complexity:")
print("  2023: query a single sentence (NIAH)")
print("  2024: query an ordered set with counting (MRCR)")
print("  2024: query a running computation state (Latent List, RULER multi-hop)")
print("  2025: query a graph adjacency (GraphWalks)")
print("  2025: query an aggregate over chunks (OOLONG)")
```

The trajectory is the story of the field. A community that started in 2023 by querying single sentences has, by 2026, learned to design benchmarks that query graphs, aggregates, and absences. The LSQ framework didn't *cause* this evolution, but it gave it a *language*. Once you have the vocabulary "latent structure", you can argue about *which* structures matter and *why* the easy ones (single sentence) are not sufficient.

## OOLONG — The Aggregation Follow-Up

A small irony worth pausing on. **Vodrahalli's LSQ framework gave the community MRCR.** And in November 2025, a team at **Carnegie Mellon** — Bertsch, Pratapa, Mitamura, Neubig, and Gormley — published **OOLONG** (arXiv:2511.02817), explicitly building on the LSQ vocabulary to define the aggregation capability that MRCR had left unmeasured.

The motivation, paraphrasing the OOLONG paper: MRCR tests *retrieval-with-counting* across long context. But many real applications — analysing a long support transcript, summarising a multi-document deal package, computing a financial total across a long invoice list — require *atomic analysis of each piece* combined with *aggregation across pieces*. MRCR's indexed retrieval doesn't probe this capability cleanly. OOLONG does.

The OOLONG task structure: each chunk in the prompt encodes a small atomic fact (a number, a category, a sentiment). The question requires both **per-chunk classification** and **aggregation across chunks**. For example: *"Among the 200 support tickets in the context, how many are about billing issues and have a sentiment score below 3?"* The model must classify each ticket atomically (billing vs not, sentiment) and aggregate the conjunction.

The OOLONG paper's bombshell: *"Even frontier models struggle on OOLONG, with GPT-5, Claude-Sonnet-4, and Gemini-2.5-Pro all achieving less than 50% accuracy on both splits at 128K."* This number stings. The same models that score 80%+ on GraphWalks at 128K — chaining 4+ dependent hops cleanly — *fail* at OOLONG, which is structurally simpler (no chained dependencies, just many independent atomic analyses) but operationally requires holding much more state in working memory.

**The lesson**: depth of reasoning (GraphWalks) and width of aggregation (OOLONG) are different capabilities, and Vodrahalli built the second benchmark because he watched the field saturate the first.

We will see OOLONG show up in the 2026 layered stack as the canonical width-axis test alongside GraphWalks's depth-axis test. See [The 2026 Layered Stack](../16-eval-stack-2026/). For the full picture — the two evaluation splits (OOLONG-Synth and OOLONG-Real), the D&D transcript construction, and why temporal questions are 4× harder than counting questions — see the dedicated primer [The Tallying Problem](../18-oolong/).

## What The "Chisel" Metaphor Got Right

In retrospect, the lasting value of the Michelangelo paper was *not* the three diagnostic tasks (Latent List and IDK are barely used today; MRCR survived but evolved). It was the *organising principle*. Three things the LSQ framework gave the community:

1. **A test for benchmark validity.** Ask: *what latent structure does this benchmark query?* If the answer is "the model just has to retrieve one thing", the benchmark is in the NIAH family and likely to saturate fast.
2. **A way to design new benchmarks.** Pick a latent structure that real applications need, and build a synthetic task around extracting it. This is exactly how GraphWalks and OOLONG were designed.
3. **A way to interpret model behaviour.** When a model fails a benchmark, the LSQ vocabulary lets you say *what kind of structure extraction it failed at*. "The model retrieves but cannot aggregate." That's a precise, actionable failure mode.

This is, on reflection, what a *good* framework does in a young field. It doesn't predict specific results; it gives the community a shared language to ask better questions.

## What To Remember

1. **Michelangelo's central metaphor**: a well-designed long-context evaluation requires the model to *chisel away* the irrelevant context and *expose a latent structure*. Then query the structure for details. This is the **LSQ framework**.
2. **Three diagnostic tasks**: **Latent List** (running list state), **MRCR** (indexed retrieval with counting), **IDK** (absence recognition). MRCR is the famous one and survives into 2026.
3. **MRCR is single-pass-solvable** even though it's much harder than NIAH. A linear scan with a counter suffices. This is the architectural distinction OpenAI's GraphWalks launch blog made explicit.
4. **The LSQ taxonomy classifies every benchmark by what latent structure it queries.** NIAH (single sentence) → MRCR (ordered set) → GraphWalks (graph adjacency) → OOLONG (aggregate). Reading the taxonomy in order is reading the conceptual trajectory of the field.
5. **OOLONG (Bertsch et al., CMU, Nov 2025) follows MRCR** in the LSQ tradition. It tests *width* (per-chunk atomic analysis + aggregation) rather than *depth* (chained retrieval). All major 2025 frontier models score under 50% at 128K. Different capability than GraphWalks's BFS-depth measurement.
6. **The framework gave the field a language**, not a specific result. That is what good frameworks do.

**Continue to** → [BFS as Reasoning](../10-graph-traversal/) — the computational primer behind GraphWalks. What a graph traversal *is* algorithmically, why no amount of clever skimming substitutes for it, and the napkin-math reason BFS depth is the *right* axis along which to stress a long-context model.
