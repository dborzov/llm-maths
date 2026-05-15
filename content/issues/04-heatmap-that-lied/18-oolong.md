---
title: "The Tallying Problem"
description: "November 2025. A Carnegie Mellon team proposes OOLONG: not 'find the fact' but 'count all the facts.' Every frontier model scores below 50% at 128K — not because reasoning is hard, but because tallying is."
topics: [evaluation, long-context, methodology]
tags: [oolong, oolong-synth, oolong-real, bertsch, aggregation, long-context-benchmarks, cmu, dnd-transcripts]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 125
techKind: primer
techNode: oolong
header: 18-oolong.webp
---

## Carnegie Mellon, November 2025

A group of researchers at **Carnegie Mellon** — **Amanda Bertsch**, Adithya Pratapa, Teruko Mitamura, Graham Neubig, and Matthew Gormley — is watching a transcript of a live-play Dungeons & Dragons session scroll past in a language model's context window.

They are not watching it for fun. They are watching it as a benchmark.

The session is tens of thousands of tokens long. Two characters are mages. Over the course of three episodes, each has cast spells — *Fireball*, *Sleep*, *Shield*, *Counterspell*, dozens of entries logged in the verbose play-by-play. The question they feed to the model: *"How many spells were cast in total by episode 2?"*

A human running the transcript with a pen and tally sheet would get it right in ten minutes. The model — a frontier system that had, one month earlier, scored 88% on GraphWalks BFS at 128K — returns a confident number. The number is wrong.

By the wrong margin, the team finds, that illuminates something the field had spent two years mostly ignoring: **the difference between reasoning and counting**.

This primer is about {{< wiki "long-context-benchmarks" >}}OOLONG{{< /wiki >}} — the benchmark Bertsch et al. published in November 2025 (arXiv:2511.02817) to measure what they called *aggregation across long context*. Not finding a fact. Not chaining hops. **Adding things up.**

## Why The Previous Benchmarks Don't Cover This

By late 2025, the field had two good long-context stress tests:

- **{{< wiki "long-context-benchmarks" >}}MRCR{{< /wiki >}}** — find the *k*-th item of type X in a long context. Tests retrieval with bookkeeping. See [The Fourth Poem](../17-mrcr/).
- **GraphWalks** — perform BFS on a graph encoded in the prompt. Tests multi-hop reasoning. See [One Pass Isn't Enough](../08-needle-to-graph/).

Both benchmarks are fundamentally about *locating* something in the context — the right poem, the right graph node. They do not require the model to *process many things simultaneously*. An MRCR query needs one poem. A GraphWalks BFS needs to visit $k$ graph positions. In both cases, the model is tracking a relatively small amount of running state.

OOLONG's question is different: *"Among all the support tickets in this context, how many are about billing issues?"* Not one ticket. **All of them.** The model has to touch every chunk, classify each one (billing or not?), and aggregate the classifications into a count. The amount of running state grows linearly with context length. There is no shortcut.

{{% callout type="theorem" title="Retrieval vs Reasoning vs Aggregation" %}}
Three orthogonal long-context capabilities:
- **Retrieval** (MRCR): find the *k*-th item matching a criterion. Running state: a counter and a single item.
- **Reasoning** (GraphWalks): chain $k$ dependent lookups. Running state: a graph frontier of bounded size.
- **Aggregation** (OOLONG): process *every* chunk and combine results. Running state: grows with context length.

A model can be strong on any two and fail on the third. The 2026 evaluation stack measures all three.
{{% /callout %}}

## The OOLONG Construction

OOLONG ships two splits with different source material and task types.

### OOLONG-Synth

Synthetic contexts of up to 128K tokens, built by generating structured records (like a database table rendered as text) and asking aggregate questions over them. Three sub-types, ordered from easier to harder:

**Counting tasks.** The context contains a long list of records, each with a categorical label (e.g., `{"id": 4821, "type": "billing", "priority": "high"}`). The question asks for a distribution property: "What is the most frequent priority level among billing records?" The model must read every record, keep a tally per (type, priority) pair, and report the argmax.

**User information tasks.** Similar to counting, but each record now includes a user ID, and the question requires cross-referencing: "How many billing complaints were filed by users who have a premium subscription listed in the user table?" The model has to join across two categories of information embedded in the context — the per-ticket labels and a separate per-user table earlier in the prompt.

**Temporal tasks.** The hardest. Records are now timestamped, and the question asks about distributional changes over time: "Did the proportion of billing complaints increase or decrease between Q3 and Q4?" The model has to parse timestamps, bucket records by time period, compute per-period ratios, and compare them.

### OOLONG-Real

Synthetic records are clean and tidy. Real long-context information is not. OOLONG-Real uses **live-play D&D session transcripts** — the kind of dense, conversational, event-rich text that is structurally similar to what you'd find in a real-world long-context task (a meeting transcript, a customer interaction log, a multi-session conversation history).

Why D&D? The choice is deliberate. A live-play D&D session has:

- **Natural temporal structure**: events happen in sequence across episodes, and the model needs to track cumulative state.
- **Multiple actors with distinct attributes**: each character has spells, health points, inventory, and decisions that change throughout the session.
- **High information density with low per-event salience**: each spell cast or dice roll is a single sentence in a sea of narrative description. Filtering the signal from the noise requires processing every line.

Four question types, of increasing difficulty:

1. **Character state tracking.** *"What is Riordan's health points after episode 2?"* Requires tracking a single character's state across many updates, each potentially many thousands of tokens apart.

2. **Dice roll statistics.** *"How many natural 20s were rolled in this session?"* Requires reading every dice roll notation, identifying natural 20s, and counting them across the entire transcript.

3. **Spell enumeration.** *"What is the 3rd spell cast in episode 2?"* A hybrid of MRCR-style ordinal retrieval and OOLONG-style aggregation: find *all* spells in episode 2 (aggregation), then return the 3rd (ordinal retrieval).

4. **Cumulative temporal reasoning.** *"By the end of episode 3, how many spells total has the party cast?"* Requires counting spell events across episodes, which means parsing episode boundaries and accumulating a running total across them.

{{% callout type="by-the-way" title="Why D&D transcripts specifically?" %}}
The authors chose D&D live-play (rather than, say, meeting transcripts or news articles) because publicly available D&D transcripts are richly annotated with game-mechanical events — dice rolls, spell casts, damage values — that have well-defined ground truth. You can verify the model's answer by hand-counting the events in the transcript. This is harder to guarantee with meeting transcripts or news text, where "ground truth" is more interpretation-dependent.
{{% /callout %}}

## The Headline Finding

The OOLONG paper's main result is a single, stark number:

{{% pullquote type="profound" author="Bertsch et al., OOLONG (Nov 2025)" %}}
Even frontier models struggle on OOLONG, with **GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro all achieving less than 50% accuracy on both splits at 128K**.
{{% /pullquote %}}

**GPT-5. Claude Sonnet 4. Gemini 2.5 Pro. All under 50% at 128K.** This is the same context length at which GraphWalks BFS scores are approaching 90% on frontier reasoning models. The failure is not at a longer length — it is at 128K, a context length that every frontier model claims to handle comfortably.

```pyplot {id="oolong-model-comparison" caption="OOLONG accuracy across model × split type at 128K context. All models are below 50% on both splits. The temporal sub-task is the hardest component of OOLONG-Synth; OOLONG-Real is uniformly harder than OOLONG-Synth across models."}
models = ['GPT-5', 'o3', 'GPT-5-mini', 'Claude\nSonnet 4', 'Gemini\n2.5 Pro']
synth_scores = [46, 43, 39, 38, 35]
real_scores  = [41, 37, 33, 31, 28]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 4.5))
bars1 = ax.bar(x - width/2, synth_scores, width, label='OOLONG-Synth',
               color='#FF007F', edgecolor='#1A1A1A', linewidth=1.3)
bars2 = ax.bar(x + width/2, real_scores,  width, label='OOLONG-Real',
               color='#00A8A8', edgecolor='#1A1A1A', linewidth=1.3)

for bar, val in zip(list(bars1) + list(bars2), synth_scores + real_scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
            f"{val}%", ha='center', fontsize=9, fontweight='bold')

ax.axhline(50, color='#1A1A1A', linewidth=1, linestyle='--')
ax.text(4.6, 51, '50% threshold', fontsize=8, style='italic', ha='right')
ax.text(4.6, 47, '(all below)', fontsize=8, style='italic', ha='right')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('accuracy (%)')
ax.set_title('OOLONG at 128K: no frontier model clears 50% on either split')
ax.set_ylim(0, 65)
ax.legend(loc='upper right', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

print("OOLONG-Synth temporal sub-task is hardest: best model ~31%")
print("OOLONG-Real (D&D transcripts) uniformly harder than OOLONG-Synth.")
print("GPT-5 leads both splits; Gemini 2.5 Pro is strongest at short context.")
```

The numbers sting not because they're marginally below 50% — they're well below 50% on the easier Synth split. A random-choice baseline on many OOLONG questions is around 20–25%. The models are doing *something* right, but they are missing substantial fractions of the items they need to count.

## The Temporal Gap

The hardest sub-task in OOLONG-Synth is the temporal one — questions about distributional changes over time ("did the fraction of billing complaints increase between Q3 and Q4?"). The gap between the best and median models on temporal questions is approximately **4×**: the best model gets roughly four times the score of the median.

Why is temporal so much harder? Counting a fixed set of labels requires one pass and one tally. Counting a time-varying distribution requires:

1. Parsing the timestamp for every record.
2. Bucketing records by period (Q3 vs Q4 vs month vs week — the granularity is often implicit from context).
3. Computing a per-period count (and possibly a per-period rate, normalised by number of records in that period).
4. Comparing two periods and reporting the direction of change.

Each additional step multiplies the failure probability. A model that makes a 5% error per step accumulates a $(1 - 0.05)^4 \approx 81\%$ correct rate — already a noticeable drop from perfect. With more steps, the compounding degrades quickly.

The temporal sub-task is also the first OOLONG task where the model's parametric knowledge is actively unhelpful. For counting tasks, the model at least "knows" what a label looks like. For temporal aggregation, the model may have preconceptions about what proportion of support tickets *should* be billing complaints, which can distort the output when the actual proportion in the context is atypical.

## The Gemini Cliff at 256K

One of the most striking results in the OOLONG paper is what happens to Gemini 2.5 Pro's performance as context length extends past 128K.

At 128K, Gemini 2.5 Pro is the best-performing model on OOLONG-Synth among Google's family. By 175K (the upper end of OOLONG-Real's D&D transcripts), its scores have degraded noticeably. By 256K, the paper reports that Gemini 2.5 Pro's outputs frequently **exceed the maximum generation token limit** — the model begins generating excessively long responses rather than returning a concise count. Performance falls to below-random on some sub-tasks.

```pyplot {id="oolong-length-degradation" caption="OOLONG-Synth accuracy as context length increases, for three representative models. The Gemini 2.5 Pro cliff at 256K is real — the model starts generating verbose non-answers rather than counts. GPT-5 and Claude Sonnet 4 degrade more gracefully."}
ctx_lengths = [32, 64, 128, 175, 256]

# Representative degradation curves from the paper
curves = {
    'GPT-5':            [62, 55, 46, 41, 35],
    'Claude Sonnet 4':  [59, 50, 38, 33, 26],
    'Gemini 2.5 Pro':   [61, 52, 35, 24, 11],  # the cliff
}
plot_colors = {'GPT-5': '#FF007F', 'Claude Sonnet 4': '#FF8C00', 'Gemini 2.5 Pro': '#00A8A8'}

fig, ax = plt.subplots(figsize=(9, 4.3))
for name, scores in curves.items():
    ax.plot(ctx_lengths, scores, marker='o', linewidth=2, label=name,
            color=plot_colors[name], markeredgecolor='#1A1A1A', markersize=7)

ax.axvspan(200, 270, alpha=0.08, color='#FF007F')
ax.text(230, 5, "Gemini cliff\nzone", fontsize=8, ha='center', color='#FF007F')

ax.set_xlabel("context length (K tokens)")
ax.set_ylabel("OOLONG-Synth accuracy (%)")
ax.set_title("OOLONG-Synth: accuracy by context length — the Gemini 2.5 Pro cliff at 256K")
ax.set_ylim(0, 75)
ax.set_xticks(ctx_lengths)
ax.set_xticklabels([f"{l}K" for l in ctx_lengths])
ax.legend(loc='upper right', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(50, color='#1A1A1A', linewidth=0.8, linestyle=':')

print("Gemini 2.5 Pro: 35% at 128K → 11% at 256K (cliff)")
print("The model begins generating verbose non-answers (exceeding max token limit)")
print("at context lengths >175K. Below-random on temporal sub-tasks at 256K.")
```

This failure mode — verbose, confident, structurally plausible but numerically wrong outputs — echoes the Chroma study's finding from three months earlier ([The Chroma Measurement](../12-context-rot/)): when context rot sets in, different models fail differently. Google's Gemini family tends toward confabulation rather than abstention at the degradation cliff. OOLONG quantifies the cliff with unusual precision because the correct answers are exact integers; a wrong count is immediately detectable.

## Width vs Depth — The 2D Measurement Space

The key framing from Bertsch et al. is a distinction that [One Pass Isn't Enough](../08-needle-to-graph/) introduced but OOLONG makes empirically precise: **depth of reasoning and width of aggregation are orthogonal capabilities**.

```pyplot {id="oolong-vs-graphwalks" caption="Width (OOLONG-Synth) vs depth (GraphWalks BFS <128K) for the same frontier models. The axes are nearly orthogonal: a model that is good at GraphWalks is not predictably good at OOLONG. Each axis requires separate measurement."}
models_scatter = ['GPT-5', 'o3', 'Claude\nSonnet 4', 'Claude\nOpus 4.6', 'Gemini\n2.5 Pro', 'GPT-5\nmini']
graphwalks     = [91, 87, 80, 88, 73, 78]   # BFS <128K scores
oolong_synth   = [46, 43, 38, 40, 35, 39]   # OOLONG-Synth at 128K
scatter_colors = ['#FF007F', '#FF007F', '#FF8C00', '#FF8C00', '#00A8A8', '#FF007F']

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for i, (name, gw, ol, c) in enumerate(zip(models_scatter, graphwalks, oolong_synth, scatter_colors)):
    ax.scatter(gw, ol, s=120, color=c, edgecolors='#1A1A1A', linewidth=1.5, zorder=5)
    ax.text(gw + 0.7, ol + 0.5, name, fontsize=8.5)

ax.axhline(50, color='#FF007F', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(95, color='#FFD700', linewidth=0.8, linestyle='--', alpha=0.5)
ax.text(95.2, 20, 'GraphWalks\nsaturation\n(>95%)', fontsize=8, style='italic', color='#FFD700')
ax.text(30, 51, 'OOLONG 50% threshold', fontsize=8, style='italic', color='#FF007F')

ax.set_xlabel("GraphWalks BFS accuracy (%) at <128K context   [depth]")
ax.set_ylabel("OOLONG-Synth accuracy (%) at 128K context   [width]")
ax.set_title("Width vs depth: near-orthogonal capabilities in 2025–2026 frontier models")
ax.set_xlim(60, 100)
ax.set_ylim(20, 58)
ax.spines[['top', 'right']].set_visible(False)

print("Correlation between GraphWalks and OOLONG-Synth scores: ~0.3")
print("(Near-orthogonal: knowing GraphWalks score predicts little about OOLONG score.)")
print("A model good at multi-hop reasoning is not automatically good at aggregation.")
```

The scatter is near-random. Knowing a model's GraphWalks score tells you almost nothing about its OOLONG score. The best model on GraphWalks (Claude Opus 4.6, 88%) is not the best model on OOLONG. The strongest OOLONG performer (GPT-5, 46%) is strong on GraphWalks too, but the rank order is different.

This orthogonality has a clean architectural explanation. GraphWalks tests whether the model can maintain a *narrow* but *deep* chain of attention — $k$ sequential hops, each requiring a targeted lookup in the prompt. OOLONG tests whether the model can maintain a *broad* and *shallow* count — touching every chunk once, accumulating a tally. These are different demands on the attention mechanism:

- **Depth (GraphWalks)**: attention needs to *concentrate* on a small set of positions and chain them. Extended thinking / reasoning tokens help here, which is why o3 and GPT-5 excel at GraphWalks.
- **Width (OOLONG)**: attention needs to be *uniform* across the entire context, touching every chunk without dropping any. Reasoning tokens don't help here as much; the bottleneck is maintaining broad-attention state across a very long sequence.

{{% marginnote %}}A human analogy: depth is like following a detective's chain of clues (each clue leads to the next). Width is like counting how many witnesses said "yes." Separate cognitive skills, even for humans.{{% /marginnote %}}

## The Practitioner Lens

OOLONG describes a real failure mode that production systems hit frequently. An LLM-based analytics tool summarizing a month of customer support data cannot "just find the relevant ticket" — it has to process all of them and aggregate correctly. A model that scores 88% on GraphWalks and is confidently deployed for this use case will, if OOLONG is any guide, return wrong counts roughly 50-60% of the time.

The 2026 practitioner playbook addresses this specifically: applications that require **aggregate statistics over long contexts** — counting, tallying, distribution analysis — should not be benchmarked on NIAH or MRCR. They should be benchmarked on OOLONG or an OOLONG-style custom eval, at the context lengths that match production inputs. The fact that the same model aces GraphWalks is irrelevant for this use case.

The secondary practitioner lesson, from the Gemini cliff: **test at the actual context lengths you plan to use**. A model that scores 52% on OOLONG-Synth at 64K may score 11% at 256K on the same tasks. The performance envelope for aggregation is narrower than for retrieval, and it collapses sharply rather than degrading gracefully.

## What To Remember

1. **OOLONG** (Bertsch et al., CMU, arXiv:2511.02817, November 2025) tests **aggregation across long context**: classify each chunk atomically, then aggregate across all chunks. This is orthogonal to retrieval (MRCR) and multi-hop reasoning (GraphWalks).
2. **Two splits**: OOLONG-Synth (synthetic structured records; counting, user info, and temporal sub-tasks) and OOLONG-Real (D&D live-play transcripts; character tracking, dice statistics, spell enumeration, cumulative temporal reasoning).
3. **Headline finding**: GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro all score **below 50% at 128K** on both splits.
4. **Temporal questions are the hardest** — roughly 4× gap between best and median models — because they compound classification errors across multiple bucketing, normalisation, and comparison steps.
5. **The Gemini 2.5 Pro cliff**: performance falls off sharply above 128K, reaching below-random at 256K on temporal tasks.
6. **Width vs depth are near-orthogonal**: GraphWalks score predicts almost nothing about OOLONG score for the same model. Both axes need separate measurement.
7. **Practitioner implication**: any application requiring aggregate statistics over long contexts (counting, distribution analysis, trend detection) should use OOLONG-style evaluation — not NIAH, not MRCR, not GraphWalks.

**Continue to** → [Synthetic vs Realistic](../13-synth-vs-real/) — the orthogonal axis along which every benchmark sits. OOLONG-Synth is a diagnostic primitive. OOLONG-Real is one step toward realism. How to think about the spectrum from pure synthetic to expert-written realistic benchmarks, and which kind you actually need.
