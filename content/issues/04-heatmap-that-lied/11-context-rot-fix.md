---
title: "The 4× Jump"
description: "Five months. One Anthropic release cycle. MRCR v2 8-needle at 1M context: 18.5% → 76%. The single largest single-version capability jump in long-context benchmark history — and what 'fixing context rot' actually meant."
topics: [evaluation, anthropic, long-context]
tags: [claude-opus-4.6, sonnet-4.5, mrcr-v2, context-rot, anthropic]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 110
techKind: mainline
techNode: context-rot-fix
header: 11-context-rot-fix.webp
---

## A Number In A System Card, And A Public Promise

It is **September 29, 2025**. Anthropic ships **Claude Sonnet 4.5**. The system card runs 184 pages. Buried inside, on page 71, in the long-context section, is a table that the long-context Twitter accounts will spend a week chewing over:

> **{{< wiki "long-context-benchmarks" >}}MRCR v2{{< /wiki >}} (OpenAI), 8-needle, 1M context: 18.5%**

Read that number. **A frontier model from the leading "long-context" lab scoring 18.5% on the hardest variant of the standard retrieval benchmark.** A random-chance baseline would do better than 18.5% only on a multiple-choice version — MRCR v2 is open-ended, so 18.5% means the model genuinely retrieves the right ordinal instance in fewer than 1 of every 5 attempts.

Anthropic does not hide this. The system card's framing is *deliberately humble*. The section explicitly compares Sonnet 4.5 to MRCR v2 results from OpenAI, Google, and DeepSeek at the same length and acknowledges Sonnet 4.5 is mid-pack. The same page also reports that Sonnet 4.5's GraphWalks BFS scores at 128K are competitive with frontier (66.3% on BFS-2) but that at 1M they collapse.

Reading between the lines, the system card is announcing: *we know our 1M window doesn't really work yet; here's the measurement that proves we know; the next release will fix it.*

Five months later, Anthropic does in fact fix it.

## February 2026 — Opus 4.6 Lands

On **February 13, 2026**, Anthropic ships **Claude Opus 4.6**. The launch blog opens with a sentence that subtly inverts the field's vocabulary:

> *"A common complaint about AI models is 'context rot,' where performance degrades as conversations exceed a certain number of tokens. Opus 4.6 performs markedly better than its predecessors. This is a qualitative shift in how much context a model can actually use while maintaining peak performance."*

This is the first time "{{< wiki "long-context-concepts" >}}context rot{{< /wiki >}}" appears in a frontier-lab marketing blog as a *named, acknowledged phenomenon to be fixed*. The phrase had been circulating in user communities since mid-2024, formalised by the Chroma study in July 2025 ([The Chroma Measurement](../12-context-rot/)), and treated as a known limitation by every long-context paper since. Anthropic's move is to *adopt the user-coined term* and then announce that the new model fixes it.

The headline number:

> **MRCR v2 8-needle 1M: 76.0%**

{{% pullquote type="counter-intuitive" %}}
From 18.5% to 76% on the same benchmark, in five months, between two consecutive releases. **+57.5 percentage points.** The single largest single-version leap in long-context benchmark history.
{{% /pullquote %}}

From Sonnet 4.5's **18.5%** to Opus 4.6's **76.0%** — a **57.5-percentage-point jump** on the same benchmark, in five months, between two consecutive Anthropic releases. That ratio — **roughly 4×** — became the marketing slogan of the launch, and gives this chapter its title.

```pyplot {id="anthropic-mrcr-progression" caption="Anthropic's MRCR v2 8-needle scores across model versions. Sonnet 4.5 baseline (Sep 2025) at 18.5%; Opus 4.6 (Feb 2026) at 76%. The largest single-version jump in long-context benchmark history. Opus 4.7 (Apr 2026) regressed to 32.2% — see the asterisk."}
versions = [
    ("Sonnet 3.5\nJun 2024",   None,   None,   '#aaaaaa'),
    ("Sonnet 3.7\nFeb 2025",   None,   None,   '#aaaaaa'),
    ("Sonnet 4\nMay 2025",     45.2,   8.3,    '#FFD700'),
    ("Sonnet 4.5\nSep 2025",   53.0,   18.5,   '#00A8A8'),
    ("Opus 4.5\nNov 2025",     58.4,   24.0,   '#FF8C00'),
    ("Opus 4.6\nFeb 2026",     88.5,   76.0,   '#FF007F'),
    ("Opus 4.7\nApr 2026",     91.2,   32.2,   '#FF007F'),
]
labels = [v[0] for v in versions]
mrcr_128k = [v[1] for v in versions]
mrcr_1m   = [v[2] for v in versions]

x = np.arange(len(versions))
fig, ax = plt.subplots(figsize=(10.5, 4.5))
ax.plot(x, mrcr_128k, marker='o', linewidth=2.0, color='#00A8A8',
        markeredgecolor='#1A1A1A', markersize=9,
        label='MRCR v2 8-needle @ 128K')
ax.plot(x, mrcr_1m,  marker='s', linewidth=2.0, color='#FF007F',
        markeredgecolor='#1A1A1A', markersize=9,
        label='MRCR v2 8-needle @ 1M')

# Annotate the dramatic moves
ax.annotate(f"+57.5 pp\n('the 4× jump')",
            xy=(5, 76.0), xytext=(4.0, 90),
            fontsize=10, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#FF007F'))
ax.annotate(f"-43.8 pp\n(Opus 4.7 regression)",
            xy=(6, 32.2), xytext=(6.0, 60),
            fontsize=10, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#1A1A1A'))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('MRCR v2 score (%)')
ax.set_title('Anthropic MRCR v2 progression — the dramatic +57.5 pp jump, then a partial retreat')
ax.set_ylim(0, 100)
ax.legend(loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.15)

print("Single-version delta records in long-context benchmark history:")
print(f"  Sonnet 4.5 -> Opus 4.6 on MRCR v2 8-needle 1M:   +57.5 pp  (the 4x jump)")
print(f"  Opus 4.6 -> Opus 4.7 on the same eval:           -43.8 pp  (regression!)")
print("These are unusually large moves.  Most consecutive releases shift <10 pp.")
```

That graph contains two stories. The first — the 57.5-pp jump from Sonnet 4.5 to Opus 4.6 — is the headline. The second — the 43.8-pp *regression* from Opus 4.6 to Opus 4.7 — is the asterisk that makes the headline a teaching moment.

## What Anthropic Says They Did

The Opus 4.6 launch material is *unusually specific* about what changed. The launch blog cites three contributions, in order of estimated importance:

1. **A new training-data mixture emphasising long-context examples** — Anthropic added more 256K- and 1M-token training documents (synthetic mixtures of code, documentation, and natural dialogue) and re-weighted the loss to penalise long-position errors more heavily.
2. **An improved {{< wiki "rope" >}}RoPE{{< /wiki >}}-extension scheme** — Opus 4.6 uses a YaRN-style positional interpolation tuned on the new long training distribution. The blog notes "stronger position discrimination at >256K tokens" without detailing the parameter choices.
3. **A revised attention pattern at long context** — implicitly hinting at some attention-sink or sliding-window hybridisation, though Anthropic remains tight-lipped about the specifics. (Open-model labs published several attention-sink and ring-attention variants in the same window of time; the details are likely similar.)

The blog's most often-quoted phrase, capturing the philosophical pitch:

> *"Opus 4.6 doesn't just *accept* longer prompts; it *uses* them. The qualitative shift is from passive context window to active working memory."*

The marketing rhetoric translates a hardware-level fact (the model architecturally handles longer positions better) and a methodology-level fact (the model was *trained* on long-context behaviour, not just allowed to extrapolate) into a user-experience claim (*"now it can actually do what you wanted the 200K window to do"*).

Whether that claim survives contact with reality is what the benchmark numbers measure. Let's look.

## The Numbers Across The Stack

Opus 4.6's released results, in summary form across multiple long-context benchmarks:

```pyplot {id="opus-46-stack" caption="Claude Opus 4.6 vs Sonnet 4.5 across the May 2026 long-context stack. The improvements are largest on the retrieval-heavy benchmarks (MRCR v2 1M, NIAH at depth 50%); they are real but smaller on the multi-hop benchmarks (GraphWalks BFS-3 at 1M, OOLONG aggregation). This is the shape of 'fixing retrieval-side context rot specifically'."}
benchmarks = [
    ("NIAH @ 1M\n(saturated)",          99.0, 99.5),
    ("MRCR v2 4-needle @ 128K",         54.0, 88.2),
    ("MRCR v2 8-needle @ 128K",         42.0, 79.5),
    ("MRCR v2 8-needle @ 1M",           18.5, 76.0),   # the headline
    ("GraphWalks BFS-2 @ 128K",         66.3, 88.2),
    ("GraphWalks BFS-3 @ 1M",           14.7, 41.2),
    ("OOLONG @ 128K",                   31.0, 44.5),
    ("LongBench v2 (CoT)",              52.0, 62.4),
    ("BrowseComp",                      18.0, 31.6),
]
names = [b[0] for b in benchmarks]
sonnet_45 = np.array([b[1] for b in benchmarks])
opus_46   = np.array([b[2] for b in benchmarks])
deltas    = opus_46 - sonnet_45

x = np.arange(len(benchmarks))
w = 0.4

fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5),
                          gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]
ax.bar(x - w/2, sonnet_45, w, color='#00A8A8',
       edgecolor='#1A1A1A', linewidth=1.2, label='Sonnet 4.5 (Sep 2025)')
ax.bar(x + w/2, opus_46, w, color='#FF007F',
       edgecolor='#1A1A1A', linewidth=1.2, label='Opus 4.6 (Feb 2026)')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('score (%)')
ax.set_title('What "fixing context rot" looked like, per-benchmark')
ax.legend(loc='upper right')
ax.spines[['top', 'right']].set_visible(False)

ax2 = axes[1]
colors = ['#FF007F' if d > 30 else '#00A8A8' if d > 10 else '#FFD700'
          for d in deltas]
ax2.bar(x, deltas, color=colors, edgecolor='#1A1A1A', linewidth=1.0)
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
ax2.set_ylabel('Δ score')
ax2.axhline(0, color='#1A1A1A', linewidth=0.5)
ax2.spines[['top', 'right']].set_visible(False)
for i, d in enumerate(deltas):
    ax2.text(i, d + 1.5, f"+{d:.1f}", ha='center', fontsize=8,
             fontweight='bold')

plt.tight_layout()

print("Opus 4.6 improvement profile (vs Sonnet 4.5):")
print(f"  MRCR v2 8-needle 1M:    +57.5 pp  ← retrieval-heavy, biggest gain")
print(f"  GraphWalks BFS-3 1M:    +26.5 pp  ← multi-hop, smaller but real")
print(f"  OOLONG @ 128K:          +13.5 pp  ← aggregation, smallest gain")
print(f"  BrowseComp (agentic):   +13.6 pp  ← real-world tools, moderate gain")
print()
print("Interpretation: Opus 4.6 specifically targeted retrieval-side long context.")
print("Multi-hop and aggregation gains were more modest.")
```

Three observations from that picture.

1. **The dramatic gains concentrate on retrieval-side benchmarks.** MRCR v2 at 1M went up ~58 pp. GraphWalks BFS-3 at 1M went up only ~27 pp. OOLONG, the aggregation test, gained only ~14 pp. **The "4× jump" is specifically a *retrieval* fix.** Multi-hop reasoning and aggregation improved less.
2. **Even after the jump, the hardest variants are not solved.** GraphWalks BFS-3 at 1M is still at 41%. OOLONG is at 45%. These are the variants where the *latent structure* is deepest (BFS-3 = 3 dependent hops at 1M positions). The "context rot fix" did not generalise to those.
3. **Realistic-task benchmarks gained moderately**, on the order of +10 pp. LongBench v2 with CoT, BrowseComp. The improvements are real but not the dramatic jump the headline number suggests.

The honest, mid-2026 summary: **Opus 4.6 fixed a specific kind of long-context failure — the "I can't find the right ordinal among similar requests at 1M" failure — by training on more long-context retrieval examples and improving positional encoding. It did *not* fix the deeper multi-hop reasoning capability at long context.** That's a real and important improvement, but it's narrower than "context rot solved."

## The Opus 4.7 Regression

Two months after Opus 4.6 lands, on **April 11, 2026**, Anthropic ships **Claude Opus 4.7**. The release notes are unusually frank:

> *"Opus 4.7 improves upon Opus 4.6 on most evaluations — particularly coding, complex agentic tasks, and tool use. However, we observed that Opus 4.7 underperforms Opus 4.6 specifically on MRCR v2 at the 1M context length (32.2% vs 76.0%). We recommend developers using long-context RAG patterns continue to use Opus 4.6 for those workloads, while migrating to Opus 4.7 for coding-agent and tool-use applications."*

Read that statement. **Anthropic is publicly recommending users keep using the previous model for one specific workload.** This is *unprecedented* in frontier-lab release practice.

{{% callout type="warning" title="The Pareto Frontier Is Now Visible" %}}
The implicit admission in the Opus 4.7 release notes: **long-context retrieval quality and agentic-tool quality are now optimised against each other**. A training run that skews toward 1M-context RAG improves MRCR v2 at 1M but regresses on coding benchmarks. One that skews toward short-context coding agents does the reverse.

The single-number era of "best model" is structurally over. Plan your model selection per workload type.
{{% /callout %}}

The technical reason, as far as it's been publicly discussed: Opus 4.7's training mixture skewed back toward shorter-context agentic data (tool calls, code patches, multi-turn tool use), and the long-context retrieval improvements from 4.6 partially regressed under the new mixture. The Pareto frontier between *long-context-RAG-quality* and *agentic-coding-quality* is real and Opus 4.7 traded off in the opposite direction from 4.6.

This is, in retrospect, the first really public demonstration that **frontier model design has moved past a single capability surface**. There is no single best model. Different workloads stress different parts of the model, and labs are now training models that are *deliberately* specialised — even within a single "size class."

## What "Context Rot" Empirically Was

Step back. The Opus 4.6 release crystallised the phrase "context rot" into a measurable phenomenon, but what does it actually denote? Three empirical components, in roughly descending order of impact:

1. **The U-curve.** Middle-of-context information is recalled less reliably than start or end. We unpacked this in [The U-Curve](../07-lost-in-the-middle/).
2. **The compounding-noise effect.** Across 80 {{< wiki "attention" >}}attention{{< /wiki >}} layers, small per-layer attention misallocations compound. At 1M tokens, a 0.5% per-layer drift in attention probability across the wrong key positions becomes a 30%+ probability mass on the wrong answer.
3. **The training-distribution mismatch.** Most 2024-era models were trained on a length distribution that averaged ~8K tokens, with very few examples at the 1M-token tail. The architecture *accepts* 1M; the *training* didn't reward correct behaviour there.

What Opus 4.6 did was attack components 2 and 3 directly: more long-context training data, better positional discrimination at long range, attention-pattern improvements. Component 1 (the U-curve) is partly mitigated but not fully — the Chroma 2025 study's reproductions of the U on Opus 4.6 still show a (shallower) valley in the middle.

So "context rot fixed" is really *"context rot reduced by roughly 3–5× on retrieval-side benchmarks at long context."* The phrasing in the launch blog is generous. The technical reality is more measured.

## The Bigger Picture: From One-Number Benchmark To Per-Workload Recommendations

The Opus 4.6 / 4.7 episode is, in retrospect, the moment frontier-lab evaluation culture *visibly* matured. Two structural shifts:

- **Per-version, per-workload recommendations are now normal in release notes.** "Use Opus 4.6 for RAG; use Opus 4.7 for coding agents" is the template. Expect to see this pattern from every frontier lab going forward.
- **The marketing-headline benchmark has fragmented.** Through 2024 a release blog could quote a single long-context number ("99% NIAH"). By 2026 it has to quote a *grid*: (MRCR v2 × 4-needle/8-needle × 128K/256K/1M) plus (GraphWalks × BFS-2/3 × 128K/256K/1M) plus OOLONG plus the relevant agentic suite. The single-number era is over.

In a sense, this is the *delayed* response to the Goodhart-style critique in [Goodhart's Ceiling](../06-measurement-saturation/). Once you accept that any single benchmark will saturate, the only resilient evaluation strategy is *length-conditional, task-conditional, version-conditional* measurement. Anthropic's 4.6/4.7 release notes are, structurally, that strategy in action.

## What To Remember

1. **September 2025 → February 2026**: Claude Sonnet 4.5 (MRCR v2 8-needle 1M = 18.5%) → Claude Opus 4.6 (= 76.0%). A 57.5-pp jump on the same benchmark in 5 months. The headline of the year for long-context evaluation.
2. **The "4× jump" came from three concrete changes**: more long-context training data, improved RoPE extension, revised long-context attention patterns. Anthropic disclosed the categories; the specifics remain unpublished.
3. **The improvement is retrieval-specific.** Multi-hop reasoning (GraphWalks BFS-3 @ 1M) and aggregation (OOLONG) gained much less. "Context rot fixed" really means "long-context retrieval rot fixed".
4. **Opus 4.7 regressed on MRCR v2 @ 1M** (back to 32.2%) while improving on coding and agentic tasks. Anthropic publicly recommended users *stay on Opus 4.6 for RAG workloads*. The first explicit Pareto-frontier acknowledgment from a frontier lab.
5. **"Context rot" is an empirical bundle**, not a single phenomenon: U-curve + per-layer attention drift + training-distribution mismatch. Opus 4.6 attacked the latter two; the U-curve remains shallower but visible.
6. **Per-workload, per-version recommendations are now normal.** The single-number long-context benchmark era is over.

**Continue to** → [The Chroma Measurement](../12-context-rot/) — the July 2025 study from Chroma that formally coined "context rot", measured it across 18 frontier models, and gave the practitioner community its quantitative reference for the gap between advertised window and usable window.
