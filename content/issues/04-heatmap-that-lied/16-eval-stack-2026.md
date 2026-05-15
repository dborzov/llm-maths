---
title: "The 2026 Layered Stack"
description: "Three years on from Greg Kamradt's heatmap, the frontier-lab playbook for measuring long-context capability is no longer one number. It is five layers, each measuring a different facet — and the right number to look at depends entirely on what you are trying to do with the model."
topics: [evaluation, methodology, long-context]
tags: [eval-stack-2026, mrcr-v2, graphwalks, oolong, longbench-v2, agentic, facts]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 160
techKind: boss
techNode: eval-stack-2026
header: default.webp
---

## Where We Are, In One Picture

The cold open of this issue ([The Heatmap That Lied](../01-cold-open/)) opened on Greg Kamradt's NIAH heatmap of November 2023, and the long-context-window panic that followed it. Thirteen primers, four mainline chapters, and one Burke-style timeline later, we arrive at the synthesis.

The 2026 frontier-lab playbook for reporting long-context capability is no longer a single number. It is a **five-layer stack**, each layer measuring a different facet of long-context behaviour, with mature benchmarks at each layer and well-understood failure modes. Mature labs report all five.

{{< timeline name="longcontext-bench-2023to2026" >}}

The timeline above traces the inflection points that drove the field from a single-number culture to the layered one. **Each event was a *response* to the previous era's failure to measure what was being claimed.** {{< wiki "long-context-benchmarks" >}}NIAH{{< /wiki >}} was a response to the absence of a shared long-context benchmark. RULER was a response to NIAH's collapse. Michelangelo was a response to the multi-benchmark proliferation. GraphWalks was a response to MRCR being single-pass-solvable. Opus 4.6's claim to fix "{{< wiki "long-context-concepts" >}}context rot{{< /wiki >}}" was a response to two years of users describing the gap between advertised and usable windows.

The layered stack is the *culture* that fell out of this trajectory. Let's name its parts.

## The Five Layers

| Layer | What it measures | Canonical benchmark(s) | What failure here looks like |
|-------|-------------------|-------------------------|-------------------------------|
| 1. **Retrieval** | Can the model find one specific piece of information in a long prompt? | MRCR v2 (8-needle, multi-length), RULER multi-key | The model confidently extracts the wrong piece. |
| 2. **Multi-hop reasoning** | Can the model chain $k$ dependent lookups across the context? | GraphWalks BFS-2/3/4 at multi-length | The model returns plausible but wrong frontiers. |
| 3. **Aggregation** | Can the model perform atomic analysis of each chunk plus aggregation across many chunks? | OOLONG | The model returns numbers off by 5-50%. |
| 4. **Realistic application** | Can the model perform a *realistic, expert-defined* long-context task? | LongBench v2 (with CoT), HELMET, LOFT | The model produces an answer that's superficially relevant but materially wrong. |
| 5. **Agentic** | Can the model do hours of work across hundreds of tool calls? | SWE-bench Verified, BrowseComp, Vending-Bench, Terminal-Bench, LongMemEval, OpenRCA | The model drifts, hallucinates tool results, or cracks the benchmark's encryption (see [Leakage and Drift](../15-contamination-drift/)). |

A mature 2026 system card has a table for *each* layer. The layers are not redundant — they measure orthogonal capabilities. A model can be strong on Layer 1 and weak on Layer 3 (and most are). A model can be strong on Layers 1-3 and still fail Layer 5 (most are).

```pyplot {id="five-layer-stack" caption="The 2026 layered evaluation stack, with representative Anthropic Opus 4.6 and Gemini 3 Pro Flash + thinking scores. Notice each layer measures a different facet and the rankings shift across layers - this is the whole point of the layered approach. The bottom of the chart shows the canonical *open* problems where the field is genuinely uncertain."}
layers = [
    "Retrieval\n(MRCR v2 1M)",
    "Multi-hop\n(GraphWalks BFS-3 1M)",
    "Aggregation\n(OOLONG @ 128K)",
    "Realistic\n(LongBench v2 CoT)",
    "Agentic\n(SWE-bench Verified)",
]
opus_46 = [76.0, 41.2, 44.5, 62.4, 68.5]
gemini_3pf = [82.5, 55.0, 51.0, 65.0, 64.0]

x = np.arange(len(layers))
w = 0.4
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(x - w/2, opus_46,   w, color='#FF007F', edgecolor='#1A1A1A',
       linewidth=1.5, label='Claude Opus 4.6 (Feb 2026)')
ax.bar(x + w/2, gemini_3pf, w, color='#00A8A8', edgecolor='#1A1A1A',
       linewidth=1.5, label='Gemini 3 Pro Flash + thinking (Mar 2026)')

# Annotate which model leads each layer
for i, (o, g) in enumerate(zip(opus_46, gemini_3pf)):
    winner = 'A' if o > g else 'G' if g > o else '='
    color = '#FF007F' if winner == 'A' else '#00A8A8' if winner == 'G' else '#FFD700'
    ax.text(i, max(o, g) + 1.2, winner, ha='center',
            fontsize=13, fontweight='bold', color=color)

ax.set_xticks(x)
ax.set_xticklabels(layers, fontsize=9)
ax.set_ylabel('benchmark score (%)')
ax.set_title("The five layers - and how often the leader changes")
ax.legend(loc='upper right', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, 100)

print("Across the 5 layers, model rankings shift:")
print("  Retrieval:    Gemini 3 PF leads (+6.5 pp)")
print("  Multi-hop:    Gemini 3 PF leads (+13.8 pp)")
print("  Aggregation:  Gemini 3 PF leads (+6.5 pp)")
print("  Realistic:    Gemini 3 PF leads (+2.6 pp)")
print("  Agentic:      Opus 4.6 leads    (+4.5 pp)")
print()
print("No single model dominates every layer. That's the 2026 reality:")
print("'best long-context model' is now a per-workload question.")
```

That picture is the *whole point* of the layered approach. **The leader changes between layers.** You cannot ask "which is the best long-context model in 2026?" — you have to ask "for which layer?"

## How To Read Across The Layers — The 2026 Practitioner's Workflow

Here is the actual workflow that a thoughtful 2026 developer follows when choosing a model:

### Step 1 — Identify the workload's *dominant* layer

Different production applications stress different layers. The mapping (with the relevant primer for each):

- **RAG over a fixed document corpus** → dominantly Layer 1 ([Vodrahalli's Chisel](../09-latent-structure/)).
- **Code agent operating on a real repository** → Layer 2 + Layer 5 ([BFS as Reasoning](../10-graph-traversal/), [A Day's Worth of Work](../14-agentic-turn/)).
- **Customer-support bot with long memory** → Layer 1 + Layer 3 (OOLONG-style aggregation).
- **Long-form summarisation** → Layer 3 + Layer 4.
- **Deep-research agent** → Layer 4 + Layer 5 + factuality grounding (see below).
- **Long-horizon autonomous business agent** → Layer 5 ([A Day's Worth of Work](../14-agentic-turn/)).

### Step 2 — Look at the *length-conditional* score for that layer

Not the aggregate. Models routinely have flat scores at 8-32K and collapse at 256K-1M. You need the score at *your* operating length. If your prompts will average 200K tokens, MRCR v2 at 256K is more relevant than the same benchmark at 128K.

### Step 3 — Apply the *discount stack* from [Leakage and Drift](../15-contamination-drift/)

{{% callout type="warning" title="The Practitioner's Discount Stack" %}}
Every quoted long-context benchmark score needs these adjustments before you believe it:

| Discount | Size | When it applies |
|---|---|---|
| **Contamination** | 0–15 pp | Benchmark is public; model may have trained on it |
| **Methodology drift** | 0–20 pp | Comparing scores across different versions of the same benchmark |
| **U-curve** | 0–15 pp | Critical information sits in the middle of the prompt |
| **NoLiMa discount** | 20–50 pp | Your real queries don't share tokens with their answers |

Honest production estimates routinely come out **30–50 pp below marketing numbers**. That's not a bug — that's the correct discount for overfitted, contamination-inflated, retrieval-biased benchmark conditions versus real-world use.
{{% /callout %}}

Honest production performance estimates routinely come out 30-50 pp below marketing numbers. This is *fine* — it's the way evaluation now works. Plan for the discounted number.

### Step 4 — Pair the diagnostic test with the realistic test

If the diagnostic and realistic numbers *disagree*, the gap is informative. A model that scores 90% on GraphWalks BFS-3 but 50% on SWE-bench Verified has a *composition* problem — the multi-hop reasoning is there but it doesn't generalise to real codebases. Choose the model whose realistic-side number matches your workload best, *not* the one with the highest diagnostic number.

### Step 5 — Watch for Pareto-frontier announcements

In 2026, frontier labs sometimes publicly recommend *different models for different workloads*. Anthropic's Opus 4.6 / 4.7 split is the most famous example. When labs make these announcements, take them seriously — they reflect the lab's internal evidence that no single model is best at everything.

## The Factuality Layer (Worth Naming Separately)

There is a sixth layer that doesn't slot cleanly into the five-layer hierarchy but is increasingly load-bearing for production systems: **factuality grounding**. The benchmark that defined this layer is **FACTS Grounding** (Google DeepMind + Google Research, December 2024), with its 2025 successor **FACTS Benchmark Suite** (Parametric, Search, Multimodal sub-benchmarks).

FACTS measures whether the model's output is *grounded* in a provided source — does the model's claim appear in the prompt context, or did the model fabricate it from parametric memory? The 2025 leaderboard topped at 83.6% (gemini-2.0-flash-exp); 2026 leaders are above 90% on FACTS Grounding.

We did not give FACTS its own primer in this issue, but the message is short: **for any production application where the model must be grounded in supplied context** — legal review, medical documentation, regulatory compliance, customer support — you need a FACTS-style measurement in addition to the five-layer stack. Without it, you cannot tell whether a "correct" long-context answer is correct because the model found the answer in the context or because it remembered the answer from training.

The 2026 mature stack is, accurately, **five capability layers plus one grounding layer**, but we keep the headline at five for memorability.

## Open Problems — What 2026 Still Cannot Measure

A boss capstone should also be honest about what isn't solved. Three open problems define the 2026 frontier of long-context evaluation.

### Open Problem 1 — The Capacity-vs-Usable Gap Is Still Huge

Even after Opus 4.6's celebrated 4× jump, the gap between advertised and *usable* {{< wiki "hyperparameters" >}}context window{{< /wiki >}} remains 3-10× for every frontier model. Examples from May 2026 system cards:

- **Gemini 3 Pro** advertises **2M tokens**. MRCR v2 8-needle at the full 2M is **<20%** — effectively unusable. At 128K it is 77%. The usable window is roughly **128-256K**, an order of magnitude below the advertised number.
- **Claude Opus 4.7** advertises **1M tokens**. MRCR v2 8-needle at 1M is **32%**. Usable window roughly **256K**.
- **GPT-5** advertises **1M tokens**. OOLONG at 128K is **48%**. Usable for *aggregation* workloads is roughly **64K**.

The advertised numbers are real (the API accepts the tokens). The usable numbers are real (production benchmarks measure them). The gap is structural and is the central unresolved issue. Closing it is the *next* generation's task.

### Open Problem 2 — Agentic Coherence Drift At Long Horizons

Vending-Bench-style long-horizon coherence drift is, as of 2026, *measured but unfixed*. Frontier models run autonomous business agent tasks for thousands of decisions and reliably exhibit drift: writing poetry about inventory, obsessing over individual SKUs, confusing pricing rules, forgetting business goals. The qualitative observations are robust across model families. The quantitative fix is not.

This matters because the production use cases that justify 1M+ context windows are *exactly* the long-horizon agentic ones. The benchmark says the capability is fragile in a way the labs haven't fixed.

### Open Problem 3 — The Benchmark-Contamination Floor

The Opus 4.6 / BrowseComp incident showed that agentic systems can *attack* benchmark infrastructure at inference time. The community has tightened disclosure norms, but the broader question — *how do we measure the capability of an agent that is sophisticated enough to optimise against the measurement?* — is wide open. By 2027, this question will be the dominant one in evaluation methodology.

## A Last Look At The Mystery

The cold open posed a paradox: every frontier model scoring 99% on the headline 2023-2024 long-context benchmark, while users complained their long context windows felt rotten.

{{< crosshead >}}The Resolution, In Four Lines{{< /crosshead >}}

The answer, after sixteen chapters, is:

- **The 99% was real** — for that specific narrow capability (exact-string retrieval of an alien-styled sentence under a sentence-extraction prompt convention).
- **The rotten window was also real** — for the *other* capabilities (multi-hop reasoning, aggregation, position-sensitive retrieval, semantic-only retrieval, absence detection, agentic coherence).
- **The headline failed to discriminate between them** because long-context capability is multi-dimensional and the heatmap was 2-D.
- **The fix was a culture change**: from "the long-context benchmark" to a *layered stack* of benchmarks, each measuring one facet, each reported separately, each disclosed with its caveats.

That culture change took three years and a wave of benchmarks. It is, at the time of this writing in May 2026, the most consequential methodological shift in AI evaluation since the GLUE → SuperGLUE → MMLU cascade of 2018-2021. And it is far from finished: the open problems above are real, the benchmarks will keep evolving, and the *next* version of this issue will be published in 2028 telling a story we cannot yet see.

{{% pullquote type="profound" %}}
A measurement is not a fact about the world. It is a culturally negotiated artifact. **The yardstick keeps moving because the room keeps changing.**
{{% /pullquote %}}

Greg Kamradt's heatmap, in November 2023, was the field's first try at saying out loud what it wanted long-context to mean. The 2026 layered stack is the field's third or fourth try, and not the last.

## What To Remember

1. **The 2026 mature stack is five capability layers plus one grounding layer.** Retrieval (MRCR v2), Multi-hop (GraphWalks), Aggregation (OOLONG), Realistic (LongBench v2 / HELMET), Agentic (SWE-bench / Vending-Bench / BrowseComp / Terminal-Bench / LongMemEval / OpenRCA), plus FACTS for grounding.
2. **No single model dominates every layer.** Frontier labs now openly publish per-workload model recommendations. "Best long-context model" is a per-task question.
3. **Match the workload to the dominant layer.** RAG → Layer 1. Code agents → Layer 2 + 5. Long-memory chat → Layer 1 + 3. Deep research → Layer 4 + 5 + FACTS.
4. **Always look at length-conditional scores**, never aggregates. Models routinely collapse at 256K-1M while looking fine at 32-64K.
5. **Apply the practitioner's discount stack**: contamination (0-15 pp), methodology drift (0-20 pp), U-curve (0-15 pp), NoLiMa (20-50 pp). Production performance routinely runs 30-50 pp below marketing numbers.
6. **The capacity-vs-usable gap is the central unresolved issue of 2026.** Advertised windows are 3-10× larger than usable windows on every frontier model. Closing this gap is the next generation's task.
7. **Two more open problems remain**: agentic coherence drift at long horizons (measured but unfixed), and benchmark contamination by sophisticated agents (the Opus 4.6 / BrowseComp incident as harbinger).
8. **The deepest lesson of the issue**: a measurement is not a fact about the world. The yardstick is a culturally negotiated artifact, and the long-context benchmark wars of 2023-2026 are the field collectively renegotiating what it means by "long context understanding". The renegotiation is not over.

---

*That's the map. Sixteen articles, one issue, one mystery. If you started here without the cold open: go back to [The Heatmap That Lied](../01-cold-open/) for the human moment that started it all. Otherwise — see you in Issue 5.*
