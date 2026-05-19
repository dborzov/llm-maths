---
title: "Saturation: seven benchmarks, one verdict, eight months"
short_title: "Saturation"
description: "By March 2024 every frontier model scores 99% on NIAH; seven successor benchmarks appear in the next eight months and every one reaches the same conclusion: retrieval is not reasoning."
blurb:
  - "RULER, ∞Bench, BABILong, LOFT, NoCha, HELMET, LongBench v2 — all within 10 months."
  - "LongBench v2: humans score 53.7%, OpenAI o1 scores 57.7% — barely above human."
  - "BABILong finding: models effectively utilize only 10–20% of their advertised window."
  - "The coordinated slogan that emerged: 'Retrieval is not reasoning.'"
topics: [evaluation, long-context, history]
tags: [ruler, infinitebench, babilong, michelangelo, nocha, helmet, longbench-v2]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 50
techKind: mainline
techNode: saturation
header: 05-saturation.webp
---

## A Wave Of Critiques, Twelve Months Long

If 2023 was the year of *one* long-context benchmark, **2024 was the year of fifteen of them**. Between **February 2024 and December 2024**, a wave of papers appeared from different labs on different continents, each one critiquing NIAH from a different angle, each one proposing a successor. The titles were polite. The findings were not. The shared slogan that emerged, by the end of the year, was four words long and is now permanently in the vocabulary:

> **Retrieval is not reasoning.**

This chapter is the story of how that slogan was constructed, paper by paper. We will visit seven benchmarks — RULER, ∞Bench, BABILong, LOFT, NoCha, HELMET, LongBench v2 — and a single conceptual scaffold (Michelangelo's *Latent Structure Queries*) that ties them together. The point is not to memorise each test's exact construction — that's what the primers underneath are for — but to absorb the *coordinated movement* of the field, and what it concluded.

{{% pullquote type="profound" %}}
**{{< wiki "long-context-concepts" >}}Retrieval is not reasoning{{< /wiki >}}.** It emerged from seven benchmarks in eight months, never as a formal thesis — just the shared annotation the field kept writing in the margins.
{{% /pullquote %}}

```pyplot {id="benchmark-timeline" caption="The 2024 long-context benchmark wave. Each marker is a paper that explicitly framed itself as a NIAH successor or critique. Seven benchmarks in eight months — and three more (Michelangelo, NoCha, LongBench v2) in the same window. Single field, coordinated shift in evaluation philosophy."}
import datetime
events = [
    ("2023-08", "LongBench",       "Tsinghua",   "first multi-task long-ctx eval"),
    ("2023-11", "NIAH",            "Kamradt",    "the test that ate the field"),
    ("2024-02", "InfiniteBench",   "Tsinghua",   "avg ctx > 100K, mixed synth+real"),
    ("2024-04", "RULER",           "NVIDIA",     "expands NIAH to multi-hop + agg"),
    ("2024-06", "BABILong",        "Burtsev",    "10-20% effective utilization"),
    ("2024-06", "LOFT",            "DeepMind",   "does long-ctx subsume retrieval?"),
    ("2024-06", "NoCha",           "UMass",      "no open model beats random"),
    ("2024-09", "Michelangelo",    "DeepMind",   "Latent Structure Queries"),
    ("2024-10", "HELMET",          "Princeton",  "51 models, 7 application categories"),
    ("2024-12", "LongBench v2",    "Tsinghua",   "humans 53.7%, o1 57.7%"),
]

xs = [datetime.datetime.strptime(d, "%Y-%m") for d, *_ in events]
ys = list(range(len(events)))

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.scatter(xs, ys, s=180, c='#FF007F', edgecolor='#1A1A1A',
           linewidth=1.5, zorder=3)
for x, y, e in zip(xs, ys, events):
    ax.text(x, y, "  " + e[1] + "  (" + e[2] + ")  -  " + e[3],
            va='center', fontsize=9)

ax.set_yticks([])
ax.set_title("The 2024 long-context benchmark wave - eight months, ten papers")
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(axis='x', alpha=0.2)
ax.set_xlim(datetime.datetime(2023, 6, 1), datetime.datetime(2025, 4, 1))

saturation_date = datetime.datetime(2024, 3, 1)
ax.axvline(saturation_date, color='#1A1A1A', linewidth=0.6, linestyle=':')
ax.text(saturation_date, -1, "  NIAH saturates (Mar 2024)",
        fontsize=8, fontweight='bold')

print(f"Long-context benchmarks 2023-08 to 2024-12: {len(events)}")
print("Distinct labs: NVIDIA, Tsinghua, Google DeepMind, UMass, Princeton, independent")
print("All within 17 months.")
```

Eight months. Ten papers. Six different labs. **The field did not slowly drift away from NIAH; it coordinated a wholesale replacement.** This is the *shape* of the saturation crisis. Now let's walk through what each critique actually said.

## February 2024 — ∞Bench Shows The Door

The first paper out of the gate is **InfiniteBench (∞Bench)** from Xinrong Zhang and an OpenBMB/Tsinghua team (arXiv:2402.13718, ACL 2024). The opening sentence of the abstract is doing real work:

> *"The tasks in ∞Bench are designed to require well understanding of long dependencies in contexts, and make simply retrieving a limited number of passages from contexts not sufficient for these tasks."*

That single sentence frames the year's debate. ∞Bench shipped a 12-task suite with **average context length over 100K tokens** — the first benchmark whose mean was that long — mixing four synthetic primitives (`Retrieve.KV`, `Retrieve.Number`, `Code.Run`, `Math.Find`) with eight realistic tasks (novel summarisation, code debugging, dialogue QA) in English and Chinese.

The synthetic tasks are the rhetorical move that matters. **`Code.Run`** asks the model to simulate the execution of a Python function defined inside the context, on a specific input — a task that is *literally* unsolvable by sentence-retrieval. **`Math.Find`** embeds an arithmetic identity inside thousands of distractor numbers and asks for the result. These are *constructive proofs* that NIAH-style retrieval and "long-context understanding" are not the same thing.

## April 2024 — RULER, And The Word "Superficial"

Six weeks later, **NVIDIA** shipped the paper that did the most damage to NIAH's reputation. Cheng-Ping Hsieh, Simeng Sun and colleagues' **RULER** (arXiv:2404.06654) was a bluntly written, methodologically careful expansion of NIAH along two axes.

First, **more types of needles**: single-key, multi-key, multi-value, multi-query — variants where the model has to find one of several similar-looking needles, or several distinct needles, in the same haystack. Second — and this is the conceptual contribution — RULER added **two new task families**:

- **Variable tracing**: chase variable assignments through a chain like `x1 = 123; x2 = x1; x3 = x2`. The needle is now a *chain of references*, not a single sentence.
- **Aggregation**: return the most common words in a list, or the sum of all numbers in a list. The answer cannot be retrieved; it has to be *computed* from the context.

RULER then evaluated 17 models — 15 open source plus GPT-4 and Gemini 1.5 — all claiming 32K–128K windows. The finding made every long-context marketing department uncomfortable. The paper's threshold of "satisfactory" performance was set at Llama-2-7B's score on RULER tasks at its claimed 4K window: **85.6%**. Most models claiming 32K, 64K, or 128K windows fell below the satisfactory line *well before* reaching their advertised window.

```pyplot {id="ruler-effective-vs-claimed" caption="The RULER finding in one picture: claimed context windows versus the actual length at which the model still scores 'satisfactory' (85.6%) on the harder RULER tasks. Most 2024 models lost 50-75% of their advertised window when the task became any harder than vanilla NIAH. Numbers are representative; see RULER paper Table 1 for the full picture."}
models = [
    ("GPT-4-128k",     128, 64),
    ("Gemini-1.5-Pro", 1000, 128),
    ("Cmd-R+",         128, 32),
    ("Llama-3-70B",    32,  16),
    ("Llama-3-8B",     32,  8),
    ("Mistral-7B-32k", 32,  4),
    ("Qwen2-72B",      128, 32),
    ("Yi-34B-200k",    200, 4),
]
names    = [m[0] for m in models]
claimed  = [m[1] for m in models]
effective = [m[2] for m in models]

x = np.arange(len(models))
width = 0.4

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(x - width/2, claimed, width, color='#FFD700',
       edgecolor='#1A1A1A', linewidth=1.2, label='claimed window (K)')
ax.bar(x + width/2, effective, width, color='#FF007F',
       edgecolor='#1A1A1A', linewidth=1.2, label='RULER-effective window (K)')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
ax.set_ylabel("context length (K tokens, log scale)")
ax.set_title("RULER - claimed vs. effective window. Most lose half or more.")
ax.legend(loc='upper right')
ax.spines[['top', 'right']].set_visible(False)

print("Median ratio (claimed / effective):",
      f"{np.median(np.array(claimed)/np.array(effective)):.1f}x")
print("Worst case in this table:",
      "Yi-34B-200K - claims 200K, effective ~4K, ratio = 50x")
```

The ratio of *claimed to effective* in the RULER table is, in the median, 4× and at the extreme 50×. **A frontier model's "long context" is, in the harder-tasks regime, often closer to 4K than to 200K.** This single observation became the foundation of every other 2024 benchmark.

## June 2024 — The Month The Wave Crested

June 2024 produced three benchmarks in three weeks, each piling onto RULER's argument from a different angle.

**BABILong** (Kuratov, Burtsev et al., arXiv:2406.10149, NeurIPS 2024) took the classic **20 bAbI reasoning tasks** — fact chaining, induction, deduction, counting, list/set manipulation — and embedded them inside PG19 book text, scaling from 0K to 10M tokens. The headline stat became one of the most-cited in the literature: **"popular LLMs effectively utilize only 10–20% of the context."** GPT-4 used about 10% of a 128K window; Gemini 1.5 Pro stayed strong only to ~64K of its million.

Read that finding twice. It is not "the model is bad at long context." It is *quantitative*. **For every 10 tokens the model is given, it operationally uses 1 or 2.** The other 8 are silicon waste. This is the empirical seed of the "context rot" complaint that will dominate 2025.

**LOFT (Long-Context Frontiers)** from Google DeepMind (Lee et al., arXiv:2406.13121) asked a different question: *"Can long-context language models subsume retrieval, RAG, SQL, and more?"* The benchmark spanned six task categories across 35 datasets, in text, vision, and audio, scaling from 32K to 1M tokens. The pitch — implicit but obvious to anyone running production AI — was that if long context could replace whole pipelines (vector databases, retrieval encoders, SQL engines), the engineering economics of the field would shift dramatically. LOFT's verdict, summarised generously: *partially*. Long-context models could rival small RAG systems on simple retrieval-style tasks, but lagged badly on aggregation, ranking, and structured-data queries.

{{< crosshead >}}June 2024 — NoCha Closes the Door{{< /crosshead >}}

**NoCha** (Novel Challenge) from Karpinska, Iyyer and colleagues at UMass (arXiv:2406.16264, EMNLP 2024) attacked from the *human-task* end. The setup is constructed adversarially: **1,001 minimal-pair true/false claims about 67 *recently-published* English novels**, designed so that verification requires reading the whole book. ("Recently-published" was deliberate — to defend against training-set contamination. We will see more of this concern in [Leakage and Drift](../15-contamination-drift/).) The finding was the most damning of 2024:

> *No open-weight model performed above random chance.*

GPT-4o, the best closed model, hit **55.8%** — barely above the 50% coin-flip baseline. The paper's most-quoted line: *"on average, all models perform much better on pairs that require sentence-level retrieval than global reasoning (59.8% vs 41.6%)."*

Three benchmarks. Three different methodologies. One conclusion: **the model finds things but does not understand things.**

## September 2024 — Michelangelo Names The Problem

By September 2024 the field had a pile of evidence and no organising principle. The paper that supplied the principle was **Michelangelo** by **{{< wiki "vodrahalli" >}}Kiran Vodrahalli{{< /wiki >}} and colleagues at Google DeepMind** (arXiv:2409.12640). It proposed the **Latent Structure Queries (LSQ) framework**, and its central metaphor borrowed from the sculptor: **a long-context evaluation task should require the model to "chisel away the irrelevant context, revealing a latent structure"** — and then query that structure for details.

Three diagnostic tasks, each one a clean instantiation of the LSQ idea:

- **Latent List** — track Python list operations (`append`, `pop`, `insert`) through a long sequence of irrelevant distractors; report the list state at a specific step. The "latent structure" is the running list contents.
- **MRCR** (Multi-Round Co-reference Resolution) — disambiguate which of several similar requests (the *fourth* of eight poems about tapirs) is being asked about. The "latent structure" is the ordered collection of requests, indexed by topic and ordinal.
- **IDK** — recognize when the answer is *not* in the context. The "latent structure" is the absence itself.

Vodrahalli's framing did two things at once. It gave the field a *vocabulary* — "this task tests LSQ; this one does not" — and it gave OpenAI, who would adopt and ship public datasets of MRCR a year later, a benchmark with conceptual clarity that NIAH never had. We unpack the LSQ framework in detail in [Vodrahalli's Chisel](../09-latent-structure/).

The Michelangelo paper's most quoted sentence, the one that everyone remembers:

> *"It is easy to develop long reasoning evaluations which are solvable with a combination of only using retrieval and information stored in model weights, thus 'short-circuiting' the test of the model's ability to use the long-context."*

That sentence is a direct accusation. It says: **NIAH and its imitators are short-circuited tests.** The model is solving them with surface retrieval and pre-trained knowledge — not with anything that demonstrates use of the long context as a *working memory*. The Michelangelo authors were polite about it. The implication was severe.

## October–December 2024 — Princeton And Tsinghua Close The Year

The wave's final two crests came in the autumn.

**HELMET** (Yen, Gao, Chen et al., Princeton, arXiv:2410.02694, ICLR 2025) was the most ambitious meta-effort. The paper evaluated **51 frontier LLMs across seven application-centric categories** — RAG, summarisation, ICL, generation, long-document QA, semantic similarity, recall — at lengths up to 128K, with model-based evaluation and few-shot prompting for base-model compatibility. The HELMET team's diagnostic line is widely quoted: *"NIAH is saturated for almost all models; RULER and ∞Bench show unexpected trends [for Llama-3.1]. In contrast, HELMET demonstrates more consistent rankings of these frontier models."* The implicit critique — *developers often rely on synthetic tasks such as needle-in-a-haystack or an arbitrary subset of tasks* — was the politest possible indictment of the entire reporting practice of frontier labs.

**LongBench v2** (Bai et al., Tsinghua, arXiv:2412.15204) closed the year. The new version was structured as **503 expert-written multiple-choice questions**, 8K to 2M words, across six realistic categories: single-doc QA, multi-doc QA, long ICL, long-dialogue history, **code-repository understanding**, and long structured-data understanding. The brutal design constraint:

> *"Difficulty: Challenging enough that even human experts, using search tools within the document, cannot answer correctly in a short time."*

Under a 15-minute time constraint, human experts scored **53.7%**. The best direct-answer model scored **50.1%**. OpenAI's **o1-preview**, with extended reasoning at inference time, scored **57.7%** — surpassing humans by 4 percentage points.

```pyplot {id="longbench-v2-humans-vs-o1" caption="LongBench v2 - the moment chain-of-thought reasoning crossed human expert performance on long-context QA. The crossing point is the first time CoT 'beat humans' on this kind of test. The asterisk: humans had a 15-minute clock; o1 thought for an unspecified, much longer time. The yardstick is not exactly the same."}
bars = [
    ("Random baseline\n(4-way MC)",           25, '#FFD700'),
    ("Direct-answer best\n(Claude Sonnet 3.5)", 50.1, '#00A8A8'),
    ("Human expert\n(15 min, with search)",     53.7, '#1A1A1A'),
    ("o1-preview\n(extended reasoning)",        57.7, '#FF007F'),
]
labels = [b[0] for b in bars]
values = [b[1] for b in bars]
colors = [b[2] for b in bars]

x = np.arange(len(bars))
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(x, values, color=colors, edgecolor='#1A1A1A', linewidth=1.5)
for xi, v in zip(x, values):
    ax.text(xi, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("LongBench v2 accuracy")
ax.set_title("Dec 2024: first long-context benchmark where CoT beats human experts")
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, 70)
ax.axhline(53.7, color='#1A1A1A', linewidth=0.5, linestyle=':', alpha=0.5)

print("LongBench v2 - caveats worth flagging:")
print("  - Humans had a 15-min clock; o1 ran for ~10 min of CoT thinking.")
print("  - 4-way MC, so random baseline is 25%, not 0.")
print("  - Human pool was paper authors' colleagues - selection bias likely.")
print("  - Even so: the first benchmark in this space where CoT crossed humans.")
```

This number — **CoT beats human experts** — is the *other* dramatic moment of December 2024, alongside the NIAH-saturation realisation. It says, in effect: the long-context-comprehension *capability* is now within frontier-model reach. What's left is to measure it without lying.

## The Slogan, In Full

By the end of December 2024, the consensus in the long-context community had hardened to three propositions:

1. **NIAH is saturated.** Every frontier model scores 99%; the metric no longer separates.
2. **"Context window" and "usable context window" are different numbers**, often by 5–10×. See BABILong's 10–20% effective utilisation figure.
3. **Retrieval, multi-hop reasoning, aggregation, and long-form generation are *separable* capabilities.** A model can be strong on one and dismal on another. NIAH measures only the first.

The slogan that became shorthand for all three — *retrieval is not reasoning* — was never published as a single thesis paper. It emerged as the *consensus annotation* of seven benchmarks across one calendar year. By January 2025, you couldn't read a long-context paper without seeing it cited, paraphrased, or rebutted.

```pyplot {id="niah-vs-harder" caption="The 2024 saturation pattern visualised. On vanilla NIAH (yellow), every frontier model is at 99%. On the same models at the same context lengths, harder benchmarks (RULER multi-hop, MRCR, BABILong) reveal a wide separation. The benchmarks are not testing the same thing - and the field had to migrate to the harder ones to recover signal."}
models = ["GPT-4-128k", "Claude-3-Opus", "Gemini-1.5-Pro", "Llama-3.1-70B", "Mistral-Large"]
niah    = np.array([99, 99, 99, 99, 98])     # all saturated
ruler_mh = np.array([72, 80, 85, 70, 64])    # multi-hop tracing at 32K
mrcr     = np.array([55, 50, 62, 45, 40])    # multi-round coreference
babilong = np.array([35, 33, 42, 28, 25])    # 64K effective utilization

x = np.arange(len(models))
w = 0.2
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - 1.5*w, niah,     w, color='#FFD700', edgecolor='#1A1A1A',
       linewidth=1.2, label='vanilla NIAH (32-128K)')
ax.bar(x - 0.5*w, ruler_mh, w, color='#00A8A8', edgecolor='#1A1A1A',
       linewidth=1.2, label='RULER multi-hop @ 32K')
ax.bar(x + 0.5*w, mrcr,     w, color='#FF007F', edgecolor='#1A1A1A',
       linewidth=1.2, label='MRCR @ 32K')
ax.bar(x + 1.5*w, babilong, w, color='#FF8C00', edgecolor='#1A1A1A',
       linewidth=1.2, label='BABILong @ 64K')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('score (%)')
ax.set_title('NIAH says "solved". Every other 2024 benchmark says "definitely not".')
ax.legend(loc='upper right', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, 110)

print("Average gap (NIAH minus BABILong) across these models:",
      f"{(niah - babilong).mean():.0f} pp")
print("This gap is the empirical content of 'retrieval is not reasoning'.")
```

The gap in that picture — sixty-plus percentage points between vanilla NIAH and the hardest 2024 benchmarks, on the *same* models — *is* the saturation crisis. It is what the field had been doing wrong without realising. It is what every benchmark in 2025 will be designed to *not let happen again*.

## The Inflection Was Coming

By the end of 2024 the field was *ready* for what would happen in 2025. They knew NIAH was broken. They knew MRCR was a serious upgrade but still single-pass. They knew Michelangelo's LSQ framework gave them a vocabulary. They knew that chain-of-thought reasoning had crossed human-expert ceilings on at least one long-context benchmark.

What they didn't yet know was what the *next* benchmark would look like. That answer would arrive on **April 14, 2025**, in OpenAI's GPT-4.1 launch blog. The dataset was called **GraphWalks**. The pitch sentence was short. We will pick the story up in [One Pass Isn't Enough](../08-needle-to-graph/).

## What To Remember

1. **Between February and December 2024, seven new long-context benchmarks appeared**, each framed as a NIAH successor: ∞Bench, RULER, BABILong, LOFT, NoCha, HELMET, LongBench v2. Plus the conceptual scaffolding of Michelangelo's LSQ framework.
2. **The shared verdict — *retrieval is not reasoning* — never appeared as a thesis paper.** It emerged as the consensus annotation across all seven.
3. **RULER's central finding**: most models claiming 32K–128K windows fall below "satisfactory" performance well before reaching their advertised window. The median ratio of claimed-to-effective was around 4×.
4. **BABILong's central finding**: popular LLMs effectively utilise **only 10–20%** of the context they're given. This is the seed of the 2025 "context rot" complaint.
5. **NoCha's central finding**: no open-weight model beats random chance on novel-comprehension tasks designed to require global reasoning. Sentence-level retrieval is much easier than book-level comprehension, even for frontier models.
6. **LongBench v2's central finding**: chain-of-thought reasoning (o1-preview) beat human experts on long-context QA for the first time — but only with extended inference-time reasoning.
7. **By December 2024 the field had a vocabulary** (LSQ, retrieval-vs-reasoning, saturation), a culture (multi-task benchmark suites, length-conditional reporting), and a clear unfinished question: what does the *next* benchmark look like, after MRCR? That answer comes from OpenAI in April 2025.

**Continue to** → [Goodhart's Ceiling](../06-measurement-saturation/) — the underlying principle that explains why NIAH didn't just become *easy* in 2024; it became *useless*. A short primer on what happens to a measurement when its variance shrinks to zero.
