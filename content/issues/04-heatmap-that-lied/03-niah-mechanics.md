---
title: "Needle in a Haystack"
description: "Late 2023, Greg Kamradt: take Paul Graham essays, glue them together, paste a sentence about a sandwich somewhere in the middle, ask the model where to eat in San Francisco. The benchmark that took over an industry."
topics: [evaluation, long-context]
tags: [niah, kamradt, claude-2.1, anthropic, paul-graham]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 30
techKind: primer
techNode: niah-mechanics
header: default.webp
---

## The Whiteboard, The Weekend, The Test

Picture **Greg Kamradt** at his kitchen table in **San Francisco**, sometime in the third week of November 2023. He is an independent AI developer with a substantial following on X — a wiry, sharp-eyed presence in the corner of the screen on every Latent Space podcast, somebody who tries every new model the day it ships. Two days earlier, on **November 19, 2023**, Anthropic put **Claude 2.1** behind their API with a **200K-token** context window — at that moment, the largest commercial model context in the world.

Kamradt wants to know: *does it actually work?* Can the model really pay attention to information buried 100,000 tokens deep, or is the window a marketing artifact?

So he writes a quick script. The script is straightforward enough that *you* could write it before lunch. Here, in spirit, is what it does:

```python
# Pseudocode, not pyplot. The actual NIAH harness, with names changed.
haystack = concat(paul_graham_essays)       # ~200K tokens of public text

for context_length in [1K, 4K, 8K, 16K, 32K, 64K, 96K, 128K]:
    for depth_pct in [0, 10, 20, ..., 90, 100]:
        prompt = (
            haystack[:context_length].splice_at(
                position = context_length * depth_pct // 100,
                inject   = "The best thing to do in San Francisco is "
                           "eat a sandwich and sit in Dolores Park on a sunny day."
            )
            + "\n\nWhat is the most fun thing to do in San Francisco?"
        )
        answer = model(prompt)
        score[context_length][depth_pct] = (
            "Dolores Park" in answer and "sandwich" in answer
        )
```

Eight context lengths × eleven depths = **88 cells**, each one a model call. At Anthropic's then-list price of roughly **$8 per million input tokens**, the full sweep on Claude 2.1 costs about $50–$100 of API credit. *That is the entire methodology.* No held-out test set. No human raters. No statistical correction for multiple comparisons. A weekend script and a credit card.

The output is a 2-D grid of pass/fail flags. Render it as a heatmap and *that* is the picture that launched a thousand product slides.

## The Anatomy Of A Needle

Let's slow down and look at what the prompt actually contains. The haystack is non-trivial: Paul Graham's essays, by 2023, had been on the internet since the early 2000s. They are exactly the kind of writing — clear, opinionated, vocab-light, paragraph-structured — that big language models have *certainly* trained on, often multiple times. The model's relationship with this text is not "novel passage I am encountering for the first time" but more like "a paraphrase of something my weights already know."

The needle sentence, by contrast, is a deliberate stylistic outlier:

> *"The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."*

Read it twice. Two things jump out. First, the *register* is wrong for Paul Graham. He writes about startups, Lisp, painters, philosophy — not sandwiches. The sentence has no business in an essay called *"How to Start a Startup"* or *"What I Did This Summer"*. Second, the *content* directly answers a question — *"what to do in San Francisco"* — that no Paul Graham essay literally asks. The needle is *engineered* to be the unambiguous answer to one specific question, given the rest of the haystack to ignore.

This is, in retrospect, the load-bearing observation of the entire NIAH paradigm. **The needle is detectable as an out-of-distribution token sequence inside the haystack.** A well-trained model could, in principle, find it without "remembering" anything — just by detecting the local stylistic seam where the embedded sentence joins the surrounding text.

```pyplot {id="needle-styled" caption="A schematic of the NIAH prompt construction. The haystack is a multi-paragraph essay; the needle is a single short, topical sentence inserted at a target depth. The visual seam — colour change here, stylistic register in practice — is doing more measurement work than 2023's framing acknowledged."}
fig, ax = plt.subplots(figsize=(9, 3.6))

# Schematic representation of a NIAH prompt at 50% depth
ctx_blocks = 30
needle_pos = 15  # 50%
for i in range(ctx_blocks):
    color = '#FFD700' if i != needle_pos else '#FF007F'
    width = 1.0 if i != needle_pos else 1.0
    ax.barh(0, width, left=i, height=0.6, color=color,
            edgecolor='#1A1A1A', linewidth=0.5)

# Annotations
ax.annotate("the haystack\n(Paul Graham essays,\nfamiliar to model)",
            xy=(7, 0.5), xytext=(7, 1.2), fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='-', color='#1A1A1A'))
ax.annotate("the needle\n('Dolores Park…sandwich')\nstylistically alien",
            xy=(needle_pos + 0.5, 0.4), xytext=(needle_pos + 0.5, -1.0),
            fontsize=10, ha='center', color='#FF007F', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FF007F'))
ax.annotate("query: 'what is the most fun thing to do in SF?'",
            xy=(ctx_blocks-0.5, 0.3), xytext=(ctx_blocks-0.5, 1.2),
            fontsize=9, ha='right', style='italic',
            arrowprops=dict(arrowstyle='-', color='#1A1A1A'))

ax.set_xlim(-1, ctx_blocks + 1)
ax.set_ylim(-1.6, 1.8)
ax.set_yticks([])
ax.set_xticks([0, ctx_blocks/2, ctx_blocks])
ax.set_xticklabels(['start', '50% depth', 'end (query here)'])
ax.set_title("NIAH prompt anatomy: a foreign sentence wedged into familiar text")
ax.spines[['top', 'right', 'left']].set_visible(False)
```

If the needle were *not* obviously foreign — if it were a Paul Graham–style aphorism about San Francisco that also mentioned sandwiches — the test would be substantially harder. **Kamradt's needle is closer to a watermark than a memory probe.**

## The Heatmap That Was Bigger Than Its Math

Now the visual. Kamradt's choice to render the (context-length × depth) grid as a 2-D heatmap with a green-red colour ramp is the design choice that *made* the benchmark. Here is why.

A scalar score — "Claude 2.1 retrieved the needle 27% of the time" — is dry. A line plot — "accuracy vs context length" — is a familiar machine-learning chart that no AI executive would have any reason to share. But a 2-D heatmap with green and red regions is *immediately legible to a non-technical audience*. It looks like a thermal image of a brain. It looks like a battery health diagnostic. It looks, fundamentally, like *a measurement of a physical system*. And critically, it has a built-in **storytelling axis**: you can point at the red corner and say *"this is where the model breaks."*

A 2023 launch deck could include this image and the slide wrote itself: *"Our model maintains 100% recall through 200K context except in a narrow band near the middle."* The visual gave a 1-D capability (does the model find the sentence?) the *visual rhetoric* of a 2-D measurement (the field's first long-context dashboard).

For 14 months in 2023–2024, the visual rhetoric ran ahead of the underlying measurement validity. We have a primer dedicated to that gap: [Goodhart's Ceiling](../06-measurement-saturation/).

## The December Footnote That Should Have Killed The Benchmark

We have already, in the cold open, told the story of Anthropic's **December 6, 2023** response post. Let's go a layer deeper.

The post is titled *"Long context prompting for Claude 2.1"*. The author byline is anonymous — "Anthropic". The structure is unusual for an AI lab blog: it opens with an admission, walks through a diagnosis, and closes with a tactical workaround. The admission is buried in paragraph three:

> *"Claude 2.1 is trained on a mix of data aimed at reducing inaccuracies. This includes not answering a question based on a document if it doesn't contain enough information to justify that answer… As a result, we believe Claude 2.1 is much more reluctant to answer when a sentence seems out of place in a longer context. **This particular cause of increased reluctance wasn't captured by evaluations targeted at real-world long context retrieval tasks**."*

That bolded sentence is the December 2023 admission that NIAH did not measure what people thought it measured. Anthropic was, in effect, conceding that **their model's *good* alignment behavior — refusing to answer when context appears tampered with — was being scored as a *bad* benchmark result.** A model trained to be cautious about adversarial-looking text scored low on a benchmark whose entire premise was inserting adversarial-looking text.

The tactical workaround, the *"add this magic incantation to the prompt"* fix, made the embarrassing number go away. The deeper observation — that **the entire benchmark was load-bearing on a prompting convention** — was politely allowed to sit in the second-to-last paragraph without elaboration. Almost nobody talked about that observation for a year.

### What The 10 Words Actually Do

Let us strip the trick to its essence. Kamradt's standard NIAH prompt has the structure:

```
{HAYSTACK including injected needle somewhere}

{QUESTION about the needle's topic}
```

Anthropic's December workaround prepends a one-line instruction immediately *before* the question:

```
{HAYSTACK including injected needle somewhere}

Here is the most relevant sentence in the context:

{QUESTION about the needle's topic}
```

What does the magic line tell the model? Mechanically, two things:

1. **The needle exists.** Without the line, the model is, on Anthropic's training distribution, performing a *judgment call* about whether the haystack contains an answer at all. The line forecloses the judgment. *Of course* there is a relevant sentence — go find it.
2. **Format your output to extract one.** "Here is..." is a sentence-fragment that demands completion. The model now has to *produce* a sentence-shaped object lifted from the prompt, not summarise or reason about it.

Both are *prompt-engineering tricks*. Neither is a measurement of the underlying model's long-context capability. Yet the benchmark score swings from **27% to 98%** — a 71-percentage-point delta — depending on which version you ran.

```pyplot {id="prompt-decomposition" caption="A schematic decomposition of the 'magic prompt' fix. Each of the two pieces — 'the needle exists' + 'lift a sentence' — closes one degree of freedom in how the model interprets the task. Together they convert a judgment-laden answer task into a verbatim-extraction task."}
fig, ax = plt.subplots(figsize=(8.5, 4.5))

# Stacked bar showing the contribution of each prompt piece (illustrative)
pieces = ['Vanilla\nNIAH prompt',
          '+ "Here is the most\nrelevant sentence..."',
          '+ Both pieces\n(Anthropic Dec 2023)']
contribution_baseline = [27, 27, 27]
contribution_exists   = [0, 40, 40]
contribution_lift     = [0,  0, 31]

x = np.arange(3)
ax.bar(x, contribution_baseline, color='#FF007F',
       edgecolor='#1A1A1A', linewidth=1.5, label='baseline (model recall)')
ax.bar(x, contribution_exists, bottom=contribution_baseline, color='#FFD700',
       edgecolor='#1A1A1A', linewidth=1.5,
       label='+ "needle exists" framing')
ax.bar(x, contribution_lift,
       bottom=np.array(contribution_baseline) + np.array(contribution_exists),
       color='#00A8A8', edgecolor='#1A1A1A', linewidth=1.5,
       label='+ "lift a sentence" formatting')

for i, total in enumerate(np.array(contribution_baseline) + np.array(contribution_exists) + np.array(contribution_lift)):
    ax.text(i, total + 2, f"{total}%", ha='center', fontweight='bold', fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(pieces, fontsize=9)
ax.set_ylim(0, 110)
ax.set_ylabel('reported NIAH accuracy (Claude 2.1)')
ax.set_title('Claude 2.1 NIAH score decomposed by what the prompt tells the model')
ax.legend(loc='upper left', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(99, color='#1A1A1A', linewidth=0.5, linestyle=':')
ax.text(0.5, 99.5, 'Claude 3 Opus, Mar 2024 (99%)', fontsize=8, alpha=0.6)
```

This is the **load-bearing observation** of the entire long-context benchmark literature. The single most important thing to notice about NIAH circa 2023 is not how high or low any specific model scored. It is that *small changes to the user-facing prompt produced enormous changes to the benchmark score*. The capability layer (what the model could in principle do) was being measured through a thick lens of *capability elicitation* (whether the user's prompt invited the model to actually do it).

We will see in [Scan vs Think](../04-retrieval-vs-reasoning/) why this gap exists at the architecture level. We will see in [The Chroma Measurement](../12-context-rot/) how the same gap reappears in 2025 under a different name.

## What NIAH Actually Measures (Strictly)

After all that — what *is* NIAH, in 2026, considered to measure?

The honest answer: **NIAH measures a single capability** — *can the model perform exact retrieval of a syntactically-alien sentence inserted at a known depth inside a stylistically-uniform haystack, given a prompt format that conventionally elicits sentence-extraction behaviour?*

That is a real capability. It is also a *narrow* capability. It is to long-context reasoning what *spelling test* is to *literacy*. A model that can find Kamradt's sandwich sentence in 128K tokens has demonstrated something — but the something is much smaller than "the model understands long context." Specifically, NIAH does **not** measure:

- **Multi-hop reasoning** — chasing co-references across multiple positions in the context. (That's MRCR and GraphWalks. See [Vodrahalli's Chisel](../09-latent-structure/).)
- **Aggregation** — counting, summing, listing, deduplicating items scattered across the context. (That's RULER's aggregation tasks. See [When Everyone Scored 99](../05-saturation/).)
- **Semantic-only retrieval** — finding information without surface-token overlap with the question. (That's NoLiMa, an explicit attack on NIAH's surface-matching nature. See [The Chroma Measurement](../12-context-rot/).)
- **Absence detection** — recognising when the answer is *not* in the context. (That's IDK from Michelangelo and AbsenceBench. See [Vodrahalli's Chisel](../09-latent-structure/).)
- **Long-form generation under constraints** — producing coherent long answers that respect many embedded constraints. (That's LongGenBench.)
- **Position-aware reasoning** — knowing not just *what* was said but *when*, *by whom*, *in what order*. (That's most of what real users actually want.)

Every one of those capabilities was *eventually* matched with a benchmark of its own. The story of how, between roughly April 2024 and the end of that year, the field went from one benchmark (NIAH) to fifteen-plus is the subject of [When Everyone Scored 99](../05-saturation/).

## The Quiet Defenders Of NIAH

A fair primer would not close without giving NIAH its due. The benchmark was not stupid, and Kamradt was not naive. Three points of nuance worth holding:

1. **As a smoke test, NIAH is still useful in 2026.** Even now, when a new model is released, a NIAH heatmap is a quick sanity check that the model has not catastrophically broken at long context. A model that *fails* vanilla NIAH at its advertised window is broken in a way the marketing team needs to know about, immediately. The community keeps running it for this reason.

2. **The visual format catalysed an industry.** Whatever NIAH did or did not measure rigorously, the heatmap was the *first* shared long-context visualization. It made the topic legible to product managers and investors. The benchmark industry that followed — RULER, ∞Bench, BABILong, NoCha, HELMET, LongBench v2 — would not have been *funded* without the cultural ground NIAH laid in late 2023.

3. **The critique was internal to the field from day one.** Anthropic's December 2023 footnote, RULER's April 2024 abstract, Michelangelo's September 2024 reframe — every major lab and every credible benchmark group acknowledged NIAH's limits publicly, often the same year they ran it. The misuse of NIAH as a marketing headline number was a *practitioner culture* failure, not a *researcher culture* failure. The papers were honest. The press releases were not.

## What To Remember

1. **NIAH is the test where a foreign sentence is buried in familiar text and the model is asked to retrieve it.** The original test used Paul Graham essays and a sandwich-Dolores-Park sentence, both choices that make the needle stylistically detectable as out-of-distribution.
2. **The heatmap visualization was load-bearing.** A 2-D green/red image made a narrow 1-D capability *feel* like a 2-D measurement of long-context behavior. It launched the long-context benchmark era.
3. **Adding 10 words to the prompt jumps Claude 2.1's NIAH score from 27% to 98%.** This is the most important fact about NIAH. The capability layer is being measured through a thick lens of capability elicitation; small prompt changes produce huge score changes.
4. **NIAH measures exact-string retrieval under a sentence-extraction prompt convention.** It does not measure multi-hop reasoning, aggregation, semantic retrieval without token overlap, absence detection, or long-form generation. Each of those needed its own benchmark, and the rest of this issue is about them.
5. **The benchmark wasn't dishonest. The reporting culture was.** Anthropic, NVIDIA, Google DeepMind, and Princeton all published the critiques openly. Frontier-lab marketing materials picked the numbers that flattered the latest release. The gap is something to remember next time you read "our model scores 99% on the long-context benchmark."

**Continue to** → [Scan vs Think](../04-retrieval-vs-reasoning/) — the deeper architectural reason a transformer can find a sentence in 200K tokens with a single forward pass, and the deeper reason that *finding the sentence* is not the same as *thinking about what the haystack contained*.
