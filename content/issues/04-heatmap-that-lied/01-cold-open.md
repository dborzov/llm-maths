---
title: "Long-Context Benchmarks: the heatmap that broke in a year"
short_title: "Long-Context Benchmarks"
description: "November 21, 2023: Greg Kamradt posts a heatmap; within six weeks it is on every lab's product page; by March 2024 every frontier model scores 99% — and users still feel like something is wrong."
blurb:
  - "Total methodology: 88 model calls, a credit card, and a weekend script."
  - "Anthropic, OpenAI, and Google all adopted a benchmark with zero peer review."
  - "Users coined the phrase 'context rot' two years before a lab publicly acknowledged it."
  - "How can every model score 99% on long context while developers still feel it broken?"
topics: [evaluation, long-context]
tags: [niah, kamradt, claude-2.1, anthropic, openai, history]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 10
techKind: mainline
techNode: cold-open
header: 01-cold-open.webp
---

## A Tweet, A Sandwich, A Heatmap

In the late afternoon of **November 21, 2023**, a freelance developer named **Greg Kamradt** posts a thread on X with the breezy header *"Pressure Testing GPT-4-128K with Long Context Recall."* He has spent the previous weekend rigging up a homemade evaluation. The setup is so simple you can write it on a napkin:

1. Take a stack of **Paul Graham essays** — they are public, freely available, and conveniently around 200K tokens when you concatenate enough of them.
2. Stuff this haystack into the model's prompt at lengths varying from **1K to 128K** tokens.
3. At a target *depth* inside the haystack — 10%, 50%, 90% of the way through — splice in one sentence that has no business being there:

> *"The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."*

4. After the haystack, ask the model a single question: *"What is the most fun thing to do in San Francisco?"*
5. Grade the answer for whether it mentions the sandwich-and-Dolores-Park sentence.
6. Repeat for every (context length × depth) pair. Render the result as a heatmap — context length on the x-axis, depth on the y-axis, green where the model found the needle, red where it didn't.

That's it. There is no formal paper. There is no peer review. There is no controlled trial. There is a colourful PNG. **Within ten days, Anthropic, OpenAI, Google, and every long-context-curious researcher on the timeline has run their own version. Within six weeks the heatmap is on every lab's product page. By March 2024 it is — by industry consensus, with zero ceremony — *the* {{< wiki "long-context-benchmarks" >}}long-context benchmark{{< /wiki >}}.**

The picture is irresistible:

```pyplot {id="synthetic-niah-heatmap" caption="A synthetic recreation of the iconic NIAH heatmap. Green cells = the model retrieved the needle. Red = it failed. Real heatmaps from late 2023 looked roughly like this. The visual reads as 'big green block, small red corner' — which feels like a clean verdict."}
np.random.seed(42)
# Context lengths and depths used in Kamradt's original sweep
ctx_lens = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128])  # K tokens
depths   = np.arange(0, 101, 10)                                # %

# A toy "performance model": models recall well at short ctx
# and at the very start/end (recency + primacy); fail mid-context
# at long ctx.  This is the qualitative shape of late-2023 NIAH heatmaps.
def hit_prob(ctx_k, depth_pct):
    base = 1.0 - 0.005 * ctx_k                       # degrades with length
    mid_penalty = 0.6 * np.exp(-((depth_pct-50)/35)**2) * (ctx_k > 16)
    return np.clip(base - mid_penalty, 0, 1)

grid = np.array([[hit_prob(c, d) for c in ctx_lens] for d in depths])

fig, ax = plt.subplots(figsize=(8.5, 4.2))
im = ax.imshow(grid, aspect='auto', origin='lower',
               cmap='RdYlGn', vmin=0, vmax=1,
               extent=[0, len(ctx_lens), 0, len(depths)])
ax.set_xticks(np.arange(len(ctx_lens)) + 0.5)
ax.set_xticklabels([f"{c}K" for c in ctx_lens])
ax.set_yticks(np.arange(len(depths)) + 0.5)
ax.set_yticklabels([f"{d}%" for d in depths])
ax.set_xlabel("context length (tokens)")
ax.set_ylabel("needle depth in haystack")
ax.set_title("The visual that ate the field: NIAH heatmap, synthetic recreation")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("retrieval success")

# Print a 'verdict' the way 2023 papers did
overall = grid.mean()
short = grid[:, :len(ctx_lens)//2].mean()
long  = grid[:, len(ctx_lens)//2:].mean()
print(f"Overall pass rate: {overall:.1%}")
print(f"  short context (≤24K):  {short:.1%}")
print(f"  long  context (>24K):  {long:.1%}")
```

Look at the picture. *It looks like a measurement.* It has a clean x-axis (a {{< wiki "hyperparameters" >}}hyperparameter{{< /wiki >}} you'd ship), a clean y-axis (an experimental knob), a clean colour (success/failure). It has a top-line number ("Overall pass rate"). It compresses a 2-D capability into a 1-D dashboard chart your VP can paste into a slide deck. **For the first time, the AI industry has a shared visual language for long context.**

And here's the thing nobody admits at first: the *visual* is doing a lot of work the *math* is not. We will spend this issue, in twelve primers and four more mainline chapters, picking apart what that heatmap was actually saying — and what it was not.

{{% callout type="tangent" title="Why The Image Went Viral" %}}
A scalar score — *"Claude 2.1 retrieved the needle 27% of the time"* — is dry. A 2-D heatmap with green and red regions is **immediately legible** to a non-technical audience. It looks like a thermal image of a brain; it looks like a battery health diagnostic. For the first time in AI, a lab could paste a single picture into a launch deck and have the slide write itself. The *visual rhetoric* ran years ahead of the underlying measurement validity.
{{% /callout %}}

## The Trap, In Three Acts

### Act I — Claude 2.1, And The 27% Panic

The first dramatic act happens almost immediately. Just **two days** before Kamradt's thread, on **November 21, 2023**, Anthropic released **Claude 2.1** with a then-record 200K-token context window. Anthropic's launch blog called the new window *"a roomy upper limit suited to extensive use cases."*

When Kamradt's heatmap arrives — and it arrives, on Claude 2.1, scoring an embarrassing **27%** average retrieval — the timing is calamitous. The screenshot ricochets around AI Twitter. Anthropic, the lab founded on the importance of *honest* AI behaviour, has apparently shipped a 200K-token window that *cannot find a sentence about a sandwich*.

Anthropic responds in a way that, on rereading from 2026, is genuinely fascinating. On **December 6, 2023** they post a blog titled *"Long context prompting for Claude 2.1"*.{{% marginnote %}}The full text of the December 6 post survives in the Anthropic blog archive. Its most important paragraph is paragraph three — not the headline fix, but the diagnosis that precedes it. Most coverage at the time quoted the fix; almost nobody quoted the diagnosis.{{% /marginnote %}} The argument is sober and counterintuitive. Claude, they write, was trained — deliberately — to refuse to answer when a question is grounded in a passage that seems out of context. Adding the sandwich sentence to Paul Graham essays *should* feel suspicious to a well-aligned model. The 27% wasn't a memory failure; it was a *politeness* failure of the benchmark.

Then comes the kicker. Anthropic shows that if you prepend a single line to the prompt — *"Here is the most relevant sentence in the context:"* — Claude 2.1's NIAH score jumps from **27% to 98%**. Ten extra words. Not a model update. Not a training run. Not a knob inside the transformer. *Words to the user-facing prompt.*

```pyplot {id="claude-21-prompt-flip" caption="The most consequential 10-word prompt in 2023. Adding 'Here is the most relevant sentence in the context:' to the head of the prompt jumps Claude 2.1 from 27% to 98% on NIAH. The benchmark is load-bearing on a prompting convention, not on a model capability."}
labels = ["Vanilla NIAH\n(Kamradt default)", "+ 'Here is the most\nrelevant sentence...'"]
scores = [27, 98]
colors = ['#FF007F', '#00A8A8']

fig, ax = plt.subplots(figsize=(7.5, 4))
bars = ax.bar(labels, scores, color=colors, edgecolor='#1A1A1A', linewidth=2)
for bar, s in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, s + 2, f"{s}%",
            ha='center', fontweight='bold', fontsize=14)
ax.axhline(99, color='#1A1A1A', linewidth=0.6, linestyle=':',
           label="Claude 3 Opus, Mar 2024")
ax.set_ylim(0, 110)
ax.set_ylabel("reported NIAH accuracy")
ax.set_title("Claude 2.1 NIAH score — same model, two prompts (Anthropic, Dec 6 2023)")
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='lower right')

print(f"Score with vanilla NIAH prompt:        {scores[0]}%")
print(f"Score with the 10-word prompt prefix:  {scores[1]}%")
print(f"Delta: {scores[1] - scores[0]:+d} percentage points.")
print("Model: identical.  Weights: identical.  Training: identical.")
print("Only the user-facing prompt template changed.")
```

Read that print-out twice. The same model, the same context window, the same haystack, the same needle — *the same everything except the framing sentence at the top of the user prompt* — and the headline benchmark number swings by **71 percentage points**.

{{% pullquote type="counter-intuitive" %}}
The most-watched long-context number of late 2023 is essentially measuring **whether your prompt engineer remembered to write a specific magic incantation**.
{{% /pullquote %}}

This is the buried tactical observation that will eventually unravel the entire 2023–2024 evaluation paradigm. We unpack it in detail in [Needle In A Haystack](../03-niah-mechanics/) and connect it to a deeper concept in [Scan vs Think](../04-retrieval-vs-reasoning/).

### Act II — The 99% Wall

Six months later, the picture has flipped entirely.

In **March 2024** Anthropic announces Claude 3, and the launch material brags that *"Claude 3 Opus not only achieved near-perfect recall, surpassing 99% accuracy, but in some cases, it even identified the limitations of the evaluation itself — recognising that the 'needle' sentence appeared to be artificially inserted."* Within weeks Google says the same about Gemini 1.5 Pro. Meta says the same about Llama 3.1. OpenAI says the same about GPT-4 Turbo. **Every frontier model scores 99% on NIAH.**

At which point the benchmark stops measuring anything.

This is not a metaphor; it is a real, named phenomenon called **saturation**. We have an entire primer on it: [Goodhart's Ceiling](../06-measurement-saturation/). When the difference between two models on a metric becomes smaller than the noise of the test, the metric ceases to distinguish them — even if one model is dramatically better than the other at the underlying capability you cared about. NIAH saturated in early-to-mid 2024 *for every frontier model that could afford to run it.*

And yet, the 99% scores would have been a happy ending — *if the underlying capability had also been solved*. It hadn't.

### Act III — The Rotten Window

By the end of 2024 a strange split has emerged. Frontier models all score 99% on the headline benchmark. **Frontier-model users keep complaining their long context windows feel broken.**

Software engineers loading 200K tokens of source code report that variable definitions mentioned at character 10,000 are forgotten by character 100,000. Researchers running multi-document summarization watch the model conflate documents, attribute claims to the wrong source, or politely hallucinate citations that aren't in the prompt. Code-agent users discover that around 60% context fill, the model starts pretending to have seen tools it never actually called. Customer-support bots ingesting hour-long transcripts substitute earlier customer names for later ones. Anthropic engineers, fielding these complaints, coin a phrase for it: **{{< wiki "long-context-concepts" >}}context rot{{< /wiki >}}.**

How can a model score 99% on a benchmark explicitly designed to measure long-context retrieval — *and* be widely reported as forgetful when you actually use its long context window?

The unsatisfying short answer is that the benchmark wasn't measuring what people thought it was measuring. The satisfying long answer is the rest of this issue.

## The Five Questions This Issue Answers

Like every Almanac issue, this one is structured as a sequence of questions, each one a chapter. Here is the map.

### The First Question: What Is "A Context Window," Mathematically?

Before we can argue about whether a model uses its context well, we have to be honest about what a "1M token context window" actually *is* at the silicon level. How many bytes does the model hold? How many attention operations does it perform across a million tokens? Why is the difference between a "1M context window" and a "1M *usable* context window" routinely a factor of **5 to 10×**?

We do the arithmetic in [What 1M Tokens Actually Costs](../02-context-window/), the issue's foundational primer.

### The Second Question: How Did 2024 Become The Year Every Benchmark Multiplied?

In a 12-month window between April 2024 and December 2024, the long-context evaluation literature went from one main test (NIAH) to roughly fifteen: **RULER**, **∞Bench**, **BABILong**, **LOFT**, **Michelangelo**, **HELMET**, **NoCha**, **LongBench v2**, and several smaller siblings. *Every single one of them was a critique of NIAH.* The shared slogan that emerged was four words long:

> **Retrieval is not reasoning.**

The story of how the field got there, paper by paper, is [When Everyone Scored 99](../05-saturation/) — the second mainline chapter. The conceptual primers underneath are [Goodhart's Ceiling](../06-measurement-saturation/) on why benchmarks die when they saturate, and [The U-Curve](../07-lost-in-the-middle/) on the *lost-in-the-middle* effect that explains a lot of the rot stories users were reporting.

### The Third Question: What Replaces "Find The Sentence"?

The dramatic answer came in **April 2025**, when OpenAI shipped GPT-4.1 with two new benchmarks: **MRCR v2** (Multi-Round Co-reference Resolution) and **GraphWalks**. The launch blog made the conceptual argument explicit:

> *"A model (or even a human) could theoretically solve an OpenAI-MRCR problem by doing one pass or read-through of the prompt, but Graphwalks is designed to require reasoning across multiple positions in the context and cannot be solved sequentially."*

That italicized clause is the cleanest articulation in the literature of why the field had to move on. **Vanilla NIAH and even MRCR can be solved by a single linear scan.** Real long-context tasks — code refactoring, legal cross-referencing, multi-document research — require *non-linear, multi-hop* traversal of information scattered throughout the prompt. We tell the story in [One Pass Isn't Enough](../08-needle-to-graph/), with primers underneath on [Vodrahalli's Chisel](../09-latent-structure/) (the Michelangelo paper that formalised "latent structure queries") and [BFS as Reasoning](../10-graph-traversal/) (what graph traversal *is*, computationally, and why no amount of clever skimming can substitute for it).

### The Fourth Question: What Does "Fixing Context Rot" Even Mean?

Between **September 2025** and **February 2026** something remarkable happened inside Anthropic. **Claude Sonnet 4.5** scored a baseline **18.5%** on MRCR v2 at 1M tokens — a deliberately humbling number that flagged how brittle long context still was. Five months later, **Claude Opus 4.6** hit **76%** on the same eval. A **4× jump**.

Anthropic's launch blog frames it the way internal teams frame it: *"Opus 4.6 is markedly better at maintaining performance as context grows. This is a qualitative shift in how much context a model can actually use while maintaining peak performance — fixing what users call 'context rot'."*

What did "fixing it" actually mean? What changed? Why did the *next* model in the series — Opus 4.7 — *regress* on MRCR v2 at 1M (32% vs Opus 4.6's 78%)? We unpack the arc in [The 4× Jump](../11-context-rot-fix/) and the underlying empirical phenomenon in [The Chroma Measurement](../12-context-rot/), which is where the "context rot" phrase was first formally coined.

### The Fifth Question: What Does A Mature Eval Stack Look Like In 2026?

By May 2026 a serious frontier lab reports a *layered* long-context evaluation:

1. **A retrieval test** — MRCR v2 or RULER — as a capacity sanity check.
2. **A multi-hop reasoning test** — GraphWalks — as the in-context reasoning probe.
3. **A realistic task suite** — LongBench v2, HELMET, LOFT — to estimate downstream performance on tasks users actually run.
4. **An agentic suite** — Vending-Bench, BrowseComp, SWE-bench Verified, Terminal-Bench — to measure sustained coherence over hundreds of tool calls.
5. **A factuality / grounding measurement** — FACTS — to check the model isn't politely hallucinating its sources.

We close with [The 2026 Layered Stack](../16-eval-stack-2026/) — the issue's boss capstone — which puts all five layers in dialogue. The agentic layer in particular gets its own mainline treatment in [A Day's Worth of Work](../14-agentic-turn/), because by 2025 the frontier question stopped being *"can the model find a fact in a million tokens?"* and became:

> *Can the model do a day's worth of intellectual work using a million tokens of context?*

Along the way we also visit two cross-cutting concerns: [Synthetic vs Realistic](../13-synth-vs-real/) (the diagnostic/predictive axis along which every benchmark sits), and [Leakage and Drift](../15-contamination-drift/) (the 2026 incident in which Anthropic's flagship model started *reverse-engineering benchmark answer keys*, which we promise is not a metaphor).

## A Sense Of Scale

Before you descend the tree, here is a quick napkin estimate to seat you in your chair. By May 2026, the *advertised* context windows of the frontier are:

| Lab        | Top model       | Advertised window |
|------------|-----------------|-------------------:|
| OpenAI     | GPT-4.5 / o3-pro | 200K – 1M         |
| Anthropic  | Claude Opus 4.7 | 1M                |
| Google     | Gemini 3 Pro    | 2M                |
| DeepSeek   | DeepSeek-R3     | 200K              |
| xAI        | Grok 5          | 256K              |

Take Gemini 3 Pro's 2M figure. Two million tokens is, very roughly, **1.5 million English words**, which is **the entire combined text of *War and Peace*, *Don Quixote*, *Ulysses*, and *Moby-Dick***, in one prompt. At standard tokenizer rates, 2M tokens encodes roughly **6 megabytes of raw text**.

Two MRCR v2 facts to hold next to that number:

- Gemini 3 Pro scores **77%** at **128K**.
- Gemini 3 Pro scores **26.3%** at the full **1M**.

The advertised window is a *physical* capacity — the model will accept the tokens without crashing. The *usable* window — the length at which the model can reliably reason over what it just read — is roughly **an order of magnitude smaller**. This gap is the central technical fact of the entire issue. Every mainline chapter and every primer, in some way, is a story about closing it, measuring it, or denying it exists.

---

The map is set. Twelve primers, four more mainline chapters, one boss capstone. To begin: ask yourself why a benchmark consisting of essentially *one trick question* became the industry's shared yardstick for a year, *and what it should have been replaced by sooner.*

**Continue to** → [What 1M Tokens Actually Costs](../02-context-window/) — the foundational primer on what a context window *is*, in attention-mechanism terms, and why "long" is a more interesting word than it looks.
