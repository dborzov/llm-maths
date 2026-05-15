---
title: "The U-Curve"
description: "Liu et al., July 2023: plot retrieval accuracy against the position of the answer inside a long prompt. The shape that comes out is a smile. The model is best at the start, best at the end, worst in the middle — and the curve is everywhere."
topics: [evaluation, long-context, attention]
tags: [lost-in-the-middle, position, liu-2023, u-curve, primacy, recency]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 70
techKind: primer
techNode: lost-in-the-middle
header: default.webp
---

## July 2023, Two Months Before NIAH

While Greg Kamradt is, in late 2023, building the benchmark that will dominate the next year, a Stanford group is publishing the *empirical finding* that will dominate the year after that.

The paper is **"Lost in the Middle: How Language Models Use Long Contexts"** by **Nelson F. Liu, Kevin Lin, John Hewitt, and colleagues** (arXiv:2307.03172, July 6, 2023). Their experiment is so clean it could fit on a postcard. Take 20 Wikipedia documents. Concatenate them into a single long prompt. Hide a "gold" document — the one that actually contains the answer to a question — at *position* 1, 5, 10, 15, or 20 in the sequence. Ask the model the question. Score whether it gets the answer right. Repeat across many questions. Plot accuracy as a function of *the position the gold document was in*.

The plot that comes out is a **smile**. Models score highest when the gold document is **at the start** (~75% on GPT-3.5-turbo at 20 docs). They score nearly as high when it is **at the end** (~63%). They score *catastrophically* when it is **in the middle** (~52%). The dip in the middle is *larger* than the dip you get if you simply remove the gold document and ask the same questions of the model from its prior knowledge alone (52% vs 56% closed-book).

{{% pullquote type="counter-intuitive" %}}
Putting the right answer in the **middle** of a long prompt was *worse* than not putting it in the prompt at all. The surrounding context distracted the model so badly that its parametric memory outperformed its in-context retrieval.
{{% /pullquote %}}

This is the **U-curve**. It is the single most reproduced finding in long-context literature, and it explains a lot of what users feel as "{{< wiki "long-context-concepts" >}}context rot{{< /wiki >}}" without needing any new vocabulary.

## The Picture, Carefully

Let's draw the canonical U.

```pyplot {id="canonical-u-curve" caption="A toy U-curve emulating the Liu et al. finding. Highest accuracy when the relevant information sits at the start or end of the prompt; a deep dip in the middle. The shape is robust across model families, prompt formats, and task types — but the depth of the dip and its precise centre move with each variable."}
positions = np.linspace(0, 1, 21)        # 0 = start, 1 = end

# Stylised U: high at edges, low at middle, with slight start bias (primacy)
def u_curve(pos, primacy=0.78, recency=0.66, valley=0.49,
            center=0.5, sharpness=4.5):
    # mix of primacy (left peak) + recency (right peak) + valley
    left_peak  = primacy  * np.exp(-((pos - 0.0) * sharpness)**2)
    right_peak = recency  * np.exp(-((pos - 1.0) * sharpness)**2)
    floor      = valley * np.ones_like(pos)
    return np.maximum.reduce([left_peak, right_peak, floor])

models = {
    "GPT-3.5 Turbo @ 20 docs":  u_curve(positions, 0.78, 0.66, 0.49, sharpness=4.0),
    "Claude-1.3 @ 20 docs":     u_curve(positions, 0.74, 0.69, 0.55, sharpness=3.5),
    "Llama-2 70B @ 20 docs":    u_curve(positions, 0.62, 0.55, 0.39, sharpness=3.0),
}
colors = ['#FF007F', '#00A8A8', '#FFD700']

fig, ax = plt.subplots(figsize=(8.5, 4.2))
for (name, y), c in zip(models.items(), colors):
    ax.plot(positions, y, marker='o', linewidth=2, color=c, label=name,
            markeredgecolor='#1A1A1A', markersize=6)
# closed-book baseline
ax.axhline(0.56, color='#1A1A1A', linewidth=0.6, linestyle=':',
           label='GPT-3.5 closed-book (no docs)')

ax.set_xlabel("position of gold document in prompt (0 = start, 1 = end)")
ax.set_ylabel("answer accuracy")
ax.set_title("'Lost in the Middle' — the canonical U-curve (Liu et al., 2023)")
ax.set_ylim(0.3, 0.9)
ax.legend(loc='lower center', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# annotate the valley
for name, y in models.items():
    idx = np.argmin(y)
    if 'GPT-3.5' in name:
        ax.annotate("the valley\n(answer here ≈ no answer)",
                    xy=(positions[idx], y[idx]),
                    xytext=(0.5, 0.32), ha='center', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#1A1A1A'))

print("Shape parameters (canonical):")
print("  primacy (left peak):  ~75%  — strong, often the global maximum")
print("  recency (right peak): ~65%  — slightly weaker")
print("  valley (middle):      ~50%  — at or below closed-book baseline")
print("This U is robust across model family, prompt format, and task.")
```

Three things to notice in that plot.

First, **the U is asymmetric**. The start of the prompt is recalled *better* than the end — by roughly 10 percentage points in the original Liu et al. data. This is called **primacy bias**, and it shows up across nearly every model family. The middle is worse than both edges, but the slope from edge to valley is *steeper* coming from the right (recency side) than from the left (primacy side).

Second, **the valley is around the closed-book baseline**. If you simply asked the model the same questions *without* any context, GPT-3.5-turbo answered correctly ~56% of the time from parametric knowledge alone. The middle-of-prompt accuracy is *at* or *below* this baseline. The retrieved context, when placed badly, **actively misleads** the model.

Third, **the shape is qualitatively robust** but the *parameters* vary. Better models have shallower valleys. Larger context windows widen the valley. Different prompt formats shift the peaks. The U is the *family* of curves; each specific (model, task, prompt) combination is one member of the family.

## Why Does It Happen? Three Explanations, All Probably Partly Right

The "lost in the middle" effect has been studied to death since 2023. No single explanation accounts for all the data. The most honest summary is that there are at least three contributing causes, each operating at different scales.

### 1. The Training-Distribution Bias

In the data the model was *trained* on — long-form text, web pages, books, code — the load-bearing information is overwhelmingly concentrated at the **start** (titles, abstracts, function signatures, opening paragraphs) and the **end** (conclusions, return values, summaries, totals). Middle-of-document material tends to be developmental, exploratory, or transitional. Models that have learned to attend to "where information lives in real text" have learned to *under-weight* the middle.

This is the dominant explanation for the asymmetry (start > end). Code, especially, has primacy-biased information structure: imports, then types, then function bodies. Models that ingest a lot of code learn to look up the top of a document first.

### 2. The RoPE-Extension Aliasing

We saw in [What 1M Tokens Actually Costs](../02-context-window/) that **{{< wiki "rope" >}}RoPE{{< /wiki >}}** encodes position by rotating query/key vectors by angles proportional to position. RoPE has a *wavelength range* — the slowest-rotating dimensions cycle every ~10,000 positions, the fastest every position. Extend past the longest wavelength the model was trained on and positions in different "octaves" start to alias.

Middle positions, in a long prompt, are *exactly* the regime where RoPE extension techniques (NTK-aware, YaRN, ABF) are doing the most interpolation work — and where they generate the most positional confusion. Empirically, the dip in the U-curve *deepens* as context length increases, which is what you'd expect from a RoPE-aliasing explanation. We will revisit this in [The Chroma Measurement](../12-context-rot/), where the empirical evidence is fleshed out across 18 models.

### 3. The Attention-Dilution Argument

Recall from [Scan vs Think](../04-retrieval-vs-reasoning/) that {{< wiki "softmax" >}}softmax{{< /wiki >}} {{< wiki "attention" >}}attention{{< /wiki >}} has to *normalise* across all keys. With more keys, the probability mass per key, on average, gets smaller. A key in the *middle* of the prompt is competing with $n - 1$ distractors, and the "right" attention weight has to win a normalisation contest against all of them.

But this should apply uniformly to all positions, not preferentially to the middle. The reason it manifests as a *U* and not a *flat line* is that the model has *learned* to amplify edge tokens — via training-distribution bias (point 1) — but has not learned, or has not been *able* to learn given its position-encoding scheme, to amplify the middle.

These three explanations are **complementary**, not competing. Each accounts for a piece of the shape. The training bias explains the asymmetry. RoPE aliasing explains why the dip deepens with length. Attention dilution explains why edges win at all.

## The U Is Everywhere

A short tour of where the U-curve shows up in the wild, post-2023:

- **{{< wiki "long-context-benchmarks" >}}NIAH{{< /wiki >}} heatmaps** (2023–2024) show *exactly* this pattern in the depth dimension. The red band that became the visual signature of NIAH is the middle of the U, plotted as a colour.
- **MRCR** (2024) finds that the *fourth* of eight tapir poems is harder to retrieve than the *first* or the *eighth*. Same U.
- **NoCha** (2024) finds that book passages quoted in the middle of the book are less reliably verified than those in the first chapter or the last.
- **Chroma's "Context Rot" study** (2025; Hong et al.) finds that the U-curve's *centre* shifts with content type — pure-text prompts have the valley around 50%; code-heavy prompts have it closer to 60%. The U is real but its *centre* is not exactly the geometric middle.
- **LongMemEval** (2025) finds the same U on chat-history retrieval — middle-of-conversation references are recalled less reliably than openings or recent messages.

The U-curve is, in short, *not a quirk of the Liu et al. experimental setup*. It is a fundamental fact about how long-context attention allocates probability across positions, and it shows up in every test that varies the position of a key piece of information.

## Why The Edges Matter, Operationally

{{% callout type="tip" title="Defeating the U-Curve in Practice" %}}
The U-curve is a training-distribution artifact — you can't fix it by changing the model. You can route around it:

- **Put the critical instruction at the very end** of the prompt. Recency bias is your friend.
- **Restate key facts at the end** if they appear in the middle of a long document.
- **Use structured headers** (`## Section`, `def function_name():`) — models trained on code and web text have learned to attend to these as anchors.
- **Split very long contexts** into two passes with an explicit hand-off if the information you need is buried past the 50% mark.

None of these require changing the model. All of them cost tokens. That's the deal.
{{% /callout %}}

Here is the practical observation for any developer using a long-context model. **The position of your most important information inside the prompt matters more than the size of the prompt.**

Concretely, two prompts containing the same 200K tokens of context — one with the critical instruction at character 80,000 (middle), one with the critical instruction at character 199,500 (end) — can produce dramatically different model behaviour. The model is the same. The information is the same. The position is different.

This is the operational mechanism behind a lot of what users call "context rot". It is not that the model *forgot* the middle (which would be a memory failure); it is that the model is *systematically under-weighting* the middle (which is an attention-allocation behaviour learned during training and amplified by positional encoding artifacts).

The fix, when one is available, is *prompt structure*:

- **Put the critical instruction at the end.** The last thing the model attends to, before generating, is the most recent context. Use recency bias deliberately.
- **Restate critical context at the end.** If you need a long document to *also* be in the model's middle, duplicate the key parts in a closing summary.
- **Use structured headers.** Models trained on web text and code have learned to attend to *labels* (`## Section`, `def function():`, `<title>`). Headers create attention anchors that partly defeat the U-curve.
- **Split very long contexts.** A 200K-token prompt with a U-curve is sometimes worse than two 100K-token prompts processed sequentially with explicit hand-off.

```pyplot {id="u-curve-fixes" caption="What 'fixing the U' looks like in practice. The baseline U (pink) is what you get with raw 200K-token retrieval. Restating the critical fact at the end (teal) recovers most of the lost middle. Using structured headers (yellow) flattens the U substantially. None of these are model changes — they are prompt-engineering changes."}
positions = np.linspace(0, 1, 21)

def u_curve(pos, primacy=0.75, recency=0.62, valley=0.50, sharpness=4.0):
    left  = primacy * np.exp(-((pos - 0.0) * sharpness)**2)
    right = recency * np.exp(-((pos - 1.0) * sharpness)**2)
    return np.maximum(np.maximum(left, right), valley)

baseline = u_curve(positions, primacy=0.78, recency=0.62, valley=0.45)
restate_end = u_curve(positions, primacy=0.78, recency=0.85, valley=0.55)
headers     = u_curve(positions, primacy=0.78, recency=0.72, valley=0.66,
                       sharpness=2.4)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(positions, baseline, marker='o', linewidth=2, color='#FF007F',
        label='baseline (raw 200K prompt)', markeredgecolor='#1A1A1A',
        markersize=5)
ax.plot(positions, restate_end, marker='s', linewidth=2, color='#00A8A8',
        label='+ restate critical fact at end', markeredgecolor='#1A1A1A',
        markersize=5)
ax.plot(positions, headers, marker='^', linewidth=2, color='#FFD700',
        label='+ structured headers throughout', markeredgecolor='#1A1A1A',
        markersize=5)

ax.set_xlabel("position of the answer in the prompt")
ax.set_ylabel("retrieval accuracy")
ax.set_title("Practical fixes don't change the model — they reshape the U-curve")
ax.set_ylim(0.3, 1.0)
ax.legend(loc='lower right', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

print("Prompt-engineering fixes that flatten the U:")
print("  Restate critical fact at the end          — gain ~10 pp at the middle")
print("  Structured headers throughout long context — gain ~15 pp at the middle")
print("  Split a 200K prompt into 2× 100K           — usually gain another 5-10 pp")
```

These are not deep capability changes. They are *workarounds* for an attention-allocation problem. Each one has a cost (latency, tokens, complexity). The 2026 takeaway is that **modern long-context inference involves a lot of prompt engineering specifically to defeat the U-curve**, which is in turn evidence that the underlying capability has not been fixed in the architecture, only in the wrapper.

## The Deeper Point: Position Is A Capability Channel

Most developers think of *attention* as a "look up the right token" operation. The U-curve is a corrective. **Position is itself a capability channel**, partly orthogonal to content. A model that can find your sentence at position 1 cannot necessarily find the same sentence at position 100,000. A model that scores 99% on NIAH at depth 0% and depth 100% might score 35% at depth 50%.

This is the operational reason every serious long-context benchmark since Liu et al. *varies the position of the relevant information* and reports the score *as a function of position*. NIAH did this from day one (it's why the heatmap had a depth axis). RULER does it. MRCR does it. GraphWalks does it. The single-number averages you see in marketing material are *averaging across position* — which means they are averaging across a curve whose tails are misleading you about the middle.

We will see this play out again in [The Chroma Measurement](../12-context-rot/), where the U-curve gets formal treatment across 18 frontier models and the position-of-the-valley itself turns out to be a measurable model property.

## What To Remember

1. **Liu et al., July 2023**: model accuracy on retrieval tasks forms a U-shaped curve as a function of where the relevant information sits in the prompt. High at start, high at end, low in the middle.
2. **The valley is at or below the closed-book baseline**. Putting the right answer in the middle of a long prompt can be *worse* than not putting it in the prompt at all.
3. **At least three causes operate**: training-distribution bias (real text concentrates information at edges), RoPE-extension aliasing (middle positions are exactly where extended positional codes get noisy), and attention dilution (more keys means lower per-key probability).
4. **The U is asymmetric** — primacy bias is usually stronger than recency bias. Better-trained, more carefully-aligned models have shallower valleys.
5. **Position is a capability channel.** Two prompts containing the same information at different positions can elicit dramatically different model behaviour. Modern long-context engineering involves *defeating* the U-curve via prompt structure, restated headers, and chunking.
6. **Every credible long-context benchmark reports as a function of position**. Single-number averages mask the curve.

**Continue to** → [One Pass Isn't Enough](../08-needle-to-graph/) — the April 2025 inflection point. OpenAI ships GPT-4.1 with a new benchmark called GraphWalks, and the launch blog spells out the conceptual move that the 2024 benchmark wave had been circling around: long-context tasks should require *non-linear* traversal, not just linear scanning.
