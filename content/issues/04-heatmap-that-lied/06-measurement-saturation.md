---
title: "Goodhart's Ceiling"
description: "When a measure becomes a target, it ceases to be a good measure. A short primer on why benchmarks die — and why NIAH dying in 2024 was inevitable, predictable, and a textbook case."
topics: [evaluation, methodology, history]
tags: [goodhart, saturation, benchmarks, ceiling-effect, measurement]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 60
techKind: primer
techNode: measurement-saturation
header: 06-measurement-saturation.webp
---

## A Bank Of England Economist In 1975

The story begins in **London, 1975**. **Charles Goodhart** is a young senior advisor at the Bank of England, watching the British government deploy *monetary aggregates* — specifically the M3 money supply — as policy tools to control inflation. The idea is straightforward: M3 is empirically correlated with inflation, so if you can control M3, you can control inflation.

Within months it stops working. The relationship between M3 and inflation breaks down. Goodhart writes a now-famous internal note explaining why, which gets published in 1981 and crystallises into a one-sentence law:

> *"Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes."*

{{% pullquote type="profound" author="Marilyn Strathern (1997)" %}}
When a measure becomes a target, it ceases to be a good measure.
{{% /pullquote %}}

Once a benchmark is being **optimised against**, the correlation that made it useful in the first place starts to erode. The number goes up; the underlying capability does not. **Eventually the number reaches a ceiling and stops moving, while the capability it was supposed to track is somewhere else entirely.**

Every benchmark in machine learning has lived this arc. NIAH lived it in 2024.

## The Three Phases Of A Benchmark's Life

Pull back and watch a benchmark over time. Almost every ML evaluation in the last 20 years has followed roughly the same three-phase trajectory.

```pyplot {id="benchmark-lifecycle" caption="Idealised benchmark lifecycle: from discriminating signal (Phase I), through productive optimisation (Phase II), to saturation and ceiling-hugging (Phase III). The X-axis is wall-clock time after the benchmark's release; the Y-axis is the gap between the best model and the chance baseline, scaled to [0, 1]."}
t = np.linspace(0, 5, 200)
# Sigmoid growth to ceiling
def lifecycle(t, midpoint=2.0, slope=2.5, ceiling=1.0):
    return ceiling / (1 + np.exp(-slope*(t - midpoint)))

phases = {
    'NIAH (2023-2024)':       (1.0, 5.0, 0.99),  # quick saturation
    'GLUE (2018-2019)':       (1.2, 4.0, 0.95),
    'ImageNet (2012-2017)':   (2.5, 1.5, 0.92),
    'MMLU (2021-2024)':       (2.8, 1.8, 0.88),
}

fig, ax = plt.subplots(figsize=(8.5, 4.3))
colors = ['#FF007F', '#00A8A8', '#FFD700', '#FF8C00']
for (name, params), color in zip(phases.items(), colors):
    midpoint, slope, ceiling = params
    ax.plot(t, lifecycle(t, midpoint, slope, ceiling),
            color=color, linewidth=2.2, label=name)

ax.axhline(0.97, color='#1A1A1A', linewidth=0.5, linestyle=':')
ax.text(0.1, 0.985, "saturation zone", fontsize=8, style='italic')
ax.set_xlabel("years after release")
ax.set_ylabel("best model accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Every benchmark has three phases: signal, optimisation, ceiling")
ax.legend(loc='lower right', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# Annotate phases on NIAH
ax.annotate("Phase I:\nsignal", xy=(0.4, 0.05), fontsize=8, ha='center')
ax.annotate("Phase II:\noptimisation", xy=(1.2, 0.3), fontsize=8, ha='center')
ax.annotate("Phase III:\nceiling", xy=(3.0, 0.97), fontsize=8, ha='center')

print("NIAH saturated in <6 months — the shortest Phase II in ML benchmark history.")
print("This is partly why the field had no time to develop a successor before the")
print("ceiling effect dominated the reporting culture.")
```

**Phase I — Signal.** The benchmark is new. Different models score wildly different numbers. Improvements on the underlying capability translate cleanly into improvements on the benchmark score. The Pearson correlation between *capability* and *score* is close to 1. The benchmark *separates* models.

**Phase II — Optimisation.** The community has rallied around the benchmark. Labs are training against it (perhaps not literally, but the marketing pressure makes it a *de facto* target). Scores climb. The benchmark is still useful — improvements still mostly mean what they look like they mean.

**Phase III — Ceiling.** Scores asymptote near 100%. The best models are all clustered within a few percentage points of each other. *The variance of the test, conditional on top-tier models, is smaller than the noise.* You cannot tell the new model is better than the old model by looking at this number alone. The correlation between capability and score collapses — not because the capability is solved (it may or may not be), but because the *measurement* is dead.

NIAH's trajectory was, by ML standards, *unusually fast*. From Kamradt's November 2023 launch to industry-wide 99% saturation in March 2024, the entire arc was **four months**. By comparison: GLUE took roughly 14 months. ImageNet took ~5 years. MMLU took ~3 years. The reason NIAH burned through Phase II so fast is the same reason it became popular: **the test was easy.** A capability that took 4 months to saturate clearly was not the capability the field thought it was measuring.

## The Math Of A Dead Benchmark

Let's be precise about what "the measurement is dead" *means*. Imagine you're trying to rank two models: Model A is genuinely better at long-context recall than Model B. The benchmark gives each model a score $s_A$ and $s_B$. The question is: **with what probability does $s_A > s_B$ on the benchmark, given that A is genuinely better?**

When the benchmark is in Phase I, the gap $s_A - s_B$ is large compared to the test's noise floor. The signal is clean. You can rank models with high confidence.

When the benchmark is in Phase III — when every model scores 98–99% — the gap collapses. Two effects compound:

1. **Compression**: the *possible* range of scores has collapsed from `[0, 100]` to roughly `[97, 100]`. There's only 3 points of headroom.
2. **Sample noise**: scoring noise (which questions happened to land in the test, which prompt template was used, which random seed for sampling) is on the order of a few percentage points for a few-hundred-example benchmark.

When the compression and the sample noise are comparable, your ability to tell A from B falls apart. Mathematically, if the gap $\Delta$ is comparable to the standard error $\sigma$, the *test statistic* $\Delta / \sigma$ is order 1, and the test is no longer informative.

```pyplot {id="saturation-statistical-power" caption="The statistical-power view of saturation. As the benchmark's mean approaches its ceiling, the variance shrinks (top panel) and the gap between two models of comparable strength (bottom panel) becomes indistinguishable from noise. Once Δ/σ falls below ~2, the benchmark can no longer reliably rank models."}
np.random.seed(7)
# Simulate a benchmark of 200 questions, scored as proportion correct
N = 200

# Two models, A slightly better than B, across three phases of difficulty
def simulate(true_skill_A, true_skill_B, N=200, trials=2000):
    a = np.random.binomial(N, true_skill_A, trials) / N
    b = np.random.binomial(N, true_skill_B, trials) / N
    return a, b

phases = [
    ("Phase I (difficult)",  0.45, 0.40),
    ("Phase II (moderate)",  0.80, 0.75),
    ("Phase III (saturated)",0.99, 0.985),
]

fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharey='row')
for col, (name, A, B) in enumerate(phases):
    a, b = simulate(A, B)
    # Top: distributions of A and B
    ax = axes[0, col]
    ax.hist(a, bins=20, color='#FF007F', alpha=0.6, label=f'Model A ({A:.2%})')
    ax.hist(b, bins=20, color='#00A8A8', alpha=0.6, label=f'Model B ({B:.2%})')
    ax.set_title(name, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    if col == 0:
        ax.set_ylabel('score frequency')
    ax.legend(fontsize=8)

    # Bottom: distribution of A - B
    ax = axes[1, col]
    diff = a - b
    ax.hist(diff, bins=30, color='#FFD700',
            edgecolor='#1A1A1A', linewidth=0.4)
    ax.axvline(0, color='#1A1A1A', linewidth=1.5)
    delta = (A - B)
    sigma = diff.std()
    z = delta / sigma if sigma > 0 else float('inf')
    ax.set_title(f"A-B   Δ={delta:.3f}  σ={sigma:.3f}  Δ/σ={z:.1f}", fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    if col == 0:
        ax.set_ylabel('A − B distribution')
    ax.set_xlim(-0.10, 0.15)

plt.tight_layout()

print("Power vs phase:")
for name, A, B in phases:
    a, b = simulate(A, B)
    win_rate = (a > b).mean()
    print(f"  {name:25s} P(A beats B on benchmark) = {win_rate:.2%}")
print("\nIn Phase III the benchmark calls A the winner only slightly more than 50%")
print("of the time, despite A being genuinely better. That's a dead benchmark.")
```

The print-out is the punchline. When a model that is *genuinely better* wins the benchmark only ~50–60% of the time, the benchmark is no longer ranking models — it's flipping a noisy coin. That's the state NIAH reached by mid-2024.

## A Brief Tour Of Past Benchmark Deaths

The ML literature has a long history of benchmarks dying. A short Burke-style tour:

- **MNIST (1998 → ~2010).** Hand-written digit recognition. By the mid-2000s, conv-nets were scoring 99.5%; by 2010, ensemble methods broke 99.8%. The test became a sanity check for new architectures, not a research target.
- **ImageNet (2010 → 2017).** The 1.2M-image classification benchmark that defined the deep-learning era. From AlexNet's 84% in 2012 to ResNet's 96.4% in 2015 — three years of dramatic progress. By 2017, top models were saturating around 88% top-1 / 99% top-5; the field migrated to ImageNet-21K, then to ImageNet-V2, then to harder downstream tasks.
- **GLUE (2018 → 2019).** A multi-task NLP benchmark. Saturated in 14 months. The same authors released **SuperGLUE** with explicitly harder tasks; that saturated within 2 years too.
- **SQuAD (2016 → 2018).** Reading comprehension benchmark. SuperHuman performance achieved in under 3 years. SQuAD 2.0 added unanswerable questions; it lasted about 2 years more.
- **MMLU (2021 → 2024).** Multi-task language understanding. By 2024, top models hit ~89% (vs ~88% human-expert estimate). Saturation. The successor is MMLU-Pro and Humanity's Last Exam.
- **GSM8K (2021 → 2023).** Grade-school math word problems. Saturated in ~18 months. Successors: MATH, then AIME, then FrontierMath.
- **NIAH (Nov 2023 → Mar 2024).** Four months. The fastest-saturating ML benchmark on record.

Notice the pattern: **each saturation triggers a new benchmark with a harder variant of the same task.** The community is using saturation as a *signal to make the test harder.* This is fine, but it has a cost — every cycle, the new benchmark has to re-prove its own validity. NIAH's saturation produced the seven 2024 successors in [When Everyone Scored 99](../05-saturation/), each one with its own assumptions to re-litigate.

## Why NIAH Saturated So Fast

NIAH's four-month death was not an accident. Three structural reasons:

1. **The task is narrow.** "Find this sentence" is a single capability. The model only has to do it well *once*, in one mode. There are no sub-skills to slowly improve on.
2. **The needle is engineered as stylistically detectable.** As we showed in [Needle in a Haystack](../03-niah-mechanics/), the embedded sentence is *intentionally* out-of-distribution within the haystack. A model with good language-modelling — which every frontier model has — can detect the local seam.
3. **The benchmark is load-bearing on prompting.** With Anthropic's 10-word fix, *every* model can score >95%. Once that fix became common knowledge, the benchmark was a 30-second prompt-engineering exercise.

None of these were *flaws* per se. They were *limits* — the benchmark measured a real capability, but a narrow one with a sharp ceiling. The reporting culture's mistake was to extrapolate from "the model can do this" to "the model can do long context." Those are different claims.

## The Anti-Saturation Toolbox

What do you *do* about a saturating benchmark? The literature has converged on a small toolbox.

- **Length-conditional reporting.** Don't quote a single number; report performance at $\{8K, 32K, 128K, 256K, 1M\}$. Saturation at one length is fine; saturation at all lengths means the test is dead. See HELMET's methodology.
- **Adversarial construction.** Build the test so that the obvious solution path fails. NoLiMa removed token overlap. AbsenceBench flipped the question. GraphWalks built in non-linear traversal.
- **Recent-data design.** Use data the model couldn't have seen during training. NoCha used novels published after the model's cutoff. GraphWalks uses random hex hashes that have no semantic prior.
- **Human-expert ceiling.** Build the test so that the *measurement floor is human expert performance*. LongBench v2 used the 15-minute human expert as the ceiling.
- **Multi-task suites.** Don't rely on one task. RULER, BABILong, HELMET all bundle multiple capabilities so saturation has to happen across all of them simultaneously.

The 2026 evaluation stack ([The 2026 Layered Stack](../16-eval-stack-2026/)) uses all five tricks at once. The structure is *designed* to resist Goodhart-style collapse.

## Goodhart Is Forever

The deep observation hiding inside Goodhart's law is that **any quantitative success criterion you can write down will eventually be gamed**, whether by deliberate gaming or by selection pressure that looks indistinguishable from gaming. The Bank of England's M3 → inflation correlation broke down not because British bankers *cheated* — they didn't have to. The correlation broke because once M3 was the target, every actor in the system optimised for whatever side of M3 they cared about, and the side that "improving M3" was supposed to capture stopped being the one that mattered.

For ML benchmarks, the same logic applies. *Optimising your model against NIAH* in 2024 was not cheating. But it was also not the same as *optimising your model for long-context understanding*. As the field gradually realised, those two optimisation targets *diverged*. The benchmark crisis of 2024 is what happens when the field collectively discovers the divergence.

{{% callout type="tip" title="Reading Benchmark Numbers After Saturation" %}}
- When a frontier model scores **above 95%** on a benchmark, treat that number as a *sanity check*, not a *measurement*. The discriminating information lives in length-conditional, task-conditional, and adversarial-variant sub-scores.
- When a benchmark is being promoted in marketing materials, expect Phase III collapse within **12–24 months**. The successor should be designed before it's urgently needed.
- A healthy benchmark should separate today's models by **>10 pp**. If it doesn't, it's a sanity check — name it that way.
{{% /callout %}}

Two practical takeaways for the rest of this issue:

- When a benchmark's headline number on a frontier model is **above 95%**, treat it as a *sanity check*, not a *measurement*. The discriminating information lives in length-conditional, task-conditional, and adversarial-variant scores beneath the headline.
- When a benchmark is being optimised against in marketing materials, expect Phase III collapse within 12–24 months. Plan the successor *before* you need it.

## What To Remember

1. **Goodhart's law**: when a measure becomes a target, it ceases to be a good measure. The original 1975 observation was about monetary policy; it generalises to every quantitative evaluation under optimisation pressure.
2. **Three benchmark phases**: Signal (test separates models) → Optimisation (test improves alongside capability) → Ceiling (test variance < noise; ranking unreliable).
3. **NIAH's four-month death was the fastest in ML benchmark history.** Three reasons: narrow task, detectable needle, prompt-load-bearing scoring.
4. **A dead benchmark is statistically detectable.** When $\Delta / \sigma$ between two genuinely-different models falls below ~2, the test is randomly ranking them. Watch for this.
5. **Anti-saturation tools**: length-conditional reporting, adversarial construction, recent-data design, human-expert ceiling, multi-task suites. The 2026 evaluation stack uses all five.

**Continue to** → [The U-Curve](../07-lost-in-the-middle/) — the position-dependent failure mode that *every* long-context benchmark eventually rediscovers. Models recall best from the start and the end of the prompt, worst from the middle, and the shape of the curve says something deep about how attention allocates.
