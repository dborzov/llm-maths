# Writing Guide

**When to read this:** You are writing a new article, polishing an existing one, or making substantive content changes. This is the authoritative source for voice, structure, and quality standards. It supersedes any briefer writing notes elsewhere.

---

## The Ultimate Goal

Every article exists to give the reader a **deeper, intuitive understanding of first-principles concepts** — the mathematical and logical essence of what is being explained. Not a survey of facts. Not a cheat sheet. A shift in how the reader *thinks* about the topic.

Specifically, the reader should finish each article able to:
- Grok deep, highly technical results and experiments and see the big-picture meaning behind the data points.
- Hold a grounded, concrete understanding — grounded in plots and stripped-to-the-essence Python, not vague hand-waving chains of fancy terms.
- Notice surprising connections between algorithms, math methods, and phenomena that initially look unrelated.
- See the topic not as a collection of recipes and facts, but as part of one massive interconnected web of human thought.

Keep this goal in mind at all times. It overrides every stylistic suggestion below.

---

## Audience

**Target level:** TowardsDataScience blog readers.

**Assume:** Foundations in linear algebra, calculus, statistics, probability, discrete math. Mechanical understanding of ML basics and LLM architecture. Familiarity with Python, NumPy, and PyTorch.

**Rule:** Leverage their existing knowledge. Use appropriate terminology without over-explaining basics.

---

## Narrative Techniques (Use Judgment — Not a Checklist)

These are tools, not rules. Think about the ultimate goal and choose the tools that serve it for this specific topic.

### The Detective Mystery Arc

Frame the article as a detective story rather than a lecture:

1. Open with a **mystery** — a technical problem or question, framed through simplified historical context. Make the reader feel the challenge. Give it human stakes.
2. Walk through **naive approaches and why they fail**. This is not wasted space — it's where the reader internalizes why the problem is hard.
3. Let the narrative **drift through related themes and subplots**, circling back to the framing story as understanding accumulates.
4. Deliver the **dramatic culmination**: the solution arrives as a natural endpoint, not a stranger. By the time the algorithm appears, its elements should feel like breadcrumbs the reader already picked up.

Example: Stuart Lloyd at Bell Labs, puzzling over how to quantize analog phone sound into a discrete set of values. Show what's hard about it. Walk through why obvious approaches fail. Let the reader follow Lloyd to the insight. By the time Lloyd-Max appears, it isn't surprising — it's inevitable.

### History Framing

Don't throw final math definitions at the reader as if they fell from the sky. Frame the material as a timeline of how humans — engineers, scientists, programmers — slowly shaped their understanding: early hunches, naive attempts, dead-ends, and eventual insight. The reader is on a journey through time, feeling the human stakes, appreciating what the material is really about.

### Hero's Journey

Technical questions are often best framed through the struggles and breakthroughs of a specific person. Include:
- A little background on the person, the place, the year.
- The relevant zeitgeist — what were the related fields grappling with?
- The human consequences of the problem: was this used in WW2 weaponry? Then say it helped win WW2. We are learning math, not testifying in court.

### The "Aha" Moment

Every article should have at least one genuine epiphany — a moment where the reader sees a core technical issue in a new light or notices a non-obvious connection. Requirements for a good epiphany:

- **Non-obvious to a technical audience.** Avoid eat-pray-love trivialities.
- **Mathematically deep.** Should result from careful empirical analysis, a surprising application of mathematical logic, or a new perspective that reframes earlier material.
- **Core to the topic.** Not a tangent.

### Theory of Mind While Writing

Be conscious of what the reader knows at each point in the article. Do not suddenly introduce terminology you have not yet used. Do not assume the reader (or the historical character facing the mystery) already knows the answer you are about to reveal. Maintain suspense.

---

## Concrete Anchoring (Required)

### One Running Python Example

Anchor abstract concepts to a single, highly specific Python toy example that you follow throughout the narrative. When explaining Vector Quantization, take a specific array of 2D float vectors and show manipulations on *those* values. Do not use generic `$v \in \mathbb{R}^n$` language as the primary vehicle.

### Napkin Math

For key quantities — parameters, bytes, FLOPs, tokens — include Fermi estimates and order-of-magnitude reasoning. Make quantitative arguments concrete.

### Metaphors and Analogies

For long stretches of abstraction, introduce a grounding metaphor early and return to it. Example: vector quantization → shoe sizes. Feet grow continuously (analog), society forces them into discrete boxes (sizes 9, 9.5, 10). Lloyd's job was finding where to place those sizes to minimize total discomfort (error) for the whole population.

---

## Ruthless Simplification

Get to the beating heart of the material as fast as possible. This means:

- **Sacrifice edge cases and the general case** — put them in an appendix ("You might have seen this as…") if needed.
- **One specific example, done well,** beats a never-ending "Also see" list of applications.
- **Delay or skip scope entirely** if it doesn't serve the core insight.

---

## Connections and Cross-Links

Every article should surface surprising connections — to other articles on this site, to other areas of mathematics, to historical threads. Consider a dedicated "Connections" appendix with a bullet list of the most interesting links, with hyperlinks.

Every concept that has a primer in this issue must link to that primer with `[label](../slug-of-primer/)`. Every primer should link forward to the mainline article that uses it.

---

## Per-Article Quality Checklist

Every article must satisfy all of these before it is done:

- [ ] **Opens with a human moment.** A specific year, a specific person, a specific failed approach. Never "In this article we will…".
- [ ] **One concrete example carried through.** Better to anchor one well than five poorly.
- [ ] **Math + intuition + picture.** After every load-bearing equation: what does it *feel like*? Where possible, a pyplot block that shows it.
- [ ] **An "aha" turn.** Set up a question, show naive approach failing, deliver the surprising insight.
- [ ] **Napkin math.** Fermi estimates for key quantities (parameters, bytes, FLOPs, tokens).
- [ ] **At least 2 pyplot blocks** for mainline articles; **1+ pyplot block** for primers. Use theme colors: `#FF007F`, `#00A8A8`, `#FFD700`, `#FF8C00`.
- [ ] **Cross-links to siblings.** Every concept with a primer must link there; every primer links forward to the mainline article that uses it.
- [ ] **Pop-art formatting variety.** Mix `**bold**`, `*emphasis*`, tables, blockquotes, `<details>` blocks, fenced code, inline HTML/SVG. Use the component library (pullquotes, callouts, margin notes, crossheads) to break up uniform pages. The page must not be one font weight on one background.
- [ ] **Closes with a forward link.** A "Continue to → [Next Article]" line, written as a cliffhanger.
- [ ] **Uses microGPT terminology where applicable, with wiki shortcodes on first mention.** See [`docs/microgpt-contract.md`](microgpt-contract.md) and [`docs/wiki.md`](wiki.md).

---

## Primer vs Mainline vs Boss — How to Decide

When sketching the tech tree, slot each article into one role deliberately:

| If the article is…                                                         | Use `techKind` | Tree node color |
|----------------------------------------------------------------------------|----------------|-----------------|
| The narrative entry point (cold open). No prereqs.                         | `mainline`     | not in tree     |
| A storyline chapter advancing the plot ("X happened, then Y broke things") | `mainline`     | pink            |
| A standalone tutorial on one math concept that other chapters build on     | `primer`       | cream           |
| The capstone "everything builds to this" finale                            | `boss`         | yellow halftone |
| Cross-issue reference to a concept covered elsewhere                       | `external`     | faint teal      |

Heuristic: **mainline chapters advance the story. Primer chapters can be read in any order. Boss chapters are the destination.**

---

## Formatting Tools

Use these as tools to make the narrative better explained and more intuitive:

- **Plots.** See [`docs/components/pyplot.md`](components/pyplot.md).
- **Interactive HTML visualizations.** Where you'd put a table of numbers, consider an interactive element with toggles/selectors that let the reader explore the data and see the argument it makes.
- **Varied formatting.** Bullet lists, nested lists, tables, code blocks, blockquotes, `<details>` elements. Vary it. Surprise the reader. Keep it fresh.
- **Component library.** Full reference at [`docs/components/README.md`](components/README.md). Quick lookup:

| You want…                                  | Reach for                                    |
|--------------------------------------------|----------------------------------------------|
| Top-of-article banner                      | `header:` front-matter field                 |
| Profound conclusion / theorem              | `{{% pullquote %}}`                          |
| Tangent / tip / warning / definition       | `{{% callout %}}`                            |
| Citation or single-sentence aside          | `{{% marginnote %}}`                         |
| Mid-section signpost (not a TOC entry)     | `{{< crosshead >}}`                          |
| Third-party image with caption + lightbox  | `{{< figure src=... caption=... >}}`         |
| Our own matplotlib plot                    | ` ```pyplot ` fenced block                   |
| Five-event chronology                      | `{{< timeline name="..." >}}`                |
| Issue-cover DAG TOC                        | `{{< techtree name="..." >}}`                |
| Interactive panel instead of a data table  | `{{< infographic >}}`                        |

---

## Reference Examples

- Good **mainline** article: `content/issues/03-sixteen-numbers/06-outliers.md`
- Good **primer** article: `content/issues/03-sixteen-numbers/03-lloyd-max.md`
