---
title: "LM Head and Sampling"
description: "One linear layer (`lm_head`) turns a 16-vector into a distribution over the vocabulary. After that comes a small zoo of tricks — temperature, top-k, top-p, min-p — to bend that distribution into the next token."
topics: [transformer, sampling]
tags: [microgpt, lm_head, sampling, temperature, top-p]
theme: cream
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 150
techKind: primer
techNode: sampling
header: 15-sampling.webp
---

## The Paper That Renamed The Knob

In April 2019, Ari Holtzman and his co-authors at the Allen Institute uploaded a paper to arXiv with a title that sounded like a Sherlock Holmes story: *The Curious Case of Neural Text Degeneration.* The "case" was a problem nobody had a clean name for. GPT-2 had just been released. People were generating text from it. And whenever they used the obvious approach — pick the most likely next token, every time — the output collapsed into loops within a paragraph:

> "The study, which was published in the journal Science, found that the study, which was published in the journal Science, found that the study, which was published in the journal Science…"

The opposite extreme — sampling proportionally from the full distribution — produced a different kind of garbage: fluent for two sentences, then a left turn into hallucinated proper nouns and dropped grammar. Holtzman's paper introduced **nucleus sampling** (now universally called *top-p*), and along the way it taught a generation of practitioners the lesson that the final step of an LLM — the bit that turns a 50,000-long vector of floats into a single integer — is not a footnote. It is a knob, and the knob has texture.

That knob is what this chapter is about. We have spent fourteen chapters earning the right to one final `linear`:

```python
return linear(x, state_dict['lm_head'])
```

After that line returns, the model is done. Everything that happens next — temperature, top-k, top-p, min-p, repetition penalty, beam search — happens **outside** the function, in seven lines of driver code that the [cold open](../01-cold-open/) printed and we never explained. Today we explain.

## What `lm_head` Actually Is

Take a moment to remember where `x` came from. After the embedding lookup, after `n_layer` blocks of attention plus MLP, after the final RMSNorm (which microGPT's listing folds into the first line of each sub-block — see [ch.5 rmsnorm](../05-rmsnorm/)), `x` is a single vector of length `n_embd`. For our toy that is 16 floats. For Llama 3 8B it is 4,096. This vector is the **residual stream** at its final state. It is what the model thinks the next token "feels like" in concept-space.

The `lm_head` is a matrix of shape `(vocab_size, n_embd)`. One row per token in the vocabulary. The line

```python
return linear(x, state_dict['lm_head'])
```

dots `x` against each of those rows. The result is a vector of length `vocab_size` — one number per vocabulary token. **These numbers are the `{{< wiki "logit" >}}logits{{< /wiki >}}`.** They are not probabilities. They have not been normalized. They can be negative, can be 47, can be -3.2. The bigger the logit, the more the model "votes" for that token.

Geometrically: each row of `lm_head` is a *reading template* for one vocabulary word. The dot product is "how aligned is the final residual with the template for the word *cat*?" High alignment, high logit. The lm head is, literally, an inner-product retrieval against a learned dictionary of tokens.

That's it. **The model returns raw logits and stops.** No softmax inside `gpt()`. The caller decides what to do with them. This separation is deliberate — it lets the same `gpt()` function feed argmax decoding, sampling, beam search, speculative decoding, classifier-free guidance, contrastive decoding, and a dozen other downstream tricks without changing one line.

## Tied vs Untied: A Symmetry Most Models Exploit

Stare at the shapes for a second:

| Tensor | Shape |
|---|---|
| `wte` (token embedding) | `vocab_size × n_embd` |
| `lm_head` (vocab projection) | `vocab_size × n_embd` |

They are **the same shape**. Both have one row per vocabulary token, each row of length `n_embd`. This is not a coincidence — they are doing dual jobs. `wte` is the *reading* table (token id → vector). `lm_head` is the *writing* table (vector → distribution over tokens). The vocabulary on both sides is the same vocabulary.

Models that *tie* these tables share storage: `lm_head` is literally a transpose-view onto `wte`. GPT-2 and most pre-2023 open models did this — it saves a `vocab × n_embd` block of parameters for free, which in Llama 3 8B would be 524 million weights or about a third of a gigabyte at fp16.

Models that *untie* them (microGPT, Llama 3, Gemma) keep two separate tables. The cost is those 524 M extra parameters. The benefit is that the model can use **different geometries for reading and writing** — the residual stream that *receives* the token "cat" at the bottom of the network does not have to point in the same direction the residual stream uses to *emit* the token "cat" at the top. Empirically this gives a small but measurable PPL bump on large vocabularies and is worth the bytes.

> The [state dict chapter](../02-state-dict/) showed `lm_head` and `wte` as separate keys in microGPT's nine-name list. The tying choice is **a property of the loader, not the model file** — a tied model just omits the `lm_head` key and the inference code aliases it to `wte` on load.

## Logits To Probabilities: The Driver Loop

Here is the line that follows `gpt()` in microGPT's driver. We have skipped past it ten times in earlier chapters. Now we read it carefully.

```python
logits = gpt(token_id, pos_id, keys, values)
probs  = softmax([l / temperature for l in logits])
token_id = random.choices(range(vocab_size), weights=probs)[0]
if token_id == BOS:
    break
```

Three operations and a stop condition. Let's take them in order.

**Step 1: divide every logit by `temperature`.** A scalar division before the softmax. We unpack what it does below.

**Step 2: softmax.** The same numerically-stable {{< wiki "softmax" >}}softmax{{< /wiki >}} from [ch.7](../07-softmax/) — subtract the max, exponentiate, normalize. The output is a length-`vocab_size` probability distribution.

**Step 3: weighted random choice.** Python's `random.choices` does inverse-CDF sampling: build the cumulative distribution, throw a uniform random number into [0, sum(weights)), find the bucket. One token returned.

**Stop condition.** If the sampled token is `BOS` (the model's tokenizer reserves index 0 for both *beginning-of-sequence* and *end-of-sequence* in microGPT — most real tokenizers split these into separate `BOS` and `EOS` ids, often token 1 and 2), we break out of the decode loop. The model has signalled that it is done.

That is the entire sampling stack in its smallest form. Every fancier scheme is a modification of step 1 or step 2.

## Temperature: The Volume Knob

Temperature `T` divides logits before the softmax:

$$
p_i = \frac{e^{\ell_i / T}}{\sum_j e^{\ell_j / T}}
$$

Three limits to keep in your back pocket:

- **`T = 1`** — identity. Softmax of the raw logits. The model's "native" distribution.
- **`T < 1`** — sharpening. Differences between logits get amplified. The biggest logit's probability drifts toward 1.
- **`T > 1`** — flattening. Differences get squashed. The distribution drifts toward uniform.
- **`T → 0`** — argmax / greedy decoding. The single largest logit wins with probability 1.
- **`T → ∞`** — uniform sampling. Every token has probability `1/vocab_size`.

The metaphor that survives every conversation about this knob: temperature is *creativity* on one end and *determinism* on the other. T=0.2 for code (you want the same answer every time). T=0.8–1.0 for prose (you want variety). T>1 only when you're trying to break a model out of a rut.

```pyplot {id="temperature-sharpness" caption="Four different temperatures applied to the same eight logits. As T shrinks toward zero the distribution collapses onto the argmax; as T grows the bars flatten toward uniform."}
logits = np.array([2.5, 2.1, 1.8, 1.5, 0.9, 0.6, 0.2, -0.4])
labels = [f'tok{i}' for i in range(len(logits))]
temperatures = [0.3, 1.0, 2.0, 5.0]
colors = ['#FF007F', '#00A8A8', '#FFD700', '#FF8C00']

fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
for ax, T, c in zip(axes, temperatures, colors):
    scaled = logits / T
    scaled -= scaled.max()
    p = np.exp(scaled); p /= p.sum()
    ax.bar(labels, p, color=c, edgecolor='#1A1A1A', linewidth=1.2)
    ax.set_title(f'T = {T}', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_ylabel('probability')
fig.suptitle('Same logits, four temperatures', fontsize=12)
fig.tight_layout()
```

A subtle point that bites engineers: temperature is applied *before* softmax, not after. Dividing the probabilities by T and re-normalizing does not give the same answer. The exponential is the whole point.

## Top-k, Top-p, Min-p: Three Ways To Cut Off The Tail

Even with a sensible temperature, there is a structural problem with sampling from `random.choices(range(vocab_size), weights=probs)`. The vocab is 128,000 tokens long. Even if the bottom 120,000 each have probability 0.00001, **their collective probability is more than 1.0%**. One time in a hundred, the sampler will pick from the long tail of obvious garbage. This is the cause of "fluent for two sentences, then a wrong proper noun" failure mode that Holtzman's paper diagnosed.

The fix is to **truncate the distribution before sampling**. Three flavors:

**Top-k.** Sort logits descending. Keep the top `k`. Set the rest to `-∞`. Softmax. Sample. Fixed cardinality. Typical `k=40` or `k=50`. Cheap but coarse — sometimes the legitimate distribution has only 3 plausible tokens and you're still giving 47 garbage tokens a vote; sometimes it has 200 plausible tokens and you're throwing 160 of them away.

**Top-p (nucleus).** Sort logits descending. Take the smallest *prefix* whose cumulative softmax probability is ≥ `p`. Set everything else to `-∞`. Re-softmax and sample. Typical `p=0.9` or `p=0.95`. **Adaptive** — when the model is confident (one token has p=0.97), you keep just one token. When it's spread out (no token above p=0.05), you keep many. This is Holtzman's contribution.

**Min-p** (Minh et al., 2023). Keep every token whose probability is at least `min_p × max_prob`. Typical `min_p=0.05`. The intuition: if the top token has probability 0.4, anything below 0.02 is noise. If the top token has probability 0.05 (flat distribution, the model is genuinely uncertain), keep everything above 0.0025. Robust to the failure case where top-p over-truncates a deliberately uncertain distribution.

```pyplot {id="topk-topp-minp" caption="Same nine-token distribution, three different filters applied. Top-k=4 keeps a fixed number. Top-p=0.9 keeps until cumulative mass reaches 0.9 (here, five tokens). Min-p=0.1 keeps tokens above 10% of the max (here, four tokens — but the threshold floats with the distribution)."}
logits = np.array([3.2, 2.8, 2.3, 1.9, 1.4, 0.8, 0.3, -0.2, -0.8])
p = np.exp(logits - logits.max()); p /= p.sum()

# top-k=4
k = 4
keep_topk = np.zeros_like(p, dtype=bool)
keep_topk[np.argsort(p)[-k:]] = True

# top-p=0.9
order = np.argsort(p)[::-1]
cum = np.cumsum(p[order])
cutoff = np.searchsorted(cum, 0.9) + 1
keep_topp = np.zeros_like(p, dtype=bool)
keep_topp[order[:cutoff]] = True

# min-p=0.1 (relative to max)
keep_minp = p >= 0.1 * p.max()

labels = [f't{i}' for i in range(len(p))]
fig, axes = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True)
for ax, keep, title, color in zip(
    axes,
    [keep_topk, keep_topp, keep_minp],
    ['top-k (k=4)', 'top-p (p=0.9)', 'min-p (min_p=0.1)'],
    ['#FF007F', '#00A8A8', '#FF8C00'],
):
    bar_colors = [color if k_ else '#D9D9D9' for k_ in keep]
    ax.bar(labels, p, color=bar_colors, edgecolor='#1A1A1A', linewidth=1.2)
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylim(0, p.max() * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[-1].set_xlabel('token')
fig.suptitle('Three ways to cut the tail', fontsize=12)
fig.tight_layout()
```

In production, these are often **stacked**: first top-k=50 to keep the sort cheap, then top-p=0.9 inside the survivors, then temperature on the result. The order matters less than the fact that the long tail is gone before `random.choices` ever sees a weight.

## Greedy, Beam, And The Argmax Edge Case

Two more sampling strategies deserve a paragraph.

**Greedy decoding** is `T → 0`, or equivalently `np.argmax(logits)`. Pick the most likely token every time. Deterministic. Used for unit tests, for code-completion models, and for any setting where you want the same output to the same input. Its weakness is Holtzman's loop case — greedy decoding will happily repeat the same sentence forever if the model has even slightly miscalibrated odds at one branch point.

**Beam search** keeps `B` candidate sequences in parallel ("beams"). At each step, expand every beam by every possible next token (`B × vocab_size` candidates total), score each by cumulative log-prob, prune to the top `B`. At the end, return the highest-scoring beam. Beam search dominated machine translation for a decade (it's natural for tasks with one right answer) and is now mostly displaced for open-ended generation, because it amplifies the same repetition pathology as greedy — beam search loves loops. Modern LLM serving stacks include it as a legacy option but rarely as a default.

## Repetition Penalty And Other Logit Hacks

One small family of tricks worth knowing because every serving stack ships them: **logit processors** that modify the raw logit vector *before* the temperature/softmax/sampling pipeline.

- **Repetition penalty** (Keskar et al., 2019): divide the logit of every token that has appeared in the context by some factor (typical: 1.1–1.3). The model votes less strongly for tokens it has already said.
- **Frequency penalty** (OpenAI API): subtract `α × count(token)` from each logit. Scales with how often the token has appeared. Useful for long completions.
- **Presence penalty**: a flat `-β` for any token that has appeared at all. Encourages bringing in new vocabulary.
- **Logit bias**: a user-supplied dict `{token_id: bias}` added to logits. Used to forbid specific words, or to nudge structured-output models toward emitting JSON.

These are pre-softmax operations. They live in the gap between `gpt()` returning logits and the driver applying temperature. **The lm head returns a clean raw vector; the rest of the world bends it.**

## Napkin Math: How Expensive Is Sampling?

For Llama 3 8B: `vocab_size = 128,000`, `n_embd = 4,096`.

**The lm head itself.** `128,000 × 4,096 = 524,288,000` parameters. About 6.5% of the model's weights. At fp16, **1 GB on its own**. The matrix-vector product is 524 M multiply-adds — by far the most expensive single op in the per-token forward pass except for the FFN.

**Softmax over 128,000 logits.** 128,000 exps + 1 sum + 128,000 divides. Roughly 384,000 floating-point ops. **Roughly 1,400× cheaper than the lm head projection that produced them.**

**Sampling itself** (`random.choices` on a 128,000-long weight vector). Inverse CDF: cumulative sum is 128,000 adds, then a binary search (17 comparisons) for the chosen bucket. **Negligible.**

The takeaway: the per-token compute cost of inference is **almost entirely in the forward pass**. The lm head is roughly one-thirty-second of the model's total FLOPs per token. Everything that happens after it — temperature, top-k, top-p, min-p, sampling — is in the noise. This is why "sampling strategy" papers rarely show wall-clock numbers; the entire pipeline of post-processing is cheaper than computing one MLP block.

## What To Remember

1. **`lm_head` is just another `linear`.** One matrix-vector product against a `vocab_size × n_embd` table. Output is `vocab_size` raw logits. No softmax inside.
2. **The model stops at logits. The caller decides everything else.** Temperature, truncation, sampling strategy — all in driver code, not in `gpt()`.
3. **Temperature divides logits before softmax.** T=1 is native, T→0 is greedy argmax, T→∞ is uniform. Always apply *before* the exponential, never after.
4. **The long tail is real and needs cutting.** Top-k keeps a fixed count, top-p keeps a cumulative-probability prefix (adaptive to model confidence), min-p keeps tokens above a fraction of the max (robust to flat distributions). Production stacks usually combine them.
5. **Sampling is dirt cheap.** The lm head projection is ~6.5% of model weights and dominates the cost; the softmax and `random.choices` after it are rounding error. Optimize the forward pass, not the sampler.
6. **The stop condition is a token.** microGPT breaks on `BOS`; real tokenizers reserve specific `EOS` ids. Either way, generation halts when an integer match happens — not when the model "decides" to stop in some abstract sense.

---

**Continue to** → [The Full Forward Pass](../16-full-forward/) — every chapter so far has unpacked one line; now we reprint microGPT one last time with the whole listing annotated, and every variable cross-linked back to the chapter that earned it.

