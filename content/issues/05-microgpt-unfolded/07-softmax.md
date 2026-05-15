---
title: "Softmax, Numerically"
description: "Why we subtract the max before exponentiating, what `flash-attention`'s online softmax is actually doing differently, and the one-line implementation that hides a decade of numerical analysis."
topics: [transformer, numerics]
tags: [microgpt, softmax, numerical-stability]
theme: cream
math: true
draft: false
date: 2026-05-14T02:48:52-04:00
issue: 5
weight: 70
techKind: primer
techNode: softmax
header: default.webp
---

## Eugene, Oregon, 1959

A 33-year-old mathematical psychologist at the University of Oregon is staring at a stack of index cards. Each card records one decision a subject made in a behavioural experiment: shown three faint tones of slightly different pitch, which one did they pick as "highest"? Some tones get picked far more often than others. The psychologist's name is **R. Duncan Luce**, and he is trying to write down, on a single line of mathematics, *how* a brain converts a vector of internal preferences into the probability of choosing one option over the others.

The constraint he is reaching for he calls — pompously, but with reason — the **choice axiom**. It says: the relative odds of preferring A over B must not depend on whether some other option C is even on the menu. If you prefer coffee to tea two-to-one when those are your options, you should still prefer coffee to tea two-to-one when juice is added to the table.

When Luce works out the equation that satisfies this axiom, he gets, up to a free parameter,

$$
P(\text{choose } i) = \frac{e^{\beta v_i}}{\sum_j e^{\beta v_j}}
$$

He publishes it in his 1959 book *Individual Choice Behavior*. The statistical-mechanics crowd had been writing the same expression — calling it the **Boltzmann distribution** — since the 1870s. The neural-network crowd will rediscover it as **softmax** in the mid-1980s. The deep-learning textbooks of 2014 will print it as if it were obvious.

You already met it in [chapter 1's listing](../01-cold-open/). It is the four-line helper at the bottom of the page:

```python
def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(val - max_val) for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Two lines of math, one line of arithmetic. But the first line — the `max_val = max(logits)` subtraction — is a numerical-analysis trick that, when forgotten, has bricked production inference servers, blown up training runs, and burned at least one engineer-week in every shop that has tried to write its own inference loop. This primer is about *why that line is there*.

## What Softmax Is Even For

Softmax takes a vector of real numbers $x \in \mathbb{R}^n$ — anything from $-\infty$ to $+\infty$ — and squeezes it into a **probability distribution**: $n$ non-negative numbers that sum to exactly 1.

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

Three properties make it the natural choice for this job, and they are worth saying out loud because every alternative people have tried fails at least one of them.

1. **All outputs are positive.** $e^{x_i} > 0$ for any real $x_i$. There is no way to accidentally produce a negative probability.
2. **They sum to 1.** The denominator is, by construction, the sum of all the numerators. Drop your hand on any softmax output and you have a valid distribution.
3. **The order is preserved.** $e^{\cdot}$ is monotonically increasing, so the largest logit always becomes the largest probability. There is no scrambling.

What softmax *doesn't* do, and this catches everyone the first time: it is not "the function that turns numbers into probabilities" — it is *one such function*, chosen because it satisfies Luce's axiom and because its derivative is suspiciously clean (you can write $\partial \text{softmax}_i / \partial x_j$ in terms of softmax outputs themselves, which is why backprop through it is one line of code). A perfectly valid alternative would be to clip everything below zero and normalise; it would just lose all the gradients and Luce would frown.

The exponential is what makes the gap between "biggest {{< wiki "logit" >}}logit{{< /wiki >}}" and "everyone else" widen as the logits get larger. Two logits differing by 1 produce probabilities in the ratio $e^1 \approx 2.72$. Differing by 5: ratio $e^5 \approx 148$. Differing by 20: ratio $\approx 5 \cdot 10^8$. *Big numbers eat small numbers.* This is the entire reason a single {{< wiki "attention" >}}attention{{< /wiki >}} head can lock onto one specific previous token while ignoring the rest.

## Where It Lives In microGPT

Open [the cold-open listing](../01-cold-open/) and search for `softmax`. Two hits.

**Hit #1 — inside each attention head, every layer, every token.** Right after [the Q·K dot products](../06-qkv-projections/) get divided by $\sqrt{d_\text{head}}$, we softmax them into attention weights:

```python
attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
               for t in range(len(k_h))]
attn_weights = softmax(attn_logits)
```

Each `attn_logits` is a vector of length $T$ (the number of tokens seen so far). One softmax per head, per layer, per generated token. We'll wire that up properly in [the next chapter on attention](../08-attention/).

**Hit #2 — the sampling loop, once per generated token.** After `lm_head` produces a vector of `vocab_size` logits, the driver code in chapter 1 does:

```python
probs = softmax([l / temperature for l in logits])
token_id = random.choices(range(vocab_size), weights=probs)[0]
```

The `temperature` is just a divisor before the softmax: small temperature sharpens the distribution (the biggest logit eats almost everything), large temperature flattens it (every token roughly equiprobable). We will earn the right to that knob in [chapter 15 on sampling](../15-sampling/).

Two appearances. Same function. Wildly different vector lengths: $T$ tokens for attention versus $|V|$ vocabulary entries for sampling.

## The Naive `exp` Blows Up

So why is the first line of the helper subtracting `max_val`? Try the textbook formula on a vector that any reasonable inference path could actually produce.

```python
>>> import math
>>> math.exp(709)
8.218407461554972e+307
>>> math.exp(710)
OverflowError: math range error
```

IEEE-754 double precision tops out at about $1.8 \cdot 10^{308}$. The exponent above which `math.exp` (and, for that matter, `numpy.exp` returning a numpy double) overflows is roughly **709.78**. For 32-bit floats — what most GPU kernels actually use — the cliff is much closer: $e^{x}$ overflows for $x \gtrsim 88.7$.

When does an LLM produce a logit that big?

The attention logits in [the next chapter](../08-attention/) are protected by the $1/\sqrt{d_\text{head}}$ scaling factor. For Llama 3 8B's `head_dim = 128`, you'd need the raw Q·K dot product to exceed $709 \cdot \sqrt{128} \approx 8020$ before overflow bites. That is possible but unusual on trained models — though it has happened, and the 2023 *Outlier Suppression* literature is partly motivated by exactly such blowups in the attention path.

The lm_head logits are not protected. A well-trained model's lm_head logits routinely span $\pm 30$ or so in float32, which is fine. But the driver scales them by `1 / temperature` before softmaxing:

$$
\text{logit}_i \mapsto \text{logit}_i \, / \, T
$$

At `temperature = 0.01` (a "near-greedy" setting that some users *will* hand you), a logit of 10 becomes 1000 and `exp(1000)` is, in IEEE-754, plus infinity. Your probability distribution is `[nan, nan, ..., nan]`. Your sampler returns garbage. Your support ticket says "the model started producing Chinese characters at low temperatures."

That is the failure mode every inference framework has shipped at least once.

## The Max-Subtract Trick

Here is the bargain that saves us. For *any* constant $c$,

$$
\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}
$$

because $e^{-c}$ factors out of every term in both the numerator and the denominator and cancels. The softmax output is **shift-invariant**: you can add or subtract any constant from every logit and get exactly the same probabilities.

Now pick $c = \max_j x_j$. After the shift, every shifted logit $x_i - c$ is **non-positive**, with the largest one being exactly zero. So:

- The numerator $e^{x_i - c}$ is in the safe range $(0, 1]$. No overflow.
- The denominator is a sum of values in $(0, 1]$, with at least one term equal to 1. It is bounded between 1 and $n$. No overflow.

Underflow can still happen — `exp(-800)` flushes to zero — but the *result* is still well-defined. If $x_i - c$ is so far below zero that `math.exp` returns 0.0, the corresponding probability genuinely *is* roughly zero, which is the answer you want. The unsafe failure mode (inf/nan, contagion) is gone.

This is the entirety of the trick. One line of code, one cancellation, an entire class of bugs eliminated:

```python
max_val = max(logits)
exps = [math.exp(val - max_val) for val in logits]
```

Worked example. Suppose `logits = [1000, 1001, 1002]` — a perfectly plausible result of a `temperature = 0.01` sampling pass. The naive formula tries to compute `exp(1000)` and overflows immediately. The safe version computes `max_val = 1002`, then `exp(-2), exp(-1), exp(0) = [0.1353, 0.3679, 1.0000]`, sums to `1.5032`, divides, and returns `[0.0900, 0.2447, 0.6652]`. Sums to 1. Largest probability on the largest logit. Done.

## Picture I: Temperature Shapes The Distribution

Before we plot the overflow, let's get an intuition for what softmax *does* to a fixed vector when you turn the temperature knob. Set `logits = [-5, -2, 0, 1, 3, 10]` — six imaginary vocabulary tokens, one ("token 5") far ahead of the pack, and see what happens at $T = 0.1, 1, 5$.

```pyplot {id="softmax-temperature" caption="One logit vector, three temperatures. Low T sharpens to a winner-take-all spike on the largest logit. High T flattens toward uniform. T=1 is the baseline."}
def softmax(logits):
    max_val = max(logits)
    exps = [np.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

logits = [-5, -2, 0, 1, 3, 10]
temperatures = [0.1, 1.0, 5.0]
colors = ['#FF007F', '#00A8A8', '#FFD700']

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, T, c in zip(axes, temperatures, colors):
    probs = softmax([l / T for l in logits])
    bars = ax.bar(range(len(logits)), probs, color=c,
                  edgecolor='#1A1A1A', linewidth=1.5)
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f'{p:.3f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(logits)))
    ax.set_xticklabels([f'{l}' for l in logits])
    ax.set_xlabel('logit value')
    ax.set_title(f'T = {T}', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('softmax probability')
plt.tight_layout()
```

At $T = 0.1$ the largest logit takes essentially the entire probability mass — a "near-greedy" decode. At $T = 1$, the baseline, you can see the exponential pulling probability toward the winner but leaving a few percent for the runners-up. At $T = 5$, the distribution is nearly flat: all six tokens are roughly equally likely and the model has been turned into a random word generator.

A useful intuition: **dividing logits by $T$ is the same as multiplying the "sharpness" of the distribution.** $T \to 0$ → argmax; $T \to \infty$ → uniform.

## Picture II: The Overflow Cliff

Now the actual stability picture. Plot `exp(x)` for $x$ from 600 to 720, alongside `exp(x - max(x))` for the same range. One curve will fly off the screen into IEEE-754 infinity; the other will stay neatly bounded inside $(0, 1]$.

```pyplot {id="overflow-cliff" caption="Why the max-subtract trick exists. exp(x) (pink) overflows past x ≈ 709.78 in float64 — the entire computation derails. exp(x - max(x)) (teal) stays bounded in (0, 1] no matter how large the inputs."}
xs = np.linspace(600, 720, 240)

# Naive: exp(x) directly. Use float64; values above ~709.78 overflow to inf.
naive = np.exp(xs.astype(np.float64))

# Safe: shift each individual x by the running max (here, just x itself
# for the single-value demo) — equivalent to exp(x - max_in_batch).
# For a softmax over [x], max(x)=x so exp(x-max)=1 always. To show a more
# realistic case, treat each x as one logit in a batch where max=720.
shifted = np.exp(xs - 720.0)

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(xs, naive, color='#FF007F', linewidth=2.4, label='exp(x) — naive')
ax.semilogy(xs, shifted, color='#00A8A8', linewidth=2.4,
            label='exp(x - max(x)) — max-subtract trick')
ax.axvline(709.78, color='#FF8C00', linewidth=1.5, linestyle='--',
           label='float64 overflow ≈ 709.78')
ax.axhline(1.8e308, color='#1A1A1A', linewidth=0.8, linestyle=':',
           alpha=0.7, label='float64 max ≈ 1.8e308')
ax.set_xlabel('x (logit value)')
ax.set_ylabel('exp(...)  [log scale]')
ax.set_title('The overflow cliff: naive exp blows up, max-subtract stays bounded')
ax.legend(loc='center left', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# Print where exactly the naive curve hits inf
first_inf = xs[np.isinf(naive)]
if len(first_inf):
    print(f"Naive exp(x) becomes inf at x = {first_inf[0]:.2f}")
print(f"Max of safe curve: {shifted.max():.3e}  (always <= 1)")
print(f"Min of safe curve: {shifted.min():.3e}")
```

The pink curve climbs without bound and then *cliffs* into `inf` at $x \approx 709.78$ — and once one entry of your `exps` array is `inf`, the sum is `inf`, every probability is `inf / inf = nan`, and the entire downstream pipeline is poisoned. The teal curve, after the max-subtract, never exceeds 1.

## Online Softmax: One Pass, No Buffer

There is a sequel to this story that we will not tell in full here but is worth flagging, because it is *the* reason flash-attention is fast.

The naive softmax we just defended needs **two passes over the logit array**: one pass to find the max, one pass to compute and sum the exponentials. For attention on a sequence of length $T$, that means materialising a length-$T$ vector of logits in memory before you can begin. For $T = 100{,}000$ tokens — well within 2024-2026 context windows — that's a 100,000-entry buffer **per head per layer**, written to and read from GPU HBM. The whole bottleneck of long-context attention turns out to be this buffer, not the actual arithmetic.

The trick, called **online softmax**, is to maintain a *running max* $m$ and a *running denominator* $\ell$ as you stream through the logits in chunks. When you process a new chunk and it contains a value larger than $m$, you rescale the running denominator by $e^{m_\text{old} - m_\text{new}}$ to account for the shift, then accumulate. The math is identical to the two-pass version — same answer, same numerical stability — but you only have to hold one chunk in fast memory at a time. NVIDIA's *Flash-Attention* (Dao et al., 2022) and *Flash-Attention 2* (Dao, 2023) build their entire attention kernel around this loop, fused with the Q·K and softmax(...)·V steps. We unpack the payoff in [chapter 8](../08-attention/).

The germ of the trick is already in the four-line `softmax` helper: *whatever constant you subtract from the logits, the answer is the same*. Online softmax just chooses that constant adaptively, chunk by chunk, instead of committing to it at the start.

## Napkin Math: How Often Does Softmax Run?

Per token of output from **Llama 3 8B** during decoding, with context length $T = 8192$:

- **Attention softmax**: one softmax per head, per layer, per generated token. With `n_layer = 32` and `n_head = 32`, that's $32 \cdot 32 = 1024$ softmaxes, each over a vector of length $T = 8192$. Total exp calls: $1024 \cdot 8192 \approx 8.4$ million.
- **Sampling softmax**: one softmax over `vocab_size = 128{,}000` logits at the end. Total exp calls: $128{,}000$.

So per generated token: **~8.4 million exp() evaluations for attention vs. ~128 thousand for sampling** — a 65× ratio. The vast majority of softmax FLOPs in inference live inside the attention loop, not in the sampler.

For a 100-token response, multiply by 100: nearly **a billion exp() calls** spent on attention softmax alone. This is why every modern attention kernel is fused, why flash-attention is online, and why the sampler — even at vocabulary 128k — barely shows up in profiles.

## What To Remember

1. **Softmax turns a real-valued vector into a probability distribution** in the only way that satisfies Luce's choice axiom (and, equivalently, the only way whose derivative is a clean function of its own outputs). All three of: all-positive, sums-to-1, order-preserving.
2. **Big numbers eat small numbers.** A gap of 20 between two logits is a ratio of $5 \cdot 10^8$ in probabilities. That is how attention heads "pick a winner" and how low-temperature sampling becomes near-deterministic.
3. **Naive `exp(x)` overflows at $x \approx 709.78$ in float64** (much earlier, $\approx 88.7$, in float32). Realistic low-temperature sampling produces logits that blow that cliff.
4. **The max-subtract trick is one line and one identity.** Subtract $\max_j x_j$ from every logit before exponentiating. The probabilities are unchanged because $e^{-\max}$ cancels; the largest exponent becomes 0; nothing overflows.
5. **The same identity powers online softmax.** Whatever you subtract doesn't matter, so flash-attention subtracts a *running* max as it streams logits in chunks. Two passes become one; the length-$T$ buffer vanishes.
6. **Softmax appears in microGPT in two roles**: turning [Q·K logits](../06-qkv-projections/) into attention weights (the expensive role — millions of evaluations per token) and turning lm_head logits into a sampling distribution (the cheap role — once per generated token).

---

**Continue to** → [Scaled Dot-Product Attention](../08-attention/) — armed with a safe softmax, we can finally assemble the per-head loop that defines how transformers remember.

