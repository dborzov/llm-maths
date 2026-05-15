---
title: "The Simpler S-Curve"
description: "Mayo Clinic, 1944: a contrarian statistician bets that nobody can tell the difference between the normal CDF and his much simpler replacement. He names it the logit."
topics: [probability, statistics]
tags: [logit, logistic, berkson, mayo-clinic, probit]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 2
weight: 30
techKind: mainline
techNode: berksons-gamble
header: default.webp
---

## Rochester, Minnesota, 1944

**Joseph Berkson** was not a man who suffered inefficiency quietly.

It was 1944, and Berkson was the head of biometry at the Mayo Clinic in Rochester, Minnesota. His job was to build statistical models for medical outcomes — which patients were likely to survive radiation therapy, what factors predicted disease, whether smoking caused cancer. For all of these questions, the tool of choice was Bliss's probit model: fit the S-curve by assuming a normally distributed threshold, apply the [probit transform](../02-probit-transform/), and regress.

Every calculation required looking up values in Bliss's probit tables. Evaluating $\Phi^{-1}(p)$ — the inverse normal CDF — had no closed form. You interpolated in the table, and if you were fitting a model to a hundred patients, you did this a hundred times. Per iteration. Per variable.

Berkson was fitting risk models every day. He found this deeply irritating.

## What Berkson Was Really Working On

The probit friction was most acute for Berkson because of what he was studying. In the early 1940s, Berkson was one of the first scientists seriously trying to model the relationship between **cigarette smoking and lung cancer** — a question that required dose-response modeling on a massive scale, using hospital records rather than controlled laboratory experiments.

He needed to fit models across hundreds of covariates, thousands of patients, and many outcome categories. The probit's computational demands were not a minor inconvenience. They were a practical barrier to the research.

He began asking a question that, once asked, sounds almost obvious: *How different are the available S-curves, really?* If you could find a function that looked like $\Phi(x)$ but was easier to compute, and the difference was smaller than your measurement error — wouldn't that be strictly better?

## The Logistic Function

The function Berkson proposed was the **logistic**:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

This function has two remarkable properties. First, it is an S-curve: it increases monotonically from 0 to 1, with an inflection point at $x = 0$ where it crosses $p = 0.5$. Second, its inverse is trivially computable:

$$
\sigma^{-1}(p) = \log\!\left(\frac{p}{1-p}\right)
$$

No table lookup. No approximation. If you have $p$, you compute $p/(1-p)$ (the odds), take the logarithm, and you're done. Berkson called this inverse the **{{< wiki "logit" >}}logit{{< /wiki >}}** — from *log-odds unit*, by exact analogy with *probability unit* (probit).

The logit of a probability is the log-odds of that probability:

$$
\text{logit}(p) = \log\!\frac{p}{1-p}
$$

## The Key Question: How Similar Are They?

Berkson's bet was that the logistic and the normal CDF were, in practice, *indistinguishable*. To make this comparison fair, you have to scale them properly. The standard normal CDF $\Phi(x)$ has variance 1. The standard logistic $\sigma(x)$ has variance $\pi^2/3$. To compare them, Berkson scaled the logistic so they have the same variance:

$$
\sigma_{\text{scaled}}(x) = \frac{1}{1 + e^{-x/s}}, \quad s = \frac{\sqrt{3}}{\pi} \approx 0.5513
$$

With this scaling, the two S-curves have identical variance. Now we can ask: how different are they?

```pyplot {id="berkson-comparison" caption="Left: normal CDF vs scaled logistic — nearly indistinguishable in the data range. Centre: the difference between them — max 0.023. Right: the tails (x from 3 to 6) where the logistic assigns more probability mass."}
def _norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def _norm_cdf(x):
    _p_ = 0.2316419
    _b_ = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + _p_ * np.abs(x))
    poly = t * (_b_[0] + t*(_b_[1] + t*(_b_[2] + t*(_b_[3] + t*_b_[4]))))
    cdf_pos = 1.0 - _norm_pdf(np.abs(x)) * poly
    return np.where(x >= 0, cdf_pos, 1.0 - cdf_pos)

s = np.sqrt(3) / np.pi

x_main = np.linspace(-4, 4, 500)
x_tail = np.linspace(3, 6, 300)

norm_main = _norm_cdf(x_main)
logi_main = 1 / (1 + np.exp(-x_main / s))
diff = logi_main - norm_main

norm_tail = _norm_cdf(x_tail)
logi_tail = 1 / (1 + np.exp(-x_tail / s))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.5))

# Panel 1: both curves
ax1.plot(x_main, norm_main, color='#FF007F', linewidth=2.5, label='Normal CDF Φ(x)')
ax1.plot(x_main, logi_main, color='#00A8A8', linewidth=2, linestyle='--',
         label='Logistic (scaled)')
ax1.set_title("Two S-curves")
ax1.set_xlabel("x")
ax1.set_ylabel("CDF value")
ax1.legend(fontsize=8)
ax1.spines[['top', 'right']].set_visible(False)

# Panel 2: difference
ax2.plot(x_main, diff, color='#FF8C00', linewidth=2)
ax2.axhline(0, color='#1A1A1A', linewidth=0.8)
max_diff = np.abs(diff).max()
ax2.set_title(f"Difference: logistic − normal\n(max |Δ| = {max_diff:.4f})")
ax2.set_xlabel("x")
ax2.set_ylabel("logistic − normal CDF")
ax2.spines[['top', 'right']].set_visible(False)
ax2.fill_between(x_main, diff, alpha=0.2, color='#FF8C00')

# Panel 3: tails
ax3.plot(x_tail, norm_tail, color='#FF007F', linewidth=2.5, label='Normal CDF')
ax3.plot(x_tail, logi_tail, color='#00A8A8', linewidth=2, linestyle='--', label='Logistic')
ax3.set_title("The tails (x = 3 to 6)")
ax3.set_xlabel("x")
ax3.set_ylabel("CDF value (1 − tail probability)")
ax3.legend(fontsize=8)
ax3.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

# Compute specific differences for annotation
for xv in [1, 2, 3, 4, 5]:
    nd = float(_norm_cdf(np.array([xv]))[0])
    ld = 1 / (1 + np.exp(-xv / s))
    print(f"x={xv}: normal={nd:.6f}, logistic={ld:.6f}, diff={ld-nd:.6f}")
```

The maximum difference is about **0.023** — and it occurs near $x = \pm 1$, well within the data range. At $x = 3\sigma$ (the edge of most real datasets), the difference is under 0.008. For the kind of noisy biological data Berkson was working with, this gap is undetectable.

{{% pullquote %}}
Berkson wasn't claiming the logistic was *true*. He was claiming it was indistinguishable from the normal CDF where your data actually lives — and it was *infinitely* easier to compute.
{{% /pullquote %}}

## The Tails: Where the Wager Has Fine Print

The logistic does depart from the normal CDF — significantly — in the extreme tails. The logistic function has **heavier tails**: at $x = 5$, the logistic assigns roughly 10× more probability than the normal CDF to observations beyond that point.

For Berkson's practical purposes, this didn't matter. You almost never have data at $5\sigma$ from the center. If you do, you have other problems.

For the theoretical statisticians who argued with Berkson throughout the 1940s and 50s — and there were several who argued loudly — this was the fatal flaw. The probit, they insisted, arose from a principled biological model (the normal threshold distribution). The logistic was a convenient approximation that happened to look similar.

Berkson's reply, which you can still feel in his papers, was essentially: *show me the data that distinguishes them*.

Nobody ever could.

## The Naming: Logit by Analogy

Berkson was deliberate about terminology. He introduced "logit" as a direct parallel to Bliss's "probit":

- **Probit**: *pro*bability *unit*. The probit of $p$ is the $z$-score that gives that cumulative probability: $\Phi^{-1}(p)$.
- **Logit**: *log*-odds *unit*. The logit of $p$ is the log-odds that gives that probability: $\log[p/(1-p)]$.

Both are transformations that map a probability $(0, 1)$ to the entire real line $(-\infty, +\infty)$. Both linearize the S-curve. The probit uses the normal quantile function; the logit uses the log-odds.

**Logistic regression** is then the model that treats the logit as a linear function of the predictors — exactly as the probit model treats the probit as linear.

The question of which was "correct" — probit or logit — became one of the great polite arguments in 20th-century biostatistics. In medicine and epidemiology, the logit won almost completely, because the logit has a beautiful interpretation via the [odds ratio](../04-log-odds/) — a quantity that physicians already understood. In certain areas of pharmacology and toxicology, the probit persists, because regulators require it and it has a biological derivation.

In deep learning, the logit won unconditionally. The reason is computational — and we'll see it in full in [The Softmax Kingdom](../06-softmax-kingdom/).

---

**Continue to** → [The Language of Risk](../04-log-odds/) — the primer on log-odds that shows why Berkson's logit transform makes medical risk factors *add together* in exactly the way doctors need, and which gives logistic regression its main advantage over any alternative.
