---
title: "The Language of Risk"
description: "Gamblers discovered log-odds centuries before statisticians did. When risk factors multiply, log-odds add — and that simple fact is why logistic regression dominates medical statistics."
topics: [probability, statistics]
tags: [log-odds, odds-ratio, logit, risk-factors, logistic-regression]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 2
weight: 40
techKind: primer
techNode: log-odds
header: default.webp
---

## Gamblers Knew First

Before statisticians invented the logit, gamblers had been working with odds for centuries.

The reason is practical. If you bet on a horse race, the question is not "what's the probability this horse wins?" The question is "at what payout ratio is this bet worth taking?" The answer depends on the *odds* — the ratio of the probability of winning to the probability of losing.

If a horse has a 75% chance of winning, its odds are:

$$
\text{odds} = \frac{P}{1-P} = \frac{0.75}{0.25} = 3 \;\text{(or "3 to 1")}
$$

A fair bet pays 3:1 on a 3:1-odds horse. Gamblers learned to think in odds rather than probabilities because odds have a magical property when combining independent events:

> **When independent events combine, their odds multiply.**

If your horse has 3:1 odds of finishing in the top two, and 2:1 odds of finishing first *given* it finishes in the top two, the joint odds of finishing first are $3 \times 2 = 6$ to 1. Probabilities don't multiply like that — you'd need to work with $P(A \cap B) = P(A) \cdot P(B|A)$. Odds multiply directly when events are independent.

This multiplicative structure is also what makes log-odds useful for modeling risk.

## From Multiplication to Addition

The **log-odds** — or logit — of a probability $p$ is:

$$
\text{logit}(p) = \log\frac{p}{1-p}
$$

Taking the log converts multiplication into addition. If independent risk factors multiply the odds by $r_1, r_2, r_3, \ldots$, then in log-odds space they *add*:

$$
\log(\text{combined odds}) = \log(\text{baseline odds}) + \log r_1 + \log r_2 + \log r_3 + \cdots
$$

This is the core of what makes the logit transform natural for medical risk modeling. Each independent risk factor contributes an **additive term** to the log-odds. A model that says "smoking adds 1.1 to your log-odds of heart disease" is equivalent to saying "smoking multiplies your odds of heart disease by $e^{1.1} \approx 3$." The two are the same statement, in different units.

## A Berkson-Style Example: The Mayo Clinic Data

Consider a stylized version of the kind of patient data Berkson was working with at the Mayo Clinic in the 1940s — modeling coronary heart disease risk from two binary factors: smoking and elevated cholesterol.

We have four groups of patients:

| Group | Smoking | High cholesterol | $n$ | Disease | $\hat{p}$ |
|---|---|---|---|---|---|
| A | No | No | 400 | 20 | 0.050 |
| B | Yes | No | 400 | 54 | 0.135 |
| C | No | Yes | 400 | 70 | 0.175 |
| D | Yes | Yes | 400 | 155 | 0.388 |

Now let's compute the logits:

$$
\text{logit}(p_A) = \log\frac{0.050}{0.950} \approx -2.944 \quad (\text{this is } \beta_0)
$$

$$
\text{logit}(p_B) = \log\frac{0.135}{0.865} \approx -1.855
$$

$$
\beta_1 = \text{logit}(p_B) - \text{logit}(p_A) \approx -1.855 - (-2.944) = +1.089 \approx \log 3
$$

$$
\text{logit}(p_C) = \log\frac{0.175}{0.825} \approx -1.553
$$

$$
\beta_2 = \text{logit}(p_C) - \text{logit}(p_A) \approx -1.553 - (-2.944) = +1.391 \approx \log 4
$$

Now predict group D:

$$
\text{logit}(\hat{p}_D) = \beta_0 + \beta_1 + \beta_2 \approx -2.944 + 1.089 + 1.391 = -0.464
$$

$$
\hat{p}_D = \sigma(-0.464) = \frac{1}{1 + e^{0.464}} \approx 0.386
$$

The actual observed proportion in group D is 0.388. The model — which assumed smoking and cholesterol act *independently on the log-odds scale* — predicts it almost exactly.

{{% callout %}}
**The odds ratio for smoking is the same in both cholesterol strata.** Among low-cholesterol patients: odds ratio = $(0.135/0.865)/(0.050/0.950) = 0.156/0.053 = 2.97 \approx 3$. Among high-cholesterol patients: $(0.388/0.612)/(0.175/0.825) = 0.634/0.212 = 2.99 \approx 3$. The odds ratio for smoking is constant across cholesterol levels — that's what "no interaction" means in the logit model. If you just compared all smokers to all non-smokers, you'd confound smoking and cholesterol. Logistic regression estimates each factor's effect *holding the others constant*.
{{% /callout %}}

The model equation is:

$$
\text{logit}(p) = \beta_0 + \beta_1 \cdot S + \beta_2 \cdot C
$$

where $S \in \{0, 1\}$ is smoking status and $C \in \{0, 1\}$ is cholesterol status. Each $\beta$ is the additive change in log-odds, and $e^\beta$ is the multiplicative change in odds — the **odds ratio**.

```pyplot {id="log-odds-additive" caption="Left: the four groups in probability space — the scale is nonlinear and the contributions of smoking and cholesterol don't add cleanly. Right: the same data in log-odds space — smoking adds exactly log(3) and cholesterol adds exactly log(4), regardless of the other factor's status."}
import numpy as np

groups = ['No smoke\nNo chol', 'Smoke\nNo chol', 'No smoke\nChol', 'Smoke\nChol']
p = np.array([0.050, 0.135, 0.175, 0.388])
logit_p = np.log(p / (1 - p))

colors = ['#00A8A8', '#FF007F', '#FFD700', '#FF8C00']
x = np.arange(4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Left: probability space
bars1 = ax1.bar(x, p, color=colors, edgecolor='#1A1A1A', linewidth=2)
ax1.set_xticks(x)
ax1.set_xticklabels(groups, fontsize=8)
ax1.set_ylabel("probability of disease")
ax1.set_title("Probability space\n(nonlinear — risks don't add cleanly)")
ax1.set_ylim(0, 0.50)
for bar, pv in zip(bars1, p):
    ax1.text(bar.get_x() + bar.get_width()/2, pv + 0.008, f'{pv:.3f}',
             ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.spines[['top', 'right']].set_visible(False)

# Right: log-odds space
bars2 = ax2.bar(x, logit_p, color=colors, edgecolor='#1A1A1A', linewidth=2)
ax2.set_xticks(x)
ax2.set_xticklabels(groups, fontsize=8)
ax2.set_ylabel("log-odds (logit) of disease")
ax2.set_title("Log-odds space\n(linear — smoking adds log(3), chol adds log(4))")
ax2.axhline(0, color='#1A1A1A', linewidth=0.8, linestyle='--')
for bar, lv in zip(bars2, logit_p):
    va = 'bottom' if lv >= 0 else 'top'
    offset = 0.05 if lv >= 0 else -0.05
    ax2.text(bar.get_x() + bar.get_width()/2, lv + offset, f'{lv:.3f}',
             ha='center', va=va, fontsize=8, fontweight='bold')

# Annotate the additive gaps
b1 = logit_p[1] - logit_p[0]
b2 = logit_p[2] - logit_p[0]
ax2.annotate('', xy=(1, logit_p[1]), xytext=(1, logit_p[0]),
             arrowprops=dict(arrowstyle='<->', color='#1A1A1A', lw=1.5))
ax2.text(1.55, (logit_p[0] + logit_p[1]) / 2, f'+{b1:.2f}\n≈log(3)', fontsize=8, va='center')
ax2.annotate('', xy=(2, logit_p[2]), xytext=(2, logit_p[0]),
             arrowprops=dict(arrowstyle='<->', color='#1A1A1A', lw=1.5))
ax2.text(2.55, (logit_p[0] + logit_p[2]) / 2, f'+{b2:.2f}\n≈log(4)', fontsize=8, va='center')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
print(f"β₀ (baseline logit) = {logit_p[0]:.3f}")
print(f"β₁ (smoking) = {logit_p[1]-logit_p[0]:.3f} → OR = {np.exp(logit_p[1]-logit_p[0]):.2f}")
print(f"β₂ (cholesterol) = {logit_p[2]-logit_p[0]:.3f} → OR = {np.exp(logit_p[2]-logit_p[0]):.2f}")
print(f"Predicted logit(pD) = {logit_p[0] + (logit_p[1]-logit_p[0]) + (logit_p[2]-logit_p[0]):.3f}")
b0, b1, b2 = logit_p[0], logit_p[1]-logit_p[0], logit_p[2]-logit_p[0]
pred_d = 1 / (1 + np.exp(-(b0 + b1 + b2)))
print(f"Predicted pD = σ(logit) = {pred_d:.3f}  (observed: 0.388)")
```

## Why Log-Odds, Not Probabilities?

The log-odds representation has two advantages over working directly with probabilities.

**First, it's unbounded.** A probability lives in $[0, 1]$. A sum of linear terms can go anywhere on the real line. Modeling probabilities directly with a linear model will eventually predict values below 0 or above 1 — as we saw Bliss's linear fit do in [the cold open](../01-cold-open/). The logit transformation maps $[0, 1]$ to $(-\infty, +\infty)$, so you can safely fit a linear model on the logit scale and transform back.

**Second, it makes independent risk factors add.** When risk factors act independently — each one multiplying the baseline odds by some amount — those multipliers become additions in log-odds space. The model $\text{logit}(p) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots$ is exactly the claim that each covariate $x_i$ independently multiplies the odds by $e^{\beta_i}$.

This is why physicians adopted logistic regression so quickly once David Cox formalized it in 1958. The coefficients $e^{\beta_i}$ are **odds ratios** — a quantity that physicians, epidemiologists, and clinical trial designers had already been using for decades. "Smoking triples your odds of heart disease" is a sentence a cardiologist in 1960 could immediately understand and act on. "Smoking increases the probit of heart disease by 0.69" is not.

## What You Carry Forward

The logit is the natural scale for any binary outcome model. Every time you see:

$$
\log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k
$$

you are modeling the **log-odds as a linear combination of features**. Solving for $p$ gives you the logistic function. The model is called logistic regression not because it uses the logistic function (it does), but because it models the *logit* (log-odds) as a linear regression.

This distinction matters later. When a neural network produces "logits" from its last layer — the raw pre-softmax scores — those are *literally* log-odds estimates for each class. The name comes directly from Berkson's 1944 paper.

---

**Continue to** → [From Insects to ImageNet](../05-logistic-regression/) — the story of how logistic regression went from a hospital statistician's notebook to the computational engine behind every binary classifier, and why the gradient turns out to be the cleanest computation in all of machine learning.
