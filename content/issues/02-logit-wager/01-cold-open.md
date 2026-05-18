---
title: "Logistic Regression: born in a 1934 bug-killing experiment"
short_title: "Logistic Regression"
description: "Chester Bliss, 1934: six dose levels, 300 aphids, and a mortality curve no straight line can fit."
blurb:
  - "The LD50 — the dose that kills exactly 50% — is the problem. Bliss needs to read it off a curve, not a table."
  - "A linear fit predicts negative deaths at low doses and over 100% mortality at high doses. It is obviously wrong."
  - "The data isn't noisy — it's S-shaped. The shape is the signal, not the error."
  - "Every neural network classifier on Earth inherits its output function from this 1934 insect experiment."
topics: [probability, statistics]
tags: [probit, ld50, dose-response, bliss]
theme: cream
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 2
weight: 10
techKind: mainline
techNode: cold-open
header: 01-cold-open-02.webp
---

## Connecticut, 1934

It is the spring of 1934, and **Chester Ittner Bliss** is having a frustrating morning at the Connecticut Agricultural Experiment Station. His job sounds straightforward: measure how deadly a given dose of insecticide is to a population of aphids. Farmers need this number to know how much to spray. Regulators need it to certify products. The number even has a name — the **LD50**, the *lethal dose* that kills exactly 50% of the test population.

The problem is the data. Bliss has run the experiment with six dose levels, exposed groups of insects at each level, and counted the dead. Here is what he has:

| Dose (mg/L) | Exposed | Dead | Proportion dead |
|---|---|---|---|
| 1.0 | 50 | 3 | 0.06 |
| 2.0 | 50 | 10 | 0.20 |
| 3.0 | 50 | 19 | 0.38 |
| 5.0 | 50 | 34 | 0.68 |
| 8.0 | 50 | 42 | 0.84 |
| 12.0 | 50 | 48 | 0.96 |

The obvious thing to do is fit a line: *proportion dead = a + b × dose*. It is 1934; fitting a line is about the only tool in the statistician's bag that doesn't require a mechanical calculator. So Bliss tries it.

```pyplot {id="cold-open-linear-fail" caption="Left: raw mortality data — unmistakably S-shaped. Right: a linear fit fails badly, predicting impossible values above 1 and below 0."}
dose = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
proportion_dead = np.array([0.06, 0.20, 0.38, 0.68, 0.84, 0.96])

# Fit a line via least squares
coeffs = np.polyfit(dose, proportion_dead, 1)
p = np.poly1d(coeffs)
dose_fine = np.linspace(0, 15, 300)
fit_line = p(dose_fine)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left panel: raw data
ax1.scatter(dose, proportion_dead, color='#FF007F', s=80, zorder=5, label='observed')
ax1.set_xlabel("dose (mg/L)")
ax1.set_ylabel("proportion dead")
ax1.set_title("The raw data")
ax1.set_ylim(-0.05, 1.05)
ax1.axhline(0, color='#1A1A1A', linewidth=0.8, linestyle='--')
ax1.axhline(1, color='#1A1A1A', linewidth=0.8, linestyle='--')
ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)

# Right panel: linear fit
ax2.scatter(dose, proportion_dead, color='#FF007F', s=80, zorder=5, label='observed')
ax2.plot(dose_fine, fit_line, color='#00A8A8', linewidth=2, label='linear fit')
# Highlight impossible region
impossible_mask_low = fit_line < 0
impossible_mask_high = fit_line > 1
ax2.fill_between(dose_fine, fit_line, 0, where=impossible_mask_low,
                 color='#FF8C00', alpha=0.35, label='impossible (<0)')
ax2.fill_between(dose_fine, fit_line, 1, where=impossible_mask_high,
                 color='#FFD700', alpha=0.35, label='impossible (>1)')
ax2.axhline(0, color='#1A1A1A', linewidth=0.8, linestyle='--')
ax2.axhline(1, color='#1A1A1A', linewidth=0.8, linestyle='--')
ax2.set_xlabel("dose (mg/L)")
ax2.set_ylabel("proportion dead")
ax2.set_title("Linear fit: two problems")
ax2.set_ylim(-0.25, 1.25)
ax2.legend(fontsize=8)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

# Print the fit
print(f"Linear fit: proportion = {coeffs[0]:.4f} × dose + {coeffs[1]:.4f}")
print(f"At dose 0: predicted = {p(0):.3f}  (impossible if negative)")
print(f"At dose 15: predicted = {p(15):.3f}  (impossible if >1)")
```

The failure is immediate and obvious. A line extended far enough in either direction will predict proportions below zero and above one — values that are not, in any physical sense, probabilities. The relationship is not linear. It is **S-shaped**: flat near zero, steep in the middle, flat again near one.

## Why Everything Is S-Shaped

The S-shape is not a quirk of this particular insecticide or this particular aphid. It shows up in:

- **Drug dose vs patient response** (the whole field of pharmacology)
- **Voltage vs neuron firing rate** (in neuroscience, the sigmoid is sometimes called the "Hodgkin-Huxley curve")
- **Advertising spend vs sales conversion** (marketing attribution models)
- **Exam difficulty vs pass rate** (item-response theory in psychometrics)
- **Signal strength vs bit-error rate** (in communications engineering)

The reason is always the same: you are measuring the fraction of a population that crosses some threshold. At very low doses, almost nobody crosses the threshold; at very high doses, almost everybody does; in between is where the action is. The S-shape is the *cumulative distribution function* of whatever threshold distribution exists across the population.

For Bliss's insects, the interpretation is direct. Each aphid has its own *personal* lethal threshold — the minimum dose that kills *that individual*. Some aphids are fragile (low threshold). Some are hardy (high threshold). The thresholds vary across the population. At dose $x$, the fraction that dies is precisely the fraction whose personal threshold is at or below $x$. That fraction is, by definition, a CDF.

**The S-shape is a cumulative distribution function in disguise.**

The question that stopped Bliss that spring morning — and which drove one of the central ideas of 20th-century statistics — is: *whose* CDF? What distribution should we assume for the thresholds?

## The Constraint The Data Cannot Tell You

This is a genuinely subtle point. Looking only at the mortality proportions, you cannot determine the shape of the threshold distribution. You can see *that* there is an S-curve. You cannot see *which* S-curve.

This is always the situation when you observe cumulative effects. You observe the proportion dead at dose $x = 3$ mg/L. You don't observe the individual thresholds. You can't reconstruct the threshold distribution from the proportions alone — there are infinitely many distributions that would produce the same six proportions.

Bliss's resolution was to **assume** a distribution and check whether it fit. His assumption: the thresholds are **normally distributed** across the population. This is not obviously true, but it has two justifications.

*First*, the normal distribution arises naturally whenever a quantity results from many small independent factors adding together — and biological variation often has this character. An aphid's lethal threshold plausibly depends on its age, weight, the thickness of its cuticle, the state of its immune system, and a dozen other factors, each contributing a small amount.

*Second*, the assumption is testable. If it's wrong, the data will not lie on the fitted curve.

It turned out — for Bliss's data and for most biological dose-response data — the normal assumption *does* fit. Not perfectly, but close enough to work.

The mathematical machine Bliss built on this assumption — the **probit transform** — linearizes the S-curve so that you can fit it with ordinary least squares. It is a beautiful trick, and it is the subject of the next chapter.

---

**Continue to** → [Cramming the Bell Curve Into a Straight Line](../02-probit-transform/) — the mathematical leap that turns a stubborn S-curve into a line you can fit with 1930s arithmetic, and introduces a tool that would travel, 90 years later, into the hidden layers of every modern language model.
