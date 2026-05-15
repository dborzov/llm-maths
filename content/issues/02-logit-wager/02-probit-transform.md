---
title: "Cramming the Bell Curve Into a Straight Line"
description: "Bliss's key insight: the S-curve IS the normal CDF. Apply its inverse to both sides and you have a straight line — one you can fit by hand in 1934."
topics: [probability, statistics]
tags: [probit, normal-distribution, cdf, dose-response, bliss]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 2
weight: 20
techKind: primer
techNode: probit-transform
header: 02-probit-transform.webp
---

## Each Insect Has a Secret

The insight that cracked open the problem came when Bliss stopped thinking about doses and started thinking about *insects*.

Not the proportion dead at a given dose. The individual insects themselves.

Each aphid, Bliss reasoned, has a personal lethal threshold — the minimum dose that kills *that particular individual*. Call it $T_i$ for aphid $i$. Some aphids are fragile. They die at very low doses. Others are robust; they survive everything but the highest doses. The variation in individual tolerance $T_i$ is not a measurement error. It is a real biological fact: insects in the same population genuinely differ in their susceptibility.

Now watch what happens when we run the dose-response experiment. At dose level $x$, the fraction that dies is:

$$
P(\text{death} \mid \text{dose} = x) = P(T_i \leq x) = F_T(x)
$$

where $F_T$ is the *cumulative distribution function* of the threshold random variable $T$. **The S-curve is the CDF of individual tolerances across the population.** The LD50 — the dose that kills 50% — is the *median* of the threshold distribution.

## The Normal Distribution Assumption

Bliss needed to choose a distribution for $T$. His choice: **Gaussian**. The thresholds are normally distributed:

$$
T_i \sim \mathcal{N}(\mu, \sigma^2)
$$

This gives:

$$
P(\text{death} \mid \text{dose} = x) = \Phi\!\left(\frac{x - \mu}{\sigma}\right)
$$

where $\Phi$ is the standard normal CDF. The mysterious S-curve has a name: it is the normal CDF, shifted by $\mu$ and scaled by $\sigma$.

Why is this a reasonable assumption? **The central limit theorem gives us a philosophical argument.** An aphid's resistance to insecticide plausibly depends on many small, independent biological factors — the thickness of its cuticle, the efficiency of its excretory system, its metabolic rate, its age, its body mass, and dozens more. When many small independent effects combine additively, the sum converges to normal. The threshold distribution *should* be approximately Gaussian, for the same reason that human heights are approximately Gaussian.

```pyplot {id="probit-normal-cdf" caption="Left: the normal PDF (bell curve) — shaded area is the fraction of the population that dies at a given dose threshold. Right: the normal CDF — the S-curve we saw in the data. The two panels are the same mathematical object viewed from different angles."}
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

z = np.linspace(-4, 4, 500)
dose_threshold = 1.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: PDF with shaded area
ax1.plot(z, _norm_pdf(z), color='#FF007F', linewidth=2.5)
ax1.fill_between(z, _norm_pdf(z), where=(z <= dose_threshold),
                 color='#FF007F', alpha=0.30,
                 label=f'P(T ≤ {dose_threshold}) = {_norm_cdf(dose_threshold):.2f}')
ax1.axvline(dose_threshold, color='#1A1A1A', linewidth=1.5, linestyle='--')
ax1.set_title("Normal PDF: threshold distribution")
ax1.set_xlabel("tolerance z-score")
ax1.set_ylabel("probability density")
ax1.legend(fontsize=9)
ax1.spines[['top', 'right']].set_visible(False)
ax1.text(dose_threshold + 0.1, 0.3, f'z = {dose_threshold}', fontsize=9)

# Right: CDF (the S-curve)
ax2.plot(z, _norm_cdf(z), color='#00A8A8', linewidth=2.5)
ax2.axvline(dose_threshold, color='#1A1A1A', linewidth=1.5, linestyle='--')
ax2.axhline(_norm_cdf(dose_threshold), color='#FFD700', linewidth=1.5, linestyle='--')
ax2.scatter([dose_threshold], [_norm_cdf(dose_threshold)],
            color='#FF007F', s=80, zorder=5)
ax2.set_title("Normal CDF: the S-curve")
ax2.set_xlabel("tolerance z-score")
ax2.set_ylabel("proportion dead = Φ(z)")
ax2.text(dose_threshold + 0.1, _norm_cdf(dose_threshold) - 0.06,
         f'Φ({dose_threshold}) = {_norm_cdf(dose_threshold):.2f}', fontsize=9)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

## The Probit Transform: Linearizing the S-Curve

Here is the problem Bliss faced in 1934. He had an equation:

$$
P = \Phi\!\left(\frac{x - \mu}{\sigma}\right)
$$

He needed to estimate $\mu$ and $\sigma$ from his six data points. He could not simply minimize a nonlinear loss function — there were no computers. He needed *ordinary least squares*, which only works for linear relationships.

His trick: **apply $\Phi^{-1}$ to both sides.**

$$
\Phi^{-1}(P) = \frac{x - \mu}{\sigma} = \underbrace{\frac{-\mu}{\sigma}}_{=\, a} + \underbrace{\frac{1}{\sigma}}_{=\, b} \cdot x
$$

The left side, $\Phi^{-1}(P)$, is called the **probit** of $P$ (from *probability unit*). The right side is a *linear function of dose $x$*. If we define:

$$
\text{probit}(P) = \Phi^{-1}(P)
$$

then plotting probit$(P_i)$ against $x_i$ should give a **straight line**. Bliss could now:

1. Apply the probit transform to each observed proportion $P_i$.
2. Regress the probits on dose $x_i$ using ordinary least squares.
3. Read off $\mu = -a/b$ (the LD50) and $\sigma = 1/b$.

The catch: computing $\Phi^{-1}$ requires inverting the normal CDF, which has no closed form. Bliss spent *years* of his career computing probit tables by hand and later by mechanical calculator. He published the first comprehensive probit tables in 1934 — a contribution that was immediately indispensable, and which statisticians would use for decades.

```pyplot {id="probit-linearization" caption="The probit transform straightens the S-curve. Left: original data with fitted probit model (S-curve through the points). Right: the same data after the probit transform — a straight line that ordinary least squares can fit."}
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

dose = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
proportion_dead = np.array([0.06, 0.20, 0.38, 0.68, 0.84, 0.96])
probits = np.array([-1.555, -0.842, -0.305, 0.468, 0.994, 1.751])

# Fit line in probit space
coeffs = np.polyfit(dose, probits, 1)
b, a = coeffs
dose_fine = np.linspace(0, 14, 300)
probit_fit = a + b * dose_fine
p_fit = _norm_cdf(probit_fit)

ld50 = -a / b
sigma_hat = 1.0 / b

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.scatter(dose, proportion_dead, color='#FF007F', s=80, zorder=5, label='observed')
ax1.plot(dose_fine, p_fit, color='#00A8A8', linewidth=2, label='probit model')
ax1.axvline(ld50, color='#FFD700', linewidth=1.5, linestyle='--',
            label=f'LD50 = {ld50:.2f} mg/L')
ax1.axhline(0.5, color='#FFD700', linewidth=1, linestyle=':')
ax1.set_xlabel("dose (mg/L)")
ax1.set_ylabel("proportion dead")
ax1.set_title("Original space: S-curve fit")
ax1.legend(fontsize=8)
ax1.spines[['top', 'right']].set_visible(False)

ax2.scatter(dose, probits, color='#FF007F', s=80, zorder=5, label='observed (probit)')
ax2.plot(dose_fine, a + b * dose_fine, color='#00A8A8', linewidth=2, label='linear fit')
ax2.axhline(0, color='#1A1A1A', linewidth=0.8, linestyle='--')
ax2.set_xlabel("dose (mg/L)")
ax2.set_ylabel("probit(proportion dead)")
ax2.set_title("Probit space: straight line")
ax2.legend(fontsize=8)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print(f"Fitted: a={a:.4f}, b={b:.4f}")
print(f"LD50 = -a/b = {ld50:.2f} mg/L")
print(f"σ = 1/b = {sigma_hat:.2f} mg/L")
```

{{% callout %}}
**Napkin math.** The fitted slope $b \approx 0.26$ and intercept $a \approx -1.40$. So:

$$
\text{LD50} = -a/b \approx 1.40/0.26 \approx \mathbf{5.4 \text{ mg/L}}
$$

This is the dose at which half the aphid population dies. The slope $b = 1/\sigma$ tells us the spread of the tolerance distribution: $\sigma = 1/0.26 \approx 3.8$ mg/L. A large $\sigma$ means the aphid population has wide variation in sensitivity — some die at 1 mg/L, others survive 12 mg/L.
{{% /callout %}}

## Was Normal Really Right?

The normal distribution assumption was Bliss's bet. It was reasonable but not inevitable. There is a nagging question: is the *dose itself* normally distributed across individuals, or is it the *log of the dose*?

Biologically, the second is more plausible. Many physiological processes are multiplicative — the body metabolizes a drug by factors, not by fixed amounts. If the lethal tolerance depends on multiplicative biological processes, then $\log(\text{tolerance})$ is more likely normal than tolerance itself.

This suggests fitting the probit model against $\log_{10}(\text{dose})$ rather than dose. Bliss himself noticed this, and modern pharmacology has largely standardized on log-dose probit models.

```pyplot {id="probit-log-dose" caption="Probits vs raw dose (left) vs log₁₀ dose (right). The log-dose version is visually straighter — biological tolerances are multiplicative, not additive."}
dose = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
probits = np.array([-1.555, -0.842, -0.305, 0.468, 0.994, 1.751])
log_dose = np.log10(dose)

c_raw = np.polyfit(dose, probits, 1)
c_log = np.polyfit(log_dose, probits, 1)

fit_raw = np.poly1d(c_raw)(dose)
fit_log = np.poly1d(c_log)(log_dose)

ss_res_raw = np.sum((probits - fit_raw)**2)
ss_res_log = np.sum((probits - fit_log)**2)
ss_tot = np.sum((probits - probits.mean())**2)
r2_raw = 1 - ss_res_raw / ss_tot
r2_log = 1 - ss_res_log / ss_tot

dose_fine = np.linspace(0.5, 13, 200)
log_dose_fine = np.log10(dose_fine)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.scatter(dose, probits, color='#FF007F', s=80, zorder=5)
ax1.plot(dose_fine, np.poly1d(c_raw)(dose_fine), color='#00A8A8', linewidth=2)
ax1.set_xlabel("dose (mg/L)")
ax1.set_ylabel("probit")
ax1.set_title(f"Probit vs raw dose")
ax1.spines[['top', 'right']].set_visible(False)
ax1.text(6, -1.3, f"R² = {r2_raw:.3f}", fontsize=12, color='#00A8A8', fontweight='bold')

ax2.scatter(log_dose, probits, color='#FF007F', s=80, zorder=5)
ax2.plot(log_dose_fine, np.poly1d(c_log)(log_dose_fine), color='#FF8C00', linewidth=2)
ax2.set_xlabel("log₁₀(dose)")
ax2.set_ylabel("probit")
ax2.set_title(f"Probit vs log₁₀ dose")
ax2.spines[['top', 'right']].set_visible(False)
ax2.text(0.5, -1.3, f"R² = {r2_log:.3f}", fontsize=12, color='#FF8C00', fontweight='bold')

plt.tight_layout()
print(f"R² (raw dose): {r2_raw:.4f}")
print(f"R² (log dose): {r2_log:.4f}")
log_ld50_val = -c_log[1] / c_log[0]
print(f"LD50 in log-dose space: 10^{log_ld50_val:.3f} = {10**log_ld50_val:.2f} mg/L")
```

The log-dose version fits better ($R^2 \approx 0.994$ vs $R^2 \approx 0.956$). The LD50 in log-dose space is $10^{-a/b}$ mg/L. Modern pharmacology calls this the **log-probit** model, and it is the standard for regulatory LD50 submissions worldwide.

## What Probit Analysis Gave the World

Bliss's probit method gave practitioners:

1. **A principled biological model**: not just "fit a curve" but "here is the mechanistic story — individual thresholds distributed across a population."
2. **A linear fitting procedure**: no computers required, just a table of $\Phi^{-1}$ values.
3. **Interpretable parameters**: LD50 is a median, $\sigma$ is a spread, both with direct biological meaning.
4. **A testable assumption**: if the normal distribution is wrong, the data won't lie on the fitted line.

The one thing it didn't give them was *computational convenience*. Every application of the probit required evaluating $\Phi^{-1}$ — the inverse normal CDF — which has no closed form. You had to look it up in Bliss's tables, carefully interpolate between entries, and pray you hadn't made an arithmetic error.

Ten years later, a physician in Minnesota would get tired of looking things up in tables. His solution would change the shape of statistics — and, eventually, of deep learning.

---

**Continue to** → [The Simpler S-Curve](../03-berksons-gamble/) — a Mayo Clinic physician bets that nobody can tell the difference between the normal CDF and a much simpler function, and turns out to be exactly right.
