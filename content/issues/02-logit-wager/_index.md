---
title: "The Logit Wager"
description: "In 1934, an entomologist needed to kill bugs scientifically. In 1944, a physician replaced his hardest integral with a clever approximation. Ninety years later, that replacement runs inside every neural network on Earth."
issue: 2
layout: issue-cover
theme: cream
math: false
header: default.webp
date: 2026-05-14T09:00:00-04:00
---

It is the spring of 1934, and **Chester Ittner Bliss** has a problem that sounds almost comic. He is an entomologist at the Connecticut Agricultural Experiment Station, and his job is to measure, with scientific precision, how deadly a given dose of insecticide is to a given pest. Farmers need the number. Regulators need the number. Everyone needs the number — but nobody can fit the data to a straight line. The relationship between dose and mortality is stubbornly, infuriatingly S-shaped: nothing happens at low doses, then deaths accelerate through the middle range, then the curve flattens again as the last survivors cling on. You cannot draw a line through an S.

Bliss's solution — the **probit transform** — would win him a footnote in statistical history. But it came with a catch: to use it, you had to evaluate a certain integral that has no closed form. Every calculation required tables. Every table took hours to produce. For a hospital statistician handling hundreds of patients a week, this was a serious problem.

**Joseph Berkson** felt that friction every day. It was 1944, and the Mayo Clinic physician was building risk models for radiation therapy patients — fitting the same S-curves that Bliss had used on insects, now applied to human survival. He was one of the first researchers trying to model whether cigarette smoking caused cancer. He needed his S-curves to be *fast*. So he made a wager: what if you replaced the normal CDF — the integral without a closed form — with the **logistic function**, which has a trivially computable inverse? The error between the two curves would be at most 0.023. Nobody would be able to tell the difference in real data. He called his substitution the **{{< wiki "logit" >}}logit{{< /wiki >}}**.

Berkson was right that nobody could tell the difference. He was wrong that his approximation would stay confined to medical statistics. Ninety years after that wager, every neural network classifier on Earth ends its forward pass with Berkson's 1944 approximation. **PyTorch calls those pre-softmax scores "logits"** — after Berkson's own term — and CrossEntropyLoss expects them as input by default. The function that a Mayo Clinic physician invented to save himself twenty minutes of table-lookup runs billions of times per second on GPU clusters worldwide.

## How To Read This Issue

Start with the cold-open. The **tech tree** below is the table of contents — each node is an article, arrows show dependencies. You can read the primers in any order; the mainline chapters are best read left-to-right.

{{< techtree name="issue02" >}}

## What You Will Walk Away With

By the end of this issue you should be able to:

- **Explain the probit transform** — why the S-curve is a normal CDF, how Bliss linearised it, and what the LD50 actually measures in the original insect mortality sense.
- **Explain Berkson's wager** — what the logistic approximation trades away for its computational convenience, and why the trade almost never matters in practice.
- **Derive the logistic regression gradient** from first principles and see why it is the same computation as backpropagation through a sigmoid output neuron.
- **Understand why PyTorch wants raw logits** rather than probabilities, and trace the 90-year chain from Connecticut insects to `torch.nn.CrossEntropyLoss`.

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
