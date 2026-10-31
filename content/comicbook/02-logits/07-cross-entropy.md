---
title: "Cross-Entropy: the bill, the overcharge, and the discount"
short_title: "Cross-Entropy"
description: "Cross-entropy is what a model pays to encode reality, entropy is the fair price, KL divergence is the overcharge, and mutual information is the discount that context buys: one logarithm, averaged four ways."
blurb:
  - "1951: Claude Shannon opens a detective novel and has Betty Shannon guess it one letter at a time. It is the first next-token benchmark."
  - "PyTorch prints `loss = 3.12`. Part of that number is the language's fault and part is the model's. Which part is which?"
  - "KL divergence is the overcharge. Solomon Kullback, a codebreaker, defined it as evidence per observation, and it refuses to be symmetric."
  - "Mutual information is a KL divergence too: the exact overcharge for pretending the context doesn't matter."
topics: [information-theory, probability, machine-learning]
tags: [cross-entropy, kl-divergence, mutual-information, entropy, perplexity, shannon, kullback, distillation, scaling-laws]
theme: cream
math: true
draft: false
date: 2026-10-31T09:00:00-04:00
issue: 2
weight: 70
techKind: mainline
techNode: cross-entropy
header: default.webp
---

## Murray Hill, 1951

Claude Shannon has a problem he invented himself.

Three years earlier, in the 1948 paper that founded information theory, he proved that every source of messages has a number attached to it: its **entropy**, the average number of bits per symbol that no code, however clever, can beat. For a source that emits symbol $x$ with probability $p(x)$, the number is

$$
H = -\sum_x p(x) \log_2 p(x).
$$

Elegant, and for English almost useless. To evaluate it you need $p$: the true probability of every next letter given *everything that came before it*. Nobody has that table. Nobody ever will.

The obvious move is to build the table piece by piece, and Shannon does. Count single letters, then pairs, then triples. Each table gives a lower estimate, because each one uses more context:

| What the table knows (27 symbols: A–Z plus space) | Bits per letter |
|---|---|
| Nothing: all symbols equally likely | 4.76 |
| Single-letter frequencies (E is common, Z is rare) | 4.03 |
| The previous letter (after Q comes U) | 3.32 |
| The previous two letters | 3.1 |

Then the method runs out of road. A table of four-letter contexts has half a million rows; a table of ten-letter contexts has more rows than there is English in print. The long-range structure of language, the part that makes "THE ROOM WAS NOT VERY L" end in "IGHT", lives exactly where counting can't reach.

So Shannon does something sly. He swaps the table for a predictor that has already been trained on a lifetime of English: a person. He opens a book (a Raymond Chandler detective story, according to his biographers) and asks **Betty Shannon**, a mathematician who had worked in Bell Labs' mathematics group, to guess the passage one letter at a time. After each guess he says *right* or *wrong*; she keeps guessing until she gets it, and he writes down how many tries each letter took. *A SMALL OBLONG READING LAMP ON THE DESK…* Most letters fall on the first try. The first letter of a new word can take many.

From those guess counts Shannon squeezes out bounds: with about a hundred letters of context, printed English carries somewhere between **0.6 and 1.3 bits per letter**. Three quarters of the alphabet's capacity is redundancy.

Look at what he actually measured. Not the entropy of English; that stays out of reach. He measured how hard *one particular predictor* found real text. A predictor that is wrong about the language can only find it harder than it really is, never easier, so its difficulty is an upper bound. Keep that asymmetry in your pocket. This whole chapter hangs on it.

Seventy-five years later the guessing game runs on GPU clusters around the clock. The predictor is a transformer, each guess is a {{< wiki "softmax" >}}softmax{{< /wiki >}} over 50,257 tokens, and the scoreboard is a line that PyTorch prints every few seconds:

```text
step 600000 | loss 3.1172
```

Four questions about that number, which the rest of this chapter answers:

1. **What is it counting?** What are the units of 3.1172?
2. **Why does it never reach zero?** Where is the floor?
3. **How much of it is the model's fault,** and how much belongs to the language itself?
4. **How much is the context worth?** The model reads thousands of previous tokens. What does that buy, in the same units?

The answers have names: **{{< wiki "information-measures" >}}cross-entropy, entropy, KL divergence, and mutual information{{< /wiki >}}**. A phone-company mathematician and two teams of government codebreakers worked them out between 1941 and 1951, mostly working apart.

{{< timeline name="information-measures" >}}

## Surprise Has Units

Before averaging anything, price a single event. If a model gave probability $q$ to the thing that actually happened, its **surprise** is

$$
s = -\log_2 q \quad \text{bits}.
$$

Why a logarithm? Two demands pin it down. A certain event should cost nothing ($q = 1 \Rightarrow s = 0$). And two independent events should cost the *sum* of their separate surprises, even though their probabilities *multiply*. Only the log turns products into sums. That is the same reason gamblers kept their books in [log-odds](../04-log-odds/): independent risk factors multiply, and a log lets you add them instead.

Three numbers worth memorising:

- A fair coin landing heads: $-\log_2 \tfrac12 = $ **1 bit**.
- A one-in-a-thousand event: about **10 bits**.
- Something your model called impossible: $-\log_2 0 = $ **∞ bits**.

The base is a choice of currency. Base 2 gives **bits**. The natural log gives **nats**, and PyTorch uses the natural log, so every `loss` you have ever printed is in nats. {{% marginnote label="exchange rate" %}}1 nat = 1/ln 2 ≈ **1.443 bits**. A loss of 3.12 nats is 4.50 bits per token.{{% /marginnote %}} This chapter counts in bits while we work by hand and switches to nats when we reach the LLM.

{{< crosshead >}}Three hundred aphids{{< /crosshead >}}

We need a world to put prices on, and this issue already has one. In the [cold open](../01-cold-open/), Chester Bliss exposed six trays of 50 aphids to six doses of insecticide and counted the dead:

| Dose (mg/L) | 1 | 2 | 3 | 5 | 8 | 12 |
|---|---|---|---|---|---|---|
| Dead out of 50 | 3 | 10 | 19 | 34 | 42 | 48 |
| Death rate $p(\text{dead} \mid \text{dose})$ | 0.06 | 0.20 | 0.38 | 0.68 | 0.84 | 0.96 |

For the rest of this chapter **this table is the world**. Pick one of the 300 aphids at random and you get two facts: its dose $D$ and its fate $Y$.

Suppose nobody tells you the dose. Then 156 of 300 aphids died, so $p(\text{dead}) = 0.52$, and the **entropy** of a random aphid's fate is

$$
H(Y) = -0.52 \log_2 0.52 - 0.48 \log_2 0.48 = 0.999 \text{ bits}.
$$

One bit is the entropy of a fair coin. Without the dose, an aphid's fate is almost exactly a coin flip.

Entropy is the average surprise of the *best possible* predictor, the one that knows the true probabilities. Shannon's source-coding theorem says no code, compressor or model can report aphid fates in fewer than $H$ bits each on average. Call it the **fair price**. Notice that it is a property of the world alone. No model appears anywhere in the formula.

## The Bill: Cross-Entropy

Now let a model in. A model $q$ prices each aphid by its own beliefs, $-\log_2 q(\text{fate})$, but it doesn't choose which aphids show up. The world does. The model's average cost on real data is the **cross-entropy**:

$$
H(p, q) = -\sum_y p(y) \log_2 q(y) = \mathbb{E}_{y \sim p}\big[-\log_2 q(y)\big].
$$

The world picks the outcomes; the model sets the prices; cross-entropy is the bill. When the model also sees the dose, it prices each aphid with $q(y \mid d)$ and we average over the trays as well.

Here is the property that put cross-entropy into every training loop: **you can compute it without knowing $p$.** You never need the true probabilities. You just walk through the real aphids and average $-\log q$ over what actually happened. That is all `F.cross_entropy` does with a batch of tokens: it averages $-\log \operatorname{softmax}(z)_y$ over the targets $y$ that really occurred.

Five candidate models get the same 300 aphids:

| Model | What it believes | Bits per aphid | Bill for all 300 |
|---|---|---|---|
| **The coin-flipper** | Every aphid: 50/50 | 1.000 | 300.0 bits |
| **The bookie** | Every aphid: 52% dead (the overall rate), dose ignored | 0.999 | 299.7 bits |
| **The ruler** | The straight-line fit from the [cold open](../01-cold-open/) | ∞ | ∞ |
| **Berkson's logistic** | $\sigma(-2.96 + 2.30 \ln \text{dose})$, fit by maximum likelihood | 0.633 | 189.9 bits |
| **The oracle** | The table itself | 0.631 | 189.4 bits |

The ruler deserves a moment. Its line, $0.096 + 0.082 \cdot \text{dose}$, predicts 108% deaths at 12 mg, which leaves "survives" with a probability of −8%. Clip it to zero if you like. Two aphids survived the 12 mg tray anyway, and each one costs $-\log_2 0 = \infty$ bits. One event the model called impossible, and the bill is infinite. Cross-entropy is a ruthless accountant: **confidently wrong costs without limit.** That is one more reason [Berkson's logistic curve](../03-berksons-gamble/) won. Its tails come arbitrarily close to 0 and 1 and never touch them.

The whole computation fits on a napkin:

```python
import numpy as np

dose = np.array([1, 2, 3, 5, 8, 12])
dead = np.array([3, 10, 19, 34, 42, 48])        # out of 50 per tray
p = dead / 50                                    # the world: P(dead | dose)

def bill(p, q):     # cross-entropy in bits per aphid (every tray holds 50)
    return np.mean(-(p * np.log2(q) + (1 - p) * np.log2(1 - q)))

q_coin    = np.full(6, 0.5)
q_bookie  = np.full(6, dead.sum() / 300)          # 0.52 everywhere
q_berkson = 1 / (1 + np.exp(2.960 - 2.296 * np.log(dose)))

print(bill(p, q_coin))     # 1.000
print(bill(p, q_bookie))   # 0.999
print(bill(p, q_berkson))  # 0.633
print(bill(p, p))          # 0.631   <- nobody pays less than this
```

## The Overcharge: KL Divergence

Look at the bottom row. The oracle pays 0.631 bits per aphid, and no model can pay less. Every other model pays a premium on top. Splitting the bill makes the premium explicit: add and subtract $\log p$ inside the sum.

$$
H(p, q)
= \underbrace{-\sum_y p(y) \log p(y)}_{H(p)\ =\ \text{fair price}}
\;+\; \underbrace{\sum_y p(y) \log \frac{p(y)}{q(y)}}_{D_{\mathrm{KL}}(p \parallel q)\ =\ \text{overcharge}}
$$

The second term is the **Kullback–Leibler divergence**. It is never negative, and it is zero only when $q = p$ (this is Gibbs' inequality, one application of Jensen's inequality to the convex $-\log$). The intuition: on data generated by $p$, *no other belief beats the truth on average.* Shift probability away from $p$ and you pay extra bits on exactly the outcomes where $p$ was right, and the outcomes you favoured instead don't come often enough to repay you.

```pyplot {id="bill-vs-belief" caption="The bill for one tray as a function of the model's belief q. It is a bowl that touches the fair price H(p) exactly at q = p. The shaded gap above the floor is the KL overcharge."}
def ce(p, q):
    return -(p * np.log2(q) + (1 - p) * np.log2(1 - q))

def berkson(dose):
    return 1 / (1 + np.exp(2.960 - 2.296 * np.log(dose)))

qs = np.linspace(0.004, 0.996, 500)
trays = [("3 mg tray: p = 0.38", 0.38, 3), ("12 mg tray: p = 0.96", 0.96, 12)]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, (title, p, d) in zip(axes, trays):
    bill = ce(p, qs)
    floor = ce(p, p)
    ax.fill_between(qs, floor, bill, color='#FF007F', alpha=0.13,
                    label=r'overcharge  $D_{KL}(p\,\|\,q)$')
    ax.plot(qs, bill, color='#FF007F', linewidth=2.6, label=r'bill  $H(p, q)$')
    ax.axhline(floor, color='#00A8A8', linewidth=2.2, linestyle='--',
               label=rf'fair price  $H(p)$ = {floor:.3f}')
    ax.axvline(p, color='#1A1A1A', linewidth=0.8, linestyle=':')
    pts = [(0.52, "bookie", '#FF8C00', (-20, 55)),
           (berkson(d), "Berkson", '#FFD700', (-75, 70) if d == 12 else (-55, 50))]
    for qq, name, col, off in pts:
        ax.scatter([qq], [ce(p, qq)], s=85, color=col, edgecolor='#1A1A1A',
                   linewidth=1.4, zorder=5)
        ax.annotate(f"{name}: {ce(p, qq):.3f} bits", (qq, ce(p, qq)),
                    textcoords='offset points', xytext=off, fontsize=9,
                    fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='-', color='#1A1A1A', lw=1))
    ax.set_ylim(0, 3.2)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    ax.set_xlabel("model's predicted death rate q")
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel("bits per aphid")
axes[0].legend(loc='upper center', fontsize=8.5, frameon=False)
plt.tight_layout()

for title, p, d in trays:
    q = berkson(d)
    print(f"{title}: H(p)={ce(p, p):.4f}  bookie={ce(p, 0.52):.4f}  Berkson={ce(p, q):.4f}")
```

The bowl is lopsided, and the lopsidedness is the point. On the 12 mg tray, a model that says 50% instead of 96% overpays about three quarters of a bit per aphid. A model that says 4% instead of 96% overpays more than 4. Timidity is cheap and overconfidence is expensive, and the price climbs without bound as $q$ heads toward 0 on an outcome that actually happens. Try it:

{{< infographic title="Fair Price vs. Overcharge" description="Pick a tray, then drag the model's belief. The bill can touch the fair price but never go below it. The gap is the KL overcharge, and the reverse-direction KL disagrees with it everywhere except at the bottom of the bowl." >}}

{{< infographic-controls >}}
  <div class="ig-field">
    <span class="ig-field-label">Which tray? (sets the true death rate p)</span>
    <div class="ig-toggle-row" role="group" aria-label="Tray">
      <button type="button" data-ce-tray="0.06" aria-pressed="false">1 mg</button>
      <button type="button" data-ce-tray="0.38" aria-pressed="true">3 mg</button>
      <button type="button" data-ce-tray="0.68" aria-pressed="false">5 mg</button>
      <button type="button" data-ce-tray="0.96" aria-pressed="false">12 mg</button>
    </div>
  </div>
  <div class="ig-slider">
    <div class="ig-slider-header">
      <span class="ig-slider-label">Model's predicted death rate q</span>
      <output class="ig-slider-readout" id="ig-ce-q-out">52%</output>
    </div>
    <input type="range" id="ig-ce-q" min="1" max="99" step="1" value="52" aria-label="Model's predicted death rate q">
  </div>
{{< /infographic-controls >}}

{{< infographic-viz >}}
  <div class="ig-stage">
    <p class="ig-stage-label" id="ig-ce-caption">Bits per aphid · true death rate p = 38%</p>

    <div class="ig-bar-row">
      <span class="ig-bar-row-label">Fair price H(p)</span>
      <div class="ig-bar">
        <div class="ig-bar-fill ig-bar-fill--teal" id="ig-ce-bar-h" style="width:0%"></div>
      </div>
      <span class="ig-bar-row-value" id="ig-ce-val-h">—</span>
    </div>

    <div class="ig-bar-row">
      <span class="ig-bar-row-label">Bill H(p,q)</span>
      <div class="ig-bar">
        <div class="ig-bar-fill" id="ig-ce-bar-ce" style="width:0%"></div>
        <div class="ig-bar-marker" id="ig-ce-mark" data-label="floor" style="left:0%"></div>
      </div>
      <span class="ig-bar-row-value" id="ig-ce-val-ce">—</span>
    </div>

    <div class="ig-bar-row">
      <span class="ig-bar-row-label">Overcharge KL(p‖q)</span>
      <div class="ig-bar">
        <div class="ig-bar-fill ig-bar-fill--orange" id="ig-ce-bar-kl" style="width:0%"></div>
      </div>
      <span class="ig-bar-row-value" id="ig-ce-val-kl">—</span>
    </div>

    <div class="ig-bar-row">
      <span class="ig-bar-row-label">Reverse KL(q‖p)</span>
      <div class="ig-bar">
        <div class="ig-bar-fill ig-bar-fill--yellow" id="ig-ce-bar-rkl" style="width:0%"></div>
      </div>
      <span class="ig-bar-row-value" id="ig-ce-val-rkl">—</span>
    </div>
  </div>

  <script>
    (function () {
      var SCALE = 7;                       // bits at 100% bar width
      var p = 0.38;
      var slider = document.getElementById('ig-ce-q');
      var buttons = document.querySelectorAll('[data-ce-tray]');
      function lg(x) { return Math.log(x) / Math.LN2; }
      function H(a) { return -(a * lg(a) + (1 - a) * lg(1 - a)); }
      function CE(a, b) { return -(a * lg(b) + (1 - a) * lg(1 - b)); }
      function KL(a, b) { return a * lg(a / b) + (1 - a) * lg((1 - a) / (1 - b)); }
      function pct(bits) { return Math.min(100, bits / SCALE * 100).toFixed(1) + '%'; }
      function fmt(bits) { return Math.max(0, bits).toFixed(3) + ' bits'; }
      function set(key, bits) {
        document.getElementById('ig-ce-bar-' + key).style.width = pct(bits);
        document.getElementById('ig-ce-val-' + key).textContent = fmt(bits);
      }
      function recalc() {
        var q = slider.value / 100;
        document.getElementById('ig-ce-q-out').textContent = slider.value + '%';
        document.getElementById('ig-ce-caption').textContent =
          'Bits per aphid · true death rate p = ' + Math.round(p * 100) + '%';
        set('h', H(p));
        set('ce', CE(p, q));
        set('kl', KL(p, q));
        set('rkl', KL(q, p));
        document.getElementById('ig-ce-mark').style.left = pct(H(p));
      }
      slider.addEventListener('input', recalc);
      buttons.forEach(function (btn) {
        btn.addEventListener('click', function () {
          buttons.forEach(function (b) { b.setAttribute('aria-pressed', 'false'); });
          btn.setAttribute('aria-pressed', 'true');
          p = parseFloat(btn.getAttribute('data-ce-tray'));
          recalc();
        });
      });
      recalc();
    })();
  </script>
{{< /infographic-viz >}}

{{< /infographic >}}

{{< crosshead >}}The constant you will never see{{< /crosshead >}}

Now line up three facts about the decomposition $H(p, q) = H(p) + D_{\mathrm{KL}}(p \parallel q)$. Together they explain why every language model on Earth trains on cross-entropy:

1. **Cross-entropy can be measured.** Average $-\log q$ over real data. No knowledge of $p$ required.
2. **Entropy cannot.** It needs the true $p$, which is Shannon's whole problem. For English it is unknown to this day.
3. **KL is what you actually want to drive to zero**, and you can't measure it directly either, because it contains $\log p(y)$.

The escape hatch: $H(p)$ does not depend on the model's parameters. So

$$
\nabla_\theta\, H(p, q_\theta) = \nabla_\theta\, D_{\mathrm{KL}}(p \parallel q_\theta).
$$

Minimising the quantity you *can* measure minimises the quantity you *care about*. They differ by a constant you will never see. Training on cross-entropy **is** minimising the KL divergence from the data distribution to the model; the training loss just reports it with the unknowable fair price added on top. It is also why Shannon's 1951 measurement could only ever be an upper bound. His predictor's bill included its own overcharge.

The gradient tells the same story. For a softmax output $q = \operatorname{softmax}(z)$ and any target distribution $p$, one-hot or soft,

$$
\frac{\partial H(p, q)}{\partial z} = q - p,
$$

the "prediction minus truth" residual that [the sigmoid gradient chapter](../05-logistic-regression/) derived for one class and [the Softmax Kingdom](../06-softmax-kingdom/) inherited for $K$. It vanishes exactly when $q = p$, which is exactly when the overcharge is zero.

**Where cross-entropy and KL part ways in practice: distillation.** When the target is a teacher model's soft distribution $p_T$ rather than a one-hot label, $H(p_T, q) = H(p_T) + D_{\mathrm{KL}}(p_T \parallel q)$ still holds. The gradients are identical, but the *numbers* are not. A cross-entropy distillation loss bottoms out at the teacher's entropy, which can be a couple of nats on open-ended text, and looks stuck. The KL version bottoms out at zero and tells you how far the student still has to go. That is why distillation code, from {{< cite text="Hinton, Vinyals & Dean (2015)" url="https://arxiv.org/abs/1503.02531" kind="paper" >}} onward, usually logs the KL.

{{% callout type="warning" title="PyTorch's KL is written backwards" %}}
`F.kl_div(input, target)` computes $\sum \text{target} \cdot (\log \text{target} - \text{input})$, and it expects **`input` to already be log-probabilities**. So `F.kl_div(q.log(), p)` is $D_{\mathrm{KL}}(p \parallel q)$: the argument order is reversed from the math. The default `reduction='mean'` divides by *every element* (batch × vocabulary), which turns the true value into a number roughly 50,000× too small on an LLM vocabulary. Pass `reduction='batchmean'`. See the {{< cite text="PyTorch docs" url="https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html" kind="doc" >}}.
{{% /callout %}}

## Kullback's Reading: Evidence per Aphid

The same formula was found a second time, for a completely different reason, by people who had never seen a compressor.

Bletchley Park, 1941. **Alan Turing** and his assistant **I. J. Good** need to decide which of two hypotheses about an Enigma setting is true, using a pile of weak clues. Their bookkeeping is to score each clue by the log of its likelihood ratio and *add the scores up*. The unit is the **ban**, named after the town of Banbury where their scoring sheets were printed, and the working unit is the **deciban**. Good later claimed a deciban is about the smallest shift in belief a human can feel. It is Bayes' rule in exactly the additive form that [the log-odds chapter](../04-log-odds/) taught:

$$
\underbrace{\log \frac{P(H_1 \mid x)}{P(H_2 \mid x)}}_{\text{posterior log-odds}}
\;=\;
\underbrace{\log \frac{P(H_1)}{P(H_2)}}_{\text{prior log-odds}}
\;+\;
\underbrace{\log \frac{p_1(x)}{p_2(x)}}_{\text{weight of evidence}}
$$

Washington, 1951. **Solomon Kullback**, one of the three junior cryptanalysts William Friedman hired in 1930 to found the Army's Signal Intelligence Service, and **Richard Leibler**, a fellow mathematician, are working at the Armed Forces Security Agency, which becomes the NSA the following year. In *On Information and Sufficiency* ({{< cite text="Kullback & Leibler, 1951" url="https://doi.org/10.1214/aoms/1177729694" kind="paper" >}}) they ask the codebreaker's question: if $H_1$ is true, how much evidence against $H_2$ does one observation give me, *on average*? The answer, which they called "the mean information for discrimination":

$$
\mathbb{E}_{x \sim p_1}\!\left[\log \frac{p_1(x)}{p_2(x)}\right] = D_{\mathrm{KL}}(p_1 \parallel p_2).
$$

It is the overcharge formula, symbol for symbol. The two stories have the same shape. Shannon's coder **overpays** because it holds the wrong beliefs; Kullback's detective **collects evidence** because the impostor holds the wrong beliefs. **KL divergence is the rate at which reality testifies against a wrong model.** {{% marginnote label="naming" %}}Kullback and Leibler used the word *divergence* for the symmetrised sum $D(p_1 \parallel p_2) + D(p_2 \parallel p_1)$. The one-directional half took the name later.{{% /marginnote %}}

Make it concrete with the aphids. Someone hands you an unlabelled tray: is it 12 mg or 1 mg? Each dead aphid is worth $10 \log_{10}(0.96 / 0.06) = +12.0$ decibans for "12 mg". Each survivor is worth $10 \log_{10}(0.04 / 0.94) = -13.7$ decibans. If the tray really is 12 mg, you expect

$$
0.96 \times (+12.0) + 0.04 \times (-13.7) = +11.0 \text{ decibans per aphid} = 3.66 \text{ bits} = D_{\mathrm{KL}}(p_{12} \parallel p_1).
$$

Three aphids put you at roughly 2,000 to 1. Now try 3 mg against 5 mg: $D_{\mathrm{KL}} \approx 0.27$ bits, or 0.8 decibans per aphid, so you need around **37 aphids** to reach 1,000 to 1 (30 decibans). KL is *distinguishability*: the number of samples you need to tell two worlds apart is roughly the evidence you want divided by the KL.

{{< crosshead >}}Why it refuses to be symmetric{{< /crosshead >}}

Here KL stops behaving like a distance. Take two trays. **Tray A** kills half its aphids. **Tray B** is the control and kills none.

- If **A is true**, the first dead aphid ends the argument: B gave death probability zero, so one observation carries infinite evidence. $D_{\mathrm{KL}}(A \parallel B) = \infty$.
- If **B is true**, every aphid survives, and each survivor is only mild evidence against A (A gave survival 50%, so one bit each). $D_{\mathrm{KL}}(B \parallel A) = 1$ bit.

One black swan refutes "all swans are white" outright. A thousand white swans only slowly wear down "half of all swans are black". Evidence is asymmetric, so KL is asymmetric. Run the comparison across all six of Bliss's trays and the matrix refuses to be symmetric:

```pyplot {id="tray-kl-matrix" caption="KL divergence in bits between every pair of Bliss's trays. Row = the true tray, column = the impostor. The matrix is not symmetric: telling 1 mg from 12 mg is easier when 1 mg is the truth."}
dose = np.array([1, 2, 3, 5, 8, 12])
p = np.array([3, 10, 19, 34, 42, 48]) / 50

def kl(a, b):
    return a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))

K = np.array([[kl(a, b) for b in p] for a in p])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'cream_pink', ['#FDF5E6', '#FFD700', '#FF8C00', '#FF007F'])
fig, ax = plt.subplots(figsize=(7.2, 6.0))
im = ax.imshow(K, cmap=cmap, vmin=0, vmax=K.max())
labels = [f"{d} mg" for d in dose]
ax.set_xticks(range(6)); ax.set_xticklabels(labels)
ax.set_yticks(range(6)); ax.set_yticklabels(labels)
ax.set_xlabel("impostor q  (the tray you are testing against)")
ax.set_ylabel("truth p  (the tray the aphids come from)")
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{K[i, j]:.2f}", ha='center', va='center', fontsize=9.5,
                fontweight='bold', color='#1A1A1A')
for (i, j) in [(0, 5), (5, 0)]:
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor='#00A8A8', linewidth=3))
ax.set_title(r"$D_{KL}(\mathrm{row}\,\|\,\mathrm{column})$ in bits per aphid",
             fontsize=11, fontweight='bold')
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("bits of evidence per aphid")
plt.tight_layout()

print("KL(1 mg || 12 mg) =", round(K[0, 5], 3), "bits")
print("KL(12 mg || 1 mg) =", round(K[5, 0], 3), "bits")
print("max asymmetry |K - K.T| =", round(np.abs(K - K.T).max(), 3), "bits")
```

## Two Directions, Two Personalities

Because $D_{\mathrm{KL}}(p \parallel q) \neq D_{\mathrm{KL}}(q \parallel p)$, *which direction you minimise* decides what kind of model you get. Ask where each direction blows up:

- **Forward KL**, $D_{\mathrm{KL}}(p \parallel q) = \mathbb{E}_{p}[\log p/q]$, is infinite wherever $q = 0$ but $p > 0$. The model is forced to **cover every outcome reality produces**. This is the mass-covering direction.
- **Reverse KL**, $D_{\mathrm{KL}}(q \parallel p) = \mathbb{E}_{q}[\log q/p]$, is infinite wherever $q > 0$ but $p = 0$. The model is forbidden from **putting mass where reality has none**. This is the mode-seeking direction.

Give a single bell curve the job of imitating a two-humped distribution and the two directions produce opposite characters:

```pyplot {id="forward-vs-reverse-kl" caption="Fitting one Gaussian q to a two-mode p. Forward KL (maximum likelihood) straddles both modes and spends mass in the empty valley. Reverse KL picks one mode and ignores the other."}
x = np.linspace(-8, 8, 4001)
dx = x[1] - x[0]

def log_gauss(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))

p = 0.5 * np.exp(log_gauss(x, -2.2, 0.6)) + 0.5 * np.exp(log_gauss(x, 2.2, 0.6))
logp = np.log(p)

mus = np.linspace(-4, 4, 161)
sigmas = np.linspace(0.3, 3.5, 161)
fwd = np.empty((mus.size, sigmas.size))
rev = np.empty_like(fwd)
for i, mu in enumerate(mus):
    logq = log_gauss(x[None, :], mu, sigmas[:, None])     # (n_sigma, n_x)
    q = np.exp(logq)
    fwd[i] = np.sum(p * (logp - logq), axis=1) * dx
    rev[i] = np.sum(q * (logq - logp), axis=1) * dx

i_f, j_f = np.unravel_index(np.argmin(fwd), fwd.shape)
i_r, j_r = np.unravel_index(np.argmin(rev), rev.shape)
fits = [("Forward KL(p || q): cover everything", mus[i_f], sigmas[j_f], '#FF007F'),
        ("Reverse KL(q || p): pick a mode", mus[i_r], sigmas[j_r], '#00A8A8')]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
for ax, (title, mu, s, col) in zip(axes, fits):
    ax.fill_between(x, p, color='#FFD700', alpha=0.55, label='reality p (two modes)')
    ax.plot(x, p, color='#1A1A1A', linewidth=1)
    ax.plot(x, np.exp(log_gauss(x, mu, s)), color=col, linewidth=3,
            label=f'best single Gaussian q\nμ = {mu:+.2f}, σ = {s:.2f}')
    ax.set_xlim(-6, 6)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8.5, frameon=False, loc='upper left')
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_xlabel("outcome")
plt.tight_layout()

print(f"forward-KL fit: mu={mus[i_f]:+.2f} sigma={sigmas[j_f]:.2f} (KL={fwd[i_f, j_f]:.3f} nats)")
print(f"reverse-KL fit: mu={mus[i_r]:+.2f} sigma={sigmas[j_r]:.2f} (KL={rev[i_r, j_r]:.3f} nats)")
```

Both personalities run inside every chat model you have used:

- **Pretraining is forward KL.** Cross-entropy on data is $H(p) + D_{\mathrm{KL}}(p_{\text{data}} \parallel q)$, so a base model is a professional hedger. It must leave room for *every* continuation that appears anywhere in the data, and a model with finite capacity can only do that by smearing a little probability over the junk in between. A standard explanation for why sampling a base model at temperature 1 drifts into nonsense is that long, thin tail, and it is why truncation samplers such as top-k and top-p exist. See [Issue 05's LM-head chapter](../../05-microgpt/15-sampling/).
- **The RLHF leash is reverse KL.** InstructGPT-style fine-tuning maximises reward minus $\beta \, D_{\mathrm{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$, with the *policy* in the first slot ({{< cite text="Ouyang et al., 2022" url="https://arxiv.org/abs/2203.02155" kind="paper" >}}). That penalty forbids the tuned model from producing what the reference model considers impossible. It does not forbid dropping whole modes the reference liked. Tuned models are noticeably less diverse than their base models, and the direction of the leash is part of the reason.

## The Discount: Mutual Information

Go back to the bookie. It pays 0.999 bits per aphid, and it is not stupid: it knows the overall death rate *perfectly*. Its only failing is that it ignores the dose. The oracle, which reads the dose, pays 0.631. The difference is exactly what knowing the dose is worth:

$$
I(Y; D) = H(Y) - H(Y \mid D) = 0.999 - 0.631 = 0.367 \text{ bits per aphid}.
$$

Here $H(Y \mid D)$, the **conditional entropy**, is the average over trays of the entropy of fate *within* each tray: the fair price once you know the context. The drop from $H(Y)$ to $H(Y \mid D)$ is the **mutual information** between dose and fate. Think of it as the discount the context buys.

Now compute the bookie's overcharge directly, tray by tray. At 1 mg the truth is 6% and the bookie says 52%, an overcharge of $D_{\mathrm{KL}}(0.06 \parallel 0.52) = 0.725$ bits. At 3 mg (38% against 52%) it is only 0.057. At 12 mg it is 0.706. Average the six overcharges and you get **0.367 bits**. That is the discount, to the last digit.

$$
I(Y; D) \;=\; \mathbb{E}_{d}\Big[ D_{\mathrm{KL}}\big(p(y \mid d) \,\parallel\, p(y)\big) \Big] \;=\; D_{\mathrm{KL}}\big(p(d, y) \,\parallel\, p(d)\,p(y)\big)
$$

{{% pullquote type="counter-intuitive" %}}
Mutual information **is** a KL divergence.

**The discount for using the context = the overcharge for ignoring it.**
{{% /pullquote %}}

The second form says it most bluntly: mutual information compares the real joint distribution of (dose, fate) with a *counterfeit* in which the two are independent, and it measures the overcharge you pay for believing the counterfeit.

```pyplot {id="bookie-overcharge" caption="Left: each tray's bill under the bookie, which ignores the dose. Teal is the fair price; pink is the overcharge. Right: averaged over trays, the pink overcharge is exactly the mutual information between dose and fate."}
dose = np.array([1, 2, 3, 5, 8, 12])
p = np.array([3, 10, 19, 34, 42, 48]) / 50
pm = p.mean()                                           # 0.52 — the bookie

def H(a):
    return -(a * np.log2(a) + (1 - a) * np.log2(1 - a))

def kl(a, b):
    return a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))

fair, over = H(p), kl(p, pm)
H_Y, H_YD = H(pm), fair.mean()
MI = H_Y - H_YD

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                               gridspec_kw={'width_ratios': [3.2, 1]}, sharey=True)
xs = np.arange(6)
ax1.bar(xs, fair, color='#00A8A8', edgecolor='#1A1A1A', linewidth=1.8,
        label='fair price  H(fate | this dose)')
ax1.bar(xs, over, bottom=fair, color='#FF007F', edgecolor='#1A1A1A', linewidth=1.8,
        label="bookie's overcharge  KL(p(fate | dose) || 0.52)")
for i in range(6):
    ax1.text(i, fair[i] + over[i] + 0.03, f"+{over[i]:.3f}", ha='center',
             fontsize=9, fontweight='bold', color='#FF007F')
ax1.set_xticks(xs)
ax1.set_xticklabels([f"{d} mg\np = {pp:.2f}" for d, pp in zip(dose, p)])
ax1.set_ylabel("bits per aphid")
ax1.set_ylim(0, 1.25)
ax1.set_title("Tray by tray", fontsize=11, fontweight='bold', loc='left')
ax1.legend(fontsize=8.5, frameon=False, loc='upper center')
ax1.spines[['top', 'right']].set_visible(False)

ax2.bar([0], [H_YD], color='#00A8A8', edgecolor='#1A1A1A', linewidth=1.8, width=0.6)
ax2.bar([0], [MI], bottom=[H_YD], color='#FF007F', edgecolor='#1A1A1A', linewidth=1.8, width=0.6)
ax2.text(0, H_YD / 2, f"H(Y|D)\n{H_YD:.3f}", ha='center', va='center',
         fontsize=10, fontweight='bold', color='white')
ax2.text(0, H_YD + MI / 2, f"I(Y;D)\n{MI:.3f}", ha='center', va='center',
         fontsize=10, fontweight='bold', color='white')
ax2.text(0, H_Y + 0.03, f"H(Y) = {H_Y:.3f}", ha='center', fontsize=9.5, fontweight='bold')
ax2.set_xticks([0]); ax2.set_xticklabels(["average\nover trays"])
ax2.set_xlim(-0.6, 0.6)
ax2.set_title("The bill, itemized", fontsize=11, fontweight='bold', loc='left')
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print(f"H(Y) = {H_Y:.4f}, H(Y|D) = {H_YD:.4f}, I(Y;D) = {MI:.4f}")
print(f"mean bookie overcharge = {over.mean():.4f}  (equal to I(Y;D))")
```

{{< crosshead >}}The symmetry nobody expects{{< /crosshead >}}

Now flip the question around. Hand me a *dead* aphid and ask which tray it came from. Before I see it, the dose is uniform over six trays: $H(D) = \log_2 6 = 2.585$ bits. Once I know it died, the higher doses become more likely and $H(D \mid Y) = 2.218$ bits. The discount is **0.367 bits again**, the same number.

That is not a coincidence. $I(Y; D) = H(Y) + H(D) - H(Y, D)$, which is symmetric by construction. The fate tells you exactly as much about the dose as the dose tells you about the fate. Mutual information is the one quantity in this chapter that doesn't care about direction. That makes it the information theorist's version of correlation, except that it detects *any* dependence, not only linear dependence: it is zero if and only if the two variables are independent.

Put Kullback's reading on it and you get Bliss's whole research question in one number. Each aphid testifies, on average, 0.367 bits in favour of "dose matters" over "dose is irrelevant". Three hundred aphids make 110 bits, a likelihood ratio of about $10^{33}$ to one. {{% marginnote label="statistics" %}}The **G-test** of independence uses exactly this: $G = 2N \cdot \hat{I}$ in nats, here about 153. Pearson's χ² test is its second-order Taylor approximation. Statisticians have been computing mutual information since the 1950s under another name.{{% /marginnote %}} Bliss could have stopped counting long before the three-hundredth aphid.

## Scaling Up: Reading an LLM's Bill

Now swap the nouns. **Dose → context**: every token before this one. **Fate → next token**: one of 50,257. **Trays → every distinct prefix** in the training data. Every quantity carries over unchanged, and each row of the aphid table gets an LLM counterpart:

| Aphid model | LLM counterpart | Loss (nats/token) | Perplexity $e^{\text{loss}}$ |
|---|---|---|---|
| Coin-flipper | Freshly initialised model: uniform over the vocabulary | $\ln 50{,}257 = 10.82$ | 50,257 |
| Bookie | Context-blind unigram model (Zipf's-law estimate) | ≈ 7.6 | ≈ 1,900 |
| Berkson's logistic | GPT-2 small (124M) on OpenWebText | ≈ 3.1 | ≈ 22 |
| Oracle | The irreducible floor: Chinchilla's fitted $E$ | ≈ 1.69 | ≈ 5.4 |

**Perplexity**, $e^{H(p,q)}$, converts the bill into a headcount: the number of equally likely tokens the model behaves as if it were choosing between. Even a perfect model of web text is, on average, torn between about five.

Read the bill top to bottom:

- **10.82 → 7.6: learning the marginal.** The first and steepest drop on almost every LM loss curve is the model learning which tokens are common at all. That is the KL between the unigram distribution and uniform, about 3.3 nats.
- **7.6 → 1.69: the context's discount.** This is $I(\text{next token}; \text{context}) \approx 5.9$ nats, or 8.5 bits per token. Reading the context lets an ideal model rule out all but about 1 in 360 of the options a unigram model keeps open ($e^{5.9} \approx 360$). This is the quantity that pretraining mines, one token at a time.
- **3.1 − 1.69 ≈ 1.4 nats: GPT-2 small's overcharge.** That is $D_{\mathrm{KL}}(p_{\text{data}} \parallel q_{\text{model}})$, everything that more parameters, more data or more training can still win back. GPT-2 XL (1.5B) reaches about 2.54, an overcharge of roughly 0.85. (GPT-2 numbers are from {{< cite text="nanoGPT's baseline table" url="https://github.com/karpathy/nanoGPT" kind="repo" >}}.)

The Chinchilla scaling law ({{< cite text="Hoffmann et al., 2022" url="https://arxiv.org/abs/2203.15556" kind="paper" >}}) is this decomposition written as a forecast:

$$
L(N, D) \;=\; \underbrace{E}_{\text{fair price: } H(\text{text})} \;+\; \underbrace{\frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}}_{\text{overcharge: } D_{\mathrm{KL}}(p_{\text{data}} \parallel q_\theta)}
$$

with a fitted $E = 1.69$ nats per token. Scaling laws forecast how fast the overcharge shrinks with parameters $N$ and tokens $D$. They also promise that nothing ever shrinks the fair price.

Finally, close Shannon's loop. At roughly 4 characters per token, GPT-2 small's 3.1 nats is 4.5 bits per token, or **about 1.1 bits per character**: inside the 0.6–1.3 range Betty Shannon set with a hundred letters of context. Chinchilla's floor of 1.69 nats is 2.4 bits per token, or **about 0.6 bits per character**, right at the bottom of Shannon's 1951 range. (Different alphabets, tokenizers and corpora, so treat this as napkin agreement, not a measurement.) A bound that one mathematician got from another's guesses at a pulp novel turned up again 71 years later in a curve fit to thousands of GPU-days of training runs.

```pyplot {id="two-bills" caption="The same bill at two scales. Every model's loss is the fair price (teal) plus an overcharge. Ignoring the context costs exactly the mutual information (pink). LLM numbers are napkin estimates: a Zipf unigram, nanoGPT's GPT-2 baselines, and Chinchilla's fitted floor."}
# ---- aphids (bits per aphid) ----
p = np.array([3, 10, 19, 34, 42, 48]) / 50
dose = np.array([1, 2, 3, 5, 8, 12])
pm = p.mean()
H = lambda a: -(a * np.log2(a) + (1 - a) * np.log2(1 - a))
kl = lambda a, b: a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))
q_berk = 1 / (1 + np.exp(2.960 - 2.296 * np.log(dose)))
floor_a, mi_a = H(p).mean(), H(pm) - H(p).mean()
aphid_rows = [  # name, [floor, context ignored (= MI), wrong marginal, model's own KL]
    ("coin-flipper", [floor_a, mi_a, kl(pm, 0.5), 0]),
    ("bookie", [floor_a, mi_a, 0, 0]),
    ("Berkson's logistic", [floor_a, 0, 0, kl(p, q_berk).mean()]),
    ("oracle", [floor_a, 0, 0, 0]),
]

# ---- LLM (nats per token) ----
V = 50257
z = 1.0 / np.arange(1, V + 1); z /= z.sum()
H_unigram = -(z * np.log(z)).sum()        # Zipf's-law estimate, ~7.57 nats
E = 1.69                                  # Chinchilla's irreducible loss
llm_rows = [
    ("uniform (init)", [E, H_unigram - E, np.log(V) - H_unigram, 0]),
    ("unigram (context-blind)", [E, H_unigram - E, 0, 0]),
    ("GPT-2 small, 124M", [E, 0, 0, 3.12 - E]),
    ("GPT-2 XL, 1.5B", [E, 0, 0, 2.54 - E]),
    ("floor E", [E, 0, 0, 0]),
]

seg_colors = ['#00A8A8', '#FF007F', '#FFD700', '#FF8C00']
seg_names = ['fair price (entropy)', 'ignoring context = mutual information',
             'wrong marginal (KL vs uniform)', "model's residual KL"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
panels = [(ax1, aphid_rows, "bits per aphid", "Bliss's aphids, 1934", 1.25, "{:.3f}"),
          (ax2, llm_rows, "nats per token", "Next-token prediction, 2019–2022", 12.8, "{:.2f}")]
for ax, rows, unit, title, xmax, fmt in panels:
    ax.set_xlim(0, xmax)
    for r, (name, segs) in enumerate(rows):
        left = 0
        for k, w in enumerate(segs):
            if w > 1e-9:
                ax.barh(r, w, left=left, color=seg_colors[k], edgecolor='#1A1A1A',
                        linewidth=1.5, height=0.62)
            left += w
        ax.text(left + 0.015 * xmax, r, fmt.format(left), va='center',
                fontsize=9.5, fontweight='bold')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel(unit)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    ax.spines[['top', 'right']].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='#1A1A1A') for c in seg_colors]
fig.legend(handles, seg_names, loc='lower center', ncol=4, fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0, 0.06, 1, 1])

print(f"aphids: floor={floor_a:.3f} MI={mi_a:.3f} Berkson KL={kl(p, q_berk).mean():.4f} bits")
print(f"LLM: lnV={np.log(V):.2f} unigram={H_unigram:.2f} MI(context)={H_unigram - E:.2f} nats")
```

## Where They Differ

Four quantities, one logarithm. Here is where they part company:

| | Entropy $H(p)$ | Cross-entropy $H(p,q)$ | KL divergence $D_{\mathrm{KL}}(p \parallel q)$ | Mutual information $I(X;Y)$ |
|---|---|---|---|---|
| **Question it answers** | How unpredictable is the world? | What does my model pay? | How much of that is my model's fault? | How much does $Y$ tell me about $X$? |
| **Built from** | One distribution | Data distribution + model | Data distribution + model | One *joint* distribution over two variables |
| **Floor** | 0 (a certain outcome) | $H(p)$ | 0, only when $q = p$ | 0, only when $X$ and $Y$ are independent |
| **Symmetric?** | — | No | No, and the direction sets the personality | **Yes** |
| **Measurable from samples alone?** | No | **Yes**: average $-\log q$ | No, but its gradient equals cross-entropy's | No, needs density estimates or bounds |
| **Metaphor** | The fair price | The bill | The overcharge | The discount |
| **Where you meet it in LLMs** | Chinchilla's $E$; the floor of every loss curve | The training loss; perplexity $= e^{H(p,q)}$ | Distillation, the RLHF leash, quantization evaluation | What context buys; word2vec's PMI; contrastive losses |

Three differences bite in practice:

1. **Measurable versus meaningful.** Cross-entropy is the only one of the four you can compute from data without knowing the true distribution, which is why it is the loss. KL is the one that means "how wrong is my model", which is why it is the *target*. They differ by a constant, so you optimise one and get the other.
2. **Direction.** Cross-entropy and KL are asymmetric, and the asymmetry has consequences: forward KL produces hedgers, reverse KL produces specialists. Mutual information is symmetric. The dose tells you about the fate exactly as much as the fate tells you about the dose.
3. **What gets compared with what.** Cross-entropy and KL compare *two distributions over the same variable* (reality versus model). Mutual information compares *one joint distribution with a counterfeit of itself*: the same marginals, with the dependence deleted.

{{% callout type="theorem" title="One logarithm, four averages" %}}
- **Entropy**, the fair price: $\;H(p) = \mathbb{E}_{x \sim p}\big[-\log p(x)\big]$
- **Cross-entropy**, the bill: $\;H(p, q) = \mathbb{E}_{x \sim p}\big[-\log q(x)\big]$
- **KL divergence**, the overcharge: $\;D_{\mathrm{KL}}(p \parallel q) = \mathbb{E}_{x \sim p}\big[\log \tfrac{p(x)}{q(x)}\big] = H(p, q) - H(p)$
- **Mutual information**, the discount: $\;I(X; Y) = \mathbb{E}_{(x,y) \sim p}\big[\log \tfrac{p(x, y)}{p(x)\,p(y)}\big] = D_{\mathrm{KL}}\big(p(x,y) \parallel p(x)\,p(y)\big) = H(X) - H(X \mid Y)$

All four are an expected value of a surprise or a difference of surprises. Change what you average over, or whose probabilities sit in the log, and you move from one to the next.
{{% /callout %}}

## Connections

- **Word2vec factorises PMI.** The pointwise term inside mutual information, $\log \frac{p(w, c)}{p(w)\,p(c)}$, is the PMI that [Issue 01's one-hot chapter](../../01-word2vec/02-one-hot-problem/) mentions as a pre-neural fix. Levy & Goldberg (2014) showed that [skip-gram with negative sampling](../../01-word2vec/04-word2vec/) is implicitly factorising a shifted PMI matrix. Word embeddings are a compressed table of pointwise mutual information.
- **Rate–distortion is minimum mutual information.** Shannon's lossy-compression floor, $R(D) = \min I(X; \hat{X})$, is the subject of [Issue 03's rate-distortion chapter](../../03-quantization/04-rate-distortion/). Quantized LLMs are graded by the KL between full-precision and quantized output distributions.
- **Contrastive learning bounds mutual information.** InfoNCE ({{< cite text="van den Oord et al., 2018" url="https://arxiv.org/abs/1807.03748" kind="paper" >}}), and with it CLIP-style training, minimises a loss that satisfies $I(X; Y) \ge \log N - \mathcal{L}_{\text{InfoNCE}}$ for a batch of $N$. Lowering the loss raises a lower bound on mutual information.
- **DPO's reward is a weight of evidence.** Direct Preference Optimization ({{< cite text="Rafailov et al., 2023" url="https://arxiv.org/abs/2305.18290" kind="paper" >}}) solves the reverse-KL-regularised RLHF objective in closed form, and its implicit reward is $\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$. That is a log-likelihood ratio, measured in Turing's units.
- **Bliss's question was mutual information all along.** "Does dose affect mortality?" means "is $I(Y; D) > 0$?", and the G-test measures exactly that.

## The Bill, Itemized

Back to the line on the training dashboard: `loss 3.1172`.

It is a **bill**, in nats per token. About **1.7** of it is the language's fault: the fair price of English being English, which Betty Shannon couldn't beat in 1951 and no model will beat either. About **1.4** is the model's fault: the KL overcharge, the part that scaling laws promise to shrink. And the roughly **5.9 nats** the model *doesn't* pay, the gap between it and a context-blind bookie, is mutual information. It is the discount mined out of the context, and it is the reason the model reads the context at all.

Cross-entropy is what you can measure, entropy is what you can't escape, KL is what you can fix, and mutual information is what the context is worth. It is one logarithm the whole way down: the surprise, $-\log q$, averaged four different ways.

---

**Continue to** → [Issue 03: Rate-Distortion, the curve that unifies MP3, JPEG, and LLMs](../../03-quantization/04-rate-distortion/), where Shannon turns the discount inside out. If mutual information is what the context is worth, what is the *least* information you can keep about an LLM's weights before the model forgets what it knows?
