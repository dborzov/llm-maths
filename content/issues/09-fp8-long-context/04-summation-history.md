---
title: "Kahan Summation: a 1965 fix inside your 2026 GPU"
description: "Floating-point addition is non-associative. The error grows with n. Wilkinson named it in 1960; Kahan fixed it in 1965. The same class of bug lived in vLLM's attention kernel."
blurb:
  - "(a + b) + c ≠ a + (b + c) in floating point. This is always true."
  - "Sum n numbers naively: error grows as O(n·ε). At n = 128,000, that error swamps the result."
  - "James Wilkinson at NPL named it in 1960. William Kahan fixed it in 1965."
  - "The same class of bug sat in vLLM's FP8 attention kernel — waiting for context windows to grow long enough."
topics: [numerical-analysis, primer]
tags: [kahan-summation, wilkinson, ieee754, floating-point, compensation]
theme: teal
math: true
draft: false
date: 2026-05-17T09:00:00-04:00
issue: 9
weight: 40
techKind: primer
techNode: summation
header: default.webp
---

## A Bug Older Than The Chip It Lives On

Before the accumulator bug in FP8 attention, there was the same bug, on a different machine, in 1962. Before that, the same bug, on a different machine, in 1955. The bug is older than the computer it currently lives on. It is older than the IEEE 754 standard. It is older than transistors. It is, in some sense, **as old as floating point itself** — which is to say, about as old as electronic computing.

This primer exists for a single purpose: by the time you finish it, you should understand the family of bugs in which the FP8 attention accumulator lives, and you should know the family of algorithms that fixes them. The next chapter ([The Accumulator Lie](../05-accumulator-lie/)) walks the specific bug in detail. To follow that argument, you need to know two facts about floating-point addition that often surprise people who have only thought about it casually.

**Fact 1.** Floating-point addition is **not associative**: $(a + b) + c$ can produce a different result than $a + (b + c)$.

**Fact 2.** When you sum $n$ numbers naively into a running accumulator, the **error grows like $O(n \cdot \epsilon)$** where $\epsilon$ is the machine precision. For $n$ in the hundreds of thousands and $\epsilon$ in the $10^{-7}$ neighborhood, the error becomes the same order of magnitude as the sum you're trying to compute.

Both of these facts have been known since at least the late 1950s. The first widely-cited reference is **James Wilkinson** at the **National Physical Laboratory** in Teddington, England. The second — and the algorithm that fixes the first — is **William Kahan**'s 1965 *Communications of the ACM* note.

The argument in this primer is that the [vLLM FP8 KV bug](../01-cold-open/) is a 21st-century instance of the same problem Wilkinson and Kahan named six decades ago, and the fix that finally landed in vLLM in April 2026 is, algebraically, a 21st-century version of the same algorithm Kahan published in 1965.

## Teddington, 1960

**James Hardy Wilkinson** runs the numerical analysis group at the National Physical Laboratory. He has spent the entire decade since 1947 working on **ACE**, Alan Turing's first major British computer design, and on its successor, **DEUCE**. Most of his job is, in modern terms, *numerical software engineering*: writing routines to solve linear systems, eigenvalue problems, polynomial roots. The routines have to be fast, robust, and — most importantly — accurate, on a machine whose arithmetic unit makes mistakes you can quantify but not avoid.

Wilkinson's central insight, which becomes the foundation of modern numerical analysis, is this:

{{% pullquote type="profound" author="J.H. Wilkinson, c. 1960" %}}
**Backward error analysis.** Don't ask "how much error did the algorithm introduce into the answer?" Ask: "what is the smallest perturbation of the *input* that would make the algorithm's actual answer correct?" If that perturbation is small, the algorithm is well-behaved, regardless of how big the forward error is.
{{% /pullquote %}}

The reframing is brilliant because it isolates the *algorithm's* contribution to error from the *problem's* sensitivity to error. A perfectly correct algorithm on an ill-conditioned problem can produce a wildly wrong answer; backward error analysis says "the algorithm did its job, the problem just demands more precision than you've got". An incorrect algorithm on a well-conditioned problem produces a wrong answer the algorithm has no excuse for.

Wilkinson works through hundreds of small numerical routines through the 1950s and 1960s and assigns each one a *backward error bound*. The routines that come out well-bounded become the workhorses of every later numerical library — LINPACK, LAPACK, every Fortran math library you have ever called into. The routines that come out badly-bounded get rewritten.

One of the routines Wilkinson catalogues has the simplest possible form: take a list of $n$ numbers and add them up. The conclusion he reaches about it is the seed for everything in this primer.

## The Bug In Naive Summation

Here is the routine, in modern Python:

```python
def naive_sum(xs):
    s = 0.0
    for x in xs:
        s = s + x
    return s
```

That looks like it works. It does work — *for most inputs*. The reason it doesn't always work is a property of floating-point addition that you have to see explicitly to believe.

In IEEE 754 floating point, every number is represented as $\pm m \cdot 2^e$ where $m$ is a fixed-width mantissa. When you add two numbers with very different exponents, you first **align** them — shifting the smaller number's mantissa right by the exponent difference — and *the bits that fall off the right end of the mantissa are gone*. They are not stored anywhere. They are not recoverable. The sum proceeds as if those bits were zero.

Concretely. Suppose you are accumulating in FP32 (24-bit mantissa, $\epsilon \approx 6 \times 10^{-8}$). Your running sum is around $10^7$. You add a number that's around $1.0$. To do the addition, you shift the $1.0$ right by about 23 bits — which is the *entire mantissa* — and you have added zero. The number you wanted to include in the sum has been silently dropped.

```pyplot {id="naive-sum-drift" caption="Naive vs Kahan summation, summing one hundred million copies of 1.0 in FP32. The naive sum stops growing once the partial sum overshoots the per-add precision; Kahan stays linear. This is the bug we're chasing in the attention accumulator."}
import numpy as np
import matplotlib.pyplot as plt

n = 10_000_000  # ten million 1.0's
xs = np.ones(n, dtype=np.float32)

# Track the running sum at exponential checkpoints
checkpoints = np.unique(np.logspace(2, np.log10(n), 50).astype(int))
naive = []
kahan = []
exact = []

s = np.float32(0.0)
for k, x in enumerate(xs, start=1):
    s = s + x
    if k in checkpoints:
        naive.append((k, float(s)))

# Kahan compensated summation
s = np.float32(0.0)
c = np.float32(0.0)
for k, x in enumerate(xs, start=1):
    y = x - c
    t = s + y
    c = (t - s) - y
    s = t
    if k in checkpoints:
        kahan.append((k, float(s)))

exact = [(k, float(k)) for k in checkpoints]

fig, ax = plt.subplots(figsize=(9.5, 4.8))
nk, nv = zip(*naive)
ax.plot(nk, nv, color='#FF007F', linewidth=2.4, marker='o', markersize=5,
        markerfacecolor='#FFD700', markeredgecolor='#1A1A1A',
        label='naive sum (FP32)')
kk, kv = zip(*kahan)
ax.plot(kk, kv, color='#00A8A8', linewidth=2.4, marker='s', markersize=5,
        markerfacecolor='#FFD700', markeredgecolor='#1A1A1A',
        label='Kahan summation (FP32)')
ek, ev = zip(*exact)
ax.plot(ek, ev, color='#FFD700', linewidth=1.5, linestyle='--',
        label='exact (each add really is +1)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('number of 1.0’s summed so far')
ax.set_ylabel('running partial sum')
ax.set_title('Naive FP32 summation stalls; Kahan keeps tracking the exact value',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper left', framealpha=1, edgecolor='#1A1A1A')
ax.grid(True, alpha=0.15, which='both')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

Stare at this picture for a moment. Both curves should sit exactly on the dashed yellow line — that's what summing $k$ copies of $1.0$ ought to produce. Naive FP32 summation tracks the line until around $k = 10^7$, then **stops growing**. After that, every additional $1.0$ you try to add is silently dropped, because the partial sum has grown large enough that $1.0$ falls off the end of the mantissa during alignment. The error is no longer "a small percentage of the answer". The error is *the answer*. You added ten million ones and the accumulator says you added eight million.

That is the failure mode the attention accumulator inherits at long context. Same algebraic structure, different numbers.

## Kahan, 1965

**William Kahan** is a young mathematician at the University of Toronto in 1965 when he publishes a five-page note in *Communications of the ACM* titled, with characteristic understatement, *"Further Remarks on Reducing Truncation Errors."* The paper proposes the following modification to naive summation:

```python
def kahan_sum(xs):
    s = 0.0  # running sum
    c = 0.0  # running compensation (the bits we lost last time)
    for x in xs:
        y = x - c            # apply previously-lost bits
        t = s + y            # the imprecise sum
        c = (t - s) - y      # capture what was lost in this add
        s = t
    return s
```

Read those four lines carefully. They are doing something almost magical and, when you see it for the first time, slightly mysterious.

- `y = x - c`. We bias the incoming value by `c`, which stores the bits we lost on the previous iteration. So we're really adding `(x - c)` to the running sum.
- `t = s + y`. The imprecise add — same as the naive version.
- `c = (t - s) - y`. Here is the trick. *Mathematically*, `(t - s)` is just `y`. But in floating point, when `s` is big and `y` is small, `(t - s)` is the *truncated* version of `y` — the part that actually survived the addition. So `(t - s) - y` is the *negative of what was lost*. The next iteration will subtract this from the incoming value, effectively re-injecting the lost bits.
- `s = t`. The running sum gets updated.

```python
# Trace one Kahan step where s is big and y is small
s = 1e7              # running sum
y = 1.0              # we're about to add 1
t = s + y            # FP32: rounds to 10000001.0 if mantissa lets it, else 1e7
# what did we actually add?
just_added = t - s   # FP32: maybe 1.0, maybe 0.0
# what should we have added?
should_have = y      # 1.0
# the lost amount, with sign:
lost = should_have - just_added   # 0.0 if we got lucky, otherwise 1.0
c = -lost            # store as the next iteration's correction
```

The compensation variable `c` is a *running record of arithmetic mistakes*. Each iteration finds and corrects last iteration's mistake. The result is that errors don't accumulate — they get neutralized. The error bound on Kahan summation is $O(\epsilon)$ regardless of how many terms you add: the algorithm pays a constant-bounded error per element instead of an error that grows with $n$.

Kahan's paper goes further. He proves that for the sum of $n$ values, the **forward error** of his algorithm is bounded by a constant times machine epsilon, independent of $n$. The naive algorithm's forward error is $O(n \epsilon)$. For $n = 10^5$ and FP32 $\epsilon \approx 6 \times 10^{-8}$, the naive error bound is $\sim 6 \times 10^{-3}$ — *three orders of magnitude* worse than Kahan's bound. For the contractions inside long-context attention, this is precisely the gap that explains why a 91% benchmark becomes 13%.

Kahan would later go on to chair the IEEE 754 floating-point standards committee. The standard he shepherded — adopted in 1985, still in use today on every CPU and GPU — was designed in significant part to make algorithms like compensated summation *implementable*. The fact that floating-point rounding is well-defined and reproducible is what makes the `c = (t - s) - y` line above provably correct. He received the **Turing Award** in 1989 for this work. {{< wiki "kahan" >}}William Kahan{{< /wiki >}} is the person you want to thank every time a serious numerical routine actually produces a correct answer on a sum of millions of values.

{{% callout type="info" title="The Algebraic Identity" %}}
The Kahan correction $c = (t - s) - y$ exploits an algebraic identity that is *exact* in floating point: when $|y| \le |s|$, the difference between the imprecise sum $t = s + y$ and the true sum $s + y$ is exactly the part of $y$ that got truncated during alignment. The IEEE 754 standard guarantees this — round-to-nearest-even rounding ensures that the operation $(t - s)$ recovers the truncated value bit-for-bit.

This is why Kahan summation is **exact in a specific structural sense**, not just an empirical heuristic. The compensation `c` is the mathematically-correct correction, computable from the operations the floating-point unit just performed.
{{% /callout %}}

## A Worked Example Pulled From The Attention Inner Loop

To anchor this to the bug we're chasing, here is a stripped-down model of what happens inside the FA3 FP8 attention kernel's inner loop. The actual kernel is much more complex; this is the algebraic skeleton.

```python
import numpy as np

def naive_attention_score(q, K, scale):
    """Compute one row of QK^T with naive accumulation. Lossy for large T."""
    T, D = K.shape
    scores = np.zeros(T, dtype=np.float32)
    for j in range(T):
        # The "tensor core" of this toy is the dot product q . k_j
        # We're simulating loss of precision by accumulating into FP16
        # instead of FP32. Real bug: even FP32 accumulator can lose
        # precision when summing >100k terms.
        s = np.float16(0.0)        # imprecise accumulator
        for d in range(D):
            s = s + np.float16(q[d] * K[j, d])
        scores[j] = float(s) * scale
    return scores

def kahan_attention_score(q, K, scale):
    """Same thing, with compensated summation in the inner product."""
    T, D = K.shape
    scores = np.zeros(T, dtype=np.float32)
    for j in range(T):
        s = np.float16(0.0)
        c = np.float16(0.0)
        for d in range(D):
            y = np.float16(q[d] * K[j, d]) - c
            t = s + y
            c = (t - s) - y
            s = t
        scores[j] = float(s) * scale
    return scores

# Quick check: long contraction
np.random.seed(0)
D = 4096            # long contraction dimension (simulating long context)
q = np.random.randn(D).astype(np.float32)
K = np.random.randn(1, D).astype(np.float32)

scores_naive = naive_attention_score(q, K, 1.0)
scores_kahan = kahan_attention_score(q, K, 1.0)
scores_exact = (q @ K.T)

print(f"exact:  {scores_exact[0]:.6f}")
print(f"naive:  {scores_naive[0]:.6f}   error = {abs(scores_naive[0]-scores_exact[0]):.6f}")
print(f"kahan:  {scores_kahan[0]:.6f}   error = {abs(scores_kahan[0]-scores_exact[0]):.6f}")
```

On a typical run of that toy, the naive accumulator loses a percentage point or two of the dot product's value. The Kahan accumulator nails it to the last bit of the underlying FP16 mantissa. The kernel that ships in FA3 FP8 is the moral equivalent of `naive_attention_score` (with FP8 multiplies and a not-quite-FP32 accumulator). The fix that lands in April 2026 is the moral equivalent of `kahan_attention_score` — though the actual implementation is *block-Kahan* or *two-level accumulation*, which we will unpack in [Chapter 6](../06-two-level-fix/).

The structure of the algebraic fix is identical to Kahan 1965. Only the tile structure and the register placement changed.

## Two-Level Accumulation: Kahan's Idea, Hardware-Tiled

The specific variant the team uses, from **SageAttention2**, is sometimes called *two-level accumulation* or *block-corrected summation*. The idea is a small generalization of Kahan: instead of compensating after every single add (which would slow the tensor core to a crawl), you accumulate into the fast tensor-core register for a *block* of operations, then periodically promote the block accumulator to a true FP32 register and reset.

```python
def two_level_attention_score(q, K, scale, block_size=64):
    """SageAttention2-style two-level accumulation: inner loop uses fast
    (low-precision) accumulator, periodically promotes to FP32."""
    T, D = K.shape
    scores = np.zeros(T, dtype=np.float32)
    for j in range(T):
        s_outer = np.float32(0.0)         # true FP32 register
        d = 0
        while d < D:
            # Inner block: use fast (FP16) accumulator
            s_inner = np.float16(0.0)
            for di in range(min(block_size, D - d)):
                s_inner = s_inner + np.float16(q[d + di] * K[j, d + di])
            # Promote to FP32 outer accumulator and reset
            s_outer += float(s_inner)
            d += block_size
        scores[j] = s_outer * scale
    return scores
```

The inner block runs fast — same number of cycles as the naive kernel, because the inner accumulator is in the same fast register. The outer accumulator runs in a true FP32 register, but only updates once per block. The cost: one extra register held permanently across the loop, plus one promote-and-add per block. The benefit: errors compound only over the inner block (small $n$), not over the full contraction (huge $n$).

This is the algorithm that landed in [`flash-attention#104`](https://github.com/vllm-project/flash-attention/pull/104) in April 2026 and brought needle-in-a-haystack accuracy from 13% back to 89%. The next chapter goes into why the original FA3 FP8 path needed this and what specifically broke without it. For now, the punchline:

{{% pullquote type="profound" %}}
The fix that solved the vLLM FP8 long-context regression in 2026 is, structurally, the same idea William Kahan published in 1965 for an entirely different reason: when you accumulate many small floating-point numbers into a running sum, the imprecise sum is fixable by storing the imprecision and re-injecting it. The 1965 trick lives inside the 2026 kernel.
{{% /pullquote %}}

## A Brief Aside: Why The Problem Did Not Go Away With Bigger Floats

A reasonable reaction to all of this is: *can't you just accumulate in a wider type?* If FP32 accumulation loses precision, accumulate in FP64. If FP64 loses precision, accumulate in some arbitrary-precision software type.

This works for some problems but is the wrong move for high-performance attention kernels, for two reasons.

1. **Wider accumulators cost silicon area and latency.** A tensor core that natively accumulates in FP32 already costs you roughly twice the chip area per multiply-accumulate as one that accumulates in FP16. An FP64 accumulator would roughly quadruple again. Hopper's FP8 tensor cores were designed specifically to *not* have to widen the accumulator beyond FP32, because the area savings is most of the point.
2. **FP64 doesn't fully solve the problem either.** It pushes the precision-loss threshold out by a factor of $2^{29}$ (the mantissa difference), which is huge — but at long enough context, you'd hit the same wall, just slower. Compensated summation removes the asymptotic dependence on $n$, which is the right fix in principle.

The point of Kahan-style compensated summation is that it gives you the *good asymptotic behavior* (error independent of $n$) with the *good silicon footprint* (a fast register that does most of the work) plus a small constant overhead. This is why the algorithm has lived in numerical libraries continuously since 1965 and keeps reappearing every time a new piece of hardware introduces a faster but slightly imprecise multiply-accumulate path.

## What To Remember

1. **Floating-point addition is non-associative.** $(a + b) + c \ne a + (b + c)$ in general. The bits that fall off the right end of the mantissa during alignment are lost.
2. **Naive summation has error growing like $O(n \epsilon)$.** For long contractions in low-precision floats, this becomes the same order of magnitude as the answer.
3. **Kahan compensated summation** stores the bits you lost on the previous add and re-injects them on the next add. Error bound becomes $O(\epsilon)$ independent of $n$.
4. **Two-level (block) accumulation** is a hardware-friendly variant: fast inner accumulator in a tensor-core register, slow outer accumulator in a true FP32 register, with periodic promotion between them. This is what landed in vLLM in April 2026.
5. **The structure of the fix is sixty years old.** The bug found by the AWS / Red Hat team in 2026 is the same family of bug Wilkinson catalogued at NPL in 1960; the algorithm that fixes it is the same family of algorithm Kahan published in 1965. The new wrinkle is what register the inner accumulator lives in.

**Continue to** → [The Accumulator Lie](../05-accumulator-lie/) — the autopsy of the FP8 FA3 inner loop, why its FP32 accumulator isn't really FP32 in the way you expect, and how the team traced the precision loss back to a specific tensor-core operation Hopper has been shipping since 2023.
