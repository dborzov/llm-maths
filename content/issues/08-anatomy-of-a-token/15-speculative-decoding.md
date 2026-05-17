---
title: "The Draft Trick"
description: "Decode is so bandwidth-bound that verifying 32 candidate tokens costs the same memory pass as verifying 1. Run a small draft model to speculate several tokens ahead, then let the target model verify all of them in a single forward pass — a 2–3× throughput gain if the draft is even modestly accurate."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T16:00:00-04:00
issue: 8
weight: 150
techKind: mainline
techNode: speculative-decoding
header: 15-speculative-decoding.webp
---

## A DeepMind Whiteboard, November 2022

In **November 2022**, three researchers at **Google DeepMind** in London — **Yaniv Leviathan**, **Matan Kalman**, and **Yossi Matias** — uploaded a short, almost casually-titled paper to arXiv: *"Fast Inference from Transformers via Speculative Decoding"* ({{< cite text="Leviathan, Kalman, Matias (2022)" url="https://arxiv.org/abs/2211.17192" kind="paper" >}}). It was thirteen pages including appendices. Less than a hundred lines of pseudocode. It described a trick that, on its face, should have been impossible: a way to make a large autoregressive transformer **emit more than one token per forward pass** — without changing its weights, without changing its output distribution, without retraining.

Three months later, at almost exactly the same moment, **Charlie Chen, Sebastian Borgeaud and collaborators** at DeepMind posted a near-identical paper ({{< cite text="Chen, Borgeaud et al. (2023)" url="https://arxiv.org/abs/2302.01318" kind="paper" >}}). The two papers had been developed in parallel inside the same building, by different teams, in the way that happens when an idea is genuinely *in the water*. The trick had a name now — **speculative decoding** — and it would dominate the inference literature for the next two years.

The trick rests on something every reader of this issue has already absorbed but probably has not yet *felt*. Decode-time forward passes are **bandwidth-bound** — see [Two Phases, Two Personalities](../07-prefill-vs-decode/). The {{< wiki "transformer-weights" >}}weight matrices{{< /wiki >}} of a 70B model take roughly 70 GB of HBM bandwidth to load once. The single token's worth of compute that you do *with* those weights — the actual matmul FLOPs — finishes long before the load. The tensor cores spend most of the forward pass watching the HBM bus drain. If you could *use* those idle FLOPs — if you could ask the same forward pass to do meaningful work on **several** candidate tokens at once, while the weights are already in flight — you would get those extra tokens almost free.

That is the entire trick. The rest is plumbing.

To make it concrete: Llama-3.1-8B at batch 4 on an H100, in BF16, does one decode step in about **18 ms**. With **EAGLE-3 speculation** at $K=4$ candidate draft tokens, on conversational workloads where the draft model gets ~70% of its guesses right, that same model does what it would have done in four decode steps in about **31 ms** total. Per-token: **7.8 ms**. A **2.3× speedup** on a workload the GPU was barely using. No model surgery, no quality loss, no quantization, no architectural change to the target model — *just letting it work harder while it was already in the act of loading itself*.

The mystery is not why this works. The mystery is why we let the tensor cores sit idle for the first three years after GPT-3 shipped.

## The Roofline, Re-Stated With $K$

In [The Roofline](../04-roofline/) we drew the diagonal that separates the **memory-bound** regime from the **compute-bound** regime. The ridge point for an H100 SXM (1.98 PFLOPs in BF16, 3.35 TB/s of HBM) sits at roughly $I = 590$ FLOP/byte. Decode at batch 1 lives at $I \approx 1$. The gap is 600×; the GPU is *barely awake*.

Speculative decoding raises $I$ in the most direct way imaginable. A decode step over $K$ tokens of the same sequence loads the same weights and computes $K\times$ as many FLOPs, because each weight is now used for $K$ outputs instead of one. The bytes loaded scale roughly linearly in $K$ for the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} (you read $K$ extra K/V vectors per layer) but not for the model weights — those are loaded once and reused. To a first approximation:

$$
I(K) \;\approx\; \frac{K \cdot F_\text{per-token}}{B_\text{weights} + K \cdot B_\text{kv-per-token}}
\;\to\; \frac{F_\text{per-token}}{B_\text{kv-per-token}}\;\; \text{as } K \text{ grows.}
$$

For Llama-70B at decode batch 4, the weights term dominates until $K$ gets around 90. So for any speculation count $K \lesssim 90$, **adding draft tokens makes the GPU work harder, not the HBM bus.** Step latency goes up only marginally — the extra FLOPs hide inside the bandwidth-stall. You are buying tokens with FLOPs the GPU was throwing away.

```pyplot {id="spec-decode-roofline" caption="ARITHMETIC INTENSITY VS K — ADDING DRAFT TOKENS PUSHES DECODE RIGHTWARD ALONG THE H100 ROOFLINE. UNDER THE RIDGE, EXTRA TOKENS ARE NEARLY FREE."}
np.random.seed(2)

# Llama-70B-ish: weights ~70 GB, decode-step FLOPs per token ~140 GFLOPs.
# H100 ridge ~ 590 FLOP/byte, peak ~1.98 PFLOPs BF16.
F_per_token = 140e9             # FLOPs
B_weights   = 70e9              # bytes (loaded once per step regardless of K)
B_kv_token  = 2e7               # bytes per extra K/V vector across all layers
peak_flops  = 1.98e15
hbm_bw      = 3.35e12
ridge       = peak_flops / hbm_bw

K = np.arange(1, 96)
I = (K * F_per_token) / (B_weights + K * B_kv_token)

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.semilogy(K, I, '-o', color='#FF007F', linewidth=2,
            markersize=4, markerfacecolor='#FF007F',
            markeredgecolor='#1A1A1A', label='I(K) for Llama-70B decode')
ax.axhline(ridge, color='#1A1A1A', linestyle='--',
           linewidth=1.5, label=f'H100 BF16 ridge ≈ {ridge:.0f}')
ax.fill_between(K, 1, ridge, color='#FFD700', alpha=0.15)
ax.fill_between(K, ridge, 5000, color='#FF8C00', alpha=0.15)
ax.text(3, 6, "memory-bound regime\n(extra K is nearly free)", fontsize=10)
ax.text(3, 1200, "compute-bound regime\n(extra K costs FLOPs)", fontsize=10)
ax.set_xlabel("K  (candidate tokens verified per step)")
ax.set_ylabel("arithmetic intensity (FLOP/byte)")
ax.set_title("Verifying K tokens raises arithmetic intensity — but stays under the ridge for K ≲ 90",
             fontsize=11, loc='left')
ax.set_ylim(1, 5000)
ax.set_xlim(0, 96)
ax.legend(loc='lower right')
ax.spines[['top','right']].set_visible(False)

print(f"H100 BF16 ridge point: {ridge:.0f} FLOP/byte")
print(f"Llama-70B decode arithmetic intensity:")
for kk in [1, 2, 4, 8, 16, 32, 64]:
    val = (kk * F_per_token) / (B_weights + kk * B_kv_token)
    regime = "memory-bound" if val < ridge else "COMPUTE-BOUND"
    print(f"  K={kk:>2}: I = {val:7.2f} FLOP/byte  ({regime})")
plt.tight_layout()
```

The plot tells a clean story. At $K = 1$, decode lives deep in the memory-bound regime — $I \approx 2$ FLOP/byte, six hundred times *under* the H100 ridge. As $K$ grows, $I$ climbs roughly linearly. By the time $K \approx 90$, $I$ touches the ridge and extra draft tokens are no longer free — you've crossed into compute-bound territory and additional candidates cost FLOPs the same way prefill does. **Every speculation method in production picks $K$ well to the left of that boundary**, typically $K = 4$ to $K = 8$, for reasons we'll see in a moment.

{{% pullquote type="counter-intuitive" %}}
The hardest thing about decode is not that the model is slow. The hardest thing is that the model is fast and the memory bus is slow. Speculative decoding lets the model speak while the bus is still inhaling.
{{% /pullquote %}}

## The Two-Model Setup

Concretely, speculative decoding needs two pieces:

- A **target model** $M$ — the real, expensive transformer you want output from. Llama-3.1-8B in our anchor case, or Llama-70B in production.
- A **draft model** $m$ — a much smaller, much faster transformer trained on similar data. Could be a 1B distilled student, a single trained "draft head" on top of $M$'s hidden states (EAGLE), or even $K$ extra prediction heads bolted onto $M$ itself (Medusa).

One **speculation round** does the following:

1. **Draft.** The draft model $m$ autoregressively samples $K$ tokens $t_1, t_2, \ldots, t_K$, conditioned on the running prefix. Each draft sample is one tiny decode step; $K$ of them in sequence still costs a fraction of one target step.
2. **Verify.** The target model $M$ runs **one forward pass** that consumes the prefix plus $t_1 \ldots t_K$ as input, and emits $K+1$ output probability distributions: one at each of the candidate positions plus one at the very end.
3. **Accept or correct.** Walk the candidates left to right. At position $i$, accept $t_i$ with probability $\min(1, p_M(t_i) / p_m(t_i))$. On rejection at position $j$, emit a corrected token sampled from a residual distribution $\propto \max(0, p_M - p_m)$, throw away $t_{j+1}, \ldots, t_K$, and end the round. If every candidate is accepted, also use the bonus distribution at position $K+1$ to emit one more free token.

```python
# The verification step, in pseudocode (Leviathan et al., Algorithm 1).
def speculate(target, draft, prefix, K):
    drafts = []                                 # candidate tokens t_1..t_K
    state = draft.state_for(prefix)
    for _ in range(K):
        p_m = draft.next_token_dist(state)      # tiny: O(small model FLOPs)
        t   = sample(p_m)
        drafts.append((t, p_m))
        state = draft.advance(state, t)

    # One target forward pass over [prefix, drafts]
    p_targets = target.forward_dist(prefix + [t for t,_ in drafts])
    # p_targets[i] is the target's dist at position i.

    accepted = []
    for i, (t, p_m) in enumerate(drafts):
        p_M = p_targets[i]
        r = uniform(0, 1)
        if r < p_M[t] / p_m[t]:                 # accept
            accepted.append(t)
        else:                                   # reject: emit correction
            residual = max_zero(p_M - p_m); residual /= residual.sum()
            accepted.append(sample(residual))
            return accepted                     # drop the rest of drafts

    # All K accepted: bonus token from p_targets[K]
    accepted.append(sample(p_targets[K]))
    return accepted
```

The non-obvious property: **the distribution of the output sequence is provably identical to running the target model autoregressively.** This is the result that gives speculative decoding its name as a *correct* algorithm — not an approximation, not a heuristic. The rejection-sampling step exactly cancels the bias introduced by sampling from $p_m$ instead of $p_M$. Whatever the draft chooses to suggest, the verification step turns it into a draw from the target's true distribution.

The proof is two lines of probability theory and Leviathan et al. give it in their Appendix A. The intuition: rejection sampling is the standard way to convert samples from a proposal $q$ into samples from a target $p$, provided you accept each sample with probability $p/q$ scaled appropriately. Speculative decoding is rejection sampling applied to a *sequence* of distributions, one per position, and the math works out the same way.

{{% callout type="theorem" %}}
**The speculative-decoding theorem (informal).** For any prefix $x_{< t}$ and any draft distribution $q(x_t)$ with full support over the target's vocabulary, the joint distribution of accepted tokens under the speculate-and-verify algorithm is identical to autoregressive sampling from the target model $p$. The draft can be *anywhere* on the spectrum from "uniform random vocabulary" to "exactly the target" and the output distribution is unchanged. What changes is only the expected acceptance rate, and therefore the speedup.
{{% /callout %}}

That callout is the punchline of the paper. Once you've internalized it, the entire optimization-space of speculative decoding opens up: any model that produces a probability distribution over the next token is a *legal* draft. Better drafts buy more speedup. Worse drafts buy less. Wrong drafts buy nothing — but they also cost nothing in quality.

## Mechanic Inside vLLM

The scheduler ([→ ch.14](../14-scheduler/)) treats a speculative-decode round as a **multi-token request**. From the scheduler's point of view, that request is asking for $K+1$ tokens of budget this step instead of one. The block manager pre-allocates space for $K+1$ provisional KV entries per layer. The forward pass runs the target over $K$ input tokens and emits $K+1$ output distributions. The sampler does the rejection-sampling math.

If $j$ out of $K$ drafts are accepted, the unused $K - j$ KV blocks at the tail are immediately returned to the free list — the block manager already has refcounted blocks ([→ ch.11](../11-block-manager/)) so this costs O(1) per dropped slot. If the draft model uses its own private KV cache, the same eviction happens there.

The beautiful part: **none of this requires new machinery**. PagedAttention already supports non-contiguous K/V. The scheduler already speaks in "tokens this request gets to spend." Chunked prefill already proved that a single request can ask for $N$ tokens of compute. Speculative decoding is just *another value of $N$*, plus a sampling rule at the tail.

This is the second time in this issue we have seen the same pattern: a feature that looks structurally novel turns out to be the same forward pass everyone else uses, with a different value of the same parameter. The token budget keeps paying rent.

## EAGLE, Medusa, MTP — The Family Tree

The Leviathan-Chen formulation uses an *independent* draft model — a different transformer, separately trained, separately weighted. That's the cleanest version of the algorithm and the first one shipped to production. But the field rapidly discovered that the best draft model for a target model is often *part of the target model itself*.

- **Medusa** ({{< cite text="Cai et al., 2024" url="https://arxiv.org/abs/2401.10774" kind="paper" >}}, January 2024). Bolt $K$ small "Medusa heads" onto the target model's final hidden state — each head predicts the token at position $t+i$ for $i = 1, \ldots, K$. No separate draft model. The target model itself produces all $K$ candidates in a single forward pass via tree attention. Acceptance is per-head and rejection at head $i$ truncates the rest.
- **EAGLE-1/2/3** ({{< cite text="Li et al., 2024 (EAGLE)" url="https://arxiv.org/abs/2401.15077" kind="paper" >}}, January 2024 onward). A more elegant move: train a single tiny autoregressive draft head that operates *in feature space* — it consumes the target model's penultimate hidden states (not just the last token's) and emits candidate features that are then projected back through the target's LM head. EAGLE-3 (late 2025) gets to roughly **70% token acceptance** at $K=4$ while costing about 1% of target compute per draft step.
- **MTP / Multi-Token Prediction** ({{< cite text="DeepSeek-V3 technical report" url="https://arxiv.org/abs/2412.19437" kind="paper" >}}, December 2024). DeepSeek bakes the multi-token-prediction objective into the *training* of V3 itself. The model has multiple output heads and is trained to predict 2–4 tokens ahead during pretraining, so the draft path is just "use the extra heads" — no separate training stage, no extra parameters except the heads.

The common thread: the draft model gets cheaper and the acceptance rate gets higher as the draft gets *more entangled* with the target. The platonic limit is a draft that costs zero and accepts with probability one — which is the same as the target model emitting $K$ tokens per forward pass, which is exactly what MTP-trained models approximate.

| Method | Year | Draft model | Acceptance @ K=4 | Speedup |
|---|---|---|---|---|
| Leviathan/Chen (independent draft) | 2022/23 | Llama-1B for Llama-70B | ~50% | 1.6–1.9× |
| Medusa | 2024 | K bolted heads, tree attn | ~55% | 1.9–2.3× |
| EAGLE-1 | 2024 | feature-space draft head | ~63% | 2.1–2.7× |
| EAGLE-3 | 2025 | improved feature draft | ~70% | 2.3–3.1× |
| MTP (trained in) | 2024 | model's own extra heads | ~75% | 2.5–3.5× |

(Numbers are rough averages from the published benchmarks; per-workload they vary 15–30%.)

{{< crosshead >}}Where the speedup ceiling lives{{< /crosshead >}}

The expected speedup under a target acceptance rate $\alpha$ and verification count $K$ is the classic Leviathan formula:

$$
\text{Speedup}(\alpha, K) \;=\; \frac{1 - \alpha^{K+1}}{(1 - \alpha)\,(1 + Kc)}
$$

where $c$ is the cost of one draft-model forward pass relative to one target-model forward pass. For an EAGLE-style head, $c \approx 0.01$. For a separate 1B draft of a 70B target, $c \approx 0.05$. As $\alpha \to 1$ and $c \to 0$, the speedup approaches $K+1$. As $\alpha \to 0$ (a useless draft), the speedup approaches zero — you've spent $K$ extra forward passes for one accepted token.

Plotted, this is the **acceptance curve** — the picture every speculative-decoding paper has on its first or second page, and the one that tells you what $K$ to pick for your workload.

```pyplot {id="spec-decode-acceptance" caption="EXPECTED INTER-TOKEN LATENCY VS K FOR THREE DRAFT-ACCEPTANCE RATES. EACH CURVE HAS AN OPTIMAL K WHERE THE GEOMETRIC-TAIL OF REJECTED DRAFTS STARTS EATING THE GAIN."}
np.random.seed(5)

# Llama-3.1-8B baseline ITL on H100 at batch 4: ~18 ms (decode-bound)
itl_base_ms = 18.0
# Draft cost fraction c: EAGLE-style ~0.01; separate-draft ~0.05
c_eagle = 0.01

K = np.arange(1, 32)

fig, ax = plt.subplots(figsize=(9, 4.6))
colors = {'0.5': '#00A8A8', '0.7': '#FF8C00', '0.9': '#FF007F'}

for alpha, color in [(0.5, '#00A8A8'), (0.7, '#FF8C00'), (0.9, '#FF007F')]:
    # Expected accepted tokens per round = (1 - alpha^(K+1)) / (1 - alpha)
    accepted = (1 - alpha**(K+1)) / (1 - alpha)
    # Step latency: target pass cost ~= itl_base, plus K*c draft passes
    step_latency = itl_base_ms * (1 + K * c_eagle)
    # ITL per accepted token
    itl = step_latency / accepted
    ax.plot(K, itl, '-o', color=color, linewidth=2,
            markersize=4, markeredgecolor='#1A1A1A',
            label=f'acceptance α = {alpha}')
    # Mark the optimum
    kstar = K[np.argmin(itl)]
    ax.scatter([kstar], [itl.min()], s=110, color=color,
               edgecolor='#1A1A1A', linewidth=1.5, zorder=4)
    ax.annotate(f'K*={kstar}\n{itl.min():.1f} ms',
                xy=(kstar, itl.min()), xytext=(kstar+1, itl.min()+1.4),
                fontsize=9, color='#1A1A1A')

ax.axhline(itl_base_ms, color='#1A1A1A', linestyle=':',
           linewidth=1.4, label=f'no spec, ITL = {itl_base_ms} ms')
ax.set_xlabel("K  (draft tokens per round)")
ax.set_ylabel("expected inter-token latency (ms)")
ax.set_title("Acceptance curve — choose K so the geometric tail just stops paying",
             fontsize=11, loc='left')
ax.set_xlim(0, 32)
ax.set_ylim(0, itl_base_ms * 1.05)
ax.legend(loc='upper right')
ax.spines[['top','right']].set_visible(False)

print("Llama-3.1-8B speculative decoding — anchor case:")
print(f"  Baseline (no spec) ITL: {itl_base_ms} ms")
for alpha in [0.5, 0.7, 0.9]:
    accepted = (1 - alpha**(K+1)) / (1 - alpha)
    step_latency = itl_base_ms * (1 + K * c_eagle)
    itl = step_latency / accepted
    kstar = K[np.argmin(itl)]
    print(f"  α={alpha:.1f}, EAGLE-style c=0.01: optimal K = {kstar}, "
          f"ITL = {itl.min():.1f} ms ({itl_base_ms/itl.min():.2f}× speedup)")
plt.tight_layout()
```

The sweet-spot $K$ sits at roughly $K \approx 4$ for $\alpha = 0.7$ and edges out to $K \approx 8$ for $\alpha = 0.9$. Beyond that the **geometric-series tail of rejected drafts** eats the gain — once you've drafted further than the acceptance rate can credibly hold, you're paying for FLOPs that the verifier throws away. This is why EAGLE-3 and DeepSeek's MTP both ship with $K = 4$ as the production default. The math says it; the benchmarks confirm it.

## When Spec Decode Doesn't Help

The arithmetic-intensity argument also tells you exactly when speculative decoding **fails to help**. Three failure modes:

1. **Heavy batches.** At batch 256, the model is already loaded once and reused across 256 sequences — arithmetic intensity has risen naturally from $I \approx 1$ to $I \approx 256$. The ridge gap has closed. Adding $K$ draft tokens per sequence pushes you over the ridge into the compute-bound regime, where extra FLOPs are *no longer free*. Speedup collapses to ~1.0×. **Speculative decoding is a single-/few-stream optimization.**
2. **Compute-bound steps.** Long-context decode at high batch — anywhere $I$ is already near the ridge — the same argument applies. Once you're computing as fast as the cores will run, $K$ more candidates costs $K\times$ more time.
3. **Adversarial drafts.** If the draft model has *zero* alignment with the target (e.g., wrong tokenizer, wrong domain, wrong instruction tuning), every candidate gets rejected and you've burned the draft FLOPs and the verification FLOPs for one autoregressive token. Worse than no speculation.

In production, this means speculative decoding is most useful for **interactive latency-sensitive workloads** — single-user chat, code completion, voice assistants — where batch is small and ITL is the metric you live or die by. It is less useful for **throughput-oriented batch jobs** — overnight evaluation runs, document processing, fine-tuning data generation — where you're already on the ridge.

## What To Remember

1. **Verifying $K$ candidates costs ~1× the HBM pass.** That's the whole trick. Memory-bound decode makes extra FLOPs cheap, and speculative decoding turns those FLOPs into accepted tokens.
2. **Rejection sampling makes it exact.** The output distribution is provably identical to the target's. No quality trade-off — just engineering complexity in the draft path and the sampler.
3. **The best draft model lives inside the target model.** EAGLE / Medusa / MTP all share weights or features with the target, getting both speed (cheaper draft passes) and higher acceptance rate (the draft is by construction aligned).
4. **Spec decode is a single-stream win.** It evaporates at high batch because the ridge gap closes naturally — fall back to plain continuous batching when you're already compute-bound.

**Continue to → [Splitting the Model](../16-tp-pp/)** — speculative decoding squeezes more tokens out of a single GPU. Splitting the model across many GPUs opens a different scaling axis: the one that lets a 70 GB model fit on an 80 GB card with room left for KV.
