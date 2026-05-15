---
title: "Scan vs Think"
description: "A transformer can find a sentence in 200,000 tokens with one forward pass. It cannot, in the same pass, follow a chain of reasoning that depends on five separately scattered facts. This is the gap the whole benchmark crisis is about."
topics: [transformers, attention, reasoning]
tags: [retrieval, reasoning, multi-hop, attention-depth, chain-of-thought]
theme: teal
math: true
draft: false
date: 2026-05-14T09:00:00-04:00
issue: 4
weight: 40
techKind: primer
techNode: retrieval-vs-reasoning
header: default.webp
---

## A Phone Call To A Detective

It is **early September 2024** and a hypothetical reader emails their friend, a software-engineering manager. The email reads:

> *Hey — quick favour. In your codebase, find the function that takes a `UserSession` as input and returns a `BillingState`. I need its name.*

The friend opens the repository, runs `grep -n "BillingState" *.py`, scans the matches, finds one function signature that matches the description, and texts back: *`finalize_session_billing`*. Total time: thirty seconds.

Two days later the same reader sends a second email:

> *Quick favour, sorry — for the same function, what does it eventually return when the user has an unpaid trial subscription, the trial ended within the last 7 days, the billing region is EU, and the previous renewal attempt failed with a 402?*

The friend opens the repository, finds `finalize_session_billing`, reads it, follows the branching logic into `compute_renewal_state`, follows *that* into `eu_dunning_policy`, traces the dunning policy through three more files of conditional rules, then comes back. Total time: half an hour, with a coffee.

The first task is **retrieval**: a single string-matching scan, one fact lifted, done. The second task is **reasoning**: a chain of dependent lookups, each one shaped by the answer to the previous, conditional logic that branches based on intermediate state. **Both tasks happen inside the same codebase.** The codebase has not changed. The model of the codebase your friend is using to answer has not changed. What changed is the *number of conditional hops* the question requires.

This primer is about why a transformer's attention mechanism is *built* for the first task — single-shot retrieval across a long context — and *not* built for the second — chained reasoning that depends on intermediate state. Once you see the gap, the entire 2024 benchmark crisis becomes inevitable in retrospect.

## What Attention *Is*, In One Picture

Strip a transformer down to bare metal. Each decoder block does two operations that take a sequence of $n$ token vectors and return another sequence of $n$ token vectors. The first is {{< wiki "attention" >}}self-attention{{< /wiki >}}: for each output token $i$, look at every previous token $j$, compute a scalar **attention weight** $\alpha_{ij}$ that says "how much should token $j$ influence token $i$'s output?", and then take a weighted sum.

The attention weight $\alpha_{ij}$ is the dot product of token $i$'s **query** vector (`q` in the [microGPT reference](../../05-microgpt-unfolded/08-attention/)) and token $j$'s **key** vector (`k`), normalised by {{< wiki "softmax" >}}softmax{{< /wiki >}} — these are the `attn_logits` and `attn_weights` from the listing:

$$
\alpha_{ij} \;=\; \frac{\exp(q_i \cdot k_j / \sqrt{\text{head\_dim}})}{\sum_{k \leq i} \exp(q_i \cdot k_k / \sqrt{\text{head\_dim}})}
$$

The output for token $i$ is a weighted blend of the value vectors (`v`, cached as `values[li]`) of every previous token:

$$
O_i = \sum_{j \leq i} \alpha_{ij} \, v_j
$$

In a single attention layer, then, *each output token sees every previous input token*, with a learned per-pair weighting. If you draw the attention pattern $\alpha$ for a typical sentence, you see something like this:

```pyplot {id="attn-patterns" caption="Two stylized attention patterns. LEFT — retrieval: one bright cell where the output token finds the single relevant context token. RIGHT — multi-hop reasoning: a chain of cells across three intermediate keys, requiring layers to chain attention over each other. A single attention layer can do the left case in one shot; the right case requires depth."}
np.random.seed(0)
n = 24
# Pattern 1: pure retrieval — output token attends sharply to one key
retrieval = np.zeros((n, n))
retrieval[-1, 8] = 1.0
# add some noise for realism
retrieval[-1] += np.random.rand(n) * 0.04
retrieval[-1, 8] = 1.0

# Pattern 2: multi-hop — final output cares about keys 3, 10, 17, blended
# (each one is itself the output of a previous reasoning step)
multihop = np.zeros((n, n))
multihop[-1, [3, 10, 17]] = [0.4, 0.35, 0.55]
multihop[-1] += np.random.rand(n) * 0.04

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, A, title in zip(axes,
                         [retrieval, multihop],
                         ["Retrieval pattern\n('find the sentence')",
                          "Multi-hop pattern\n('chase the chain')"]):
    im = ax.imshow(A, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("attended-to position (key)")
    ax.set_ylabel("attending position (query)")
    ax.set_xticks([0, n-1])
    ax.set_xticklabels(['token 0', f'token {n-1}'])
    ax.set_yticks([0, n-1])
    ax.set_yticklabels(['token 0', f'token {n-1}'])
plt.tight_layout()

print("Retrieval: probability mass concentrated on ONE position.")
print("Multi-hop: probability mass spread across MANY positions —")
print("but you can't tell from the pattern alone WHY each was selected.")
print("The 'why' would have been computed in earlier layers and re-encoded")
print("into the query vector at position -1.  Multi-hop needs depth.")
```

This single picture is doing a lot of conceptual work. Read it twice.

In the **retrieval** case (left panel), one query token sends almost all of its probability mass to one key token. The other 23 keys contribute essentially nothing. This is the attention pattern of a NIAH-style lookup: query says "what does the model want?", and the relevant prior token says "I'm here." One step. Done.

In the **multi-hop** case (right panel), the same query attends to multiple keys. But — and this is the subtle part — *each of those keys was itself the product of earlier attention operations* in earlier decoder layers. The "fact" the query is now retrieving is not a fact that was literally in the original input. It is a *synthesized* fact, computed across multiple layers of attention, each layer chaining onto the previous. **A single attention pattern image cannot show you the chain.** The chain is across layers, not across positions.

## The "Hops" Argument From Architecture

Let's formalise. Suppose a question requires $k$ sequential hops to answer. *Alice has a parent named Bob. Bob has a parent named Carol. Carol has a parent named Diana. Diana has a parent named Eve. Who is Alice's great-great-grandmother?* That's a 4-hop reasoning chain. To answer it from a long prompt that contains the facts scattered separately:

1. **Hop 1**: locate "Alice's parent" → return Bob.
2. **Hop 2**: locate "Bob's parent" → return Carol. (Required Hop 1's output to know to look up Bob.)
3. **Hop 3**: locate "Carol's parent" → return Diana. (Required Hop 2's output.)
4. **Hop 4**: locate "Diana's parent" → return Eve. (Required Hop 3's output.)

How many attention layers does this take? In the worst case — when none of the facts can be co-located or precomputed in earlier layers — **at least one attention layer per hop**. The first layer does Hop 1 and writes "Alice → Bob" into Alice's {{< wiki "residual-stream" >}}residual stream{{< /wiki >}}. The second layer reads "Alice → Bob" out of the residual, does Hop 2 (look up Bob's parent), and writes "Alice → Carol" back. And so on.

This is a real, formal result. It's been studied under various names — *circuit depth*, *reasoning depth*, *attention compositionality* — by interpretability researchers like Sanford et al. (2024) and the Anthropic interpretability group.

{{% pullquote type="technical" %}}
**With $L$ decoder layers, a transformer can chain at most $L$ dependent attention hops within a single forward pass.** Beyond that, it must either parallelise the hops (which only works if they're independent) or defer the answer to generation time (chain-of-thought).
{{% /pullquote %}}

Beyond that, it has to either *parallelize* the hops (which only works if they're independent) or *defer* the answer until generation time (which only works if it can write intermediate tokens).

```pyplot {id="hops-vs-layers" caption="The reasoning-depth ceiling. For a model with L decoder layers, the number of sequential dependent hops it can chain in a single forward pass is bounded by L. Llama-3.1 70B has L=80, which sounds like plenty — but real reasoning chains overflow this rapidly when the chain has branches, when hops are noisy, and when multiple sub-questions compete for the same layers."}
L_values = [16, 32, 64, 80, 96]   # decoder depths of common models
hops_possible = L_values            # 1 hop per layer in the optimistic case

# typical real-world reasoning chain depths
chains = {
    "single fact lookup":           1,
    "compare two facts":            2,
    "Alice→Bob→Carol":              3,
    "5-step legal reasoning":       5,
    "code: trace 3 calls":          8,
    "BFS of depth 4 on a graph":   15,
    "long agentic task":           50,
}

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(L_values, hops_possible, color='#00A8A8',
       edgecolor='#1A1A1A', linewidth=1.5, width=8,
       label='max chainable hops (= L)')

for name, depth in chains.items():
    ax.axhline(depth, color='#FF007F', linewidth=0.6, alpha=0.7)
    ax.text(L_values[-1] + 4, depth + 0.5, name, fontsize=8, va='center')

ax.set_xlabel("model decoder depth L")
ax.set_ylabel("dependent attention hops")
ax.set_title("Reasoning-depth budget — what fits inside one forward pass?")
ax.set_xticks(L_values)
ax.set_xticklabels([f"L={l}\n({['gpt-2','gpt-3 6.7B','llama-2 70B','llama-3.1 70B','o3'][i]})"
                    for i,l in enumerate(L_values)], fontsize=8)
ax.legend(loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(L_values[0] - 8, L_values[-1] + 35)
ax.set_ylim(0, max(L_values) + 5)

print("Hops budget vs typical task depth:")
print(f"  Llama-3.1 70B (L=80) — chainable hops ≤ 80")
print(f"  BFS depth-4 on a graph of ~50 nodes ≈ 15-20 dependent hops")
print(f"  An agentic task with 50 tool calls ≈ 50+ dependent hops")
print(f"  → the model has to use chain-of-thought to break the chain into pieces")
```

This bar chart is the architecture-level reason **GraphWalks** exists. A graph BFS of depth $d$ requires roughly $d$ dependent hops just to enumerate the frontier; an even moderately complex graph traversal demands more hops than any model has layers. The model has to either spread the computation across multiple decoding steps (chain-of-thought), or fail.

## Chain-Of-Thought: The Hop-Multiplier

The most famous response to the hop-depth ceiling is **chain-of-thought** (CoT). The 2022 Wei et al. paper showed that if a model is encouraged to *write out intermediate reasoning steps* before producing the final answer, performance on multi-step problems jumps dramatically. The reason is direct: CoT *converts depth into width*. Instead of one forward pass with 80 layers, the model gets to do many forward passes, each of 80 layers, with the previous pass's *output* as part of the next pass's *input*.

If a single forward pass can do at most $L$ hops, then $T$ forward passes of chain-of-thought can do at most $T \cdot L$ hops. For modern reasoning models that emit hundreds or thousands of CoT tokens, this is effectively unlimited.

This is why **o1, Claude Sonnet 4.5 thinking, Gemini 2.5 Pro Deep Think, and the rest of the late-2024-onwards reasoning models** simultaneously crushed earlier multi-hop benchmarks. They didn't fundamentally change the attention mechanism. They just got better at *spending tokens on intermediate reasoning*, which is the only way the hop ceiling moves up.

This is also why the late-2024 long-context benchmarks split into two categories: those that *allow* extended reasoning at inference time, and those that don't. LongBench v2's most famous result is that **o1-preview's 57.7% beat human experts' 53.7%** at multi-hop long-context QA — but only when o1 was allowed to think for an unspecified amount of time. Direct-answer models scored 50.1%. The chain-of-thought multiplier *is the difference* between the new benchmarks and the old.

## Two Different Capabilities, Two Different Benchmarks

So when **OpenAI's April 2025 GPT-4.1 launch** described MRCR as a benchmark that can be solved *"by doing one pass or read-through of the prompt"* — that was a precise architectural claim. Even **MRCR** ("retrieve the *fourth* tapir poem among 8 candidates"), despite being substantially harder than vanilla NIAH, fundamentally has the structure: scan, count, retrieve. A clever single-pass implementation suffices.

**GraphWalks** explicitly violates this. The task is: given a graph of hex-hash node ids, perform BFS starting from a specific node and return all nodes at a given depth. A correct BFS at depth $k$ requires at least $k$ dependent hops to enumerate. **No amount of cleverness lets a single linear pass solve it.** Either the model uses chain-of-thought to spread the BFS across many forward passes (and CoT tokens make the inference *very* expensive), or it gets the answer wrong.

This is the architectural reason GraphWalks is the *right* benchmark for the next era. Not because it's flashier or harder per token, but because **it cannot be solved without exercising the capability the field actually cares about**.

We unpack the graph traversal mechanics in [BFS as Reasoning](../10-graph-traversal/), and the mainline narrative of how GraphWalks emerged in [One Pass Isn't Enough](../08-needle-to-graph/).

## A Useful Test At The Kitchen Table

If you want to feel the difference yourself, try this exercise the next time you're using a long-context model. Take the same document — a contract, a codebase, a research paper — and ask the model two questions about it:

1. **A retrieval question**: *"What is the deductible amount specified in Section 4.2?"* or *"What does the function `compute_renewal_state` return when the user is a paid subscriber?"* — a fact that lives in one place in the prompt.
2. **A reasoning question**: *"If a user signed up on a 7-day trial that just ended on 2025-12-10, has an active EU billing address, and a failed renewal attempt last week, what state does `compute_renewal_state` return?"* — a fact that requires chaining through several conditional branches.

You will, reliably, find that the first kind of question is answered almost flawlessly. The second kind degrades much faster as the document gets longer, even when all the relevant logic is *in the prompt*. The model is finding things, but is not *thinking* about them at the same scale.

This is the gap. Every benchmark in the 2024-onwards literature is, in some sense, a more rigorous version of this kitchen-table test.

## What To Remember

1. **Retrieval and reasoning are different capabilities even when they happen inside the same model.** Retrieval: find one fact at one position. Reasoning: chain multiple lookups whose inputs depend on prior outputs.
2. **A single attention layer cannot do multi-hop reasoning.** Each hop typically requires its own decoder layer (or its own forward pass via chain-of-thought).
3. **The hop budget is bounded by depth $L$ in a single forward pass.** With $L = 80$, you can chain ~80 dependent hops at most. Real reasoning chains routinely exceed this when there's branching or noise.
4. **Chain-of-thought converts depth into width.** Emit intermediate reasoning tokens; each token's forward pass adds another $L$ hops to the available chain. Reasoning models (o1, Sonnet 4.5 thinking, Gemini 2.5 Pro Deep Think) lean hard on this.
5. **The 2024–2025 benchmark turnover happened because NIAH and MRCR can be solved by single-pass retrieval, while GraphWalks and friends cannot.** That architectural distinction is the reason the cold open's mystery resolves the way it does.

**Continue to** → [When Everyone Scored 99](../05-saturation/) — the 2024 saturation crisis, told as a parade of seven benchmarks that all reached the same conclusion in the same twelve months.
