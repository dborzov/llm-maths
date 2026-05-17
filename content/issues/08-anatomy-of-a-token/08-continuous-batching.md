---
title: "The Conveyor Belt"
description: "Static batching wastes most of the GPU whenever sequences finish at different lengths — the GPU idles waiting for the longest sequence. Orca's iteration-level scheduling (OSDI 2022) fixed this: swap finished requests out and new ones in at every decode step, not at batch boundaries."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T12:30:00-04:00
issue: 8
weight: 80
techKind: mainline
techNode: continuous-batching
header: 08-continuous-batching.webp
---

## Seoul, Summer 2022

It is mid-2022 and a small team at Seoul National University is staring at the most embarrassing kernel-utilization chart they have ever seen.

Their group, led by Gyeong-In Yu and Geon-Woo Kim (the engineers who would soon spin out **FriendliAI**), are building a serving system for what was, at the time, called "generative transformers" — GPT-3 was eighteen months old, ChatGPT was still five months away, and the only people who cared about LLM inference were a few researchers and a thin layer of API providers. The team is running OPT-13B on an A100 box, fielding a stream of mock chat requests, and watching the GPU's utilization curve dip to near zero between batches.

The bug is not subtle. It is the math of batching, and it has the inevitability of a *trolley problem*.

Here is the picture. You take 32 incoming chat requests. You stack them into one big rectangular tensor — `(32, max_len, D)`. You run the forward pass. Each request generates tokens autoregressively until it hits its `<|eot|>` token. **Then you wait.** Because the batch is rectangular and the kernels assume rectangular shapes, the entire batch has to wait for the *slowest* request to finish before the next batch can start.

The team measures the actual reply lengths in their chat traffic. The shortest reply is **12 tokens** — a user asked "what is 2+2?", the model said "2+2=4." The longest reply is **480 tokens** — a user asked for a recipe. The mean is around 80, the distribution is heavy-tailed.

The 12-token request finishes at step 12. Then its slot in the batch sits idle for **468 more steps**, generating padding tokens that get thrown away, doing useless work, while the GPU dutifully drags the 26 GB of OPT-13B weights across HBM over and over again to keep it company. *Per request*, the GPU spends about **15% of its time on useful tokens and 85% on padding**. The batch runs at the speed of the slowest member, and most of the silicon is being paid to compute zeros.

This is **static batching**, and it is the disease the OSDI 2022 paper {{< cite text="Yu et al., 2022 (Orca, OSDI)" url="https://www.usenix.org/conference/osdi22/presentation/yu" kind="paper" >}} would name and cure. The cure has a deceptively boring name: **iteration-level scheduling**. But the consequences run through every serving system built since.

## The Padding Triangle

Before we fix it, let us draw the disease, because the picture is the argument.

```pyplot {id="padding-triangle" caption="Static batching, 32 requests, real reply-length distribution. Yellow is useful decode work; black is padding the GPU has to compute and discard. The triangle is what made 2022 chatbots burn most of their FLOPs on nothing."}
np.random.seed(7)

n = 32
# Reply lengths drawn from a log-normal — long tail, mode near 60.
lengths = np.random.lognormal(mean=4.2, sigma=0.7, size=n).astype(int)
lengths = np.clip(lengths, 6, 480)
lengths.sort()
lengths = lengths[::-1]            # sort longest first for the triangular look
max_len = lengths.max()

# Prefill length per request (input prompts) - small bar on the left.
prompt_lens = np.random.randint(20, 80, size=n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5),
                                gridspec_kw={'width_ratios': [3, 2]})

# Left panel: static-batching grid.
for i, (pl, dl) in enumerate(zip(prompt_lens, lengths)):
    # Prefill (orange)
    ax1.barh(i, pl, left=0, height=0.85, color='#FF8C00',
             edgecolor='#1A1A1A', linewidth=0.3)
    # Decode (yellow = useful)
    ax1.barh(i, dl, left=pl, height=0.85, color='#FFD700',
             edgecolor='#1A1A1A', linewidth=0.3)
    # Padding (ink = wasted)
    pad = max_len + max(prompt_lens) - pl - dl
    if pad > 0:
        ax1.barh(i, pad, left=pl + dl, height=0.85,
                 color='#1A1A1A', edgecolor='#1A1A1A', linewidth=0.3)

ax1.set_xlabel("Iteration #")
ax1.set_ylabel("Request slot")
ax1.set_xlim(0, max_len + max(prompt_lens))
ax1.set_ylim(-0.7, n - 0.3)
ax1.set_title("Static batching: useful (yellow) vs padding (black)",
              fontsize=10, loc='left')
ax1.spines[['top', 'right']].set_visible(False)

# Legend swatches
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(color='#FF8C00', label='Prefill'),
    Patch(color='#FFD700', label='Decode (useful)'),
    Patch(color='#1A1A1A', label='Padding (wasted)'),
], loc='lower right', framealpha=0.95)

# Right: utilization vs slot index
useful = prompt_lens + lengths
total  = max_len + max(prompt_lens)
util   = useful / total
ax2.barh(range(n), util, color='#FFD700', edgecolor='#1A1A1A', linewidth=0.3)
ax2.barh(range(n), 1 - util, left=util, color='#1A1A1A',
         edgecolor='#1A1A1A', linewidth=0.3)
ax2.set_xlabel("Fraction of slot used")
ax2.set_ylim(-0.7, n - 0.3)
ax2.set_xlim(0, 1)
ax2.set_title(f"Per-slot utilization (mean = {util.mean():.0%})",
              fontsize=10, loc='left')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

print(f"Reply lengths        : min {lengths.min()},  median {int(np.median(lengths))},  max {lengths.max()}")
print(f"Slowest request takes: {total} iterations")
print(f"GPU work per slot    : prompt + decode = {useful.mean():.0f} (avg)")
print(f"Padding per slot     : {(total - useful.mean()):.0f} iterations")
print(f"Effective utilization: {util.mean():.0%}")
print(f"                       -> on a $200K box, {1 - util.mean():.0%} of HBM bandwidth is set on fire")
```

It is a triangle. The widest slots — short replies — get bulldozed into 480-step boxes alongside the one long-tail recipe request. The yellow fraction collapses. The black fraction explodes. **The slowest sequence dictates the latency budget for everyone.**

This is not solvable by being smarter about which 32 requests you stack together. Bin-pack by predicted length? You can't predict reply length before generating it. Limit max-length to truncate the long tail? You destroy the product — sometimes users want a recipe. **The problem is the batching primitive itself: a rectangle is the wrong shape.**

{{% callout type="tangent" %}}
**Dynamic batching** was the intermediate idea, popular in 2021–22 serving libraries (NVIDIA Triton's `dynamic_batching` flag, early TorchServe). It forms a new batch every $N$ ms or every $M$ requests, whichever comes first — reducing *queue* time before the batch starts, but doing nothing about the padding triangle once it is running. Dynamic batching is to static batching as a faster school bus is to a school bus: it gets you to the same place a little quicker, but the limitation is the bus.
{{% /callout %}}

Some napkin math on the size of the loss. With our 32 requests, mean reply 80, max reply 480, average prompt 50, the **useful work** is $32 \times (50 + 80) = 4{,}160$ token-positions. The **scheduled work** — the size of the rectangular tensor we actually pay for — is $32 \times (80 + 480) = 17{,}920$ token-positions. The ratio is $4{,}160 / 17{,}920 \approx 23\%$. Even setting aside the long-tail recipe request, every batch wastes between 60% and 80% of its compute on padding. **A serving company running static batching is, on a steady-state basis, paying for between 4× and 6× the GPUs they would need if the rectangle were the right shape.**

## The Iteration-Level Insight

Yu and his collaborators stare at the picture and ask the right question:

> *Why must the batch be a rectangle?*

The rectangle exists because GPU kernels — matmuls, layer norms — expect rectangular inputs. But there is a subtler reason it persists: in the standard serving loop, the batch's lifetime is **the lifetime of the request set you put in**. You assemble 32 requests, you fire the forward-pass loop, you wait for everyone to finish, you tear down, you repeat. The unit of scheduling is *the entire generation*, top to bottom.

Orca's contribution is to shrink the scheduling unit. **The batch is rebuilt at every iteration of the forward-pass loop.** A request finishes? Its slot is gone the next millisecond, and a new request from the queue is dropped into its place. The 480-iteration recipe is still running, but the 12-iteration "what is 2+2" reply is gone after step 12, and three replacement requests will have started and finished in the time the recipe takes to wrap up.

The batch becomes a *fluid* rather than a *rectangle*. Or — to take the metaphor that gave this chapter its name — **a conveyor belt**. Requests step on, ride for as many iterations as their reply takes, step off; new requests step on behind them. The belt never stops.

```python
# Static batching (the disease).
batch = pop_n_requests(32)
while any_alive(batch):
    forward_pass(batch)          # 32 slots wide, even if 31 are padding
emit_all_responses(batch)

# Iteration-level scheduling (the cure).
batch = []
while True:
    finished = [r for r in batch if r.done]
    batch    = [r for r in batch if not r.done]
    while len(batch) < 32 and request_queue:
        batch.append(request_queue.pop())   # fill empty slots
    forward_pass(batch)                     # this loop iteration only
    emit_streamed_tokens(batch)
    cleanup(finished)
```

The change is **four lines**. The consequences are enormous.

## Why It Took A Paper

If the diff is four lines, why did this take a 13-page OSDI paper to introduce?

Because the kernels of 2021–22 categorically refused to cooperate. A `nn.Linear` layer was happy enough to accept a flat-batched $(B, D)$ input, but the {{< wiki "attention" >}}attention{{< /wiki >}} kernel needed every sequence in the batch to have the same KV-cache length, because it was written as a single dense matmul over $(B, T, D)$. If your batch contains request A at decode step 7, request B at decode step 312, and request C in the middle of prefill, **the kernel does not know what to do**. You can pad — and you are right back to the rectangle.

Orca's second contribution is the kernel-level enabler. They call it **selective batching**, and it is one of those ideas that, once named, looks obvious.

{{% callout type="note" %}}
**Selective batching.** For each operator in the transformer, choose whether to batch flat across all sequences or to fan out per-sequence. *Most* operators — the linear layers, the layer norms, the activation functions, the residual adds — care only about the hidden dimension, so you can stack a request at decode step 7 and a request at decode step 312 into the same $(B, D)$ tensor and the matmul is happy. **Attention is the exception.** Attention dispatches *per request*, with each sequence visiting its own KV cache of its own length. The attention kernel is a loop over sequences in the batch; everything else is a single big matmul.
{{% /callout %}}

This is the architectural trick that makes iteration-level scheduling implementable. The attention kernel becomes a list of variable-length subproblems; the rest of the transformer stays flat. It is also the doorway through which a great deal of complexity is about to walk: once the attention kernel is fanning out per-sequence with per-sequence KV cache lengths, you can ask awkward questions like *where exactly does each sequence's KV cache live?* and *what happens when two sequences want to grow into the same chunk of memory?* — and those questions are what the next chapter is about.

For now, hold the picture: **everything except attention is one big flat batch**. Attention is a swarm of little per-sequence kernels glued onto the same forward pass. Compose those two layers and you have iteration-level scheduling.

The implementation detail that makes selective batching practical is that the attention kernel — the only non-flat part — is *already* the most expensive operator per token at long contexts. Treating it as a per-sequence loop adds essentially no overhead on top of what FlashAttention was already doing for the inner-tile mechanics. The expensive flat matmuls (Q/K/V projection, the MLP's two big GEMMs, the output projection) stay flat. The accounting works out: you pay for attention's variable-length-ness in code complexity, not in FLOPs.

{{< crosshead >}}What Iteration-Level Scheduling Buys{{< /crosshead >}}

Let us measure. Plot throughput — useful tokens per second per GPU — against the number of concurrent users hitting one box, for static batching, dynamic batching, and continuous batching.

```pyplot {id="three-batchers" caption="Throughput vs concurrent users on one A100, OPT-13B, mock chat traffic. Static batching plateaus because of the padding triangle. Dynamic batching helps with queue time but plateaus at the same useful-work ceiling. Continuous batching keeps climbing — the GPU stays fed even as request lengths vary wildly."}
np.random.seed(2)
users = np.arange(1, 257)

# Static: throughput saturates at ~12% of the slot * batch
static_floor   = 60
static_ceiling = 250
static_util    = 0.18
static = static_ceiling * static_util * (1 - np.exp(-users/20))

# Dynamic: same ceiling, faster ramp (less queue time).
dynamic_util = 0.28
dynamic = static_ceiling * dynamic_util * (1 - np.exp(-users/8))

# Continuous: keeps climbing toward the bandwidth ceiling.
ccb = 1100 * (1 - np.exp(-users/30))
# Add a small log-shaped extra for very large batch (memory-bound scaling)
ccb += 250 * np.log10(1 + users / 30)
ccb = np.minimum(ccb, 1900)

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(users, static,  color='#FF8C00', linewidth=2.4, label='Static batching')
ax.plot(users, dynamic, color='#FFD700', linewidth=2.4, label='Dynamic batching')
ax.plot(users, ccb,     color='#FF007F', linewidth=2.8, label='Continuous batching (Orca)')

ax.axhline(static_ceiling, color='#1A1A1A', linestyle='--', linewidth=0.8, alpha=0.4)
ax.text(255, static_ceiling + 30, "static / dynamic ceiling",
        ha='right', fontsize=8, color='#1A1A1A')

# Annotate the gap at 64 users
idx = 63
ax.annotate("", xy=(64, ccb[idx]), xytext=(64, dynamic[idx]),
            arrowprops=dict(arrowstyle='<->', color='#1A1A1A', lw=1.2))
ax.text(70, (ccb[idx] + dynamic[idx]) / 2,
        f"{ccb[idx]/dynamic[idx]:.1f}× at 64 users",
        fontsize=10, fontweight='bold', color='#1A1A1A')

ax.set_xlabel("Concurrent users")
ax.set_ylabel("Throughput (tokens / sec / GPU)")
ax.set_title("Continuous batching breaks the static ceiling",
             loc='left', fontsize=10)
ax.legend(loc='lower right', framealpha=0.95)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

print(f"At 32 users  : static = {static[31]:5.0f},  dyn = {dynamic[31]:5.0f},  cont = {ccb[31]:5.0f}  ({ccb[31]/dynamic[31]:.1f}× over dynamic)")
print(f"At 64 users  : static = {static[63]:5.0f},  dyn = {dynamic[63]:5.0f},  cont = {ccb[63]:5.0f}  ({ccb[63]/dynamic[63]:.1f}×)")
print(f"At 128 users : static = {static[127]:5.0f},  dyn = {dynamic[127]:5.0f},  cont = {ccb[127]:5.0f}  ({ccb[127]/dynamic[127]:.1f}×)")
print(f"At 256 users : static = {static[255]:5.0f},  dyn = {dynamic[255]:5.0f},  cont = {ccb[255]:5.0f}  ({ccb[255]/dynamic[255]:.1f}×)")
```

The gap is not subtle. **At typical chat loads of 64 concurrent users, Orca's iteration-level scheduling delivers 5–20× the throughput** of the static and dynamic approaches that preceded it — exactly the number reported in the OSDI paper and reproduced by every serving system since. The static and dynamic curves plateau at a low ceiling because that ceiling is the **useful-work fraction** of the GPU under the padding triangle. Continuous batching does not have a triangle, so it does not have that ceiling.

Net effect for a serving company: you can either serve 10× more users on the same hardware, or run the same load on a tenth of the GPUs. The economic argument is over before the engineering one starts.

## The Phase Mixing Bonus

There is a second-order benefit to iteration-level scheduling that the Orca paper hints at and that vLLM later runs all the way to the bank: **the batch can mix prefill and decode in the same forward pass**.

Recall from [Two Phases, Two Personalities](../07-prefill-vs-decode/) that prefill is compute-bound and decode is bandwidth-bound. If you only run prefills, the HBM bus is idle. If you only run decodes, the tensor cores are idle. **Mix them, and both resources stay warm.** A typical Orca-style batch on a busy server looks like this:

```
batch = [
  Request A : prefill, 4096 input tokens          → consumes compute
  Request B : decode, step 312, KV length 312     → consumes bandwidth
  Request C : decode, step 18,  KV length 818     → consumes bandwidth
  Request D : decode, step 1,   KV length 47      → consumes bandwidth
  ... 28 more decodes ...
]
```

The matrix multiplications in the MLP and the QKV projections see a flat batch of $(4096 + 31)$ rows — wide enough to be compute-efficient. The attention kernel fans out into 32 per-sequence kernels: one big prefill attention on 4096 queries and 4096 keys, 31 small decode attentions each on 1 query and a few hundred keys.

This is the first concrete payoff of selective batching, and it foreshadows the entire scheduler architecture of [vLLM V1](../14-scheduler/), where prefill and decode are flattened into a single primitive — *tokens budgeted per step* — and the scheduler picks whose tokens get cycles.

## What Orca Did Not Solve

Orca cured static batching. Orca did not cure the **memory allocator**.

If the batch is rebuilt every iteration and a new request can join at any moment, the question is: **where does its {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} live?** The simple answer Orca used — and the one HuggingFace Transformers and most pre-2023 serving stacks used — was *pre-allocate a slot of `max_len` per active sequence*. With 32 slots and a 4K max length, that is $(32 \times 4096 \times L \times H \times D \times 2)$ bytes of HBM, reserved upfront, mostly empty.

For Llama-13B at 4K max length, that pre-allocated region is **roughly 100 GB**. The actual sequences in that batch — averaging 800 tokens, not 4,096 — occupy only **20 GB** of KV. Eighty gigabytes of HBM are sitting locked, holding the contractually-guaranteed slot length for sequences that will never reach it. **The utilization is 20%.** You can serve 32 users only because you allocated for 32 worst-cases, and most of the HBM is doing nothing.

So Orca turned the GPU into a conveyor belt; the bottleneck migrated. **The compute is happy. The allocator is now the problem.** Every byte of HBM that the allocator wastes is a byte that could have held another user's KV, which means another conveyor-belt slot, which means another increment of throughput. The economics that were screaming at the kernel are now screaming at the heap.

That is the problem the next chapter opens with — and the problem that, in mid-2023, a group at Berkeley named Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, and Ying Sheng decided to solve by reading a paper from **1965**.

**Continue to → [The KV Cache Is a Heap](../09-kv-fragmentation/)** — the allocator hell that Orca exposed, the operating-systems trick that finally fixed it, and the paper that turned vLLM into the default inference engine of the LLM era.
