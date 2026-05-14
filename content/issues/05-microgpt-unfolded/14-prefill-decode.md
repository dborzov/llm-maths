---
title: "Prefill vs Decode"
description: "Two loops sitting around `gpt()`. The first is bandwidth-bound; the second is latency-bound. They are so different they get their own GPUs in modern serving clusters, and the entire vLLM/SGLang/TensorRT-LLM stack is a fight over this distinction."
topics: [transformer, inference]
tags: [microgpt, prefill, decode, serving]
theme: teal
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 140
techKind: mainline
techNode: prefill-decode
header: default.webp
---

## The Berkeley Team That Looked At The Wrong Graph

It is March 2023 and Woosuk Kwon, a graduate student in Ion Stoica's lab at Berkeley, is staring at an `nvidia-smi` dashboard he has had open for six hours. They are running an LLM serving benchmark on a single A100. The GPU is, on paper, capable of 312 TFLOPs of bfloat16 matrix multiplication. The dashboard says it is doing somewhere between **3% and 5%** of that.

The team's first instinct, naturally, was that something was wrong with the framework — maybe an unfused kernel, maybe a synchronization bug, maybe the wrong CUDA stream layout. They rewrote the attention call three different ways. The number did not move.

What finally cracked it, the story goes, was that someone — Kwon, or possibly Zhuohan Li — printed two histograms next to each other. The first was the **per-step compute** during the *prompt-ingestion* phase: a single fat bar, the GPU pegged at 90% utilization, doing one enormous matmul. The second was the **per-step compute** during the *token-generation* phase: a forest of tiny bars, each one a couple of milliseconds long, the GPU mostly idle in between, doing one *thin* matmul per layer over and over again.

It was not one inference workload. It was **two workloads pretending to share a GPU**, and one of them was starving the other. The paper that came out of staring at those two histograms became vLLM. The split they had stumbled into has a name now — **prefill versus decode** — and the entire modern serving stack (vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo) is, fundamentally, a fight over how to manage it.

The reason you can see the split *exactly* is that you have already read the microGPT driver. Pull it up again.

## Two Loops, Mechanically

```python
keys = [[] for _ in range(n_layer)]
values = [[] for _ in range(n_layer)]
prompt_tokens = [uchars.index(ch) for ch in prompt]

token_id = BOS
pos_id = 0

# === PREFILL ===
for p_token in prompt_tokens:
    _ = gpt(token_id, pos_id, keys, values)  # discard logits
    token_id = p_token
    pos_id += 1

# === DECODE ===
for _ in range(block_size - pos_id):
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax([l / temperature for l in logits])
    token_id = random.choices(range(vocab_size), weights=probs)[0]
    if token_id == BOS: break
    sample.append(uchars[token_id])
    pos_id += 1
```

Two `for` loops. Same function `gpt()` in both. **Wildly different semantics.**

**Prefill** walks through `prompt_tokens` and, on every step, calls `gpt(token_id, pos_id, keys, values)` *for its side effect*. The returned `logits` are assigned to `_` — thrown away. The only thing prefill cares about is that inside `gpt()`, the lines

```python
keys[li].append(k)
values[li].append(v)
```

have run for every layer and every prompt position. The cache is the deliverable. The logits are scaffolding.

**Decode** then enters a second loop that *does* keep the logits. Every iteration: one call to `gpt()`, divide by temperature, softmax, sample one `token_id`, append the character, increment `pos_id`, repeat. The cache continues to grow — one more entry per layer per step — but most of the *work* is now in the read path: each new query has to dot-product against every previous key, weight-average every previous value.

If your prompt is 2,000 tokens and you generate 200 tokens, prefill runs `gpt()` 2,000 times and decode runs it 200 times. **Eleven-to-one in call count.** And yet, in wall-clock seconds on a production GPU, the two phases often take *the same amount of time*. That is the inversion that broke the Berkeley team's intuition. To see why, look at what is in each call.

## Why They Are Different Beasts

Look at one line inside the attention block:

$$
\text{attn\_logits}_t = \frac{q \cdot k_t}{\sqrt{D}}, \quad t = 0, 1, \ldots, T-1
$$

During **prefill**, you do not actually run this `T` separate times. Production engines pack all `T` prompt tokens into a single tensor of shape `(T, n_embd)` and run *one* big matmul to produce all queries, *one* big matmul to produce all keys, and *one* triangular-masked $QK^\top$ matmul of shape `(T, T)`. The arithmetic intensity — FLOPs per byte read from memory — is enormous. The matmul units saturate. The GPU is **compute-bound**.

During **decode**, the new query is shape `(1, n_embd)`. The new key is shape `(1, n_embd)`. But to compute attention you must still read *every cached key* — a tensor of shape `(T, n_embd)` per layer — from HBM. The matmul is shaped `(1, n_embd) × (n_embd, T)`, an extremely thin slab. You are now reading more bytes than you are doing FLOPs on. The matmul units sit idle while the memory subsystem grinds. The GPU is **memory-bandwidth-bound**.

Here is the arithmetic in numbers you can hold in your head.

```pyplot {id="prefill-decode-flops" caption="FLOPs per step. Prefill processes the entire prompt in one shot; decode processes one token. The vertical axis is log-scaled because we span three orders of magnitude."}
# Llama-3-8B-ish: n_layer=32, n_embd=4096, head_dim=128, n_head=32
n_layer = 32
n_embd = 4096

# Per-layer FLOPs as a function of "tokens processed in this step"
# Dominant terms: QKV projection (3 * 2 * n_embd^2 per token),
#                 MLP (2 * 2 * n_embd * 4*n_embd per token),
#                 attention dot-products (~ 2 * n_embd * cache_len per token)
def step_flops(tokens_this_step, cache_len):
    proj_per_tok = 2 * (3 * n_embd * n_embd + n_embd * n_embd)  # QKV + out
    mlp_per_tok = 2 * (2 * n_embd * 4 * n_embd)
    attn_per_tok = 2 * 2 * n_embd * cache_len  # QK^T and AV
    per_tok = proj_per_tok + mlp_per_tok + attn_per_tok
    return n_layer * per_tok * tokens_this_step

prefill_len = 2048
prefill_flops = step_flops(prefill_len, prefill_len)            # one big step
decode_flops_per_tok = [step_flops(1, prefill_len + t) for t in range(256)]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar([0], [prefill_flops / 1e12], width=0.6,
       color='#FF007F', edgecolor='#1A1A1A', linewidth=2, label='prefill (2048 tokens, one step)')
ax.bar([1], [sum(decode_flops_per_tok) / 1e12], width=0.6,
       color='#00A8A8', edgecolor='#1A1A1A', linewidth=2, label='decode (256 tokens, 256 steps)')
ax.bar([2], [decode_flops_per_tok[0] / 1e12], width=0.6,
       color='#FFD700', edgecolor='#1A1A1A', linewidth=2, label='one decode step')
ax.set_yscale('log')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Prefill\n(whole prompt)', 'Decode\n(all 256 steps)', 'Decode\n(one step)'])
ax.set_ylabel('TFLOPs (log scale)')
ax.set_title('Llama-3-8B: FLOPs budget by phase')
ax.legend(loc='upper right', frameon=True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
```

Prefill on 2,048 tokens is about **33 TFLOPs**. Decode on 256 tokens is about **4 TFLOPs total** — but spread across 256 separate calls of ~16 GFLOPs each. On an H100 doing 1,000 TF/s of FP8 matmul:

- **Prefill** at 33 TFLOPs / (1,000 TF/s) ≈ **30 ms** — and that's compute-bound, so the H100 does run close to peak here.
- **Decode** at 16 GFLOPs/step *should* take 16 µs/step — but decode is **bandwidth-bound**. To compute one decode step you must read all 16 GB of model weights (a Llama-3-8B at FP8 is ~8 GB; KV cache adds more) from HBM. H100 HBM bandwidth is 3.35 TB/s, so just *reading* the model weights costs ~5 ms per step.

That's a **300× gap** between what the math says decode "should" take and what the memory says it actually takes. Every "decode optimization" paper from 2023 onward is, fundamentally, a way to read fewer bytes per step.

## The Picture That Cracked The Whole Field

Drop the FLOPs framing for a second. What does a *single inference request* look like on a wall-clock timeline?

```pyplot {id="ttft-tpot" caption="Time breakdown of one inference request (2048-token prompt, 256-token generation, H100 FP8). Prefill is one fat block: that is your time-to-first-token. Decode is a fence of thin slats: each slat is one time-per-output-token."}
prefill_ms = 30
tpot_ms = 15
n_decode = 256

fig, ax = plt.subplots(figsize=(10, 3.2))

# Prefill bar
ax.barh([0], [prefill_ms], left=[0], height=0.8,
        color='#FF007F', edgecolor='#1A1A1A', linewidth=2)
ax.text(prefill_ms / 2, 0, 'PREFILL\n(TTFT)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')

# Decode bars
x = prefill_ms
for i in range(n_decode):
    ax.barh([0], [tpot_ms], left=[x], height=0.8,
            color='#00A8A8', edgecolor='#1A1A1A', linewidth=0.3)
    x += tpot_ms

# Annotations
total = prefill_ms + n_decode * tpot_ms
ax.annotate('', xy=(prefill_ms, 0.6), xytext=(0, 0.6),
            arrowprops=dict(arrowstyle='<->', color='#1A1A1A', lw=1.5))
ax.text(prefill_ms / 2, 0.75, f'TTFT = {prefill_ms} ms', ha='center', fontsize=10)

ax.annotate('', xy=(total, -0.6), xytext=(prefill_ms, -0.6),
            arrowprops=dict(arrowstyle='<->', color='#1A1A1A', lw=1.5))
ax.text((prefill_ms + total) / 2, -0.78,
        f'{n_decode} × TPOT = {n_decode * tpot_ms} ms', ha='center', fontsize=10)

ax.set_xlim(-20, total + 20)
ax.set_ylim(-1.1, 1.1)
ax.set_yticks([])
ax.set_xlabel('milliseconds')
ax.set_title(f'One request: {prefill_ms} ms prefill + {n_decode}×{tpot_ms} ms decode = {total/1000:.1f} s total')
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
```

Two metrics fell out of that picture, and they have driven LLM serving-product strategy ever since:

- **TTFT** — **T**ime **T**o **F**irst **T**oken. The width of the pink bar. This is the prefill bottleneck. Users perceive this as *how long until something starts happening*. On a chatbot, anything above ~500 ms feels broken.
- **TPOT** — **T**ime **P**er **O**utput **T**oken. The width of *one* teal slat. This is the decode bottleneck. Users perceive this as *how fast it talks*. Below ~50 ms/token you cannot read fast enough to notice; above ~200 ms/token it feels like a slow modem.

These have separate SLAs. They have separate hardware preferences. They have separate optimization techniques. They are, increasingly, run on **separate GPUs**.

## What This Buys The Serving Stack

Once you can see prefill and decode as two distinct workloads, every production trick falls into one of three buckets: *make prefill faster*, *make decode faster*, *or stop letting them step on each other*.

**Continuous batching.** The classic batched-inference assumption — line up `B` requests, run them through the model in lockstep, write `B` outputs — breaks down because different requests finish decoding at different times. Continuous batching, introduced in Orca (2022) and made famous by vLLM, *swaps requests in and out of the batch at every decode step*. The batch is no longer aligned by sequence — it is aligned by *step*. This pushes decode-phase GPU utilization from 5% to closer to 60% on bursty traffic.

**Chunked prefill.** A 32K-token prompt's prefill takes ~500 ms on an H100 — long enough that any decode steps for other requests queued behind it stall for half a second. Chunked prefill (SARATHI, 2023; default in vLLM today) breaks the prefill into chunks of, say, 512 tokens, and interleaves them with decode steps from other requests. The TTFT for the long-prompt user goes up slightly; TPOT for everyone else stops being held hostage.

**Disaggregated serving.** Prefill is compute-bound; decode is bandwidth-bound. Why are they sharing a GPU? Disaggregated serving (DistServe, 2024; NVIDIA Dynamo, 2025) runs the two phases on *different machines*: prefill on H100s with maximum FLOPs, decode on hardware optimized for memory bandwidth (e.g. H200, B200). When prefill finishes, the populated KV cache is shipped over the interconnect — sometimes NVLink, sometimes InfiniBand — to a decode node, which takes over generation.

**Speculative decoding.** Decode runs one token per step because the model is autoregressive — you cannot start token $t+1$ before you sample token $t$. Speculative decoding cheats: a tiny "draft" model proposes the next $k$ tokens; the big model verifies all $k$ in a single forward pass (the same memory read serves $k$ tokens worth of compute). Accept the longest validated prefix, advance, repeat. On easy tokens you get $k \approx 4$ speedup for free. This trick is *only* worthwhile during decode — prefill is already running every token in parallel.

| Trick | What it accelerates | Where in the driver |
|---|---|---|
| Continuous batching | Decode | the second `for` loop, across many concurrent requests |
| Chunked prefill | Prefill / decode coexistence | the first `for` loop, split into sub-batches |
| Disaggregated serving | Both, by separating | one driver per machine, KV cache shipped between them |
| Speculative decoding | Decode | replaces single `gpt()` call with draft+verify pair |

## One Subtlety: The Discarded Logits

Read the prefill loop one more time. The line

```python
_ = gpt(token_id, pos_id, keys, values)
```

throws away the logits. For plain generation, this is correct: you only need the *last* prefill logits to predict the first new token, and even those can be folded into the first decode step. So engines like vLLM perform a deliberate optimization where prefill runs *with the LM head turned off* for all but the last position. You save one `linear(x, lm_head)` per prompt token — and `lm_head` is the largest single matmul in the network.

But some workloads *do* want the intermediate logits:

- **Logprob inspection.** OpenAI's `logprobs` API parameter returns per-token probabilities, which means the prefill logits are now part of the deliverable. The optimization above turns off.
- **Grammar-constrained decoding.** Tools like Outlines or `lm-format-enforcer` mask logits against a regex/JSON grammar. If the prompt itself contains constrained portions (rare, but happens), prefill logits matter.
- **Token healing.** Some tokenizers produce different token sequences depending on whether you tokenize the prompt + completion together or separately. Engines that "heal" the seam between prompt and completion peek at the last prefill logit to choose.

microGPT, being a teaching artifact, takes the simplest possible stance: discard everything during prefill, generate fresh during decode. That is the default. The exceptions are why production engines have option flags for it.

## Napkin Math

A concrete request on a Llama-3-8B at FP8 on a single H100:

- Prompt: 2,048 tokens. Per-token model FLOPs (rough) ≈ 16 GFLOPs. Prefill total: $2{,}048 \times 16 \approx 33$ TFLOPs.
- H100 FP8 peak: ~1,000 TFLOPs/s. Prefill is compute-bound, so realistic throughput is ~70% of peak = 700 TFLOPs/s.
- **TTFT** $\approx 33 / 700 \approx 47$ ms (we'll call it 50 ms).
- Decode reads ~8 GB of FP8 weights per step (plus a growing KV slice). H100 HBM: 3.35 TB/s. Reading 8 GB: $8 / 3{,}350 \approx 2.4$ ms. Add overhead and growing cache reads → ~15 ms/step.
- 256 generated tokens × 15 ms = **3,840 ms = 3.84 s**.
- Total wall-clock: 50 ms TTFT + 3.84 s decode ≈ **3.9 seconds**.

Notice the asymmetry. Prefill is **1.3%** of the wall-clock, even though it processed **89%** of the tokens (2,048 / 2,304). One block of compute, brief and bright, followed by 256 slow drips. *This* is the shape that every modern serving optimization is trying to flatten.

## What To Remember

1. **microGPT has two loops, doing two different things.** Prefill calls `gpt()` for its side-effect on `keys` and `values` — it discards logits. Decode calls `gpt()` for the logits and samples from them.
2. **Prefill is compute-bound; decode is memory-bandwidth-bound.** Same code, same `gpt()`, opposite hardware bottlenecks. This is *the* asymmetry of LLM serving.
3. **TTFT vs TPOT are two different SLAs.** Time-to-first-token is the prefill bottleneck; time-per-output-token is the decode bottleneck. Each has its own optimization techniques.
4. **Modern serving tricks each target one phase.** Chunked prefill interleaves the two; continuous batching amortizes decode; disaggregated serving runs them on separate hardware; speculative decoding multiplies decode throughput by ~4×.
5. **Prefill logits are usually thrown away.** Only the last one matters for plain generation, so production engines skip the LM head on inner prefill positions — unless `logprobs`, grammar constraints, or token healing are on.

---

**Continue to** → [LM Head and Sampling](../15-sampling/) — we have spent fourteen chapters producing one logit vector per call; now we will turn that vector into the integer that becomes the next character.

