---
title: "LLM Inference: one packet, one box, eighteen layers"
short_title: "LLM Inference"
description: "One HTTP request enters an 8×H200 chassis in Ohio shared by 793 users — this chapter names every layer it touches, from tokenizer to streamed reply, before the rest of the issue explains each one."
blurb:
  - "User #347 of 793 on a single box: Anya's 12-token prompt joins a 312-token system prompt and runs through a 70B FP8 model sliced across four GPUs."
  - "325 milliseconds from send to first character — each millisecond is a different piece of the stack."
  - "Every optimization in the issue is already running: paged KV, continuous batching, prefix caching, CUDA graphs."
  - "The cold open names them all. The other seventeen chapters explain why each one had to be invented."
topics: [inference, vllm]
tags: [vllm, gpu, inference, serving, cold-open]
theme: cream
math: true
draft: false
date: 2026-05-16T09:00:00-04:00
issue: 8
weight: 10
techKind: mainline
techNode: cold-open
header: 01-cold-open-08.webp
---

## Anya, Eight Hundred Strangers, And A Box In Ohio

It is **3:42 PM in Toronto on Tuesday, June 9, 2026**. Anya Sokolova, twenty-eight, an editor at a small history magazine, is on the Yonge subway line writing a piece on collapse narratives. She opens the ChatGPT app on her phone, taps the input field, and types:

> *Why did the Roman Empire fall?*

Forty-seven characters. She hits send.

Three hundred and twenty-five milliseconds later, the word **"The"** appears in the reply pane. Two seconds after that, a paragraph is streaming into her phone at sixty words per second, a fluent essay on Edward Gibbon, on grain shortages, on the Crisis of the Third Century, on currency debasement. By the time the train pulls into Bloor station the answer is finished. Anya copies a phrase into her notes and moves on with her day.

She has no idea that her request just lived, for those three seconds, inside an **8×NVIDIA H200 chassis** in an OpenAI data center in **New Albany, Ohio**, sharing a 632-watt thermal envelope with **seven hundred and ninety-two** other concurrent users. Nor that her twelve-token prompt set off a cascade of operating-system tricks invented in 1965, a 2022 batching algorithm born in Korea, a 2023 paper from Berkeley about page tables, and a tiled attention kernel that one Stanford PhD student wrote on a whiteboard in 2022. None of this existed five years ago. Most of it did not exist three years ago.

The plan of this issue is to take Anya's three seconds apart, frame by frame, and explain every component that touched her token. This chapter is the **establishing shot**: every layer named, no mechanism unpacked. The other seventeen chapters are the unpacking.

{{% marginnote %}}Throughout the issue we use **vLLM V1** (late 2025) as the canonical reference implementation, because it is open, widely deployed, and converges on the same architectural pattern as TensorRT-LLM, SGLang, and TGI.{{% /marginnote %}}

## The Box

The chassis in Ohio is a Supermicro AS-8125GS-TNHR. Eight **H200 SXM modules** ({{< cite text="NVIDIA H200 product brief, 2024" url="https://www.nvidia.com/en-us/data-center/h200/" kind="doc" >}}), each with **141 GB of HBM3e** and **4.8 TB/s of memory bandwidth**, are bolted to a single board through a **third-generation NVSwitch** fabric that lets any GPU talk to any other at 900 GB/s. Two AMD EPYC 9654 CPUs sit underneath, mostly as glorified PCIe hubs. Four ConnectX-7 NICs sit on the back, each one a 400 Gb/s pipe to the data-center spine.

The whole machine draws 6.5 kilowatts at the wall. Two of those kilowatts are HBM. The rest is logic. Air-cooled racks like this one are now considered *quaint*; the next generation of these boxes is liquid-immersion.

Loaded on this particular chassis, sliced four-ways tensor-parallel across the first four GPUs, is **Llama-3.1-70B** quantized to **FP8** ({{< cite text="Touvron et al., Llama 3 herd, 2024" url="https://arxiv.org/abs/2407.21783" kind="paper" >}}). That's 70 GB of {{< wiki "transformer-weights" >}}weights{{< /wiki >}}, sliced into ~17.5 GB per GPU, leaving roughly **120 GB of free HBM on each GPU** for the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}. The other four GPUs are running a different model entirely — a smaller mixture-of-experts for image captioning — but we will pretend they are not there.

Anya is user **#347** of the **793** currently in flight on the LLama-3 half of this box. We will follow her token.

## Step 0 — Wire (the part we will not explain)

Anya's phone sends a TLS 1.3-encrypted HTTP/2 request through a Toronto Cogeco cable head-end, across the Equinix TR2 internet exchange, down a leased fiber pair into a Microsoft Azure edge POP in Chicago, then through OpenAI's backbone to a load balancer in front of the Ohio cluster. The load balancer terminates the TLS, looks at the URL path (`/v1/chat/completions`), and forwards the request to one of several hundred FastAPI worker processes attached to a serving cluster.

This is forty milliseconds of travel time and we will say nothing more about it. Our box opens at the API.

## Step 1 — Tokenizer And The Async Front Door

Inside the vLLM process, the FastAPI handler converts the JSON body into a `ChatCompletionRequest`. The user's text — *"Why did the Roman Empire fall?"* — is concatenated with a baked-in system prompt (*"You are ChatGPT, a large language model trained by OpenAI…"*, 312 tokens) and passed through the model's tokenizer.

The tokenizer is **tiktoken** for OpenAI-flavoured Llama-3: a byte-pair-encoding table with 128,256 entries. It walks the byte string, greedily merging adjacent pairs that exist in the merge table, until no more merges fire. Anya's question collapses to twelve token IDs:

```
[6822, 1550, 279, 12131, 21080, 11299, 30]
[ Why  did  the  Roman  Empire  fall  ? ]
```

(seven for her sentence, plus five wrapping the chat template `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`.) Total prompt length after system prepend: **324 tokens**.

The request now lands inside **AsyncLLM**, vLLM's asynchronous front door. AsyncLLM does not run any GPU code itself; it allocates a `request_id`, wraps a `RequestOutput` async iterator that the FastAPI handler will yield tokens from, and ships the prompt across a `multiprocessing.Queue` to the **EngineCore** process — the single Python process that actually owns the eight GPUs.

## Step 2 — The Scheduler Picks A Slot

EngineCore wakes up every iteration of its main loop (every ~30 ms during decode, every ~80 ms when a long prefill is mid-flight) and asks one question: *who runs next?* The answer is computed by the **V1 unified scheduler** ({{< cite text="vLLM V1 architecture, 2025" url="https://docs.vllm.ai/en/latest/design/v1/" kind="doc" >}}).

There are currently **53 requests** in some stage of life on this box's Llama-3 half — 738 more are queued upstream. The scheduler does not distinguish prefill from decode; it allocates a single number, the **token budget**, currently set to **8,192 tokens per iteration**. Each in-flight request is asking for some number of tokens this step:

- 47 requests are decoding: they want 1 token each.
- 5 requests are in the middle of chunked prefill: they want 1,024 tokens each.
- 1 request just arrived: it wants its full 324-token prompt prefilled.

That's $47 + 5{,}120 + 324 = 5{,}491$ tokens this iteration — well under the budget. Anya's request fits. The scheduler tags it `prefill`, allocates her 324 tokens of work to this iteration, and stops accepting new requests because there is no room left in the **KV pool** until the decoding requests release some blocks. Forward link → [The Token Budget](../14-scheduler/).

## Step 3 — KV Cache Allocation

Anya's prompt needs working memory. For each of the 324 tokens, the model will produce one key vector and one value vector at each of the 80 layers, with 8 KV heads per layer at 128 dimensions per head, stored in FP8 (1 byte). The arithmetic is

$$
324 \;\times\; 80 \;\times\; 8 \;\times\; 128 \;\times\; 2 \;\times\; 1 \;=\; 53{,}084{,}160\;\text{bytes} \;\approx\; \textbf{53 MB}.
$$

The KV pool on this GPU is a contiguous **45 GB slab** allocated at server startup and chopped into fixed-size **blocks of 16 tokens** apiece. Each block holds 16 × 80 × 8 × 128 × 2 = 2.6 MB. Anya needs $\lceil 324 / 16 \rceil = 21$ blocks. The **block manager** ({{< cite text="Kwon et al., PagedAttention, 2023" url="https://arxiv.org/abs/2309.06180" kind="paper" >}}) pops 21 free block IDs from a doubly-linked free list, records them in Anya's **block table** (a 21-long array of physical block IDs), and returns. The allocation took roughly 4 microseconds.

Why is the KV cache *paged* rather than contiguous? Because Anya's request will eventually generate an unknown number of tokens — 247 in this case — and naive contiguous allocation would force vLLM to either over-reserve (wasting 70% of HBM) or perform expensive defragmenting copies. The page-table trick borrows from operating-systems work that **Peter Denning** wrote up in 1965. Forward link → [Borrowing From 1965](../10-paged-attention/), [The Block Manager](../11-block-manager/).

## Step 4 — Prefix Cache Hit

Before Anya's prefill kernel runs, the scheduler does one more check: a **prefix cache lookup**. Anya's first 312 tokens are the standard ChatGPT system prompt, identical byte-for-byte to roughly 60% of all incoming requests on this box. The block manager hashes the first block of the prompt (`hash(tokens[0:16]) = 0xc4e1...`) and finds an entry in the prefix cache: **19 blocks** of pre-computed K/V for the system prompt, resident since 6:14 AM, reference count currently at 412.

vLLM bumps the refcount to 413, points Anya's block table to those 19 shared physical blocks, and reduces her *actual* prefill work from 324 tokens to **20 new tokens** — the 12 tokens of her question plus the 8-token assistant header. Of her allocated 21 blocks, 19 are now shared and 2 are private. Forward link → [Reusing The Prologue](../12-prefix-caching/).

This single optimization is the largest free latency win in the entire inference stack. Without it, Anya's prefill would be 16× more work, and TTFT would be closer to **800 ms** than 280.

## Step 5 — The Forward Pass

The model worker process — one per GPU, four total for this tensor-parallel slice — finally has a job. The 20 new tokens are embedded, passed through 80 transformer layers (TP-sharded so each GPU owns 1/4 of every weight matrix), and the final layer's hidden state becomes a 4096-dim vector for each of the 20 positions. The whole thing is a single **CUDA graph replay**.

```pyplot {id="ttft-latency-stack" caption="Where Anya's 325 ms of time-to-first-token actually goes. The forward pass is a sliver. The wire dominates."}
np.random.seed(8)
labels = [
    "Wire + TLS + LB\n(Toronto → Ohio)",
    "Tokenize + JSON",
    "Queue / wait for slot",
    "KV alloc + prefix\ncache lookup",
    "Forward pass\n(20 tokens, GPU)",
    "Sample top-p\n(softmax + RNG)",
    "Detokenize + HTTP/2\nstream back",
]
times_ms = np.array([42, 4, 28, 1, 5, 0.4, 12])

# Color: GPU work in pink, everything else in teal/yellow/orange
colors = ['#00A8A8', '#FFD700', '#FF8C00', '#FFD700',
          '#FF007F', '#FF007F', '#00A8A8']

fig, ax = plt.subplots(figsize=(9, 4.5))
left = 0.0
for label, t, c in zip(labels, times_ms, colors):
    ax.barh(0, t, left=left, color=c, edgecolor='#1A1A1A', linewidth=1.5, height=0.6)
    if t > 2.5:
        ax.text(left + t/2, 0, f"{t} ms", ha='center', va='center',
                fontsize=9, fontweight='bold', color='#1A1A1A')
    left += t

# Component labels under the bar
left = 0.0
for label, t in zip(labels, times_ms):
    if t > 2.0:
        ax.text(left + t/2, -0.55, label, ha='center', va='top',
                fontsize=8, color='#1A1A1A')
    left += t

total = times_ms.sum()
ax.set_xlim(0, total + 5)
ax.set_ylim(-1.2, 0.7)
ax.set_yticks([])
ax.set_xlabel("milliseconds since 'send' tap")
ax.set_title(f"TTFT stack-up: {total:.0f} ms from Toronto to first character", loc='left')
ax.spines[['top', 'right', 'left']].set_visible(False)

print("Time-to-first-token breakdown:")
for label, t in zip(labels, times_ms):
    short = label.split('\n')[0]
    print(f"  {short:35s}  {t:6.1f} ms   ({t/total:5.1%})")
print(f"  {'TOTAL TTFT':35s}  {total:6.1f} ms")
print()
print(f"Of which: GPU compute = {times_ms[4] + times_ms[5]:.1f} ms ({(times_ms[4]+times_ms[5])/total:.1%})")
print(f"          Wire        = {times_ms[0]:.1f} ms ({times_ms[0]/total:.1%})")
print(f"          Everything else is queueing, allocation, and serialization.")
```

The forward pass itself: roughly **5 milliseconds**. Inside that 5 ms, on each of the four GPUs:

- **132 streaming multiprocessors** ({{< wiki "transformer-weights" >}}weights{{< /wiki >}} streaming in from HBM at 4.8 TB/s) execute matmul kernels on tensor cores. See [Inside the Silicon](../02-gpu-anatomy/).
- A **FlashAttention-3 kernel** ({{< wiki "flash-attention" >}}FlashAttention{{< /wiki >}}; {{< cite text="Shah et al., FlashAttention-3, 2024" url="https://arxiv.org/abs/2407.08608" kind="paper" >}}) computes attention by tiling K and V through SRAM. The 20×324 score matrix is never written to HBM. See [Attention in SRAM](../06-flash-attention/).
- An **all-reduce** runs after every block to combine the tensor-parallel slices over NVLink. ~25 µs per all-reduce, 80 of them, ~2 ms cumulative.
- The whole sequence of 1,200-odd CUDA kernel launches is replayed from a pre-recorded **CUDA graph**, costing 180 µs of CPU overhead instead of the 4 ms the launches would cost individually. See [Launches Aren't Free](../05-cuda-graphs/).

The four GPUs cooperate to produce a single 4096-dim hidden state for Anya's last token position. That hidden state is multiplied by the LM head to yield a **128,256-dim {{< wiki "logit" >}}logit{{< /wiki >}} vector**, the unnormalized probabilities for the next token. See [The Pyramid of Speed](../03-memory-hierarchy/) for why most of those 5 ms are spent not computing but waiting for memory.

## Step 6 — Sampling And The Return Trip

The logit vector lands on GPU 0. The sampler applies temperature 0.7, then top-p truncation at $p = 0.95$ — the smallest cumulative-probability set of tokens that sums to 0.95 gets kept, the rest get zeroed. A {{< wiki "softmax" >}}softmax{{< /wiki >}} normalizes the survivors. A CUDA RNG draws one sample. The winning token ID is **791** — `The`.

That single token ID travels back across the IPC bus from EngineCore to AsyncLLM, gets detokenized to the string `"The"`, gets wrapped in a server-sent-event JSON envelope, gets pushed through HTTP/2 back to Anya's phone, gets decrypted by her phone's TLS stack, and gets rendered in the chat UI's `UITextView`. **TTFT — time to first token — was 325 ms.** The user feels this as instant.

## Step 7 — The Decode Loop

Anya's request now leaves the prefill set and joins the **decode set**: every ~28 ms, EngineCore wakes, the scheduler hands one new-token slot to Anya's request, and the model worker runs a single forward pass that consumes only her previous-token's hidden state and her existing KV cache. This is the **autoregressive loop**, and it is fundamentally different from prefill in one way that this whole issue is about: it loads ~140 GB of model weights and ~12 MB of KV cache from HBM to produce one token. The arithmetic intensity is roughly **1 FLOP per byte**, ninety times below the H200's break-even point.

Forward link → [Two Phases, Two Personalities](../07-prefill-vs-decode/), [The Conveyor Belt](../08-continuous-batching/).

To squeeze more tokens out of the same memory pass, a 1.4B-parameter **draft model** runs ahead of the target, speculating four tokens at a time; the 70B target verifies all four in a single forward pass and accepts whichever prefix it agrees with (typically three of four). See [The Draft Trick](../15-speculative-decoding/).

This loop runs **247 times** for Anya's reply. At ~28 ms per token, the full generation takes **6.9 seconds**. She perceives it as smooth streaming because each token shows up on her screen 30 ms after the previous one — well below the human flicker-fusion threshold for text.

```pyplot {id="hbm-occupancy" caption="KV cache footprint on one H200 GPU during the inference window. Anya is request #347. The cache breathes — allocating during prefill, growing through decode, freeing on completion."}
np.random.seed(13)

# Simulate a 60-second window of KV cache occupancy on one GPU
T = 60
dt = 0.05  # 50ms resolution
t = np.arange(0, T, dt)
N = len(t)

# Background: 53 concurrent users, each with a random prefill/decode trajectory
def request_trace(t_start, prefill_tok, decode_tok, decode_rate=33):
    """Returns KV bytes used at each timestep."""
    bytes_per_token = 80 * 8 * 128 * 2  # ~163 KB at FP8
    trace = np.zeros_like(t)
    prefill_dur = prefill_tok / 40000.0  # ~40k tok/s prefill
    decode_dur = decode_tok / decode_rate
    end = t_start + prefill_dur + decode_dur
    in_window = (t >= t_start) & (t <= end)
    # Linear ramp during prefill, slower ramp during decode
    for i in np.where(in_window)[0]:
        elapsed = t[i] - t_start
        if elapsed < prefill_dur:
            tok = prefill_tok * elapsed / prefill_dur
        else:
            tok = prefill_tok + decode_rate * (elapsed - prefill_dur)
        trace[i] = tok * bytes_per_token
    return trace, end

total = np.zeros_like(t)
anya_trace = np.zeros_like(t)
for i in range(60):
    start = np.random.uniform(0, T - 5)
    prefill = int(np.random.choice([200, 400, 800, 2000, 8000],
                                    p=[0.4, 0.3, 0.15, 0.1, 0.05]))
    decode = int(np.random.uniform(80, 600))
    trace, _ = request_trace(start, prefill, decode)
    total += trace

# Anya: starts at t=20, prefill 324 tok, decode 247 tok
anya_trace, _ = request_trace(20.0, 324, 247, decode_rate=33)
total += anya_trace

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(t, 0, (total - anya_trace) / 1e9, color='#00A8A8',
                alpha=0.6, label='other 52 requests')
ax.fill_between(t, (total - anya_trace) / 1e9, total / 1e9,
                color='#FF007F', alpha=0.95, label='Anya (#347)')
ax.axhline(45, color='#1A1A1A', linewidth=1.5, linestyle='--',
           label='KV pool cap (45 GB)')
ax.set_xlim(0, T)
ax.set_ylim(0, 50)
ax.set_xlabel("time (s)")
ax.set_ylabel("KV cache used (GB)")
ax.set_title("KV cache occupancy on one H200 (one slice of the TP=4 group)", loc='left')
ax.legend(loc='upper right', framealpha=0.95)
ax.spines[['top', 'right']].set_visible(False)

print(f"Peak KV occupancy on this GPU during the window: {total.max()/1e9:.1f} GB")
print(f"Anya's own peak footprint:                       {anya_trace.max()/1e6:.1f} MB")
print(f"Free HBM still available at peak:                {141 - 17.5 - total.max()/1e9:.1f} GB")
print()
print("The pool BREATHES. Without paging, those 60 requests would force")
print("vLLM to reserve max-context per request — easily 4× this footprint.")
```

## Step 8 — The Future, Already Here

In another data center, three states away, the same model is now running in a *disaggregated* topology: the **prefill machines** (compute-heavy, tensor-core saturated) are physically distinct from the **decode machines** (bandwidth-heavy, mostly waiting on HBM). When a request arrives, prefill runs on machine A, the entire KV cache is shipped over **NIXL / Mooncake** ({{< cite text="Qin et al., Mooncake, 2024" url="https://arxiv.org/abs/2407.00079" kind="paper" >}}) RDMA fabric to machine B, and decode continues from there. Three network hops, one hidden state, zero milliseconds of compute idle. Forward link → [Two Houses, Divided](../17-disagg-pd/).

This is the topology OpenAI is rolling out *as Anya is finishing her sentence*. By 2027 it will be the default.

## Step 9 — Done

After 247 tokens, the sampler picks token ID 128009 — `<|eot|>`, end-of-turn. The scheduler marks Anya's request complete, returns her 21 block IDs to the free pool (the 19 shared system-prompt blocks are reference-decremented from 413 to 412; the 2 private blocks are freed outright), and pushes a final empty SSE frame to her phone. The slot is open. Within microseconds, request **#794** from someone in Berlin slots in.

Total elapsed: **7.2 seconds**. Total GPU compute: roughly **2.1 seconds of forward-pass time** across four GPUs. The rest was the queueing, the scheduling, the wire.

{{< crosshead >}}What Just Happened, In One Sentence{{< /crosshead >}}

Anya's twelve-token question was tokenized, scheduled into a token budget, allocated 21 paged blocks of KV cache (19 of which were already in a prefix cache from 9 hours ago), prefilled in a single CUDA graph replay on four tensor-parallel GPUs running FlashAttention-3 kernels, sampled by top-p softmax, and then decoded 247 times in a continuous-batching loop with speculative decoding accepting three of every four guesses, while 792 strangers shared the same hardware and never noticed each other.

## What To Remember

1. **Every engineering decision in modern inference is downstream of one fact**: at decode time, the GPU is loading bytes faster than it is multiplying them. The {{< wiki "transformer-weights" >}}weights{{< /wiki >}} cost ~140 GB to read; the matmul against a single token costs ~140 GFLOPs. The ratio is one. The H200 wants a ratio of ~290. The factor of 290 gap is where the whole rest of the issue lives.

2. **Every layer in the stack is named after a specific 2022–2026 paper.** Orca (continuous batching), PagedAttention (vLLM), Hydragen (prefix caching), Sarathi-Serve (chunked prefill), EAGLE / Medusa (speculative decoding), DistServe / Mooncake (disaggregated P/D). The history is recent, traceable, and unfinished — at least three of these papers will be obsolete by the time you read this.

3. **The tech tree on the cover is the map.** Read the five GPU primers first if names like *streaming multiprocessor* or *HBM bandwidth* feel hand-wavy. Otherwise jump straight to [Two Phases, Two Personalities](../07-prefill-vs-decode/). Everything downstream connects back to the napkin math in this chapter.

{{% pullquote type="counter-intuitive" %}}
The hardest problem in modern LLM serving is not "making the model smarter". It is making seven hundred and ninety-three strangers share the same 1.1 terabytes of HBM bandwidth without any of them noticing the other seven hundred and ninety-two.
{{% /pullquote %}}

**Continue to → [Inside the Silicon](../02-gpu-anatomy/)** — before we can dissect Anya's three seconds, we need to know what kind of machine is on the operating table. Eight of them, in fact, in a chassis in Ohio.
