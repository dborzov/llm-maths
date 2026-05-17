---
title: "Splitting the Model"
description: "Tensor parallelism splits weight matrices column-by-column across GPUs, with one all-reduce per transformer block. Pipeline parallelism stacks layers across machines. Both fit large models into finite HBM — the trade-offs are latency (all-reduce cost) versus throughput (micro-batch fill)."
topics: []
tags: []
theme: teal
math: true
draft: false
date: 2026-05-16T16:30:00-04:00
issue: 8
weight: 160
techKind: primer
techNode: tp-pp
header: 16-tp-pp.webp
---

## NVIDIA, October 2019: A Model That Won't Fit

It is **October 2019**. **Mohammad Shoeybi**, **Mostofa Patwary**, and a small team of engineers at NVIDIA Research are trying to train a transformer with **8.3 billion parameters**. The standard practice at the time, two years after the Vaswani et al. paper, is "make sure your model fits in one GPU's memory." The V100 they're using has 32 GB. A naive BF16 store of the model is 16.6 GB, which fits — until you add the optimizer state (Adam keeps two extra momentum tensors per parameter, so a 16.6 GB model becomes a 67 GB working set) and the activations of a long training sequence.

Their model does not fit. Not by a factor of two.

You can imagine the conversations. The obvious answer is "buy a bigger GPU." It does not exist yet. The next obvious answer is "use a smaller model." That's the answer they are trying to *avoid*. The third answer — the one that ends up on the whiteboard — is **break the model**. Slice the weight matrices in half. Put one half on GPU 0, the other half on GPU 1. Have them work together on the same forward pass.

The paper they write, *"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"* ({{< cite text="Shoeybi et al., 2019" url="https://arxiv.org/abs/1909.08053" kind="paper" >}}), describes a recipe for slicing a transformer block that minimizes inter-GPU communication. They call it **tensor parallelism**. By the end of 2022 it is the default way every research lab on Earth trains a model larger than 10B parameters. By 2024 it is the default way every *inference* serving system handles a model larger than its biggest GPU.

The reason this primer exists is that the next chapter on disaggregated prefill/decode assumes you already know what TP and PP mean, and the chapter before it on speculative decoding ran on a model that was already split. Let's catch up.

## The Forcing Function: 140 GB Doesn't Fit On 80

Concrete anchor: **Llama-3-70B in BF16**. The weights are roughly $70 \times 10^9$ parameters $\times\,2$ bytes = **140 GB**. The largest single GPU you can buy in mid-2024 is the **H100 SXM with 80 GB of HBM3**. The H200 (141 GB) just barely fits the weights but leaves no room for KV cache. The B200 (192 GB) leaves 50 GB — okay-ish for low concurrency.

For most production deployments in 2024–25, the math doesn't close on a single GPU. You need either:

- **Tensor parallelism (TP).** Split each layer's weight matrices across $K$ GPUs. They share input and output activations but each holds only $1/K$ of every weight.
- **Pipeline parallelism (PP).** Give each GPU a contiguous block of *layers*. Activations flow through the pipeline like a conveyor belt.
- **A combination.** TP within a node (where NVLink is fast), PP across nodes (where Ethernet/IB is slower).

The two techniques have opposite trade-offs and you cannot pick one without doing some napkin math first.

## Tensor Parallelism: The Megatron Trick

Pick any linear layer in a transformer: $Y = XW$, where $X$ has shape $(B, T, D)$ and $W$ has shape $(D, 4D)$ for the {{< wiki "mlp-block" >}}MLP up-projection{{< /wiki >}}. The output $Y$ has shape $(B, T, 4D)$.

The trick: shard $W$ along its **output dimension**. GPU 0 holds $W_0$ of shape $(D, 2D)$ — the first half of the columns. GPU 1 holds $W_1$ of shape $(D, 2D)$ — the second half. Both GPUs receive the *full* input $X$. Each computes its half of the output:

$$
Y_0 = X W_0,\qquad Y_1 = X W_1, \qquad Y = [Y_0 \,\Vert\, Y_1].
$$

GPU 0 holds $Y_0$ of shape $(B, T, 2D)$; GPU 1 holds $Y_1$ similarly. **No communication needed yet** — each GPU has done its half independently and we have not concatenated the result.

This is the **column-parallel** pattern: shard $W$ along output, no comm at the end if the next op is row-friendly.

Now the {{< wiki "mlp-block" >}}MLP down-projection{{< /wiki >}} comes next: $Z = Y W'$ where $W'$ has shape $(4D, D)$. The clever move is to shard $W'$ along its **input dimension** — its rows. GPU 0 holds $W'_0$ of shape $(2D, D)$; GPU 1 holds $W'_1$ of shape $(2D, D)$. Each GPU computes a *partial* result using its half of $Y$:

$$
Z_0 = Y_0 W'_0,\qquad Z_1 = Y_1 W'_1, \qquad Z = Z_0 + Z_1.
$$

**Now** we must communicate: the final $Z$ is the *sum* of partial results from both GPUs. That sum is an **all-reduce** over the GPUs in the TP group. After the all-reduce, both GPUs hold the full $Z$ of shape $(B, T, D)$ and the next layer can begin.

This is the **row-parallel** pattern: shard $W$ along input, all-reduce at end.

The two patterns chain together beautifully. **One column-parallel layer followed by one row-parallel layer requires exactly one all-reduce** for the pair. A transformer block has two such pairs — one in the attention block (Q/K/V column-parallel; output projection row-parallel) and one in the MLP block (up column-parallel; down row-parallel). So **two all-reduces per transformer block**, regardless of TP degree.

{{% marginnote %}}There is a third "internal" all-reduce in the attention block if you compute softmax across sharded query heads — but a common simplification is to assign whole heads to specific GPUs, so the softmax is local and the only post-attention comm is the row-parallel output projection's reduce. Megatron's original paper does it this way.{{% /marginnote %}}

That is **Megatron-style tensor parallelism**: column-parallel + row-parallel, in alternation, with one all-reduce at the junction. The pattern is so clean it deserves a picture.

```python
# Tensor parallelism, schematically (TP=2). Each GPU holds half the weights.
# The two GPUs execute these lines together; the all-reduce is the join point.

# Attention block (Q, K, V are column-parallel; O is row-parallel)
q = X @ Wq_local           # X: (B,T,D)   Wq_local: (D, D/2 per head shard)
k = X @ Wk_local
v = X @ Wv_local
attn_out = attention(q, k, v)      # local to each GPU's head shard
y = attn_out @ Wo_local            # row-parallel; partial sum per GPU
y = all_reduce(y)                  # NCCL: now both GPUs have full y
y = layernorm(y + X)

# MLP block (up is column-parallel; down is row-parallel)
h = y @ W_up_local         # local
h = gelu(h)                # element-wise, no comm
out = h @ W_down_local     # row-parallel; partial sum per GPU
out = all_reduce(out)      # NCCL: full output
out = layernorm(out + y)
```

## The Cost: All-Reduce Bandwidth

The all-reduce moves $(B, T, D)$ floats per call across the TP group. For Llama-70B at $D = 8192$, BF16, batch $B = 8$, $T = 1$ (decode): $8 \times 1 \times 8192 \times 2$ bytes = **131 KB per all-reduce**. Two all-reduces per layer × 80 layers = **21 MB per forward pass**.

On an **NVLink-4** fabric (H100 SXM, NVSwitch-3): aggregate bandwidth per GPU is roughly 900 GB/s. 21 MB / 900 GB/s = **23 microseconds** of total comm time per forward pass — across an 80-layer model, on a TP=4 group, at decode batch 8. Negligible compared to the ~25 ms HBM-bound decode time.

On an **NVLink-5** fabric (B200, NVSwitch-4): 1.8 TB/s. Even cheaper.

This is the reason tensor parallelism *works at decode time*: NVLink is fast enough that the per-layer all-reduce hides behind the HBM-bound matmuls. Across regular Ethernet — even 100 Gb/s — it would not.

```pyplot {id="tp-comm-cost" caption="ALL-REDUCE WALL TIME VS TP DEGREE AT THREE BATCH SIZES, LLAMA-70B DECODE. NVLINK STAYS SUB-MILLISECOND OUT TO TP=8."}
np.random.seed(7)

# Llama-70B-ish: D = 8192, 80 layers, BF16 (2 bytes)
D = 8192
n_layers = 80
bytes_per_elem = 2
T = 1               # decode step

# Per-step comm bytes per GPU (ring all-reduce: 2*(N-1)/N * tensor_size)
tp_degrees = np.array([1, 2, 4, 8, 16])
batches = [1, 8, 64]

# NVLink-4 effective bandwidth (per GPU, aggregate over the ring)
bw_nvlink = 900e9    # 900 GB/s aggregate
# Inter-node IB-HDR
bw_ib     = 25e9     # 25 GB/s effective

fig, ax = plt.subplots(figsize=(9, 4.6))
colors = {'1': '#FFD700', '8': '#FF8C00', '64': '#FF007F'}

for B in batches:
    tensor_bytes = B * T * D * bytes_per_elem
    # 2 all-reduces per layer (attn out + mlp out)
    comm_per_step = []
    for N in tp_degrees:
        if N == 1:
            comm_per_step.append(0.0)
            continue
        # Ring all-reduce: data volume per GPU = 2*(N-1)/N * tensor_bytes per AR
        per_ar = 2 * (N - 1) / N * tensor_bytes
        total_bytes = per_ar * 2 * n_layers
        # NVLink within a node up to TP=8; IB beyond
        bw = bw_nvlink if N <= 8 else bw_ib
        comm_per_step.append(total_bytes / bw * 1000)  # ms
    ax.plot(tp_degrees, comm_per_step, '-o', linewidth=2,
            markersize=7, color=colors[str(B)],
            markeredgecolor='#1A1A1A',
            label=f'batch B={B}')

ax.axhline(0.5, color='#1A1A1A', linestyle='--', linewidth=1,
           label='0.5 ms (decode budget)')
ax.set_xlabel("TP degree")
ax.set_ylabel("all-reduce wall time per forward pass (ms)")
ax.set_xscale('log', base=2)
ax.set_xticks(tp_degrees)
ax.set_xticklabels([str(n) for n in tp_degrees])
ax.set_ylim(0, max(2.5, comm_per_step[-1]*1.1))
ax.set_title("Tensor-parallel comm cost: cheap on NVLink, expensive across the IB hop",
             fontsize=11, loc='left')
ax.legend(loc='upper left')
ax.spines[['top','right']].set_visible(False)

print("Llama-70B decode, per-step all-reduce cost across the TP group:")
for B in batches:
    tensor_bytes = B * T * D * bytes_per_elem
    print(f"  B={B}:")
    for N in tp_degrees:
        if N == 1: continue
        per_ar = 2 * (N - 1) / N * tensor_bytes
        total_bytes = per_ar * 2 * n_layers
        bw = bw_nvlink if N <= 8 else bw_ib
        print(f"    TP={N}: {total_bytes/bw*1000:6.3f} ms  "
              f"({'NVLink' if N<=8 else 'IB-HDR'})")
plt.tight_layout()
```

The story the plot tells: on NVLink, TP scales painlessly to TP=8. The moment you cross the node boundary into Infiniband-HDR territory at TP=16, comm time jumps two orders of magnitude. **That is the rule of thumb in one picture.** TP within the NVLink island; pipeline (or something else) across the island boundary.

{{% callout type="tangent" %}}
**NCCL: latency or bandwidth?** The NVIDIA Collective Communication Library implements ring all-reduce — $N-1$ steps of sending $1/N$ of the data to your right neighbour and receiving the same from your left. Per-step latency is around 1 µs per hop on NVLink. So a ring of $N = 8$ GPUs is at minimum $8 - 1 = 7$ µs of *latency* just to start the reduce. For small messages (decode batch 1, tensor under 100 KB), you are entirely **latency-bound** — bandwidth doesn't help. For large messages (prefill at batch 64, tensor in the MB range), you are **bandwidth-bound** and the ring fully exploits the fabric. NCCL has a tree all-reduce variant that lowers latency for small messages, but it costs bandwidth — and the tuning of which collective to use when is one of those low-level details that production inference operators end up obsessed with.
{{% /callout %}}

## Pipeline Parallelism: The GPipe Bubble

Now suppose the model is so large that even TP=8 within a single NVLink island leaves no room for KV cache. (DeepSeek-V3 in BF16 is 1.3 TB. A B200 has 192 GB. You need 8 GPUs just for the weights, and then another 8 for KV — across two nodes.)

The other axis is **pipeline parallelism**, popularized by **GPipe** ({{< cite text="Huang et al., 2018" url="https://arxiv.org/abs/1811.06965" kind="paper" >}}, Google Brain, 2018) and **PipeDream** (Microsoft Research, 2018). Instead of slicing each layer, slice the *model* into contiguous chunks of layers and assign each chunk to a GPU.

GPU 0 holds layers 0–19. GPU 1 holds layers 20–39. GPU 2 holds 40–59. GPU 3 holds 60–79.

A forward pass becomes a *pipeline*: tokens flow into GPU 0, which produces hidden states; those flow to GPU 1, then GPU 2, then GPU 3, which produces the final logits. Each GPU works only on its slice of layers. No weight is duplicated.

The catch is the **pipeline bubble**. When the first batch enters the pipeline, GPUs 1, 2, 3 are idle. When the last batch exits, GPUs 0, 1, 2 are idle. To keep the pipeline busy you split the batch into $m$ **micro-batches** and feed them sequentially. The bubble fraction (idle time / total time) is approximately:

$$
\text{bubble} \;\approx\; \frac{n_\text{stages} - 1}{n_\text{stages} + m - 1}.
$$

For 4 stages with 1 micro-batch: bubble = 3/4 = **75% idle**. With 16 micro-batches: 3/19 = **16% idle**. With 64: 3/67 = **4% idle**.

```pyplot {id="pipeline-bubble" caption="PIPELINE BUBBLE FRACTION VS MICRO-BATCH COUNT FOR 4, 8, 16-STAGE PIPELINES. MORE MICRO-BATCHES → LESS WASTED GPU."}
np.random.seed(13)
m = np.arange(1, 64)
stages_list = [4, 8, 16]
colors = {'4': '#FF007F', '8': '#FF8C00', '16': '#00A8A8'}

fig, ax = plt.subplots(figsize=(9, 4.6))
for n in stages_list:
    bubble = (n - 1) / (n + m - 1)
    ax.plot(m, bubble * 100, '-o', linewidth=2, markersize=4,
            color=colors[str(n)], markeredgecolor='#1A1A1A',
            label=f'PP={n}')

ax.axhline(10, color='#1A1A1A', linestyle='--', linewidth=1,
           label='10% bubble (practical floor)')
ax.set_xlabel("micro-batches per scheduler step (m)")
ax.set_ylabel("pipeline bubble (% of GPU time idle)")
ax.set_title("Pipeline bubble decays as 1/m — need many micro-batches to fill it",
             fontsize=11, loc='left')
ax.set_ylim(0, 100)
ax.legend(loc='upper right')
ax.spines[['top','right']].set_visible(False)

print("Pipeline bubble examples:")
for n in stages_list:
    print(f"  PP={n}:")
    for mm in [1, 4, 16, 64]:
        b = (n - 1) / (n + mm - 1)
        print(f"    m={mm:>2}: bubble = {b*100:5.1f}%")

plt.tight_layout()
```

The lesson: pipeline parallelism only earns its rent at **high micro-batch counts**. For *training*, where you naturally have a big global batch you can slice up, this works. For *inference* — especially low-latency, low-concurrency inference where one user wants the next token fast — micro-batches are scarce and the bubble eats you alive.

Pipeline parallelism's natural home is **throughput-oriented serving with many concurrent users**: each user's request is its own micro-batch, $m \gg n_\text{stages}$, the bubble is small, and the model gets to be enormous.

## Expert Parallelism: The New Collective

Mixture-of-experts models — **Mixtral**, **DeepSeek-V3**, **Qwen3-MoE**, **GPT-4o**'s rumored architecture — have a wrinkle that neither TP nor PP cleanly handles.

In an MoE layer, the MLP block is replaced by $E$ separate "expert" MLPs and a small **router** that picks $k$ of them per token (usually $k = 2$). Each token's hidden state is dispatched to its chosen experts, the experts compute, and the results are combined.

If you have 256 experts (DeepSeek-V3's number) and 8 GPUs, the obvious thing is to put 32 experts on each GPU. That's **expert parallelism (EP)**. But now the routing produces a problem: every token, on every GPU, has to *travel* to the GPU that owns its chosen experts. The collective operation that moves tokens to their experts is **all-to-all** — every GPU sends some tokens to every other GPU and receives some from each.

All-to-all is fundamentally different from all-reduce in two ways:

- The data volume scales with **batch × top-k**, not just hidden size.
- It is a **bandwidth-stress** collective rather than a latency-stress one. Every GPU's outgoing pipe is fully utilized in both directions.

For MoE serving, the all-to-all is *the* critical path. NVLink switches were redesigned (NVLink-5 in Blackwell) specifically to make all-to-all faster. The DeepSeek-V3 paper spends pages on the all-to-all kernel and the dispatch scheduling.

The rule of thumb for MoE: **use EP for the experts, TP for the dense parts (attention, router, shared MLPs), PP if you must.** Hybrid TP+EP+PP topologies are the norm for the largest open-weights MoE models.

## Sequence Parallelism: For Long Contexts

One last variant, especially important for the kind of long-context inference that this issue cares about. **Sequence parallelism (SP)** slices the sequence dimension of activations across the TP group.

The motivation: in a standard TP setup, the **non-tensor-parallelized operations** — layer norm, dropout, residual add — still touch the full $(B, T, D)$ activation tensor on every GPU. For $T = 100{,}000$ (a long-context user) the activation tensor is massive and gets duplicated across all TP GPUs.

SP splits the sequence axis $T$ instead. Each GPU in the TP group owns a different chunk of the sequence. The non-parallel ops become local. At the boundary between SP regions and TP regions, a small all-gather or reduce-scatter converts between layouts.

For long-context inference workloads — RAG with massive context, agentic loops with growing conversation history — SP is the difference between fitting a single sequence at all and not. Most modern serving stacks (vLLM, SGLang, TensorRT-LLM) include SP in their long-context modes.

## Rule Of Thumb For Inference

Putting all of the above into a one-paragraph operating manual:

- **Up to 30B params (Llama-3-8B, Mistral-12B, Llama-3-70B with quantization).** TP=1 or 2. Latency is dominated by per-layer launch and HBM. TP overhead is small but unnecessary.
- **30B to 200B dense (Llama-3-70B at full precision, DeepSeek-V2 dense).** TP=4 or 8 inside one NVLink island. NCCL all-reduce on NVLink is fast enough that decode is not noticeably hurt; KV cache splits proportionally.
- **200B+ dense, or multi-node deployments.** TP within a node; PP across nodes. PP eats throughput on small batches but is the only option when one node's worth of HBM doesn't hold the weights. Disaggregated prefill/decode ([next chapter](../17-disagg-pd/)) is the modern alternative.
- **MoE models.** TP for dense parts, EP for the experts, PP if model is so big it spans nodes. The all-to-all kernel dominates the optimization budget.
- **Long context (>32K tokens).** Add sequence parallelism on top of whatever else you have.

This is what every serving operator's `--tp-size`, `--pp-size`, `--ep-size`, `--sp-size` flags actually mean. The flags are not orthogonal — every combination is a different point on a Pareto frontier of latency vs throughput vs memory.

## What To Remember

1. **TP keeps latency low; PP keeps the model big.** TP within an NVLink island, PP across nodes when one node can't hold the model. Hybrid for the largest builds.
2. **One all-reduce per attention block, one per MLP block.** That's the entire TP critical path. NCCL on NVLink makes it sub-millisecond at any practical batch.
3. **The pipeline bubble decays as $1/m$.** PP needs many concurrent requests (or many micro-batches) to fill the pipe. It is naturally a throughput optimization, not a latency one.
4. **MoE introduces a new collective.** All-to-all for token-to-expert routing. Expert parallelism is its own discipline and the all-to-all kernel is the new critical path.

**Continue to → [Two Houses, Divided](../17-disagg-pd/)** — the most architecturally radical recent change in LLM serving: prefill and decode no longer share a machine at all. The parallelism axes we just covered get reshuffled across two physically separate clusters connected by an RDMA KV-cache fabric.
