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


## Anchor and Frame

- **Anchor case.** Llama-3-70B in BF16 is 140 GB. Doesn't fit on an 80 GB H100. Choose between: TP=2 (split the layers in half, two GPUs work as one), PP=2 (each GPU runs half the layers, micro-batch flow), or sharding the weights with ZeRO-style offload. For inference: TP for low latency, PP for big models / large batches.
- **History.** Megatron-LM (NVIDIA, 2019) for TP; GPipe (Google, 2018) and PipeDream (MSR, 2018) for PP.

## Outline

### Tensor Parallelism

- The trick: a linear $Y = XW$ where $W$ is $D \times 4D$ can be sharded column-wise. Each GPU holds $D \times D$, computes its slice of $Y$, then all-reduces if the next op needs the full $Y$.
- Two patterns in transformer blocks:
  - **Column-parallel** (e.g., the up-projection of the MLP): shard $W$ along output dim, no comm at end if the next op is row-parallel.
  - **Row-parallel** (e.g., the down-projection): shard $W$ along input dim, all-reduce at end.
- One block uses one column-parallel + one row-parallel → exactly one all-reduce per attention + one per MLP. Two all-reduces per transformer layer.
- Cost: NCCL all-reduce of a $(B, T, D)$ tensor at every layer. At TP=4 across an H100 NVLink switch, this is microseconds — fast enough that decode benefits.

### Pipeline Parallelism

- Each GPU owns a contiguous slice of layers. Activations stream through the pipeline.
- The pipeline bubble: when the pipeline is starting/draining, some stages are idle. Bubble fraction ≈ (n_stages − 1) / (n_stages + n_microbatches − 1).
- For inference with many concurrent requests, micro-batches come naturally; PP works well at *high* batch and large models.

### Expert Parallelism (MoE)

- For mixture-of-experts models (Mixtral, DeepSeek-V3, Qwen3-MoE): experts are sharded across GPUs. Routing dispatches each token to its chosen experts via all-to-all. **All-to-all is the new collective on the critical path** — and it is what makes MoE serving fundamentally different from dense.

### Sequence Parallelism

- Variant of TP that also slices the sequence axis through the norm / dropout / residual ops; reduces activation memory. Used in long-context inference.

### Rule of Thumb For Inference

- ≤30B params: TP=1 or 2. Latency dominated by per-layer overhead.
- 30–200B dense: TP=4 to 8 within one NVLink island. NCCL bandwidth is enough.
- 200B+ or multi-node: TP within a node (NVLink) + PP across nodes (Ethernet/IB).
- MoE: TP for the dense parts, EP for the experts.

### Communication Stack

- NCCL on NVLink/NVSwitch within a node; on IB/RDMA across nodes. CUDA-aware MPI on some HPC clusters.
- Latency vs bandwidth: NVLink-5 (Blackwell): 1.8 TB/s aggregate per GPU. IB HDR: 200 Gb/s = 25 GB/s.

## Connections

- ← [Inside the Silicon](../02-gpu-anatomy/) — multi-GPU is a chassis topology question.
- → [Two Houses, Divided](../17-disagg-pd/) — disagg P/D is a different axis of parallelism: across role, not across layers.

## What To Remember

1. **TP keeps the latency low; PP keeps the model big.** TP within an NVLink island; PP across nodes. Hybrid for the largest models.
2. **One all-reduce per attention, one per MLP.** That's the TP critical path. NCCL on NVLink makes it sub-millisecond.
3. **MoE introduces a new collective**: all-to-all for token-to-expert routing. Expert parallelism is its own discipline.

**Continue to → [Two Houses, Divided](../17-disagg-pd/)** — the most architecturally radical recent change in LLM serving: prefill and decode no longer share a machine at all.

