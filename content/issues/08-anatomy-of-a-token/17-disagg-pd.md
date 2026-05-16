---
title: "Two Houses, Divided"
description: "Prefill needs high compute throughput; decode needs high memory bandwidth. Running them on the same GPU is the worst of both worlds. Disaggregated prefill/decode splits the job across two purpose-built machine types connected by an RDMA KV-cache fabric — DistServe, Mooncake, and NIXL are the field's answers."
topics: []
tags: []
theme: cream
math: true
draft: false
date: 2026-05-16T17:00:00-04:00
issue: 8
weight: 170
techKind: mainline
techNode: disagg-pd
header: default.png
---


## Anchor and Frame

- **History line.** Patel et al., Microsoft, *Splitwise* (2023). Zhong et al., PKU/Berkeley, *DistServe* (OSDI 2024). Moonshot, *Mooncake* (Kimi's serving platform, 2024). NVIDIA, *NIXL* (2025). The arc: research idea → production deployment → standardized transport layer.
- **Anchor case.** A workload mix: 30% of requests have long prompts (RAG, code review). Putting them on the same GPUs as the decode workers means decode workers get blocked by long prefills (head-of-line). With disagg: prefill cluster of compute-rich GPUs (H100/B200), decode cluster of bandwidth-rich GPUs (H200/MI300), connected by RDMA.

## Outline

### Why Disaggregate

- Prefill is compute-bound; decode is bandwidth-bound. Their *ideal hardware* is different:
  - **Prefill machine:** maximize FLOPs per dollar (tensor cores, high TDP).
  - **Decode machine:** maximize HBM bandwidth per dollar (HBM3e capacity, NVLink interconnect).
- Even on identical hardware, *isolation* helps: a long prefill on a decode machine ruins its ITL SLA.

### What Has To Move

- After prefill, the KV cache for the prompt lives on the prefill machine's HBM. It must be shipped to the decode machine *before the first decode step can start*.
- Size: for Llama-70B at 4K tokens, that's ~640 MB of KV. At RDMA wire speed (200 Gb/s = 25 GB/s), ~25 ms. Acceptable for TTFT; intolerable if synchronous on critical path.

### Hiding The Transfer

- **Layer-by-layer streaming.** As prefill computes layer $\ell$, ship layer $\ell$'s KV blocks while computing layer $\ell+1$. By the time prefill finishes, transfer is mostly done.
- **Block-by-block from prefix cache.** If a prefix-cache hit lives on the prefill cluster, it can be shipped first and overlapped with the truncated prefill.

### NIXL: NVIDIA Inference Xfer Library

- A unified abstraction over RDMA, NVLink, and PCIe. Page-aligned, zero-copy, exposes KV blocks as a memory region for one-sided RDMA reads/writes.
- The interface is per-block: prefill registers `(prefill_request_id, block_id) → physical_block_addr`; decode requests a read; data lands at decode's block manager-allocated address.

### Mooncake: KVCache-Centric Architecture

- Moonshot's serving platform. Adds a *third* role: a KV cache fabric — DRAM + NVMe — that holds warm KV cache across machines and decouples both prefill and decode workers from owning persistent state.
- Useful for *prefix-cache aware* routing: route a request to the prefill machine that already has the most prefix cached.

### The Scheduler, Distributed

- Two coordinating schedulers (one per cluster). The control plane decides which prefill machine handles a request based on cache locality and load; once prefill is done, hands off to a decode machine with available KV capacity.

### When Disagg Is Not Worth It

- Small models (≤13B): the gains don't justify the operational complexity.
- Workloads with very short prompts: prefill is cheap, mixing is fine.
- The economics flip with workload mix. Most production teams run *both* a co-located fleet and a disaggregated fleet and route accordingly.

## Connections

- ← [The Token Budget](../14-scheduler/), [Borrowing from 1965](../10-paged-attention/) — paged KV is what makes the per-block transfer practical.
- ← [Splitting the Model](../16-tp-pp/) — TP/PP is parallelism over layers/dims; this is parallelism over *role*.
- → [The Full Anatomy of a Token](../18-full-anatomy/) — the boss chapter spans this whole topology.

## What To Remember

1. **Prefill and decode want different machines.** Compute-rich vs bandwidth-rich. Disagg pairs them with the right SKU.
2. **The KV cache is the unit of transfer.** Paged blocks plus RDMA make this a layer-by-layer streamed handoff, not a stop-the-world copy.
3. **Mooncake adds a third tier**: a KV cache fabric that lives between prefill and decode and decouples both from ownership of state. The cache is now part of the storage hierarchy.

**Continue to → [The Full Anatomy of a Token](../18-full-anatomy/)** — back to Anya's request, with every layer of the stack now named.

