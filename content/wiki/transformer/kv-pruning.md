---
title: "KV Cache Pruning"
slug: "kv-pruning"
description: "Methods that reduce KV cache memory by evicting low-importance token entries: H₂O, SnapKV, KVzip, KVzap, and the landscape of T-axis compression."
category: "transformer"
also_known_as: ["KV cache compression", "KV eviction", "token eviction", "T-axis compression", "H₂O", "KVzip", "KVzap", "heavy hitters", "SnapKV", "StreamingLLM", "attention sinks", "DuoAttention", "Expected Attention", "KVzip+", "KVzap-Linear", "KVzap-MLP", "compression ratio", "eviction policy", "sliding window", "threshold tau", "top-k eviction", "importance score", "surrogate model"]
source_of_truth: "/comicbook/06-kvcache-pruning/06-kvzap/"
source_of_truth_title: "ch.6 KVzap: The Final Zap (Issue 06)"
related: ["kv-cache", "attention", "residual-stream"]
draft: false
---

KV cache pruning compresses the KV cache along the **T (sequence length) axis** by identifying and discarding token entries whose KV pairs are unlikely to be attended to again.

| Method | Key idea | Limitation |
|---|---|---|
| **H₂O** (2023) | Keep "heavy hitters" with highest accumulated attention | Fixed global budget; kernel-invasive |
| **StreamingLLM** (2023) | Sliding window + attention sinks | Not true pruning; loses long-range info |
| **SnapKV** (2024) | Cluster KV pairs, keep representatives | Prefill-only; clustering overhead |
| **DuoAttention** (2024) | Some heads full KV, some sliding window | Can't adapt to decoding |
| **KVzip** (2025) | Copy-and-paste pretext task for oracle scores | 2× prefill cost; decode-incompatible |
| **KVzap** (2026) | Surrogate model on hidden states | Requires PagedAttention for variable-length caches |

**The four production criteria** (from KVzap paper): fast & lightweight · phase-agnostic (prefill + decode) · optimization-friendly (FlashAttention/PagedAttention compatible) · faithful (minimal accuracy loss).

Full timeline and analysis → [ch.10 — The KV Pruning Family Tree](/comicbook/06-kvcache-pruning/10-pruning-landscape/)
