---
title: "Long-Context Evaluation Concepts"
slug: "long-context-concepts"
description: "Conceptual vocabulary for long-context evaluation: context rot, LSQ, benchmark saturation, retrieval vs reasoning, the 2026 eval stack."
category: "long-context-eval"
also_known_as: ["context rot", "LSQ", "Latent Structure Queries", "benchmark saturation", "Goodhart's ceiling", "retrieval is not reasoning", "2026 layered eval stack", "Michelangelo framework"]
source_of_truth: "/comicbook/04-long-context-bench/12-context-rot/"
source_of_truth_title: "Issue 04 — The Heatmap That Lied"
related: ["long-context-benchmarks"]
draft: false
---

| Term | Definition | Source |
|---|---|---|
| **Context rot** | Performance degrades as input length grows: plateau at short lengths, steep drop at 50–75% window, partial recency reflex at end. (Chroma report, July 2025) | [ch.12](/comicbook/04-long-context-bench/12-context-rot/) |
| **LSQ** (Latent Structure Queries) | Michelangelo framework: a good long-context task requires the model to chisel away irrelevant context to expose a latent structure, then query it. (Vodrahalli et al., 2024) | [ch.9](/comicbook/04-long-context-bench/09-latent-structure/) |
| **Benchmark saturation** | Every frontier model clusters near the ceiling; variance < noise; ranking becomes unreliable. | [ch.6](/comicbook/04-long-context-bench/06-measurement-saturation/) |
| **Retrieval is not reasoning** | A model that retrieves single facts does not necessarily chain dependent lookups or aggregate across many facts. | [ch.5](/comicbook/04-long-context-bench/05-saturation/) |
| **2026 layered eval stack** | Five-layer framework: (1) Retrieval, (2) Multi-hop reasoning, (3) Aggregation, (4) Realistic application, (5) Agentic. | [ch.16](/comicbook/04-long-context-bench/16-eval-stack-2026/) |
