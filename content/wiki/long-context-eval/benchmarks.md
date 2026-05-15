---
title: "Long-Context Benchmarks"
slug: "long-context-benchmarks"
description: "Named benchmarks for evaluating long-context model capability: NIAH, U-curve, MRCR, GraphWalks, OOLONG."
category: "long-context-eval"
also_known_as: ["NIAH", "Needle in a Haystack", "U-curve", "lost in the middle", "MRCR", "Multi-Round Co-reference Resolution", "GraphWalks", "OOLONG"]
source_of_truth: "/issues/04-heatmap-that-lied/03-niah-mechanics/"
source_of_truth_title: "Issue 04 — The Heatmap That Lied"
related: ["long-context-concepts"]
draft: false
---

| Benchmark | What it tests | Source |
|---|---|---|
| **NIAH** | Retrieve a single planted sentence from a long haystack. Industry standard 2023–2024; saturated early 2024. | [ch.3](/issues/04-heatmap-that-lied/03-niah-mechanics/) |
| **U-curve** | Position-dependent accuracy — highest at start and end, lowest in the middle. (Liu et al., 2023) | [ch.7](/issues/04-heatmap-that-lied/07-lost-in-the-middle/) |
| **MRCR** | Disambiguate the *k*-th of *N* similar requests in a long context. Standard retrieval benchmark from 2025. | [ch.9](/issues/04-heatmap-that-lied/09-latent-structure/) |
| **GraphWalks** | BFS on a hex-hash directed graph encoded in the prompt — requires multi-hop reasoning. (OpenAI, April 2025) | [ch.8](/issues/04-heatmap-that-lied/08-needle-to-graph/) |
| **OOLONG** | Per-chunk classification then aggregation across all chunks — orthogonal to GraphWalks. (Vodrahalli et al., Nov 2025) | [ch.9](/issues/04-heatmap-that-lied/09-latent-structure/) |
