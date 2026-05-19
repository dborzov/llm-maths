---
title: "Long-Context Benchmarks"
slug: "long-context-benchmarks"
description: "Named benchmarks for evaluating long-context model capability: NIAH, U-curve, MRCR, GraphWalks, OOLONG."
category: "long-context-eval"
also_known_as: ["NIAH", "Needle in a Haystack", "U-curve", "lost in the middle", "MRCR", "Multi-Round Co-reference Resolution", "GraphWalks", "OOLONG", "NoLiMa", "No Literal Match"]
source_of_truth: "/comicbook/04-long-context-bench/03-niah-mechanics/"
source_of_truth_title: "Issue 04 — The Heatmap That Lied"
related: ["long-context-concepts"]
draft: false
---

| Benchmark | What it tests | Source |
|---|---|---|
| **NIAH** | Retrieve a single planted sentence from a long haystack. Industry standard 2023–2024; saturated early 2024. | [ch.3](/comicbook/04-long-context-bench/03-niah-mechanics/) |
| **U-curve** | Position-dependent accuracy — highest at start and end, lowest in the middle. (Liu et al., 2023) | [ch.7](/comicbook/04-long-context-bench/07-lost-in-the-middle/) |
| **MRCR** | Disambiguate the *k*-th of *N* similar requests in a long context. 9-cell grid: 2/4/8 needles × 128K/256K/1M. Standard retrieval benchmark from 2025. | [ch.17 — dedicated](/comicbook/04-long-context-bench/17-mrcr/) · [ch.9 — LSQ framework](/comicbook/04-long-context-bench/09-latent-structure/) |
| **GraphWalks** | BFS on a hex-hash directed graph encoded in the prompt — requires multi-hop reasoning. (OpenAI, April 2025) | [ch.8](/comicbook/04-long-context-bench/08-needle-to-graph/) |
| **OOLONG** | Per-chunk classification then aggregation across all chunks — the *width* axis orthogonal to GraphWalks's *depth* axis. All frontier models &lt;50% at 128K. (Bertsch et al., CMU, Nov 2025, arXiv:2511.02817) | [ch.18 — dedicated](/comicbook/04-long-context-bench/18-oolong/) · [ch.9 — LSQ framework](/comicbook/04-long-context-bench/09-latent-structure/) |
| **NoLiMa** | Needle-in-a-haystack with *no* literal token overlap between query and needle — tests semantic retrieval only. (Modarressi et al., Adobe Research, ICML 2025) | [ch.12](/comicbook/04-long-context-bench/12-context-rot/) |
