---
title: "DeepSeek"
slug: "deepseek"
description: "DeepSeek model family — open-weight LLMs from DeepSeek AI notable for the MLA → DSA → CSA/HCA attention arc, mixture-of-experts scaling, and million-token context."
category: "models"
also_known_as: ["DeepSeek-V2", "DeepSeek-V3", "DeepSeek-V3.1", "DeepSeek-V3.2-Exp", "DeepSeek-V4", "DeepSeek-V4-Pro", "DeepSeek-V4-Flash", "DeepSeek-R1", "DeepSeek-Coder", "MLA", "DSA", "CSA", "HCA", "Native Sparse Attention", "NSA", "lightning indexer", "Multi-head Latent Attention", "DeepSeek Sparse Attention", "Compressed Sparse Attention", "Heavily Compressed Attention", "Liang Wenfeng", "High-Flyer"]
source_of_truth: "/comicbook/07-deepseek-attn/"
source_of_truth_title: "Issue 7: The Sparse Lab"
related: ["attention", "qwen3", "kv-cache", "kv-pruning", "rope"]
draft: false
---

Open-weight model family from **DeepSeek AI**, the research arm of the Hangzhou-based hedge fund **High-Flyer** (幻方量化), founded by **Liang Wenfeng** in 2016.

DeepSeek is most notable for a series of stacked architectural innovations in **attention** and **long-context scaling**. The full arc is the subject of [Issue 7: The Sparse Lab](/comicbook/07-deepseek-attn/).

## Release Timeline

| Date | Release | Key innovation |
|---|---|---|
| May 2024 | **DeepSeek-V2** | **MLA** (Multi-head Latent Attention): low-rank KV via a single latent $c_t$, absorption-identity decompression, ~30× cache shrink. See [Issue 5 ch.19](/comicbook/05-microgpt/19-mla/) for microGPT-level mechanics and [Issue 7 ch.2](/comicbook/07-deepseek-attn/02-mla-rewind/) for the historical arc. |
| Dec 2024 | **DeepSeek-V3** | 671B-parameter MoE on top of MLA. Trained for ~$6M. |
| Jan 2025 | **DeepSeek-R1** | Pure-RL reasoning fine-tune. Same MLA backbone. |
| Feb 2025 | **NSA paper** | "Native Sparse Attention" preprint. Three-branch architecture (compression, selection, sliding window). [Issue 7 ch.5](/comicbook/07-deepseek-attn/05-nsa-paper/). |
| Aug 2025 | DeepSeek-V3.1 | 128K context. Same MLA. |
| Sept 2025 | **DeepSeek-V3.2-Exp** | **DSA** (DeepSeek Sparse Attention): the "lightning indexer" scores past tokens and selects top-$k$. 50% API price cut. [Issue 7 ch.6](/comicbook/07-deepseek-attn/06-lightning-indexer/). |
| 2026 | **DeepSeek-V4 (Pro, Flash)** | **CSA + HCA** hybrid attention. V4-Pro = 1.6T params (49B active), 61 layers. V4-Flash = 284B params (13B active), 43 layers. Native 1M-token context. [Issue 7 ch.7](/comicbook/07-deepseek-attn/07-csa/) — [ch.9](/comicbook/07-deepseek-attn/09-hybrid-pattern/). Also introduces **mHC** (Manifold-Constrained Hyper-Connections) for residual stream stability and the **Muon** optimizer for most parameters. |

## Vocabulary Index

| Term | Defined in | What it is |
|---|---|---|
| **MLA** | [Issue 5 ch.19](/comicbook/05-microgpt/19-mla/) | Low-rank latent KV cache + absorption identity |
| **NSA** | [Issue 7 ch.5](/comicbook/07-deepseek-attn/05-nsa-paper/) | Three-branch sparse attention research preprint |
| **DSA** | [Issue 7 ch.6](/comicbook/07-deepseek-attn/06-lightning-indexer/) | Lightning indexer + top-k selection (V3.2-Exp) |
| **CSA** | [Issue 7 ch.7](/comicbook/07-deepseek-attn/07-csa/) | Compressed Sparse Attention: token compression + DSA (V4) |
| **HCA** | [Issue 7 ch.8](/comicbook/07-deepseek-attn/08-hca/) | Heavily Compressed Attention: heavy compression, no sparse selection (V4) |
| **Lightning indexer** | [Issue 7 ch.6](/comicbook/07-deepseek-attn/06-lightning-indexer/) | Bilinear scoring head that ranks past tokens for top-k |
| **mHC** | DeepSeek-V4 paper §2.2 | Manifold-Constrained Hyper-Connections (residual upgrade) |
| **Muon optimizer** | DeepSeek-V4 paper §2.4 | Newton-Schulz iteration-based update; replaces AdamW |

## Strategic Note

DeepSeek publishes 30-50-page technical reports under permissive (MIT) licenses with weights, inference code, and kernel patches merged upstream into vLLM/SGLang on release day. This publication discipline — more than any single architectural innovation — is arguably the lab's biggest contribution to the field.
