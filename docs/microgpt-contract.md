# microGPT Terminology Contract

**When to read this:** You are writing or editing any article that touches transformer internals — attention, embeddings, MLP blocks, KV cache, inference phases, or any concept that has a name in the microGPT reference listing.

---

## Why This Exists

Issue 5 (`content/issues/05-microgpt-unfolded/`) introduces a 60-line plain-Python transformer — **microGPT** — that is the shared reference implementation for the entire project. A reader who has read issue 5 once can pick up any article in any later issue and know that `attn_wk` always means the same thing.

This contract enforces that consistency.

---

## The Canonical Names

| Symbol | What it is | Where in microGPT | Primer to link |
|---|---|---|---|
| `wte` | token embedding table | `state_dict['wte']` | [ch.3 Tokens & Positions](../05-microgpt-unfolded/03-embeddings/) |
| `wpe` | positional embedding table | `state_dict['wpe']` | [ch.3 Tokens & Positions](../05-microgpt-unfolded/03-embeddings/) |
| `attn_wq` / `attn_wk` / `attn_wv` | per-layer Q/K/V projections | `state_dict[f'layer{li}.attn_w?']` | [ch.6 Q, K, V](../05-microgpt-unfolded/06-qkv-projections/) |
| `attn_wo` | attention output projection | `state_dict[f'layer{li}.attn_wo']` | [ch.9 Multi-Head Attention](../05-microgpt-unfolded/09-multi-head/) |
| `mlp_fc1` / `mlp_fc2` | MLP fatten / skinny matrices | `state_dict[f'layer{li}.mlp_fc?']` | [ch.11 The MLP Block](../05-microgpt-unfolded/11-mlp-block/) |
| `lm_head` | vocab projection | `state_dict['lm_head']` | [ch.15 LM Head & Sampling](../05-microgpt-unfolded/15-sampling/) |
| `q`, `k`, `v` | current-token query/key/value | local in `gpt()` | [ch.6](../05-microgpt-unfolded/06-qkv-projections/) |
| `q_h`, `k_h`, `v_h` | per-head slices | local in head loop | [ch.9](../05-microgpt-unfolded/09-multi-head/) |
| `attn_logits` | raw `q · k / √d` scores | local | [ch.8 Scaled Dot-Product Attention](../05-microgpt-unfolded/08-attention/) |
| `attn_weights` | softmaxed scores | local | [ch.8](../05-microgpt-unfolded/08-attention/) |
| `head_out` | per-head output vector | local | [ch.8](../05-microgpt-unfolded/08-attention/) |
| `x_attn` | concatenated head outputs | local | [ch.9](../05-microgpt-unfolded/09-multi-head/) |
| `x_residual` | residual-stream copy held aside | local | [ch.10 The Residual Stream](../05-microgpt-unfolded/10-residual-stream/) |
| `keys[li]`, `values[li]` | KV cache for layer `li` | function argument | [ch.13 The KV Cache](../05-microgpt-unfolded/13-kv-cache/) |
| `n_layer`, `n_embd`, `block_size`, `n_head`, `head_dim` | model hyperparameters | `const.py` | [ch.2 The State Dict](../05-microgpt-unfolded/02-state-dict/) |
| **prefill** / **decode** | the two inference phases | driver loop | [ch.14 Prefill vs Decode](../05-microgpt-unfolded/14-prefill-decode/) |
| KV cache shape `(2, L, H, T, D)` | the 5D tensor view of the cache | implicit | [ch.17 The Three Axes](../05-microgpt-unfolded/17-kv-axes/) |
| `n_kv_head`, `group_size` | GQA/MQA sharing factor | (extension) | [ch.18 Grouped-Query Attention](../05-microgpt-unfolded/18-gqa/) |
| `kv_down`, `d_c`, latent `c` | MLA low-rank cache form | (extension) | [ch.19 Multi-head Latent Attention](../05-microgpt-unfolded/19-mla/) |
| sliding window `W` | per-layer attention window | (extension) | [ch.20 Sliding-Window Attention](../05-microgpt-unfolded/20-sliding-window/) |
| `state[li]` for SSM layers | fixed-size recurrent state replacing KV | (extension) | [ch.21 State-Space Hybrids](../05-microgpt-unfolded/21-ssm-hybrids/) |

For shape reference, defer to the table in [ch.1 Sixty Lines, One LLM](../05-microgpt-unfolded/01-cold-open/#the-names-you-should-tattoo) — don't re-derive `wte` is `vocab_size × n_embd` in every article.

---

## What This Means In Practice

When writing a new article that touches any of these concepts:

1. **Use the microGPT name on first mention** — not a synonym. Write "the `attn_wk` projection" rather than "the key matrix `W_K`" or "the key weights". If the source paper uses different notation, introduce both: *"the K projection (`attn_wk` in our reference listing)"*.

2. **Link to the issue 5 primer** the first time a microGPT name appears. From another issue: `[the \`attn_wk\` projection](../../05-microgpt-unfolded/06-qkv-projections/)`. From within issue 5: `../06-qkv-projections/`.

3. **Show the relevant slice of the listing** when discussing one specific line. Copy-paste the 1–4 line excerpt from microGPT verbatim. Do not rewrite it in a different style.

4. **For shape questions, defer to the canonical table** in ch.1. Don't re-derive it locally.

---

## Continuously-expanding terminology 

What we did with microGPT, we would want to do with the other terminology about the other terms and notations that are not limited to a single article / issue but are recurrent themes.

When in your work, you identify and notice such terms, update this docs section and document these terms, and point to the article / page that will be the source-of-truth to be referenced by other articles when mentioned.

---

### Long-context evaluation vocabulary (Issue 04 canonical definitions)

These terms recur across articles in Issue 04 and across the site wherever long-context evaluation is discussed. Use these definitions consistently; link to the source-of-truth page on first mention.

| Term | Definition | Source-of-truth |
|---|---|---|
| **NIAH** (Needle in a Haystack) | Benchmark where a single out-of-distribution sentence is embedded in a long haystack and the model is asked to retrieve it. The 2023–2024 industry standard, saturated by early 2024. | [ch.3 Needle in a Haystack](../content/issues/04-heatmap-that-lied/03-niah-mechanics/) |
| **Context rot** | The observed degradation in model performance as input length grows, characterised by a plateau at short lengths, a steep drop in the 50–75% window range, and a partial recency reflex near the end. Coined by the Chroma report, July 2025. | [ch.12 The Chroma Measurement](../content/issues/04-heatmap-that-lied/12-context-rot/) |
| **U-curve** (lost-in-the-middle) | Position-dependent retrieval accuracy curve: highest at the start, nearly as high at the end, lowest at the middle of a long prompt. Valley is often at or below the closed-book baseline. Liu et al., 2023. | [ch.7 The U-Curve](../content/issues/04-heatmap-that-lied/07-lost-in-the-middle/) |
| **LSQ** (Latent Structure Queries) | Michelangelo framework (Vodrahalli et al., 2024): a well-designed long-context task requires the model to chisel away irrelevant context to expose a latent structure, then query that structure. | [ch.9 Vodrahalli's Chisel](../content/issues/04-heatmap-that-lied/09-latent-structure/) |
| **MRCR** (Multi-Round Co-reference Resolution) | Retrieval-with-counting benchmark: disambiguate the *k*-th of *N* similar requests in a long context. Single-pass solvable. The standard retrieval-side benchmark from 2025 onward. | [ch.9 Vodrahalli's Chisel](../content/issues/04-heatmap-that-lied/09-latent-structure/) |
| **GraphWalks** | Multi-hop reasoning benchmark: directed graph of hex-hash nodes encoded in the prompt; model must perform BFS from a start node and return all nodes at depth *k*. Cannot be solved by a single linear scan. OpenAI, April 2025. | [ch.8 One Pass Isn't Enough](../content/issues/04-heatmap-that-lied/08-needle-to-graph/) |
| **OOLONG** | Aggregation benchmark: per-chunk atomic classification + aggregation across all chunks. Orthogonal to GraphWalks (tests width, not depth). Vodrahalli et al., November 2025. | [ch.9 Vodrahalli's Chisel](../content/issues/04-heatmap-that-lied/09-latent-structure/) |
| **Benchmark saturation** | Phase III of a benchmark lifecycle: every frontier model clusters near the ceiling, variance < noise, ranking becomes unreliable. Antidote: length-conditional reporting, adversarial construction, recent-data design. | [ch.6 Goodhart's Ceiling](../content/issues/04-heatmap-that-lied/06-measurement-saturation/) |
| **Retrieval is not reasoning** | Slogan summarising the 2024 benchmark-wave finding: a model that retrieves single facts from a long context does not necessarily chain dependent lookups or aggregate across many facts. | [ch.5 When Everyone Scored 99](../content/issues/04-heatmap-that-lied/05-saturation/) |
| **The 2026 layered eval stack** | Five-layer long-context evaluation framework: (1) Retrieval, (2) Multi-hop reasoning, (3) Aggregation, (4) Realistic application, (5) Agentic, plus FACTS for grounding. | [ch.16 The 2026 Layered Stack](../content/issues/04-heatmap-that-lied/16-eval-stack-2026/) |
