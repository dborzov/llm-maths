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
