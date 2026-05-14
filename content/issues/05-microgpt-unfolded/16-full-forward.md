---
title: "The Full Forward Pass"
description: "The capstone. The microGPT listing reprinted one final time, with every variable and every loop hyperlinked to the chapter that explains it. This is the index to the rest of the project."
topics: [transformer, inference]
tags: [microgpt, reference]
theme: teal
math: true
draft: false
date: 2026-05-14T02:48:59-04:00
issue: 5
weight: 160
techKind: boss
techNode: full-forward
header: default.webp
---

## The Index

Fifteen chapters in, you have unpacked every line of microGPT once. This page is the **assembled view**: the same listing, one more time, with each variable name and each control-flow construct cross-linked to the chapter that earned the right to say what it means.

Treat this page as a **lookup table**. When a later issue says "the K-channel outliers in `attn_wk` are why naive 4-bit quantization breaks", come back here, find `attn_wk` in the listing, click through to chapter 6, and the sentence will already be unambiguous before you finish reading it.

## Constants And Helpers

The hyperparameters of the toy model. Real models scale these into the thousands; the structure is identical.

```python
n_layer    = 2     # see ch.16 (this page) — depth of the network
n_embd     = 16    # see ch.2  — width of the residual stream
block_size = 16    # see ch.3  — max number of positions in wpe
n_head     = 4     # see ch.9  — number of attention heads
head_dim   = 4     # = n_embd / n_head
```

The four helper functions referenced by the forward pass:

```python
def linear(x, w):                                          # → ch.4
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def rmsnorm(x):                                            # → ch.5
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def softmax(logits):                                       # → ch.7
    max_val = max(logits)
    exps = [math.exp(val - max_val) for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def relu(x_val):                                           # → ch.12
    return max(0.0, x_val)
```

## The Forward Pass, Annotated

Each comment points at the chapter that introduced the concept. If you can read this listing top-to-bottom and the comments stop being necessary, you have internalized the issue.

```python
def gpt(token_id, pos_id, keys, values):
    # ─── input embedding ────────────────────────────────────────────
    tok_emb = state_dict['wte'][token_id]                  # → ch.3
    pos_emb = state_dict['wpe'][pos_id]                    # → ch.3
    x = [t + p for t, p in zip(tok_emb, pos_emb)]          # → ch.10 (residual init)
    x = rmsnorm(x)                                         # → ch.5

    for li in range(n_layer):
        # ─── attention sub-block ──────────────────────────────────
        x_residual = x                                     # → ch.10
        x = rmsnorm(x)                                     # → ch.5 (pre-norm)

        q = linear(x, state_dict[f'layer{li}.attn_wq'])    # → ch.6
        k = linear(x, state_dict[f'layer{li}.attn_wk'])    # → ch.6
        v = linear(x, state_dict[f'layer{li}.attn_wv'])    # → ch.6

        keys[li].append(k)                                 # → ch.13 (KV cache)
        values[li].append(v)                               # → ch.13 (KV cache)

        x_attn = []
        for h in range(n_head):                            # → ch.9
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]                        # → ch.9
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]  # → ch.9
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]

            attn_logits = [                                # → ch.8
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)            # → ch.7, ch.8
            head_out = [                                   # → ch.8
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)                        # → ch.9 (concat heads)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])  # → ch.9 (output proj)
        x = [a + b for a, b in zip(x, x_residual)]            # → ch.10 (residual add)

        # ─── MLP sub-block ────────────────────────────────────────
        x_residual = x                                     # → ch.10
        x = rmsnorm(x)                                     # → ch.5 (pre-norm)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])    # → ch.11 (fattening)
        x = [relu(xi) for xi in x]                         # → ch.12 (nonlinearity)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])    # → ch.11 (skinnying)
        x = [a + b for a, b in zip(x, x_residual)]         # → ch.10 (residual add)

    return linear(x, state_dict['lm_head'])                # → ch.15 (project to vocab)
```

That is the whole model, with every nontrivial operation pointing at the chapter that took it apart.

## The Driver Loop

The function above generates **one logit vector per call**. To produce text we need a wrapping loop. There are two distinct phases — see [Prefill vs Decode](../14-prefill-decode/) for the full story.

```python
# initialize the KV cache: one empty list per layer
keys   = [[] for _ in range(n_layer)]                      # → ch.13
values = [[] for _ in range(n_layer)]                      # → ch.13

token_id = BOS
pos_id   = 0
sample   = list(prompt)

# ─── prefill ──────────────────────────────────────────────────────
# walk the prompt forward to populate the cache. discard logits.
for p_token in prompt_tokens:                              # → ch.14
    _ = gpt(token_id, pos_id, keys, values)
    token_id = p_token
    pos_id  += 1

# ─── decode ───────────────────────────────────────────────────────
# generate one token at a time, sampling from the logit distribution
for _ in range(block_size - pos_id):                       # → ch.14
    logits = gpt(token_id, pos_id, keys, values)
    probs  = softmax([l / temperature for l in logits])    # → ch.15 (temperature)
    token_id = random.choices(range(vocab_size), weights=probs)[0]  # → ch.15
    if token_id == BOS:
        break
    sample.append(uchars[token_id])
    pos_id += 1
```

## Lines Of Code That Quietly Run The Industry

Some lines look unremarkable in microGPT but bloom into entire research subfields in production. A pointer to where each thread leads:

| Line in microGPT | What it becomes in the real world |
|---|---|
| `tok_emb = state_dict['wte'][token_id]` | Tied vs. untied embeddings; vocab size 32K → 256K trade-offs |
| `pos_emb = state_dict['wpe'][pos_id]` | Sinusoidal → learned → RoPE → ALiBi → YaRN; long-context extension |
| `rmsnorm(x)` | RMSNorm vs LayerNorm; pre-norm vs post-norm; QK-norm |
| `linear(x, state_dict[f'layer{li}.attn_wq'])` | Quantized linears (issue 03); LoRA adapters; tensor parallelism shards |
| `attn_logits / head_dim**0.5` | Flash-attention; sliding-window attention; sparse attention masks |
| `softmax(attn_logits)` | Numerically stable online softmax; logit-soft-capping (Gemma) |
| `keys[li].append(k)` | Paged KV cache; KV quantization (issue 03 ch.15); KV cache compression |
| `for h in range(n_head)` | MQA (multi-query) and GQA (grouped-query) attention |
| `linear(x_attn, attn_wo)` | The "output projection bottleneck" — where many quantization schemes break |
| `relu(xi)` | GELU → SiLU → SwiGLU; gated MLPs that double the parameter count |
| `linear(x, mlp_fc2)` | Mixture-of-Experts: replace one big `fc2` with K small ones + router |
| `linear(x, state_dict['lm_head'])` | Speculative decoding; medusa heads; tied vocab matrices |
| `random.choices(...)` | Temperature, top-k, top-p, min-p, mirostat, repetition penalty, beam search |

Each row of that table is, somewhere in this project or the projects to come, an entire chapter of its own. The point of microGPT is that every one of them is a **localized modification of a single line you can already read.**

## What To Remember

1. **The forward pass is six things in a loop.** Embed → norm → attention → MLP → ... → project to vocab. The "loop" runs `n_layer` times. Everything else is detail.
2. **The KV cache is `list.append`.** Mechanically. Everything sophisticated about KV cache management is an optimization of where those lists live in memory and how they get reused.
3. **Two phases of inference.** Prefill walks the prompt forward to fill the cache; decode generates one token per call. Most serving optimizations target one or the other.
4. **Every modern trick is a line replacement.** RoPE replaces the `wpe` lookup. Flash-attention replaces the `attn_logits` + `softmax` + `head_out` block. SwiGLU replaces `relu(...)`. MoE replaces `mlp_fc2`. Find the line first; *then* read the paper.
5. **Read this listing before reading any other paper about LLM internals.** The half-second of "wait, where in the forward pass is this?" is what separates skimming from understanding.

---

*This is the baseline. From here, the rest of the issue (and every other issue in the project) assumes you can find your way around this listing without needing the comments. When in doubt, come back to this page.*

**Continue to** → [The Three Axes of KV Compression](../17-kv-axes/) — once you can read the baseline, the obvious next question is which lines the 2023–2025 frontier-architecture papers changed, and why. The answer organizes itself along three axes of the KV cache tensor.
