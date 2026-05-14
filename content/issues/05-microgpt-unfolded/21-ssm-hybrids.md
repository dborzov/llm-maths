---
title: "State-Space Hybrids"
description: "Replace some attention layers entirely with state-space blocks (Mamba, RWKV) that maintain a *fixed-size* recurrent state instead of a growing KV cache. Jamba's 1:7 attention-to-Mamba ratio gives 8× compression; Nemotron 3 Nano gets 4.8×; the design space is large and 2026's frontier is busy mapping it."
topics: [transformer, ssm, kv-cache]
tags: [microgpt, mamba, jamba, nemotron, kimi-linear, rwkv]
theme: cream
math: true
draft: false
date: 2026-05-14T02:57:00-04:00
issue: 5
weight: 210
techKind: mainline
techNode: ssm-hybrids
header: default.webp
---

## A Paper Nobody Quite Believed

On the 1st of December 2023, two researchers — Albert Gu (Carnegie Mellon) and Tri Dao (Princeton, also the author of Flash-Attention) — pushed a paper to arXiv with a title that read like a polite provocation: *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."* The claim, baldly stated, was that you could throw the attention mechanism away. No `attn_wq`, no `attn_wk`, no `attn_wv`, no softmax over a growing list of past tokens. Replace the entire attention sub-block with a recurrent state-space layer that maintained a **fixed-size** internal state, and the model would *still* match a transformer of equal parameter count on the language-modeling tasks people cared about.

The deep-learning community had heard versions of this story before. RNN revivalists had been promising "linear-time, transformer-quality" sequence models for five straight years — RetNet, Linear Transformer, S4, RWKV, Hyena — and each one had cracked at the seams as soon as anyone scaled it past 7B parameters. The default reaction to the Mamba abstract was a polite eye-roll. Yann LeCun's quote tweet, lightly paraphrased: *"will believe it when someone trains a serious one."*

Three months later, in March 2024, AI21 Labs published **Jamba** — a 52-billion-parameter mixed-expert model in which seven out of every eight blocks were Mamba layers and only one was a transformer attention block. It scored competitively with Mixtral 8×7B on every long-context benchmark they ran. The KV cache for a 128K-token Jamba inference was *one-eighth* the size of a comparable Mixtral run.

The eye-rolls stopped. By the end of 2025, NVIDIA had shipped **Nemotron 3 Nano** (Mamba-2 hybrid, 4.8× cache compression), Moonshot AI had shipped **Kimi-Linear** (linear-attention hybrid, 4× compression), and every major lab had at least one hybrid in research. This chapter is the **L-axis story** for the family that finally made the prophecy stick.

## The Lever We're Pulling On

From [ch.17 three axes](../17-kv-axes/) we know the KV cache is a 5-D tensor of shape $(2, L, H, T, D)$. The L-axis trick is: **don't compress every layer the same way. Compress some of them to nothing.**

A pure-attention layer in microGPT contributes a cache entry every single token:

```python
keys[li].append(k)        # grows with T
values[li].append(v)      # grows with T
```

A sliding-window layer, as we saw in [ch.20 sliding-window](../20-sliding-window/), keeps only the last $W$ entries. That bounds the growth but doesn't eliminate it: each layer still owns $W$ K/V pairs.

A **state-space layer** is more radical. It maintains exactly one fixed-size object per layer — call it `state[li]` — that gets *overwritten* (not appended-to) on every new token. The number of bytes that layer contributes to the cache is **independent of $T$**. At $T = 100$ it costs the same as at $T = 100{,}000$.

If 7 out of 8 layers in a model behave that way, the per-token cost of growing the cache drops by a factor of 8 — and at 128K context, that's the difference between a single H100 and a small rack.

## What An SSM Block Actually Computes

Strip away the hype and a state-space layer is two lines of linear algebra. Given a per-token input vector $x_t$, you maintain a hidden state $h_t \in \mathbb{R}^{d_{\text{state}}}$ and produce an output $y_t$ via:

$$
h_t = A\, h_{t-1} + B\, x_t \qquad y_t = C\, h_t
$$

That is **literally an RNN**. The $A$, $B$, $C$ matrices are learned. The state $h_t$ has a small fixed dimension — typically $d_{\text{state}} = 16, 32,$ or $64$ — and is *the entire memory* of the layer. The previous 100,000 tokens have been compressed, recursively, into that one short vector.

Compare to attention's contract. In microGPT, the equivalent of "memory" is the per-layer KV cache list:

```python
# attention's "state" at token T:
keys[li]   # length T, each entry length n_embd
values[li] # length T, each entry length n_embd
```

That state has size $2 \cdot T \cdot n_{\text{embd}}$ — it grows linearly with the sequence length. The SSM's state has size $d_{\text{state}}$ — a constant. The trade is **stark**: attention can look up *any* past token exactly; the SSM can only look at whatever happens to be encoded in its current short hidden vector.

### Selectivity: Mamba's contribution

Classic state-space models — S4, S5 — fixed $A$, $B$, $C$ as data-independent matrices. That made them fast (the recurrence could be unrolled as a convolution) but bad at language: a constant filter has no way to *decide* which token to remember versus forget.

Mamba's headline trick was to make $B$ and $C$ functions of the input itself: $B_t = B(x_t)$, $C_t = C(x_t)$. This is called the **selective scan**. The state-update equation becomes input-conditioned:

$$
h_t = A\, h_{t-1} + B(x_t)\, x_t
$$

In plain English: the model can now look at the current token and decide *how much of it to write into the state, and along which axes*. That's the property that distinguishes a Mamba layer from a vanilla RNN — and the property that lets it match attention on the kinds of pattern-matching tasks language models actually need.

```pyplot {id="ssm-state-flow" caption="A state-space block consumes a token sequence and maintains a fixed-size hidden state. Each new token READS the previous state, MUTATES it, and emits an output. The state vector's dimensionality does not grow with T."}
np.random.seed(7)
T = 20
d_state = 8

# simulate state evolution with a simple stable A matrix
A = np.eye(d_state) * 0.85 + np.random.randn(d_state, d_state) * 0.03
B = np.random.randn(d_state, 1) * 0.5
x_seq = np.sin(np.linspace(0, 6, T)) + np.random.randn(T) * 0.2

h = np.zeros(d_state)
states = []
for t in range(T):
    h = A @ h + (B * x_seq[t]).flatten()
    states.append(h.copy())
states = np.array(states)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={'height_ratios': [1, 2]})

ax1.stem(range(T), x_seq, linefmt='#FF8C00', markerfmt='o', basefmt=' ')
ax1.set_xticks(range(T))
ax1.set_xticklabels([f't{i}' for i in range(T)], fontsize=8)
ax1.set_ylabel('input x_t', fontsize=10)
ax1.set_title('Token stream flowing into the SSM layer', fontsize=11)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

im = ax2.imshow(states.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
ax2.set_xticks(range(T))
ax2.set_xticklabels([f't{i}' for i in range(T)], fontsize=8)
ax2.set_yticks(range(d_state))
ax2.set_yticklabels([f'h[{i}]' for i in range(d_state)], fontsize=8)
ax2.set_xlabel('time step (token index)')
ax2.set_ylabel('state component')
ax2.set_title(f'Fixed-size state h_t (d_state={d_state}) overwritten at every step', fontsize=11)
for spine in ax2.spines.values():
    spine.set_edgecolor('#1A1A1A')
    spine.set_linewidth(1.2)

plt.tight_layout()
```

The picture above is the entire pitch in one image. The top panel is the token stream entering the layer. The bottom panel is the hidden state $h_t$ — the width of the heatmap stretches with $T$ only because we're showing the *history* of a vector that, at any one instant, has exactly $d_{\text{state}}$ entries. The state never grows. The cache footprint never grows.

## The microGPT Diff

In microGPT's forward pass, a transformer block is a *normalize → attention → residual → normalize → MLP → residual* sandwich. A **hybrid model** swaps the attention sub-block for an SSM sub-block in some layers, and leaves the MLP untouched in *all* layers. The dispatch table from [ch.17](../17-kv-axes/) becomes concrete:

```python
state = [None] * n_layer        # NEW: one fixed-size state vector per Mamba layer
keys   = [[] for _ in range(n_layer)]
values = [[] for _ in range(n_layer)]

for li in range(n_layer):
    # ── attention OR state-space sub-block ────────────────
    x_residual = x
    x = rmsnorm(x)

    if attention_type[li] == 'full':
        # ── unchanged from microGPT ────────────────────
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        # ... multi-head attention as in the canonical listing ...
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])

    elif attention_type[li] == 'mamba':
        # ── fixed-size recurrent state, NO keys/values list ──
        x, state[li] = mamba_block(x, state[li], state_dict, li)

    x = [a + b for a, b in zip(x, x_residual)]

    # ── MLP sub-block: UNCHANGED in both branches ─────────
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
    x = [relu(xi) for xi in x]
    x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
    x = [a + b for a, b in zip(x, x_residual)]
```

That is the **entire architectural diff**. Two consequences worth staring at:

- For Mamba layers, `keys[li]` and `values[li]` are *never appended to*. They contribute zero bytes to the KV cache.
- The MLP sandwich is untouched. SwiGLU, GeGLU, MoE — whichever flavor of MLP the model uses — runs exactly the same in attention layers and in SSM layers. The hybrid story is *only* about the attention slot.

## Three Hybrids You Have Already Heard Of

**Jamba (AI21, March 2024).** 52B total parameters, 12B active per token (MoE). The architecture interleaves blocks in a strict **1 attention : 7 Mamba** ratio, repeated four times for 32 blocks total. The single attention layer per group is what preserves Jamba's ability to do exact in-context retrieval; the seven Mamba layers do the bulk summarization. **Cache compression: 8×** versus an all-attention model with the same parameter count.

**Kimi-Linear (Moonshot AI, 2025).** Uses *linear attention* — a kernel-trick reformulation of attention where each new token updates a fixed-size $D \times D$ accumulator matrix rather than appending to a list. It's an SSM in disguise (the accumulator is the "state"). Compression: **4×**.

**Nemotron 3 Nano (NVIDIA, 2025).** Uses Mamba-2, a refinement of Mamba where the $A$ matrix is structured to allow a much faster matmul-friendly parallel scan on GPUs. Optimized for **edge inference** — the fixed-state property means a phone has a hope of running it at long context. Compression: **4.8×**.

```pyplot {id="hybrid-cache-comparison" caption="KV cache footprint at T=128K, n_embd≈8192, FP16, for an 8B-class model. Pure attention pays the full 16 GB; hybrids amortize away most of it by replacing attention layers with fixed-state SSM layers."}
labels = ['Llama 3 8B\n(pure attention, GQA 4x)', 'Jamba\n(1:7 attn:Mamba)', 'Kimi-Linear\n(linear-attn hybrid)', 'Nemotron 3 Nano\n(Mamba-2 hybrid)']
# baseline: 16 GB approx for Llama 3 8B at 128K with GQA already applied
baseline_gb = 16.0
sizes_gb = [baseline_gb, baseline_gb / 8, baseline_gb / 4, baseline_gb / 4.8]
colors = ['#FF007F', '#00A8A8', '#FFD700', '#FF8C00']

fig, ax = plt.subplots(figsize=(10, 4.5))
y = list(range(len(labels)))
bars = ax.barh(y, sizes_gb, color=colors, edgecolor='#1A1A1A', linewidth=2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
for i, s in enumerate(sizes_gb):
    ax.text(s + 0.2, i, f'{s:.2f} GB', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('KV cache size (GB) @ T=128K')
ax.set_title('Hybrid models pay a fraction of pure-attention cache cost', fontsize=12)
ax.set_xlim(0, baseline_gb * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axvline(baseline_gb, color='#1A1A1A', linestyle='--', linewidth=1, alpha=0.4)
ax.text(baseline_gb, -0.6, 'pure-attention baseline', ha='right', fontsize=8, style='italic', color='#555')
```

A napkin-math check on the leftmost bar: Llama 3 8B has 32 layers, GQA group size 4, head_dim 128, $n_{\text{kv\_head}} = 8$. The cache shape at $T = 128{,}000$ in FP16 is

$$
2 \cdot 32 \cdot 8 \cdot 128{,}000 \cdot 128 \cdot 2 \text{ bytes} \approx 16 \text{ GB}.
$$

Jamba's 1:7 ratio replaces 7 of every 8 layers with zero-cache Mamba blocks, so it pays roughly $16 / 8 = 2$ GB at the same context. That's the difference between a model that needs a server and a model that runs on a workstation.

## The Retrieval Catch

Hybrid architectures exist for one specific reason: **pure SSMs are bad at exact recall**. The classic stress test is the *needle-in-a-haystack* benchmark — bury a random fact like "the secret code is 73-Q-violet" at token 50,000 of a 100,000-token document, then ask the model to retrieve it. Pure-Mamba models score near zero. Pure-transformer models score near one.

The reason is mechanical. To retrieve "73-Q-violet", a model needs to do something like: *"My current query is asking for a secret code. Scan the past tokens for a key that says 'secret code is'. Read the value attached to it."* That is **content-addressable lookup over the entire history** — the exact thing softmax-over-keys was designed to do. An SSM's state has, by construction, lossy-compressed every past token into a $d_{\text{state}} = 64$ vector. The fingerprint of "73-Q-violet" was overwritten three thousand tokens ago.

Hybrids exploit this asymmetry. The few remaining attention layers — even one per eight, as in Jamba — preserve enough exact-lookup capability that needle-in-a-haystack scores recover. The SSM layers handle the *flow* — summarization, syntactic structure, long-range coherence — that attention is wasteful on anyway. It is a clean division of labor:

> Attention is for **lookup**. SSMs are for **flow**. Hybrids let each layer do what it's best at.

## The Wider Family

Mamba was not the first attempt at a fixed-state alternative, just the first that worked at scale. The family tree is worth a glance:

- **Linear Attention** (Katharopoulos et al., 2020). Reformulate the softmax-attention dot product using a kernel feature map; the result is a fixed-size accumulator matrix that updates token-by-token. Mathematically equivalent to an SSM with a particular structured $A$.
- **RWKV** (Peng et al., 2023). A hand-designed hybrid of RNN and attention with linear-time inference. Trained at 14B and remains in active production (RWKV-7 shipped in late 2025).
- **RetNet** (Sun et al., 2023). Microsoft's "Retentive Networks." Same structural insight as Mamba — fixed-size state — but with a closed-form parallel formulation that, at the time, didn't *quite* match transformer quality on language. The lessons fed forward into Mamba-2.
- **Mamba / Mamba-2** (Gu & Dao, 2023, 2024). The selective-scan trick. The architecture that finally cleared the bar.

The recurring shape of the design problem is: **how much "state" can you carry forward, and how do you decide what to put in it?** Attention's answer is "all of it, never compress." SSMs' answer is "a fixed amount, learn the policy." Hybrids sit somewhere in between, exploiting the fact that *most* layers don't need full recall, and the few that do can be allocated as a budgeted resource.

## What To Remember

1. **A state-space block has a fixed-size recurrent state.** $h_t = A h_{t-1} + B x_t$, $y_t = C h_t$. The state lives in $\mathbb{R}^{d_{\text{state}}}$ for $d_{\text{state}} \sim 64$. It never grows with $T$.
2. **The microGPT diff is per-layer dispatch.** Replace the attention sub-block with `mamba_block(x, state[li])` for some layers. The MLP sub-block is untouched. `keys[li]` and `values[li]` for SSM layers stay empty.
3. **Mamba's selectivity is the key.** Input-dependent $B$ and $C$ matrices let the model *choose* what to keep. Without selectivity, you have an S4-style data-independent filter and quality suffers.
4. **Hybrids exist because SSMs lose at exact recall.** Needle-in-a-haystack collapses without at least a few attention layers. The hybrid's attention layers do retrieval; the SSM layers do flow.
5. **Production numbers in 2026:** Jamba 8×, Nemotron 3 Nano 4.8×, Kimi-Linear 4×. All L-axis interventions; all stackable with H-axis (GQA) and D-axis (MLA) tricks on top.
6. **The 2027 question is genuinely open.** Pure transformer, pure SSM, or interleaved hybrid — every major lab has a horse in this race and the leaderboard hasn't settled.

---

**Continue to** → [the issue cover](../) — and from there, [the next issue](../../). Flip back through the tech tree one final time: the foundations at the bottom row now ladder all the way up to a working understanding of every contemporary frontier architecture, and the rest of this project is going to drill into each rung in detail.

