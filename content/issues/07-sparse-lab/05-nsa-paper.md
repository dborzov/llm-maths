---
title: "The NSA Blueprint"
description: "February 16, 2025. DeepSeek publishes 'Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention'. Three branches — compression, selection, sliding window — that compose. The paper that becomes the blueprint for DSA, CSA, and HCA all at once."
topics: [attention, sparse-attention, deepseek]
tags: [nsa, native-sparse-attention, deepseek, hardware-aligned, three-branch]
theme: cream
math: true
draft: false
date: 2026-05-16T09:40:00-04:00
issue: 7
weight: 50
techKind: mainline
techNode: nsa-paper
header: default.png
---

## A Preprint, Not A Product

**February 16, 2025.** Eight DeepSeek authors post a paper to arXiv: *"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"* (arXiv:2502.11089). No model release. No weights. Just the architecture.

The paper is structured as a critique of every sparse-{{< wiki "attention" >}}attention{{< /wiki >}} attempt that came before — the criteria from [the last chapter](../04-sparse-detour/) appear almost verbatim. The contribution is a single new architecture, **NSA**, that the authors claim passes all the criteria. Three branches, one composite output, fully trainable from scratch.

The paper is doing two things at once. It is a research contribution — a real algorithm with real ablations. But it is also a *blueprint*. Read carefully, and you can see the lab telegraphing what they are about to ship. Every design decision maps to a downstream product: the selection branch will become [the lightning indexer](../06-lightning-indexer/). The compression branch will become [CSA](../07-csa/). The sliding window survives intact into everything. Seven months before V3.2-Exp ships, the architecture is already visible in this arXiv preprint.

Let us build it up from first principles.

{{< crosshead >}}Why Three Branches?{{< /crosshead >}}

The dead end from the [previous chapter](../04-sparse-detour/) was sharp: every single sparse attention method failed either on quality (fixed patterns) or on kernel compatibility (learned patterns). The tension seemed fundamental — selecting which tokens to attend to requires a discrete decision, and discrete decisions break differentiability.

NSA's first move is to reframe the problem. Instead of asking "how do we make token selection differentiable?", ask a different question: **what kinds of context does an attention head actually need?**

When a transformer processes a long document — say, a 32K-token legal contract — and a query token at position 28,000 fires an attention head, what does that head actually need to attend to?

Three things, roughly:
1. **Something global.** The big-picture meaning of the whole document. Topic, structure, the argument being made. You can't get this from a local window.
2. **Something specific.** A particular clause that was defined at position 4,000 and is being referenced at position 28,000. A named entity. A number. This requires fine-grained retrieval — finding the exact right token among 28,000 candidates.
3. **Something local.** The immediately preceding tokens that provide grammatical and semantic context. Without these, the model can't form coherent continuations.

{{% pullquote type="standard" %}}
Global context, fine-grained retrieval, and local coherence are three distinct information needs. No single sparse pattern can satisfy all three simultaneously. NSA stops trying to satisfy all three with one mechanism and builds three.
{{% /pullquote %}}

These three information needs map directly to the three branches.

{{< crosshead >}}Branch 1: Compression (Global Context){{< /crosshead >}}

The compression branch aggregates the entire past context into a compact representation. Every $m$ consecutive tokens are pooled into a single compressed KV entry — a learned, content-aware blend of those $m$ tokens. The model then runs dense {{< wiki "attention" >}}attention{{< /wiki >}} over the compressed sequence.

The compressor is not a simple mean pool. It uses a learned softmax-weighted blend:

```python
# NSA compression branch
# h: (T, d_model) — all hidden states
# W_kv: (d_model, d_head) — project to KV space
# W_z: (d_model, d_head) — compression weight predictor

def nsa_compress(h, W_kv, W_z, m=32):
    """
    h: (T, d_model)
    Returns C_comp: (T//m, d_head) — compressed KV entries
    """
    T = h.shape[0]
    C = h @ W_kv          # (T, d_head): project to KV space
    Z = h @ W_z           # (T, d_head): compression weights (unnormalized)

    C_comp = []
    for i in range(T // m):
        start, end = i * m, (i + 1) * m
        S_i = np.exp(Z[start:end]) / np.exp(Z[start:end]).sum(axis=0)  # softmax over block
        # S_i: (m, d_head) — per-token weight within the block
        # C[start:end]: (m, d_head) — KV projections for the block
        C_i = (S_i * C[start:end]).sum(axis=0)  # (d_head,) — weighted sum
        C_comp.append(C_i)

    return np.stack(C_comp)  # (T//m, d_head)

# At T=32768 and m=32: compressed sequence length = 32768/32 = 1024 tokens
# Dense attention over 1024 entries: O(1024^2) instead of O(32768^2)
T, m = 32768, 32
T_comp = T // m
flops_dense   = T * T          # 1,073,741,824 — full dense
flops_compress = T_comp * T_comp  # 1,048,576 — dense over compressed
print(f"Dense attention FLOPs:       {flops_dense:>15,}")
print(f"Compression branch FLOPs:    {flops_compress:>15,}")
print(f"Compression ratio:           {flops_dense/flops_compress:.0f}×")
# → Compression ratio: 1024× reduction in attention FLOPs
# The compressor cost (T*d*m per block) is O(T*d), much cheaper
```

The compressed sequence has length $T/m$. Dense attention over $T/m$ entries is $O((T/m)^2)$, which at $m=32$ is $1024\times$ cheaper than full dense attention. The compressor itself costs $O(T \cdot d)$ — linear in sequence length. Net result: global context at $O(T/m)^2 + O(T \cdot d)$ rather than $O(T^2)$.

**What this branch provides:** the model can always attend to any part of the sequence, just at block-level granularity. No part of the document is invisible. This is the global antenna.

**What this branch does NOT provide:** token-level precision. If the exact right token is at position 7,432, the compression branch sees it blended with tokens 7,424–7,455. Good enough for global context; not good enough for precise retrieval.

{{< crosshead >}}Branch 2: Selection (Fine-Grained Retrieval){{< /crosshead >}}

The selection branch is where the innovation lives. It is the mechanism that solves the [six-year failure](../04-sparse-detour/) of learned sparse patterns.

The setup: for every query token at position $t$, we want to identify which specific past tokens are most relevant. Not which block — which *tokens*. We want token-level precision. But we want it cheaply, without computing full dot-product attention against all $T$ past tokens.

NSA's approach: **score past blocks, not past tokens.** Divide past tokens into blocks of size $m$. For each block $B_i$, compute a single summary score indicating how relevant block $B_i$ is for query $q_t$. Keep the top-$k$ scoring blocks. Then attend to all tokens in those $k$ blocks with full attention.

```python
def nsa_select_and_attend(q, K_blocks, V_blocks, W_score, top_k=16, m=32):
    """
    q: (d_head,) — query vector
    K_blocks: (n_blocks, m, d_head) — all past tokens organized into blocks
    V_blocks: (n_blocks, m, d_head)
    W_score: (d_head, d_score) — projection to scoring space
    top_k: number of blocks to keep
    m: tokens per block
    """
    n_blocks = K_blocks.shape[0]

    # Step 1: Compute block summary scores via bilinear product
    # Each block's "summary key" = mean of its keys projected to scoring space
    block_keys = K_blocks.mean(axis=1)        # (n_blocks, d_head) — block summaries
    q_score = q @ W_score                     # (d_score,) — query in scoring space
    block_scores = block_keys @ q_score        # (n_blocks,) — one score per block

    # Step 2: Top-k block selection (discrete, non-differentiable forward pass)
    top_block_idx = np.argsort(block_scores)[-top_k:]   # (top_k,) indices

    # Step 3: Attend to all tokens in selected blocks
    K_selected = K_blocks[top_block_idx].reshape(top_k * m, -1)   # (top_k*m, d_head)
    V_selected = V_blocks[top_block_idx].reshape(top_k * m, -1)   # (top_k*m, d_head)

    d_head = q.shape[-1]
    attn_scores = q @ K_selected.T / np.sqrt(d_head)   # (top_k*m,)
    weights = np.exp(attn_scores - attn_scores.max())
    weights /= weights.sum()
    return weights @ V_selected   # (d_head,)

# Napkin math at T=32768, m=32, top_k=16:
T, m, top_k = 32768, 32, 16
n_blocks = T // m                         # = 1024 blocks
scoring_flops = n_blocks * 32             # block scoring: n_blocks × d_score=32
attend_flops  = top_k * m * 128           # attend: top_k*m tokens × d_head=128
total_selection = scoring_flops + attend_flops
full_attn = T * 128
print(f"Scoring phase:    {scoring_flops:>8,} FLOPs (d_score=32, n_blocks={n_blocks})")
print(f"Attend phase:     {attend_flops:>8,} FLOPs (top_k={top_k} blocks × m={m} × d=128)")
print(f"Total selection:  {total_selection:>8,} FLOPs")
print(f"Full attention:   {full_attn:>8,} FLOPs")
print(f"Selection ratio:  {total_selection/full_attn:.3f}× dense")
# → 0.049× dense — 20× cheaper than full attention per query
```

The scoring is cheap because the block summary dimension $d_{\text{score}}$ can be small (the paper uses $d_{\text{score}} = 32$, vs. $d_{\text{head}} = 128$). The scoring step is $O(T/m \times d_{\text{score}})$ — linear in sequence length with a small constant. The attend step is $O(k \times m \times d_{\text{head}})$ — constant in sequence length (it depends on how many blocks you keep, not on $T$).

{{% callout type="counterintuitive" %}}
**The block granularity is a feature, not a bug.** Earlier methods (Longformer, BigBird) operated at token granularity, which is precise but forces the pattern to be pre-specified. NSA operates at block granularity for scoring, which lets the model learn "which region of the sequence matters" without needing to identify the exact token. The exact-token selection happens *after* the block is retrieved, via full attention within the block.
{{% /callout %}}

{{< crosshead >}}The Natively Trainable Innovation{{< /crosshead >}}

Here is the differentiability problem in its sharpest form. The top-$k$ selection in Step 2 is a **hard discrete decision**: you either include block $B_i$ in the selected set or you don't. The gradient of "was this block selected?" with respect to the block score $s_i$ is zero almost everywhere (the argmax has zero gradient outside of ties).

NSA solves this with two mechanisms working in tandem.

**Mechanism 1: Block-level Straight-Through Estimator (STE).**

During the forward pass, the block selection is hard top-$k$ — discrete. During the backward pass, gradients are allowed to flow through the scoring function *as if* the selection were a continuous soft-max over all blocks. The "straight-through" approximation pretends the discrete decision didn't happen:

```python
# Conceptual STE implementation for block selection
class HardTopKWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, top_k):
        # Hard selection: discrete, correct, non-differentiable
        top_idx = torch.topk(scores, top_k).indices
        mask = torch.zeros_like(scores)
        mask[top_idx] = 1.0
        ctx.save_for_backward(scores, mask)
        return mask  # binary: 1 if selected, 0 if not

    @staticmethod
    def backward(ctx, grad_output):
        scores, mask = ctx.saved_tensors
        # Straight-through: pass gradient back as if selection were identity
        # i.e., pretend every block contributes proportionally to its score
        return grad_output, None  # gradient flows to scores unchanged

# Effect: the scorer weights W_score receive gradient signals from all
# blocks that *should* have been selected (because the output quality
# suffered from selecting the wrong blocks). The model learns to score
# blocks more accurately.
```

The STE is an approximation — the "true" gradient is zero for blocks that weren't selected. But in practice, it works: the scorer receives a learning signal, and it learns to rank blocks by relevance. The [top-k routing primer](../13-topk-and-routing/) has the full theoretical treatment of why STE converges despite the approximation.

**Mechanism 2: Auxiliary Distillation Loss.**

The second mechanism is more aggressive. Run full dense attention in parallel with NSA during training. Use the dense attention output as a *teacher*, and add a distillation loss that pushes the NSA output to match it:

```python
def nsa_training_step(q, k, v, model, lm_targets, distill_weight=0.01):
    # Student: sparse NSA attention
    nsa_output = nsa_forward(q, k, v, model.nsa_params)

    # Teacher: full dense attention (no gradient — just a supervision signal)
    with torch.no_grad():
        dense_output = full_attention(q, k, v)

    # Main LM objective: next-token prediction quality
    logits = model.head(nsa_output)
    lm_loss = F.cross_entropy(logits, lm_targets)

    # Distillation objective: NSA output should look like dense output
    distill_loss = F.mse_loss(nsa_output, dense_output)

    # Combined: the scorer learns to select blocks that
    # produce outputs matching what dense attention would produce
    total_loss = lm_loss + distill_weight * distill_loss
    return total_loss
```

This distillation signal is direct. If the selection branch picks the wrong blocks — blocks that are not actually relevant for the current query — the NSA output will diverge from the dense output, and the distillation loss will push the block scorer to score the *right* blocks higher. The scorer gets explicit training signal about which blocks matter, rather than inferring it only from downstream LM loss.

{{% pullquote type="theorem" %}}
STE + auxiliary distillation is the combination that makes NSA "natively trainable." STE provides the gradient pathway through discrete selection. Auxiliary distillation provides a direct supervision signal for the scorer. Together, they close the quality gap that killed every previous learned-selection method.
{{% /pullquote %}}

{{< crosshead >}}Branch 3: Sliding Window (Local Coherence){{< /crosshead >}}

The third branch is the simplest: always attend to the last $W = 512$ tokens. No learning. No selection. A hard window.

The sliding window exists because compression and selection both have blind spots at very short ranges. The compression branch sees each block as a single averaged entry — fine for global structure, but if you're at token 32,500 and need to attend precisely to token 32,490, the compression branch can't give you that precision (32,490 and 32,500 are in the same block). The selection branch doesn't bother scoring very recent blocks because their scores are always high — any recent token is relevant for local coherence. Including them in the top-$k$ wastes retrieval budget on trivially relevant positions.

The sliding window handles this efficiently. It is FlashAttention-compatible by construction: a fixed causal window is exactly what FlashAttention's tiling algorithm expects. Cost: $O(W)$ per query, constant in sequence length.

```pyplot {id="nsa-three-branches" caption="NSA'S THREE-BRANCH ARCHITECTURE. EACH QUERY READS FROM ALL THREE PATHS. THE COMPRESSION BRANCH IS GLOBAL, THE SELECTION BRANCH IS FINE-GRAINED, THE SLIDING WINDOW IS LOCAL."}
fig, ax = plt.subplots(figsize=(10, 5))
# Draw three boxes for branches with arrows feeding into a join node
boxes = [
    (0.1, 0.65, 'compression\n(every m tokens → 1)', '#FF007F'),
    (0.1, 0.4, 'selection\n(top-k blocks)', '#00A8A8'),
    (0.1, 0.15, 'sliding window\n(last W tokens)', '#FFD700'),
]
for (x, y, label, c) in boxes:
    ax.add_patch(plt.Rectangle((x, y), 0.35, 0.18, facecolor=c, alpha=0.6, edgecolor='#1A1A1A', linewidth=2))
    ax.text(x + 0.175, y + 0.09, label, ha='center', va='center', fontsize=11)
    ax.annotate('', xy=(0.7, 0.5), xytext=(x + 0.35, y + 0.09),
                arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.5))

ax.add_patch(plt.Rectangle((0.7, 0.41), 0.18, 0.18, facecolor='#FF8C00', alpha=0.5, edgecolor='#1A1A1A', linewidth=2))
ax.text(0.79, 0.5, 'sum /\nconcat', ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title("NSA: three parallel attention paths, one summed output")
```

{{< crosshead >}}The Hardware Alignment{{< /crosshead >}}

The phrase "hardware-aligned" in the paper title is doing real work. Let us be concrete about what it means.

A GPU processes data in **warps** of 32 threads that execute in lockstep. Memory accesses are most efficient when all 32 threads in a warp access consecutive memory addresses — this is **coalesced access**. FlashAttention's tiling algorithm depends on this: it processes attention in fixed-size tiles (typically 128×128 or 64×64 elements) where all memory accesses within a tile are coalesced.

The reason Reformer and BigBird failed kernel compatibility was memory access patterns. Reformer's sort-by-bucket step brought non-consecutively-addressed tokens into proximity in a logical sense, but their physical memory layout remained scattered — the sort reindexed logical order without moving physical data. BigBird's random attention pattern gathered keys from arbitrary positions with no locality structure at all.

NSA's design choices are explicitly backwards-engineered from the GPU memory hierarchy:

| Design parameter | NSA value | Rationale |
|---|---|---|
| Block size $m$ | 32 tokens | Matches warp size (32 threads) — one block per warp |
| Sliding window $W$ | 512 tokens | Multiple of SM tile size (128); 4 full FlashAttention tiles |
| Top-$k$ blocks | 16 blocks | 16 × 32 = 512 tokens; same as sliding window; symmetric tile load |
| Scoring dimension $d_\text{score}$ | 32 | One warp-worth of dot products per block score computation |

The selection branch selects 16 blocks of 32 tokens = 512 contiguous tokens per selection. Those 512 tokens are the contents of 16 consecutive block regions. They are contiguous in memory *within each block* (tokens in a block are allocated sequentially). FlashAttention can tile each selected block natively as a 32-token tile.

{{% callout type="tip" %}}
Compare this with Reformer: after sorting by hash bucket, tokens in the same bucket are logically adjacent but physically scattered in the original key matrix. You would need to physically gather them into a new buffer before FlashAttention could tile them — an extra O(T) memory copy per forward pass, per layer, per head. At 128K tokens, 80 layers, 128 heads: this is a significant memory bandwidth overhead on every forward pass. NSA avoids this entirely because blocks are already contiguous.
{{% /callout %}}

{{< crosshead >}}Output Combination{{< /crosshead >}}

The three branches run simultaneously for every query. Each branch produces an output vector of dimension $d_{\text{head}}$:
- $\mathbf{o}_\text{comp} \in \mathbb{R}^{d_\text{head}}$ — compression branch output
- $\mathbf{o}_\text{sel} \in \mathbb{R}^{d_\text{head}}$ — selection branch output  
- $\mathbf{o}_\text{win} \in \mathbb{R}^{d_\text{head}}$ — sliding window output

These are concatenated along the head dimension and projected to $d_\text{model}$:

$$\mathbf{o} = W_o \cdot [\mathbf{o}_\text{comp}; \mathbf{o}_\text{sel}; \mathbf{o}_\text{win}]$$

The projection $W_o \in \mathbb{R}^{d_\text{model} \times 3d_\text{head}}$ is trained normally — backpropagation through $W_o$ distributes gradients to all three branches.

This is critical: the three branches are **not** alternatives. They are not routed. Every query reads from all three simultaneously. The model learns, via $W_o$, how to weight the contributions of global context, fine-grained retrieval, and local coherence for each position.

{{< crosshead >}}Ablation Results{{< /crosshead >}}

The paper includes training experiments on models up to 7B parameters, trained from scratch with NSA replacing standard attention in all layers. The key findings:

```pyplot {id="nsa-branch-ablation" caption="NSA BRANCH CONTRIBUTION ABLATION: REMOVING SELECTION COSTS THE MOST PERPLEXITY. EACH BRANCH PROVIDES NON-REDUNDANT INFORMATION."}
# Stylized ablation results from the NSA paper
branch_configs = [
    "Full NSA\n(all 3 branches)",
    "No selection\n(comp + window)",
    "No compression\n(select + window)",
    "No window\n(comp + select)",
    "Window only\n(baseline)",
    "Dense\nattention",
]

# Perplexity (lower is better). Dense is baseline=10.0 (stylized)
perplexity = [10.4, 13.4, 11.9, 12.4, 16.8, 10.0]
colors = ['#FF007F', '#FF8C00', '#FFD700', '#00A8A8', '#1A1A1A', '#1A1A1A']
alphas = [1.0, 0.8, 0.8, 0.8, 0.6, 0.4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bars = ax1.bar(range(len(branch_configs)), perplexity,
               color=colors, alpha=0.85, edgecolor='#1A1A1A', linewidth=1.5)
ax1.axhline(y=10.0, color='#1A1A1A', linestyle='--', alpha=0.5, label='Dense baseline')
ax1.set_xticks(range(len(branch_configs)))
ax1.set_xticklabels(branch_configs, fontsize=8.5)
ax1.set_ylabel("Perplexity (lower = better)")
ax1.set_title("NSA branch ablation: perplexity on long-context tasks")
ax1.set_ylim(9, 18)
for bar, val in zip(bars, perplexity):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.15,
             f"{val:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax1.spines[['top', 'right']].set_visible(False)

# Block size ablation
block_sizes = [8, 16, 32, 64, 128]
perp_by_block = [11.2, 10.6, 10.4, 11.0, 12.8]  # stylized from paper
ax2.plot(block_sizes, perp_by_block, 'o-', color='#FF007F',
         linewidth=2, markersize=8, markeredgecolor='#1A1A1A', markeredgewidth=1.5)
ax2.axhline(y=10.0, color='#1A1A1A', linestyle='--', alpha=0.5, label='Dense baseline')
ax2.set_xlabel("Block size m (tokens per block)")
ax2.set_ylabel("Perplexity")
ax2.set_title("Block size ablation: m=32 is approximately optimal")
ax2.set_xticks(block_sizes)
ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The key findings from the ablation:
- **Removing the selection branch** costs ~3 perplexity points — it is the most important branch.
- **Removing compression** costs ~1.5 points — global context matters but less than fine-grained retrieval.
- **Removing the sliding window** costs ~2 points — local coherence is more important than compression alone.
- **Window-only** baseline (effectively StreamingLLM) loses ~6 points vs. NSA — fixed patterns are genuinely worse.
- **NSA vs. dense**: ~0.4 perplexity points gap with 30–40% FLOP reduction. **95–99% quality retention at 60–70% of the compute.**

The block size $m = 32$ is approximately optimal. Smaller blocks ($m = 8, 16$) recover more quality in the scoring step but add overhead; larger blocks ($m = 64, 128$) lose too much token-level precision in the selection step.

{{< crosshead >}}What Was Missing (Why This Was A Preprint){{< /crosshead >}}

The NSA paper is intellectually complete. The algorithm is correct, the ablations are real, the kernels work. So why didn't DeepSeek ship a model with NSA in February 2025?

**Problem 1: Training-from-scratch experiments only.**

Every result in the paper is from models trained with NSA from initialization — random weights, NSA attention, train until convergence. No result shows what happens if you take an existing pretrained model (say, DeepSeek V3) and continue-train it with NSA replacing its attention. This matters enormously. Retraining a frontier model from scratch costs $5M+. Continue-training might cost $50K–200K. If the team can't show a continue-training path, the economics don't work.

**Problem 2: Three branches per layer is heavier than one.**

Even though each branch is individually sparse, running three branches simultaneously adds latency versus running one. The compression branch needs its compressor network. The selection branch needs its scoring step plus the block-gather. The sliding window is cheap but still a third pass. For a model that already has MLA adding latency per layer, adding three-branch NSA on top of MLA is a significant per-layer cost increase.

**Problem 3: The compression branch is redundant with MLA.**

This one is visible in hindsight. The NSA compression branch learns a softmax-weighted blend of past tokens into a compressed representation. MLA (Multi-head Latent Attention) already produces exactly such a compressed representation: the **latent vector** $c^{KV}$ is a low-rank projection of the past hidden state, carrying compressed content information for all heads.

If you already have MLA, you already have a compressor. The NSA compression branch is relearning something MLA has already done. By May 2025, the DeepSeek team realizes this: **the MLA latent IS the compressor.** You don't need a separate compression branch — you just need a scoring mechanism that queries the MLA latent to find relevant past positions.

This insight collapses NSA's three branches to effectively one-and-a-half for a MLA-based model: the selection branch (using MLA latents as block keys instead of learned compressor outputs) plus the sliding window. That streamlined architecture is [DSA](../06-lightning-indexer/).

{{% callout type="tangent" %}}
If you're curious why MLA was built the way it was, the [MLA: The First Cut](../02-mla-rewind/) chapter has the full derivation. The key property for DSA is that the latent vector $c^{KV}$ is a 512-dimensional projection of the full hidden state — a learned compressed summary of what that token has seen. NSA's compressor does the same thing with a different architecture. When you already have one, you don't need the other.
{{% /callout %}}

{{< crosshead >}}The Blueprint → Product Mapping{{< /crosshead >}}

Read NSA carefully and you can trace every design decision forward to a downstream product:

| NSA Component | Downstream in |
|---|---|
| Branch 2: Selection (top-k blocks) | DSA — lightning indexer over MLA latents |
| Branch 3: Sliding window | DSA, CSA, HCA — appears in all three |
| Branch 1: Compression (m=32 softmax pooler) | CSA (reappears with 2-stream pooler) |
| Heavy compression (m'=128 variants in ablations) | HCA — heavy compression only, no selection |
| Block size m=32, warp-aligned | Retained throughout |
| STE + distillation training | Retained in DSA |

The NSA paper is not an intermediate research step that got abandoned. It is a design document for the entire attention stack of V3.2-Exp, written seven months before the model ships.

```pyplot {id="nsa-to-dsa-lineage" caption="THE NSA BLUEPRINT TRACES FORWARD: EACH DOWNSTREAM MECHANISM INHERITS ONE OR MORE NSA BRANCHES. NOTHING IS INVENTED FROM SCRATCH IN SEPTEMBER 2025."}
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')

# NSA box
nsa_x, nsa_y = 0.05, 0.35
ax.add_patch(plt.Rectangle((nsa_x, nsa_y), 0.22, 0.3, facecolor='#FF007F',
                            alpha=0.3, edgecolor='#1A1A1A', linewidth=2))
ax.text(nsa_x + 0.11, nsa_y + 0.15, 'NSA\n(Feb 2025)\n3 branches', ha='center',
        va='center', fontsize=10, fontweight='bold')

# Three branch labels inside NSA
for i, (label, y_off) in enumerate([('Compression', 0.24), ('Selection', 0.15), ('Window', 0.06)]):
    ax.text(nsa_x + 0.22, nsa_y + y_off, f'→ {label}', ha='left', va='center',
            fontsize=8.5, color='#1A1A1A')

# Downstream boxes
downstream = [
    (0.55, 0.62, 'DSA', 'Selection + Window\n(MLA latent as key)', '#FF007F'),
    (0.55, 0.35, 'CSA', 'Compression + Window\n(2-stream pooler)', '#00A8A8'),
    (0.55, 0.08, 'HCA', 'Heavy Compression\n(no selection)', '#FFD700'),
]
for (x, y, name, desc, c) in downstream:
    ax.add_patch(plt.Rectangle((x, y), 0.28, 0.22, facecolor=c,
                                alpha=0.3, edgecolor='#1A1A1A', linewidth=2))
    ax.text(x + 0.14, y + 0.15, name, ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(x + 0.14, y + 0.07, desc, ha='center', va='center', fontsize=7.5)

# Arrows from NSA to downstream
for (x, y, *_) in downstream:
    ax.annotate('', xy=(x, y + 0.11), xytext=(nsa_x + 0.22, nsa_y + 0.15),
                arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.5,
                                connectionstyle='arc3,rad=0.1'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("NSA (Feb 2025) → V3.2-Exp attention stack (Sep 2025): blueprint-to-product", fontsize=11)
```

{{< crosshead >}}What To Remember{{< /crosshead >}}

1. **NSA = three branches:** compression (global context at $O(T/m)^2$), selection (fine-grained retrieval at $O(T/m)$ scoring + $O(km)$ attention), sliding window (local coherence at $O(W)$).

2. **Hardware alignment is what made it real.** Block size $m=32$ matches warp size. Top-$k=16$ blocks × $m=32$ = 512 tokens matches FlashAttention tile size. The architecture was designed backwards from the kernel, not the other way around.

3. **Natively trainable** means two things working together: block-level straight-through estimators for gradient flow through discrete selection, and auxiliary distillation loss that explicitly trains the scorer against dense attention.

4. **It was a research preprint.** No model shipped on it. The lab was building toward something. The three blockers (no continue-training path, three-branch overhead, redundant compression vs. MLA) were the engineering problems to solve before shipping.

5. **All three branches recur downstream.** DSA inherits selection + window. CSA reintroduces compression explicitly. HCA inherits compression alone. Nothing in the September 2025 stack was invented from scratch — it was all visible in this February preprint.

**Continue to** → [Lightning Strikes Twice](../06-lightning-indexer/) — September 29, 2025. V3.2-Exp ships. The selection branch survives as the lightning indexer. Compression folds into MLA's latent. The three-branch overhead becomes one-and-a-half branches. The 50% price cut.
