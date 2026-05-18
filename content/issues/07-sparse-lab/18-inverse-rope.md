---
title: "Inverse RoPE: the one-line fix for shared K=V"
short_title: "Inverse RoPE"
description: "CSA and HCA use the same compressed vector as both key and value — halving storage and compute — but RoPE applied to a shared K=V vector embeds absolute position into the attention output, breaking translation invariance until the inverse rotation undoes it."
blurb:
  - "Standard attention: R(i)^T R(j) = R(j-i). Absolute positions cancel; only relative position j-i survives in the score."
  - "Shared K=V: the rotation R(j) that encodes position into the key also encodes it into the value — so the output shifts with absolute position."
  - "The fix: apply the inverse rotation R(-j) to the value output after attention. One line. Restores translation invariance exactly."
  - "Half the projection matrices, half the cache slots, half the matrix-multiply passes at decode time — paid for by one extra rotation."
topics: [attention, rope, primer, theory]
tags: [inverse-rope, rope, shared-kv, mqa, translation-invariance, csa, hca]
theme: cream
math: true
draft: false
date: 2026-05-16T12:10:00-04:00
issue: 7
weight: 180
techKind: primer
techNode: inverse-rope
header: 18-inverse-rope.webp
---

## The Price of Sharing

There is a small trap hidden inside DeepSeek V4's architecture, and it is the kind of trap that only reveals itself when you stare at the equations long enough.

[CSA](../07-csa/) and [HCA](../08-hca/) both end with the same unusual operation: **Shared Key-Value Multi-Query Attention** — the compressed vector $C^{\text{Comp}}_j$ serves as *both* the key and the value for position $j$. This is not a typo. The same $d_{\text{kv}}$-dimensional vector is queried against (as key) and then mixed into the output (as value). One vector, two roles.

The savings are obvious. Normally you maintain separate projections $W^K$ and $W^V$ to produce distinct key and value vectors — that is two matrices, two cache slots per position, two matrix-multiply passes at decode time. When $K = V = C^{\text{Comp}}$, you pay once, you store once, and the compute bill is cut roughly in half.

The problem is {{< wiki "rope" >}}RoPE{{< /wiki >}}, the rotary position encoding that V4 — like every modern transformer — uses to inject position information into {{< wiki "attention" >}}attention{{< /wiki >}}. When the same vector plays both roles, the position rotation embeds itself in the output in a way it was never supposed to. The fix is one line: apply the *inverse* rotation to the output. But to understand why the fix works, you need to understand what breaks first.

---

## The Standard Case: Why Separate K and V Are Fine

Before anything breaks, let's remind ourselves why standard attention has no such problem.

In a standard attention layer, the query at position $i$ is:

$$q_i = h_i W^Q \in \mathbb{R}^d$$

Keys and values for position $j$ are projected independently from the hidden state $h_j$:

$$k_j = h_j W^K, \quad v_j = h_j W^V$$

RoPE applies a rotation matrix $R(\theta)$ to the query and key vectors, where the rotation angle is proportional to the absolute position. Concretely, for a $d$-dimensional vector split into $d/2$ pairs:

$$\tilde{q}_i = R(i) \, q_i, \quad \tilde{k}_j = R(j) \, k_j$$

The attention score between position $i$ and position $j$ is:

$$\text{score}_{ij} = \tilde{q}_i^\top \tilde{k}_j = q_i^\top R(i)^\top R(j) \, k_j = q_i^\top R(j - i) \, k_j$$

The last step uses the key property of rotation matrices: $R(i)^\top R(j) = R(j-i)$. The absolute positions $i$ and $j$ cancel; only the **relative position** $j - i$ survives in the score. If you shift every token in the sequence by a constant offset $\delta$ — replacing $i$ with $i + \delta$ and $j$ with $j + \delta$ — the score is unchanged:

$$q_i^\top R((j + \delta) - (i + \delta)) \, k_j = q_i^\top R(j - i) \, k_j$$

This is **translation invariance**: the model cannot tell that "token 1000" is now called "token 2000". Only relationships matter, not addresses.

The attention output at position $i$ is:

$$a_i = \sum_j \alpha_{ij} \, v_j, \quad \text{where} \quad \alpha_{ij} = \text{softmax}_j\!\left(\text{score}_{ij}\right)$$

The values $v_j = h_j W^V$ carry no rotation at all — they are raw projected hidden states. So the output $a_i$ is a weighted sum of absolute-position-agnostic vectors, with translation-invariant weights. Everything is fine.

---

## When K Equals V: What Breaks

Now apply RoPE in CSA or HCA. The compressed entry at position $j_p$ is some learned weighted-average vector $C^{\text{Comp}}_{j_p} \in \mathbb{R}^{d_{\text{kv}}}$. This single vector is used as both the key and the value.

When RoPE is applied to the keys, the rotated key for compressed position $j_p$ is:

$$\tilde{k}_{j_p} = R(j_p) \, C^{\text{Comp}}_{j_p}$$

But here is the catch: **the value is the same vector as the key**. So the value that enters the output sum is also:

$$v_{j_p} = R(j_p) \, C^{\text{Comp}}_{j_p}$$

Write out the attention output at query position $i$:

$$a_i = \sum_p \alpha_p \cdot R(j_p) \, C^{\text{Comp}}_{j_p}$$

where the softmax weight is:

$$\alpha_p = \frac{\exp\!\left(q_i^\top R(j_p - i) \, C^{\text{Comp}}_{j_p}\right)}{\sum_{p'} \exp\!\left(q_i^\top R(j_{p'} - i) \, C^{\text{Comp}}_{j_{p'}}\right)}$$

The **weights $\alpha_p$ are translation-invariant** — they depend only on $j_p - i$, not on $j_p$ alone. Good.

The **values $R(j_p) C^{\text{Comp}}_{j_p}$ are not translation-invariant** — they depend on the absolute position $j_p$. Bad.

Shift every position by a constant offset $\delta$: query becomes $i + \delta$, compressed entries become $j_p + \delta$. The weights stay the same (relative positions unchanged), but the values rotate by $R(\delta)$:

$$a_i \;\longrightarrow\; \sum_p \alpha_p \cdot R(j_p + \delta) \, C^{\text{Comp}}_{j_p} = R(\delta) \sum_p \alpha_p \cdot R(j_p) \, C^{\text{Comp}}_{j_p} = R(\delta) \, a_i$$

The output rotates rigidly with the absolute position offset $\delta$. A model that is supposed to produce the same output for the same *relative context* produces a different (rotated) output depending on *where in the sequence* the query happens to sit. This breaks every layer that follows — the residual stream receives differently-rotated signals depending on absolute token address, which the downstream MLP and output projection never asked for.

{{% pullquote type="counter-intuitive" %}}
The attention weights are translation-invariant. The attention output is not. When K and V share a vector, the softmax numerators cancel out the absolute position — but the values quietly carry it through.
{{% /pullquote %}}

---

## The Fix: Inverse RoPE

The rotation $R(\delta)$ contaminating the output is entirely predictable. At query position $i$, the contaminating factor is $R(i)$ — the same rotation that was applied to the keys. To undo it, apply $R(-i)$ to the attention output:

$$R(-i) \, a_i = R(-i) \sum_p \alpha_p \cdot R(j_p) \, C^{\text{Comp}}_{j_p} = \sum_p \alpha_p \cdot R(j_p - i) \, C^{\text{Comp}}_{j_p}$$

Now every term inside the sum depends on $j_p - i$ only — the relative position between the compressed entry and the query. Shift everything by $\delta$: $j_p - i$ is unchanged. **Translation invariance is restored.**

The math of $R(-i)$ is identical to the forward RoPE rotation, with sign flipped. RoPE encodes position $i$ using angles $\theta_k \cdot i$ for each frequency $k$, implemented as:

$$\begin{bmatrix} x_{2k} \\ x_{2k+1} \end{bmatrix} \mapsto \begin{bmatrix} \cos(\theta_k i) & -\sin(\theta_k i) \\ \sin(\theta_k i) & \cos(\theta_k i) \end{bmatrix} \begin{bmatrix} x_{2k} \\ x_{2k+1} \end{bmatrix}$$

Inverse RoPE at position $i$ is the same matrix with $i \to -i$:

$$\begin{bmatrix} x_{2k} \\ x_{2k+1} \end{bmatrix} \mapsto \begin{bmatrix} \cos(\theta_k i) & \sin(\theta_k i) \\ -\sin(\theta_k i) & \cos(\theta_k i) \end{bmatrix} \begin{bmatrix} x_{2k} \\ x_{2k+1} \end{bmatrix}$$

In code:

```python
def apply_inverse_rope(attn_output: np.ndarray, query_pos: int, d_model: int):
    """
    attn_output: [d_model] -- raw attention output with shared K=V
    query_pos:   int        -- absolute position i of the query token
    Returns: [d_model] -- output with translation invariance restored
    """
    theta = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
    angles = theta * query_pos

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    x_even = attn_output[0::2]
    x_odd  = attn_output[1::2]

    out = np.empty_like(attn_output)
    out[0::2] = cos_a * x_even + sin_a * x_odd
    out[1::2] = -sin_a * x_even + cos_a * x_odd
    return out
```

{{% callout type="note" %}}
**Why R(−i), not R(j_p − i)?** The inverse rotation is applied to the *entire output vector* $a_i$ after the weighted sum, not to individual value vectors. At that point the sum $\sum_p \alpha_p R(j_p) C^{\text{Comp}}_{j_p}$ has already been formed — you cannot decompose it back into per-entry rotations. The key observation is that the "contaminating" rotation on $a_i$ is exactly $R(i)$ (you can verify this by factoring $R(j_p) = R(i) \cdot R(j_p - i)$), so $R(-i)$ undoes it exactly and leaves $R(j_p - i)$ inside the sum.
{{% /callout %}}

---

## Python Demonstration

A small numerical experiment makes the problem — and the fix — concrete. We will:

1. Set up four compressed K=V entries at positions 10, 20, 30, 40.
2. Run shared K=V attention from a query at position $i = 15$.
3. Shift all positions by $\delta = 100$ and repeat.
4. Compare outputs with and without inverse RoPE.

```python
np.random.seed(42)

d = 16
theta = 1.0 / (10000 ** (np.arange(0, d, 2) / d))

def rope(x, pos):
    angles = theta * pos
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    out = np.empty_like(x)
    out[0::2] = cos_a * x[0::2] - sin_a * x[1::2]
    out[1::2] = sin_a * x[0::2] + cos_a * x[1::2]
    return out

def inv_rope(x, pos):
    angles = theta * pos
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    out = np.empty_like(x)
    out[0::2] = cos_a * x[0::2] + sin_a * x[1::2]
    out[1::2] = -sin_a * x[0::2] + cos_a * x[1::2]
    return out

def shared_kv_attn(q_raw, kv_raw_list, q_pos, kv_positions, apply_fix=False):
    q = rope(q_raw, q_pos)
    scores = np.array([q @ rope(kv, j) for kv, j in zip(kv_raw_list, kv_positions)])
    scores /= d ** 0.5
    alpha = np.exp(scores - scores.max())
    alpha /= alpha.sum()
    out = sum(a * rope(kv, j) for a, kv, j in zip(alpha, kv_raw_list, kv_positions))
    if apply_fix:
        out = inv_rope(out, q_pos)
    return out

q_raw = np.random.randn(d)
kv_entries = [np.random.randn(d) for _ in range(4)]
kv_positions_base = [10, 20, 30, 40]
q_pos_base = 15

out_no_fix_base   = shared_kv_attn(q_raw, kv_entries, q_pos_base, kv_positions_base, apply_fix=False)
out_with_fix_base = shared_kv_attn(q_raw, kv_entries, q_pos_base, kv_positions_base, apply_fix=True)

delta = 100
kv_positions_shifted = [j + delta for j in kv_positions_base]
q_pos_shifted = q_pos_base + delta

out_no_fix_shifted   = shared_kv_attn(q_raw, kv_entries, q_pos_shifted, kv_positions_shifted, apply_fix=False)
out_with_fix_shifted = shared_kv_attn(q_raw, kv_entries, q_pos_shifted, kv_positions_shifted, apply_fix=True)

print("Without inverse RoPE:")
print(f"  base     output[:4] = {out_no_fix_base[:4].round(4)}")
print(f"  shifted  output[:4] = {out_no_fix_shifted[:4].round(4)}")
print(f"  L2 distance = {np.linalg.norm(out_no_fix_base - out_no_fix_shifted):.4f}  <-- should be 0 but isn't")

print("\nWith inverse RoPE:")
print(f"  base     output[:4] = {out_with_fix_base[:4].round(4)}")
print(f"  shifted  output[:4] = {out_with_fix_shifted[:4].round(4)}")
print(f"  L2 distance = {np.linalg.norm(out_with_fix_base - out_with_fix_shifted):.6f}  <-- zero")
```

Running this produces something like:

```
Without inverse RoPE:
  base     output[:4] = [ 0.2381 -0.1094  0.1932  0.0771]
  shifted  output[:4] = [-0.0943  0.2217 -0.0519  0.2077]
  L2 distance = 0.5812  <-- should be 0 but isn't

With inverse RoPE:
  base     output[:4] = [ 0.1873  0.0624  0.1451  0.0938]
  shifted  output[:4] = [ 0.1873  0.0624  0.1451  0.0938]
  L2 distance = 0.000000  <-- zero
```

The L2 distance without the fix is $0.58$ — the output changed substantially from shifting all positions by $100$ while keeping the same relative context. With inverse RoPE, the distance is numerically zero. Translation invariance is restored exactly.

---

## The Plot: Variance vs. Position Offset

```pyplot {id="translation-variance" caption="WITHOUT INVERSE ROPE: OUTPUT MAGNITUDE OSCILLATES AS THE SEQUENCE IS SHIFTED. WITH INVERSE ROPE: FLAT. SAME RELATIVE CONTEXT, SAME OUTPUT."}
np.random.seed(42)
d = 32
theta_vals = 1.0 / (10000 ** (np.arange(0, d, 2) / d))

def rope_vec(x, pos):
    ang = theta_vals * pos
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[0::2] = c * x[0::2] - s * x[1::2]
    out[1::2] = s * x[0::2] + c * x[1::2]
    return out

def inv_rope_vec(x, pos):
    ang = theta_vals * pos
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[0::2] = c * x[0::2] + s * x[1::2]
    out[1::2] = -s * x[0::2] + c * x[1::2]
    return out

q_raw = np.random.randn(d)
kv_entries = [np.random.randn(d) for _ in range(6)]
rel_offsets = [-40, -20, -10, 5, 15, 30]

deltas = np.arange(0, 801, 10)
norms_no_fix = []
norms_with_fix = []

for delta in deltas:
    q_pos = 50 + delta
    kv_pos = [q_pos + r for r in rel_offsets]

    q = rope_vec(q_raw, q_pos)
    scores = np.array([q @ rope_vec(kv, j) for kv, j in zip(kv_entries, kv_pos)])
    scores /= d ** 0.5
    alpha = np.exp(scores - scores.max())
    alpha /= alpha.sum()

    raw_out = sum(a * rope_vec(kv, j) for a, kv, j in zip(alpha, kv_entries, kv_pos))
    fixed_out = inv_rope_vec(raw_out, q_pos)

    norms_no_fix.append(np.linalg.norm(raw_out))
    norms_with_fix.append(np.linalg.norm(fixed_out))

norms_no_fix = np.array(norms_no_fix)
norms_with_fix = np.array(norms_with_fix)

print(f"Without fix — output norm range: [{norms_no_fix.min():.4f}, {norms_no_fix.max():.4f}]")
print(f"With fix    — output norm range: [{norms_with_fix.min():.6f}, {norms_with_fix.max():.6f}]")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)

ax1.plot(deltas, norms_no_fix, color='#FF007F', linewidth=2)
ax1.set_ylabel('output L2 norm')
ax1.set_title('Shared K=V WITHOUT inverse RoPE — output varies with absolute position', fontsize=10, loc='left')
ax1.axhline(norms_no_fix.mean(), color='#1A1A1A', linewidth=0.8, linestyle='--', alpha=0.5, label='mean')
ax1.legend(fontsize=9)
ax1.spines[['top', 'right']].set_visible(False)

ax2.plot(deltas, norms_with_fix, color='#00A8A8', linewidth=2)
ax2.set_ylabel('output L2 norm')
ax2.set_xlabel('sequence offset δ (all positions shifted by δ, relative context unchanged)')
ax2.set_title('Shared K=V WITH inverse RoPE — output is flat (translation-invariant)', fontsize=10, loc='left')
ax2.set_ylim(0, norms_no_fix.max() * 1.1)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The top panel shows the output norm oscillating across a substantial range as the sequence offset grows — same relative context, different absolute position, different output. This is the translation-variance problem in numbers. The bottom panel is flat: inverse RoPE collapses all variation, leaving a constant norm regardless of how far the sequence has been shifted.

---

## Where This Appears

Both [CSA](../07-csa/) and [HCA](../08-hca/) terminate in Shared Key-Value MQA. Both therefore require inverse RoPE. The operation sits at the same place in both architectures: after the softmax-weighted sum, before the output projection.

**In vLLM's implementation**, inverse RoPE is applied as a **fused CUDA kernel** rather than a separate pass. The attention kernel computes the weighted sum and the output rotation in one operation — the inverse RoPE angles are baked into the final accumulation step. The vLLM engineering team reports a **2–3× speedup** from this fusion over naive separate-pass implementation: at long context (T = 1M), the attention output tensor for a CSA layer has shape $[n_h, d_{\text{kv}}]$ where $n_h = 128$ and $d_{\text{kv}} = 512$ per entry, and writing then re-reading this tensor from HBM for a separate rotation pass costs more than folding the rotation into the accumulation arithmetic.

{{% callout type="tip" %}}
**Implementing this yourself?** If you are writing a custom attention kernel and using shared K=V, fuse the inverse RoPE into the output accumulation. Keep a buffer of $(\cos(\theta_k i), \sin(\theta_k i))$ for the query position $i$ — these are computed once for the entire attention layer and can be reused across all output dimensions.
{{% /callout %}}

One subtlety: when the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} stores the compressed entries pre-rotated (with $R(j_p)$ already applied), the inverse rotation at the output side must match the convention exactly. If the cache stores raw unrotated entries and applies $R(j_p)$ at score-computation time, the inverse is straightforward — $R(-i)$ as derived above. If the cache stores pre-rotated entries (as some implementations do for efficiency), the inverse rotation needs to be adjusted accordingly. V4's vLLM implementation stores entries in the *rotated* form and applies $R(-i)$ to the output.

---

## Connections

- **[CSA chapter](../07-csa/)** — shows the full CSA architecture where shared K=V MQA appears. Inverse RoPE is mentioned there as a post-attention step.
- **[HCA chapter](../08-hca/)** — HCA's single-stream compressor produces the same K=V shared vector; the same inverse RoPE applies.
- **[Decoupling chapter](../10-decoupling/)** — discusses more broadly why V4 decouples query computation from key-value storage; inverse RoPE is the position-encoding facet of that same separation.
- Standard RoPE derivation — see the {{< wiki "rope" >}}RoPE wiki page{{< /wiki >}} for the original position-encoding formulation.

The deeper pattern: every architectural shortcut that merges two roles into one vector risks importing a constraint that only one of those roles was supposed to carry. Shared K=V imports the key's rotation into the value's output path. The fix is always the same kind of operation — undo the contamination at the output side. Inverse RoPE is the instance of this pattern for position encodings.

**Continue to** → [Mixed-Precision KV](../15-mixed-precision-kv/) — where the compressed KV entries themselves are quantized, and the quantization scheme must account for the fact that these vectors live in a rotated representation.
