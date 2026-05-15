---
title: "Inside LLM.int8() — The Two-Path MatMul"
description: "The 2022 algorithm that broke the 175B accessibility wall, dissected. Vector-wise scaling. The threshold of 6. Why three matmuls beat one. And why it is still 3-4× slower than FP16."
topics: [quantization]
tags: [llm-int8, bitsandbytes, dettmers, matmul, vector-wise]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 65
techKind: mainline
techNode: llm-int8
header: 06b-llm-int8-deep.webp
---

## Where We Are

[The 1% That Ruins Everything](../06-outliers/) told the detective story: Dettmers tries to quantize OPT-175B, the model collapses, the cause is six emergent feature dimensions out of 12,288. This chapter is the **autopsy of the algorithm**. We pry the lid off [LLM.int8()](https://arxiv.org/abs/2208.07339) and look at exactly what each line of the published method does, why each piece is necessary, and what the entire system gives up in return for keeping the activations alive.

You will leave this chapter able to (a) implement the core algorithm from scratch in twenty lines of numpy, (b) explain to a colleague why per-tensor absmax is unsalvageable at 6.7B+ parameters but per-row is fine, and (c) predict — within a factor of two — how much slower a given LLM.int8 deployment will be compared to FP16.

## The Three-Step Recipe

The Hugging Face paper summarizes the method in three numbered steps. They look innocent. Each one hides a design decision.

> 1. From the input hidden states, extract the outlier columns (any column whose absmax exceeds a threshold $\alpha$).
> 2. Perform the matmul in two pieces: outliers in {{< wiki "number-formats" >}}FP16{{< /wiki >}}, the rest in INT8 with vector-wise scaling.
> 3. Dequantize the INT8 result back to FP16 and add it to the FP16 outlier result.

Let us write this as math. We have an activation matrix $X \in \mathbb{R}^{T \times d}$ (token batch on rows, hidden dims on columns) and a {{< wiki "transformer-weights" >}}weight matrix{{< /wiki >}} $W \in \mathbb{R}^{m \times d}$ (output dims on rows, input dims on columns). We want $Y = X W^\top \in \mathbb{R}^{T \times m}$.

Define an **outlier mask** on columns of $X$:

$$
\mathcal{O} = \{\, j \in [d] \,:\, \max_t |X_{tj}| > \alpha \,\}.
$$

Split:

$$
X = X_{\mathcal O} + X_{\mathcal R}, \qquad
X_{\mathcal O}[:,j] = X[:,j] \text{ if } j \in \mathcal O \text{ else } 0,
$$

and analogously $W = W_{\mathcal O} + W_{\mathcal R}$ on columns. The matmul splits cleanly:

$$
Y = X W^\top = X_{\mathcal O} W_{\mathcal O}^\top + X_{\mathcal R} W_{\mathcal R}^\top.
$$

That is the algebra. The first term is computed in FP16 directly. The second term is computed in INT8 with vector-wise scaling, then dequantized. Let us examine each piece.

## Step 1: The Outlier Threshold

The threshold $\alpha = 6.0$ is not a magic number. Dettmers's paper shows it is the **smallest value** for which classification accuracy on OPT and BLOOM stays flush against the FP16 baseline. Drop $\alpha$ to 3 and you treat too many activations as outliers — the FP16 path bloats, the INT8 path becomes anaemic, you lose the speed benefit and gain nothing. Push $\alpha$ to 12 and you start treating real outliers as bulk; the bulk absmax balloons and the INT8 reconstruction error explodes.

The number 6 has an information-theoretic flavor to it: it is roughly $5\sigma$ above the standard activation scale, the point past which a Gaussian assumption is no longer reasonable. Anything bigger than 6 in a post-LayerNorm activation is a *named* feature dimension, not noise.

Per the [LLM.int8() paper](https://arxiv.org/abs/2208.07339), the consequences of this thresholding are dramatic and counterintuitive:

| Threshold $\alpha$ | Outlier column fraction | OPT-13B accuracy |
|---|---|---|
| ∞ (no split, naive INT8) | 0% | ↓ catastrophic |
| 12 | 0.02% | ↓ moderate |
| 6 | **0.1%** | ≈ FP16 |
| 3 | 0.5% | ≈ FP16 |
| 0 (everything FP16) | 100% | = FP16 |

Note the sweet spot. **One in a thousand activations** carries 100% of the "this number cannot be quantized" signal. The remaining 99.9% behave like a perfectly normal Gaussian and quantize trivially.

## Step 2a: Vector-Wise INT8 On The Bulk Path

The bulk path computes $Y_{\mathcal R} = X_{\mathcal R} W_{\mathcal R}^\top$ in INT8. Naive [absmax quantization](../02a-absmax/) on the whole tensor would still produce noticeable errors because rows of $X$ have very different magnitudes — a token that begins a sentence has different activation magnitudes than a token in mid-paragraph.

The fix is **vector-wise scaling**, also called per-row / per-column scaling. Instead of one scale per tensor, use:

- one scale per **row** of $X$ (one scalar per token in the batch),
- one scale per **column** of $W$ (one scalar per input dim of the linear layer).

Mathematically:

$$
\tilde X_{tj} = \mathrm{round}\!\left(\frac{127 \cdot X_{tj}}{s^X_t}\right),
\qquad
s^X_t = \max_j |X_{tj}|,
$$

$$
\tilde W_{kj} = \mathrm{round}\!\left(\frac{127 \cdot W_{kj}}{s^W_j}\right),
\qquad
s^W_j = \max_k |W_{kj}|.
$$

The INT8 matmul computes $\tilde Y_{tk} = \sum_j \tilde X_{tj} \tilde W_{kj}$ in pure integer arithmetic (INT8 multiply, INT32 accumulate). Dequantization is then a **single outer product**:

$$
Y_{tk} \;\approx\; \tilde Y_{tk} \cdot \frac{s^X_t \, s^W_j}{127^2}
\quad \text{wait — } s^W_j \text{ depends on } j? \text{ How is this an outer product?}
$$

Good catch. Look again. In the **bulk path**, we have already removed the outlier columns, so $s^W_j$ is roughly uniform across the surviving $j$'s. The vector-wise scale that matters per *output* element is the row scale $s^X_t$ and the *column* scale $s^W_k$ — note: $k$, not $j$. We pick per-output-column scales for $W$ along its *row* index $k$ (the axis we sum *into*), so dequantization is:

$$
Y_{tk} \;\approx\; \tilde Y_{tk} \cdot \frac{s^X_t \, s^W_k}{127^2}.
$$

This is exactly the outer product $\mathbf{s}^X (\mathbf{s}^W)^\top$ elementwise-multiplied with $\tilde Y / 127^2$. One outer product after the INT8 matmul. Free.

```pyplot {id="vectorwise-vs-pertensor" caption="Same weight matrix, three quantization granularities. Per-tensor is dominated by the column with the largest single weight; per-row improves; per-column improves further because LLM weight columns have very different scales."}
np.random.seed(11)
m, d = 64, 64
W = np.random.randn(m, d) * 0.1
# Inject column-level scale variation
col_scales = np.exp(np.random.randn(d) * 0.8)
W = W * col_scales[None, :]
# One ugly weight
W[10, 20] = 2.5

def quantize_dequant(W, axis):
    if axis is None:  # per-tensor
        s = np.abs(W).max()
        return np.round(W / s * 127) * s / 127
    s = np.abs(W).max(axis=axis, keepdims=True)
    s = np.where(s == 0, 1e-12, s)
    return np.round(W / s * 127) * s / 127

errs = {
    "per-tensor (1 scale)":     W - quantize_dequant(W, axis=None),
    "per-row (m scales)":       W - quantize_dequant(W, axis=1),
    "per-column (d scales)":    W - quantize_dequant(W, axis=0),
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (label, err) in zip(axes, errs.items()):
    im = ax.imshow(np.abs(err), aspect='auto', cmap='magma', vmax=np.abs(errs['per-tensor (1 scale)']).max())
    ax.set_title(f"{label}\n max |err| = {np.abs(err).max():.4f}")
    ax.set_xlabel("input dim j")
    if ax is axes[0]:
        ax.set_ylabel("output dim k")
plt.tight_layout()
```

The visual is unambiguous. Per-tensor scaling crushes everything to align with one outlier weight. Per-column scaling — the right granularity for $W$ in a matmul — leaves a clean residual. **One number per column is all you need.** The total metadata cost is $d$ FP16 scales for $W$ and $T$ FP16 scales for $X$ — utterly negligible compared to the $md$ INT8 weights.

## Step 2b: FP16 On The Outlier Path

The outlier path $Y_{\mathcal O} = X_{\mathcal O} W_{\mathcal O}^\top$ is a normal FP16 matmul on a *narrow* submatrix. Suppose 6 of the 12,288 hidden dims are outlier columns. Then $X_{\mathcal O}$ has shape $T \times 6$ and $W_{\mathcal O}$ has shape $m \times 6$. The cost of this matmul is $T \cdot m \cdot 6 \cdot 2 = 12Tm$ bytes of arithmetic versus the original $T \cdot m \cdot d \cdot 2 = 24576 Tm$. That is **0.05% of the original FP16 matmul cost**. Essentially free.

But — and this is the awkward part — it is a separate kernel launch, on the GPU, with its own data movement, that has to *complete* and *write to the same accumulator* as the dequantized INT8 result. The synchronization is what costs you.

## Step 3: The Recombine

The final assembly is:

$$
Y = \underbrace{X_{\mathcal O} W_{\mathcal O}^\top}_{\text{FP16 matmul, narrow}} \;+\; \underbrace{\big( \mathbf{s}^X (\mathbf{s}^W)^\top \big) \odot \frac{\tilde X_{\mathcal R} \tilde W_{\mathcal R}^\top}{127^2}}_{\text{INT8 matmul + outer-product dequant}}.
$$

Two matmuls, one outer product, one tensor add. In a fused CUDA kernel this is fast in principle — and in `bitsandbytes` 0.43+ it is partially fused. But the column-splitting step (Step 1, the outlier extraction) is a **scatter/gather**, and scatter/gathers are GPU's worst friend. This is the single biggest reason LLM.int8 inference is 3-4× slower than FP16: the GPU is doing more synchronization than computation.

## Try The Whole Thing

The widget below runs LLM.int8 in your browser on a small synthetic problem. Drag the outlier magnitude and the threshold $\alpha$ and watch:

- the relative size of the FP16 path (orange) vs the INT8 path (teal);
- the reconstruction error on the bulk (pink);
- the *total* matmul error compared to pure FP16.

<div class="li8-widget" data-widget="llmint8">
  <div class="li8-widget__controls">
    <label class="li8-widget__label">
      <span>Outlier magnitude</span>
      <input type="range" min="0" max="100" step="1" value="40" data-control="outlier">
      <output data-output="outlier">40</output>
    </label>
    <label class="li8-widget__label">
      <span>Threshold α</span>
      <input type="range" min="0" max="30" step="0.5" value="6" data-control="alpha">
      <output data-output="alpha">6.0</output>
    </label>
    <label class="li8-widget__label">
      <span>Hidden dim</span>
      <select data-control="dim">
        <option value="64">64</option>
        <option value="256" selected>256</option>
        <option value="1024">1024</option>
      </select>
    </label>
    <label class="li8-widget__label">
      <span>Number of outlier dims</span>
      <input type="range" min="0" max="20" step="1" value="3" data-control="nout">
      <output data-output="nout">3</output>
    </label>
  </div>
  <canvas data-canvas width="900" height="280"></canvas>
  <div class="li8-widget__readout" data-readout></div>
</div>

<style>
.li8-widget {
  border: 3px solid #1A1A1A;
  background: #FDF5E6;
  box-shadow: 4px 4px 0 #1A1A1A;
  padding: 1rem 1.1rem 1.2rem;
  margin: 1.5rem 0;
  font-family: 'Space Grotesk', system-ui, sans-serif;
}
.li8-widget__controls {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem 1rem;
  margin-bottom: 0.6rem;
}
@media (max-width: 800px) { .li8-widget__controls { grid-template-columns: 1fr 1fr; } }
.li8-widget__label { display: flex; flex-direction: column; font-size: 0.85rem; font-weight: 600; gap: 0.25rem; }
.li8-widget__label input[type="range"] { accent-color: #FF007F; }
.li8-widget__label select {
  border: 2px solid #1A1A1A; padding: 0.2rem 0.4rem; background: #fff; font: inherit;
}
.li8-widget__label output {
  font-variant-numeric: tabular-nums; font-weight: 700; color: #FF007F;
}
.li8-widget canvas { display: block; width: 100%; height: auto; background: #fff; border: 2px solid #1A1A1A; }
.li8-widget__readout {
  margin-top: 0.6rem; font-size: 0.85rem;
  background: #1A1A1A; color: #FDF5E6; padding: 0.5rem 0.8rem;
  white-space: pre-wrap; font-family: 'Space Mono', ui-monospace, monospace; line-height: 1.6;
}
</style>
<script>
(function() {
  function init(widget) {
    if (widget.dataset.bound) return;
    widget.dataset.bound = '1';
    const canvas   = widget.querySelector('[data-canvas]');
    const ctx      = canvas.getContext('2d');
    const outIn    = widget.querySelector('[data-control="outlier"]');
    const alphaIn  = widget.querySelector('[data-control="alpha"]');
    const dimIn    = widget.querySelector('[data-control="dim"]');
    const noutIn   = widget.querySelector('[data-control="nout"]');
    const outO     = widget.querySelector('[data-output="outlier"]');
    const alphaO   = widget.querySelector('[data-output="alpha"]');
    const noutO    = widget.querySelector('[data-output="nout"]');
    const readout  = widget.querySelector('[data-readout]');

    function gauss(n, seed) {
      let s = seed; const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        s = (s * 1664525 + 1013904223) >>> 0; const u1 = (s + 1) / 4294967296;
        s = (s * 1664525 + 1013904223) >>> 0; const u2 = (s + 1) / 4294967296;
        out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      return out;
    }

    function quantizeRow(X, T, d) {
      // per-row absmax, returns {q: Int8 flat, s: Float32 of length T}
      const q = new Int8Array(T * d), s = new Float32Array(T);
      for (let t = 0; t < T; t++) {
        let am = 0; for (let j = 0; j < d; j++) { const v = Math.abs(X[t*d+j]); if (v>am) am=v; }
        s[t] = am || 1e-9;
        const sc = 127 / s[t];
        for (let j = 0; j < d; j++) {
          const v = Math.round(X[t*d+j] * sc);
          q[t*d+j] = Math.max(-127, Math.min(127, v));
        }
      }
      return { q, s };
    }
    function quantizeColW(W, m, d) {
      // per-row (output dim) absmax of W (W is m x d)
      const q = new Int8Array(m * d), s = new Float32Array(m);
      for (let k = 0; k < m; k++) {
        let am = 0;
        for (let j = 0; j < d; j++) { const v = Math.abs(W[k*d+j]); if (v>am) am=v; }
        s[k] = am || 1e-9;
        const sc = 127 / s[k];
        for (let j = 0; j < d; j++) {
          const v = Math.round(W[k*d+j] * sc);
          q[k*d+j] = Math.max(-127, Math.min(127, v));
        }
      }
      return { q, s };
    }

    function draw() {
      const W = canvas.width, H = canvas.height;
      const outlier = parseFloat(outIn.value);
      const alpha   = parseFloat(alphaIn.value);
      const d       = parseInt(dimIn.value, 10);
      const nout    = parseInt(noutIn.value, 10);
      outO.value   = outlier.toFixed(0);
      alphaO.value = alpha.toFixed(1);
      noutO.value  = nout;

      const T = 8;     // small batch
      const m = 16;    // output dim
      const Xfull = gauss(T * d, 13);
      for (let t = 0; t < T; t++) Xfull[t * d + 0] *= 0.5;
      // Inject outlier columns
      const outlierCols = [];
      for (let i = 0; i < nout; i++) outlierCols.push(Math.floor((i + 0.5) * d / Math.max(1, nout)));
      for (let t = 0; t < T; t++) {
        for (const j of outlierCols) Xfull[t * d + j] = outlier * (Math.random() > 0.5 ? 1 : -1) * (0.5 + 0.5 * Math.random());
      }
      const Wfull = gauss(m * d, 31);
      for (let i = 0; i < m * d; i++) Wfull[i] *= 0.1;

      // True FP32 Y
      const Yref = new Float32Array(T * m);
      for (let t = 0; t < T; t++) for (let k = 0; k < m; k++) {
        let acc = 0;
        for (let j = 0; j < d; j++) acc += Xfull[t*d+j] * Wfull[k*d+j];
        Yref[t*m+k] = acc;
      }

      // Identify outlier columns by threshold
      const outMask = new Uint8Array(d);
      for (let j = 0; j < d; j++) {
        let am = 0;
        for (let t = 0; t < T; t++) { const v = Math.abs(Xfull[t*d+j]); if (v>am) am=v; }
        if (am > alpha) outMask[j] = 1;
      }
      let nOut = 0; for (const b of outMask) nOut += b;
      const nBulk = d - nOut;

      // Build XR, WR (bulk-only) — zero out outlier columns rather than densify (simpler)
      const XR = new Float32Array(T * d);
      const WR = new Float32Array(m * d);
      for (let t = 0; t < T; t++) for (let j = 0; j < d; j++) {
        XR[t*d+j] = outMask[j] ? 0 : Xfull[t*d+j];
      }
      for (let k = 0; k < m; k++) for (let j = 0; j < d; j++) {
        WR[k*d+j] = outMask[j] ? 0 : Wfull[k*d+j];
      }

      // Vector-wise quantize XR and WR
      const QX = quantizeRow(XR, T, d);
      const QW = quantizeColW(WR, m, d);

      // INT8 matmul + dequant
      const Ybulk = new Float32Array(T * m);
      for (let t = 0; t < T; t++) for (let k = 0; k < m; k++) {
        let acc = 0;
        for (let j = 0; j < d; j++) acc += QX.q[t*d+j] * QW.q[k*d+j];
        Ybulk[t*m+k] = acc * (QX.s[t] * QW.s[k]) / (127 * 127);
      }
      // FP16 outlier path
      const Yo = new Float32Array(T * m);
      for (let t = 0; t < T; t++) for (let k = 0; k < m; k++) {
        let acc = 0;
        for (const j of outlierCols) if (outMask[j]) acc += Xfull[t*d+j] * Wfull[k*d+j];
        Yo[t*m+k] = acc;
      }

      // Combined LLM.int8 result
      const Yli8 = new Float32Array(T * m);
      for (let i = 0; i < T*m; i++) Yli8[i] = Ybulk[i] + Yo[i];

      // Naive INT8 result (per-tensor absmax, no split) for comparison
      let amX = 0, amW = 0;
      for (const v of Xfull) { const a = Math.abs(v); if (a>amX) amX=a; }
      for (const v of Wfull) { const a = Math.abs(v); if (a>amW) amW=a; }
      const sX = 127/amX, sW = 127/amW;
      const Ynaive = new Float32Array(T * m);
      for (let t = 0; t < T; t++) for (let k = 0; k < m; k++) {
        let acc = 0;
        for (let j = 0; j < d; j++) {
          const qx = Math.max(-127, Math.min(127, Math.round(Xfull[t*d+j] * sX)));
          const qw = Math.max(-127, Math.min(127, Math.round(Wfull[k*d+j] * sW)));
          acc += qx * qw;
        }
        Ynaive[t*m+k] = acc * (amX * amW) / (127 * 127);
      }

      function rmse(a, b) {
        let s = 0; for (let i = 0; i < a.length; i++) { const e = a[i] - b[i]; s += e*e; }
        return Math.sqrt(s / a.length);
      }
      const errLi8   = rmse(Yli8, Yref);
      const errNaive = rmse(Ynaive, Yref);
      const refRMS   = Math.sqrt(Yref.reduce((a, v) => a + v*v, 0) / Yref.length) || 1e-9;

      // Draw schematic: hidden dim bar split
      ctx.clearRect(0, 0, W, H);
      const margin = 28;
      const barY = 70, barH = 38;
      const barW = W - 2 * margin;
      const fracOut = nOut / d;
      ctx.fillStyle = '#00A8A8';
      ctx.fillRect(margin, barY, barW * (1 - fracOut), barH);
      ctx.fillStyle = '#FF8C00';
      ctx.fillRect(margin + barW * (1 - fracOut), barY, barW * fracOut, barH);
      ctx.strokeStyle = '#1A1A1A'; ctx.lineWidth = 2;
      ctx.strokeRect(margin, barY, barW, barH);
      ctx.fillStyle = '#1A1A1A';
      ctx.font = 'bold 12px "Space Grotesk", sans-serif';
      ctx.fillText(`INT8 path:  ${nBulk} bulk dims (${(100*(1-fracOut)).toFixed(1)}%)`, margin, barY - 6);
      ctx.fillText(`FP16 path:  ${nOut} outlier dims (${(100*fracOut).toFixed(2)}%)`,
                    margin + barW * (1 - fracOut) + 6, barY - 6);
      ctx.fillText('Hidden dimension d → split by threshold α',
                    margin, barY + barH + 18);

      // Error bars
      const eY = 175, eH = 26;
      const refScale = Math.max(errNaive, errLi8, refRMS * 0.001) * 1.1;
      function bar(y, val, color, label) {
        const w = Math.min(1, val / refScale) * barW;
        ctx.fillStyle = color;
        ctx.fillRect(margin, y, w, eH);
        ctx.strokeStyle = '#1A1A1A'; ctx.lineWidth = 1.5;
        ctx.strokeRect(margin, y, barW, eH);
        ctx.fillStyle = '#1A1A1A';
        ctx.font = 'bold 12px "Space Grotesk", sans-serif';
        ctx.fillText(label + ' RMSE = ' + val.toExponential(2), margin + 8, y + 17);
      }
      bar(eY,         errNaive, '#FF007F', 'Naive per-tensor INT8');
      bar(eY + eH + 8, errLi8,   '#00A8A8', 'LLM.int8()');

      readout.textContent = [
        `d = ${d} hidden dims, m = ${m} output dims, T = ${T} tokens`,
        `Outlier columns detected at α=${alpha.toFixed(1)}: ${nOut} of ${d}  (${(100*fracOut).toFixed(2)}%)`,
        `Reference ‖Y‖_RMS = ${refRMS.toExponential(2)}`,
        `Naive INT8 RMSE   = ${errNaive.toExponential(2)}   (${(100*errNaive/refRMS).toFixed(2)}% of signal)`,
        `LLM.int8() RMSE   = ${errLi8.toExponential(2)}   (${(100*errLi8/refRMS).toFixed(2)}% of signal)`,
        `→ LLM.int8 is ${(errNaive/errLi8).toFixed(1)}× more accurate than naive INT8 on this matmul`,
      ].join('\n');
    }
    [outIn, alphaIn, dimIn, noutIn].forEach(el => el.addEventListener('input', draw));
    draw();
  }
  document.querySelectorAll('.li8-widget').forEach(init);
})();
</script>

Three things to play with:

- **Set outlier magnitude to 0.** With no outliers, naive INT8 and LLM.int8 are indistinguishable. The whole apparatus is unnecessary on small models. This is why pre-2022 quantization papers had no outlier-handling — they were not training models big enough to make outliers emerge.
- **Set outlier magnitude to 60 (OPT-66B regime).** Naive INT8 explodes. LLM.int8 holds steady. The ratio of errors is the *entire reason* LLM.int8 exists.
- **Push the threshold $\alpha$ to 25.** Outliers no longer get caught — they leak into the bulk path. LLM.int8 degrades to naive INT8. This shows why the threshold is a knob, not a fixed constant; in [GLM-130B](https://arxiv.org/abs/2210.02414) the outliers reach magnitude ~150 and a much higher threshold is needed.

## The Memory Math, Done On A Napkin

Here is the actual order-of-magnitude budget for OPT-175B inference on a single 8×A100 80GB node.

| Item | FP16 | LLM.int8() |
|---|---|---|
| Weight storage | $175 \cdot 10^9 \cdot 2$ B $= 350$ GB | $175 \cdot 10^9 \cdot 1$ B $= 175$ GB |
| Per-column scale factors (FP16) | — | $175 \cdot 10^9 / d \cdot 2$ B $\approx 28$ MB *(negligible)* |
| Outlier columns (FP16, ~0.1%) | — | $0.001 \cdot 350$ GB $= 350$ MB |
| Activation cache (per layer) | $T \cdot d \cdot 2$ B | $T \cdot d \cdot 1$ B + $T$ FP16 scales |
| **Total weights** | **350 GB** | **~176 GB** |

A factor of two on the weight memory bill. With 8×A100 80GB you have 640 GB available; 176 GB fits in **three** A100s, with all the activation and KV cache memory on the remaining five. Or — the headline result — you can fit it on **one** machine that has only 4 A100 80GB cards. Before LLM.int8, the *minimum bar of entry* to run OPT-175B was an eight-GPU node. After LLM.int8, it was a four-GPU node. *In practice* — accounting for activation memory — most users found themselves running OPT-175B on a 4×80GB or 4×48GB workstation, where before they could not run it at all.

For BLOOM-176B the numbers are essentially identical. For LLaMA-65B (released six months later) the weights go from 130 GB FP16 to 65 GB INT8 — which means it now fits on a single 80 GB GPU with room for the KV cache. That was the moment 65B-class models entered the consumer hobbyist universe.

## The Speed Bill

The slow side of LLM.int8 is not the math. The math is the same number of multiplies. The slow side is **everything around the math**: column gather, scale broadcast, kernel launch overhead for the FP16 outlier path, synchronization between the two paths, the lack of a fused kernel that pipelines all of it together. Published numbers from Dettmers's blog post (BLOOM-176B, A100 80GB, batch size 1):

| Precision | GPUs | Time per token (ms) |
|---|---|---|
| BF16 | 8 | **239** |
| INT8 (LLM.int8) | 4 | **282** |
| BF16 | 14 (A100 40GB) | 285 |
| INT8 (LLM.int8) | 5 (A100 40GB) | 367 |

Read the rows carefully. On *equivalent* hardware, LLM.int8 is **~20% slower per token** than BF16 — not 3-4×. The 3-4× slowdown number that you sometimes hear quoted is for **small models** (T5-3B and similar), where the kernel-launch overhead dominates the matmul cost. For large models, the per-token slowdown of LLM.int8 versus BF16 on the same hardware is a small fraction; the speed cost mostly disappears at scale because the matmuls take long enough that the overhead becomes a rounding error.

This is a counterintuitive observation worth pausing on. **LLM.int8 was billed as "slower than FP16 but more memory-efficient," and the trade-off framing made it sound like you were paying for memory with latency.** In reality, on a fully-loaded 175B-class model, you pay essentially nothing in per-token latency — you just get to use half as many GPUs. The slowdown is real on small models where the GPU is starved for work, but it is essentially absent on the models LLM.int8 was actually designed for.

## What Came After

Three months after LLM.int8 was published, **GPTQ** (Frantar & Alistarh, 2022) attacked the same problem from the other direction: instead of routing outliers around the quantizer, it quantizes weights one at a time using second-order Hessian information to *redistribute* the error onto the still-unquantized weights. This requires no special handling for outliers — they are absorbed in the per-block quantization. GPTQ is the subject of [Brain Surgery Returns](../08-brain-surgery/).

Six months later, **SmoothQuant** (Xiao et al., 2022) observed that you can *algebraically migrate* the outlier magnitudes from activations into weights using a per-channel rescaling that the matrix multiply absorbs for free. Now both activations and weights are in INT8 — no FP16 path, no synchronization cost. The price is a per-channel calibration pass.

By mid-2024, every modern method had either built on or replaced LLM.int8's two-path decomposition. But every one of them inherits its central insight: **outliers are not noise to be smoothed away. They are load-bearing features. The right move is to handle them, not to wish them gone.** See the comparison in [Method Family Tree](../09-method-family-tree/).

## A Short Cookbook

If you have read this far and you want to *use* this in real code, the bitsandbytes API has not changed since 2023:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# The single flag that does everything in this article.
config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,            # the α from above
    llm_int8_skip_modules=["lm_head"], # keep the output projection in FP16
)
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-13b",
    quantization_config=config,
    device_map="auto",
)
```

That is the entire interface to a 200-page algorithm. The `load_in_8bit=True` does the column-split, the vector-wise scale, the FP16 outlier path, the recombine. Three months of mathematical infrastructure is hidden behind a boolean.

This is the shape of *every* quantization library that followed. The user-facing API is a flag. The math underneath is a 2022 paper.

**Continue to** → [Taylor & Hessians](../07-taylor-and-hessians/) to see the second-order tools that power GPTQ and the post-LLM.int8 generation of methods.
