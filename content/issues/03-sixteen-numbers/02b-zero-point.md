---
title: "Zero-Point Quantization"
description: "The asymmetric cousin of absmax. Two parameters instead of one — a scale and an offset — bought at the cost of an extra add per multiply. The default for ReLU activations and the UINT8 of mobile inference."
topics: [quantization, number-formats]
tags: [zero-point, asymmetric, uint8, qnn, tflite]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 27
techKind: primer
techNode: zero-point
header: default.webp
---

## The Wasted Half

In 2017, a team at Google published a paper with the dry title [*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*](https://arxiv.org/abs/1712.05877). The result was the **gemmlowp** library and, downstream, every quantized MobileNet you have run on your phone. The whole technique runs on one observation about [absmax quantization](../02a-absmax/):

> If your data lives in $[0, 6]$ and you quantize symmetrically around zero, **half of your integer codes go to numbers that cannot occur.**

This is not a minor point. Modern transformers contain layers that produce only non-negative outputs — anything downstream of a `ReLU`, `GELU`-with-clamp, or `sigmoid` lives in a one-sided range. For these tensors, symmetric absmax is wasting *one whole bit* of precision out of eight. That is the difference between {{< wiki "number-formats" >}}INT8{{< /wiki >}} and INT7, paid silently every layer.

The fix is to drop symmetry. Allow the quantization grid to slide off-zero. This is **zero-point quantization** — sometimes called *affine* or *asymmetric* quantization — and it is the default integer format on every mobile inference runtime: TFLite, QNN, Core ML, ONNX Runtime.

## The Math

You want to map a real interval $[\alpha, \beta]$ (the observed range of your tensor — possibly entirely positive, possibly straddling zero, possibly entirely negative) onto an integer interval $[q_{\min}, q_{\max}]$ (typically $[0, 255]$ for UINT8 or $[-128, 127]$ for INT8). The map is affine:

$$
q = \mathrm{round}\!\left(\frac{x}{s}\right) + z, \qquad
\hat x = s \cdot (q - z).
$$

Two parameters now: a **scale** $s$ and a **zero-point** $z$. We pick them by demanding that the two ends of the real interval map exactly to the two ends of the integer interval:

$$
s = \frac{\beta - \alpha}{q_{\max} - q_{\min}}, \qquad
z = \mathrm{round}\!\left(q_{\min} - \frac{\alpha}{s}\right).
$$

The zero-point $z$ is the integer code that represents the real value zero. It is **not** zero in general. For a tensor with $\alpha = 0, \beta = 6$ quantized into UINT8, $z = 0$. For a tensor with $\alpha = -1, \beta = 1$ quantized into UINT8, $z = 128$. The grid has been shifted so that whichever side of zero your data lives on, **none of the codes are wasted**.

Compare this to [absmax](../02a-absmax/), which is the special case where $\alpha = -\beta = -s$ and $z = 0$. Absmax is the symmetric collapse of zero-point quantization. Zero-point is strictly more flexible.

## Why You Need The Offset

It is tempting to ask: why not just use absmax with a per-channel scale and call it done? Two pictures explain why the offset is non-negotiable.

```pyplot {id="zeropoint-vs-absmax" caption="Same post-ReLU activation tensor. Absmax (top) places its grid symmetrically around zero — half its codes sit in the negative half-line where no value can ever appear. Zero-point (bottom) shifts the grid to cover only the populated region."}
np.random.seed(3)
# Post-ReLU activations: one-sided, exponential-tailed.
acts = np.clip(np.random.randn(50000) + 1.0, 0, None) * 2.0
qmin, qmax = 0, 15  # INT4 to make the wasted codes visible

# Absmax
absmax = np.abs(acts).max()
scale_a = absmax / qmax     # only the positive half is used; -qmax..0 wasted
levels_a = (np.arange(-qmax, qmax + 1)) * scale_a

# Zero-point on [0, max]
alpha, beta = acts.min(), acts.max()
scale_z = (beta - alpha) / (qmax - qmin)
zp = round(qmin - alpha / scale_z)
levels_z = (np.arange(qmin, qmax + 1) - zp) * scale_z

fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
for ax, levels, title, used_color in [
    (axes[0], levels_a, "Absmax INT4 grid: half the codes are wasted on the empty negative half-line", '#FF007F'),
    (axes[1], levels_z, "Zero-point INT4 grid: every code falls inside the populated region", '#00A8A8'),
]:
    ax.hist(acts, bins=120, color='#FFD700', edgecolor='#1A1A1A',
            linewidth=0.3, alpha=0.7, density=True)
    for lv in levels:
        is_used = (lv >= alpha - 0.1) and (lv <= beta + 0.1)
        ax.axvline(lv, color=used_color if is_used else '#888',
                   linestyle='-' if is_used else '--',
                   linewidth=1.4 if is_used else 0.8,
                   alpha=0.9 if is_used else 0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-7, 7)
    ax.spines[['top', 'right']].set_visible(False)
axes[1].set_xlabel("activation value")
plt.tight_layout()
```

Solid lines are grid points that *can be reached* by the data; dashed grey lines are grid points that exist in the integer codebook but are unreachable. Absmax forfeits eight of sixteen codes. Zero-point uses all sixteen. **Same number of bits stored on disk. Double the effective resolution where the data lives.**

## Worked Example

Take the toy vector again, but from a one-sided distribution: $\mathbf{x} = [0.1, 0.4, 1.2, 0.0, 5.0, 0.3, 2.8, 4.7]$, target UINT8.

- $\alpha = 0$, $\beta = 5.0$.
- $s = (5.0 - 0) / (255 - 0) = 0.0196$.
- $z = \mathrm{round}(0 - 0 / 0.0196) = 0$. (Zero-point is exactly the integer code at the left edge.)
- Quantize: $q_i = \mathrm{round}(x_i / 0.0196) + 0 = [5, 20, 61, 0, 255, 15, 143, 240]$.
- Dequantize: $\hat x_i = 0.0196 \cdot (q_i - 0) = [0.098, 0.392, 1.196, 0, 5.00, 0.294, 2.804, 4.706]$.

Maximum error is $0.008$. The same data through symmetric absmax (with absmax = 5.0, into the range $[-127, 127]$) would have a step size of $5.0/127 \approx 0.039$ — *five times worse*. The asymmetric format simply gives the data more room.

## Try It Yourself

The widget below quantizes a stream of points through both schemes side-by-side. Drag the distribution off-center and watch absmax waste codes while zero-point absorbs the shift.

<div class="zp-widget" data-widget="zeropoint">
  <div class="zp-widget__controls">
    <label class="zp-widget__label">
      <span>Distribution mean</span>
      <input type="range" min="-3" max="3" step="0.05" value="1.5" data-control="mean">
      <output data-output="mean">1.50</output>
    </label>
    <label class="zp-widget__label">
      <span>Distribution std</span>
      <input type="range" min="0.05" max="2" step="0.05" value="0.6" data-control="std">
      <output data-output="std">0.60</output>
    </label>
    <label class="zp-widget__label">
      <span>Clip negative? (simulates ReLU)</span>
      <input type="checkbox" data-control="relu" checked>
    </label>
    <label class="zp-widget__label">
      <span>Bit width</span>
      <select data-control="bits">
        <option value="3">INT3 (8 levels)</option>
        <option value="4" selected>INT4 (16 levels)</option>
        <option value="5">INT5 (32 levels)</option>
        <option value="8">INT8 (256 levels)</option>
      </select>
    </label>
  </div>
  <canvas data-canvas width="900" height="360"></canvas>
  <div class="zp-widget__readout" data-readout></div>
</div>

<style>
.zp-widget {
  border: 3px solid #1A1A1A;
  background: #FDF5E6;
  box-shadow: 4px 4px 0 #1A1A1A;
  padding: 1rem 1.1rem 1.2rem;
  margin: 1.5rem 0;
  font-family: 'Space Grotesk', system-ui, sans-serif;
}
.zp-widget__controls {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  gap: 0.8rem 1rem;
  margin-bottom: 0.6rem;
}
@media (max-width: 700px) {
  .zp-widget__controls { grid-template-columns: 1fr 1fr; }
}
.zp-widget__label {
  display: flex;
  flex-direction: column;
  font-size: 0.85rem;
  font-weight: 600;
  gap: 0.25rem;
}
.zp-widget__label input[type="range"] { accent-color: #00A8A8; }
.zp-widget__label select {
  border: 2px solid #1A1A1A; padding: 0.2rem 0.4rem; background: #fff; font: inherit;
}
.zp-widget__label output {
  font-variant-numeric: tabular-nums;
  font-weight: 700; color: #00A8A8;
}
.zp-widget canvas {
  display: block; width: 100%; height: auto;
  background: #fff; border: 2px solid #1A1A1A;
}
.zp-widget__readout {
  margin-top: 0.6rem;
  font-size: 0.85rem;
  background: #1A1A1A; color: #FDF5E6;
  padding: 0.5rem 0.8rem;
  white-space: pre-wrap;
  font-family: 'Space Mono', ui-monospace, monospace;
  line-height: 1.6;
}
</style>
<script>
(function() {
  function init(widget) {
    if (widget.dataset.bound) return;
    widget.dataset.bound = '1';
    const canvas = widget.querySelector('[data-canvas]');
    const ctx    = canvas.getContext('2d');
    const meanIn = widget.querySelector('[data-control="mean"]');
    const stdIn  = widget.querySelector('[data-control="std"]');
    const reluIn = widget.querySelector('[data-control="relu"]');
    const bitsIn = widget.querySelector('[data-control="bits"]');
    const meanO  = widget.querySelector('[data-output="mean"]');
    const stdO   = widget.querySelector('[data-output="std"]');
    const readout= widget.querySelector('[data-readout]');

    function gauss(n, seed) {
      let s = seed; const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        s = (s * 1664525 + 1013904223) >>> 0; const u1 = (s + 1) / 4294967296;
        s = (s * 1664525 + 1013904223) >>> 0; const u2 = (s + 1) / 4294967296;
        out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      return out;
    }
    const base = gauss(2000, 7);

    function draw() {
      const W = canvas.width, H = canvas.height;
      const mean = parseFloat(meanIn.value);
      const std  = parseFloat(stdIn.value);
      const relu = reluIn.checked;
      const bits = parseInt(bitsIn.value, 10);
      meanO.value = mean.toFixed(2);
      stdO.value  = std.toFixed(2);

      const data = new Float32Array(base.length);
      for (let i = 0; i < base.length; i++) {
        let v = base[i] * std + mean;
        if (relu) v = Math.max(0, v);
        data[i] = v;
      }

      let mn = Infinity, mx = -Infinity, am = 0;
      for (const v of data) {
        if (v < mn) mn = v; if (v > mx) mx = v;
        const a = Math.abs(v); if (a > am) am = a;
      }

      // Absmax (signed): levels at q * (am / qmax) for q in [-qmax..qmax]
      const qmax = (1 << (bits - 1)) - 1;
      const levelsA = qmax * 2 + 1;
      const sA = am / qmax;

      // Zero-point: map [mn, mx] to [0, 2^bits - 1]
      const uqmax = (1 << bits) - 1;
      const sZ = (mx - mn) / uqmax;
      const z  = Math.round(0 - mn / sZ);  // integer code for value 0

      // Compute reconstruction errors
      let sseA = 0, sseZ = 0, usedA = new Set(), usedZ = new Set();
      for (const v of data) {
        const qa = Math.max(-qmax, Math.min(qmax, Math.round(v / sA)));
        usedA.add(qa);
        const ra = qa * sA; sseA += (v - ra) ** 2;
        const qz = Math.max(0, Math.min(uqmax, Math.round(v / sZ) + z));
        usedZ.add(qz);
        const rz = sZ * (qz - z); sseZ += (v - rz) ** 2;
      }
      const rmseA = Math.sqrt(sseA / data.length);
      const rmseZ = Math.sqrt(sseZ / data.length);

      ctx.clearRect(0, 0, W, H);
      const margin = 28;
      const halfH = (H - 3 * margin) / 2;

      const viewLo = Math.min(-am, mn) * 1.05;
      const viewHi = Math.max( am, mx) * 1.05;
      const xOf = (v) => margin + ((v - viewLo) / (viewHi - viewLo)) * (W - 2 * margin);

      // Histogram once (shared shape)
      const bins = 60;
      const counts = new Uint32Array(bins);
      for (const v of data) {
        const b = Math.floor(((v - viewLo) / (viewHi - viewLo)) * bins);
        if (b >= 0 && b < bins) counts[b]++;
      }
      let cmax = 1; for (const c of counts) if (c > cmax) cmax = c;
      const barW = (W - 2 * margin) / bins;

      function panel(yTop, label, levels, color, used) {
        const yBase = yTop + halfH;
        // axis
        ctx.strokeStyle = '#1A1A1A'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(margin, yBase); ctx.lineTo(W - margin, yBase); ctx.stroke();
        // hist
        ctx.fillStyle = '#FFD700'; ctx.globalAlpha = 0.75;
        for (let i = 0; i < bins; i++) {
          const h = (counts[i] / cmax) * halfH;
          ctx.fillRect(margin + i * barW, yBase - h, barW - 1, h);
        }
        ctx.globalAlpha = 1;
        // levels
        const drawStep = Math.max(1, Math.floor(levels.length / 200));
        for (let i = 0; i < levels.length; i += drawStep) {
          const x = xOf(levels[i]);
          const isUsed = used.has(i - (color === '#00A8A8' ? 0 : qmax));
          ctx.strokeStyle = color;
          ctx.lineWidth = isUsed ? 1.5 : 0.8;
          ctx.globalAlpha = isUsed ? 0.9 : 0.35;
          ctx.beginPath();
          ctx.moveTo(x, yBase - 6); ctx.lineTo(x, yBase + 6); ctx.stroke();
        }
        ctx.globalAlpha = 1;
        // zero marker
        const xz = xOf(0);
        ctx.strokeStyle = '#1A1A1A'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(xz, yTop); ctx.lineTo(xz, yBase + 10); ctx.stroke();
        ctx.setLineDash([]);
        // label
        ctx.fillStyle = '#1A1A1A';
        ctx.font = 'bold 12px "Space Grotesk", sans-serif';
        ctx.fillText(label, margin, yTop + 12);
      }

      // Build level arrays
      const levelsArrA = [];
      for (let q = -qmax; q <= qmax; q++) levelsArrA.push(q * sA);
      const levelsArrZ = [];
      for (let q = 0; q <= uqmax; q++) levelsArrZ.push(sZ * (q - z));

      panel(margin, `absmax (1 param: scale=${sA.toExponential(2)})  →  ${usedA.size}/${levelsA} codes used`,
            levelsArrA, '#FF007F', usedA);
      panel(2 * margin + halfH, `zero-point (2 params: scale=${sZ.toExponential(2)}, z=${z})  →  ${usedZ.size}/${uqmax + 1} codes used`,
            levelsArrZ, '#00A8A8', usedZ);

      readout.textContent = [
        `data range = [${mn.toFixed(3)}, ${mx.toFixed(3)}]    absmax = ${am.toFixed(3)}`,
        `absmax      step = ${sA.toExponential(2)}    RMSE = ${rmseA.toExponential(2)}    codes used = ${usedA.size} / ${levelsA}`,
        `zero-point  step = ${sZ.toExponential(2)}    RMSE = ${rmseZ.toExponential(2)}    codes used = ${usedZ.size} / ${uqmax + 1}`,
        `→ zero-point is ${(rmseA / rmseZ).toFixed(2)}× lower RMSE on this distribution`
      ].join('\n');
    }
    [meanIn, stdIn, bitsIn].forEach(el => el.addEventListener('input', draw));
    reluIn.addEventListener('change', draw);
    draw();
  }
  document.querySelectorAll('.zp-widget').forEach(init);
})();
</script>

With "Clip negative" enabled and the mean dragged to $1.5$, the absmax grid is using maybe half of its codes; the zero-point grid is using essentially all of them. The error ratio in the readout is the bit you save: roughly **2× lower RMSE for one extra parameter per tensor**.

## The Hidden Cost: One Extra Add

Nothing in this world is free. The cost of zero-point's flexibility shows up in the *matrix multiplication* itself. To compute $Y = X W^\top$ in integer arithmetic with both operands stored as $(q, s, z)$ pairs, you need to expand:

$$
X_{ij} W_{kj} = s_X s_W (q^X_{ij} - z_X)(q^W_{kj} - z_W).
$$

Expanding that product gives four terms:

$$
q^X_{ij} q^W_{kj} \;-\; z_W\, q^X_{ij} \;-\; z_X\, q^W_{kj} \;+\; z_X z_W.
$$

The first term is the integer matmul you want. The next two are **per-element correction terms** that have to be subtracted out. The last is a constant. In gemmlowp's optimized kernels, the correction terms collapse into per-row and per-column sums you precompute, but they exist. Symmetric absmax (where $z_X = z_W = 0$) annihilates these entire correction terms — you get $q^X_{ij} q^W_{kj}$ for free, with no extra arithmetic. This is exactly the trade.

On a 2017 mobile CPU with a thin INT8 dot-product unit, the corrections were a meaningful fraction of the runtime cost. On a 2024 datacenter GPU with bottomless INT8 tensor cores, they are nothing. As hardware got better, the cost of the asymmetric scheme evaporated, and zero-point became viable for *both* weights and activations — though, in practice, modern LLM stacks still prefer symmetric absmax for **weights** (they are roughly Gaussian, so symmetry helps) and zero-point for **activations** (often one-sided, so asymmetry helps).

## When Each One Wins

A quick decision table — internalize this; you will revisit it constantly.

| Tensor source | Distribution shape | Use |
|---|---|---|
| Linear-layer weights (Gaussian-ish, symmetric) | Symmetric around 0 | **Absmax** |
| {{< wiki "activations" >}}Post-ReLU activations{{< /wiki >}} | One-sided $[0, +\infty)$ | **Zero-point** |
| Post-GELU activations | Nearly one-sided, slight negative tail | **Zero-point** |
| {{< wiki "normalization" >}}Post-LayerNorm activations{{< /wiki >}} | Approximately Gaussian (centered) | **Absmax** |
| {{< wiki "attention" >}}Attention{{< /wiki >}} scores (pre-{{< wiki "softmax" >}}softmax{{< /wiki >}}) | Roughly symmetric, heavy-tailed | **Absmax**, but watch the [outliers](../06-outliers/) |
| {{< wiki "embeddings" >}}Embedding{{< /wiki >}} tables | Per-row distributions vary wildly | **Per-row zero-point** |
| {{< wiki "kv-cache" >}}KV cache{{< /wiki >}} (post-softmax × V) | Per-channel, often skewed | Per-row [zero-point](../10-kv-cache/) |

A modern quantization framework — [LLM.int8](../06-outliers/), TensorRT, ONNX Runtime — does not pick one and stick with it. It picks **per tensor** based on the distribution shape.

## What This Buys You In Bits

A back-of-envelope answer to "how much does asymmetry actually help?"

For a Gaussian centered at $\mu$ with standard deviation $\sigma$, absmax sets the scale to $|\mu| + k\sigma$ for some $k$ that depends on the tail; zero-point sets it to the *empirical range* $\beta - \alpha \approx 6\sigma$. So:

$$
\frac{\text{absmax step}}{\text{zero-point step}} \;\approx\; \frac{|\mu| + k\sigma}{6\sigma}.
$$

When $|\mu| \gg \sigma$ — a strongly off-center distribution — this ratio grows linearly with $|\mu|/\sigma$. For a post-ReLU activation with $\mu \approx 2$ and $\sigma \approx 0.5$, you save a factor of *roughly four* in step size, or **two effective bits**. That is enormous. For a centered weight tensor with $\mu \approx 0$, you save *nothing*. The asymmetric format is optional precisely where it matters least, and mandatory where the data is most skewed.

This is the deep reason every mobile inference runtime defaults to zero-point for activations and absmax for weights. The hardware does not pick the format; the *shape of the data* picks the format, and the format follows.

**Continue to** → [Geometry Of Weights](../05-geometry-of-weights/) for what real LLM weight distributions actually look like, or jump ahead to → [The 1% That Ruins Everything](../06-outliers/) to see what happens when neither absmax nor zero-point can save you.
