---
title: "Absmax — The One-Line Quantizer"
description: "The simplest quantization scheme that works: find the biggest number, scale, round. Three lines of numpy that became the default 8-bit baseline of the LLM era."
topics: [quantization, number-formats]
tags: [absmax, int8, scaling, baseline]
theme: cream
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 25
techKind: primer
techNode: absmax
header: default.webp
---

## The Boring Champion

Before the clever methods — before [Lloyd-Max](../03-lloyd-max/), before [GPTQ](../08-brain-surgery/), before [LLM.int8()](../06-outliers/) — there is a three-line algorithm that does most of what quantization needs to do. It has no parameters. It has no calibration. It has no nuance. It is so dumb it does not even know there is such a thing as an outlier. And yet **every modern LLM quantization paper benchmarks against it** because if you cannot beat absmax, you have nothing.

The algorithm has a name that sounds like a brand of acne cream. **Absmax** (short for *absolute maximum*) quantization. Here it is in its entirety:

```python
def absmax_quantize(x, bits=8):
    qmax = 2**(bits-1) - 1          # 127 for INT8
    scale = qmax / np.abs(x).max()  # one number per tensor
    return np.round(x * scale).astype(np.int8), scale

def absmax_dequantize(q, scale):
    return q.astype(np.float32) / scale
```

That is it. That is the algorithm. It fits on a business card. Now let us understand exactly what it gives you and exactly where it fails — because the failure mode is what motivates **every** improvement that came after.

## The Geometry

Picture a number line in the original `float32` range. Suppose the values in your tensor span $[-s, s]$ where $s = \max_i |x_i|$. The integer grid you target spans $[-127, +127]$ for signed INT8. Absmax stretches the float interval like a rubber band onto the integer interval. Every float gets snapped to the nearest integer code.

$$
q_i = \mathrm{round}\!\left(\frac{127}{s} \cdot x_i\right), \qquad
\hat x_i = q_i \cdot \frac{s}{127}.
$$

Two things follow from this geometry instantly:

1. **Zero is preserved exactly.** Because the map is linear and symmetric, the float $0$ goes to integer $0$ and back to float $0$. No bias is introduced at the origin.
2. **The grid is uniform.** The reconstruction levels are evenly spaced by a step of $\Delta = s / 127$. Every representable value is $\Delta$ apart from its neighbours.

That second property is both absmax's blessing and its curse. Uniform spacing is the simplest possible thing the hardware can do — a single multiply-add, no lookup table, no branching. It also means we are **wasting bits** when the data is non-uniform, which it always is. We will quantify that waste below.

## Worked Example

Take the toy vector from the Hugging Face blog post that introduced this method to a wider audience:

$$
\mathbf{x} = [\,1.2,\ -0.5,\ -4.3,\ 1.2,\ -3.1,\ 0.8,\ 2.4,\ 5.4\,]
$$

Absmax is $|\!-\!4.3|, |1.2|, |5.4|, \ldots = 5.4$. The scale factor is $127 / 5.4 = 23.519$. Multiply through:

$$
\mathbf{x} \cdot 23.519 = [28.2,\ -11.8,\ -101.1,\ 28.2,\ -72.9,\ 18.8,\ 56.4,\ 127.0]
$$

Round to the nearest integer:

$$
\mathbf{q} = [28,\ -12,\ -101,\ 28,\ -73,\ 19,\ 56,\ 127]
$$

Dequantize by dividing each integer by $23.519$:

$$
\hat{\mathbf{x}} = [1.190,\ -0.510,\ -4.295,\ 1.190,\ -3.103,\ 0.808,\ 2.381,\ 5.400]
$$

The biggest reconstruction error is on $2.4 \to 2.381$, off by $0.019$. The worst-case error of any well-designed uniform quantizer is **half a step**, $\Delta/2$. Here $\Delta/2 = 5.4 / (2 \cdot 127) \approx 0.021$. Our worst observed error sits right at that ceiling. Predicted and measured agree to the third decimal.

## Try It Yourself

Drag the slider to change the absmax of the input. The grid below shows where the 256 INT8 codes land on the original number line. Watch what happens when one outlier ten times larger than everything else sneaks in.

<div class="absmax-widget" data-widget="absmax">
  <div class="absmax-widget__controls">
    <label class="absmax-widget__label">
      <span>Bulk std (the typical "normal" weight)</span>
      <input type="range" min="0.01" max="2" step="0.01" value="0.3" data-control="sigma">
      <output data-output="sigma">0.30</output>
    </label>
    <label class="absmax-widget__label">
      <span>Outlier magnitude (a single big value)</span>
      <input type="range" min="0" max="200" step="1" value="0" data-control="outlier">
      <output data-output="outlier">0</output>
    </label>
    <label class="absmax-widget__label">
      <span>Bit width</span>
      <select data-control="bits">
        <option value="4">INT4 (16 levels)</option>
        <option value="6">INT6 (64 levels)</option>
        <option value="8" selected>INT8 (256 levels)</option>
      </select>
    </label>
  </div>
  <canvas data-canvas width="900" height="320"></canvas>
  <div class="absmax-widget__readout" data-readout></div>
</div>

<style>
.absmax-widget {
  border: 3px solid #1A1A1A;
  background: #FDF5E6;
  box-shadow: 4px 4px 0 #1A1A1A;
  padding: 1rem 1.1rem 1.2rem;
  margin: 1.5rem 0;
  font-family: 'Space Grotesk', system-ui, sans-serif;
}
.absmax-widget__controls {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.8rem 1rem;
  margin-bottom: 0.6rem;
}
@media (max-width: 600px) {
  .absmax-widget__controls { grid-template-columns: 1fr; }
}
.absmax-widget__label {
  display: flex;
  flex-direction: column;
  font-size: 0.85rem;
  font-weight: 600;
  gap: 0.25rem;
}
.absmax-widget__label input[type="range"] {
  width: 100%;
  accent-color: #FF007F;
}
.absmax-widget__label select {
  border: 2px solid #1A1A1A;
  padding: 0.2rem 0.4rem;
  background: #fff;
  font: inherit;
}
.absmax-widget__label output {
  font-variant-numeric: tabular-nums;
  font-size: 0.85rem;
  font-weight: 700;
  color: #FF007F;
}
.absmax-widget canvas {
  display: block;
  width: 100%;
  height: auto;
  background: #fff;
  border: 2px solid #1A1A1A;
}
.absmax-widget__readout {
  margin-top: 0.6rem;
  font-size: 0.85rem;
  font-variant-numeric: tabular-nums;
  line-height: 1.6;
  background: #1A1A1A;
  color: #FDF5E6;
  padding: 0.5rem 0.8rem;
  white-space: pre-wrap;
  font-family: 'Space Mono', ui-monospace, monospace;
}
</style>
<script>
(function() {
  const root = document.currentScript.previousElementSibling.previousElementSibling
    ? document.querySelector('.absmax-widget:not([data-bound])')
    : null;
  function init(widget) {
    if (!widget || widget.dataset.bound) return;
    widget.dataset.bound = '1';
    const canvas = widget.querySelector('[data-canvas]');
    const ctx    = canvas.getContext('2d');
    const sigmaIn = widget.querySelector('[data-control="sigma"]');
    const outIn   = widget.querySelector('[data-control="outlier"]');
    const bitsIn  = widget.querySelector('[data-control="bits"]');
    const sigmaO  = widget.querySelector('[data-output="sigma"]');
    const outO    = widget.querySelector('[data-output="outlier"]');
    const readout = widget.querySelector('[data-readout]');

    // Pre-generate a fixed bulk sample so the picture is stable as you slide.
    function gaussian(n, seed) {
      let s = seed; const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        s = (s * 1664525 + 1013904223) >>> 0;
        const u1 = (s + 1) / 4294967296;
        s = (s * 1664525 + 1013904223) >>> 0;
        const u2 = (s + 1) / 4294967296;
        out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      return out;
    }
    const bulkBase = gaussian(800, 42);

    function draw() {
      const W = canvas.width, H = canvas.height;
      const sigma   = parseFloat(sigmaIn.value);
      const outlier = parseFloat(outIn.value);
      const bits    = parseInt(bitsIn.value, 10);
      const qmax    = (1 << (bits - 1)) - 1;
      const levels  = qmax * 2 + 1;
      sigmaO.value = sigma.toFixed(2);
      outO.value   = outlier.toFixed(0);

      const bulk = new Float32Array(bulkBase.length);
      for (let i = 0; i < bulk.length; i++) bulk[i] = bulkBase[i] * sigma;
      const all  = outlier > 0 ? Float32Array.from([...bulk, outlier]) : bulk;

      let absmax = 0;
      for (const v of all) { const a = Math.abs(v); if (a > absmax) absmax = a; }
      if (absmax < 1e-6) absmax = 1e-6;
      const scale = qmax / absmax;
      const step  = absmax / qmax;

      // Quant error on bulk
      let sse = 0, maxErr = 0;
      for (const v of bulk) {
        const q = Math.max(-qmax, Math.min(qmax, Math.round(v * scale)));
        const r = q / scale;
        const e = v - r;
        sse += e * e;
        if (Math.abs(e) > maxErr) maxErr = Math.abs(e);
      }
      const rmse = Math.sqrt(sse / bulk.length);

      // Effective unique codes used by bulk
      const used = new Set();
      for (const v of bulk) {
        used.add(Math.max(-qmax, Math.min(qmax, Math.round(v * scale))));
      }

      // Draw
      ctx.clearRect(0, 0, W, H);
      // axes
      const cy = Math.round(H * 0.55);
      ctx.strokeStyle = '#1A1A1A';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(20, cy); ctx.lineTo(W - 20, cy); ctx.stroke();

      // mapping
      const span = absmax * 1.05;
      const xOf  = (v) => 20 + ((v + span) / (2 * span)) * (W - 40);

      // INT codes as tick marks (limit visual density)
      const drawStep = Math.max(1, Math.floor(levels / 200));
      ctx.strokeStyle = '#00A8A8';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.55;
      for (let q = -qmax; q <= qmax; q += drawStep) {
        const x = xOf(q / scale);
        ctx.beginPath(); ctx.moveTo(x, cy - 8); ctx.lineTo(x, cy + 8); ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // Bulk distribution as a faint histogram above the axis
      const binCount = 60;
      const bins = new Uint32Array(binCount);
      const lo = -span, hi = span;
      for (const v of bulk) {
        const b = Math.floor(((v - lo) / (hi - lo)) * binCount);
        if (b >= 0 && b < binCount) bins[b]++;
      }
      let maxBin = 1; for (const b of bins) if (b > maxBin) maxBin = b;
      ctx.fillStyle = '#FF007F';
      ctx.globalAlpha = 0.65;
      const histH = cy - 30;
      const barW  = (W - 40) / binCount;
      for (let i = 0; i < binCount; i++) {
        const h = (bins[i] / maxBin) * histH;
        ctx.fillRect(20 + i * barW, cy - h, barW - 1, h);
      }
      ctx.globalAlpha = 1;

      // Outlier marker
      if (outlier > 0) {
        const x = xOf(outlier);
        ctx.strokeStyle = '#FF8C00';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, cy); ctx.stroke();
        ctx.fillStyle = '#FF8C00';
        ctx.font = 'bold 12px "Space Grotesk", sans-serif';
        ctx.fillText('outlier', x + 6, 20);
      }

      // x-axis labels
      ctx.fillStyle = '#1A1A1A';
      ctx.font = '11px "Space Mono", monospace';
      ctx.fillText((-absmax).toFixed(2), 20, cy + 28);
      ctx.fillText('0', W / 2 - 4, cy + 28);
      ctx.fillText(absmax.toFixed(2), W - 60, cy + 28);
      ctx.font = '11px "Space Grotesk", sans-serif';
      ctx.fillText('INT codes (teal ticks)', 20, H - 22);
      ctx.fillText('bulk distribution (pink)', 20, H - 8);

      // Order-of-magnitude readout
      const wasted = 1 - used.size / levels;
      const lines = [
        `absmax = ${absmax.toFixed(2)}     step Δ = ${step.toExponential(2)}     ${levels} codes total`,
        `bulk values within ±${(3*sigma).toFixed(2)} reach only ≈${(3*sigma/step).toFixed(1)} codes`,
        `bulk RMSE = ${rmse.toExponential(2)}   |   max bulk error = ${maxErr.toExponential(2)}   |   theoretical Δ/2 = ${(step/2).toExponential(2)}`,
        `codes actually used by bulk: ${used.size} / ${levels}  (${(100*wasted).toFixed(1)}% wasted)`
      ];
      readout.textContent = lines.join('\n');
    }
    [sigmaIn, outIn, bitsIn].forEach(el => el.addEventListener('input', draw));
    draw();
  }
  document.querySelectorAll('.absmax-widget').forEach(init);
})();
</script>

The takeaway is uncomfortable. With the bulk standard deviation set to $\sigma = 0.3$ and an outlier of $30$, the bulk hits only **two or three** of the 256 available INT8 codes. Two hundred and fifty-three codes sit unused in the empty stretch between $\pm 0.3$ and $\pm 30$. We are paying eight bits per weight to use roughly **one bit's worth** of resolution where it matters.

## The Order-of-Magnitude View

For a Gaussian-distributed tensor with standard deviation $\sigma$ and absmax $s$, the number of INT codes that the **bulk** of values (say within $\pm 3\sigma$) actually touches is approximately:

$$
N_{\text{used}} \approx \frac{6 \sigma}{s / 127} = \frac{6 \sigma \cdot 127}{s}
$$

A handful of regimes:

| Scenario | $\sigma$ | $s$ | Bulk codes used | Wasted INT8 codes |
|---|---|---|---|---|
| Well-behaved layer (no outlier) | $0.1$ | $0.4$ | $\approx 190$ | $\approx 25\%$ |
| Mild outlier (10× bulk) | $0.1$ | $3.0$ | $\approx 25$ | $\approx 90\%$ |
| OPT-6.7B-style outlier (50× bulk) | $0.1$ | $15$ | $\approx 5$ | $\approx 98\%$ |
| GPT-3 175B-scale outlier (200× bulk) | $0.1$ | $60$ | $\approx 1$ | $\approx 99.6\%$ |

Read the last row carefully. With one outlier of magnitude 60 in a tensor of bulk-$\sigma$ 0.1, **the absmax INT8 quantizer reduces the bulk to a single bit of information** — sign-only. Every dot product the layer computes is now powered by ternary $\{-1, 0, +1\}$ arithmetic on what used to be a learned distribution. The transformer collapses. This is exactly the wall Dettmers hit in summer 2022, the wall we tell the story of in [The 1% That Ruins Everything](../06-outliers/).

## Why Anyone Uses It Anyway

Knowing all this, why is absmax still the universal baseline? Three reasons.

1. **Symmetric range matches the hardware**. INT8 multiply-accumulate units expect signed integers in $[-128, +127]$. The symmetric absmax map maps cleanly. A more flexible asymmetric map ([zero-point quantization](../02b-zero-point/)) needs an extra add per element, which used to matter when the math units were thin.
2. **Zero is exact**. In neural networks the scalar value $0$ appears constantly — ReLU outputs, padding tokens, masked positions, pruned weights, attention masks. Any quantizer that introduces a bias at zero will silently corrupt all of these. Absmax preserves zero by construction. (Asymmetric schemes have to work harder to do this — see [Zero-Point Quantization](../02b-zero-point/).)
3. **It has no parameters to calibrate**. No data needed, no offline pass. A model is loaded, scale is computed in one CUDA reduction, weights are cast. The latency budget of a deployment script can absorb this.

That third property is why even [LLM.int8()](../06-outliers/) uses absmax internally — applied **per row of the activation matrix and per column of the weight matrix** rather than per tensor, but absmax nonetheless. The genius of LLM.int8 is not abandoning absmax. The genius is **isolating the outliers so absmax can do its job on the rest**.

## The Granularity Knob

The single most important practical decision when using absmax is: *how much data do you compute one absmax over?* This is called the **granularity** of the quantization.

```pyplot {id="absmax-granularity" caption="Same data, four granularities. Per-tensor absmax is dominated by the single outlier and wastes the dynamic range; per-block absmax (block=32) recovers most of it."}
np.random.seed(0)
n = 1024
# Most values small, one ugly outlier.
x = np.random.randn(n) * 0.15
x[256] = 12.0   # one outlier in block #8 if blocks of 32

def absmax_q(v, qmax=127):
    s = np.abs(v).max()
    if s < 1e-12:
        return v.copy()
    q = np.round(v / s * qmax)
    return q * s / qmax

# Per-tensor
recon_tensor = absmax_q(x)

# Per-row (8 rows of 128)
recon_row = np.empty_like(x)
for i in range(8):
    sl = slice(i * 128, (i + 1) * 128)
    recon_row[sl] = absmax_q(x[sl])

# Per-block-128
recon_b128 = np.empty_like(x)
for i in range(8):
    sl = slice(i * 128, (i + 1) * 128)
    recon_b128[sl] = absmax_q(x[sl])

# Per-block-32
recon_b32 = np.empty_like(x)
for i in range(n // 32):
    sl = slice(i * 32, (i + 1) * 32)
    recon_b32[sl] = absmax_q(x[sl])

fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), sharex=True, sharey=True)
configs = [
    (recon_tensor, "per-tensor (1 scale)", '#FF007F'),
    (recon_row,    "per-row of 128 (8 scales)", '#00A8A8'),
    (recon_b128,   "per-block of 128 (8 scales, same here)", '#FF8C00'),
    (recon_b32,    "per-block of 32 (32 scales)", '#FFD700'),
]
for ax, (rec, title, color) in zip(axes.flat, configs):
    err = x - rec
    bulk_err = np.delete(err, 256)
    ax.plot(bulk_err, color=color, linewidth=0.5)
    ax.set_title(f"{title}\nbulk RMSE = {np.sqrt(np.mean(bulk_err**2)):.4f}")
    ax.spines[['top', 'right']].set_visible(False)
axes[1, 0].set_xlabel("index")
axes[1, 1].set_xlabel("index")
axes[0, 0].set_ylabel("error")
axes[1, 0].set_ylabel("error")
plt.tight_layout()
```

Per-tensor absmax pays one outlier-tax over the entire matrix. Per-block-32 absmax pays it only inside the one block that *contains* the outlier; the other 31 blocks quantize freely. The cost is storing one extra scale factor (an FP16 number) per block — for blocks of 32, that's an overhead of about $16/32 \cdot 32 = 0.5$ bits per weight. In modern formats (NF4, MXFP4, K-quants) this granularity is taken to its extreme: blocks as small as **16 or 32 weights**, with the scale factors themselves sometimes quantized to FP8 in another layer of compression (we cover this in [Calibration & Blocks](../11-calibration-and-blocks/)).

The lesson the field absorbed by 2023 is uncomfortable and simple. **Per-tensor absmax is a museum piece.** Real systems quantize per block.

## What Absmax Cannot Do

Absmax has three known weaknesses, and every modern method addresses at least one of them:

1. **Outliers murder the bulk** — as shown above. Fixed by mixed precision ([LLM.int8](../06-outliers/)), by migration (SmoothQuant), or by activation-aware re-scaling (AWQ). See the [Method Family Tree](../09-method-family-tree/).
2. **The grid is uniform but the data is not** — Gaussian-distributed weights spend most of their mass near zero, where uniform absmax under-allocates precision. Fixed by quantile-spaced grids ([NF4 via Lloyd-Max](../03-lloyd-max/)).
3. **The scale is symmetric, but real activation distributions are not** — post-ReLU activations are $\geq 0$; their absmax wastes half the codes on negative numbers that cannot appear. Fixed by [zero-point quantization](../02b-zero-point/).

We will meet each fix in turn. But every one of them is, structurally, a fix *on top of* absmax. The simplest algorithm in the family is also the chassis the family is built on.

**Continue to** → [Zero-Point Quantization](../02b-zero-point/) for the asymmetric cousin that recovers the wasted half of the grid.
