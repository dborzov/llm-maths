---
title: "The 1% That Ruins Everything"
description: "August 2022. Tim Dettmers quantizes OPT-175B and the model collapses. The cause is six dimensions out of 12,288 — and the fix invents the modern field."
topics: [quantization]
tags: [llm-int8, dettmers, outliers, opt-175b]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 60
techKind: mainline
techNode: outliers
header: default.webp
---

## Saskatchewan To Seattle

In August 2022, **Tim Dettmers** was a PhD student at the University of Washington, working under Luke Zettlemoyer. He had a side project that everyone he knew thought was crazy: he was going to get **OPT-175B** to run on a single GPU.

The reason this was crazy is that OPT-175B doesn't *fit* on a single GPU. As we noted in [the cold open](../01-cold-open/), at 16 bits per weight the model weighs 350 GB, and the largest single GPU you could buy in 2022 was 80 GB. Dettmers's plan was to quantize the weights to INT8 — half the bytes, half the storage, model fits. Standard procedure for CNNs since 2017.

He ran the quantization. The model output collapsed.

Not "got 3% worse on benchmarks" collapsed — *catastrophically* collapsed. Perplexity exploded by orders of magnitude. The model could no longer construct coherent sentences. It was as if quantization had erased something essential.

What followed is one of the cleanest examples of empirical detective work in modern ML. Dettmers wrote about it later in a [long blog post](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) that you should read if you can. We'll summarize the detective story here.

## What Naïve INT8 Does

Standard 8-bit quantization works like this — the full primer is at [Absmax Quantization](../02a-absmax/), but in one paragraph: for a {{< wiki "transformer-weights" >}}weight matrix{{< /wiki >}} $W$,

1. Find the largest absolute value in the matrix: $s = \max |W_{ij}|$.
2. Scale to fit in $[-127, 127]$: $\tilde{W}_{ij} = \text{round}(127 \cdot W_{ij} / s)$.
3. To dequantize: $W_{ij} \approx \tilde{W}_{ij} \cdot s / 127$.

This is **per-tensor symmetric quantization**, also known as absmax. It works fine on a CNN. It works fine on a small transformer. **On OPT-6.7B, it suddenly stops working.** The reason — as we will see below — is not a *quantization* problem in the abstract sense. It is a problem about the *distribution of activations* that emerges only when transformers cross a particular scale threshold. Below that threshold, every activation is well-behaved Gaussian noise, and absmax happily reduces it to INT8 with $\Delta/2$ error per value. Above the threshold, six out of twelve thousand hidden dimensions develop magnitudes 50× larger than everything else, and the whole scheme falls apart.

```pyplot {id="naive-quant-error" caption="Naïve per-tensor INT8 quantization on synthetic outlier-containing data. The scale is dominated by the outliers, so the bulk of values gets crushed."}
np.random.seed(42)
# 99% of values are small (Gaussian, sigma=0.1)
bulk = np.random.randn(99000) * 0.1
# 1% are huge (mean ~5, sigma=1)
outliers = np.random.randn(1000) * 1.0 + np.sign(np.random.randn(1000)) * 5
values = np.concatenate([bulk, outliers])

# Naive INT8 quantization
s = np.abs(values).max()
quantized = np.round(values / s * 127)
dequantized = quantized * s / 127
errors = values - dequantized

print(f"max value: {s:.2f}")
print(f"effective resolution (1 step) in original units: {s/127:.4f}")
print(f"bulk values have std 0.1 — that's only {0.1 / (s/127):.1f} discrete steps!")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.hist(bulk, bins=80, color='#FF007F', alpha=0.7, label='bulk (99%)', density=True)
ax1.hist(outliers, bins=30, color='#1A1A1A', alpha=0.7, label='outliers (1%)', density=True)
ax1.set_yscale('log')
ax1.set_xlabel("value")
ax1.set_ylabel("density (log)")
ax1.set_title("The data: 99% small + 1% huge")
ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)

bulk_errors = errors[:99000]
ax2.hist(bulk_errors, bins=80, color='#00A8A8', edgecolor='#1A1A1A', linewidth=0.3)
ax2.set_xlabel("quantization error (bulk values)")
ax2.set_ylabel("count")
ax2.set_title(f"Bulk error after INT8: huge — bulk std={bulk.std():.3f}, error std={bulk_errors.std():.3f}")
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
```

The outliers eat the entire dynamic range. With max-value scaling, the gap between adjacent INT8 levels is $s/127$, which for our synthetic case is around $0.06$. Most weights have magnitude around $0.1$ — barely two quantization steps away from zero. We've gone from a real number to one of three values: $-0.06$, $0$, or $+0.06$. That's not enough resolution to do anything useful.

This is the mechanism. **In the presence of outliers, max-based scaling makes the bulk of the data unrepresentable.**

### Order Of Magnitude: How Bad Is It?

Let us do the napkin math. For a Gaussian tensor with standard deviation $\sigma_\text{bulk}$ and a single rare outlier of magnitude $M$, the absmax $s \approx M$ for any reasonable $M / \sigma_\text{bulk}$. The INT8 step size $\Delta = s/127 \approx M/127$. The number of distinct integer codes that the bulk distribution actually touches — call it the **effective resolution** — is roughly

$$
R \;\approx\; \frac{6\sigma_\text{bulk}}{\Delta} \;=\; \frac{6 \cdot 127 \cdot \sigma_\text{bulk}}{M} \;=\; \frac{762 \, \sigma_\text{bulk}}{M}.
$$

Concrete regimes from real transformers:

| Model | $\sigma_\text{bulk}$ | $M$ | Effective codes | Effective bits |
|---|---|---|---|---|
| GPT-2 (125M, pre-emergence) | 0.5 | 3.0 | $\approx 127$ | 7.0 |
| GPT-Neo 1.3B | 0.3 | 8 | $\approx 29$ | 4.8 |
| OPT-6.7B (just past phase shift) | 0.2 | 18 | $\approx 8.5$ | 3.1 |
| OPT-13B | 0.15 | 50 | $\approx 2.3$ | **1.2** |
| OPT-66B | 0.1 | 95 | $\approx 0.8$ | **near zero** |

The last column is the bit you should engrave somewhere. At OPT-66B scale, per-tensor INT8 absmax delivers **less than one bit of information about the bulk activation**. The transformer is being asked to do {{< wiki "mlp-block" >}}FFN{{< /wiki >}} matmuls where 99% of the inputs are routed through what is essentially a ternary $\{-1, 0, +1\}$ representation. Of course it collapses. The question is not "why does it collapse"; the question is "why does it not collapse *worse*."

Watch the same calculation in real time. The widget below lets you slide outlier magnitude and bulk std and reads off the effective bits.

<div class="resolution-widget" data-widget="resolution">
  <div class="resolution-widget__controls">
    <label class="resolution-widget__label">
      <span>Bulk std σ</span>
      <input type="range" min="0.01" max="2" step="0.01" value="0.15" data-control="sigma">
      <output data-output="sigma">0.15</output>
    </label>
    <label class="resolution-widget__label">
      <span>Outlier magnitude M</span>
      <input type="range" min="0.5" max="200" step="0.5" value="50" data-control="M">
      <output data-output="M">50</output>
    </label>
    <label class="resolution-widget__label">
      <span>Bit width</span>
      <select data-control="bits">
        <option value="4">INT4</option>
        <option value="8" selected>INT8</option>
        <option value="16">FP16-as-int (range only)</option>
      </select>
    </label>
  </div>
  <div class="resolution-widget__bars" data-bars></div>
  <div class="resolution-widget__readout" data-readout></div>
</div>

<style>
.resolution-widget {
  border: 3px solid #1A1A1A;
  background: #007A7A;
  color: #FDF5E6;
  box-shadow: 4px 4px 0 #1A1A1A;
  padding: 1rem 1.1rem 1.2rem;
  margin: 1.5rem 0;
  font-family: 'Space Grotesk', system-ui, sans-serif;
}
.resolution-widget__controls {
  display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem 1rem;
  margin-bottom: 0.8rem;
}
@media (max-width: 600px) { .resolution-widget__controls { grid-template-columns: 1fr; } }
.resolution-widget__label {
  display: flex; flex-direction: column; gap: 0.25rem;
  font-size: 0.85rem; font-weight: 600;
}
.resolution-widget__label input[type="range"] { accent-color: #FFD700; }
.resolution-widget__label select {
  border: 2px solid #1A1A1A; padding: 0.2rem 0.4rem;
  background: #FDF5E6; color: #1A1A1A; font: inherit;
}
.resolution-widget__label output {
  font-variant-numeric: tabular-nums; font-weight: 700; color: #FFD700;
}
.resolution-widget__bars {
  display: grid; gap: 0.4rem;
  background: #1A1A1A; padding: 0.6rem;
  border: 2px solid #1A1A1A;
}
.resolution-widget__bar {
  display: grid; grid-template-columns: 12rem 1fr 6rem;
  align-items: center; gap: 0.6rem;
  font-family: 'Space Mono', monospace; font-size: 0.8rem;
}
.resolution-widget__bar-fill {
  background: linear-gradient(90deg, #FF8C00 0%, #FFD700 100%);
  height: 16px;
  border: 1px solid #FDF5E6;
  min-width: 1px;
}
.resolution-widget__readout {
  margin-top: 0.6rem;
  background: #1A1A1A; color: #FFD700;
  padding: 0.5rem 0.8rem;
  white-space: pre-wrap;
  font-family: 'Space Mono', ui-monospace, monospace;
  font-size: 0.85rem; line-height: 1.6;
}
</style>
<script>
(function() {
  function init(widget) {
    if (widget.dataset.bound) return;
    widget.dataset.bound = '1';
    const sIn = widget.querySelector('[data-control="sigma"]');
    const mIn = widget.querySelector('[data-control="M"]');
    const bIn = widget.querySelector('[data-control="bits"]');
    const sO  = widget.querySelector('[data-output="sigma"]');
    const mO  = widget.querySelector('[data-output="M"]');
    const bars= widget.querySelector('[data-bars]');
    const out = widget.querySelector('[data-readout]');

    function render() {
      const sigma = parseFloat(sIn.value);
      const M     = parseFloat(mIn.value);
      const bits  = parseInt(bIn.value, 10);
      sO.value = sigma.toFixed(2);
      mO.value = M.toFixed(1);

      const qmax = (1 << (bits - 1)) - 1;
      const levels = qmax * 2 + 1;
      const absmaxV = Math.max(M, 4 * sigma);
      const step    = absmaxV / qmax;
      const eff     = (6 * sigma) / step;
      const effBits = eff > 0 ? Math.log2(Math.max(1, eff)) : 0;

      // Scenarios
      const scenarios = [
        { label: "GPT-2 125M",            sigma: 0.5,  M: 3   },
        { label: "GPT-Neo 1.3B",          sigma: 0.3,  M: 8   },
        { label: "OPT-6.7B",              sigma: 0.2,  M: 18  },
        { label: "OPT-13B",               sigma: 0.15, M: 50  },
        { label: "OPT-66B",               sigma: 0.1,  M: 95  },
        { label: "Your slider →",         sigma,       M       },
      ];
      const rows = scenarios.map(s => {
        const am = Math.max(s.M, 4 * s.sigma);
        const st = am / qmax;
        const e  = (6 * s.sigma) / st;
        const eb = Math.log2(Math.max(1, e));
        const w  = Math.min(100, eb / bits * 100);
        return `<div class="resolution-widget__bar">
            <div>${s.label}</div>
            <div><div class="resolution-widget__bar-fill" style="width:${w}%"></div></div>
            <div>${eb.toFixed(2)} bit</div>
          </div>`;
      });
      bars.innerHTML = rows.join('');

      const codesUsed = Math.max(1, Math.round(eff));
      out.textContent = [
        `Stored bits per weight  = ${bits}`,
        `INT step size Δ          = absmax / ${qmax} = ${step.toExponential(2)}`,
        `Bulk reaches ±3σ = ±${(3*sigma).toFixed(2)}, which spans ${eff.toFixed(2)} integer codes`,
        `Effective bits delivered to bulk = log₂(${eff.toFixed(2)}) ≈ ${effBits.toFixed(2)}`,
        `→ At outlier M=${M.toFixed(1)} with σ=${sigma.toFixed(2)}, you store ${bits} bits and use ${effBits.toFixed(2)}. ${(bits - effBits).toFixed(1)} bits per weight wasted.`,
      ].join('\n');
    }
    [sIn, mIn, bIn].forEach(el => el.addEventListener('input', render));
    render();
  }
  document.querySelectorAll('.resolution-widget').forEach(init);
})();
</script>

The takeaway: **the effective bits delivered to your bulk activations is not the number of bits you store.** It is the number of bits you store minus the bits stolen by the outliers. For an OPT-13B-scale outlier ($M \approx 50$, $\sigma \approx 0.15$), you store eight bits and you deliver one. Seven bits per weight are wasted, every layer, every forward pass. The mystery is not the model collapsing — the mystery is anything still being computable at all.

## What Dettmers Saw

So Dettmers ran a series of measurement experiments. He instrumented OPT models at every layer, looking at the activation magnitudes feeding into each linear layer.

He found something that nobody had reported before. In OPT-6.7B and larger, **certain specific feature dimensions** consistently had activations 20-100× larger than the rest. *The same dimensions*, batch after batch. These dimensions weren't noise — they were systematic features the model had learned. Below the 6.7B parameter scale, this phenomenon did not appear; the activations looked uniformly Gaussian-ish across all dimensions.

Then a sharper observation: when these "outlier dimensions" appear, they appear *suddenly*, as the model crosses a scale threshold. They were an **emergent phenomenon** of large-scale training.

He recreated the picture from [Geometry Of Weights](../05-geometry-of-weights/) with real numbers. Plotted with care, the histogram of OPT-13B's activations has a fat dense peak near zero — and seven thin spikes way out in the tails, each corresponding to a single hidden dimension hitting magnitude ±50.

The natural follow-up question was: what happens if we just *delete* those outlier dimensions?

The answer: **everything collapses**. Zeroing out the 7 outlier dimensions (out of 5120) caused OPT-13B's perplexity on Wikipedia to explode by 40x. Ablate-on-substantive-things-only: the outlier dimensions weren't noise; they were carrying load-bearing computation.

## The First Working Method: LLM.int8()

So Dettmers needed a quantization scheme that (a) didn't crush the bulk values and (b) didn't damage the outlier dimensions. His solution was a **mixed-precision decomposition**, and it has the bone-clean structure of a really good idea.

For each linear layer $Y = X W^\top$:

1. **Identify outlier columns of $X$**. Use a simple threshold: any column whose absolute max exceeds $\alpha$ (typically 6).
2. **Split $X$ into two parts**: $X = X_{\text{regular}} + X_{\text{outlier}}$, where $X_{\text{outlier}}$ keeps only the outlier columns and zeros out the rest.
3. **Compute the matmul in two pieces**:
   - $Y_{\text{regular}} = X_{\text{regular}} W^\top_{\text{regular}}$ — done in **INT8**, using vector-wise scaling (one scale per row of $X$, one per column of $W$).
   - $Y_{\text{outlier}} = X_{\text{outlier}} W^\top_{\text{outlier}}$ — done in **FP16**, because the outliers refuse to be quantized cleanly.
4. **Sum them**: $Y = Y_{\text{regular}} + Y_{\text{outlier}}$.

```pyplot {id="llm-int8-decomp" caption="LLM.int8() splits activations into a regular path (quantized) and an outlier path (kept in FP16). Combined output is mathematically equivalent up to quantization error."}
np.random.seed(0)
hidden = 16
batch = 8
# Activations: most dims small, 2 dims with huge outliers
X = np.random.randn(batch, hidden) * 0.5
outlier_dims = [3, 11]
X[:, outlier_dims] *= 60

# Outlier mask
threshold = 6.0
outlier_cols = np.abs(X).max(axis=0) > threshold

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, data, title, vmax in [
    (axes[0], X, "Original X", 60),
    (axes[1], X * ~outlier_cols, "X_regular (INT8 path)", 6),
    (axes[2], X * outlier_cols, "X_outlier (FP16 path)", 60),
]:
    im = ax.imshow(np.abs(data), aspect='auto', cmap='magma', vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("hidden dim")
    if ax == axes[0]:
        ax.set_ylabel("batch row")
plt.tight_layout()

print(f"outlier dim indices: {np.where(outlier_cols)[0]}")
print(f"fraction of X that is outlier: {outlier_cols.mean():.1%}")
```

Two things to notice:

- **The outlier columns are a tiny fraction** — typically <0.1% of all activations, often just 6–10 specific dimensions out of thousands. The FP16 path is short and cheap.
- **The regular path is now quantization-friendly**: with the outliers carved out, the bulk distribution is well-behaved enough that vector-wise INT8 (one scale per row of $X$, one per column of $W$) gives essentially lossless reconstruction.

The whole layer's output is **mathematically equivalent** to a normal matmul, up to the INT8 quantization error on the regular path — which is small because we removed what was making it large.

## Why This Was The Move

There were several earlier attempts to quantize LLMs that *didn't* work, and each one fails for a reason that becomes obvious once you see LLM.int8().

- **"Just use INT8 everywhere."** Fails because outliers eat the dynamic range.
- **"Just use FP16 everywhere."** Doesn't save memory (still 2 bytes/weight).
- **"Clip the outliers."** Destroys load-bearing features.
- **"Re-train the model to not have outliers."** Tried by several groups; impossibly expensive at 175B scale, and the outliers come back during training anyway (they're *useful*).

LLM.int8 sidesteps all of these. It says: stop trying to make outliers go away. Build a system that *handles them with the right precision*, leaves them alone, and quantizes everything else.

```python
# What it looks like to use it (real code with the bitsandbytes library)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-175b",
    quantization_config=bnb_config,
    device_map="auto",
)
# At this point OPT-175B is loaded in INT8 across however many GPUs you have,
# with vector-wise quantization for the bulk and FP16 for the outlier columns.
```

Two lines of code. OPT-175B on a single 8-GPU node, where it used to need eight. By the end of 2022 the `bitsandbytes` library had hundreds of thousands of downloads and the field had shifted irrevocably.

## What LLM.int8() Quietly Established

Reading the LLM.int8 paper now, with hindsight, three things stand out as foundational moves that the rest of the field built on:

1. **Quantize at a finer granularity than the whole tensor.** Vector-wise scaling — one scale per row of $X$, one per column of $W$ — is a free precision win because real LLM weight rows have different magnitudes. (Per-row/per-column scales become per-block scales in later methods; we revisit this in [Calibration & Blocks](../11-calibration-and-blocks/).)

2. **Mixed-precision is acceptable.** Before LLM.int8, the field had an implicit aesthetic that "real" quantization meant "every weight is in the same low-bit format." Dettmers showed that running 0.1% of your activations in FP16 and the rest in INT8 produces a model that is **both** memory-efficient *and* lossless — and there's no shame in the asymmetry.

3. **The right unit of analysis is the activation, not the weight.** Older quantization papers focused on quantizing weights. Dettmers showed that for LLMs, the *activations* are what break things first. Weights are usually well-behaved; activations are the wild ones.

This last point is the one to internalize. Many later methods come back to it. **SmoothQuant** (Xiao et al., 2022) takes it even further: rather than handling outliers separately, it *mathematically migrates* the outlier magnitudes from activations into weights, where they can be quantized more easily. AWQ uses activation magnitudes to decide which weight columns to *protect* during quantization. Every successor method, in some way, builds on the insight that the activations tell you where to be careful.

## The Catch: LLM.int8() Is Slow

The mixed-precision decomposition is mathematically beautiful but *operationally awkward*. You're doing a matmul in two paths and a separate GEMM kernel for the outlier path. The overhead of the column-splitting, the precision conversion, and the synchronization adds up. Inference with LLM.int8 was typically **3-4× slower** per token than pure FP16.

This was the trade: you got OPT-175B onto a single node, at the cost of being slow on it. For a long time the joke was "LLM.int8 turns 8 GPUs into 1 slow GPU."

The next-generation methods — **GPTQ, AWQ, NF4** — would all attempt to recover the speed *without* losing the precision gains. They take a different angle: rather than handle outliers with a runtime mixed-precision split, they do the harder work of **quantizing weights more cleverly** during a one-time calibration step. That story needs second-order math — the kind that powers [Brain Surgery Returns](../08-brain-surgery/) — so we'll detour for a primer on Hessians first.

If you would like the **inside-the-algorithm** version of this chapter — every step of the LLM.int8 matmul written out, the per-row / per-column scale arithmetic, the threshold-tuning experiments, and an interactive widget that runs the whole pipeline in your browser — read [Inside LLM.int8()](../06b-llm-int8-deep/) before moving on. It is the implementation companion to this detective story.

**Continue to** → [Inside LLM.int8()](../06b-llm-int8-deep/) for the algorithm autopsy, or skip ahead to → [Taylor & Hessians](../07-taylor-and-hessians/) for the mathematical tools that power the next generation.
