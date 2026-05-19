---
title: "KV Distributions: outlier channels in K, outlier tokens in V"
short_title: "KV Distributions"
description: "Three independent research groups in early 2023 pointed microscopes at the KV cache and found the same thing: K has persistent outlier feature channels, V has persistent outlier sequence positions, and the BOS token absorbs a wildly disproportionate share of attention."
blurb:
  - "K outliers: the same feature dimensions, every input, every layer, with magnitudes 10–50× the bulk."
  - "V outliers: the same sequence positions across heads — concentrated at the BOS token and a few early positions."
  - "Guangxuan Xiao's team at MIT names the BOS pattern 'attention sinks'. Mingjie Sun's team at CMU finds the residual-stream version: 'massive activations'."
  - "The asymmetry (per-channel K, per-token V) is the empirical fact that drives every method in the KV family tree."
topics: [quantization, attention, statistics]
tags: [kv-cache, kivi, attention-sinks, streamingllm, massive-activations, distribution]
theme: teal
math: true
draft: false
date: 2026-05-13T09:00:00-04:00
issue: 3
weight: 160
techKind: mainline
techNode: kv-distribution
header: 16-kv-distribution.webp
---

## Three Microscopes

You can spend a year working with transformers without ever looking *inside* the {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}. The cache is just a buffer; the model writes to it, the model reads from it, the wrapper logic feeds it back into {{< wiki "attention" >}}attention{{< /wiki >}}. From outside the box, the bits that go in and out are floating-point numbers and that is that.

Then in **early 2023**, three different research groups, working independently, point microscopes at the same buffer and start reporting strange things. **Yujun Liu's group at MIT** notices that K vectors have a few channels with persistently giant values, while V vectors have a few sequence positions with persistently giant values. **Guangxuan Xiao's team at MIT** notices that the *first token* of every sequence — the BOS token — receives a wildly disproportionate fraction of attention from later positions. **Mingjie Sun's team at CMU** notices that *some hidden states*, in the residual stream itself, develop magnitudes a thousand times larger than their neighbors.

Three observations. Three teams. They turn out to be the same observation.

{{% pullquote type="profound" author="The Quantization Slogan" %}}
Most of the KV cache is well-behaved Gaussian noise. The trouble is that **the bits that aren't noise are doing all the work.**
{{% /pullquote %}}

This article is the story of how the field assembled the picture, and why the picture matters for every method in [The KV Method Family Tree](../15-kv-method-family/). It is a detective story with three witnesses, all unreliable in different ways.

## The Boring Picture: What You'd Naïvely Expect

Open up a trained LLM, run a forward pass, and pull out the K and V tensors at, say, the 16th transformer layer. Each is shaped `[batch, num_heads, seq_len, head_dim]`. Squash a single head and you get a 2D matrix: rows are sequence positions, columns are feature dimensions.

If transformers were well-behaved Gaussian objects, this matrix would look like white noise. Roughly zero mean, roughly the same variance everywhere, no structure.

```pyplot {id="naive-kv-expectation" caption="What you'd expect K and V to look like if transformers were well-behaved. Spoiler: they aren't."}
np.random.seed(0)
seq_len, d_head = 64, 32

K_imagined = np.random.randn(seq_len, d_head) * 0.5
V_imagined = np.random.randn(seq_len, d_head) * 0.5

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, data, title in [(axes[0], K_imagined, "K (imagined)"),
                         (axes[1], V_imagined, "V (imagined)")]:
    im = ax.imshow(np.abs(data), aspect='auto', cmap='magma', vmin=0, vmax=2.0)
    ax.set_title(title)
    ax.set_xlabel("feature dim")
    ax.set_ylabel("token position")
plt.tight_layout()
```

This was the working assumption of every quantization scheme prior to 2023. The naive INT8 KV cache implementation in early HuggingFace `transformers` used a per-tensor scale, and it *almost worked*: a few percent perplexity loss, easy to explain away as "well, you're throwing away half the bits."

## Witness #1 — KIVI's K-Channel / V-Token Asymmetry

In **June 2023**, Liu, Wang, and collaborators at MIT post **KIVI** to arXiv. The paper does the simplest possible empirical study: histogram every K and V tensor across many layers and many inputs of a Llama-2 forward pass.

The histograms are not Gaussian. They are **asymmetric**. K shows a fat tail concentrated in **specific feature dimensions** — the same dimension indices, every input, every layer, with magnitudes 10–50× the bulk. V shows a fat tail concentrated in **specific sequence positions**.

{{< infographic title="The Asymmetric Grid" description="K and V outliers follow different geometric rules. Toggle to see how quantization axes align." >}}
  {{< infographic-controls >}}
    <div class="ig-toggle-row" role="group" aria-label="Tensor Type">
      <button type="button" id="toggle-k" aria-pressed="true">KEY (K)</button>
      <button type="button" id="toggle-v" aria-pressed="false">VALUE (V)</button>
    </div>
    <p class="ig-description">
      K outliers are <strong>vertical</strong>: specific channels are large for all tokens.<br><br>
      V outliers are <strong>horizontal</strong>: specific tokens are large for all channels.
    </p>
  {{< /infographic-controls >}}

  {{< infographic-viz >}}
    <div class="ig-stage">
      <div id="ig-asym-viz" style="width: 100%; max-width: 400px; margin: 0 auto;">
        <svg viewBox="0 0 200 200" class="ig-svg" id="asym-svg">
          <!-- Grid background -->
          <rect x="10" y="10" width="180" height="180" data-fill="cream" data-stroke="ink" stroke-width="2" />
          
          <!-- Key Outliers (Vertical) -->
          <g id="k-outliers">
            <rect x="40" y="10" width="15" height="180" data-fill="pink" opacity="0.8" />
            <rect x="110" y="10" width="15" height="180" data-fill="pink" opacity="0.8" />
            <rect x="160" y="10" width="15" height="180" data-fill="pink" opacity="0.8" />
          </g>

          <!-- Value Outliers (Horizontal) -->
          <g id="v-outliers" style="display:none">
            <rect x="10" y="10" width="180" height="15" data-fill="teal" opacity="0.8" />
            <rect x="10" y="80" width="180" height="15" data-fill="teal" opacity="0.8" />
            <rect x="10" y="140" width="180" height="15" data-fill="teal" opacity="0.8" />
          </g>

          <!-- Axis Labels -->
          <text x="100" y="205" text-anchor="middle" font-size="10" font-weight="bold">CHANNELS</text>
          <text x="-100" y="5" text-anchor="middle" font-size="10" font-weight="bold" transform="rotate(-90)">TOKENS</text>
        </svg>
      </div>
      <p class="ig-stage-label" id="asym-label">OPTIMAL: PER-CHANNEL QUANTIZATION</p>
    </div>

    <script>
      (function() {
        const btnK = document.getElementById('toggle-k');
        const btnV = document.getElementById('toggle-v');
        const kGrp = document.getElementById('k-outliers');
        const vGrp = document.getElementById('v-outliers');
        const label = document.getElementById('asym-label');

        btnK.addEventListener('click', () => {
          btnK.setAttribute('aria-pressed', 'true');
          btnV.setAttribute('aria-pressed', 'false');
          kGrp.style.display = 'block';
          vGrp.style.display = 'none';
          label.textContent = 'OPTIMAL: PER-CHANNEL QUANTIZATION';
        });

        btnV.addEventListener('click', () => {
          btnV.setAttribute('aria-pressed', 'true');
          btnK.setAttribute('aria-pressed', 'false');
          kGrp.style.display = 'none';
          vGrp.style.display = 'block';
          label.textContent = 'OPTIMAL: PER-TOKEN QUANTIZATION';
        });
      })();
    </script>
  {{< /infographic-viz >}}
{{< /infographic >}}

The KIVI paper's table says it precisely:

| Tensor | Axis with persistent outliers | Axis with no special structure |
|---|---|---|
| Key (**K**) | **Channel** (Feature dim) | Token |
| Value (**V**) | **Token** (Position) | Channel |

In [microGPT's listings](../../05-microgpt/16-full-forward/), the `keys[li].append(k)` and `values[li].append(v)` lines are where these shapes are born. This asymmetry is the load-bearing fact for [per-channel KV quantization](../../03-quantization/18-quantization-axes/).

## Witness #2 — StreamingLLM And The Softmax Factor

In **September 2023**, the Xiao team publishes **StreamingLLM**. They aren't looking at values; they're looking at the **attention weights** $\text{softmax}(QK^\top)$.

They find the **attention sink**: the first few tokens (usually the BOS) receive a huge fraction of attention mass even when semantically irrelevant. 

{{% marginnote %}}
**Wait — really?** Yes. In a 4K-token document about quantum mechanics, the BOS token might get 30% of the attention from token 3,500. The model is simply "parking" weight there.
{{% /marginnote %}}

{{< crosshead >}}THE MECHANISTIC ARGUMENT{{< /crosshead >}}

The reason is mathematical. In the [microGPT attention loop](../../05-microgpt/08-attention/), the `softmax(attn_logits)` function forces the weights to sum exactly to **1.0**. 
Every attention head **must** spend its full budget of attention on *something*. If a head has nothing relevant to attend to, it needs a "junk drawer" to dump its mass. The BOS token, being visible to every subsequent query, becomes that drawer.

{{< callout type="info" title="The Ghost Token" >}}
Think of the BOS as a **Ghost Token**. It doesn't carry meaning; it carries the **numerical residue** of a head that wants to stay quiet but isn't allowed to by the `softmax` sum-to-one constraint.
{{< /callout >}}

```pyplot {id="softmax-vs-softmax1" caption="Standard Softmax vs Softmax-1. Standard softmax (left) forces attention onto a sink even when all scores are low. Softmax-1 (right) allows the total attention mass to drop to near zero, providing a 'quiet' option."}
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def softmax_minus_1(x):
    e = np.exp(x)
    return e / (1 + e.sum())

# Case: All tokens are irrelevant (low scores)
low_scores = np.array([-5.0, -5.5, -6.0, -5.2])

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axes[0].bar(range(4), softmax(low_scores), color='#FF007F', edgecolor='#1A1A1A')
axes[0].set_title("Standard Softmax\n(Forces mass even on low scores)")
axes[0].set_ylabel("Attention Weight")

axes[1].bar(range(4), softmax_minus_1(low_scores), color='#00A8A8', edgecolor='#1A1A1A')
axes[1].set_title("Softmax-1\n(Allows 'Quiet Attention')")

for ax in axes:
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['BOS', 't1', 't2', 't3'])
plt.tight_layout()
```

## Witness #3 — Sun Et Al. And Dark Signals

In **April 2024**, Mingjie Sun and collaborators at CMU publish **"Massive Activations in Large Language Models"**. They point the microscope at the **{{< wiki "residual-stream" >}}residual stream{{< /wiki >}}** (the `x` variable in [microGPT](../../05-microgpt/10-residual-stream/)).

They find **Massive Activations**: specific (token, channel) entries with magnitudes 100,000× the median. These act as **Implicit Bias** terms.

{{< callout type="warning" title="Dark Signals" >}}
These outliers are often created by the {{< wiki "mlp-block" >}}Feed-Forward Networks (MLPs){{< /wiki >}} specifically to be used as sinks. Removing them breaks the model instantly.
{{< /callout >}}

1. **They are persistent:** Once an anchor cell develops (often at BOS), it stays massive through 20+ layers.
2. **Projected Outliers:** project a Massive Activation at BOS through $W_K$ and you get the **K-channel outliers**. Project it through $W_V$ and you get the **V-token outliers**.

```pyplot {id="massive-activation-cascade" caption="The Unified Theory: A single massive activation in the residual stream cascades into three downstream symptoms."}
np.random.seed(11)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')

def box(x, y, w, h, label, color='#FFD700', big=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, color=color, ec='#1A1A1A', lw=2.5))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=11 if not big else 13, fontweight='bold')

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=2))

box(0.5, 2, 2.5, 1, "Residual\nMASSIVE\nactivation", color='#FF007F', big=True)
ax.text(1.75, 3.2, "(BOS token, channel 1415)", ha='center', fontsize=8, style='italic')

arrow(3.0, 2.5, 4.5, 4.2); arrow(3.0, 2.5, 4.5, 2.5); arrow(3.0, 2.5, 4.5, 0.8)

box(4.5, 3.6, 3.0, 1.0, "K with outlier\nCHANNEL")
box(4.5, 2.0, 3.0, 1.0, "V with outlier\nTOKEN")
box(4.5, 0.4, 3.0, 1.0, "Attention\nSINK at BOS")

arrow(7.5, 4.1, 9.0, 4.1); arrow(7.5, 2.5, 9.0, 2.5); arrow(7.5, 0.9, 9.0, 0.9)

box(9.0, 3.6, 4.5, 1.0, "KIVI: per-CHANNEL K scale")
box(9.0, 2.0, 4.5, 1.0, "KIVI: per-TOKEN V scale")
box(9.0, 0.4, 4.5, 1.0, "StreamingLLM: keep BOS K/V")
```

## A Five-Year Timeline Of What We Learned

The lineage of these discoveries:

{{< timeline name="kv-distribution-discoveries" >}}

## What This Means For Quantization

1. **The asymmetric quantization axis (per-channel K, per-token V) is empirically correct because of how massive activations propagate.** Don't fight it. KIVI was right to special-case the geometry.
2. **Keep the first ~4 tokens of K and V at full precision.** They are the attention sinks. Quantizing them is asking for trouble. Almost every modern method does this.
3. **Outliers are not noise.** They are a *load-bearing feature* the model uses to manage attention. Treat them as first-class, not as values to clip.
4. **Rotation methods sidestep the asymmetry but pay a kernel cost.** A Hadamard rotation flattens out the per-channel and per-token outliers, making everything quantize uniformly — at the cost of an in-line transform during attention. See [Rotations](../17-rotations/).
5. **The story is not done.** If the field figures out how to *train without* attention sinks, the entire KV-quantization rulebook changes.

**Continue to** → [Rotations](../17-rotations/) — the geometric trick that bypasses the K/V asymmetry by spreading outliers uniformly across the whole basis. Used by the most recent KV methods (TurboQuant) and the most recent weight methods (QuaRot, SpinQuant) alike.

---

## Learning Resources

- **[Massive Activations in Large Language Models (Arxiv)](https://arxiv.org/abs/2402.17762)**  
  The "Witness #3" paper identifying the root cause in the residual stream. Essential for understanding outliers.
- **[MIT HAN Lab — StreamingLLM Project](https://hanlab.mit.edu/projects/streamingllm)**  
  Original source for the "Attention Sink" discovery with excellent visual animations.
- **[Adam Butterworth — StreamingLLM Deep Dive](https://www.adambutterworth.com/blog/streamingllm)**  
  A fantastic pedagogical breakdown of *why* the softmax constraint forces sinks to happen.
- **[Evan Miller — Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html)**  
  The influential post proposing **Softmax-1** (Quiet Attention) to fix the sink phenomenon.
- **[Neel Nanda — Mechanistic Interpretability: Attention Sinks](https://www.neelnanda.io/mechanistic-interpretability/glossary#attention-sinks)**  
  Expert "gears-level" walkthrough of how models hack their own math to create null operations.
- **[Hugging Face — KV Cache Quantization](https://huggingface.co/blog/kv-cache-quantization)**  
  The industry-standard guide on implementing KIVI and other distribution-aware schemes.
- **[Databricks — LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)**  
  Covers the "Memory Wall" and why the KV cache distribution is the primary bottleneck for serving.
- **[Nicola Cancedda — Spectral filters, dark signals, and attention sinks](https://arxiv.org/abs/2406.16743)**  
  The mid-2024 sequel explaining how MLPs generate these "dark signals" to use as sinks.
- **[Reddit r/MachineLearning — Massive Activations Discussion](https://www.reddit.com/r/MachineLearning/comments/1b1p9v6/r_massive_activations_in_large_language_models/)**  
  High-signal community discussion connecting the dots between sinks, outliers, and quantization.
- **[Umar Jamil — StreamingLLM (YouTube Walkthrough)](https://www.youtube.com/watch?v=gZ9N_4f-xIA)**  
  Line-by-line paper explanation showing how the attention mass explodes at sequence start.
