---
title: "Component Library & Style Guide"
description: "The visual building blocks every article uses — pullquotes, callouts, margin notes, timelines, infographics, and friends. Live examples and copy-paste recipes."
topics: [meta]
tags: [style-guide, components, shortcodes]
theme: cream
math: true
draft: false
unfolded: true
---

This page is the **human-facing showcase** for the almanac's component library. Every visual building block is rendered live below — same CSS, same shortcodes, same neubrutalist sensibility you'll see across the articles. Use it as a reference when you write.

The **AI-agent companion documentation** lives at [`docs/components/`](https://github.com/) in the repo. When you're prompting an LLM to write or edit an article, point it at `docs/components/README.md` first.

## The catalog at a glance

| Component | When to reach for it |
|---|---|
| **Hero image** | Top of every article — sets the mood |
| **Pullquote** | A profound conclusion, theorem, or famous figure's voice |
| **Callout** | Tangents, pro-tips, warnings, definitions |
| **Margin note** | Citation or single-sentence aside |
| **Crosshead** | Signpost mid-section, NOT a TOC entry |
| **Figure** | Third-party image with caption + lightbox |
| **Pyplot block** | Our own matplotlib plot, regenerated at build |
| **Code block** | Syntax-highlighted Python listing |
| **Timeline** | Five-event chronological strip |
| **Tech tree** | Issue-cover TOC (DAG of articles) |
| **Infographic** | Two-pane interactive — replaces a wall-of-numbers table |

## Hero image

Set via front matter, not a shortcode. Every article gets one — see the top of this page.

```yaml
---
header: lloyd-max.webp   # file lives at static/header-illustrations/<name>
---
```

**Format:** WebP, ≤1600 px wide, ≤300 KB. Aspect ~8:3.

---

## Pullquote — three moods

For a profound conclusion, a counter-intuitive result, or a famous figure's voice. **One per article max.**

{{% pullquote type="profound" author="Grace Hopper" %}}
The most dangerous phrase in the language is, "We've always done it this way."
{{% /pullquote %}}

{{% pullquote type="technical" author="Transformer 101" %}}
The KV cache scales linearly with context length but consumes GPU memory exponentially.
{{% /pullquote %}}

{{% pullquote type="counter-intuitive" author="StreamingLLM, 2023" %}}
The first four tokens of any sequence act as "attention sinks" — they absorb significant weight regardless of their semantic content.
{{% /pullquote %}}

**Recipe**

```markdown
{{%/* pullquote author="Grace Hopper" type="profound" */%}}
The most dangerous phrase in the language is, "We've always done it this way."
{{%/* /pullquote */%}}
```

Types: `profound` (pink), `technical` (teal), `counter-intuitive` (orange).

---

## Callout — five variants

Boxed sidebars for tangents, tips, warnings, definitions. Use the `%%` form so Markdown inside renders.

{{% callout type="note" %}}
A **note** is the neutral teal variant. Use it for clarifications and quiet definitions. Markdown like *italic*, **bold**, and `inline code` all work inside.
{{% /callout %}}

{{% callout type="tip" title="Pro-Tip" %}}
A **tip** uses pop pink. Practical advice — "if you're implementing this, do X" — goes here.
{{% /callout %}}

{{% callout type="warning" title="Gotcha" %}}
A **warning** uses pop orange. Reserve for sharp edges that crash quietly: "this fails silently when `n_kv_head` is set."
{{% /callout %}}

{{% callout type="tangent" title="Historical Aside" %}}
A **tangent** uses cream + ink. The term *entropy* was suggested to Shannon by Von Neumann because "no one really knows what it is anyway."
{{% /callout %}}

{{% callout type="theorem" title="Theorem 3.2" %}}
A **theorem** uses pop yellow. For formal definitions or named results: *"If the attention scores satisfy $\sum_i s_i = 1$ and $s_i \geq 0$, then the softmax output preserves convex combinations."*
{{% /callout %}}

**Recipe**

```markdown
{{%/* callout type="tip" title="Pro-Tip" */%}}
When implementing eviction, always preserve the first 4 tokens (the *attention sinks*).
{{%/* /callout */%}}
```

Types: `note` (default), `tip`, `warning`, `tangent`, `theorem`.

---

## Margin note

Floats in the right margin on wide screens; inlines on tablets and phones. Use for citations and optional asides.

We could afford to store the entire context.{{% marginnote label="memory" %}}At 32k context on Llama-3 70B the KV cache exceeds **16 GB per user** — see [Dao 2023](https://arxiv.org/abs/2307.08691).{{% /marginnote %}} But in 2024 context is the new scarcity, and the cache is the first thing to saturate.

**Recipe**

```markdown
We could afford to store the entire context.{{%/* marginnote */%}}At 32k context the KV cache exceeds **16 GB per user**.{{%/* /marginnote */%}} But in 2024 context is the new scarcity.
```

Place the shortcode right where you want the inline marker ⊕. The aside floats up to align on desktop.

---

## Crosshead

Mid-section signpost — visually distinct from `<h2>`, doesn't appear in the TOC. Drop one every 5–8 paragraphs of long prose.

{{< crosshead >}}The Heavy-Hitter Hypothesis{{< /crosshead >}}

The core intuition behind nearly all eviction algorithms is simple: not all tokens are created equal. In a sentence, the period at the end often accumulates more attention than a generic "the" in the middle.

**Recipe**

```markdown
{{</* crosshead */>}}The Heavy-Hitter Hypothesis{{</* /crosshead */>}}
```

---

## Figure

Image + caption + click-to-fullscreen lightbox. For our own plots, use a pyplot block instead.

{{< figure src="header-illustrations/lloyd-max.webp"
           alt="Lloyd–Max banner"
           caption="**Stand-in caption** — for a real figure you'd describe what the reader should notice. Markdown *inline* works." >}}

**Recipe**

```markdown
{{</* figure src="diagrams/attention-map.webp"
           alt="Heatmap of attention scores"
           caption="Attention scores over a 2048-token window. Notice the spike at the first 4 tokens." */>}}
```

---

## Pyplot block — our own plots

Reproducible matplotlib code rendered at build time. The source listing collapses under the image.

```pyplot {id="component-demo-tanh" caption="tanh — the reference S-curve"}
x = np.linspace(-4, 4, 200)
plt.figure(figsize=(7, 3.5))
plt.plot(x, np.tanh(x), color='#FF007F', linewidth=2.5)
plt.axhline(0, color='#1A1A1A', linewidth=0.8)
plt.axvline(0, color='#1A1A1A', linewidth=0.8)
plt.title("tanh", fontsize=14)
plt.grid(alpha=0.3)
```

**Strict rules**: `plt` and `np` are pre-imported; do not call `plt.show()` or `plt.savefig()`; the isolated env has only `numpy` + `matplotlib`.

---

## Code block

Standard fenced code with Chroma syntax highlighting.

```python
def h2o_evict(cache, scores, k):
    sinks = cache[:4]
    recent = cache[-k:]
    mid_scores = scores[4:-k]
    _, idx = torch.topk(mid_scores, k=32)
    return torch.cat([sinks, cache[idx], recent])
```

Use ```` ```python ```` for code that isn't executed. For runnable matplotlib, use a pyplot block instead.

---

## Timeline

Five-event chronological strip. Data lives in `data/timelines/<name>.toml`. Hard cap of 5.

{{< timeline name="quantization2019to2026" >}}

**Recipe**

```markdown
{{</* timeline name="quantization2019to2026" */>}}
```

---

## Infographic — interactive

Two-pane panel: controls on the left, live readout on the right. Use it to replace a wall-of-numbers static table with something the reader can poke.

**The basic unit is always a comparison.** Either two things side-by-side, or one thing changing in reaction to one or two inputs. A single number in isolation tells the reader nothing about scale — pair it with a baseline, a reference shape, or a second variant.

Three demos follow, each illustrating a different visualization pattern:

1. **Comparison bars** — two architectures, one slider, watch the bars race
2. **Scaled circles** — a quantity scales by area, paired with a baseline
3. **Stacked segments** — a budget split, segments reshape as you toggle

### Demo 1 — comparison bars (HBM under context-window pressure)

The point isn't the number 3.5 GB. It's how **70B's bar overruns the 80 GB HBM ceiling** long before 8B does.

{{< infographic title="KV Cache vs. HBM Budget" description="HBM footprint scales linearly with sequence length × architecture size. The dashed marker is one H100 (80 GB)." >}}

  {{< infographic-controls >}}
    <div class="ig-slider">
      <div class="ig-slider-header">
        <span class="ig-slider-label">Context length</span>
        <output class="ig-slider-readout" id="ig-d1-seq-out">4,096</output>
      </div>
      <input type="range" id="ig-d1-seq" min="1024" max="131072" step="1024" value="4096">
    </div>

    <div class="ig-toggle-row" role="group" aria-label="Precision">
      <button type="button" data-d1-prec="float16" aria-pressed="true">FP16</button>
      <button type="button" data-d1-prec="int8" aria-pressed="false">INT8</button>
    </div>

    <div class="ig-legend" style="margin-top:0.5rem">
      <span class="ig-legend-item"><span class="ig-legend-swatch" style="background:var(--accent-1)"></span>Llama 3 8B</span>
      <span class="ig-legend-item"><span class="ig-legend-swatch" style="background:var(--accent-2)"></span>Llama 3 70B</span>
    </div>
  {{< /infographic-controls >}}

  {{< infographic-viz >}}
    <div class="ig-stage">
      <p class="ig-stage-label">HBM footprint (GB)</p>

      <div class="ig-bar-row">
        <span class="ig-bar-row-label">Llama 3 8B</span>
        <div class="ig-bar">
          <div class="ig-bar-fill" id="ig-d1-bar-8b" style="width:1%"></div>
          <div class="ig-bar-marker" data-label="80 GB · H100" id="ig-d1-mark" style="left:50%"></div>
        </div>
        <span class="ig-bar-row-value" id="ig-d1-val-8b">0.5 GB</span>
      </div>

      <div class="ig-bar-row">
        <span class="ig-bar-row-label">Llama 3 70B</span>
        <div class="ig-bar">
          <div class="ig-bar-fill ig-bar-fill--teal" id="ig-d1-bar-70b" style="width:5%"></div>
          <div class="ig-bar-marker" data-label="" style="left:50%"></div>
        </div>
        <span class="ig-bar-row-value" id="ig-d1-val-70b">2.5 GB</span>
      </div>
    </div>

    <script>
      (function() {
        const models = {
          '8b':  { layers: 32, kv_heads: 8, head_dim: 128 },
          '70b': { layers: 80, kv_heads: 8, head_dim: 128 }
        };
        const $ = (sel) => document.querySelector(sel);
        const ceiling = 80; // one H100, GB
        const scaleMax = 320; // x-axis tops out at 320 GB so even 70B@128k fits
        let prec = 'float16';

        function recalc() {
          const seq = +$('#ig-d1-seq').value;
          $('#ig-d1-seq-out').textContent = seq.toLocaleString();
          const bpe = prec === 'float16' ? 2 : 1;
          for (const key of ['8b', '70b']) {
            const m = models[key];
            const bytes = bpe * 2 * m.layers * m.kv_heads * m.head_dim * seq;
            const gb = bytes / 1024 / 1024 / 1024;
            $('#ig-d1-bar-' + key).style.width =
              Math.min(100, (gb / scaleMax) * 100) + '%';
            $('#ig-d1-val-' + key).textContent = gb.toFixed(1) + ' GB';
          }
          // Marker position is fixed at 80 GB / scaleMax.
          $('#ig-d1-mark').style.left = (ceiling / scaleMax) * 100 + '%';
        }

        $('#ig-d1-seq').addEventListener('input', recalc);
        document.querySelectorAll('[data-d1-prec]').forEach((btn) => {
          btn.addEventListener('click', () => {
            document.querySelectorAll('[data-d1-prec]').forEach((b) =>
              b.setAttribute('aria-pressed', 'false'));
            btn.setAttribute('aria-pressed', 'true');
            prec = btn.dataset.d1Prec;
            recalc();
          });
        });
        recalc();
      })();
    </script>
  {{< /infographic-viz >}}

{{< /infographic >}}

### Demo 2 — scaled circles (parameter count by area)

Bars are great for "how much" of a budget. **Circles are great when the relationship is multiplicative** — a 70B model isn't 8× a 9B model along one axis, it's 8× more *stuff* in total. Sizing by area makes that legible at a glance.

{{< infographic title="Parameter Density — by Area" description="Disc area is proportional to total parameter count. Drag the slider to add a third comparator." >}}

  {{< infographic-controls >}}
    <div class="ig-slider">
      <div class="ig-slider-header">
        <span class="ig-slider-label">Third model size</span>
        <output class="ig-slider-readout" id="ig-d2-third-out">8 B</output>
      </div>
      <input type="range" id="ig-d2-third" min="1" max="120" step="1" value="8">
    </div>

    <p style="font-size:0.7rem;line-height:1.45;margin-top:0.6rem;opacity:0.7">
      Each disc's <em>area</em> ≈ parameter count, so doubling params widens the disc by √2, not 2×. The visual penalty for "just one more zero" is the whole point.
    </p>
  {{< /infographic-controls >}}

  {{< infographic-viz >}}
    <div class="ig-circle-stage" id="ig-d2-stage">
      <div class="ig-circle-bundle">
        <div class="ig-circle" style="--r:36;--color:var(--accent-2)"></div>
        <span class="ig-circle-label">Llama 8B</span>
        <span class="ig-circle-value">8 B</span>
      </div>
      <div class="ig-circle-bundle">
        <div class="ig-circle" style="--r:108;--color:var(--accent-1)"></div>
        <span class="ig-circle-label">Llama 70B</span>
        <span class="ig-circle-value">70 B</span>
      </div>
      <div class="ig-circle-bundle">
        <div class="ig-circle" id="ig-d2-circle" style="--r:36;--color:var(--accent-4)"></div>
        <span class="ig-circle-label">Your model</span>
        <span class="ig-circle-value" id="ig-d2-third-label">8 B</span>
      </div>
    </div>

    <script>
      (function() {
        const $ = (sel) => document.querySelector(sel);
        // Map params (B) -> radius in px. r = sqrt(params) * k, anchored so
        // 70B == 108 px so it visually matches the static circle above.
        const k = 108 / Math.sqrt(70);

        function recalc() {
          const p = +$('#ig-d2-third').value;
          const r = Math.max(12, Math.sqrt(p) * k);
          $('#ig-d2-circle').style.setProperty('--r', r.toFixed(1));
          $('#ig-d2-third-label').textContent = p + ' B';
          $('#ig-d2-third-out').textContent = p + ' B';
        }
        $('#ig-d2-third').addEventListener('input', recalc);
        recalc();
      })();
    </script>
  {{< /infographic-viz >}}

{{< /infographic >}}

### Demo 3 — stacked segments (a single budget split)

When you have **one quantity (here: bytes of HBM per request) split across categories**, a stacked bar shows the trade-off shape. Notice how aggressively the KV cache devours the budget as you raise the context length — model weights stay fixed, activations grow slowly, the cache grows linearly.

{{< infographic title="HBM Budget Allocation per Inference Request" description="Llama 3 70B in FP16. Total budget shown is HBM consumed per request at the chosen context length." >}}

  {{< infographic-controls >}}
    <div class="ig-slider">
      <div class="ig-slider-header">
        <span class="ig-slider-label">Context length</span>
        <output class="ig-slider-readout" id="ig-d3-seq-out">8,192</output>
      </div>
      <input type="range" id="ig-d3-seq" min="1024" max="65536" step="1024" value="8192">
    </div>

    <div class="ig-slider">
      <div class="ig-slider-header">
        <span class="ig-slider-label">Concurrent users</span>
        <output class="ig-slider-readout" id="ig-d3-users-out">4</output>
      </div>
      <input type="range" id="ig-d3-users" min="1" max="32" step="1" value="4">
    </div>

    <p style="font-size:0.7rem;line-height:1.45;margin-top:0.6rem;opacity:0.7">
      Weights are fixed (≈140 GB) and shared across users; the KV cache is per-user and scales linearly with context.
    </p>
  {{< /infographic-controls >}}

  {{< infographic-viz >}}
    <div class="ig-stage">
      <p class="ig-stage-label" id="ig-d3-total">Total: 0 GB</p>

      <div class="ig-stack" id="ig-d3-stack">
        <div class="ig-stack-seg ig-stack-seg--ink" id="ig-d3-seg-w" style="flex-basis:60%">Weights</div>
        <div class="ig-stack-seg ig-stack-seg--1" id="ig-d3-seg-kv" style="flex-basis:30%">KV</div>
        <div class="ig-stack-seg ig-stack-seg--3" id="ig-d3-seg-a" style="flex-basis:10%">Act</div>
      </div>

      <div class="ig-legend">
        <span class="ig-legend-item"><span class="ig-legend-swatch" style="background:var(--color-ink)"></span>Weights (140 GB, fixed)</span>
        <span class="ig-legend-item"><span class="ig-legend-swatch" style="background:var(--accent-1)"></span>KV cache (per user × ctx)</span>
        <span class="ig-legend-item"><span class="ig-legend-swatch" style="background:var(--accent-3)"></span>Activations</span>
      </div>

      <p class="ig-stage-label" style="margin-top:1rem">Headroom on a 4× H100 node (320 GB)</p>
      <div class="ig-bar">
        <div class="ig-bar-fill ig-bar-fill--orange" id="ig-d3-headroom" style="width:50%"></div>
      </div>
    </div>

    <script>
      (function() {
        const $ = (sel) => document.querySelector(sel);
        // 70B FP16 weights ≈ 140 GB; KV cache per user (8 kv_heads × 128 head_dim
        // × 80 layers × 2 (k+v) × 2 bytes) = 327_680 B per token ≈ 0.305 MB/token.
        const W = 140; // GB
        const KV_PER_TOKEN_GB = (8 * 128 * 80 * 2 * 2) / 1024 / 1024 / 1024;
        const ACT = 6; // GB rough constant
        const HOST = 320; // GB total HBM

        function recalc() {
          const seq = +$('#ig-d3-seq').value;
          const users = +$('#ig-d3-users').value;
          $('#ig-d3-seq-out').textContent = seq.toLocaleString();
          $('#ig-d3-users-out').textContent = users;

          const kv = users * seq * KV_PER_TOKEN_GB;
          const total = W + kv + ACT;
          const pct = (x) => Math.max(0.02, x / total) * 100;
          $('#ig-d3-seg-w').style.flexBasis  = pct(W) + '%';
          $('#ig-d3-seg-kv').style.flexBasis = pct(kv) + '%';
          $('#ig-d3-seg-a').style.flexBasis  = pct(ACT) + '%';
          $('#ig-d3-seg-w').textContent  = 'W ' + W.toFixed(0);
          $('#ig-d3-seg-kv').textContent = 'KV ' + kv.toFixed(1);
          $('#ig-d3-seg-a').textContent  = 'A ' + ACT;
          $('#ig-d3-total').textContent = 'Total: ' + total.toFixed(1) + ' GB / ' + HOST + ' GB on a 4× H100 node';
          const headroom = Math.min(100, (total / HOST) * 100);
          $('#ig-d3-headroom').style.width = headroom + '%';
        }

        $('#ig-d3-seq').addEventListener('input', recalc);
        $('#ig-d3-users').addEventListener('input', recalc);
        recalc();
      })();
    </script>
  {{< /infographic-viz >}}

{{< /infographic >}}

**Recipe** — outline of any infographic:

```markdown
{{</* infographic title="..." description="..." */>}}

  {{</* infographic-controls */>}}
    <!-- .ig-slider, .ig-toggle-row, .ig-field, <select>, etc. -->
  {{</* /infographic-controls */>}}

  {{</* infographic-viz */>}}
    <div class="ig-stage">
      <!-- .ig-bar-row, .ig-circle, .ig-stack — geometric primitives -->
    </div>
    <script> ... vanilla JS in an IIFE; updates widths/radii ... </script>
  {{</* /infographic-viz */>}}

{{</* /infographic */>}}
```

Full vocabulary in [`docs/components/infographic.md`](https://github.com/anthropics/) — `.ig-bar` (width-driven), `.ig-circle` (area-driven), `.ig-stack` (proportional segments), `.ig-slider` (labelled range), plus an SVG primitive set for arrows and custom shapes.

---

## Design tokens

All component colours come from CSS custom properties at the top of `themes/almanac/assets/css/main.css`. Never hard-code hex values in shortcodes or articles.

| Token | Value (cream theme) | Used for |
|---|---|---|
| `--accent-1` | `#FF007F` (pop pink) | Primary accent, pullquote/profound, marker dots |
| `--accent-2` | `#00A8A8` (pop teal) | Secondary accent, notes, italic emphasis |
| `--accent-3` | `#FFD700` (pop yellow) | Highlight stripes, theorem headers, hover states |
| `--accent-4` | `#FF8C00` (pop orange) | Warnings, counter-intuitive quotes, blockquote bar |
| `--color-ink` | `#1A1A1A` | Text, borders, hard shadow |
| `--color-cream` | `#FDF5E6` | Page background (cream theme) |
| `--shadow-pop` | `4px 4px 0 ink` | Standard hard shadow |
| `--shadow-pop-lg` | `8px 8px 0 ink` | Bigger hard shadow for stage-stealing components |
| `--border` | `3px solid ink` | Default component border |

## Cardinal rules

1. **Don't reach for a component on the first draft.** Write the prose; see if it stands. Components are seasoning.
2. **One pullquote, ≤2 callouts, ≤3 margin notes per article.** Rarity is the punch.
3. **Hero is automatic.** Don't repeat it mid-article.
4. **Captions and labels stay terse.** 4–8 words, UPPERCASE-friendly.
5. **Don't author CSS in articles.** Fix the stylesheet instead.
