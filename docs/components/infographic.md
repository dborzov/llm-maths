# Infographic

**Two-pane interactive panel: controls on the left, live geometric readout on the right.** Use when a static table of numbers would convey the technical point less worse than a reader poking toggles and watching shapes resize.

## The basic unit is a *comparison*

A single number in isolation tells the reader nothing about scale. **Every infographic must compare at least two things** — or show one thing reacting visibly to a slider. Patterns that work:

| Pattern | When to use it | Vocabulary |
|---|---|---|
| **Comparison bars** — two stacked horizontal bars, one slider | Two architectures, models, or settings racing the same budget. Whoever overruns a marker first is the lede. | `.ig-bar-row`, `.ig-bar`, `.ig-bar-marker` |
| **Scaled circles** — two or three discs sized by area | Multiplicative comparisons: param count, FLOPs, memory volume. Area is perceptually accurate; height alone exaggerates. | `.ig-circle`, `.ig-circle-stage` |
| **Stacked segments** — single bar split into proportional pieces | A single budget (HBM, vocab, compute) split across categories. Watch the trade-off reshape. | `.ig-stack`, `.ig-stack-seg` |
| **Slider → shape** — one input drives one geometric reaction | "As X grows, see Y morph." The simplest causality demo. | `.ig-slider` + any primitive |
| **Custom SVG** — inline `<svg>` with theme-coloured shapes | When the picture is genuinely diagrammatic (boxes and arrows). | `.ig-svg` + `data-fill`/`data-stroke` attributes |

If your infographic shows **one number**, you're using the component wrong. Find a baseline to compare against, or a slider to drive the number around, or both.

**Scope is deliberately limited.** No bundled charts. No line plots. No animations beyond width/height transitions. If you need a chart, ship a static pyplot block instead.

**One infographic per article maximum.** They are heavy.

## How to invoke

A three-shortcode nest. Use the **`<>` delimiter form** — the body is raw HTML and inline `<script>`. The canonical recipe is a two-thing comparison driven by one slider:

```markdown
{{< infographic title="KV Cache vs. HBM Budget"
                description="Two architectures, one slider, one budget ceiling." >}}

  {{< infographic-controls >}}
    <div class="ig-slider">
      <div class="ig-slider-header">
        <span class="ig-slider-label">Context length</span>
        <output class="ig-slider-readout" id="kv-seq-out">4,096</output>
      </div>
      <input type="range" id="kv-seq" min="1024" max="131072" step="1024" value="4096">
    </div>

    <div class="ig-toggle-row" role="group" aria-label="Precision">
      <button type="button" data-prec="float16" aria-pressed="true">FP16</button>
      <button type="button" data-prec="int8" aria-pressed="false">INT8</button>
    </div>
  {{< /infographic-controls >}}

  {{< infographic-viz >}}
    <div class="ig-stage">
      <p class="ig-stage-label">HBM footprint (GB)</p>

      <div class="ig-bar-row">
        <span class="ig-bar-row-label">Llama 3 8B</span>
        <div class="ig-bar">
          <div class="ig-bar-fill" id="kv-bar-8b" style="width:1%"></div>
          <div class="ig-bar-marker" data-label="80 GB · H100" id="kv-mark" style="left:25%"></div>
        </div>
        <span class="ig-bar-row-value" id="kv-val-8b">0.5 GB</span>
      </div>

      <div class="ig-bar-row">
        <span class="ig-bar-row-label">Llama 3 70B</span>
        <div class="ig-bar">
          <div class="ig-bar-fill ig-bar-fill--teal" id="kv-bar-70b" style="width:5%"></div>
        </div>
        <span class="ig-bar-row-value" id="kv-val-70b">2.5 GB</span>
      </div>
    </div>

    <script>
      (function() {
        const models = {
          '8b':  { layers: 32, kv_heads: 8, head_dim: 128 },
          '70b': { layers: 80, kv_heads: 8, head_dim: 128 }
        };
        const $ = (sel) => document.querySelector(sel);
        const ceiling = 80, scaleMax = 320;   // GB
        let prec = 'float16';

        function recalc() {
          const seq = +$('#kv-seq').value;
          $('#kv-seq-out').textContent = seq.toLocaleString();
          const bpe = prec === 'float16' ? 2 : 1;
          for (const key of ['8b', '70b']) {
            const m = models[key];
            const gb = bpe * 2 * m.layers * m.kv_heads * m.head_dim * seq / 2**30;
            $('#kv-bar-' + key).style.width = Math.min(100, gb / scaleMax * 100) + '%';
            $('#kv-val-' + key).textContent = gb.toFixed(1) + ' GB';
          }
          $('#kv-mark').style.left = ceiling / scaleMax * 100 + '%';
        }
        $('#kv-seq').addEventListener('input', recalc);
        document.querySelectorAll('[data-prec]').forEach((btn) => {
          btn.addEventListener('click', () => {
            document.querySelectorAll('[data-prec]').forEach((b) =>
              b.setAttribute('aria-pressed', 'false'));
            btn.setAttribute('aria-pressed', 'true');
            prec = btn.dataset.prec;
            recalc();
          });
        });
        recalc();
      })();
    </script>
  {{< /infographic-viz >}}

{{< /infographic >}}
```

The point isn't the number 2.5 GB. It's that **the 70B bar crashes through the 80 GB H100 marker** at a context the 8B bar barely notices. The bars *and* the marker make that visible.

## Parameters

### `infographic` (outer)

| Param | Required | Description |
|---|---|---|
| `title` | **yes** | Header label in the black chrome bar. |
| `description` | no | Small fine-print line under the controls. |
| `id` | no | Anchor id. |

### `infographic-controls` and `infographic-viz` (slots)

No parameters. Body is raw HTML.

## Pre-styled helper classes

### Form controls (controls slot)

| Class | Element | Purpose |
|---|---|---|
| `.ig-field` | `<label>` block | One labelled form control row |
| `.ig-field-label` | `<span>` | The uppercase label above the control |
| `.ig-toggle-row` | `<div role="group">` | Side-by-side toggle buttons (use `aria-pressed`) |
| `.ig-slider` | `<div>` wrapping `.ig-slider-header` + `<input type="range">` | Labelled slider with a yellow readout pill. Most ergonomic control type — prefer it over `.ig-field` with a plain range. |
| `.ig-slider-header` | `<div>` | Flex row holding the label and the readout pill |
| `.ig-slider-label` | `<span>` | The uppercase label |
| `.ig-slider-readout` | `<output>` | The yellow pill showing the current value (update via JS) |

Selects, number inputs, and text inputs inside `.ig-field` are auto-styled.

### Visualization primitives (viz slot)

| Class | Drives | Purpose |
|---|---|---|
| `.ig-stage` | layout | Flex column inside `.infographic-viz` — gives consistent vertical rhythm. Wrap your viz in this. |
| `.ig-stage-label` | layout | Small uppercase caption above a primitive. |
| `.ig-bar` + `.ig-bar-fill` | `style="width:XX%"` on the fill | Horizontal bar. Default fill is pink — modifier classes `--teal`, `--yellow`, `--orange` switch colour. |
| `.ig-bar-marker` | `style="left:XX%"` | Vertical tick across a bar (e.g. "80 GB H100 line"). `data-label` attr renders a small label above. |
| `.ig-bar-row` | grid | Three-column row: `label | bar | numeric readout`. Stack two rows for the canonical two-thing comparison. |
| `.ig-bar-row-label` / `.ig-bar-row-value` | — | The left label and right numeric readout. |
| `.ig-circle` | `--r` custom property (radius in px), `--color` | A disc sized by radius. Use `Math.sqrt(quantity) * k` so disc *area* tracks the quantity (perceptually accurate). |
| `.ig-circle-stage` | layout | Flex row that holds multiple `.ig-circle-bundle`s side-by-side. |
| `.ig-circle-bundle` | layout | One disc + its label + value, stacked vertically. |
| `.ig-stack` + `.ig-stack-seg` | `style="flex-basis:XX%"` per segment | Stacked horizontal bar with up to 5 colour-coded segments (`--1`/`--2`/`--3`/`--4`/`--ink`). Use to show one quantity split across categories. |
| `.ig-legend` + `.ig-legend-item` + `.ig-legend-swatch` | — | Colour legend strip. |
| `.ig-bignum` | text content | Big single-number readout. **Avoid using this alone** — pair with a primitive. |
| `.ig-svg` | inline `<svg>` | Theme-coloured SVG. Use `data-stroke="ink|pink"` and `data-fill="pink|teal|yellow|orange|cream"` on child shapes for pop-art colours without inline styles. |

### Driving the primitives from JS

```js
// Width-driven bar
$('#bar-fill').style.width = (value / max * 100) + '%';

// Area-driven circle: keep area proportional to value
const k = ANCHOR_RADIUS / Math.sqrt(ANCHOR_VALUE);   // calibrate once
$('#circle').style.setProperty('--r', (Math.sqrt(value) * k).toFixed(1));

// Stack segment
$('#seg-kv').style.flexBasis = (kv / total * 100) + '%';
```

Use `style.setProperty('--r', ...)` for custom properties; use `style.width`/`flexBasis` for normal sizing. Both transition smoothly thanks to the CSS rules.

## When to use

- A 4-row × 3-col static table that the reader is supposed to "understand the shape of." Replace with the table's *input* on the left and one *derived number* on the right.
- A scaling-law graph reduced to its punchline: "as N grows, X grows like…"
- A model-comparison matrix where the reader will care about 2–3 dimensions at a time, not all of them at once.

## When NOT to use

- For decorative interactivity. If a static table conveys the point in 3 seconds, leave the table.
- For multi-step interactions. If a reader has to do more than two toggles to get the insight, you've built a tool, not a infographic.
- For anything that requires a chart. We don't ship a charting library — render the chart with [pyplot](pyplot.md) and skip the interactivity.

## Constraints

- **Vanilla JS only.** No frameworks, no bundlers.
- **Inline `<script>` inside the shortcode is fine** and the canonical pattern. Scope variables in an IIFE so multiple infographics on one page don't collide.
- **ID-prefix every selector** with the infographic name (e.g. `#ig-kv-...`). Two infographics on a page must not share ids.
- **Keep total JS under ~100 lines.** Push complex logic into a static pyplot block instead.

## Where it renders

- Shortcodes: `themes/almanac/layouts/shortcodes/infographic.html`, `infographic-controls.html`, `infographic-viz.html`
- CSS: search `INFOGRAPHIC` in `themes/almanac/assets/css/main.css`

## Pitfalls

- **Hugo's `<>` delimiter passes inner HTML through unchanged** — that's why this shortcode uses it. But it also means **Markdown inside is NOT processed**. If you need bold/italic in your labels, write `<strong>` / `<em>`.
- **Don't put `<script>` outside `infographic-viz`.** The shortcode body runs at page render and is included once — keep all JS in one place.
