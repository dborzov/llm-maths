# LLM Maths Comics — Claude Code Instructions

A static website: a serialized collection of long-form deep-learning mathematics articles structured as **comic-book issues**. Each issue groups multiple articles around a single theme and presents them via a **tech tree graph** as table of contents. Pop-art neubrutalist visual design. Hugo + custom theme + Python pre-build script.

Live at: `https://borzov.ca/llm-maths/`

**Canonical reference implementation:** `content/issues/03-sixteen-numbers/` — when in doubt, copy what issue 03 does.

**Canonical LLM vocabulary:** `content/issues/05-microgpt-unfolded/` introduces a 60-line plain-Python transformer (microGPT) whose variable names (`wte`, `wpe`, `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`, `mlp_fc1`, `mlp_fc2`, `lm_head`, `q`, `k`, `v`, `keys[li]`, `values[li]`, `x_residual`, `attn_logits`, `attn_weights`, `head_out`, `x_attn`, `prefill`, `decode`, …) are the **canonical names** every article must use. See [`docs/microgpt-contract.md`](docs/microgpt-contract.md) for the full contract.

---

## Which Doc to Read

| Task | Read |
|---|---|
| Write or outline a new issue from scratch | [`docs/new-issue.md`](docs/new-issue.md) |
| Write, polish, or deepen an article | [`docs/writing.md`](docs/writing.md) — authoritative |
| Add components or make an article visually richer | [`docs/components/README.md`](docs/components/README.md) |
| Front matter fields, file naming, directory layout | [`docs/issue-format.md`](docs/issue-format.md) |
| microGPT variable names and cross-linking rules | [`docs/microgpt-contract.md`](docs/microgpt-contract.md) |
| CSS, templates, scripts, validation, CI, pitfalls | [`docs/dev.md`](docs/dev.md) |
| Tech tree TOML schema | [`docs/techtrees-schema.md`](docs/techtrees-schema.md) |
| Timeline TOML schema | [`docs/timelines-schema.md`](docs/timelines-schema.md) |

---

## Quick Commands

```bash
make validate        # lint front matter, tech tree, links, pyplot rules
make preview         # validate + plots + serve at localhost:1313/llm-maths/
make build           # validate + plots + minified build to public/
```

Full command reference in [`docs/dev.md`](docs/dev.md).

---

## When You Are Stuck

- "What does a good article look like?" → read `content/issues/03-sixteen-numbers/06-outliers.md` (mainline) and `content/issues/03-sixteen-numbers/03-lloyd-max.md` (primer).
- "Which component do I use here?" → [`docs/components/README.md`](docs/components/README.md). Then drill into the per-component file.
- "What does a good tech tree look like?" → see `data/techtrees/issue03.toml`; schema in [`docs/techtrees-schema.md`](docs/techtrees-schema.md).
- "What does a good timeline look like?" → see `data/timelines/quantization2019to2026.toml`; schema in [`docs/timelines-schema.md`](docs/timelines-schema.md).
- "What conventions am I forgetting?" → run `make validate`. It tells you.
