# Code Block

**Syntax-highlighted source listing.** Standard Hugo fenced code blocks rendered via Chroma. Use for Python snippets that *aren't* executed (executed plots go in [pyplot blocks](pyplot.md)).

## How to invoke

Just a standard Markdown fenced code block with a language tag:

````markdown
```python
def evict_kv(cache, scores, k):
    sinks = cache[:4]
    recent = cache[-k:]
    return torch.cat([sinks, cache[topk(scores[4:-k], k)], recent])
```
````

For shell commands, use ```` ```bash ````. For TOML, ```` ```toml ````. For raw text without highlighting, ```` ``` ```` with no language.

## Rendering

- **Dark theme** (the ink `#1A1A1A` background) with `#F8F8F2` text and a pop-pink language label tab in the top-right corner.
- **3px ink border + hard shadow** so it pops off the cream page background.
- **Horizontal scroll** for lines wider than the container.
- **Inline `code`** (single backticks) renders ink-background + yellow-text, with a thin ink border. Use for variable names, paths, and short command names.

## Conventions

- **Python listings:** use `python`. Keep them short — under 20 lines. Long listings break narrative; split them with prose between sections.
- **Don't dump full files.** Show the 3–10 lines that matter; reference the file path elsewhere.
- **Use the canonical microGPT names** (`attn_wk`, `wpe`, `q`, `k`, etc.) — see the microGPT terminology contract in the root `CLAUDE.md`.
- **Comments inside code are fine** — Chroma will style them. But don't pad short snippets with explanation comments; that's what the surrounding prose is for.

## When to use vs. pyplot

| You have… | Use |
|---|---|
| Reproducible numpy plot | [pyplot block](pyplot.md) |
| Code excerpt from a paper, reference impl, or 3rd-party library | ` ```python ` code block |
| Snippet illustrating an algorithm in pseudo-Python | ` ```python ` code block |
| Shell commands the reader should run | ` ```bash ` code block |
| Config file content (TOML, YAML, JSON) | ` ```toml/yaml/json ` code block |

## Where it renders

- Chroma config: `hugo.toml`
- CSS: search `CODE BLOCKS` in `themes/almanac/assets/css/main.css`
- Language label: emitted via `.highlight::before { content: attr(data-lang); }` — Hugo sets `data-lang` automatically.
