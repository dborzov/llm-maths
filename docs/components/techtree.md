# Tech Tree

**The DAG of articles inside an issue, rendered as the issue's visual table of contents.** One per issue, on the issue cover page.

## How to invoke

```markdown
{{< techtree name="issue03" >}}
```

The `name` argument selects a TOML file at `data/techtrees/<name>.toml`. By convention `name = issue<NN>`.

## Layout convention

Y-axis = dependency direction. Boss / mainline chapters at the **top**; prerequisites at the **bottom**. Arrows point upward (from prerequisite to user). Every node maps to one article.

## When to use

- **On the issue cover only.** Never inside an article — the tree is the TOC, not body content.
- **Exactly once per issue.** A second tree would conflict with the first.

## Where it renders

- Shortcode: `themes/almanac/layouts/shortcodes/techtree.html`
- Schema: [`docs/techtrees-schema.md`](../techtrees-schema.md)
- CSS: search `TECHTREE WIDGET` in `themes/almanac/assets/css/main.css`
- Existing example: `data/techtrees/issue03.toml`

## Pitfalls

- **Relative `link` values resolve against the issue cover's `RelPermalink`**, *not* against the page calling the shortcode. So `link = "06-outliers/"` works regardless of where the tree is rendered.
- **Inside `range` blocks, `$` is the root context — not the parent iterator.** Capture the current node into a local variable (`{{ $node := . }}`) before referencing fields.
- **Don't try to hide an article from the tree by setting `techKind: mainline`.** Mainline cold-opens are deliberately tree-less, but every primer / boss must have a `techNode` matching a tree node — `make validate` enforces this.
