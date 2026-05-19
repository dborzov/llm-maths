# `data/techtrees/` — tech-tree data files

Each `issueNN.toml` here is the **data backing the tech tree** rendered on issue `NN`'s cover page. The shortcode lives in `themes/almanac/layouts/shortcodes/techtree.html`; this directory only holds data.

**Layout is auto-computed.** `techtree.js` assigns node positions using a hierarchical layout algorithm — no x/y coordinates in the TOML. Just declare nodes and edges; the JS handles the rest.

## Canonical Example

See `issue03.toml` for a complete, working tree. That issue is the reference implementation.

## Schema (TOML)

```toml
title   = "Tech tree — display caption"    # optional
caption = "optional footer HTML"           # rendered with safeHTML

[[nodes]]
id    = "unique-id"                        # required; used by edges and techNode front matter
label = "Display Label\nTwo Lines OK"      # \n in TOML becomes a literal newline → two-line label
kind  = "mainline"                         # mainline | primer | boss | external
link  = "06-some-article/"                 # relative URL to article (resolves from the issue cover dir)
sub   = "ch.6"                             # small uppercase label above the box (optional)

[[edges]]
from = "prerequisite-id"   # the article that must be read first
to   = "depends-on-it-id"  # the article that needs the prerequisite; arrow points here

[[legend]]
kind  = "primer"
label = "Tutorial primer"
```

**No coordinates, no canvas dimensions needed.** The JS computes layout automatically.

## Edge Philosophy

Edges encode **reading prerequisites** — not "related articles" and not "mentioned in".

An edge `from A to B` means: *a reader who has not read A will likely be confused by B*.

Guidelines:
- Include only **non-obvious** prerequisites. If A and B both cover the same broad domain, that's not a useful edge.
- Limit to ~3–5 incoming edges per node to prevent the graph from becoming a ball of yarn.
- Do NOT add edges just because articles share a topic.
- DO add edges when B uses a specific concept, formula, or result from A without re-explaining it.
- Cold-open articles have no dependencies — they are the narrative entrance to the issue.

## Node Kinds At A Glance

| `kind`     | Visual              | Meaning                                    |
|------------|---------------------|--------------------------------------------|
| `mainline` | pink fill           | A storyline chapter advancing the narrative |
| `primer`   | cream fill          | A standalone tutorial on one concept       |
| `boss`     | yellow + halftone   | Capstone that synthesises the whole issue  |
| `external` | faint teal          | Cross-issue concept (not a local chapter)  |

## Layout Algorithm

`techtree.js` uses a simplified Sugiyama-style hierarchical layout:

1. **Layer assignment**: layer 0 = nodes with no prerequisites (sources); each dependent node gets `max(parent_layer) + 1`.
2. **Source compaction**: source nodes (no prerequisites) are pulled up to `min(child_layer) − 1` so primers sit adjacent to the first chapter they enable rather than isolated at the bottom.
3. **Barycenter ordering**: within each layer, nodes are sorted by the average position of their neighbours in the layer below, reducing edge crossings.

The graph shows in the **Graph tab** (default on tablet/desktop) or **List tab** (default on mobile). Users can switch at any time.

## How Links Resolve

Inside the shortcode, `link` values without a leading `/` are joined to the **issue cover's RelPermalink**:

- `link = "06-outliers/"` from `issue03.toml` → `/llm-maths/comicbook/03-quantization/06-outliers/`
- Works correctly whether called from the issue cover OR from inside an article (the shortcode walks `.Page.Parent` for articles).

## Validation

`make validate` checks that:
- Every node `id` referenced by an edge actually exists in the same file
- Every node `link` resolves to an existing article file
- Every article's `techNode` front-matter field matches a node `id` here (unless `techKind=mainline` for cold-open)
