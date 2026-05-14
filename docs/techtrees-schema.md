# `data/techtrees/` — tech-tree data files

Each `issueNN.toml` here is the **data backing the tech tree** rendered on issue `NN`'s cover page. The shortcode lives in `themes/almanac/layouts/shortcodes/techtree.html`; this directory only holds data.

## Canonical Example

See `issue03.toml` for a complete, working tree (11 nodes, 13 edges, legend, caption). That issue is the reference implementation — copy its structure when starting a new tree.

## Schema (TOML)

```toml
title  = "Tech tree — display caption"     # optional
width  = 1280                              # SVG viewBox width  (default 1200)
height = 740                               # SVG viewBox height (default 700)
node_w = 200                               # node box width  (default 200)
node_h = 70                                # node box height (default 70)
caption = "optional footer HTML"           # rendered with safeHTML

[[nodes]]
id    = "unique-id"                        # required, used by edges
label = "Display Label\nTwo Lines OK"      # \n in TOML becomes a literal newline → two-line label
kind  = "mainline"                         # mainline | primer | boss | external
x     = 600                                # center x in viewBox coords
y     = 220                                # center y in viewBox coords
link  = "06-some-article/"                 # relative URL to article (resolves to the issue cover dir)
sub   = "ch.6"                             # small uppercase label above the box (optional)

[[edges]]
from = "prerequisite-id"                   # required, points UPWARD in the diagram
to   = "uses-it-id"                        # arrow head sits here

[[legend]]
kind  = "primer"
label = "Tutorial primer"
```

## Layout Conventions

- **Y-axis is the dependency direction.** Put boss/mainline at the **top** (small y), primers at the **bottom**. Arrows point UPWARD.
- **Lay out by hand.** Hugo does not auto-layout. Pick coords that produce a readable, columnar structure. Sketch it on paper before writing TOML.
- **Cold-open belongs in the cover prose, not the tree.** The tree shows mathematical dependencies; the cold-open is the narrative entrance with no prereqs.

## Node Kinds At A Glance

| `kind`     | Visual              | Meaning                              |
|------------|---------------------|--------------------------------------|
| `mainline` | pink fill           | A storyline chapter advancing plot   |
| `primer`   | cream fill          | A standalone tutorial on one concept |
| `boss`     | yellow + halftone   | The "everything builds to this" cap  |
| `external` | faint teal          | Cross-issue concept, not a chapter   |

## How Links Resolve

Inside the shortcode, `link` values without a leading `/` are joined to the **issue cover's RelPermalink**. This means:

- `link = "06-outliers/"` from `issue03.toml` resolves to `/llm-maths/issues/03-sixteen-numbers/06-outliers/`
- The shortcode works correctly whether called from the issue cover OR from inside an article (it walks `.Page.Parent` for articles)

## Validation

`make validate` checks that:
- Every node `id` referenced by an edge actually exists
- Every node `link` resolves to an existing article file
- Every article's `techNode` front-matter field matches a node `id` here (unless `techKind=mainline` for cold-open or `techKind=external`)
