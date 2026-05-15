# Component Library — AI-Agent Reference

**The complete catalog of reusable article components.** Every visual building block we encourage authors to reach for. Read this file end-to-end the first time you write or edit an article. Once you know what's here, return only to the per-component files when you need exact syntax.

The corresponding human-facing showcase lives at `/llm-maths/docs/components/` (source: `content/docs/components.md`). Live examples render there.

## How to pick a component

| You want to… | Use |
|---|---|
| Set the **mood** at the top of an article | **Hero image** — `header:` front-matter field |
| Show **chronologically ordered field history** (≤5 events) | `{{< timeline name="..." >}}` |
| Show the **DAG** of articles inside an issue | `{{< techtree name="..." >}}` (issue cover only) |
| **Highlight a profound conclusion / theorem result** | `{{% pullquote %}}` |
| Add a **tangent / "by the way" / pro-tip / warning** | `{{% callout %}}` |
| Drop an **optional citation or aside** in the margin | `{{% marginnote %}}` |
| Drop a **signpost mid-section** without claiming a TOC entry | `{{< crosshead >}}` |
| Embed a **third-party image / diagram** with caption | `{{< figure >}}` |
| Embed our own **executable matplotlib plot** | fenced ` ```pyplot ` block |
| Show a **dark-themed Python listing** | fenced ` ```python ` code block |
| Replace a **wall-of-numbers table** with toggles + live readouts | `{{< infographic >}}` |

## The components

Every file below is self-contained. Read the one you need; you can ignore the rest.

| Component | Detail file | One-line description |
|---|---|---|
| **Hero image** | [hero.md](hero.md) | Full-width banner at top of every article + issue cover. Set via `header:` front-matter. WebP-only. |
| **Timeline** | [timeline.md](timeline.md) | Five-event milestone strip from `data/timelines/<name>.toml`. Hard 5-event cap. |
| **Tech tree** | [techtree.md](techtree.md) | DAG TOC on issue covers. Data in `data/techtrees/<name>.toml`. |
| **Pullquote** | [pullquote.md](pullquote.md) | Stage-stealing excerpt panel. Three colour variants. |
| **Callout** | [callout.md](callout.md) | Bordered sidebar for tangents, tips, warnings, definitions. Five variants. |
| **Margin note** | [marginnote.md](marginnote.md) | Tufte-style sidenote — margin on desktop, inline block on mobile. |
| **Crosshead** | [crosshead.md](crosshead.md) | Mid-section signpost. Not a real heading. |
| **Figure** | [figure.md](figure.md) | Image + caption + click-to-fullscreen lightbox. |
| **Pyplot block** | [pyplot.md](pyplot.md) | Executable matplotlib code fences. Pre-rendered at build. |
| **Code block** | [codeblock.md](codeblock.md) | Standard fenced code with Chroma syntax highlighting. |
| **Infographic** | [infographic.md](infographic.md) | Two-pane interactive panel: controls + live readout. |

## Cardinal rules

1. **Don't reach for a component on the first draft.** Write the prose, see if it stands. Components are seasoning, not substance.
2. **One pullquote per article, two callouts max, three margin notes max.** They lose impact when stacked.
3. **Hero is automatic.** Every article gets one via `header:` — don't add a second hero shortcode mid-article.
4. **Captions and labels stay terse.** UPPERCASE 4–8 words. Long captions read as paragraphs and break the comic-book voice.
5. **Use the `%%` shortcode form whenever the body contains Markdown.** The `<>` form skips Markdown processing. Each detail file says which form to use.
6. **Don't author CSS in articles.** If a component's appearance is wrong, fix it in `themes/almanac/assets/css/main.css` — never with inline `<style>`.

## Editing the library

| If you want to… | Do this |
|---|---|
| Add a new component | Create `themes/almanac/layouts/shortcodes/<name>.html` + CSS in `main.css` under "COMPONENT LIBRARY" + a `docs/components/<name>.md` reference + a row above + a showcase block in `content/docs/components.md`. |
| Change a component's look | Edit only the CSS block in `main.css`. Don't touch the shortcode HTML unless you're changing the API. |
| Add a new variant to an existing component | Update both the shortcode HTML enum + the CSS modifier class + the detail file. |
| Rename a component | Search-and-replace across `themes/almanac/layouts/shortcodes/`, CSS, `docs/components/`, and `content/`. Then update this README's table. |

## Cross-references

- **Design tokens** (colors, shadows, borders, fonts) live in CSS custom properties at the top of `themes/almanac/assets/css/main.css`. Use them, don't hard-code hex values.
- **KaTeX math** works inside every component that uses the `%%` form. Math: set `math: true` in article front matter.
- **microGPT terminology contract** (variable names like `attn_wk`, `wpe`, etc.) is documented in the root `CLAUDE.md` and is independent of components.
