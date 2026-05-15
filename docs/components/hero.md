# Hero Image

**Full-width banner illustration at the top of every article and issue cover.** Eye-catching visual candy that sets the mood. Mandatory — every article has one, even if it's just `default.webp`.

## How to set it

Hero images are configured via front matter, not a shortcode:

```yaml
---
title: "Outliers, Anomalies, and the Long Tail"
header: lloyd-max.webp     # optional; defaults to default.webp
---
```

The file lives at `static/header-illustrations/<header>`. The `single.html` and `issue-cover.html` templates render it inside an `.article-header-illustration` block below the article title.

## File contract

| Property | Value |
|---|---|
| Format | **WebP** (lossy VP8). PNG/JPEG are forbidden — they are 5–60× larger at the same quality. |
| Max width | 1600 px (height auto-scales) |
| Preferred aspect | ~8:3 (1600×600) — the comic-strip banner look |
| Target size | **<150 KB.** Hard cap 300 KB. |
| Metadata | Stripped |
| Naming | `<slug>.webp` (e.g. `lloyd-max.webp`) — keep it short |
| Originals | High-res PNGs go in `raw-sort-me/<slug>/` (gitignored). Only the optimized `.webp` is committed. |

## Canonical encoding command

```bash
convert <source.png> \
  -resize 1600x -strip \
  -define webp:lossless=false -define webp:method=6 \
  static/header-illustrations/<slug>.webp
```

If the result exceeds 300 KB, drop the resize width to 1200 px before raising compression — visual fidelity matters more than the last 50 KB.

## When to commission a custom hero

| Situation | Header to use |
|---|---|
| Tentpole mainline chapter | Custom illustration. Worth the effort. |
| Primer chapter | Either a custom illustration *or* the theme's `default.webp`. Skip the front-matter line to take the default. |
| Cold-open / boss | Always custom. These pages are the doors readers walk through. |

## Where it renders

- `themes/almanac/layouts/_default/single.html` — at the top of every article, below the title and topic chip
- `themes/almanac/layouts/_default/issue-cover.html` — at the top of every issue cover
- CSS class: `.article-header-illustration` (with `.article-hero` as a forward-compatible alias)

## Pitfalls

- **Do not embed a second hero mid-article.** If you need a second large image in the body, use [figure.md](figure.md).
- **Do not commit PNG originals.** They balloon the repo. Put them in `raw-sort-me/` and gitignore.
- **Do not change the aspect ratio per-article.** Pages will jolt as readers scroll between articles.
