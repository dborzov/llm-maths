# LLM Maths Almanac — Visual Style Reference

This is the **design system reference app** for [Ye Olde LLM Maths Almanac](https://borzov.ca/llm-maths/). It is a Vite + React app that renders the pop-art neubrutalist style guide and a wiki mockup, both of which were used as the visual source of truth when building the site's custom Hugo theme.

**This app is not the website.** It is a living style reference — the equivalent of a Figma file, but in runnable code.

---

## The Two Views

Toggle between them using the button bar at the bottom of the screen.

### Style Guide

An annotated showcase of every design token and UI pattern:

- **Colour palette** — Pop Pink `#FF007F`, Pop Teal `#00A8A8`, Pop Yellow `#FFD700`, Pop Orange `#FF8C00`, Pop Cream `#FDF5E6`, Ink Black `#1A1A1A`
- **Typography** — Bangers (display), Space Grotesk (body/UI)
- **Interactive states** — default, hover, pressed, disabled
- **Recurring motifs** — halftone dots, checkered grid, hard offset shadows

These values are directly replicated in `themes/almanac/assets/css/main.css` and the data theme files `data/themes/cream.toml` and `data/themes/teal.toml`.

### Wiki Mockup

A full-page simulation of an almanac article, showing:

- Sticky navigation bar with thick bottom border
- Article header with topic label and large Bangers title
- Prose with drop-cap first letter
- Side-by-side image + key-characteristics panel
- Sidebar with article metadata and dark newsletter widget

The **cream theme** (`data/themes/cream.toml`) is derived from the Style Guide view (cream `#FDF5E6` background, Pop Pink primary accent).

The **teal theme** (`data/themes/teal.toml`) is derived from the dark sidebar element in the Wiki Mockup (and reinforced by `concept-art2.webp` in the `concept-art2/` folder): deep teal background `#007A7A`, Pop Yellow `#FFD700` primary accent.

---

## Design Principles (from `StyleGuide.tsx`)

> "A bold visual framework inspired by 1950s pop-art, comic aesthetics, and neubrutalist precision. Designed to be the antithesis of corporate minimalism."

| Principle | Implementation |
|---|---|
| No rounded corners | Zero `border-radius` everywhere |
| Thick black borders | `3px solid #1A1A1A` on all structural elements |
| Hard shadows | `box-shadow: 4px 4px 0px #1A1A1A` (no blur radius) |
| High-saturation solid colour | Large blocks of Pop Pink, Teal, Yellow, Orange |
| Visible grid | Halftone dot and checkered patterns as decorative overlays |

---

## Concept Art

The `concept-art/` and `concept-art2/` folders contain the visual identity references:

- **`concept-art/original.webp`** — Hot pink background, pop-art ink style. Basis for the **cream** theme's energy (pink dominant accent).
- **`concept-art2/`** — Teal background, pop-art ink style, with "LLM Maths Almanac" branding. Basis for the **teal** theme.

---

## Run Locally

**Prerequisites:** Node.js (v18+)

```bash
cd styles/
npm install
npm run dev
```

The app opens at `http://localhost:5173`.

> **Note:** The app requires a `GEMINI_API_KEY` in `.env.local` only if you use the AI generation features. The style guide and wiki mockup views work without it.

---

## Relationship to the Hugo Site

| Style App | Hugo Site |
|---|---|
| `src/index.css` → colour tokens, Tailwind config | `data/themes/cream.toml`, `data/themes/teal.toml` |
| `.card-pop` class | `.card-pop` utility in `main.css` |
| `.btn-pop` class | `.btn-primary`, `.btn-secondary` in `main.css` |
| `.bg-halftone`, `.bg-checkered` | `.halftone`, `.checkered` utilities in `main.css` |
| `border-pop` (3px solid black) | `--border: 3px solid var(--color-ink)` in `main.css` |
| `shadow-pop` (4px offset) | `--shadow-pop: 4px 4px 0px var(--color-ink)` in `main.css` |
| Bangers + Space Grotesk fonts | `--font-display` + `--font-body` in `main.css` |

When changing the visual design, update both the style app (for reference) and the Hugo theme CSS (for production). The style app is the source of truth; the Hugo theme is the implementation.
