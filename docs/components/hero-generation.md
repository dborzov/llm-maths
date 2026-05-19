# Hero Image Generation Workflow

This project uses a custom AI-driven workflow to generate "Hero" (header) illustrations that match our pop-art neubrutalist visual style.

## The Vibe (Art Direction)

Our illustrations follow a specific aesthetic: pixel-grainy 1950s comic book panel with larger-than-life scene.

### Visual Rules
- **Neubrutalist:** Thick black ink outlines, hard shadows (no gradients), and halftone dots.
-**Cool badasses**. Visually appealing characters. Think "Pulp Fiction" meets "GTA San Andreas" loading screens.
- **No Rounded Corners:** Everything should feel sharp and structural.
- **High Contrast:** Leverage the theme-specific palettes.
- **Cinematic Framing:** Wide, edge-to-edge "full-bleed" composition (21:9 aspect ratio).
- **No Text:** Explicitly forbid the AI from adding headers, labels, captions, or digital artifacts.

### Thematic Guidelines
- **Cheeky & Inventive:** Don't depict literal tech objects (no "glowing brains", no "blue circuit boards").
- **Human-Centric Noir:** Use larger-than-life 1950s/60s characters (detectives, gangsters, high-fashion ladies, gamblers) to personify technical concepts.
- - **No Computers, Robots, future Tech stuff**. Its a metaphor. Think a relevant screenshot from 1950s movie. 
- **Metaphor over Literalism.**
- Examples: 
    - Instead of "Retrieval," you can show a gritty detective with a flashlight.
    - Instead of "Quantization," show an accountant with a briefcase of golden numbers.
    - Instead of "NIAH," show a fancy lady repulsed by a literal pile of hay.

## Technical Workflow

### 1. The Script
The generation is handled by `scripts/generate_hero.py`. It uses the **Google GenAI "Nano Banana"** (Gemini 2.5 Flash Image) API.

### 2. Manual Overrides
To ensure high creativity, the script uses a `VISUAL_CONCEPTS` dictionary. **Always add a concept here before generating.** 
- Keys should be the article slug (e.g., `04-retrieval-vs-reasoning`) or the issue slug with `-cover` (e.g., `04-long-context-bench-cover`).
- Values should be a descriptive, mood-focused scene.

### 3. Running the Generator
Ensure your `GOOGLE_API_KEY` is set in your environment, then run:

```bash
uv run scripts/generate_hero.py content/comicbook/NN-slug/article.md
```

The script will:
1. Detect the theme (Cream/Teal) and set the color palette.
2. Request a **21:9** image from the API.
3. Perform a high-quality **center-crop to 800×300** (see sizing note below).
4. Save the output as a **WebP** file in `static/header-illustrations/`.

### 4. Updating Front Matter
Once the image is generated, update the article's front matter:

```yaml
header: slug-of-article.webp
```

## Maintenance

### Target Image Size
Hero images are decorative ("visual sugar") — high fidelity is not needed.

- **Target dimensions:** 800 × 300 px (8:3 aspect ratio)
- **Format:** WebP, quality 80
- **Max file size target:** ~100 KB

If you need to manually resize/re-encode existing images, use the Pillow script pattern:
```python
from PIL import Image
with Image.open('in.webp') as img:
    img.convert('RGB').resize((800, 300), Image.LANCZOS).save('out.webp', 'WEBP', quality=80, method=6)
```

### Colors
- **Cream:** Vibrant Pop Pink (`#FF007F`) and warm Cream (`#FDF5E6`).
- **Teal:** Golden Yellow (`#FFD700`) and deep Dark Teal (`#007A7A`).
