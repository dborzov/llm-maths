import os
import sys
import re
import argparse
from io import BytesIO
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not found. Please install it with: pip install google-genai")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow package not found. Please install it with: pip install Pillow")
    sys.exit(1)

# Configuration
MODEL_ID = "gemini-2.5-flash-image"
STYLE_PROMPT = (
    "A striking, simple, and cheeky pop-art illustration in the style of an old-school 1950s comic book panel. "
    "Vibe: 'Pulp Fiction' meets 'GTA San Andreas' loading screens. Cool, confident, slightly 'gangster' or 'superhero' aesthetic. "
    "Visual Rules: Neubrutalist. Thick black ink outlines, hard shadows, halftone dots, high contrast, no rounded corners. "
    "NO TEXT, NO HEADERS, NO LABELS, NO CAPTIONS. "
    "Avoid literal tech objects or boring digital artifacts. "
    "Color palette: {theme_colors}. "
    "Composition: Wide cinematic aspect ratio (8:3), 1600x600 equivalent. "
)

# Article-specific visual concepts (manual overrides for better creativity)
VISUAL_CONCEPTS = {
    "02-logit-wager-cover": (
        "High-stakes 1950s casino scene. A 'The Gambler' style character with slicked-back hair and a sharp tuxedo "
        "is coolly pushing a massive pile of glowing neon-blue chips into the center of a roulette table. "
        "The dealer is a calm, mysterious lady in a cocktail dress. Noir lighting, smoke in the air, Pulp Fiction vibe."
    ),
    "03-sixteen-numbers-cover": (
        "A 'The Accountant' character in a sharp, thin-lapel suit and skinny tie (Pulp Fiction style) is opening "
        "a glowing briefcase on a table. Inside the briefcase are rows of perfectly uniform, glowing golden numbers. "
        "He's looking at them with meticulous, slightly menacing precision. Hard shadows, halftone dots."
    ),
    "04-heatmap-that-lied-cover": (
        "The 'Boss' of the city. A powerful, older gangster-style character in a pinstripe suit sits at a massive "
        "mahogany desk in a high-rise office at night, overlooking a neon-drenched city. On his desk is a large, "
        "framed '99% PASS' heatmap with vibrant green and red cells. He is lighting a cigar, looking cynical and powerful."
    ),
    "05-microgpt-unfolded-cover": (
        "A cool, James Dean-style character in a leather jacket and white t-shirt leans against a massive, "
        "60-foot tall neon sign that glows with code snippets. He's holding a blueprint that shows a complex, "
        "beautiful machine labeled 'TRANSFORMER'. Sunset-noir LA vibe, GTA San Andreas loading screen style."
    ),
    "06-eviction-notice-cover": (
        "The Bouncer. A massive, tough-looking bouncer in a black suit stands at a velvet rope of an exclusive "
        "neon club called 'THE CACHE'. He is physically tossing out a group of 'Weak Token' characters into the street, "
        "while a 'Heavy Hitter' character in a flashy suit walks past him into the club. High-contrast noir."
    ),
    # Issue 06 — The Eviction Notice articles
    "01-cold-open": (
        "A sharp-suited 1950s engineer at a gleaming control console is staring at two monitors side-by-side. "
        "The left monitor glows with a triumphant green bar chart. The right monitor shows three rejection letters "
        "in red. His tie is loosened, cigarette dangling. Behind him, a wall of tape drives spins frantically. "
        "Pulp noir, hard shadows, halftone dots."
    ),
    "02-kv-crisis": (
        "A glamorous 1950s executive lady on a transatlantic plane, first-class cabin. She has spread a napkin "
        "across the fold-out tray and is scrawling enormous, alarming calculations with a fountain pen. "
        "Columns of numbers march off the napkin onto her seat, the armrest, the window glass. "
        "The number at the bottom is circled three times. Her expression is controlled alarm. Noir style."
    ),
    "03-heavy-hitter": (
        "A classic 1950s gangster spotlight scene. One massive, broad-shouldered mob boss in a pinstripe suit "
        "stands alone in a bright circle of light, commanding the room. Around him in the shadows: a hundred "
        "identical, pale, faceless figures barely visible. The contrast is extreme — one heavy hitter, "
        "a sea of ghosts. Film noir, halftone dots, ink outlines."
    ),
    "04-copy-paste": (
        "A 1950s office scene: a determined secretary in cat-eye glasses and a pencil skirt is furiously "
        "copy-typing a stack of documents onto carbon paper. As she types certain words they glow hot pink. "
        "Other words stay grey and fade. She's completely focused, one eyebrow raised — she knows which "
        "words matter. Retro-cool, high-contrast, pulp comic style."
    ),
    "05-ghost-state": (
        "A moody 1950s detective's office at night. Filing cabinets line the walls, one drawer slightly ajar "
        "with a faint golden glow spilling out — the hidden state. A detective in a fedora sits at a desk, "
        "studying a glowing dossier labeled 'IMPORTANCE'. Outside the rain-streaked window, ghost-like "
        "silhouettes drift past. The answer was inside all along. Hard shadows, halftone."
    ),
    "06-kvzap": (
        "A 1960s mission-control room at the moment of triumph. A lone scientist in a white coat stands at "
        "a giant wall of blinking lights and toggles. She has just flipped one big red switch. Half the lights "
        "go dark — the cache is compressed. The remaining lights glow brighter, more intense. "
        "She turns to camera with a cool, knowing smile. GTA loading screen energy, pop-art halftone."
    ),
    "07-attention-sparsity": (
        "A 1950s Las Vegas casino roulette table. All the stacked chips — an enormous glittering mountain — "
        "are piled on just two numbers. Every other number on the wheel is empty. The croupier in a bow tie "
        "looks unsurprised, almost bored. A few wide-eyed observers in the background. "
        "Winner-take-all, high contrast, pulp style."
    ),
    "08-contribution-norm": (
        "A 1950s boardroom post-mortem. At the head of the table, an overbearing executive in a loud suit "
        "is gesturing dramatically at a blank whiteboard. His name is on every slide. At the far end, a quietly "
        "exhausted staff engineer with rolled sleeves holds up a model of an entire skyscraper he built alone. "
        "Everyone is looking at the executive. Nobody sees the engineer. Satirical, noir, comic contrast."
    ),
    "09-threshold-topk": (
        "Two 1950s managers at a performance review desk, side by side. The first has a calculator showing '50%' "
        "and is mechanically pointing at half the employees in a line — they look shocked. The second manager "
        "holds a clipboard with a single quality bar drawn on it and is calmly waving away only a few people. "
        "His team looks relieved. Sharp suits, hard shadows, halftone."
    ),
    "10-pruning-landscape": (
        "A 1950s detective's crime-board wall. Photographs of seven research papers pinned with red string, "
        "each labeled with a year. Five of the photos have large red X stamps: REJECTED. Two at the end glow — "
        "one in hot pink, one stamped PRODUCTION READY. The detective stands back, arms crossed, satisfied. "
        "Pulp noir thriller energy."
    ),
    "11-bandwidth-wall": (
        "A 1950s construction worker in a hard hat stands at the base of an enormous brick wall that fills "
        "the entire frame. A tiny door at ground level is labeled '3.35 TB/s'. The worker holds blueprints "
        "and stares up at the wall with resigned understanding. Behind the wall, an H100-shaped machine "
        "sits almost completely idle. Von Neumann's bottleneck made physical. Hard ink, halftone, noir."
    ),
    "02-context-window": (
        "A flashy, 1950s used-car salesman with slicked-back hair and a loud plaid suit stands in front "
        "of a giant, chrome-heavy luxury car with a '1,000,000 TOKENS' sign on the roof. "
        "The car's hood is open, revealing an engine that is actually a massive, smoking, "
        "overloaded pile of heavy metal weights and glowing GPU chips. The salesman is sweating."
    ),
    "03-niah-mechanics": (
        "An overly fancy, James Bond-style lady in high heels and a glamorous dress stands in front of a rustic farm barn. "
        "She is gingerly poking at a literal massive pile of hay with a long, elegant finger, looking visibly repulsed "
        "and far too sophisticated for such a mundane task. High-fashion 'noir' vibe."
    ),
    "04-retrieval-vs-reasoning": (
        "A gritty 1950s private eye detective with a fedora and a lit cigarette stands between two desks. "
        "Desk 1 is labeled 'SCAN': he points a single, bright flashlight at a single red folder. "
        "Desk 2 is labeled 'THINK': a chaotic, tangled spiderweb of red strings connects a dozen different photos, "
        "maps, and notes on a wall. The detective is rubbing his temples in frustration."
    ),
    "05-saturation": (
        "A group of stylish, futuristic-looking characters (the 'Benchmark Gang') are standing on a street corner, "
        "pointing and laughing at a single, rusty, outdated 1950s robot labeled 'NIAH-2023' that is falling apart. "
        "One gang member holds a megaphone. Pulp Fiction street scene vibe."
    ),
    "06-measurement-saturation": (
        "A high-striking fairground 'test your strength' machine in a retro carnival. "
        "A cool-looking character in a leather jacket has just hit the base with a giant hammer, "
        "and the metal puck has flown right off the top of the machine, smashing through the '100%' sign "
        "and into the neon sky. The character has a smug, 'that was easy' smirk."
    ),
    "07-lost-in-the-middle": (
        "A stylish traveler in a trench coat stands at a crossroads in a vast desert. "
        "One sign points to 'START', another to 'END'. But the road between them has vanished into a "
        "giant, bottomless, neon-lit canyon. She's looking into the abyss with a map that says 'YOU ARE HERE' "
        "in the middle of a blank space."
    ),
    "08-needle-to-graph": (
        "A master thief in a sleek suit and sunglasses stands in a high-tech vault. "
        "Instead of lasers, the vault is protected by a glowing, complex 3D spiderweb of nodes and edges. "
        "He is holding a single, glowing 'needle' and looking for the one specific node it plugs into. "
        "Mission Impossible meets GTA."
    ),
    "09-latent-structure": (
        "A futuristic sculptor with glowing goggles in a neon-lit studio. "
        "She is using a laser-chisel to carve a beautiful, complex crystalline structure out of a giant, "
        "rough block of 'data' marble. High-tech craftsmanship vibe."
    ),
    "10-graph-traversal": (
        "A high-speed GTA-style chase scene where a cool sports car navigates a complex, multi-level "
        "neon highway system that looks like a directed graph. The car is drifting through a node at 100mph."
    ),
    "11-context-rot-fix": (
        "A cool 'tech-mechanic' with a gangster aesthetic and a high-tech soldering iron. "
        "He is coolly smoking a cigarette while fixing a glowing, complex machine that is leaking "
        "neon sparks. He's making an 'easy fix' gesture."
    ),
    "12-context-rot": (
        "A once-glamorous 1950s Hollywood mansion that is now overgrown with glowing, neon-green "
        "vines that are literally dissolving the walls. A resident in a bathrobe is looking at a "
        "melting wall in horror. Decay and rot vibe."
    ),
    "13-synth-vs-real": (
        "A Matrix-style diner scene. A character is sitting at a booth where half the room is "
        "gritty 1950s realism and the other half is glowing green binary code and wireframes. "
        "Even their coffee cup is half-real, half-code."
    ),
    "14-agentic-turn": (
        "A group of cool 'Agent' characters in black suits and sunglasses (Reservoir Dogs style) "
        "sitting around a table in a smoky backroom, planning a heist using a massive, 10-foot-long "
        "scroll of 'Context' spread out before them."
    ),
    "15-contamination-drift": (
        "A high-end 1950s laboratory. A scientist in a lab coat and sunglasses is looking at a "
        "petri dish through a microscope. The dish is full of 'contaminated' glowing snippets of "
        "popular internet memes and old essay fragments leaking into the experiment."
    ),
    "16-eval-stack-2026": (
        "A futuristic, neon-lit skyline of a city in 2026. Massive billboards show different "
        "glowing benchmark scores. A cool superhero-like character stands on a rooftop overlooking "
        "the city. Epic finale vibe."
    )
}

THEME_COLORS = {
    "cream": "vibrant pop pink (#FF007F) and warm cream (#FDF5E6)",
    "teal": "golden yellow (#FFD700) and deep dark teal (#007A7A)"
}

def get_article_info(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract front matter (simple regex to avoid PyYAML dependency)
    fm_match = re.search(r'^---(.*?)---', content, re.DOTALL)
    if not fm_match:
        return None, content
    
    fm_text = fm_match.group(1)
    fm = {}
    for line in fm_text.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            fm[key.strip()] = val.strip().strip('"').strip("'")
            
    return fm, content

def generate_hero(article_path, api_key, output_path=None):
    p = Path(article_path)
    if p.name == "_index.md":
        file_stem = f"{p.parent.name}-cover"
    else:
        file_stem = p.stem
        
    fm, _ = get_article_info(article_path)
    if not fm:
        print(f"Error: Could not parse front matter from {article_path}")
        return

    title = fm.get('title', 'Untitled')
    description = fm.get('description', '')
    theme = fm.get('theme', 'cream')
    
    colors = THEME_COLORS.get(theme, THEME_COLORS['cream'])
    
    # Use manual override if available, otherwise fall back to basic description
    illustration_idea = VISUAL_CONCEPTS.get(file_stem, f"A scene depicting {title}: {description}")
    
    full_prompt = (
        STYLE_PROMPT.format(theme_colors=colors) + 
        f"Scene: {illustration_idea} "
        "The illustration must be full-bleed, edge-to-edge, filling the entire frame completely with no borders or letterboxing."
    )
    
    print(f"Using Model: {MODEL_ID}")
    print(f"Generated Prompt: {full_prompt}")
    
    client = genai.Client(api_key=api_key)
    
    print("Requesting image from Nano Banana (Gemini 2.5) API...")
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="21:9",
                    image_size="1K"
                )
            )
        )
    except Exception as e:
        print(f"API Error: {e}")
        return None
    
    if not response.candidates or not response.candidates[0].content.parts:
        print("Error: No image candidates returned.")
        return None

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            img_data = part.inline_data.data
            img = Image.open(BytesIO(img_data))
            
            print(f"Original image size: {img.size}")
            
            # Target 1600x600
            target_w, target_h = 1600, 600
            target_ratio = target_w / target_h
            
            orig_w, orig_h = img.size
            orig_ratio = orig_w / orig_h
            
            if orig_ratio > target_ratio:
                # Original is wider than target - scale by height
                scale = target_h / orig_h
                new_w = int(orig_w * scale)
                img = img.resize((new_w, target_h), Image.Resampling.LANCZOS)
                # Center crop width
                left = (new_w - target_w) // 2
                img = img.crop((left, 0, left + target_w, target_h))
            else:
                # Original is narrower than target - scale by width
                scale = target_w / orig_w
                new_h = int(orig_h * scale)
                img = img.resize((target_w, new_h), Image.Resampling.LANCZOS)
                # Center crop height
                top = (new_h - target_h) // 2
                img = img.crop((0, top, target_w, top + target_h))

            print(f"Final processed size: {img.size}")
            
            if not output_path:
                output_path = f"static/header-illustrations/{file_stem}.webp"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            img.save(output_path, "WEBP", quality=85)
            print(f"Success! Image saved as {output_path}")
            return output_path
    
    print("Error: No image was found in the response parts.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hero illustration for an article using Nano Banana.")
    parser.add_argument("article", help="Path to the article markdown file.")
    parser.add_argument("--api-key", help="Gemini API Key.")
    parser.add_argument("--output", help="Output path for the image.")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set and --api-key not provided.")
        sys.exit(1)
        
    generate_hero(args.article, api_key, args.output)
