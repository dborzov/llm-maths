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
    # Issue 02: Logit Wager
    "02-logit-wager-cover": (
        "High-stakes 1950s casino scene. A 'The Gambler' style character with slicked-back hair and a sharp tuxedo "
        "is coolly pushing a massive pile of glowing neon-blue chips into the center of a roulette table. "
        "The dealer is a calm, mysterious lady in a cocktail dress. Noir lighting, smoke in the air, Pulp Fiction vibe."
    ),
    "01-cold-open-02": (
        "An intense 1930s entomologist in a dusty, dimly lit laboratory, surrounded by specimen jars. "
        "He is meticulously trying to draw a perfect straight line through a glowing, translucent S-curve "
        "on a glass drafting table. Cinematic noir vibe."
    ),
    "02-probit-transform": (
        "A muscular character in a 1950s gym is physically wrestling with a giant, glowing, elastic "
        "bell-curve (Normal distribution). He is trying to straighten it out into a perfectly horizontal "
        "steel bar. Neon brutalist lighting."
    ),
    "03-berksons-gamble": (
        "A contrarian statistician at a smoke-filled 1940s poker table. He is coolly playing a hand "
        "with a much simpler deck of cards (labeled 'LOGIT') while other players are struggling with "
        "complex, oversized 'PROBIT' scrolls. Cheeky smirk."
    ),
    "04-log-odds": (
        "A classic 1950s gambler at a neon-lit racetrack. He is writing in a ledger, adding up simple "
        "integers (log-odds) while other bettors are frantically trying to multiply long decimals. "
        "Confident, 'gangster' cool."
    ),
    "05-logistic-regression": (
        "A scientist in a lab coat is observing a giant, 10-foot tall insect through a high-tech "
        "microscope. The insect is made of glowing pixels and binary code. The scientist is pointing "
        "at its 'gradient' which looks like a beam of light. Pulp Fiction style."
    ),
    "06-softmax-kingdom": (
        "A powerful Queen on a neon throne in a futuristic 1950s palace. Ten subjects (the digits 0-9) "
        "stand before her, all adjusting their heights to ensure they sum to exactly one. "
        "Striking composition, high contrast."
    ),

    # Issue 03: Sixteen Numbers
    "03-sixteen-numbers-cover": (
        "A 'The Accountant' character in a sharp, thin-lapel suit and skinny tie (Pulp Fiction style) is opening "
        "a glowing briefcase on a table. Inside the briefcase are rows of perfectly uniform, glowing golden numbers. "
        "He's looking at them with meticulous, slightly menacing precision. Hard shadows, halftone dots."
    ),
    "01-cold-open-03": (
        "A massive, mountain-sized model made of trillions of numbers is being crushed and squeezed "
        "into a tiny, elegant, glowing chrome suitcase by a team of futuristic characters in 1950s jumpsuits. "
        "Epic scale, sunset-noir."
    ),
    "02-numbers-in-boxes": (
        "A character in a stylish 1950s hardware store, pointing at a giant wall of small, identical "
        "neon-lit boxes. Each box contains a single floating number. The character is looking for "
        "the 'perfect' box. GTA loading screen style."
    ),
    "02a-absmax": (
        "A character at a giant industrial scale in a neon-lit shipyard. They are pointing at the "
        "heaviest metal weight (the max) while a massive hydraulic press crushes all the other "
        "smaller weights into dust. High impact."
    ),
    "02b-zero-point": (
        "A technician in a high-tech laboratory carefully adjusting two delicate, glowing knobs "
        "(labeled 'SCALE' and 'OFFSET') on a massive, complex machine that is recalibrating "
        "a sequence of numbers. Scientific noir."
    ),
    "04-rate-distortion": (
        "A magnificent suspension bridge spanning a deep, foggy neon canyon. One side of the bridge "
        "is labeled 'FILE SIZE', the other 'FIDELITY'. A cool character is walking across the 'bend' "
        "where the bridge is most stable. Cinematic wide shot."
    ),
    "05-geometry-of-weights": (
        "A stylish character in a modern art museum, standing before a series of abstract, "
        "geometric sculptures that represent mathematical distributions. High-contrast lighting, "
        "noir shadows."
    ),
    "06-outliers": (
        "A stylish character in a leather jacket and sunglasses (the outlier) coolly crashing a very polite, "
        "uniform party of identical people in grey suits. He's lighting a cigarette and the whole room "
        "is looking at him in shock. Pulp Fiction vibe."
    ),
    "06b-llm-int8-deep": (
        "A double-track neon railway at night. Most trains (the common values) go down one track, "
        "while a few high-speed, glowing express trains (the outliers) roar down a separate, "
        "elevated track. High speed, high contrast."
    ),
    "07-taylor-and-hessians": (
        "A surveyor in a 1950s hat measuring the curvature of a neon-drenched landscape with a "
        "complex, glowing tripod device. He is looking for the 'smooth' spots. Cinematic noir."
    ),
    "08-brain-surgery": (
        "A character in surgical scrubs carefully using a tiny, high-tech neon needle to prune "
        "a single glowing wire from a massive, complex artificial brain on a table. Intense focus, "
        "hard shadows."
    ),
    "09-method-family-tree": (
        "A classic 1950s family tree hanging on a wood-paneled wall, but the family members are "
        "all stylized, gangster-like variants of quantization algorithms. The 'Godfather' is at the top."
    ),
    "10-kv-cache": (
        "A character struggling to carry a giant, overflowing glowing bucket of 'KV CACHE' data "
        "while their other luggage (labeled 'WEIGHTS') is tiny and weightless. 'The Burden' vibe."
    ),
    "11-calibration-and-blocks": (
        "A construction worker in a stylish 1950s outfit meticulously stacking identical "
        "neon blocks, each with a small glowing scale label. He's using a level to make sure "
        "they're perfect. GTA style."
    ),
    "12-hardware-horizon": (
        "A character looking out at a futuristic neon skyline where the skyscrapers are shaped "
        "like massive H100 and Blackwell GPU architectures. A 'New World' vibe."
    ),
    "13-calibration-survey": (
        "An explorer in a tropical neon jungle, cataloging a series of strange, glowing "
        "'Calibration' plants with a high-tech scanner. Scientific adventure vibe."
    ),
    "14-compression-roots": (
        "A dusty 1940s archive room where engineers in shirts and ties are working on massive "
        "vacuum-tube computers and early magnetic tapes. A 'Heritage' noir vibe."
    ),
    "15-kv-method-family": (
        "A stylish 1950s office with a large wall map connecting various 'KV Cache' crime families "
        "with red string. A detective is pointing at the 'KIVI' family. Noir intrigue."
    ),
    "16-kv-distribution": (
        "A detective in a dark interrogation room with two suspects (labeled 'K' and 'V'). "
        "On the chalkboard behind them are their unique 'outlier' fingerprints. Intense drama."
    ),
    "17-rotations": (
        "A stylish character in a mid-century modern living room, coolly rotating a giant, "
        "abstract geometric painting on the wall to hide a secret safe. 'The Heist' vibe."
    ),
    "18-quantization-axes": (
        "A character standing at the center of a giant, multi-dimensional neon compass. "
        "The axes point to 'CHANNEL', 'TOKEN', 'BLOCK', and 'TENSOR'. He is choosing his path."
    ),

    # Issue 04: The Heatmap That Lied
    "04-heatmap-that-lied-cover": (
        "The 'Boss' of the city. A powerful, older gangster-style character in a pinstripe suit sits at a massive "
        "mahogany desk in a high-rise office at night, overlooking a neon-drenched city. On his desk is a large, "
        "framed '99% PASS' heatmap with vibrant green and red cells. He is lighting a cigar, looking cynical and powerful."
    ),
    "01-cold-open": (
        "A sneaky-looking man in a long trench coat stands in a dark, neon-lit rainy alleyway. "
        "He is opening one side of his coat to reveal a glowing, neon-green-and-red heatmap grid "
        "pinned to the lining. He looks like a black-market dealer. "
        "GTA San Andreas loading screen style, very cool and mysterious."
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
    ),

    # Issue 05: MicroGPT Unfolded
    "05-microgpt-unfolded-cover": (
        "A cool, James Dean-style character in a leather jacket and white t-shirt leans against a massive, "
        "60-foot tall neon sign that glows with code snippets. He's holding a blueprint that shows a complex, "
        "beautiful machine labeled 'TRANSFORMER'. Sunset-noir LA vibe, GTA San Andreas loading screen style."
    ),
    "01-cold-open-05": (
        "A character in a lab coat holding a tiny, intricate clockwork mechanism in their hand. "
        "The mechanism has exactly 60 glowing gears and parts. They are looking at it with wonder. "
        "High-contrast, noir background."
    ),
    "02-state-dict": (
        "A character in a dark 1950s filing room opening a drawer labeled 'STATE DICT'. "
        "Inside are exactly nine perfectly organized, glowing folders. Noir lighting, hard shadows."
    ),
    "03-embeddings": (
        "Two separate glowing conveyor belts merging into one. One belt carries stylized people (tokens), "
        "the other carries glowing hats (positions). After the merge, everyone is wearing a hat. "
        "GTA loading screen style."
    ),
    "04-linear": (
        "A character at a giant industrial press in a neon-lit factory. The press is transforming "
        "a flat piece of glowing metal into a specific, complex shape. High-impact noir."
    ),
    "05-rmsnorm": (
        "A cool-looking technician with a pair of scissors, coolly cutting a wire labeled 'SUBTRACTION' "
        "out of a complex, glowing circuit board. He has a 'less is more' smirk."
    ),
    "06-qkv-projections": (
        "A character standing between three different glowing mirrors (labeled Q, K, and V). "
        "Each mirror reflects a different, specialized aspect of the same character. Cinematic wide shot."
    ),
    "07-softmax": (
        "A character in a 1950s control room, adjusting a giant 'MAX' slider on a massive thermometer "
        "to keep the glowing liquid from boiling over. Scientific noir."
    ),
    "08-attention": (
        "A cool-looking sniper with a high-tech scope (the query) aiming at a single, bright "
        "target (the key) while the rest of the world is blurred and out of focus. Noir action vibe."
    ),
    "09-multi-head": (
        "A stylish character with several translucent, glowing heads (like a cool noir hydra), "
        "each looking in a different direction and focusing on different things simultaneously."
    ),
    "10-residual-stream": (
        "A character in a sleek kayak, paddling down a glowing, neon-blue 'stream'. Information is "
        "being poured into and scooped out of the stream as they move. Sunset-noir vibe."
    ),
    "11-mlp-block": (
        "A character in a neon-lit balloon shop. They are inflating a long balloon (fatten), "
        "then squeezing the middle (relu), then letting it shrink back down (skinny). Cheeky vibe."
    ),
    "12-activations": (
        "A cool character at a neon-lit bar, being served a tray of colorful, glowing cocktails "
        "by a mysterious bartender. Each drink is a different 'activation'. Pulp Fiction vibe."
    ),
    "13-kv-cache": (
        "A character meticulously putting glowing folders into an ever-growing, infinite "
        "filing cabinet labeled 'CACHE'. Hard shadows, high contrast."
    ),
    "14-prefill-decode": (
        "Two racers on a neon track. One is a massive giant who takes one huge, ground-shaking "
        "step (prefill); the other is a fast sprinter taking hundreds of tiny, rapid steps (decode)."
    ),
    "15-sampling": (
        "A character at a high-stakes dice table in a neon casino. They are using different "
        "glowing 'weights' on the dice to control the probability. Noir cool."
    ),
    "16-full-forward": (
        "A character at the controls of a massive, beautiful Rube Goldberg machine. "
        "They are watching a single glowing ball (the input) roll through every complex part "
        "of the machine. Epic scale."
    ),
    "17-kv-axes": (
        "A technician in a high-tech workshop, using a 3D holographic coordinate system to "
        "prune and shrink a complex, glowing structure. GTA loading screen style."
    ),
    "18-gqa": (
        "Several cool-looking characters (the queries) all sharing a single set of high-tech "
        "headphones connected to a glowing box labeled 'K/V'. Efficiency vibe."
    ),
    "19-mla": (
        "A master thief with a 'shrinking ray' gun, compressing a massive, glowing safe "
        "into a tiny, portable 'latent' version. He's looking at the camera with a wink."
    ),
    "20-sliding-window": (
        "A traveler on a high-speed neon train, looking through a narrow, sliding window. "
        "They can only see a specific 'window' of the landscape at any time. Cinematic wide."
    ),
    "21-ssm-hybrids": (
        "A cool character at a futuristic DJ console, seamlessly switching between 'ATTENTION' "
        "and 'RECURRENCE' sliders to mix a glowing beat. High-energy noir."
    ),

    # Issue 06: Eviction Notice
    "06-eviction-notice-cover": (
        "The Bouncer. A massive, tough-looking bouncer in a black suit stands at a velvet rope of an exclusive "
        "neon club called 'THE CACHE'. He is physically tossing out a group of 'Weak Token' characters into the street, "
        "while a 'Heavy Hitter' character in a flashy suit walks past him into the club. High-contrast noir."
    ),
    "01-cold-open-06": (
        "A sharp-suited 1950s engineer at a gleaming control console is staring at two monitors side-by-side. "
        "The left monitor glows with a triumphant green bar chart. The right monitor shows three rejection letters "
        "in red. His tie is loosened, cigarette dangling. Behind him, a wall of tape drives spins frantically."
    ),
    "02-kv-crisis": (
        "A glamorous 1950s executive lady on a transatlantic plane, first-class cabin. She has spread a napkin "
        "across the fold-out tray and is scrawling enormous, alarming calculations with a fountain pen. "
        "Columns of numbers march off the napkin onto her seat, the armrest, the window glass."
    ),
    "03-heavy-hitter": (
        "A classic 1950s gangster spotlight scene. One massive, broad-shouldered mob boss in a pinstripe suit "
        "stands alone in a bright circle of light, commanding the room. Around him in the shadows: a hundred "
        "identical, pale, faceless figures barely visible."
    ),
    "04-copy-paste": (
        "A 1950s office scene: a determined secretary in cat-eye glasses and a pencil skirt is furiously "
        "copy-typing a stack of documents onto carbon paper. As she types certain words they glow hot pink."
    ),
    "05-ghost-state": (
        "A moody 1950s detective's office at night. Filing cabinets line the walls, one drawer slightly ajar "
        "with a faint golden glow spilling out — the hidden state. A detective in a fedora sits at a desk studying a dossier."
    ),
    "06-kvzap": (
        "A 1960s mission-control room at the moment of triumph. A lone scientist in a white coat stands at "
        "a giant wall of blinking lights and toggles. She has just flipped one big red switch."
    ),
    "07-attention-sparsity": (
        "A 1950s Las Vegas casino roulette table. All the stacked chips — an enormous glittering mountain — "
        "are piled on just two numbers. Every other number on the wheel is empty."
    ),
    "08-contribution-norm": (
        "A 1950s boardroom post-mortem. At the head of the table, an overbearing executive in a loud suit "
        "is gesturing dramatically at a blank whiteboard. His name is on every slide."
    ),
    "09-threshold-topk": (
        "Two 1950s managers at a performance review desk, side by side. The first has a calculator showing '50%' "
        "and is mechanically pointing at half the employees in a line — they look shocked."
    ),
    "10-pruning-landscape": (
        "A 1950s detective's crime-board wall. Photographs of seven research papers pinned with red string, "
        "each labeled with a year. Five of the photos have large red X stamps: REJECTED."
    ),
    "11-bandwidth-wall": (
        "A 1950s construction worker in a hard hat stands at the base of an enormous brick wall that fills "
        "the entire frame. A tiny door at ground level is labeled '3.35 TB/s'."
    ),

    # Issue 07: The Sparse Lab
    "07-sparse-lab-cover": (
        "A shadowy research laboratory at midnight in a high-rise building, venetian blinds casting noir stripes "
        "across the walls. A lone researcher in a sharp double-breasted suit pins the final piece of a massive "
        "attention-architecture diagram to a corkboard surrounded by rejected competitor papers stamped 'FAILED'. "
        "A single neon spotlight glows on the breakthrough. Outside the window: a neon city skyline. Spy-thriller "
        "meets GTA loading screen."
    ),
    "01-cold-open-07": (
        "A 1950s newsroom at midnight. The ticker-tape machine explodes into life. A reporter in rolled-up "
        "shirtsleeves rips the tape, reads it with widening eyes — his cigarette drops. On the tape: 'API PRICE: -50%'. "
        "Through the window: a neon city skyline. Every journalist frozen in shock. The biggest scoop of the year, "
        "dropped at midnight."
    ),
    "02-mla-rewind": (
        "A 1950s couture fashion designer at a massive cutting table makes the first decisive scissor-cut into "
        "an enormous bolt of fabric labeled 'MEMORY'. Thirty percent of the fabric falls away cleanly as offcuts. "
        "She looks up with cool, precise satisfaction: this was always obvious. High-fashion noir studio."
    ),
    "03-quadratic-wall": (
        "A lone athlete in 1950s track gear has just run full-speed into a massive, perfectly vertical concrete wall "
        "that fills the entire frame. Stamped on the wall: 'T²'. He's staring up at it, hands on knees. Every runner "
        "who tried this path hit the same wall. Cinematic wide shot, hard noir shadows."
    ),
    "04-sparse-detour": (
        "A weary detective in a trench coat stands before a corkboard covered in ten years of pinned research papers — "
        "each one stamped with a large red 'FAILED'. Empty coffee cups surround him. Dozens of red strings lead nowhere. "
        "He lights another cigarette and stares at the wall. A decade of dead ends. Cold-case forensic noir."
    ),
    "05-nsa-paper": (
        "A master architect at a 1950s drafting room unrolls a massive blueprint across the table. The design shows "
        "three distinct wings labeled 'COMPRESS', 'SELECT', 'SLIDE'. Engineers crowd around, stunned — this solves "
        "everything. The architect has a cool, I-told-you-so expression. Cinematic hard shadows."
    ),
    "06-lightning-indexer": (
        "A 1950s telegraph operator at midnight — the machine suddenly roars to life at full speed. She reads "
        "the incoming tape with wide eyes: prices are collapsing in real-time. A clock on the wall strikes twelve. "
        "Outside: neon lightning flashes over a city skyline. Speed, drama, irreversibility. Pulp Fiction energy."
    ),
    "07-csa": (
        "A 1950s factory floor. A mechanical press runs at full speed — for every four items entering the conveyor "
        "belt, it stamps them into a single dense cube. Three inputs disappear, one compressed output emerges. "
        "The supervisor watches with clipboard satisfaction. Industrial precision, GTA loading screen aesthetic."
    ),
    "08-hca": (
        "A 1950s archaeologist at a museum display table, methodically condensing 128 unearthed artifacts into "
        "a single dense, labeled display case. He distills an entire excavation into one representative object. "
        "'128 into 1,' his notepad reads. Sparse and contemplative — the extreme version. Noir museum atmosphere."
    ),
    "09-hybrid-pattern": (
        "A 1950s jazz orchestra conductor in a neon-lit concert hall, alternating his baton between two sections — "
        "'CSA' brass on the left, 'HCA' strings on the right — in precise alternating sequence. The interleaving "
        "creates something richer than either section alone. Cinematic wide, viewed from the back of the hall."
    ),
    "10-decoupling": (
        "A 1950s wrecking-ball operator swinging a massive chrome ball into a ceiling labeled 'QUADRATIC WALL'. "
        "The ceiling shatters — fragments rain down, and beyond the rubble: an infinite open sky. The operator "
        "is grinning. Boss-chapter finale energy. Epic scale, triumph, hard noir shadows."
    ),
    "11-attention-compute": (
        "A sharp-suited 1950s accountant at a massive mechanical adding machine, the paper receipt unspooling "
        "across the floor and down the hall. He's jabbing one line with a sharp finger: 'QK^T' — ten times longer "
        "than everything else on the receipt. Absurdist bureaucratic noir, precision arithmetic."
    ),
    "12-token-compression": (
        "A 1950s industrial juice-press operator feeding a long chain of individual fruit pieces into a giant "
        "compression machine. Out the other end: a single dense, glowing cube. A sign reads '4 INTO 1'. "
        "The operator watches with satisfaction. Factory noir, high contrast."
    ),
    "13-topk-and-routing": (
        "A 1950s judge slams a gavel: from a lineup of twenty, exactly three are selected for trial. The rest "
        "are dismissed with a wave. But the jury box is in uproar — the selection cannot be appealed through "
        "normal legal channels, the decision is irreversible. Dramatic courtroom noir."
    ),
    "14-indexer-pattern": (
        "A 1950s talent scout at the back of a smoky nightclub, quietly watching acts and making tiny tick marks "
        "on a notepad. He hands his short list to the big boss at the front table — who only auditions the "
        "top-scored performers. Efficient two-tier selection. Cool, precise, understated noir."
    ),
    "15-mixed-precision-kv": (
        "A 1950s master jeweler at three sorting trays — platinum on the left (finest), gold in the center, "
        "silver on the right. Each gemstone gets exactly the tray its grade deserves, assessed with jeweler's loupes. "
        "Meticulous sorting of different-value materials. Noir gem-dealer precision."
    ),
    "16-empirical-sparsity": (
        "A 1950s nightclub — a hundred people in the audience, but only one performer under a single spotlight "
        "absorbs all the applause. The rest of the stage is dark and empty. Power law: five percent of the "
        "performers, ninety percent of the attention. High-contrast lighting, dramatic spotlight."
    ),
    "17-attention-sink": (
        "A 1950s plumbing engineer watching water flow through a complex pipe junction. Almost all the water "
        "funnels into one massive outlet labeled 'SINK' in the corner of the system. The engineer studies it "
        "carefully, noting it on his clipboard. Not a bug — a feature. Pipes and noir."
    ),
    "18-inverse-rope": (
        "A 1950s telephone exchange operator who has just spotted two lines crossed — the same wire is serving "
        "two signals (K and V) simultaneously. She makes one swift, precise cable swap with a single hand motion "
        "and the system snaps back into order. The one-line fix. Telephone switchboard noir."
    ),
    "19-cache-engineering": (
        "A 1950s railroad switching yard manager directing three separate tracks: fast express trains on the right "
        "(raw tokens), mid-speed commuter cars in the center (4× compressed), slow heavy freight on the left "
        "(128× compressed). He dispatches each with a different signal flag. Systems logistics noir, cinematic wide."
    ),

    # Issue 08: Anatomy of a Token
    "08-anatomy-of-a-token-cover": (
        "A glamorous 1950s anatomist in a lab coat over a pinstripe suit, standing over a dissection table "
        "lit by a single harsh spotlight. On the table: a single glowing token the size of a human, fully "
        "opened and labeled inside. She gestures with an elegant pointer, cool and clinical. Medical precision "
        "meets GTA loading screen style."
    ),
    "01-cold-open-08": (
        "A single small package on a 1950s factory receiving dock, the shipping label reading 'HTTP REQUEST'. "
        "Behind it: a cathedral-sized factory stretching to the horizon labeled 'H200 INFERENCE ENGINE'. "
        "The scale contrast is staggering — one tiny package, one enormous machine waiting to process it. "
        "Cinematic wide, noir industrial."
    ),
    "02-gpu-anatomy": (
        "A 1950s anatomy lecturer stands at a massive chalkboard showing a cutaway schematic of a machine "
        "instead of a human body. 132 identical compartments are labeled 'SM'. He traces the interconnections "
        "with a pointer, explaining the floor plan to stunned colleagues. Engineering-as-dissection noir."
    ),
    "03-memory-hierarchy": (
        "A 1950s archaeologist standing in an Egyptian desert before a massive four-level stone pyramid. "
        "Tiny gold peak at the top ('REGISTERS — blazing fast'), vast stone base at the bottom ('HBM — slow'). "
        "He gestures upward with a torch. The trade-off is architectural and ancient. Desert noir, cinematic wide."
    ),
    "04-roofline": (
        "A 1950s factory architect inside a cavernous building, pointing at a sloped roofline overhead. "
        "She's marking a line on the floor with chalk: 'RIDGE POINT'. 'To the left: memory-bound. "
        "To the right: compute-bound.' She's explaining the building's fate in one gesture. Architectural industrial noir."
    ),
    "05-cuda-graphs": (
        "A 1950s Hollywood director on a soundstage. First take: twelve technicians scramble chaotically. "
        "He shouts 'CUT!' then presses a single button — the entire crew replays the scene automatically, "
        "perfectly synchronized. One button for a thousand actions. The director's confident smirk. GTA loading screen style."
    ),
    "06-flash-attention": (
        "A 1950s mental prodigy — a woman in a sharp suit — sitting at a bare table, eyes closed, computing "
        "an enormous calculation entirely in her head. Other scientists fill ten blackboards around her with "
        "partial equations. She opens her eyes and writes just one line: the final answer. Never touched paper until the end."
    ),
    "07-prefill-vs-decode": (
        "Two 1950s athletes on a track side by side: a massive weightlifter (labeled 'PREFILL') taking one "
        "single earth-shaking stride; a sleek sprinter (labeled 'DECODE') taking hundreds of tiny rapid steps. "
        "Same race, completely different gaits. Perfectly contrasted. Cinematic wide, GTA loading screen style."
    ),
    "08-continuous-batching": (
        "A 1950s assembly-line manager at a non-stop conveyor belt factory. He's pulling finished parts off "
        "without stopping the belt and slapping new ones on in the same motion. Behind him: the old system "
        "where the belt grinds to a halt — crossed out in chalk. 'Keep it moving.' Clever efficiency noir."
    ),
    "09-kv-fragmentation": (
        "A 1950s property developer walking through a massive warehouse chopped into hundreds of oddly-shaped, "
        "incompatible storage compartments. Most are empty but nothing useful fits the gaps. She gestures at "
        "the wasted space with visible frustration. Memory fragmentation as real-estate disaster. Noir."
    ),
    "10-paged-attention": (
        "A 1960s IBM mainframe operator in thick glasses connecting cables at a massive switchboard. "
        "Top panel: logical order (VIRTUAL). Bottom panel: physical machine chaos (REAL). The cables cross "
        "and redirect — she knows exactly where everything lands. 'Borrowing from 1965.' Elegant cable-management noir."
    ),
    "11-block-manager": (
        "A 1950s warehouse foreman with a clipboard at a block-storage yard, assigning numbered slots to "
        "incoming pallets, chalk-marking reference counts on each. His filing system is flawless — every "
        "block accounted for, free slots known instantly. Efficient, calm, noir logistics precision."
    ),
    "12-prefix-caching": (
        "A 1950s librarian who has pre-assembled the first three chapters of every book as starter kits on "
        "a special shelf. When a reader walks in, she's already handing them chapter four. "
        "'The prologue is ready,' she says with a knowing smile. Cache hit as librarian elegance. Noir efficiency."
    ),
    "13-chunked-prefill": (
        "A 1950s short-order chef slicing one enormous steak into equal portions, grilling them one at a time "
        "while smaller orders interleave on the same grill. Nobody monopolizes the cook. He surveys the diner "
        "with calm authority — fair, fast, no single customer hogging the heat. Pulp Fiction diner efficiency."
    ),
    "14-scheduler": (
        "A 1950s government budget director at a massive mahogany desk, distributing tokens from a single fat "
        "envelope across a row of competing requests. Calm, powerful, impartial — each department gets exactly "
        "its allocation. Hard shadows, noir bureaucracy as precision art."
    ),
    "15-speculative-decoding": (
        "A 1950s heist crew: a fast sketch artist rapidly draws 32 rough scene thumbnails while the master "
        "planner reviews all 32 in one sweep. 'If 28 are right, we proceed.' Speed through parallel speculation. "
        "The sketch artist works in a blur; the master verifies in seconds. Cool noir workflow."
    ),
    "16-tp-pp": (
        "A 1950s moving crew using a crane to split a massive safe into two halves — one team hoists the "
        "top half across the street, another takes the bottom half. Both halves will reunite at the destination. "
        "Coordinated distributed heist. Powerful noir logistics, cinematic wide."
    ),
    "17-disagg-pd": (
        "Two separate 1950s factories connected by a pneumatic tube system. Factory A ('PREFILL PLANT') is "
        "a massive compute-heavy foundry. Factory B ('DECODE WORKS') is a sleek, high-speed assembly line. "
        "Between them: bronze tubes firing glowing canisters of KV data at high velocity. Industrial disaggregation noir."
    ),
    "18-full-anatomy": (
        "A 1950s surgeon stepping back from a completed operation, gesturing over a patient whose body is "
        "now fully labeled — every organ named, every nerve mapped, every system annotated in chalk. "
        "The full picture, at last. He looks at the camera with quiet triumph. Grand finale, complete mastery. Noir medical."
    ),
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

    # Handle multiple cold-opens by appending issue number to slug if needed
    issue_num = fm.get('issue', '')
    lookup_key = file_stem
    if file_stem == "01-cold-open" and issue_num:
        lookup_key = f"01-cold-open-{int(issue_num):02d}"

    title = fm.get('title', 'Untitled')
    description = fm.get('description', '')
    theme = fm.get('theme', 'cream')
    
    colors = THEME_COLORS.get(theme, THEME_COLORS['cream'])
    
    # Use manual override if available, otherwise fall back to basic description
    illustration_idea = VISUAL_CONCEPTS.get(lookup_key, f"A scene depicting {title}: {description}")
    
    full_prompt = (
        STYLE_PROMPT.format(theme_colors=colors) + 
        f"Scene: {illustration_idea} "
        "The illustration must be full-bleed, edge-to-edge, filling the entire frame completely with no borders or letterboxing."
    )
    
    print(f"Using Model: {MODEL_ID}")
    print(f"Generated Prompt: {full_prompt}")
    
    client = genai.Client(api_key=api_key)
    
    print(f"Requesting image for {lookup_key} from Nano Banana (Gemini 2.5) API...")
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
                output_path = f"static/header-illustrations/{lookup_key}.webp"
            
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
