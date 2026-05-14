import React from 'react';
import ReactMarkdown from 'react-markdown';
import { FileText, Sparkles } from 'lucide-react';

const sampleMarkdown = `
# Pop-Art Markdown Explosion!

Welcome to the **vivid** world of styled syntax. This page demonstrates how every element of a standard markdown document is transformed into a *bold, pop-art masterpiece*.

## Section Headers: The Teal Anchor
Headers are no longer just text; they are structural anchors with background highlights and thick borders.

### Sub-headers: Orange Accents
Even smaller headers get the royal treatment with orange tones and uppercase tracking.

---

### Lists with Personality

*   **Bullet Points**: Custom square markers with hard shadows.
*   **Vivid Colors**: Each point is designed to stand out.
*   **Consistency**: Maintaining the 1950s comic vibe.

1.  **Numbered Lists**: Large teal badges for every step.
2.  **Scannability**: Numbers are easy to find and read.
3.  **Interaction**: Clean, bold, and functional.

---

### Data in the Grid

| Character | Role | Special Move |
| :--- | :--- | :--- |
| **Pink Panther** | The Lead | Halftone Fade |
| **Teal Titan** | The Muscle | Shadow Slam |
| **Yellow Flash** | The Speed | Neon Dash |

---

### Quotes & Code

> "Design is not just what it looks like and feels like. Design is how it works... especially when it looks like a comic book!"
> — *The Pop-Art Manifesto*

You can also include technical snippets:

\`\`\`javascript
function activatePopArt() {
  const vibe = "MAXIMUM";
  console.log(\`Vibe level: \${vibe}\`);
  return {
    borders: "thick",
    shadows: "hard",
    colors: "vivid"
  };
}
\`\`\`

Inline code like \`const pop = true\` is also styled to match the ink-black aesthetic.
`;

export const MarkdownShowcase = () => {
  return (
    <div className="max-w-4xl mx-auto p-8 pb-32">
      <header className="mb-12 space-y-4 border-b-8 border-pop-black pb-8 relative overflow-hidden">
        <div className="absolute inset-0 bg-halftone opacity-10" />
        <div className="relative z-10 flex items-center gap-4">
          <div className="p-3 bg-pop-orange border-pop shadow-pop">
            <FileText className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-6xl font-display text-pop-black">Syntax Style</h1>
            <p className="text-xl font-bold text-pop-pink uppercase tracking-widest flex items-center gap-2">
              <Sparkles className="w-5 h-5" />
              Markdown Rendered Vividly
            </p>
          </div>
        </div>
      </header>

      <div className="card-pop bg-white p-12 relative">
        <div className="absolute top-4 right-4 text-xs font-black uppercase opacity-20 tracking-tighter">
          Render Engine v1.0 // Pop-Art Edition
        </div>
        <div className="markdown-body">
          <ReactMarkdown>{sampleMarkdown}</ReactMarkdown>
        </div>
      </div>

      <footer className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card-pop bg-pop-teal text-white">
          <h4 className="text-xl font-display mb-2">Why this style?</h4>
          <p className="text-sm font-medium opacity-90">
            Standard markdown is often boring. By applying vivid colors to specific syntax elements, we improve readability through visual hierarchy and make the act of reading a technical document an aesthetic experience.
          </p>
        </div>
        <div className="card-pop bg-pop-yellow">
          <h4 className="text-xl font-display mb-2">Technical Note</h4>
          <p className="text-sm font-medium">
            This uses a custom CSS layer targeting standard HTML tags produced by the markdown parser, ensuring that even dynamically loaded content maintains the brand identity.
          </p>
        </div>
      </footer>
    </div>
  );
};
