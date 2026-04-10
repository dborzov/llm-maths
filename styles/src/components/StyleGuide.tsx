import React from 'react';
import { motion } from 'motion/react';
import { Palette, Type, Square, MousePointer2, Layout, Layers } from 'lucide-react';

const ColorSwatch = ({ color, name, hex }: { color: string; name: string; hex: string }) => (
  <div className="flex flex-col gap-2">
    <div 
      className={`h-24 w-full border-pop shadow-pop`} 
      style={{ backgroundColor: color }}
    />
    <div>
      <p className="font-bold uppercase text-sm">{name}</p>
      <p className="font-mono text-xs opacity-60">{hex}</p>
    </div>
  </div>
);

export const StyleGuide = () => {
  return (
    <div className="max-w-6xl mx-auto p-8 space-y-16 pb-32">
      <header className="space-y-4 border-b-4 border-pop-black pb-8">
        <h1 className="text-7xl font-display text-pop-pink drop-shadow-[4px_4px_0px_rgba(0,0,0,1)]">
          Wiki Style Guide
        </h1>
        <p className="text-xl font-medium max-w-2xl">
          A bold visual framework inspired by 1950s pop-art, comic aesthetics, and neubrutalist precision. 
          Designed to be the antithesis of corporate minimalism.
        </p>
      </header>

      {/* Color Palette */}
      <section className="space-y-8">
        <div className="flex items-center gap-3">
          <Palette className="w-8 h-8" />
          <h2 className="text-4xl">Color Palette</h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
          <ColorSwatch color="#FF007F" name="Pop Pink" hex="#FF007F" />
          <ColorSwatch color="#00A8A8" name="Pop Teal" hex="#00A8A8" />
          <ColorSwatch color="#FFD700" name="Pop Yellow" hex="#FFD700" />
          <ColorSwatch color="#FF8C00" name="Pop Orange" hex="#FF8C00" />
          <ColorSwatch color="#FDF5E6" name="Pop Cream" hex="#FDF5E6" />
          <ColorSwatch color="#1A1A1A" name="Ink Black" hex="#1A1A1A" />
        </div>
      </section>

      {/* Typography */}
      <section className="space-y-8">
        <div className="flex items-center gap-3">
          <Type className="w-8 h-8" />
          <h2 className="text-4xl">Typography</h2>
        </div>
        <div className="grid md:grid-cols-2 gap-12">
          <div className="card-pop space-y-4">
            <p className="text-xs uppercase font-bold opacity-50">Display Font: Bangers</p>
            <h3 className="text-6xl font-display text-pop-teal">POW! ZAP! WIKI!</h3>
            <p className="text-sm">Used for high-impact accents, brand elements, and decorative headings.</p>
          </div>
          <div className="card-pop space-y-4">
            <p className="text-xs uppercase font-bold opacity-50">UI Font: Space Grotesk</p>
            <h3 className="text-4xl font-sans">The Quick Brown Fox</h3>
            <p className="font-sans leading-relaxed">
              Used for all functional UI elements, navigation, and secondary headings. 
              Its geometric yet quirky nature complements the pop-art aesthetic.
            </p>
          </div>
        </div>
      </section>

      {/* Interactive Elements */}
      <section className="space-y-8">
        <div className="flex items-center gap-3">
          <MousePointer2 className="w-8 h-8" />
          <h2 className="text-4xl">Interactive States</h2>
        </div>
        <div className="grid md:grid-cols-4 gap-6">
          <div className="space-y-4">
            <p className="text-xs font-bold uppercase opacity-50">Default</p>
            <button className="btn-pop bg-pop-pink text-white w-full">Action</button>
          </div>
          <div className="space-y-4">
            <p className="text-xs font-bold uppercase opacity-50">Hover</p>
            <button className="btn-pop bg-pop-pink text-white w-full hover:bg-pop-teal transition-colors">Hover Me</button>
          </div>
          <div className="space-y-4">
            <p className="text-xs font-bold uppercase opacity-50">Active / Pressed</p>
            <button className="btn-pop bg-pop-pink text-white w-full translate-x-[2px] translate-y-[2px] shadow-none">Pressed</button>
          </div>
          <div className="space-y-4">
            <p className="text-xs font-bold uppercase opacity-50">Disabled</p>
            <button className="border-pop px-6 py-2 font-bold uppercase tracking-wider bg-gray-200 text-gray-400 cursor-not-allowed w-full" disabled>Locked</button>
          </div>
        </div>
      </section>

      {/* Recurring Motifs */}
      <section className="space-y-8">
        <div className="flex items-center gap-3">
          <Layers className="w-8 h-8" />
          <h2 className="text-4xl">Recurring Motifs</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="relative h-48 card-pop overflow-hidden group">
            <div className="absolute inset-0 bg-halftone group-hover:opacity-30 transition-opacity" />
            <div className="relative z-10 flex items-center justify-center h-full">
              <span className="font-bold uppercase bg-white px-4 py-2 border-pop">Halftone Dots</span>
            </div>
          </div>
          <div className="relative h-48 card-pop overflow-hidden group">
            <div className="absolute inset-0 bg-checkered group-hover:opacity-20 transition-opacity" />
            <div className="relative z-10 flex items-center justify-center h-full">
              <span className="font-bold uppercase bg-white px-4 py-2 border-pop">Checkered Grid</span>
            </div>
          </div>
          <div className="relative h-48 card-pop flex items-center justify-center bg-pop-yellow">
            <div className="absolute inset-0 flex items-center justify-center opacity-10">
               <div className="w-full h-full border-4 border-dashed border-pop-black animate-spin-slow" />
            </div>
            <span className="font-bold uppercase bg-white px-4 py-2 border-pop shadow-pop">Hard Shadows</span>
          </div>
        </div>
      </section>

      {/* Spacing & Layout */}
      <section className="space-y-8">
        <div className="flex items-center gap-3">
          <Layout className="w-8 h-8" />
          <h2 className="text-4xl">Layout Principles</h2>
        </div>
        <div className="card-pop bg-pop-teal text-white space-y-6">
          <ul className="list-disc list-inside space-y-2 font-medium">
            <li>No rounded corners. Sharp 90-degree angles only.</li>
            <li>Thick black borders (3px-4px) on all structural elements.</li>
            <li>Hard shadows with no blur, offset by 4px or 8px.</li>
            <li>Large areas of solid, high-saturation color.</li>
            <li>Visible grids and structural lines.</li>
          </ul>
        </div>
      </section>
    </div>
  );
};
