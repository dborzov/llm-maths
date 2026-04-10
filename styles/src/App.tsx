import React, { useState } from 'react';
import { StyleGuide } from './components/StyleGuide';
import { WikiMockup } from './components/WikiMockup';
import { Eye, Palette as PaletteIcon } from 'lucide-react';

export default function App() {
  const [view, setView] = useState<'guide' | 'mockup'>('guide');

  return (
    <div className="min-h-screen">
      {/* View Toggle */}
      <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-[100] flex gap-2 bg-white border-pop p-2 shadow-pop-lg">
        <button 
          onClick={() => setView('guide')}
          className={`flex items-center gap-2 px-6 py-2 font-bold uppercase text-sm transition-all ${
            view === 'guide' 
              ? 'bg-pop-pink text-white border-pop' 
              : 'hover:bg-pop-cream'
          }`}
        >
          <PaletteIcon className="w-4 h-4" />
          Style Guide
        </button>
        <button 
          onClick={() => setView('mockup')}
          className={`flex items-center gap-2 px-6 py-2 font-bold uppercase text-sm transition-all ${
            view === 'mockup' 
              ? 'bg-pop-teal text-white border-pop' 
              : 'hover:bg-pop-cream'
          }`}
        >
          <Eye className="w-4 h-4" />
          Wiki Mockup
        </button>
      </div>

      {/* Content Rendering */}
      {view === 'guide' ? <StyleGuide /> : <WikiMockup />}
    </div>
  );
}
