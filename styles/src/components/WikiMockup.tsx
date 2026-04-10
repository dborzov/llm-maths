import React from 'react';
import { Search, Menu, BookOpen, Clock, User, Share2, Bookmark, MessageSquare } from 'lucide-react';

export const WikiMockup = () => {
  return (
    <div className="min-h-screen bg-pop-cream flex flex-col">
      {/* Navigation */}
      <nav className="border-b-4 border-pop-black bg-white sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-20 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-pop-pink border-pop shadow-pop flex items-center justify-center">
                <BookOpen className="text-white w-6 h-6" />
              </div>
              <span className="text-2xl font-display uppercase tracking-tighter">PopWiki</span>
            </div>
            <div className="hidden md:flex items-center gap-1">
              {['Articles', 'Random', 'Contribute', 'About'].map((item) => (
                <button key={item} className="px-4 py-2 font-bold uppercase text-sm hover:bg-pop-yellow transition-colors">
                  {item}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="relative hidden sm:block">
              <input 
                type="text" 
                placeholder="Search the archives..." 
                className="border-pop px-4 py-2 w-64 focus:outline-none focus:bg-pop-yellow transition-colors pr-10"
              />
              <Search className="absolute right-3 top-2.5 w-5 h-5 opacity-50" />
            </div>
            <button className="btn-pop bg-pop-teal text-white py-1.5 px-4 text-sm">Login</button>
            <button className="md:hidden">
              <Menu className="w-8 h-8" />
            </button>
          </div>
        </div>
      </nav>

      <div className="flex-1 max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-8 p-4 md:p-8">
        {/* Main Content */}
        <main className="space-y-8">
          <article className="card-pop relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-pop-pink -mr-16 -mt-16 rotate-45 opacity-10" />
            
            <header className="space-y-6 relative z-10">
              <div className="flex items-center gap-4 text-sm font-bold uppercase">
                <span className="bg-pop-yellow px-2 py-0.5 border-pop">History</span>
                <span className="opacity-50">Article #4029</span>
              </div>
              
              <h1 className="text-5xl md:text-7xl font-display leading-none text-pop-black">
                The Golden Age of <span className="text-pop-pink underline decoration-4 underline-offset-8">Retro-Futurism</span>
              </h1>

              <div className="flex flex-wrap items-center gap-6 py-4 border-y-2 border-pop-black/10 text-sm font-medium">
                <div className="flex items-center gap-2">
                  <User className="w-4 h-4 text-pop-teal" />
                  <span>By Agent 007</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-pop-teal" />
                  <span>Updated 2 hours ago</span>
                </div>
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-pop-teal" />
                  <span>24 Comments</span>
                </div>
              </div>
            </header>

            <div className="mt-8 space-y-6 text-lg leading-relaxed font-sans">
              <p className="first-letter:text-7xl first-letter:font-display first-letter:float-left first-letter:mr-3 first-letter:text-pop-pink">
                Retro-futurism is a movement in the creative arts which shows the influence of depictions of the future produced in an earlier era. If futurism is sometimes called a "science" bent on anticipating what will come, retro-futurism is the remembering of that anticipation.
              </p>
              
              <div className="grid md:grid-cols-2 gap-8 my-12">
                <div className="space-y-4">
                  <div className="aspect-video bg-pop-teal border-pop shadow-pop relative overflow-hidden">
                    <div className="absolute inset-0 bg-halftone" />
                    <img 
                      src="https://picsum.photos/seed/vintage/800/450" 
                      alt="Vintage Concept" 
                      className="w-full h-full object-cover mix-blend-multiply grayscale contrast-125"
                      referrerPolicy="no-referrer"
                    />
                  </div>
                  <p className="text-xs font-bold uppercase opacity-60 italic">Fig 1.1: A conceptual drawing of the "City of Tomorrow" (circa 1954).</p>
                </div>
                <div className="bg-pop-yellow/20 p-6 border-pop border-dashed space-y-4">
                  <h3 className="text-xl font-bold">Key Characteristics</h3>
                  <ul className="space-y-2 text-sm font-medium">
                    <li className="flex items-start gap-2">
                      <span className="w-2 h-2 bg-pop-pink mt-1.5 shrink-0" />
                      <span>Optimistic technological progress</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="w-2 h-2 bg-pop-pink mt-1.5 shrink-0" />
                      <span>Streamlined aesthetic curves</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="w-2 h-2 bg-pop-pink mt-1.5 shrink-0" />
                      <span>Atomic-age iconography</span>
                    </li>
                  </ul>
                </div>
              </div>

              <h2 className="text-3xl font-display border-b-4 border-pop-pink inline-block pb-1">The Aesthetic Conflict</h2>
              <p>
                The style often combines the "low-tech" materials of the past with the "high-tech" concepts of the future. This creates a unique visual tension that is both nostalgic and forward-looking. In the context of modern UI, this translates to heavy borders, vibrant palettes, and a rejection of the "glassy" minimalism that dominated the 2010s.
              </p>
            </div>

            <footer className="mt-12 pt-8 border-t-4 border-pop-black flex items-center justify-between">
              <div className="flex items-center gap-2">
                <button className="btn-pop bg-white text-sm py-1 px-3">Edit Page</button>
                <button className="btn-pop bg-white text-sm py-1 px-3">History</button>
              </div>
              <div className="flex items-center gap-4">
                <Share2 className="w-5 h-5 cursor-pointer hover:text-pop-pink" />
                <Bookmark className="w-5 h-5 cursor-pointer hover:text-pop-pink" />
              </div>
            </footer>
          </article>

          {/* Related Articles */}
          <section className="space-y-6">
            <h2 className="text-2xl font-display">Related Archives</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                { title: 'Atomic Design', color: 'bg-pop-orange' },
                { title: 'Pop Art Origins', color: 'bg-pop-pink' },
                { title: 'The Grid System', color: 'bg-pop-teal' }
              ].map((item) => (
                <div key={item.title} className={`card-pop ${item.color} text-white group cursor-pointer hover:-translate-y-1 transition-transform`}>
                  <h3 className="text-xl font-display mb-2">{item.title}</h3>
                  <p className="text-xs font-bold uppercase opacity-80">Read Article →</p>
                </div>
              ))}
            </div>
          </section>
        </main>

        {/* Sidebar */}
        <aside className="space-y-8">
          <div className="card-pop bg-white space-y-6">
            <div className="aspect-square bg-pop-yellow border-pop shadow-pop relative overflow-hidden">
               <div className="absolute inset-0 bg-checkered" />
               <img 
                src="https://picsum.photos/seed/popart/400/400" 
                alt="Article Subject" 
                className="w-full h-full object-cover mix-blend-multiply contrast-150"
                referrerPolicy="no-referrer"
               />
            </div>
            <div className="space-y-4">
              <h3 className="text-2xl font-display text-center">Retro-Futurism</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between border-b border-pop-black/10 pb-1">
                  <span className="font-bold uppercase opacity-50">Era</span>
                  <span className="font-bold">1950s - 1960s</span>
                </div>
                <div className="flex justify-between border-b border-pop-black/10 pb-1">
                  <span className="font-bold uppercase opacity-50">Medium</span>
                  <span className="font-bold">Mixed Media</span>
                </div>
                <div className="flex justify-between border-b border-pop-black/10 pb-1">
                  <span className="font-bold uppercase opacity-50">Status</span>
                  <span className="font-bold text-pop-teal">Active</span>
                </div>
              </div>
            </div>
          </div>

          <div className="card-pop bg-pop-black text-white space-y-4">
            <h3 className="text-xl font-display text-pop-yellow">Weekly Almanac</h3>
            <p className="text-sm leading-relaxed opacity-80">
              Subscribe to get the latest dispatches from the pop-culture frontlines.
            </p>
            <div className="space-y-2">
              <input 
                type="email" 
                placeholder="Your email..." 
                className="w-full border-2 border-white bg-transparent px-3 py-2 text-sm focus:outline-none focus:bg-white focus:text-pop-black transition-colors"
              />
              <button className="w-full bg-pop-pink py-2 font-bold uppercase text-sm border-2 border-white shadow-[4px_4px_0px_white]">
                Join Now
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-xs font-bold uppercase opacity-50 tracking-widest">Trending Topics</h3>
            <div className="flex flex-wrap gap-2">
              {['Design', 'Vintage', 'Comic', 'Art', 'Future', 'Style'].map(tag => (
                <span key={tag} className="px-3 py-1 bg-white border-2 border-pop-black text-xs font-bold uppercase cursor-pointer hover:bg-pop-teal hover:text-white transition-colors">
                  #{tag}
                </span>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};
