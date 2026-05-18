/**
 * toc.js — Table of contents behaviour:
 *
 *   1. Mobile UX:   a sticky pill-shaped toggle expands/collapses the TOC,
 *                   and shows the current section name when collapsed.
 *   2. Scroll-spy:  highlights the active TOC entry as the user scrolls.
 *   3. Auto-close:  tapping a link on mobile closes the panel.
 *
 * Wiring runs whenever the document is ready. Each piece is independent —
 * a missing TOC body must NOT prevent the toggle from binding, and a
 * broken toggle must not prevent scroll-spy from working.
 */
function setupToc() {
  const toc = document.querySelector('.toc');
  if (!toc) return;

  /* ---------- 1. mobile open/close toggle (top priority) ---------- */
  const toggle = toc.querySelector('.toc-toggle');
  const setOpen = (open) => {
    toc.classList.toggle('toc-open', open);
    if (toggle) toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  };

  if (toggle) {
    /* Use a pointer-event handler so a single user gesture toggles once
       across mouse, touch, and stylus inputs. `pointerup` is the most
       reliable cross-browser tap event for non-link / non-form elements
       on iOS Safari, which historically swallows synthetic click events
       under certain sticky-positioning conditions. We still keep a
       fallback `click` listener so keyboard activation (Enter / Space)
       continues to work. */
    let pointerHandled = false;
    toggle.addEventListener('pointerup', (e) => {
      // Only main button or touch — skip right-click / middle-click.
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      pointerHandled = true;
      setOpen(!toc.classList.contains('toc-open'));
      // Reset the flag so the matching click event is ignored, then
      // re-enable for the next gesture.
      setTimeout(() => { pointerHandled = false; }, 350);
    });
    toggle.addEventListener('click', (e) => {
      if (pointerHandled) { e.preventDefault(); return; }
      setOpen(!toc.classList.contains('toc-open'));
    });

    /* Close on Escape so keyboard users aren't trapped in the panel. */
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && toc.classList.contains('toc-open')) setOpen(false);
    });

    /* Close when tapping outside the panel on mobile (better UX than
       leaving a half-page overlay behind). */
    document.addEventListener('pointerdown', (e) => {
      if (!window.matchMedia('(max-width: 1024px)').matches) return;
      if (!toc.classList.contains('toc-open')) return;
      if (toc.contains(e.target)) return;
      setOpen(false);
    });
  }

  /* ---------- 2. TOC links ---------- */
  const tocLinks = toc.querySelectorAll('.toc-body a');

  /* Tapping a TOC link should close the panel on mobile. */
  tocLinks.forEach(link => {
    link.addEventListener('click', () => {
      if (window.matchMedia('(max-width: 1024px)').matches) setOpen(false);
    });
  });

  /* If there are no link targets at all, scroll-spy has nothing to do. */
  if (tocLinks.length === 0) return;

  /* Map anchor-id → TOC link element. */
  const linkMap = {};
  tocLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href && href.startsWith('#')) linkMap[href.slice(1)] = link;
  });

  /* ---------- 3. scroll-spy ---------- */
  const currentLabel = document.querySelector('[data-toc-current]');
  let activeId = null;

  const setActive = (id) => {
    if (id === activeId) return;
    activeId = id;
    tocLinks.forEach(l => l.classList.remove('toc-active'));
    const link = linkMap[id];
    if (!link) return;
    link.classList.add('toc-active');

    if (currentLabel) currentLabel.textContent = link.textContent.trim();

    const tocBody = document.querySelector('.toc-body');
    if (tocBody) {
      const linkTop = link.offsetTop;
      const bodyHeight = tocBody.clientHeight;
      if (linkTop < tocBody.scrollTop || linkTop > tocBody.scrollTop + bodyHeight - 60) {
        tocBody.scrollTo({ top: linkTop - bodyHeight / 2, behavior: 'smooth' });
      }
    }
  };

  const targets = [
    ...document.querySelectorAll('.article-content details[id]'),
    ...document.querySelectorAll('.article-content h3[id]'),
  ];
  if (targets.length === 0) return;

  const observer = new IntersectionObserver(entries => {
    let topEntry = null;
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        if (!topEntry || entry.boundingClientRect.top < topEntry.boundingClientRect.top) {
          topEntry = entry;
        }
      }
    });
    if (topEntry) setActive(topEntry.target.getAttribute('id'));
  }, {
    rootMargin: '-8% 0px -75% 0px',
    threshold: 0,
  });
  targets.forEach(el => observer.observe(el));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupToc);
} else {
  setupToc();
}
