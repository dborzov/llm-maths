/**
 * toc.js — Table of contents behaviour:
 *
 *   1. Scroll-spy:  highlights the active TOC entry as the user scrolls.
 *   2. Mobile UX:   a sticky pill-shaped toggle expands/collapses the TOC,
 *                   and shows the current section name when collapsed.
 *   3. Auto-close:  tapping a link on mobile closes the panel.
 */
document.addEventListener('DOMContentLoaded', () => {
  const toc = document.querySelector('.toc');
  const tocLinks = document.querySelectorAll('.toc-body a');
  if (tocLinks.length === 0 || !toc) return;

  /* Map anchor-id → TOC link element. */
  const linkMap = {};
  tocLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href && href.startsWith('#')) linkMap[href.slice(1)] = link;
  });

  /* ---------- 1. scroll-spy ---------- */
  const currentLabel = document.querySelector('[data-toc-current]');
  let activeId = null;

  const setActive = (id) => {
    if (id === activeId) return;
    activeId = id;
    tocLinks.forEach(l => l.classList.remove('toc-active'));
    const link = linkMap[id];
    if (!link) return;
    link.classList.add('toc-active');

    // Mirror the active heading text into the mobile toggle.
    if (currentLabel) currentLabel.textContent = link.textContent.trim();

    // Keep the active link visible inside the scrolling panel.
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

  if (targets.length > 0) {
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

  /* ---------- 2. mobile open/close toggle ---------- */
  const toggle = toc.querySelector('.toc-toggle');
  if (toggle) {
    const setOpen = (open) => {
      toc.classList.toggle('toc-open', open);
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    };

    toggle.addEventListener('click', () => {
      const isOpen = toc.classList.contains('toc-open');
      setOpen(!isOpen);
    });

    /* Tapping a TOC link should close the panel on mobile. */
    tocLinks.forEach(link => {
      link.addEventListener('click', () => {
        if (window.matchMedia('(max-width: 1024px)').matches) setOpen(false);
      });
    });

    /* Close on Escape so keyboard users aren't trapped. */
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && toc.classList.contains('toc-open')) setOpen(false);
    });
  }
});
