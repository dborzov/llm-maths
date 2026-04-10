/**
 * toc.js — Scroll-spy: highlights the active TOC entry as the user scrolls.
 * Uses IntersectionObserver for performance.
 */
document.addEventListener('DOMContentLoaded', () => {
  const tocLinks = document.querySelectorAll('.toc-body a');
  if (tocLinks.length === 0) return;

  // Build a map: anchor-id → TOC link element
  const linkMap = {};
  tocLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href && href.startsWith('#')) {
      linkMap[href.slice(1)] = link;
    }
  });

  let activeId = null;

  const setActive = (id) => {
    if (id === activeId) return;
    activeId = id;
    tocLinks.forEach(l => l.classList.remove('toc-active'));
    if (id && linkMap[id]) {
      linkMap[id].classList.add('toc-active');
      // Scroll TOC sidebar to keep active link visible
      const tocBody = document.querySelector('.toc-body');
      if (tocBody) {
        const activeLink = linkMap[id];
        const linkTop = activeLink.offsetTop;
        const bodyHeight = tocBody.clientHeight;
        if (linkTop < tocBody.scrollTop || linkTop > tocBody.scrollTop + bodyHeight - 60) {
          tocBody.scrollTo({ top: linkTop - bodyHeight / 2, behavior: 'smooth' });
        }
      }
    }
  };

  // Observe all heading elements that have an id (either on h2/h3 or on details wrapping them)
  const targets = [
    ...document.querySelectorAll('.article-content details[id]'),
    ...document.querySelectorAll('.article-content h3[id]'),
  ];

  if (targets.length === 0) return;

  const observer = new IntersectionObserver(entries => {
    // Find the topmost intersecting element
    let topEntry = null;
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        if (!topEntry || entry.boundingClientRect.top < topEntry.boundingClientRect.top) {
          topEntry = entry;
        }
      }
    });
    if (topEntry) {
      setActive(topEntry.target.getAttribute('id'));
    }
  }, {
    rootMargin: '-8% 0px -75% 0px',
    threshold: 0,
  });

  targets.forEach(el => observer.observe(el));

  // Mobile: make TOC header clickable to toggle
  const tocHeader = document.querySelector('.toc-header');
  const toc = document.querySelector('.toc');
  if (tocHeader && toc && window.innerWidth <= 900) {
    tocHeader.style.cursor = 'pointer';
    tocHeader.addEventListener('click', () => {
      toc.classList.toggle('toc-open');
    });
  }
});
