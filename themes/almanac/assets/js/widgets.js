/**
 * widgets.js — interactive bits for the issue cover and timeline shortcodes.
 * Tech tree toggle + rendering lives in techtree.js.
 */
function setupWidgets() {
  /* Issue-cover article list: keep title / "Read →" links clean. */
  document.querySelectorAll('.issue-cover-article-open, a.issue-cover-article-title').forEach(a => {
    a.addEventListener('click', (e) => e.stopPropagation());
  });

  /* Timeline scroll-position dots */
  document.querySelectorAll('.timeline').forEach(timeline => {
    const rail   = timeline.querySelector('.timeline-rail');
    const events = timeline.querySelectorAll('.timeline-event');
    if (!rail || events.length === 0) return;

    const dots = document.createElement('div');
    dots.className = 'timeline-dots';
    dots.setAttribute('aria-hidden', 'true');
    events.forEach((_, i) => {
      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'timeline-dot';
      dot.dataset.index = String(i);
      dot.setAttribute('aria-label', 'Jump to event ' + (i + 1));
      dots.appendChild(dot);
    });
    rail.insertAdjacentElement('afterend', dots);

    const updateActive = () => {
      const railRect = rail.getBoundingClientRect();
      const center = railRect.left + railRect.width / 2;
      let activeIdx = 0;
      let bestDist = Infinity;
      events.forEach((ev, i) => {
        const r = ev.getBoundingClientRect();
        const c = r.left + r.width / 2;
        const d = Math.abs(c - center);
        if (d < bestDist) { bestDist = d; activeIdx = i; }
      });
      dots.querySelectorAll('.timeline-dot').forEach((d, i) =>
        d.classList.toggle('is-active', i === activeIdx)
      );
    };

    rail.addEventListener('scroll', updateActive, { passive: true });
    window.addEventListener('resize', updateActive);
    updateActive();

    dots.addEventListener('click', (e) => {
      const btn = e.target.closest('.timeline-dot');
      if (!btn) return;
      const i = Number(btn.dataset.index);
      const ev = events[i];
      if (ev) ev.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupWidgets);
} else {
  setupWidgets();
}
