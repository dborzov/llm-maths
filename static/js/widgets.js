/**
 * widgets.js — wire up interactive bits of the techtree and timeline
 * shortcodes:
 *
 *   • Tech tree: clicking the Graph/List buttons swaps which pane is
 *     visible. On mobile (<700px) we default to the list pane.
 *
 *   • Timeline: a small dot-row reflects scroll position on narrow screens
 *     so readers can see how many milestones remain.
 */
document.addEventListener('DOMContentLoaded', () => {
  /* ---------- Issue-cover article list: keep "Open →" link clean ----------
     A link inside <summary> normally still toggles the parent <details>.
     For the explicit "Open →" link we want a pure navigation. */
  document.querySelectorAll('.issue-cover-article-open, a.issue-cover-article-title').forEach(a => {
    a.addEventListener('click', (e) => e.stopPropagation());
  });

  /* ---------- Tech tree ---------- */
  document.querySelectorAll('[data-techtree]').forEach(tree => {
    const buttons = tree.querySelectorAll('[data-techtree-view]');
    const panes   = tree.querySelectorAll('[data-techtree-pane]');

    const setView = (view) => {
      buttons.forEach(b => {
        const active = b.dataset.techtreeView === view;
        b.classList.toggle('is-active', active);
        b.setAttribute('aria-selected', active ? 'true' : 'false');
      });
      panes.forEach(p => {
        p.hidden = p.dataset.techtreePane !== view;
      });
      tree.dataset.activeView = view;
    };

    /* Default per screen size: list on mobile, graph on desktop. */
    const initial = window.matchMedia('(max-width: 700px)').matches ? 'list' : 'graph';
    setView(initial);

    buttons.forEach(b => b.addEventListener('click', () => setView(b.dataset.techtreeView)));
  });

  /* ---------- Timeline scroll-position dots ---------- */
  document.querySelectorAll('.timeline').forEach(timeline => {
    const rail   = timeline.querySelector('.timeline-rail');
    const events = timeline.querySelectorAll('.timeline-event');
    if (!rail || events.length === 0) return;

    /* Build a dot per event — purely a position indicator on mobile. */
    const dots = document.createElement('div');
    dots.className = 'timeline-dots';
    dots.setAttribute('aria-hidden', 'true');
    events.forEach((_, i) => {
      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'timeline-dot';
      dot.dataset.index = String(i);
      dot.setAttribute('aria-label', `Jump to event ${i + 1}`);
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
});
