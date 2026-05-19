/**
 * fold.js — Auto-wrap H2 sections in <details>/<summary> for collapsible sections,
 * and inject hierarchical section numbers ("1.", "1.2", ...) into headings and
 * matching TOC links so readers always know where they are.
 *
 * Runs once on DOMContentLoaded. Requires no markup changes from authors.
 */
document.addEventListener('DOMContentLoaded', () => {
  const body = document.querySelector('.article-content');
  if (!body) return;

  // Pages can opt out of the auto-fold by setting `unfolded: true` in front
  // matter (showcase / reference pages where every H2 should be visible).
  const unfolded = body.classList.contains('unfolded');

  /* ------------------------------------------------------------ *
   * 1. Number every H2 and H3 in document order. We do this BEFORE
   *    auto-folding so the numbers are computed against the original
   *    DOM and applied to the (about-to-be-rewritten) headings.
   * ------------------------------------------------------------ */
  const numberFor = new Map();   // anchor id -> "1.", "1.2"
  let majorIdx = 0;
  let minorIdx = 0;

  /* Only count headings that are part of the prose narrative — skip
     headings nested inside shortcode widgets (timeline cards, callouts,
     pullquotes, infographics, etc.) so the auto-numbering reflects the
     real section structure of the article. */
  const COMPONENT_SELECTOR = '.timeline, .callout, .pullquote, .infographic, .techtree, .figure-block, .marginnote, .wiki-sot';
  const headings = [...body.querySelectorAll('h2, h3')].filter(
    h => !h.closest(COMPONENT_SELECTOR)
  );
  /* Detect manually-numbered headings ("### 1. Foo", "### (2) Bar") so we
     don't double-number them. We still tick the index forward so the
     auto-numbering of siblings stays consistent. */
  const MANUAL_NUMBER_RE = /^\s*(\(?\d+[.)])\s*/;

  headings.forEach(h => {
    let number;
    if (h.tagName === 'H2') {
      majorIdx += 1;
      minorIdx = 0;
      number = `${majorIdx}.`;
    } else {
      if (majorIdx === 0) return;
      minorIdx += 1;
      number = `${majorIdx}.${minorIdx}`;
    }
    /* Skip both the heading prefix AND the TOC injection when the author
       already wrote a number — that count IS the section number, and we
       don't want to stamp another one in front of it. */
    if (MANUAL_NUMBER_RE.test(h.textContent)) return;

    const id = h.getAttribute('id');
    if (id) numberFor.set(id, number);

    const span = document.createElement('span');
    span.className = 'section-number';
    span.textContent = number;
    h.insertBefore(span, h.firstChild);
  });

  /* ------------------------------------------------------------ *
   * 2. Mirror those numbers in the TOC sidebar so the panel matches
   *    the headings exactly.
   * ------------------------------------------------------------ */
  document.querySelectorAll('.toc-body a[href^="#"]').forEach(link => {
    const id = link.getAttribute('href').slice(1);
    const number = numberFor.get(id);
    if (!number) return;
    const span = document.createElement('span');
    span.className = 'toc-number';
    span.textContent = number;
    link.insertBefore(span, link.firstChild);
  });

  /* ------------------------------------------------------------ *
   * 3. Auto-fold every H2 section into a <details>/<summary>.
   * ------------------------------------------------------------ */
  if (unfolded) return;

  const h2s = headings.filter(h => h.tagName === 'H2');
  if (h2s.length === 0) return;

  h2s.forEach((h2, index) => {
    const details  = document.createElement('details');
    const summary  = document.createElement('summary');

    if (index === 0) details.setAttribute('open', '');

    const id = h2.getAttribute('id');
    if (id) {
      details.setAttribute('id', id);
      h2.removeAttribute('id');
    }

    summary.innerHTML = h2.innerHTML;
    details.appendChild(summary);

    const siblings = [];
    let next = h2.nextSibling;
    while (next && !(next.nodeType === 1 && next.tagName === 'H2')) {
      siblings.push(next);
      next = next.nextSibling;
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'section-content';
    siblings.forEach(sib => contentDiv.appendChild(sib));
    details.appendChild(contentDiv);

    h2.parentNode.replaceChild(details, h2);
  });
});
