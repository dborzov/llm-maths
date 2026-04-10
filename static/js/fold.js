/**
 * fold.js — Auto-wrap H2 sections in <details>/<summary> for collapsible sections.
 * Runs once on DOMContentLoaded. Requires no markup changes from authors.
 */
document.addEventListener('DOMContentLoaded', () => {
  const body = document.querySelector('.article-content');
  if (!body) return;

  const headings = [...body.querySelectorAll('h2')];
  if (headings.length === 0) return;

  headings.forEach((h2, index) => {
    const details  = document.createElement('details');
    const summary  = document.createElement('summary');

    // First section starts open
    if (index === 0) details.setAttribute('open', '');

    // Preserve the heading's anchor id on the details element
    const id = h2.getAttribute('id');
    if (id) {
      details.setAttribute('id', id);
      h2.removeAttribute('id');
    }

    // Move heading text into summary
    summary.innerHTML = h2.innerHTML;
    details.appendChild(summary);

    // Collect all following siblings until the next H2
    const siblings = [];
    let next = h2.nextSibling;
    while (next && !(next.nodeType === 1 && next.tagName === 'H2')) {
      siblings.push(next);
      next = next.nextSibling;
    }

    // Create a wrapper div for the section content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'section-content';
    siblings.forEach(sib => contentDiv.appendChild(sib));
    details.appendChild(contentDiv);

    // Replace the original H2 with the details element
    h2.parentNode.replaceChild(details, h2);
  });
});
