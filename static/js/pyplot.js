/**
 * pyplot.js — UX enhancement for pyplot blocks.
 *
 *  - Click on the plot image opens a fullscreen <dialog> lightbox.
 *    Clicking the image inside the lightbox toggles between "fit to viewport"
 *    and "actual pixels" (which makes the wrapper scrollable for inspection).
 *  - The <details> summary label toggles between "show python source" and
 *    "hide python source" so the affordance is obvious.
 *
 * No-JS fallback: each image is wrapped in an <a href="…" target="_blank">
 * so it still opens at full size; <details> toggles natively.
 */
document.addEventListener('DOMContentLoaded', () => {
    const blocks = document.querySelectorAll('.pyplot-block');
    if (blocks.length === 0) return;

    // Build one lightbox dialog reused by every plot on the page.
    const dialog = document.createElement('dialog');
    dialog.className = 'pyplot-lightbox';
    dialog.innerHTML = `
        <div class="pyplot-lightbox-body">
            <div class="pyplot-lightbox-bar">
                <span class="pyplot-lightbox-caption"></span>
                <button type="button" class="pyplot-lightbox-close" aria-label="Close">× Close</button>
            </div>
            <div class="pyplot-lightbox-img-wrap">
                <img class="pyplot-lightbox-img" alt="">
            </div>
        </div>
    `;
    document.body.appendChild(dialog);

    const lightboxImg     = dialog.querySelector('.pyplot-lightbox-img');
    const lightboxCaption = dialog.querySelector('.pyplot-lightbox-caption');
    const closeBtn        = dialog.querySelector('.pyplot-lightbox-close');

    const openLightbox = (src, alt, caption) => {
        lightboxImg.src = src;
        lightboxImg.alt = alt || '';
        lightboxImg.classList.remove('is-actual-size');
        lightboxCaption.textContent = caption || '';
        if (typeof dialog.showModal === 'function') {
            dialog.showModal();
        } else {
            dialog.setAttribute('open', '');
        }
    };
    const closeLightbox = () => {
        if (typeof dialog.close === 'function') dialog.close();
        else dialog.removeAttribute('open');
        // Free memory — large PNGs are heavy.
        lightboxImg.removeAttribute('src');
    };

    closeBtn.addEventListener('click', closeLightbox);
    // Click on the dim backdrop (i.e. on the <dialog> element itself, outside body) closes.
    dialog.addEventListener('click', (e) => {
        if (e.target === dialog) closeLightbox();
    });
    // Clicking the image inside the lightbox toggles 1:1 view.
    lightboxImg.addEventListener('click', () => {
        lightboxImg.classList.toggle('is-actual-size');
    });

    blocks.forEach((block) => {
        const link    = block.querySelector('.pyplot-image');
        const caption = block.querySelector('.pyplot-caption')?.textContent.trim();

        if (link) {
            link.addEventListener('click', (e) => {
                // Respect modifier keys so cmd/ctrl/shift-click still opens in a new tab.
                if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
                e.preventDefault();
                const img = link.querySelector('img');
                openLightbox(link.getAttribute('href'), img?.alt, caption);
            });
        }

        const details = block.querySelector('.pyplot-code');
        if (details) {
            const label = details.querySelector('.pyplot-code-label');
            const update = () => {
                if (!label) return;
                label.textContent = details.open ? 'hide python source' : 'show python source';
            };
            details.addEventListener('toggle', update);
            update();
        }
    });
});
