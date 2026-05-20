// @ts-check
/**
 * The mobile TOC has a sticky pill button that opens a panel of in-page
 * links. This file pins down the user-facing contract:
 *   • the button is tappable from anywhere along its surface,
 *   • tapping opens the panel; tapping again (or outside) closes it,
 *   • desktop has no toggle UX — the panel is always visible.
 *
 * Regression target: the bug where pointer-events on a child span made
 * the iOS Safari tap miss the underlying <button>.
 */
const { test, expect } = require('@playwright/test');

const ARTICLE = '/comicbook/03-quantization/03-lloyd-max/';

test.describe('Article TOC', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(ARTICLE);
    await page.waitForLoadState('networkidle');
  });

  test('the TOC pill is sticky and renders the toggle controls', async ({ page }) => {
    const toggle = page.locator('.toc-toggle');
    await expect(toggle).toBeVisible();
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });

  test('mobile-only: tapping the bar opens the panel; tap again closes it', async ({ page, isMobile }) => {
    test.skip(!isMobile, 'desktop has the panel open permanently');
    const toc = page.locator('.toc');
    const toggle = page.locator('.toc-toggle');
    const panel = page.locator('.toc-panel');

    // Initial state — closed.
    await expect(toc).not.toHaveClass(/toc-open/);

    await toggle.tap();
    await expect(toc).toHaveClass(/toc-open/);
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    // Panel becomes scrollable when open — poll until the 0.18s max-height
    // transition has settled rather than reading mid-animation.
    await expect(async () => {
      const openHeight = await panel.evaluate(el => parseFloat(getComputedStyle(el).maxHeight));
      expect(openHeight).toBeGreaterThan(100);
    }).toPass({ timeout: 1000 });

    await toggle.tap();
    await expect(toc).not.toHaveClass(/toc-open/);
  });

  test('mobile-only: tapping any child of the bar still opens the panel', async ({ page, isMobile }) => {
    test.skip(!isMobile, 'desktop has no toggle');
    const toc = page.locator('.toc');

    // The chevron sits at the far right of the bar — historically the
    // most likely to get its tap eaten by an overlay.
    await page.locator('.toc-toggle-chev').tap();
    await expect(toc).toHaveClass(/toc-open/);

    // The hamburger icon at the far left.
    await page.locator('.toc-toggle').tap(); // close
    await expect(toc).not.toHaveClass(/toc-open/);

    await page.locator('.toc-toggle-icon').tap();
    await expect(toc).toHaveClass(/toc-open/);

    // The label text in the middle.
    await page.locator('.toc-toggle').tap(); // close
    await page.locator('.toc-toggle-label').tap();
    await expect(toc).toHaveClass(/toc-open/);
  });

  test('mobile-only: tapping a TOC link closes the panel and jumps the page', async ({ page, isMobile }) => {
    test.skip(!isMobile, 'desktop has no panel to close');
    const toc = page.locator('.toc');
    await page.locator('.toc-toggle').tap();
    await expect(toc).toHaveClass(/toc-open/);

    const link = page.locator('.toc-body a').first();
    const href = await link.getAttribute('href');
    expect(href).toMatch(/^#/);
    await link.tap();

    await expect(toc).not.toHaveClass(/toc-open/);
    // Allow scroll-spy a moment to settle, then verify URL fragment.
    await page.waitForTimeout(150);
    expect(new URL(page.url()).hash).toBe(href);
  });

  test('mobile-only: tapping outside the panel closes it', async ({ page, isMobile }) => {
    test.skip(!isMobile, 'desktop has no panel');
    const toc = page.locator('.toc');
    await page.locator('.toc-toggle').tap();
    await expect(toc).toHaveClass(/toc-open/);

    // Dispatch an outside pointerdown synthetically — the dropdown panel
    // covers most of the screen when open, so a real tap on most coords
    // would land on the TOC itself.
    await page.evaluate(() => {
      document.dispatchEvent(new PointerEvent('pointerdown', {
        bubbles: true,
        clientX: 1,
        clientY: 1,
      }));
    });
    await expect(toc).not.toHaveClass(/toc-open/);
  });

  test('desktop-only: panel is always visible and toggle is inert', async ({ page, isMobile }) => {
    test.skip(isMobile, 'mobile uses a collapsible panel');
    const panel = page.locator('.toc-panel');
    await expect(panel).toBeVisible();
    const display = await panel.evaluate(el => getComputedStyle(el).display);
    expect(display).toBe('block');

    // The toggle is pointer-events: none on desktop so the visual styling
    // stays but clicks are inert. We can still check via JS.
    const pe = await page.locator('.toc-toggle').evaluate(el => getComputedStyle(el).pointerEvents);
    expect(pe).toBe('none');
  });

  test('Escape closes an open panel', async ({ page, isMobile }) => {
    test.skip(!isMobile, 'desktop has no panel to close');
    const toc = page.locator('.toc');
    await page.locator('.toc-toggle').tap();
    await expect(toc).toHaveClass(/toc-open/);
    await page.keyboard.press('Escape');
    await expect(toc).not.toHaveClass(/toc-open/);
  });
});
