// @ts-check
/**
 * fold.js injects hierarchical section numbers ("1.", "1.2") into H2/H3
 * headings and mirrors them into TOC links. These tests pin down:
 *   • numbers appear on every section heading,
 *   • the same numbers appear on the corresponding TOC link,
 *   • shortcode-internal headings (timeline cards) are NOT numbered,
 *   • author-numbered headings ("### 1. Foo") are not double-prefixed.
 */
const { test, expect } = require('@playwright/test');

test('section numbers are injected into article headings', async ({ page }) => {
  await page.goto('/comicbook/03-quantization/03-lloyd-max/');
  await page.waitForLoadState('networkidle');

  // First H2 (now wrapped in a <details><summary>) gets "1."
  const firstSummary = page.locator('.article-content summary').first();
  await expect(firstSummary.locator('.section-number')).toHaveText('1.');

  // Section number color and font are part of the visual contract.
  const colour = await firstSummary
    .locator('.section-number')
    .evaluate(el => getComputedStyle(el).color);
  // Pink-ish (--accent-1 on cream theme).
  expect(colour).toMatch(/rgb\(255,\s*0,\s*127\)/);
});

test('TOC links carry the same section number prefix', async ({ page }) => {
  await page.goto('/comicbook/03-quantization/03-lloyd-max/');
  await page.waitForLoadState('networkidle');

  const firstTocLink = page.locator('.toc-body a').first();
  await expect(firstTocLink.locator('.toc-number')).toHaveText('1.');
});

test('shortcode-internal headings are NOT auto-numbered', async ({ page }) => {
  await page.goto('/comicbook/03-quantization/12-hardware-horizon/');
  await page.waitForLoadState('networkidle');

  // Timeline event titles are H3 inside .timeline — they must stay clean.
  const timelineTitles = page.locator('.timeline-event-title');
  const count = await timelineTitles.count();
  expect(count).toBeGreaterThan(0);
  for (let i = 0; i < count; i++) {
    await expect(timelineTitles.nth(i).locator('.section-number')).toHaveCount(0);
  }
});

test('manually numbered headings ("### 1. Foo") are not double-numbered', async ({ page }) => {
  await page.goto('/comicbook/03-quantization/03-lloyd-max/');
  await page.waitForLoadState('networkidle');

  // The article has "### 1. The Robot-Voice Limit (rate vs. fidelity)".
  // Look for that text in any h3 — it should NOT have a .section-number span.
  const h3 = page.locator('.article-content h3', { hasText: 'Robot-Voice Limit' });
  await expect(h3).toHaveCount(1);
  await expect(h3.locator('.section-number')).toHaveCount(0);
});
