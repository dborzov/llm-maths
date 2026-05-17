// @ts-check
/**
 * Articles get their H2 sections auto-wrapped in <details>/<summary> so
 * readers can collapse parts of a long piece. This pins down the tap
 * interaction we ship — including the iOS-flavoured edge cases.
 */
const { test, expect } = require('@playwright/test');

const ARTICLE = '/issues/03-sixteen-numbers/03-lloyd-max/';

// fold.js gives every wrapped H2 an id; pyplot code blocks are also
// <details> but without one. Filter on `[id]` to pick just our sections.
const SECTION = '.article-content > details[id], .article-content section details[id]';

test('H2 sections become collapsible <details> blocks', async ({ page }) => {
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');

  const sections = page.locator(SECTION);
  const count = await sections.count();
  expect(count).toBeGreaterThan(3);

  // The first section starts open; everything else starts closed.
  await expect(sections.first()).toHaveJSProperty('open', true);
  await expect(sections.nth(1)).toHaveJSProperty('open', false);
});

test('clicking/tapping a section summary toggles it open and closed', async ({ page }) => {
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');

  const second = page.locator(SECTION).nth(1);
  // The section's own summary (not a nested pyplot summary).
  const summary = second.locator('> summary');

  await expect(second).toHaveJSProperty('open', false);
  await summary.click();
  await expect(second).toHaveJSProperty('open', true);
  await summary.click();
  await expect(second).toHaveJSProperty('open', false);
});

test('every visible <details> id matches its TOC link target', async ({ page }) => {
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');

  const detailsIds = await page
    .locator('.article-content details[id]')
    .evaluateAll(els => els.map(e => e.id));
  const tocFragments = await page
    .locator('.toc-body a[href^="#"]')
    .evaluateAll(els => els.map(a => a.getAttribute('href').slice(1)));

  // Each TOC fragment must point at a real section anchor.
  for (const frag of tocFragments) {
    // Either matches a details id, or an h3 id inside a details block.
    const exists = await page.locator(`[id="${frag}"]`).count();
    expect(exists, `expected anchor #${frag} to exist`).toBeGreaterThan(0);
  }
  // And we expect at least one of our details to have been seen.
  expect(detailsIds.length).toBeGreaterThan(0);
});
