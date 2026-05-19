// @ts-check
/**
 * The {{< figure >}} shortcode renders an image with a caption block.
 * Three contracts this pins down:
 *
 *   1. The image src must resolve (no 404 — i.e. naturalWidth > 0). A
 *      broken src would otherwise silently ship as the screenshot bug
 *      that motivated this spec.
 *   2. On phone-narrow viewports, the caption text gets its own row
 *      and is NOT squeezed into a single-word column by the uppercase
 *      `figure-credit` next to it.
 *   3. On wider screens, the caption text and credit share a row.
 */
const { test, expect } = require('@playwright/test');

const ARTICLE = '/comicbook/07-deepseek-attn/07-csa/';

test('figure image src resolves (no 404)', async ({ page }) => {
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');
  // Open all collapsible H2 sections so the figure under "Figure 3, As
  // Promised" is laid out.
  await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));

  const img = page.locator('.figure-block img').first();
  await expect(img).toBeVisible();
  const dims = await img.evaluate((el) => ({
    nw: /** @type {HTMLImageElement} */ (el).naturalWidth,
    nh: /** @type {HTMLImageElement} */ (el).naturalHeight,
  }));
  // naturalWidth === 0 is how the browser exposes a failed image load.
  expect(dims.nw, 'image failed to load (naturalWidth=0)').toBeGreaterThan(0);
  expect(dims.nh).toBeGreaterThan(0);
});

test('figure caption text gets enough room (not squeezed by the credit)', async ({ page }) => {
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');
  await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));

  const figure = page.locator('.figure-block').first();
  await expect(figure).toBeVisible();
  const captionText = figure.locator('.figure-caption-text');
  const credit = figure.locator('.figure-credit');
  await expect(captionText).toBeVisible();
  await expect(credit).toBeVisible();

  const figBox = await figure.boundingBox();
  const textBox = await captionText.boundingBox();
  expect(figBox).not.toBeNull();
  expect(textBox).not.toBeNull();
  if (!figBox || !textBox) return;

  // The caption text must occupy a real readable column — at least
  // ~55% of the figure's width on every supported viewport. On the
  // pre-fix mobile layout it was squeezed below 30%.
  const ratio = textBox.width / figBox.width;
  expect(
    ratio,
    `caption text width = ${textBox.width.toFixed(0)}px of ${figBox.width.toFixed(0)}px (${(ratio * 100).toFixed(0)}%)`,
  ).toBeGreaterThan(0.55);
});

test('phone-narrow: caption text and credit stack on separate rows', async ({ page }) => {
  const vp = page.viewportSize();
  test.skip(!vp || vp.width > 600, 'phone-only layout (max-width: 600px)');

  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');
  await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));

  const figure = page.locator('.figure-block').first();
  const textBox = await figure.locator('.figure-caption-text').boundingBox();
  const creditBox = await figure.locator('.figure-credit').boundingBox();
  expect(textBox).not.toBeNull();
  expect(creditBox).not.toBeNull();
  if (!textBox || !creditBox) return;

  // Credit must sit BELOW the caption text, not beside it.
  expect(
    creditBox.y,
    `credit y=${creditBox.y}, text bottom=${textBox.y + textBox.height}`,
  ).toBeGreaterThanOrEqual(textBox.y + textBox.height - 1);
});

test('wider screens: caption text and credit share a row', async ({ page }) => {
  const vp = page.viewportSize();
  test.skip(!vp || vp.width <= 600, 'desktop/tablet-only layout (min-width: 601px)');

  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');
  await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));

  const figure = page.locator('.figure-block').first();
  const textBox = await figure.locator('.figure-caption-text').boundingBox();
  const creditBox = await figure.locator('.figure-credit').boundingBox();
  expect(textBox).not.toBeNull();
  expect(creditBox).not.toBeNull();
  if (!textBox || !creditBox) return;

  // Their vertical centers should overlap — they're on the same row.
  const textCenter = textBox.y + textBox.height / 2;
  const creditCenter = creditBox.y + creditBox.height / 2;
  const sameRow = Math.abs(textCenter - creditCenter) < textBox.height;
  expect(sameRow, `text center y=${textCenter}, credit center y=${creditCenter}`).toBe(true);
});
