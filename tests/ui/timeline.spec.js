// @ts-check
/**
 * The timeline shortcode lays out up to five milestone cards. On mobile
 * it becomes a horizontally scrollable strip with a dot-row that
 * indicates which card is centred.
 */
const { test, expect } = require('@playwright/test');

const ARTICLE = '/comicbook/03-quantization/12-hardware-horizon/';

test('timeline renders 1–5 events in a grid rail', async ({ page }) => {
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');

  // Make sure all collapsibles are open so the timeline is visible.
  await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));

  const events = page.locator('.timeline-event');
  const count = await events.count();
  expect(count).toBeGreaterThan(0);
  expect(count).toBeLessThanOrEqual(5);
});

test('phone-narrow: dot row appears and reflects scroll position', async ({ page }) => {
  // CSS shows dots only at viewport widths ≤ 700px.
  test.skip(page.viewportSize().width > 700, 'dots are hidden above 700px');
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');
  await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));

  const dots = page.locator('.timeline-dots .timeline-dot');
  const count = await dots.count();
  expect(count).toBeGreaterThan(1);

  // First dot is active to start.
  await expect(dots.nth(0)).toHaveClass(/is-active/);

  // Tapping a later dot scrolls it into view and marks it active.
  await dots.nth(2).click();
  await page.waitForTimeout(400);
  await expect(dots.nth(2)).toHaveClass(/is-active/);
});

test('above-700px: dot row is hidden (the rail itself shows everything)', async ({ page }) => {
  test.skip(page.viewportSize().width <= 700, 'phone-narrow shows the dots');
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');
  await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));

  const dots = page.locator('.timeline-dots');
  if (await dots.count() > 0) {
    await expect(dots).toBeHidden();
  }
});
