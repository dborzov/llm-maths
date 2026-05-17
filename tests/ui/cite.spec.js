// @ts-check
/**
 * The {{< cite >}} shortcode produces an external link with a coloured
 * kind badge ("paper", "blog", ...). It must open in a new tab and
 * carry a noopener relationship.
 */
const { test, expect } = require('@playwright/test');

test('cite shortcode renders as an external link with a kind badge', async ({ page }) => {
  await page.goto('/issues/03-sixteen-numbers/03-lloyd-max/');
  await page.waitForLoadState('networkidle');
  await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));

  const cites = page.locator('a.cite');
  const count = await cites.count();
  expect(count).toBeGreaterThan(0);

  const first = cites.first();
  await expect(first).toHaveAttribute('target', '_blank');
  const rel = await first.getAttribute('rel');
  expect(rel).toContain('noopener');
  expect(rel).toContain('external');

  // Badge must reflect the kind in the data-attribute / class.
  const className = await first.getAttribute('class');
  expect(className).toMatch(/cite--/);
  await expect(first.locator('.cite-badge')).toBeVisible();
});
