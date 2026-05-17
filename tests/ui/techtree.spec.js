// @ts-check
/**
 * The techtree shortcode ships two views: an SVG dependency graph and
 * a tap-friendly card list. The toolbar swaps between them; the default
 * depends on viewport width (list on mobile, graph on desktop).
 */
const { test, expect } = require('@playwright/test');

const COVER = '/issues/03-sixteen-numbers/';

test('techtree renders both panes; phone-narrow defaults to list, wider screens to graph', async ({ page }) => {
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  const tree = page.locator('[data-techtree]').first();
  await expect(tree).toBeVisible();

  const graph = tree.locator('[data-techtree-pane="graph"]');
  const list  = tree.locator('[data-techtree-pane="list"]');

  // Default behaviour follows the CSS breakpoint (max-width: 700px),
  // not the Playwright `isMobile` flag — tablets are touch but wide
  // enough to display the graph by default.
  const viewportW = page.viewportSize().width;
  if (viewportW <= 700) {
    await expect(list).toBeVisible();
    await expect(graph).toBeHidden();
  } else {
    await expect(graph).toBeVisible();
    await expect(list).toBeHidden();
  }
});

test('clicking Graph/List buttons swaps the active pane', async ({ page }) => {
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  const tree  = page.locator('[data-techtree]').first();
  const graph = tree.locator('[data-techtree-pane="graph"]');
  const list  = tree.locator('[data-techtree-pane="list"]');

  await tree.locator('[data-techtree-view="list"]').click();
  await expect(list).toBeVisible();
  await expect(graph).toBeHidden();

  await tree.locator('[data-techtree-view="graph"]').click();
  await expect(graph).toBeVisible();
  await expect(list).toBeHidden();
});

test('every list-view card is a tappable link to the article', async ({ page }) => {
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  const tree = page.locator('[data-techtree]').first();
  await tree.locator('[data-techtree-view="list"]').click();

  const links = tree.locator('.techtree-list-link[href]');
  const count = await links.count();
  expect(count).toBeGreaterThan(5);
  for (let i = 0; i < count; i++) {
    const href = await links.nth(i).getAttribute('href');
    expect(href).toMatch(/^\//); // resolved absolute
  }
});

test('the legend remains visible regardless of which pane is active', async ({ page }) => {
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  const tree = page.locator('[data-techtree]').first();
  await expect(tree.locator('.techtree-legend')).toBeVisible();
  await tree.locator('[data-techtree-view="list"]').click();
  await expect(tree.locator('.techtree-legend')).toBeVisible();
});
