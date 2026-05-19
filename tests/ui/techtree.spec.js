// @ts-check
/**
 * The techtree shortcode ships two views: an SVG dependency graph and
 * a tap-friendly card list. The toolbar swaps between them; the default
 * depends on viewport width (list on mobile, graph on desktop).
 */
const { test, expect } = require('@playwright/test');

const COVER = '/comicbook/03-quantization/';

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

test('graph pane actually renders nodes (not just an empty SVG container)', async ({ page }) => {
  // Regression: Hugo's html/template engine was JS-escaping the JSON
  // inside <script type="application/json">, turning the data into a
  // string literal so JSON.parse returned a string and data.nodes was
  // undefined — the graph showed only the legend, no nodes.
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  const tree = page.locator('[data-techtree]').first();
  await tree.locator('[data-techtree-view="graph"]').click();

  const svg = tree.locator('.techtree-svg');
  await expect(svg).toBeVisible();
  await expect(tree.locator('.techtree-node')).not.toHaveCount(0);
  await expect(tree.locator('.techtree-edges path')).not.toHaveCount(0);
});

test('cross-issue (external) nodes are tappable links in both views', async ({ page }) => {
  // Regression: external nodes in issue06/issue07 had no `link` field
  // so they rendered as inert chips and unlinked SVG groups. They
  // should resolve to the article they reference, like any other node.
  await page.goto('/comicbook/06-kvcache-pruning/');
  await page.waitForLoadState('networkidle');

  const tree = page.locator('[data-techtree]').first();

  // List view: every external chip must be a real <a href>.
  await tree.locator('[data-techtree-view="list"]').click();
  const extChips = tree.locator('.techtree-list-item--external a.techtree-list-link');
  const chipCount = await extChips.count();
  expect(chipCount).toBeGreaterThan(0);
  for (let i = 0; i < chipCount; i++) {
    const href = await extChips.nth(i).getAttribute('href');
    expect(href, `external chip #${i} should have href`).toBeTruthy();
    expect(href).toMatch(/^\//);
  }

  // Graph view: every external SVG node must be wrapped in an <a>.
  await tree.locator('[data-techtree-view="graph"]').click();
  const extSvgLinks = tree.locator('.techtree-node--external a.techtree-node-link');
  await expect(extSvgLinks).toHaveCount(chipCount);
});
