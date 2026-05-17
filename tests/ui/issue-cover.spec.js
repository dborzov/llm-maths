// @ts-check
/**
 * Issue cover page: the "Recommended Reading Order" section is a list
 * of <details> cards. Each one expands to show that article's own TOC
 * with absolute-URL fragments. The title is a real link; the chevron
 * area toggles the card.
 */
const { test, expect } = require('@playwright/test');

const COVER = '/issues/03-sixteen-numbers/';

test('cover renders a numbered reading list', async ({ page }) => {
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  const items = page.locator('.issue-cover-article-item');
  const count = await items.count();
  // Issue 03 has 21 articles.
  expect(count).toBeGreaterThanOrEqual(20);
});

test('clicking an article TITLE navigates without toggling the card', async ({ page }) => {
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  const firstCard = page.locator('.issue-cover-article-item').first();
  const title = firstCard.locator('a.issue-cover-article-title');
  const href = await title.getAttribute('href');
  expect(href).toBeTruthy();

  await title.click();
  await page.waitForLoadState('networkidle');
  expect(page.url()).toContain(href);
});

test('clicking an article CARD (not the title) expands its outline', async ({ page }) => {
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  // Pick a card with a known TOC.
  const card = page.locator('.issue-cover-article-item').nth(4); // Lloyd-Max
  const detail = card.locator('.issue-cover-article-detail');
  const chevron = card.locator('.issue-cover-article-chev');

  await expect(detail).toHaveJSProperty('open', false);
  await chevron.click();
  await expect(detail).toHaveJSProperty('open', true);

  // The nested TOC must show at least one link.
  const innerToc = card.locator('.issue-cover-article-toc a');
  await expect(innerToc.first()).toBeVisible();
});

test('nested TOC links point at absolute URLs (not bare fragments)', async ({ page }) => {
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  const card = page.locator('.issue-cover-article-item').nth(4);
  await card.locator('.issue-cover-article-detail').evaluate(d => d.open = true);

  const links = card.locator('.issue-cover-article-toc a[href]');
  const count = await links.count();
  expect(count).toBeGreaterThan(0);
  for (let i = 0; i < count; i++) {
    const href = await links.nth(i).getAttribute('href');
    // Must include a slash before the fragment — proves it's absolute.
    expect(href, `link #${i}: ${href}`).toMatch(/\/[^#]*#/);
  }
});

test('mobile-only: the reading list reaches the bottom of the page', async ({ page, isMobile }) => {
  test.skip(!isMobile, 'desktop reading list is checked above');
  await page.goto(COVER);
  await page.waitForLoadState('networkidle');

  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(200);
  const lastVisible = await page.locator('.issue-cover-article-item').last().isVisible();
  expect(lastVisible).toBe(true);
});
