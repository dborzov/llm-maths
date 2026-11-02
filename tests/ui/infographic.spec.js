// @ts-check
/**
 * {{< infographic >}} panels are hand-wired vanilla JS that lives inside an
 * article. This spec pins down the contract of the "Fair Price vs.
 * Overcharge" panel in Issue 02 ch.7 (cross-entropy / KL divergence):
 *
 *   1. The inline script runs: readouts are filled in on load (no "—"
 *      placeholders) and match the article's numbers for the default tray.
 *   2. Every tray toggle is visible, fits inside the viewport, and works on
 *      phone, tablet and desktop. Clicking one makes it the ONLY pressed
 *      button and re-prices the bars.
 *   3. Setting the model's belief q equal to the tray's true rate p puts the
 *      bill on the floor: overcharge (KL) reads exactly zero and the bill
 *      equals the fair price. This is the identity H(p,q) = H(p) + KL(p||q)
 *      the article is built around.
 *   4. Away from q = p, forward KL(p||q) and reverse KL(q||p) disagree.
 */
const { test, expect } = require('@playwright/test');

const ARTICLE = '/comicbook/02-logits/07-cross-entropy/';

/** Load the article, open every folded H2 section, return the panel. */
async function openPanel(page) {
  await page.goto(ARTICLE);
  await page.waitForLoadState('networkidle');
  await page.evaluate(() => document.querySelectorAll('details').forEach((d) => (d.open = true)));
  const panel = page.locator('aside.infographic', { has: page.locator('#ig-ce-q') });
  await expect(panel).toHaveCount(1);
  await panel.scrollIntoViewIfNeeded();
  return panel;
}

/** Drive the range input the way a drag does: set value, fire `input`. */
async function setQ(page, percent) {
  await page.locator('#ig-ce-q').evaluate((el, v) => {
    /** @type {HTMLInputElement} */ (el).value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, percent);
}

/** Read a readout like "0.958 bits" as a number. */
async function bits(page, key) {
  return parseFloat((await page.locator(`#ig-ce-val-${key}`).textContent()) || 'NaN');
}

const TRAYS = [
  { label: '1 mg', percent: 6, fair: '0.327 bits' },
  { label: '3 mg', percent: 38, fair: '0.958 bits' },
  { label: '5 mg', percent: 68, fair: '0.904 bits' },
  { label: '12 mg', percent: 96, fair: '0.242 bits' },
];

test('cross-entropy panel fills its readouts on load', async ({ page }) => {
  await openPanel(page);
  await expect(page.locator('#ig-ce-q-out')).toHaveText('52%');
  await expect(page.locator('#ig-ce-caption')).toContainText('38%');
  // Default: 3 mg tray (p = 0.38) priced by the bookie (q = 0.52).
  await expect(page.locator('#ig-ce-val-h')).toHaveText('0.958 bits');
  await expect(page.locator('#ig-ce-val-ce')).toHaveText('1.015 bits');
  await expect(page.locator('#ig-ce-val-kl')).toHaveText('0.057 bits');
  for (const key of ['h', 'ce', 'kl', 'rkl']) {
    const width = await page.locator(`#ig-ce-bar-${key}`).evaluate((el) => el.style.width);
    expect(width, `bar ${key} never got a width`).toMatch(/%$/);
  }
});

test('every tray toggle is on-screen and re-prices the bars', async ({ page }) => {
  const panel = await openPanel(page);
  const vw = page.viewportSize()?.width ?? 0;
  await expect(panel.locator('[data-ce-tray]')).toHaveCount(TRAYS.length);

  for (const tray of TRAYS) {
    const btn = panel.getByRole('button', { name: tray.label, exact: true });
    await expect(btn).toBeVisible();
    const box = await btn.boundingBox();
    expect(box, `${tray.label} has no layout box`).not.toBeNull();
    if (box) {
      expect(box.x, `${tray.label} starts off-screen`).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width, `${tray.label} overflows the viewport`).toBeLessThanOrEqual(vw + 1);
    }

    await btn.click();
    await expect(btn).toHaveAttribute('aria-pressed', 'true');
    await expect(panel.locator('[data-ce-tray][aria-pressed="true"]')).toHaveCount(1);
    await expect(page.locator('#ig-ce-caption')).toContainText(`${tray.percent}%`);
    await expect(page.locator('#ig-ce-val-h')).toHaveText(tray.fair);
  }
});

test('tapping a tray toggle works on touch screens', async ({ page, hasTouch }) => {
  test.skip(!hasTouch, 'touch-only contract');
  const panel = await openPanel(page);
  const btn = panel.getByRole('button', { name: '5 mg', exact: true });
  await btn.tap();
  await expect(btn).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#ig-ce-val-h')).toHaveText('0.904 bits');
});

test('q = p puts the bill on the floor: zero overcharge', async ({ page }) => {
  const panel = await openPanel(page);
  for (const tray of TRAYS) {
    await panel.getByRole('button', { name: tray.label, exact: true }).click();
    await setQ(page, tray.percent);
    await expect(page.locator('#ig-ce-q-out')).toHaveText(`${tray.percent}%`);
    await expect(page.locator('#ig-ce-val-kl')).toHaveText('0.000 bits');
    await expect(page.locator('#ig-ce-val-rkl')).toHaveText('0.000 bits');
    await expect(page.locator('#ig-ce-val-ce')).toHaveText(tray.fair);
    // The floor marker on the bill bar sits exactly where the bill ends.
    const markLeft = await page.locator('#ig-ce-mark').evaluate((el) => el.style.left);
    const billWidth = await page.locator('#ig-ce-bar-ce').evaluate((el) => el.style.width);
    expect(markLeft).toBe(billWidth);
  }
});

test('forward and reverse KL disagree away from the floor', async ({ page }) => {
  const panel = await openPanel(page);
  await panel.getByRole('button', { name: '12 mg', exact: true }).click();
  await setQ(page, 10);
  const fwd = await bits(page, 'kl');
  const rev = await bits(page, 'rkl');
  expect(fwd).toBeCloseTo(2.953, 2);
  expect(rev).toBeCloseTo(3.717, 2);
  // The bill is always fair price + overcharge (up to display rounding).
  expect(await bits(page, 'ce')).toBeCloseTo((await bits(page, 'h')) + fwd, 2);
});
