# UI Tests

Playwright tests for the Hugo site. They pin down the interactive
contract — what taps, clicks, keyboard shortcuts, and viewport
breakpoints are supposed to do — across three viewports (`mobile`,
`tablet`, `desktop`).

The full operating-procedure summary lives in
[`docs/dev.md`](../docs/dev.md) → "Testing". This README is the
quickstart and the per-spec contract reference.

---

## TL;DR

```bash
make test-ui-install   # one-time: npm install + browser
make test-ui           # run mobile/tablet/desktop suites
```

A green run is part of "done" for any UI-touching change. The full
loop — when to run, when to add, what to do when it fails — is in
[`docs/dev.md`](../docs/dev.md).

---

## What each spec pins down

- `toc.spec.js` — the sticky mobile TOC pill: tap opens, tap closes,
  every child element of the bar (icon, label, chevron) still opens
  it, tapping a TOC link closes the panel, tapping outside closes,
  Escape closes, desktop stays inert.
- `section-numbering.spec.js` — H2/H3 get `"1."` / `"1.2"` prefixes;
  matching TOC links carry the same; shortcode-internal headings stay
  unnumbered; author-numbered headings are not double-prefixed.
- `article-fold.spec.js` — H2 sections become `<details>`; clicking the
  summary toggles them; every TOC fragment points at a real anchor.
- `issue-cover.spec.js` — reading list reaches the bottom; clicking a
  title navigates without toggling the card; clicking the chevron
  expands the nested outline; nested TOC links are absolute URLs.
- `techtree.spec.js` — Graph/List toolbar swaps panes; phone-narrow
  defaults to list, wider screens to graph; every list card is a
  real link; the legend stays visible.
- `timeline.spec.js` — 1–5 events render; phone-narrow shows a dot
  row that tracks scroll; wider screens hide the dots.
- `cite.spec.js` — `{{< cite >}}` renders as an external link with
  `target="_blank" rel="noopener external"` and a kind badge.
- `figure.spec.js` — `{{< figure >}}` image actually loads (no 404 src);
  caption text gets enough horizontal room on every viewport; on phones
  the `figure-credit` wraps onto its own row instead of squeezing the
  caption into a one-word column.

---

## Adding a new spec

1. **One feature, one file.** Name it after the CSS class or shortcode
   you're testing, not after the bug or task that motivated it.
2. **Open with a doc-comment** that states the contract in plain
   English. A reader six months from now should be able to tell, in
   ten seconds, what behaviour this file is guarding.
3. **Use `page.click()` for cross-viewport interactions**; use
   `page.tap()` only inside a `test.skip(!hasTouch, …)` guard.
4. **Gate viewport-specific assertions on `page.viewportSize().width`**
   (not Playwright's `isMobile` flag) — the CSS uses pixel widths and
   that's what readers experience.
5. **For UI bug fixes, write the test first** in red, then fix. If the
   bug was "tap on X did nothing", write `page.tap('selector-for-X')`
   and assert the expected state change. If the bug had a specific
   cause (e.g. taps on child spans not bubbling), write one test per
   child — the point is to catch the next variant of the same mistake.
6. **Don't loosen an assertion** to make a test pass. If you genuinely
   think the test is wrong, stop and ask before changing it.

---

## Running specific subsets

```bash
cd tests
npm test                                # all projects, all specs
npm test -- --project=mobile            # just mobile
npm test -- ui/toc.spec.js              # one file, all projects
npm test -- --project=desktop --grep="Escape"   # filter by test name
```

Debug mode opens the Playwright inspector:

```bash
make test-ui-debug
```

Tracing is on by default for failures — the failure message includes a
path under `tests/test-results/`. Open it with:

```bash
cd tests && npx playwright show-trace test-results/<dir>/trace.zip
```

---

## Running against an already-running Hugo

By default Playwright starts its own Hugo on port 1314 so the suite
runs in isolation. If you already have `make preview` going on port
1313, point the tests at it instead:

```bash
TEST_BASE_URL=http://localhost:1313/llm-maths cd tests && npm test
```

This skips the per-run Hugo spin-up and gives a tight edit/run loop.

---

## CI / non-default browser locations

- `HUGO_BIN=/path/to/hugo` — use a specific Hugo binary if `hugo` is
  not on `PATH`.
- `CHROMIUM_BIN=/path/to/chrome` — use a pre-installed Chromium
  instead of the one Playwright would download. Useful in sandboxed
  environments where `npx playwright install` can't reach the network.
- `TEST_BASE_URL=…` — see above; bypasses the embedded Hugo server.
- `CI=1` — Playwright enables one retry and the HTML reporter.
