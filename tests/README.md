# UI Tests

Playwright tests for the Hugo site. They run in three viewports
(`mobile`, `tablet`, `desktop`) so a regression in one form factor is
caught even if the others still look fine.

## What's covered

- `toc.spec.js` — the sticky mobile TOC pill: tap to open, tap to
  close, child-element taps still hit the button, link tap closes the
  panel, outside tap closes, desktop stays inert.
- `section-numbering.spec.js` — H2/H3 get `"1."` / `"1.2"` prefixes;
  matching TOC links carry the same; shortcode-internal headings stay
  unnumbered; author-numbered headings are not double-prefixed.
- `article-fold.spec.js` — H2 sections become `<details>`; tapping the
  summary toggles them; every TOC fragment points at a real anchor.
- `issue-cover.spec.js` — reading list reaches the bottom; clicking a
  title navigates without toggling the card; clicking the chevron
  expands the nested outline; nested TOC links are absolute URLs.
- `techtree.spec.js` — Graph/List toolbar swaps panes; mobile defaults
  to list, desktop to graph; every list card is a real link.
- `timeline.spec.js` — 1–5 events render; mobile shows a dot row that
  tracks scroll; desktop hides it.
- `cite.spec.js` — `{{< cite >}}` renders as an external link with
  `target="_blank" rel="noopener external"` and a kind badge.

## Running

The first time, install deps inside `tests/`:

```bash
cd tests && npm install
```

Then from the repo root:

```bash
make test-ui                 # builds + serves + runs tests
make test-ui-debug           # same, with PWDEBUG inspector
```

Or directly:

```bash
cd tests
npm test                     # all three viewports
npm test -- --project=mobile # just the mobile suite
```

By default Playwright starts its own Hugo server on port `1314`. To run
against an already-running server, set `TEST_BASE_URL`:

```bash
TEST_BASE_URL=http://localhost:1313 npm test
```
