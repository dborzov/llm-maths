# Developer and Meta Reference

**When to read this:** You are working on CSS, Hugo templates, scripts, validation, GitHub Actions, or any housekeeping task. Also read this before running any build command.

---

## Essential Commands

```bash
make help            # list all targets with their purposes
make preview         # validate, run plots, serve at http://localhost:1313/llm-maths/
make build           # validate, run plots, build minified site to public/
make validate        # run the linter on its own
make plots           # only run pyplot executor (skips unchanged blocks)
make plots-force     # re-run all pyplot blocks regardless of cache
make clean           # delete public/ + resources/
make clean-all       # + delete plot cache and generated PNGs

make test-ui-install # one-time: npm install + Playwright chromium
make test-ui         # run UI tests across mobile/tablet/desktop
make test-ui-debug   # same, with PWDEBUG inspector

make new-issue NN=04 SLUG=foo TITLE='Foo'
make new-article ISSUE=04-foo SLUG=cold-open TITLE='Cold Open' KIND=mainline
```

**Python tooling:** always use `uv` — never bare `python` or `pip`.

---

## Sanity-Check Protocol

After every significant change (CSS, templates, content, scripts), run in order:

1. `make validate` — must exit clean.
2. `uv run scripts/run_plots.py` — must exit with 0 errors.
3. `hugo` — must build cleanly.
4. **If you touched CSS, JS, templates, shortcodes, or layouts:**
   `make test-ui` — must exit with 0 failed.
5. `make preview` and visually inspect affected pages at `http://localhost:1313/llm-maths/`.

`make build` chains steps 1–3, so it's the single command for a full
pre-deploy check. The Playwright UI suite (step 4) is intentionally a
separate target — it's slower (~45s) and only relevant when you've
changed something interactive — but it is **not optional** when you
have. See the "Testing" section below for the loop.

---

## Validation

`make validate` runs `scripts/validate.py`, which checks four invariants:

1. **Front matter contract**: every article has all required fields with valid types and enum values.
2. **Tech tree consistency**: every issue's `{{< techtree >}}` shortcode references a real data file; every article's `techNode` matches a node id; every tree-node `link` resolves to an actual article file.
3. **Cross-link integrity**: every `[text](../slug/)` link inside an article resolves to a real sibling article in the same issue.
4. **Pyplot rules**: unique ids, no forbidden imports (scipy/torch/etc.), no `plt.show`/`plt.savefig`.

`make build` and `make preview` both run validate first. A failed validate fails the build.

---

## Testing (Playwright UI suite)

`make validate` covers *content* invariants. The Playwright suite under
`tests/` covers *interaction* invariants — clicks, taps, keyboard
shortcuts, viewport breakpoints, and the things that go wrong silently
on mobile when no human is looking. The full reference, file layout,
and how to install the browser lives in [`tests/README.md`](../tests/README.md).
This section is the operating-procedure summary.

### When you must run it

- You changed anything in `static/js/`, `themes/almanac/assets/css/`,
  `themes/almanac/layouts/`, or any shortcode template.
- You fixed a UI bug.
- You changed a CSS breakpoint, a viewport rule, or a JS event handler.
- You're about to mark a UI-touching task as complete.

If you only changed Markdown content (no front-matter fields, no
shortcode arguments, no templates), `make validate` is enough.

### When you must add or expand a test

- **New interactive component** — add a `tests/ui/<feature>.spec.js`
  that exercises every tap target, every state transition, and the
  viewport breakpoints the component cares about. Cover each of the
  three projects (`mobile`, `tablet`, `desktop`) it claims to support.
- **UI bug fix** — write a test that fails on the unfixed code first,
  then make it pass. The test must reproduce the user's reported
  failure mode, not a paraphrase. Example: "tapping the Contents pill
  doesn't open the panel" → a `page.tap('.toc-toggle')` followed by an
  assertion that the panel is now open. Then for good measure, the
  same assertion after tapping each child element, since that was the
  *specific* cause.
- **The user's prompt implies a contract you didn't have before** —
  "must work on phones", "every node should be tappable", "nothing
  should require a mouse" — turn that intent into one or more assertions
  before you call the task done. The intent is the spec.
- **You added a breakpoint or default-state rule** — assert it. If
  mobile defaults to list view and desktop to graph view, you need one
  test per case.

### How tests are structured

```
tests/
├── playwright.config.js     # three projects: mobile / tablet / desktop
├── ui/
│   ├── article-fold.spec.js     # H2 auto-fold details
│   ├── cite.spec.js             # {{< cite >}} shortcode
│   ├── issue-cover.spec.js      # nested reading list
│   ├── section-numbering.spec.js
│   ├── techtree.spec.js         # Graph/List toggle
│   ├── timeline.spec.js         # scroll-snap + dots
│   └── toc.spec.js              # sticky mobile pill
└── README.md
```

Conventions:

- One spec file per feature. Name it after the CSS class / shortcode
  it tests, not after the task that motivated it.
- Top-of-file docstring states exactly what contract this file pins down.
- Use `page.click()` for interactions that work on every viewport;
  use `page.tap()` only inside `test.skip(!hasTouch, …)` guards.
- Gate viewport-specific assertions on `page.viewportSize().width`,
  not on Playwright's `isMobile` flag — the CSS uses pixel widths.
- If a regression had a specific cause ("clicks on child spans missed
  the button"), write a test per child element. The point is to catch
  the next variant of the same mistake.

### The work-until-green loop

`make test-ui` either reports `… N passed (Xs)` or fails. Failure is
the start of the loop, not the end of the task:

1. **Read the failure.** Playwright prints the failing assertion,
   the call log, and the path to a `.zip` trace under
   `tests/test-results/`. Open the trace with
   `npx playwright show-trace <path>` for a full screencast.
2. **Reproduce manually.** Run `make preview`, open the same URL the
   test loads, do the same action in your browser. Confirm the bug
   exists outside the test runner.
3. **Fix the underlying code.** Never weaken the assertion, increase
   the timeout, or wrap the test in `test.skip` to make the red go
   away. If you genuinely believe the test is wrong, stop and ask.
4. **Re-run the failing project only**, fast:
   ```bash
   cd tests && npx playwright test --project=mobile ui/toc.spec.js
   ```
5. **Re-run the full suite** before you commit:
   ```bash
   make test-ui
   ```
   It must finish `0 failed` across all three projects. `N skipped`
   is fine — that means the test correctly opted out of a viewport
   where it doesn't apply.
6. Commit and push.

### Running against an existing Hugo server

The Playwright config starts its own Hugo on port 1314, but if you
already have `make preview` running, point the tests at it:

```bash
TEST_BASE_URL=http://localhost:1313/llm-maths cd tests && npm test
```

This skips the per-run Hugo spin-up and gives you a tight edit/run loop.

---

## Dependency Management

```bash
uv run <command>          # runs in project venv
uv run --isolated ...     # throwaway venv (used by run_plots.py internally)
uv add <package>          # add to pyproject.toml
```

Never use bare `pip install`. Never use `apt` for Python packages.

---

## Hugo Notes

- `enableGitInfo = true` in `hugo.toml`. CI also passes `--enableGitInfo`.
- `baseURL = "https://borzov.ca/llm-maths/"` — all URLs include the `/llm-maths/` prefix.
- **CSS pipeline:** `themes/almanac/assets/css/main.css` → Hugo Pipes `minify | fingerprint` → served as `/css/main.<hash>.css`.
- **JS pipeline:** `themes/almanac/assets/js/*.js` → Hugo Pipes `minify | fingerprint` per file. Add new JS files here (not in `static/js/`). The old `static/js/` copies are kept for backwards compatibility but are unused by Hugo.
- **KaTeX:** loaded from jsDelivr CDN only on pages with `math: true` front matter. Do not add `.IsHome` back — the homepage has no math.
- The `prompts/` section lives **outside** `content/` (at repo root) — Hugo never sees it.

---

## GitHub Actions

`.github/workflows/deploy.yml` triggers on push to `main`:
1. `uv run scripts/run_plots.py`
2. `hugo --minify --enableGitInfo`
3. Deploy to GitHub Pages via `actions/deploy-pages@v4`

---

## Design System

From the `styles/` app (React/Vite reference of the visual language):

- **No rounded corners.** Zero `border-radius` everywhere.
- **Hard shadows.** `4px 4px 0px #1A1A1A` — no blur.
- **Thick borders.** `3px solid #1A1A1A` on structural elements.
- **Halftone dots.** `radial-gradient(#1A1A1A 1px, transparent 0); background-size: 5px 5px`
- **Bangers** (Google Fonts) for display/headings, **Space Grotesk** for body text.
- No JavaScript framework on the site — vanilla JS only.

---

## Pitfalls — Don't Re-Discover Them

These are mistakes from prior sessions that took non-trivial time to debug.

1. **Shortcode `relURL` is rooted at the site, not the calling page.** Writing `{{ .link | relURL }}` in a shortcode resolves to `/llm-maths/06-foo/` rather than `/llm-maths/issues/03-NN/06-foo/`. **Fix**: capture `.Page.RelPermalink` (or `.Page.Parent.RelPermalink` if the calling page is an article) into a local variable before entering the range loop, and prepend it to relative links. See `techtree.html` and `timeline.html` for the pattern.

2. **Inside Go template `range` blocks, `$` is the root context — NOT the parent iterator.** `$.x` will fail if `.` was rebound. **Fix**: capture the current item with `{{- $node := . -}}` immediately upon entering the range, then use `$node.x`.

3. **`_build.render: never` in `cascade` is unreliable across Hugo versions.** Trying to hide a content section via cascade left the child pages building anyway. **Fix**: keep non-publishable files OUTSIDE `content/`. Authoring prompts live at the repo root in `prompts/`, not in `content/prompts/`.

4. **Pyplot isolated env has only numpy + matplotlib.** Importing `scipy.stats.norm` will pass local linting but fail the plot run. **Fix**: use numpy-only equivalents (e.g. hardcoded quantile values). `make validate` catches this preemptively.

5. **Hugo template variable `:=` re-declaration in the same scope is an error.** If you copy a `{{ $x := ... }}` from one block to another in the same template, the second one fails. **Fix**: declare once at the top, reuse below.

6. **Hugo `Parent.RelPermalink` for the top-level issues section is `/llm-maths/issues/`, NOT `/issues/`.** Comparing against the literal `/issues/` will fail. **Fix**: compare against `(site.GetPage "/issues").RelPermalink`.
