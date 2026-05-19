# Borzov's LLM Maths — Claude Code Instructions

**Project name:** "LLM Maths" (short) / "Borzov's LLM Maths" (full). Do not use any other name variant ("Comics", "Almanac", "LLM Maths Comics", etc.).

A static website: a serialized collection of long-form deep-learning mathematics articles structured as **issues** (like comic-book issues). Each issue groups multiple articles around a single theme and presents them via a **tech tree graph** as table of contents. Pop-art neubrutalist visual design. Hugo + custom theme + Python pre-build script.

Live at: `https://borzov.ca/llm-maths/`

**Canonical reference implementation:** `content/comicbook/03-quantization/` — when in doubt, copy what issue 03 does.

**Canonical LLM vocabulary:** `content/comicbook/05-microgpt/` introduces **microGPT** — the shared reference implementation. Every transformer concept is named after its variable there. Canonical names and aliases live in the wiki (`content/wiki/transformer/`); look them up with `uv run scripts/wiki_index.py search <term>`. Writing rules are in [`docs/microgpt-contract.md`](docs/microgpt-contract.md).

**Wiki / glossary:** Recurring concepts have stub pages at `content/wiki/` — each stub holds the canonical slug, aliases, and a link to where the concept is defined. Use `{{< wiki "slug" >}}term{{< /wiki >}}` shortcode on first mention in any article. See [`docs/wiki.md`](docs/wiki.md).

---

## Which Doc to Read

| Task | Read |
|---|---|
| Write or outline a new comic book from scratch | [`docs/new-issue.md`](docs/new-issue.md) |
| Write, polish, or deepen an article | [`docs/writing.md`](docs/writing.md) — authoritative |
| **Write a good title and blurb for an article** | [`docs/writing-titles.md`](docs/writing-titles.md) — the pattern, worked example, failure modes |
| Add components or make an article visually richer | [`docs/components/README.md`](docs/components/README.md) |
| Front matter fields (`blurb`, `short_title`, etc.) | [`docs/issue-format.md`](docs/issue-format.md) |
| microGPT naming rules + wiki shortcode usage | [`docs/microgpt-contract.md`](docs/microgpt-contract.md) + [`docs/wiki.md`](docs/wiki.md) |
| Wiki / glossary — add or update concept stubs | [`docs/wiki.md`](docs/wiki.md) |
| CSS, templates, scripts, validation, CI, pitfalls | [`docs/dev.md`](docs/dev.md) |
| **Adding or running UI tests** | [`tests/README.md`](tests/README.md) + [`docs/dev.md`](docs/dev.md) → "Testing" |
| Tech tree TOML schema | [`docs/techtrees-schema.md`](docs/techtrees-schema.md) |
| Timeline TOML schema | [`docs/timelines-schema.md`](docs/timelines-schema.md) |

---

## Quick Commands

```bash
make validate        # lint front matter, tech tree, links, pyplot rules, wiki
make preview         # validate + plots + serve at localhost:1313/llm-maths/
make build           # validate + plots + minified build to public/

make test-ui-install # one-time: install Playwright + chromium under tests/
make test-ui         # run the mobile/tablet/desktop UI test suite

uv run scripts/wiki_index.py search <term>   # find the right wiki slug for a concept
uv run scripts/wiki_index.py show <slug>     # full record + usages
uv run scripts/wiki_index.py lint            # wiki-only warnings (also run by make validate)
```

Full command reference in [`docs/dev.md`](docs/dev.md).

---

## Testing — Non-Negotiable Rules

The `tests/` directory holds a Playwright UI test suite that pins down
how every interactive component is supposed to behave. Treat the suite
the way you treat `make validate`: **a green test run is part of
"done", not an optional extra**.

**When to run tests:**

- Before you call any task involving CSS, templates, JS, shortcodes, or
  layouts complete — `make test-ui` must pass on all three viewports.
- After every fix you push in response to a regression — the failure
  that motivated the fix should now be covered by a test.

**When to add or expand tests:**

- You add a new interactive component or shortcode → add a new
  `tests/ui/<feature>.spec.js` covering its tap, click, keyboard,
  and viewport-breakpoint behaviour.
- You fix a UI bug → write a test that fails on the unfixed code and
  passes after your fix. This is non-optional. If the bug was "tapping
  on X did nothing", the test taps on X and asserts the expected state
  change. See `tests/ui/toc.spec.js` for the pattern that the iOS
  Safari toggle bug spawned.
- You change a layout breakpoint or a JS handler → update the relevant
  spec to assert the new behaviour, don't loosen it to "make it pass".
- The user's intent in the prompt implies a contract ("must work on
  phones", "every node should be tappable", "all UI is expandable")
  → translate that intent into a test before declaring the task
  complete.

**Work-until-green loop.** If `make test-ui` reports failures:

1. Read the failure (Playwright prints the call log + a trace path).
2. Open the failing page in `make preview` and reproduce it manually.
3. Fix the underlying code — never weaken or skip the assertion to
   "make the red go away".
4. Re-run `make test-ui`. Repeat until 0 failed.
5. If you cannot make a test pass and you believe the test is wrong,
   stop and ask the user — don't quietly delete or disable it.

The test infrastructure, file conventions, and worked examples live in
[`tests/README.md`](tests/README.md). The "Testing" section of
[`docs/dev.md`](docs/dev.md) explains the loop in more depth.

---

## When You Are Stuck

- "What does a good article look like?" → read `content/comicbook/03-quantization/06-outliers.md` (mainline) and `content/comicbook/03-quantization/03-lloyd-max.md` (primer).
- "Which component do I use here?" → [`docs/components/README.md`](docs/components/README.md). Then drill into the per-component file.
- "What does a good tech tree look like?" → see `data/techtrees/issue03.toml`; schema in [`docs/techtrees-schema.md`](docs/techtrees-schema.md).
- "What does a good timeline look like?" → see `data/timelines/quantization2019to2026.toml`; schema in [`docs/timelines-schema.md`](docs/timelines-schema.md).
- "What conventions am I forgetting?" → run `make validate`. It tells you.
- "What's the canonical slug/name for this concept?" → `uv run scripts/wiki_index.py search <term>`. If not found, add a wiki page per [`docs/wiki.md`](docs/wiki.md).
- "Is my UI change actually working on phones?" → `make test-ui`. If you changed anything interactive and didn't add a test for it, you're not done.
