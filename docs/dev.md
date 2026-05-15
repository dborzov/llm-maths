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
4. `make preview` and visually inspect affected pages at `http://localhost:1313/llm-maths/`.

`make build` chains steps 1–3, so it's the single command for a full pre-deploy check.

---

## Validation

`make validate` runs `scripts/validate.py`, which checks four invariants:

1. **Front matter contract**: every article has all required fields with valid types and enum values.
2. **Tech tree consistency**: every issue's `{{< techtree >}}` shortcode references a real data file; every article's `techNode` matches a node id; every tree-node `link` resolves to an actual article file.
3. **Cross-link integrity**: every `[text](../slug/)` link inside an article resolves to a real sibling article in the same issue.
4. **Pyplot rules**: unique ids, no forbidden imports (scipy/torch/etc.), no `plt.show`/`plt.savefig`.

`make build` and `make preview` both run validate first. A failed validate fails the build.

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

- `enableGitInfo = false` in `hugo.toml`. CI passes `--enableGitInfo` explicitly.
- `baseURL = "https://borzov.ca/llm-maths/"` — all URLs include the `/llm-maths/` prefix.
- CSS via Hugo Pipes: `resources.Get | minify | fingerprint`.
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
