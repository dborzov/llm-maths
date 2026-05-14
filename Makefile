.PHONY: help preview build plots plots-force validate clean clean-all new-issue new-article

help:
	@echo "Common targets:"
	@echo "  make preview     validate, run plots (cached), serve at http://localhost:1313/llm-maths/"
	@echo "  make build       validate, run plots, build minified site to public/"
	@echo "  make validate    run the convention linter (front-matter, tech tree, cross-links, pyplot)"
	@echo "  make plots       run pyplot pre-build (cached)"
	@echo "  make plots-force re-run all pyplot blocks regardless of cache"
	@echo "  make clean       delete public/ + resources/"
	@echo "  make clean-all   + delete plot cache and generated PNGs"
	@echo ""
	@echo "Scaffolding (see CLAUDE.md → 'Authoring A New Issue'):"
	@echo "  make new-issue NN=NN SLUG=slug TITLE='Title'"
	@echo "  make new-article ISSUE=NN-slug SLUG=slug TITLE='Title' [KIND=primer|mainline|boss]"

preview: validate
	uv run scripts/run_plots.py && hugo serve --buildDrafts --navigateToChanged

build: validate
	uv run scripts/run_plots.py && hugo --minify

plots:
	uv run scripts/run_plots.py

plots-force:
	uv run scripts/run_plots.py --force

validate:
	@uv run scripts/validate.py

clean:
	rm -rf public/ resources/

clean-all: clean
	rm -f scripts/.plot_cache.json
	find static/plots -name "*.png" -delete

# ---------------------------------------------------------------------------
# Scaffolding wrappers around scripts/new_issue.py
#
#   make new-issue NN=04 SLUG=attention-anatomy TITLE='Attention, Anatomized'
#   make new-article ISSUE=04-attention-anatomy SLUG=cold-open TITLE='Cold Open' KIND=mainline
# ---------------------------------------------------------------------------

KIND ?= primer

new-issue:
	@test -n "$(NN)"    || (echo 'usage: make new-issue NN=NN SLUG=slug TITLE="Title"'; exit 1)
	@test -n "$(SLUG)"  || (echo 'usage: make new-issue NN=NN SLUG=slug TITLE="Title"'; exit 1)
	@test -n "$(TITLE)" || (echo 'usage: make new-issue NN=NN SLUG=slug TITLE="Title"'; exit 1)
	uv run scripts/new_issue.py issue $(NN) $(SLUG) $(TITLE)

new-article:
	@test -n "$(ISSUE)" || (echo 'usage: make new-article ISSUE=NN-slug SLUG=slug TITLE="Title" [KIND=primer|mainline|boss]'; exit 1)
	@test -n "$(SLUG)"  || (echo 'usage: make new-article ISSUE=NN-slug SLUG=slug TITLE="Title" [KIND=primer|mainline|boss]'; exit 1)
	@test -n "$(TITLE)" || (echo 'usage: make new-article ISSUE=NN-slug SLUG=slug TITLE="Title" [KIND=primer|mainline|boss]'; exit 1)
	uv run scripts/new_issue.py article $(ISSUE) $(SLUG) $(TITLE) --kind $(KIND)
