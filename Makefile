.PHONY: preview build plots plots-force clean clean-all

preview:
	uv run scripts/run_plots.py && hugo serve --buildDrafts --navigateToChanged

build:
	uv run scripts/run_plots.py && hugo --minify

plots:
	uv run scripts/run_plots.py

plots-force:
	uv run scripts/run_plots.py --force

clean:
	rm -rf public/ resources/

clean-all: clean
	rm -f scripts/.plot_cache.json
	find static/plots -name "*.png" -delete
