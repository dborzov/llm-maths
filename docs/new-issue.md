# Authoring a New Issue

**When to read this:** You are creating a brand-new issue from scratch — picking a topic, scaffolding files, and writing the articles.

---

## The Fast Path

```bash
# 1. Pick a playful title and a slug. Sketch the dependency DAG on paper.
make new-issue NN=04 SLUG=attention-anatomy TITLE='Attention, Anatomized'
# → creates content/issues/04-attention-anatomy/_index.md
# → creates data/techtrees/issue04.toml (stub with example nodes)

# 2. Edit data/techtrees/issue04.toml — define the actual nodes and edges.
#    Schema reference: docs/techtrees-schema.md

# 3. Scaffold each article in the order the reader encounters them.
#    The script auto-increments NN and weight, and alternates the theme.
make new-article ISSUE=04-attention-anatomy SLUG=cold-open      TITLE='The Cold Open'          KIND=mainline
make new-article ISSUE=04-attention-anatomy SLUG=dot-products   TITLE='Dot Products Revisited'  KIND=primer
make new-article ISSUE=04-attention-anatomy SLUG=softmax-primer TITLE='Softmax & Friends'       KIND=primer
# ... etc

# 4. Fill in each article. Match the canonical style (docs/writing.md).
# 5. Validate continuously while you work.
make validate

# 6. Preview and ship.
make preview
```

The scaffolding scripts live at `scripts/new_issue.py`. Both `new_issue.py` and `validate.py` are idempotent — they refuse to overwrite existing files.

---

## What an Issue Contains

An **issue** is the unit of release. It contains multiple articles sharing one theme:

- A short, playful **issue title** that hooks readers.
- A **mystery / cold open** article that sets up the question the whole issue answers.
- A handful of **mainline narrative chapters** that progress the story.
- A handful of **primer / tech-tree chapters** — standalone tutorials on each prerequisite mathematical concept, in the spirit of James Burke's *Connections*.
- A **tech tree graph** rendered on the issue cover page, acting as the table of contents.
- **Heavy internal linking**: every time a mainline chapter mentions a load-bearing concept, link to the primer. Every primer links forward to the mainline article that uses it.

---

## Issue Cover (`_index.md`) Body

The cover's markdown body should:
1. Open with a few paragraphs setting up the mystery / cold-open framing.
2. Embed the tech-tree graph: `{{< techtree name="issueNN" >}}`.
3. Optionally end with a short reading-order note. The article list renders automatically below.

---

## Tech Tree Design

See `docs/techtrees-schema.md` for the full TOML schema.

Reference implementation: `data/techtrees/issue03.toml`.

Sketch the dependency DAG first, on paper. Nodes are articles; edges express "you should read X before Y". The cold-open article is typically not in the tree (it has no prereqs). Primer nodes are cream-colored; mainline nodes are pink; the boss node is yellow halftone.

---

## Writing the Articles

For all writing guidance — voice, narrative arc, quality checklist, formatting tools — see [`docs/writing.md`](writing.md).

For microGPT naming rules and wiki shortcode usage (required for any article touching transformer internals) — see [`docs/microgpt-contract.md`](microgpt-contract.md) and [`docs/wiki.md`](wiki.md).

For front matter schemas and file naming — see [`docs/issue-format.md`](issue-format.md).

---

## Canonical Reference

Issue 03 (`content/issues/03-sixteen-numbers/`) is the reference implementation. When in doubt, copy what it does.
