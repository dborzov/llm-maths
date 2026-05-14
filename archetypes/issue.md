---
title: "{{ replace .File.ContentBaseName "-" " " | title }}"
description: ""
issue:
layout: issue-cover
theme: cream
math: false
header: default.webp
date: {{ .Date }}
---

## The Mystery

[Open with the framing question — the specific moment, the human stakes, what made the field move. See content/issues/03-sixteen-numbers/_index.md for the canonical example.]

## How To Read This Issue

Start with the cold-open. The **tech tree** below is the table of contents — each node is an article, arrows show dependencies.

{{ "{{< techtree name=\"issueNN\" >}}" | safeHTML }}

## What You Will Walk Away With

By the end of this issue you should be able to:

- ...

---

*Each chapter is self-contained. Mainline chapters link liberally to the primer chapters they build on — and each primer links forward to the mainline chapter where it pays off.*
