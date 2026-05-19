---
title: "Word2vec: the geometry of meaning"
description: "How a Google researcher's 2013 insight turned words into arrows — and why high-dimensional geometry turned out to be the perfect language for meaning."
issue: 1
layout: issue-cover
theme: cream
math: false
header: word2vec.webp
date: 2026-04-01T09:00:00-04:00
---

## The Mystery

September 2012. Tomáš Mikolov is staring at his training logs and thinking about something that shouldn't matter.

He is running a recurrent neural network language model at Google — one of the best such models in the world. The network is processing billions of words, and somewhere deep inside it, a 300-dimensional matrix is quietly learning to represent the entire vocabulary. This matrix is not the point of the model. It is just the input layer, a lookup table that converts each word into a starting vector before the *real* computation begins.

But the vectors it is learning are beautiful.

Mikolov had noticed this before. The embedding matrix always converged to something structured, something *geometric*. Words that appeared near similar words ended up near each other. The word "cat" drifted toward "dog" and away from "democracy." The word "king" moved toward "queen," "throne," and "royal." And strangely — he had spotted this while debugging — if you subtracted the vector for "man" from "king" and added the vector for "woman," you ended up frighteningly close to "queen."

He had a realization. The network was doing the easy part wrong. The embedding matrix was doing the hard part right.

What if the embedding matrix was the whole point?

What if the right question wasn't *"how do we build a good language model?"* but *"how do we build the smallest possible model that forces the embedding matrix to learn a good geometry?"*

That question, and the embarrassingly simple answer Mikolov found, changed natural language processing permanently.

## How To Read This Issue

Start with the [cold open](01-cold-open/) — a short scene-setter that shows, concretely, why the representation that everyone was using in 2012 was broken. After that, the **tech tree** below is your table of contents.

Arrows show which articles depend on which. The two **primers** (cream nodes) can be read in any order — [The Geometry of Agreement](03-dot-product/) teaches you the dot product as a measuring tool; [The Stranger Country](05-high-dimensional/) introduces the counterintuitive world of high-dimensional geometry. The **mainline chapters** (pink) tell the story; the **boss capstone** (yellow) connects everything to the LLMs of today.

{{< techtree name="issue01" >}}

## The Timeline

{{< timeline name="word2vec-history" >}}

## What You Will Walk Away With

By the end of this issue you should be able to answer, with confidence:

- Why **one-hot vectors** are mathematically broken — not just "bad practice" but actively lying about the relationship between words.
- What the **distributional hypothesis** is, why linguists believed it for sixty years before anyone could compute with it, and why the obvious computational approach was a dead end.
- Exactly how **Word2Vec's skip-gram model** works — what the two matrices are, why the training objective is a dot product comparison, and why removing the hidden layer was genius rather than laziness.
- Why high-dimensional space is **not like 3D space** — why random vectors are nearly perpendicular, why all the volume lives near the surface, and why this is a *gift* for embedding large vocabularies.
- What the **king − man + woman ≈ queen** equation actually means geometrically, and why vector arithmetic can express analogy.
- How Word2Vec's embedding layer became the **direct ancestor** of the embedding tables in every modern LLM — and how attention transformed static vectors into contextual ones.
- Why 300 dimensions, chosen somewhat arbitrarily in 2013, turns out to be more than sufficient for a vocabulary of a million words — and why modern LLMs use 4,096 to 16,384 dimensions anyway.

---

*Each chapter is self-contained. Mainline chapters link to the primers they build on — and each primer links forward to the mainline chapter where the math pays off.*
