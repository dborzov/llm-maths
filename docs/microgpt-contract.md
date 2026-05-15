# microGPT Terminology Contract

**When to read this:** You are writing or editing any article that touches transformer internals — attention, embeddings, MLP blocks, KV cache, inference phases, or any concept covered in Issue 5.

---

## Why This Exists

Issue 5 (`content/issues/05-microgpt-unfolded/`) introduces a 60-line plain-Python transformer — **microGPT** — that is the shared reference implementation for the entire project. Every transformer concept on this site is named after its variable in microGPT.

The canonical names, aliases, and source-of-truth links now live in the **wiki**. Look them up there:

```bash
uv run scripts/wiki_index.py search "key projection"   # find a slug by alias
uv run scripts/wiki_index.py show attention             # full record with all aliases
```

Relevant wiki pages: [`transformer-weights`](/wiki/transformer/transformer-weights/), [`attention`](/wiki/transformer/attention/), [`kv-cache`](/wiki/transformer/kv-cache/), [`hyperparameters`](/wiki/transformer/hyperparameters/).

---

## Rules When Writing an Article

1. **Use the microGPT name on first mention** — not a synonym. Write "the `attn_wk` projection" not "the key matrix `W_K`". If the source paper uses different notation, introduce both: *"the K projection (`attn_wk` in our reference listing)"*.

2. **Wrap the first mention in a wiki shortcode** so readers can follow to the canonical definition:
   ```markdown
   The {{< wiki "attention" >}}`attn_wk` projection{{< /wiki >}} is computed as…
   ```
   Run `uv run scripts/wiki_index.py search <term>` to find the right slug.

3. **Show the relevant microGPT slice** when discussing a specific line — copy-paste the 1–4 line excerpt verbatim, do not rewrite it.

4. **For shape questions, defer to the canonical table** in [ch.1 Sixty Lines, One LLM](/issues/05-microgpt-unfolded/01-cold-open/#the-names-you-should-tattoo). Do not re-derive shapes per article.

---

## Adding New Recurring Terms

When you identify a term that will recur across multiple articles:

1. Add a wiki stub: `content/wiki/<category>/<slug>.md` — see [`docs/wiki.md`](wiki.md) for the format.
2. Run `uv run scripts/wiki_index.py rebuild` to index it.
3. Use `{{< wiki "slug" >}}` on first mention in each article.

The wiki is the continuously-expanding terminology registry. Do not add new term tables to this file.
