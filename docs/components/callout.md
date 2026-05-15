# Callout

**Boxed sidebar for content that runs alongside the main narrative.** The all-purpose "footnote that needs more than a sentence" — tangents, tips, warnings, formal definitions.

## How to invoke

Use the **`%%` delimiter form** so Markdown inside renders (bold, lists, KaTeX, code):

```markdown
{{% callout type="tip" title="Pro-Tip" %}}
When implementing eviction, always preserve the first 4 tokens (the *attention sinks*).
Otherwise perplexity spikes exponentially.
{{% /callout %}}
```

## Parameters

| Param | Required | Values | Default |
|---|---|---|---|
| `type` | no | `note` \| `tip` \| `warning` \| `tangent` \| `theorem` | `note` |
| `title` | no | header label (uppercased automatically) | the `type`, uppercased |
| `id` | no | anchor id | — |

## Variants

| Type | Header colour | Use for |
|---|---|---|
| `note` | Pop teal | Neutral clarifications. Definitions. "What we mean by X is…" |
| `tip` | Pop pink | Practical advice. Recommended approaches. "If you're implementing this…" |
| `warning` | Pop orange | Gotchas. Sharp edges. "This crashes silently when…" |
| `tangent` | Ink (black) on cream | Anecdotes. Historical asides. James Burke moments. The "by the way" detour. |
| `theorem` | Pop yellow | Formal definitions or named results. "Theorem 3.2 (Heavy Hitters):" |

## When to use

- A factoid that interrupts the narrative but is genuinely interesting — that's a `tangent`.
- A definition that *every* reader will need but expert readers can skip — that's a `note`.
- A piece of advice for an implementer following along — that's a `tip`.
- A warning about a non-obvious failure mode — that's a `warning`.
- A formal mathematical result that you'll refer back to — that's a `theorem`.

## When NOT to use

- For prose. If the content is two paragraphs of running narrative, it belongs in the body — not a callout.
- For single citations or one-line clarifications. Use a [margin note](marginnote.md) instead.
- For "did you know" trivia unconnected to the argument. Cut it.
- For the punchline. The punchline goes in a [pullquote](pullquote.md) or in the body.

## Limit per article

**Maximum two callouts.** Three callouts in 1500 words read as a listicle — they break narrative flow more than they help.

## Where it renders

- Shortcode: `themes/almanac/layouts/shortcodes/callout.html`
- CSS: search `CALLOUT` in `themes/almanac/assets/css/main.css`
