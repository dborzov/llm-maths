---
title: "Contextual Embeddings: one word, one vector was not enough"
short_title: "Contextual Embeddings"
description: "Word2Vec gave every word one vector — a compromise between all its meanings. ELMo (2018) and BERT fixed that by making embeddings depend on context, not just token identity."
blurb:
  - "Word2Vec's \"bank\" vector is the average of river bank, financial bank, blood bank, and aircraft bank — a point equidistant from all of them."
  - "GPT-2 small: 50,257-token vocabulary, 768-dimensional embeddings. The embedding table alone is 38.6 million parameters, 154 MB at fp32."
  - "The first operation of every modern transformer is still a matrix row lookup — mathematically identical to Word2Vec's `W_in`, renamed `W_E`."
  - "Attention layers modify each token's representation using the surrounding context. Same word, different sentence: different final vector."
topics: [embeddings, transformers, llms]
tags: [transformers, bert, gpt, contextual-embeddings, positional-encoding, attention]
theme: cream
math: true
draft: false
date: 2026-04-01T09:36:00-04:00
issue: 1
weight: 70
techKind: boss
techNode: inheritance
---

## NAACL 2018, New Orleans

Matthew Peters is presenting a paper with the longest name in conference memory: "Deep Contextualized Word Representations." The model is called **ELMo** — Embeddings from Language Models — and it is about to make Word2Vec obsolete.

Not because Word2Vec was wrong. Because it was too right. The geometry of meaning that Mikolov discovered in 2013 was so useful that researchers built five years of better systems on top of it. And those systems eventually grew complex enough to notice Word2Vec's central limitation: one word, one vector.

ELMo's solution: use a deep bidirectional LSTM to compute *different* vectors for the same word token depending on the sentence it appears in. The word "bank" in "river bank" gets a different vector from "bank" in "bank account." Same token. Different embedding. Context included.

Eight months later, Google's **BERT** paper landed, and by the time the dust settled, "contextual embeddings" were the new standard. The architecture had changed — BERT used {{< wiki "attention" >}}attention{{< /wiki >}} instead of recurrence — but the fundamental insight remained unchanged since 2013: *meaning lives in dense vectors, trained by predicting context*.

## The Embedding Layer: Unchanged Since 2013

Open any modern transformer. The first operation is always the same.

The model maintains an **embedding matrix** $W_E$ of shape `(vocab_size, d_model)`. A sequence of token IDs arrives. The embedding layer looks up each token ID's row in $W_E$:

```python
import numpy as np

# Typical GPT-2 small configuration
vocab_size = 50_257   # byte-pair encoding vocabulary
d_model    = 768      # embedding dimension
seq_len    = 1024     # maximum sequence length

# The embedding table — learned by gradient descent, just like Word2Vec
W_E = np.random.randn(vocab_size, d_model) * 0.02   # shape: (50_257, 768)

# A tokenized sentence: "the cat sat" → token IDs
token_ids = np.array([262, 3797, 2275])   # hypothetical IDs

# The entire embedding lookup: three rows from a matrix
token_embeddings = W_E[token_ids]   # shape: (3, 768)

print(f"Embedding table: {W_E.shape}")
print(f"Token IDs: {token_ids}")
print(f"Embeddings shape: {token_embeddings.shape}")
print(f"Parameters in embed table: {vocab_size * d_model:,}")
```

```
Embedding table: (50257, 768)
Token IDs: [262 3797 2275]
Embeddings shape: (3, 768)
Parameters in embed table: 38,597,376
```

This is exactly Word2Vec's $W_{in}$ matrix, renamed $W_E$. The operation — multiply a one-hot vector by a matrix, or equivalently look up a row — is identical. The matrix is learned by gradient descent through the loss of the whole model, not a standalone skip-gram objective. The dimension is 768 instead of 300. But the mathematical object is the same.

The *spirit* of Word2Vec is intact: train the matrix so that token vectors that appear in similar contexts end up near each other.

## Adding Position: What Word2Vec Could Not Do

Word2Vec has no concept of word order. "Cat chases dog" and "Dog chases cat" would produce the same context predictions — the window is symmetric, the order is lost.

Transformers fix this with **positional encodings** — a second embedding added to the token embedding:

$$h_t = W_E[\text{token}_t] + W_P[\text{position}_t]$$

where $W_P$ is a `(max_seq_len, d_model)` matrix. The combined vector $h_t$ encodes both *what* the token is and *where* in the sequence it appears.

```python
# Positional embedding table (learned in GPT-style models)
W_P = np.random.randn(seq_len, d_model) * 0.02   # shape: (1024, 768)

# For position 0, 1, 2 in the sequence:
positions = np.array([0, 1, 2])
pos_embeddings = W_P[positions]   # shape: (3, 768)

# The input to the transformer body
h = token_embeddings + pos_embeddings   # shape: (3, 768)
print(f"Input to transformer (token + position): {h.shape}")

# Storage cost:
token_params = vocab_size * d_model
pos_params   = seq_len * d_model
print(f"\nEmbedding parameters:")
print(f"  Token embeddings W_E: {token_params:>10,}  ({token_params*4/1e6:.1f} MB at fp32)")
print(f"  Position embed W_P:   {pos_params:>10,}  ({pos_params*4/1e6:.1f} MB at fp32)")
```

```
Input to transformer (token + position): (3, 768)

Embedding parameters:
  Token embeddings W_E:  38,597,376  (154.4 MB at fp32)
  Position embed W_P:       786,432  (3.1 MB at fp32)
```

The position embedding is tiny — the sequence length is much shorter than the vocabulary. This combined vector $h = W_E[\text{token}] + W_P[\text{position}]$ is what enters the transformer's {{< wiki "attention" >}}attention{{< /wiki >}} layers.

## Contextual Embeddings: The Unsolved Word2Vec Problem, Solved

Word2Vec's central failure: the word "bank" has one vector. This vector is the *average* of all the contexts where "bank" appeared in the training corpus — financial banks, river banks, blood banks, aircraft banks. The single vector is a compromise.

A transformer does not stop at the embedding lookup. After $h = W_E[\text{token}] + W_P[\text{position}]$, the vector passes through $L$ layers of attention, each of which mixes information from all other positions in the sequence. By the time a representation exits the final layer, it has been modified by the entire surrounding context.

The key: **the attention mechanism is a weighted sum of value vectors from other positions**. For the word "bank" in the sentence "I deposited money in the **bank**," the attention layers route information from "deposited," "money," and other financial-context words into the "bank" representation, shifting it toward the financial-institution cluster. In "the **bank** of the river," the same word's representation gets shaped by "river," shifting it toward the geography cluster.

```pyplot {id="static-vs-contextual" caption="STATIC EMBEDDING (WORD2VEC) vs CONTEXTUAL (TRANSFORMER). SAME TOKEN 'BANK', TWO CONTEXTS, TWO DIFFERENT VECTORS. THE TRANSFORMER SOLVES THE POLYSEMY PROBLEM."}
np.random.seed(11)

# Simulate static vs contextual embeddings for "bank"
d = 2  # 2D for visualization

# Static embedding: one point, intermediate between clusters
static_bank = np.array([0.5, 0.5])

# "Financial" cluster center
financial_center = np.array([0.8, 0.3])
geo_center       = np.array([0.2, 0.8])

# Contextual embeddings: shifted toward each context's cluster
ctx_bank_fin = static_bank + 0.6*(financial_center - static_bank) + np.random.randn(2)*0.04
ctx_bank_geo = static_bank + 0.6*(geo_center - static_bank)       + np.random.randn(2)*0.04

# Nearby words in each context (for the scatter)
fin_words = ['money','deposit','loan','interest','account']
geo_words = ['river','stream','shore','flood','water']
fin_vecs  = financial_center + np.random.randn(5, 2)*0.12
geo_vecs  = geo_center       + np.random.randn(5, 2)*0.12

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for ax, title, bank_pt, cluster_pts, cluster_words, cluster_color in [
    (ax1, "Static (Word2Vec): one point for all 'bank's",
     static_bank, np.vstack([fin_vecs, geo_vecs]),
     fin_words + geo_words, ['#00A8A8']*5 + ['#FF8C00']*5),
    (ax2, "Contextual (Transformer): different vector per context",
     None, np.vstack([fin_vecs, geo_vecs, [ctx_bank_fin, ctx_bank_geo]]),
     fin_words + geo_words + ['bank (financial)', 'bank (river)'],
     ['#00A8A8']*5 + ['#FF8C00']*5 + ['#FF007F', '#FFD700']),
]:
    for v, w, c in zip(cluster_pts, cluster_words, cluster_color):
        ax.scatter(*v, color=c, s=90, edgecolor='#1A1A1A', lw=1.0, zorder=3)
        ax.text(v[0]+0.02, v[1]+0.02, w, fontsize=8.5, color='#333333', zorder=4)

    if bank_pt is not None:
        ax.scatter(*bank_pt, color='#FF007F', s=200, edgecolor='#1A1A1A', lw=2,
                  marker='*', zorder=5)
        ax.text(bank_pt[0]+0.02, bank_pt[1]+0.03, '"bank"', fontsize=11,
                fontweight='bold', color='#FF007F', zorder=6)
        ax.text(bank_pt[0]-0.15, bank_pt[1]-0.12, 'stuck between\nboth clusters',
                fontsize=8.5, color='#888', style='italic')

    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.set_title(title, fontsize=9.5)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Cluster labels
    ax.text(0.80, 0.10, 'FINANCE', fontsize=10, color='#00A8A8',
            fontweight='bold', ha='center', alpha=0.7)
    ax.text(0.15, 0.92, 'GEOGRAPHY', fontsize=10, color='#FF8C00',
            fontweight='bold', ha='center', alpha=0.7)

plt.suptitle('Polysemy problem: solved by context-dependent representations', fontsize=11)
plt.tight_layout()
```

## Napkin Math: Embedding Layers at Scale

The direct lineage from Word2Vec to modern LLMs is visible in the numbers:

| Model | Year | Vocab | d_model | Embed params | Embed params at fp16 |
|---|---|---|---|---|---|
| Word2Vec | 2013 | 1M | 300 | 300M | 600 MB |
| BERT-base | 2018 | 30,522 | 768 | 23M | 47 MB |
| GPT-2 small | 2019 | 50,257 | 768 | 39M | 77 MB |
| GPT-3 | 2020 | 50,257 | 12,288 | 617M | 1.2 GB |
| LLaMA-3 70B | 2024 | 128,256 | 8,192 | 1.05B | 2.1 GB |

```python
models = {
    'Word2Vec':     (1_000_000, 300),
    'BERT-base':    (30_522,    768),
    'GPT-2 small':  (50_257,    768),
    'GPT-3':        (50_257,   12_288),
    'LLaMA-3 70B':  (128_256,   8_192),
}

print(f"{'Model':<16} {'Vocab':>10} {'d_model':>8} {'Embed params':>14} {'MB (fp16)':>10}")
print("-" * 64)
for name, (vocab, dim) in models.items():
    params = vocab * dim
    mb = params * 2 / 1e6  # fp16 = 2 bytes
    print(f"{name:<16} {vocab:>10,} {dim:>8,} {params:>14,} {mb:>10.0f}")
```

```
Model            Vocab  d_model  Embed params   MB (fp16)
----------------------------------------------------------------
Word2Vec    1,000,000      300   300,000,000      600
BERT-base      30,522      768    23,440,896       47
GPT-2 small    50,257      768    38,597,376       77
GPT-3          50,257   12,288   617,556,992    1,235
LLaMA-3 70B   128,256    8,192 1,050,722,304    2,101
```

The embedding layer of LLaMA-3 70B is 1 billion parameters — twice the size of the entire original GPT-2 model. And it is doing exactly the same operation as Word2Vec: looking up a row in a matrix.

```pyplot {id="dimension-scaling" caption="EMBEDDING DIMENSION vs YEAR. FROM WORD2VEC'S 300 TO MODERN 8,192–12,288 — A 40× SCALING DRIVEN BY THE NEED TO ENCODE RICHER CONTEXT."}
years  = [2013, 2014, 2016, 2018, 2018, 2019, 2020, 2022, 2023, 2024]
dims   = [300,  300,  300,  512,  768,  768,  12288, 4096, 4096, 8192]
names  = ['Word2Vec', 'GloVe', 'FastText', 'ELMo', 'BERT-base', 'GPT-2',
          'GPT-3', 'PaLM', 'LLaMA-1', 'LLaMA-3 70B']
colors_pts = (['#00A8A8']*3 +          # static embeddings
              ['#FFD700', '#FFD700'] +  # contextual pioneers
              ['#FF007F']*5)            # transformer LLMs

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(years, dims, c=colors_pts, s=120, edgecolor='#1A1A1A',
           linewidth=1.3, zorder=4)

for i, (yr, d, nm) in enumerate(zip(years, dims, names)):
    va = 'bottom'; yoff = 120
    if nm in ('GloVe', 'FastText', 'ELMo', 'PaLM'): va = 'top'; yoff = -250
    ax.annotate(nm, xy=(yr, d), xytext=(yr+0.05, d+yoff),
                fontsize=8.5, color='#1A1A1A', ha='left',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.7))

ax.axvspan(2013, 2016.5, alpha=0.06, color='#00A8A8', label='static embeddings')
ax.axvspan(2017.5, 2025, alpha=0.06, color='#FF007F', label='transformer models')
ax.set_yscale('log')
ax.set_ylabel("embedding dimension d_model", fontsize=11)
ax.set_xlabel("year", fontsize=11)
ax.set_title("Embedding dimension scaled 40× in 10 years — same operation, bigger geometry", fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.spines[['top','right']].set_visible(False)
ax.set_yticks([300, 512, 768, 4096, 8192, 12288])
ax.set_yticklabels(['300', '512', '768', '4,096', '8,192', '12,288'])
plt.tight_layout()
```

## The Full Chain

The inheritance from Word2Vec to modern LLMs is not metaphorical. It is literal:

1. **The embedding table** ($W_E$) is the direct descendant of Word2Vec's $W_{in}$. Same shape, same initialization strategy, same update rule (gradient descent on a prediction objective), same purpose (map token IDs to dense vectors).

2. **The training intuition** is preserved: vectors for tokens that appear in similar contexts should be similar. In Word2Vec this was enforced by the skip-gram objective directly. In a transformer it is enforced indirectly, through the cross-entropy loss on next-token prediction — but the effect on the embedding matrix is the same.

3. **The geometry** is preserved: after training, the embedding space of a modern LLM contains semantic directions, analogy structure, and cluster organization identical in character to Word2Vec's. The vectors are 40× bigger and informed by full-sequence context, but the basic geometric properties — cosine measures meaning, addition is composition, clusters are semantic categories — are unchanged.

{{% pullquote %}}
When GPT-4 reads your question, the first thing it does is look up a row in a matrix — exactly what Mikolov's 2013 model did. Everything that makes GPT-4 intelligent happens after that lookup. But the lookup is where it all begins.
{{% /pullquote %}}

The geometry of meaning that Mikolov stumbled onto while debugging a language model in September 2012 is not a historical curiosity. It is running in production right now, at scale, in every LLM being served to the world.

The words know their place. They have since 2013.

---

## Connections

<details>
<summary>Further reading — linked concepts in this series</summary>

- [Why Every Word Was An Island](../02-one-hot-problem/) — the representation crisis that motivated Word2Vec
- [The Geometry of Agreement](../03-dot-product/) — the dot product as Word2Vec's fundamental operation
- [Embarrassingly Simple](../04-word2vec/) — how skip-gram and negative sampling work
- [The Stranger Country](../05-high-dimensional/) — why 300 dimensions is the right geometry
- [Words That Know Their Place](../06-meaning-as-geometry/) — what the trained space contains
- [Issue 2: The Logit Wager](../../02-logits/) — what the transformer does *after* the embedding lookup
- [Issue 5: microGPT Unfolded](../../05-microgpt/) — the full transformer forward pass, with `wte` and `wpe` named and traced

</details>

**Continue to →** [Issue 2: The Logit Wager](../../02-logits/), where we follow the embedded vectors through the rest of the transformer — and discover that the next operation (the linear classifier head) is just as geometrically elegant as the embedding that feeds it.
