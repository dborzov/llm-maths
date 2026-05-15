---
title: "The State Dict, Demystified"
description: "The model is a Python dictionary of tensors. What's in it, what each entry is shaped like, and why these are the only nine names you need to memorize for the rest of the project."
topics: [transformer]
tags: [microgpt, state-dict, parameters]
theme: teal
math: true
draft: false
date: 2026-05-14T02:48:52-04:00
issue: 5
weight: 20
techKind: primer
techNode: state-dict
header: default.webp
---

## A Suspicious File On A Friend's Laptop

In February 2023, a week after the Llama 1 weights leaked onto the internet via that infamous 4chan torrent, a friend of mine — a backend engineer who had never trained a model in his life — sent me a giddy message. He had pulled the torrent, watched the progress bar tick to 100%, and was looking at a directory full of files named `consolidated.00.pth`, `consolidated.01.pth`, and so on. Total size: 13 GB for the 7B variant.

"It's just one file," he wrote. "I expected, I don't know, a binary executable? A program? But it's just one big number-blob."

He was right, in a way no university course had ever told him. He fired up Python:

```python
>>> import torch
>>> sd = torch.load('consolidated.00.pth', map_location='cpu')
>>> type(sd)
<class 'collections.OrderedDict'>
>>> list(sd.keys())[:5]
['tok_embeddings.weight',
 'layers.0.attention.wq.weight',
 'layers.0.attention.wk.weight',
 'layers.0.attention.wv.weight',
 'layers.0.attention.wo.weight']
```

Then he sent me the screenshot with three exclamation marks. "It's literally a **dict**. The whole model is a dict of tensors!?"

It is. That is the entire trick. A "trained LLM" is not a program, not a virtual machine, not a magic artifact. It is a Python dictionary whose keys are strings and whose values are rectangular arrays of floats. The cleverness — every last drop of it — lives in the **code that interprets that dictionary**. The dictionary itself is dead matter. It is, in the words of an old Bell Labs colleague of mine, *the world's most expensive lookup table*.

This is chapter two of an issue spent unpacking [microGPT, the 60-line transformer from chapter one](../01-cold-open/). Before we run a single token through that listing, we need to know exactly what is in the `state_dict` it keeps fishing tensors out of. By the end of this article you will be able to recite the keys in your sleep and tell me, for any one of them, the shape, the bytes, and the role.

## The Nine Names

Open microGPT and grep for `state_dict`. You will find exactly **nine distinct key patterns**:

```python
state_dict['wte']                       # token embedding
state_dict['wpe']                       # position embedding
state_dict[f'layer{li}.attn_wq']        # query projection
state_dict[f'layer{li}.attn_wk']        # key projection
state_dict[f'layer{li}.attn_wv']        # value projection
state_dict[f'layer{li}.attn_wo']        # attention output projection
state_dict[f'layer{li}.mlp_fc1']        # MLP fattening
state_dict[f'layer{li}.mlp_fc2']        # MLP skinnying
state_dict['lm_head']                   # vocabulary projection
```

That is it. Two top-level keys, seven per-layer keys repeated `n_layer` times. For our toy model with `n_layer = 2`, the full key set has length $2 + 7 \cdot 2 = 16$.

A Llama 3 8B `consolidated.00.pth` has more keys than that — closer to 290 — because real transformers also store **RMSNorm gain vectors** (one or two per layer) and split the MLP into three projections instead of two (the `SwiGLU` trick that we'll meet in [chapter 12](../12-activations/)). But every one of those extra keys is a refinement of, or addition to, the nine names above. The skeleton is identical.

> The state dict is a **flat namespace**. There is no hierarchy on disk, no nesting, no schema. It is `{str: ndarray}`, full stop. The string keys *imply* a structure ("this one belongs to layer 7's attention block") but the file format does not enforce one. The implication only becomes truth when the inference code agrees to read those keys in a particular order.

## Hyperparameters Are Not In The File

Here is the moment of confusion that catches every newcomer. **The integers `n_layer`, `n_embd`, `block_size`, `n_head`, `head_dim` are NOT in the state dict.**

Read that twice. The most-talked-about numbers in any model card — "32 layers, 4096 dim, 32 heads" — do not appear inside the dictionary of weights. They live next to it, in a tiny companion blob called `params.json` or `config.json` or, in microGPT's case, three lines at the top of `const.py`:

```python
n_layer    = 2
n_embd     = 16
block_size = 16
n_head     = 4
head_dim   = n_embd // n_head   # = 4
```

Why this separation? Because those integers are not learned. They were chosen by a human before training started, and the *shapes of the tensors* in `state_dict` already encode them implicitly. If you hand me `state_dict['wte']` and tell me its shape is `27 × 16`, I know `n_embd = 16` and `vocab_size = 27` without you saying a word. If you hand me `state_dict['layer0.attn_wq']` and it's `16 × 16`, I learn nothing new. If you hand me `state_dict['layer7.attn_wq']` — *and the key exists at all* — I learn that `n_layer ≥ 8`.

The hyperparameter blob is **metadata that tells you how to interpret the shapes**. It is the equivalent of the sample rate and bit depth printed on the side of an old audio CD: not part of the music, but without it the music is gibberish.

This is also why the same `.pth` file can be loaded by twelve different inference engines (vLLM, llama.cpp, HuggingFace `transformers`, MLX, exllama, …) and produce the *same logits to the last bit*. They all agree on the interpretation. The dictionary is the contract.

## The Table That Holds The Whole Thing

Here are the nine shapes, side by side, for our toy and for a real frontier model. We'll use **Llama 3 8B** as the reference — 32 layers, 4096 embedding dim, 32 query heads, 8 key/value heads (it uses Grouped-Query {{< wiki "attention" >}}Attention{{< /wiki >}}, which compresses K and V; we'll explain it in [chapter 9](../09-multi-head/) and [chapter 18](../18-gqa/)), 128000-token vocab.

| Key | Role | microGPT shape | Llama 3 8B shape |
|---|---|---|---|
| `wte` | token → embedding lookup | `vocab × n_embd` = `27 × 16` | `128000 × 4096` |
| `wpe` | position → embedding lookup | `block_size × n_embd` = `16 × 16` | *(absent — uses RoPE)* |
| `layer{li}.attn_wq` | query projection | `n_embd × n_embd` = `16 × 16` | `4096 × 4096` |
| `layer{li}.attn_wk` | key projection | `n_embd × n_embd` = `16 × 16` | `1024 × 4096` (GQA: 8 KV heads) |
| `layer{li}.attn_wv` | value projection | `n_embd × n_embd` = `16 × 16` | `1024 × 4096` (GQA: 8 KV heads) |
| `layer{li}.attn_wo` | attention output | `n_embd × n_embd` = `16 × 16` | `4096 × 4096` |
| `layer{li}.mlp_fc1` | MLP fattening | `4·n_embd × n_embd` = `64 × 16` | `14336 × 4096` |
| `layer{li}.mlp_fc2` | MLP skinnying | `n_embd × 4·n_embd` = `16 × 64` | `4096 × 14336` |
| `lm_head` | residual → vocab logits | `vocab × n_embd` = `27 × 16` | `128000 × 4096` |

Two columns of dictionary entries. Same nine names. **The only thing that changed between a toy you can write on a napkin and one of the most expensive trained artifacts in the world is the integers in the shapes.**

> *Caveat for the pedant in row 2:* Llama 3 has no `wpe` because position information enters via **Rotary Position Embeddings** (RoPE) applied to `q` and `k` at attention time, rather than via a lookup table added to the embedding. That swap is one of the "localized line replacements" we promised in the cold open — RoPE replaces the `wpe` line. We unpack it in [chapter 3](../03-embeddings/).

## Napkin Math: Bytes Per Bucket

Let us count *bytes*, not just parameters, because bytes are what fit (or don't fit) on your GPU.

**microGPT.** Two layers, each with four `16×16` attention projections (1024 params each) and one `16×64` + one `64×16` MLP pair (1024 params each). Plus the two embedding tables and the head. Total $\approx 7{,}264$ floats. At fp32 that's $\approx 29$ KB. The entire model fits in the L1 cache of a 2010-era CPU.

**Llama 3 8B.** Let's do it per-layer:

$$
\underbrace{4096^2}_{\text{wq}} + \underbrace{4096 \cdot 1024}_{\text{wk}} + \underbrace{4096 \cdot 1024}_{\text{wv}} + \underbrace{4096^2}_{\text{wo}} = 41{,}943{,}040
$$

That is attention. Then SwiGLU MLPs (three projections in real Llama, but pretend two for now):

$$
\underbrace{4096 \cdot 14336}_{\text{fc1}} + \underbrace{14336 \cdot 4096}_{\text{fc2}} \approx 117{,}440{,}512
$$

Per-layer total: $\approx 159$ M params. Times 32 layers: $\approx 5.1$ B. Add the embedding and head tables ($128000 \cdot 4096 \cdot 2 = 1.05$ B if they're untied), and you land at roughly **6–8 B parameters** depending on whether `wte` and `lm_head` share storage. At fp16 (2 bytes/param), that's **12–16 GB on disk**. This is why a single 24 GB RTX 4090 can host Llama 3 8B in fp16 with room left for {{< wiki "kv-cache" >}}KV cache{{< /wiki >}}, but not a 70B sibling.

The same nine names. The same six lines of arithmetic. Different integers.

```pyplot {id="param-share-llama3" caption="Where Llama 3 8B's ~8 B parameters live, by state-dict category. The MLP fattening/skinnying pair dominates; attention is the second largest chunk; embedding tables look big but are a single-digit slice."}
labels = ['wte (128K x 4096)', 'lm_head (128K x 4096)',
          'attn_wq (32x)', 'attn_wk (32x, GQA)', 'attn_wv (32x, GQA)', 'attn_wo (32x)',
          'mlp_fc1 (32x)',  'mlp_fc2 (32x)']
sizes  = [128000*4096, 128000*4096,
          32*4096*4096, 32*4096*1024, 32*4096*1024, 32*4096*4096,
          32*4096*14336, 32*4096*14336]
colors = ['#FFD700','#FFD700',
          '#FF007F','#FF007F','#FF007F','#FF007F',
          '#00A8A8','#00A8A8']

# Convert to gigabytes at fp16 (2 bytes/param)
sizes_gb = [s * 2 / 1e9 for s in sizes]

fig, ax = plt.subplots(figsize=(8, 4.5))
y = list(range(len(labels)))
ax.barh(y, sizes_gb, color=colors, edgecolor='#1A1A1A', linewidth=1.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
for i, s in enumerate(sizes_gb):
    ax.text(s + 0.08, i, f'{s:.2f} GB', va='center', fontsize=9)
ax.set_xlabel('bytes at fp16 (GB)')
ax.set_title('Llama 3 8B state_dict — bytes per category')
ax.set_xlim(0, max(sizes_gb) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

total = sum(sizes_gb)
print(f"Total state_dict size at fp16: {total:.2f} GB")
print(f"MLP share:        {(sizes_gb[6]+sizes_gb[7]) / total * 100:.1f} %")
print(f"Attention share:  {sum(sizes_gb[2:6]) / total * 100:.1f} %")
print(f"Embedding share:  {(sizes_gb[0]+sizes_gb[1]) / total * 100:.1f} %")
```

The picture tells the same story at every scale: **the MLP is the elephant, attention is the rhino, and the embedding tables are a pair of large but tidy filing cabinets.** In Llama 3 8B, the MLP eats $\sim$60% of the bytes, attention takes $\sim$25%, and the embeddings + head together make up the remaining $\sim$15%.

## What Happens When You Call `torch.load`

This is the bit my friend stumbled into and could not let go of. So let's say it plainly.

`torch.load('consolidated.00.pth')` does, mechanically, the following:

1. Open the file.
2. Detect that it's a **pickle** archive wrapping tensor data (older `.pth`) or, for newer `.safetensors` files, a length-prefixed JSON header followed by raw byte buffers.
3. Reconstruct a Python `OrderedDict`.
4. For each entry, mmap or copy the raw bytes into a `torch.Tensor` and stash it in the dict.
5. Return the dict.

There is no executable code in the file. There is no graph definition. There is no virtual machine state. There is no "model" in any animated sense. There is a **dictionary of arrays**. The "model" only exists once your inference script — whether it's microGPT's `gpt()` function or a 37,000-line vLLM serving stack — *reads keys out of that dictionary and threads them through arithmetic in a specific order*. The dictionary is the noun. The forward pass is the verb.

<details>
<summary>What about `.safetensors`? Same thing, no pickle.</summary>

The `.safetensors` format from HuggingFace is the modern preferred container. It removes the security footgun in pickle (arbitrary code execution at load time) by replacing the pickle stream with a fixed-format header:

```
[ 8 bytes: header length N ]
[ N bytes: JSON header — names, dtypes, shapes, byte offsets ]
[ rest of file: raw tensor bytes, contiguous, in the order declared ]
```

Loading a `.safetensors` is read-JSON-then-mmap. No Python execution. But the **resulting object is the same**: a string-keyed dictionary of tensors. The container changed; the contents didn't.

</details>

## A Worked Example: Pulling Out One Tensor

Suppose microGPT is trained and `state_dict` is in memory. We are processing the third token of a sequence. We want to compute the query projection of layer 1 for that token. Here is the literal lookup:

```python
li = 1
W_q = state_dict[f'layer{li}.attn_wq']      # shape (16, 16) — a list of 16 rows
                                            #   each row is a list of 16 floats
x = [...]                                   # the layer's input, length 16

q = linear(x, W_q)                          # length 16
```

The `linear` helper from chapter one is the simplest matrix-vector product imaginable:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

For each of the 16 rows `wo` in `W_q`, we dot it with the 16-element input `x`, producing one output number. Sixteen rows in, sixteen numbers out. That's 256 multiplies and 240 adds, for **one token, one layer, one projection**. The whole microGPT forward pass for one token runs roughly:

$$
\underbrace{4 \cdot 16 \cdot 16}_{\text{4 attn projs}} + \underbrace{2 \cdot 16 \cdot 64}_{\text{2 MLP projs}} = 3{,}072 \text{ multiplies per layer}
$$

Times two layers, plus the embedding lookups (effectively free) and the final `lm_head` projection ($16 \cdot 27 = 432$): **about 6,576 multiplies per generated token**. A 2GHz laptop CPU running interpreted Python does that in roughly 2 ms once you remove the overhead. The model is, computationally, *trivial*.

For Llama 3 8B, the same arithmetic gives you about **8 billion multiplies per token** ($\sim 2N$ where $N$ is the parameter count, a famous Fermi rule that drops out of this exercise). That is why a GPU is useful and a CPU is sad.

## What This Buys You

Once you internalize that a model is a dictionary, several confusing things become obvious:

- **LoRA adapters** are just *additional* state-dict entries — small low-rank matrices added to the original projections at load time. The base dict + the adapter dict get merged before inference. No magic.
- **Model merging** ("Frankenmerge", SLERP, task arithmetic) is literally `merged[k] = alpha * sd_a[k] + (1-alpha) * sd_b[k]` for every key. You can write it in five lines.
- **Quantization** (issue 3's whole obsession) replaces a `float16` tensor with a `int4` tensor plus a small scale vector — that is, it replaces one dict entry with two. The forward-pass code learns one new line: dequantize before `linear`.
- **MoE** (Mixture-of-Experts) adds, per layer, a *family* of `mlp_fc1` and `mlp_fc2` tensors plus a tiny router. Same idea, just K copies of the MLP keys and a `router.weight`.
- **Tied embeddings** is the trick where `lm_head` and `wte` share storage. The state dict simply omits `lm_head` and the loader aliases it to `wte` at runtime. One key disappears; the file shrinks by a vocab-table's worth of bytes.

Every "innovation" in the modern LLM stack lives somewhere in the diff between two state dicts. If you can read a dict, you can read a research paper.

## What To Remember

1. **A trained LLM is a `dict[str, Tensor]`.** Nine key patterns in microGPT. A few more in real models. That's the whole artifact on disk.
2. **Hyperparameters live outside the dict.** `n_layer`, `n_embd`, `n_head`, `block_size` are in a separate metadata blob and *implied* by the tensor shapes.
3. **Shapes are the contract.** Given the nine shapes and the integers, the forward pass is fully determined. Different inference engines reading the same dict produce bit-identical logits.
4. **MLP is the elephant.** $\sim 60$% of bytes in modern frontier models live in `mlp_fc1` and `mlp_fc2`. Attention is second, embeddings are third.
5. **From here on, "the model" means the dict.** When [chapter 4 talks about `linear`](../04-linear/) and [chapter 6 talks about the Q/K/V projections](../06-qkv-projections/), they mean the rows of these specific dict entries. There is nothing else.

---

**Continue to** → [The Embedding Tables](../03-embeddings/) — now that you know `wte` and `wpe` are dictionary entries shaped `vocab × n_embd` and `block_size × n_embd`, the next question is what the *rows* of those tables actually mean, and why two tables get added together at the top of every forward pass.

