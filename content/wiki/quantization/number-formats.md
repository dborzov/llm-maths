---
title: "Number Formats"
slug: "number-formats"
description: "Floating-point and integer number formats used in LLM training and inference: FP32, FP16, BF16, FP8, FP4, INT8, INT4, NF4."
category: "quantization"
also_known_as: ["FP32", "FP16", "BF16", "FP8", "FP8 E4M3", "FP8 E5M2", "FP4", "FP4 E2M1", "INT8", "INT4", "NF4", "IEEE 754", "bfloat16", "half precision", "float16", "float32", "normal float 4", "mixed precision"]
source_of_truth: "/issues/03-sixteen-numbers/02-numbers-in-boxes/"
source_of_truth_title: "Numbers In Boxes (Issue 03)"
related: ["transformer-weights"]
draft: false
---

Floats are a **logarithmic** grid — spacing doubles each power-of-two octave.
FP16 and BF16 are not smaller FP32 — they are different geometric objects.

| Format | Exponent | Mantissa | Total | Key trait |
|---|---|---|---|---|
| FP32 | 8 | 23 | 32 | Standard training reference |
| FP16 | 5 | 10 | 16 | More precision, limited range |
| BF16 | 8 | 7 | 16 | FP32 range, less precision — wins for training |
| FP8 E4M3 | 4 | 3 | 8 | More precision, smaller range — weights/activations |
| FP8 E5M2 | 5 | 2 | 8 | More range — gradients |
| FP4 E2M1 | 2 | 1 | 4 | 16 values total — needs microscaling |
| INT8 | — | — | 8 | 256 uniform levels |
| INT4 | — | — | 4 | 16 uniform levels |
| NF4 | — | — | 4 | 16 values at normal-distribution quantiles (QLoRA) |

INT4 and FP4 both have 16 levels; INT4 spaces them uniformly, FP4 logarithmically.
Optimal level placement for a given data distribution → [The Lloyd-Max Bargain](/issues/03-sixteen-numbers/03-lloyd-max/).
