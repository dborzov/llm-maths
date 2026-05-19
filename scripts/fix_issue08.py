"""Fix issue 08 articles: remove scaffold notes, add descriptions, fix dates."""
import re
import sys
from pathlib import Path

BASE = Path("/attic/olde-llm-maths/content/comicbook/08-vLLM")

# Descriptions derived from scaffold note content and titles
DESCRIPTIONS = {
    "01-cold-open.md": None,  # already has a good description
    "02-gpu-anatomy.md": (
        "A GPU is a throughput machine: 132 streaming multiprocessors, "
        "thousands of simultaneous threads, and a memory hierarchy that spans four "
        "orders of magnitude. Understanding the hardware is the foundation for every "
        "inference optimization that follows."
    ),
    "03-memory-hierarchy.md": (
        "Registers, shared memory, L2, and HBM: four orders of magnitude of bandwidth "
        "separate the fastest storage from the slowest. Every inference optimization is "
        "a trade along this pyramid — capacity for speed, or speed for capacity."
    ),
    "04-roofline.md": (
        "The roofline model tells you in one number — arithmetic intensity — whether a "
        "kernel is compute-bound or memory-bound. Decode attention sits at roughly "
        "1 FLOP/byte, catastrophically to the left of the ridge point. That single fact "
        "predicts the shape of every optimization in modern LLM serving."
    ),
    "05-cuda-graphs.md": (
        "A decode-step forward pass launches hundreds of tiny CUDA kernels. The Python "
        "scheduling overhead can eat the entire latency budget before the GPU starts. "
        "CUDA graphs capture the launch sequence and replay it as a single GPU command, "
        "dropping CPU-side overhead from milliseconds to microseconds."
    ),
    "06-flash-attention.md": (
        "FlashAttention is the most important inference kernel of the decade: it "
        "computes attention without ever materializing the score matrix in HBM, using "
        "tiled SRAM computation and online softmax. It is also the direct skeleton that "
        "PagedAttention extends with paged indirection."
    ),
    "07-prefill-vs-decode.md": (
        "Prefill is compute-bound; decode is bandwidth-bound. They have opposite "
        "hardware personalities, opposite bottlenecks, and opposite optimal batch sizes — "
        "and every major inference engine decision flows from that asymmetry."
    ),
    "08-continuous-batching.md": (
        "Static batching wastes most of the GPU whenever sequences finish at different "
        "lengths — the GPU idles waiting for the longest sequence. Orca's "
        "iteration-level scheduling (OSDI 2022) fixed this: swap finished requests out "
        "and new ones in at every decode step, not at batch boundaries."
    ),
    "09-kv-fragmentation.md": (
        "The KV cache grows one token at a time, to unpredictable lengths, and must be "
        "contiguous in naive implementations. The result is a fragmented heap that "
        "wastes 60–80 percent of HBM in the worst case. This is the problem "
        "PagedAttention was built to solve."
    ),
    "10-paged-attention.md": (
        "PagedAttention is virtual memory for KV caches: a per-request block table maps "
        "logical token positions to non-contiguous physical KV blocks, eliminating "
        "fragmentation. The same abstraction enables prefix sharing, copy-on-write "
        "branching for beam search, and tiered KV offload — all for free."
    ),
    "11-block-manager.md": (
        "The block manager is the data structure that implements paging for KV caches: "
        "a free-block pool, per-request block tables with reference counting, and an "
        "O(1) LRU eviction policy. Understanding it is the key to understanding how "
        "prefix caching, copy-on-write, and KV offload work at the implementation level."
    ),
    "12-prefix-caching.md": (
        "Every conversation in a deployment starts with the same system prompt. Hashing "
        "prefix blocks and re-using their physical memory turns a full prefill into a "
        "cache hit — the single highest-leverage optimization in modern LLM serving, "
        "and an almost embarrassingly simple one in hindsight."
    ),
    "13-chunked-prefill.md": (
        "A 100K-token prefill can monopolize the GPU for seconds, wrecking "
        "inter-token latency for every other user in the batch. Chunked prefill "
        "slices long prompts into token-budget-sized pieces that interleave with decode "
        "steps, giving every user bounded and predictable time-to-first-token."
    ),
    "14-scheduler.md": (
        "vLLM V1's scheduler is deliberately simple: allocate a fixed token budget per "
        "step, let each request spend it on prefill or decode tokens, and let the "
        "prefill-versus-decode distinction dissolve. This chapter traces the years of "
        "layered complexity that one clean abstraction quietly deletes."
    ),
    "15-speculative-decoding.md": (
        "Decode is so bandwidth-bound that verifying 32 candidate tokens costs the same "
        "memory pass as verifying 1. Run a small draft model to speculate several tokens "
        "ahead, then let the target model verify all of them in a single forward pass — "
        "a 2–3× throughput gain if the draft is even modestly accurate."
    ),
    "16-tp-pp.md": (
        "Tensor parallelism splits weight matrices column-by-column across GPUs, "
        "with one all-reduce per transformer block. Pipeline parallelism stacks "
        "layers across machines. Both fit large models into finite HBM — the "
        "trade-offs are latency (all-reduce cost) versus throughput (micro-batch fill)."
    ),
    "17-disagg-pd.md": (
        "Prefill needs high compute throughput; decode needs high memory bandwidth. "
        "Running them on the same GPU is the worst of both worlds. Disaggregated "
        "prefill/decode splits the job across two purpose-built machine types connected "
        "by an RDMA KV-cache fabric — DistServe, Mooncake, and NIXL are the field's answers."
    ),
    "18-full-anatomy.md": (
        "The cold-open trace, re-annotated end to end: the same packet, the same "
        "H200 box, the same user — but every layer now named, every component explained, "
        "every latency budget itemized. Everything the issue introduced, assembled "
        "into one complete, labelled picture."
    ),
}

# Dates: spread from 09:00 to 17:30 in 30-min steps
DATES = {
    "01-cold-open.md":         "2026-05-16T09:00:00-04:00",
    "02-gpu-anatomy.md":       "2026-05-16T09:30:00-04:00",
    "03-memory-hierarchy.md":  "2026-05-16T10:00:00-04:00",
    "04-roofline.md":          "2026-05-16T10:30:00-04:00",
    "05-cuda-graphs.md":       "2026-05-16T11:00:00-04:00",
    "06-flash-attention.md":   "2026-05-16T11:30:00-04:00",
    "07-prefill-vs-decode.md": "2026-05-16T12:00:00-04:00",
    "08-continuous-batching.md":"2026-05-16T12:30:00-04:00",
    "09-kv-fragmentation.md":  "2026-05-16T13:00:00-04:00",
    "10-paged-attention.md":   "2026-05-16T13:30:00-04:00",
    "11-block-manager.md":     "2026-05-16T14:00:00-04:00",
    "12-prefix-caching.md":    "2026-05-16T14:30:00-04:00",
    "13-chunked-prefill.md":   "2026-05-16T15:00:00-04:00",
    "14-scheduler.md":         "2026-05-16T15:30:00-04:00",
    "15-speculative-decoding.md":"2026-05-16T16:00:00-04:00",
    "16-tp-pp.md":             "2026-05-16T16:30:00-04:00",
    "17-disagg-pd.md":         "2026-05-16T17:00:00-04:00",
    "18-full-anatomy.md":      "2026-05-16T17:30:00-04:00",
}

# Regex to match scaffold note paragraph (the > **Scaffold note.** ... line and a trailing blank line)
SCAFFOLD_RE = re.compile(r'^> \*\*Scaffold note\.\*\*.*\n\n?', re.MULTILINE)

changed = 0
for fname, desc in DESCRIPTIONS.items():
    path = BASE / fname
    text = path.read_text()

    # Remove scaffold note (the whole blockquote line + following blank line)
    new_text = SCAFFOLD_RE.sub('\n', text, count=1)

    # Fix description if needed
    if desc is not None:
        new_text = re.sub(
            r'^(description:\s*)""',
            f'\\1"{desc}"',
            new_text,
            count=1,
            flags=re.MULTILINE,
        )

    # Fix date
    new_date = DATES[fname]
    new_text = re.sub(
        r'^(date:\s*).*$',
        f'\\g<1>{new_date}',
        new_text,
        count=1,
        flags=re.MULTILINE,
    )

    if new_text != text:
        path.write_text(new_text)
        print(f"  fixed: {fname}")
        changed += 1
    else:
        print(f"  unchanged: {fname}")

print(f"\nDone. {changed} files updated.")
