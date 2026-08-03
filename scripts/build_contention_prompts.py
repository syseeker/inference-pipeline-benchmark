#!/usr/bin/env python3
"""Emit the contention study's prompt files from experiment_config.json.

The customer's prompts live in `workspace/contention/experiment_config.json`
under the top-level `prompts` key. The GPU yaml's `workloads:` block points at
`workspace/contention/prompts/<name>.jsonl` — this script is what puts those
files on disk, in the shape aiperf's `single_turn` custom dataset loader wants:

    {"text": "What is CUDA?"}

One JSON object per line, TEXT ONLY. The media file (video clip / document
image) is NOT baked in here: `vlm_video_short` and `vlm_video_long` share one
prompts file and differ only by their `data:` clip, so the text+media pairing
has to happen per run. benchmarks/coloc.py does that combining into the run's
artifact dir (materialise_workload_input).

Idempotent: re-running rewrites the same bytes, so the generated .jsonl can be
checked in and a regeneration shows up as an empty diff.

Usage:
    python3 scripts/build_contention_prompts.py            # write + report
    python3 scripts/build_contention_prompts.py --check    # verify only, no write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "workspace" / "contention" / "experiment_config.json"
OUT_DIR = REPO_ROOT / "workspace" / "contention" / "prompts"


# ─────────────────────────── prompt overrides ──────────────────────────────
#
# `llm_long` is REPLACED here rather than edited in experiment_config.json.
# That file is the customer's brief as delivered and stays a pristine record —
# and being JSON, it cannot carry the explanation of why we departed from it.
#
# WHY: the yaml's workloads block described llm_long as "~1000 tok", and the
# whole point of the workload is to be the long-prefill arm of the
# prompt-length dimension. The customer's actual entries measure 142 and 109
# tokens. That is not a long prompt: `secondary-input-size-llm` would sweep a
# ~10x span instead of the intended ~20x, and `cross-vlm-prefill-vs-llm` would
# pit a ~140-token LLM prefill against a VLM's several-thousand-token
# 40-frame burst — making the LLM side effectively short-prompt in the one
# experiment built to contrast prefill lengths.
#
# The customer's originals, preserved verbatim (the two entries this replaces):
#
#   1. "Explain the complete memory hierarchy of a modern GPU, covering
#       registers, shared memory, L1 cache, L2 cache, and global memory.
#       Describe how data flows between these levels during a typical matrix
#       multiplication kernel. Include the role of memory coalescing and bank
#       conflicts. Discuss how the programmer can optimize memory access
#       patterns to maximize throughput. Cover the differences between compute
#       capability generations in terms of cache sizes and bandwidth. Finally,
#       explain how unified memory and managed memory simplify programming but
#       may impact performance."
#
#   2. "Write a detailed technical specification for a distributed inference
#       system that serves multiple large language models across a GPU cluster.
#       Cover model placement strategies, request routing, load balancing,
#       fault tolerance, autoscaling policies, and latency budgets. Include
#       specific numerical targets for tail latency percentiles and throughput
#       requirements. Describe the monitoring and alerting infrastructure
#       needed to maintain SLOs."
#
# The replacements below keep both topics and both intents — they are the same
# two questions asked at the depth the "~1000 tok" label implied. Prefill
# length is the measured quantity, so what matters is that the text is real,
# coherent and long; padding would have produced a different prefill shape
# than a genuine request of this size.
#
# To go back to the customer's originals, delete the llm_long entry below and
# re-run this script.

PROMPT_OVERRIDES: dict[str, list[str]] = {
    "llm_long": [
        (
            "Explain the complete memory hierarchy of a modern NVIDIA GPU in depth, "
            "and then use it to reason about the performance of two concrete kernels.\n\n"
            "Start with the register file. Describe its size per streaming multiprocessor, "
            "how registers are allocated per thread, what register pressure means in "
            "practice, and how the compiler's decision to spill to local memory changes a "
            "kernel's performance characteristics. Explain occupancy carefully: how "
            "per-block register and shared-memory usage bound the number of resident warps, "
            "why higher occupancy is not always better, and how latency hiding actually "
            "works when a warp stalls on a memory operation.\n\n"
            "Move to shared memory. Cover its banked structure, what a bank conflict is, "
            "how an n-way conflict serialises accesses, and the standard padding trick used "
            "to avoid conflicts in tiled matrix multiplication. Explain the configurable "
            "split between shared memory and L1 on the architectures that support it, and "
            "when you would bias the split one way or the other.\n\n"
            "Then the cache hierarchy: L1 per SM and L2 shared across the device. Give "
            "typical sizes and latencies for Ampere, Hopper and Blackwell, say what changed "
            "between those generations, and describe the L2 persisting-access window and "
            "the access patterns it helps.\n\n"
            "Cover global memory and the HBM or GDDR device memory behind it. Give bandwidth "
            "figures per generation, explain what memory coalescing means at warp level, "
            "what happens when a warp's 32 lanes touch scattered addresses, and how the "
            "memory controller's transaction granularity determines the real cost of a "
            "strided access pattern.\n\n"
            "Now apply all of that to a concrete case. Walk through a tiled FP16 matrix "
            "multiplication of two 4096x4096 matrices. Describe how data flows from global "
            "memory into shared memory into registers, where the tensor cores enter, what "
            "tile sizes you would choose and why, and how you would compute the kernel's "
            "arithmetic intensity. Use the roofline model to decide whether it is "
            "compute-bound or memory-bound on a card with 1.6 TB/s of bandwidth, and show "
            "the arithmetic rather than asserting the conclusion.\n\n"
            "Then contrast it with a memory-bound case: autoregressive decode in a large "
            "language model at batch size one, where every generated token requires "
            "streaming the entire weight matrix out of device memory. Explain why arithmetic "
            "intensity is so low in that regime, why the SMs sit largely idle, and what that "
            "implies for how much tensor cores, weight quantization and batching can each "
            "help. Be specific about which of those three attacks the actual bottleneck.\n\n"
            "Finally, explain unified memory and managed memory: what page migration costs, "
            "when prefetching helps, how oversubscription behaves under pressure, and why a "
            "workload that fits comfortably in device memory should usually avoid both.\n\n"
            "Cover the asynchronous data-movement path as well. Explain what the async copy "
            "instruction changed relative to the older load-to-register-then-store-to-shared "
            "idiom, why it matters that the copy bypasses the register file, and how software "
            "pipelining with multiple buffer stages hides the latency of the next tile's "
            "fetch behind the current tile's compute. Describe the tensor memory accelerator "
            "on Hopper and later: what it does that async copy did not, how descriptors "
            "work, and why bulk asynchronous copies matter more as tile sizes grow.\n\n"
            "Explain thread block clusters and distributed shared memory. Describe what "
            "guarantee the cluster provides about co-scheduling, how one block reads another "
            "block's shared memory, and what class of algorithm this makes practical that "
            "was awkward before.\n\n"
            "Extend the picture to multiple GPUs. Compare NVLink and PCIe as interconnects in "
            "bandwidth and latency terms for the current generation, explain what peer-to-peer "
            "access means and when it is available, and describe how a collective such as "
            "all-reduce actually moves bytes across a node. Explain why tensor parallelism is "
            "sensitive to interconnect bandwidth in a way that pipeline parallelism is not, "
            "and what that implies for a workstation card with no NVLink at all.\n\n"
            "Close with methodology. Given a kernel you believe is memory-bound, describe the "
            "specific counters you would collect to confirm it, which profiler surfaces them, "
            "and how you would distinguish a bandwidth limit from a latency limit from an "
            "occupancy limit — three conditions that look similar in a wall-clock number and "
            "call for entirely different fixes.\n\n"
            "One more contrast before you finish. Take a convolution over a 224x224 image "
            "batch and a fused attention kernel over a 8192-token sequence, and place both on "
            "the same roofline you used earlier. Explain which one is closer to the ridge "
            "point, how the answer changes with batch size, and why a kernel can move from "
            "memory-bound to compute-bound purely by batching without a single line of its "
            "code changing.\n\n"
            "Throughout, give concrete numbers and state which architecture generation they "
            "apply to. Where a figure varies by product within a generation, say so rather "
            "than collapsing it to a single number. Where you are uncertain about a specific "
            "figure, say so explicitly rather than presenting an estimate as measured fact."
        ),
        (
            "Write a detailed technical specification for a distributed inference system that "
            "serves multiple large language models across a GPU cluster. Assume 64 nodes of "
            "8 GPUs each, a mixed fleet of models from 7B to 400B parameters, and a tenant "
            "population with materially different latency requirements.\n\n"
            "Begin with model placement. Describe how you decide which models are resident on "
            "which GPUs, how you handle models too large for a single device, and the "
            "trade-offs between tensor parallelism, pipeline parallelism and expert "
            "parallelism for the sizes above. Explain when replication beats sharding for a "
            "small model, and how you would decide the replica count from an offered-load "
            "estimate. Address whether models should be co-resident on one GPU at all, what "
            "that does to each model's KV cache, and how you would decide the memory split "
            "between co-tenants.\n\n"
            "Specify request routing. Cover how a request finds a replica holding the right "
            "model, how you exploit prefix-cache locality in routing decisions, and how "
            "routing interacts with continuous batching in the serving engine. Explain the "
            "queueing discipline: is it FIFO, priority-based, or deadline-aware, and what "
            "happens to a request that will clearly miss its deadline. Describe admission "
            "control and how you shed load without destabilising the tail.\n\n"
            "Cover load balancing under heterogeneity. Requests differ by orders of magnitude "
            "in prompt length and generation length, so least-connections and round-robin both "
            "behave badly. Propose a scheme that accounts for the work a request represents "
            "rather than its count, and explain how you estimate that work before it runs.\n\n"
            "Specify the latency budget. Give numerical targets for time-to-first-token and "
            "inter-token latency at p50, p95 and p99, and justify each number against a "
            "concrete user-facing requirement. Explain why p99 is the number that governs "
            "capacity and why an average is nearly useless here. State the throughput target "
            "in requests per second and tokens per second, and show how the two relate given "
            "your assumed generation length.\n\n"
            "Cover fault tolerance: node loss mid-request, GPU fallen off the bus, a model "
            "that fails to load after a rolling deploy, and a slow node that is not dead. "
            "Explain how each is detected, how long detection takes, and what the user sees "
            "in each case.\n\n"
            "Specify autoscaling. Give the signal you scale on, the thresholds, the cooldowns, "
            "and how you handle the multi-minute cost of loading a large model into GPU memory "
            "— which makes reactive scaling nearly useless at the top of the size range. "
            "Describe what you do instead.\n\n"
            "Finally, describe the monitoring and alerting needed to hold these SLOs: which "
            "metrics you collect, at what cardinality, which ones page a human, and which "
            "exist only for post-incident analysis. Include the GPU-level telemetry that "
            "distinguishes a saturated cluster from a slow one — utilisation, memory "
            "bandwidth, power draw and clock throttling — and explain what each one rules in "
            "or out when you are debugging a latency regression at three in the morning.\n\n"
            "Specify KV cache management in detail, since it governs how many requests can be "
            "in flight at once. Explain paged attention and what problem it solves relative to "
            "contiguous allocation, how block size trades internal fragmentation against "
            "bookkeeping cost, and what happens when the cache fills: which requests are "
            "preempted, whether their state is recomputed or swapped to host memory, and how "
            "that choice shows up in tail latency. Give the arithmetic for KV cache bytes per "
            "token for a 70B model with grouped-query attention, and use it to compute how "
            "many concurrent 8k-token sessions fit in a given amount of spare device memory.\n\n"
            "Describe prefix caching across the fleet. Explain how shared system prompts are "
            "detected and reused, what the cache key must include for reuse to be safe, how "
            "hit rate interacts with the routing scheme you specified earlier, and the failure "
            "mode where routing for cache locality concentrates load onto one replica.\n\n"
            "Add a cost model. Express cost per million tokens as a function of GPU-hour price, "
            "achieved throughput and utilisation, and use it to compare two deployments: many "
            "small replicas of a quantized model, versus fewer sharded replicas at full "
            "precision. State which assumptions the comparison is most sensitive to.\n\n"
            "Finally, address multi-tenancy directly. If two models share a GPU, describe how "
            "you would bound the blast radius of a noisy tenant, what isolation the hardware "
            "and the serving stack can each provide, which metrics reveal that one tenant is "
            "degrading another, and how you would decide from measurements whether co-location "
            "is economically worth the interference it causes.\n\n"
            "Present the whole thing as a specification a team could build from, not as an "
            "essay. Use sections, state requirements as testable assertions with numbers "
            "attached, and where you make an assumption about the workload, mark it as an "
            "assumption and say how you would validate it against production traffic before "
            "committing to the design."
        ),
    ],
}

# Only used for the report that lets a human sanity-check the yaml's "~N tok"
# comments — never for anything a run depends on, so pulling in a real tokenizer
# and its model download is not worth it.
#
# 5.1, not the usual 4.0: measured against cl100k_base on these actual prompts,
# where the ratio came out 4.9-5.4 chars/token. Technical prose tokenizes more
# efficiently than general English — long domain words are single tokens — so
# 4.0 over-estimated the count by about 30%. The yaml comments carry the real
# tokenizer numbers; this constant only has to be close enough to catch an
# order-of-magnitude drift.
CHARS_PER_TOKEN = 5.1


def load_prompts(config_path: Path) -> dict[str, list[str]]:
    """The `prompts` block, validated as name -> non-empty list of strings."""
    data = json.loads(config_path.read_text())
    prompts = data.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise SystemExit(f"{config_path}: no top-level `prompts` object")
    out: dict[str, list[str]] = {}
    for name, entries in prompts.items():
        if not isinstance(entries, list) or not entries or not all(
            isinstance(e, str) and e.strip() for e in entries
        ):
            raise SystemExit(f"{config_path}: prompts.{name} must be a non-empty list of strings")
        out[name] = [e.strip() for e in entries]
    # Applied last so an override always wins, and so a name that exists only
    # as an override still lands. See PROMPT_OVERRIDES for why llm_long is one.
    for name, entries in PROMPT_OVERRIDES.items():
        out[name] = [e.strip() for e in entries]
    return out


def render_jsonl(entries: list[str]) -> str:
    """The exact bytes for one prompt file. `ensure_ascii=False` keeps any
    non-ASCII prompt readable in the committed artefact; aiperf reads UTF-8."""
    return "".join(json.dumps({"text": e}, ensure_ascii=False) + "\n" for e in entries)


def est_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="Fail if any file is missing or stale; write nothing.")
    args = ap.parse_args(argv)

    prompts = load_prompts(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stale: list[str] = []
    for name, entries in sorted(prompts.items()):
        path = args.out_dir / f"{name}.jsonl"
        body = render_jsonl(entries)
        current = path.read_text() if path.exists() else None
        if current != body:
            if args.check:
                stale.append(str(path.relative_to(REPO_ROOT)))
            else:
                path.write_text(body)
        toks = [est_tokens(e) for e in entries]
        print(
            f"{path.relative_to(REPO_ROOT)}: {len(entries)} prompt(s), "
            f"~{min(toks)}-{max(toks)} tok each (~{sum(toks)} tok total)"
        )

    if stale:
        print(
            "stale or missing prompt files: " + ", ".join(stale)
            + " — re-run scripts/build_contention_prompts.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
