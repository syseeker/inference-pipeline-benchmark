# Per-GPU capacity for Qwen3-VL checkpoints

How big a Qwen3-VL checkpoint fits on each candidate GPU at each
quantisation level, and where multi-GPU starts to make sense.

> Numbers are **production-style sizing**: enough headroom for a real KV
> cache, the vision tower, framework overhead, and activations — not the
> theoretical "weights only" minimum. Verify on the actual host before
> committing; the back-of-envelope here is meant to gate decisions, not
> replace `nvidia-smi`.

## Assumptions

| Field | Value | Why |
| --- | --- | --- |
| Reserve for KV / activations / framework | **25–30 % of VRAM** | KV at batch ≈ 8, ctx ≈ 8K dominates this budget for 8B–32B models |
| BF16 weight bytes/param | 2 | reference precision |
| FP8 weight bytes/param | 1 | E4M3 weights, FP8 activations (TRT-LLM / vLLM / SGLang) |
| W8A8 (INT8) weight bytes/param | 1 | SmoothQuant / ModelOpt INT8; same memory as FP8 |
| INT4 / AWQ weight bytes/param | ≈ 0.6 | 4-bit weights + per-group scales/zeros |
| MoE memory | **total params** | all experts must be resident on the GPU(s); active params drive compute, not memory |
| Vision tower | counted in the param totals below | fused into the dense weight memory for the table |

## Per-precision precision/notes

- **BF16** — official HF `Instruct` / `Thinking` checkpoints; the
  precision baseline for accuracy comparisons.
- **FP8** — HF `…-FP8` checkpoints (Qwen ships these). Drops weights to
  1 B/param with very small accuracy loss on Qwen-VL benchmarks; widely
  supported on Hopper / Blackwell tensor cores.
- **W8A8 (INT8)** — same memory cost as FP8. Different precision
  profile: SmoothQuant / ModelOpt-INT8 produces W8A8 calibrated weights.
  Historically more lossy on VLMs than FP8 (vision tower is sensitive to
  activation outliers), so the typical recipe is **W8A8 on the language
  tower + BF16 on the vision tower** — still ≈ 1 B/param overall.
  Treat W8A8 capacity as identical to FP8 in this table.
- **INT4 / AWQ** — Qwen3-VL does **not** ship official INT4 checkpoints
  on HF (BF16 / FP8 / GGUF only). Either calibrate via ModelOpt
  (`AWQ`/`INT4-W4A16`) or use a community quant. GGUF Q4 only flows
  through llama.cpp, not vLLM / SGLang / TRT-LLM.

## Memory math reference

Approximate weight memory per checkpoint (incl. vision tower, before
KV / activations):

| Checkpoint | Params (incl. vision) | BF16 | FP8 / W8A8 | INT4 / AWQ |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B | ~2.5 B | 5 GB | 2.5 GB | 1.5 GB |
| Qwen3-VL-4B | ~4.5 B | 9 GB | 4.5 GB | 2.5 GB |
| Qwen3-VL-8B | ~9 B | 18 GB | 9 GB | 5 GB |
| Qwen3-VL-32B | ~33 B | 66 GB | 33 GB | 17 GB |
| Qwen3-VL-30B-A3B (MoE) | ~31 B total | 62 GB | 31 GB | 16 GB |
| Qwen3-VL-235B-A22B (MoE) | ~236 B total | 472 GB | 236 GB | 118 GB |

## Single-GPU capacity (1× GPU, no tensor parallelism)

| GPU | VRAM | Usable for weights† | Max BF16 | Max FP8 | Max W8A8 | Max INT4 / AWQ‡ |
| --- | --- | --- | --- | --- | --- | --- |
| **RTX 5090** (Blackwell) | 32 GB GDDR7 | ~22 GB | **8B** (18 GB; tight on KV — drop to FP8 to free ~9 GB) | **8B** (9 GB; comfortable) | **8B** (9 GB; comfortable) | **32B / 30B-A3B** (16–17 GB) |
| **RTX PRO 6000 Blackwell Server** | 96 GB GDDR7 | ~70 GB | **32B / 30B-A3B** (62–66 GB) | **32B / 30B-A3B** (31–33 GB) | **32B / 30B-A3B** (31–33 GB) | **32B** (235B does **not** fit) |
| **H200** | 141 GB HBM3e | ~105 GB | **32B / 30B-A3B** (62–66 GB; ample KV) | **32B / 30B-A3B** (31–33 GB; trivial) | **32B / 30B-A3B** | **32B**; 235B-INT4 (118 GB) is borderline |
| **B300** (Blackwell Ultra) | 288 GB HBM3e | ~210 GB | **32B / 30B-A3B** (235B-BF16 = 472 GB needs TP) | **235B-A22B** (236 GB; tight at high batch / long ctx) | **235B-A22B** (236 GB; tight at high batch / long ctx) | **235B-A22B** (118 GB; comfortable) |

† 70 % of VRAM, leaving 30 % for KV cache / vision tower / framework
overhead at batch ≈ 8 and ctx ≈ 8K. Reduce reservation if you
specifically know batch and context will be lower.
‡ Qwen3-VL ships no official INT4 — assumes ModelOpt or community quant.

## Single-GPU edge cases worth calling out

- **RTX 5090 + 8B BF16** is the headline single-GPU consumer recipe.
  KV is tight: plan batch ≤ 4 at 8K context. **Drop to 8B-FP8** and you
  reclaim ~9 GB → batch ≥ 16 at 8K is realistic. This is the **first
  optimisation lever** Karan flagged: prove single-GPU FP8 + KV reuse
  before considering multi-GPU.
- **30B-A3B vs. 32B dense** at the same VRAM tier: MoE costs ~the same
  memory but only ~3B params are active per token. It is the cleanest
  **memory-bandwidth** stress test in the matrix and worth running at
  every tier.
- **H200 cannot host 235B-A22B-FP8 on 1×** (236 GB > 141 GB). Either
  TP=2 across 2× H200 (NVLink-friendly) or move to B300.
- **B300 + 235B-A22B-FP8** fits with ~50 GB headroom. Fine for low
  concurrency / short context; tight for production batch sizes. For
  headroom, prefer 235B-INT4 on 1× B300 **or** 235B-BF16 across 2× B300
  (TP=2).

---

## Multi-GPU applicability

Multi-GPU is not free — the value depends on whether the **interconnect
between cards** can keep up with the per-token all-reduce cost of TP.
Karan's bandwidth thesis applies here too: don't add a second GPU until
the single-GPU FP8 + KV-reuse + CUDA-graph baseline is published.

### When multi-GPU helps

| Reason | Helped by | Notes |
| --- | --- | --- |
| Model + KV doesn't fit on 1× | TP=2/4/8 | Each rank holds 1/N of weights and KV |
| Per-token latency is bandwidth-bound | TP (NVLink) | Matmul is split N ways; only worth it if interconnect ≪ matmul time |
| MoE expert capacity exceeds VRAM | EP (expert parallel) | Splits experts across ranks; works well when active set is local to a rank |
| Throughput is request-bound | Replicas (no TP) | Run N independent servers behind a load balancer; almost always preferred when 1× already fits the model |

### Per-GPU multi-GPU notes

| GPU | Inter-card link | Multi-GPU verdict |
| --- | --- | --- |
| **RTX 5090** | **No NVLink** — PCIe Gen 5 only (~64 GB/s/direction) | TP across PCIe is dominated by all-reduce overhead. Expected outcome: 2× 5090 with TP=2 is **slower** than 1× 5090 with FP8 + KV reuse for 4B/8B. Treat any TP on 5090 as an experiment, not a default. **Replicas** for throughput are fine. |
| **RTX PRO 6000 Blackwell** | PCIe Gen 5 (workstation/server SKU does **not** ship NVLink bridges) | Same caveat as 5090 but the larger 96 GB per card means you're rarely forced into TP — single-card already fits 32B/30B-A3B. Use replicas for scale-out. |
| **H200** | **NVLink** (900 GB/s/GPU on HGX H200 SXM5; NVSwitch in 8× systems) | Designed for TP. TP=2 / 4 / 8 are clean. Pick this when you need 235B serving, or when 32B at very high batch is bandwidth-bound. |
| **B300** (Blackwell Ultra) | **NVLink 5** (~1.8 TB/s/GPU; GB300 NVL72 with NVSwitch up to 72 GPUs in one domain) | The unconstrained multi-GPU target. TP=2 of 235B-BF16, EP across 8× for 235B serving, full 72-way racks for hundreds-of-billions-class models. |

### Multi-GPU capacity matrix (FP8, with KV headroom)

Cells show "fits / tight / no" for the largest Qwen3-VL checkpoint at FP8.

| Config | Aggregate VRAM | 8B | 32B | 30B-A3B | 235B-A22B |
| --- | --- | --- | --- | --- | --- |
| 1× RTX 5090 | 32 GB | fits | no | no | no |
| 2× RTX 5090 (TP=2, PCIe) | 64 GB | fits | tight (PCIe loss likely) | tight (PCIe loss likely) | no |
| 1× RTX PRO 6000 | 96 GB | fits | fits | fits | no |
| 2× RTX PRO 6000 (TP=2, PCIe) | 192 GB | fits | fits | fits | tight (PCIe loss likely) |
| 1× H200 | 141 GB | fits | fits | fits | no |
| 2× H200 (TP=2, NVLink) | 282 GB | fits | fits | fits | **fits** |
| 4× H200 (TP=4, NVLink) | 564 GB | overkill | fits | fits | fits (BF16 also possible) |
| 1× B300 | 288 GB | fits | fits | fits | fits (tight) |
| 2× B300 (TP=2, NVLink 5) | 576 GB | overkill | fits | fits | **fits comfortably (BF16 also possible)** |

### Recommended configurations

| Goal | Suggested config | Why |
| --- | --- | --- |
| Razer-relevant consumer baseline | **1× RTX 5090, Qwen3-VL-8B-FP8** | Matches the customer device. TP=2 on PCIe is risky; first prove single-GPU FP8 + prefix-cache + CUDA graphs. |
| Server-class POC, single GPU | **1× RTX PRO 6000, Qwen3-VL-32B-FP8** *or* **30B-A3B-FP8** | 96 GB lets you run the bigger reasoner at FP8 with a real KV budget; MoE variant exercises the bandwidth thesis. |
| Memory-bandwidth ceiling reference | **1× H200, Qwen3-VL-32B-BF16** | Cleanest read on bandwidth-bound performance — HBM3e at 4.8 TB/s, no quant artefacts. |
| 235B-A22B serving | **2× H200 TP=2 (FP8)** *or* **1× B300 (FP8)** *or* **2× B300 TP=2 (BF16)** | Single-card fits 235B only on B300; H200 needs TP=2; B300 NVLink 5 handles 2× cleanly for the BF16 reference. |
| Throughput at scale (any size) | **N× single-GPU replicas behind a load balancer** | If 1× already fits the model, replicas beat TP — no all-reduce, linear scaling, simpler ops. |

### What to NOT do

- **Don't** TP RTX 5090 / RTX PRO 6000 to "scale up performance" on a
  model that already fits one card — the PCIe round-trip on every
  attention layer will erase the win on the latency-sensitive
  command-sequence path.
- **Don't** assume MoE EP "magically" reduces memory: the experts still
  have to live somewhere. EP shards them across ranks; total memory is
  unchanged.
- **Don't** report tokens/sec from a multi-GPU run without recording the
  interconnect overhead. Always cite **TP efficiency** =
  `latency(TP=1) / (N × latency(TP=N))`. A value ≤ 1 means parallelism
  is hurting you.

## Source-of-truth caveats

- Per-GPU VRAM and bandwidth numbers are **target specs**, not
  validated. Run `scripts/gpu_probe.sh` on the actual host and copy the
  results into `benchmarks/results/host_<hostname>.json` before
  publishing.
- HF param counts include the vision tower for Qwen3-VL; the Qwen team
  may update vision-tower architectures between releases. Re-verify
  with `transformers.AutoConfig.from_pretrained(...)` before relying on
  the numeric column.
- NIM cloud's catalogue does not currently include Qwen3-VL or
  Qwen2.5-VL; for live Qwen-VL inference you must self-host. See
  [QUICKSTART.md](../QUICKSTART.md) Mode B.
