# Per-GPU capacity for the headline checkpoints

How big each headline checkpoint (Qwen3-VL, Qwen3.5/3.6, Nemotron-3-Nano-Omni)
fits on each candidate GPU at each quantisation level, and where multi-GPU
starts to make sense.

> Numbers are **production-style sizing**: enough headroom for a real KV
> cache, the vision/audio tower, framework overhead, and activations — not
> the theoretical "weights only" minimum. Verify on the actual host before
> committing; the back-of-envelope here is meant to gate decisions, not
> replace `nvidia-smi`.

## Assumptions

| Field | Value | Why |
| --- | --- | --- |
| Reserve for KV / activations / framework | **25–30 % of VRAM** | KV at batch ≈ 8, ctx ≈ 8K dominates this budget for 8B–35B models |
| BF16 weight bytes/param | 2 | reference precision |
| FP8 weight bytes/param | 1 | E4M3 weights, FP8 activations (TRT-LLM / vLLM / SGLang) |
| NVFP4 weight bytes/param | ≈ 0.7 | Blackwell tensor-core 4-bit format with FP8 scale; ~5 effective bits |
| MoE memory | **total params** | all experts must be resident on the GPU(s); active params drive compute, not memory |
| Vision / audio tower | counted in the param totals below | fused into the dense weight memory for the table |

> **Note on the KV-reservation knob.** Every "Max …" cell in the tables
> below assumes the 25–30 % reservation above. Tune to your workload:
> - **More aggressive** (10–15 % reservation; low batch ≤ 2, ctx ≤ 2K,
>   single-stream) → cells push **up**.
> - **More conservative** (40–50 % reservation; batch ≥ 16, ctx ≥ 16K,
>   long prefix caching) → cells push **down**.
>
> The cells are **decision-grade defaults**, not strict limits — verify
> against your real batch and context budget before sizing the order.

## Per-precision notes

- **BF16** — official HF base / Instruct checkpoints; the precision
  baseline for accuracy comparisons.
- **FP8** — HF `…-FP8` checkpoints (Qwen ships these for 3-VL, 3.6).
  Drops weights to 1 B/param with very small accuracy loss; widely
  supported on Hopper / Blackwell tensor cores.
- **NVFP4** — Blackwell tensor-core-native 4-bit format (~0.7 B/param).
  Used here for the 5090 Nemotron Omni pick where nothing else fits.
  **Hopper does not support NVFP4 natively** — do not enable on H200.

W8A8 (INT8) and INT4/AWQ are intentionally **out of scope** for the
headline benchmark — not apples-to-apples across the three serving
frameworks. See [models.md](models.md) for the rationale.

## Memory math reference

Approximate weight memory per headline checkpoint (incl. vision/audio
tower, before KV / activations):

| Checkpoint | Params (incl. encoders) | BF16 | FP8 | NVFP4 |
| --- | --- | --- | --- | --- |
| Qwen3-VL-8B (dense) | ~9 B | 18 GB | 9 GB | 6 GB |
| Qwen3-VL-32B (dense) | ~33 B | 66 GB | 33 GB | 21 GB |
| Qwen3.5-9B (dense, text) | ~9 B | 18 GB | 9 GB | 6 GB |
| Qwen3.6-27B (dense, text) | ~27 B | 54 GB | 27 GB | 17 GB |
| Qwen3.6-35B-A3B (MoE) | ~35 B total | 70 GB | 35 GB | 22 GB |
| Nemotron-3-Nano-Omni-30B-A3B (MoE) | ~30 B total | 60 GB | 33 GB | 21 GB |
| Gemma-4-31B (dense, VLM, image+video) | ~31 B | 61 GB | 31 GB | 21 GB |

## Single-GPU capacity (1× GPU, no tensor parallelism)

| GPU | VRAM | Usable for weights† | Headline pick (precision) | Fits? |
| --- | --- | --- | --- | --- |
| **RTX 5090** (Blackwell) | 32 GB GDDR7 | ~22 GB | Qwen3-VL-8B-FP8 (9 GB); Qwen3.5-9B-FP8 (9 GB); Nemotron Omni @ NVFP4 (21 GB) | All three fit; Nemotron at NVFP4 is the only quant that works |
| **RTX PRO 6000 Blackwell Server** | 96 GB GDDR7 | ~70 GB | Qwen3-VL-32B-FP8 (33 GB); Qwen3.6-27B-FP8 (27 GB); Nemotron Omni @ FP8 (33 GB); Gemma-4-31B @ FP8 (31 GB) | All four fit comfortably with KV/ctx room |
| **H200** (Hopper) | 141 GB HBM3e | ~105 GB | Qwen3-VL-32B-BF16 (66 GB); Qwen3.6-35B-A3B-FP8 (35 GB) + 27B-FP8 (27 GB); Nemotron Omni @ BF16 (60 GB) | All three fit at the BF16 accuracy baseline; FP8 sweep has trivial headroom |
| **B300** (Blackwell Ultra) | 288 GB HBM3e | ~210 GB | reference; not in the headline matrix | All three fit at any precision; reserved for 235B-class follow-ups |

† 70 % of VRAM, leaving 30 % for KV cache / vision tower / framework
overhead at batch ≈ 8 and ctx ≈ 8K.

## Single-GPU edge cases worth calling out

- **RTX 5090 + Qwen3-VL-8B-FP8** is the consumer-baseline headline. KV at
  FP8 is comfortable; batch ≥ 16 at 8K context is realistic. Drop to BF16
  only as an accuracy reference — not the daily driver.
- **RTX 5090 + Nemotron Omni** must run at NVFP4. FP8 (~33 GB) does not
  fit. NVFP4 needs Blackwell — this exact recipe is not portable to H200.
- **H200 + Qwen3-VL-32B-BF16** is the cleanest read on memory-bandwidth
  performance: HBM3e at 4.8 TB/s with no quant artefacts. The accuracy
  baseline for any FP8 / NVFP4 delta study.
- **H200 + Qwen3.6-35B-A3B-FP8** is the **flagship TRT-LLM-vs-vLLM/SGLang
  comparison** on a dense-text-class workload. Hopper FP8 + MoE at high
  batch is what the trt-engine path was designed for; if TRT-LLM is going
  to win cleanly anywhere, it's here.

---

## Multi-GPU applicability

Multi-GPU is not free — the value depends on whether the **interconnect
between cards** can keep up with the per-token all-reduce cost of TP.
Don't add a second GPU until the single-GPU FP8 + KV-reuse + CUDA-graph
baseline is published.

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
| **RTX 5090** | **No NVLink** — PCIe Gen 5 only (~64 GB/s/direction) | TP across PCIe is dominated by all-reduce overhead. Treat any TP on 5090 as an experiment, not a default. **Replicas** for throughput are fine. |
| **RTX PRO 6000 Blackwell** | PCIe Gen 5 (workstation/server SKU does **not** ship NVLink bridges) | Same caveat as 5090 but the larger 96 GB per card means you're rarely forced into TP — single-card already fits everything in the headline matrix. Use replicas for scale-out. |
| **H200** | **NVLink** (900 GB/s/GPU on HGX H200 SXM5; NVSwitch in 8× systems) | Designed for TP. TP=2 / 4 / 8 are clean. Pick this when scaling beyond the headline picks (e.g. Nemotron Super 120B-A12B, Qwen3-VL-235B-A22B). |
| **B300** (Blackwell Ultra) | **NVLink 5** (~1.8 TB/s/GPU; GB300 NVL72 with NVSwitch up to 72 GPUs in one domain) | The unconstrained multi-GPU target for 235B / 397B / Nemotron Ultra. Out of scope for this single-GPU benchmark. |

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
- HF param counts include vision/audio towers; vendors update tower
  architectures between releases. Re-verify with
  `transformers.AutoConfig.from_pretrained(...)` before relying on the
  numeric column.
- NIM cloud's catalogue does not currently expose Qwen3-VL, Qwen3.5/3.6,
  or Nemotron Nano Omni endpoints reliably; for live inference you must
  self-host. See [INFERENCE_BACKENDS.md](../INFERENCE_BACKENDS.md) Mode B.

## NitroGen policy (not a headline checkpoint)

The [NitroGen](nitrogen.md) diffusion policy is ~500M params — negligible
against any of these GPUs (≈1 GB at BF16, less at FP8/NVFP4). It needs no
capacity planning: it fits trivially on the RTX 5090's 32 GB with room for
batching. The interesting axis for NitroGen is **execution backend × precision ×
denoise steps**, not fit — see [nitrogen.md](nitrogen.md) and
[frameworks.md](frameworks.md#nitrogen-execution-backends).
