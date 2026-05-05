# GPU strategy

> **Sizing reference:** for the per-GPU "what Qwen3-VL checkpoint fits"
> table (BF16 / FP8 / W8A8 / INT4) and the multi-GPU capacity matrix
> with NVLink-vs-PCIe applicability notes, see
> [capacity.md](capacity.md).


## Thesis

> **The bottleneck for this VLM-action use case is memory bandwidth, not
> VRAM capacity** (per Karan).

Therefore: do not jump to tensor parallelism on consumer GPUs. Prove the
single-GPU optimised baseline first. Only test multi-GPU if profiling
shows the model is still bandwidth-bound **and** the interconnect overhead
does not erase the win.

## Stages

| Stage | GPU | Memory | Bandwidth | NVLink | Why |
| --- | --- | --- | --- | --- | --- |
| Consumer-target baseline | 1× RTX 5090 | 32 GB GDDR7 | ~1.79 TB/s | **No** | Razer-relevant target device. PCIe Gen 5 only. Risky for TP. |
| Server POC | 1× RTX PRO 6000 Blackwell (Server Edition) | 96 GB GDDR7 | ~1.6 TB/s | – | Server workflow, MIG, video encode/decode. Bridge between consumer and DC. |
| Bandwidth ceiling | 1× H200 | 141 GB HBM3e | **~4.8 TB/s** | NVLink | Cleanest benchmark for the bandwidth thesis. |

(Bandwidth numbers are approximate; pull authoritative values from the
NVIDIA datasheet at run time and bake into the per-GPU yaml under
`benchmarks/configs/`.)

## Owners

- This user (`boonpingl@nvidia.com`): RTX PRO 6000 + H200 runs.
- Peer at Razer / partner: RTX 5090 runs.
- Each side publishes raw + summarised results into
  `benchmarks/results/<gpu>/<run-id>/` and the summary table here.

## Test matrix (single GPU)

For each `(gpu, framework, model, quantization)` cell we want:

- TTFT (p50/p95/p99)
- Inter-token latency (p50/p95/p99)
- End-to-end command-sequence latency
- Vision-encoder latency (when CV stage exists)
- Memory bandwidth utilisation (DCGM / `nvidia-smi dmon` / Nsight)
- KV-cache hit rate (framework-reported)
- CUDA-graph on/off delta
- Quantisation accuracy delta vs. BF16 reference

Configs live in `benchmarks/configs/{rtx5090,rtx_pro6000,h200}.yaml`.

## When to introduce multi-GPU

**Only after the single-GPU optimised baseline is published.** Then test:

- 2× RTX 5090 (PCIe-only) — TP=2 on vLLM and SGLang. Expect interconnect
  to be the dominant cost; document the win/loss honestly.
- 2× RTX PRO 6000 — same shape, with the server-class PCIe topology.
- 2× / 4× H200 — NVLink-class scaling reference.

If the single-GPU runs already meet the latency budget, the recommendation
to Razer is *do not buy two GPUs*.

## Probes

`scripts/gpu_probe.sh` (placeholder) records, per host, the following so
that result rows are self-describing:

- `nvidia-smi -q` (driver, GPU model, P-state, ECC)
- `nvcc --version`
- `nvidia-smi topo -m`
- `dcgmi diag -r 1` summary
- CUDA / TRT / TRT-LLM / vLLM / SGLang versions in the active env
