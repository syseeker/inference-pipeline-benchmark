# Contention measurement — the decisions, and why

Design decisions for the multi-model GPU contention study, with the reasoning.
Written to be readable by someone who did not attend the design discussion.

The customer's original brief is `workspace/contention/experiment_design.md`.
Where this document departs from it, the departure is stated and justified.

---

## 1. Open-loop load, not concurrency

**The customer's design** drives LLMs by *concurrency* ("keep 4 requests in
flight") and CV by *request rate* ("50 images per second"). Those are two
different experiments, and comparing across them is invalid.

### Why it matters — a game example

Picture a game. The world sends the AI a decision request every 100 ms, whether
or not the previous one has come back. That is **open-loop**: the world does not
wait. If inference slows down, requests pile up and the character visibly lags.

Now the closed-loop version: "always keep exactly 4 requests in flight." When the
GPU slows down, the client **automatically sends more slowly**, because it is
waiting for replies before issuing new work. Per-request latency looks almost
unchanged. The system appears healthy.

That is the trap. **A closed-loop baseline hides the damage**, because the client
throttles itself in exact proportion to the slowdown you are trying to measure.
The degradation ratio ends up describing the test harness, not the GPU.

No real workload behaves that way. Game engines, video pipelines and camera feeds
all push at their own rate and do not wait for you.

### Decision

Open-loop fixed rate for every tenant, solo and co-resident. Record both
`offered_rps` and `achieved_rps`.

**Bonus:** the point where `achieved_rps` falls below `offered_rps` *is* the
safe-operating-envelope boundary the design asks for. It comes free — no extra
experiment needed.

### Consequence for baselines

A solo baseline must run at the **same offered rate** as the contention run it is
compared against. A baseline collected at a different rate makes every ratio
derived from it an artifact.

---

## 2. Clock throttling is not contention

Under co-residency, power draw rises, the GPU hits its power cap, SM clocks drop,
and every tenant slows down. That is a real effect — but it is **not contention**,
and a contention matrix that silently reports it is wrong.

### Decision

- Pin the power limit first, then lock graphics clocks at **60–80% of max boost**.
  Locking at max under a power cap just gets you silently throttled anyway.
- **Record the achieved clock distribution with every result.** On GeForce the
  lock is advisory, not guaranteed — NVIDIA is explicit that clock requests never
  override the card's self-protection.
- **Assert on `clocks_throttle_reasons.active`.** Fail the run if `SwPowerCap` or
  `HwThermalSlowdown` fired, rather than publishing a throttled number as though
  it were a contention finding.

### Per-GPU support

| | RTX 5090 | RTX PRO 6000 | H200 |
|---|---|---|---|
| `-lgc` lock graphics clock | Yes, **advisory** | Yes | Yes |
| `-lmc` lock memory clock | No-op (GDDR7 fixed in P0) | Yes | **Unsupported** — needs `--lock-memory-clocks-deferred` |
| `-pl` power limit | Yes (575 W default) | Yes (600 W; 300 W Max-Q) | Yes (700 W SXM) |

All of these require root.

---

## 3. One GPU sampler per window, not one per tenant

The single-model runner creates a `GpuSampler` per round. With N co-resident
tenants that would mean N `dcgmi dmon` subprocesses on one GPU, N different
sampling windows, and — worst of all — **every tenant reporting the whole GPU's
memory as its own**, because DCGM is device-scoped and has no per-process view.

### Decision

The orchestrator owns exactly one sampler spanning the union of all tenant
windows. Whole-GPU numbers attach to the **colocation**, not to individual tenant
rows. Per-tenant VRAM, where needed, comes from
`nvidia-smi --query-compute-apps`, not from the sampler.

---

## 4. Per-request timestamps

Durations alone cannot be aligned. Two tenants' request streams need a **shared
wall clock** before you can show that tenant A's p99 spike lands inside tenant B's
prefill window — and that alignment is the entire causal claim of the study.

### Decision

Record `start_epoch_ms` and `end_epoch_ms` per request.

- `time.time()` for the shared timeline — comparable across processes.
- `perf_counter()` for durations — monotonic, immune to clock adjustment.
- **Never conflate them.** One is for alignment, the other for measurement.

---

## 5. Repetition policy comes from measurement

The customer's design assumes <5% run-to-run variance and repeats only the
memory-pressure scenarios. That figure is for **solo** inference on dedicated
hardware. Co-residency adds scheduler non-determinism, allocator ordering, and
thermal coupling between tenants — all of which raise variance.

### Decision

Run one colocation 5× in Phase 0 and report the actual spread. Set the repetition
policy from that measurement rather than from assumption.

The customer's instinct about memory pressure is right and is kept: near-OOM
scenarios are genuinely **bimodal** (the model either fits or thrashes), so the
mean is misleading. Where variance is high, report both modes.

---

## 6. Sampling interval

The default 250 ms sampling interval gives 1–2 samples across a 200–500 ms VLM
prefill burst — too coarse to attribute a latency spike to it.

### Decision

50 ms for alignment runs. 250 ms is adequate for aggregate resource reporting.

---

## 7. Dimensions dropped, and why

| Dropped | Reason |
|---|---|
| **Quantization (q4 vs fp16)** | The design specifies "Q4_0", which is a **llama.cpp GGUF** format. vLLM cannot load it — GGUF moved out-of-tree to `vllm-gguf-plugin` and is documented as highly experimental. Replaced by choosing the best-fitting weight format per model per GPU (AWQ where an official checkpoint exists, else BF16/FP8). |
| **llama.cpp backend** | No Triton backend exists, and it adds no new contention axis that vLLM/SGLang/TRT-LLM do not already cover. |
| **MPS / MIG as a swept dimension** | Only RTX PRO 6000 has MIG, so the dimension is not comparable across the test bed. Fixed at the best available setting per GPU instead; MIG kept as one hardware-isolated reference run. |

## 8. Dimensions that became free

| Dimension | How |
|---|---|
| **Arrival pattern** (burst / uniform / Poisson) | Native to both drivers: `--arrival-pattern` in AIPerf, `--request-distribution` in perf_analyzer. Poisson added at no cost — it is the realistic middle the design was interpolating. |
| **Load asymmetry** (1:1, 4:1, 16:1) | Two independent rate-controlled generators — set different rates. |

---

## 9. What the customer got right

Worth stating, because most of the design is sound and is adopted unchanged:

- **Phase 0 as a hard gate.** If the serving layer serialises rather than
  overlaps, the whole study changes meaning. Front-loading that check is correct.
- **Solo baselines before anything else.** Degradation ratios are meaningless
  without them.
- **The dual-baseline design in Phase 6** (compute-bound vs memory-bound) is what
  surfaces interaction effects — "backend choice matters 3× more under memory
  pressure" is only visible with two contrasting baselines.
- **Metrics reporting policy** — P50/P95/max always, P99 and mean±std only where
  sample counts justify them. Statistically honest.
- **Model load measurement** (`time_to_first_ready_s`, `vram_after_load_gb`).
  Cold-start cost never appears in inference metrics but decides whether a
  co-residency plan is deployable at all.
