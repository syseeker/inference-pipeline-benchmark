"""In-run /metrics poller — captures peak gauge values during the timed loop.

The post-run `scrape()` call only sees gauges *after* requests have drained,
so `kv_cache_usage_pct` and (for sglang/trtllm) `prefix_cache_hit_rate` read
back as 0 / None. This poller scrapes on a fixed cadence in a sidecar thread
and keeps the running max for each gauge field so the BenchmarkResult sees
the actual run-time peak.

Pattern mirrors `GpuSampler`: context manager, soft failure (errors recorded,
run continues), `peaks` dict read after exit.
"""

from __future__ import annotations

import threading

from benchmarks.probes.prom_scrape import ScrapeError, scrape


# Gauge fields whose peak we want to track. Histogram percentiles and
# counter-ratio fields stay on the post-run scrape (they're cumulative).
_PEAK_FIELDS = ("kv_cache_usage_pct", "prefix_cache_hit_rate")


class PromPoller:
    """Background poller for the model server's /metrics endpoint.

    Args:
        base_url: same OpenAI-style URL the runner uses (e.g.
            `http://host:8000/v1`); `scrape()` strips `/v1` internally.
        framework: "vllm" | "sglang" | "trtllm". Other values disable polling.
        interval_s: cadence between scrapes. Default 0.5s — short enough to
            catch peaks on sub-second scenarios, long enough that the HTTP
            cost is negligible compared to inference work.
    """

    def __init__(
        self,
        *,
        base_url: str,
        framework: str,
        interval_s: float = 0.5,
    ) -> None:
        self.base_url = base_url
        self.framework = framework
        self.interval_s = interval_s

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peaks: dict[str, float | None] = {k: None for k in _PEAK_FIELDS}
        self.n_samples: int = 0
        self.n_errors: int = 0
        self.last_error: str | None = None

    # ---- context manager ---------------------------------------------------

    def __enter__(self) -> "PromPoller":
        if self.framework not in ("vllm", "sglang", "trtllm"):
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2.0)

    # ---- worker ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                m = scrape(self.base_url, self.framework, timeout_s=2.0)
                self.n_samples += 1
                for key in _PEAK_FIELDS:
                    v = getattr(m, key, None)
                    if v is None:
                        continue
                    cur = self.peaks[key]
                    if cur is None or v > cur:
                        self.peaks[key] = v
            except ScrapeError as e:
                self.n_errors += 1
                self.last_error = str(e)
            # Use Event.wait so __exit__ can interrupt the sleep promptly.
            self._stop.wait(self.interval_s)
