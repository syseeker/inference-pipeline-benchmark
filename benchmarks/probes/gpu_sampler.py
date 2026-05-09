"""GPU sampler — DCGM if available, nvidia-smi as fallback.

Used as a context manager around the scenario loop:

    with GpuSampler(gpu_index=0) as gs:
        for sc in scenarios: ...
    summary = gs.summary  # dict of mem_bw / gpu_util / fb / power aggregates

The sampler runs a sidecar subprocess that emits one line per interval;
a thread drains stdout into a sample list. On exit we SIGTERM the
subprocess, drain the rest, compute p50/peak/avg, and return.

Failure modes are *soft*: if the binary is missing, the subprocess dies,
or no samples land, `summary` returns the same shape with all values
None and a `sampler_backend = "none"`. The benchmark run continues.

DCGM gives mem-bw % (`DCGM_FI_PROF_DRAM_ACTIVE`, field 1005) which is
the bandwidth-thesis metric. `nvidia-smi` cannot expose this — when we
fall back, `mem_bw_util_pct` stays None and a note is left.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from typing import IO


# DCGM field IDs we ask for. Order must match parsing in _parse_dcgmi_line.
#   203  GPU_UTIL                  %
#   252  FB_USED                   MiB
#   155  POWER_USAGE               W
#   1005 PROF_DRAM_ACTIVE          0-1 (mem-bw util)
#   1001 PROF_GR_ENGINE_ACTIVE     0-1 (graphics engine, ~ GPU util)
_DCGM_FIELDS = "203,252,155,1005,1001"


@dataclass
class _Samples:
    gpu_util_pct: list[float] = field(default_factory=list)       # 0-100
    mem_bw_util_pct: list[float] = field(default_factory=list)    # 0-100 (DCGM only)
    fb_used_mib: list[float] = field(default_factory=list)        # MiB
    power_w: list[float] = field(default_factory=list)            # watts
    gr_engine_pct: list[float] = field(default_factory=list)      # 0-100 (DCGM only)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    import math
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _to_float(token: str) -> float | None:
    """Parse a number; return None for 'N/A', '-', empty, etc."""
    t = token.strip().rstrip(",")
    if not t or t in ("N/A", "n/a", "-"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


class GpuSampler:
    """Context manager that samples GPU stats during a benchmark run.

    Args:
        gpu_index: which GPU to sample. Default 0.
        interval_ms: cadence. Default 250ms.
        prefer_dcgm: try DCGM first. Set False to force nvidia-smi.
    """

    def __init__(
        self,
        *,
        gpu_index: int = 0,
        interval_ms: int = 250,
        prefer_dcgm: bool = True,
    ) -> None:
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.prefer_dcgm = prefer_dcgm

        self._samples = _Samples()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self.backend: str = "none"   # "dcgm" | "nvidia-smi" | "none"
        self.error: str | None = None

    # ---- subprocess plumbing -----------------------------------------------

    def _start_dcgm(self) -> bool:
        if not shutil.which("dcgmi"):
            return False
        cmd = [
            "dcgmi", "dmon",
            "-i", str(self.gpu_index),
            "-d", str(self.interval_ms),
            "-e", _DCGM_FIELDS,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except (OSError, ValueError) as e:
            self.error = f"dcgmi spawn failed: {e}"
            return False
        self.backend = "dcgm"
        return True

    def _start_nvidia_smi(self) -> bool:
        if not shutil.which("nvidia-smi"):
            return False
        cmd = [
            "nvidia-smi",
            "-i", str(self.gpu_index),
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-lms", str(self.interval_ms),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except (OSError, ValueError) as e:
            self.error = f"nvidia-smi spawn failed: {e}"
            return False
        self.backend = "nvidia-smi"
        return True

    # ---- parsers -----------------------------------------------------------

    def _parse_dcgmi_line(self, line: str) -> None:
        # Format: "GPU 0   42   12345   210.5   0.310   0.420"
        # Fields: GPU id GPUTL FBUSD POWER  DRAMA  GRACT
        line = line.strip()
        if not line.startswith("GPU"):
            return
        parts = line.split()
        # Expect at least: GPU <id> <5 metric values>
        if len(parts) < 7:
            return
        gpu_util = _to_float(parts[2])
        fb_used = _to_float(parts[3])
        power = _to_float(parts[4])
        dram_active = _to_float(parts[5])    # 0-1
        gr_engine = _to_float(parts[6])      # 0-1
        if gpu_util is not None:
            self._samples.gpu_util_pct.append(gpu_util)
        if fb_used is not None:
            self._samples.fb_used_mib.append(fb_used)
        if power is not None:
            self._samples.power_w.append(power)
        if dram_active is not None:
            self._samples.mem_bw_util_pct.append(dram_active * 100.0)
        if gr_engine is not None:
            self._samples.gr_engine_pct.append(gr_engine * 100.0)

    def _parse_nvidia_smi_line(self, line: str) -> None:
        # Format: "55, 4096, 220.5"
        line = line.strip()
        if not line:
            return
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return
        gpu_util = _to_float(parts[0])
        fb_used = _to_float(parts[1])
        power = _to_float(parts[2])
        if gpu_util is not None:
            self._samples.gpu_util_pct.append(gpu_util)
        if fb_used is not None:
            self._samples.fb_used_mib.append(fb_used)
        if power is not None:
            self._samples.power_w.append(power)

    def _drain_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        parser = self._parse_dcgmi_line if self.backend == "dcgm" else self._parse_nvidia_smi_line
        stdout: IO[str] = self._proc.stdout
        try:
            for line in stdout:
                if self._stop.is_set():
                    break
                parser(line)
        except Exception:
            # If the pipe breaks (subprocess died), just stop cleanly.
            pass

    # ---- context manager ---------------------------------------------------

    def __enter__(self) -> "GpuSampler":
        started = (self.prefer_dcgm and self._start_dcgm()) or self._start_nvidia_smi()
        if not started:
            self.backend = "none"
            self.error = self.error or "no GPU sampler binary found (dcgmi / nvidia-smi)"
            return self
        self._reader = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                self._proc.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
            finally:
                # Close pipes so the reader thread unblocks.
                if self._proc.stdout is not None:
                    try:
                        self._proc.stdout.close()
                    except OSError:
                        pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)

    # ---- summary -----------------------------------------------------------

    @property
    def summary(self) -> dict[str, float | int | str | None]:
        s = self._samples
        n_samples = max(
            len(s.gpu_util_pct), len(s.mem_bw_util_pct), len(s.fb_used_mib), len(s.power_w),
        )
        fb_used_peak_gb = (max(s.fb_used_mib) / 1024.0) if s.fb_used_mib else None
        return {
            "sampler_backend": self.backend,
            "n_samples": n_samples,
            "mem_bw_util_pct_p50": _percentile(s.mem_bw_util_pct, 0.50),
            "mem_bw_util_pct_peak": (max(s.mem_bw_util_pct) if s.mem_bw_util_pct else None),
            "gpu_util_pct_p50": _percentile(s.gpu_util_pct, 0.50),
            "gpu_util_pct_peak": (max(s.gpu_util_pct) if s.gpu_util_pct else None),
            "fb_used_peak_gb": fb_used_peak_gb,
            "power_avg_w": (sum(s.power_w) / len(s.power_w)) if s.power_w else None,
            "power_peak_w": (max(s.power_w) if s.power_w else None),
        }
