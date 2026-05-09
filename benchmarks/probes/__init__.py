"""Out-of-band benchmark probes (Prometheus scrape, GPU sampler).

These are best-effort: scrape failures degrade silently to None fields
on `BenchmarkResult` rather than aborting the run.
"""
