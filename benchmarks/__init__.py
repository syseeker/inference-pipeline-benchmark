"""Benchmark harness for the VLM-to-action pipeline.

Each framework adapter implements `BenchmarkAdapter` and emits
`BenchmarkResult` rows. The runner orchestrates a (framework, gpu, model,
quantization) sweep and writes raw + summarised results.
"""
