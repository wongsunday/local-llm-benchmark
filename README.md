# DGX Spark vLLM serving experiments

Small, repeatable benchmarks and serving notes for the local DGX Spark.

## Run the quick benchmark on the Spark

The benchmark should run on the same host as vLLM so the timing is local to
the serving machine:

```bash
python3 bench_vllm_quick.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-tokens 256 \
  --concurrency 1 2 4 8 \
  --repeats 5
```

The benchmark performs one warm-up, then repeats each concurrency wave. It
reports TTFT, per-stream decode tok/s, aggregate decode tok/s, completion
status, token source, and a median summary across successful repeats.

Raw benchmark evidence is intentionally ignored by Git. By default each run
is written as `results/run-<UTC timestamp>-<model slug>.json`; this keeps
machine-specific endpoint/configuration metadata out of the public source
history. Keep a selected, sanitized result file only when it is useful as a
durable comparison artifact, under a separate documented baseline path. Add
`--require-usage` when a run must use server-reported
`usage.completion_tokens`; otherwise the script explicitly marks an SSE-event
count fallback as approximate.

Run the offline tests without a live model server:

```bash
python3 -m unittest -v test_bench_vllm_quick.py
```

## Git workflow

This repository is designed to be public: it contains no hostnames, LAN
addresses, credentials, SSH configuration, or machine-specific paths. Keep
actual benchmark result files private or sanitize them before committing.

Clone it on the Spark and pull changes before running a benchmark.
