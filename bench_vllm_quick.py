#!/usr/bin/env python3
"""Small, repeatable OpenAI-compatible streaming benchmark for a vLLM endpoint."""

import argparse
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCRIPT_VERSION = "0.2.0"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180
DEFAULT_WAVE_TIMEOUT_SECONDS = 300


PROMPTS = [
    "Write a concise technical explanation of paged attention, with numbered sections and one short example.",
    "Generate a JSON object containing 20 synthetic GPU telemetry records with timestamp, temperature, power, and utilization fields.",
    "Write a small Python implementation of a bounded worker queue, followed by a brief explanation of its shutdown behavior.",
    "Explain how speculative decoding works, including draft tokens, verification, acceptance, and fallback.",
    "Produce a compact HTML report with a heading, a table of five servers, and a short operations summary.",
    "List practical vLLM tuning steps for improving decode throughput while keeping time-to-first-token reasonable.",
    "Describe the difference between prefill throughput, decode throughput, per-request throughput, and aggregate throughput.",
    "Write a structured troubleshooting checklist for a GPU inference server showing high latency and low utilization.",
]


def read_sse(resp, started, wave_started):
    first = None
    last = None
    delta_events = 0
    usage_tokens = None
    model = None
    finish_reason = None
    stream_done = False
    buf = b""
    while True:
        block = resp.readline()
        if not block:
            break
        buf += block
        if not buf.endswith(b"\n\n") and not buf.endswith(b"\r\n\r\n"):
            continue
        event = buf.decode("utf-8", "replace")
        buf = b""
        for line in event.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                stream_done = True
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            model = obj.get("model") or model
            usage = obj.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                usage_tokens = int(usage["completion_tokens"])
            choice = (obj.get("choices") or [{}])[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            # vLLM model adapters vary: Nemotron emits `reasoning`, while
            # other OpenAI-compatible servers use `reasoning_content`.
            text = (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or ""
            )
            if text:
                now = time.perf_counter()
                first = first or now
                last = now
                delta_events += 1
    ended = time.perf_counter()
    token_source = "usage.completion_tokens" if usage_tokens is not None else "sse_delta_events"
    approximate_token_count = usage_tokens is None
    tokens = usage_tokens if usage_tokens is not None else delta_events
    decode_tokens = max(0, tokens - 1)
    decode_ms = ((last - first) * 1000) if first and last and last > first else 0
    return {
        "ttft_ms": ((first - started) * 1000) if first else 0,
        "decode_ms": decode_ms,
        "completion_tokens": tokens,
        "decode_tokens": decode_tokens,
        "decode_tps": (decode_tokens / decode_ms * 1000) if decode_ms else 0,
        "total_ms": (ended - started) * 1000,
        # Relative to the wave start, so concurrent aggregation uses a
        # common clock while the saved result remains portable.
        "first_at_ms": ((first - wave_started) * 1000) if first else None,
        "last_at_ms": ((last - wave_started) * 1000) if last else None,
        "delta_events": delta_events,
        "token_source": token_source,
        "approximate_token_count": approximate_token_count,
        "finish_reason": finish_reason,
        "stream_done": stream_done,
        "completed_normally": bool(stream_done and finish_reason),
        "model": model,
    }


def one_request(
    base_url,
    model,
    prompt,
    max_tokens,
    wave_started=None,
    request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = Request(base_url.rstrip("/") + "/chat/completions", data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })
    started = time.perf_counter()
    wave_started = wave_started if wave_started is not None else started
    try:
        with urlopen(req, timeout=request_timeout) as resp:
            result = read_sse(resp, started, wave_started)
            result["http_status"] = getattr(resp, "status", None)
            return result
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return {
            "error": f"HTTP {exc.code}: {detail}",
            "http_status": exc.code,
            "total_ms": (time.perf_counter() - started) * 1000,
        }
    except URLError as exc:
        return {
            "error": f"URL error: {exc.reason}",
            "total_ms": (time.perf_counter() - started) * 1000,
        }
    except Exception as exc:
        return {"error": str(exc), "total_ms": (time.perf_counter() - started) * 1000}


def load_prompts(path):
    """Read a prompt set from a JSON array or a newline-delimited file.

    Acceptance length — and therefore decode throughput — depends heavily on how
    predictable the output is, so comparing workloads requires swapping the
    prompt set while holding every other setting constant.
    """
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        prompts = json.loads(raw)
    else:
        prompts = [line.strip() for line in raw.splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def wave(
    base_url,
    model,
    concurrency,
    max_tokens,
    prompts=None,
    require_usage=False,
    request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    wave_timeout=DEFAULT_WAVE_TIMEOUT_SECONDS,
):
    started = time.perf_counter()
    prompt_set = prompts or PROMPTS
    prompts = [prompt_set[i % len(prompt_set)] for i in range(concurrency)]
    pool = ThreadPoolExecutor(max_workers=concurrency)
    futures = {
        pool.submit(
            one_request,
            base_url,
            model,
            prompt,
            max_tokens,
            started,
            request_timeout,
        ): index
        for index, prompt in enumerate(prompts)
    }
    rows = [None] * concurrency
    timed_out = False
    try:
        for future in as_completed(futures, timeout=wave_timeout):
            rows[futures[future]] = future.result()
    except FuturesTimeoutError:
        timed_out = True
    finally:
        for future in futures:
            future.cancel()
        # Do not make the caller wait for a request that has already exceeded
        # the wave budget. Individual urlopen calls still have their own cap.
        pool.shutdown(wait=not timed_out, cancel_futures=True)
    if timed_out:
        for index, row in enumerate(rows):
            if row is None:
                rows[index] = {
                    "error": f"Wave timed out after {wave_timeout}s",
                    "wave_timeout": True,
                }
    ended = time.perf_counter()
    ok = [
        r for r in rows
        if "error" not in r
        and r.get("decode_tokens", 0) > 0
        and r.get("first_at_ms") is not None
        and r.get("last_at_ms") is not None
        and (not require_usage or r.get("token_source") == "usage.completion_tokens")
    ]
    if not ok:
        return {
            "concurrency": concurrency,
            "ok": 0,
            "failed": len(rows),
            "error": next((r.get("error") for r in rows if r.get("error")), "No valid streams"),
            "wave_ms": (ended - started) * 1000,
            "streams": rows,
        }
    first = min(r["first_at_ms"] for r in ok)
    last = max(r["last_at_ms"] for r in ok)
    total_tokens = sum(r["decode_tokens"] for r in ok)
    aggregate = total_tokens / (last - first) * 1000 if last > first else 0
    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "wave_ms": (ended - started) * 1000,
        "mean_decode_tps": statistics.mean(r["decode_tps"] for r in ok),
        "median_decode_tps": statistics.median(r["decode_tps"] for r in ok),
        "median_ttft_ms": statistics.median(r["ttft_ms"] for r in ok),
        "aggregate_decode_tps": aggregate,
        "approximate_streams": sum(r.get("approximate_token_count", False) for r in ok),
        "incomplete_streams": sum(not r.get("completed_normally", False) for r in ok),
        "streams": rows,
    }


def summarize_results(results):
    """Summarize only fully successful waves for primary comparisons."""
    summary = []
    levels = sorted({r["concurrency"] for r in results})
    for concurrency in levels:
        rows = [r for r in results if r["concurrency"] == concurrency]
        complete = [r for r in rows if r.get("ok") == concurrency and not r.get("error")]
        if not complete:
            summary.append({"concurrency": concurrency, "complete_repeats": 0})
            continue
        tps = [r["aggregate_decode_tps"] for r in complete]
        per_stream = [r["mean_decode_tps"] for r in complete]
        ttft = [r["median_ttft_ms"] for r in complete]
        summary.append({
            "concurrency": concurrency,
            "complete_repeats": len(complete),
            "failed_repeats": len(rows) - len(complete),
            "median_per_stream_tps": statistics.median(per_stream),
            "median_aggregate_tps": statistics.median(tps),
            "min_aggregate_tps": min(tps),
            "max_aggregate_tps": max(tps),
            "median_ttft_ms": statistics.median(ttft),
            "approximate_waves": sum(bool(r.get("approximate_streams")) for r in complete),
        })
    return summary


def default_output_path(model):
    """Return a private, ignored path for raw evidence from this run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower() or "model"
    return os.path.join("results", f"run-{stamp}-{slug}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    ap.add_argument("--wave-timeout", type=float, default=DEFAULT_WAVE_TIMEOUT_SECONDS)
    ap.add_argument(
        "--prompts-file",
        default=None,
        help="JSON array or newline-delimited prompts; defaults to the built-in mix",
    )
    ap.add_argument(
        "--require-usage",
        action="store_true",
        help="Reject streams unless usage.completion_tokens is reported",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Raw evidence path; defaults to results/run-<UTC>-<model>.json",
    )
    args = ap.parse_args()
    if args.output is None:
        args.output = default_output_path(args.model)

    prompts = load_prompts(args.prompts_file) if args.prompts_file else PROMPTS

    print("warmup")
    warmup = one_request(
        args.base_url,
        args.model,
        prompts[0],
        args.max_tokens,
        request_timeout=args.request_timeout,
    )
    print(json.dumps(warmup, indent=2))
    if warmup.get("error") or warmup.get("decode_tokens", 0) <= 0:
        raise SystemExit(f"warmup failed: {warmup.get('error', 'no decoded tokens')}")
    results = []
    for c in args.concurrency:
        for repeat in range(args.repeats):
            print(f"concurrency={c} repeat={repeat + 1}/{args.repeats}", flush=True)
            result = wave(
                args.base_url,
                args.model,
                c,
                args.max_tokens,
                prompts=prompts,
                require_usage=args.require_usage,
                request_timeout=args.request_timeout,
                wave_timeout=args.wave_timeout,
            )
            result["repeat"] = repeat + 1
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "streams"}, indent=2))
    output = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "config": vars(args),
        # The prompt text is the workload. Record it so a rate is never read
        # without knowing what produced it.
        "prompt_set": prompts,
        "warmup": warmup,
        "results": results,
        "summary": summarize_results(results),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"wrote {args.output}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
