import io
import json
import unittest

from bench_vllm_quick import default_output_path, read_sse, summarize_results, wave


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.stream = io.BytesIO(body.encode())

    def readline(self):
        return self.stream.readline()


def event(payload):
    return f"data: {json.dumps(payload)}\n\n"


class BenchmarkTests(unittest.TestCase):
    def test_default_output_path_is_standardized_and_ignored(self):
        path = default_output_path("nvidia/Nemotron 30B/NVFP4")
        self.assertRegex(path, r"^results/run-\d{8}T\d{6}Z-nvidia-nemotron-30b-nvfp4\.json$")

    def test_usage_tokens_and_reasoning_stream_are_measured(self):
        body = "".join([
            event({"model": "test", "choices": [{"delta": {"reasoning": "one"}}]}),
            event({"choices": [{"delta": {"content": "two"}, "finish_reason": "stop"}]}),
            event({"choices": [], "usage": {"completion_tokens": 12}}),
            "data: [DONE]\n\n",
        ])
        result = read_sse(FakeResponse(body), 1.0, 1.0)
        self.assertEqual(result["completion_tokens"], 12)
        self.assertEqual(result["decode_tokens"], 11)
        self.assertEqual(result["token_source"], "usage.completion_tokens")
        self.assertFalse(result["approximate_token_count"])
        self.assertTrue(result["completed_normally"])
        self.assertEqual(result["model"], "test")

    def test_missing_usage_is_explicitly_approximate(self):
        body = "".join([
            event({"choices": [{"delta": {"content": "one"}}]}),
            event({"choices": [{"delta": {"content": "two"}, "finish_reason": "length"}]}),
            "data: [DONE]\n\n",
        ])
        result = read_sse(FakeResponse(body), 1.0, 1.0)
        self.assertEqual(result["token_source"], "sse_delta_events")
        self.assertTrue(result["approximate_token_count"])

    def test_summary_reports_median_and_absolute_concurrent_window(self):
        rows = [
            {
                "decode_tokens": 99,
                "decode_tps": 99.0,
                "ttft_ms": 10.0,
                "decode_ms": 90.0,
                "first_at_ms": 10.0,
                "last_at_ms": 100.0,
                "token_source": "usage.completion_tokens",
                "approximate_token_count": False,
                "completed_normally": True,
            },
            {
                "decode_tokens": 199,
                "decode_tps": 95.0,
                "ttft_ms": 20.0,
                "decode_ms": 200.0,
                "first_at_ms": 20.0,
                "last_at_ms": 220.0,
                "token_source": "usage.completion_tokens",
                "approximate_token_count": False,
                "completed_normally": True,
            },
        ]
        # Run the real wave aggregator without making HTTP requests.
        import bench_vllm_quick

        original = bench_vllm_quick.one_request
        try:
            bench_vllm_quick.one_request = lambda *args, **kwargs: rows.pop(0)
            result = wave("http://unused/v1", "test", 2, 256)
        finally:
            bench_vllm_quick.one_request = original
        self.assertAlmostEqual(result["aggregate_decode_tps"], 298 / 210 * 1000)

        summary = summarize_results([result])
        self.assertEqual(summary[0]["complete_repeats"], 1)
        self.assertEqual(summary[0]["median_aggregate_tps"], result["aggregate_decode_tps"])


if __name__ == "__main__":
    unittest.main()
