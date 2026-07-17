import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import pytest


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return " ".join(str(item) for item in ids)

    def encode(self, prompt, add_special_tokens=False):
        return [int(item) for item in prompt.split()]


def test_rotating_prompt_variants_keep_length_and_change_prefix() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")

    variants = bench._build_rotating_prompt_variants(
        tokenizer=FakeTokenizer(),
        safe_token_ids=[101, 202, 303, 404],
        target_len=8,
        variant_count=4,
    )

    assert [ids for _, ids in variants] == [
        [101, 202, 303, 404, 101, 202, 303, 404],
        [202, 303, 404, 101, 202, 303, 404, 101],
        [303, 404, 101, 202, 303, 404, 101, 202],
        [404, 101, 202, 303, 404, 101, 202, 303],
    ]
    assert len({prompt for prompt, _ in variants}) == 4


def test_diagnostic_request_headers_are_opt_in_and_unique() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")

    assert bench._request_headers("", 8000, 7) is None
    assert bench._request_headers("phase462-8k", 8000, 7) == {
        "X-AIC-Prompt-Tokens": "8000",
        "X-Request-Id": "phase462-8k-000007",
    }


def test_streaming_payload_requests_usage() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")

    payload = bench._make_payload(
        model="kimi-k2.5",
        output_len=2_000,
        temperature=0.0,
        ignore_eos=True,
        prompt="hello",
        prompt_token_ids=None,
        stream=True,
    )

    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_stream_state_uses_first_nonempty_text_and_done_for_latency() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")
    state = bench.StreamResponseState(started_at=10.0)

    state.observe({"choices": [{"text": ""}]}, observed_at=10.5)
    state.observe({"choices": [{"text": "a"}]}, observed_at=11.5)
    state.observe({"choices": [{"text": "b"}]}, observed_at=12.0)
    state.observe(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8_000,
                "completion_tokens": 2,
                "total_tokens": 8_002,
            },
        },
        observed_at=12.1,
    )
    state.finish(observed_at=13.0)

    record = state.to_record(
        request_index=7,
        status_code=200,
        expected_prompt_tokens=8_000,
        expected_completion_tokens=2,
    )

    assert record.ttft_ms == pytest.approx(1_500.0)
    assert record.latency_ms == pytest.approx(3_000.0)
    assert record.tpot_ms == pytest.approx(1_500.0)
    assert record.stream_chunk_count == 2


def test_stream_state_rejects_missing_usage() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")
    state = bench.StreamResponseState(started_at=10.0)
    state.observe({"choices": [{"text": "a"}]}, observed_at=11.0)
    state.finish(observed_at=12.0)

    with pytest.raises(ValueError, match="stream_usage_missing"):
        state.to_record(
            request_index=0,
            status_code=200,
            expected_prompt_tokens=8_000,
            expected_completion_tokens=1,
        )


def test_stream_state_rejects_completion_token_mismatch() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")
    state = bench.StreamResponseState(started_at=10.0)
    state.observe({"choices": [{"text": "a"}]}, observed_at=11.0)
    state.observe(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8_000,
                "completion_tokens": 1,
                "total_tokens": 8_001,
            },
        },
        observed_at=11.1,
    )
    state.finish(observed_at=12.0)

    with pytest.raises(ValueError, match="completion_tokens_mismatch"):
        state.to_record(
            request_index=0,
            status_code=200,
            expected_prompt_tokens=8_000,
            expected_completion_tokens=2,
        )


def test_parse_sse_line_distinguishes_data_done_and_keepalive() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")

    assert bench._parse_sse_line(b"\n") is None
    assert bench._parse_sse_line(b": keepalive\n") is None
    assert bench._parse_sse_line(b"data: [DONE]\n") is bench.STREAM_DONE
    assert bench._parse_sse_line(b'data: {"choices":[{"text":"x"}]}\n') == {
        "choices": [{"text": "x"}]
    }


class FakeContent:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = iter(lines)

    async def readline(self) -> bytes:
        return next(self._lines, b"")


class FakeResponse:
    status = 200

    def __init__(self, lines: list[bytes]) -> None:
        self.content = FakeContent(lines)

    async def text(self) -> str:
        return ""


class FakeRequestContext:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeSession:
    def __init__(self, lines: list[bytes]) -> None:
        self._response = FakeResponse(lines)

    def post(self, endpoint, json, headers=None):
        return FakeRequestContext(self._response)


def test_post_streaming_completion_builds_request_record(monkeypatch) -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")
    usage = {
        "prompt_tokens": 8_000,
        "completion_tokens": 2,
        "total_tokens": 8_002,
    }
    lines = [
        b'data: {"choices":[{"text":"a"}]}\n',
        f"data: {json.dumps({'choices': [], 'usage': usage})}\n".encode(),
        b"data: [DONE]\n",
    ]
    ticks = iter([10.0, 11.0, 11.1, 12.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))

    record = asyncio.run(
        bench._post_streaming_completion(
            FakeSession(lines),
            "http://example/v1/completions",
            {"stream": True},
            request_index=3,
            expected_prompt_tokens=8_000,
            expected_completion_tokens=2,
        )
    )

    assert record.ok == 1
    assert record.ttft_ms == pytest.approx(1_000.0)
    assert record.latency_ms == pytest.approx(2_000.0)
    assert record.tpot_ms == pytest.approx(1_000.0)


def test_stream_metric_summary_reports_all_percentiles() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")
    records = [
        bench.RequestRecord(0, 1, 200, 100.0, 8, 2, 10, "", 10.0, 45.0, 2),
        bench.RequestRecord(1, 1, 200, 200.0, 8, 2, 10, "", 20.0, 90.0, 2),
    ]

    summary = bench._stream_metric_summary(records)

    assert summary == pytest.approx({
        "mean_ttft_ms": 15.0,
        "p50_ttft_ms": 15.0,
        "p90_ttft_ms": 19.0,
        "p99_ttft_ms": 19.9,
        "mean_tpot_ms": 67.5,
        "p50_tpot_ms": 67.5,
        "p90_tpot_ms": 85.5,
        "p99_tpot_ms": 89.55,
    })


def test_stream_cli_can_skip_prompt_token_id_probe() -> None:
    sys.modules.setdefault("aiohttp", types.SimpleNamespace())
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=object()))
    bench = importlib.import_module("scripts.run_openai_fixed_shape_benchmark")

    args = bench._build_parser().parse_args(
        [
            "--tokenizer",
            "/model",
            "--input-len",
            "8000",
            "--output-len",
            "2000",
            "--result-json",
            "/tmp/result.json",
            "--stream",
            "--skip-prompt-token-id-probe",
        ]
    )

    assert args.stream is True
    assert args.skip_prompt_token_id_probe is True
    assert args.result_json == Path("/tmp/result.json")
