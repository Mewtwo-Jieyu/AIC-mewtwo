import importlib
import sys
import types


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
