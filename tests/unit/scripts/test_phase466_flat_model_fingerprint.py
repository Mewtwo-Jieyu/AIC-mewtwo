from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "run_phase466_low_overhead_probe.py"
FIXED_MTIME_NS = 1_700_000_000_000_000_000


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_safetensors(
    path: Path,
    *,
    header: dict[str, object],
    payload: bytes = b"payload",
) -> None:
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    os.utime(path, ns=(FIXED_MTIME_NS, FIXED_MTIME_NS))


def _make_flat_model(root: Path, *, reverse_creation: bool = False) -> Path:
    root.mkdir()
    files = [
        (".msc", b"modelscope-state"),
        (".mv", b"Revision:master,CreatedAt:1\n"),
        ("config.json", b'{"model_type":"kimi_k25"}\n'),
        ("tokenizer_config.json", b'{"tokenizer_class":"KimiTokenizer"}\n'),
        ("modeling_kimi.py", b"MODEL = 'kimi'\n"),
        ("chat_template.jinja", b"{{ messages }}\n"),
        ("tiktoken.model", b"tokenizer-model"),
    ]
    for name, content in reversed(files) if reverse_creation else files:
        (root / name).write_bytes(content)
    index = {
        "metadata": {"total_size": 14},
        "weight_map": {
            "layer.0": "model-00001-of-000002.safetensors",
            "layer.1": "model-00002-of-000002.safetensors",
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, sort_keys=True), encoding="utf-8"
    )
    shard_specs = [
        (
            "model-00001-of-000002.safetensors",
            {"layer.0": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
        ),
        (
            "model-00002-of-000002.safetensors",
            {"layer.1": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
        ),
    ]
    for name, header in reversed(shard_specs) if reverse_creation else shard_specs:
        _write_safetensors(root / name, header=header)
    return root


def test_flat_model_fingerprint_is_stable_and_order_independent(tmp_path: Path) -> None:
    runner = _load("phase466_flat_fingerprint_stable")
    first = runner.fingerprint_flat_model(_make_flat_model(tmp_path / "first"))
    second = runner.fingerprint_flat_model(
        _make_flat_model(tmp_path / "second", reverse_creation=True)
    )

    assert first["schema"] == "phase466_flat_model_fingerprint_v1"
    assert first["identity_scope"] == "same_flat_mirror_instance"
    assert first["fingerprint_sha256"] == second["fingerprint_sha256"]
    assert [item["path"] for item in first["shards"]] == [
        "model-00001-of-000002.safetensors",
        "model-00002-of-000002.safetensors",
    ]
    assert {item["path"] for item in first["runtime_files"]} == {
        "chat_template.jinja",
        "config.json",
        "model.safetensors.index.json",
        "modeling_kimi.py",
        "tiktoken.model",
        "tokenizer_config.json",
    }


@pytest.mark.parametrize(
    "mutation",
    ["runtime", "msc", "mv", "header", "size", "mtime"],
)
def test_flat_model_fingerprint_captures_all_identity_changes(
    tmp_path: Path, mutation: str
) -> None:
    runner = _load(f"phase466_flat_fingerprint_change_{mutation}")
    model = _make_flat_model(tmp_path / "model")
    before = runner.fingerprint_flat_model(model)["fingerprint_sha256"]
    shard = model / "model-00001-of-000002.safetensors"

    if mutation == "runtime":
        (model / "config.json").write_text('{"changed":true}\n', encoding="utf-8")
    elif mutation == "msc":
        (model / ".msc").write_bytes(b"changed-msc")
    elif mutation == "mv":
        (model / ".mv").write_bytes(b"changed-mv")
    elif mutation == "header":
        _write_safetensors(
            shard,
            header={
                "layer.0": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
            },
        )
    elif mutation == "size":
        shard.write_bytes(shard.read_bytes() + b"more")
        os.utime(shard, ns=(FIXED_MTIME_NS, FIXED_MTIME_NS))
    else:
        os.utime(shard, ns=(FIXED_MTIME_NS + 1, FIXED_MTIME_NS + 1))

    after = runner.fingerprint_flat_model(model)["fingerprint_sha256"]
    assert after != before


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing", "flat_model_shards_missing"),
        ("extra", "flat_model_shards_unexpected"),
        ("extra_symlink", "flat_model_shard_not_regular"),
        ("invalid_header", "safetensors_header_invalid"),
        ("wrong_index", "safetensors_index_header_mismatch"),
        ("path_traversal", "safetensors_index_shard_path_invalid"),
        ("duplicate_key", "json_duplicate_key"),
    ],
)
def test_flat_model_fingerprint_rejects_invalid_shard_sets_and_indexes(
    tmp_path: Path, mutation: str, error: str
) -> None:
    runner = _load(f"phase466_flat_fingerprint_invalid_{mutation}")
    model = _make_flat_model(tmp_path / "model")
    if mutation == "missing":
        (model / "model-00002-of-000002.safetensors").unlink()
    elif mutation == "extra":
        _write_safetensors(
            model / "model-00003-of-000002.safetensors",
            header={"extra": {}},
        )
    elif mutation == "extra_symlink":
        (model / "model-00003-of-000002.safetensors").symlink_to(
            model / "model-00001-of-000002.safetensors"
        )
    elif mutation == "invalid_header":
        (model / "model-00001-of-000002.safetensors").write_bytes(
            (9999).to_bytes(8, "little") + b"{}"
        )
    elif mutation == "wrong_index":
        index = json.loads((model / "model.safetensors.index.json").read_text())
        index["weight_map"] = {
            "layer.0": "model-00002-of-000002.safetensors",
            "layer.1": "model-00001-of-000002.safetensors",
        }
        (model / "model.safetensors.index.json").write_text(json.dumps(index))
    elif mutation == "path_traversal":
        index = json.loads((model / "model.safetensors.index.json").read_text())
        index["weight_map"]["layer.0"] = "../outside.safetensors"
        (model / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        (model / "model.safetensors.index.json").write_text(
            '{"weight_map":{"layer.0":"model-00001-of-000002.safetensors",'
            '"layer.0":"model-00002-of-000002.safetensors"}}',
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match=error):
        runner.fingerprint_flat_model(model)
