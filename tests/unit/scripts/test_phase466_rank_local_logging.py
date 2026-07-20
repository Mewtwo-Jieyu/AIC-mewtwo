from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "phase466_rank_local_logging.py"
CONFIG_PATH = REPO_ROOT / "scripts" / "phase466_rank_local_logging.json"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "phase466_rank_local_logging", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(*, process_name: str, message: str, name: str = "vllm.v1.engine.core"):
    record = logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.processName = process_name
    record.process = 1200 if process_name.endswith("0") else 1201
    return record


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_handler_writes_dp_ranks_to_separate_files(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    monkeypatch.setenv("PHASE466_RANK_LOG_DIR", str(tmp_path))
    handler = module.RankLocalIterationHandler()

    handler.emit(_record(process_name="EngineCore_DP0", message="Iteration(7): x"))
    handler.emit(_record(process_name="EngineCore_DP1", message="Iteration(8): y"))
    handler.close()

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "rank-0.jsonl",
        "rank-1.jsonl",
    ]
    assert _read_jsonl(tmp_path / "rank-0.jsonl") == [
        {
            "rank": 0,
            "pid": 1200,
            "process_name": "EngineCore_DP0",
            "message": "Iteration(7): x",
        }
    ]
    assert _read_jsonl(tmp_path / "rank-1.jsonl") == [
        {
            "rank": 1,
            "pid": 1201,
            "process_name": "EngineCore_DP1",
            "message": "Iteration(8): y",
        }
    ]


def test_handler_maps_single_dp_engine_core_to_rank_zero(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    monkeypatch.setenv("PHASE466_RANK_LOG_DIR", str(tmp_path))
    handler = module.RankLocalIterationHandler()

    handler.emit(_record(process_name="EngineCore", message="Iteration(1): x"))
    handler.close()

    row = _read_jsonl(tmp_path / "rank-0.jsonl")[0]
    assert row["rank"] == 0
    assert row["process_name"] == "EngineCore"


def test_handler_ignores_non_iteration_and_unresolved_rank_records(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    monkeypatch.setenv("PHASE466_RANK_LOG_DIR", str(tmp_path))
    handler = module.RankLocalIterationHandler()

    handler.emit(_record(process_name="EngineCoreWorker", message="Iteration(1): x"))
    handler.emit(_record(process_name="EngineCore_DP0", message="Starting engine"))
    handler.emit(
        _record(
            process_name="EngineCore_DP0",
            message="Iteration(1): x",
            name="vllm.v1.core.sched.scheduler",
        )
    )
    handler.close()

    assert list(tmp_path.iterdir()) == []


def test_console_filter_suppresses_only_rank_local_iteration_records() -> None:
    module = _load_module()
    record_filter = module.NonIterationConsoleFilter()

    assert not record_filter.filter(
        _record(process_name="EngineCore_DP0", message="Iteration(2): x")
    )
    assert record_filter.filter(
        _record(process_name="EngineCore_DP0", message="Starting engine")
    )
    assert record_filter.filter(
        _record(process_name="EngineCoreWorker", message="Iteration(2): x")
    )


def test_logging_config_uses_rank_handler_and_console_filter() -> None:
    config = json.loads(CONFIG_PATH.read_text())

    assert config["handlers"]["rank_local"]["()"] == (
        "phase466_rank_local_logging.RankLocalIterationHandler"
    )
    assert config["handlers"]["console"]["filters"] == ["non_iteration"]
    assert config["loggers"]["vllm"]["handlers"] == ["console", "rank_local"]


def test_handler_requires_rank_dir_only_when_iteration_is_emitted(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.delenv("PHASE466_RANK_LOG_DIR", raising=False)

    handler = module.RankLocalIterationHandler()
    with pytest.raises(RuntimeError, match="PHASE466_RANK_LOG_DIR is required"):
        handler.emit(
            _record(process_name="EngineCore_DP0", message="Iteration(1): x")
        )
