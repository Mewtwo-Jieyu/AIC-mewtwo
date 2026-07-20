"""Rank-local transport for Phase466 vLLM iteration records."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TextIO


LOGGER_NAME = "vllm.v1.engine.core"
PROCESS_NAME_RE = re.compile(r"^EngineCore(?:_DP(?P<rank>\d+))?$")
ITERATION_RE = re.compile(r"^Iteration\(\d+\):\s+")


def _record_rank(record: logging.LogRecord) -> int | None:
    if record.name != LOGGER_NAME:
        return None
    match = PROCESS_NAME_RE.fullmatch(record.processName)
    if match is None or ITERATION_RE.match(record.getMessage()) is None:
        return None
    rank = match.group("rank")
    return int(rank) if rank is not None else 0


class NonIterationConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _record_rank(record) is None


class RankLocalIterationHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        rank_log_dir = os.environ.get("PHASE466_RANK_LOG_DIR")
        self._streams: dict[int, TextIO] = {}
        self._rank_log_dir = Path(rank_log_dir) if rank_log_dir else None

    def emit(self, record: logging.LogRecord) -> None:
        rank = _record_rank(record)
        if rank is None:
            return
        if self._rank_log_dir is None:
            rank_log_dir = os.environ.get("PHASE466_RANK_LOG_DIR")
            if not rank_log_dir:
                raise RuntimeError("PHASE466_RANK_LOG_DIR is required")
            self._rank_log_dir = Path(rank_log_dir)
        stream = self._streams.get(rank)
        if stream is None:
            self._rank_log_dir.mkdir(parents=True, exist_ok=True)
            stream = (self._rank_log_dir / f"rank-{rank}.jsonl").open(
                "a", encoding="utf-8"
            )
            self._streams[rank] = stream
        stream.write(
            json.dumps(
                {
                    "rank": rank,
                    "pid": record.process,
                    "process_name": record.processName,
                    "message": record.getMessage(),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        stream.flush()

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()
        super().close()
