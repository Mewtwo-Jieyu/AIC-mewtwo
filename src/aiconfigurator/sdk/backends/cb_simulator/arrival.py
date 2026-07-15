"""Measured tokenizer arrival primitives for the CB engine loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Iterable


_RESOURCE_PACKAGE = "aiconfigurator.sdk.backends.cb_simulator"
_RESOURCE_PATH = "resources/tokenizer_primitives.json"


@dataclass(frozen=True)
class TokenizerPrimitive:
    resource_version: int
    model_path: str
    system: str
    backend: str
    backend_version: str
    prompt_tokens: tuple[int, ...]
    min_batch_size: int
    max_batch_size: int
    wait_timeout_ms: float
    workers: int
    intercept_ms: float
    ms_per_prompt_token: float
    ms_per_request: float
    weighted_mape: float
    provenance: str

    def predict_ms(self, total_prompt_tokens: int, batch_size: int) -> float:
        if not self.min_batch_size <= batch_size <= self.max_batch_size:
            raise ValueError(f"unsupported tokenizer batch size: {batch_size}")
        predicted = (
            self.intercept_ms
            + self.ms_per_prompt_token * total_prompt_tokens
            + self.ms_per_request * batch_size
        )
        if predicted <= 0:
            raise ValueError("tokenizer primitive produced non-positive latency")
        return predicted


def _load_primitives() -> tuple[TokenizerPrimitive, ...]:
    resource = files(_RESOURCE_PACKAGE).joinpath(_RESOURCE_PATH)
    data = json.loads(resource.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported tokenizer primitive resource schema")

    primitives = []
    for row in data.get("primitives", []):
        primitive = TokenizerPrimitive(
            resource_version=int(row["resource_version"]),
            model_path=str(row["model_path"]),
            system=str(row["system"]),
            backend=str(row["backend"]),
            backend_version=str(row["backend_version"]),
            prompt_tokens=tuple(int(value) for value in row["support"]["prompt_tokens"]),
            min_batch_size=int(row["support"]["batch_size"][0]),
            max_batch_size=int(row["support"]["batch_size"][1]),
            wait_timeout_ms=float(row["runtime"]["wait_timeout_ms"]),
            workers=int(row["runtime"]["workers"]),
            intercept_ms=float(row["formula"]["intercept_ms"]),
            ms_per_prompt_token=float(row["formula"]["ms_per_prompt_token"]),
            ms_per_request=float(row["formula"]["ms_per_request"]),
            weighted_mape=float(row["weighted_mape"]),
            provenance=str(row["provenance"]),
        )
        if primitive.workers != 1:
            raise ValueError("only the measured single-worker tokenizer is supported")
        if primitive.weighted_mape > 0.10:
            raise ValueError("tokenizer primitive did not pass its WMAPE gate")
        primitives.append(primitive)
    if not primitives:
        raise ValueError("tokenizer primitive resource is empty")
    return tuple(primitives)


def resolve_tokenizer_primitive(
    *,
    model_path: str,
    system: str,
    backend: str,
    version: str,
    prompt_tokens: int,
) -> TokenizerPrimitive | None:
    deployment_matches = [
        primitive
        for primitive in _load_primitives()
        if primitive.model_path == model_path
        and primitive.system == system
        and primitive.backend == backend
        and primitive.backend_version == version
    ]
    if not deployment_matches:
        return None
    if len(deployment_matches) != 1:
        raise ValueError("ambiguous tokenizer primitive deployment match")
    primitive = deployment_matches[0]
    if prompt_tokens not in primitive.prompt_tokens:
        raise ValueError(f"unsupported prompt length: {prompt_tokens}")
    return primitive


@dataclass(frozen=True)
class _PendingRequest:
    request_id: object
    prompt_tokens: int
    arrival_ms: float
    sequence: int


@dataclass(frozen=True)
class _TokenizerBatch:
    requests: tuple[_PendingRequest, ...]
    complete_ms: float


class TokenizerArrivalLayer:
    """One-worker asynchronous tokenizer microbatch queue."""

    def __init__(self, primitive: TokenizerPrimitive) -> None:
        self.primitive = primitive
        self._pending: list[_PendingRequest] = []
        self._inflight: _TokenizerBatch | None = None
        self._ready: list[object] = []
        self._worker_available_ms = 0.0
        self._sequence = 0

    def submit_many(
        self,
        requests: Iterable[tuple[object, int]],
        *,
        now_ms: float,
    ) -> None:
        for request_id, prompt_tokens in requests:
            if prompt_tokens not in self.primitive.prompt_tokens:
                raise ValueError(f"unsupported prompt length: {prompt_tokens}")
            self._pending.append(
                _PendingRequest(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    arrival_ms=now_ms,
                    sequence=self._sequence,
                )
            )
            self._sequence += 1
        self._pending.sort(key=lambda item: (item.arrival_ms, item.sequence))
        self._schedule_next()

    def _schedule_next(self) -> None:
        if self._inflight is not None or not self._pending:
            return

        first_arrival_ms = self._pending[0].arrival_ms
        full_batch_arrival_ms = float("inf")
        if len(self._pending) >= self.primitive.max_batch_size:
            full_batch_arrival_ms = self._pending[
                self.primitive.max_batch_size - 1
            ].arrival_ms
        aggregation_start_ms = max(self._worker_available_ms, first_arrival_ms)
        start_ms = max(
            aggregation_start_ms,
            min(
                full_batch_arrival_ms,
                aggregation_start_ms + self.primitive.wait_timeout_ms,
            ),
        )
        eligible = [request for request in self._pending if request.arrival_ms <= start_ms]
        selected = tuple(eligible[: self.primitive.max_batch_size])
        if not selected:
            raise RuntimeError("tokenizer batch has no eligible requests")
        selected_sequences = {request.sequence for request in selected}
        self._pending = [
            request
            for request in self._pending
            if request.sequence not in selected_sequences
        ]
        total_prompt_tokens = sum(request.prompt_tokens for request in selected)
        service_ms = self.primitive.predict_ms(total_prompt_tokens, len(selected))
        complete_ms = start_ms + service_ms
        self._worker_available_ms = complete_ms
        self._inflight = _TokenizerBatch(selected, complete_ms)

    def _advance(self, now_ms: float) -> None:
        while self._inflight is not None and self._inflight.complete_ms <= now_ms:
            self._ready.extend(
                request.request_id for request in self._inflight.requests
            )
            self._inflight = None
            self._schedule_next()

    def drain(self, now_ms: float) -> list[object]:
        self._advance(now_ms)
        ready = self._ready
        self._ready = []
        return ready

    def next_event_ms(self) -> float | None:
        self._schedule_next()
        if self._inflight is None:
            return None
        return self._inflight.complete_ms
