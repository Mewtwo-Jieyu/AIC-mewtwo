# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed provenance records for performance estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence


_SOURCE_TYPES = frozenset({"measured_exact", "measured_interp", "structural", "calibrated"})
_TRANSFORM_OPERATIONS = frozenset({"multiply", "divide"})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("performance source values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported performance source value: {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if value is None:
        return ()
    return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))


def _mapping_dict(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(value)


def _thaw_value(value: Any) -> Any:
    if not isinstance(value, tuple):
        return value
    if value and all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        for item in value
    ):
        return {key: _thaw_value(item) for key, item in value}
    return [_thaw_value(item) for item in value]


def _thaw_mapping(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {key: _thaw_value(item) for key, item in value}


@dataclass(frozen=True)
class PerfSourceTransform:
    """A numeric transform applied after a source value was obtained."""

    operation: str
    factor: float

    def __post_init__(self) -> None:
        if self.operation not in _TRANSFORM_OPERATIONS:
            raise ValueError(f"unknown performance source transform: {self.operation!r}")
        if not math.isfinite(float(self.factor)):
            raise ValueError("performance source transform factor must be finite")
        if self.operation == "divide" and float(self.factor) == 0.0:
            raise ValueError("performance source divide transform cannot use zero")


@dataclass(frozen=True)
class PerfSourceRecord:
    """One measured, structural, or calibrated performance source."""

    source_type: str
    source_id: str
    data_file_sha256: str | None = None
    query_key: tuple[tuple[str, Any], ...] = ()
    row_key: tuple[tuple[str, Any], ...] = ()
    domain: tuple[tuple[str, Any, Any], ...] = ()
    support_keys: tuple[tuple[tuple[str, Any], ...], ...] = ()
    formula_id: str | None = None
    formula_inputs: tuple[tuple[str, Any], ...] = ()
    original_source: str | None = None
    anchor_id: str | None = None
    scale: float | None = None
    scope: tuple[tuple[str, Any], ...] = ()
    approved: bool = True
    transforms: tuple[PerfSourceTransform, ...] = ()

    def __post_init__(self) -> None:
        if self.source_type not in _SOURCE_TYPES:
            raise ValueError(f"unknown performance source type: {self.source_type!r}")
        if not self.source_id:
            raise ValueError("performance source_id must be non-empty")
        if self.source_type in {"measured_exact", "measured_interp"}:
            if self.data_file_sha256 is None or len(self.data_file_sha256) != 64:
                raise ValueError("measured performance source requires a SHA256 digest")
            int(self.data_file_sha256, 16)
            if not self.query_key:
                raise ValueError("measured performance source requires a query key")
        if self.source_type == "measured_exact" and not self.row_key:
            raise ValueError("measured_exact source requires a row key")
        if self.source_type == "measured_interp":
            if not self.domain or not self.support_keys:
                raise ValueError("measured_interp source requires finite domain and support keys")
            query = _mapping_dict(self.query_key)
            for name, lower, upper in self.domain:
                if lower > upper:
                    raise ValueError(f"invalid interpolation domain [{lower}, {upper}] for {name}")
                if name not in query:
                    raise ValueError(f"interpolation domain field missing from query: {name}")
                value = query[name]
                if not isinstance(value, (int, float)) or not lower <= value <= upper:
                    raise ValueError(
                        f"query field {name}={value!r} is outside measured interpolation domain [{lower}, {upper}]"
                    )
                for frozen_support in self.support_keys:
                    support = _mapping_dict(frozen_support)
                    if name not in support:
                        raise ValueError(f"interpolation support field missing: {name}")
                    support_value = support[name]
                    if not isinstance(support_value, (int, float)) or not lower <= support_value <= upper:
                        raise ValueError(
                            f"support field {name}={support_value!r} is outside measured interpolation domain "
                            f"[{lower}, {upper}]"
                        )
        if self.source_type == "structural":
            if not self.formula_id or not self.formula_inputs:
                raise ValueError("structural source requires formula ID and all formula inputs")
        if self.source_type == "calibrated":
            if not self.original_source or not self.anchor_id or self.scale is None or not self.scope:
                raise ValueError("calibrated source requires origin, anchor, scale, and scope")
            if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
                raise ValueError("calibration scale must be finite and positive")

    @classmethod
    def measured_exact(
        cls,
        *,
        source_id: str,
        data_file_sha256: str,
        query_key: Mapping[str, Any],
        row_key: Mapping[str, Any],
    ) -> PerfSourceRecord:
        return cls(
            source_type="measured_exact",
            source_id=source_id,
            data_file_sha256=data_file_sha256,
            query_key=_freeze_mapping(query_key),
            row_key=_freeze_mapping(row_key),
        )

    @classmethod
    def measured_interp(
        cls,
        *,
        source_id: str,
        data_file_sha256: str,
        query_key: Mapping[str, Any],
        domain: Mapping[str, tuple[int | float, int | float]],
        support_keys: Sequence[Mapping[str, Any]],
    ) -> PerfSourceRecord:
        frozen_domain = tuple(
            sorted((str(name), _freeze_value(bounds[0]), _freeze_value(bounds[1])) for name, bounds in domain.items())
        )
        return cls(
            source_type="measured_interp",
            source_id=source_id,
            data_file_sha256=data_file_sha256,
            query_key=_freeze_mapping(query_key),
            domain=frozen_domain,
            support_keys=tuple(_freeze_mapping(key) for key in support_keys),
        )

    @classmethod
    def structural(
        cls,
        *,
        formula_id: str,
        formula_inputs: Mapping[str, Any],
        approved: bool = True,
        source_id: str | None = None,
    ) -> PerfSourceRecord:
        return cls(
            source_type="structural",
            source_id=source_id or formula_id,
            formula_id=formula_id,
            formula_inputs=_freeze_mapping(formula_inputs),
            approved=approved,
        )

    @classmethod
    def calibrated(
        cls,
        *,
        source_id: str,
        original_source: str,
        anchor_id: str,
        scale: float,
        scope: Mapping[str, Any],
    ) -> PerfSourceRecord:
        return cls(
            source_type="calibrated",
            source_id=source_id,
            original_source=original_source,
            anchor_id=anchor_id,
            scale=float(scale),
            scope=_freeze_mapping(scope),
        )

    def with_transform(self, operation: str, factor: float) -> PerfSourceRecord:
        return replace(
            self,
            transforms=(*self.transforms, PerfSourceTransform(operation, float(factor))),
        )

    def validate_scope(self, actual_scope: Mapping[str, Any]) -> None:
        if self.source_type != "calibrated":
            return
        actual = _mapping_dict(_freeze_mapping(actual_scope))
        mismatches = {
            key: (expected, actual.get(key))
            for key, expected in self.scope
            if actual.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"calibration scope mismatch: {mismatches}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "data_file_sha256": self.data_file_sha256,
            "query_key": _thaw_mapping(self.query_key),
            "row_key": _thaw_mapping(self.row_key),
            "domain": {name: [lower, upper] for name, lower, upper in self.domain},
            "support_keys": [_thaw_mapping(key) for key in self.support_keys],
            "formula_id": self.formula_id,
            "formula_inputs": _thaw_mapping(self.formula_inputs),
            "original_source": self.original_source,
            "anchor_id": self.anchor_id,
            "scale": self.scale,
            "scope": _thaw_mapping(self.scope),
            "approved": self.approved,
            "transforms": [
                {"operation": transform.operation, "factor": transform.factor}
                for transform in self.transforms
            ],
        }


def validate_performance_result_sources(result: Any, actual_scope: Mapping[str, Any] | None = None) -> None:
    """Apply the strict source contract to a float-like performance result."""

    sources = tuple(getattr(result, "sources", ()))
    if float(result) != 0.0 and not sources:
        raise ValueError("non-zero performance result has no sources")
    for source in sources:
        if not isinstance(source, PerfSourceRecord):
            raise ValueError(f"unknown performance source record: {type(source).__name__}")
        if source.source_type == "structural" and not source.approved:
            raise ValueError(f"unapproved structural formula: {source.formula_id}")
        if source.source_type == "calibrated" and actual_scope is not None:
            source.validate_scope(actual_scope)
