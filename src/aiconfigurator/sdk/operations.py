# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Literal, Optional

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError, PerfDatabase
from aiconfigurator.sdk.perf_source import PerfSourceRecord
from aiconfigurator.sdk.performance_result import PerformanceResult

logger = logging.getLogger(__name__)

_VLLM_MODULE_RUNTIME_MODEL = "moonshotai/Kimi-K2.5"
_VLLM_MODULE_PERFDB_MODEL = "kimi-k2.5"
_VLLM_MODULE_HARDWARE = "h200_sxm"
_VLLM_MODULE_VERSION = "0.19.0"
_VLLM_MODULE_TOPOLOGY = "tp4dp2ep8"
_VLLM_MODULE_QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"
_VLLM_MODULE_FUSEDMOE_BUCKETS = frozenset(
    {1, 2, 15, 16, 30, 32, 241, 482, 1808, 2048, 3616, 4096, 8192, 16384}
)
_VLLM_MODULE_EP8_COMM_BUCKETS = frozenset({1, 15, 16, 241, 1808, 2048, 8192})
_VLLM_MODULE_BUCKETS = _VLLM_MODULE_FUSEDMOE_BUCKETS | _VLLM_MODULE_EP8_COMM_BUCKETS
_VLLM_MODULE_BUCKETS_BY_BOUNDARY = {
    "fusedmoe_runner_compute": _VLLM_MODULE_FUSEDMOE_BUCKETS,
    "ep8_comm_dispatch_combine": _VLLM_MODULE_EP8_COMM_BUCKETS,
}


def _scale_result(result: PerformanceResult, scale_factor: float) -> PerformanceResult:
    if isinstance(result, PerformanceResult):
        scaled = result * scale_factor
    else:
        scaled = PerformanceResult(
            float(result) * scale_factor,
            energy=getattr(result, "energy", 0.0) * scale_factor,
            sources=tuple(
                source.with_transform("multiply", scale_factor)
                for source in getattr(result, "sources", ())
            ),
        )
    if hasattr(result, "provenance"):
        scaled.provenance = result.provenance
    return scaled


def _validate_vllm_module_bucket(bucket_tokens: int) -> None:
    if bucket_tokens not in _VLLM_MODULE_BUCKETS:
        raise ValueError(f"bucket_tokens must be one of {sorted(_VLLM_MODULE_BUCKETS)}, got {bucket_tokens}")


def _has_vllm_module_exact_bucket(module_boundary: str, bucket_tokens: int) -> bool:
    return bucket_tokens in _VLLM_MODULE_BUCKETS_BY_BOUNDARY[module_boundary]


def _is_vllm_module_scope(database: PerfDatabase, model_name: str, topology: str) -> bool:
    return (
        getattr(database, "backend", None) == common.BackendName.vllm.value
        and getattr(database, "system", None) == _VLLM_MODULE_HARDWARE
        and getattr(database, "version", None) == _VLLM_MODULE_VERSION
        and model_name == _VLLM_MODULE_RUNTIME_MODEL
        and topology == _VLLM_MODULE_TOPOLOGY
    )


def _require_vllm_module_topology(database: PerfDatabase, model_name: str, topology: str) -> None:
    if (
        getattr(database, "backend", None) == common.BackendName.vllm.value
        and getattr(database, "system", None) == _VLLM_MODULE_HARDWARE
        and getattr(database, "version", None) == _VLLM_MODULE_VERSION
        and model_name == _VLLM_MODULE_RUNTIME_MODEL
        and not topology
    ):
        raise ValueError("Kimi vLLM module binding requires vllm_module_topology metadata")


def _query_vllm_module(
    database: PerfDatabase,
    *,
    bucket_tokens: int,
    module_boundary: str,
    scale_factor: float,
) -> PerformanceResult:
    if getattr(database, "system", None) != _VLLM_MODULE_HARDWARE:
        raise ValueError(
            f"vLLM module perf runtime binding requires system={_VLLM_MODULE_HARDWARE!r}, "
            f"got {getattr(database, 'system', None)!r}"
        )
    if getattr(database, "version", None) != _VLLM_MODULE_VERSION:
        raise ValueError(
            f"vLLM module perf runtime binding requires version={_VLLM_MODULE_VERSION!r}, "
            f"got {getattr(database, 'version', None)!r}"
        )
    _validate_vllm_module_bucket(bucket_tokens)

    result = database.query_vllm_module(
        model=_VLLM_MODULE_PERFDB_MODEL,
        hardware=database.system,
        vllm_version=database.version,
        topology=_VLLM_MODULE_TOPOLOGY,
        bucket_tokens=bucket_tokens,
        module_boundary=module_boundary,
        quant_runtime=_VLLM_MODULE_QUANT_RUNTIME,
    )
    scaled = _scale_result(result, scale_factor)
    scaled.provenance = "vllm_module_measured"
    return scaled


def _query_vllm_ep8_alltoall(
    database: PerfDatabase,
    *,
    bucket_tokens: int,
    hidden_size: int,
    topk: int,
    scale_factor: float,
    source: Literal["measured", "structural"],
) -> PerformanceResult:
    if source == "measured":
        result = database.query_vllm_ep8_a2a_decode(
            bucket_tokens=bucket_tokens,
            hidden_size=hidden_size,
            topk=topk,
            moe_ep_size=8,
        )
        scaled = _scale_result(result, scale_factor)
        scaled.provenance = "phase431_ep8_a2a_measured"
        return scaled
    if source != "structural":
        raise ValueError(f"unknown vLLM EP8 source: {source!r}")

    node_spec = getattr(database, "system_spec", {}).get("node", {})
    if "intra_node_bw" not in node_spec:
        raise PerfDataNotAvailableError("structural vLLM EP8 source requires node.intra_node_bw metadata")
    # h200_sxm records single-direction intra-node bandwidth; dispatch+combine can use both directions.
    bidirectional_bw = float(node_spec["intra_node_bw"]) * 2.0
    if bidirectional_bw <= 0.0:
        raise ValueError("structural vLLM EP8 source requires positive node.intra_node_bw")
    bytes_per_token = hidden_size * topk * common.CommQuantMode.half.value.memory * 2.0
    latency_ms = bucket_tokens * bytes_per_token / bidirectional_bw * 1000.0
    result = PerformanceResult(
        latency_ms,
        energy=0.0,
        sources=(
            PerfSourceRecord.structural(
                formula_id="vllm_ep8_bidirectional_bandwidth_v1",
                formula_inputs={
                    "bucket_tokens": bucket_tokens,
                    "hidden_size": hidden_size,
                    "topk": topk,
                    "bytes_per_element": common.CommQuantMode.half.value.memory,
                    "directions": 2.0,
                    "bidirectional_bw": bidirectional_bw,
                },
            ),
        ),
    )
    result = _scale_result(result, scale_factor)
    result.provenance = "structural_bidirectional_bandwidth"
    return result


def _resolve_vllm_ep8_source(
    database: PerfDatabase,
    *,
    bucket_tokens: int,
    hidden_size: int,
    topk: int,
) -> Literal["measured", "structural"]:
    coverage_query = getattr(database, "get_vllm_ep8_a2a_decode_coverage", None)
    if not callable(coverage_query):
        raise PerfDataNotAvailableError("vLLM EP8 source selection requires explicit measured coverage metadata")
    bucket_min, bucket_max = coverage_query(hidden_size=hidden_size, topk=topk, moe_ep_size=8)
    return "measured" if bucket_min <= bucket_tokens <= bucket_max else "structural"


class Operation:
    """
    Base operation class.

    Note: query() now returns PerformanceResult (float-like) instead of plain float.
    This maintains backward compatibility while adding power data.
    """

    def __init__(self, name: str, scale_factor: float) -> None:
        self._name = name
        self._scale_factor = scale_factor

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query operation latency with power data.

        Returns:
            PerformanceResult: PerformanceResult (behaves like float) with latency in milliseconds
                   (scaled by scale_factor). Power data available via .power attribute.
        """
        raise NotImplementedError

    def get_weights(self, **kwargs):
        raise NotImplementedError


class CustomAllReduce(Operation):
    """
    Custom AllReduce operation with power tracking.
    """

    def __init__(self, name: str, scale_factor: float, h: int, tp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._h = h
        self._tp_size = tp_size
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query custom allreduce latency with power data."""
        if self._tp_size == 1:
            return PerformanceResult(0.0, 0.0)
        # count, not size in bytes
        size = kwargs.get("x") * self._h

        result = database.query_custom_allreduce(common.CommQuantMode.half, self._tp_size, size)
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class P2P(Operation):
    """
    P2P operation with power tracking.
    """

    def __init__(self, name: str, scale_factor: float, h: int, pp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._h = h
        self._pp_size = pp_size
        self._bytes_per_element = 2
        # self._empirical_scaling_factor = 1.1
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query P2P latency with power data."""
        if self._pp_size == 1:
            return PerformanceResult(0.0, 0.0)

        size = kwargs.get("x") * self._h
        p2p_bytes = size * 2

        result = database.query_p2p(p2p_bytes)
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class NCCL(Operation):
    """
    NCCL operation with power tracking.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        nccl_op: str,
        num_elements_per_token: int,
        num_gpus: int,
        comm_quant_mode: common.CommQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._nccl_op = nccl_op
        self._num_elements_per_token = num_elements_per_token
        self._num_gpus = num_gpus
        self._comm_quant_mode = comm_quant_mode
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query NCCL latency with power data."""
        message_size = kwargs.get("x") * self._num_elements_per_token

        result = database.query_nccl(self._comm_quant_mode, self._num_gpus, self._nccl_op, message_size)
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GEMM(Operation):
    """
    GEMM operation with power tracking.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._k = k
        self._quant_mode = quant_mode
        self._weights = self._n * self._k * quant_mode.value.memory
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)
        self._low_precision_input = kwargs.get("low_precision_input", False)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query GEMM latency with energy data.

        For `fp8_static` quant mode, subtracts compute_scale overhead.
        For GEMMs marked as low-precision input under `fp8_static`, also subtract scale_matrix.

        Returns:
            PerformanceResult: Behaves like float (scaled latency in ms).
                              Energy data accessible via .energy attribute.
                              Power can be derived as energy/latency.
        """
        x = kwargs.get("x")
        x //= self._scale_num_tokens
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode
        model_name = str(kwargs.get("model_name", ""))
        is_fp8_static = quant_mode == common.GEMMQuantMode.fp8_static

        # Query with energy
        result = database.query_gemm(x, self._n, self._k, quant_mode)
        latency = float(result)
        energy = result.energy
        sources = list(result.sources)

        # Adjust for fp8_static: subtract compute_scale overhead, only fix for trtllm now
        if is_fp8_static:
            compute_scale_result = database.query_compute_scale(x, self._k, quant_mode)
            latency -= float(compute_scale_result)
            energy -= compute_scale_result.energy
            sources.extend(getattr(compute_scale_result, "sources", ()))
            scale_matrix_latency = 0.0
            scale_matrix_energy = 0.0
            if self._low_precision_input:
                scale_matrix_result = database.query_scale_matrix(x, self._k, quant_mode)
                latency -= float(scale_matrix_result)
                energy -= scale_matrix_result.energy
                scale_matrix_latency = float(scale_matrix_result)
                scale_matrix_energy = scale_matrix_result.energy
                sources.extend(getattr(scale_matrix_result, "sources", ()))
            sources.append(
                PerfSourceRecord.structural(
                    formula_id="gemm_fp8_scale_overhead_subtraction_v1",
                    formula_inputs={
                        "base_latency_ms": float(result),
                        "base_energy_wms": result.energy,
                        "compute_scale_latency_ms": float(compute_scale_result),
                        "compute_scale_energy_wms": compute_scale_result.energy,
                        "scale_matrix_latency_ms": scale_matrix_latency,
                        "scale_matrix_energy_wms": scale_matrix_energy,
                    },
                )
            )

        # Ensure non-negative latency and energy
        latency_clamped = max(0.0, latency)
        energy_clamped = max(0.0, energy)
        if latency_clamped != latency or energy_clamped != energy:
            logger.warning(
                "GEMM.query clamped latency/energy to 0.0. "
                "op=%s model=%s m=%s n=%s k=%s quant_mode=%s post_sub(lat=%.6f, eng=%.6f)",
                self._name,
                model_name,
                x,
                self._n,
                self._k,
                quant_mode.name,
                latency,
                energy,
            )
            sources.append(
                PerfSourceRecord.structural(
                    formula_id="gemm_nonnegative_clamp_v1",
                    formula_inputs={
                        "input_latency_ms": latency,
                        "input_energy_wms": energy,
                        "output_latency_ms": latency_clamped,
                        "output_energy_wms": energy_clamped,
                    },
                )
            )

        latency = latency_clamped
        energy = energy_clamped

        return _scale_result(PerformanceResult(latency=latency, energy=energy, sources=sources), self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class TrtLLMWideEPMoE(Operation):
    """
    TensorRT-LLM WideEP MoE operation with configurable EPLB modes.

    This class is specifically designed for TensorRT-LLM backend's WideEP MoE computation.
    It handles the pure computation aspect of MoE, excluding All2All communication which
    is handled by TrtLLMWideEPMoEDispatch.

    Supports three EPLB modes:
    - EPLB off: workload_distribution without "_eplb" suffix, num_slots = num_experts
    - EPLB on: workload_distribution with "_eplb" suffix, num_slots = num_experts
    - EPLB redundant: workload_distribution with "_eplb" suffix, num_slots > num_experts

    Args:
        name: Operation name
        scale_factor: Scaling factor for the operation
        hidden_size: Hidden dimension size
        inter_size: Intermediate dimension size
        topk: Number of top experts to select
        num_experts: Total number of experts
        num_slots: Number of expert slots (= num_experts for EPLB off/on, > num_experts for redundant)
        moe_tp_size: MoE tensor parallelism size
        moe_ep_size: MoE expert parallelism size
        quant_mode: Quantization mode for MoE computation
        workload_distribution: Workload distribution pattern (e.g., "power_law_1.01" or "power_law_1.01_eplb")
        attention_dp_size: Attention data parallelism size (scales input tokens)
        is_gated: Whether MoE uses gated activation (default: True)
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        attention_dp_size: int,
        num_slots: Optional[int] = None,  # EPLB slots, defaults to num_experts
        is_gated: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._quant_mode = quant_mode
        self._topk = topk
        self._num_experts = num_experts
        self._num_slots = num_slots if num_slots is not None else num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._workload_distribution = workload_distribution
        self._is_gated = is_gated

        # Calculate weights: 3 GEMMs for gated (gate, up, down), 2 GEMMs for non-gated (up, down)
        num_gemms = 3 if is_gated else 2
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            * num_gemms
            // self._moe_ep_size
            // self._moe_tp_size
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query TrtLLM WideEP MoE compute latency with energy data.

        Supports three EPLB modes based on workload_distribution and num_slots:
        - EPLB off: distribution without "_eplb" suffix, num_slots = num_experts
        - EPLB on: distribution with "_eplb" suffix, num_slots = num_experts
        - EPLB redundant: distribution with "_eplb" suffix, num_slots > num_experts

        Args:
            database: Performance database instance
            **kwargs: Additional arguments including:
                - x: Number of input tokens (will be scaled by attention_dp_size)
                - quant_mode: Optional override for quantization mode

        Returns:
            PerformanceResult with latency and energy data
        """
        # Scale input tokens by attention_dp_size
        x = kwargs.get("x") * self._attention_dp_size
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode

        logger.debug(f"TrtLLMWideEPMoE: Querying compute with num_slots={self._num_slots}")

        # Query WideEP MoE compute performance
        result = database.query_wideep_moe_compute(
            num_tokens=x,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            num_slots=self._num_slots,
            moe_tp_size=self._moe_tp_size,
            moe_ep_size=self._moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=self._workload_distribution,
        )

        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        """Get the weight memory size for this MoE layer."""
        return self._weights * self._scale_factor


class TrtLLMWideEPMoEDispatch(Operation):
    """
    TensorRT-LLM WideEP MoE dispatch operation using NVLink Two-Sided All2All.

    This class handles WideEP-specific All2All communication for expert parallelism
    in TensorRT-LLM, including prepare, dispatch, and combine phases.

    Communication phases:
    - Pre-dispatch: prepare + dispatch operations
    - Post-dispatch: combine or combine_low_precision operation

    Args:
        name: Operation name
        scale_factor: Scaling factor for the operation
        hidden_size: Hidden dimension size
        topk: Number of top experts to select
        num_experts: Total number of experts
        moe_tp_size: MoE tensor parallelism size
        moe_ep_size: MoE expert parallelism size
        attention_dp_size: Attention data parallelism size
        pre_dispatch: If True, performs prepare+dispatch; if False, performs combine
        quant_mode: Quantization mode for All2All operations (required)
        use_low_precision_combine: If True, uses FP8 optimized combine (default: False)
        node_num: Explicit node count for All2All; None means auto-compute from EP size
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        attention_dp_size: int,
        pre_dispatch: bool,
        quant_mode: common.MoEQuantMode,
        use_low_precision_combine: bool = False,
        node_num: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._pre_dispatch = pre_dispatch
        self._quant_mode = quant_mode
        self._use_low_precision_combine = use_low_precision_combine
        self._node_num = node_num
        self._weights = 0.0  # MoEDispatch has no weight memory
        self.num_gpus = self._moe_ep_size * self._moe_tp_size

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query TrtLLM WideEP All2All communication latency.

        Args:
            database: Performance database instance
            **kwargs: Additional arguments including:
                - x: Number of input tokens

        Returns:
            PerformanceResult with latency (no energy for communication ops)
        """
        num_tokens = kwargs.get("x")

        phase = "Pre-dispatch" if self._pre_dispatch else "Post-dispatch"
        precision = (
            "low-precision combine"
            if self._use_low_precision_combine and not self._pre_dispatch
            else "standard precision"
        )
        logger.debug(f"TrtLLMWideEPMoEDispatch: {phase} with {precision}")

        comm_latency = 0.0

        if self._pre_dispatch:
            # Pre-dispatch phase: prepare + dispatch
            prepare_result = database.query_wideep_alltoall(
                op_name="alltoall_prepare",
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                node_num=self._node_num,
            )
            dispatch_result = database.query_wideep_alltoall(
                op_name="alltoall_dispatch",
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                node_num=self._node_num,
            )
            comm_latency = float(prepare_result) + float(dispatch_result)
        else:
            # Post-dispatch phase: combine or combine_low_precision
            combine_op = "alltoall_combine_low_precision" if self._use_low_precision_combine else "alltoall_combine"
            combine_result = database.query_wideep_alltoall(
                op_name=combine_op,
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                node_num=self._node_num,
            )
            comm_latency = float(combine_result)

        # MoEDispatch returns no energy (communication ops don't track energy)
        return PerformanceResult(comm_latency * self._scale_factor, energy=0.0)

    def get_weights(self, **kwargs):
        """MoE dispatch has no weight memory."""
        return 0.0


class MoE(Operation):
    """
    MoE operation with power tracking.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        attention_dp_size: int,
        is_context: bool = True,
        is_gated: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._quant_mode = quant_mode
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._workload_distribution = workload_distribution
        self._is_context = is_context
        self._is_gated = is_gated
        self._moe_backend = kwargs.get("moe_backend")
        self._enable_eplb = kwargs.get("enable_eplb", False)
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)
        # 3 GEMMs for gated (gate, up, down), 2 GEMMs for non-gated (up, down)
        num_gemms = 3 if is_gated else 2
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            * num_gemms
            // self._moe_ep_size
            // self._moe_tp_size
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MoE latency with energy data."""
        # attention dp size will scale up the total input tokens.
        x = kwargs.get("x")
        context_prefill_tokens = kwargs.get("context_prefill_tokens")
        if self._is_context and context_prefill_tokens is not None:
            x = int(context_prefill_tokens)
        elif self._is_context:
            x = max(1, x // self._scale_num_tokens)
            x *= self._attention_dp_size
        else:
            x *= self._attention_dp_size
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode
        model_name = str(kwargs.get("model_name", ""))
        vllm_module_topology = str(kwargs.get("vllm_module_topology", ""))
        _require_vllm_module_topology(database, model_name, vllm_module_topology)

        if _is_vllm_module_scope(database, model_name, vllm_module_topology) and _has_vllm_module_exact_bucket(
            "fusedmoe_runner_compute",
            x,
        ):
            return _query_vllm_module(
                database,
                bucket_tokens=x,
                module_boundary="fusedmoe_runner_compute",
                scale_factor=self._scale_factor,
            )

        result = database.query_moe(
            num_tokens=x,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            moe_tp_size=self._moe_tp_size,
            moe_ep_size=self._moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=self._workload_distribution,
            is_context=self._is_context,
            moe_backend=self._moe_backend,
            is_gated=self._is_gated,
            enable_eplb=self._enable_eplb,
            model=model_name,
        )

        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# a comm op to deduce the communication cost of MoE
class MoEDispatch(Operation):
    """
    MoE dispatch operation. For fine grained moe dispatch
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        attention_dp_size: int,
        pre_dispatch: bool,
        enable_fp4_all2all: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._weights = 0.0
        self._enable_fp4_all2all = enable_fp4_all2all
        self._pre_dispatch = pre_dispatch
        self.num_gpus = self._moe_ep_size * self._moe_tp_size
        self._attention_tp_size = moe_tp_size * moe_ep_size // self._attention_dp_size
        self._sms = kwargs.get("sms", 12)
        self._moe_backend = kwargs.get("moe_backend")
        self._is_context = kwargs.get("is_context", True)
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        num_tokens = kwargs.get("x")
        volume = num_tokens * self._hidden_size
        _sm_version = database.system_spec["gpu"]["sm_version"]
        _num_gpus_per_node = database.system_spec["node"]["num_gpus_per_node"]
        _node_num = self.num_gpus / _num_gpus_per_node

        if database.backend == common.BackendName.trtllm.value:
            assert self._attention_tp_size == 1 or self._attention_dp_size == 1, (
                "trtllm does not support TP>1 and DP>1 for attn simultaneously"
            )
            if _sm_version == 100:
                logger.debug("MoEDispatch: In trtllm SM100 execution path")
                if self._pre_dispatch:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        if _num_gpus_per_node == 72 and self.num_gpus > 4:  # to do: nvl72, node per gpu
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "all_reduce", volume
                            )
                        else:
                            comm_latency = database.query_custom_allreduce(
                                common.CommQuantMode.half, self.num_gpus, volume
                            )
                    elif self._attention_dp_size > 1:
                        if self._enable_fp4_all2all:
                            # Calculate all2all communication volume for nvfp4 all2all operation
                            # Volume calculation considers the average case between best and worst
                            # scenarios:
                            # - Best case: volume * 1/4 (all selected experts are in one GPU for
                            #   all tokens)
                            # - Worst case: volume * min(topk, attention_dp_size)/4 (every selected
                            #   expert is in different GPU)
                            # - Final volume: average of best and worst cases, divided by 4 for
                            #   nvfp4 quantization
                            all2all_volume = (
                                volume * (1 + min(self._topk, self._attention_dp_size)) / 2 / 4
                            )  # mean of best and worst
                            # to do: nvfp4 custom all2all
                            all2all_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "alltoall", all2all_volume
                            )
                            all2all_sf_latency = database.query_nccl(
                                common.CommQuantMode.half,
                                self.num_gpus,
                                "alltoall",
                                all2all_volume / 8,
                            )  # volume_scale_factor = 1/8 volume
                            comm_latency = all2all_latency + all2all_sf_latency + 1e-2  # msg size static latency 10us
                        else:
                            all_gather_volume = volume * self._attention_dp_size / 4
                            all_gather_latency = database.query_nccl(
                                common.CommQuantMode.half,
                                self.num_gpus,
                                "all_gather",
                                all_gather_volume,
                            )  # nvfp4 allgather
                            all_gather_sf_latency = database.query_nccl(
                                common.CommQuantMode.half,
                                self.num_gpus,
                                "all_gather",
                                all_gather_volume / 8,
                            )  # volume_scale_factor = 1/8 volume
                            comm_latency = all_gather_latency + all_gather_sf_latency
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        if _num_gpus_per_node == 72 and self.num_gpus > 4:  # to do: nvl72, node per gpu
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "all_reduce", volume
                            )
                        else:
                            comm_latency = database.query_custom_allreduce(
                                common.CommQuantMode.half, self.num_gpus, volume
                            )
                    elif self._attention_dp_size > 1:
                        if self._enable_fp4_all2all:
                            # to do: nvfp4 all2all
                            all2all_volume = (
                                volume * (1 + min(self._topk, self._attention_dp_size)) / 2 / 4
                            )  # nvfp4 all2all
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "alltoall", all2all_volume
                            )
                        else:
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half,
                                self.num_gpus,
                                "reduce_scatter",
                                volume * self._attention_dp_size,
                            )
                    else:
                        comm_latency = 0
            else:  # sm < 100 or > 100 (for now)
                logger.debug("MoEDispatch: In trtllm SM<100 or >100 execution path")
                if self._pre_dispatch:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "all_gather",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
        elif database.backend == common.BackendName.vllm.value:
            assert self._moe_tp_size == 1 or self._moe_ep_size == 1, (
                "vllm does not support MoE TP and MoE EP at the same time"
            )
            context_prefill_tokens = kwargs.get("context_prefill_tokens")
            if self._is_context and context_prefill_tokens is not None:
                scaled_num_tokens = int(context_prefill_tokens)
            else:
                scaled_num_tokens = max(1, num_tokens // self._scale_num_tokens)
            model_name = str(kwargs.get("model_name", ""))
            vllm_module_topology = str(kwargs.get("vllm_module_topology", ""))
            _require_vllm_module_topology(database, model_name, vllm_module_topology)
            if _is_vllm_module_scope(database, model_name, vllm_module_topology) and _has_vllm_module_exact_bucket(
                "ep8_comm_dispatch_combine",
                scaled_num_tokens,
            ):
                if not self._pre_dispatch:
                    result = PerformanceResult(0.0, energy=0.0)
                    result.provenance = "vllm_module_measured_combined_in_pre_dispatch"
                    return result
                return _query_vllm_module(
                    database,
                    bucket_tokens=scaled_num_tokens,
                    module_boundary="ep8_comm_dispatch_combine",
                    scale_factor=self._scale_factor,
                )
            if _is_vllm_module_scope(database, model_name, vllm_module_topology) and self._moe_ep_size > 1:
                source = _resolve_vllm_ep8_source(
                    database,
                    bucket_tokens=scaled_num_tokens,
                    hidden_size=self._hidden_size,
                    topk=self._topk,
                )
                if not self._pre_dispatch:
                    result = PerformanceResult(0.0, energy=0.0)
                    result.provenance = f"{source}_combined_in_pre_dispatch"
                    return result
                return _query_vllm_ep8_alltoall(
                    database,
                    bucket_tokens=scaled_num_tokens,
                    hidden_size=self._hidden_size,
                    topk=self._topk,
                    scale_factor=self._scale_factor,
                    source=source,
                )

            volume = scaled_num_tokens * self._hidden_size
            comm_latency = 0

            if self._attention_tp_size > 1:
                comm_latency += database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)

            if self._attention_dp_size > 1:
                comm_latency += database.query_nccl(
                    common.CommQuantMode.half,
                    self.num_gpus,
                    "all_gather" if self._pre_dispatch else "reduce_scatter",
                    volume * self._attention_dp_size,
                )
        elif database.backend == common.BackendName.sglang.value:
            if self._moe_backend == "deepep_moe":
                logger.debug("MoEDispatch: In SGLang DeepEP execution path")
                num_tokens = num_tokens // self._scale_num_tokens
                if self._is_context:
                    comm_latency = database.query_wideep_deepep_normal(
                        node_num=_node_num,
                        num_tokens=num_tokens,
                        num_experts=self._num_experts,
                        topk=self._topk,
                        hidden_size=self._hidden_size,
                        sms=self._sms,
                    )
                else:
                    comm_latency = database.query_wideep_deepep_ll(
                        node_num=_node_num,
                        num_tokens=num_tokens,
                        num_experts=self._num_experts,
                        topk=self._topk,
                        hidden_size=self._hidden_size,
                    )
            else:
                assert self._attention_tp_size == 1 or self._attention_dp_size == 1, (
                    "We don't enable the path for non-wideep SGLang to support TP>1 and DP>1 for attn simultaneously"
                )
                # TODO: support TP+DP
                logger.debug("MoEDispatch: In SGLang non-DeepEP execution path")
                if self._pre_dispatch:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "all_gather",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
        else:  # other backends
            raise NotImplementedError(f"MoEDispatch: Not implemented for backend {database.backend}")

        if isinstance(comm_latency, PerformanceResult):
            return _scale_result(comm_latency, self._scale_factor)
        return PerformanceResult(comm_latency * self._scale_factor, energy=0.0)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

    def query_ideal(self, database: PerfDatabase, **kwargs):
        """
        Ideal communication cost for MoE dispatch. For reference only.
        """
        num_tokens = kwargs.get("x")
        volume = num_tokens * self._hidden_size

        if self._pre_dispatch:
            reduce_scatter1_v = volume / self.num_gpus
            reduce_scatter1_num_gpus = self._attention_tp_size

            all2all1_v = volume * self._topk / self.num_gpus
            all2all1_num_gpus = self.num_gpus

            allgather1_v = volume / self._moe_tp_size
            allgather1_num_gpus = self._moe_tp_size

            comm_latency = (
                database.query_nccl(
                    common.CommQuantMode.half,
                    reduce_scatter1_num_gpus,
                    "reduce_scatter",
                    reduce_scatter1_v,
                )
                + database.query_nccl(common.CommQuantMode.half, all2all1_num_gpus, "alltoall", all2all1_v)
                + database.query_nccl(common.CommQuantMode.half, allgather1_num_gpus, "all_gather", allgather1_v)
            )
        else:
            reduce_scatter2_v = volume
            reduce_scatter2_num_gpus = self._moe_tp_size

            all2all2_v = volume * self._topk / self.num_gpus
            all2all2_num_gpus = self.num_gpus

            allgather2_v = volume / self.num_gpus
            allgather2_num_gpus = self._attention_tp_size

            comm_latency = (
                database.query_nccl(
                    common.CommQuantMode.half,
                    reduce_scatter2_num_gpus,
                    "reduce_scatter",
                    reduce_scatter2_v,
                )
                + database.query_nccl(common.CommQuantMode.half, all2all2_num_gpus, "alltoall", all2all2_v)
                + database.query_nccl(common.CommQuantMode.half, allgather2_num_gpus, "all_gather", allgather2_v)
            )

        return comm_latency * self._scale_factor


class ContextAttention(Operation):
    """
    Context attention operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        window_size: int = 0,
        head_size: int = 128,
    ) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._window_size = window_size
        self._head_size = head_size

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context attention latency with energy data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        result = database.query_context_attention(
            batch_size,
            isl,
            prefix,
            self._n,
            self._n_kv,
            self._kvcache_quant_mode,
            self._fmha_quant_mode,
            window_size=self._window_size,
            head_size=self._head_size,
        )
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationAttention(Operation):
    """
    Generation attention operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        window_size: int = 0,
        head_size: int = 128,
    ) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype
        self._window_size = window_size
        self._head_size = head_size

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation attention latency with energy data."""
        beam_width = kwargs.get("beam_width")
        assert beam_width == 1, "only support beam_width=1"
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_attention(
            batch_size,
            s,
            self._n,
            self._n_kv,
            self._kv_cache_dtype,
            window_size=self._window_size,
            head_size=self._head_size,
        )
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class ContextMLA(Operation):
    """
    Context MLA operation. now only contains MHA part.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        # 2*(1536*24576/tp_size + 128/tp_size*512*128+128/tp_size*512*128)
        # up q, up k, up v  float16 # 104MB / tpsize per layer
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context MLA latency with energy data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        result = database.query_context_mla(
            b=batch_size,
            s=isl,
            prefix=prefix,
            num_heads=self._num_heads,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
        )
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationMLA(Operation):
    """
    Generation MLA operation. now only contains MQA part.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        # 2*(1536*24576/tp_size + 128/tp_size*512*128+128/tp_size*512*128)
        # up q, up k, v up float16
        self._weights = 0.0
        self._kv_cache_dtype = kv_cache_dtype

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation MLA latency with energy data."""
        beam_width = kwargs.get("beam_width")
        assert beam_width == 1, "only support beam_width=1"
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_mla(batch_size, s, self._num_heads, self._kv_cache_dtype)
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class MLABmm(Operation):
    """
    MLABmm operation. consider to be contained by mla op. for now, keep it as a separate op to
    show the cost of bmm
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0.0
        self._quant_mode = quant_mode
        self._if_pre = if_pre

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MLA BMM latency with power data."""
        beam_width = kwargs.get("beam_width")
        assert beam_width == 1, "only support beam_width=1"
        batch_size = kwargs.get("batch_size")

        result = database.query_mla_bmm(batch_size, self._num_heads, self._quant_mode, self._if_pre)
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class Embedding(Operation):
    """
    Embedding operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        row_size: int,
        column_size: int,
        empirical_bw_scaling_factor: float = 0.3,
    ) -> None:
        super().__init__(name, scale_factor)
        self._row_size = row_size
        self._column_size = column_size
        self._weights = row_size * column_size * 2
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6  # 5us

    # sol only
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query embedding latency with power data."""
        x = kwargs.get("x")
        d2d_bytes = x * self._column_size * 2

        result = database.query_mem_op(d2d_bytes)
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class ElementWise(Operation):
    """
    Element-wise operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        dim_in: int,
        dim_out: int,
        empirical_bw_scaling_factor: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._weights = 0.0
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6  # 5us
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)

    # sol only
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query element-wise operation latency with power data."""
        x = kwargs.get("x")  # num tokens
        x //= self._scale_num_tokens
        read_bytes = x * self._dim_in * 2  # fp16 for act
        write_bytes = x * self._dim_out * 2

        result = database.query_mem_op(read_bytes + write_bytes)
        return _scale_result(result, self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class WideEPGenerationMLA(Operation):
    """
    WideEP Generation MLA operation.
    This handles the MLA operations in generation/decoding mode.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query WideEP generation MLA latency with power data."""
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_wideep_generation_mla(
            batch_size,
            s,
            self._tp_size,
            self._kvcache_quant_mode,
            self._fmha_quant_mode,
            self._attn_backend,
        )
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class WideEPContextMLA(Operation):
    """
    WideEP Context MLA operation.
    This handles the MLA operations in context/prefill mode.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query WideEP context MLA latency with power data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        result = database.query_wideep_context_mla(
            b=batch_size,
            s=isl,
            prefix=prefix,
            tp_size=self._tp_size,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            attention_backend=self._attn_backend,
        )
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class Mamba2Kernel(Operation):
    """
    Single Mamba2 kernel op (Conv1D or SSM) using collected mamba2_perf data.

    One of four kernels: causal_conv1d_fn, mamba_chunk_scan_combined (context),
    causal_conv1d_update, selective_state_update (generation).
    Uses full (unsharded) dimensions for lookup; collector data is per-layer.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        kernel_source: str,
        phase: str,
        hidden_size: int,
        nheads: int,
        head_dim: int,
        d_state: int,
        d_conv: int,
        n_groups: int,
        chunk_size: int,
    ) -> None:
        super().__init__(name, scale_factor)
        self._kernel_source = kernel_source
        self._phase = phase
        self._hidden_size = hidden_size
        self._nheads = nheads
        self._head_dim = head_dim
        self._d_state = d_state
        self._d_conv = d_conv
        self._n_groups = n_groups
        self._chunk_size = chunk_size
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")
        seq_len = s if self._phase == "context" else None
        result = database.query_mamba2(
            phase=self._phase,
            kernel_source=self._kernel_source,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=self._hidden_size,
            d_state=self._d_state,
            d_conv=self._d_conv,
            nheads=self._nheads,
            head_dim=self._head_dim,
            n_groups=self._n_groups,
            chunk_size=self._chunk_size,
        )
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class Mamba2(Operation):
    """
    Mamba2 operation for NemotronH hybrid models.

    Models the Mamba2Mixer layer which consists of:
    - in_proj: Linear projection (hidden_size -> expanded_size)
    - conv1d: Causal 1D convolution
    - SSM: Selective State Space Model (scan operation)
    - norm: RMSNorm with gating
    - out_proj: Linear projection back to hidden_size

    This is a SOL-based approximation that models:
    - Two GEMMs for in_proj and out_proj
    - Memory operations for conv1d and SSM scan

    The internal state dimension is calculated as:
    expanded_size = 2 * (nheads * head_dim + 2 * n_groups * d_state)
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        nheads: int,
        head_dim: int,
        d_state: int,
        d_conv: int,
        n_groups: int,
        chunk_size: int,
        tp_size: int,
        quant_mode: common.GEMMQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._nheads = nheads
        self._head_dim = head_dim
        self._d_state = d_state
        self._d_conv = d_conv
        self._n_groups = n_groups
        self._chunk_size = chunk_size
        self._tp_size = tp_size
        self._quant_mode = quant_mode

        # Calculate dimensions matching TensorRT-LLM mamba2_mixer.py lines 76-78:
        # d_inner = head_dim * nheads
        # d_in_proj = 2 * d_inner + 2 * n_groups * d_state + nheads
        # conv_dim = d_inner + 2 * n_groups * d_state
        self._d_inner = nheads * head_dim
        self._conv_dim = self._d_inner + 2 * n_groups * d_state
        self._in_proj_out_size = 2 * self._d_inner + 2 * n_groups * d_state + nheads

        # Calculate weights (in_proj + conv1d + out_proj + A + D + dt_bias + norm)
        # in_proj: hidden_size * in_proj_out_size (Linear d_model -> d_in_proj)
        # conv1d: d_conv * conv_dim (Linear d_conv -> conv_dim, stored as Linear for TP)
        # out_proj: d_inner * hidden_size (Linear d_inner -> d_model)
        # A, D, dt_bias: nheads each (small, ignored for weight calculation)
        # norm: d_inner (small, ignored)
        self._weights = (
            (
                hidden_size * self._in_proj_out_size  # in_proj
                + d_conv * self._conv_dim  # conv1d
                + self._d_inner * hidden_size  # out_proj
            )
            * quant_mode.value.memory
            // tp_size
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query Mamba2 latency using SOL-based approximation.

        Models the operation as:
        1. in_proj GEMM: (x, hidden_size) @ (hidden_size, in_proj_out_size)
        2. conv1d: Memory-bound operation
        3. SSM scan: Memory-bound recurrent operation
        4. out_proj GEMM: (x, d_inner) @ (d_inner, hidden_size)
        """
        x = kwargs.get("x")  # num tokens

        # Apply TP sharding (matching TensorRT-LLM mamba2_mixer.py lines 81-84)
        # tp_nheads = nheads // tp_size
        # tp_d_inner = d_inner // tp_size
        # tp_ngroups = n_groups // tp_size
        # tp_conv_dim = conv_dim // tp_size
        nheads_per_gpu = self._nheads // self._tp_size
        d_inner_per_gpu = nheads_per_gpu * self._head_dim
        n_groups_per_gpu = self._n_groups // self._tp_size
        conv_dim_per_gpu = d_inner_per_gpu + 2 * n_groups_per_gpu * self._d_state
        in_proj_out_per_gpu = 2 * d_inner_per_gpu + 2 * n_groups_per_gpu * self._d_state + nheads_per_gpu

        total_latency = 0.0
        total_energy = 0.0

        # 1. in_proj GEMM: hidden_size -> in_proj_out_size
        in_proj_result = database.query_gemm(x, in_proj_out_per_gpu, self._hidden_size, self._quant_mode)
        total_latency += float(in_proj_result)
        total_energy += in_proj_result.energy

        # 2. conv1d: Memory-bound operation on conv_dim (not just d_inner)
        # conv1d operates on xbc which has dimension conv_dim
        # Read: x * conv_dim * d_conv (for conv states) + x * conv_dim (input)
        # Write: x * conv_dim (output)
        conv_read_bytes = x * conv_dim_per_gpu * (self._d_conv + 1) * 2  # fp16
        conv_write_bytes = x * conv_dim_per_gpu * 2
        conv_result = database.query_mem_op(conv_read_bytes + conv_write_bytes)
        total_latency += float(conv_result)
        total_energy += conv_result.energy

        # 3. SSM scan: Memory-bound recurrent operation
        # For prefill (context), uses chunked scan
        # For decode (generation), uses selective_state_update
        # Approximate as memory operation:
        # Read: x * (d_inner + n_groups * d_state * 2 + nheads) for x, B, C, dt
        # Write: x * d_inner for output
        ssm_read_bytes = (
            x
            * (
                d_inner_per_gpu
                + n_groups_per_gpu * self._d_state * 2  # B and C
                + nheads_per_gpu  # dt
            )
            * 2
        )
        ssm_write_bytes = x * d_inner_per_gpu * 2
        ssm_result = database.query_mem_op(ssm_read_bytes + ssm_write_bytes)
        total_latency += float(ssm_result)
        total_energy += ssm_result.energy

        # 4. norm: RMSNormGated on d_inner (TRT-LLM mamba2_mixer.py line 315)
        # Read SSM output, apply norm with gating, write normalized output
        norm_read_bytes = x * d_inner_per_gpu * 2  # fp16
        norm_write_bytes = x * d_inner_per_gpu * 2  # fp16
        norm_result = database.query_mem_op(norm_read_bytes + norm_write_bytes)
        total_latency += float(norm_result)
        total_energy += norm_result.energy

        # 5. out_proj GEMM: d_inner -> hidden_size
        out_proj_result = database.query_gemm(x, self._hidden_size, d_inner_per_gpu, self._quant_mode)
        total_latency += float(out_proj_result)
        total_energy += out_proj_result.energy

        return PerformanceResult(
            latency=total_latency * self._scale_factor,
            energy=total_energy * self._scale_factor,
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
