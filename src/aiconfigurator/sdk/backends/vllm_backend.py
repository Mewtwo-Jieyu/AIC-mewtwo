# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


class VLLMBackend(BaseBackend):
    """
    VLLM backend.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self._agg_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
        self._agg_cache_v2: dict[tuple, InferenceSummary] = {}
        self.name = common.BackendName.vllm

    _DEFAULT_METHOD = "batch_sync"
    _CALIBRATED_METHOD = "batch_sync_calibrated"
    _VLLM_MAX_CHUNK_TOKENS: int = 8192
    _CB_SIM_DEFAULT_OVERLAP_FACTOR: float = 0.0
    _CB_SIM_DEFAULT_DECODE_OVERHEAD_MS: float = 0.0
    # phase397m: the 90ms ep8 decode "overhead" was a magic-number placeholder for
    # the unmodeled TP all-reduce communication (K2.5 generation_ar was commented
    # out). phase397l profiling showed real decode comm is ~5.9ms all-reduce (no
    # all2all). Communication is now modeled structurally via the re-enabled
    # CustomAllReduce ops in DeepSeekModel, so this iteration-level constant is 0.
    _CB_SIM_8GPU_EP_DECODE_OVERHEAD_MS: float = 0.0
    _CB_SIM_EP8_DECODE_OVERHEAD_SOURCE_KEY = (
        "phase148_h200_vllm_ep8_all2all_decode_candidate"
    )

    @staticmethod
    def _get_cb_efficiency_factor(b: int, isl: int, osl: int) -> float:
        """Continuous batching efficiency correction factor (v2).

        AIC's batch model assumes synchronous execution of all requests in a batch,
        while vLLM's continuous batching allows pipeline overlap. This factor corrects
        the throughput prediction to account for the scheduling efficiency gain.

        v2: Fitted at tp=16 dp=1, same config as benchmark, to isolate CB
        efficiency from parallelism config differences.

        Fitted from 7 Kimi-K2.5 + H200 benchmark scenarios using model:
            factor = a * (ISL/OSL)^b_exp * bs^c + d

        LOOCV max error: 1.29x (vs uncorrected 5x-84x at tp16dp1).
        See scripts/fit_cb_factor.py for full derivation.

        Calibrated range: ISL=3k-32k, OSL=1k-3k, b=8-256.
        """
        if b <= 1:
            return 1.0
        # Fitted parameters (see scripts/fit_cb_factor.py, v2 same-config)
        a, b_exp, c, d = 0.881800, 1.171579, 0.165407, 2.904901
        isl_osl_ratio = isl / max(osl, 1)
        factor = a * (isl_osl_ratio ** b_exp) * (b ** c) + d
        return max(factor, 1.0)

    @staticmethod
    def _get_ttft_cb_correction(b: int, isl: int, osl: int) -> float:
        """TTFT correction factor for vLLM Continuous Batching mode (B1b).

        AIC's batch model computes TTFT ∝ batch_prefill_time (all b requests
        prefill simultaneously). In vLLM CB, each request starts prefilling
        immediately on arrival, so TTFT ≈ single_request_prefill + queue_wait,
        which is much lower at small-to-moderate concurrency.

        This factor corrects AIC's overestimate:
            corrected_ttft = aic_ttft / ttft_cb_correction

        Formula (log-linear, Model G):
            correction = 2.186 * ln(ISL/OSL) + 2.109 * ln(b) - 1.940

        Fitted from 7 Kimi-K2.5 + H200 SXM benchmark scenarios (tp=16, dp=1).
        LOOCV max error: 1.73x (vs uncorrected 32x resource calculation error).
        See scripts/fit_ttft_cb_factor.py for full derivation.

        Fitted range: ISL=16k-30k, b=4-32 (clean data, see fit_ttft_cb_factor.py).
        ISL 3k-16k is intentional extrapolation — validated on noisy 3k-3k data,
        acceptable for scenario-A use cases but not formally in the calibrated set.
        At low concurrency (b ≤ 4) or ISL/OSL ≈ 1: correction approaches 1.0.
        At very high concurrency (b > 64): queue-wait dominates; correction
          may over-predict — B2 queue-aware modeling is the proper long-term fix.
        """
        if b <= 1:
            return 1.0
        import math
        # Fitted parameters (see scripts/fit_ttft_cb_factor.py, Model G)
        a = 2.186094       # ISL/OSL log coefficient
        b_coeff = 2.108914  # concurrency log coefficient
        c = -1.939795      # constant offset
        isl_osl_ratio = isl / max(osl, 1)
        factor = a * math.log(max(isl_osl_ratio, 0.01)) + b_coeff * math.log(max(b, 1)) + c
        return max(factor, 1.0)

    @staticmethod
    def _should_apply_cb_calibration(
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
    ) -> bool:
        isl = runtime_config.isl
        osl = runtime_config.osl
        b = runtime_config.batch_size

        # ISL lower bound is 3000 to cover scenario-A (3k-3k) as intentional
        # extrapolation beyond the fitted range (16k-30k clean data).
        return (
            model.model_path == "moonshotai/Kimi-K2.5"
            and database.system == "h200_sxm"
            and model.config.tp_size == 16
            and model.config.pp_size == 1
            and model.config.attention_dp_size == 1
            and 3000 <= isl <= 32000
            and 1000 <= osl <= 3000
            and 4 <= b <= 256
        )

    @classmethod
    def _resolve_chunked_prefill_tokens(cls, **kwargs) -> int:
        return int(kwargs.get("chunked_prefill_tokens") or cls._VLLM_MAX_CHUNK_TOKENS)

    @classmethod
    def _uses_cb_sim_ep8_decode_overhead(
        cls,
        model: BaseModel,
        database: PerfDatabase,
    ) -> bool:
        total_gpus = (
            model.config.tp_size
            * model.config.pp_size
            * model.config.attention_dp_size
        )
        return (
            database.system == "h200_sxm"
            and database.backend == common.BackendName.vllm.value
            and total_gpus == 8
            and model.config.moe_ep_size == 8
        )

    @classmethod
    def _get_cb_sim_decode_overhead_ms(
        cls,
        model: BaseModel,
        database: PerfDatabase,
    ) -> float:
        if cls._uses_cb_sim_ep8_decode_overhead(model, database):
            return cls._CB_SIM_8GPU_EP_DECODE_OVERHEAD_MS
        return cls._CB_SIM_DEFAULT_DECODE_OVERHEAD_MS

    @classmethod
    def _get_cb_sim_decode_overhead_source_key(
        cls,
        model: BaseModel,
        database: PerfDatabase,
        overhead_ms: float,
    ) -> str:
        if overhead_ms <= 0.0:
            return "none"
        if cls._uses_cb_sim_ep8_decode_overhead(model, database):
            return cls._CB_SIM_EP8_DECODE_OVERHEAD_SOURCE_KEY
        return "manual_cb_sim_decode_overhead_candidate"

    @staticmethod
    def _make_cb_sim_decode_overhead_shape_key(
        runtime_config: RuntimeConfig,
        ctx_tokens: int,
        cb_config,
    ) -> str:
        phase = "decode_present" if runtime_config.osl > 0 else "prefill_only"
        return (
            f"{phase}:isl{runtime_config.isl}:osl{runtime_config.osl}:"
            f"bs{runtime_config.batch_size}:ctx{ctx_tokens}:"
            f"max_bt{cb_config.max_num_batched_tokens}:max_seqs{cb_config.max_num_seqs}"
        )

    def run_agg(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Run the agg inference. TODO: add vLLM's own implementation
        """
        method = kwargs.get("method", self._DEFAULT_METHOD)
        if method not in {self._DEFAULT_METHOD, self._CALIBRATED_METHOD, "cb_sim"}:
            raise ValueError(f"Unsupported vLLM agg method: {method}")

        # Dispatch to CB simulator if requested
        if method == "cb_sim":
            isl = runtime_config.isl
            osl = runtime_config.osl
            b = runtime_config.batch_size
            ctx_tokens = kwargs.get("ctx_tokens", self._VLLM_MAX_CHUNK_TOKENS)
            cb_config = kwargs.get("cb_config")
            default_overlap_factor = self._CB_SIM_DEFAULT_OVERLAP_FACTOR
            default_overhead_ms = self._get_cb_sim_decode_overhead_ms(
                model,
                database,
            )
            overlap_factor = float(
                getattr(
                    cb_config,
                    "overlap_factor",
                    kwargs.get("overlap_factor", default_overlap_factor),
                )
            )
            per_iteration_overhead_ms = float(
                getattr(
                    cb_config,
                    "per_iteration_overhead_ms",
                    kwargs.get("per_iteration_overhead_ms", default_overhead_ms),
                )
            )
            cb_config_key = (
                "default",
                overlap_factor,
                per_iteration_overhead_ms,
            )
            if cb_config is not None:
                cb_config_key = (
                    cb_config.max_num_batched_tokens,
                    cb_config.max_num_seqs,
                    cb_config.num_requests,
                    cb_config.warmup_requests,
                    cb_config.long_prefill_token_threshold,
                    cb_config.num_gpu_blocks,
                    cb_config.block_size,
                    cb_config.overlap_factor,
                    cb_config.per_iteration_overhead_ms,
                )
            model_key = (
                model.model_path,
                database.system,
                database.backend,
                database.version,
                model.config.tp_size,
                model.config.pp_size,
                model.config.attention_dp_size,
                model.config.moe_tp_size,
                model.config.moe_ep_size,
                model.config.gemm_quant_mode.name,
                model.config.kvcache_quant_mode.name,
                model.config.fmha_quant_mode.name,
                model.config.moe_quant_mode.name,
                model.config.comm_quant_mode.name,
            )
            cache_key = (
                model_key, isl, osl, b, ctx_tokens, "cb_sim", cb_config_key,
            )
            if cache_key in self._agg_cache_v2:
                return self._agg_cache_v2[cache_key]
            summary = self._run_agg_cb_sim(model, database, runtime_config, **kwargs)
            self._agg_cache_v2[cache_key] = summary
            return summary

        isl = runtime_config.isl
        osl = runtime_config.osl
        prefix = runtime_config.prefix
        b = runtime_config.batch_size
        ctx_seq_imbalance_correction_scale = runtime_config.seq_imbalance_correction_scale
        gen_seq_imbalance_correction_scale = runtime_config.gen_seq_imbalance_correction_scale
        ctx_tokens = kwargs.get("ctx_tokens")
        assert ctx_tokens is not None, "ctx_tokens is required"
        balance_score = isl * b / ctx_tokens / osl

        try:
            summary = self._agg_cache[isl][osl][b][ctx_tokens]
        except KeyError:
            # we would like to calculate num_mix_steps and num_genonly_steps based on
            # isl, osl, b, ctx_tokens within osl steps, need to finish all the ctx tokens
            steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)
            num_mix_steps = num_genonly_steps = 0
            num_mix_steps_for_tpot_calc = 0  # this is a correction for tpot calc only.
            if b > 1:
                if steps_to_finish_ctx >= osl:
                    num_mix_steps = steps_to_finish_ctx
                    num_mix_ctx_tokens = ctx_tokens
                    num_mix_gen_tokens = max(1, b // (steps_to_finish_ctx / osl))
                    num_genonly_steps = 0
                    num_genonly_tokens = 0
                    num_mix_steps_for_tpot_calc = num_mix_steps
                else:
                    # 3-step is an empirical correction for pipelining requests where new requests
                    # cannot be enqueued immediately after last request's exit
                    num_mix_steps = steps_to_finish_ctx
                    num_mix_ctx_tokens = ctx_tokens
                    num_mix_gen_tokens = b - np.ceil(ctx_tokens / isl)  # the error check is outside
                    assert num_mix_gen_tokens >= 1, (
                        f"num_mix_gen_tokens: {num_mix_gen_tokens}, b: {b}, ctx_tokens: {ctx_tokens}, isl: {isl}"
                    )
                    num_genonly_steps = osl - num_mix_steps
                    num_genonly_tokens = b
                    num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)
            elif b == 1:
                # special case for b=1
                num_mix_steps = 1
                num_mix_ctx_tokens = ctx_tokens
                num_mix_gen_tokens = 0
                num_genonly_steps = osl - 1
                num_genonly_tokens = 1
                num_mix_steps_for_tpot_calc = 0

            # Per-ops latency collection
            per_ops_data = {}

            # FIXME, fix for DS. DS has different ops for attn in ctx and gen.
            def _get_mix_step_latency(
                model: BaseModel,
                database: PerfDatabase,
                ctx_tokens: int,
                gen_tokens: int,
                isl: int,
                osl: int,
                prefix: int,
            ) -> tuple[float, float]:
                """
                Get mixed step latency and energy.

                Returns:
                    tuple: (latency in ms, energy in watt-milliseconds)
                """
                num_tokens = ctx_tokens + gen_tokens
                # treat this as a combined single batch inference, extract non-attention latency
                summary = self.run_static(
                    model,
                    database,
                    # num tokens for gemm needs to be adjusted for prefix, depends on the avg prefix len per request
                    RuntimeConfig(
                        batch_size=1,
                        beam_width=1,
                        isl=num_tokens,
                        osl=1,
                        prefix=prefix * np.floor(ctx_tokens / isl),
                        seq_imbalance_correction_scale=ctx_seq_imbalance_correction_scale,
                    ),
                    mode="static_ctx",
                )
                latency_dict = summary.get_context_latency_dict()
                energy_wms_dict = summary.get_context_energy_wms_dict()
                non_attention_latency_ms = 0.0
                non_attention_energy_wms = 0.0
                mix_non_attn_ops = {}
                for layer_name, latency in latency_dict.items():
                    if layer_name != "context_attention":
                        non_attention_latency_ms += latency
                        non_attention_energy_wms += energy_wms_dict.get(layer_name, 0.0)
                        mix_non_attn_ops[layer_name] = latency

                # second pass to get ctx attn, split full isl over
                # num_steps(=np.ceil(isl/ctx_tokens))
                # average the ctx attn latency with num_steps to get the ctx_attention_latency
                num_tokens = isl
                batch_size = np.ceil(ctx_tokens / isl)
                summary = self.run_static(
                    model,
                    database,
                    RuntimeConfig(
                        batch_size=batch_size,
                        beam_width=1,
                        isl=num_tokens,
                        osl=1,
                        prefix=prefix,
                        seq_imbalance_correction_scale=ctx_seq_imbalance_correction_scale,
                    ),
                    mode="static_ctx",
                )
                latency_dict = summary.get_context_latency_dict()
                energy_wms_dict = summary.get_context_energy_wms_dict()
                scale_factor = np.ceil(isl / ctx_tokens)
                ctx_attention_latency_ms = latency_dict["context_attention"] / scale_factor
                ctx_attention_energy_wms = energy_wms_dict.get("context_attention", 0.0) / scale_factor

                # third pass to get generation attn. use isl+osl//2 for avg generation attn latency.
                gen_attention_latency_ms = 0.0
                gen_attention_energy_wms = 0.0
                if gen_tokens > 0:
                    num_tokens = gen_tokens
                    summary = self.run_static(
                        model,
                        database,
                        RuntimeConfig(
                            batch_size=num_tokens,
                            beam_width=1,
                            isl=isl + osl // 2,
                            osl=2,
                            gen_seq_imbalance_correction_scale=gen_seq_imbalance_correction_scale,
                        ),
                        mode="static_gen",
                    )
                    latency_dict = summary.get_generation_latency_dict()
                    energy_wms_dict = summary.get_generation_energy_wms_dict()
                    gen_attention_latency_ms = latency_dict["generation_attention"]
                    gen_attention_energy_wms = energy_wms_dict.get("generation_attention", 0.0)

                # Collect per-op breakdown for mix step
                per_ops_data["mix_step"] = {
                    **mix_non_attn_ops,
                    "context_attention (scaled)": ctx_attention_latency_ms,
                    "generation_attention": gen_attention_latency_ms,
                }

                # Combine all components (simple addition)
                total_latency_ms = non_attention_latency_ms + ctx_attention_latency_ms + gen_attention_latency_ms
                total_energy_wms = non_attention_energy_wms + ctx_attention_energy_wms + gen_attention_energy_wms

                return total_latency_ms, total_energy_wms

            def _get_genonly_step_latency(
                model: BaseModel, database: PerfDatabase, gen_tokens: int, isl: int, osl: int
            ) -> tuple[float, float]:
                """
                Get generation-only step latency and energy.

                Returns:
                    tuple: (latency in ms, energy in watt-milliseconds)
                """
                if gen_tokens <= 0:
                    return 0.0, 0.0
                num_tokens = gen_tokens
                summary = self.run_static(
                    model,
                    database,
                    RuntimeConfig(
                        batch_size=num_tokens,
                        beam_width=1,
                        isl=isl + osl // 2,
                        osl=2,
                        gen_seq_imbalance_correction_scale=gen_seq_imbalance_correction_scale,
                    ),
                    mode="static_gen",
                )
                latency_dict = summary.get_generation_latency_dict()
                energy_wms_dict = summary.get_generation_energy_wms_dict()
                genonly_step_latency_ms = 0.0
                genonly_step_energy_wms = 0.0
                genonly_ops = {}
                for layer_name, latency in latency_dict.items():
                    genonly_step_latency_ms += latency
                    genonly_step_energy_wms += energy_wms_dict.get(layer_name, 0.0)
                    genonly_ops[layer_name] = latency

                per_ops_data["genonly_step"] = genonly_ops

                return genonly_step_latency_ms, genonly_step_energy_wms

            # Call helpers (now return energy in W·ms instead of power)
            mix_step_latency_ms, mix_step_energy_wms = _get_mix_step_latency(
                model, database, num_mix_ctx_tokens, num_mix_gen_tokens, isl, osl, prefix
            )
            genonly_step_latency_ms, genonly_step_energy_wms = _get_genonly_step_latency(
                model, database, num_genonly_tokens, isl, osl
            )

            # Calculate timing (unchanged)
            ttft = mix_step_latency_ms * np.ceil(isl / ctx_tokens)
            # correction for ttft in trtllm agg mode, assume we have requests 10x of concurrency
            # (batch size here) to mitigate the impact of first round latency
            # assume we need to increase x of requests when concurrency gets larger.
            # thus capped to 4 to make it reasonable.
            correction_factor = min(2 + (steps_to_finish_ctx - 3) / 2 / 10, 4)
            ttft *= correction_factor
            logger.debug(
                f"ttft correction factor: {2 + (steps_to_finish_ctx - 3) / 2 / 10} capped to "
                f"{correction_factor} when b: {b}, ctx_tokens: {ctx_tokens} isl {isl}"
            )

            tpot = (mix_step_latency_ms * num_mix_steps_for_tpot_calc + genonly_step_latency_ms * num_genonly_steps) / (
                num_mix_steps_for_tpot_calc + num_genonly_steps
            )
            output_throughput = (
                1000
                / (num_mix_steps * mix_step_latency_ms + num_genonly_steps * genonly_step_latency_ms)
                * b
                * (osl - 1)
            )

            calibration_applied = False
            calibration_reason = "disabled"
            ttft_cb_correction = 1.0
            cb_factor = 1.0
            if method == self._CALIBRATED_METHOD:
                if self._should_apply_cb_calibration(model, database, runtime_config):
                    ttft_cb_correction = self._get_ttft_cb_correction(b, isl, osl)
                    cb_factor = self._get_cb_efficiency_factor(b, isl, osl)
                    ttft /= ttft_cb_correction
                    output_throughput *= cb_factor
                    tpot /= cb_factor
                    calibration_applied = True
                    calibration_reason = "applied"
                else:
                    calibration_reason = "out_of_calibrated_regime"
                logger.debug(
                    "vLLM CB calibration %s: ttft_correction=%.2f cb_factor=%.2f (b=%s, isl=%s, osl=%s)",
                    calibration_reason,
                    ttft_cb_correction,
                    cb_factor,
                    b,
                    isl,
                    osl,
                )
            logger.debug(
                f"ctx_tokens: {ctx_tokens}, b: {b}, osl: {osl}, isl: {isl}, "
                f"num_mix_steps: {num_mix_steps}, num_genonly_steps: {num_genonly_steps}, "
                f"num_mix_ctx_tokens: {num_mix_ctx_tokens}, "
                f"num_mix_gen_tokens: {num_mix_gen_tokens}, "
                f"num_genonly_tokens: {num_genonly_tokens}"
            )
            logger.debug(
                f"mix_step_latency: {mix_step_latency_ms} ms, genonly_step_latency: {genonly_step_latency_ms} ms"
            )
            logger.debug(
                f"mix_step_energy: {mix_step_energy_wms} W·ms, genonly_step_energy: {genonly_step_energy_wms} W·ms"
            )
            logger.debug(f"ttft: {ttft}, tpot: {tpot}, output_throughput: {output_throughput}")

            # Calculate weighted average power (SIMPLIFIED!)
            # Step 1: Calculate total energy (simple multiplication and addition)
            total_mix_energy_wms = num_mix_steps * mix_step_energy_wms
            total_genonly_energy_wms = num_genonly_steps * genonly_step_energy_wms
            total_energy_wms = total_mix_energy_wms + total_genonly_energy_wms

            # Step 2: Calculate total latency (simple multiplication and addition)
            total_latency_ms = num_mix_steps * mix_step_latency_ms + num_genonly_steps * genonly_step_latency_ms

            # Step 3: Derive average power (single division)
            if total_latency_ms > 0:
                agg_power_avg_w = total_energy_wms / total_latency_ms
            else:
                agg_power_avg_w = 0.0

            logger.debug(f"Aggregated power: {agg_power_avg_w}W (from {total_energy_wms} W·ms / {total_latency_ms} ms)")

            num_ctx_requests = np.ceil(ctx_tokens / isl)
            num_gen_requests = b - num_ctx_requests
            if b == 1:
                num_ctx_requests = 1
                num_gen_requests = 1

            # correct output_throughput and concurrency for attention dp (global batch)
            scale_factor = model.config.pp_size * model.config.attention_dp_size
            output_throughput = output_throughput * scale_factor
            concurrency = b * scale_factor

            request_rate = output_throughput / (osl - 1)
            if b > 1:
                # will not be corrected by balance score when it's larger than 1.0
                # in order to indicate what's happening
                num_tokens = num_gen_requests + ctx_tokens
            else:
                num_tokens = ctx_tokens
            memory = self._get_memory_usage(
                model,
                database,
                b,
                1,
                isl,
                osl,
                num_tokens,
                prefix=prefix,
                enable_chunked_prefill=kwargs.get("enable_chunked_prefill", False),
                chunked_prefill_tokens=self._resolve_chunked_prefill_tokens(**kwargs),
            )
            logger.debug(
                f"Memory (b={b}, isl={isl}, osl={osl}): total={memory['total']:.2f} GiB "
                f"weights={memory['weights']:.2f} act={memory['activations']:.2f} kv={memory['kvcache']:.2f}"
            )
            tp = model.config.tp_size

            pp = model.config.pp_size
            dp = model.config.attention_dp_size
            moe_tp = model.config.moe_tp_size
            moe_ep = model.config.moe_ep_size
            tokens_s_gpu = output_throughput / pp / tp / dp
            tokens_s_user = 1000 / tpot
            seq_s = request_rate
            seq_s_gpu = seq_s / pp / tp / dp
            tokens_s = output_throughput
            request_latency = ttft + tpot * max(osl - 1, 0)
            num_total_gpus = tp * pp * dp
            parallel = f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}"
            gemm = model.config.gemm_quant_mode.name
            kvcache = model.config.kvcache_quant_mode.name
            fmha = model.config.fmha_quant_mode.name
            moe = model.config.moe_quant_mode.name
            comm = model.config.comm_quant_mode.name
            mem = memory["total"]

            result_dict = {
                "model": model.model_path,
                "isl": isl,
                "osl": osl,
                "prefix": prefix,
                "concurrency": concurrency,
                "request_rate": request_rate,
                "bs": b,
                "global_bs": b * model.config.attention_dp_size,
                "ttft": ttft,
                "tpot": tpot,
                "seq/s": seq_s,
                "seq/s/gpu": seq_s_gpu,
                "tokens/s": tokens_s,
                "tokens/s/gpu": tokens_s_gpu,
                "tokens/s/user": tokens_s_user,
                "request_latency": request_latency,
                "num_total_gpus": num_total_gpus,
                "tp": tp,
                "pp": pp,
                "dp": dp,
                "moe_tp": moe_tp,
                "moe_ep": moe_ep,
                "parallel": parallel,
                "gemm": gemm,
                "kvcache": kvcache,
                "fmha": fmha,
                "moe": moe,
                "comm": comm,
                "memory": mem,
                "balance_score": balance_score,
                "num_ctx_reqs": num_ctx_requests,
                "num_gen_reqs": num_gen_requests,
                "num_tokens": num_tokens,
                "ctx_tokens": ctx_tokens,
                "gen_tokens": num_gen_requests,
                "backend": database.backend,
                "version": database.version,
                "system": database.system,
                "power_w": agg_power_avg_w,  # Weighted average power for AGG mode
            }
            result = pd.DataFrame([result_dict], columns=common.ColumnsAgg).round(3)
            summary = InferenceSummary(RuntimeConfig(isl=isl, osl=osl))
            summary.set_memory_and_check_oom(memory, database.system_spec["gpu"]["mem_capacity"])
            summary.set_summary_df(result)
            summary.set_result_dict(result_dict)

            # Store per-ops latency breakdown
            per_ops_data["scheduling"] = {
                "num_mix_steps": float(num_mix_steps),
                "num_genonly_steps": float(num_genonly_steps),
                "mix_step_latency_ms": float(mix_step_latency_ms),
                "genonly_step_latency_ms": float(genonly_step_latency_ms),
            }
            per_ops_data["batch_sync_boundary"] = {
                "method": method,
                "cb_calibration_applied": calibration_applied,
                "cb_calibration_reason": calibration_reason,
                "ttft_cb_correction": float(ttft_cb_correction),
                "throughput_cb_factor": float(cb_factor),
            }
            summary.set_per_ops_data(per_ops_data)

            # caching
            self._agg_cache[isl][osl][b][ctx_tokens] = summary

        return summary

    def _run_agg_cb_sim(
        self, model: BaseModel, database: PerfDatabase,
        runtime_config: RuntimeConfig, **kwargs,
    ) -> InferenceSummary:
        """CB simulation-based agg prediction (B2).

        Produces a standard ColumnsAgg result, compatible with existing
        CLI/webapp/pareto/InferenceSession consumers.
        """
        from aiconfigurator.sdk.backends.cb_simulator import CBSimConfig, CBSimulator
        from aiconfigurator.sdk.backends.cb_simulator.forward_descriptor import make_topology_key

        isl, osl, prefix, b = (
            runtime_config.isl, runtime_config.osl,
            runtime_config.prefix, runtime_config.batch_size,
        )
        ctx_tokens = kwargs.get("ctx_tokens", self._resolve_chunked_prefill_tokens(**kwargs))
        cb_config = kwargs.get("cb_config")
        if cb_config is None:
            cb_config = CBSimConfig(
                max_num_batched_tokens=ctx_tokens,
                overlap_factor=float(
                    kwargs.get("overlap_factor", self._CB_SIM_DEFAULT_OVERLAP_FACTOR)
                ),
                per_iteration_overhead_ms=float(
                    kwargs.get(
                        "per_iteration_overhead_ms",
                        self._get_cb_sim_decode_overhead_ms(model, database),
                    )
                ),
            )

        tp = model.config.tp_size
        pp = model.config.pp_size
        dp = model.config.attention_dp_size
        num_gpus = tp * pp * dp

        # phase397j: `b` (runtime_config.batch_size) is the GLOBAL concurrency of
        # the deployment. Under attention data parallelism the engine splits that
        # concurrency across `dp` replicas, so a single replica (a tp group) only
        # decodes b/dp requests per iteration -- matching a real DP benchmark that
        # specifies a global max-concurrency (e.g. tp4dp2 at 128 total ->
        # ~64 requests/replica; measured DP0 72 / DP1 56). Simulating the full `b`
        # on one replica over-batched decode by dp and charged the MoE op at
        # b*attention_dp tokens (2x the real gathered token count for dp=2),
        # which inverted the tp-vs-tp+dp throughput ranking. Split per replica.
        # For dp=1 this is a no-op (per_replica == b), so non-DP results are
        # bit-for-bit unchanged.
        per_replica_concurrency = int(np.ceil(b / dp))

        # Simulator returns TP-group throughput for dp=1. For dp>1, use the
        # multi-replica path only in the one-chunk regime whose DP composition
        # fingerprint was validated in Phase449. Larger batched-token regimes
        # (for example bt65536) keep the legacy per-replica path until they have
        # their own fingerprint gate, otherwise a small unvalidated regression
        # leaks into the six-point table.
        sim = CBSimulator(self, model, database, cb_config)
        use_dp_lockstep = dp > 1 and ctx_tokens == isl
        if use_dp_lockstep:
            result = sim.run_multi_replica(
                isl=isl,
                osl=osl,
                concurrency=b,
                data_parallel_size=dp,
                prefix=prefix,
                num_gpus=tp * dp,
                lockstep=True,
            )
            scale_factor = pp
        else:
            result = sim.run(
                isl=isl, osl=osl, concurrency=per_replica_concurrency,
                prefix=prefix, num_gpus=tp,
            )
            scale_factor = pp * dp

        # CB sim throughput validated against real benchmarks (Phase 2).
        # Output-only throughput now comes directly from the simulator.
        # Legacy path: result is per TP group, so scale across pp/dp stages.
        # DP lockstep path: result already includes all DP replicas, so scale pp only.
        raw_output_throughput = result.throughput_tok_s * scale_factor
        output_throughput = raw_output_throughput
        # `b` is already the global concurrency; only pp adds pipeline replicas.
        concurrency_scaled = b * pp

        ttft = result.mean_ttft_ms
        tpot = result.mean_tpot_ms
        request_latency = ttft + tpot * max(osl - 1, 0)
        request_rate = output_throughput / max(osl - 1, 1)
        tokens_s_gpu = output_throughput / num_gpus
        tokens_s_user = 1000 / tpot if tpot > 0 else 0.0
        seq_s = request_rate
        seq_s_gpu = seq_s / num_gpus
        balance_score = isl * b / ctx_tokens / osl

        # Memory check (same as batch_sync path). Per-GPU KV cache holds only the
        # per-replica in-flight requests, so use the dp-split concurrency (phase397j).
        # dp=1 -> per_replica_concurrency == b, unchanged.
        num_tokens = result.peak_tokens_per_iter if result.peak_tokens_per_iter > 0 else ctx_tokens
        memory = self._get_memory_usage(
            model,
            database,
            per_replica_concurrency,
            1,
            isl,
            osl,
            num_tokens,
            prefix=prefix,
            enable_chunked_prefill=kwargs.get("enable_chunked_prefill", False),
            chunked_prefill_tokens=self._resolve_chunked_prefill_tokens(**kwargs),
        )

        moe_tp = model.config.moe_tp_size
        moe_ep = model.config.moe_ep_size
        overhead_topology_key = make_topology_key(tp, dp, moe_tp, moe_ep)
        overhead_shape_key = self._make_cb_sim_decode_overhead_shape_key(
            runtime_config,
            ctx_tokens,
            cb_config,
        )
        overhead_source_key = self._get_cb_sim_decode_overhead_source_key(
            model,
            database,
            float(cb_config.per_iteration_overhead_ms),
        )

        avg_ctx_reqs = result.avg_prefill_reqs_per_iter
        avg_gen_reqs = result.avg_decode_reqs_per_iter
        avg_tokens = result.avg_tokens_per_iter

        result_dict = {
            "model": model.model_path,
            "isl": isl, "osl": osl, "prefix": prefix,
            "concurrency": concurrency_scaled,
            "request_rate": request_rate,
            "bs": per_replica_concurrency,
            "global_bs": b,
            "ttft": ttft, "tpot": tpot,
            "seq/s": seq_s, "seq/s/gpu": seq_s_gpu,
            "tokens/s": output_throughput,
            "tokens/s/gpu": tokens_s_gpu,
            "tokens/s/user": tokens_s_user,
            "request_latency": request_latency,
            "num_total_gpus": num_gpus,
            "tp": tp, "pp": pp, "dp": dp,
            "moe_tp": moe_tp, "moe_ep": moe_ep,
            "parallel": f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}",
            "gemm": model.config.gemm_quant_mode.name,
            "kvcache": model.config.kvcache_quant_mode.name,
            "fmha": model.config.fmha_quant_mode.name,
            "moe": model.config.moe_quant_mode.name,
            "comm": model.config.comm_quant_mode.name,
            "memory": memory["total"],
            "balance_score": balance_score,
            "num_ctx_reqs": avg_ctx_reqs,
            "num_gen_reqs": avg_gen_reqs,
            "num_tokens": avg_tokens,
            "ctx_tokens": ctx_tokens,
            "gen_tokens": avg_gen_reqs,
            "backend": database.backend,
            "version": database.version,
            "system": database.system,
            "power_w": 0.0,  # TODO: energy tracking in simulator
        }

        summary_df = pd.DataFrame([result_dict], columns=common.ColumnsAgg).round(3)
        summary = InferenceSummary(RuntimeConfig(isl=isl, osl=osl))
        summary.set_memory_and_check_oom(memory, database.system_spec["gpu"]["mem_capacity"])
        summary.set_summary_df(summary_df)
        summary.set_result_dict(result_dict)

        per_ops_data = {
            "cb_sim_scheduling": {
                "avg_prefill_reqs_per_iter": float(avg_ctx_reqs),
                "avg_decode_reqs_per_iter": float(avg_gen_reqs),
                "avg_tokens_per_iter": float(avg_tokens),
                "peak_prefill_reqs_per_iter": float(result.peak_prefill_reqs_per_iter),
                "peak_decode_reqs_per_iter": float(result.peak_decode_reqs_per_iter),
                "peak_tokens_per_iter": float(result.peak_tokens_per_iter),
                "steady_state_iterations": result.steady_state_iterations,
                "steady_state_time_ms": float(result.steady_state_time_ms),
                "total_iterations": result.total_iterations,
                "overlap_factor": float(cb_config.overlap_factor),
                "per_iteration_overhead_ms": float(cb_config.per_iteration_overhead_ms),
                "per_iteration_overhead_source_key": overhead_source_key,
                "per_iteration_overhead_topology_key": overhead_topology_key,
                "per_iteration_overhead_shape_key": overhead_shape_key,
                "per_iteration_overhead_diagnostic_only": True,
                "per_iteration_overhead_valid_for_default": False,
                "per_iteration_overhead_perf_database": False,
            },
            "cb_sim_boundary": {
                "ttft_source": "cb_sim",
                "tpot_source": "cb_sim",
                "throughput_source": "cb_sim",
                "cb_sim_raw_tokens_s": float(raw_output_throughput),
                "cb_sim_tokens_s": float(output_throughput),
                "cb_sim_tokens_s_gpu": float(tokens_s_gpu),
            },
        }
        summary.set_per_ops_data(per_ops_data)

        return summary

    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints.

        Args:
            model: the model to be tested
            database: the database to be tested
            runtime_config: the runtime configuration
            top_k: the number of best results to return
            max_batch_size: the maximum batch size to test
            ctx_stride: the stride of ctx tokens to test, it will impact the time to run the test.
            enable_chunked_prefill: whether to enable chunked prefill, it will impact the time to
                run the test while have little impact on the result. Default off.

        Returns:
            A summary of the best agg result under constraints.
        """
        isl = runtime_config.isl
        osl = runtime_config.osl
        ttft = runtime_config.ttft
        tpot = runtime_config.tpot
        prefix = runtime_config.prefix
        top_k = kwargs.get("top_k", 1)
        max_batch_size = kwargs.get("max_batch_size", 512)
        ctx_stride = kwargs.get("ctx_stride", 512)
        enable_chunked_prefill = kwargs.get("enable_chunked_prefill", False)

        # when b is larger than 1024, the result is not good as the data collection is not enough
        # to cover this.
        b_list_default = (
            list(range(1, 16, 1))
            + list(range(16, 32, 4))
            + list(range(32, 64, 8))
            + list(range(64, 256, 16))
            + list(range(256, 512, 32))
            + list(range(512, 1024, 256))
            + [1024]
        )

        # sweep for batch_size and ctx_tokens
        # ctx_tokens will have a step of ctx_stride. When it's larger than 8192, we will increase
        # the step to ctx_stride_large.
        # outer_loop is over batch_size dimention, from 1 to max_batch_size
        # inner_loop is over ctx_tokens dimention, from 0 to max_ctx_tokens where it's
        # max(8192, 4*isl).
        # during the loop, as b, ctx_tokens and system memory are monotonic, we can break the
        # inner loop when the system is oom.
        b_list = [b for b in b_list_default if b <= max_batch_size]
        ctx_tokens_list = self._get_ctx_tokens_list_for_agg_sweep(isl, ctx_stride, enable_chunked_prefill)

        results_df = pd.DataFrame(columns=common.ColumnsAgg)
        results_dict_list = []
        capped_b = []
        all_oom = True
        for b in b_list:
            for ctx_tokens in ctx_tokens_list:
                if b - np.ceil(ctx_tokens / isl) < 0:  # allow b==1
                    break

                if b > 1 and (
                    b - np.ceil(ctx_tokens / isl) < 1
                ):  # general case, to ensure there's at least one gen req
                    break

                # filter out repeated records for balance score correction
                balance_score = isl * b / ctx_tokens / osl
                if balance_score > 1:
                    gen_tokens = b // balance_score
                    if gen_tokens > 1 and gen_tokens in capped_b:
                        continue
                    else:
                        capped_b.append(gen_tokens)

                # Forward non-sweep kwargs (e.g. method="cb_sim") to run_agg
                _sweep_keys = {"top_k", "max_batch_size", "ctx_stride", "enable_chunked_prefill"}
                fwd_kwargs = {k: v for k, v in kwargs.items() if k not in _sweep_keys}
                summary = self.run_agg(
                    model=model,
                    database=database,
                    runtime_config=RuntimeConfig(batch_size=b, isl=isl, osl=osl, prefix=prefix),
                    ctx_tokens=ctx_tokens,
                    enable_chunked_prefill=enable_chunked_prefill,
                    **fwd_kwargs,
                )

                if summary.check_oom():
                    break  # larger ctx tokens will cause oom
                all_oom = False
                result_dict = summary.get_result_dict()
                if result_dict and result_dict["tpot"] <= tpot and result_dict["ttft"] <= ttft:
                    results_dict_list.append(result_dict)

        if results_dict_list:
            results_df = pd.DataFrame(results_dict_list, columns=common.ColumnsAgg).round(3)

        sorted_results_df = results_df.sort_values(by="seq/s", ascending=False).round(3)
        if top_k > 0:
            sorted_results_df = sorted_results_df.head(top_k)

        summary = InferenceSummary(runtime_config)
        summary.set_summary_df(sorted_results_df)
        summary.set_oom(all_oom)
        return summary

    def _get_memory_usage(
        self,
        model: BaseModel,
        database: PerfDatabase,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        num_tokens: int = 0,
        prefix: int = 0,
        *,
        enable_chunked_prefill: bool = False,
        chunked_prefill_tokens: int | None = None,
    ) -> dict[str, float]:
        from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend

        if num_tokens == 0:
            num_tokens = (isl - prefix) * batch_size

        if not enable_chunked_prefill:
            return TRTLLMBackend()._get_memory_usage(
                model, database, batch_size, beam_width, isl, osl, num_tokens,
            )

        chunk_window = int(chunked_prefill_tokens or self._VLLM_MAX_CHUNK_TOKENS)
        num_tokens_for_act = min(num_tokens, chunk_window)
        return TRTLLMBackend()._get_memory_usage(
            model, database, batch_size, beam_width, isl, osl, num_tokens_for_act,
        )
