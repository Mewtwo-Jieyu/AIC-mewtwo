from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "phase442_event_timing_patch.py"
    spec = importlib.util.spec_from_file_location("phase442_event_timing_patch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


SAMPLE_GPU_MODEL_RUNNER = '''
import torch


class GPUModelRunner:
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
    ):
        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        num_tokens_padded = batch_desc.num_tokens
        with (
            set_forward_context(),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        return model_output
'''


def test_phase442_patch_inserts_event_timing_and_jsonl_writer():
    patcher = _load_module()

    patched = patcher.patch_source(SAMPLE_GPU_MODEL_RUNNER)

    assert "AIC PHASE442 EVENT TIMING BEGIN" in patched
    assert "AIC_PHASE442_EVENT_JSONL" in patched
    assert "forward_busy_ms" in patched
    assert "torch.cuda.Event(enable_timing=True)" in patched
    assert "aic_phase442_forward_start.record()" in patched
    assert "aic_phase442_forward_end.synchronize()" in patched
    assert "self._aic_phase442_write_event_timing(" in patched


def test_phase442_patch_is_reversible_and_idempotent():
    patcher = _load_module()

    patched = patcher.patch_source(SAMPLE_GPU_MODEL_RUNNER)
    assert patcher.patch_source(patched) == patched
    assert patcher.restore_source(patched).strip() == SAMPLE_GPU_MODEL_RUNNER.strip()
