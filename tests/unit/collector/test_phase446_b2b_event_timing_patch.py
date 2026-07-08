from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "phase446_b2b_event_timing_patch.py"
    spec = importlib.util.spec_from_file_location("phase446_b2b_event_timing_patch", path)
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


def test_phase446_patch_records_events_without_runtime_sync_or_step_io():
    patcher = _load_module()

    patched = patcher.patch_source(SAMPLE_GPU_MODEL_RUNNER)

    assert "AIC PHASE446 B2B EVENT TIMING BEGIN" in patched
    assert "AIC_PHASE446_EVENT_JSONL" in patched
    assert "phase446_graph_outer_event_v2" in patched
    assert "torch.cuda.Event(enable_timing=True)" in patched
    assert "aic_phase446_forward_start.record()" in patched
    assert "aic_phase446_forward_end.record()" in patched
    assert "aic_phase446_records.append(" in patched
    assert "atexit.register(self._aic_phase446_flush_event_timing)" in patched
    assert ".synchronize()" not in patched
    assert "with path_obj.open(\"a\"" not in patched


def test_phase446_patch_is_reversible_and_idempotent():
    patcher = _load_module()

    patched = patcher.patch_source(SAMPLE_GPU_MODEL_RUNNER)

    assert patcher.patch_source(patched) == patched
    assert patcher.restore_source(patched).strip() == SAMPLE_GPU_MODEL_RUNNER.strip()


if __name__ == "__main__":
    test_phase446_patch_records_events_without_runtime_sync_or_step_io()
    test_phase446_patch_is_reversible_and_idempotent()
