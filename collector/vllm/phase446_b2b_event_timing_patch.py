#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


HELPER_BEGIN = "    # AIC PHASE446 B2B EVENT TIMING BEGIN\n"
HELPER_END = "    # AIC PHASE446 B2B EVENT TIMING END\n"
FORWARD_BEGIN = "            # AIC PHASE446 B2B FORWARD EVENT BEGIN\n"
FORWARD_END = "            # AIC PHASE446 B2B FORWARD EVENT END\n"

EXECUTE_MODEL_ANCHOR = "    @torch.inference_mode()\n    def execute_model(\n"
FORWARD_CALL_ANCHOR = """            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
"""


def _helper_block() -> str:
    return (
        HELPER_BEGIN
        +
        "    def _aic_phase446_event_jsonl_path(self):\n"
        "        import os\n"
        "        return os.environ.get(\"AIC_PHASE446_EVENT_JSONL\")\n"
        "\n"
        "    def _aic_phase446_ensure_event_state(self):\n"
        "        if getattr(self, \"_aic_phase446_event_state_ready\", False):\n"
        "            return\n"
        "        import atexit\n"
        "        import os\n"
        "        import signal\n"
        "        self._aic_phase446_pending_events = []\n"
        "        self._aic_phase446_records = []\n"
        "        self._aic_phase446_event_pool = []\n"
        "        self._aic_phase446_flush_interval = int(\n"
        "            os.environ.get(\"AIC_PHASE446_FLUSH_INTERVAL\", \"64\")\n"
        "        )\n"
        "        self._aic_phase446_previous_sigterm = signal.getsignal(signal.SIGTERM)\n"
        "        self._aic_phase446_previous_sigint = signal.getsignal(signal.SIGINT)\n"
        "        self._aic_phase446_previous_sigusr1 = signal.getsignal(signal.SIGUSR1)\n"
        "        self._aic_phase446_event_state_ready = True\n"
        "        atexit.register(self._aic_phase446_flush_event_timing)\n"
        "        signal.signal(signal.SIGTERM, self._aic_phase446_signal_flush)\n"
        "        signal.signal(signal.SIGINT, self._aic_phase446_signal_flush)\n"
        "        signal.signal(signal.SIGUSR1, self._aic_phase446_signal_flush)\n"
        "\n"
        "    def _aic_phase446_signal_flush(self, signum, frame):\n"
        "        import signal\n"
        "        self._aic_phase446_flush_event_timing()\n"
        "        previous = None\n"
        "        if signum == signal.SIGTERM:\n"
        "            previous = getattr(self, \"_aic_phase446_previous_sigterm\", None)\n"
        "        elif signum == signal.SIGINT:\n"
        "            previous = getattr(self, \"_aic_phase446_previous_sigint\", None)\n"
        "        elif signum == signal.SIGUSR1:\n"
        "            previous = getattr(self, \"_aic_phase446_previous_sigusr1\", None)\n"
        "            if callable(previous):\n"
        "                previous(signum, frame)\n"
        "            return\n"
        "        if callable(previous):\n"
        "            previous(signum, frame)\n"
        "            return\n"
        "        raise SystemExit(0)\n"
        "\n"
        "    def _aic_phase446_borrow_event_pair(self):\n"
        "        import torch\n"
        "        self._aic_phase446_ensure_event_state()\n"
        "        if self._aic_phase446_event_pool:\n"
        "            return self._aic_phase446_event_pool.pop()\n"
        "        return (\n"
        "            torch.cuda.Event(enable_timing=True),\n"
        "            torch.cuda.Event(enable_timing=True),\n"
        "        )\n"
        "\n"
        "    def _aic_phase446_capture_event_timing(\n"
        "        self,\n"
        "        scheduler_output,\n"
        "        *,\n"
        "        num_tokens_unpadded,\n"
        "        num_tokens_padded,\n"
        "        cudagraph_mode,\n"
        "        start_event,\n"
        "        end_event,\n"
        "    ):\n"
        "        self._aic_phase446_ensure_event_state()\n"
        "        try:\n"
        "            from vllm.v1.utils import compute_iteration_details\n"
        "            details = compute_iteration_details(scheduler_output)\n"
        "            ctx_requests = details.num_ctx_requests\n"
        "            ctx_tokens = details.num_ctx_tokens\n"
        "            gen_requests = details.num_generation_requests\n"
        "            gen_tokens = details.num_generation_tokens\n"
        "        except Exception:\n"
        "            ctx_requests = None\n"
        "            ctx_tokens = None\n"
        "            gen_requests = None\n"
        "            gen_tokens = None\n"
        "        parallel = getattr(self, \"parallel_config\", None)\n"
        "        payload = {\n"
        "            \"schema\": \"phase446_graph_outer_event_v2\",\n"
        "            \"measurement\": \"model_forward_cuda_event_outer_delayed\",\n"
        "            \"ctx_requests\": ctx_requests,\n"
        "            \"ctx_tokens\": ctx_tokens,\n"
        "            \"generation_requests\": gen_requests,\n"
        "            \"generation_tokens\": gen_tokens,\n"
        "            \"num_tokens_unpadded\": int(num_tokens_unpadded),\n"
        "            \"num_tokens_padded\": int(num_tokens_padded),\n"
        "            \"cudagraph_mode\": str(cudagraph_mode),\n"
        "            \"attention_busy_ms\": None,\n"
        "            \"non_attn_busy_ms\": None,\n"
        "            \"dp_rank\": getattr(parallel, \"data_parallel_rank\", None),\n"
        "            \"tp_rank\": getattr(parallel, \"tensor_parallel_rank\", None),\n"
        "        }\n"
        "        self._aic_phase446_pending_events.append((start_event, end_event, payload))\n"
        "        if len(self._aic_phase446_pending_events) >= self._aic_phase446_flush_interval:\n"
        "            self._aic_phase446_drain_ready_events(force=False)\n"
        "\n"
        "    def _aic_phase446_drain_ready_events(self, *, force=False):\n"
        "        if not getattr(self, \"_aic_phase446_event_state_ready\", False):\n"
        "            return\n"
        "        next_pending = []\n"
        "        for start_event, end_event, payload in self._aic_phase446_pending_events:\n"
        "            if not force and not end_event.query():\n"
        "                next_pending.append((start_event, end_event, payload))\n"
        "                continue\n"
        "            if force and not end_event.query():\n"
        "                next_pending.append((start_event, end_event, payload))\n"
        "                continue\n"
        "            payload[\"forward_busy_ms\"] = float(start_event.elapsed_time(end_event))\n"
        "            self._aic_phase446_records.append(payload)\n"
        "            self._aic_phase446_event_pool.append((start_event, end_event))\n"
        "        self._aic_phase446_pending_events = next_pending\n"
        "\n"
        "    def _aic_phase446_flush_event_timing(self):\n"
        "        path = self._aic_phase446_event_jsonl_path()\n"
        "        if not path:\n"
        "            return\n"
        "        self._aic_phase446_drain_ready_events(force=True)\n"
        "        records = getattr(self, \"_aic_phase446_records\", [])\n"
        "        if not records:\n"
        "            return\n"
        "        import json\n"
        "        import os\n"
        "        from pathlib import Path\n"
        "        path_obj = Path(path)\n"
        "        path_obj.parent.mkdir(parents=True, exist_ok=True)\n"
        "        data = \"\".join(\n"
        "            json.dumps(record, sort_keys=True) + \"\\n\" for record in records\n"
        "        ).encode(\"utf-8\")\n"
        "        fd = os.open(str(path_obj), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)\n"
        "        try:\n"
        "            os.write(fd, data)\n"
        "        finally:\n"
        "            os.close(fd)\n"
        "        self._aic_phase446_records = []\n"
        "\n"
        + HELPER_END
    )


def _forward_event_block() -> str:
    return (
        FORWARD_BEGIN
        +
        "            aic_phase446_event_path = self._aic_phase446_event_jsonl_path()\n"
        "            if aic_phase446_event_path:\n"
        "                aic_phase446_forward_start, aic_phase446_forward_end = (\n"
        "                    self._aic_phase446_borrow_event_pair()\n"
        "                )\n"
        "                aic_phase446_forward_start.record()\n"
        + FORWARD_CALL_ANCHOR
        +
        "            if aic_phase446_event_path:\n"
        "                aic_phase446_forward_end.record()\n"
        "                self._aic_phase446_capture_event_timing(\n"
        "                    scheduler_output,\n"
        "                    num_tokens_unpadded=num_tokens_unpadded,\n"
        "                    num_tokens_padded=num_tokens_padded,\n"
        "                    cudagraph_mode=cudagraph_mode,\n"
        "                    start_event=aic_phase446_forward_start,\n"
        "                    end_event=aic_phase446_forward_end,\n"
        "                )\n"
        + FORWARD_END
    )


def _remove_marked_block(text: str, begin: str, end: str) -> str:
    while begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + text[stop:]
    return text


def _replace_marked_block(text: str, begin: str, end: str, replacement: str) -> str:
    while begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + replacement + text[stop:]
    return text


def restore_source(text: str) -> str:
    text = _remove_marked_block(text, HELPER_BEGIN, HELPER_END)
    text = _replace_marked_block(text, FORWARD_BEGIN, FORWARD_END, FORWARD_CALL_ANCHOR)
    return text


def patch_source(text: str) -> str:
    if HELPER_BEGIN in text and FORWARD_BEGIN in text:
        return text
    text = restore_source(text)
    if EXECUTE_MODEL_ANCHOR not in text:
        raise ValueError("execute_model_anchor_not_found")
    if FORWARD_CALL_ANCHOR not in text:
        raise ValueError("forward_call_anchor_not_found")
    text = text.replace(EXECUTE_MODEL_ANCHOR, _helper_block() + EXECUTE_MODEL_ANCHOR, 1)
    text = text.replace(FORWARD_CALL_ANCHOR, _forward_event_block(), 1)
    return text


def apply_patch_file(target: Path, backup: Path | None = None) -> None:
    original = target.read_text(encoding="utf-8")
    patched = patch_source(original)
    if backup is not None and not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")


def restore_patch_file(target: Path, backup: Path | None = None) -> None:
    if backup is not None and backup.exists():
        target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        return
    target.write_text(restore_source(target.read_text(encoding="utf-8")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch vLLM gpu_model_runner for Phase446 B2b event timing.")
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--backup", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--restore", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.apply:
        apply_patch_file(args.target, args.backup)
        return 0
    if args.restore:
        restore_patch_file(args.target, args.backup)
        return 0
    text = args.target.read_text(encoding="utf-8")
    if HELPER_BEGIN in text and FORWARD_BEGIN in text:
        print("phase446_b2b_event_timing_patch=applied")
        return 0
    print("phase446_b2b_event_timing_patch=missing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
