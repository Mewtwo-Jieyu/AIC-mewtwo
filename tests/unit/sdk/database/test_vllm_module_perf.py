from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import (
    PerfDataNotAvailableError,
    PerfDatabase,
    load_vllm_module_data,
)
from aiconfigurator.sdk.performance_result import PerformanceResult


REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_SYSTEMS_ROOT = REPO_ROOT / "src/aiconfigurator/systems"
REAL_VLLM_MODULE_PERF = (
    REAL_SYSTEMS_ROOT / "data/h200_sxm/vllm/0.19.0/vllm_module_perf.txt"
)
REAL_BUCKETS = [1, 15, 16, 241, 1808, 2048, 8192]


def _write_system(tmp_path):
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "vllm" / "0.19.0"
    data_dir.mkdir(parents=True)
    system_spec = {
        "data_dir": "data",
        "misc": {"nccl_version": "missing"},
        "gpu": {"float16_tc_flops": 1_000.0, "mem_bw": 100.0},
        "node": {"num_gpus_per_node": 8, "inter_node_bw": 100.0, "intra_node_bw": 100.0},
    }
    (systems_root / "test_system.yaml").write_text(yaml.safe_dump(system_spec), encoding="utf-8")
    return systems_root, data_dir


def _write_vllm_module_perf(data_dir, rows: str) -> None:
    (data_dir / common.PerfDataFilename.vllm_module.value).write_text(rows, encoding="utf-8")


def _row(
    *,
    bucket_tokens: int,
    module_boundary: str = "fusedmoe_runner_compute",
    latency: str = "1.25",
    power: str | None = "200.0",
    kernel_source: str = "CompressedTensorsWNA16MarlinMoEMethod:Marlin",
) -> list[str]:
    values = [
        "kimi-k2.5",
        "h200_sxm",
        "0.19.0",
        "tp4dp2ep8",
        str(bucket_tokens),
        module_boundary,
        "CompressedTensorsWNA16MarlinMoEMethod",
        latency,
        kernel_source,
    ]
    if power is not None:
        values.append(power)
    return values


def test_vllm_module_filename_is_independent_from_moe_perf() -> None:
    assert common.PerfDataFilename.vllm_module.value == "vllm_module_perf.txt"
    assert common.PerfDataFilename.vllm_module.value != common.PerfDataFilename.moe.value


def test_load_vllm_module_data_reads_exact_rows_and_energy(tmp_path) -> None:
    csv_file = tmp_path / "vllm_module_perf.txt"
    csv_file.write_text(
        "\n".join(
            [
                "model,hardware,vllm_version,topology,bucket_tokens,module_boundary,quant_runtime,latency,kernel_source,power",
                ",".join(_row(bucket_tokens=1, latency="1.25", power="200.0")),
                ",".join(_row(bucket_tokens=15, module_boundary="ep8_comm_dispatch_combine", latency="0.5", power="10.0")),
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = load_vllm_module_data(csv_file)

    key = (
        "kimi-k2.5",
        "h200_sxm",
        "0.19.0",
        "tp4dp2ep8",
        1,
        "fusedmoe_runner_compute",
        "CompressedTensorsWNA16MarlinMoEMethod",
    )
    assert data[key]["latency"] == pytest.approx(1.25)
    assert data[key]["energy"] == pytest.approx(250.0)
    assert data[key]["kernel_source"] == "CompressedTensorsWNA16MarlinMoEMethod:Marlin"


def test_load_vllm_module_data_defaults_missing_power_to_zero(tmp_path) -> None:
    csv_file = tmp_path / "vllm_module_perf.txt"
    csv_file.write_text(
        "\n".join(
            [
                "model,hardware,vllm_version,topology,bucket_tokens,module_boundary,quant_runtime,latency,kernel_source",
                ",".join(_row(bucket_tokens=16, latency="0.75", power=None)),
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = load_vllm_module_data(csv_file)
    result = next(iter(data.values()))

    assert result["power"] == pytest.approx(0.0)
    assert result["energy"] == pytest.approx(0.0)


def test_load_vllm_module_data_rejects_duplicate_exact_key(tmp_path) -> None:
    csv_file = tmp_path / "vllm_module_perf.txt"
    csv_file.write_text(
        "\n".join(
            [
                "model,hardware,vllm_version,topology,bucket_tokens,module_boundary,quant_runtime,latency,kernel_source,power",
                ",".join(_row(bucket_tokens=1, latency="1.25", power="200.0", kernel_source="kernel_a")),
                ",".join(_row(bucket_tokens=1, latency="1.50", power="201.0", kernel_source="kernel_b")),
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate vllm module perf exact key"):
        load_vllm_module_data(csv_file)


def test_load_vllm_module_data_missing_file_fails_fast(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_vllm_module_data(tmp_path / "missing_vllm_module_perf.txt")


def test_query_vllm_module_exact_lookup_returns_performance_result(tmp_path) -> None:
    systems_root, data_dir = _write_system(tmp_path)
    _write_vllm_module_perf(
        data_dir,
        "\n".join(
            [
                "model,hardware,vllm_version,topology,bucket_tokens,module_boundary,quant_runtime,latency,kernel_source,power",
                ",".join(_row(bucket_tokens=241, latency="2.5", power="20.0")),
                "",
            ]
        ),
    )
    database = PerfDatabase("test_system", "vllm", "0.19.0", str(systems_root))

    result = database.query_vllm_module(
        model="kimi-k2.5",
        hardware="h200_sxm",
        vllm_version="0.19.0",
        topology="tp4dp2ep8",
        bucket_tokens=241,
        module_boundary="fusedmoe_runner_compute",
        quant_runtime="CompressedTensorsWNA16MarlinMoEMethod",
    )

    assert isinstance(result, PerformanceResult)
    assert float(result) == pytest.approx(2.5)
    assert result.energy == pytest.approx(50.0)


def test_query_vllm_module_rejects_128_unknown_module_and_missing_exact_key(tmp_path) -> None:
    systems_root, data_dir = _write_system(tmp_path)
    _write_vllm_module_perf(
        data_dir,
        "\n".join(
            [
                "model,hardware,vllm_version,topology,bucket_tokens,module_boundary,quant_runtime,latency,kernel_source,power",
                ",".join(_row(bucket_tokens=1, latency="1.0", power="10.0")),
                "",
            ]
        ),
    )
    database = PerfDatabase("test_system", "vllm", "0.19.0", str(systems_root))

    with pytest.raises(ValueError, match="bucket_tokens"):
        database.query_vllm_module(
            "kimi-k2.5",
            "h200_sxm",
            "0.19.0",
            "tp4dp2ep8",
            128,
            "fusedmoe_runner_compute",
            "CompressedTensorsWNA16MarlinMoEMethod",
        )

    with pytest.raises(ValueError, match="module_boundary"):
        database.query_vllm_module(
            "kimi-k2.5",
            "h200_sxm",
            "0.19.0",
            "tp4dp2ep8",
            1,
            "bare_fused_experts",
            "CompressedTensorsWNA16MarlinMoEMethod",
        )

    with pytest.raises(PerfDataNotAvailableError, match="exact vLLM module perf key"):
        database.query_vllm_module(
            "kimi-k2.5",
            "h200_sxm",
            "0.19.0",
            "tp8ep8",
            1,
            "fusedmoe_runner_compute",
            "CompressedTensorsWNA16MarlinMoEMethod",
        )


def test_query_vllm_module_missing_table_fails_fast(tmp_path) -> None:
    systems_root, _ = _write_system(tmp_path)
    database = PerfDatabase("test_system", "vllm", "0.19.0", str(systems_root))

    with pytest.raises(PerfDataNotAvailableError, match="vLLM module perf table is missing"):
        database.query_vllm_module(
            "kimi-k2.5",
            "h200_sxm",
            "0.19.0",
            "tp4dp2ep8",
            1,
            "fusedmoe_runner_compute",
            "CompressedTensorsWNA16MarlinMoEMethod",
        )


def test_real_vllm_module_perf_file_loads_14_exact_keys() -> None:
    data = load_vllm_module_data(REAL_VLLM_MODULE_PERF)

    assert len(data) == 14
    expected_keys = {
        (
            "kimi-k2.5",
            "h200_sxm",
            "0.19.0",
            "tp4dp2ep8",
            bucket,
            module_boundary,
            "CompressedTensorsWNA16MarlinMoEMethod",
        )
        for bucket in REAL_BUCKETS
        for module_boundary in (
            "fusedmoe_runner_compute",
            "ep8_comm_dispatch_combine",
        )
    }
    assert set(data) == expected_keys

    fused_key = (
        "kimi-k2.5",
        "h200_sxm",
        "0.19.0",
        "tp4dp2ep8",
        8192,
        "fusedmoe_runner_compute",
        "CompressedTensorsWNA16MarlinMoEMethod",
    )
    ep8_key = (
        "kimi-k2.5",
        "h200_sxm",
        "0.19.0",
        "tp4dp2ep8",
        2048,
        "ep8_comm_dispatch_combine",
        "CompressedTensorsWNA16MarlinMoEMethod",
    )
    assert data[fused_key]["latency"] == pytest.approx(20.612520)
    assert data[fused_key]["kernel_source"] == "CompressedTensorsWNA16MarlinMoEMethod:Marlin"
    assert data[ep8_key]["latency"] == pytest.approx(1.237760)
    assert data[ep8_key]["kernel_source"] == "allgather_reducescatter:AgRsAll2AllManager"
    assert all(row["power"] == pytest.approx(0.0) for row in data.values())
    assert all(row["energy"] == pytest.approx(0.0) for row in data.values())


def test_real_perf_database_queries_two_vllm_module_boundaries() -> None:
    database = PerfDatabase("h200_sxm", "vllm", "0.19.0", str(REAL_SYSTEMS_ROOT))

    fused = database.query_vllm_module(
        "kimi-k2.5",
        "h200_sxm",
        "0.19.0",
        "tp4dp2ep8",
        241,
        "fusedmoe_runner_compute",
        "CompressedTensorsWNA16MarlinMoEMethod",
    )
    ep8 = database.query_vllm_module(
        "kimi-k2.5",
        "h200_sxm",
        "0.19.0",
        "tp4dp2ep8",
        1,
        "ep8_comm_dispatch_combine",
        "CompressedTensorsWNA16MarlinMoEMethod",
    )

    assert isinstance(fused, PerformanceResult)
    assert float(fused) == pytest.approx(1.889204)
    assert fused.energy == pytest.approx(0.0)
    assert isinstance(ep8, PerformanceResult)
    assert float(ep8) == pytest.approx(0.157088)
    assert ep8.energy == pytest.approx(0.0)


def test_real_perf_database_vllm_module_exact_lookup_fail_fast() -> None:
    database = PerfDatabase("h200_sxm", "vllm", "0.19.0", str(REAL_SYSTEMS_ROOT))

    with pytest.raises(ValueError, match="bucket_tokens"):
        database.query_vllm_module(
            "kimi-k2.5",
            "h200_sxm",
            "0.19.0",
            "tp4dp2ep8",
            128,
            "fusedmoe_runner_compute",
            "CompressedTensorsWNA16MarlinMoEMethod",
        )

    with pytest.raises(PerfDataNotAvailableError, match="exact vLLM module perf key"):
        database.query_vllm_module(
            "kimi-k2.5",
            "h200_sxm",
            "0.19.0",
            "wrong_topology",
            1,
            "fusedmoe_runner_compute",
            "CompressedTensorsWNA16MarlinMoEMethod",
        )

    with pytest.raises(ValueError, match="module_boundary"):
        database.query_vllm_module(
            "kimi-k2.5",
            "h200_sxm",
            "0.19.0",
            "tp4dp2ep8",
            1,
            "bare_fused_experts",
            "CompressedTensorsWNA16MarlinMoEMethod",
        )
