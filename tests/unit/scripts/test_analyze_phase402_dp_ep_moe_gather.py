from types import SimpleNamespace

import pytest

from scripts import analyze_phase402_dp_ep_moe_gather as phase402


def _fake_counterfactual(scenario: str, moe64: float, moe128: float):
    return SimpleNamespace(
        scenario=scenario,
        generation_moe_local64_ms=moe64,
        generation_moe_gather128_ms=moe128,
        generation_moe_dp0_72_ms=moe128 * 0.56,
        generation_moe_dp1_56_ms=moe128 * 0.44,
    )


def test_phase402_rejects_local64_undercharge_when_runtime_already_gathers() -> None:
    rows = phase402.build_phase402_rows(
        counterfactuals=[
            _fake_counterfactual("K2.5-tp8ep8-8k2k", 10.0, 10.0),
            _fake_counterfactual("K2.5-tp8ep8-32k3k", 10.0, 10.0),
            _fake_counterfactual("K2.5-tp4ep8dp2-8k2k", 6.0, 12.0),
            _fake_counterfactual("K2.5-tp4ep8dp2-32k3k", 7.0, 14.0),
        ]
    )
    by_scenario = {row["scenario"]: row for row in rows}

    dp2 = by_scenario["K2.5-tp4ep8dp2-8k2k"]
    assert dp2["nominal_local_decode_tokens"] == "64"
    assert dp2["nominal_gathered_decode_tokens"] == "128"
    assert dp2["runtime_effective_moe_tokens"] == "128"
    assert dp2["hypothesis_verdict"] == "rejected_current_runtime_already_gathers_moe_tokens"
    assert float(dp2["forced_moe64_error_ratio"]) > float(dp2["forced_moe128_error_ratio"])

    tp8 = by_scenario["K2.5-tp8ep8-8k2k"]
    assert tp8["hypothesis_verdict"] == "control_no_dp_gather_delta"


def test_phase402_records_dp_imbalance_as_secondary_not_primary() -> None:
    rows = phase402.build_phase402_rows(
        counterfactuals=[
            _fake_counterfactual("K2.5-tp8ep8-8k2k", 10.0, 10.0),
            _fake_counterfactual("K2.5-tp8ep8-32k3k", 10.0, 10.0),
            _fake_counterfactual("K2.5-tp4ep8dp2-8k2k", 6.0, 12.0),
            _fake_counterfactual("K2.5-tp4ep8dp2-32k3k", 7.0, 14.0),
        ]
    )
    dp2 = {row["scenario"]: row for row in rows}["K2.5-tp4ep8dp2-8k2k"]

    assert dp2["dp0_decode_tokens"] == "72"
    assert dp2["dp1_decode_tokens"] == "56"
    assert dp2["dp_max_vs_mean_ratio"] == "1.125000"
    assert dp2["dp_max_vs_min_ratio"] == "1.285714"
    assert dp2["primary_driver"] == "not_moe_local64_undercharge"
    assert dp2["secondary_imbalance"] == "dp0_72_dp1_56_secondary_12p5pct_vs_mean"


def test_phase402_writer_rejects_default_upgrade(tmp_path) -> None:
    rows = phase402.build_phase402_rows(
        counterfactuals=[
            _fake_counterfactual("K2.5-tp8ep8-8k2k", 10.0, 10.0),
            _fake_counterfactual("K2.5-tp8ep8-32k3k", 10.0, 10.0),
            _fake_counterfactual("K2.5-tp4ep8dp2-8k2k", 6.0, 12.0),
            _fake_counterfactual("K2.5-tp4ep8dp2-32k3k", 7.0, 14.0),
        ]
    )
    rows[0]["valid_for_default"] = "true"

    with pytest.raises(ValueError, match="valid_for_default"):
        phase402.write_phase402_csv(tmp_path / "bad.csv", rows)
