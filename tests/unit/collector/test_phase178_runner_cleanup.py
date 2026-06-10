from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector" / "vllm" / "run_phase178_budget_mechanism_holdout.sh"


def _fake_cleanup_path(tmp_path: Path) -> tuple[Path, Path]:
    marker = tmp_path / "cleanup_marker.log"
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for name in ("pkill", "ray"):
        script = fakebin / name
        script.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s %s\\n' '{name}' \"$*\" >> '{marker}'\n"
            "exit 0\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
    return fakebin, marker


def _env_with_fake_cleanup(fakebin: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    return env


def test_preflight_does_not_call_cleanup_commands(tmp_path: Path) -> None:
    fakebin, marker = _fake_cleanup_path(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "preflight"],
        cwd=REPO_ROOT,
        env=_env_with_fake_cleanup(fakebin),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "preflight_benchmark_started=false" in result.stdout
    assert not marker.exists()


def test_cleanup_mode_calls_cleanup_commands(tmp_path: Path) -> None:
    fakebin, marker = _fake_cleanup_path(tmp_path)

    result = subprocess.run(
        ["bash", str(RUNNER), "cleanup"],
        cwd=REPO_ROOT,
        env=_env_with_fake_cleanup(fakebin),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    marker_text = marker.read_text(encoding="utf-8")
    assert "pkill -f vllm.entrypoints.cli.main serve" in marker_text
    assert "ray stop --force" in marker_text
