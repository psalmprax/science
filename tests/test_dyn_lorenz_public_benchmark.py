"""Pins for the frozen Lorenz public-benchmark harness (PREREG-LORENZ-PUB-V1).

The prereg is the spec (docs/PREREGISTRATION_LORENZ_PUBLIC_BENCHMARK_V1.md);
these pins cover the harness WIRING in smoke mode only (single seed, referee
bootstrap 100) — never a confirmation verdict:

  W1  the harness runs end-to-end and writes its payload in the frozen schema;
  W2  PATH-B (greedy forward training-BIC) is never referee-promoted against
      the true model (H1-margin wiring: status != unique_support);
  W3  PATH-C (the transcript chimera) is scored without referee certification
      and its divergence-horizon ratio is below the H6 catch bound (< 0.5),
      i.e. the registered instruments DO catch the chimera at smoke settings.
Full confirmation behavior is the harness's own registered run, not these pins.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "benchmarks" / "run_lorenz_public_benchmark.py"
RESULTS = REPO_ROOT / "benchmarks" / "lorenz_public_benchmark_results.json"


def _run_smoke() -> dict:
    proc = subprocess.run([sys.executable, str(HARNESS), "--smoke"],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-500:]
    return json.loads(RESULTS.read_text())


def test_smoke_payload_schema_and_paths():  # W1
    payload = _run_smoke()
    assert payload["prereg_id"] == "PREREG-LORENZ-PUB-V1-2026-10-20"
    assert payload["mode"] == "smoke"
    paths = {r["path"] for r in payload["rows"]}
    assert paths == {"B", "B-prime", "R", "C"}
    head = payload["frozen"]
    assert head["world"]["rho"] == 28.0 and head["referee"]["n_boot"] == 1000
    # smoke artifact must not be mistakable for a confirmation run
    assert payload["git_head"]


def test_path_b_is_never_referee_promoted():  # W2 (H1 wiring margin)
    payload = _run_smoke()
    for row in payload["rows"]:
        if row["path"] == "B":
            assert row["referee_status"] != "unique_support", row
            assert row["referee_promoted"] != "winner"
            # the naive path must load spurious structure (its raison d'etre)
            assert row["structure_clean"] is False


def test_chimera_caught_by_registered_instruments():  # W3 (H6 wiring margin)
    payload = _run_smoke()
    for row in payload["rows"]:
        if row["path"] == "C":
            assert row["referee_status"] == "not-adjudicated-chimera"
            assert row["divergence_horizon_ratio"] is not None
            assert row["divergence_horizon_ratio"] < 0.5, (
                "the chimera must NOT clear the same-integrator divergence "
                "bound at smoke settings; if it does, the H6 instrument set "
                "needs reinspection BEFORE the full confirmation run")


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
