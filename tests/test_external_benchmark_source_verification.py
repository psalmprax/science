"""Source verification for the two experimental benchmark tables.

The measurement lives in ``benchmarks/external_benchmarks_source_check.json``,
produced by ``scripts/verify_external_benchmark_sources.py``. These tests are the
offline half: they pin what was measured, so the verdict cannot drift silently,
and they pin the repository's dataset hashes, so the tables cannot be edited
while the record still says "unverified at these numbers".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from apedp.ingestion.external_benchmarks import (
    NACA0012TransonicBenchmark,
    NanocrystallineMicrocompressionBenchmark,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORD = REPO_ROOT / "benchmarks" / "external_benchmarks_source_check.json"
DOC = REPO_ROOT / "docs" / "BENCHMARK_SOURCE_VERIFICATION.md"


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_and_page_exist(record):
    assert DOC.is_file(), DOC
    assert record["unreachable_sources"] == []
    assert record["script"] == "scripts/verify_external_benchmark_sources.py"


def test_every_source_is_identified_by_a_hash(record):
    citations = {
        "harris_tm81927": "TM-81927",
        "ladson_cp": "100526",
        "gregory_cp": "3726",
        "meyers_review": "Meyers",
    }
    for key, marker in citations.items():
        source = record["sources"][key]
        assert marker in source["citation"], key
        assert source["url"].startswith("https://"), key
        assert len(source["sha256"]) == 64, key
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), key
        assert source["bytes"] > 1000, key


def test_both_tables_are_marked_unverified(record):
    assert record["naca0012_transonic"]["verdict"] == "unverified"
    assert record["nanocrystalline_copper"]["verdict"] == "unverified"
    for benchmark in (NACA0012TransonicBenchmark, NanocrystallineMicrocompressionBenchmark):
        provenance = benchmark.provenance()
        assert provenance["verified"] is False, benchmark.__name__
        assert provenance["record"] == "benchmarks/external_benchmarks_source_check.json"
        assert provenance["reason"].strip(), benchmark.__name__


def test_recorded_hashes_are_the_live_datasets(record):
    """A table edit changes its hash and must re-open the verification."""
    _, _, _, naca_hash = NACA0012TransonicBenchmark.get_dataset()
    _, _, _, copper_hash = NanocrystallineMicrocompressionBenchmark.get_dataset()
    assert naca_hash == record["naca0012_transonic"]["repo_dataset_sha256"]
    assert copper_hash == record["nanocrystalline_copper"]["repo_dataset_sha256"]


def test_harris_report_has_no_cp_versus_mach_table(record):
    """The cited report's only table is the airfoil section coordinates."""
    harris = record["naca0012_transonic"]["harris_report"]
    assert harris["section_coordinates_table_present"] is True
    assert harris["cp_vs_mach_table_present"] is False
    assert any("SECTION COORDINATES" in title for title in harris["table_titles"])
    assert harris["text_chars"] > 20000


def test_naca_values_are_a_constant_multiple_of_the_measured_point(record):
    """The signature of a constructed series: one ratio, five Mach numbers."""
    naca = record["naca0012_transonic"]
    assert naca["pg_reference"]["incompressible_cp0"] == pytest.approx(-0.3387, abs=5e-4)
    low_band = [row for row in naca["comparison"] if row["mach"] <= 0.80]
    ratios = [row["ratio_table_over_measurement"] for row in low_band]
    assert len(ratios) == 5
    assert max(ratios) - min(ratios) < 0.005, ratios
    assert naca["ratio_band_m0.50_to_m0.80"] == pytest.approx([1.475, 1.476], abs=1e-3)

    # And the drift at higher Mach, which no single angle of attack produces.
    high = {row["mach"]: row["ratio_table_over_measurement"]
            for row in naca["comparison"] if row["mach"] >= 0.88}
    assert max(high.values()) > 1.6
    assert high[0.95] < 1.3


def test_naca_measurement_is_alpha_dependent_at_one_mach(record):
    """Cp(0.25) moves by 2.8x between alpha~0 and alpha=10 at M=0.30."""
    measured = {entry["zone"]: entry["cp_at_x_over_c_025"]
                for entry in record["naca0012_transonic"]["measured_sources"]
                if entry["file"] == "ladson_cp.dat" and "Re=6 million" in entry["zone"]}
    alpha_zero = next(value for zone, value in measured.items() if "alpha=.0169" in zone)
    alpha_ten = next(value for zone, value in measured.items() if "alpha=10.0254" in zone)
    assert alpha_zero == pytest.approx(-0.3551, abs=5e-4)
    assert alpha_ten == pytest.approx(-0.9964, abs=5e-4)
    assert alpha_ten / alpha_zero == pytest.approx(2.8, abs=0.1)


def test_copper_table_sits_outside_the_published_range_at_20nm(record):
    copper = record["nanocrystalline_copper"]
    assert copper["published_points"] == [
        {"sample": "Cu1", "compaction_c": 106.0, "density_pct": 92.5,
         "grain_size_nm": 19.0, "sigma_y_mpa": 650.0, "hardness_over_three_mpa": 770.0},
        {"sample": "Cu2", "compaction_c": 106.0, "density_pct": 98.4,
         "grain_size_nm": 20.0, "sigma_y_mpa": 850.0, "hardness_over_three_mpa": 870.0},
    ]
    assert copper["published_envelope"]["sigma_mpa"] == [100, 900]
    at_20 = copper["at_20_nm"]
    assert at_20["repo_mpa"] == 910.0
    assert at_20["published_mpa"] == [650.0, 850.0]
    assert at_20["above_best_published_pct"] == pytest.approx(7.1, abs=0.05)
    assert at_20["above_lowest_published_pct"] == pytest.approx(40.0, abs=0.05)
    # The table's peak exceeds the review's own envelope for the whole range.
    assert copper["repo_peak"]["grain_size_nm"] == 12.0
    assert copper["repo_peak"]["sigma_y_mpa"] > copper["published_envelope"]["sigma_mpa"][1]


def test_page_states_both_verdicts_and_what_would_change_them():
    text = DOC.read_text(encoding="utf-8")
    assert text.count("unverified") >= 3
    for name in ("NACA0012TransonicBenchmark", "NanocrystallineMicrocompressionBenchmark",
                 "AGARD-AR-138", "inert-gas-condensed"):
        assert name in text, name
    assert "What would verify them" in text
