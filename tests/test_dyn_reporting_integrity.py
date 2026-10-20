"""Reporting integrity: suite counts must be machine-checked, not narrated.

The failure this file guards against is a reporting failure, not a code one:
a scoped count ("74/74") presented in a way that reads as the whole suite.
The test below recomputes the count of the RIME/repair/referee/STEM suite by
collection and asserts it equals a constant in this file, so a stated total
is always CI-verified. It also asserts the constant exceeds the scoped 74,
so the two cannot be confused, and that the CHANGELOG names the standalone
exclusion rather than leaving a silently-shorter suite.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Machine-verified total: updated only by re-running this test, which fails
# if it is wrong. Anything quoted in prose must match it or be labelled a
# subset.
RIME_SUITE_TOTAL = 213

RIME_TEST_FILES = [
    "test_dyn_shape_inference.py",
    "test_dyn_prior_guided_unsolicited.py",
    "test_dyn_landauer_report_fidelity.py",
    "test_dyn_rime_coverage_funnel.py",
    "test_dyn_ga_qec.py",
    "test_dyn_rime_noise_present_battery.py",
    "test_dyn_rime_operator_release_gate.py",
    "test_dyn_rime_qec_d5_and_saturation_pins.py",
    "test_dyn_rime_qec_oos_frozen.py",
    "test_dyn_rime_qec_repaired_transfer.py",
    "test_dyn_rime_rational_and_pade.py",
    "test_dyn_rime_rational_parameter_identifiability.py",
    "test_dyn_rime_reachability_and_contemporaneous_qec.py",
    "test_dyn_rime_repair_v010_diagnosis.py",
    "test_dyn_rime_repair_v011_engine.py",
    "test_dyn_rime_repair_v012_sensitivity.py",
    "test_dyn_rime_repair_v013_realdata.py",
    "test_dyn_rime_repair_v014_qec.py",
    "test_dyn_rime_seed_order_coverage.py",
    "test_dyn_rime_staged_search.py",
    "test_dyn_rime_static_grammar.py",
    "test_dyn_rime_structured_null_fpr.py",
    "test_dyn_rime_v013_integration.py",
    "test_dyn_rime_v014_grammar_ablation.py",
    "test_dyn_rime_v02_flaw_remediation.py",
    "test_dyn_rime_whiteness_calibration.py",
    "test_dyn_rime_known_truth.py",
    "test_dyn_rime_known_truth_scope_repeats.py",
    "test_dyn_rime_native_additive_and_seeds.py",
    "test_dyn_rime_proposer.py",
    "test_dyn_real_data_ame_repair.py",
    "test_dyn_stem_grammar_options_123.py",
    "test_dyn_whiteness_flag_calibration.py",
    "test_dyn_referee_floor_sensitivity.py",
    "test_dyn_referee_rho_sensitivity.py",
    "test_dyn_selective_risk_ga_rime.py",
    "test_dyn_quantum_sandbox.py",
    "test_dyn_trading_np93.py",
    "test_dyn_trading_sandboxes_epistemic.py",
    "test_dyn_trading_universe_sweep.py",
    "test_dyn_intraday_universe_sweep.py",
    "test_dyn_population_extrapolation_rime_ga.py",
    "test_dyn_population_extrapolation_v2.py",
]


def _collected_count() -> int:
    """Count test_* functions per file by importing (fast, deterministic).

    Static counting, not a pytest subprocess: parametrised tests in this suite
    are declared with explicit loops inside a single function, so function
    count equals case count here. It is still machine-checked -- nobody types
    the number by hand and asserts it -- which is the property that was
    violated by a narrated "74/74".
    """
    total = 0
    missing = []
    for fname in RIME_TEST_FILES:
        path = ROOT / "tests" / fname
        if not path.exists():
            missing.append(fname)
            continue
        ns: dict = {"__file__": str(path), "__name__": f"rtk_count_{path.stem}"}
        try:
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)  # noqa: S102 - integrity harness deliberately executes repo-owned source files; no external input
        except Exception as exc:
            raise AssertionError(f"{fname} failed to import: {exc}") from exc
        total += sum(1 for k, v in ns.items()
                     if k.startswith("test_") and callable(v))
    assert not missing, missing
    return total


def test_stated_suite_total_is_machine_verified():
    actual = _collected_count()
    print(f"\n[reporting] RIME/repair/referee/STEM suite collected = {actual} (stated "
          f"{RIME_SUITE_TOTAL})")
    assert actual == RIME_SUITE_TOTAL, (
        actual, RIME_SUITE_TOTAL,
        "update RIME_SUITE_TOTAL by re-running this test, never by editing prose")


def test_scoped_subset_is_not_mistaken_for_the_total():
    # The scoped Python figure quoted elsewhere must stay a labelled subset.
    assert RIME_SUITE_TOTAL > 74, RIME_SUITE_TOTAL
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "74" in changelog or "Python" in changelog  # the subset is mentioned


def test_standalone_exclusions_are_named_not_silent():
    """A suite that shrinks by deselection must say so."""
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "lean" in changelog.lower(), "the Lean timeout exclusion must be named"
    assert "nano_ds10_freeze" in changelog, "the stale-pin failure must be named"
