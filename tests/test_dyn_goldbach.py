"""Goldbach epistemic trap pins: finite verification justifies C2, never C6.

Promoted from scratch/test_goldbach_epistemic_adjudication.py so the
finite-evidence-vs-proof distinction is gated, not illustrative.
Deterministic (sieve + exhaustive check, fixed limit).
"""

import importlib.util


def _load():
    spec = importlib.util.spec_from_file_location(
        "goldbach_case", "scratch/test_goldbach_epistemic_adjudication.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_goldbach_universal_claim_abstains():
    mod = _load()
    r = mod.run_goldbach_epistemic_adjudication(limit=100000)
    assert r["total_even_numbers_tested"] == 49999
    assert r["counterexamples_found"] == 0
    assert r["eedr_final_status"] == "TERMINAL_ABSTAIN"
    assert r["justified_claim_tier"] == "C2_PREDICTIVE"


def test_goldbach_audit_flags():
    mod = _load()
    r = mod.run_goldbach_epistemic_adjudication(limit=100000)
    inv = r["epistemic_audit_invariants"]
    assert inv["finite_induction_fallacy_detected"] is True
    assert inv["universal_proof_rejected"] is True
    assert inv["bounded_empirical_regularity_preserved"] is True
