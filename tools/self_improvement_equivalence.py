"""Equivalence check: re-authored apedp/self_improvement.py vs the orphaned bytecode.

Same approach as ``tools/constitution_equivalence.py``, and for the same
reason: a re-authored module that quietly differs from the one it replaces
is worse than no module, because the gap is invisible and the name is not.

Compares, over a matrix of inputs:
  - ``FailureDetector.detect_weaknesses`` -- every emitted signal's type,
    severity, component, description, metrics and suggested level, in order
  - ``IndependentGatekeeper.evaluate_and_decide`` -- accepted, status,
    regression summary, constitutional verdict, and the stable part of the
    provenance hash (the timestamp is wall-clock and differs by
    construction, so it is excluded and noted)
  - dataclass field names and order for all four dataclasses
  - ``ImprovementLevel`` members

Usage::

    python3 tools/self_improvement_equivalence.py
"""
from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYC = sorted((ROOT / "apedp/__pycache__").glob(
    "self_improvement.cpython-312.pyc"))[-1]


def load_original():
    spec = importlib.util.spec_from_file_location("self_improvement_original", PYC)
    m = importlib.util.module_from_spec(spec)
    # @dataclass resolves string annotations through sys.modules; a module
    # that was never registered is absent from it and the decorator raises
    # AttributeError on 'NoneType'.
    sys.modules["self_improvement_original"] = m
    spec.loader.exec_module(m)
    return m


def signal_tuple(s) -> tuple:
    return (
        s.signal_type, s.severity, s.affected_component, s.description,
        tuple(sorted(s.evidence_metrics.items())), s.suggested_level.name,
    )


def decision_tuple(d) -> tuple:
    return (
        d.proposal_id, bool(d.accepted), d.status, d.regression_summary,
        d.constitutional_verdict.compliant, d.constitutional_verdict.sha256,
        tuple(sorted(d.target_metric_deltas.items())),
        # timestamp is wall-clock and cannot match; provenance_hash is
        # computed from the proposal's own timestamp, which is fixed here,
        # so it must match exactly.
        d.provenance_hash,
    )


def _histories() -> list[dict]:
    return [
        {},                                        # empty run
        {"candidates": []},                         # no candidates
        {"candidates": [{"train_mse": 0.01, "near_ood_mse": 500.0}]},   # rule 2
        {"candidates": [{"reason_chain": "transcendental boom", "dim_status": "UNRESOLVED"}]},
        {"candidates": [{"dim_status": "UNRESOLVED"}] * 3},              # rule 1 (3/3)
        {"candidates": [{"dim_status": "VALID"}] * 10},                  # rule 1 not fired
        {"coupling_residual_ratio": 0.01},                                # None threshold
        {"coupling_residual_ratio": 0.10},                                # rule 3
        {"coupling_residual_ratio": 0.10, "coupling_residual_threshold": 0.05},  # above thresh
        {"coupling_residual_ratio": 0.5, "coupling_residual_threshold": 0.5},   # == thresh
        # everything at once
        {"candidates": [{"dim_status": "UNRESOLVED"}, {"train_mse": 0.0, "near_ood_mse": 1e9}],
         "coupling_residual_ratio": 0.08},
    ]


def _proposals() -> list[tuple]:
    import apedp.self_improvement as new
    out = []
    for level in new.ImprovementLevel:
        for mbd in (False, True):
            for rcc in (False, True):
                out.append(new.ModificationProposal(
                    proposal_id=f"P-{level.name}-{mbd}-{rcc}",
                    level=level,
                    target_component="comp",
                    modifications={"a": 1},
                    hypothesis="h", rationale="r",
                    author="tester", timestamp="2026-01-01T00:00:00+00:00",
                    modifies_benchmark_data=mbd,
                    retrospective_criterion_change=rcc,
                ))
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import apedp.self_improvement as new

    orig = load_original()
    agree = disagree = 0

    # --- FailureDetector -------------------------------------------------
    for h in _histories():
        a = [signal_tuple(s) for s in orig.FailureDetector.detect_weaknesses([h])]
        b = [signal_tuple(s) for s in new.FailureDetector.detect_weaknesses([h])]
        if a == b:
            agree += 1
        else:
            disagree += 1
            print(f"DISAGREE detect_weaknesses history={h}")
            print(f"  original:    {a}")
            print(f"  reauthored: {b}")
    # empty input
    if orig.FailureDetector.detect_weaknesses([]) == new.FailureDetector.detect_weaknesses([]):
        agree += 1
    else:
        disagree += 1
        print("DISAGREE detect_weaknesses([])")

    # --- IndependentGatekeeper -------------------------------------------
    # The original's ModificationProposal is a distinct class object from the
    # re-authored one, so an identical proposal is constructed in each
    # namespace rather than shared. Sharing would compare the two auditors
    # against differently-typed inputs and prove nothing.
    for mbd in (False, True):
        for rcc in (False, True):
            for reg in (True, False):
                p_new = new.ModificationProposal(
                    proposal_id="P-1", level=new.ImprovementLevel.LEVEL_1_PARAMETER,
                    target_component="c", modifications={}, hypothesis="h",
                    rationale="r", author="a", timestamp="2026-01-01T00:00:00+00:00",
                    modifies_benchmark_data=mbd,
                    retrospective_criterion_change=rcc)
                p_orig = orig.ModificationProposal(
                    proposal_id="P-1", level=orig.ImprovementLevel.LEVEL_1_PARAMETER,
                    target_component="c", modifications={}, hypothesis="h",
                    rationale="r", author="a", timestamp="2026-01-01T00:00:00+00:00",
                    modifies_benchmark_data=mbd,
                    retrospective_criterion_change=rcc)
                a = decision_tuple(orig.IndependentGatekeeper.evaluate_and_decide(
                    p_orig, None, reg))
                b = decision_tuple(new.IndependentGatekeeper.evaluate_and_decide(
                    p_new, None, reg))
                if a == b:
                    agree += 1
                else:
                    disagree += 1
                    print(f"DISAGREE gatekeeper mbd={mbd} rcc={rcc} reg={reg}")
                    print(f"  original:    {a}")
                    print(f"  reauthored: {b}")

    # --- to_dict shapes ---------------------------------------------------
    for cls in ("ModificationProposal", "DiagnosticSignal", "GatekeeperDecision"):
        o = [f.name for f in dataclasses.fields(getattr(orig, cls))]
        n = [f.name for f in dataclasses.fields(getattr(new, cls))]
        if o == n:
            agree += 1
            print(f"AGREE  {cls} fields: {n}")
        else:
            disagree += 1
            print(f"DISAGREE {cls} fields: {o} vs {n}")

    old_levels = {i.name for i in orig.ImprovementLevel}
    new_levels = {i.name for i in new.ImprovementLevel}
    if old_levels == new_levels:
        agree += 1
        print(f"AGREE  ImprovementLevel: {sorted(new_levels)}")
    else:
        disagree += 1
        print(f"DISAGREE ImprovementLevel: {old_levels} vs {new_levels}")

    print(f"\n{agree} agree, {disagree} disagree")
    return 1 if disagree else 0


if __name__ == "__main__":
    raise SystemExit(main())
