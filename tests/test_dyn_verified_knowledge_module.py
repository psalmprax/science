"""Pins for the Verified Knowledge Module Library (Gap-3, ledger-mapped).

Acceptance criteria:
  K1  Evidence gating: registration refuses candidates whose experiment-ledger
      state is not CONFIRMED_VALID, and candidates whose promotion tracker has
      not passed STAGE_4_OUT_OF_SAMPLE_TRANSFER — error names the gate.
  K2  Metadata integrity: preconditions required; validity ranges must be
      finite and ordered; uncertainty needs value + method (no naked floats);
      counterexamples REQUIRED non-empty — a module without a documented
      failure is not verified knowledge.
  K3  Dependency closure: registration refuses unknown dependencies.
  K4  Composition: validity domains INTERSECT on shared inputs (safe subset
      exported); disjoint domains refuse; dependencies must be in the chain.
  K5  Export determinism + disclosure: deterministic ordering; every record
      carries the stage-5/6 non-inclusion disclosure (economic stages are NOT
      implied by library entry).
"""

from __future__ import annotations

import json

import pytest

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    FullLineageExperimentLedger,
)
from apedp_dyn.epistemic.promotion_lifecycle import (
    PromotionStage,
    PromotionTracker,
)
from apedp_dyn.epistemic.verified_knowledge_module import (
    ModuleCompositionError,
    ModuleIntegrityError,
    ValidatedModuleSpec,
    VerifiedModuleLibrary,
)


def _confirmed_candidate(ledger: FullLineageExperimentLedger, cid: str, formula: str):
    ledger.register_candidate(cid, proposer_id="test_proposer", formula=formula)
    ledger.update_lifecycle_state(cid, CandidateLifecycleState.CONFIRMED_VALID)
    return ledger.get_candidate(cid)


def _tracker_through_stage_4(cid: str = "c1") -> PromotionTracker:
    t = PromotionTracker(candidate_id=cid)
    t.verify_stage_1(lean_status="P2_VERIFIED", dimensions_consistent=True, bounded=True)
    t.verify_stage_2(in_sample_ic=0.05, cv_stable=True)
    t.verify_stage_3(dm_p_value=0.001, max_literature_r2=0.4)
    # Stage 4 real signature: (in_sample_ic, oos_ic, oos_p_value, ...); passes
    # with retention 0.60 >= MIN_OOS_TRANSFER_RETENTION 0.50, oos IC 0.06 >= 0.015.
    t.verify_stage_4(in_sample_ic=0.10, oos_ic=0.06, oos_p_value=0.001)
    return t


def _spec(mid: str = "m1", cid: str = "c1", **kw) -> ValidatedModuleSpec:
    base = dict(
        module_id=mid,
        candidate_id=cid,
        formula="y = a0 + a1*x",
        preconditions=("unit_interval_inputs",),
        validity_domain={"x": (0.0, 1.0)},
        uncertainty={"oos_rmse": {"value": 0.05, "method": "sealed_holdout_bootstrap"}},
        counterexamples=("diverges for x > 1.0 (documented, domain-closed)",),
        dependencies=(),
    )
    base.update(kw)
    return ValidatedModuleSpec(**base)


@pytest.fixture()
def artifact(tmp_path):
    led = FullLineageExperimentLedger(persistence_path=tmp_path / "led.jsonl")
    _confirmed_candidate(led, "c1", "y = a0 + a1*x")
    return led


def test_refusal_unconfirmed_ledger_state(tmp_path):  # K1a
    led = FullLineageExperimentLedger(persistence_path=tmp_path / "led.jsonl")
    led.register_candidate("c1", proposer_id="p", formula="y=x")
    lib = VerifiedModuleLibrary()
    with pytest.raises(ModuleIntegrityError) as ei:
        lib.register_module(_spec(), led, _tracker_through_stage_4())
    assert "confirmed_valid" in str(ei.value)


def test_refusal_below_stage_4(artifact):  # K1b
    t = PromotionTracker(candidate_id="c1")
    t.verify_stage_1(lean_status="P2_VERIFIED", dimensions_consistent=True, bounded=True)
    t.verify_stage_2(in_sample_ic=0.05, cv_stable=True)
    t.verify_stage_3(dm_p_value=0.001, max_literature_r2=0.4)
    lib = VerifiedModuleLibrary()
    with pytest.raises(ModuleIntegrityError) as ei:
        lib.register_module(_spec(), artifact, t)
    assert "stage 4" in str(ei.value)
    assert PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER == 4


def test_registration_success_gated(artifact):  # K1 positive path
    lib = VerifiedModuleLibrary()
    reg = lib.register_module(_spec(), artifact, _tracker_through_stage_4())
    assert reg.evidence["experiment_node_state"] == "confirmed_valid"
    assert "STAGE_4" in reg.evidence["promotion_stage_passed"]
    assert "NOT required" in reg.evidence["disclosure"]


@pytest.mark.parametrize("bad", [
    {"preconditions": ()},
    {"validity_domain": {"x": (1.0, 0.0)}},
    {"uncertainty": {"oos_rmse": {"value": 0.05}}},
    {"counterexamples": ()},
])
def test_metadata_integrity_refusals(artifact, bad):  # K2
    lib = VerifiedModuleLibrary()
    with pytest.raises(ModuleIntegrityError):
        lib.register_module(_spec(**bad), artifact, _tracker_through_stage_4())


def test_dependency_closure_register(artifact):  # K3
    lib = VerifiedModuleLibrary()
    lib.register_module(_spec(mid="m0"), artifact, _tracker_through_stage_4())
    with pytest.raises(ModuleIntegrityError) as ei:
        lib.register_module(_spec(mid="m1", dependencies=("ghost",)), artifact, _tracker_through_stage_4())
    assert "ghost" in str(ei.value)
    # after m0 is registered, the dependency resolves
    reg = lib.register_module(_spec(mid="m1", dependencies=("m0",)), artifact, _tracker_through_stage_4())
    assert reg.spec.module_id == "m1"


def test_composition_intersects_domain():
    lib = VerifiedModuleLibrary()

    class _Stub(VerifiedModuleLibrary):
        def register_module(self, spec, ledger, promotion):  # bypass gates for composition pins
            self._modules[spec.module_id] = type(
                "R", (), {"spec": spec, "evidence": {}, "registered_at": "x"},
            )()
            return self._modules[spec.module_id]

    lib = _Stub()
    lib.register_module(_spec(mid="a", validity_domain={"x": (0.0, 2.0)}), None, None)
    lib.register_module(_spec(mid="b", validity_domain={"x": (1.0, 3.0)}), None, None)
    out = lib.check_compatible(["a", "b"])
    assert out["validity_domain"]["x"] == [1.0, 2.0]  # K4 intersection


def test_composition_refuses_disjoint_domain():
    lib = VerifiedModuleLibrary()

    class _Stub(VerifiedModuleLibrary):
        def register_module(self, spec, ledger, promotion):
            self._modules[spec.module_id] = type(
                "R", (), {"spec": spec, "evidence": {}, "registered_at": "x"},
            )()
            return self._modules[spec.module_id]

    lib = _Stub()
    lib.register_module(_spec(mid="a", validity_domain={"x": (0.0, 1.0)}), None, None)
    lib.register_module(_spec(mid="b", validity_domain={"x": (2.0, 3.0)}), None, None)
    with pytest.raises(ModuleCompositionError) as ei:
        lib.check_compatible(["a", "b"])
    assert "DISJOINT" in str(ei.value)


def test_composition_refuses_unclosed_dependencies():
    lib = VerifiedModuleLibrary()

    class _Stub(VerifiedModuleLibrary):
        def register_module(self, spec, ledger, promotion):
            self._modules[spec.module_id] = type(
                "R", (), {"spec": spec, "evidence": {}, "registered_at": "x"},
            )()
            return self._modules[spec.module_id]

    lib = _Stub()
    lib.register_module(_spec(mid="a", dependencies=("b",)), None, None)
    lib.register_module(_spec(mid="b"), None, None)
    with pytest.raises(ModuleCompositionError) as ei:
        lib.check_compatible(["a"])  # b exists but not in the chain
    assert "not in the chain" in str(ei.value)
    ok = lib.check_compatible(["a", "b"])  # closed chain composes
    assert ok["chain"] == ["a", "b"]


def test_export_deterministic_and_disclosed(artifact):  # K5
    lib = VerifiedModuleLibrary()
    lib.register_module(_spec(mid="b"), artifact, _tracker_through_stage_4())
    lib.register_module(_spec(mid="a", candidate_id="c1"), artifact, _tracker_through_stage_4())
    e1 = lib.export()
    e2 = lib.export()
    assert json.dumps(e1) == json.dumps(e2)
    assert [m["module_id"] for m in e1] == ["a", "b"]
    for m in e1:
        assert "NOT required" in m["evidence"]["disclosure"]
        assert m["counterexamples"], "export carries documented failures"
