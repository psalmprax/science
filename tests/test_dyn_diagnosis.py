"""Diagnostic abstention composer: machine-readable WHY, not bare abstention."""
from __future__ import annotations

from apedp_dyn.epistemic.adjudicate import EpistemicStatus
from apedp_dyn.epistemic.diagnosis import diagnose
from apedp_dyn.epistemic.transfer_scope import TransferScope


class _FakePromoted:
    def __init__(self, cid):
        self.candidate = type("C", (), {"candidate_id": cid})()


class _FakeAdj:
    def __init__(self, **kw):
        self.status = kw.get("status", EpistemicStatus.MULTI_HYPOTHESIS)
        self.epistemic_attribution = kw.get(
            "attribution", "intrinsic_observational_nonidentifiability")
        self.promoted = kw.get("promoted", None)
        self.best_rmse = kw.get("best_rmse", 10.71)
        self.second_rmse = kw.get("second_rmse", 14.36)
        self.delta_rmse = kw.get("delta_rmse", -3.65)
        self.tau_epi = kw.get("tau_epi", 11.9)
        self.contract = kw.get("contract", "full")
        self.reason = kw.get("reason", "rival within tau")
        self.ranking = kw.get("ranking", [])


def test_multi_block_shape():
    d = diagnose(_FakeAdj())
    assert d["status"] == "multi_hypothesis"
    assert d["diagnosis"]["separation"]["insufficient"] is True
    assert d["transfer"] == "NOT_ESTABLISHED"
    assert "unique_structure" in d["claim_boundary"]["unsupported"]
    assert "observed_regime" in d["claim_boundary"]["supported"]


def test_promoted_block_shape():
    d = diagnose(_FakeAdj(status=EpistemicStatus.UNIQUE_SUPPORT,
                          promoted=_FakePromoted("hp_true")))
    assert d["diagnosis"]["separation"]["insufficient"] is False
    assert d["diagnosis"]["evidence"] == ["hp_true"]


def test_transfer_composes():
    scope = TransferScope(candidate_id="hp_global", rival_id="constant_baseline",
                          interp_rmse_best=3.56, interp_rmse_rival=10.53,
                          extra_rmse_best=93.80, extra_rmse_rival=14.30)
    d = diagnose(_FakeAdj(), transfer=scope)
    assert d["transfer"]["transfer"]["status"] == "cross_domain_failed"
    assert d["transfer"]["transfer"]["leadership_reversed"] is True


def test_full_state_unifies_dimensions():
    from apedp_dyn.epistemic.diagnosis import full_state
    from apedp_dyn.epistemic.transfer_scope import TransferScope
    scope = TransferScope(candidate_id="hp_global", rival_id="constant_baseline",
                          interp_rmse_best=3.56, interp_rmse_rival=10.53,
                          extra_rmse_best=93.80, extra_rmse_rival=14.30)
    stab = {"distribution": {"piecewise_hp_inv": 1.0}, "entropy_nats": 0.0}
    d = full_state(_FakeAdj(), transfer=scope, stability=stab)
    assert d["status"] == "multi_hypothesis"
    assert d["discovery"]["structural_recurrence"] == 1.0
    assert d["transfer"]["transfer"]["status"] == "cross_domain_failed"
    assert "unique_structure" in d["claim_boundary"]["unsupported"]


def test_full_state_without_stability():
    from apedp_dyn.epistemic.diagnosis import full_state
    d = full_state(_FakeAdj())
    assert d["discovery"] == "NOT_ASSESSED"
    assert d["transfer"] == "NOT_ESTABLISHED"
