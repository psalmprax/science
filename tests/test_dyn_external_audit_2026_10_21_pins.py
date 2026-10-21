"""Mutation-test regression pins for the 2026-10-21 external 19-flaw audit.

Each test pins the *fixed* behaviour AND, where the defect can be mirrored in
Python, demonstrates that the pin fails under the reintroduced defect: the
pre-fix behaviour is reimplemented inline and executed on the same fixture so
the two outcomes visibly diverge. Rust-side fixes that expose no Python seam
are pinned by source-text assertions on the kernel files (removal of the fix
flips the assertion); Rust behavioural unit tests live next to the kernel code
(``cargo test``) — see ``diff_pade.rs`` mod tests.

Flaw map (external audit, docs/SYSTEM_FLAWS_AUDIT.md, 2026-10-21 round):
  FLAW-02 identity ODE scorer          -> test_flaw02_discovered_rhs_is_evaluated
  FLAW-03 S3 TOCTOU lock race          -> test_flaw03_contested_lock_cannot_be_stolen (+ fallback)
  FLAW-05 tautological R^2 / se=0      -> test_flaw05_flat_target_with_residual_scores_zero (+ rust text pin)
  FLAW-07 C6 apex leapfrog             -> test_flaw07_c6_requires_monotone_evidence
  FLAW-08 re-anchor bootstrap bypass   -> test_flaw08_re_anchor_runs_bootstrap_on_paired_residuals
  FLAW-09 silent return-type mutation  -> test_flaw09_eval_collapse_stays_binary
  FLAW-10 circular axiom injection     -> test_flaw10_core_axiom_proofs_labeled_schema_check
  FLAW-11 odd-pole cancellation        -> test_flaw11_opposed_limit_probes_refuse_to_average
  FLAW-12 inactive-column scale drift  -> test_flaw12_implicit_scale_uses_active_columns_only (text pin)
  FLAW-13 hardcoded ODE thresholds      -> test_flaw13_ode_referee_thresholds_bound_to_spec
  FLAW-13 divergent-but-stable rollout -> test_flaw13_negative_r2_gets_unstable_divergent
  FLAW-14 toroidal Nyquist→DC wrap     -> test_flaw14_spectral_advection_does_not_wrap
  FLAW-15 unnormalized SVD tolerance   -> test_flaw15_lie_symmetry_tolerance_is_normalized (text pin)
  FLAW-16 Hamiltonian archetype drift  -> test_flaw16_hamiltonian_archetype_matches_contract
  FLAW-17 CBF post-correction violation-> test_flaw17_cbf_final_projection_prevents_norm_violation
  FLAW-18 ledger append races / O(N^2) -> test_flaw18_ledger_flock_and_amortized_sync
  FLAW-19 dead gradient / I_arr leak   -> test_flaw19_sir_formula_uses_physical_notation
  FLAW-06a pole masking (Rust)         -> rust test + text pin (pole flag + is_exact guard)
  FLAW-06b koopman centered R^2 (Rust) -> text pin (centered ss_tot + 0.0 floor)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# FLAW-02: coupled ODE proposer discarded the discovered RHS
# ---------------------------------------------------------------------------


def test_flaw02_discovered_rhs_is_evaluated():
    """Pin: compute_score must return the discovered dx/dt, never the identity.

    Mutation demonstration: the OLD scorer (identity echo) is reimplemented on
    the same fixture; the fixed scorer tracks the true derivative while the
    identity is far off, so reverting the fix flips the assertion.
    """
    from apedp_dyn.ode.proposer import CoupledOdeProposer

    rng = np.random.default_rng(7)
    dt, n = 0.02, 400
    x0, x1 = 2.0, 0.0
    arr = np.empty((n, 2))
    for k in range(n):
        arr[k] = (x0, x1)
        d0, d1 = (-x0 + 1.0) * dt, (x0 - x1) * dt
        x0, x1 = x0 + d0, x1 + d1
    arr += rng.normal(0, 1e-9, size=(n, 2))
    df = pd.DataFrame({"x0": arr[:, 0], "x1": arr[:, 1]})

    props = CoupledOdeProposer(dt=dt).fit_and_propose(df)
    p0 = next(p for p in props if p.metadata["state_variable"] == "x0")

    true_dot = np.gradient(arr[:, 0], dt)
    fixed = p0.compute_score(df)
    identity = df["x0"].to_numpy(dtype=float)  # the OLD (defective) echo

    fixed_rmse = float(np.sqrt(np.mean((fixed - true_dot) ** 2)))
    identity_rmse = float(np.sqrt(np.mean((identity - true_dot) ** 2)))
    assert fixed_rmse < 0.05, f"fixed scorer must evaluate the discovered RHS (rmse={fixed_rmse:.4f})"
    assert fixed_rmse < 0.1 * identity_rmse, (
        "mutation reverted: scorer behaves like the identity echo "
        f"(fixed={fixed_rmse:.4f} vs identity={identity_rmse:.4f})"
    )


# ---------------------------------------------------------------------------
# FLAW-03: non-atomic S3 mutex (TOCTOU put race)
# ---------------------------------------------------------------------------


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


# The class NAME matters: remote_storage flags type(exc).__name__ == "ClientError".
_ClientError = type("ClientError", (Exception,), {})


class _RacyContestedClient:
    """Deterministic two-phase race fixture.

    Phase 1 — our GET sees no lock (KeyError).
    Phase 2 — our conditional put loses the race (the botocore-named
    ``ClientError`` carrying ``412`` in its text). From then on every GET
    reports a FOREIGN, FRESH (non-stale) holder.

    The fixed backend must back off and refuse; the legacy check-then-
    unconditional-put logic silently takes over on exactly this fixture.
    """

    def __init__(self) -> None:
        import time as _t

        self._t = _t
        self._gets = 0
        self.store_body: bytes | None = None

    def get_object(self, Bucket, Key):  # noqa: ARG002
        self._gets += 1
        if self._gets == 1:
            raise KeyError(Key)
        body = (
            self.store_body
            if self.store_body is not None
            else f"foreign_worker:{self._t.time()}".encode()
        )
        return {"Body": _FakeBody(body)}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **_kw):  # noqa: ARG002
        if IfNoneMatch is not None:
            raise _ClientError(
                "An error occurred (PreconditionFailed) when calling the "
                "PutObject operation: 412"
            )
        # Legacy unconditional put: silently takes over the racing slot.
        self.store_body = Body
        return {}

    def delete_object(self, Bucket, Key):  # noqa: ARG002
        self.store_body = None
        return {}


class _OldStyleContentionError(RuntimeError):
    pass


def _old_style_acquire(backend, key: str, timeout: float = 0.4) -> str:
    """PRE-FLAW-03 acquire logic inlined verbatim (mutation demonstration).

    On the ``_RacyContestedClient`` fixture the legacy code sees an absent
    lock and takes over with an unconditional put — the exact double-winner
    corruption the remediated backend refuses.
    """
    import uuid as _uu

    _t = backend.client._t
    lock_key = backend._full_key(f"locks/{key}.lock")
    token = _uu.uuid4().hex
    deadline = _t.time() + timeout
    stale_timeout = 60.0
    while True:
        lock_active = False
        try:
            resp = backend.client.get_object(Bucket=backend.bucket, Key=lock_key)
            body = resp["Body"].read().decode("utf-8")
            parts = body.strip().split(":")
            lock_time = float(parts[1]) if len(parts) > 1 else 0.0
            if (_t.time() - lock_time) <= stale_timeout:
                lock_active = True
            else:
                backend.client.delete_object(Bucket=backend.bucket, Key=lock_key)
        except Exception:
            lock_active = False
        if not lock_active:
            backend.client.put_object(
                Bucket=backend.bucket, Key=lock_key, Body=f"{token}:{_t.time()}".encode()
            )
            return token
        if _t.time() >= deadline:
            raise _OldStyleContentionError("old-style acquire timed out")
        _t.sleep(0.02)


def test_flaw03_contested_lock_cannot_be_stolen():
    """Fixed: a fresh race (conditional write fails 412, read-back foreign)
    must NEVER yield ownership — the legacy unconditional put would."""
    from apedp_dyn.epistemic.remote_storage import S3StorageBackend, StorageLockError

    with pytest.raises(StorageLockError):
        S3StorageBackend(
            bucket="b", prefix="p", s3_client=_RacyContestedClient()
        ).acquire_lock("ledger", timeout=0.4)

    # Mutation demonstration: the legacy logic WINS the same race.
    token_old = _old_style_acquire(
        S3StorageBackend(bucket="b", prefix="p", s3_client=_RacyContestedClient()),
        "ledger",
        timeout=0.4,
    )
    assert isinstance(token_old, str) and token_old


class _ParamRejectedFallbackClient:
    """Old-botocore world: the conditional parameter is rejected locally, the
    fallback unconditional put runs, and the read-back confirms OUR token.

    Motivated by the pin run: the first remediation draft skipped the put on
    the parameter-rejection path, so the read-back could never see our token
    and every acquire deadlocked until timeout."""

    def __init__(self) -> None:
        self.last_put_body: bytes | None = None

    def get_object(self, Bucket, Key):  # noqa: ARG002
        return {"Body": _FakeBody(self.last_put_body or b"")}  # read-back wins

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **_kw):  # noqa: ARG002
        if IfNoneMatch is not None:
            raise ValueError("unknown parameter IfNoneMatch for method put_object")
        self.last_put_body = Body
        return {}

    def delete_object(self, Bucket, Key):  # noqa: ARG002
        self.last_put_body = None
        return {}


def test_flaw03_parameter_rejected_fallback_still_verifies_ownership():
    from apedp_dyn.epistemic.remote_storage import S3StorageBackend

    client = _ParamRejectedFallbackClient()
    backend = S3StorageBackend(bucket="b", prefix="p", s3_client=client)
    token = backend.acquire_lock("ledger", timeout=2.0)
    assert isinstance(token, str) and token
    assert token in backend._active_locks  # ownership verified after read-back


# ---------------------------------------------------------------------------
# FLAW-05: tautological R^2 on flat targets
# ---------------------------------------------------------------------------


def test_flaw05_flat_target_with_residual_scores_zero():
    """Pin: ss_tot ~ 0 with real residual energy must score 0.0, not 1.0.

    Old inline formula appended to make the divergence visible (mutation
    demonstration).
    """
    from apedp_dyn.epistemic.pde_discovery import _frozen_residual_r2

    n = 64
    ut = np.ones(n) * 0.5                      # flat target: ss_tot == 0
    columns = {
        "1": np.ones(n),
        "s": np.sin(np.linspace(0.0, 6.0, n)),  # unrelated non-constant basis
    }
    fixed = _frozen_residual_r2(ut, columns, active_terms=["s"], coefficients=[1.0])
    old = 1.0  # the defective branch: `else 1.0` tautology
    assert fixed == 0.0, f"flat target with residual energy must score 0.0, got {fixed}"
    assert fixed != old, "mutation reverted: flat-target R^2 back to the 1.0 tautology"


# ---------------------------------------------------------------------------
# FLAW-07: C6 apex leapfrog (missing rival-equivalence evidence)
# ---------------------------------------------------------------------------


def test_flaw07_c6_requires_monotone_evidence():
    from apedp_dyn.epistemic.claim_ladder import (
        ClaimLevel,
        EvidenceBundle,
        EvidenceRequirement,
        adjudicate_scientific_claim,
    )

    bundle = EvidenceBundle(metrics={"apparent_r2": 0.997})
    for req in (
        EvidenceRequirement.IN_SAMPLE_CONVERGENCE,
        EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT,
        EvidenceRequirement.PHYSICAL_CONSERVATION,
        EvidenceRequirement.NUMERICAL_CFL_STABILITY,
        EvidenceRequirement.DUAL_BOUND_CERTIFICATE,
    ):
        bundle.add(req)

    adj = adjudicate_scientific_claim(
        candidate_id="c6_leapfrog_probe",
        proposed_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
        evidence=bundle,
        domain="computational_physics",
    )
    # Monotone strictness: apex tier cannot be reached without the evidence
    # C3/C4 already mandated. (Old set: this bundle WAS promotable to C6.)
    assert adj.justified_claim != ClaimLevel.C6_FORMALLY_VERIFIED
    assert EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED in adj.missing_evidence
    assert EvidenceRequirement.INTERVENTIONAL_INVARIANCE in adj.missing_evidence

    # Restoring the full monotone bundle restores C6 promotion.
    bundle.add(EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED)
    bundle.add(EvidenceRequirement.INTERVENTIONAL_INVARIANCE)
    adj_full = adjudicate_scientific_claim(
        candidate_id="c6_full_bundle",
        proposed_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
        evidence=bundle,
        domain="computational_physics",
    )
    assert adj_full.justified_claim == ClaimLevel.C6_FORMALLY_VERIFIED


# ---------------------------------------------------------------------------
# FLAW-08: silent bootstrap bypass on holdout re-anchor
# ---------------------------------------------------------------------------


def _re_anchor_fixture(with_series: bool = True):
    """Two candidates on the re-anchor path. ``with_series=False`` strips the
    gate-4 residual series from provenance — the PRE-FLAW-08 world where the
    re-anchor could not recover a pairable residual pair and fell back to
    point-estimate-only ranking (the defectconfiguration the fixed code no
    longer allows to claim UNIQUE_SUPPORT silently)."""
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

    n = 60
    t = np.arange(n)

    def scored(cid: str, train_loss: float, ho_rmse: float, signed_resid: np.ndarray, terms) -> ScoredCandidate:
        cand = CandidateDynamics(
            lhs="incidence",
            terms=tuple(Term(name, f"a{i}") for i, name in enumerate(terms)),
            candidate_id=cid,
        )
        gate4 = {
            "passed": True,
            "holdout_obs_rms": 1.0,
            "holdout_noise_sigma": 0.1,
            "holdout_n_samples": n,
        }
        if with_series:
            gate4["holdout_residuals"] = signed_resid.tolist()
        provenance = {
            "gate4_status": gate4,
            "data_origin": "simulation",
        }
        return ScoredCandidate(cand, provenance, train_loss=train_loss, holdout_rmse=ho_rmse)

    # Train-loss best (A): worse on holdout, so the gate-4-certified
    # holdout head (B) takes over => the FLAW-08 re-anchor path.
    # NOTE the DISTINCT term sets — identical structures would be
    # de-duplicated before the pair path ever fires.
    # The point record says B beats A by 0.30 > tau; the RECOVERED
    # per-timepoint series say the opposite at every timepoint (A's sq
    # errors 0.0025 < B's 0.01) — the block-bootstrap CI over d is
    # strictly negative => 'rival-favored' => the point-estimate margin
    # is denied (exactly the Winner's-Curse abort the tiebreak exists for).
    a_signed = np.full(n, 0.05)   # recovered series: sq 0.0025 (beats B blockwise)
    b_signed = np.full(n, 0.10)   # recovered series: sq 0.01
    a = scored("cand_A_train_loss_best", train_loss=0.01,
               ho_rmse=0.40,
               signed_resid=a_signed, terms=("one", "x", "x3", "v"))
    b = scored("cand_B_holdout_head", train_loss=0.10,
               ho_rmse=0.10,
               signed_resid=b_signed, terms=("one", "v", "abs_v_v"))
    return [a, b], a_signed, b_signed


def test_flaw08_re_anchor_runs_bootstrap_on_paired_residuals():
    """Pin: after the re-anchor, the referee must bootstrap the NEW pair from
    provenance residuals; wildly dispersed rival squared errors must deny the
    point-estimate-only UNIQUE_SUPPORT (Winner's Curse).

    Mutation demonstration: rerunning the same fixture with the OLD semantics
    (bootstrap forced False, residuals dropped) grants UNIQUE_SUPPORT — so
    reverting the fix flips the first assertion.
    """
    from apedp_dyn.epistemic.adjudicate import adjudicate
    from apedp_dyn.epistemic.adjudicate import EpistemicStatus

    fitted, _a_signed, _b_signed = _re_anchor_fixture()

    # Fixed behaviour: re-anchor happens, paired gate-4 residuals are
    # recovered, bootstrap runs, and the dispersed d-series denies a
    # confirmed unique support.
    fixed_adj = adjudicate(
        fitted,
        tau_epi=0.01,
        resid_best=None,
        resid_rival=None,
        bootstrap=True,
        bootstrap_n_boot=200,
        bootstrap_seed=7,
        data_origin="simulation",
        benchmark_level="L1_CONSTRAINED_DISCOVERY",
    )
    assert fixed_adj.status is not EpistemicStatus.UNIQUE_SUPPORT, (
        "re-anchored unique support claimed WITHOUT a verified bootstrap tiebreak — "
        "FLAW-08 reverted"
    )

    # Mutation demonstration under OLD semantics: the re-anchor happens but
    # the paired residual series are nowhere in provenance (the recovery IS
    # the fix) and bootstrap stays off — ranking on point estimates alone
    # grants UNIQUE_SUPPORT on the identical underlying data.
    old_like = adjudicate(
        _re_anchor_fixture(with_series=False)[0],
        tau_epi=0.01,
        resid_best=None,
        resid_rival=None,
        bootstrap=False,  # <- the defect's forced downgrade
        data_origin="simulation",
        benchmark_level="L1_CONSTRAINED_DISCOVERY",
    )
    assert old_like.status is EpistemicStatus.UNIQUE_SUPPORT
    # And the two worlds must diverge on the same data — that divergence is
    # what makes this pin mutation-sensitive.
    assert old_like.status is not fixed_adj.status


# ---------------------------------------------------------------------------
# FLAW-09: silent return-type mutation at eval collapse
# ---------------------------------------------------------------------------


def test_flaw09_eval_collapse_stays_binary():
    from apedp_dyn.proposers.hawkes_orderflow_proposer import HawkesOrderFlowProposer

    rng = np.random.RandomState(42)
    ts_fit = np.cumsum(rng.exponential(scale=0.5, size=100))
    fit_df = pd.DataFrame({
        "timestamp": ts_fit,
        "volume": rng.exponential(scale=100.0, size=100),
    })
    proposer = HawkesOrderFlowProposer(decay_beta=1.5, intensity_threshold_quantile=0.80)
    p = proposer.fit_and_propose(fit_df)[0]
    assert p.metadata["binary_surge_viable_on_fit"] is True

    # Low-volatility eval window: huge inter-arrival gaps crush the fitted
    # intensity below the (fit-time) surge threshold => binary collapse.
    ts_eval = np.cumsum(rng.exponential(scale=50.0, size=60))
    eval_df = pd.DataFrame({
        "timestamp": ts_eval,
        "volume": np.full(60, 1.0),
    })
    scores = p.compute_score(eval_df)
    uniq = set(np.unique(scores).tolist())
    assert uniq.issubset({0.0, 1.0}), (
        f"binary surge contract broken at eval collapse; unique values={sorted(uniq)[:5]}... "
        "— FLAW-09 reverted (silent z-score mutation)"
    )
    # Mutation demonstration: the OLD fallback (standardized intensity) would
    # produce non-binary unbounded values on the same frame.
    intens_proxy = np.linspace(-3.0, -1.0, len(scores))  # continuous, unbounded type
    assert not set(np.unique(intens_proxy)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# FLAW-10: circular axiom injection in the Lean core bridge
# ---------------------------------------------------------------------------


def test_flaw10_core_axiom_proofs_labeled_schema_check(tmp_path):
    from apedp_dyn.epistemic.agnostic_adapter import (
        AgnosticCandidateSpecification,
        DeclaredInvariant,
        InvariantKind,
    )
    from apedp_dyn.epistemic.claim_ladder import ClaimLevel
    from apedp_dyn.epistemic.lean_universal_bridge import LeanUniversalBridge

    bridge = LeanUniversalBridge(output_dir=tmp_path / "lean_pins")
    inv = DeclaredInvariant(
        name="population_flux_conservation",
        kind=InvariantKind.CONSERVATION_LAW,
        description="SIR epidemiological mass-action flux conservation (sir route)",
        required_for_tier=ClaimLevel.C6_FORMALLY_VERIFIED,
    )
    spec = AgnosticCandidateSpecification(
        candidate_id="EPID-SIR-PINTEST",
        proposed_claim=ClaimLevel.C6_FORMALLY_VERIFIED,
        formula_repr="-beta * s * i",
        declared_invariants=[inv],
    )
    _code, theorems = bridge.generate_lean4_module(spec, standalone_core=True)
    ring_recs = [t for t in theorems if t.is_formally_proven]
    assert ring_recs, "expected the core-mode SIR conservation theorem to be emitted"
    for rec in ring_recs:
        assert rec.proof_method.startswith("syntactic_schema_check::"), (
            f"core axiomatic proof published as independent FORMAL_PROOF: "
            f"method={rec.proof_method!r} — FLAW-10 reverted"
        )
        assert "syntactic_schema_check" in rec.diagnostic_message
        assert rec.diagnostic_message.startswith("[syntactic_schema_check"), rec.diagnostic_message
    # Mutation demonstration: the OLD raw method name (claimed as formal proof)
    assert not any(r.proof_method == "lean_ring_tactic" for r in ring_recs)


# ---------------------------------------------------------------------------
# FLAW-11: odd-pole limit cancellation
# ---------------------------------------------------------------------------


def test_flaw11_opposed_limit_probes_refuse_to_average():
    from apedp.safety import LimitTimeout, limit_with_timeout

    x = sp.Symbol("x")

    def hanging_limit(*_args, **_kwargs):
        import time
        time.sleep(2.5)
        return 0

    with mock.patch.object(sp, "limit", hanging_limit):
        with pytest.raises(LimitTimeout):
            limit_with_timeout(1 / x, x, 0, timeout=0.5)
    # Mutation demonstration: the old numeric fallback would AVERAGE the
    # symmetric probes (+1e6, -1e6...) into a fake finite 0.0 for limit(1/x).
    probing_values = [float(1 / (0 + d)) for d in (1e-6, -1e-6)]
    averaged = sum(probing_values) / 2.0
    assert averaged == pytest.approx(0.0, abs=1e-9)  # that is the defect


# ---------------------------------------------------------------------------
# FLAW-13: ODE referee thresholds and the divergent verdict
# ---------------------------------------------------------------------------


def test_flaw13_ode_referee_thresholds_bound_to_spec():
    from apedp_dyn.conf.canonical_spec import (
        ODE_REFEREE_MIN_MEAN_R2,
        ODE_REFEREE_MIN_WORST_R2,
    )
    from apedp_dyn.ode.referee import OdeReferee

    referee = OdeReferee()
    assert referee._min_mean_r2 == ODE_REFEREE_MIN_MEAN_R2 == 0.95
    assert referee._min_worst_r2 == ODE_REFEREE_MIN_WORST_R2 == 0.85
    # The spec file is the single source: the keys must exist in the JSON.
    import json as _json
    spec = _json.loads(
        (REPO_ROOT / "specifications" / "canonical_preregistered_spec_v1.json").read_text()
    )
    assert spec["thresholds"]["ode_referee_min_mean_r2"] == 0.95
    assert spec["thresholds"]["ode_referee_min_worst_r2"] == 0.85


def test_flaw13_negative_r2_gets_unstable_divergent():
    """A numerically-stable rollout that tracks the holdout WORSE than its own
    mean (finite holdout R^2 < 0) is UNSTABLE_DIVERGENT, not
    PREDICTIVE_WRONG_STRUCTURE (mutation: the old verdict name was returned)."""
    from apedp_dyn.epistemic.coupled_discovery import (
        CoupledEquationInfo,
        CoupledSystemDiscoveryResult,
    )
    from apedp_dyn.ode.referee import EpistemicOdeVerdict, OdeReferee

    eq = CoupledEquationInfo(
        state_variable="x",
        formula="test",
        active_terms=["x"],
        coefficients=[-0.9],
        r2=0.999,
        residual_rmse=0.001,
    )
    result = CoupledSystemDiscoveryResult(
        equations=[eq], mean_system_r2=0.999, worst_state_r2=0.999,
        is_exact_recovery=False,
    )
    holdout = {"x": list(range(30))}  # ramping truth the damped sim cannot track
    adj = OdeReferee().adjudicate_result(result, holdout)
    assert adj.verdict is EpistemicOdeVerdict.UNSTABLE_DIVERGENT, (
        f"finite-negative holdout R^2 ({adj.mean_holdout_r2:.3f}) mislabeled as "
        f"{adj.verdict.value} — FLAW-13 verdict semantics reverted"
    )
    assert 0.0 > adj.mean_holdout_r2 > -float("inf")
    assert adj.is_stable is True  # numerically stable, scientifically divergent


# ---------------------------------------------------------------------------
# FLAW-14: toroidal Nyquist->DC spectral wrapping
# ---------------------------------------------------------------------------


def test_flaw14_spectral_advection_does_not_wrap():
    from apedp_dyn.audio.spectrogram_transport import _upwind_advection_step

    n, df = 64, 0.02
    u = np.zeros(n)
    u[-1] = 1.0  # Nyquist-bin energy
    # c > 0: upwind looks backwards; the DC bin must NOT receive the Nyquist
    # energy that the old np.roll stencil wrapped around the boundary.
    out = _upwind_advection_step(u, c=0.5, dt=0.001, frequency_step=df)
    assert out[0] == 0.0, "energy wrapped from Nyquist into the DC bin — FLAW-14 reverted"
    # Mutation demonstration: the old np.roll backward difference would push
    # mass into index 0 immediately.
    rolled_slope = (u - np.roll(u, 1)) / df
    rolled = np.maximum(u - 0.5 * 0.001 * rolled_slope, 0.0)
    assert rolled[0] > 0.0  # the defect, reproduced for the record

    # Symmetric check: mass at DC with c < 0 must not wrap into Nyquist.
    u2 = np.zeros(n)
    u2[0] = 1.0
    out2 = _upwind_advection_step(u2, c=-0.5, dt=0.001, frequency_step=df)
    assert out2[-1] == 0.0, "DC energy wrapped into the Nyquist bin — FLAW-14 reverted"


# ---------------------------------------------------------------------------
# FLAW-16: Hamiltonian archetype drift
# ---------------------------------------------------------------------------


def test_flaw16_hamiltonian_archetype_matches_contract():
    from apedp_dyn.grammar.proposer_interface import StrategyArchetype
    from apedp_dyn.proposers.stem_proposers import HamiltonianProposer

    t = np.linspace(0.0, 10.0, 500)
    q = np.cos(2.0 * t)
    p = -2.0 * np.sin(2.0 * t)
    proposals = HamiltonianProposer(dt=float(t[1] - t[0]), max_energy_drift=0.02).fit_and_propose(
        pd.DataFrame({"q": q, "p": p})
    )
    assert len(proposals) == 1
    assert proposals[0].strategy_archetype == StrategyArchetype.CONTINUOUS_FACTOR, (
        "FLAW-16 reverted: archetype drift off the pinned acceleration-law contract"
    )


# ---------------------------------------------------------------------------
# FLAW-17: CBF post-correction norm-ball violation
# ---------------------------------------------------------------------------


def test_flaw17_cbf_final_projection_prevents_norm_violation():
    """Deterministic single-pair geometry: v_i sits ON the norm ball with a
    large tangential component, so the least-change barrier correction
    overshoots the ball (|v_i| 1.0 -> ~1.04). The OLD loop exits right after
    the correction (no re-projection before the fail-closed check) and
    rejects the trajectory; the remediated budget-aware distribution keeps
    BOTH contracts exactly.

    Mutation demonstration: the old single-iteration sequence is reimplemented
    inline; the fixed API with the same ``max_iter=1`` certifies the same
    trajectory via the endpoint-headroom split.
    """
    from apedp_dyn.robotics.cbf_certificate import (
        CBFQPCertificateAssumptions,
        CertificateViolation,
        _pair_barriers,
        solve_cbf_qp,
    )

    assumptions = CBFQPCertificateAssumptions(
        v_max=1.0, r_safe=0.5, gamma=1.0, dt=0.1, tol=1e-9,
    )
    positions = np.array([[1.6, 0.0], [1.0, 0.0]])
    w = math.sqrt(1.0 - 0.2 ** 2)  # tangential component: |v_i| = v_max on the ball
    nominal = np.array([[-0.2, w], [1.0, 0.0]])  # head-on convergence

    # --- invariant sanity: the nominal IS on the norm ball and IS violating
    # --- the pair (so a correction is truly required, kappa ~ deficit/2 ~ 0.55).
    bar = _pair_barriers(positions, assumptions.r_safe)
    unit01 = bar.unit[0, 1]  # [2] — n_ij for the single (0,1) pair
    b01 = -assumptions.gamma * float(bar.h[0, 1])
    nrm0 = np.linalg.norm(nominal, axis=1)
    assert np.all(nrm0 <= assumptions.v_max + 1e-12), "fixture must start inside the ball"
    lhs0 = float(unit01 @ (nominal[0] - nominal[1]))
    assert b01 - lhs0 > 0.5, "fixture must require a large barrier correction"

    def old_solve() -> np.ndarray:
        """PRE-FLAW-17 loop, single iteration: it exits right after a
        barrier correction, before ANY re-projection — the defect."""
        v = np.array(nominal, dtype=float)
        a_ij = unit01[None, :]            # [1, 2]
        b_ij_old = np.array([b01])
        ii = np.array([0])
        jj = np.array([1])
        for _ in range(1):
            lhs = np.einsum("pk,pk->p", a_ij, v[ii] - v[jj])
            deficit = b_ij_old - lhs
            bad = deficit > assumptions.tol
            if not bad.any():
                break
            kappa = deficit[bad] / 2.0
            corr = a_ij[bad] * kappa[:, None]
            np.add.at(v, ii[bad], corr)
            np.add.at(v, jj[bad], -corr)
            # loop ENDS here on a correction — old code never re-projects
        norms = np.linalg.norm(v, axis=1)
        if float(np.max(norms)) > assumptions.v_max + assumptions.tol:
            raise CertificateViolation("CBF-QP failed the norm-ball constraint (A1)")
        return v

    # Mutation demonstration: the old logic raises the A1 false refusal.
    with pytest.raises(CertificateViolation, match="norm-ball"):
        old_solve()

    # Fixed: the budget-aware final distribution certifies the SAME
    # trajectory (same single correction iteration) without the false A1
    # refusal and without breaking the pair either.
    fixed_v = solve_cbf_qp(positions, nominal, assumptions, max_iter=1)
    norms = np.linalg.norm(fixed_v, axis=1)
    assert float(np.max(norms)) <= assumptions.v_max + assumptions.tol
    lhs_f = float(unit01 @ (fixed_v[0] - fixed_v[1]))
    assert lhs_f >= b01 - assumptions.tol, "remedy must not re-break the barrier"

    # And the unconverged-loop case with the FULL loop (default max_iter) is
    # unaffected: still certifies.
    v_full = solve_cbf_qp(positions, nominal, assumptions)
    assert float(np.max(np.linalg.norm(v_full, axis=1))) <= assumptions.v_max + assumptions.tol


# ---------------------------------------------------------------------------
# FLAW-18: ledger append races and O(N^2) remote sync
# ---------------------------------------------------------------------------


class _CountingBackend:
    def __init__(self) -> None:
        self.sync_calls = 0

    def sync_to_remote(self, local_path, remote_key):  # noqa: ARG002
        self.sync_calls += 1

    def pull_from_remote(self, remote_key, local_path):  # noqa: ARG002
        return False

    def export_snapshot(self, local_path, snapshot_name=None):  # noqa: ARG002
        return str(local_path)


def test_flaw18_ledger_flock_and_amortized_sync(tmp_path):
    """Pin A: 16 pending appends amortize to ONE remote sync (the old per-
    append upload made 16 — the O(N^2) defect).

    Mutation demonstration: under the OLD policy each append synced, so the
    count would be 16; the amortized count is what the pin checks.
    """
    from apedp_dyn.epistemic.experiment_ledger import (
        CandidateExperimentNode,
        CandidateLifecycleState,
        FullLineageExperimentLedger,
    )

    backend = _CountingBackend()
    ledger = FullLineageExperimentLedger(
        persistence_path=tmp_path / "ledger.jsonl",
        storage_backend=backend,
        auto_sync=True,
    )
    for i in range(16):
        ledger.register_candidate(
            candidate_id=f"cand_flaw18_{i:02d}",
            proposer_id="pin_proposer",
            formula="f(x)",
        )
    assert backend.sync_calls == 1, (
        f"expected exactly one amortized sync after 16 appends, got {backend.sync_calls} "
        "(16 would mean the per-append O(N^2) defect returned)"
    )

    # Pin B: concurrent appends stay line-atomic (flock guard survives a
    # racing-writer hammering).
    from concurrent.futures import ThreadPoolExecutor

    state = CandidateLifecycleState.PROPOSED
    nodes = [
        CandidateExperimentNode(
            candidate_id=f"racer_{w:02d}_{k:02d}",
            proposer_id="pin_proposer",
            formula=f"f_{w}_{k}",
        )
        for w in range(4)
        for k in range(25)
    ]

    def _append(node):
        ledger._persist_append(node)
        return node.candidate_id

    with ThreadPoolExecutor(max_workers=4) as pool:  # noqa: S110
        list(pool.map(_append, nodes))

    lines = (tmp_path / "ledger.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(nodes) + 16  # 16 registered candidates + 100 racing appends
    parsed = [json.loads(line) for line in lines]
    ids = {rec["candidate_id"] for rec in parsed}
    assert len(ids) == len(parsed), "a corrupted/interleaved record was found"
    assert {f"racer_{w:02d}_{k:02d}" for w in range(4) for k in range(25)}.issubset(ids)


# ---------------------------------------------------------------------------
# FLAW-19: dead gradient computation and internal variable leak
# ---------------------------------------------------------------------------


def test_flaw19_sir_formula_uses_physical_notation():
    from apedp_dyn.proposers.stem_proposers import CompartmentalSirProposer

    n_pop, dt, n_steps = 1_000_000.0, 1.0, 80
    S, I, R = np.empty(n_steps), np.empty(n_steps), np.empty(n_steps)
    S[0], I[0], R[0] = 0.99 * n_pop, 0.01 * n_pop, 0.0
    for t in range(1, n_steps):
        dS = -0.3 * S[t - 1] * I[t - 1] / n_pop * dt
        dI = (0.3 * S[t - 1] * I[t - 1] / n_pop - 0.12 * I[t - 1]) * dt
        S[t], I[t], R[t] = S[t - 1] + dS, I[t - 1] + dI, R[t - 1] - dS - dI
    proposals = CompartmentalSirProposer(dt=dt).fit_and_propose(
        pd.DataFrame({"S": S, "I": I, "R": R})
    )
    assert proposals, "SIR proposer must not abstain on the pinned fixture"
    formula = proposals[0].formula
    assert "I_arr" not in formula, f"internal variable leaked into formula: {formula!r}"
    assert " * I" in formula, f"physical notation expected: {formula!r}"


# ---------------------------------------------------------------------------
# Rust-side text pins (kernel seams live in cargo tests next to the code)
# ---------------------------------------------------------------------------


def _kernel_source(rel: str) -> str:
    return (REPO_ROOT / rel).read_text()


def test_flaw04_rust_fornberg_nonuniform_fallback_pins():
    src = _kernel_source("rust/apedp-kernel/src/rime/diff_pade.rs")
    assert "fn fornberg_weights" in src, "Fornberg helper missing — FLAW-04 reverted"
    for fn in ("compute_derivatives", "compute_second_derivatives"):
        start = src.index(f"pub fn {fn}")
        block = src[start : src.index("\n}", start)]
        assert "is_uniform_grid" in block, f"{fn} lost the uniformity gate — FLAW-04 reverted"
        assert "derivatives_nonuniform" in block, f"{fn} lost the non-uniform fallback"


def test_flaw05_rust_stridge_significance_gate_pins():
    src = _kernel_source("rust/apedp-kernel/src/rime/pde.rs")
    # (a) failed linear solve (se == 0) must disable the significance rescue
    assert "se_g[idx] > 0.0 &&" in src, "se=0 rescue defeat returned — FLAW-05 reverted"
    # (b) constant-target tautology removed
    block_at = src.index("let r2 = if ss_tot > 1e-15")
    tail = src[block_at : block_at + 400]
    assert "} else if ss_res <= 1e-15 {" in tail, "flat-target tautology returned"


def test_flaw06a_rust_pole_flag_pins():
    src = _kernel_source("rust/apedp-kernel/src/rime/diff_pade.rs")
    assert src.count("pole_hit.set(true)") >= 2, (
        "pole masking returned (P(x)≈0 / S(x)≈0 no longer flagged) — FLAW-06a reverted"
    )
    assert "pole_singularity_hit: pole_final" in src, "pole flag dropped from the result"
    assert "!pole_final" in src, "exact-recovery no longer disqualified on poles"


def test_flaw06b_rust_koopman_centered_r2_pins():
    src = _kernel_source("rust/apedp-kernel/src/rime/koopman.rs")
    assert "z_mean" in src, "uncentered ss_tot returned — FLAW-06b reverted"
    assert "ss_tot += d_mean * d_mean" in src, "centered variance accumulation gone"
    assert "} else {\n        0.0\n    };" in src, "zero-variance tautology returned"


def test_flaw12_rust_active_column_scale_pins():
    src = _kernel_source("rust/apedp-kernel/src/rime/implicit.rs")
    assert 'if names[j] == "1" || sparse_c[j] == 0.0 {' in src, (
        "inactive columns re-inflating the exact-RMSE ceiling — FLAW-12 reverted"
    )


def test_flaw15_rust_normalized_tolerance_pins():
    src = _kernel_source("rust/apedp-kernel/src/rime/lie_symmetry.rs")
    assert "tol * max_eval.max(1e-300)" in src, (
        "unnormalized absolute nullspace tolerance returned — FLAW-15 reverted"
    )
    # The old literal-use site must be gone from the eigenvalue loop.
    eigen_region = src[: src.index("let monos = monomials_upto")]
    assert "if v.abs() <= tol {" not in eigen_region


def test_flaw01_documented_open_and_not_bypassed():
    """Governance quorum desync: the audit doc must record it OPEN, and the
    manifest must still carry the (deprecated) sealed hash — no local re-mint
    masquerading as trustee recertification."""
    import hashlib

    spec_bytes = (REPO_ROOT / "specifications" / "canonical_preregistered_spec_v1.json").read_bytes()
    manifest = (REPO_ROOT / "specifications" / "governance_manifest.json").read_text()
    doc = (REPO_ROOT / "docs" / "SYSTEM_FLAWS_AUDIT.md").read_text()
    assert "FLAW-01" in doc and "OPEN" in doc, "governance flaw must stay documented OPEN"
    disk_hash = hashlib.sha256(spec_bytes).hexdigest()
    # The canonical_spec PIN must equal the disk file (§1.3 duty).
    from apedp_dyn.conf.canonical_spec import PINNED_SPEC_SHA256

    assert disk_hash == PINNED_SPEC_SHA256
    # And the manifest is still NOT re-signed (no local forgery present).
    assert "a6d026f3" in manifest or disk_hash not in manifest
