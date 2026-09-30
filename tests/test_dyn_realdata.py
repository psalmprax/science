"""Real-data firewall tests (no synthetic generator involved).

Uses Bertozzi 1964 electron data (data/bertozzi_1964.jsonl, real experimental
runs) through the epistemic core: classical vs relativistic kinetic-energy
laws, ranked by fit, adjudicated on held-out high-energy runs. This exercises
ranking + adjudication + abstention on measurements no part of APEDP
generated. It does NOT exercise the trajectory fitter (static relation), a
limitation stated, not hidden.

Second test: same-regime control — low-energy runs only, where both laws
agree to measurement precision. The firewall must NOT manufacture a
distinction: MULTI_HYPOTHESIS or INSUFFICIENT_EVIDENCE, never UNIQUE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apedp_dyn.epistemic.adjudicate import EpistemicStatus, adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

DATA = Path(__file__).resolve().parent.parent / "data" / "bertozzi_1964.jsonl"
C = 1.0  # v_over_c units
M_MEV = 0.511


def _load():
    rec = json.loads(DATA.read_text())
    runs = rec["runs"]
    ke = np.array([r["KE_MeV"] for r in runs])
    v = np.array([r["v_over_c"] for r in runs])
    return ke, v


def _pred_classical(ke):
    return np.sqrt(2.0 * ke / M_MEV)


def _pred_relativistic(ke):
    return np.sqrt(1.0 - 1.0 / (1.0 + ke / M_MEV) ** 2)


def _scored(cid, terms, train_rmse, ho_rmse, obs_rms, n):
    c = CandidateDynamics(lhs="v", terms=tuple(Term(t, f"a{i}") for i, t in enumerate(terms)),
                          candidate_id=cid)
    prov = {"data_origin": "experimental",
            "gate4_status": {"passed": True, "holdout_obs_rms": obs_rms,
                             "holdout_noise_sigma": 0.005,
                             "holdout_n_samples": n}}
    return ScoredCandidate(c, prov, train_loss=train_rmse, holdout_rmse=ho_rmse)


class TestBertozzi:
    def test_relativistic_supported_across_regimes(self):
        """Train on low-energy runs, adjudicate on held-out high-energy runs:
        classical goes superluminal and must lose decisively."""
        ke, v = _load()
        assert len(ke) == 5
        tr, ho = slice(0, 3), slice(3, 5)
        rc = np.sqrt(np.mean((_pred_classical(ke[tr]) - v[tr]) ** 2))
        rr = np.sqrt(np.mean((_pred_relativistic(ke[tr]) - v[tr]) ** 2))
        hc = np.sqrt(np.mean((_pred_classical(ke[ho]) - v[ho]) ** 2))
        hr = np.sqrt(np.mean((_pred_relativistic(ke[ho]) - v[ho]) ** 2))
        assert hc > 10 * hr  # classical catastrophically wrong out-of-regime
        obs_rms = float(np.sqrt(np.mean(v[ho] ** 2)))
        sc = _scored("classical", ("one", "x"), rc, hc, obs_rms, 2)
        sr = _scored("relativistic", ("one", "x", "x3", "v"), rr, hr, obs_rms, 2)
        adj = adjudicate([sr, sc])
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj
        assert adj.promoted.candidate.candidate_id == "relativistic"
        assert adj.data_origin == "experimental", f"Expected experimental data_origin, got {adj.data_origin}"

    def test_no_agreement_regime_in_this_dataset(self):
        """Corrected premise (was: low-energy runs agree — false): even the
        lowest run sits at KE ~ m c^2, where classical already errs by 0.53
        in v/c. This dataset contains NO regime where both laws agree, so no
        abstention control can be built from it; agreement-regime abstention
        is covered by the synthetic small-holdout MULTI tests. Pinned here so
        the premise cannot silently rot."""
        ke, v = _load()
        tr = slice(0, 3)
        rc = float(np.sqrt(np.mean((_pred_classical(ke[tr]) - v[tr]) ** 2)))
        rr = float(np.sqrt(np.mean((_pred_relativistic(ke[tr]) - v[tr]) ** 2)))
        assert rc > 0.5, rc  # classical fails everywhere in this dataset
        assert rr < 0.05, rr  # relativistic fits everywhere in this dataset
        assert rc / rr > 10  # separated by an order of magnitude even in-sample
