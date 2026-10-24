"""Active Cross-Regime Invariant Transfer Proposer.

Searches for representations that are invariant across multiple environments
or regimes, enforcing Invariant Risk Minimization (IRM) penalties, strict
worst-regime retention gating, and active extrapolation barrier checks.

Disclosure contract (AGENTS.md §1.1): every metric in a proposal's metadata
(``worst_env_r2``, ``r2_dispersion_std``, ``cgi_score``, ...) is computed on the
SAME development environments the search selected over. This proposer ranks
candidates but never decides acceptance: that is the epistemic referee's job on
a sealed holdout. Search size (number of degree trials, gate-passing pool, the
selected rank, and a shared per-call ``search_id``) is disclosed in every
emitted candidate's metadata so ``CandidateTrialLedger.effective_trial_count``
charges the whole search, not just the emitted survivor.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype

logger = logging.getLogger("apedp_dyn.proposers.invariant_transfer")

try:
    import apedp_kernel
    _KERNEL_AVAILABLE = True
except ImportError:
    _KERNEL_AVAILABLE = False

PROPOSER_ID = "invariant_transfer_proposer"


class InvariantTransferProposer(ProposerBase):
    """Proposer actively seeking invariant causal representations across environments."""

    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        target_col: str = "target",
        env_col: str | None = "regime",
        min_worst_r2: float = 0.50,
        max_r2_dispersion: float = 0.25,
        extrap_margin_ratio: float = 0.50,
        max_gradient_ratio: float = 10.0,
        max_poly_degree: int = 3,
        strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL,
    ):
        self._feature_cols = list(feature_cols) if feature_cols else None
        self._target_col = target_col
        self._env_col = env_col
        self._min_worst_r2 = float(min_worst_r2)
        self._max_r2_dispersion = float(max_r2_dispersion)
        self._extrap_margin_ratio = float(extrap_margin_ratio)
        self._max_gradient_ratio = float(max_gradient_ratio)
        self._max_poly_degree = int(max_poly_degree)
        self._strategy_archetype = strategy_archetype

    @property
    def proposer_id(self) -> str:
        return PROPOSER_ID

    def _split_environments(
        self, train_data: Any
    ) -> list[dict[str, Any]]:
        """Extract environment dictionaries from DataFrame or list of DataFrames."""
        envs: list[dict[str, Any]] = []

        if isinstance(train_data, (list, tuple)):
            for idx, df in enumerate(train_data):
                if not isinstance(df, pd.DataFrame) or self._target_col not in df.columns:
                    continue
                name = f"env_{idx}"
                envs.append({"name": name, "df": df})
            return envs

        if not isinstance(train_data, pd.DataFrame) or self._target_col not in train_data.columns:
            return envs

        df = train_data
        if self._env_col and self._env_col in df.columns:
            for val, group in df.groupby(self._env_col):
                envs.append({"name": str(val), "df": group})
        else:
            # Fallback: bisect into early/late temporal regimes
            n = len(df)
            mid = n // 2
            envs.append({"name": "regime_early", "df": df.iloc[:mid]})
            envs.append({"name": "regime_late", "df": df.iloc[mid:]})

        return envs

    def fit_and_propose(
        self,
        train_data: Any,
        feature_cols: Sequence[str] | None = None,
        target_col: str | None = None,
        **kwargs: Any,
    ) -> list[CandidateProposal]:
        if not _KERNEL_AVAILABLE:
            return []

        t_col = target_col or self._target_col
        raw_envs = self._split_environments(train_data)
        if len(raw_envs) < 2:
            return []

        # Resolve feature columns
        sample_df = raw_envs[0]["df"]
        f_cols = list(feature_cols) if feature_cols else (
            self._feature_cols or [
                c for c in sample_df.columns
                if c != t_col and c != self._env_col and np.issubdtype(sample_df[c].dtype, np.number)
            ]
        )
        if not f_cols:
            return []

        proposals: list[CandidateProposal] = []
        evaluated_fit_count = 0

        for feat in f_cols:
            # Build environment payload for native Rust
            rust_envs: list[dict[str, Any]] = []
            for env in raw_envs:
                df = env["df"]
                if feat not in df.columns or t_col not in df.columns:
                    continue
                x_vals = df[feat].to_numpy(dtype=float)
                y_vals = df[t_col].to_numpy(dtype=float)
                mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                if int(mask.sum()) < 4:
                    continue
                rust_envs.append({
                    "name": env["name"],
                    "x": x_vals[mask].tolist(),
                    "y": y_vals[mask].tolist(),
                })

            if len(rust_envs) < 2:
                continue

            envs_json = json.dumps(rust_envs)

            # Fit candidate polynomials on pooled dev data
            x_pool = np.concatenate([np.array(e["x"]) for e in rust_envs])
            y_pool = np.concatenate([np.array(e["y"]) for e in rust_envs])

            for deg in range(1, self._max_poly_degree + 1):
                evaluated_fit_count += 1
                try:
                    poly_fit = np.polyfit(x_pool, y_pool, deg)
                    coeffs = poly_fit[::-1].tolist() # ascending order: c0 + c1*x + ...
                except (np.linalg.LinAlgError, ValueError, TypeError) as exc:
                    # Pool degeneracies (collinear columns, non-finite data)
                    # are data conditions, not verdicts: skip this degree.
                    logger.debug("polyfit degree %d skipped: %s", deg, exc)
                    continue

                formula_terms = []
                for p, c in enumerate(coeffs):
                    if abs(c) < 1e-6:
                        continue
                    if p == 0:
                        formula_terms.append(f"{c:.4f}")
                    elif p == 1:
                        formula_terms.append(f"{c:.4f}*{feat}")
                    else:
                        formula_terms.append(f"{c:.4f}*{feat}^{p}")
                formula = " + ".join(formula_terms) or "0.0"

                # Evaluate via native Rust invariant transfer engine
                raw_res = apedp_kernel.rime_invariant_transfer_evaluate_json(
                    f"poly_deg_{deg}_{feat}",
                    formula,
                    envs_json,
                    coeffs,
                    "poly",
                    self._min_worst_r2,
                    self._max_r2_dispersion,
                    self._extrap_margin_ratio,
                    self._max_gradient_ratio,
                )
                res = json.loads(raw_res)
                if not res:
                    continue

                # Enforce active retention gate & extrapolation safety
                if not res["passed_retention_gate"] or not res["extrapolation_safe"]:
                    continue

                captured_coeffs = list(coeffs)
                captured_feat = feat

                def _make_scorer(c_list: list[float], f_name: str):
                    def compute_score(frame: pd.DataFrame) -> np.ndarray:
                        if f_name not in frame.columns:
                            return np.zeros(len(frame))
                        xv = frame[f_name].to_numpy(dtype=float)
                        out = np.zeros_like(xv)
                        x_pow = np.ones_like(xv)
                        for c in c_list:
                            out += c * x_pow
                            x_pow *= xv
                        return np.asarray(np.asarray(np.asarray(out)))
                    return compute_score

                proposals.append(
                    CandidateProposal(
                        proposer_id=self.proposer_id,
                        candidate_id=f"invariant_poly_{deg}_{feat}",
                        formula=formula,
                        strategy_archetype=self._strategy_archetype,
                        declared_scale_invariant=False,
                        declared_nominal_dependencies=(feat,),
                        compute_score=_make_scorer(captured_coeffs, captured_feat),
                        parameters={
                            f"c{i}": float(c) for i, c in enumerate(captured_coeffs)
                        },
                        metadata={
                            "cgi_score": res["cgi_score"],
                            "worst_env_r2": res["worst_env_r2"],
                            "r2_dispersion_std": res["r2_dispersion_std"],
                            "irm_penalty": res["irm_penalty"],
                            "extrapolation_verdict": res["extrapolation_verdict"],
                            "degree_trials": self._max_poly_degree,
                            "dev_side_gates": "passed_retention_gate & extrapolation_safe (pooled dev environments)",
                        },
                    )
                )

        # Sort proposals by Cross-Regime Generalization Index (CGI) descending
        proposals.sort(
            key=lambda p: p.metadata.get("cgi_score", -float("inf")),
            reverse=True,
        )
        # Disclose the selection pool and rank: metrics above were selected over
        # this many order-statistics, so they are optimistically biased without
        # holdout confirmation.
        search_id = f"{self.proposer_id}:{uuid.uuid4().hex[:12]}"
        selection_pool_size = max(evaluated_fit_count, len(proposals))
        for rank, proposal in enumerate(proposals):
            proposal.metadata["search_id"] = search_id
            proposal.metadata["selection_pool_size"] = selection_pool_size
            proposal.metadata["evaluated_fit_count"] = evaluated_fit_count
            proposal.metadata["selection_rank"] = rank
        return proposals
