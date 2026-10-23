"""RIME-AD Autonomous Multi-Representation Proposer.

Integrates the native Rust engines in `apedp_kernel`:
- Pillar 1: Variational BV-ROF Total Variation filtering & step extraction (`total_variation.rs`)
- Pillar 2: Sparse Implicit SVD null space for rational poles and algebraic constraints (`implicit.rs`)
- Pillar 3: Differential Padé & Jet algebra polynomial ODE solver (`diff_pade.rs`)
- Pillar 4: Shared-computation canonical DAG enumeration (`enumerate.rs`)
- Additive Multi-Term regression for composite relations

Autonomous selection protocol:
1. Profiles the dataset invariants (quantized plateaus, dynamic range, Arrhenius tunneling).
2. Generates candidate hypotheses across active representations.
3. Ranks candidates on dev data by Bayesian Information Criterion (BIC / MDL).
4. Emits standardized `CandidateProposal` objects with out-of-sample `compute_score` callables.

Representation 8 adds the saturating rational family with a *fitted* half-
saturation constant -- `y = q_max * x / (c + x) + offset` (Langmuir isotherm,
Michaelis-Menten) and its cooperative Hill n=2 form. The native rational-pair
search enumerates DAG nodes with integer constants only, so a two-parameter law
like Langmuir is outside its emit set; here `c` is fitted by a golden-section
search on log(c) with the affine readout solved exactly at each trial.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised only on kernel-less installs
    import apedp_kernel as ak
except ImportError:  # kernel-only engine: fails at call time with an actionable error
    from apedp_dyn.kernel_shim import missing_kernel as ak  # type: ignore[assignment]
from apedp.rust_bridge import RustKernelBridge
from apedp_dyn.epistemic.autonomous_profiler import profile_dataset
from apedp_dyn.epistemic.shape_inference import staircase_prior
from apedp_dyn.grammar.floor_rules import fit_floor_rule, floor_rule_candidate
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype

PROPOSER_ID = "rime_ad_proposer"


# Differential-Pade integrator bounds. The target step keeps the RK4 error small;
# the cap refuses to burn an unbounded pure-Python loop (a single unit of
# distance at the target step is 2000 iterations; 20000 is a 10x safety margin).
_PADE_TARGET_STEP = 0.0005
_PADE_MAX_STEPS = 20_000


class RimeAdProposer(ProposerBase):
    """Unified autonomous proposer combining all RIME-AD representations."""

    def __init__(
        self,
        target_col: str = "target",
        max_degree: int = 2,
        strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL,
        max_univariate_features: int = 12,
        screen_top_k: int = 5,
    ):
        self._target_col = target_col
        self._max_degree = max_degree
        self._strategy_archetype = strategy_archetype
        self._max_univariate_features = max_univariate_features
        self._screen_top_k = screen_top_k

    def _screen_features(
        self, df: pd.DataFrame, feature_cols: list[str], y: np.ndarray
    ) -> list[str]:
        """Rank features by univariate [1, x, x^2] OLS dev R² (deterministic).

        The quadratic term keeps even-symmetric dependence (y ~ x^2, whose
        linear correlation is ~0) screen-visible. Returns the top
        screen_top_k names bounded by max_univariate_features, so fan-out
        exclusion is by measurement rather than silent truncation.
        """
        scored: list[tuple[float, str]] = []
        for feat in feature_cols:
            try:
                x = df[feat].to_numpy(dtype=float)
            except Exception:  # noqa: BLE001, S112 - try-continue by design: best-effort path skips this candidate without aborting  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                continue
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < 30:
                continue
            xm, ym = x[mask], y[mask]
            try:
                basis = np.column_stack([np.ones_like(xm), xm, xm ** 2])
                coef, *_ = np.linalg.lstsq(basis, ym, rcond=None)
                pred = basis @ coef
            except Exception:  # noqa: BLE001, S112 - try-continue by design: best-effort path skips this candidate without aborting  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                continue
            vy = float(np.var(ym))
            r2 = 1.0 - float(np.mean((ym - pred) ** 2)) / vy if vy > 1e-18 else 0.0
            scored.append((r2, feat))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        keep = max(0, min(self._screen_top_k, self._max_univariate_features))
        return [feat for _, feat in scored[:keep]]

    @property
    def proposer_id(self) -> str:
        return PROPOSER_ID

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame] | pd.DataFrame,
        target_col: str | None = None,
    ) -> Sequence[CandidateProposal]:
        t_col = target_col or self._target_col
        if isinstance(train_data, dict):
            dfs = [d for d in train_data.values() if isinstance(d, pd.DataFrame) and not d.empty]
            if not dfs:
                return ()
            df = pd.concat(dfs, axis=0, ignore_index=True)
        else:
            df = train_data

        if t_col not in df.columns:
            return ()

        feature_cols = [c for c in df.columns if c != t_col]
        if not feature_cols:
            return ()

        y_dev = df[t_col].to_numpy(dtype=float)
        candidates: list[CandidateProposal] = []

        # Case 1: Univariate discovery
        if len(feature_cols) == 1:
            feat = feature_cols[0]
            x_dev = df[feat].to_numpy(dtype=float)
            candidates.extend(self._propose_univariate(x_dev, y_dev, feat))
        else:
            # Case 2: Multi-variate additive discovery
            candidates.extend(self._propose_multivariate(df, feature_cols, y_dev))
            # Screening-gated fan-out: rank features by univariate dev R²
            # and run the univariate representations (TV/staircase,
            # implicit SVD, differential Padé, shared-DAG enumeration) over
            # the measured top features, so the Rust engines engage on
            # multi-feature frames instead of only the OLS baseline firing.
            # Each representation keeps its own gates; ranking below stays
            # BIC on dev rows.
            for feat in self._screen_features(df, feature_cols, y_dev):
                try:
                    x_col = df[feat].to_numpy(dtype=float)
                except Exception:  # noqa: BLE001, S112 - try-continue by design: best-effort path skips this candidate without aborting  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                    continue
                mask = np.isfinite(x_col) & np.isfinite(y_dev)
                if int(mask.sum()) < 30:
                    continue
                try:
                    candidates.extend(
                        self._propose_univariate(x_col[mask], y_dev[mask], feat)
                    )
                except Exception:  # noqa: BLE001, S112 - try-continue by design: best-effort path skips this candidate without aborting  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                    continue

        if not candidates:
            return ()

        # Rank candidates by dev BIC on the training slice
        n_samples = len(y_dev)
        ranked = []
        for cand in candidates:
            try:
                pred = np.asarray(cand.compute_score(df), dtype=float).ravel()
                mask = np.isfinite(y_dev) & np.isfinite(pred)
                if np.sum(mask) < 4:
                    continue
                sse = float(np.sum((y_dev[mask] - pred[mask]) ** 2))
                k_params = max(1, len(cand.parameters))
                bic = n_samples * np.log(max(sse / n_samples, 1e-18)) + k_params * np.log(n_samples)
                ranked.append((bic, cand))
            except Exception:  # noqa: BLE001, S112 - try-continue by design: best-effort path skips this candidate without aborting  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                continue

        ranked.sort(key=lambda item: item[0])
        return [cand for _, cand in ranked]

    def _propose_univariate(
        self, x: np.ndarray, y: np.ndarray, feat_name: str
    ) -> list[CandidateProposal]:
        candidates: list[CandidateProposal] = []
        profile = profile_dataset(x, y)
        prior, seeds = staircase_prior(x, y)

        # Representation 1: Step / Plateau (BV-ROF)
        if prior.family == "piecewise_constant" or profile.flat_derivative_fraction > 0.35:
            if seeds is not None and len(seeds.thresholds) >= 2:
                rule = fit_floor_rule(x, y, seeds.thresholds)
                if rule is not None:
                    q0, w, off, b, r2 = rule
                    cand: CandidateProposal | None = floor_rule_candidate(x, q0, w, off, b, r2, feat_name)
                    if cand is not None:
                        candidates.append(cand)

        # Representation 2: Sparse Implicit SVD (Rational Poles / Constraints)
        cfg_imp = {
            "max_degree": self._max_degree,
            "include_cross_terms": True,
            "include_rational_products": True,
            "sparsity_threshold": 0.02,
            "rcond_threshold": 1e-12,
        }
        try:
            res_imp = json.loads(
                ak.rime_implicit_discover_json(
                    json.dumps(x.tolist()), json.dumps(y.tolist()), json.dumps(cfg_imp)
                )
            )
            if res_imp.get("algebraic_rmse", 1.0) < 0.05:
                cand = self._make_implicit_candidate(res_imp, feat_name)
                if cand is not None:
                    candidates.append(cand)
        except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            pass

        # Representation 3: Differential Padé (Polynomial ODEs / Singularities)
        if profile.is_arrhenius_tunneling or profile.dynamic_range > 10.0:
            cfg_diff = {
                "max_x_degree": 2,
                "max_diff_order": 1,
                "smooth_window": 5,
                "sparsity_threshold": 0.05,
                "rcond_threshold": 1e-12,
            }
            try:
                res_diff = json.loads(
                    ak.rime_diff_pade_discover_json(
                        json.dumps(x.tolist()), json.dumps(y.tolist()), json.dumps(cfg_diff)
                    )
                )
                if res_diff.get("integrated_trajectory_r2", 0.0) > 0.80:
                    cand = self._make_diff_pade_candidate(res_diff, x, y, feat_name)
                    if cand is not None:
                        candidates.append(cand)
            except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                pass

        # Representation 4: Shared-Computation DAG Enumeration (Harmonics & Powers)
        try:
            raw_cands = json.loads(
                ak.rime_enumerate_search_json(
                    json.dumps([[float(v) for v in x]]),
                    json.dumps(y.tolist()),
                    json.dumps([feat_name]),
                    True,  # allow trig
                    False,
                    False,  # no exp (keeps the frozen depth-2 search space)
                    False,  # no extended misc ops
                    False,  # no advanced ops
                    2,
                )
            )
            for item in raw_cands[:3]:
                cand = self._make_dag_candidate(item, feat_name)
                if cand is not None:
                    candidates.append(cand)
        except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            pass

        # Representation 5: Multi-term rational powers basis [1, 1/x, 1/x^2]
        if np.all(x > 0):
            try:
                X_basis = np.column_stack([np.ones_like(x), 1.0 / x, 1.0 / (x ** 2)])
                coef, _residuals, _rank, _s = np.linalg.lstsq(X_basis, y, rcond=None)
                pred = X_basis @ coef
                var_y = np.var(y)
                r2 = 1.0 - (np.mean((y - pred) ** 2) / var_y) if var_y > 1e-12 else 0.0
                if r2 > 0.95:
                    def _scorer(df: pd.DataFrame) -> np.ndarray:
                        xv = df[feat_name].to_numpy(dtype=float)
                        Xb = np.column_stack([np.ones_like(xv), 1.0 / xv, 1.0 / (xv ** 2)])
                        return np.asarray(np.asarray(np.asarray(Xb @ coef)))

                    formula = f"{coef[0]:.4f} + {coef[1]:.4f}*(1/{feat_name}) + {coef[2]:.4f}*(1/{feat_name}^2)"
                    cand = CandidateProposal(
                        proposer_id=self.proposer_id,
                        candidate_id="multi_rational_powers",
                        formula=formula,
                        strategy_archetype=self._strategy_archetype,
                        declared_scale_invariant=False,
                        declared_nominal_dependencies=(feat_name,),
                        compute_score=_scorer,
                        parameters={"b0": float(coef[0]), "b1": float(coef[1]), "b2": float(coef[2])},
                        metadata={"r2": float(r2), "method": "multi_rational_basis"},
                    )
                    candidates.append(cand)
            except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                pass

        # Representation 6: Rational Pairs Search with Exp (Dual-Asymptotics / Planck Radiance)
        try:
            rat_cands = json.loads(
                ak.rime_rational_search_json(
                    json.dumps([[float(v) for v in x]]),
                    json.dumps(y.tolist()),
                    json.dumps([feat_name]),
                    False,
                    False,
                    False,
                    False,
                    2,
                    3,
                )
            )
            for item in rat_cands:
                if item.get("r2", 0.0) > 0.95:
                    cand = self._make_rational_pair_candidate(item, feat_name)
                    if cand is not None:
                        candidates.append(cand)
        except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            pass

        # Representation 7: Multi-Stage Heterogeneous Pipeline Composition (BV-ROF TV Jump + Smooth Law)
        try:
            tv_res = json.loads(
                ak.rime_total_variation_json(
                    json.dumps(x.tolist()), json.dumps(y.tolist()), 0.1, 0.4
                )
            )
            jumps = tv_res.get("jumps", [])
            if len(jumps) >= 2 or tv_res.get("is_quantized_staircase"):
                step = float(tv_res.get("estimated_quantum_step", 1.0))
                jump_xs = [j["x_location"] for j in jumps]
                if len(jump_xs) >= 2:
                    w = float(np.median(np.diff(jump_xs)))
                    off = float(jump_xs[0] - w)
                else:
                    w = 1.0
                    off = 0.0
                q0 = step
                b = float(y[0])
                y_jump = q0 * np.floor((x - off) / w) + b
                y_smooth = y - y_jump
                var_smooth = float(np.var(y_smooth))
                var_y = float(np.var(y))
                if 1e-4 < var_smooth < var_y:
                    X_poly = np.column_stack([np.ones_like(x), x, x ** 2])
                    c_poly, _, _, _ = np.linalg.lstsq(X_poly, y_smooth, rcond=None)
                    y_pred = y_jump + (X_poly @ c_poly)
                    r2_comp = 1.0 - (float(np.mean((y - y_pred) ** 2)) / var_y)
                    if r2_comp > 0.95:
                        cand = self._make_tv_composite_candidate(
                            q0, w, off, b + float(c_poly[0]), c_poly[1:], feat_name, r2_comp
                        )
                        if cand is not None:
                            candidates.append(cand)
        except Exception:  # noqa: BLE001, S110 - optional probe may fail silently by design (best-effort stage skip)  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            pass

        # Representation 8: fitted saturating rational (Langmuir / Michaelis-Menten)
        # and cooperative Hill n=2. The native rational-pair search can only
        # combine DAG nodes with integer constants, so a two-parameter law whose
        # half-saturation constant is free is outside its reachable set; this
        # representation fits that interior constant directly.
        for hill_n, cid in ((1.0, "saturating_rational"), (2.0, "hill_rational")):
            cand = self._fit_saturating_rational(x, y, feat_name, hill_n, cid)
            if cand is not None:
                candidates.append(cand)

        return candidates

    def _fit_saturating_rational(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feat_name: str,
        hill_n: float,
        candidate_id: str,
    ) -> CandidateProposal | None:
        """Fit `y = q_max * x^n / (c^n + x^n) + offset` with a free `c`.

        For fixed `c` the readout in `(q_max, offset)` is linear, so the fit is a
        1-D search over `log(c)` (grid + golden-section refinement) with an exact
        least-squares solve at each trial -- the separable structure that makes
        Langmuir (`n=1`, `c = 1/K`) and Michaelis-Menten reachable without
        handing the proposer the answer. `n=2` is the cooperative Hill form.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 6:
            return None
        xs, ys = x[mask], y[mask]
        scale = float(np.median(np.abs(xs)))
        if not np.isfinite(scale) or scale <= 0.0:
            return None

        def basis(c: float) -> np.ndarray | None:
            if not (np.isfinite(c) and c > 0.0):
                return None
            with np.errstate(all="ignore"):
                if hill_n == 1.0:
                    den = c + xs
                    if np.any(np.abs(den) <= 1e-12):
                        return None
                    b = xs / den
                else:
                    xn = xs ** hill_n
                    b = xn / (c ** hill_n + xn)
            if not np.all(np.isfinite(b)) or float(np.var(b)) <= 1e-15:
                return None
            return b

        def solve(c: float) -> tuple[float, float, float] | None:
            b = basis(c)
            if b is None:
                return None
            design = np.column_stack([b, np.ones_like(b)])
            coef, _res, _rank, _sv = np.linalg.lstsq(design, ys, rcond=None)
            resid = ys - design @ coef
            sse = float(resid @ resid)
            if not np.isfinite(sse):
                return None
            return sse, float(coef[0]), float(coef[1])

        lo, hi = np.log(scale * 1e-3), np.log(scale * 1e3)
        grid = np.exp(np.linspace(lo, hi, 48))
        best: tuple[float, float] | None = None  # (sse, c)
        for c in grid:
            out = solve(float(c))
            if out is not None and (best is None or out[0] < best[0]):
                best = (out[0], float(c))
        if best is None:
            return None

        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = np.log(best[1] / 3.0), np.log(best[1] * 3.0)
        x1, x2 = b - phi * (b - a), a + phi * (b - a)

        def sse_at(log_c: float) -> float:
            out = solve(float(np.exp(log_c)))
            return out[0] if out is not None else float("inf")

        f1, f2 = sse_at(x1), sse_at(x2)
        for _ in range(40):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - phi * (b - a)
                f1 = sse_at(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + phi * (b - a)
                f2 = sse_at(x2)
        c_best = float(np.exp(0.5 * (a + b)))
        fitted = solve(c_best)
        if fitted is None:
            return None
        sse, q_max, offset = fitted
        var_y = float(np.var(ys))
        if var_y <= 1e-15:
            return None
        r2 = 1.0 - sse / (var_y * len(ys))
        # A saturating form with an amplitude at the noise floor carries no law.
        if not np.isfinite(r2) or r2 <= 0.95:
            return None
        if abs(q_max) <= 1e-12 * max(float(np.max(np.abs(ys))), 1e-12):
            return None

        def _scorer(df: pd.DataFrame) -> np.ndarray:
            xv = df[feat_name].to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                if hill_n == 1.0:
                    out = q_max * xv / (c_best + xv) + offset
                else:
                    xn = xv ** hill_n
                    out = q_max * xn / (c_best ** hill_n + xn) + offset
            return np.asarray(out, dtype=float)

        if hill_n == 1.0:
            formula = (f"{q_max:.6g} * ({feat_name} / ({c_best:.6g} + {feat_name})) "
                       f"+ {offset:.6g}")
        else:
            formula = (f"{q_max:.6g} * ({feat_name}**2 / ({c_best:.6g}**2 + "
                       f"{feat_name}**2)) + {offset:.6g}")
        return CandidateProposal(
            proposer_id=PROPOSER_ID,
            candidate_id=candidate_id,
            formula=formula,
            strategy_archetype=self._strategy_archetype,
            declared_scale_invariant=False,
            declared_nominal_dependencies=(feat_name,),
            compute_score=_scorer,
            parameters={
                "q_max": q_max,
                "half_saturation": c_best,
                "offset": offset,
                "hill_n": hill_n,
            },
            metadata={
                "r2": r2,
                "method": "saturating_rational_golden_section",
                "interpretation": ("Langmuir / Michaelis-Menten" if hill_n == 1.0
                                   else "cooperative Hill (n=2)"),
            },
        )

    def _propose_multivariate(
        self, df: pd.DataFrame, feature_cols: list[str], y: np.ndarray
    ) -> list[CandidateProposal]:
        # Multi-variate linear + interaction baseline
        X_mat = [np.ones(len(df))]
        col_names = ["const"]
        for c in feature_cols:
            X_mat.append(df[c].to_numpy(dtype=float))
            col_names.append(c)

        # Add pairwise interactions if small
        if len(feature_cols) <= 4:
            for i in range(len(feature_cols)):
                for j in range(i, len(feature_cols)):
                    c1, c2 = feature_cols[i], feature_cols[j]
                    X_mat.append(df[c1].to_numpy(dtype=float) * df[c2].to_numpy(dtype=float))
                    col_names.append(f"{c1}*{c2}")

        X_arr = np.column_stack(X_mat)
        coef, *_ = np.linalg.lstsq(X_arr, y, rcond=None)
        pred = X_arr @ coef
        var_y = np.var(y)
        r2 = 1.0 - (np.mean((y - pred) ** 2) / var_y) if var_y > 1e-12 else 0.0

        def _scorer(test_df: pd.DataFrame) -> np.ndarray:
            tm = [np.ones(len(test_df))]
            for c in feature_cols:
                tm.append(test_df[c].to_numpy(dtype=float))
            if len(feature_cols) <= 4:
                for i in range(len(feature_cols)):
                    for j in range(i, len(feature_cols)):
                        c1, c2 = feature_cols[i], feature_cols[j]
                        tm.append(test_df[c1].to_numpy(dtype=float) * test_df[c2].to_numpy(dtype=float))
            return np.asarray(np.asarray(np.asarray(np.column_stack(tm) @ coef)))

        formula = " + ".join(f"{c:.4f}*{name}" for c, name in zip(coef, col_names, strict=False) if abs(c) > 1e-4)
        return [
            CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id="multivariate_ridge",
                formula=formula or "const",
                strategy_archetype=self._strategy_archetype,
                declared_scale_invariant=False,
                declared_nominal_dependencies=tuple(feature_cols),
                compute_score=_scorer,
                parameters={name: float(c) for name, c in zip(col_names, coef, strict=False)},
                metadata={"r2": float(r2), "method": "multivariate_ols"},
            )
        ]

    def _make_implicit_candidate(
        self, res_imp: dict[str, Any], feat_name: str
    ) -> CandidateProposal | None:
        c_dict = dict(zip(res_imp["feature_names"], res_imp["coefficients"], strict=False))
        # For P(x) + Q(x)*y = 0 => y = -P(x) / Q(x)
        # Check if y is linear in feature tensor
        c_1 = c_dict.get("1", 0.0)
        c_x = c_dict.get("x", 0.0)
        c_x2 = c_dict.get("x^2", 0.0)
        c_y = c_dict.get("y", 0.0)
        c_xy = c_dict.get("x*y", 0.0)
        c_x2y = c_dict.get("x^2*y", 0.0)

        # Denominator Q(x)
        def _scorer(df: pd.DataFrame) -> np.ndarray:
            xv = df[feat_name].to_numpy(dtype=float)
            num = c_1 + c_x * xv + c_x2 * (xv ** 2)
            den = c_y + c_xy * xv + c_x2y * (xv ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = -num / den
                out[np.abs(den) < 1e-12] = np.nan
            return np.asarray(np.asarray(np.asarray(out)))

        return CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id="implicit_rational_pole",
            formula=res_imp["formula"],
            strategy_archetype=self._strategy_archetype,
            declared_scale_invariant=False,
            declared_nominal_dependencies=(feat_name,),
            compute_score=_scorer,
            parameters={k: float(v) for k, v in c_dict.items() if abs(v) > 1e-6},
            metadata={"algebraic_rmse": res_imp["algebraic_rmse"], "method": "implicit_svd"},
        )

    def _make_diff_pade_candidate(
        self, res_diff: dict[str, Any], x_dev: np.ndarray, y_dev: np.ndarray, feat_name: str
    ) -> CandidateProposal | None:
        p_c = res_diff["p_coefficients"]
        q_c = res_diff["q_coefficients"]
        r_c = res_diff.get("r_coefficients", [])
        x0 = float(x_dev[-1])
        y0 = float(y_dev[-1])

        def _scorer(df: pd.DataFrame) -> np.ndarray:
            xv = df[feat_name].to_numpy(dtype=float)
            # RK4 forward/backward simulation from boundary (x0, y0) with fine step control
            idx = np.argsort(xv)
            xs_sorted = xv[idx]
            ys_sorted = np.zeros_like(xs_sorted)

            def f(x_val: float, y_val: float) -> float:
                pval = sum(c * (x_val ** p) for p, c in enumerate(p_c))
                qval = sum(c * (x_val ** p) for p, c in enumerate(q_c))
                rval = sum(c * (x_val ** p) for p, c in enumerate(r_c)) if r_c else 0.0
                if abs(pval) < 1e-12:
                    # Denominator vanishes: the ODE is singular here. Returning
                    # 0.0 used to hide the pole and poison the trajectory.
                    return float("nan")
                return float((-qval * y_val - rval * (y_val ** 2)) / pval)

            # Bounded, pole-aware integration.
            #
            # Two defects are fixed here:
            #  1. A pole (|p| ~ 0) used to return 0.0 from `f`, so the RK4
            #     marched through the singularity and every later point was
            #     computed from a corrupted state. A pole now yields NaN, and a
            #     non-finite state ends the integration: the remaining sorted
            #     points are scored NaN (fail honest) instead of inheriting the
            #     poisoned trajectory.
            #  2. `steps` was unbounded (up to ~200k Python iterations per unit
            #     distance), which hung the scorer. The step budget is capped;
            #     exceeding it marks the suffix unreachable rather than spinning.
            max_steps = _PADE_MAX_STEPS
            curr_x, curr_y = x0, y0
            broken = False
            for i, target_x in enumerate(xs_sorted):
                if broken:
                    ys_sorted[i] = np.nan
                    continue
                dist = target_x - curr_x
                steps = max(20, int(abs(dist) / _PADE_TARGET_STEP))
                if steps > max_steps:
                    broken = True
                    ys_sorted[i] = np.nan
                    continue
                h = dist / steps
                for _ in range(steps):
                    k1 = f(curr_x, curr_y)
                    k2 = f(curr_x + 0.5 * h, curr_y + 0.5 * h * k1)
                    k3 = f(curr_x + 0.5 * h, curr_y + 0.5 * h * k2)
                    k4 = f(curr_x + h, curr_y + h * k3)
                    curr_y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    curr_x += h
                    if not (np.isfinite(curr_y) and np.isfinite(curr_x)):
                        broken = True
                        break
                if broken:
                    ys_sorted[i] = np.nan
                    continue
                ys_sorted[i] = curr_y

            out = np.zeros_like(ys_sorted)
            out[idx] = ys_sorted
            return np.asarray(out)

        return CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id="diff_pade_ode",
            formula=res_diff["formula"],
            strategy_archetype=self._strategy_archetype,
            declared_scale_invariant=False,
            declared_nominal_dependencies=(feat_name,),
            compute_score=_scorer,
            parameters={"r2_traj": res_diff.get("integrated_trajectory_r2", 0.0)},
            metadata={"method": "differential_pade_rk4"},
        )

    def _make_dag_candidate(
        self, item: dict[str, Any], feat_name: str
    ) -> CandidateProposal | None:
        formula = item.get("formula", "")
        slope = item.get("slope", 1.0)
        intercept = item.get("intercept", 0.0)
        bridge = RustKernelBridge()

        # Safe evaluator for DAG expressions
        def _scorer(df: pd.DataFrame) -> np.ndarray:
            xv = df[feat_name].to_numpy(dtype=float)
            try:
                pred = bridge.evaluate_expression(formula, [feat_name], [xv])
                return np.asarray(pred, dtype=float)
            except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                # The bridge already includes its own sympy reference evaluator,
                # so reaching this point means BOTH engines could not evaluate
                # the formula. Never fabricate a different function from
                # substring matching: fail honest with NaN so downstream
                # finite-masks drop the candidate instead of scoring garbage.
                return np.full_like(xv, np.nan)

        return CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id="dag_expression",
            formula=formula,
            strategy_archetype=self._strategy_archetype,
            declared_scale_invariant=False,
            declared_nominal_dependencies=(feat_name,),
            compute_score=_scorer,
            parameters={"slope": float(slope), "intercept": float(intercept)},
            metadata={"r2": item.get("r2", 0.0), "method": "shared_dag_enumerate"},
        )

    def _make_rational_pair_candidate(
        self, item: dict[str, Any], feat_name: str
    ) -> CandidateProposal | None:
        formula = item.get("formula", "")
        slope = item.get("slope", 1.0)
        intercept = item.get("intercept", 0.0)
        bridge = RustKernelBridge()

        def _scorer(df: pd.DataFrame) -> np.ndarray:
            xv = df[feat_name].to_numpy(dtype=float)
            try:
                pred = bridge.evaluate_expression(formula, [feat_name], [xv])
                return np.asarray(pred, dtype=float)
            except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
                return np.full_like(xv, np.nan)

        return CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id="rational_pair_exp",
            formula=formula,
            strategy_archetype=self._strategy_archetype,
            declared_scale_invariant=False,
            declared_nominal_dependencies=(feat_name,),
            compute_score=_scorer,
            parameters={"slope": float(slope), "intercept": float(intercept)},
            metadata={"r2": item.get("r2", 0.0), "method": "rational_pair_search"},
        )

    def _make_tv_composite_candidate(
        self,
        q0: float,
        w: float,
        off: float,
        b: float,
        c_poly: np.ndarray,
        feat_name: str,
        r2: float,
    ) -> CandidateProposal | None:
        formula = f"{q0:.4f} * floor(({feat_name} - {off:.4f}) / {w:.4f}) + {b:.4f} + ({c_poly[0]:.4f}*{feat_name} + {c_poly[1]:.4f}*{feat_name}^2)"

        def _scorer(df: pd.DataFrame) -> np.ndarray:
            xv = df[feat_name].to_numpy(dtype=float)
            y_jump = q0 * np.floor((xv - off) / w) + b
            y_smooth = c_poly[0] * xv + c_poly[1] * (xv ** 2)
            return np.asarray(np.asarray(np.asarray(y_jump + y_smooth)))

        return CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id="tv_smooth_composite",
            formula=formula,
            strategy_archetype=self._strategy_archetype,
            declared_scale_invariant=False,
            declared_nominal_dependencies=(feat_name,),
            compute_score=_scorer,
            parameters={
                "q0": float(q0),
                "w": float(w),
                "off": float(off),
                "b": float(b),
                "c1": float(c_poly[0]),
                "c2": float(c_poly[1]),
            },
            metadata={"r2": float(r2), "method": "tv_smooth_composite_pipeline"},
        )


