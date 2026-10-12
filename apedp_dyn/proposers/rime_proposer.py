"""RIME symbolic-mechanism proposer (Stage-2 adapter).

Wraps the native RIME engine (``rust/apedp-kernel/src/rime/``, spec
``docs/RIME_V01_MASTER_SPEC.md``) behind the standard proposer interface, so a
symbolically discovered mechanism competes with the statistical proposers and is
adjudicated by the same frozen referee.

Boundary discipline (AGENTS.md §1.1, RIME spec §11):

  * This module only *proposes*. It fits on development data, emits
    ``CandidateProposal`` objects, and never scores itself against a holdout
    slice, promotes itself, or touches the referee's thresholds.
  * A proposal carries a **closed-form equation** (the RIME graph rendered as a
    string) plus the content-addressed ``graph_id``, so the referee can hash,
    re-evaluate and audit exactly what was proposed.
  * ``compute_score`` re-evaluates the *same* graph through the native kernel on
    whatever frame it is handed, which makes the proposal causally usable on
    unseen rows (the referee's out-of-sample check) rather than a fitted
    in-sample vector.

Fitting summary: the mechanism is discovered on dev data with a
ridge least-squares readout (``scale``, ``intercept``) which is reported in
``parameters`` so the equation is exactly executable.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)

try:  # pragma: no cover - exercised by the import guard test
    import apedp_kernel

    _RIME_AVAILABLE = hasattr(apedp_kernel, "rime_search_json")
except ImportError:  # pragma: no cover
    apedp_kernel = None
    _RIME_AVAILABLE = False


def is_rime_available() -> bool:
    """True when the native kernel exposing the RIME entry points is loaded."""
    return _RIME_AVAILABLE


class RimeProposer(ProposerBase):
    """Search typed mechanism graphs, then emit the accepted mechanisms.

    Parameters
    ----------
    feature_cols:
        Columns used as mechanism inputs. RIME's seed catalogue is calibrated
        for one or two inputs (it truncates at 60 forms); more inputs still work
        but the tail families are dropped, so pass the two you actually intend.
    target_col:
        Column the mechanism must predict. Use a *forward* quantity (e.g. a
        forward return or change), never a level that contains its own past.
    max_depth / beam / budget:
        Search shape (spec §7/§8). Defaults: depth 2, beam 8, 200 candidates.
    rel_penalty:
        Complexity penalty as a fraction of the target variance per charged
        degree of freedom (edge or readout parameter). Ignored when `gamma` is
        set. The audit's failure was a penalty that was either far above or far
        below every achievable MSE change; scaling to the target keeps the
        charge comparable to signal.
    gamma:
        Sample-size-aware penalty per charged degree of freedom:
        ``gamma * var(y_dev) / n_dev``. Preferred over ``rel_penalty`` because
        it moves with the sample size; 1.5 is a safety factor calibration.
    per_frame_envs:
        When True and a dict of frames is passed, score each frame as a SEPARATE
        development environment (spec §8 "mean loss reduction across dev
        environments") instead of pooling them. Use with ``aggregation``.
    aggregation:
        Frozen multi-environment rule: ``"mean"`` (equal weight per env — the
        literal §8 reading) or ``"median_worst_gated"`` (median of env MSEs plus
        a worst-environment improvement gate, so no env's raw size can outvote
        another). Fixed before scoring; see ``search_envs`` docs.
    max_proposals:
        Cap on emitted candidates, best first.
    """

    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        target_col: str = "target",
        max_depth: int = 2,
        beam: int = 8,
        budget: int = 200,
        rel_penalty: float | None = 0.01,
        gamma: float | None = None,
        per_frame_envs: bool = False,
        aggregation: str = "mean",
        gamma_gate: float = 1.5,
        ridge: float = 1e-9,
        seed: int = 0,
        max_proposals: int = 5,
        min_r2: float = 0.0,
        strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL,
    ) -> None:
        self._feature_cols = list(feature_cols) if feature_cols else None
        self._target_col = target_col
        self._max_depth = int(max_depth)
        self._beam = int(beam)
        self._budget = int(budget)
        self._rel_penalty = None if rel_penalty is None else float(rel_penalty)
        self._gamma = None if gamma is None else float(gamma)
        self._per_frame_envs = bool(per_frame_envs)
        self._aggregation = str(aggregation)
        self._gamma_gate = float(gamma_gate)
        self._ridge = float(ridge)
        self._seed = int(seed)
        self._max_proposals = int(max_proposals)
        self._min_r2 = float(min_r2)
        self._archetype = strategy_archetype

    @property
    def proposer_id(self) -> str:
        return "rime_symbolic_mechanism_search"

    # ── design-matrix assembly ───────────────────────────────────────────────

    def _frames(self, train_data: Any) -> list[pd.DataFrame]:
        if isinstance(train_data, pd.DataFrame):
            return [train_data]
        return [df for df in train_data.values() if isinstance(df, pd.DataFrame)]

    def _columns(
        self, frames: list[pd.DataFrame]
    ) -> tuple[list[str], list[np.ndarray], np.ndarray] | None:
        first = frames[0]
        feats = self._feature_cols or [
            c for c in first.columns if c != self._target_col and pd.api.types.is_numeric_dtype(first[c])
        ]
        if not feats or self._target_col not in first.columns:
            return None

        cols: list[np.ndarray] = [[] for _ in feats]
        target: list[float] = []
        for df in frames:
            if self._target_col not in df.columns:
                continue
            block = df[feats].to_numpy(dtype=float)
            y = df[self._target_col].to_numpy(dtype=float)
            if len(y) < 10:
                continue
            for j in range(len(feats)):
                cols[j].extend(block[:, j].tolist())
            target.extend(y.tolist())
            # Frame separator: one NaN row so that a lag can never read across a
            # dataset boundary. The scorer skips (and counts) NaN rows, so this
            # is a boundary, not an imputation.
            for j in range(len(feats)):
                cols[j].append(float("nan"))
            target.append(float("nan"))
        if not target:
            return None
        return feats, [np.asarray(c, dtype=float) for c in cols], np.asarray(target, dtype=float)

    # ── proposal construction ────────────────────────────────────────────────

    @staticmethod
    def _scorer(feats: list[str], graph_json: str, scale: float, intercept: float):
        def compute_score(df: pd.DataFrame) -> np.ndarray:
            """Evaluate the discovered mechanism on a frame.

            The signal is PAST-INFORMATION-ONLY with respect to the row it is
            computed on: the graph reads only the feature columns at lags the
            search discovered (or the current value, explicitly, when the
            mechanism contains a contemporaneous `Input`), and the readout
            (``scale``, ``intercept``) is refrozen on development data. This is
            a *temporal* property — it prevents look-ahead in the signal path —
            NOT a causal claim: an association that survives out-of-sample
            placement is not evidence of intervention, and the referee decides
            what the proposal is evidence of.
            """
            n = len(df)
            if not _RIME_AVAILABLE or n == 0:
                return np.zeros(n)
            cols = [df[c].to_numpy(dtype=float) if c in df.columns else np.full(n, np.nan) for c in feats]
            out = np.asarray(apedp_kernel.rime_evaluate_series_json(graph_json, cols), dtype=float)
            return scale * out + intercept

        return compute_score

    def _env_frames(self, frames: list[pd.DataFrame]) -> list[tuple[list[np.ndarray], np.ndarray]]:
        """Per-frame environment assembly (NaN-free; the kernel adds boundaries)."""
        first = frames[0]
        feats = self._feature_cols or [
            c for c in first.columns if c != self._target_col and pd.api.types.is_numeric_dtype(first[c])
        ]
        out = []
        for df in frames:
            if self._target_col not in df.columns or len(df) < 10:
                continue
            out.append(
                (
                    [df[c].to_numpy(dtype=float) for c in feats],
                    df[self._target_col].to_numpy(dtype=float),
                )
            )
        return out or []

    def fit_and_propose(
        self,
        train_data: Any,
        feature_cols: Sequence[str] | None = None,
        target_col: str | None = None,
    ) -> Sequence[CandidateProposal]:
        """Search on development data and emit the accepted mechanisms.

        An empty list is a valid, meaningful answer: RIME abstains when no
        candidate beats the mean predictor by more than its complexity charge
        (spec §8), which is the behaviour the referee expects on a null world.

        With ``per_frame_envs=True`` and a dict of frames, each frame is scored
        as its own development environment under the frozen ``aggregation``
        rule, so one long frame cannot outvote the others (the pooled-scoring
        dominance trap).
        """
        if feature_cols:
            self._feature_cols = list(feature_cols)
        if target_col:
            self._target_col = target_col
        if not _RIME_AVAILABLE:
            return []

        frames = self._frames(train_data)
        if not frames:
            return []

        use_envs = self._per_frame_envs and len(frames) > 1
        if use_envs:
            env_frames = self._env_frames(frames)
            if len(env_frames) < 2:
                use_envs = False
        if use_envs:
            feats = self._feature_cols or [
                c
                for c in frames[0].columns
                if c != self._target_col and pd.api.types.is_numeric_dtype(frames[0][c])
            ]
            envs_json = json.dumps([
                {"cols": [c.tolist() for c in cols], "target": y.tolist()}
                for cols, y in env_frames
            ])
            config = {
                "max_depth": self._max_depth,
                "beam": self._beam,
                "budget": self._budget,
                "ridge": self._ridge,
                "seed": self._seed,
                "aggregation": self._aggregation,
                "gamma_gate": self._gamma_gate,
            }
            if self._gamma is not None:
                config["gamma"] = self._gamma
            packet = json.loads(apedp_kernel.rime_search_envs_json(envs_json, json.dumps(config), feats))
        else:
            assembled = self._columns(frames)
            if assembled is None:
                return []
            feats, cols, target = assembled
            finite = np.isfinite(target)
            if int(finite.sum()) < 30:
                return []
            config = {
                "max_depth": self._max_depth,
                "beam": self._beam,
                "budget": self._budget,
                "ridge": self._ridge,
                "seed": self._seed,
            }
            if self._gamma is not None:
                config["gamma"] = self._gamma
            elif self._rel_penalty is not None:
                config["rel_penalty"] = self._rel_penalty
            packet = json.loads(
                apedp_kernel.rime_search_json(cols, target, json.dumps(config), feats)
            )

        # The complexity charge actually used, recorded in every packet.
        if use_envs:
            # The kernel computes gamma * pooled_var / pooled_n internally.
            lambda_used = (self._gamma if self._gamma is not None else 0.0) * float(
                packet["baseline_mse"]
            ) / max(int(packet["evaluated"]), 1)
        elif self._gamma is not None:
            lambda_used = self._gamma * float(np.var(target[finite])) / int(finite.sum())
        else:
            lambda_used = float(self._rel_penalty or 0.0) * float(np.var(target[finite]))

        proposals: list[CandidateProposal] = []
        for rank, p in enumerate(packet.get("proposals", [])):
            if len(proposals) >= self._max_proposals:
                break
            ev = p["dev_evidence"]
            if float(ev["r2"]) < self._min_r2:
                continue
            scale = float(ev["scale"])
            intercept = float(ev["intercept"])
            # The kernel ships the runnable graph with the packet, so the
            # proposal never has to re-derive the arena from the canonical form.
            graph_json = p["graph_json"]
            formula = f"{scale:+.6g} * ({p['equation']}) {intercept:+.6g}"
            dep_vars = tuple(feats)
            proposals.append(
                CandidateProposal(
                    proposer_id=self.proposer_id,
                    candidate_id=f"rime_{p['graph_id'][:12]}_{rank}",
                    formula=formula,
                    strategy_archetype=self._archetype,
                    declared_scale_invariant=True,
                    declared_nominal_dependencies=dep_vars,
                    compute_score=self._scorer(feats, graph_json, scale, intercept),
                    parameters={
                        "scale": scale,
                        "intercept": intercept,
                        "dev_mse": float(ev["mse"]),
                        "dev_r2": float(ev["r2"]),
                        "complexity": float(ev["complexity"]),
                        "n_eff": int(ev["n_eff"]),
                    },
                    metadata={
                        "graph_id": p["graph_id"],
                        "parent_id": p["parent_id"],
                        "rewrite_op": p["rewrite_op"],
                        "depth": int(p["depth"]),
                        "equation": p["equation"],
                        "dev_score": float(ev["score"]),
                        "ref_score": float(ev["ref_score"]),
                        "baseline_mse": float(packet["baseline_mse"]),
                        "evaluated": int(packet["evaluated"]),
                        "accepted": int(packet["accepted"]),
                        "lambda_c": lambda_used,
                        "engine": "rime_v0.1.1",
                        "signal_semantics": "past_information_only_not_causal",
                        "per_frame_envs": bool(use_envs),
                        "aggregation": str(packet.get("aggregation", self._aggregation))
                        if use_envs
                        else "single_dev_pool",
                        "worst_env_mse": float(packet["worst_env_mse"])
                        if "worst_env_mse" in packet
                        else None,
                        "worst_env_passed": bool(packet["worst_env_passed"])
                        if "worst_env_passed" in packet
                        else None,
                    },
                )
            )
        return proposals


