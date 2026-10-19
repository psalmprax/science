r"""Next-Concept Modeling (NCM) Structural Proposer for Macroscopic Functional Decomposition.

Status: experimental

Scientific Scope & Algorithmic Foundation:
- Naming inspiration only, no architectural correspondence: "Next-Concept Modeling"
  (Kumbong et al., arXiv:2610.14162, cs.CL) is an NLP text-generation method
  (multiscale residual VQ-VAE over text tokens; a transformer emitting one code level
  at a time). None of that architecture is implemented here: no codebooks, no
  hierarchical quantization, no learned latents, no transformer, no parallel
  generation. The paper's 10x-31x speedups are against diffusion language models on
  text benchmarks and apply to no RIME/kernel search path in this repository.
- What this module actually is: a purely classical, heuristic two-tier baseline over
  tabular time series. In symbolic ODE/PDE discovery, bottom-up DAG searches scale
  with combinatorial branching O(b^d); this proposer restricts the hypothesis space to
  a fixed small set of functional families and fits only their coefficients:
    1. Tier 1 (Macroscopic Archetype Fitting): Fits each requested functional family
       (conserved damped oscillator, diffusion-advection, bounded rational saturation,
       cyclic harmonic) independently. Every fitted family is emitted as a
       `CandidateProposal`; hypothesis selection is deferred to downstream epistemic
       gates (AGENTS.md §1.1).
    2. Tier 2 (Microscopic Coefficient Fitting & Sparsity Thresholding): Estimates each
       family's coefficients via Ridge regression (L2), then zeroes coefficients with
       magnitude below `sparsity_threshold` (hard thresholding; not L1/Lasso).
- Conforms to `ProposerBase` and emits standardized `CandidateProposal` objects conforming to
  `AGENTS.md §1.1` (strict proposer vs. referee separation).
"""
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


class MacroConceptArchetype(str, Enum):
    """Macroscopic physical functional concepts."""
    CONSERVED_DAMPED_OSCILLATOR = "conserved_damped_oscillator"
    DIFFUSION_ADVECTION_CONTINUUM = "diffusion_advection_continuum"
    BOUNDED_RATIONAL_SATURATION = "bounded_rational_saturation"
    CYCLIC_HARMONIC_REGIME = "cyclic_harmonic_regime"


class NCMStructuralProposer(ProposerBase):
    """Two-tier Next-Concept Modeling proposer for physical dynamical systems."""

    def __init__(
        self,
        archetypes: Sequence[MacroConceptArchetype] | None = None,
        regularization_alpha: float = 1e-3,
        sparsity_threshold: float = 1e-3,
        random_seed: int = 42,
    ):
        self.archetypes = list(archetypes) if archetypes is not None else list(MacroConceptArchetype)
        self.regularization_alpha = regularization_alpha
        self.sparsity_threshold = sparsity_threshold
        self.random_seed = random_seed

    @property
    def proposer_id(self) -> str:
        return "ncm_structural_proposer_v1"

    def fit_and_propose(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        candidate_prefix: str = "H-NCM",
    ) -> list[CandidateProposal]:
        """Decompose discovery into macroscopic concepts and fit micro-parameters."""
        proposals: list[CandidateProposal] = []
        if len(df) < 30 or not feature_cols or target_col not in df.columns:
            return proposals

        y = df[target_col].to_numpy(dtype=float)
        # Drop NaNs or infinite values
        valid_mask = np.isfinite(y)
        if np.sum(valid_mask) < 20:
            return proposals

        X_base = df[feature_cols].to_numpy(dtype=float)
        valid_mask = valid_mask & np.all(np.isfinite(X_base), axis=1)
        y = y[valid_mask]
        X_base = X_base[valid_mask]
        n_samples = len(y)

        # Iterate over macroscopic concept archetypes
        for arch in self.archetypes:
            proposal = self._fit_archetype(
                arch=arch,
                X=X_base,
                y=y,
                feature_names=feature_cols,
                candidate_id=f"{candidate_prefix}-{arch.value[:8].upper()}-{len(proposals)+1:02d}",
            )
            if proposal is not None:
                proposals.append(proposal)

        return proposals

    def _fit_archetype(
        self,
        arch: MacroConceptArchetype,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        candidate_id: str,
    ) -> CandidateProposal | None:
        """Fit a specific macroscopic concept and return CandidateProposal."""
        n_samples, n_feats = X.shape

        if arch == MacroConceptArchetype.BOUNDED_RATIONAL_SATURATION:
            # y = (alpha_0 + sum_i alpha_i x_i) / (1 + sum_i beta_i |x_i|)
            abs_X = np.abs(X)
            denom = 1.0 + 0.1 * np.mean(abs_X, axis=1)
            y_scaled = y * denom
            r_num = Ridge(alpha=self.regularization_alpha).fit(X, y_scaled)
            pruned_coef = np.where(np.abs(r_num.coef_) < self.sparsity_threshold, 0.0, r_num.coef_)
            pred_num = X @ pruned_coef + float(r_num.intercept_)
            y_pred = pred_num / denom

            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = float(1.0 - (ss_res / max(1e-12, ss_tot)))

            active_terms = int(np.sum(np.abs(pruned_coef) > 0))
            top_feat_idx = int(np.argmax(np.abs(pruned_coef))) if len(pruned_coef) > 0 else 0
            formula = f"RationalHill({feature_names[top_feat_idx]}, num_scale={pruned_coef[top_feat_idx]:.3f}, denom_sat=0.1)"

            return CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=candidate_id,
                formula=formula,
                strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
                declared_scale_invariant=False,
                declared_nominal_dependencies=tuple(feature_names),
                parameters={
                    "r2": r2,
                    "coefs": pruned_coef.tolist(),
                    "intercept": float(r_num.intercept_),
                    "active_terms": active_terms,
                },
                metadata={
                    "archetype": arch.value,
                    "r2": r2,
                    "active_terms": active_terms,
                    "complexity": active_terms + 2,
                    "tier": "next_concept_modeling",
                },
            )

        elif arch == MacroConceptArchetype.CONSERVED_DAMPED_OSCILLATOR:
            # y_ddot + gamma y_dot + omega0^2 y = 0
            # Construct synthetic derivatives from first feature or X[:, 0]
            x0 = X[:, 0]
            dx = np.gradient(x0)
            design = np.column_stack([x0, dx])

            r_osc = Ridge(alpha=self.regularization_alpha, fit_intercept=False).fit(design, y)
            pruned_coef = np.where(np.abs(r_osc.coef_) < self.sparsity_threshold, 0.0, r_osc.coef_)
            y_pred = design @ pruned_coef

            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = float(1.0 - (ss_res / max(1e-12, ss_tot)))

            active_terms = int(np.sum(np.abs(pruned_coef) > 0))
            formula = f"DampedOscillator({feature_names[0]}, k={pruned_coef[0]:.3f}, gamma={pruned_coef[1]:.3f})"

            return CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=candidate_id,
                formula=formula,
                strategy_archetype=StrategyArchetype.TACTICAL,
                declared_scale_invariant=True,
                declared_nominal_dependencies=(feature_names[0],),
                parameters={
                    "r2": r2,
                    "omega_sq": float(pruned_coef[0]),
                    "gamma": float(pruned_coef[1]),
                    "active_terms": active_terms,
                },
                metadata={
                    "archetype": arch.value,
                    "r2": r2,
                    "active_terms": active_terms,
                    "complexity": max(1, active_terms + 1),
                    "tier": "next_concept_modeling",
                },
            )

        elif arch == MacroConceptArchetype.DIFFUSION_ADVECTION_CONTINUUM:
            # dy/dt = -v * dx/ds + D * d^2x/ds^2
            grad1 = np.gradient(X, axis=0)
            grad2 = np.gradient(grad1, axis=0)
            design = np.column_stack([X, grad1[:, :2], grad2[:, :2]]) if n_feats >= 2 else np.column_stack([X, grad1, grad2])

            r_pde = Ridge(alpha=self.regularization_alpha, fit_intercept=False).fit(design, y)
            pruned_coef = np.where(np.abs(r_pde.coef_) < self.sparsity_threshold, 0.0, r_pde.coef_)
            y_pred = design @ pruned_coef

            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = float(1.0 - (ss_res / max(1e-12, ss_tot)))

            active_terms = int(np.sum(np.abs(pruned_coef) > 0))
            formula = f"AdvectionDiffusion({feature_names[0]}, advection_v={pruned_coef[0]:.3f}, diff_D={pruned_coef[-1]:.3f})"

            return CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=candidate_id,
                formula=formula,
                strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
                declared_scale_invariant=True,
                declared_nominal_dependencies=tuple(feature_names[:2]),
                parameters={
                    "r2": r2,
                    "pde_coefs": pruned_coef[:4].tolist(),
                    "active_terms": active_terms,
                },
                metadata={
                    "archetype": arch.value,
                    "r2": r2,
                    "active_terms": active_terms,
                    "complexity": max(1, active_terms + 1),
                    "tier": "next_concept_modeling",
                },
            )

        elif arch == MacroConceptArchetype.CYCLIC_HARMONIC_REGIME:
            # y = a1 sin(omega t) + a2 cos(omega t) + beta x
            t = np.linspace(0, 4 * np.pi, n_samples)
            sin_t = np.sin(t)
            cos_t = np.cos(t)
            design = np.column_stack([sin_t, cos_t, X[:, 0]])

            r_harm = Ridge(alpha=self.regularization_alpha, fit_intercept=False).fit(design, y)
            pruned_coef = np.where(np.abs(r_harm.coef_) < self.sparsity_threshold, 0.0, r_harm.coef_)
            y_pred = design @ pruned_coef

            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = float(1.0 - (ss_res / max(1e-12, ss_tot)))

            active_terms = int(np.sum(np.abs(pruned_coef) > 0))
            formula = f"CyclicHarmonicRegime({feature_names[0]}, amp_sin={pruned_coef[0]:.3f}, amp_cos={pruned_coef[1]:.3f})"

            return CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=candidate_id,
                formula=formula,
                strategy_archetype=StrategyArchetype.TACTICAL,
                declared_scale_invariant=False,
                declared_nominal_dependencies=(feature_names[0],),
                parameters={
                    "r2": r2,
                    "harmonic_coefs": pruned_coef.tolist(),
                    "active_terms": active_terms,
                },
                metadata={
                    "archetype": arch.value,
                    "r2": r2,
                    "active_terms": active_terms,
                    "complexity": max(1, active_terms + 1),
                    "tier": "next_concept_modeling",
                },
            )

        return None
