"""Numeric Fingerprinting for fast zero-cost candidate expression deduplication.

Avoids CPU-expensive SymPy canonicalization during expression generation:
1. Evaluates candidate AST expressions over a deterministic, fixed probe grid.
2. Normalizes, quantizes, and hashes the output vector.
3. Detects trivial constant outputs, singularities, and numerical duplicates in nanoseconds.
"""
from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from apedp_dyn.discovery_l2.ast_generator import ASTNode


def make_probe_grid(
    variables: Sequence[str] | Mapping[str, Any],
    n_points: int = 32,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a deterministic, non-degenerate probe evaluation grid.

    Samples points in [0.6, 3.4] so that operators like sqrt, log, pow,
    and division evaluate without trivial domain singularities, while
    giving distinct non-collinear values across distinct variables.
    """
    var_names = list(variables.keys()) if isinstance(variables, Mapping) else list(variables)
    rng = np.random.RandomState(seed)

    grid: dict[str, np.ndarray] = {}
    base = np.linspace(0.6, 3.4, n_points)

    for i, name in enumerate(var_names):
        shift = (i * 0.37) % 0.8
        noise = rng.uniform(-0.05, 0.05, size=n_points)
        vals = base + shift + noise
        grid[name] = np.clip(vals, 0.2, 4.5)

    return grid


def make_bipolar_probe_grid(
    variables: Sequence[str] | Mapping[str, Any],
    n_points: int = 32,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a bipolar probe grid spanning both negative and positive values.

    Allows distinguishing abs(x) from x, even powers from odd powers, and
    sign-preserving nonlinearities (e.g. v*|v|), avoiding domain zeros.
    """
    var_names = list(variables.keys()) if isinstance(variables, Mapping) else list(variables)
    rng = np.random.RandomState(seed + 101)

    grid: dict[str, np.ndarray] = {}
    half = n_points // 2
    neg_base = np.linspace(-3.2, -0.4, half)
    pos_base = np.linspace(0.4, 3.2, n_points - half)
    base = np.concatenate([neg_base, pos_base])

    for i, name in enumerate(var_names):
        shift = ((i * 0.41) % 0.6) - 0.3
        noise = rng.uniform(-0.04, 0.04, size=n_points)
        vals = base + shift + noise
        # Ensure values stay away from zero singularity
        vals = np.where(np.abs(vals) < 0.15, np.sign(vals) * 0.25, vals)
        grid[name] = np.clip(vals, -4.5, 4.5)

    return grid


def compute_fingerprint(
    node: ASTNode,
    probe_env: Mapping[str, np.ndarray],
    bipolar_env: Mapping[str, np.ndarray] | None = None,
) -> tuple[bool, int, str | None]:
    """Compute 64-bit normalized numeric fingerprint for an AST node.

    Evaluates candidate on bipolar grid when algebraically valid (to distinguish
    abs(x) from x and odd/even functions), falling back to positive-only probe_env
    when domain singularities occur (e.g. log(x), x^(1/2)).

    Returns:
      (is_valid, fingerprint_hash, rejection_reason)
    """
    arr: np.ndarray | None = None
    domain_prefix = 0

    def _finite_count(a: np.ndarray) -> int:
        return int(np.isfinite(a).sum()) if a.size > 0 else 0

    if bipolar_env is not None:
        try:
            val_bipolar = np.asarray(node.evaluate(bipolar_env), dtype=float)
            # Accept only if enough finite response; otherwise fall through to
            # the positive grid (partial-NaN bipolar must not shadow it).
            if val_bipolar.size > 0 and _finite_count(val_bipolar) >= 8:
                arr = val_bipolar
                domain_prefix = 1
        except (ValueError, TypeError, ArithmeticError, KeyError, ZeroDivisionError):
            pass

    if arr is None:
        try:
            val_pos = np.asarray(node.evaluate(probe_env), dtype=float)
            if val_pos.size > 0:
                arr = val_pos
                domain_prefix = 2
        except (ValueError, TypeError, ArithmeticError, KeyError, ZeroDivisionError) as exc:
            return False, 0, f"eval_error: {exc}"

    if arr is None or arr.size == 0:
        return False, 0, "empty_eval"

    # Causally-honest nodes (rolling windows) legitimately emit leading NaNs;
    # overflow is judged on the finite response, not the NaN padding.
    # Behavior for fully-finite responses is unchanged.
    if np.any(np.abs(arr[np.isfinite(arr)]) > 1e12):
        return False, 0, "overflow"
    finite = arr[np.isfinite(arr)]
    if finite.size < 8:
        return False, 0, "non_finite"
    arr = finite

    std_y = float(np.std(arr))
    if std_y < 1e-11:
        return False, 0, "trivial_constant"

    # Scale-invariant standardization (affine normalization)
    norm = (arr - np.mean(arr)) / std_y

    # Sign-orientation: make first significant nonzero entry positive
    # so that theta * f(x) and -theta * (-f(x)) map to identical basis fingerprint
    first_sig_idx = np.where(np.abs(norm) > 1e-4)[0]
    if len(first_sig_idx) > 0 and norm[first_sig_idx[0]] < 0:
        norm = -norm

    quantized = np.round(norm, decimals=3)
    raw_bytes = bytes([domain_prefix]) + quantized.tobytes()
    fp_hash = int(zlib.crc32(raw_bytes))
    return True, fp_hash, None



class FingerprintDeduplicator:
    """Stateful cache of observed numeric fingerprints for high-speed candidate pruning."""

    def __init__(
        self,
        probe_env: Mapping[str, np.ndarray] | None = None,
        bipolar_env: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.probe_env = probe_env
        self.bipolar_env = bipolar_env
        self.seen_fingerprints: dict[int, int] = {}
        self.n_checked: int = 0
        self.n_pruned: int = 0

    def is_novel(
        self,
        node: ASTNode,
        probe_env: Mapping[str, np.ndarray] | None = None,
        bipolar_env: Mapping[str, np.ndarray] | None = None,
    ) -> bool:
        """Return True if candidate produces a novel, non-trivial numeric response."""
        env = probe_env if probe_env is not None else self.probe_env
        b_env = bipolar_env if bipolar_env is not None else self.bipolar_env
        if env is None and b_env is None:
            return True

        self.n_checked += 1
        is_valid, fp_hash, _reason = compute_fingerprint(node, env if env is not None else {}, b_env)
        if not is_valid:
            self.n_pruned += 1
            return False

        prev_comp = self.seen_fingerprints.get(fp_hash)
        if prev_comp is not None and node.complexity >= prev_comp:
            self.n_pruned += 1
            return False

        self.seen_fingerprints[fp_hash] = node.complexity
        return True


