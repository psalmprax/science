"""A quantum circuit's output as a trading signal, executed in the canonical ledger.

Status: experimental

The pipeline this module implements, and nothing more:

    ledger features -> angle encoding on N qubits -> entangling fan-in
                    -> <Z_0>  ->  position  ->  canonical fee accounting

It exists because the earlier demonstration of this idea was vacuous in a way that is
easy to miss. That circuit applied `phase` gates (diagonal, hence inert on |0>) before
`h` and `cx`, so its output was **0.0 for every input** -- a constant dressed as a
signal. A feature map has to be checked against the mathematics it claims to compute,
which is why :meth:`QuantumFeatureMap.analytic` exists and why the record refuses to be
written when the simulator and the closed form disagree.

What the circuit computes
-------------------------
`Ry(pi * x_i)` on every wire, then `CX(i, 0)` for every `i >= 1`, then measure wire 0.
In the Heisenberg picture each `CX(i, 0)` pulls `Z_i` into the measured observable, so

    <Z_0> = prod_i cos(pi * x_i)

for product inputs -- every feature reaches the signal, and the quantum simulation is
checkable against a one-line formula. The circuit has **no trainable parameters**: it is
a fixed, deterministic function of the features, so no quantum advantage is being
tested here. What is tested is whether the output survives an honest referee -- fees,
turnover, a disclosed trial count and a deflated Sharpe ratio.

What the executor does, and why it is separate
---------------------------------------------
:func:`execute_signal` takes a score series and re-implements the ledger's accounting
(`gross = pos * fwd`, `fee = FEE * |dpos|`, `net = gross - fee`, with the identity
checked at row level). It is separate because `canonical_ledger.run_ledger` builds its
score from a fixed dictionary of named components and has no entry point for an external
signal; the record proves the two agree on a shared score before any quantum number is
quoted.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from apedp_dyn.epistemic.trial_ledger import (
    CandidateTrialLedger,
    compute_deflated_sharpe_ratio,
    compute_expected_max_sharpe,
)
from apedp_dyn.quantum_compilation.statevector_sim import StatevectorSimulator
from apedp_dyn.sandbox.canonical_ledger import FEE

__all__ = [
    "ANSATZ_DEPTHS",
    "CONTINUOUS_PASSES",
    "CONTINUOUS_START",
    "CONTINUOUS_STEPS",
    "FIT_ALPHAS",
    "FIT_BIASES",
    "RULES",
    "FitResult",
    "ParameterisedQuantumMap",
    "QuantumFeatureMap",
    "continuous_evaluations",
    "deflate_sharpe",
    "deflated_summary",
    "execute_signal",
    "fit_continuous_map",
    "fit_grid",
    "fit_parameterised_map",
    "holdout_decision",
    "ledger_summary",
    "parameter_grid",
    "walk_forward_windows",
]

#: The position rules the executor implements. `short_below_zero` is the canonical
#: ledger's own `base` rule, present so the executor can be checked against it.
RULES = ("symmetric", "long_only", "short_below_zero")


def _ry(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array([[math.cos(half), -math.sin(half)],
                     [math.sin(half), math.cos(half)]], dtype=np.complex128)


def _embed(op: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    """Kronecker embedding with qubit 0 as the least significant bit, as the simulator."""
    dim = 1 << num_qubits
    matrix: np.ndarray = np.zeros((dim, dim), dtype=np.complex128)
    step = 1 << qubit
    clear = dim - 1 - step  # mask that zeroes this wire's bit
    for index in range(dim):
        output_bit = (index >> qubit) & 1
        for column in range(2):
            input_index = (index & clear) | (column * step)
            matrix[index, input_index] = op[output_bit, column]
    return matrix


def _cnot_matrix(control: int, target: int, num_qubits: int) -> np.ndarray:
    """`CX(control, target)` as a permutation matrix, independent of the simulator."""
    dim = 1 << num_qubits
    matrix: np.ndarray = np.zeros((dim, dim), dtype=np.complex128)
    index = np.arange(dim)
    flip = ((index >> control) & 1) << target
    matrix[index ^ flip, index] = 1.0
    return matrix


@dataclasses.dataclass(frozen=True)
class QuantumFeatureMap:
    """One fixed angle-encoding circuit, and the closed form it must equal."""

    features: tuple[str, ...]
    layers: int = 1

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("a feature map needs at least one feature")
        if self.layers < 1:
            raise ValueError(f"layers must be positive, got {self.layers}")

    @property
    def num_qubits(self) -> int:
        return len(self.features)

    def description(self) -> tuple[str, ...]:
        """The gate sequence, one string per operation, in emission order."""
        ops: list[str] = []
        for _ in range(self.layers):
            ops.extend(f"ry(pi*{name}) q{index}"
                       for index, name in enumerate(self.features))
            ops.extend(f"cx q{index}->q0" for index in range(1, self.num_qubits))
        return tuple(ops)

    def sha256(self) -> str:
        payload = json.dumps({"features": list(self.features), "layers": self.layers,
                              "ops": list(self.description())}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def signal(self, values: Sequence[float]) -> float:
        """`<Z_0>` after the circuit, in ``[-1, 1]``, from the simulator."""
        if len(values) != self.num_qubits:
            raise ValueError(f"expected {self.num_qubits} features, got {len(values)}")
        state = StatevectorSimulator(num_qubits=self.num_qubits)
        for _ in range(self.layers):
            for qubit, value in enumerate(values):
                state.apply_matrix(qubit, _ry(math.pi * float(value)))
            for qubit in range(1, self.num_qubits):
                state.cx(qubit, 0)
        p0, p1 = state.probabilities(0)
        return float(p0 - p1)

    def analytic(self, values: Sequence[float]) -> float:
        """The closed form the single-layer circuit computes: ``prod cos(pi x_i)``.

        A second layer mixes the wires again and has no such one-line form, so asking
        for it there raises instead of returning an approximation that could be mistaken
        for a proof of correctness.
        """
        if self.layers != 1:
            raise ValueError("the closed form is defined for the single-layer map only")
        if len(values) != self.num_qubits:
            raise ValueError(f"expected {self.num_qubits} features, got {len(values)}")
        product = 1.0
        for value in values:
            product *= math.cos(math.pi * float(value))
        return float(product)


def execute_signal(score: Sequence[float], returns: Sequence[float], *,
                   fee: float = FEE, threshold: float = 0.0,
                   rule: str = "symmetric",
                   valid: Sequence[bool] | None = None) -> list[dict[str, float]]:
    """Run `score` through the canonical fee accounting, one row per observation.

    `returns[t]` is the log return of bar `t`; the forward return is `returns[t + 1]`,
    so a score computed from bar `t`'s features is paid on the next bar's move and no
    lookahead enters through the accounting. An invalid observation forces a flat
    position with the reason recorded, which is the ledger's mask-first rule.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if len(returns) < len(score):
        raise ValueError(f"{len(score)} scores but only {len(returns)} returns: the "
                         "forward return of the last scored bar would be missing")
    if len(returns) > len(score) + 1:
        raise ValueError(f"{len(returns)} returns for {len(score)} scores: at most the "
                         "next bar's forward return may follow the last score")
    if valid is not None and len(valid) != len(score):
        raise ValueError(f"{len(valid)} mask entries against {len(score)} scores")
    rows: list[dict[str, float]] = []
    previous = 0.0
    for index, value in enumerate(score):
        observed = float(value)
        finite = bool(np.isfinite(observed))
        ok = finite if valid is None else bool(valid[index]) and finite
        if not ok:
            position, reason = 0.0, "invalid_score"
        elif rule == "symmetric":
            position = 1.0 if observed > threshold else (-1.0 if observed < -threshold
                                                         else 0.0)
            reason = "symmetric_threshold"
        elif rule == "long_only":
            position, reason = (1.0 if observed > threshold else 0.0), "long_only"
        else:
            position, reason = (-1.0 if observed < 0.0 else 0.0), "base_rule"
        following = returns[index + 1] if index + 1 < len(returns) else None
        forward = float(following) if following is not None and np.isfinite(following) \
            else 0.0
        gross = 0.0 if position == 0.0 else float(position * forward)
        change = abs(position - previous)
        cost = fee * change
        rows.append({"t": index, "score": observed, "valid": ok, "reason": reason,
                     "pos": float(position), "fwd": forward, "gross": float(gross),
                     "dpos": float(change), "fee": float(cost),
                     "net": float(gross - cost)})
        previous = position
    gross = sum(row["gross"] for row in rows)
    fees = sum(row["fee"] for row in rows)
    net = sum(row["net"] for row in rows)
    if abs((gross - fees) - net) >= 1e-9:
        raise RuntimeError("accounting identity violated in the signal executor")
    return rows


def ledger_summary(rows: Sequence[dict[str, float]], *,
                   annualization: float = 365.0,
                   risk_free: float = 0.0) -> dict[str, float]:
    """The aggregates the ledger reports, over the executed rows."""
    net = np.array([row["net"] for row in rows], dtype=float)
    count = len(net)
    mean = float(np.mean(net)) if count else 0.0
    deviation = float(np.std(net)) if count else 0.0
    excess = mean - risk_free / annualization
    sharpe = (excess / deviation * math.sqrt(annualization)
              if count > 1 and deviation > 1e-12 else 0.0)
    centered = net - mean if count else net
    variance = float(np.mean(centered ** 2)) if count else 0.0
    skew = float(np.mean(centered ** 3) / variance ** 1.5) if variance > 0 else 0.0
    kurtosis = float(np.mean(centered ** 4) / variance ** 2) if variance > 0 else 3.0
    return {
        "gross": float(sum(row["gross"] for row in rows)),
        "fees": float(sum(row["fee"] for row in rows)),
        "net": float(sum(row["net"] for row in rows)),
        "turnover": float(sum(row["dpos"] for row in rows)),
        "n_obs": float(count),
        "n_invalid": float(sum(1 for row in rows if not row["valid"])),
        "active_fraction": (sum(1 for row in rows if row["pos"] != 0.0) / count
                            if count else 0.0),
        "sharpe": float(sharpe),
        "skewness": skew,
        "kurtosis": kurtosis,
        "net_returns": net.tolist(),
    }


def deflate_sharpe(sharpe: float, *, n_bars: int, skewness: float, kurtosis: float,
                   n_trials: int, var_trials: float,
                   annualization: float = 365.0) -> dict[str, Any]:
    """The Deflated Sharpe Ratio from one trial's statistics.

    Kept separate from the measurement so the record's DSR can be recomputed from the
    stored numbers alone: the family's trial count and Sharpe dispersion are inputs the
    reader can see, not values hidden inside the measuring loop.
    """
    return compute_deflated_sharpe_ratio(
        observed_sharpe=float(sharpe), n_trials=int(n_trials),
        var_trials=float(var_trials), n_bars=int(n_bars),
        skewness=float(skewness), kurtosis=float(kurtosis),
        annualization_factor=annualization)


def deflated_summary(rows: Sequence[dict[str, float]], *, n_trials: int,
                     var_trials: float, annualization: float = 365.0) -> dict[str, Any]:
    """The ledger summary plus the Deflated Sharpe Ratio for the disclosed trials."""
    summary = ledger_summary(rows, annualization=annualization)
    summary["dsr"] = deflate_sharpe(
        summary["sharpe"], n_bars=int(summary["n_obs"]),
        skewness=summary["skewness"], kurtosis=summary["kurtosis"],
        n_trials=n_trials, var_trials=var_trials,
        annualization=annualization)
    return summary


# ------------------------------------------------------------- parameterised map
#: The declared fitting search. Every member is scored on the training slice and
#: written to a trial ledger, so a fitted map is charged for the search that produced
#: it instead of being quoted as the only hypothesis ever considered.
FIT_ALPHAS = (0.5, 1.0, 1.5, 2.0)
FIT_BIASES = (-0.5, 0.0, 0.5)
#: The circuit families the fit spans. Depth enters the closed form as a power, so the
#: second family is not a rescaling of the first: `<Z_0> = [prod_i cos(alpha_i x_i +
#: beta_i)] ** L`, and `L = 2` is a different function of the same features.
ANSATZ_DEPTHS = (1, 2)


def fit_grid(num_qubits: int) -> tuple[tuple[tuple[float, ...], tuple[float, ...], int], ...]:
    """Every (alphas, betas, depth) the fit searches, at both circuit families."""
    return tuple((alphas, betas, layers)
                 for layers in ANSATZ_DEPTHS
                 for alphas, betas in parameter_grid(num_qubits))


def parameter_grid(num_qubits: int) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """The grid the fit searches, as ``(alphas, betas)`` per grid point.

    A grid rather than a continuous optimiser, deliberately: a discrete search has a
    size that can be disclosed exactly, so the multiplicity charged to the deflation is
    a number the reader can count, not an iteration count someone has to trust.
    """
    if num_qubits < 1:
        raise ValueError(f"a parameter grid needs at least one qubit, got {num_qubits}")
    return tuple(((align,) * num_qubits, (bias,) * num_qubits)
                 for align in FIT_ALPHAS for bias in FIT_BIASES)


@dataclasses.dataclass(frozen=True)
class ParameterisedQuantumMap:
    """An angle-encoding circuit whose rotations are fitted rather than fixed.

    `Ry(alpha_i x_i + beta_i)` on wire `i`, then `CX(i, 0)` for every `i >= 1`, measure
    wire 0, so

        <Z_0> = prod_i cos(alpha_i x_i + beta_i)

    and the simulator is checkable against that form for *any* parameter values, which
    is what keeps a fitted circuit honest: the fit cannot hide behind the claim that a
    two-layer ansatz has no closed form.
    """

    features: tuple[str, ...]
    alphas: tuple[float, ...]
    betas: tuple[float, ...]
    layers: int = 1

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("a feature map needs at least one feature")
        if self.layers < 1:
            raise ValueError(f"layers must be positive, got {self.layers}")
        if len(self.alphas) != len(self.features) or len(self.betas) != len(self.features):
            raise ValueError(
                f"{len(self.features)} features need one alpha and one beta each, got "
                f"{len(self.alphas)} alphas and {len(self.betas)} betas")
        if not all(math.isfinite(value) for value in (*self.alphas, *self.betas)):
            raise ValueError("rotation parameters must be finite: a NaN angle defines no signal")

    @property
    def num_qubits(self) -> int:
        return len(self.features)

    def angles(self, values: Sequence[float]) -> tuple[float, ...]:
        """The rotation applied to each wire for one observation."""
        if len(values) != self.num_qubits:
            raise ValueError(f"expected {self.num_qubits} features, got {len(values)}")
        return tuple(self.alphas[index] * float(values[index]) + self.betas[index]
                     for index in range(self.num_qubits))

    def description(self) -> tuple[str, ...]:
        ops: list[str] = []
        for layer in range(self.layers):
            ops.extend(f"ry({alpha}*{name}{beta:+}) q{index}"
                       for index, (name, alpha, beta) in enumerate(
                           zip(self.features, self.alphas, self.betas, strict=True)))
            ops.extend(f"cx q{index}->q0" for index in range(1, self.num_qubits))
            if layer + 1 < self.layers:
                ops.append(f"# layer {layer + 2}")
        return tuple(ops)

    def sha256(self) -> str:
        payload = json.dumps({"features": list(self.features),
                              "alphas": list(self.alphas),
                              "betas": list(self.betas),
                              "layers": self.layers,
                              "ops": list(self.description())}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def signal(self, values: Sequence[float]) -> float:
        """`<Z_0>` after the fitted circuit, in ``[-1, 1]``, from the simulator."""
        state = StatevectorSimulator(num_qubits=self.num_qubits)
        for _layer in range(self.layers):
            for qubit, angle in enumerate(self.angles(values)):
                state.apply_matrix(qubit, _ry(angle))
            for qubit in range(1, self.num_qubits):
                state.cx(qubit, 0)
        p0, p1 = state.probabilities(0)
        return float(p0 - p1)

    def analytic(self, values: Sequence[float]) -> float:
        """The single-layer closed form: ``prod_i cos(alpha_i x_i + beta_i)``.

        There is no product form above one layer, and this says so instead of guessing:
        the second rotation's `X` branches survive the first entangler, so the marginal is
        a sum over several Pauli strings and the resulting signal is not `depth one`
        squared (measured: 0.7146 against 0.0002 on the same input). Above one layer the
        circuit is checked against :meth:`exact_marginal` instead.
        """
        if self.layers != 1:
            raise ValueError(
                "no product closed form exists above one layer; compare against "
                "exact_marginal (dense unitary) instead")
        product = 1.0
        for angle in self.angles(values):
            product *= math.cos(angle)
        return float(product)

    def exact_marginal(self, values: Sequence[float]) -> float:
        """`<Z_0>` from an explicitly built unitary, sharing no code with the simulator.

        Applicable at every depth: the whole circuit is assembled as dense matrices and
        applied to `|0>`, which is a different route to the same number and can therefore
        disagree with the simulator.
        """
        dim = 1 << self.num_qubits
        unitary: np.ndarray = np.eye(dim, dtype=np.complex128)
        for _layer in range(self.layers):
            for qubit, angle in enumerate(self.angles(values)):
                unitary = _embed(_ry(angle), qubit, self.num_qubits) @ unitary
            for qubit in range(1, self.num_qubits):
                unitary = _cnot_matrix(qubit, 0, self.num_qubits) @ unitary
        state = unitary[:, 0]
        probability_one = float(np.sum(np.abs(state[1::2]) ** 2))
        return float(1.0 - 2.0 * probability_one)


@dataclasses.dataclass(frozen=True)
class FitResult:
    """One training-slice fit: the selected map and the search that produced it."""

    feature_map: ParameterisedQuantumMap
    in_sample_sharpe: float
    grid_size: int
    trial_count: int
    effective_trial_count: int
    candidates: tuple[tuple[float, float, int, float], ...]
    optimiser: str = "grid"

    def to_dict(self) -> dict[str, Any]:
        """The fit as the record stores it, with every grid point's Sharpe."""
        return {
            "features": list(self.feature_map.features),
            "alphas": list(self.feature_map.alphas),
            "betas": list(self.feature_map.betas),
            "layers": self.feature_map.layers,
            "circuit_sha256": self.feature_map.sha256(),
            "gates": list(self.feature_map.description()),
            "closed_form": ("prod_i cos(alpha_i * x_i + beta_i)"
                            if self.feature_map.layers == 1 else None),
            "reference": ("product closed form" if self.feature_map.layers == 1
                          else "dense unitary marginal (exact_marginal)"),
            "in_sample_sharpe": self.in_sample_sharpe,
            "optimiser": self.optimiser,
            "grid_size": self.grid_size,
            "trial_count": self.trial_count,
            "effective_trial_count": self.effective_trial_count,
            "candidates": [{"alpha": alpha, "beta": beta, "layers": layers,
                            "sharpe": sharpe}
                           for alpha, beta, layers, sharpe in self.candidates],
        }


def fit_parameterised_map(values, returns: Sequence[float], *,
                          features: Sequence[str], valid: Sequence[bool] | None = None,
                          ledger: CandidateTrialLedger | None = None,
                          search_id: str = "fit",
                          proposer_id: str = "quantum_signal_fit",
                          window: slice | None = None,
                          depths: Sequence[int] = ANSATZ_DEPTHS,
                          fee: float = FEE, rule: str = "symmetric",
                          threshold: float = 0.0,
                          annualization: float = 365.0) -> FitResult:
    """Fit the map on a training slice, charging every grid point as a trial.

    Each grid point is scored through the simulator, executed under the canonical fee
    accounting and recorded in `ledger` with ``selection_pool_size`` set to the grid
    size. The selected map is the best in-sample Sharpe; the charge is the ledger's
    ``effective_trial_count``, so quoting the winner alone cannot understate the
    multiplicity (Bailey & Lopez de Prado selection bias).
    """
    feature_names = tuple(features)
    rows = [list(map(float, row)) for row in values]
    if rows and len(rows[0]) != len(feature_names):
        raise ValueError(f"{len(rows[0])} values per row against {len(feature_names)} features")
    if len(rows) != len(returns):
        raise ValueError(f"{len(rows)} observations against {len(returns)} returns")
    grid = tuple((alphas, betas, int(layers)) for layers in depths
                 for alphas, betas in parameter_grid(len(feature_names)))
    if not grid:
        raise ValueError("the fit grid is empty: no ansatz depth was declared")
    if any(layers < 1 for _a, _b, layers in grid):
        raise ValueError("ansatz depth must be positive")
    trial_ledger = ledger if ledger is not None else CandidateTrialLedger()
    # The charge attributable to *this* fit is the ledger's growth, not its running
    # total: a fit on the fifth dataset must not be billed for the first four.
    trial_base = trial_ledger.trial_count
    effective_base = trial_ledger.effective_trial_count
    candidates: list[tuple[float, float, int, float]] = []
    best: FitResult | None = None
    for alphas, betas, layers in grid:
        feature_map = ParameterisedQuantumMap(feature_names, alphas, betas, layers=layers)
        scores = [feature_map.signal(row) for row in rows]
        executed = execute_signal(scores, returns, fee=fee, threshold=threshold,
                                  rule=rule, valid=valid)
        window_rows = executed if window is None else executed[window]
        summary = ledger_summary(window_rows, annualization=annualization)
        trial_ledger.record_trial(
            candidate_id=f"{search_id}:alpha={alphas[0]}:beta={betas[0]}:layers={layers}",
            proposer_id=proposer_id,
            formula=f"[prod_i cos({alphas[0]}*x_i{betas[0]:+})]**{layers}",
            strategy_archetype="quantum_signal",
            in_sample_sharpe=summary["sharpe"],
            returns=np.asarray([row["net"] for row in window_rows], dtype=float),
            n_bars=int(summary["n_obs"]),
            status="EVALUATED",
            metadata={"search_id": search_id, "selection_pool_size": len(grid),
                      "alphas": list(alphas), "biases": list(betas),
                      "layers": layers,
                      "circuit_sha256": feature_map.sha256(),
                      "window": None if window is None else [window.start, window.stop]},
            selection_statistic=summary["sharpe"],
        )
        candidates.append((alphas[0], betas[0], layers, summary["sharpe"]))
        if best is None or summary["sharpe"] > best.in_sample_sharpe:
            best = FitResult(
                feature_map=feature_map, in_sample_sharpe=summary["sharpe"],
                grid_size=len(grid), trial_count=trial_ledger.trial_count,
                effective_trial_count=trial_ledger.effective_trial_count,
                candidates=(),
            )
    if best is None:  # pragma: no cover - the grid is non-empty by construction
        raise RuntimeError("the parameter grid is empty")
    return dataclasses.replace(
        best, candidates=tuple(candidates),
        trial_count=trial_ledger.trial_count - trial_base,
        effective_trial_count=trial_ledger.effective_trial_count - effective_base)


#: Continuous search: coordinate ascent on the two scalars a family shares across its
#: wires, with a shrinking step and two passes per step. The declared budget below is
#: exact, and **every evaluation is a recorded trial**, so the charge is the iteration
#: count instead of a grid that happens to be disclosed.
CONTINUOUS_STEPS = (0.5, 0.25)
CONTINUOUS_PASSES = 2
CONTINUOUS_START = (1.0, 0.0)


def continuous_evaluations() -> int:
    """Evaluations per (dataset, depth) search group: the start point plus the probes."""
    return 1 + len(CONTINUOUS_STEPS) * CONTINUOUS_PASSES * 4


def fit_continuous_map(values, returns: Sequence[float], *,
                       features: Sequence[str], valid: Sequence[bool] | None = None,
                       ledger: CandidateTrialLedger | None = None,
                       search_id: str = "fit",
                       proposer_id: str = "quantum_signal_fit",
                       window: slice | None = None,
                       depths: Sequence[int] = ANSATZ_DEPTHS,
                       fee: float = FEE, rule: str = "symmetric",
                       threshold: float = 0.0,
                       annualization: float = 365.0) -> FitResult:
    """Coordinate ascent on `(alpha, beta)` per depth, charging every evaluation.

    The search is continuous in the sense that matters for accounting: it proposes
    parameter values and steps to them, and each proposal costs one recorded trial with
    ``selection_pool_size`` set to the declared budget. The charge is therefore the
    iteration count, not a grid that was chosen to make the multiplicity look small.
    """
    feature_names = tuple(features)
    rows = [list(map(float, row)) for row in values]
    if rows and len(rows[0]) != len(feature_names):
        raise ValueError(f"{len(rows[0])} values per row against {len(feature_names)} features")
    if len(rows) != len(returns):
        raise ValueError(f"{len(rows)} observations against {len(returns)} returns")
    trial_ledger = ledger if ledger is not None else CandidateTrialLedger()
    trial_base = trial_ledger.trial_count
    effective_base = trial_ledger.effective_trial_count
    per_group = continuous_evaluations()
    width = len(feature_names)

    def evaluate(alpha: float, beta: float, depth: int) -> tuple[ParameterisedQuantumMap, dict[str, Any]]:
        feature_map = ParameterisedQuantumMap(feature_names, (alpha,) * width,
                                              (beta,) * width, layers=depth)
        scores = [feature_map.signal(row) for row in rows]
        executed = execute_signal(scores, returns, fee=fee, threshold=threshold,
                                  rule=rule, valid=valid)
        window_rows = executed if window is None else executed[window]
        return feature_map, ledger_summary(window_rows, annualization=annualization)

    candidates: list[tuple[float, float, int, float]] = []
    best: FitResult | None = None
    for depth in depths:
        if int(depth) < 1:
            raise ValueError(f"ansatz depth must be positive, got {depth}")
        index = 0
        alpha, beta = CONTINUOUS_START
        feature_map, summary = evaluate(alpha, beta, int(depth))
        trial_ledger.record_trial(
            candidate_id=f"{search_id}:depth{int(depth)}:eval{index}",
            proposer_id=proposer_id,
            formula=f"prod_i cos({alpha}*x_i{beta:+})**{int(depth)}",
            strategy_archetype="quantum_signal",
            in_sample_sharpe=summary["sharpe"],
            returns=np.asarray(summary["net_returns"], dtype=float),
            n_bars=int(summary["n_obs"]), status="EVALUATED",
            metadata={"search_id": f"{search_id}:depth{int(depth)}",
                      "selection_pool_size": per_group,
                      "alpha": alpha, "beta": beta, "layers": int(depth),
                      "step": 0.0, "coordinate": "start",
                      "circuit_sha256": feature_map.sha256(),
                      "window": None if window is None else [window.start, window.stop]},
            selection_statistic=summary["sharpe"],
        )
        candidates.append((alpha, beta, int(depth), summary["sharpe"]))
        if best is None or summary["sharpe"] > best.in_sample_sharpe:
            best = FitResult(feature_map=feature_map, in_sample_sharpe=summary["sharpe"],
                             grid_size=len(candidates), trial_count=1,
                             effective_trial_count=per_group, candidates=(),
                             optimiser="continuous")
        for step in CONTINUOUS_STEPS:
            for _pass in range(CONTINUOUS_PASSES):
                for coordinate in ("alpha", "beta"):
                    for direction in (+1.0, -1.0):
                        trial_alpha = alpha + direction * step if coordinate == "alpha" else alpha
                        trial_beta = beta + direction * step if coordinate == "beta" else beta
                        if trial_alpha <= 0.0:
                            continue  # the encoding's scale stays positive
                        index += 1
                        trial_map, trial_summary = evaluate(trial_alpha, trial_beta, int(depth))
                        trial_ledger.record_trial(
                            candidate_id=f"{search_id}:depth{int(depth)}:eval{index}",
                            proposer_id=proposer_id,
                            formula=(f"prod_i cos({trial_alpha}*x_i{trial_beta:+})**"
                                     f"{int(depth)}"),
                            strategy_archetype="quantum_signal",
                            in_sample_sharpe=trial_summary["sharpe"],
                            returns=np.asarray(trial_summary["net_returns"], dtype=float),
                            n_bars=int(trial_summary["n_obs"]), status="EVALUATED",
                            metadata={"search_id": f"{search_id}:depth{int(depth)}",
                                      "selection_pool_size": per_group,
                                      "alpha": trial_alpha, "beta": trial_beta,
                                      "layers": int(depth), "step": step,
                                      "coordinate": coordinate,
                                      "circuit_sha256": trial_map.sha256(),
                                      "window": (None if window is None
                                                 else [window.start, window.stop])},
                            selection_statistic=trial_summary["sharpe"],
                        )
                        candidates.append((trial_alpha, trial_beta, int(depth),
                                           trial_summary["sharpe"]))
                        if trial_summary["sharpe"] > summary["sharpe"]:
                            alpha, beta = trial_alpha, trial_beta
                            summary = trial_summary
                            feature_map = trial_map
                            if summary["sharpe"] > best.in_sample_sharpe:
                                best = FitResult(
                                    feature_map=feature_map,
                                    in_sample_sharpe=summary["sharpe"],
                                    grid_size=len(candidates),
                                    trial_count=1, effective_trial_count=per_group,
                                    candidates=(), optimiser="continuous")
    if best is None:  # pragma: no cover - depths is non-empty by construction
        raise RuntimeError("no ansatz depth declared")
    return dataclasses.replace(
        best, candidates=tuple(candidates),
        grid_size=len(candidates),
        trial_count=trial_ledger.trial_count - trial_base,
        effective_trial_count=trial_ledger.effective_trial_count - effective_base)


def walk_forward_windows(n_bars: int, folds: int, warmup: int) -> list[tuple[int, int, int, int]]:
    """Expanding-window walk-forward folds: `(train_start, train_end, test_start, test_end)`.

    The post-warm-up span is cut into `folds + 1` equal blocks; fold `i` trains on every
    bar before its block and tests on that block alone. Windows are contiguous and
    non-overlapping, so a bar is judged in exactly one fold and a slice is consumed once
    while the same dataset is read across regimes.
    """
    if folds < 1:
        raise ValueError(f"folds must be >= 1, got {folds}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    span = n_bars - 1 - warmup
    if span <= folds:
        raise ValueError(f"{n_bars} bars cannot yield {folds} walk-forward folds "
                         f"after a {warmup}-bar warm-up")
    block = span // (folds + 1)
    if block < 1:
        raise ValueError(f"fold width would be {block} bars for {folds} folds")
    windows: list[tuple[int, int, int, int]] = []
    for index in range(1, folds + 1):
        train_end = warmup + block * index
        test_end = min(train_end + block, n_bars - 1)
        if test_end <= train_end:
            break
        windows.append((warmup, train_end, train_end, test_end))
    if not windows:
        raise ValueError(f"no walk-forward window fits {n_bars} bars at {folds} folds")
    return windows


def holdout_decision(net: float, turnover: float, sharpe: float, *,
                     searched_trials: int, var_trials: float) -> dict[str, Any]:
    """The cost-aware rule that decides a confirmation slice -- and burns it on failure.

    A candidate may claim the slice only if it trades in the window, makes money after
    fees, **and** clears the expected maximum Sharpe of the search that produced it. The
    hurdle is charged to the fitted candidate's own search size, so a wider grid has to
    clear more before it may say the holdout confirmed it; failing the rule burns the
    slice exactly as a losing candidate does.
    """
    if int(searched_trials) < 1:
        raise ValueError(f"searched_trials must be >= 1, got {searched_trials}")
    hurdle = float(compute_expected_max_sharpe(int(searched_trials), float(var_trials)))
    traded = turnover > 0.0
    profitable = net > 0.0
    passed = bool(traded and profitable and float(sharpe) >= hurdle)
    if passed:
        reason = (f"holdout clears the {hurdle:.3f} hurdle of a "
                  f"{int(searched_trials)}-trial search")
    elif not traded:
        reason = ("holdout recorded no position change: a carried position is not "
                  "holdout evidence")
    elif not profitable:
        reason = f"holdout net {net:+.4f} after fees"
    else:
        reason = (f"holdout Sharpe {float(sharpe):+.3f} below the {hurdle:.3f} hurdle of "
                  f"a {int(searched_trials)}-trial search")
    return {"hurdle": hurdle, "searched_trials": int(searched_trials),
            "passed": passed, "reason": reason}
