"""APEDP Self-Validation Harness — tests the theoretical backbone of the platform.

Every check here is expected to name its **oracle**: the source of the value it
asserts against, obtained without running the code under test. Closed-form
results, hand-derived arithmetic and outcomes generated outside APEDP are
oracles. Re-deriving the implementation's own formula inside the check is not —
such a check can only fail if the code changes into something else that still
agrees with itself — so it is recorded as self-referential instead.
:class:`SelfValidationReport` stores the oracle of each check and reports how
many checks have one, which makes the circularity a measured quantity rather
than a hidden assumption. A harness cannot be fully independent of the code it
ships with; it can be explicit about which of its checks are.

Checks:
1. Bayesian Model Selection — vs the Gaussian-conjugate closed form
2. Evolutionary Search — vs the generating law, off-sample
3. Dimensional Analysis — vs the SI base-dimension definitions (pint)
4. Buckingham Pi — vs the rank-nullity count and dimensionlessness
5. Physical Anomaly Repair — vs the pole of the published law
6. Epistemic Tracking — structural invariants
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from itertools import pairwise
from tempfile import TemporaryDirectory

import numpy as np
import sympy as sp

from apedp.core import (
    BuckinghamPiEngine,
    ClosedLoopAssuranceSystem,
    DimensionalAssuranceEngine,
    ureg,
)
from apedp.discovery import BayesianRefinementEngine, ModelEvidence, MultiVariableSRE
from apedp.epistemic import (
    AdversarialDeliberationTournament,
    EpistemicGraph,
    LifecycleManager,
    RequirementsTraceabilityMatrix,
)
from apedp.metrology import AnalyticalUncertaintyPropagator
from apedp.persistence import (
    ModelStore,
    deserialize_model,
    from_json,
    serialize_model,
    to_json,
)
from apedp.repair import PhysicalAnomalyRepairEngine
from apedp.scenarios import APEDPScenarioSuite

logger = logging.getLogger(__name__)


class SelfValidationReport:
    """Aggregates self-validation results."""

    def __init__(self):
        self.results: list[dict] = []

    def add(
        self,
        test_name: str,
        passed: bool,
        details: str = "",
        error: str = "",
        oracle: str = "",
    ) -> None:
        """Record one check.

        ``oracle`` names where the asserted value came from, independently of the
        code under test (e.g. "Gaussian-Gaussian conjugacy, derived by hand"). An
        empty oracle marks the check as self-referential.
        """
        passed = bool(passed)
        self.results.append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "error": error,
            "oracle": oracle,
        })
        status = "PASS" if passed else "FAIL"
        logger.info("[%s] %s — %s", status, test_name, details)

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        with_oracle = sum(1 for r in self.results if r.get("oracle"))
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed/total*100:.1f}%" if total > 0 else "N/A",
            "checks_with_independent_oracle": with_oracle,
            "self_referential_checks": total - with_oracle,
            "results": self.results,
        }

    def to_json(self, output_path: str = "self_validation_report.json") -> str:
        report = self.summary()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        return json.dumps(report, indent=2)


class SelfValidator:
    """Validates the theoretical backbone of APEDP."""

    def __init__(self) -> None:
        self.report = SelfValidationReport()
        self.engine = DimensionalAssuranceEngine()
        self.platform = ClosedLoopAssuranceSystem(self.engine)
        self.bayes = BayesianRefinementEngine()
        self.repair = PhysicalAnomalyRepairEngine(self.platform)
        self.propagator = AnalyticalUncertaintyPropagator()
        self.graph = EpistemicGraph()
        self.scenarios = APEDPScenarioSuite()

    # ------------------------------------------------------------------
    # Independent oracles
    # ------------------------------------------------------------------

    # Gaussian-Gaussian conjugacy: for y_i ~ N(mu, sigma^2) iid with prior
    # mu ~ N(mu0, tau^2), the marginal likelihood integrates in closed form. The
    # derivation uses no part of APEDP, which is what makes it an oracle rather
    # than a second copy of the implementation.
    _CONJUGATE_Y = np.array([1.3, 0.7, 2.1, 1.9, 0.4, 1.1])
    _CONJUGATE_SIGMA = 2.0

    @staticmethod
    def _conjugate_log_evidence(y: np.ndarray, sigma: float, mu0: float, tau: float) -> float:
        """Closed-form ``log p(D)`` for a Gaussian likelihood and a Gaussian prior.

        ``log p(D) = -(n/2)ln(2 pi sigma^2) - S/(2 sigma^2)
        + (1/2)ln(2 pi sigma^2/n) - (1/2)ln(2 pi v) - (ybar - mu0)^2/(2v)``
        with ``v = sigma^2/n + tau^2`` and ``S = sum_i (y_i - ybar)^2``.
        """
        y = np.asarray(y, dtype=float)
        n = int(y.size)
        ybar = float(np.mean(y))
        s_total = float(np.sum((y - ybar) ** 2))
        v = sigma**2 / n + tau**2
        return float(
            -0.5 * n * np.log(2.0 * np.pi * sigma**2)
            - s_total / (2.0 * sigma**2)
            + 0.5 * np.log(2.0 * np.pi * sigma**2 / n)
            - 0.5 * np.log(2.0 * np.pi * v)
            - 0.5 * (ybar - mu0) ** 2 / v
        )

    @staticmethod
    def _gaussian_logpdf(centre: float, width: float) -> Callable[[np.ndarray], float]:
        def logpdf(mu: np.ndarray) -> float:
            return float(
                -0.5 * (float(mu[0]) - centre) ** 2 / width**2
                - 0.5 * np.log(2.0 * np.pi * width**2)
            )
        return logpdf

    def _expect_raises(
        self,
        name: str,
        call: Callable[[], object],
        exc: type[Exception],
        needle: str,
        oracle: str,
    ) -> None:
        """A refused computation is a result; an unexplained crash is a failure."""
        try:
            call()
        except exc as e:
            self.report.add(
                name, needle in str(e), f"raised {type(e).__name__}: {e}", oracle=oracle
            )
        except Exception as e:  # any other exception type is a failure, not a crash
            logger.exception("Unexpected exception type from '%s'", name)
            self.report.add(
                name,
                False,
                f"raised {type(e).__name__} instead of {exc.__name__}: {e}",
                oracle=oracle,
            )
        else:
            self.report.add(
                name, False, f"expected {exc.__name__}, nothing was raised", oracle=oracle
            )

    def validate_bayesian_selection(self) -> None:
        """Check the evidence machinery against a closed-form oracle it cannot share code with.

        The previous version of this check re-derived the implementation's own
        arithmetic (a ratio of two sums) and asserted it matched — a check that
        passes for any implementation consistent with itself. Here the expected
        log-evidence comes from Gaussian conjugacy, derived by hand, so a wrong
        integrator, a missing prior normalizer or a ratio-of-sums masquerading as
        a Bayes factor all fail.
        """
        y = self._CONJUGATE_Y
        sigma = self._CONJUGATE_SIGMA
        mu0, tau = 1.0, 0.5
        n = int(y.size)
        oracle = self._conjugate_log_evidence(y, sigma, mu0, tau)
        oracle_label = "Gaussian-Gaussian conjugacy, closed form derived by hand"

        def log_likelihood(mu: np.ndarray) -> float:
            return float(
                -0.5 * np.sum((y - float(mu[0])) ** 2) / sigma**2
                - 0.5 * n * np.log(2.0 * np.pi * sigma**2)
            )

        matched = self.bayes.marginal_likelihood(
            log_likelihood,
            self._gaussian_logpdf(mu0, tau),
            bounds=np.array([[mu0 - 9 * tau, mu0 + 9 * tau]]),
            grid_points=2001,
            prior_description=f"mu ~ N({mu0}, {tau}^2)",
        )
        self.report.add(
            "Bayesian: quadrature evidence equals the conjugate closed form",
            abs(matched.log_evidence - oracle) < 1e-9,
            f"log p(D)={matched.log_evidence:.12f}, closed form={oracle:.12f}, "
            f"delta={abs(matched.log_evidence - oracle):.2e}",
            oracle=oracle_label,
        )

        mode = (n * float(np.mean(y)) / sigma**2 + mu0 / tau**2) / (n / sigma**2 + 1 / tau**2)
        laplace = self.bayes.marginal_likelihood(
            log_likelihood,
            self._gaussian_logpdf(mu0, tau),
            map_params=np.array([mode]),
            prior_description=f"mu ~ N({mu0}, {tau}^2)",
        )
        self.report.add(
            "Bayesian: Laplace evidence equals the closed form for a Gaussian posterior",
            abs(laplace.log_evidence - oracle) < 1e-6,
            f"Laplace={laplace.log_evidence:.9f}, closed form={oracle:.9f} "
            f"(exact for this posterior, so any gap is approximation error)",
            oracle=oracle_label,
        )

        mismatch_centre = -1.5
        mismatch = self.bayes.marginal_likelihood(
            log_likelihood,
            self._gaussian_logpdf(mismatch_centre, tau),
            bounds=np.array([[mismatch_centre - 9 * tau, mismatch_centre + 9 * tau]]),
            grid_points=2001,
            prior_description=f"mu ~ N({mismatch_centre}, {tau}^2)",
        )
        oracle_bf = float(
            np.exp(oracle - self._conjugate_log_evidence(y, sigma, mismatch_centre, tau))
        )
        bf, verdict, log_bf = self.bayes.compute_bayes_factor(matched, mismatch)
        self.report.add(
            "Bayesian: Bayes factor equals the ratio of the closed-form evidences",
            abs(bf - oracle_bf) <= 1e-6 * oracle_bf,
            f"BF={bf:.9f}, closed form={oracle_bf:.9f}, log10BF={log_bf:.6f}",
            oracle=oracle_label,
        )
        self.report.add(
            "Bayesian: matching prior is preferred and the verdict names its method",
            bf > 10.0 and "Model 1" in verdict and "quadrature" in verdict,
            f"BF={bf:.4g}, verdict={verdict}",
            oracle="the data were generated under the Model 1 prior, not the Model 2 prior",
        )

        identical = np.array([float(np.exp(log_likelihood(np.array([mode]))))])
        ratio, description, _ = self.bayes.compute_likelihood_ratio(identical, identical)
        self.report.add(
            "Bayesian: a summed-likelihood ratio is not reported as a Bayes factor",
            abs(ratio - 1.0) < 1e-12 and "NOT a Bayes factor" in description and bf > 10.0,
            f"ratio={ratio:.12f} (same point for both models) while BF={bf:.4g}; "
            "the two statistics disagree because only one integrates over a prior",
            oracle="definition: equal inputs give ratio 1, a Bayes factor still discriminates",
        )

        contract = "contract: the refusal is required; there is no value to compare"
        self._expect_raises(
            "Bayesian: grid quadrature refuses >2 parameters",
            lambda: self.bayes.marginal_likelihood(
                lambda p: 0.0, lambda p: 0.0, bounds=np.array([[0.0, 1.0]] * 3)
            ),
            ValueError,
            "1-2 parameters",
            contract,
        )
        self._expect_raises(
            "Bayesian: negative likelihoods are rejected",
            lambda: self.bayes.compute_likelihood_ratio(np.array([-1.0]), np.array([0.5])),
            ValueError,
            "non-negative",
            contract,
        )
        self._expect_raises(
            "Bayesian: Laplace refuses a point that is not a mode",
            lambda: self.bayes.marginal_likelihood(
                lambda p: float(p[0]), self._gaussian_logpdf(0.0, 1.0), map_params=np.array([0.0])
            ),
            ValueError,
            "mode",
            contract,
        )
        self._expect_raises(
            "Bayesian: unmeasured evidence cannot form a Bayes factor",
            lambda: self.bayes.compute_bayes_factor(
                ModelEvidence(float("nan"), "quadrature", 1),
                ModelEvidence(0.0, "quadrature", 1),
            ),
            ValueError,
            "not finite",
            contract,
        )

    def validate_evolutionary_search(self) -> None:
        engine = DimensionalAssuranceEngine()
        engine.register_variable("x", "dimensionless")
        system = ClosedLoopAssuranceSystem(engine)
        sre = MultiVariableSRE(system, independent_vars=["x"])
        x = np.linspace(1.0, 5.0, 20)
        held_out = np.array([0.75, 2.25, 4.75, 6.0])
        for name, target, expected in [
            ("quadratic", x**2, held_out**2),
            ("affine", 2 * x + 1, 2 * held_out + 1),
        ]:
            expr, score, history = sre.run_discovery(
                {"x": x}, target, generations=20, pop_size=50, seed=42
            )
            if expr is None or not np.isfinite(score):
                self.report.add(f"Evolutionary: {name} held-out fit", False, "No finite candidate")
                continue
            numeric = expr.subs({sp.Symbol(k): v for k, v in system.constants_values.items()})
            predictions = np.array([float(numeric.subs(sp.Symbol("x"), v)) for v in held_out])
            r2 = 1.0 - float(np.sum((predictions - expected)**2) / np.sum((expected - expected.mean())**2))
            self.report.add(
                f"Evolutionary: {name} held-out fit",
                np.isfinite(r2) and r2 > 0.95,
                f"R²={r2:.6f}, penalized MSE={score:.6g}, expression={expr}",
                oracle="the generating law (x^2, 2x+1) evaluated at held-out points",
            )
            scores = [entry["best_score"] for entry in history]
            self.report.add(
                f"Evolutionary: {name} best score never worsens",
                bool(scores) and all(b <= a for a, b in pairwise(scores)),
                oracle="monotonicity invariant of the search loop (no numeric oracle)",
            )

    def validate_dimensional_analysis(self) -> None:
        """Test dimensional consistency checking."""
        # Test 1: F = ma is dimensionally correct
        m = self.engine.register_variable('m', 'kilogram')
        a = self.engine.register_variable('a', 'meter / second**2')
        F = m * a
        result = self.engine.check_expression_consistency(F)
        self.report.add(
            "Dimensional: F = ma is consistent",
            result.dimensionality == ureg.newton.dimensionality,
            f"Got {result.dimensionality}, expected {ureg.newton.dimensionality}",
            oracle="pint SI base-dimension definitions (external to APEDP)",
        )

        # Test 2: F + m should raise TypeError
        try:
            self.engine.check_expression_consistency(F + m)
            self.report.add("Dimensional: F + m raises TypeError", False, "Should have raised")
        except TypeError:
            self.report.add(
                "Dimensional: F + m raises TypeError",
                True,
                "Correctly rejected",
                oracle="pint refuses addition of inconsistent dimensions (external)",
            )

        # Test 3: E = mc^2 with SI units
        c = self.engine.register_variable('c_light', 'meter / second')
        E = m * c**2
        result2 = self.engine.check_expression_consistency(E)
        self.report.add(
            "Dimensional: E = mc^2 is consistent",
            result2.dimensionality == ureg.joule.dimensionality,
            f"Got {result2.dimensionality}, expected {ureg.joule.dimensionality}",
            oracle="pint SI base-dimension definitions (external to APEDP)",
        )

        # Test 4: Dimensionless check
        pi = self.engine.register_variable('pi', 'dimensionless')
        r = self.engine.register_variable('r', 'meter')
        circumference = 2 * pi * r
        result3 = self.engine.check_expression_consistency(circumference)
        self.report.add(
            "Dimensional: 2πr is consistent",
            result3.dimensionality == ureg.meter.dimensionality,
            f"Got {result3.dimensionality}",
            oracle="pint SI base-dimension definitions (external to APEDP)",
        )

    def validate_buckingham_pi(self) -> None:
        """Pendulum Pi groups: check the property (count, dimensionlessness), not the string.

        The oracle is the Pi theorem itself — ``n_vars - rank = 3 - 2 = 1``
        independent group — together with the SI dimension table, which decides
        whether the reported group really is dimensionless. Asserting the
        engine's own formatted string would only test that the formatter is
        stable.
        """
        dimensions = {"T": {"T": 1}, "L": {"L": 1}, "g": {"L": 1, "T": -2}}
        groups = BuckinghamPiEngine.compute_pi_groups(dimensions)
        self.report.add(
            "Buckingham-Pi: group count equals n_vars - rank",
            len(groups) == 1,
            f"count={len(groups)}, expected 3 variables - rank 2 = 1; groups={groups}",
            oracle="rank-nullity applied to the 2x3 dimension matrix (rank 2)",
        )

        checker = DimensionalAssuranceEngine()
        checker.register_variable("T", "second")
        checker.register_variable("L", "meter")
        checker.register_variable("g", "meter / second**2")
        dimensionless_flags = []
        for group in groups:
            rhs = group.split("=", 1)[-1].strip()
            try:
                expr = sp.sympify(rhs)
                result = checker.check_expression_consistency(expr)
                dimensionless_flags.append(
                    result.dimensionality == ureg.dimensionless.dimensionality
                )
            except (TypeError, ValueError, sp.SympifyError):
                dimensionless_flags.append(False)
        self.report.add(
            "Buckingham-Pi: every reported group is dimensionless under SI units",
            bool(dimensionless_flags) and all(dimensionless_flags),
            f"dimensionless flags={dimensionless_flags} for groups={groups}",
            oracle="SI base-dimension definitions (second, meter) via pint",
        )

    def validate_anomaly_repair(self) -> None:
        """Check repairs by their effect at the singularity, not by completing.

        The oracle is the pole of the published law: each repair is claimed to
        regularize a specific divergence, so the check evaluates the repaired
        model *at that point* and requires a finite value where the
        unregularized law diverges. "The audit ran without raising" is a
        presence test and would pass for a no-op.
        """
        oracle = "finiteness at the modelled singularity, where the published law has a pole"

        underlying = [
            ("Bayes factor", sp.Integer(1) / sp.Symbol("Z_j", positive=True), "Z_j", 0.0),
            ("Newtonian gravity", sp.Integer(1) / sp.Symbol("r", positive=True) ** 2, "r", 0.0),
            ("Coulomb field", sp.Integer(1) / sp.Symbol("r", positive=True) ** 2, "r", 0.0),
        ]
        for label, expr, symbol, value in underlying:
            try:
                diverges = not bool(np.isfinite(float(expr.subs(sp.Symbol(symbol), value))))
            except (TypeError, ValueError, ZeroDivisionError):
                diverges = True
            self.report.add(
                f"Anomaly: unregularized {label} diverges at the singular point",
                diverges,
                f"{expr} at {symbol}={value} is not finite",
                oracle="the published law's own pole (algebra, not the repair engine)",
            )

        repaired = [
            (
                "Bayes factor Z_i/Z_j at Z_j -> 0",
                self.repair.audit_and_repair_bayes_factor,
                {"Z_i": 1.0, "Z_j": 0.0, "beta_reg": 1e-12},
            ),
            (
                "Newtonian gravity 1/r^2 at r -> 0",
                self.repair.audit_and_repair_gravity,
                {"r_dist": 0.0, "epsilon_soft": 1.0, "m_phys": 1.0, "m_target": 1.0},
            ),
            (
                "Coulomb 1/r^2 at r -> 0",
                self.repair.audit_and_repair_coulomb,
                {
                    "q_source": 1.0,
                    "q_test": 1.0,
                    "lambda_debye": 1.0,
                    "r_dist": 0.0,
                    "epsilon_soft": 1.0,
                },
            ),
        ]
        for label, audit, inputs in repaired:
            try:
                model_name = audit()
                value = self.platform.evaluate_model(model_name, inputs)
            except (ArithmeticError, KeyError, TypeError, ValueError) as e:
                self.report.add(
                    f"Anomaly: repaired {label} is finite at the singularity",
                    False,
                    f"evaluation failed: {type(e).__name__}: {e}",
                    oracle=oracle,
                )
                continue
            self.report.add(
                f"Anomaly: repaired {label} is finite at the singularity",
                bool(np.isfinite(value)),
                f"{model_name} at the singular point evaluates to {value:g}",
                oracle=oracle,
            )

    def validate_epistemic_tracking(self) -> None:
        """Test knowledge graph and traceability."""
        # Test: Add nodes and edges
        self.graph.add_node("claim_energy", "Claim", {"text": "Energy is conserved", "confidence": 0.95})
        self.graph.add_node("claim_momentum", "Claim", {"text": "Momentum is conserved", "confidence": 0.98})
        self.graph.add_edge("claim_energy", "claim_momentum", "DERIVED_FROM")

        has_nodes = len(self.graph.graph.nodes) >= 2
        has_edges = len(self.graph.graph.edges) >= 1
        self.report.add(
            "Epistemic: Knowledge graph adds claims and edges",
            has_nodes and has_edges,
            f"Nodes={len(self.graph.graph.nodes)}, Edges={len(self.graph.graph.edges)}",
            oracle="structural round-trip invariant (no numeric oracle)",
        )

        # Test: Find contradictions and validate acyclicity
        acyclic = self.graph.validate_graph_acyclicity()
        self.report.add(
            "Epistemic: Graph is acyclic",
            acyclic is True,
            oracle="graph-theoretic property of the constructed DAG",
        )

        # Test: Lifecycle manager
        lifecycle = LifecycleManager()
        lifecycle.register_model("model_1")
        lifecycle.promote("model_1", "Candidate", checks_passed=True)
        lifecycle.promote("model_1", "Validated", checks_passed=True)
        state = lifecycle.get_state("model_1")
        self.report.add(
            "Epistemic: Lifecycle state transition",
            state is not None,
            f"State: {state}",
            oracle="declared lifecycle transition table",
        )

        # Test: Requirements traceability
        matrix = RequirementsTraceabilityMatrix()
        matrix.register_requirement(
            "REQ-001", "Validate model", "tester", "ADR-001", "model.py", "test_model", "run-001"
        )
        self.report.add(
            "Epistemic: Complete traceability accepted",
            matrix.verify_completeness() == (True, []),
            oracle="traceability completeness rule applied to a fully populated record",
        )
        matrix.register_requirement(
            "REQ-002", "Validate model", "tester", "ADR-002", "model.py", "test_model", ""
        )
        complete, missing = matrix.verify_completeness()
        self.report.add(
            "Epistemic: Missing evidence rejected",
            not complete and "REQ-002 missing validation_evidence" in missing,
            oracle="traceability completeness rule applied to a record with a blank field",
        )
        tournament = AdversarialDeliberationTournament(self.graph, lifecycle)
        result = tournament.run_tournament("trial", "1/x", "zero", True, "1/(x+epsilon)")
        self.report.add(
            "Epistemic: Tournament lifecycle bookkeeping (not mathematical proof)",
            result["outcome"] == "REPAIRED_AND_VALIDATED"
            and result["hypothesis_state"] == "Retired"
            and result["final_state"] == "Validated"
            and bool(self.graph.find_contradictions()),
            oracle="declared lifecycle states; the tournament proves no physics",
        )

    def validate_persistence(self) -> None:
        """Test model storage and serialization round-trips."""
        expr = sp.symbols('x')**2 + sp.symbols('y')
        with TemporaryDirectory() as tmp:
            store = ModelStore(str(tmp))
            model_id = store.save_model("test_model", expr, ["x", "y"], 0.99, 3)
            loaded = store.load_model(model_id)
            self.report.add(
                "Persistence: ModelStore save/load",
                loaded is not None and loaded["expression"] == str(expr),
                f"Loaded: {loaded['expression'] if loaded else 'None'}",
                oracle="round-trip invariant: the expression read back equals the one written",
            )
            models = store.list_models()
            self.report.add(
                "Persistence: List models",
                len(models) > 0,
                f"Count: {len(models)}",
                oracle="round-trip invariant (no numeric oracle)",
            )
            self.report.add(
                "Persistence: Delete model",
                store.delete_model(model_id) and not store.load_model(model_id),
                oracle="round-trip invariant: a deleted record must not load",
            )

        json_str = serialize_model(expr, ["x", "y"], {"r2": 0.99})
        restored = deserialize_model(json_str)
        self.report.add(
            "Persistence: Serialization round-trip",
            restored["expression"] == str(expr),
            f"Restored: {restored['expression']}",
            oracle="round-trip invariant: serialize then deserialize is the identity",
        )

        data = {"key": "value"}
        data_back = from_json(to_json(data))
        self.report.add(
            "Persistence: to_json/from_json",
            data == data_back,
            f"Round-trip: {data_back}",
            oracle="round-trip invariant: serialize then deserialize is the identity",
        )

    def validate_metrology(self) -> None:
        x, y = sp.symbols("x y")
        result = self.propagator.propagate_uncertainty(
            2*x + 3*y, {"x": 10.0, "y": 4.0}, {"x": 0.5, "y": 0.25}
        )
        self.report.add(
            "Metrology: Independent linear uncertainty",
            abs(result.evaluated_value - 32.0) < 1e-12
            and abs(result.combined_uncertainty - 1.25) < 1e-12,
            f"Value={result.evaluated_value}, uncertainty={result.combined_uncertainty}",
            oracle=(
                "hand-derived: 2*10 + 3*4 = 32, and "
                "sqrt((2*0.5)^2 + (3*0.25)^2) = 1.25 for independent inputs"
            ),
        )

    def validate_scenarios(self) -> None:
        """Run the full 22-scenario suite."""
        try:
            results = self.scenarios.run_all_scenarios()
            passed = sum(1 for r in results.values() if r['status'] == 'PASS')
            total = len(results)
            self.report.add(
                f"Scenarios: {passed}/{total} scenarios pass (self-referential)",
                passed == total,
                f"Pass rate: {passed/total*100:.1f}%; no oracle is declared: this check "
                "asserts the system agrees with itself, not that it is right",
            )
        except Exception as e:
            logger.exception("Validation check raised an exception")
            self.report.add("Scenarios: Full suite runs", False, str(e))

    def run_all(self) -> SelfValidationReport:
        """Run all self-validation groups, recording but surviving per-group errors."""
        groups = [
            self.validate_bayesian_selection,
            self.validate_evolutionary_search,
            self.validate_dimensional_analysis,
            self.validate_buckingham_pi,
            self.validate_anomaly_repair,
            self.validate_epistemic_tracking,
            self.validate_persistence,
            self.validate_metrology,
            self.validate_scenarios,
        ]
        for group in groups:
            try:
                group()
            except Exception as e:
                logger.exception("Validation group failed: %s", group.__name__)
                self.report.add(f"{group.__name__} (group crashed)", False, "", str(e))

        summary = self.report.summary()
        logger.info("=" * 60)
        logger.info(f"SUMMARY: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']})")
        logger.info(
            "Independence: %s checks name an independent oracle, %s are self-referential",
            summary["checks_with_independent_oracle"],
            summary["self_referential_checks"],
        )
        logger.info("=" * 60)

        self.report.to_json("self_validation_report.json")
        return self.report


def run_self_validation() -> SelfValidationReport:
    """Convenience function to run the full self-validation suite."""
    validator = SelfValidator()
    return validator.run_all()


def main() -> int:
    report = run_self_validation()
    summary = report.summary()
    for result in summary["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}: {result['test']} {result['details']} {result['error']}")
    print(
        f"SUMMARY: {summary['passed']}/{summary['total']} passed; "
        f"{summary['checks_with_independent_oracle']} checks name an independent oracle, "
        f"{summary['self_referential_checks']} are self-referential"
    )
    return int(summary["failed"] > 0 or summary["total"] == 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(main())