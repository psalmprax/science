"""Contract tests for the EXT-BENCH-60 competitor adapter.

The adapter is the boundary where APEDP's answers become benchmark output, and
it had no tests at all. That is how two contract violations survived: an
abstaining model shipped a numeric curve anyway, and nothing checked that the
adapter stays out of ``hidden/``.

Kept fast by stubbing the synthesizer rather than running synthesis; the point
is the output contract, not the search.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.external_60_adapters import APEDPAdapter, load_public_grids

CORPUS = Path(__file__).resolve().parent.parent / "benchmark_corpus_external_60"

pytestmark = pytest.mark.skipif(
    not (CORPUS / "public").is_dir(),
    reason="EXT-BENCH-60 corpus not present",
)


class _StubModel:
    def __init__(self, state: str, expr: str = "1.0*x + 0.0",
                 sympy_expr=None):
        import sympy as sp

        self.epistemic_state = _Enum(state)
        self.expression_str = expr
        self.sympy_expr = sympy_expr if sympy_expr is not None else sp.sympify(expr)
        self.structural_family = "linear_baseline"
        self.val_mse = 0.0
        # Present on the real SynthesizedModel; the adapter reads them.
        self.near_ood_mse = None
        self.far_ood_mse = None


class _Enum:
    """Minimal stand-in for EpistemicState; the adapter reads .value only."""

    def __init__(self, value: str):
        self.value = value


def _stubbed(monkeypatch, state: str, expr: str = "1.0*x + 0.0"):
    import apedp.model_synthesis as ms

    class _Stub:
        def synthesize(self, *_a, **_k):
            return [_StubModel(state, expr)]

    monkeypatch.setattr(ms, "AutonomousModelSynthesizer", _Stub)
    return APEDPAdapter().run(CORPUS, "P01")


def _capturing_stub(monkeypatch) -> dict:
    """Stubs the synthesizer and records how it was called."""
    import apedp.model_synthesis as ms

    seen: dict = {}

    class _Stub:
        def synthesize(self, *a, **k):
            seen["args"] = a
            seen["kwargs"] = k
            return [_StubModel("UNIQUE_SUPPORT")]

    monkeypatch.setattr(ms, "AutonomousModelSynthesizer", _Stub)
    APEDPAdapter().run(CORPUS, "P01")
    return seen


class TestNearOodIsActuallyForwarded:
    """The near-OOD grid is loaded and then, until now, discarded.

    synthesize() takes x_near_ood/y_near_ood. The adapter read the CSV at
    line 111 and never passed it, so near_ood_mse was None on all 60
    problems and every consumer of that field was running on a permanently
    absent input: the parsimony tiebreak's OOD guard could never fire, the
    Stage C near-OOD trade-off probe could never record one, and ood_tested
    was False for the whole round.
    """

    def test_near_ood_arrays_reach_the_synthesizer(self, monkeypatch):
        import numpy as np

        seen = _capturing_stub(monkeypatch)
        args, kwargs = seen["args"], seen["kwargs"]

        near_x = args[4] if len(args) > 4 else kwargs.get("x_near_ood")
        near_y = args[5] if len(args) > 5 else kwargs.get("y_near_ood")
        assert near_x is not None and near_y is not None, (
            "near-OOD was not forwarded to synthesize(); the OOD parsimony "
            "guard is dead code without it")

        grids = load_public_grids(CORPUS, "P01")["near_ood.csv"]
        assert np.array_equal(np.asarray(near_x), grids[:, 0])
        assert np.array_equal(np.asarray(near_y), grids[:, 1])

    def test_far_ood_is_withheld_from_selection(self, monkeypatch):
        """far-OOD MSE is block X's primary metric, and X is a re-split of P/E/R.

        Handing the selector the split it is scored on turns the headline
        number into a training target. Withheld on purpose, not overlooked.
        """
        seen = _capturing_stub(monkeypatch)
        assert len(seen["args"]) <= 6, (
            f"far-OOD reached synthesize(): {seen['args'][6:]}")
        assert "x_far_ood" not in seen["kwargs"]
        assert "y_far_ood" not in seen["kwargs"]

    def test_near_ood_scores_are_reported_in_metadata(self, monkeypatch):
        """The engine's own OOD score is kept beside the grid MSE.

        The blind evaluator recomputes OOD error from predictions against
        hidden truth. Carrying the engine's number too makes a divergence
        between the two visible instead of silent.
        """
        import apedp.model_synthesis as ms

        class _Stub:
            def synthesize(self, *_a, **_k):
                m = _StubModel("UNIQUE_SUPPORT")
                m.near_ood_mse = 0.25
                m.far_ood_mse = None
                return [m]

        monkeypatch.setattr(ms, "AutonomousModelSynthesizer", _Stub)
        res = APEDPAdapter().run(CORPUS, "P01")
        assert res.metadata["near_ood_mse"] == 0.25
        assert res.metadata["far_ood_mse"] is None, (
            "far_ood_mse is withheld from selection, so it must stay None -- "
            "a value here would mean far-OOD leaked in after all")


def test_abstaining_model_emits_no_predictions(monkeypatch):
    """A model that refuses must not also ship a confident numeric curve.

    Emitting both is two contradictory claims about one problem, and the number
    is what a scorer picks up. Measured on EXT-BENCH-60: abstained problems'
    predictions ran a median NRMSE of 1.26 against 0.005 for non-promoted ones
    -- all of it the output of models that had already declined to answer.

    The rule is deliberately about the abstention flag, not about accuracy:
    abstaining is not rewarded for happening to predict well.
    """
    for state in ("INSUFFICIENT_EVIDENCE", "NO_VALID_CANDIDATE"):
        res = _stubbed(monkeypatch, state)
        assert res.abstained is True, state
        assert res.predictions == {}, (
            f"{state} abstained yet still emitted "
            f"{sorted(res.predictions)}")


def test_multihypothesis_is_not_abstention_and_does_predict(monkeypatch):
    """MULTI_HYPOTHESIS promotes nothing but still answers the question.

    The adapter counts only INSUFFICIENT_EVIDENCE and NO_VALID_CANDIDATE as
    abstentions. That is the right contract, but it means this case must keep
    predicting -- it is the one where the system has a defensible answer it
    declines to make uniquely.
    """
    res = _stubbed(monkeypatch, "MULTI_HYPOTHESIS")
    assert res.abstained is False
    assert res.predictions, "MULTI_HYPOTHESIS should still carry a curve"


def test_unique_support_emits_finite_predictions(monkeypatch):
    res = _stubbed(monkeypatch, "UNIQUE_SUPPORT")
    assert res.abstained is False
    assert res.predictions
    for split, vals in res.predictions.items():
        assert len(vals) > 0, split


def test_result_is_json_serialisable(monkeypatch):
    """These records are the scored submission; they must round-trip."""
    res = _stubbed(monkeypatch, "MULTI_HYPOTHESIS")
    blob = json.dumps(res.to_dict())
    assert json.loads(blob)["problem_id"] == "P01"


def test_adapter_never_reads_hidden_ground_truth():
    """Structural guard: no string literal in the adapter may name hidden/.

    AST-based rather than a text scan. A text scan flags the module's own
    docstring ("Never touches hidden/ ground truth") and any comment that
    mentions the rule -- i.e. it fails precisely when someone documents the
    constraint properly. Walking the tree and collecting non-docstring string
    constants tests what the code actually does.
    """
    import ast

    path = (Path(__file__).resolve().parent.parent
            / "benchmarks" / "external_60_adapters.py")
    tree = ast.parse(path.read_text())

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)

    offenders = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "hidden" in node.value
        and node.value not in docstrings
    ]
    assert not offenders, f"adapter references hidden/ ground truth: {offenders}"


def test_public_grids_load_for_every_problem():
    """Every manifest problem must have the four public splits present."""
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    for p in manifest["problems"]:
        grids = load_public_grids(CORPUS, p["problem_id"])
        for split in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
            assert grids[split].ndim == 2 and grids[split].shape[1] == 2, (
                p["problem_id"], split)
