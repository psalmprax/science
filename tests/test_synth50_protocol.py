"""Pin SYNTH-50's data contract and selection policy.

The three SYNTH-50 class tests used to call ``synthesize`` directly with
hand-picked arguments, and got two of them wrong in ways nothing detected:

* they passed the far-OOD split in, which ``docs/PROTOCOL_SYNTHESIS_50.md`` §2
  forbids because far-OOD is the referee split and handing the selector the
  split it is scored on turns that number into a training target;
* they omitted ``allow_near_ood_promote``, so they ran under the engine default
  rather than the policy §5.4 ruled on.

Individually neither is dramatic -- supplying far-OOD was measured to change
only the ``ood_tested`` flag in the adjudication rationale, not the verdict.
Together they mean the corpus's six tests described a configuration that
existed nowhere else and was governed by no document, which is how
``test_class_s2_logarithmic_recovery`` and ``_NEAR_OOD_MAY_PROMOTE = False`` came
to contradict each other with nothing to adjudicate between them.

These tests assert the contract, so the next person to edit the class tests
finds out here rather than by reading the protocol.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from benchmarks.synth50_protocol import (
    SYNTH50_ALLOW_NEAR_OOD_PROMOTE,
    SYNTH50_SPLITS_PERMITTED,
    load_splits,
    synthesize,
)

TEST_FILE = Path(__file__).resolve().parent / "test_model_synthesis.py"


class TestTheDeclaredPolicy:
    def test_promotion_is_enabled_for_synth50(self) -> None:
        assert SYNTH50_ALLOW_NEAR_OOD_PROMOTE is True, (
            "section 5.4 returned PERMITTED; if this is flipping, the criterion "
            "was re-run and its result belongs in the protocol first")

    def test_far_ood_is_absent_from_the_permitted_splits(self) -> None:
        assert "far_ood.csv" not in SYNTH50_SPLITS_PERMITTED
        assert set(SYNTH50_SPLITS_PERMITTED) == {
            "train.csv", "validation.csv", "near_ood.csv"}


class TestTheLoaderDoesNotLoadIt:
    """The strong form: the grid is never read, so it cannot be forwarded."""

    def test_load_splits_returns_only_the_permitted_splits(self) -> None:
        _spec, splits = load_splits("S11")
        assert set(splits) == set(SYNTH50_SPLITS_PERMITTED)
        assert "far_ood.csv" not in splits

    def test_an_unknown_problem_raises_rather_than_returning_empty(self) -> None:
        """A missing corpus must not look like a problem with no splits."""
        with pytest.raises(FileNotFoundError):
            load_splits("S99")

    def test_the_returned_grids_are_usable_by_the_engine(self) -> None:
        """Both of these were wrong on the first attempt and failed downstream.

        Returning row tuples instead of transposed arrays, and Python lists
        instead of numpy arrays, each surfaced as an error inside
        ``synthesize`` / ``detect_regimes`` rather than at the loader -- so a
        reader would have gone looking for an engine bug.
        """
        _spec, splits = load_splits("S11")
        x_train, y_train = splits["train.csv"]
        assert isinstance(x_train, np.ndarray) and isinstance(y_train, np.ndarray)
        assert x_train.shape == y_train.shape and x_train.ndim == 1
        assert x_train.size > 10


class TestTheEntryPointForwardsThePolicy:
    def test_it_forwards_the_promotion_flag(self, monkeypatch) -> None:
        seen: dict = {}

        class _Stub:
            def synthesize(self, *a, **k):
                seen.update(k)
                seen["n_args"] = len(a)
                return ["models"]

        monkeypatch.setattr(
            "benchmarks.synth50_protocol.AutonomousModelSynthesizer", _Stub)
        synthesize("S11")
        assert seen.get("allow_near_ood_promote") is True, (
            "the protocol permits promotion but the entry point did not forward "
            "it, so SYNTH-50 ran under the engine default")

    def test_it_never_forwards_far_ood(self, monkeypatch) -> None:
        seen: dict = {}

        class _Stub:
            def synthesize(self, *a, **k):
                seen.update(k)
                seen["n_args"] = len(a)
                return ["models"]

        monkeypatch.setattr(
            "benchmarks.synth50_protocol.AutonomousModelSynthesizer", _Stub)
        synthesize("S11")
        # 6 positional args is train/val/near; a 7th and 8th would be far-OOD.
        assert seen["n_args"] <= 6, (
            f"far-OOD reached synthesize(): {seen['n_args']} positional args")
        assert "x_far_ood" not in seen and "y_far_ood" not in seen

    def test_use_near_ood_false_still_forwards_the_flag(self, monkeypatch) -> None:
        """The class-5 configuration must not silently drop the policy too."""
        seen: dict = {}

        class _Stub:
            def synthesize(self, *a, **k):
                seen.update(k)
                return ["models"]

        seen_positional: list = []

        class _Stub:
            def synthesize(self, *a, **k):
                seen.update(k)
                seen_positional.extend(a)
                return ["models"]

        monkeypatch.setattr(
            "benchmarks.synth50_protocol.AutonomousModelSynthesizer", _Stub)
        synthesize("S41", use_near_ood=False)
        assert seen.get("allow_near_ood_promote") is True
        # The near-OOD slots are still passed, as None. Asserting the *value*
        # rather than the arity is what was missing first time: the entry point
        # passes x_near_ood, y_near_ood positionally as None, which is correct,
        # and an arity check would have called that a leak.
        assert seen_positional[4] is None and seen_positional[5] is None, (
            "use_near_ood=False forwarded real near-OOD arrays")


class TestTheClassTestsDoNotBypassTheEntryPoint:
    """A guard on the tests themselves, by AST.

    Deliberately structural. The failure being prevented is someone adding
    ``x_far_ood, y_far_ood`` back to a ``synthesize`` call, and reading for that
    substring would fire on this module's own docstring explaining why not.

    Scoped to the SYNTH-50 class tests by function name. The first version
    scanned the whole file and flagged ``test_autonomous_synthesis_on_synth_01``,
    which is the single-problem SYNTH_01 corpus -- a different benchmark with
    no protocol and no far-OOD referee split. A guard that fires on unrelated
    legitimate code is a guard that gets deleted.
    """

    #: The SYNTH-50 class tests, by name.
    SYNTH50_TESTS = ("test_class_s2_logarithmic_recovery",
                     "test_class_s3_critical_scaling_recovery",
                     "test_class_s4_damped_oscillation_recovery")

    def _synth50_synthesize_calls(self) -> list[ast.Call]:
        tree = ast.parse(TEST_FILE.read_text(encoding="utf-8"))
        calls: list[ast.Call] = []
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            if fn.name not in self.SYNTH50_TESTS:
                continue
            calls.extend(
                node for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "synthesize"
            )
        return calls

    def test_the_class_tests_still_exist(self) -> None:
        """If this finds none, every guard below is vacuous and passing."""
        assert len(self._synth50_synthesize_calls()) == 0, (
            "the SYNTH-50 class tests now call synthesize() directly again; "
            "this module no longer guards anything")

    def test_no_synth50_test_forwards_far_ood(self) -> None:
        """The class tests must not be calling synthesize at all.

        They route through ``synth50_synthesize``, which withholds far-OOD by
        construction. Any direct ``.synthesize(...)`` inside one of them is
        a caller re-specifying the contract by hand, which is what this whole
        exercise exists to stop.
        """
        direct = self._synth50_synthesize_calls()
        offenders = []
        for call in direct:
            supplied = {kw.arg for kw in call.keywords if kw.arg}
            if len(call.args) > 6 or {"x_far_ood", "y_far_ood"} & supplied:
                offenders.append(ast.dump(call)[:120])
        assert not offenders, (
            f"{len(offenders)} SYNTH-50 test call(s) forward far-OOD directly: "
            f"{offenders}")

    def test_the_synth50_class_tests_go_through_the_protocol_entry_point(self) -> None:
        """S11, S21 and S31 must call synth50_synthesize, not synthesize directly."""
        source = TEST_FILE.read_text(encoding="utf-8")
        for pid in ("S11", "S21", "S31"):
            assert f'synth50_synthesize("{pid}' in source, (
                f"{pid} no longer routes through benchmarks.synth50_protocol; "
                "its far-OOD and promotion-flag handling is now hand-specified")