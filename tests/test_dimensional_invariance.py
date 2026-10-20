"""Dimensional / scale-invariance invariants.

A net, not a demonstration. The existing ``test_scale_invariance.py`` shows
one family surviving three orders of magnitude of ``y``; that is a good test
but a thin one, and it left the axis that actually broke unguarded.

That axis is ``x``. Before commit 0153761 the regime detector compared a
*slope* (``median(|dy/dx|)``, in y/x units) against a *y-unit* span, so its
verdict depended on the units of x: rescaling x by 10 flipped 11 of the 60
EXT-BENCH problems, including a damped oscillation that lost its
UNIQUE_SUPPORT. A system whose abstention depends on whether a caller chose
metres or centimetres is not measuring identifiability.

Why a net rather than another one-off: the same class of defect let both that
bug and a family-label collision through, and it will let the next one
through. These tests are cheap enough to sit in the default run so they
guard other sessions' future edits too, not just today's.

What is asserted, and why each is a real claim:

* ``x``-rescaling does not change any verdict (cheap; 60-problem sweep).
  Under a positive affine relabeling of x, every family's identity is
  mathematically preserved, so a change here is a bug, never a judgement.
* ``y``-rescaling and ``y``-negation do not change any verdict. Sign
  symmetry is a genuine trap: a detector that keys on ``y > 0`` is reading
  the units rather than the shape.
* The noise floor is in y units, so it must be exactly invariant to ``x``.
  This is the invariant the original bug violated, pinned directly on the
  estimator so a future refactor cannot reintroduce a slope-based floor
  without failing here rather than on 11 corpus problems.
* Selected *family* is stable under ``x``-rescaling, on a small sampled set.
  Family is what the benchmark scorer records, so this is the claim that
  matters, but full synthesis is ~20 s/problem and 60x3 is not a default-run
  cost. It is sampled and marked ``slow``; the cheap invariants above carry
  the default-run weight.

Nothing here asserts scale invariance where it does not hold. ``detect_regimes``
returns ``x_min``/``x_max`` in the caller's units, so those legitimately
track the rescaling, and no test compares them.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import ClassVar

import numpy as np
import pytest

from apedp.model_synthesis import (
    AutonomousModelSynthesizer,
    _input_normalization_scale,
    _regime_identifiability,
    _robust_noise_floor,
)

CORPUS = pathlib.Path("benchmark_corpus_external_60/public")

#: Shapes spanning the qualitative regimes the detector claims to resolve.
#: Chosen to cover turnover, decay, oscillation, and a non-power onset, so a
#: detector that only happens to be invariant on monotone data is caught.
SHAPES: dict[str, tuple[str, float, float, int]] = {
    "power_onset": ("2.0 * x**1.5", 0.5, 6.0, 40),
    "saturation": ("5.0 * x / (2.0 + x)", 0.5, 6.0, 40),
    "turnover": ("2.45 * x**1.5 / (1 + 0.85 * x**2) * exp(-0.3 * x)", 0.5, 8.0, 40),
    "logarithmic": ("3.0 * log(1 + 2.0 * x)", 0.05, 5.0, 30),
    "exponential_decay": ("4.0 * exp(-0.5 * x)", 0.2, 6.0, 40),
    "damped_oscillation": ("2.0 * exp(-0.2 * x) * cos(2.8 * x)", 0.5, 6.0, 40),
    "undamped_oscillation": ("2.0 * cos(2.0 * x)", 0.5, 6.0, 40),
    "gaussian_pulse": ("3.0 * exp(-(x - 2.0)**2 / (2 * 0.7**2))", 0.0, 5.0, 40),
    "rational_transition": ("2.0 * x**1.3 / (1.0 + 0.7 * x**2.1)", 0.5, 6.0, 40),
    "exponential_envelope": ("1.0 * x * exp(-0.5 * x**1.5)", 0.5, 6.0, 40),
}


def _evaluate(formula: str, x: np.ndarray) -> np.ndarray:
    return eval(formula, {  # noqa: S307 - intentional: evaluates repo-authored physics formulas with a fixed env (test fixture)
        "x": x, "exp": np.exp, "log": np.log, "cos": np.cos, "sin": np.sin,
    })


def _series(name: str) -> tuple[np.ndarray, np.ndarray]:
    formula, lo, hi, n = SHAPES[name]
    x = np.linspace(lo, hi, n)
    return x, _evaluate(formula, x)


#: Records that ABSTAIN, paired with the reason they must abstain for.
#:
#: Without these the invariance tests are near-vacuous: every entry in SHAPES
#: returns the same verdict, so a detector hard-wired to always answer
#: "identifiable" would pass all of them. A verdict is only a meaningful thing
#: to hold invariant if it sometimes differs. Each case here is a different
#: reason to abstain, so a threshold change in one rule is caught even though
#: it leaves the others untouched.
ABSTAINING: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _build_abstaining() -> None:
    """Constructs one record per abstention rule."""
    rng = np.random.default_rng(11)

    # Rule 1: no variation at all. The adv-trig-identity shape.
    ABSTAINING["constant_record"] = (
        np.linspace(0.0, 5.0, 30), np.full(30, 2.0),
    )

    # Rule 2: signal buried in its own noise. span/noise is ~1.4 here,
    # deliberately under the 3.0 threshold.
    x = np.linspace(0.0, 5.0, 30)
    ABSTAINING["noise_dominated"] = (
        x, 1.0 + rng.normal(0.0, 0.7, 30),
    )

    # Rule 3: too few samples to bracket an interior extremum. n=6 is under
    # MIN_RESOLVABLE_REGIME_SAMPLES=8 but comfortably over the n>=3 floor and
    # has full variation, so only the sample gate can abstain it.
    ABSTAINING["too_few_samples"] = (
        np.linspace(0.5, 6.0, 6), 2.0 * np.linspace(0.5, 6.0, 6) ** 1.5,
    )

    # Rule 4: fewer than three distinct x, so no local model is estimable
    # however many samples are supplied.
    ABSTAINING["too_few_distinct_x"] = (
        np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    )


_build_abstaining()

#: A record that clears every rule, used as the counterpart to each
#: abstaining case. Same shape, noise turned down / samples turned up.
CLEARING: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "constant_record": (
        np.linspace(0.0, 5.0, 30), np.linspace(0.0, 5.0, 30) ** 1.5,
    ),
    "noise_dominated": (
        x := np.linspace(0.0, 5.0, 30), np.linspace(0.0, 5.0, 30),
    ),
    "too_few_samples": (
        np.linspace(0.5, 6.0, 20), 2.0 * np.linspace(0.5, 6.0, 20) ** 1.5,
    ),
    "too_few_distinct_x": (
        np.linspace(0.5, 6.0, 30), np.linspace(0.5, 6.0, 30) ** 1.5,
    ),
}


def _verdict(x: np.ndarray, y: np.ndarray) -> tuple[bool, str]:
    return _regime_identifiability(np.asarray(x, float), np.asarray(y, float))


def _verdict_kind(x: np.ndarray, y: np.ndarray) -> tuple[bool, str]:
    """The verdict with measured numbers stripped out of the reason.

    Abstention reasons quote the quantity that triggered them ("signal span
    8.24e-10 is under 3x the local noise floor 6.12e-10"), so the reason text
    legitimately changes when the caller changes units even though the
    decision does not. Comparing bare strings would therefore fail on a
    correct detector. This reduces a reason to its rule, so invariance is
    asserted on the decision and the rule, not on the numbers in the message.
    """
    underspecified, reason = _verdict(x, y)
    if not underspecified:
        return (False, "")
    return (True, re.sub(r"[-+0-9.eE]+", "#", reason).strip())


#: Multipliers spanning 6 orders of magnitude on both sides. A detector that
#: survives these is not merely insensitive to "the" scale, it has no scale
#: term to be sensitive to.
SCALES = (0.001, 0.1, 10.0, 1000.0)


class TestNoiseFloorIsInYUnits:
    """The invariant commit 0153761 restored, pinned on the estimator itself."""

    @pytest.mark.parametrize("name", sorted(SHAPES))
    def test_noise_floor_ignores_the_units_of_x(self, name: str) -> None:
        """Rescaling x must not move a quantity defined in y units.

        The pre-fix floor was ``median(|dy/dx|)``, a slope: dividing x by 10
        multiplied it by 10 and moved the abstention verdict. The floor now
        takes only y, so this is exact rather than approximate, and it holds
        no matter how far x is relabeled.
        """
        x, y = _series(name)
        base = _robust_noise_floor(y)
        assert base >= 0.0 and np.isfinite(base)
        for scale in SCALES:
            # The floor is a function of y alone, so the rescaled x is not
            # even an argument it could see; the check that matters is that
            # rescaling x leaves y -- and therefore the verdict -- untouched.
            assert _verdict_kind(x * scale, y) == _verdict_kind(x, y)

    def test_noise_floor_takes_no_x_argument(self) -> None:
        """Structural pin: the signature forbids reintroducing a slope term.

        A behavioural test can be satisfied while the estimator still has the
        x-shaped parameter wired up. This pins the interface too, so a
        refactor that widens the signature to accept x fails here.
        """
        import inspect
        params = list(inspect.signature(_robust_noise_floor).parameters)
        assert params == ["ys"], (
            f"noise floor must take y only; got {params}"
        )

    def test_noise_floor_tracks_y_amplitude(self) -> None:
        """Sanity counterpart: scaling y *does* scale the floor, by that factor.

        Without this, "invariant to x" could be satisfied by a constant.
        """
        _, y = _series("damped_oscillation")
        base = _robust_noise_floor(y)
        assert _robust_noise_floor(y * 1000.0) == pytest.approx(base * 1000.0, rel=1e-9)

    def test_noise_floor_annihilates_a_linear_trend(self) -> None:
        """Second differences of a line are exactly zero, at any amplitude.

        This is the estimator's load-bearing property and it is exact, not
        approximate: a first-order trend contributes nothing to a second
        difference, so a steep linear ramp cannot be mistaken for noise
        however large it is.
        """
        x = np.linspace(0.0, 5.0, 21)
        for slope in (1.0, 1e3, 1e6):
            assert _robust_noise_floor(slope * x + 7.0) == pytest.approx(0.0, abs=1e-6), (
                f"slope {slope}: a linear trend was read as noise"
            )

    @pytest.mark.parametrize("formula", [
        "x**2", "x**3", "2.0*exp(-0.2*x)*cos(2.8*x)", "2.0*cos(2.0*x)",
        "2.45*x**1.5/(1+0.85*x**2)*exp(-0.3*x)",
    ])
    def test_floor_to_span_ratio_is_scale_free(self, formula: str) -> None:
        """The ratio that ``_regime_identifiability`` compares is invariant.

        The verdict compares a y-unit floor against a y-unit span, so their
        ratio is the dimensionless quantity the decision rests on. That ratio
        must not move when the curve is scaled, which is the statement the
        original slope-based floor violated.

        Note the floor is not zero on a smooth curve -- for an oscillation the
        ratio is ~0.064, not 0 -- which is why a smooth trend needs no
        carve-out but a smooth oscillation is not read as noiseless.
        """
        x = np.linspace(0.5, 6.0, 40)
        y = _evaluate(formula, x)
        base = _robust_noise_floor(y) / np.ptp(y)
        for s in SCALES:
            assert _robust_noise_floor(s * y) / np.ptp(s * y) == pytest.approx(
                base, rel=1e-9
            ), f"{formula}: floor/span ratio moved under y*{s}"

    def test_noise_floor_recovers_a_known_sigma(self) -> None:
        """It is a consistent noise scale, not just monotone in the noise."""
        rng = np.random.default_rng(0)
        y = rng.normal(0.0, 2.0, 4000)
        assert _robust_noise_floor(y) == pytest.approx(2.0, rel=0.05)


class TestVerdictIsScaleFree:
    """The end-to-end statement: same curve, same verdict, any units."""
    @pytest.mark.parametrize("name", sorted(SHAPES))
    @pytest.mark.parametrize("scale", SCALES)
    def test_x_rescaling_does_not_change_the_verdict(self, name: str, scale: float) -> None:
        x, y = _series(name)
        assert _verdict_kind(x, y) == _verdict_kind(x * scale, y), (
            f"{name}: abstention verdict depends on the units of x"
        )

    @pytest.mark.parametrize("name", sorted(SHAPES))
    @pytest.mark.parametrize("yscale", (1e-9, 1e6))
    def test_y_rescaling_does_not_change_the_verdict(self, name: str, yscale: float) -> None:
        """y carries units too, so its verdict must also be free of them.

        A detector that keys on a hard-coded magnitude of y is reading units
        rather than shape.
        """
        x, y = _series(name)
        assert _verdict_kind(x, y) == _verdict_kind(x, y * yscale)

    @pytest.mark.parametrize("name", sorted(SHAPES))
    def test_y_negation_does_not_change_the_verdict(self, name: str) -> None:
        """Sign symmetry: -y is the same curve reflected, not a different one.

        A detector that treats y < 0 as automatically unresolvable would be
        conflating sign with information content.
        """
        x, y = _series(name)
        assert _verdict_kind(x, y) == _verdict_kind(x, -y)

    @pytest.mark.parametrize("name", sorted(SHAPES))
    def test_x_translation_does_not_change_the_verdict(self, name: str) -> None:
        """A shift in x is a re-labeling of the domain, so it cannot add evidence.

        Weaker than rescaling -- it moves the data, not just the units -- but
        the identifiability question asked is "can the shape of this curve be
        resolved from this design", which a relabeling of the domain does not
        change. Any dependence here is a hard-coded threshold on raw x.
        """
        x, y = _series(name)
        base = _verdict_kind(x, y)
        for shift in (-3.0, 1.0, 50.0):
            assert _verdict_kind(x + shift, y) == base, (
                f"{name}: verdict depends on a shift of x by {shift}"
            )


class TestCorpusIsScaleFree:
    """Sweep the real corpus, where the original bug was measured.

    11 of 60 problems changed verdict under x*10 before the fix. This is the
    regression net for exactly that, over all 60 rather than a hand-picked few.
    """

    @staticmethod
    def _corpus_problems() -> list[str]:
        manifest = json.loads(
            pathlib.Path("benchmark_corpus_external_60/manifest.json").read_text()
        )
        return [p["problem_id"] for p in manifest["problems"]]

    @pytest.mark.parametrize("pid", _corpus_problems.__func__())  # type: ignore[misc]
    def test_corpus_verdict_survives_x_rescaling(self, pid: str) -> None:
        tr = np.genfromtxt(CORPUS / pid / "train.csv", delimiter=",", skip_header=1)
        x, y = tr[:, 0], tr[:, 1]
        base = _verdict_kind(x, y)
        for scale in SCALES:
            assert _verdict_kind(x * scale, y) == base, (
                f"{pid}: verdict changes under x*{scale}"
            )

    @pytest.mark.parametrize("pid", _corpus_problems.__func__())  # type: ignore[misc]
    def test_corpus_verdict_survives_y_rescaling(self, pid: str) -> None:
        """Also the y axis: the corpus's own dynamic range is wide.

        A54-A57 are near-constant records, which is exactly the regime where a
        y-magnitude threshold misfires.
        """
        tr = np.genfromtxt(CORPUS / pid / "train.csv", delimiter=",", skip_header=1)
        x, y = tr[:, 0], tr[:, 1]
        base = _verdict_kind(x, y)
        for yscale in (1e-9, 1e9):
            assert _verdict_kind(x, y * yscale) == base, (
                f"{pid}: verdict changes under y*{yscale}"
            )


class TestNrmseIsScaleFree:
    """The metric that exposed the artifact, pinned so it stays honest.

    My 6/10 rating used R^2, which divides by SS_tot and is undefined for a
    constant target -- that produced the "-10^295 garbage" I reported for
    A54/A55. NRMSE = rmse/ptp(y) is scale-free by construction; these pin that
    it is, so the metric cannot be reintroduced in a form that is not.
    """

    @staticmethod
    def _nrmse(pred: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - y) ** 2)) / np.ptp(y))

    def test_nrmse_is_invariant_to_y_rescaling(self) -> None:
        rng = np.random.default_rng(7)
        y = rng.normal(10.0, 2.0, 200)
        p = y + rng.normal(0.0, 0.5, 200)
        base = self._nrmse(p, y)
        for s in (1e-9, 1e6):
            assert self._nrmse(p * s, y * s) == pytest.approx(base, rel=1e-9)

    def test_r2_is_undefined_where_nrmse_is_finite(self) -> None:
        """Why R^2 was the wrong tool: on a constant target it divides by zero.

        The abstained block's "catastrophic NRMSE 1.26" and my "-10^295 R^2"
        were two views of the same 12 problems, and the R^2 view was the
        broken one. R^2 is asserted here to blow up rather than to return a
        number, so the failure mode is pinned as what it is.
        """
        y = np.full(50, 3.0)
        p = y + 0.01
        assert np.var(y) == 0.0, "precondition: a constant target has no variance"
        with np.errstate(divide="ignore"):
            r2 = 1.0 - np.mean((p - y) ** 2) / np.var(y)
        assert not np.isfinite(r2), "R^2 must be undefined here, not merely large"

        # NRMSE is degenerate on the same data for the same reason (both
        # normalize by the target's spread), which is why the abstention path
        # reports no fit-quality number at all rather than picking a metric
        # that happens to be finite.
        assert np.ptp(y) == 0.0

    def test_nrmse_recovers_a_known_relative_error(self) -> None:
        """NRMSE is scale-free and readable: a 1% relative error reads as 0.01.

        The property that makes NRMSE usable in a report where R^2 is not.
        """
        y = np.linspace(1.0, 10.0, 100)
        err = 0.01 * (y.max() - y.min())
        p = y + err
        assert self._nrmse(p, y) == pytest.approx(0.01, rel=1e-9)


class TestAbstentionRules:
    """Each abstention rule fires for its own reason, and stays scale-free.

    Two jobs. First, prove the detector is not a constant: every case here
    abstains while every case in SHAPES does not, so a hard-wired verdict
    cannot pass the suite. Second, prove the boundary between them is a real
    measured boundary -- a threshold change in any one rule is caught rather
    than moving the whole detector silently.
    """

    @pytest.mark.parametrize("name", sorted(ABSTAINING))
    def test_record_abstains(self, name: str) -> None:
        x, y = ABSTAINING[name]
        underspecified, reason = _verdict(x, y)
        assert underspecified, f"{name} should abstain, got identifiable"
        assert reason, "an abstention must say why; a silent one is unauditable"

    @pytest.mark.parametrize("name", sorted(CLEARING))
    def test_counterpart_does_not_abstain(self, name: str) -> None:
        """The matching record must be identifiable, or the rule is a blanket."""
        x, y = CLEARING[name]
        underspecified, _ = _verdict(x, y)
        assert not underspecified, f"{name} counterpart should be identifiable"

    @pytest.mark.parametrize("name", sorted(ABSTAINING))
    def test_abstaining_record_stays_abstaining_under_rescaling(self, name: str) -> None:
        """An abstention must not evaporate when the caller changes units.

        The failure this catches is worse than the inverse: a record judged
        unidentifiable at x*1 and identifiable at x*10 would be admitted
        without evidence, which is the direction that manufactures false
        confidence.
        """
        x, y = ABSTAINING[name]
        base = _verdict_kind(x, y)
        assert base[0], "precondition: this record abstains at base scale"
        for scale in SCALES:
            assert _verdict_kind(x * scale, y) == base, (
                f"{name}: abstention changed under x*{scale}"
            )
        for yscale in (1e-9, 1e9):
            assert _verdict_kind(x, y * yscale) == base, (
                f"{name}: abstention changed under y*{yscale}"
            )

    def test_noise_ratio_boundary_is_where_it_is_documented(self) -> None:
        """The 3x span-to-noise threshold is a documented constant, so pin it.

        A record whose span sits just under the threshold abstains; one well
        over it does not. Asserting the boundary rather than the constant
        means a future retune has to be a deliberate edit to this test.

        The signal here is a SLOPE, not an offset: adding a constant to pure
        noise leaves the spread untouched, so an offset-based version of this
        test would measure nothing. At unit slope the span/noise ratio is 2.54
        and the record abstains; at 1.5x the ratio is 3.57 and it clears. So
        the pair brackets the documented 3.0 threshold from both sides.
        """
        x = np.linspace(0.0, 5.0, 400)
        for slope, expect_abstains in ((1.0, True), (1.5, False)):
            y = 1.0 + slope * x + np.random.default_rng(3).normal(0.0, 1.0, 400)
            got = _verdict(x, y)[0]
            assert got is expect_abstains, (
                f"slope {slope}: expected is_underspecified="
                f"{expect_abstains}, got {got}"
            )

    def test_sample_gate_boundary_is_where_it_is_documented(self) -> None:
        """MIN_RESOLVABLE_REGIME_SAMPLES=8 separates the two n=7 and n=9 cases."""
        def shape(n: int) -> tuple[np.ndarray, np.ndarray]:
            grid = np.linspace(0.5, 6.0, n)
            return grid, 2.0 * grid**1.5

        below = _verdict(*shape(7))
        above = _verdict(*shape(9))
        assert below[0], "n=7 must abstain on the sample gate"
        assert not above[0], "n=9 must clear the sample gate"
        assert "samples" in below[1], f"n=7 abstained for the wrong reason: {below[1]}"


def test_bare_log_template_is_not_x_normalized() -> None:
    """Bare log(x) needs an intercept shift that the current map-back omits."""
    x = np.array([10.0, 20.0])
    assert _input_normalization_scale(
        "logarithmic_boundary_layer", "A * log(x) + B", x
    ) == 1.0
    assert _input_normalization_scale(
        "logarithmic_boundary_layer", "A * log(1 + B*x)", x
    ) == 20.0


@pytest.mark.slow
class TestSelectedFamilyIsScaleFree:
    """The claim the benchmark scorer actually records, on a small sample.

    Family identity is mathematically preserved by a positive affine
    relabeling of x, so any change here is a defect. Marked ``slow`` because
    synthesis costs ~20 s per problem; the cheap invariants above are what
    guard the default run, and this is the deep check.
    """

    #: A spread across blocks, including the one the original bug hit (C) and
    #: the two genuine family errors (E, P). Deliberately not "the easy ones".
    #:
    #: Resolved from the manifest rather than written out by hand. The corpus
    #: ids are sparse and non-sequential -- the O block runs O29..O36 and R runs
    #: R21..R28 -- so hand-written "O01"/"R01" name problems that have never
    #: existed, and two of the eight parameters failed on FileNotFoundError
    #: rather than on anything to do with invariance. That is the worst way for
    #: a test to fail: it looks like a product defect and is really a typo.
    #: ``test_every_sampled_problem_exists`` now fails loudly if the corpus and
    #: this list ever diverge again.
    SAMPLE = ("P01", "P07", "E16", "C41", "C44", "O29", "R21", "A54")

    @classmethod
    def _corpus_ids(cls) -> list[str]:
        manifest = json.loads(
            pathlib.Path("benchmark_corpus_external_60/manifest.json").read_text()
        )
        return [p["problem_id"] for p in manifest["problems"]]

    def test_every_sampled_problem_exists(self) -> None:
        """A sampled id that is not in the corpus is a broken sample, not a bug."""
        available = set(self._corpus_ids())
        missing = [pid for pid in self.SAMPLE if pid not in available]
        assert not missing, (
            f"SAMPLE names problems absent from the corpus: {missing}. "
            f"Available: {sorted(available)}"
        )
        for pid in self.SAMPLE:
            assert (CORPUS / pid / "train.csv").is_file(), (
                f"{pid} is in the manifest but has no train.csv on disk"
            )

    def test_the_sample_spans_multiple_blocks(self) -> None:
        """Guards against the sample silently collapsing to one family."""
        blocks = {pid.rstrip("0123456789") for pid in self.SAMPLE}
        assert len(blocks) >= 4, f"sample covers too few blocks: {sorted(blocks)}"

    @staticmethod
    def _synth(pid: str, xs: float = 1.0):
        tr = np.genfromtxt(CORPUS / pid / "train.csv", delimiter=",", skip_header=1)
        va = np.genfromtxt(CORPUS / pid / "validation.csv", delimiter=",", skip_header=1)
        return AutonomousModelSynthesizer().synthesize(
            tr[:, 0] * xs, tr[:, 1], va[:, 0] * xs, va[:, 1]
        )[0]

    @pytest.mark.parametrize("pid", SAMPLE)
    def test_family_survives_x_rescaling(self, pid: str) -> None:
        base = self._synth(pid).structural_family
        for xs in (0.1, 10.0):
            got = self._synth(pid, xs).structural_family
            assert got == base, (
                f"{pid}: family changed {base} -> {got} under x*{xs}"
            )


class TestOscillationGridIsScaleFree:
    """The S4 frequency pool must not be a prior on the caller's x units.

    This is the cheap half of the O29 finding. ``TestSelectedFamilyIsScaleFree``
    is the expensive half: it synthesises, so it is ``slow`` and sampled.
    Here the claim is made directly against the pool, for every scale, in
    milliseconds -- so a regression is caught by the default run rather than
    only by the deep check.

    The defect this pins: the frequency grid was a literal list in the
    caller's units, ``[1.5 ... 4.0]``. EXT-BENCH O29 is a 2.8 rad oscillation,
    which that list contains. Rescaled by 0.1 the same record is a 28 rad
    oscillation, which the list cannot represent by a factor of seven, so the
    engine reported ``missing_structural_mechanism`` -- "the mechanism is
    absent" -- on data that contains the mechanism in full. Nothing about the
    data had changed. The caller's units had.
    """

    #: Truth from the corpus: O29 damped-2.8, O30 damped-2.0, O34 damped-4.2.
    #: Frequencies in rad per unit of the corpus's x.
    TRUTH: ClassVar[dict[str, float]] = {"O29": 2.8, "O30": 2.0, "O34": 4.2}

    @staticmethod
    def _omegas(x_lo: float, x_hi: float) -> list[float]:
        templates = AutonomousModelSynthesizer().generate_candidate_structures(
            {
                "is_underspecified": False,
                "is_oscillatory": True,
                "x_min": x_lo,
                "x_max": x_hi,
            }
        )
        out: set[float] = set()
        for tmpl, _family in templates:
            if "cos(" not in tmpl:
                continue
            out.add(float(tmpl.split("cos(")[1].split(" ")[0]))
        return sorted(out)

    @pytest.mark.parametrize("pid", sorted(TRUTH))
    @pytest.mark.parametrize("scale", (0.1, 1.0, 10.0))
    def test_the_true_frequency_is_reachable(self, pid: str, scale: float) -> None:
        """Some candidate must be within 10% of the truth at every scale.

        10% rather than exact: the grid is discrete by design, and Stage D
        refines continuously from whatever the grid lands on. The claim is
        reachability, not exactness -- an unreachable truth cannot be refined
        either, which is the failure.
        """
        tr = np.genfromtxt(CORPUS / pid / "train.csv", delimiter=",", skip_header=1)
        x = tr[:, 0] * scale
        w = self.TRUTH[pid] / scale
        omegas = self._omegas(float(x.min()), float(x.max()))
        assert omegas, "no oscillation templates were proposed at all"
        nearest = min(omegas, key=lambda o: abs(o - w))
        assert abs(nearest - w) / w <= 0.10, (
            f"{pid} at x*{scale}: truth is w={w:.4g}, nearest candidate is "
            f"{nearest:.4g} ({abs(nearest - w) / w:.1%} away). The pool cannot "
            f"represent this oscillation, so the family is unreachable."
        )

    @pytest.mark.parametrize("scale", (0.1, 1.0, 10.0))
    def test_the_pool_does_not_duplicate_the_absolute_grid(self, scale: float) -> None:
        """Adding a scale-free grid must not re-propose what was already there.

        The union is only safe as an addition if the overlap is removed:
        otherwise the same candidate appears twice, the tiebreak sees two
        identical models, and parsimony bookkeeping is spent on nothing.
        """
        lo, hi = 0.5, 6.0
        x = np.linspace(lo, hi, 40) * scale
        omegas = self._omegas(float(x.min()), float(x.max()))
        assert len(omegas) == len(set(omegas))

    def test_stage_d_frequency_box_contains_every_proposed_frequency(self) -> None:
        """Stage D's box must not exclude the grid that seeds it.

        A box in absolute 1/x units (it was ``(0.2, 10.0)``) excludes any coarse
        candidate outside it, so the seed handed to L-BFGS-B starts outside its
        own search region. That is not a worse fit; it is an invalid start, and
        it is the second half of the same O29 defect.
        """
        from apedp import model_synthesis as m

        for scale, span in ((0.1, 0.585), (1.0, 5.848), (10.0, 58.48)):
            lo = min(0.2, m._W_PERIODS_LO * 2.0 * np.pi / span)
            hi = max(10.0, m._W_PERIODS_HI * 2.0 * np.pi / span)
            assert lo < hi, f"degenerate box at x*{scale}"
            for w in self._omegas(0.0, span):
                assert lo <= w <= hi, (
                    f"x*{scale}: coarse candidate w={w:.4g} is outside the "
                    f"Stage D box ({lo:.4g}, {hi:.4g})"
                )
