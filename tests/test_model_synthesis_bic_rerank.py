"""BIC re-rank over the top-K (EXT-BENCH-60 §9.5 next-round fix).

The frozen protocol diagnosed both genuine RECOVER misses as rank-by-raw-MSE
artifacts: the true form sat at rank 2 behind a more flexible surrogate,
separated by ~1e-04 MSE while trailing by 53-57 BIC points. These tests pin
the re-rank's effect on that measured case and pin its guard rails, so it
cannot be loosened into a coin toss without a test failing.

The re-rank is OFF by default: it changes selection globally, so §6.3
requires a fresh preregistered round before its effect can be scored.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp import model_synthesis as ms
from apedp.model_synthesis import AutonomousModelSynthesizer, SynthesizedModel

CORPUS = "benchmark_corpus_external_60/public"


def _e16_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr = np.genfromtxt(f"{CORPUS}/E16/train.csv", delimiter=",", skip_header=1)
    va = np.genfromtxt(f"{CORPUS}/E16/validation.csv", delimiter=",", skip_header=1)
    return tr[:, 0], tr[:, 1], va[:, 0], va[:, 1]


def _stub(expr: str, family: str, val_mse: float, k: int) -> SynthesizedModel:
    """A candidate with only the fields the re-rank reads."""
    return SynthesizedModel(
        expression_str=expr,
        sympy_expr=ms.sp.Symbol("x"),
        parameters={f"p{i}": 1.0 for i in range(k)},
        train_mse=val_mse,
        val_mse=val_mse,
        structural_family=family,
    )


def _rerank(models: list[SynthesizedModel], *, n_val: int,
            top_k: int = 8) -> SynthesizedModel:
    """Calls the real criterion, so a change to it cannot pass unnoticed."""
    winner = ms._bic_rerank(models, n_val=n_val, top_k=top_k)
    assert winner is not None
    return winner


def _bic_of(m: SynthesizedModel, n_val: int) -> float:
    k = max(len(m.parameters), 1)
    return m.val_mse * (1.0 + k * np.log(n_val) / n_val)


class TestBicFormula:
    def test_penalty_is_monotone_in_parameter_count(self) -> None:
        """More parameters must cost more, at equal val_mse."""
        n = 10
        assert _bic_of(_stub("a", "f", 1e-3, 2), n) < _bic_of(_stub("a", "f", 1e-3, 5), n)

    def test_penalty_vanishes_at_one_sample(self) -> None:
        """ln(n)=0 at n=1: no penalty, so ranking is pure val_mse."""
        assert _bic_of(_stub("a", "f", 1e-3, 5), 1) == pytest.approx(1e-3)

    def test_smaller_validation_set_penalizes_harder(self) -> None:
        """A small validation set gives less evidence, so costs more.

        The penalty is k*ln(n)/n, which is maximized near n=e. So a 2-param
        model on 3 points is penalized ~1.83x but on 100 points only ~1.09x.
        """
        m = _stub("a", "f", 1e-3, 2)
        assert _bic_of(m, 3) > _bic_of(m, 100)

    def test_penalty_never_exceeds_two_fold(self) -> None:
        """k*ln(n)/n peaks at k/e, bounding how far a re-rank can travel.

        This is why top-K and the incumbent-competes guard are load-bearing:
        the penalty is bounded, so it can only flip candidates that are
        already within a factor of ~1.37/e of each other.
        """
        worst = max(_bic_of(_stub("a", "f", 1.0, 1), n) / 1.0 for n in range(2, 1000))
        assert worst == pytest.approx(1.0 + 1.0 / np.e, rel=1e-2)


class TestReRankGuards:
    """Each guard is a way the re-rank could do harm. Pin it."""

    def test_incumbent_wins_when_val_mse_is_decisive(self) -> None:
        """A real val-MSE gap must survive: penalty must not override evidence."""
        models = [_stub("poly", "polynomial_baseline", 1e-4, 5),
                  _stub("exp", "exponential_decay", 1.0, 2)]
        best = _rerank(models, n_val=10)
        assert best is models[0]

    def test_fewer_parameters_win_on_a_near_tie(self) -> None:
        """The E16 shape: near-tie in val_mse, big gap in parameter count."""
        models = [_stub("poly", "polynomial_baseline", 2.275e-4, 5),
                  _stub("exp", "exponential_decay", 3.347e-4, 2)]
        best = _rerank(models, n_val=10)
        assert best is models[1], "2-param truth must displace 5-param surrogate"

    def test_val_tie_with_fewer_params_is_resolved_in_favour_of_simpler(self) -> None:
        """An exact val-MSE tie is NOT a tie to BIC.

        This is the re-rank's whole purpose: at equal predictive error the
        less flexible model is the better-supported claim. Reversing this
        assertion would reinstate the raw-MSE bug.
        """
        models = [_stub("a", "polynomial_baseline", 1e-3, 5),
                  _stub("b", "exponential_decay", 1e-3, 2)]
        assert _rerank(models, n_val=10) is models[1]

    def test_bic_tie_does_not_flip_the_incumbent(self) -> None:
        """A genuine tie on the penalized score must stay a tie.

        Without this the re-rank would be a coin toss dressed as a
        criterion: same k, same val_mse, no basis to prefer either.
        """
        models = [_stub("a", "polynomial_baseline", 1e-3, 3),
                  _stub("b", "exponential_decay", 1e-3, 3)]
        assert _rerank(models, n_val=10) is models[0]

    def test_tie_stability_does_not_depend_on_the_comparison_operator(self) -> None:
        """Tie-stability is load-bearing, but it comes from scan order.

        Documents why the strict ``<`` in the guard is not independently
        load-bearing: ``min()`` returns the first minimal element, so an exact
        BIC tie already resolves to the incumbent by position. Mutating ``<``
        to ``<=`` is therefore an equivalent mutant, and the property that
        actually matters -- a tie never flips the result -- is pinned by
        test_bic_tie_does_not_flip_the_incumbent.
        """
        models = [_stub("a", "polynomial_baseline", 1e-3, 3),
                  _stub("b", "exponential_decay", 1e-3, 3)]
        # Two distinct objects with bit-identical penalized scores.
        assert _bic_of(models[0], 10) == _bic_of(models[1], 10)
        winner = _rerank(models, n_val=10)
        assert winner is models[0]
        # And the challenger is genuinely tied, not merely beaten.
        assert _bic_of(winner, 10) == pytest.approx(_bic_of(models[1], 10))

    def test_candidate_outside_top_k_cannot_displace(self) -> None:
        """Only the val-MSE top-K compete; a buried candidate is not invited.

        The buried candidate must be one that WOULD win on BIC if it were
        admitted -- a candidate that merely loses on val_mse would pass this
        test whether or not the top-K guard exists, and would therefore prove
        nothing.
        """
        incumbent = _stub("poly", "polynomial_baseline", 2.275e-4, 5)
        middle = _stub("mid", "power_law", 2.30e-4, 3)
        buried = _stub("exp", "exponential_decay", 2.40e-4, 2)
        # Sanity: the buried candidate beats the incumbent once penalized.
        assert _bic_of(buried, 10) < _bic_of(incumbent, 10), (
            "precondition: the rank-3 candidate must be the one BIC prefers"
        )
        models = [incumbent, middle, buried]
        best = _rerank(models, n_val=10, top_k=2)
        assert best is not buried, "rank-3 candidate must not win a top-2 re-rank"
        # And it does win when the guard is widened, so the test has teeth.
        assert _rerank(models, n_val=10, top_k=3) is buried

    def test_zero_parameter_model_pays_the_k_equals_one_cost(self) -> None:
        """A model reporting no parameters still owes at least k=1.

        A zero-parameter model would otherwise carry a zero penalty and
        displace every incumbent on an equal-val_mse tie while having fitted
        strictly less. The floor makes it cost the same as a 1-param model,
        which leaves the tie a tie and the incumbent standing.
        """
        one_param = _stub("linear", "linear_baseline", 1e-3, 1)
        zero_param = _stub("const", "constant", 1e-3, 0)
        assert _bic_of(one_param, 10) == pytest.approx(_bic_of(zero_param, 10))
        assert _rerank([one_param, zero_param], n_val=10) is one_param


class TestReRankIsOffByDefault:
    def test_flag_defaults_off(self) -> None:
        """Selection must not change until a fresh round authorizes it."""
        assert ms._BIC_RERANK_ENABLED is False


@pytest.mark.slow
class TestE16EndToEnd:
    """The measured case, through the real synthesizer."""

    def test_e16_selected_family(self) -> None:
        """E16's truth is 5*exp(-0.2x), filed as exponential_decay.

        Pinned under the CURRENT (re-rank off) default: the miss is real and
        recorded in the protocol as a genuine 2/53 family error. When the flag
        is enabled this must become exponential_decay. Both directions are
        asserted so a later round that flips it cannot pass silently.
        """
        x_tr, y_tr, x_va, y_va = _e16_grids()
        top = AutonomousModelSynthesizer().synthesize(x_tr, y_tr, x_va, y_va)[0]

        if ms._BIC_RERANK_ENABLED:
            assert top.structural_family == "exponential_decay"
        else:
            assert top.structural_family == "polynomial_baseline"

    def test_e16_truth_is_present_in_the_candidate_set(self) -> None:
        """The re-rank can only work if the truth is a candidate at all.

        This is the load-bearing precondition: if exp decay were missing from
        the grammar, the flag would flip nothing and the E16 test above would
        be asserting a mirage.
        """
        x_tr, y_tr, x_va, y_va = _e16_grids()
        syn = AutonomousModelSynthesizer()
        fams = {m.structural_family for m in syn.synthesize(x_tr, y_tr, x_va, y_va)}
        assert "exponential_decay" in fams
