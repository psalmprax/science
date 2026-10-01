"""SYNTH-50's data contract and selection policy, in one place.

Everything here comes from ``docs/PROTOCOL_SYNTHESIS_50.md``. The protocol
exists because SYNTH-50 had no written rules, and its six tests were the only
artefact stating them -- which is how a test asserting ``logarithmic_boundary_layer``
and an engine constant forbidding near-OOD promotion came to contradict each
other with nothing to adjudicate between them.

Two rules, and the reason each is here rather than in the test that needs it:

**far-OOD is never handed to the selector** (§2). The tests were passing
``x_far_ood, y_far_ood`` into ``synthesize``. Measured: that does not change
which model is selected or what epistemic state is reached -- ``far_ood_mse``
appears twice in ``apedp/decision/stage_c.py``, as the ``CandidateProfile``
default and in the ``ood_tested`` boolean, and reaches neither the best-candidate
choice nor the state. It flips only whether the adjudication rationale claims
extrapolation was tested. So this is a reporting inconsistency rather than a
metric leak, but it is the opposite of what EXT-BENCH-60's adapter does
deliberately, and this loader makes the correct behaviour the default rather
than something each caller has to remember.

**near-OOD promotion is enabled** (§5.4). The referee applied the protocol's
pre-committed criterion to all 50 problems in both conditions and returned
PERMITTED: family accuracy 0.560 -> 0.640, median far-OOD NRMSE 0.0617 ->
0.0210, worst individual regression 7.92x against a 10x bound, zero new
abstentions. The engine default stays ``False`` -- EXT-BENCH-60's frozen adapter
still pins promotion off, and SYNTH-50's permission does not extend to callers
with no protocol of their own.

The caveats on that verdict live in §5.4 and are not summarised here on
purpose: the far-OOD conditions covered 31 of 50 problems and none of class 5,
and the 10x bound cleared by only 1.26x. Read them before quoting the verdict.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from apedp.model_synthesis import AutonomousModelSynthesizer, SynthesizedModel
from benchmarks.corpus_paths import CORPUS_50_DIR

__all__ = [
    "SYNTH50_ALLOW_NEAR_OOD_PROMOTE",
    "SYNTH50_SPLITS_PERMITTED",
    "load_splits",
    "synthesize",
]

#: §5.4 verdict. Permitted for SYNTH-50 only.
SYNTH50_ALLOW_NEAR_OOD_PROMOTE = True

#: The splits §2 permits the selector to see. ``far_ood`` is absent by
#: construction rather than by convention: ``load_splits`` never opens it, so a
#: caller cannot pass it even by accident.
SYNTH50_SPLITS_PERMITTED = ("train.csv", "validation.csv", "near_ood.csv")

#: Opened and scored only by the referee harness, never by a SYNTH-50 caller.
_SCORING_ONLY = "far_ood.csv"


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One ``x,y`` CSV as ``(x_column, y_column)`` numpy arrays.

    Both of these details were got wrong first time and cost a test run each,
    and in both cases the failure surfaced deep inside ``synthesize`` rather
    than at the loader:

      * returning the raw row tuples instead of transposing them -- a
        "too many values to unpack" from three call sites away;
      * returning Python lists instead of numpy arrays -- a
        "only integer scalar arrays can be converted to a scalar index" from
        inside ``detect_regimes``, which is where the previous loader's
        ``np.array(rows)[:, 0]`` mattered.

    A loader that is wrong in a way the engine can absorb reports the mistake
    as an engine bug, which is the wrong place to look.
    """
    rows = list(csv.reader(path.open(encoding="utf-8")))
    grid = np.array([[float(v) for v in row] for row in rows[1:]], dtype=float)
    return grid[:, 0], grid[:, 1]


def load_splits(problem_id: str) -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """``(spec, splits)`` for one SYNTH-50 problem, with far-OOD withheld.

    ``spec`` is the problem's ``problem_spec.json``, passed through because
    S21's synthesis is configured from it (``sampling_gaps``).

    The far-OOD grid is not read. That is the point: the previous tests loaded
    it and forwarded it, and a loader that simply did not have it available
    cannot reintroduce the mistake.
    """
    problem_dir = CORPUS_50_DIR / "public" / problem_id
    if not problem_dir.is_dir():
        raise FileNotFoundError(f"SYNTH-50 problem not found: {problem_dir}")

    import json

    spec = json.loads((problem_dir / "problem_spec.json").read_text(encoding="utf-8"))
    splits = {name: _read(problem_dir / name) for name in SYNTH50_SPLITS_PERMITTED}
    return spec, splits


def synthesize(problem_id: str, *, synthesizer: AutonomousModelSynthesizer | None = None,
               use_near_ood: bool = True) -> tuple[dict, list[SynthesizedModel]]:
    """Run SYNTH-50 synthesis under this corpus's protocol.

    Returns ``(spec, models)`` because callers assert against both.

    ``use_near_ood=False`` reproduces the configuration the class-5
    abstention tests have always used, which feeds only train and validation.
    It exists because passing near-OOD to an abstention adversary is a
    behavioural change those tests were not written to cover, and quietly
    altering an abstention assertion is exactly the kind of edit that should be
    explicit rather than incidental.
    """
    spec, splits = load_splits(problem_id)
    synth = synthesizer or AutonomousModelSynthesizer()

    x_train, y_train = splits["train.csv"]
    x_val, y_val = splits["validation.csv"]
    if use_near_ood:
        x_near, y_near = splits["near_ood.csv"]
    else:
        x_near = y_near = None

    models = synth.synthesize(
        x_train, y_train, x_val, y_val, x_near, y_near,
        # far-OOD is never forwarded: see the module docstring.
        allow_near_ood_promote=SYNTH50_ALLOW_NEAR_OOD_PROMOTE,
    )
    return spec, models