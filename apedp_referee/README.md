# apedp-referee

Decides whether a claimed effect may be promoted, from evidence, against
thresholds fixed before the data was seen.

It is the referee of the APEDP-X repository, split out so it can be installed,
read and tested without the rest of that repository: no symbolic algebra, no
native kernel, no machine-learning framework. It needs numpy, scipy and pandas.

```bash
pip install ./apedp_referee            # from a checkout of the repository
```

## What it does

| Module | Decides |
| :--- | :--- |
| `canonical_spec` | Nothing; it loads the preregistered thresholds and refuses to import if their file's SHA-256 is not the pinned one. |
| `promotion_lifecycle` | `PromotionTracker`: six stages, in order, none skippable. Stage 1 formal soundness; 2 in-sample information coefficient from the signal and returns; 3 novelty against the incumbent (Diebold-Mariano) and literature factors; 4 out-of-sample transfer on a holdout claimed in the experiment ledger; 5 deflated Sharpe ratio after fees on an audited universe; 6 prospective performance. Each stage computes its statistic from the evidence it is given; a stage asked to accept a reported number refuses unless the caller says so explicitly. |
| `experiment_ledger` | `FullLineageExperimentLedger`: a holdout window is claimed by a fingerprint of its contents, so the same rows under a new name are the same window; a failed confirmation burns it for good (`HoldoutContaminationError` on reuse). The file on disk is hash-chained and refused on load if edited. |
| `trial_ledger` | `CandidateTrialLedger`: every variant tried is journaled, and the deflated Sharpe ratio uses that count. |
| `novelty_adjudication` | Stage 3's tests. |
| `universe` | `audit_universe`: whether the asset list was chosen with hindsight, and stale or unadjusted bars in the panel. |
| `proof_claim`, `proof_evidence` | The types a formal-verification run reports to stage 1. |

## Example

```python
import numpy as np
from apedp_referee.experiment_ledger import FullLineageExperimentLedger
from apedp_referee.promotion_lifecycle import PromotionTracker

rng = np.random.default_rng(0)
signal = rng.normal(size=600)
returns = 0.3 * signal + rng.normal(size=600)

tracker = PromotionTracker("my-signal", allow_reported_metrics=True)
tracker.verify_stage_1("MATHEMATICALLY_CONSISTENT", True, True)  # reported: no Lean run here
record = tracker.verify_stage_2_from_predictions(signal, returns)  # computed
print(record.passed, record.metrics)

ledger = FullLineageExperimentLedger(persistence_path="experiments.jsonl")
ledger.register_candidate("my-signal", "my-proposer", "0.3 * x")
# Stage 3 needs forecasts against an incumbent and literature factors, stage 4
# a holdout window claimed in `ledger`, stage 5 returns, turnover, a trial
# ledger and a universe audit; see tests/test_referee.py.
```

`tests/test_referee.py` walks every stage end to end and is the fuller
example. Stage 1 normally takes a `LeanVerificationResult` from a real proof
run (`verify_stage_1_from_lean`); reported numbers are accepted only with
`allow_reported_metrics=True`, and the record says they were reported.

## What it does not do

It does not propose anything, fit anything or fetch data. It does not know
whether the thresholds are the right ones, only that they are the ones that
were fixed in advance. The ledgers make holdout reuse and trial-count
omission detectable in the files they keep; they cannot detect a holdout read
outside them.

## Relation to the repository

Inside the repository the same modules are part of the full `apedp`
distribution, and each is also importable under its old name
(`apedp_dyn.epistemic.promotion_lifecycle`, `apedp_dyn.conf.canonical_spec`, ...),
which is the same module object, not a copy. Install either `apedp` or
`apedp-referee` into an environment, not both.

`canonical_preregistered_spec_v1.json` here is a byte-identical copy of the
repository's `specifications/canonical_preregistered_spec_v1.json`, the file
the governance trustees sign. Both must hash to `PINNED_SPEC_SHA256`; the
repository's tests check that they are identical.

Three conveniences come from the full distribution and are unavailable when
this package is installed alone: a default ledger location
(`FullLineageExperimentLedger.from_env`, `CandidateTrialLedger.from_env`),
remote ledger storage selected by `APEDP_STORAGE_BACKEND`, and the governance
manifest status. Each says what to pass instead.
