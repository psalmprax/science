"""apedp-referee: decides whether a claimed effect may be promoted.

Eight modules, numpy, scipy and pandas, nothing else:

- ``canonical_spec`` -- the preregistered thresholds, verified against a pinned
  SHA-256 at import.
- ``promotion_lifecycle`` -- ``PromotionTracker``, six stages in order, each
  computed from evidence rather than reported numbers.
- ``experiment_ledger`` -- ``FullLineageExperimentLedger``: holdout windows
  claimed by content fingerprint, burned on failure, hash-chained on disk.
- ``trial_ledger`` -- ``CandidateTrialLedger``: every trial counted, and the
  Sharpe ratio deflated for the count.
- ``novelty_adjudication`` -- stage 3: Diebold-Mariano against the incumbent,
  overlap with literature factors.
- ``universe`` -- ``audit_universe``: hindsight selection, stale bars and
  unadjusted splits in the panel a backtest ran on.
- ``proof_claim``, ``proof_evidence`` -- what a formal-verification run
  reports, for stage 1.

The package imports nothing at top level; import the module you need. It is
developed inside the APEDP-X repository, where ``apedp_dyn`` re-exports every
module under its old name (``apedp_dyn.epistemic.promotion_lifecycle`` and so
on) as the same object.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
