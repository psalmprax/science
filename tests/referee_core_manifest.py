"""The referee core, declared in one place.

"The referee" is the part of this repository that decides whether a claim may
be promoted: the preregistered thresholds, the promotion state machine, the
holdout and trial ledgers, the universe audit, and the evidence types those
consume. Everything else -- proposers, discovery engines, the native kernel,
sandboxes, the UI -- produces candidates and evidence for it to judge.

This file states which modules those are and what they may depend on.
``tests/test_referee_core_boundary.py`` enforces the statement three ways (the
import statements in the source, what importing the core loads, and what
running the core's own tests loads), and CI's ``referee-core`` job runs
``CORE_TESTS`` in an environment that has nothing installed but
``CORE_REQUIREMENTS`` and pytest. Adding a dependency to the referee therefore
means editing this file, in a diff a reviewer will see, and not only adding an
import.

It is a boundary inside one repository and one distribution. The files have
not moved and there is no separate ``apedp-referee`` package; see
docs/SYSTEM_FLAWS_AUDIT.md (G-01) for what that would still take.
"""

from __future__ import annotations

#: The referee itself.
CORE_MODULES: tuple[str, ...] = (
    "apedp_dyn.conf.canonical_spec",  # preregistered thresholds and their pin
    "apedp_dyn.epistemic.proof_claim",  # P0 / P1 / P2
    "apedp_dyn.epistemic.proof_evidence",  # what a formal-verification run reports
    "apedp_dyn.epistemic.promotion_lifecycle",  # the six-stage state machine
    "apedp_dyn.epistemic.experiment_ledger",  # holdout budget, burn, fingerprints, hash chain
    "apedp_dyn.epistemic.trial_ledger",  # trial count and deflated Sharpe
    "apedp_dyn.epistemic.novelty_adjudication",  # stage 3: Diebold-Mariano, literature overlap
    "apedp_dyn.market.universe",  # hindsight and stale-bar audit of a universe
)

#: Loaded as a consequence of importing the core: its parent packages, whose
#: ``__init__`` modules re-export lazily, and the helper that makes them lazy.
SUPPORT_MODULES: tuple[str, ...] = (
    "apedp_dyn",
    "apedp_dyn._lazy",
    "apedp_dyn.conf",
    "apedp_dyn.epistemic",
    "apedp_dyn.market",
)

#: Third-party distributions the core may import. Nothing else.
CORE_REQUIREMENTS: tuple[str, ...] = ("numpy", "scipy", "pandas")

#: Imports from the core to the rest of the repository that are allowed
#: because they are optional paths: written inside a function, reached only
#: when the caller asks for the behaviour named here, and never when the core
#: is merely imported or given explicit local paths.
OPTIONAL_EDGES: dict[tuple[str, str], str] = {
    ("apedp_dyn.epistemic.experiment_ledger", "apedp_dyn.env"): (
        "default ledger location, when the caller supplies no persistence path"
    ),
    ("apedp_dyn.epistemic.experiment_ledger", "apedp_dyn.epistemic.remote_storage"): (
        "remote storage backend, when APEDP_STORAGE_BACKEND or a backend object is given"
    ),
    ("apedp_dyn.epistemic.trial_ledger", "apedp_dyn.env"): (
        "default ledger location, when the caller supplies no persistence path"
    ),
}

#: Must never be loaded by importing the core or by running CORE_TESTS. A
#: deny-list beside the allow-list above, so a failure names the offender.
FORBIDDEN_AT_RUNTIME: tuple[str, ...] = (
    "apedp",  # the legacy root package
    "apedp_kernel",
    "apedp_vqc",
    "torch",
    "jax",
    "flax",
    "sympy",
    "pint",
    "sklearn",
    "xgboost",
    "matplotlib",
    "boto3",
    "google",
    "redis",
    "fastapi",
)

#: The core's own tests: module-level imports are core-only, so they collect
#: and run where nothing but CORE_REQUIREMENTS is installed.
CORE_TESTS: tuple[str, ...] = (
    "tests/test_referee_core_standalone.py",
    "tests/test_lean_model_conformance.py",
)

#: Test-side modules those tests import.
CORE_TEST_SUPPORT: tuple[str, ...] = (
    "tests",
    "tests.conftest",
    "tests.promotion_evidence",
    "tests.referee_core_manifest",
    "tests.test_referee_core_standalone",
    "tests.test_lean_model_conformance",
)
