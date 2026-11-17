"""The referee core, declared in one place.

"The referee" is the part of this repository that decides whether a claim may
be promoted: the preregistered thresholds, the promotion state machine, the
holdout and trial ledgers, the universe audit, and the evidence types those
consume. Everything else -- proposers, discovery engines, the native kernel,
sandboxes, the UI -- produces candidates and evidence for it to judge.

Since 2026-11-17 it is its own package and distribution, ``apedp_referee``
(``apedp_referee/pyproject.toml``, name ``apedp-referee``). This file states
which modules it has and what they may depend on.
``tests/test_referee_core_boundary.py`` enforces the statement four ways (the
import statements in the source, what importing the core loads, what running
the core's own tests loads, and the package's own declared dependencies), and
CI's ``referee-core`` job installs the package from its directory into an
environment that has nothing but ``CORE_REQUIREMENTS`` and pytest and runs its
tests from outside the checkout. Adding a dependency to the referee therefore
means editing this file and the package's ``pyproject.toml``, in a diff a
reviewer will see, and not only adding an import.

Every module also keeps its old name in ``apedp_dyn`` (``ALIASES``): importing
the old name returns the same module object, so nothing that imported it, or
patches it by name, changed behaviour.
"""

from __future__ import annotations

#: The referee itself.
CORE_MODULES: tuple[str, ...] = (
    "apedp_referee.canonical_spec",  # preregistered thresholds and their pin
    "apedp_referee.proof_claim",  # P0 / P1 / P2
    "apedp_referee.proof_evidence",  # what a formal-verification run reports
    "apedp_referee.promotion_lifecycle",  # the six-stage state machine
    "apedp_referee.experiment_ledger",  # holdout budget, burn, fingerprints, hash chain
    "apedp_referee.trial_ledger",  # trial count and deflated Sharpe
    "apedp_referee.novelty_adjudication",  # stage 3: Diebold-Mariano, literature overlap
    "apedp_referee.universe",  # hindsight and stale-bar audit of a universe
    "apedp_referee._host",  # the three conveniences that come from apedp_dyn, on request
)

#: Loaded as a consequence of importing the core.
SUPPORT_MODULES: tuple[str, ...] = ("apedp_referee",)

#: Third-party distributions the core may import. Nothing else. They are also
#: exactly the package's declared dependencies.
CORE_REQUIREMENTS: tuple[str, ...] = ("numpy", "scipy", "pandas")

#: Imports from the core to the rest of the repository that are allowed
#: because they are optional paths: written inside a function, reached only when the caller asks for the behaviour named
#: here, and never when the core is merely imported or given explicit paths.
OPTIONAL_EDGES: dict[tuple[str, str], str] = {
    ("apedp_referee._host", "apedp_dyn.env"): (
        "default ledger location and remote-cache directory, when the caller supplies no path"
    ),
    ("apedp_referee._host", "apedp_dyn.epistemic.remote_storage"): (
        "remote storage backend, when APEDP_STORAGE_BACKEND is set"
    ),
}

#: Old name -> the module it is. Each old file is a four-line alias.
ALIASES: dict[str, str] = {
    "apedp_dyn.conf.canonical_spec": "apedp_referee.canonical_spec",
    "apedp_dyn.epistemic.proof_claim": "apedp_referee.proof_claim",
    "apedp_dyn.epistemic.proof_evidence": "apedp_referee.proof_evidence",
    "apedp_dyn.epistemic.promotion_lifecycle": "apedp_referee.promotion_lifecycle",
    "apedp_dyn.epistemic.experiment_ledger": "apedp_referee.experiment_ledger",
    "apedp_dyn.epistemic.trial_ledger": "apedp_referee.trial_ledger",
    "apedp_dyn.epistemic.novelty_adjudication": "apedp_referee.novelty_adjudication",
    "apedp_dyn.market.universe": "apedp_referee.universe",
}

#: What importing an old name costs on top of the core: the parent packages,
#: whose ``__init__`` modules re-export lazily, and the helper that makes them
#: lazy. Before 2026-11-16 importing the promotion lifecycle by its old name
#: executed 74 repository modules.
ALIAS_SUPPORT: tuple[str, ...] = (
    "apedp_dyn",
    "apedp_dyn._lazy",
    "apedp_dyn.conf",
    "apedp_dyn.epistemic",
    "apedp_dyn.market",
)

#: Must never be loaded by importing the core or by running CORE_TESTS. A
#: deny-list beside the allow-list above, so a failure names the offender.
FORBIDDEN_AT_RUNTIME: tuple[str, ...] = (
    "apedp",  # the legacy root package
    "apedp_dyn",  # everything the referee judges
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

#: The core's tests. The first ships with the package and imports nothing but
#: the package, its requirements and the helpers beside it; the second checks
#: the state machine and holdout guard against the Lean models in
#: lean_checked/, so it needs the checkout.
PACKAGE_TESTS: tuple[str, ...] = ("apedp_referee/tests/test_apedp_referee.py",)
CORE_TESTS: tuple[str, ...] = (*PACKAGE_TESTS, "tests/test_lean_model_conformance.py")

#: Test-side modules those tests import (as named in ``sys.modules``).
CORE_TEST_SUPPORT: tuple[str, ...] = (
    "tests",
    "tests.conftest",
    "tests.referee_core_manifest",
    "tests.test_lean_model_conformance",
    "test_apedp_referee",
    "referee_evidence",
)
