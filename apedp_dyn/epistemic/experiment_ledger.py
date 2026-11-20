"""``apedp_dyn.epistemic.experiment_ledger`` is ``apedp_referee.experiment_ledger``.

The referee core is its own distribution, ``apedp-referee`` (directory
``apedp_referee/``). This name is kept so existing imports, string
monkeypatch targets and pickles keep working: importing it returns the very
same module object, not a copy, so ``apedp_dyn.epistemic.experiment_ledger.X is apedp_referee.experiment_ledger.X``
for every attribute, and patching either name patches both.
"""

import sys

from apedp_referee import experiment_ledger as _module

sys.modules[__name__] = _module
