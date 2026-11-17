"""``apedp_dyn.epistemic.novelty_adjudication`` is ``apedp_referee.novelty_adjudication``.

The referee core is its own distribution, ``apedp-referee`` (directory
``apedp_referee/``). This name is kept so existing imports, string
monkeypatch targets and pickles keep working: importing it returns the very
same module object, not a copy, so ``apedp_dyn.epistemic.novelty_adjudication.X is apedp_referee.novelty_adjudication.X``
for every attribute, and patching either name patches both.
"""

import sys

from apedp_referee import novelty_adjudication as _module

sys.modules[__name__] = _module
