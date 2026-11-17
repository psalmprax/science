"""``apedp_dyn.market.universe`` is ``apedp_referee.universe``.

The referee core is its own distribution, ``apedp-referee`` (directory
``apedp_referee/``). This name is kept so existing imports, string
monkeypatch targets and pickles keep working: importing it returns the very
same module object, not a copy, so ``apedp_dyn.market.universe.X is apedp_referee.universe.X``
for every attribute, and patching either name patches both.
"""

import sys

from apedp_referee import universe as _module

sys.modules[__name__] = _module
