"""``apedp_dyn.conf.canonical_spec`` is ``apedp_referee.canonical_spec``.

The referee core is its own distribution, ``apedp-referee`` (directory
``apedp_referee/``). This name is kept so existing imports, string
monkeypatch targets and pickles keep working: importing it returns the very
same module object, not a copy, so ``apedp_dyn.conf.canonical_spec.X is apedp_referee.canonical_spec.X``
for every attribute, and patching either name patches both.
"""

import sys

from apedp_referee import canonical_spec as _module

sys.modules[__name__] = _module
