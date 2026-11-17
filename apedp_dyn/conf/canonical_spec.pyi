# The type checker's view of ``apedp_dyn.conf.canonical_spec``: at run time the .py beside this
# replaces itself with ``apedp_referee.canonical_spec`` in sys.modules, which static tools
# cannot follow. Every public name of that module, nothing of its own.
from apedp_referee.canonical_spec import *  # noqa: F403
