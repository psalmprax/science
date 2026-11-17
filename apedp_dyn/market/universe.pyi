# The type checker's view of ``apedp_dyn.market.universe``: at run time the .py beside this
# replaces itself with ``apedp_referee.universe`` in sys.modules, which static tools
# cannot follow. Every public name of that module, nothing of its own.
from apedp_referee.universe import *  # noqa: F403
