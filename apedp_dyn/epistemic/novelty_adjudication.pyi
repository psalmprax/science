# The type checker's view of ``apedp_dyn.epistemic.novelty_adjudication``: at run time the .py beside this
# replaces itself with ``apedp_referee.novelty_adjudication`` in sys.modules, which static tools
# cannot follow. Every public name of that module, nothing of its own.
from apedp_referee.novelty_adjudication import *  # noqa: F403
