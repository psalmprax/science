# The type checker's view of ``apedp_dyn.epistemic.proof_evidence``: at run time the .py beside this
# replaces itself with ``apedp_referee.proof_evidence`` in sys.modules, which static tools
# cannot follow. Every public name of that module, nothing of its own.
from apedp_referee.proof_evidence import *  # noqa: F403
