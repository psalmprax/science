"""D1 benchmark problems."""
from apedp_dyn.problems import coupled, duffing, mass_spring_damper, pendulum
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

__all__ = ["DynProblem", "coupled", "duffing", "finalize", "make_signal",
           "mass_spring_damper", "pendulum"]
