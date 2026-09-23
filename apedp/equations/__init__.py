"""APEDP equations corpus generator: seed -> fetch -> normalize -> canonicalize -> classify -> emit."""
from apedp.equations.pipeline import run_pipeline
from apedp.equations.schema import EquationRecord

__all__ = ["EquationRecord", "run_pipeline"]
