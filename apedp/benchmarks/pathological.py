"""The 100-case pathological equation benchmark suite."""
from __future__ import annotations

from . import cases
from .results import (
    PathologicalCase,
)


class PathologicalEquationBenchmark:
    """Curated 25-case and 100-case benchmark suites spanning solvable and unrepairable scientific models.

    The case tables themselves live in :mod:`apedp.benchmarks.cases`; this class
    keeps the selection and running behaviour, so a subclass can still override
    either suite.
    """

    @classmethod
    def get_benchmark_cases(cls, suite_size: int = 25) -> list[PathologicalCase]:
        if suite_size == 100:
            return cls.get_100_case_suite()
        return cls._get_25_case_suite()

    @classmethod
    def _get_25_case_suite(cls) -> list[PathologicalCase]:
        """The 25-case suite: solvable breakdowns plus the unrepairable ones."""
        return cases._get_25_case_suite()

    @classmethod
    def get_100_case_suite(cls) -> list[PathologicalCase]:
        """Frozen 100-case cross-disciplinary benchmark suite spanning 10 distinct pathology categories."""
        return cases._get_100_case_suite()
