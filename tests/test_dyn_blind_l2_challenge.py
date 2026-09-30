"""CI test verifying the 10/10 Blind L2 Challenge Suite."""
from __future__ import annotations

from benchmarks.run_l2_blind_challenge import run_blind_challenge


class TestBlindL2ChallengeCI:
    """Verifies all 10 problems of the frozen Blind L2 Challenge Suite pass."""

    def test_all_ten_challenge_problems_pass(self):
        results = run_blind_challenge()
        assert len(results) == 10
        failed = [r for r in results if not r.passed_honest_contract]
        assert not failed, f"Failed challenge problems: {[f.problem_id for f in failed]}"
