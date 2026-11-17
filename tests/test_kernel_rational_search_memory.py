"""The native rational search does not keep every candidate's values.

``rime_rational_search_json`` pairs each numerator with each denominator and
scores the quotient. Until 2026-11-16 the quotient's n-vector stayed in the
evaluation cache for the life of the search. On the 200-row, one-feature frame
of ``tests/test_dyn_rime_ad_screening.py`` one call took the process from
262 MB to 2,992 MB. That test "failed only in the parallel run" on an 8 GB
machine because its worker was killed, and it took four other workers' tests
with it in the next full run.

The call is measured in a child process so that the peak belongs to it alone,
and by the child's ``VmHWM``, not ``getrusage``: on Linux ``ru_maxrss`` survives
fork and exec, so a child of a pytest worker that had reached 3.2 GB reported
3.2 GB for a search that peaks at 376 MB (full run of 2026-11-16).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("apedp_kernel")
if not Path("/proc/self/status").exists():
    pytest.skip("VmHWM is read from /proc/self/status, which this platform lacks", allow_module_level=True)

#: Peak resident set allowed for the child, in MB. Measured: 2,992 before the
#: fix, about 600 after for the whole screening test and less for one call. The
#: limit sits between the two with room for a debug build and another
#: allocator.
PEAK_LIMIT_MB = 1200

_CHILD = r"""
import json
import numpy as np
import apedp_kernel as ak

rng = np.random.RandomState(7)
n = 200
data = {f"x{k}": rng.randn(n) for k in range(10)}
target = data["x3"] ** 2 + 0.01 * rng.randn(n)
raw = ak.rime_rational_search_json(
    json.dumps([[float(v) for v in data["x3"]]]), json.dumps(target.tolist()),
    json.dumps(["x3"]), False, False, False, False, 2, 3,
)
with open("/proc/self/status", encoding="ascii") as status:
    peak_kb = next(int(line.split()[1]) for line in status if line.startswith("VmHWM:"))
print(json.dumps({"candidates": len(json.loads(raw)), "peak_mb": peak_kb // 1024}))
"""


def _run_child() -> dict[str, int]:
    result = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.slow  # one real search: seconds on a release build, longer on a debug one
def test_one_rational_search_stays_under_the_memory_limit():
    measured = _run_child()
    assert measured["candidates"] > 0, "the search returned nothing, so the peak measures nothing"
    assert measured["peak_mb"] < PEAK_LIMIT_MB, (
        f"peak resident set {measured['peak_mb']} MB for a 200-row, one-feature search; "
        "the per-pair values are being retained again (rust/apedp-kernel/src/rime/enumerate.rs)"
    )
