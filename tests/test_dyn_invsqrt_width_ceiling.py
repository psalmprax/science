"""Pins Benchmark I's width ceiling and the reason it is a ceiling.

Status: audit-only

`docs/INVSQRT_SUB62_DESIGN.md` §2.1 records that segmenting the champion's 21-bit
adder ``built`` a 76-qubit circuit but could not compile it: the exact-domain
compiler asserts ``Scratch function is not zero`` because the hand-off carry is
consumed and preserved by the next segment, and this build has no way to clear a
majority-valued qubit (``del_and`` clears an AND; ``mr`` does not exist here).

That failure is load-bearing evidence for the design note's verdict, so two
things are pinned:

* the note keeps saying it — the block, the exception text, and both measured
  ceilings (138,837 for the segmented build, 130,260 for record-restoring the
  registers), so an edit cannot quietly turn a blocked lever into an available one;
* the build keeps failing **for that reason**. If a future ``kickmix`` gains a
  Z-basis clear the segmented build will compile, and this test then fails on
  purpose: the 76-qubit variant has become real and has to be measured and the
  note updated, rather than the ceiling silently staying on the record.

The second test needs the ``kickmix`` extension (CPython 3.13) and is opt-in:

    APEDP_Q_KICKMIX_PYTHON=/path/to/python3.13 pytest tests/test_dyn_invsqrt_width_ceiling.py
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTE = REPO_ROOT / "docs/INVSQRT_SUB62_DESIGN.md"
SCRIPT = REPO_ROOT / "scripts/build_invsqrt_chunked.py"
KICKMIX_PYTHON = os.environ.get("APEDP_Q_KICKMIX_PYTHON")


def test_note_records_the_block_and_both_ceilings() -> None:
    text = NOTE.read_text()
    assert "Scratch function is not zero" in text, "the compiler's refusal is no longer quoted"
    assert "blocked, not expensive" in text, "the lever's status must stay explicit"
    assert "138,837" in text, "the segmented-build ceiling must stay on the record"
    assert "130,260" in text, "the record-restore ceiling must stay on the record"
    assert "portable_exact_domain.py:285" in text, "the failing assertion must stay sourced"


@pytest.mark.skipif(
    not KICKMIX_PYTHON,
    reason="set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter with kickmix",
)
def test_segmented_build_still_fails_for_the_documented_reason() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [KICKMIX_PYTHON, str(SCRIPT), "--chunk", "7", "--shots", "25600"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = result.stdout + result.stderr

    if result.returncode == 0:
        pytest.fail(
            "the segmented build now compiles: a Z-basis clear has probably appeared upstream.\n"
            "Re-measure the 76-qubit variant's Q, T and official score, then update\n"
            "docs/INVSQRT_SUB62_DESIGN.md §2.1 and the frontier note's §5 — the ceiling\n"
            "recorded there (~138,837 at Q=76) is no longer blocked."
        )

    assert "Scratch function is not zero" in output, (
        "the segmented build failed for a different reason than the note records:\n"
        + output[-3000:]
    )
