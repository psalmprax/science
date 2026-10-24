"""Mutation-equivalence harness for reconstructed-from-bytecode test files.

Answers one question: is a re-authored test file actually equivalent to the
orphaned ``.pyc`` it was reconstructed from, or did the reconstruction
weaken it?

Both versions are run against the same mutated subject module and their
per-test verdicts are compared. A mutant that the original kills and the
reconstruction passes is a test that got weaker in the round trip, which is
the failure mode worth catching -- a reconstructed suite that looks green
and asserts less is worse than no suite, because it launders the appearance
of coverage.

Design notes, both learned the hard way here:

- Each side runs in a **fresh subprocess**. The first version of this harness
  imported the bytecode module in-process, so ``apedp.singularity_probing``
  stayed cached in ``sys.modules`` after the first mutant and every later
  mutation was silently a no-op. Two mutants were reported as disagreements
  that were not disagreements at all, and the mutation that actually applied
  looked like a real finding.
- Verdict comparison normalises the failure reason. The bytecode runner
  reports ``FAIL:AssertionError``; pytest reports ``FAIL``. Comparing the
  raw strings manufactures disagreements out of formatting.

Restores are hash-verified after every mutant. The repository is shared
across concurrent sessions, and a partial restore would silently break
someone else's work.

Usage::

    python3 tools/orphan_equivalence.py singularity

Restricts the comparison to a reconstructed file that still runs against
today's source. Files whose subject API has drifted are excluded, and the
module docstring of the reconstruction should say so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Runs an orphaned bytecode test module in isolation and prints per-test
# verdicts as JSON. A subprocess because sys.modules caching silently
# defeats in-process mutation (see module docstring). No __file__ reference:
# ``python3 -c`` does not define one.
BYTECODE_RUNNER = r'''
import importlib.util, json, pathlib, sys
ROOT = pathlib.Path({root!r})
sys.path.insert(0, str(ROOT))
c = ROOT / {pyc!r}
if not c.exists():
    raise SystemExit("missing bytecode: " + str(c))
spec = importlib.util.spec_from_file_location("orphan_under_test", c)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
out = {{}}
for n in sorted(dir(m)):
    if n.startswith("test_"):
        try:
            getattr(m, n)()
            out[n] = "PASS"
        except Exception as e:
            out[n] = "FAIL:" + type(e).__name__
print("RESULT " + json.dumps(out))
'''

REAUTHORED_RUNNER = r'''
import importlib.util, json, pathlib, sys
ROOT = pathlib.Path({root!r})
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("reauthored_under_test", ROOT / {py!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
out = {{}}
for n in sorted(dir(m)):
    if n.startswith("test_"):
        fn = getattr(m, n)
        try:
            fn()
            out[n] = "PASS"
        except Exception as e:
            out[n] = "FAIL:" + type(e).__name__
print("RESULT " + json.dumps(out))
'''

SUBJECTS = {
    "singularity": {
        # The pytest-9.1.1 bytecode tag is the *pre-existing* orphan (mtime
        # 2026-09-25). The pytest-8.1.1 tag is what the installed pytest
        # writes when the reconstructed .py is collected, so a bare glob or a
        # "largest file wins" heuristic picks the reconstruction and compares
        # it against itself, which passes trivially and proves nothing. The
        # tag is pinned deliberately.
        "pyc": "tests/__pycache__/"
               "test_singularity_probing.cpython-312-pytest-9.1.1.pyc",
        "module": "apedp/singularity_probing.py",
        "reauthored": "tests/test_singularity_probing.py",
        "mutants": [
            ("risk_type AVOIDED_POLE_IN_GAP -> UNVERIFIED",
             'risk_type="AVOIDED_POLE_IN_GAP",', 'risk_type="UNVERIFIED",'),
            ("risk_type UNOBSERVED_CRITICAL_REGIME -> UNVERIFIED",
             'risk_type="UNOBSERVED_CRITICAL_REGIME",', 'risk_type="UNVERIFIED",'),
            ("gap threshold -> 99 (never fires)",
             "self.gap_factor_threshold = gap_factor_threshold",
             "self.gap_factor_threshold = 99.0"),
            ("gap threshold -> 0 (always fires)",
             "self.gap_factor_threshold = gap_factor_threshold",
             "self.gap_factor_threshold = 0.0"),
            ("blowup threshold -> 1e9 (never fires)",
             "self.blowup_threshold_factor = blowup_threshold_factor",
             "self.blowup_threshold_factor = 1e9"),
            ("blowup threshold -> 0 (always divergent)",
             "self.blowup_threshold_factor = blowup_threshold_factor",
             "self.blowup_threshold_factor = 0.0"),
            ("root search returns nothing",
             "        roots: list[float] = []\n        failures: list[str] = []",
             "        roots: list[float] = []\n"
             "        return roots, failures\n        failures: list[str] = []"),
            ("smooth-alternative requirement dropped",
             "has_smooth_alternative = any(len(self.extract_singular_components(e)) "
             "== 0 for e in competing_expressions)",
             "has_smooth_alternative = True"),
            ("min-distance gate -> 1e9 (never fires)",
             "if min_dist_to_obs > 0.005 and has_smooth_alternative:",
             "if min_dist_to_obs > 1e9 and has_smooth_alternative:"),
            ("domain buffer ratio -> 0",
             "self.domain_buffer_ratio = domain_buffer_ratio",
             "self.domain_buffer_ratio = 0.0"),
            ("root dedup tolerance -> 0",
             "and not any(abs(val - r) < 1e-4 for r in roots):",
             "and not any(abs(val - r) < 0.0 for r in roots):"),
            ("is_divergent loses the non-finite clause",
             "is_divergent = max_probe_val > self.blowup_threshold_factor * y_max"
             " or not math.isfinite(max_probe_val)",
             "is_divergent = max_probe_val > self.blowup_threshold_factor * y_max"),
            ("y_max floor 1e-6 -> 1e6",
             "y_max = max(y_max, 1e-6)", "y_max = max(y_max, 1e6)"),
        ],
    },
}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bytecode_verdicts(cfg: dict) -> dict[str, str]:
    runner = BYTECODE_RUNNER.format(root=str(ROOT), pyc=cfg["pyc"])
    r = subprocess.run([sys.executable, "-c", runner], cwd=ROOT,  # noqa: S603
                       capture_output=True, text=True, timeout=300)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"__error__": r.stderr[-400:]}


def reauthored_verdicts(cfg: dict) -> dict[str, str]:
    # Symmetric with the bytecode side: import the module and call each test
    # directly. An earlier version shelled out to pytest and scraped FAILED
    # lines, which only ever recorded the tests that failed -- every passing
    # test came back absent and manufactured a disagreement on 8 of 13
    # mutants. Both sides now report every test, pass or fail.
    runner = REAUTHORED_RUNNER.format(root=str(ROOT), py=cfg["reauthored"])
    r = subprocess.run([sys.executable, "-c", runner], cwd=ROOT,  # noqa: S603
                       capture_output=True, text=True, timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"__error__": r.stderr[-400:]}


def norm(d: dict[str, str]) -> dict[str, str]:
    return {k.split("[")[0]: v.split(":")[0] for k, v in d.items()}


def run(name: str) -> int:
    cfg = SUBJECTS[name]
    target = ROOT / cfg["module"]
    src = target.read_text()
    original_hash = sha(target)
    agree = disagree = skipped = 0
    aborted = False
    try:
        for label, old, new in cfg["mutants"]:
            if aborted:
                break
            if old not in src:
                print(f"SKIP      {label}  (anchor absent)")
                skipped += 1
                continue
            target.write_text(src.replace(old, new, 1))
            try:
                a = norm(bytecode_verdicts(cfg))
                b = norm(reauthored_verdicts(cfg))
            finally:
                target.write_text(src)
                if sha(target) != original_hash:
                    print("RESTORE MISMATCH -- aborting, tree is shared")
                    aborted = True
            if aborted:
                break
            if a == b:
                agree += 1
                print(f"AGREE     {label}")
            else:
                disagree += 1
                print(f"DISAGREE  {label}")
                for k in sorted(set(a) | set(b)):
                    if a.get(k) != b.get(k):
                        print(f"            {k}: bytecode={a.get(k)} "
                              f"reauthored={b.get(k)}")
    finally:
        target.write_text(src)
        ok = sha(target) == original_hash
        print("final restore:", "OK" if ok else "MISMATCH")
        if not ok:
            aborted = True

    total = agree + disagree
    print(f"\n{agree}/{total} mutants verdict-identical, {disagree} disagreed,"
          f" {skipped} skipped")
    if total == 0:
        print("no mutant anchors matched -- the subject file has moved on")
        return 2
    if aborted:
        return 2
    return 1 if disagree else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subject", nargs="?", default="singularity",
                    choices=sorted(SUBJECTS),
                    help="which reconstructed file to verify")
    args = ap.parse_args()
    return run(args.subject)


if __name__ == "__main__":
    raise SystemExit(main())
