#!/usr/bin/env python3
"""Create the interpreter the official VQC verifiers need, and print how to use it.

The competition verifiers import ``numpy`` (and ``dfthcsos`` imports ``scipy``), but
the only interpreter that can load the shipped ``kickmix.so`` is the CPython 3.13
build that ``uv`` manages, and that interpreter is PEP 668 "externally managed":
``pip install`` refuses to touch it. ``--break-system-packages`` would work and is
the wrong answer -- it mutates a toolchain that other builds on this host share.

So the packages go in a virtual environment that inherits the same 3.13 ABI and
therefore loads the same extension. ``kickmix`` itself is not installed: it stays
beside the competition bundle, and ``PYTHONPATH`` puts it on the path, which keeps
the pinned ``kickmix.so`` in the one place ``contracts.py`` hashes it.

Usage::

    python3.13 scripts/setup_verifier_env.py          # create + report
    python3.13 scripts/setup_verifier_env.py --check  # report only, non-zero if absent

Then, from the bundle's ``verifiers/`` directory::

    export PYTHONPATH=$PWD/../kickmix
    ../../../../build/qvcc-venv313/bin/python dqi/verify_submission_dqi.py \\
        CIRCUIT.kmb --m 15 --n 6 --degree 4 --num-shots 1536
    ../../../../build/qvcc-venv313/bin/python dfthcsos/verify.py ../sample_submission/dfthcsos

Two acceptance facts, both measured and neither obvious:

* ``--num-shots`` must be a positive multiple of ``batch_size`` (256); anything else
  is rejected before a single shot is simulated.
* A run with zero errors can still FAIL. The score gate is the one-sided
  Clopper-Pearson upper bound ``eps_plus``, not the empirical rate, so 0/256 is
  ``eps_plus`` 0.2929 >= 1/16 and 0/1536 is 0.0561 < 1/16. Use at least 1,536 shots.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV = ROOT / "build" / "qvcc-venv313"
PACKAGES = ("numpy", "scipy")


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def probe(python: Path) -> dict[str, str]:
    """Return the versions of ``PACKAGES`` importable under ``python``."""
    found: dict[str, str] = {}
    for name in PACKAGES:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(python), "-c", f"import {name};print({name}.__version__)"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            found[name] = result.stdout.strip()
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV,
                        help=f"environment directory (default: {DEFAULT_VENV.relative_to(ROOT)})")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter that can load kickmix.so (default: this one)")
    parser.add_argument("--check", action="store_true",
                        help="report only; exit non-zero if the environment is incomplete")
    args = parser.parse_args()

    python = venv_python(args.venv)
    if not python.exists():
        if args.check:
            print(f"missing: {args.venv} (run without --check to create it)")
            return 1
        print(f"creating {args.venv} from {args.python}")
        subprocess.run([args.python, "-m", "venv", str(args.venv)], check=True)  # noqa: S603

    have = probe(python)
    missing = [name for name in PACKAGES if name not in have]
    if missing and args.check:
        print(f"incomplete: {args.venv} is missing {', '.join(missing)}")
        return 1
    if missing:
        print(f"installing {', '.join(missing)}")
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(python), "-m", "pip", "install", "--quiet", *missing], check=True)
        have = probe(python)

    print()
    print(f"interpreter  {python}")
    for name in PACKAGES:
        print(f"{name:12s} {have.get(name, 'MISSING')}")
    print()
    print("run the verifiers from the bundle's verifiers/ with:")
    print(f'  export PYTHONPATH="$PWD/../kickmix"')
    print(f"  {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
