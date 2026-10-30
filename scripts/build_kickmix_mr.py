#!/usr/bin/env python3
"""Build a local, MR-capable ``kickmix`` extension from the vendored sources.

The competition build in ``kickmix/kickmix.so`` has no Z-basis measurement, so the
MR-gated DQI arms cannot run on it. The sources under
``kickmix/_qualtran_src/kickmix`` do carry the full instruction set, and
``scripts/kickmix_mr_opcode.patch`` adds the ``MR``/``MR_IF`` opcodes
(``docs/MR_INSTRUCTION_SPEC.md``) to them. This script exports those sources at
their recorded commit into ``build/kickmix_mr``, applies the patch, and compiles
the extension there.

The pinned binary is never touched: ``apedp_dyn/quantum_compilation/contracts.py``
hashes it, and only that build may produce a verifier receipt. A circuit compiled
against this extension is a local correctness experiment, not a scored arm.

The build needs ``pybind11`` and ``setuptools`` for the interpreter that will load
the result. Run it with the interpreter you intend to test under, e.g.::

    /dev/shm/qvcc-venv313/bin/python scripts/build_kickmix_mr.py

It prints the directory to put on ``sys.path`` (default
``build/kickmix_mr/lib``).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
VENDOR = BUNDLE / "kickmix" / "_qualtran_src" / "kickmix"
PATCH = ROOT / "scripts" / "kickmix_mr_opcode.patch"


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} is required to build kickmix and is not on PATH")
    return path


def export_sources(destination: Path) -> None:
    """Write the vendored tree at its recorded commit into ``destination``."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [_tool("git"), "-C", str(VENDOR), "archive", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
    )
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        [_tool("tar"), "-x", "-C", str(destination)], input=archive.stdout, check=True)


def apply_patch(source_root: Path) -> None:
    """Apply the MR opcode patch, refusing a tree it does not match.

    ``patch`` rather than ``git apply``: the scratch tree sits inside this
    repository, and ``git apply`` silently ignores patches whose paths fall
    outside the current subdirectory.
    """
    with PATCH.open("rb") as patch_file:
        applied = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [_tool("patch"), "-p1", "--forward", "--batch"],
            cwd=source_root,
            stdin=patch_file,
            capture_output=True,
            text=True,
        )
    if applied.returncode != 0:
        raise SystemExit(f"patch did not apply:\n{applied.stdout}{applied.stderr}")


def build(source_root: Path, lib_dir: Path, temp_dir: Path) -> Path:
    lib_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - this interpreter, fixed argv, no shell
        [
            sys.executable,
            "setup.py",
            "build_ext",
            f"--build-lib={lib_dir}",
            f"--build-temp={temp_dir}",
        ],
        cwd=source_root,
        check=True,
    )
    built = sorted(lib_dir.glob("kickmix*.so"))
    if not built:
        raise SystemExit(f"no kickmix extension was produced in {lib_dir}")
    return built[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "kickmix_mr",
                        help="scratch directory for sources and the built extension")
    args = parser.parse_args()

    for module in ("pybind11", "setuptools"):
        try:
            __import__(module)
        except ImportError:
            raise SystemExit(
                f"{module} is missing for {sys.executable}; "
                f"install it or run this with an interpreter that has it"
            ) from None

    if args.out.exists():
        shutil.rmtree(args.out)
    source_root = args.out / "src"
    export_sources(source_root)
    apply_patch(source_root)
    source = source_root / "src" / "kickmix" / "sim" / "sim.h"
    if "OpType::MR" not in source.read_text():
        raise SystemExit(f"patch did not add MR; check {PATCH}")
    extension = build(source_root, args.out / "lib", args.out / "tmp")
    print(extension.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
