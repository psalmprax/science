#!/usr/bin/env python3
"""Build the reproducible, MR-capable ``kickmix`` extension and verify it against its pin.

The competition build in ``kickmix/kickmix.so`` has no Z-basis measurement, so the
MR-gated DQI arms and the measured clear cannot run on it. The vendored sources under
``kickmix/_qualtran_src`` do carry the instruction, and
``scripts/kickmix_mr_opcode.patch`` adds the ``MR`` opcode and the 54-AND multiply
policy. This script exports those sources **at their recorded commit** (not from the
working tree, which carries unrelated edits), applies the patch, compiles, and then
proves the result is the pinned build in
:mod:`apedp_dyn.quantum_compilation.kickmix_builds`:

* the built extension's SHA-256 must equal ``MR_BUILD.sha256`` — a mismatch is an error,
  not a warning, because a receipt naming an unpinned build proves nothing;
* ``--compare`` re-hashes other builds and probes them for the same instruction surface,
  which is how byte-reproducibility across output directories is recorded rather than
  asserted.

The pinned competition binary is never touched.

Paths are resolved to absolute before ``setup.py`` runs: ``build_ext`` resolves
``--build-lib``/``--build-temp`` against its own working directory, which is the
extracted source tree, so a relative path silently puts the extension somewhere the
caller does not look.

    /dev/shm/qvcc-venv313/bin/python scripts/build_kickmix_mr.py \\
        --record benchmarks/quantum_compilation/kickmix_mr_build.json

``--check`` verifies an existing build against the pin without compiling.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
VENDOR = BUNDLE / "kickmix" / "_qualtran_src" / "kickmix"
PATCH = ROOT / "scripts" / "kickmix_mr_opcode.patch"
DEFAULT_OUT = ROOT / "build" / "kickmix_mr"
DEFAULT_RECORD = ROOT / "benchmarks" / "quantum_compilation" / "kickmix_mr_build.json"

# Loaded by path: importing the `apedp_dyn` package pulls the whole scientific stack
# (pandas and friends), and the interpreter that can build a CPython 3.13 extension need
# not be the one that has it. The pin module itself is stdlib-only.
def _load_pins():
    path = ROOT / "apedp_dyn" / "quantum_compilation" / "kickmix_builds.py"
    spec = importlib.util.spec_from_file_location("kickmix_builds", path)
    module = importlib.util.module_from_spec(spec)
    # `dataclasses` resolves a class's defining module through `sys.modules`, so the module
    # has to be registered before it runs, not after.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pins = _load_pins()

#: Reports the pinned build's instruction surface, so a rebuild that loses `mr` or gains
#: a different multiply policy is visible in the same record as its digest.
_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import kickmix as km
field = km.GF2Field(12)
builder = km.CircuitBuilder()
source = builder.create_quantum_register(12, name="A")
target = builder.create_quantum_register(12, name="B")
bits = builder.alloc_bits(12)
_, chain = builder.init_gf2_inverse_with_scaffold(source, target=target, field=field)
has_mr = hasattr(km.CircuitBuilder, "mr")
if has_mr:
    builder.mr(km.array(list(target)), bits)
builder.del_gf2_inverse_with_scaffold(source, target=target, scaffold=chain, field=field)
builder.free(chain)
circuit = builder.finish_circuit()
print(json.dumps({
    "has_mr": has_mr,
    "has_mr_if": hasattr(km.CircuitBuilder, "mr_if"),
    "public_methods": len([m for m in dir(km.CircuitBuilder) if not m.startswith("_")]),
    "qubits": circuit.num_qubits,
    "ops": len(circuit),
    "instructions": {k: int(v) for k, v in circuit.operation_statistics().items() if int(v)},
}))
"""


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} is required to build kickmix and is not on PATH")
    return path


def _sha256(path: Path) -> str:
    return pins.sha256_file(path)


def _git(*args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [_tool("git"), "-C", str(VENDOR), *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _extension(directory: Path) -> Path:
    """The built extension in ``directory``, refusing an ambiguous match."""
    directory = Path(directory)
    if directory.is_file():
        return directory
    found = sorted(directory.glob("kickmix*.so"))
    if not found:
        raise SystemExit(f"no kickmix extension in {directory}")
    if len(found) > 1:
        raise SystemExit(f"more than one kickmix extension in {directory}: {found}")
    return found[0]


def probe(extension: Path) -> dict:
    """Import ``extension`` in a subprocess and report its instruction surface."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", _PROBE, str(Path(extension).parent)],
        capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def export_sources(destination: Path) -> None:
    """Write the vendored tree at its recorded commit into ``destination``."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [_tool("git"), "-C", str(VENDOR), "archive", "HEAD"], check=True, stdout=subprocess.PIPE)
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        [_tool("tar"), "-x", "-C", str(destination)], input=archive.stdout, check=True)


def apply_patch(source_root: Path) -> None:
    """Apply the MR patch, after proving it is the patched-in file the pin names."""
    digest = _sha256(PATCH)
    pinned = pins.MR_BUILD.provenance["patch_sha256"]
    if digest != pinned:
        raise SystemExit(
            f"{PATCH.name} hashes to {digest}, but the pin was produced with {pinned}; "
            f"a different patch cannot reproduce build {pins.MR_BUILD.sha256[:12]}…"
        )
    with PATCH.open("rb") as patch_file:
        applied = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [_tool("patch"), "-p1", "--forward", "--batch"],
            cwd=source_root, stdin=patch_file, capture_output=True, text=True)
    if applied.returncode != 0:
        raise SystemExit(f"patch did not apply:\n{applied.stdout}{applied.stderr}")


def check_toolchain() -> dict:
    """Refuse a toolchain the pin was not produced with."""
    try:
        import pybind11  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            f"pybind11 is missing for {sys.executable}; the pin needs pybind11 2.12.0"
        ) from None
    wanted_python = "3.13"
    wanted_pybind11 = pins.MR_BUILD.provenance["pybind11"]
    if platform.python_version().split(".")[:2] != wanted_python.split("."):
        raise SystemExit(
            f"this interpreter is CPython {platform.python_version()}; the pin was produced "
            f"with {wanted_python} (the extension's ABI tag depends on it)"
        )
    if pybind11.__version__ != wanted_pybind11:
        raise SystemExit(
            f"pybind11 {pybind11.__version__} is installed; the pin was produced with "
            f"{wanted_pybind11}"
        )
    compiler = shutil.which("c++")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True,  # noqa: S603
                             check=False).stdout.splitlines()[:1] if compiler else []
    return {
        "python": platform.python_version(),
        "pybind11": pybind11.__version__,
        "compiler": version[0].strip() if version else "unknown",
        "platform": platform.platform(),
    }


def build(source_root: Path, lib_dir: Path, temp_dir: Path) -> Path:
    """Compile into absolute ``lib_dir``; ``setup.py`` would resolve relatives wrongly."""
    lib_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "setup.py",
        "build_ext",
        f"--build-lib={lib_dir.resolve()}",
        f"--build-temp={temp_dir.resolve()}",
    ]
    subprocess.run(command, cwd=source_root, check=True)  # noqa: S603 - fixed argv, no shell
    return _extension(lib_dir)


def record_receipt(*, extension: Path, source_commit: str, dirty: bool, toolchain: dict,
                   compares: list[dict], record: Path, command: str, mode: str) -> dict:
    built = probe(extension)
    digest = _sha256(extension)
    payload = {
        "receipt": "the MR-capable kickmix extension, rebuilt from this repository",
        "why": "the shipped competition build has no `mr`, so every MR-gated arm and every "
               "measured clear runs on this extension; until now its bytes had no name, so "
               "a receipt citing it proved nothing",
        "script": "scripts/build_kickmix_mr.py",
        "script_sha256": _sha256(Path(__file__)),
        "command": command,
        "mode": mode,
        "source": {
            "tree": str(VENDOR.relative_to(ROOT)),
            "commit": source_commit,
            "pinned_commit": pins.MR_BUILD.provenance["source_commit"],
            "checkout_has_uncommitted_edits": dirty,
            "note": "exported with `git archive HEAD`, so uncommitted edits cannot leak in",
        },
        "patch": {"file": str(PATCH.relative_to(ROOT)), "sha256": _sha256(PATCH)},
        "toolchain": toolchain,
        "build": {
            "extension": str(extension.relative_to(ROOT)) if extension.is_relative_to(ROOT)
                         else str(extension),
            "bytes": extension.stat().st_size,
            "sha256": digest,
            "instructions": built,
        },
        "pin": {
            "registry": "apedp_dyn/quantum_compilation/kickmix_builds.py",
            "expected_sha256": pins.MR_BUILD.sha256,
            "matched": digest == pins.MR_BUILD.sha256,
            "allowed_for_verifier": pins.MR_BUILD.allowed_for_verifier,
        },
        "comparisons": compares,
        "determinism": {
            "claim": "two builds of the pinned commit plus patch in different output "
                     "directories produce identical bytes",
            "evidence": [row for row in compares if row["identical_bytes"]] or None,
        },
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def compare_builds(extension: Path, digest: str, targets: list[Path]) -> list[dict]:
    """Hash and probe other builds, refusing one whose instruction surface differs."""
    rows: list[dict] = []
    mine = probe(extension)
    for target in targets:
        other = Path(target).resolve()
        if not other.exists():
            raise SystemExit(f"--compare target does not exist: {other}")
        other_extension = _extension(other)
        other_digest = _sha256(other_extension)
        theirs = probe(other_extension)
        known = pins.identify(other_extension)
        row = {
            "path": str(other_extension.relative_to(ROOT))
                    if other_extension.is_relative_to(ROOT) else str(other_extension),
            "sha256": other_digest,
            "identical_bytes": other_digest == digest,
            "same_instructions": theirs["instructions"] == mine["instructions"],
            "same_public_methods": theirs["public_methods"] == mine["public_methods"],
            "pinned_build": known.to_dict() if known else None,
            "other_instructions": theirs["instructions"],
        }
        rows.append(row)
        print(f"compare {row['path']}: bytes "
              f"{'identical' if row['identical_bytes'] else 'differ'}, instructions "
              f"{'identical' if row['same_instructions'] else 'DIFFER'}, "
              f"methods {theirs['public_methods']}")
        if not row["same_instructions"]:
            raise SystemExit(
                f"a build with different instructions cannot stand in for the pin: {other}"
            )
    return rows


def _write_record(args: argparse.Namespace, extension: Path, *, mode: str) -> int:
    """Record provenance for a build that has already been shown to match the pin."""
    commit = _git("rev-parse", "HEAD")
    pinned_commit = pins.MR_BUILD.provenance["source_commit"]
    if commit != pinned_commit:
        raise SystemExit(
            f"the vendored tree is at {commit[:12]}, but the pin was produced at "
            f"{pinned_commit[:12]}; the receipt would name the wrong inputs"
        )
    record = args.record.resolve() if args.record else DEFAULT_RECORD
    payload = record_receipt(
        extension=extension,
        source_commit=commit,
        dirty=bool(_git("status", "--porcelain")),
        toolchain=check_toolchain(),
        compares=compare_builds(extension, _sha256(extension), args.compare),
        record=record,
        command=" ".join([Path(sys.executable).name, *sys.argv[1:]]),
        mode=mode,
    )
    print(f"record: pin matched={payload['pin']['matched']}, "
          f"{len(payload['comparisons'])} comparison(s) -> {record}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="scratch directory for sources and the built extension")
    parser.add_argument("--record", type=Path, nargs="?", const=DEFAULT_RECORD,
                        help="write the build receipt here (default: the benchmark record)")
    parser.add_argument("--compare", type=Path, action="append", default=[],
                        help="another build (or its lib directory) to hash and probe; repeatable")
    parser.add_argument("--check", action="store_true",
                        help="verify the existing build against the pin without compiling")
    args = parser.parse_args()

    out = args.out.resolve()
    lib_dir = out / "lib"

    if args.check:
        extension = _extension(lib_dir)
        build_row = pins.assert_allowed(extension, role="mr")
        print(f"{extension} is the pinned {build_row.name!r} build "
              f"({build_row.sha256[:16]}…)")
        if args.record:
            return _write_record(args, extension, mode="verified")
        return 0

    toolchain = check_toolchain()
    commit = _git("rev-parse", "HEAD")
    pinned_commit = pins.MR_BUILD.provenance["source_commit"]
    if commit != pinned_commit:
        raise SystemExit(
            f"the vendored tree is at {commit[:12]}, but the pin was produced at "
            f"{pinned_commit[:12]}; check out the pinned commit to reproduce the build"
        )
    dirty = bool(_git("status", "--porcelain"))

    source_root = out / "src"
    export_sources(source_root)
    apply_patch(source_root)
    if "OpType::MR" not in (source_root / "src" / "kickmix" / "sim" / "sim.h").read_text():
        raise SystemExit(f"patch did not add MR; check {PATCH}")
    extension = build(source_root, lib_dir, out / "tmp")

    digest = _sha256(extension)
    if digest != pins.MR_BUILD.sha256:
        raise SystemExit(
            f"built {digest}, but the pinned MR build is {pins.MR_BUILD.sha256}.\n"
            f"The bytes differ, so this extension may not be cited by a receipt. Either the "
            f"toolchain differs from {pins.MR_BUILD.provenance} or the recipe changed; \n"
            f"rebuild with the pinned toolchain, or update the pin and its receipt together."
        )
    print(f"built {extension} ({digest[:16]}…) — matches the pinned mr build")

    if args.record:
        return _write_record(args, extension, mode="rebuilt")
    compare_builds(extension, digest, args.compare)
    print(extension.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
