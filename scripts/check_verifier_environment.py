#!/usr/bin/env python3
"""Fail-fast native verifier preflight without importing the scientific stack."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation"


def _load_pins():
    spec = importlib.util.spec_from_file_location(
        "kickmix_preflight_pins", ROOT / "apedp_dyn/quantum_compilation/kickmix_builds.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_environment(executable: str, extension: Path) -> dict:
    module = _load_pins()
    if not extension.is_file():
        return {"ready": False, "reason": f"Missing approved extension: {extension}"}
    digest = hashlib.sha256(extension.read_bytes()).hexdigest()
    build = module.identify(extension)
    if build is None or not build.allowed_for_verifier:
        return {
            "ready": False, "reason": "Extension is not an approved verifier build",
            "extension_sha256": digest,
        }
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy, scipy, kickmix
print(json.dumps({
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "extension": kickmix.__file__,
    "has_circuit_builder": hasattr(kickmix, "CircuitBuilder"),
}))
"""
    try:
        result = subprocess.run(
            [executable, "-c", script, str(extension.resolve().parent)],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "ready": False, "reason": str(error),
            "stderr": error.stderr if isinstance(error, subprocess.CalledProcessError) else "",
            "extension_sha256": digest,
        }
    runtime = json.loads(result.stdout.strip().splitlines()[-1])
    imported_path = Path(runtime["extension"]).resolve()
    imported_digest = hashlib.sha256(imported_path.read_bytes()).hexdigest()
    ready = (
        runtime["has_circuit_builder"] and runtime["python"].startswith("3.13.")
        and imported_digest == digest
    )
    return {
        "ready": ready, "build": build.name, "extension_sha256": digest,
        "runtime": runtime,
        "reason": "ready" if ready else "Interpreter, imported extension, or API mismatch",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--extension", type=Path, default=BUNDLE / "kickmix/kickmix.so")
    args = parser.parse_args(argv)
    report = check_environment(args.python_executable, args.extension)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
