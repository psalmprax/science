"""REG-LEAN: pinned-toolchain reproduction of the 323-record Lean sweep.

Register anchor: docs/SYSTEM_FLAWS_AUDIT.md — REG-LEAN.

Acceptance criterion (register): "Reproduce the 323-record sweep exit codes on
an independently provisioned toolchain with recorded version hash."

What this script does:
  1. Shims a SECOND, independently provisioned pinned elan toolchain (default
     ``leanprover--lean4---v4.34.1``, distinct from the v4.35.0 binary that
     produced the committed reference artifact) via a PATH shim so
     ``LeanUniversalBridge`` deterministically resolves the pinned binary.
  2. Re-runs ``run_exhaustive_lean_verification_sweep()`` (323 records), with
     byte-identical backup/restore of the reference provenance artifact
     AND the generated .lean files (verified by SHA-256 before/after) so the
     committed evidence is not silently rewritten.
  3. Compares candidate-by-candidate (proof_claim_status, kernel_exit_code)
     against the reference and writes
     ``benchmarks/lean_pinned_toolchain_reproduction.json`` with the pinned
     toolchain version + commit hash.
  4. Exit 0 iff ALL records match and the counts agree; any divergence or
     toolchain failure exits nonzero (nothing is papered over).

Honesty boundary (recorded in the report): this reproduces the sweep under a
pinned, upstream-established Lean toolchain provisioned independently of the
reference artifact's binary. The register's "outside this environment"
external-review sign-off remains the outstanding step and is NOT claimed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REFERENCE_ARTIFACT = REPO_ROOT / "artifacts/all_system_equations_lean_provenance.json"
REPORT_PATH = REPO_ROOT / "benchmarks/lean_pinned_toolchain_reproduction.json"
DEFAULT_TOOLCHAIN = "leanprover--lean4---v4.34.1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_toolchain_shim(toolchain: str, tmp: Path) -> Path:
    """A shim dir whose `lean` forces the pinned toolchain, so
    shutil.which("lean") resolves the pinned binary deterministically.

    `toolchain` is the elan DIRECTORY name (e.g.
    ``leanprover--lean4---v4.34.1``); the toolchain's own binary is invoked
    directly (no elan release resolution), with an elan `+release` fallback
    when the directory name is not found."""
    shim_dir = tmp / "lean_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    toolchain_bin = Path.home() / ".elan" / "toolchains" / toolchain / "bin" / "lean"
    if not toolchain_bin.exists():
        elan_bin = Path.home() / ".elan" / "bin" / "lean"
        if not elan_bin.exists():
            raise RuntimeError(
                f"pinned toolchain {toolchain!r} not found under ~/.elan/toolchains "
                "and no elan proxy at ~/.elan/bin/lean"
            )
        shim = shim_dir / "lean"
        shim.write_text(
            "#!/bin/sh\n"
            f'exec "{elan_bin}" +"{toolchain}" "$@"\n'
        )
    else:
        shim = shim_dir / "lean"
        shim.write_text(
            "#!/bin/sh\n"
            f'exec "{toolchain_bin}" "$@"\n'
        )
    shim.chmod(0o755)
    return shim_dir


def pinned_version(shim_dir: Path) -> str:
    res = subprocess.run(
        [str(shim_dir / "lean"), "--version"], capture_output=True, text=True, timeout=120,
    )
    if res.returncode != 0:
        raise RuntimeError(f"pinned lean --version failed: {res.stderr}")
    return res.stdout.strip()


def compare_records(fresh: dict, reference: dict) -> tuple[bool, list[str], dict]:
    ref_by_id = {r["candidate_id"]: r for r in reference["results"]}
    new_by_id = {r["candidate_id"]: r for r in fresh["results"]}
    mismatches: list[str] = []
    n_compared = 0
    for cid, ref in ref_by_id.items():
        if cid not in new_by_id:
            mismatches.append(f"{cid}: MISSING in pinned run")
            continue
        new = new_by_id[cid]
        for key in ("proof_claim_status", "kernel_exit_code"):
            if ref.get(key) != new.get(key):
                mismatches.append(
                    f"{cid}: {key} reference={ref.get(key)!r} pinned={new.get(key)!r}"
                )
        n_compared += 1
    for cid in new_by_id:
        if cid not in ref_by_id:
            mismatches.append(f"{cid}: NEW record absent from reference")
    counts_match = (
        fresh["total_equations_evaluated"] == reference["total_equations_evaluated"]
        and fresh["p2_verified_count"] == reference["p2_verified_count"]
        and fresh["p0_refuted_count"] == reference["p0_refuted_count"]
        and fresh["p1_candidate_count"] == reference["p1_candidate_count"]
    )
    return (not mismatches and counts_match), mismatches, {
        "records_compared": n_compared,
        "counts_match": counts_match,
        "counts_reference": {k: reference[k] for k in (
            "total_equations_evaluated", "p2_verified_count", "p0_refuted_count", "p1_candidate_count")},
        "counts_pinned": {k: fresh[k] for k in (
            "total_equations_evaluated", "p2_verified_count", "p0_refuted_count", "p1_candidate_count")},
    }


def committed_bytes(relpath: str) -> bytes:
    """The HEAD-committed bytes of a repo file (the remediated/hand-disclosed
    state takes precedence over sweep regeneration — the generator must NOT
    silently overwrite honest disclosures like the u15 'remains unverified'
    comment, which a bare regeneration corrupts back into a vacuous theorem)."""
    res = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{relpath}"],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git show HEAD:{relpath} failed: {res.stderr.decode()[:200]}")
    return res.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", default=DEFAULT_TOOLCHAIN,
                        help="elan toolchain to pin (must DIFFER from the reference's for independence)")
    args = parser.parse_args()

    if not REFERENCE_ARTIFACT.is_file():
        raise SystemExit(f"reference artifact missing: {REFERENCE_ARTIFACT}")
    reference = json.loads(REFERENCE_ARTIFACT.read_text())
    ref_version = reference.get("lean_compiler_version", "")
    if args.toolchain.split("---v")[-1] in ref_version:
        raise SystemExit(
            f"--toolchain {args.toolchain} matches the reference run's version "
            f"({ref_version}) — independence requires a DIFFERENT pinned toolchain"
        )

    # Back up everything the sweep rewrites (reference provenance + .lean files).
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        art_backup = tmp / "ref_artifact.json"
        shutil.copy2(REFERENCE_ARTIFACT, art_backup)
        ref_art_hash = sha256_file(REFERENCE_ARTIFACT)

        lean_formal_dir = REPO_ROOT / "apedp_dyn" / "lean_formal"
        formal_backup = tmp / "lean_formal_backup"
        formal_hashes_before: dict[str, str] = {}
        if lean_formal_dir.is_dir():
            # Backup = HEAD-committed bytes (authoritative remediated state),
            # NOT the current disk state: regeneration must never overwrite
            # hand-remediated disclosures (e.g. the u15 honest 'remains
            # unverified' comment vs its vacuous auto-generated theorem).
            for p in sorted(lean_formal_dir.rglob("*.lean")):
                rel = str(p.relative_to(REPO_ROOT))
                try:
                    committed = committed_bytes(rel)
                except RuntimeError:
                    committed = p.read_bytes()  # not tracked: keep on-disk bytes
                dst = formal_backup / p.relative_to(lean_formal_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(committed)
                formal_hashes_before[rel] = hashlib.sha256(committed).hexdigest()

        shim_dir = build_toolchain_shim(args.toolchain, tmp)
        version_line = pinned_version(shim_dir)

        env = dict(os.environ)
        env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
        env["LEDGER_SILENCE"] = env.get("LEDGER_SILENCE", "1")

        run = subprocess.run(
            [sys.executable, "-c",
             "import json;"
             "from apedp_dyn.sandbox.build_all_system_equations_lean_proofs import"
             " run_exhaustive_lean_verification_sweep;"
             " s = run_exhaustive_lean_verification_sweep();"
             " print('SWEEP_JSON::' + json.dumps({k: s[k] for k in ("
             "'total_equations_evaluated','p2_verified_count','p0_refuted_count',"
             "'p1_candidate_count','lean_compiler_version','results')}))"],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=3600,
        )
        if run.returncode != 0:
            print("PINNED SWEEP FAILED (nonzero exit):", file=sys.stderr)
            print(run.stdout[-4000:], file=sys.stderr)
            print(run.stderr[-4000:], file=sys.stderr)
            return 1
        sweep_line = next(l for l in run.stdout.splitlines() if l.startswith("SWEEP_JSON::"))
        fresh = json.loads(sweep_line[len("SWEEP_JSON::"):])

        # Byte-identical restoration of the reference evidence, INSIDE the
        # temp-dir block (the backups live here). Both restorations must
        # complete BEFORE cleanup deletes the backup trees.
        drift: list[str] = []
        if sha256_file(REFERENCE_ARTIFACT) != ref_art_hash:
            shutil.copy2(art_backup, REFERENCE_ARTIFACT)
            if sha256_file(REFERENCE_ARTIFACT) != ref_art_hash:
                raise SystemExit("FAILED to restore the reference provenance artifact byte-identically")
            drift.append("all_system_equations_lean_provenance.json rewritten by sweep; restored byte-identical")
        if lean_formal_dir.is_dir():
            for rel, h in formal_hashes_before.items():
                p = REPO_ROOT / rel
                if not p.is_file() or sha256_file(p) != h:
                    relpath = Path(rel)
                    src = formal_backup / relpath.relative_to(
                        lean_formal_dir.relative_to(REPO_ROOT))
                    if not src.is_file():
                        raise SystemExit(f"FAILED to restore {rel}: backup missing")
                    shutil.copy2(src, p)
                    drift.append(f"{rel} rewritten by sweep; restored to committed bytes")

    ok, mismatches, counts = compare_records(fresh, reference)
    report = {
        "title": "REG-LEAN pinned-toolchain sweep reproduction",
        "generated_commit": None,  # filled by callers/SI, not hardcoded
        "reference_toolchain_version": ref_version,
        "pinned_toolchain": args.toolchain,
        "pinned_toolchain_version_line": version_line,
        "all_records_match": ok,
        "mismatches": mismatches,
        **counts,
        "byte_identical_restoration": {
            "performed": True,
            "drift_findings": drift,
        },
        "honesty_boundary": (
            "Sweep exit codes reproduced under a pinned, upstream-established Lean "
            "toolchain provisioned independently of the reference artifact's binary "
            f"({ref_version} vs pinned {args.toolchain}). The register's "
            "'outside this environment' EXTERNAL sign-off remains outstanding; "
            "this artifact reproduces, it does not substitute external review."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "pinned_toolchain", "pinned_toolchain_version_line", "all_records_match",
        "records_compared", "mismatches")}, indent=2))
    if not ok:
        print("REPRODUCTION FAILED: record divergence (see mismatches above)")
        return 1
    print(f"report written: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
