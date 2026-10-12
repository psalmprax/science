"""Provenance runner: execute a command and bind its identity.

Records script hashes, dataset hashes, git commit, env versions, full
command line, timestamp, and stdout/stderr into one directory. Tables must
be generated from results artifacts, never transcribed.

Usage:
  python3 provenance/make_record.py <record-dir> --data f1 f2 -- <command...>
"""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    args = sys.argv[1:]
    try:
        sep = args.index("--")
    except ValueError:
        print("usage: make_record.py <record-dir> --data f1 f2 -- <command...>")
        raise SystemExit(2)
    recdir = Path(args[0])
    rest = args[1:sep]
    cmd = args[sep + 1:]
    data_files: list[str] = []
    if rest and rest[0] == "--data":
        data_files = rest[1:]
    recdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (recdir / "stdout.log").write_text(proc.stdout)
    (recdir / "stderr.log").write_text(proc.stderr)
    pyfiles = sorted({a for a in cmd if a.endswith(".py")})
    manifest = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": cmd,
        "returncode": proc.returncode,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "script_hashes": {p: sha(p) for p in pyfiles},
        "dataset_hashes": {p: sha(p) for p in data_files},
        "note": "Tables must be generated from stdout.log/results artifacts in this directory.",
    }
    (recdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"record: {recdir} returncode={proc.returncode}")


if __name__ == "__main__":
    main()
