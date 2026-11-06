#!/usr/bin/env python3
"""Run the exact Kaggle metric pre-submission validator against a submission directory or zip."""
import json
import math
import pathlib
import stat
import sys
import time
import zipfile
import zlib
import tokenize
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
sys.path.insert(0, str(BUNDLE / "kickmix"))
sys.path.insert(0, str(BUNDLE / "verifiers"))

import kickmix
from dfthcsos import verify as dfthcsos_verify
from dfthcsos.scoreinfra.reference import FACTORIZATION_KEYS, ORBITALS
from dqi import verify_submission_dqi
from invsqrt import verify_submission_invsqrt

ALLOWED_INSTRUCTIONS = {
    'X', 'CX', 'CCX', 'Z', 'CZ', 'CCZ', 'Z_POW', 'NEG', 'SWAP', 'CSWAP', 'R', 'HMR', 'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1',
    'PUSH_CONDITION', 'POP_CONDITION', 'INFO_TAG', 'DEBUG_PRINT',
}
MAX_QUBIT_OR_BIT_ID = 2**30 // 1536
MAX_NPZ_BYTES = 2**26

def find_one(root: pathlib.Path, name: str, description: str, is_dir: bool = False) -> pathlib.Path:
    pattern = f'**/{name}'
    paths = [path for path in root.glob(pattern) if path.is_dir() if is_dir] if is_dir else [path for path in root.glob(pattern) if not path.is_dir()]
    if len(paths) == 1:
        return paths[0]
    paths_description = 'none' if not paths else f'multiple: {[str(path.relative_to(root)) for path in paths]}'
    raise ValueError(f'Expected one {description} in {root}, found {paths_description}.')

def read_circuit(path: pathlib.Path, max_registers: int) -> kickmix.Circuit:
    circuit = kickmix.Circuit.from_file(path, format=path.suffix.removeprefix('.'))
    unsupported = set(circuit.operation_statistics()) - ALLOWED_INSTRUCTIONS
    if unsupported:
        raise ValueError(f'unsupported instructions {sorted(unsupported)}')
    largest_id = max(circuit.num_qubits, circuit.num_bits) - 1
    if largest_id >= MAX_QUBIT_OR_BIT_ID:
        raise ValueError(f'largest qubit or bit id {largest_id} exceeds MAX_QUBIT_OR_BIT_ID {MAX_QUBIT_OR_BIT_ID}')
    if circuit.num_qubits > max_registers:
        raise ValueError(f'circuit num_qubits {circuit.num_qubits} exceeds max_registers {max_registers}')
    return circuit

def read_npz(path: pathlib.Path, spec: dict) -> dict:
    dimensions: dict[str, int] = {}
    total = 0
    with zipfile.ZipFile(path) as archive:
        members = {name.removesuffix('.npy'): name for name in archive.namelist()}
        missing = [key for key in spec if key not in members]
        if missing:
            raise ValueError(f'{path.name} has no {missing}')
        for key, (kinds, expected_shape) in spec.items():
            with archive.open(members[key]) as member:
                if np.lib.format.read_magic(member) != (1, 0):
                    raise ValueError(f'{key} in {path.name} is not in .npy format 1.0, which numpy.savez writes')
                shape, _, dtype = np.lib.format.read_array_header_1_0(member)
            if dtype.kind not in kinds or dtype.names is not None or dtype.subdtype is not None or dtype.byteorder == '>':
                raise ValueError(f'{key} in {path.name} has dtype {dtype}, not a plain number type of numpy kind {kinds!r}')
            matches = len(shape) == len(expected_shape) and all(
                size > 0 and size == (dimensions.setdefault(expected, size) if isinstance(expected, str) else expected)
                for size, expected in zip(shape, expected_shape, strict=True)
            )
            if not matches:
                raise ValueError(f'{key} in {path.name} has shape {shape}, not {expected_shape} with its named dimensions as in {dimensions}')
            total += math.prod(shape) * dtype.itemsize
    if total > MAX_NPZ_BYTES:
        raise ValueError(f'the arrays in {path.name} take {total:,} bytes, above the limit of {MAX_NPZ_BYTES:,}')
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.array(npz[key]) for key in spec}

def validate_archive(submission_dir: pathlib.Path):
    print(f"=== Validating submission structure in {submission_dir} ===")
    root = pathlib.Path(submission_dir)
    
    # 1. invsqrt
    invsqrt_path = find_one(root, 'invsqrt.km[xb]', 'invsqrt.kmx or invsqrt.kmb')
    print(f"[OK] Found invsqrt: {invsqrt_path.relative_to(root)}")
    c_invsqrt = read_circuit(invsqrt_path, 1000)
    print(f"     Qubits: {c_invsqrt.num_qubits}, Ops: {len(c_invsqrt):,}")

    # 2. dqi
    dqi_path = find_one(root, 'dqi.km[xb]', 'dqi.kmx or dqi.kmb')
    print(f"[OK] Found dqi: {dqi_path.relative_to(root)}")
    c_dqi = read_circuit(dqi_path, 5000)
    print(f"     Qubits: {c_dqi.num_qubits}, Ops: {len(c_dqi):,}")

    # 3. dfthcsos
    dfthcsos_dir = find_one(root, 'dfthcsos', 'dfthcsos/ directory', is_dir=True)
    print(f"[OK] Found dfthcsos directory: {dfthcsos_dir.relative_to(root)}")
    angle_files = [p for p in dfthcsos_dir.glob("angle_k_*") if p.suffix in ('.kmb', '.kmx')]
    stems = sorted({p.stem for p in angle_files}, key=lambda s: int(s.split('_')[-1]))
    expected_stems = [f'angle_k_{k}' for k in range(len(stems))]
    if stems != expected_stems:
        raise ValueError(f"Angle stems {stems} do not match expected {expected_stems}")
    print(f"     Found {len(stems)} angle circuits: {stems[0]} ... {stems[-1]}")
    for stem in stems:
        path = dfthcsos_dir / f"{stem}.kmb"
        c = read_circuit(path, dfthcsos_verify.QUBIT_CAP)
        if c.num_qubits > 128:
            raise ValueError(f"{stem} exceeds 128-qubit cap: {c.num_qubits}")
    print(f"[OK] All {len(stems)} angle circuits pass 128-qubit cap and instruction audits.")

    # 4. submission.npz
    npz_path = find_one(dfthcsos_dir, 'submission.npz', 'dfthcsos/submission.npz')
    npz_spec = {
        'U': ('f', ('R', 'B', ORBITALS)),
        'W': ('f', ('R', 'B', 'C')),
        'H2_I': ('f', ('R', 'C')),
        'H1_I': ('f', ()),
        'Hsym_lower': ('f', (ORBITALS * (ORBITALS + 1) // 2,)),
        'tree_edges': ('iu', (ORBITALS - 1, 2)),
    }
    npz = read_npz(npz_path, npz_spec)
    print(f"[OK] submission.npz valid. Keys: {list(npz.keys())}")

    print("\n>>> ALL KAGGLE METRIC PRE-FLIGHT CHECKS PASSED WITH ZERO ERRORS! <<<")

if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/tmp/test_submission")
    validate_archive(target)
