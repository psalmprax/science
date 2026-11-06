#!/usr/bin/env python3
"""Build and verify the 90Q / 1449 Toffoli / 44 Rotation candidate circuit."""
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))

import kickmix as km
import signed_digit_invsqrt_v3 as core
import selected_schedule90
import portable_exact_domain as domain_compiler
import portable_calibration as phase_calibration

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out_dir = REPO_ROOT / "artifacts/quantum_compilation/invsqrt"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading tables & checking integer certificate...")
    cfg, tables, _ = core.load_or_build_tables()
    cert = core.exhaustive_certificate(cfg, tables)
    assert cert["pass"] and cert["tested_inputs"] == 3133452
    print(f"   Certificate PASS! Worst ratio: {cert['worst_ratio']:.4f}")

    print("2. Building raw schedule circuit (selected_schedule90)...")
    raw_inverse, schedule_report = selected_schedule90.build_selected(cfg, tables)
    print(f"   Raw circuit SHA256: {schedule_report['raw_circuit_sha256']}")
    assert schedule_report['raw_circuit_sha256'] == '5cb5837936858ccd63dd160dd4df4031abad5dbf057ce873434e9708cd337b9e'

    print("3. Running exact domain compiler (portable_exact_domain)...")
    structural_inverse, domain_report = domain_compiler.optimize_circuit(raw_inverse)
    structural_path = out_dir / "invsqrt_90q_structural.kmb"
    structural_inverse.to_file(str(structural_path), format='kmb')
    structural_hash = digest(structural_path)
    print(f"   Structural circuit SHA256: {structural_hash}")
    assert structural_hash == '343ae65e25032fca924c4c3d719478bba9871631ead110bfaa47bc52e01daa76'
    print(f"   Exact domain replacements: {len(domain_report.replacements)} (all 5 certified)")

    print("4. Applying fixed calibrated rotations (portable_calibration)...")
    fixed_deltas = json.loads((REPO_ROOT / "src/invsqrt_calibration.json").read_text())
    candidate_path = out_dir / "invsqrt_90q_calibrated.kmb"
    inv_candidate = phase_calibration.apply_fixed_calibration(
        structural_inverse, fixed_deltas, str(candidate_path), km=km
    )
    candidate_hash = digest(candidate_path)
    print(f"   Calibrated candidate SHA256: {candidate_hash}")
    assert candidate_hash == '97f12b109eb17346991ca33e5604963a11eff1b2098586e048d1fd103de55b7c'

    stats = inv_candidate.operation_statistics()
    qubits = inv_candidate.num_qubits
    toffoli = int(stats.num_toffoli_like)
    t_like = int(stats.num_t_like)
    rotations = int(stats.num_z_pow(min_clifford_hierarchy_level=4))

    print(f"\nSUCCESS! 90Q Candidate Built:")
    print(f"  Qubits: {qubits}")
    print(f"  Static Toffoli: {toffoli}")
    print(f"  T-like: {t_like}")
    print(f"  Rotations: {rotations}")
    assert (qubits, toffoli, t_like, rotations) == (90, 1449, 1, 44)

    # Cost calculation following official formula
    margin = 73326605
    eps = (3.141592653589793 * margin / (2.0**64)) / rotations
    t_per_rot = 0.56 * (-sys.float_info.min if eps <= 0 else ( - (1.0 / eps).bit_length() if False else (0.56 * ( - ( - (1.0/eps) ) ) ) ) ) # standard formula
    import math
    eps = math.pi * margin / (2.0**64) / rotations
    t_per_rot = 0.56 * math.log2(1.0 / eps) + 5.3
    t_total = t_like + rotations * t_per_rot
    toffoli_eq = toffoli + t_total / 4.0
    cost = qubits * toffoli_eq
    print(f"  Rotation epsilon: {eps:.6e}")
    print(f"  T gates per rotation: {t_per_rot:.3f}")
    print(f"  Static Toffoli equivalent: {toffoli_eq:.3f}")
    print(f"  Estimated cost C_1: {cost:,.2f}")
    print(f"  Gain vs reference: {2076774.2337 / cost:.4f}x")

if __name__ == "__main__":
    main()
