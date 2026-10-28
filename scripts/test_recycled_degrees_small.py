#!/usr/bin/env python3
"""Test degree-register recycling on small DQI benchmark (m=15, n=6, degree=4)."""
from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi"))

import kickmix as km
from verify_submission_dqi import verify, ProblemSpecDQI
import dqi_reference_circuit as DR

def test_reference_small():
    gf = km.GF2Field(4)
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(4, name=f'S{i}') for i in range(6)]
    err = [b.create_classical_register(4, name=f'E{j}') for j in range(15)]
    dec = DR.ProosAndZalkaDecoder(6, gf)
    dec.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err, n_iters=30)
    c = b.finish_circuit()
    spec = ProblemSpecDQI(m=15, n=6, degree=4, r=6, num_shots=256)
    res = verify(c, spec, max_workers=1, batch_size=256, log2_delta=-10.0)
    print("Reference small DQI circuit:")
    print(f"  Status:       {res['status']}")
    print(f"  Num Qubits:   {res['num_qubits']}")
    print(f"  Toffoli mean: {res['toffoli_count']}")
    print(f"  Score:        {res['score']}")
    return res

if __name__ == "__main__":
    test_reference_small()
