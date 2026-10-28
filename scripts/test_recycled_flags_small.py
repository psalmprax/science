#!/dev/shm/qvcc-venv313/bin/python
"""Prototype Recycled-Degree DQI Decoder on small instance (m=15, n=6, degree=4)."""
from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/dqi"))

import kickmix as km
from verify_submission_dqi import verify, ProblemSpecDQI
import dqi_reference_circuit as DR

# We test replacing _update_flag and _update_halting_counter first
class RecycledFlagDecoder(DR.ProosAndZalkaDecoder):
    def _update_flag(self, builder: km.CircuitBuilder, reg_b,
                     n_A: km.array, n_B: km.array, n_a: km.array, n_b: km.array,
                     n_q: km.array, flag, ctrl) -> None:
        """Toggles `flag` using derived degree invariants without reading n_a or n_b."""
        with builder.mark('update_flag_recycled'):
            c0, c1, c2, c3 = ctrl
            m = len(reg_b)
            n = self.n_syndromes
            aux = builder.alloc_qubits(1)[0]

            # Sub-step 0: unchanged
            builder.flip_if_less_than(n_A, n_B, target=aux, control=c0)
            builder.flip_if_equal(n_q, 0, target=flag, control=aux)
            builder.flip_if_less_than(n_A, n_B, target=aux, control=c0)
            builder.flip_if_equal(n_A, n_B, target=flag, control=c0)

            # Sub-step 1: unchanged
            builder.flip_if_equal(n_A, n_B, target=flag, control=c1)
            builder.cx(c1, flag)
            builder.flip_if_equal(n_B, 1, target=aux, control=c1)
            builder.cx(c1, aux)
            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)
            builder.cx(c1, aux)
            builder.flip_if_equal(n_B, 1, target=aux, control=c1)

            # Sub-step 2: RECYCLED
            builder.cx(c2, flag)
            # n_b == 0 <=> n_B == n + 1
            builder.flip_if_equal(n_B, n + 1, target=aux, control=c2)
            builder.cx(c2, aux)
            builder.flip_if_equal(reg_b[0], 0, target=flag, control=aux)
            builder.cx(c2, aux)
            builder.flip_if_equal(n_B, n + 1, target=aux, control=c2)
            # n_a == n_b + 1 <=> n_B == n_A + 2
            builder.iadd(2, target=n_A)
            builder.flip_if_equal(n_A, n_B, target=flag, control=c2)
            builder.isub(2, target=n_A)

            # Sub-step 3: RECYCLED
            # n_a == n_b <=> n_B == n_A + 1
            builder.iadd(1, target=n_A)
            builder.flip_if_equal(n_A, n_B, target=flag, control=c3)
            builder.isub(1, target=n_A)
            builder.flip_if_equal(n_q, 1, target=flag, control=c3)

            builder.free(aux)

    def _update_halting_counter(self, builder: km.CircuitBuilder, n_A: km.array,
                                n_B: km.array, n_b: km.array,
                                halting_counter: km.array, flag, is_c3, *,
                                is_adjoint: bool) -> None:
        """Increments halting_counter using derived condition 2*n_B > n + 1 without reading n_b."""
        with builder.mark('update_halting_counter_recycled'):
            work = builder.alloc_qubits(3)
            ctrl, busy, running = work
            n = self.n_syndromes
            builder.init_and(is_c3, flag, ctrl)
            # n_b < n_B <=> 2*n_B > n + 1 <=> n_B >= (n + 2) // 2
            threshold = (n + 2) // 2
            builder.flip_if_greater_than(n_B, threshold - 1, target=busy, control=ctrl)
            builder.cx(busy, running)
            builder.flip_if_equal(n_A, 0, target=running, control=busy)

            builder.cx(running, ctrl)
            if is_adjoint:
                builder.isub(1, target=halting_counter, control=ctrl)
            else:
                builder.iadd(1, target=halting_counter, control=ctrl)
            builder.cx(running, ctrl)

            builder.flip_if_equal(n_A, 0, target=running, control=busy)
            builder.cx(busy, running)
            builder.flip_if_greater_than(n_B, threshold - 1, target=busy, control=ctrl)
            builder.del_and(is_c3, flag, ctrl)
            builder.free(work)


def run_test():
    gf = km.GF2Field(4)
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(4, name=f'S{i}') for i in range(6)]
    err = [b.create_classical_register(4, name=f'E{j}') for j in range(15)]
    dec = RecycledFlagDecoder(6, gf)
    dec.build_syndrome_decoding_mbuc(b, syndromes=syn, error_mbuc=err, n_iters=30)
    c = b.finish_circuit()
    spec = ProblemSpecDQI(m=15, n=6, degree=4, r=6, num_shots=256)
    res = verify(c, spec, max_workers=1, batch_size=256, log2_delta=-10.0)
    print("Recycled Flag & Halting Counter small DQI circuit:")
    print(f"  Status:       {res['status']}")
    print(f"  Num Qubits:   {res['num_qubits']}")
    print(f"  Toffoli mean: {res['toffoli_count']}")
    print(f"  Score:        {res['score']}")
    return res

if __name__ == "__main__":
    run_test()
