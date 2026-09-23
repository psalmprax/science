"""Hilbert-space ontology: states, observables, measurement, composition."""
import math

from apedp.hilbert_space import (
    Observable,
    QuantumMeasurement,
    QuantumOperator,
    QuantumState,
    QuantumSystem,
    compose_states,
    reduced_purity,
)


def test_state_normalization_gate():
    assert QuantumState([1.0, 0.0]).is_normalized()
    assert not QuantumState([2.0, 0.0]).is_normalized()
    assert QuantumState([1 / math.sqrt(2)] * 2).is_normalized()


def test_observable_requires_hermitian():
    pauli_z = QuantumOperator([[1, 0], [0, -1]], label="Z")
    assert pauli_z.is_hermitian()
    assert pauli_z.eigenvalues() == [complex(-1), complex(1)] or \
        sorted(pauli_z.eigenvalues(), key=abs) == [complex(-1), complex(1)]
    obs = Observable(pauli_z, name="Z", unit="dimensionless")
    assert obs.is_valid()
    plus = QuantumState([1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert abs(obs.expectation(plus) - 0.0) < 1e-9
    probs = obs.born_probabilities(plus)
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(abs(p - 0.5) < 1e-9 for p in probs)


def test_non_hermitian_rejected():
    bad = QuantumOperator([[0, 1], [0, 0]])
    assert not bad.is_hermitian()
    try:
        Observable(bad).born_probabilities(QuantumState([1.0, 0.0]))
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_measurement_samples_born_rule():
    pauli_z = QuantumOperator([[1, 0], [0, -1]])
    up = QuantumState([1.0, 0.0])
    out = QuantumMeasurement(Observable(pauli_z), up, seed=7).sample(200)
    # eigenorder ascending: index 0 is eigenvalue -1 (prob 0), index 1 is +1
    assert out["counts"] == [0, 200], out["counts"]
    assert out["n"] == 200


def test_composition_and_entanglement():
    bell = QuantumState([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                        label="bell")
    assert abs(reduced_purity(bell, 2) - 0.5) < 1e-9
    prod = compose_states(QuantumState([1.0, 0.0], label="u"),
                          QuantumState([1.0, 0.0], label="u"))
    assert prod.dim == 4
    assert abs(reduced_purity(prod, 2) - 1.0) < 1e-9


def test_truncated_ladder_error_localized_at_cutoff():
    from apedp.hilbert_space import truncated_ladder
    for dim in (8, 32):
        t = truncated_ladder(dim)
        assert t["bulk_defect"] < 1e-9, (dim, t["bulk_defect"])
        assert t["commutator_defect"] > 0.0, dim


def test_lindblad_decay_and_trace():
    import numpy as np

    from apedp.hilbert_space import lindblad_step
    excited = np.array([[0, 0], [0, 1]], dtype=complex)
    h = np.array([[0, 0], [0, 1.0]])
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    rho = excited
    for _ in range(200):
        rho = lindblad_step(rho, h, [sigma_minus], 0.05,
                            [1.0])["rho"]
    assert abs(rho[1, 1]) < 0.05
    assert abs(float(np.real(np.trace(rho))) - 1.0) < 1e-9


def test_grw_type_decoherence_scales_with_rate():
    """GRW mathematics (not interpretation): localization Lindbladian
    decoheres monotonically in the collapse-rate parameter."""
    import numpy as np

    from apedp.hilbert_space import lindblad_step
    def coherence_after(lam):
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        h = np.zeros((2, 2))
        l0 = np.array([[1, 0], [0, 0]], dtype=complex)
        l1 = np.array([[0, 0], [0, 1]], dtype=complex)
        for _ in range(100):
            rho = lindblad_step(rho, h, [l0, l1], 0.01, [lam, lam])["rho"]
        return abs(rho[0, 1])
    c1, c2, c3 = (coherence_after(lam) for lam in (0.1, 0.5, 1.0))
    assert c1 > c2 > c3, (c1, c2, c3)


def test_vacuum_energy_diverges_with_cutoff():
    """The UV problem itself, numerically: zero-point sum grows ~N^2."""
    import numpy as np
    energies = [sum(0.5 * n * np.pi for n in range(1, N + 1))
                for N in (10, 100, 1000)]
    ratios = [e / N ** 2 for e, N in zip(energies, (10, 100, 1000))]
    assert all(abs(r - 0.785) < 0.1 for r in ratios), ratios
    assert energies[0] < energies[1] < energies[2]


def test_system_governance():
    sys = QuantumSystem()
    sys.add_state("up", QuantumState([1.0, 0.0]))
    sys.add_observable("Z", Observable(QuantumOperator([[1, 0], [0, -1]])))
    try:
        sys.add_state("bad", QuantumState([2.0, 0.0]))
        raise AssertionError("should have raised")
    except ValueError:
        pass
