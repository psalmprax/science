#!/usr/bin/env python3
"""Numerical and native checks for the experimental parity phase oracle."""
from fractions import Fraction
import unittest

import numpy as np

from prototype_parity_phase import audit_terms, build_parity_circuit, sparse_terms, walsh_transform


class ParityPhaseTests(unittest.TestCase):
    def test_transform_matches_explicit_parities(self):
        values = np.array([0.3, 0.1, 0.7, -0.2, 0.4, 0.6, -0.8, 0.2])
        explicit = np.array([
            sum(value * (-1) ** ((mask & x).bit_count()) for x, value in enumerate(values))
            for mask in range(values.size)
        ])
        np.testing.assert_allclose(walsh_transform(values), explicit, atol=1e-14)
        np.testing.assert_allclose(
            walsh_transform(walsh_transform(values)) / values.size, values, atol=1e-14
        )

    def test_all_terms_reconstruct_and_sparse_errors_are_reported(self):
        reference = 1 / np.sqrt(np.maximum(1, np.arange(16, dtype=float)))
        coefficients = walsh_transform(reference) / reference.size
        indices, weights = sparse_terms(coefficients, coefficients.size)
        self.assertTrue(audit_terms(indices, weights, reference, 15)["passes_model_tolerance"])
        indices, weights = sparse_terms(coefficients, 1)
        self.assertFalse(audit_terms(indices, weights, reference, 15)["passes_model_tolerance"])

    def test_native_phase_matches_selected_model_and_restores_qubits(self):
        import kickmix as km
        indices = np.array([0, 1, 3, 6])
        weights = np.array([0.125, 0.0625, -0.03125, 0.015625])
        circuit = build_parity_circuit(indices, weights, 3)
        self.assertEqual(int(circuit.operation_statistics().num_toffoli_like), 0)
        self.assertLessEqual(circuit.num_qubits, 4)
        simulator = km.Simulator(batch_size=256)
        simulator.use_same_registers_as(circuit)
        for shot in range(256):
            simulator.write_within_shot("x", shot, shot % 8)
        simulator.do(circuit)
        for shot in range(256):
            x = shot % 8
            phase = sum(
                (Fraction(float(w)) * (-1) ** ((int(mask) & x).bit_count())
                 for mask, w in zip(indices, weights, strict=True)),
                Fraction(),
            )
            self.assertEqual(simulator.read_shot_phase(shot) % 2, (2 * phase) % 2)
            self.assertEqual(simulator.read_within_shot("x", shot, out=int), x)
            self.assertEqual(simulator.read_within_shot(km.q(3), shot, out=int), 0)

    def test_invalid_transform_and_term_count(self):
        with self.assertRaises(ValueError):
            walsh_transform(np.ones(3))
        with self.assertRaises(ValueError):
            sparse_terms(np.ones(8), 0)


if __name__ == "__main__":
    unittest.main()
