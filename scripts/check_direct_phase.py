#!/usr/bin/env python3
"""Native regression checks for the experimental direct phase prototype."""
from fractions import Fraction
import unittest

import kickmix as km

from prototype_direct_phase import build_direct, phase_words


class DirectPhaseTests(unittest.TestCase):
    def test_exact_phase_input_restoration_and_clean_scratch(self):
        words = phase_words(15, 40)
        circuit = build_direct(words, 40)
        self.assertLessEqual(circuit.num_qubits, 50)
        inputs = list(range(17)) + [32, 4096, 1 << 21, (1 << 22) - 1]
        simulator = km.Simulator(batch_size=256)
        simulator.use_same_registers_as(circuit)
        scratch = km.array([km.q(i) for i in range(22, circuit.num_qubits)])
        for shot in range(256):
            simulator.write_within_shot("x", shot, inputs[shot % len(inputs)])
        simulator.do(circuit)
        for shot in range(256):
            x = inputs[shot % len(inputs)]
            expected = Fraction(2 * words[x], 1 << 40) if x < len(words) else 0
            self.assertEqual(simulator.read_within_shot("x", shot, out=int), x)
            self.assertEqual(simulator.read_within_shot(scratch, shot, out=int), 0)
            self.assertEqual(simulator.read_shot_phase(shot) % 2, expected % 2)

    def test_zero_phase_table(self):
        circuit = build_direct([0, 0], 40)
        self.assertEqual(int(circuit.operation_statistics().num_z_pow()), 0)

    def test_invalid_table_and_precision(self):
        for words, precision in (([], 40), ([0, 1], 10), ([0, -1], 40)):
            with self.assertRaises(ValueError):
                build_direct(words, precision)


if __name__ == "__main__":
    unittest.main()
