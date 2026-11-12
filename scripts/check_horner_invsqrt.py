#!/usr/bin/env python3
"""Independent checks for integer Horner rounding and reversibility limits."""
from fractions import Fraction
import unittest

import numpy as np

from audit_horner_invsqrt import coefficient_table, evaluate_integer, round_shift
from audit_normalized_invsqrt import FULL_MAX_X


def scalar_phase(x: int, table: np.ndarray, bits: int) -> int:
    exponent = x.bit_length() - 1
    segments = table.shape[1]
    scaled = x << (bits - exponent)
    index = min(((scaled - 2**bits) * segments) >> bits, segments - 1)
    local = 2 * (scaled - 2**bits) * segments - (2 * index + 1) * 2**bits
    coefficients = table[exponent & 1, index]
    state = int(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        state = round(Fraction(local * state, 2**bits)) + int(coefficient)
    return state << (48 - bits - exponent // 2)


class HornerModelTests(unittest.TestCase):
    def test_signed_rounding_matches_exact_fractions(self):
        values = np.arange(-1024, 1025, dtype=np.int64)
        for bits in range(8):
            expected = [round(Fraction(int(value), 2**bits)) for value in values]
            np.testing.assert_array_equal(round_shift(values, bits), expected)

    def test_vector_recurrence_matches_independent_integer_scalar(self):
        inputs = {1, FULL_MAX_X}
        for exponent in range(22):
            for quarter in range(4, 9):
                center = quarter * (1 << exponent) // 4
                inputs.update(x for x in (center - 1, center, center + 1)
                              if 1 <= x <= FULL_MAX_X)
        xs = np.array(sorted(inputs), dtype=np.int64)
        for bits in (25, 27, 28):
            table = coefficient_table(5, 4, bits)
            actual, maximum = evaluate_integer(xs, table, bits)
            expected = [scalar_phase(int(x), table, bits) for x in xs]
            np.testing.assert_array_equal(actual, expected)
            self.assertGreater(maximum, 0)

    def test_rounded_overwrite_has_collisions(self):
        # At t=1/2, distinct old states 0 and 1 both round to 0.
        # An in-place quantum update must retain enough information to distinguish them.
        old_states = np.array([0, 1], dtype=np.int64)
        new_states = round_shift(old_states, 1) + 7
        np.testing.assert_array_equal(new_states, [7, 7])

    def test_rejects_invalid_precision_and_domain(self):
        with self.assertRaises(ValueError):
            coefficient_table(5, 4, 30)
        table = coefficient_table(5, 4, 27)
        for xs in (np.array([0]), np.array([1.5]), np.array([FULL_MAX_X + 1])):
            with self.assertRaises(ValueError):
                evaluate_integer(xs, table, 27)


if __name__ == "__main__":
    unittest.main()
