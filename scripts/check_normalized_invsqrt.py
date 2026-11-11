#!/usr/bin/env python3
"""Regression checks for the normalized numerical approximation audit."""
import unittest

import numpy as np

from audit_normalized_invsqrt import FULL_MAX_X, evaluate, fit_segments


class NormalizedApproximationTests(unittest.TestCase):
    def test_dyadic_and_segment_boundaries(self):
        xs = {1, FULL_MAX_X}
        for exponent in range(22):
            for numerator in range(4, 9):
                boundary = numerator * (1 << exponent) // 4
                xs.update(x for x in (boundary - 1, boundary, boundary + 1)
                          if 1 <= x <= FULL_MAX_X)
        inputs = np.array(sorted(xs), dtype=float)
        target = 1 / np.sqrt(inputs)
        for segments, degree in ((1, 9), (2, 7), (4, 5), (8, 5)):
            approximation = evaluate(inputs, fit_segments(degree, segments), 28)
            self.assertTrue(np.all(np.abs(approximation - target) < 2**-24 * target))

    def test_rounded_arithmetic_changes_predictions(self):
        inputs = np.array([3, 17, 173477, FULL_MAX_X], dtype=float)
        coefficients = fit_segments(7, 2)
        exact = evaluate(inputs, coefficients)
        rounded = evaluate(inputs, coefficients, 24)
        self.assertTrue(np.any(exact != rounded))
        self.assertTrue(np.all(rounded * 2**48 == np.rint(rounded * 2**48)))

    def test_invalid_parameters_and_inputs(self):
        for degree, segments in ((0, 1), (3, 0), (3, 3)):
            with self.assertRaises(ValueError):
                fit_segments(degree, segments)
        coefficients = fit_segments(5, 4)
        for value in (0, 1.5, FULL_MAX_X + 1):
            with self.assertRaises(ValueError):
                evaluate(np.array([value]), coefficients)


if __name__ == "__main__":
    unittest.main()
