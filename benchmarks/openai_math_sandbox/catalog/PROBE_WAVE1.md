# Probe wave 1 — openai/math families through the system

Illustration only: finite probes can **never** verify the manuscripts' claims, and no verdict here is `verified` (vocabulary excludes it by construction).

Probed: {'consistent_finite_range': 2, 'no_refutation': 1, 'not_probeable': 1}

## Family 3 — The quasi-Riemann hypothesis

- **Verdict:** `consistent_finite_range`
- Engine: rust/riemann_verifier (Euler-Maclaurin, from scratch)
- Domain: zeta zeros on the critical line; off-line grid scan t in [10, 40], sigma in {0.2..0.8} (hardcoded window)
- Observed: `{"primary_T60": {"max_t": 60.0, "roots_isolated": 13, "critical_line_recovered": 13, "approximate_tally": 13, "calibration_gate": "PASS", "max_reference_delta": 1.15e-09, "off_line_grid_scan": "OFF_LINE_GRID_SCAN_PASS", "coverage_pct": 100.0}, "coverage_sweep": [{"max_t": 100.0, "roots_isolated": 29, "critical_line_recovered": 29, "approximate_tally": 29, "calibration_gate": "PASS", "max_reference_delta": 1.15e-09, "off_line_grid_scan": "OFF_LINE_GRID_SCAN_PASS", "coverage_pct": 100.0}, {"max_t": 150.0, "roots_isolated": 52, "critical_line_recovered": 52, "approximate_tally": 53, "calibration_gate": "PASS", "max_reference_delta": 1.15e-09, "off_line_grid_scan": "OFF_LINE_GRID_SCAN_PASS", "coverage_pct": 98.1}, {"max_t": 200.0, "roots_isolated": 77, "critical_line_recovered": 77, "approximate_tally": 79, "calibration_gate": "PASS", "max_reference_delta": 1.15e-09, "off_line_grid_scan": "OFF_LINE_GRID_SCAN_PASS", "coverage_pct": 97.5}, {"max_t": 250.0, "roots_isolated": 97, "critical_line_recovered": 97, "approximate_tally": 108, "calibration_gate": "PASS", "max_reference_delta": 1.15e-09, "off_line_grid_scan": "OFF_LINE_GRID_SCAN_PASS", "coverage_pct": 89.8}, {"max_t": 300.0, "roots_isolated": 118, "critical_line_recovered": 118, "approximate_tally": 138, "calibration_gate": "PASS", "max_reference_delta": 1.15e-09, "off_line_grid_scan": "OFF_LINE_GRID_SCAN_PASS", "coverage_pct": 85.5}]}`
- Can say: at T=60 all 13 roots are isolated on the critical line, calibrated to 1.15e-9, and no off-line zero appears in the scanned window
- Cannot say: quasi-RH is about every Dirichlet L-function and all heights; the engine checks zeta(s) only, discovery misses zeros at higher T, and the off-line window is [10, 40] only
- **Engine limitation found:** Root-discovery completeness degrades with height at fixed cost: the scan step is hardcoded at 0.05 and the miss rate grows from 0% (T<=100) to 14.5% (T=300, 118/138 isolated). The CLI sample-count argument does not affect discovery, so the deficit is inherent to the engine, not to the invocation.
- **Engine limitation found:** The OFF_LINE_GRID_SCAN_PASS verdict is scoped to a hardcoded window t in [10, 40] regardless of the requested height; it does not scan the requested range.

## Family 5 — Irrationality of Catalan's constant

- **Verdict:** `no_refutation`
- Engine: rime_number_theory_integer_relation_json (PSLQ/lattice null-space)
- Domain: small-coefficient (|m_i| <= 4) integer relations among {1, G, pi, pi^2, log2, zeta3, sqrt2, e}, f64 precision, eps = 1e-11
- Observed: `{"relations_found": []}`
- Can say: no small-coefficient integer relation among these constants at f64 precision (a relation would have been a refutation-grade finding)
- Cannot say: irrationality of Catalan's constant; f64 precision and bounded coefficients exclude only a tiny subspace of possible refutations

## Family 13 — Ostmann's inverse Goldbach conjecture

- **Verdict:** `not_probeable` — Ostmann's inverse Goldbach claim quantifies over ALL finite modifications of the primes ('no finite modification is additively decomposable') and over all large integers. The committed goldbach_verifier only certifies the classical finite range [4, 2e6]; no engine enumerates modification classes, so the statement has no finite probe path in the current system.

## Family 17 — The irrationality exponent of pi is 2

- **Verdict:** `consistent_finite_range`
- Engine: convergent continued fraction of pi (mpmath, high precision)
- Domain: first 2000 partial quotients / convergents of pi
- Observed: `{"max_partial_quotient": 20776, "top_empirical_exponents": [{"exponent": 3.4293, "q": 7, "convergent_index": 2}, {"exponent": 3.202, "q": 113, "convergent_index": 4}, {"exponent": 2.1964, "q": 6701487259, "convergent_index": 21}], "max_exponent": 3.4293}`
- Can say: the best rational approximations found up to this depth never beat the reported max exponent; the exponent curve decreases toward 2 as q grows (no anomalous sequence)
- Cannot say: mu(pi) = 2 exactly; a finite convergent list cannot rule out arbitrarily late exceptions
