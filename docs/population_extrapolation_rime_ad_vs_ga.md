# Population extrapolation: RIME and RIME-AD against a genetic term search

> **Correction, 2026-11-19 (SYSTEM_FLAWS_AUDIT M-05).** Earlier versions of this page reported
> a result as RIME-AD's that the script had produced with `RimeProposer` (RIME); RIME-AD was
> never run. Each engine now has its own row. The "GA" rows are a genetic search over which
> columns of a fixed, hand-built term library enter a least-squares fit; the large library
> contains the true interaction term, so those rows measure term selection, not discovery.

Ground truth: $y[t] = 2 x_1[t-1] x_2[t-1] + 3 x_3[t-2] + 10 + \varepsilon$.
Populations: A-train (400 rows, seed 500), A-test (200 rows), B (fresh seed 501), C (parameter
shift, $c_1 = 2.5$, $c_2 = 2.0$), D (Laplace(0, 1.5) inputs). Machine-readable results:
[`benchmarks/population_extrapolation_results.json`](../benchmarks/population_extrapolation_results.json).

| Engine | A-train $R^2$ | A-test $R^2$ | B $R^2$ | C (shift) $R^2$ | D (heavy tail) $R^2$ | Wall time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GA term selection, small library | `0.6809` | `0.6450` | `0.7006` | `0.4185` | `0.3212` | `503.0 ms` |
| GA term selection, large library | `0.9998` | `0.9998` | `0.9998` | `0.8961` | `1.0000` | `796.6 ms` |
| RIME (`RimeProposer`) | `0.9998` | `0.9998` | `0.9998` | `0.8961` | `1.0000` | `7.0 ms` |
| RIME-AD (`RimeAdProposer`) | `0.0108` | `-0.2829` | `-0.0258` | `-0.5292` | `-0.0199` | `19085.4 ms` |

- RIME's top formula: `+1 * (((2.995647 * x3[t-2]) + (1.996585 * (x1[t-1] * x2[t-1])))) +9.99877`
- RIME-AD's top formula: `-0.0302 * ((1 / x3) / cos(x3)) + 10.0538`
- GA large-library terms selected (20): `['1', 'x1', 'x1[t-1]', 'x2', 'x2[t-1]', 'x3', 'x3[t-1]', 'x3[t-2]', 'x1^2', 'x1[t-1]^2', 'x1[t-2]^2', 'x2^2', 'x2[t-1]^2', 'x2[t-2]^2', 'x3^2', 'x1[t-1]*x2[t-1]', 'x1[t-2]*x2[t-2]', 'x1*x2', 'x1*x3', 'x2*x3']`

## Distractor challenge (RIME, seven irrelevant inputs $x_4 \dots x_{10}$)

Inputs used: `['x1', 'x2', 'x3']`; distractors admitted: `[]`;
development $R^2$ `0.999969`; wall time `10.2 ms`.
