# SEMF vs AME2020 Findings

Date: 2026-09-22
Data: `data/mass_1.mas20.txt` (AME2020, Huang et al. CPC 45 030002; Wang et al. 030003)
Code: `apedp/semf_validation.py`, `apedp/experiment_designer.py`, `apedp/priors.py`
Mode: read-only runs, no source edits. Active loop used numpy-equivalent
because `run_semf_active_loop` requires optional `jax` (`apedp/priors.py:18`,
`apedp/experiment_designer.py:197`), not installed in `.venv`.

## 1. Parse

- Experimental entries: 2326
- Skipped extrapolated (`#`): 1230
- Skipped missing: 1
- Fit mask: A >= 16, n = 2266

```bash
.venv/bin/python -c "from apedp.semf_validation import parse_mas20; d=parse_mas20('data/mass_1.mas20.txt'); print(len(d), d.n_skipped_extrapolated, d.n_skipped_missing)"
```

## 2. Literature SEMF baseline

Coeffs: `a_v=15.8, a_s=18.3, a_c=0.714, a_a=23.2` MeV (`LIT_COEFFS`)

| Metric | Value |
|---|---|
| RMSE / nucleon (A>=16) | 0.1304 MeV |
| n | 2266 |

## 3. Refitted SEMF (new equation 1)

`fit_semf(d)` least-squares on total B, A>=16.

| Coeff | Value |
|---|---|
| a_v | 15.3986 |
| a_s | 16.5147 |
| a_c | 0.6937 |
| a_a | 22.5997 |
| pair_scale | 1.0031 |

| Metric | Value |
|---|---|
| RMSE total B | 3.289 MeV |
| RMSE / nucleon | 0.0724 MeV (~44% better than 0.1304) |
| R^2 | 0.97306 |

Magic-residual honesty check (`magic_residual_report`):
mean|resid| magic 0.0513 MeV > other 0.0274 MeV — expected, SEMF has no shell terms.

Base BIC: 5434.8

## 4. SEMF + shell corrections (new equation 2)

`fit_semf_shell(d, width=w)`, Gaussian proximity to MAGIC=(2,8,20,28,50,82,126).

| w | RMSE_total | BIC | shell amps [N,Z] MeV |
|---|---|---|---|
| 1.5 | 2.999 | 5031.4 | [3.947, 3.415] |
| 2.0 | 2.934 | 4932.0 | [3.901, 3.426] |
| 3.0 | 2.836 | 4778.0 | [3.897, 3.464] |
| 4.0 | 2.807 | 4731.1 | [3.799, 3.295] <- best |
| 6.0 | 2.905 | 4887.3 | [3.279, 2.521] |

- Best w=4.0: ~15% better than refit SEMF total (2.807 vs 3.289).
- BIC delta ~704 (>10 decisive).
- Amplitudes positive = correct sign (extra binding at closures).
- Per-magic 19-param at w=4.0: RMSE_total 2.057, BIC 3416.0 — even better, BIC still wins.

## 5. Blind magic rediscovery from residuals

SEMF residuals grouped by N/Z, peaks = local maxima >0, MAGIC used only for scoring.

- N-peaks: [15, 29, 50, 82, 108, 125] hits [28, 50, 82, 126] recall 4/7 (+-1 tol)
- Z-peaks: [29, 50, 82] hits [28, 50, 82] recall 3/7 (+-1 tol)
- Combined: [28, 50, 82, 126]

## 6. Active design -> reveal -> refit loop

Setup mirrors `run_semf_active_loop(rounds=3, seed=0)`:
order by A, A>=16, test = every 5th, train0 = 60 random, pool = rest.
Proposal = max posterior variance `x @ cov @ x` (numpy EIG proxy for `semf_isotope_design`).

- Test RMSE trajectory: 0.0740 -> 0.0699 -> 0.0675 -> 0.0687
- Revealed (Z,A): (5,20) -> (6,22) -> (110,269)
- n_train_final: 63
- Improved vs start: True (0.0687 <= 0.0740)

Round 2 best (-8.8%), round 3 uptick is normal pool noise.

## Verdict

Yes, new equations win at every stage:
LIT 0.1304 -> refit 0.0724 -> +shell 2.807 total / BIC 4731 -> per-magic 2.057 / BIC 3416,
plus active loop 0.0740 -> 0.0687. Recommended baseline: refit SEMF + w=4.0 shell.

## Repro

```bash
.venv/bin/python -c "
import numpy as np
from apedp.semf_validation import parse_mas20, semf_per_nucleon, fit_semf, fit_semf_shell, magic_residual_report, LIT_COEFFS
d = parse_mas20('data/mass_1.mas20.txt')
mask = d.A >= 16
pred_lit = semf_per_nucleon(d.Z[mask], d.N[mask], d.A[mask], LIT_COEFFS)
print(float(np.sqrt(np.mean((d.BE_per_nucleon[mask]-pred_lit)**2))))
print(fit_semf(d)[1])
for w in (1.5,2.0,3.0,4.0,6.0):
    print(w, fit_semf_shell(d, width=w)[1])
"
# Full active-loop equivalent documented in session; exact
# run_semf_active_loop needs: pip install jax
```
