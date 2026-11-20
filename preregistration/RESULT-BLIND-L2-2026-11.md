# Result: PREREG-BLIND-L2-2026-11

**Run:** once, 2026-11-19, ending 02:36 UTC, at commit `e2a37cb`, which froze the preregistration
and the script (`preregistration/DIGEST-BLIND-L2.json`; the script checks it before running).
Python 3.13.18, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0. 288 runs, median 0.66 s each, longest 1.6 s,
on one CPU core.

Outputs, unedited, in [`RESULT-BLIND-L2-2026-11/`](RESULT-BLIND-L2-2026-11/): `results.json` (the
laws and every summary cell), `runs.jsonl` (each run's truth, distractor, status, promoted law and
timing) and `run.log`.

## Outcome

Recovered / false discovery / abstained, with the 95 % Clopper–Pearson interval on recovery.

| Noise | Known laws (24 runs each) | Invented laws (48 runs each) |
| :--- | :--- | :--- |
| 0 | 15 / 0 / 9 — 62 % (41–81 %) | 32 / 0 / 16 — 67 % (52–80 %) |
| 1 % | 21 / 0 / 3 — 88 % (68–97 %) | 30 / 3 / 15 — 62 % (47–76 %) |
| 5 % | 21 / 0 / 3 — 88 % (68–97 %) | 25 / 1 / 22 — 52 % (37–67 %) |
| 10 % | 22 / 0 / 2 — 92 % (73–99 %) | 25 / 3 / 20 — 52 % (37–67 %) |
| All | 79 / 0 / 17 — 82 % (73–89 %) | 112 / 7 / 73 — 58 % (51–65 %) |

## Against the expectations

1. **False discoveries at noise ≤ 5 % at most 5 %: held.** 4 of 216 (1.9 %, interval 0.5–4.7 %).
   None on known laws.
2. **Recovery at least 80 % at noise 0 and 1 % in each family: missed** in three of the four
   cells (known at 0: 62 %; invented at 0: 67 %; invented at 1 %: 62 %). Known at 1 % reached 88 %.
3. **Known and invented within 15 points at each noise ≤ 5 %: missed** at 1 % (25 points) and 5 %
   (35 points). Held at 0 (4 points).
4. At 10 % noise: 92 % known, 52 % invented, 3 false discoveries, all invented.

## What the misses are (analysis after the run, not preregistered)

Everything below reads `runs.jsonl` after the fact. It explains the numbers above; it does not
replace them.

**The familiarity gap is a shape gap.** The invented family contains 8 laws of the form
c₁·x^p + c₂·y^q with unequal coefficients; the known family contains none. Split by shape:

| Noise | Invented, one term (32 runs) | Invented, two terms (16 runs) |
| :--- | :--- | :--- |
| 0 | 30 recovered (94 %) | 2 |
| 1 % | 26 (81 %) | 4, plus 3 false |
| 5 % | 24 (75 %) | 1, plus 1 false |
| 10 % | 25 (78 %) | 0, plus 3 false |

On one-term laws the invented family is 7 points behind the known one at 1 % noise and 12 at 5 %
(both inside the 15-point margin), and ahead at 0, where the known laws lose runs to the exact-fit
refusal below. The design meant to compare familiarity and compared grammar shape as well, which
expectation 3 did not anticipate.

The two-term laws are inside the space the engine searches: each additive term gets its own fitted
coefficient. Re-running the eight at 1 % noise, seed 0, for diagnosis: the true law was generated in
all eight and ranked first by holdout error in six. The refusals come from the referee's resolution
criterion. Two candidates count as distinct only if their holdout errors differ by more than τ, and
τ is never below `REL_FLOOR` = 5 % of the signal's RMS (`epistemic/adjudicate.py`, calibrated against
the Duffing pairs). In every two-term law the generator drew, the smaller term moves the holdout
error by less than that, so the law with the term and the law without it are "observationally
indistinguishable" and the referee abstains (`MULTI_HYPOTHESIS`). That is the referee working as
calibrated: a term contributing under 5 % of the signal cannot be confirmed from passive data of
this kind, whatever the noise. It limits what this engine can discover, and only an intervention
that makes the small term large (`discovery_l2/active_intervention.py`) can get past it.

**All seven false discoveries are the dominant term of a two-term law** (`f6**3` for
2.71·f6³ + 2.91·v6⁻², `q5**3` for 0.945·p1^(−1/2) + 1.331·q5³, `sqrt(f9)` for
2.374·√f9 + 2.493·k3⁻²). In the six from laws 14 and 20 the omitted term varies by at most 3 against
a dominant term reaching the thousands, well below the noise at 1 %: no method could recover it from
these data, and the generator did not check that both terms are identifiable. The seventh (law 11,
10 % noise, one of eight runs) is a real miss: the omitted term's spread (sd 0.41) is close to the
noise (about 0.56) but not below it. All seven are counted as false discoveries, as preregistered.

**The zero-noise dip on known laws is the referee refusing exact fits.** All 9 known-law misses at
noise 0 are `INSUFFICIENT_EVIDENCE` with the reason "residual series is constant: |rho1| undefined,
whiteness unassessable" — the right law fitted with RMSE 0, and a residual of zeros has no
autocorrelation to test, so `adjudicate.py` treats it as unassessed and abstains. Real data are never
noise-free, so this costs nothing outside simulation, but it is a mislabel: a perfect fit is reported
as structured residuals. Re-running every zero-noise `INSUFFICIENT_EVIDENCE` run confirms it: all
11 (the 9 known-law misses and 2 invented-law runs, `invented_18` and `invented_21`) ranked the true
law first with holdout RMSE exactly 0 and were refused for this reason. The other zero-noise runs
of the same laws passed because rounding left the residuals nonzero.

## What follows

- The discoverer's error rate is the strong result: 7 promotions of a wrong law in 288 runs, each
  the dominant term of a two-term law (six of them with the other term below the noise), and none
  on a textbook law.
- Its reach is the weak one: on passive data it confirms power laws and sums whose every term
  carries at least about 5 % of the signal, and abstains on the rest. The next preregistration
  should draw two-term laws whose smaller term clears that floor, and separately measure designed
  interventions on laws whose smaller term does not, keeping this run's numbers as the baseline.
- The benchmark's comparison function raises on a promoted law that is not a sum of monomials
  (such as `1/(x + y)**3`). No such law was promoted in this run, so no result is affected; a
  successor must count it as a false discovery instead of crashing.
- The exact-fit refusal is recorded as an audit finding (`docs/SYSTEM_FLAWS_AUDIT.md`, M-01) and was
  fixed on 2026-11-19 at the owner's request (adjudication contract `solefid-v6`). The numbers above
  were measured under `solefid-v5` and stand as run.
