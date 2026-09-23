# Warp-Drive Equation Study (APEDP) — paused, resumable

**Status:** paused after literature-data validation. Resume entry points at bottom.
**Rule for this file:** every number below was measured or read from a cited
source. Speculative candidates are labeled SPECULATION throughout.

## 1. Recognition battery (measured via `PriorArtAuditor.audit_novelty`)

| Probe | Result | Time |
|-------|--------|------|
| Lorentz factor `1/sqrt(1-(v/c)^2)` | KNOWN_MODEL / LORENTZ_GAMMA, exact | 1.7s |
| Expansion scalar `v*x*df/r` | KNOWN_MODEL / CONDUCTION (structural match only — same transport-gradient shape, not same semantics) | 18.3s |
| Energy scaling `v^2 R^2 s` | APPARENTLY_NOVEL_WITHIN_CORPUS | 30s budget |
| Shape fn `(tanh(s(r+R))-tanh(s(r-R)))/2tanh(sR)` | APPARENTLY_NOVEL_WITHIN_CORPUS, time-box noted | 30.7s bounded |

Note: the 15s equivalence budget cost a genuine CONDUCTION match (found at
16.5s unbounded); default is now 30s (`equivalence_budget_s`). The tanh audit
previously hung (>120s); fixed by func-name ordering + shared deadline
(`apedp/epistemic.py:audit_novelty`).

## 2. Data-free generation (parents: `v_s^2R^2s`, expansion, Lorentz)

| # | Generated | Op | Verdict / Repair |
|---|-----------|----|------------------|
| 1 | `R^2 df s v_s^3 x/r` | mul | SPECULATION / REPAIRED |
| 2 | `R^2 r s v_s/(df x)` | div | SPECULATION / REPAIRED |
| 3 | `df v_s x/(r sqrt(1-v^2/c^2))` | mul | SPECULATION / REPAIRED (v->c singularity = expected physics) |
| 4 | `df v_s x sqrt(1-v^2/c^2)/r` | div | SPECULATION / REPAIRED |
| 5 | `R^2 s v_s^2/sqrt(1-v^2/c^2)` | mul | SPECULATION / APPROVED (zero anomalies) |
| 6 | `R^2 s v_s^2 sqrt(1-v^2/c^2)` | div | SPECULATION / REPAIRED |

Best validated: #5. Most suggestive: #3 (gamma-corrected expansion scalar).
All audited samples: APPARENTLY_NOVEL_WITHIN_CORPUS with time-box flag set
(10s probe budget) — weak novelty verdicts by construction.

## 3. Correspondence limits (measured via `HypothesisComposer.correspondence_holds`)

- #5 → parent `R^2 s v_s^2` as c→∞: True
- #3 → parent `df v_s x/r` as c→∞: True
- Gamma growth probe: 1.01 → 1.15 → 2.29 as v→c; v=c singularity flagged.

## 4. Literature energy dataset (`data/warp_energy_estimates.jsonl`, 9 rows)

Ford-Pfenning 1997 (−6.2e62·v_s kg, R=100m) · Hyperwave-2023 review (>1e80 J)
· Van den Broeck 1999 final (few M_sun; v1 "grams" superseded — row-noted)
· Krasnikov 2003 mg (secondary, low confidence) · White 2012 ~700 kg +
2011 assist (conference/disputed — quarantined from fits) · Bobrick-Martire
2021 (100× optimization + positive-energy subluminal class) · Lentz 2021
(`E ~ C v_s^2 R^2/w`, R=100m → ~0.3 M_sun) · structural scaling consensus row.

**Key measurement** (`tests/test_warp_energy_dataset.py`): Ford-Pfenning vs
Lentz at same R=100m differ by 1.04e33; wall-thickness ratio 1m/Lp predicts
6.19e34 — agreement within factor ~60 across 33 orders of magnitude,
corroborating the `v_s^2 R^2/w` scaling (candidate #5's core form, s~1/w).

**Candidate #5 checks:** finite everywhere tested; exact v_s^2 doubling
(4.00×); R^2 scaling; gamma growth — all pass.

## 5. Standing caveats

- No experimental warp data exists; validation ceiling is literature +
  numerical probes (SPECULATION cap holds — no derivation, no promotion).
- White rows excluded from fits; Krasnikov low-confidence; `s` defaulted
  dimensionless in pipeline (logged warnings) thinning dimensional approval.
- Tensor/subscript LaTeX quarantines; A2-style evolution not yet run on warp
  candidates (no warp training data exists to run it on).

## 6. Resume entry points

- `tests/test_warp_energy_dataset.py` (3 tests) — dataset integrity + scaling
- `data/warp_energy_estimates.jsonl` — extend with simulation θ(r)/T00(r) profiles
- Open: digitize published expansion/energy-density curves → fit #3's form;
  Natario condition vs continuity template; Lentz relation as Buckingham-Pi case.
