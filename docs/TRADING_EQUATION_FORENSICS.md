# Trading-Equation Forensics: Full Session Record

**Scope:** APEDP L2 symbolic discovery applied to daily crypto price data —
does the search find stable, tradeable, non-redundant structure?
**Method:** read-only execution of existing fit/evaluate paths. No source file
was modified during this session.
**Data:** `data/market/crypto_top15.csv` (daily closes, 2021-01-01–2023-12-31;
15 assets + COMPOSITE_TOP15) for discovery/selection; Binance public daily
klines (BTC/ETH/SOL, Dec 2023–Dec 2024) fetched to `/tmp` for the 2024
out-of-sample test. 2024 data was never written into the repo.

**Frozen protocol used throughout:** fit/select on 2022 crash window
(2022-04-15–2022-07-15) and/or H1-2023; trade/evaluate on unseen H2-2023
(2023-07-01–2023-12-31); daily long/short on predicted sign; zero costs
(costs would only worsen every number below). Strict-causal features
(`r_lag`, `vol` = 10d std, `ma_dev` = (P−MA20)/MA20) unless noted
"repo-faithful" (teacher-forced gradient states, as the tournament evaluates).

---

## 1. M13 — the crash-fitted composite

Equation: `M13 = c + θ5·v + θ6·|v|·v + θ4·sin(x)`
(Tier B hand-composed: drift + momentum + quadratic drag + wave.)
Frozen 2022 params (COMPOSITE_TOP15): c=−0.00096, θ5=+0.0414, θ6=−1.5673,
θ4=−0.00358. Drag term dominates; wave/drift near-zero.

| Construction | Trades (W/L) | Hit | Cum. return |
|---|---|---|---|
| Repo-faithful, 2023 FY | 139 / 225 | 0.382 | **−390%** |
| Strict-causal, 2023 FY | 192 / 172 | 0.527 | **−31.5%** (avg win +219bp vs avg loss −263bp: wins small, loses big) |
| Baselines | buy-and-hold **+71.3%**; always-long 55% hit | | |

Repo's own Gate-4 on 2023: RMSE 0.54 vs tolerance 0.10, R² = −0.04
(worse than predicting the mean) → **FAIL, no promotion**.
Verdict: M13 fit 2022 panic dynamics and did not transfer. Correctly refused.

## 2. Search-depth wall (complexity-10 request)

Measured generation costs, market feature set (3 dimensionless vars):

| Config | Nodes | Time |
|---|---|---|
| Sandbox config: 5 ops / c5 | 1,535 | ~4 s |
| Full vocab (9 ops) / c5 | 4,241 | ~27 s |
| Full vocab / c6+ | — | unfinished past 110 s |
| Depth-3 / c10 | — | killed at 570 s timeout |

Bottleneck: per-node SymPy canonicalization; dimensionless features prune
nothing. **Exhaustive complexity-10 is infeasible** on this setup; max
feasible = depth-2/c5 with full vocabulary (~35 s). Complexity-10 needs
stochastic search = new algorithm = code change (forbidden this session).

## 3. The "67% candidate" — full adjudication

Discovered form: `f = 1/sqrt(r_lag²·vol)` → 103W/80L, hit 0.563, +67% on
unseen H2 after H1 selection.

| Test | Result | Interpretation |
|---|---|---|
| Direction | sign(f) ≡ +1 always | Degenerate: always-long in a +71% buy-and-hold year |
| Magnitude `\|r\|` | ρ=−0.05, p=0.54 | Nothing |
| Future 5d vol | ρ=−0.154, p=0.04 | Weak negative |
| Incremental R² | ~1pp | Trivial |
| Driver | corr(f, 1/\|r_lag\|) = **+0.982** | 98% = inverted existing feature |
| Decisive control | raw `\|r_lag\|` → futvol **+0.167, p=0.03** | Original feature strictly better |

Boxed result: **67% return ≠ 67% discovered signal.** Downgraded to
"positive-valued state variable that reproduced always-long," then to
"convoluted inverse-|r_lag| with no independent content."

## 4. Four-asset replication (D_P null benchmark)

Select H1-2023 → trade unseen H2-2023, per asset:

| Asset (B&H H2) | Discovered #1 | W/L · cum | M13 frozen-2022 | W/L · cum |
|---|---|---|---|---|
| BTC (+32.4%) | `r_lag³/vol³` | 85/98 · +32.0% | M13 | 113/70 · **+143.8%** |
| ETH (+17.0%) | `vol³/ma_dev` | 95/88 · +11.5% | M13 | 73/110 · −0.5% |
| SOL (+169.4%) | `r_lag³/vol⁶` | 89/92 · +36.3% | M13 | 53/129 · **−409.9%** |
| COMP (+71% FY) | `1/sqrt(r_lag²·vol)` | 103/80 · +67% | M13 | 192/172 · −31.5% |

Full 15-asset generation rerun (frozen depth-2/c5/full-vocab, train 2022 +
select H1-2023; top-1 per asset, sign-twins throughout): BNB/ADA
`±ma_dev + r_lag·vol`; XRP `±r_lag² + sin(r_lag)`; DOGE `±ma_dev³ +
ma_dev⁻²`; AVAX/MATIC `(r_lag−vol)/ma_dev`; DOT `ma_dev/vol ± r_lag`;
LINK `r_lag ± sin(r_lag)`; NEAR `ma_dev·vol ± r_lag`; ATOM
`exp(r_lag)·sin(r_lag)`; UNI `cos(r_lag/ma_dev²)`; LTC `±ma_dev³ +
1/ma_dev`. BTC/ETH/SOL replicate prior runs exactly (seed-stable). No
cross-asset winner; all RMSEs at noise floor; every top cluster is twins.

H1-selection R² for the 15 generated #1s (post-hoc OLS scale on 2022-train,
evaluated H1-2023; engine RMSEs definitive, R² computed consistently here):
UNI +0.03 (sole positive); BTC −0.13, AVAX −0.14, MATIC −0.04, LINK −0.07,
XRP −0.07, ATOM −0.09, SOL −0.09, BNB −0.17, DOGE −0.19, NEAR −0.20,
ETH −0.23, ADA −0.36, LTC −0.68, DOT −2.12. Every generated #1 predicts
worse than the H1 mean — selection among noise-floor ties, confirmed
numerically.

Sandbox-native runner (`run_crypto_experiment`, own windows: train 2021
bull → Terra-LUNA shock holdout, 12-term FIN_TERMS, bootstrap
adjudication), all 15 + composite: 13× MULTI_HYPOTHESIS, 2×
INSUFFICIENT_EVIDENCE (BNB, LINK), 1× NO_VALID_CANDIDATE (LTC), 1×
UNIQUE_SUPPORT — UNI `cryp-0044` (sin_x + v, train_loss 5.93,
holdout_rmse 0.30), promoted as sole survivor ("no structurally distinct
rival survived scoring"), i.e. promotion by elimination, unforensicked.
Different regime, different verdicts vs my harness — the window shapes the
verdict more than the engine does.

Sandbox-native rank-1 table (equation + RMSE + trajectory-R² + whiteness;
R²/whiteness computed post-hoc, runner emits train/RMSE only):
COMP abs_v_v+x3 0.193/+0.09/0.59; BTC sin_x+v 0.354/+0.87/0.45; ETH one+v
0.266/+0.17/0.48; SOL abs_v_v+x 0.126/+0.43/0.63; BNB abs_v_v+sat_x
0.120/−0.57/0.58; XRP one+x 0.264/+0.81/0.62; DOGE sat_x+x 0.150/+0.57/0.64;
ADA one+x 0.219/+0.92/0.64; AVAX abs_v_v+sat_x 0.130/+0.72/0.58; DOT
abs_v_v+inv_x2 0.344/+0.90/0.53; LINK abs_v_v+sin_x 0.580/+0.93/0.64;
MATIC v+x3 0.124/−0.14/0.56; NEAR one+v 0.243/+0.74/0.53; ATOM one+x3
0.321/+0.61/0.70; UNI sin_x+v 0.302/+0.97/0.49 (promoted, sole survivor);
LTC empty (0/55 gate-4). Whiteness fails on 15/16 (|ρ₁| 0.45–0.70);
rank-1 = lowest train-loss among gate-4 survivors, not the promoted form.

L4 residual-layer probe (disclosed OLS-algebraic substitution — NOT the
sandbox's dynamical fits, whose residuals are far less white; UNI/SOL/COMP
over sandbox windows, layer-2 = AR(1)+OFI+vol on signed residuals): UNI
0.506→0.249 but ALL layer-2 coeffs ≈0 (pure intercept/de-meaning, R²=0.31
from bias alone — no AR/OF structure); SOL 0.114→0.081 (phi 0.11, bOFI
0.26, bVol 0.21, R²=0.26); COMP 0.046→0.034 (bOFI 0.39, bVol 0.38,
R²=0.29). Two-layer hypothesis gets qualified algebraic-level support
(flow/vol levels explain ~30% residual variance; AR dynamics ~nothing);
UNI's fixable residual is a bias term — promotion-as-law further
unwarranted. Motif counts verified: abs_v_v in 6/16 rank-1s, sat_x in 3.

Dynamical L4 loop (simulator in the loop; UNI+SOL; A=rank-1, B=+AR(1)
post-hoc correction with φ from train rollout, C=+sat_u flow term,
metric = forward trajectory RMSE on LUNA-shock holdout): UNI
A=0.306/B=0.090/C=2.97 (φ=+0.46); SOL A=0.126/C=0.334, B unavailable
(train rollout unstable — fit a0=−10.6 knife-edge, explodes off-holdout).
Findings: (i) in-model flow via sat_u catastrophically worse on both
(10×/3×) — OFI-as-drive destabilizes fits as currently typed; u_relax
itself is dimensionally unrepresentable (u−v Add rejected), which is why
no rank-1 ever contained it; (ii) AR post-hoc wins big on UNI but is
correction, not mechanism; (iii) SOL's rank-1 fit is rollout-fragile —
fine on 21d, blows up on 365d. L4-dynamical verdict: neither B nor C as
constructed earns promotion; C fails hard.

Pipeline gates executed (proposed Discover→Stability→Integrate→Forward):
TEST1 identification (quarterly 2021 fits): UNI CV 1.2/1.3 with 30×
coefficient swings → PARAMETER_UNSTABLE; SOL CV 0.73/0.88 passes the
CV<1 rule but a0 swings 50× (−20.5→−0.42) — CV is scale-deceptive,
max/min-ratio is the better gate metric (flags both). TEST2 rollout
ladder (train x0 vs train truth): UNI 0.33→1.65→2.42→2.33 (dead by 60d);
SOL 0.144→0.118→0.128→0.484 (viable through 180d, degrades 365d —
CORRECTION of earlier "blows up" claim: an earlier train-sim returned
empty, not reproduced; current measurement is degraded-not-exploded).
TEST3 motif ablation H1(|v|v+x) vs H2(v+x) vs H3(x³+v), LUNA-holdout
trajectory RMSE: UNI linear wins (1.208 < 1.266 < 1.947); SOL cubic wins
(0.087 < 0.126 < 0.177). |v|v wins NEITHER asset head-to-head — recurrence
(6/16) does not survive controlled ablation. Statuses: |v|v, sat_x →
FIT_ONLY/RECURRING_MOTIF (mechanism unsupported); UNI rank-1 →
PARAMETER_UNSTABLE + dead-by-60d; OFI graft → DYNAMICALLY_REJECTED;
UNI AR correction → DIAGNOSTIC_ONLY. L1→L4 directionality preserved: L4
never rescued anything here.

Findings:
- **No structure replicates.** Each asset's winner is a different expression
  at its base rate. Every top-2 pair is sign-identical twins (MULTI-style
  indistinguishability across independent searches).
- **M13: +144 / −0.5 / −410 / −32.** Radically asset-dependent despite frozen
  params — lottery ticket, not law. The BTC cell alone would be selection
  fraud; the panel is the finding.
- Formal D_P record: *"Under frozen H1→H2 validation, independently
  discovered expressions did not replicate across BTC/ETH/SOL/composite. No
  cross-asset persistence."*

## 5. ETH `vol³/ma_dev` forensics (T1→T6 pipeline)

Unseen H2 (n=178), two-sided (53% short):

| Test | Result |
|---|---|
| T1 OOS loss | RMSE 0.02474 vs train-mean 0.02424 vs **zero 0.02205** — loses to naive |
| T2 Direction | ρ=−0.04, n.s. |
| T3 Magnitude | ρ=**+0.18, p=0.015** — one surviving signal |
| T4 Future vol | +0.06 n.s.; dominated by raw `vol` (+0.20) and `\|ma_dev\|` (+0.19) |
| T5 Incremental R² on `\|r\|` | vol+ma 0.0077 → +f 0.0176 (~1pp; f-alone 0.0095 vs `\|ma\|` 0.0064) |
| T6 Drivers | corr(f, 1/ma_dev) = **+0.889** — echo of inverse-MA-deviation |

Verdict: H1 "lowest loss" was selection luck; residue = ~1pp magnitude echo
whose carrier is `|ma_dev|`, already in the feature set. Interesting, not
promotable. (Canonical form verified exact: `vol**3/ma_dev`; the reciprocal
`ma_dev/vol**3` below is investigator-derived, never discovered.)

## 6. Reciprocal paradox + collapse panel

`ma_dev/vol³` (unstudied reciprocal): future-vol **+0.29, p<0.005** — the
panel's strongest correlation, from the orientation the engine discarded.
Loss-rank ≠ information-rank. Status: hypothesis only (≈35 implicit tests;
needs unseen window).

Four-way driver panel (all RMSE lose to predict-zero):

| Form | Top driver | Futures |
|---|---|---|
| BTC `r_lag³/vol³` | r_lag +0.99 | all n.s. |
| SOL `r_lag³/vol⁶` | r_lag +0.93 | all n.s. |
| COMP `1/sqrt(...)` | 1/\|r_lag\| −0.98 | nothing independent |
| ETH `vol³/ma_dev` | ma_dev +0.61 | \|r\| +0.18 only |

Then the +0.29 dissolved: Q3 +0.04 / Q4 +0.17 (both n.s.) → pooled
significance is a **between-quarter level effect**, not a mechanism. Same
instability disease as M13, one level subtler.

## 7. Derivative family — pre-registered screen

Frozen in-conversation before running: 7 forms × 4 assets on H2, primary
endpoint = same-sign Spearman p<0.05 vs future-5d-vol on ≥3/4 series:

| Form | BTC | ETH | SOL | COMP | Verdict |
|---|---|---|---|---|---|
| `ma·vol` | +0.20ᵖ | +0.24ᵖ | +0.55ᵖ | +0.47ᵖ | PASS |
| `ma/vol` | +0.20ᵖ | +0.29ᵖ | +0.50ᵖ | +0.48ᵖ | PASS |
| all 5 others | — | — | — | — | fail |

Control killed the interaction reading: raw signed `ma_dev` alone gives
+0.19/+0.27/+0.54/+0.48 — ×vol and ÷vol add **zero**. Pipeline self-check:
`rl/v` ≡ `rl³/vol³` matched to the digit (rank-monotonicity demands it).

## 8. Signed-MA → future-vol staircase (final hypothesis)

Shape (pooled quartiles, normalized): deep-below 0.75 → below 0.89 →
above 1.05 → far-above 1.32. Monotonic dose-response: elevation, not distance.

Stability correction ("4/4" overstated):

| Asset | Q3 | Q4 | Full H2 |
|---|---|---|---|
| BTC | −0.05 | −0.03 | +0.19ᵖ (pooling artifact) |
| ETH | −0.04 | +0.19 | +0.27ᵖ (Q4-driven, unconfirmed) |
| SOL | +0.33ᵖ | +0.31ᵖ | +0.54ᵖ (stable) |
| COMP | +0.31ᵖ | +0.44ᵖ | +0.48ᵖ (stable) |

Vol-clustering partial (current vol residualized): SOL +0.54→**+0.38**,
COMP +0.48→**+0.49** — survives; not GARCH in disguise.

Literal linear form `futvol = a + b·ma_dev + c·vol`:

| Sample | b̂ | t | b>0? |
|---|---|---|---|
| SOL H2-2023 | +0.068 | +6.1 | YES |
| COMP H2-2023 | +0.070 | +6.6 | YES |
| SOL 2024 (Binance) | +0.011 | +1.4 | no (p=0.16) |

Full fitted equations (H2-2023, n=178; RV_5d(t) = std of log-returns over
days t+1…t+5; ma_dev(t) = (P_{t−1}−MA20)/MA20 signed dimensionless; vol(t) =
10d trailing std floored 0.002; all vol terms in daily-return units):

- SOL: `RV5d(t) = 0.0275 + 0.068·ma_dev(t) + 0.117·vol(t)`, R² = 0.30
- COMP: `RV5d(t) = 0.0241 + 0.070·ma_dev(t) − 0.117·vol(t)`, R² = 0.20

Caution recorded: the b̂ slopes match (+0.068/+0.070) but the vol loadings
mirror (+0.117 vs −0.117), suggesting multicollinearity reshuffling between
ma_dev and vol rather than cleanly separated effects. Portability narrowing:
**only b travels, not (a, c).** The portable content of the session may be
exactly one number, b ≈ +0.07, valid in 2023, dead in 2024.

## 8b. Cross-asset-class extension — the sign flips

Same frozen spec (`futvol = a + b·ma_dev + c·vol`, partial-out-vol control)
on repo S&P 500 / Nasdaq-100 (H2-2023) and PAXG gold-token (Binance):

| Sample | n | raw ρ | partial-vol | b̂ (t) |
|---|---|---|---|---|
| SPX H2-2023 | 120 | −0.41, p<0.001 | −0.40 | −0.038 (−4.6) |
| NDX H2-2023 | 120 | −0.29, p=0.001 | −0.29 | −0.039 (−4.0) |
| PAXG H2-2023 | 178 | −0.02 n.s. | −0.04 | +0.015 (+1.1) n.s. |
| PAXG 2024 | 360 | +0.16, p=0.003 | +0.14 | +0.033 (+3.4) |

Reading: US equities show a strong, partialled-surviving **negative**
relationship — above-MA predicts *lower* future vol. That is the textbook
asymmetric-volatility (leverage) effect, and it is the mirror image of the
crypto-2023 positive sign. Gold is null in 2023, positive in 2024.
Consequence: b>0 is falsified as universal. Refined statement: the
signed-deviation→vol slope exists across asset classes but its **sign is
market-structure-dependent** (crypto chase + vs equity leverage −).
Multiple-comparison note: dozens of session tests run; SPX p<0.001 survives
Bonferroni-scale scrutiny, NDX p=0.001 is borderline — treat NDX as
supporting, SPX as decisive.

Discriminant vs the textbook leverage effect (futvol ~ r_lag + vol, then
+ ma_dev; H2-2023): SPX R² 0.045 → 0.179 (Δ+0.135, ma t=−4.36, p<0.0001);
NDX R² 0.019 → 0.136 (Δ+0.117, ma t=−3.97, p=0.0001). The slow 20-day
deviation adds ~12pp of R² beyond yesterday's return — distinct slow
component, not repackaged leverage. Refined hypothesis:
b_market with b_crypto>0, b_equities<0, b_gold=? (sign is
market-structure-dependent).

Crypto home-turf discriminant (same spec, H2-2023): SOL Δ+0.122 (t=+5.52);
COMP Δ+0.158 (t=+5.85); ETH Δ+0.062 (t=+3.49, p=0.0006); BTC Δ+0.005 n.s.
Slow component adds 6–16pp with positive sign on 3/4 crypto series —
same order of incremental content as equities (~12pp), opposite sign.
Verdict: one slow mechanism, sign-flipped by market structure. ("Dominates"
now licensed on both turfs: slow Δ dwarfs fast-variable R² everywhere
except BTC, where nothing exists.)

Full 15-asset sandbox rerun (frozen spec, H2-2023): positive sign 15/15,
significant-positive 12/15 (exceptions: BTC/XRP/LTC null; UNI/LINK
borderline under Bonferroni 0.0033, 10+ survive cleanly). Deltas +0.04 to
+0.14. Within-2023 crypto the effect is sign-universal; combined with the
2024 death this sharpens the regime reading (2023 post-crash chase year vs
2024 ETF year) without confirming it.

2024 equities leg (Yahoo daily closes, `/tmp` only): SPX Δ+0.014, b=−0.028
(t=−1.90, p=0.059); NDX Δ+0.005, b=−0.016 (t=−1.16, p=0.25). Sign retained
(negative both indices, both years) but significance mostly gone — consistent
with a stress-dependent effect (H2-2023 held the Aug–Oct −10% correction;
2024 was a low-vol grind). Post-hoc interpretation, not pre-registered:
asymmetry bites hardest when there is something to be asymmetric about.
Sign-flip scoreboard on unseen 2024: direction persists on both legs,
strength persists on neither.

Stress-conditioning test (pre-registered; repo 2018–2023 history; stress =
trailing-60d vol-mean above 2018–2022 median; interaction
frv ~ ma + rl + vol + ma·S + S): prediction was |b| larger under stress.
Result — prediction FALSIFIED in direction: SPX b_calm=−0.147,
e=+0.121 (t=+10.6) → b_stressed=−0.026; NDX b_calm=−0.106, e=+0.078
(t=+7.6); SOL b_calm=+0.028, e=−0.020 (t=−1.9, p=0.06). The positional
signal is STRONGER in calm markets and drowns under stress (ceiling effect:
when everything is volatile, MA position stops discriminating). Revised
statement: |b| larger in calm, on all three tested series. The
reinterpretation is post-hoc — the failed prediction is the recorded result.

## 8c. Post-failure hypotheses (registered, not found)

**H-CALM-01 — Calm-regime MA-position discrimination.**
Origin: pre-registered "|b| increases under stress" falsified in direction
(SPX/NDX/SOL all attenuate toward zero under stress).
Claim: |b_calm| > |b_stress| for b on 20-day signed MA-deviation in
`FV = a + b·MADev + c·r_{t−1} + d·Vol + ε`.
**H-BASE-01 — No regime dependence (null competitor):**
|b_calm| ≈ |b_stress|; apparent attenuation was sampling variation.

Pre-registered support conditions (all required): same MA20 definition
(P_{t−1}−MA)/MA; same future-5d-vol target; same controls (r_{t−1}, Vol);
stress from a pre-specified vol threshold, never tuned post-hoc;
interaction e moves b toward zero under stress with statistical support
(not mere numeric shrinkage); per-asset-class direction preserved
(b_crypto>0, b_equities<0 — attenuation, not reversal); replication on
genuinely unseen data. Sequence: prediction A → falsified → H-CALM-01 vs
H-BASE-01 → unseen test (below).

H-CALM-01 forward test (other assets): in-window 21–23 (repo CSV): BNB
atten-YES, AVAX atten-YES, DOGE/LINK null, XRP base-negative (direction
violated), ADA pro-stress (anti-attenuation, t=+5.1). Unseen 2024–2026H1
(Binance, n≈902/name, pre-2024 threshold): DOGE SUPPORT (b +0.143→+0.015,
t=−4.95); AVAX significant but sign-flips under stress (no — direction
condition fails); BNB base-negative (no); XRP/ADA/LINK n.s.
Score: 1/6 support, supporter null in-window — noise signature.
**H-BASE-01 wins this round.** Refinement: even within crypto, b_crypto>0
fails per-name-per-window (XRP, BNB negatives); attenuation is not a
general crypto property. Status: H-CALM-01 demoted to unconfirmed;
H-BASE-01 (no regime dependence) is the standing null.

Formal ledger:

| Stage | Hypothesis | Outcome | Status |
|---|---|---|---|
| A | Stress amplifies \|b\| | Interaction went toward zero | **FALSIFIED** |
| B | Calm regime systematically larger \|b\| | 1/6 forward support | **FAILED FORWARD** |
| C | No systematic regime dependence | Not rejected by these tests | **STANDING NULL** |

> **H-CALM-01 — FAILED FORWARD.** BNB and AVAX supported attenuation
> in-window but did not reproduce it on unseen 2024–2026H1 data. DOGE was
> the sole forward supporter (1/6) despite being null in-window. Others were
> null, directionally inconsistent, or sign-changing. The reshuffling
> supporters set across windows is the signature of window-specific
> correlation, not a stable mechanism.

> **H-BASE-01 — STANDING NULL.** Insufficient evidence for stable
> regime-dependent magnitude; regime dependence is not an established
> property.

Scope precision (not "ceiling effect disproven"): what failed is the
**pre-registered version under the specified assets, windows, model, stress
definition, and forward test**. No evidential basis exists to promote any
other regime-dependence formulation now. Hierarchy demonstrated:
observation → mechanism → pre-registration → development evidence →
unseen test → rejection. Each failed forward test removes a story from the
active hypothesis space — the referee's actual product.

## 9. 2024 decisive test (new data, Binance klines, `/tmp` only)

| Asset | 2024 raw | 2024 partial-vol | 2023-H2 |
|---|---|---|---|
| SOL | −0.035 n.s. | −0.031 | +0.54ᵖ |
| ETH | +0.057 n.s. | +0.070 | +0.27ᵖ |
| BTC | −0.029 n.s. | −0.027 | artifact |

Pooled 2024 staircase: 0.98/1.05/0.95/1.03 — flat. **No replication.**
Same-source H1-2023 check (hypothesis-unseen): SOL +0.29ᵖ / ETH +0.20ᵖ /
BTC null — phenomenon existed across both 2023 halves, absent 2024.
Caveats: source change (repo index vs Binance; discounted — daily
returns-dynamics shouldn't care) and regime change (2024 ETF year;
respectable refuge, needs a pre-specified regime classifier to stay
falsifiable).

## 10. Final statuses

| Item | Status | Provenance |
|---|---|---|
| M13 | Dead law: crash-fitted, transfer-failed, gate-4 refused | APEDP Tier B (hand-composed), repo tournament |
| 67% candidate | Downgraded: degenerate always-long + redundant transform | APEDP L2 search (depth-2/c5, full vocab) |
| BTC/SOL #1s | Collapsed: ≥93% = lagged return itself | APEDP L2 search |
| ETH `vol³/ma_dev` | Documented ~1pp echo, carrier = `\|ma_dev\|`, not promotable | APEDP L2 search |
| ETH reciprocal `ma_dev/vol³` | Investigator-derived hypothesis, between-regime confound found, unconfirmed | Investigator transform, never discovered |
| Derivative family (5 of 7 forms) | Screened, 2 passed criterion, carrier traced to raw `ma_dev` | Investigator-proposed (2 coincided with discovered forms) |
| Linear form `futvol=a+b·ma+c·vol` | Estimated relationship, not a law (§8 table) | Investigator specification; b̂ estimated on APEDP features |
| Signed-MA → futvol staircase | **2023 phenomenon, unconfirmed outside 2023.** Witnesses SOL+COMP; BTC artifact-control; ETH ambiguous | APEDP features + investigator test spec |
| D_P program | Null benchmark, 4× documented: no stable promoted structure in price-only daily space | Session result |
| Order-flow program (D_O/D_P+O) | Blocked: no intraday/book data on disk; spec ready (A/B/C ablation, X≤t→Y>t contract, pre-registered horizons) | Proposal, not executed |

**On the linear form specifically:** the regression *form* is textbook; the
empirical content is the estimated b̂>0 on 20-day signed MA-deviation with a
vol covariate (t>6, two witnesses) and its disappearance in 2024 (p=0.16).
That contrast is evidence of **regime-dependence, not universality** — the
t-statistics must not be read as law-certificates. Status: explicitly
estimated empirical relationship, unconfirmed outside 2023.

## 12. OF-01 — order-flow incremental information (Binance public data)

Pre-registered: OF/positioning vars add OOS info on future-5d-vol beyond
(ma_dev, r_lag, vol). Variables (daily, causal; OI dropped — history
endpoint dead): volume z-score, signed taker fraction, 5d CVD slope,
trade-count z, funding rate. Fit 2024 (frozen) → evaluate 2025–mid-2026
(n_te=537); ΔR²_OOS = full − base.

| Asset | OOS-R² base | OOS-R² full+OF | Δ | Train t-stats of note |
|---|---|---|---|---|
| SOL | −0.116 | −0.000 | **+0.115** | ma +4.8, cvd-slope −3.1, fund +2.3, vo +2.3 |
| BTC | −0.038 | +0.009 | **+0.047** | fund +5.4, ma −3.1 |

**OF-01 — SUPPORTED, QUALIFIED:** order-flow information provides
incremental OOS information vs the L1 baseline on SOL and BTC, carried by
funding rate (both) and CVD slope (SOL). Absolute OOS performance ≈ 0 —
supports incremental content, not a validated predictor.
**OF-MECH-01 — UNRESOLVED:** crowded-positioning→unwind reading is
mechanistic interpretation, not established causation; kept separate.
**OBS-MA-REGIME-01:** SOL shows +4.8σ MA coefficient in 2024-train despite
flat pooled-2024 — heterogeneity worth investigating, not a regime
mechanism (H-CALM-01's fate forbids instant re-hypothesizing).

L3 (APEDP-generated interactions on {ma,rl,vo,fund,cvd}, depth-2/c6,
selected within-2024, OLS scale frozen 2024, OOS 2025+, n=536): funding in
every top form; sign-twins identical in-selection. OOS: SOL `fund/vo +
sqrt(vo)` +0.021; SOL `fund − ma + sqrt(vo)` **−0.181**; BTC
`fund + vo − 1/vo²` −0.016. Interactions do not beat additive OF-01; the
ma-interaction is actively harmful. Note: ±ma twins indistinguishable in
selection (+same RMSE) diverge to +0.02 vs −0.18 OOS — selection-ties are
not equivalence, another MULTI exhibit. L3 verdict: failed to improve;
additive OF-01 stands as the furthest supported point.

## 12b. Rolling frozen trading protocol (H1/H2/H3, autonomous integration)

Pre-registered: train 180d → freeze → autonomous 20d rollout from OOS-start
state (model's own state feeds forward; no teacher forcing); signal = sign
of predicted 20d drift held whole window; baselines buy-hold + flat; 5bps
per flip; 60d steps through 2023 (6 windows); BTC/SOL/ETH. Mean net 20d
return (wins/windows): BTC H1 −6.26% (1/4), H2 −2.90% (3/5), H3 +2.61%
(3/5), BH +7.55% (3/6); SOL H1 −12.59% (2/3), H2 +9.55% (3/6), H3 −1.54%
(2/6), BH +27.84% (4/6); ETH H1 −4.50% (2/5), H2 +1.63% (3/6), H3 +1.63%
(3/6), BH +6.97% (3/6). No structure beats buy-hold anywhere; winners
differ per asset; H1-drag (6/16 selection motif) is worst-or-second on all
three (3/3 last-or-near-last). Caveat: 5–6 windows per asset — directionally
devastating for |v|v-as-trading-structure, not a tight estimate of anything.
Scope discipline: "buy-hold sweeps" holds for these six frozen 2020-recovery
windows only, not as a general superiority claim.

State-augmentation probe (Model C with vol-as-drive via u-channel, same
frozen forward-trajectory test): UNI A=0.306→Cvol=2.67 (9× worse, same
detonation as OFI graft); SOL A=0.126→Cvol=0.109 (14% better, suggestive
not established). Same augmentation, opposite outcomes per asset.
Failure taxonomy recorded: January = Representation failure (cannot fit
regime, H1/H2 non-convergence); October = Extrapolation failure (fits,
autonomous evolution wrong; all structures agree, regime transition
missed) — REGIME_FAILURE logged, not repaired post-hoc. Evidence chain
for |v|v closed: recurrence → ablation 0/6 → rollout fragility → frozen
OOS last-or-near-last 3/3 → mechanism hypothesis CLOSED (as trading
structure; FIT_ONLY/RECURRING_MOTIF retained as approximator status).

## 20. Denominator ablation (1/Volume) — H4 verdict

V0 (persist+Amihud+r+|r|) → V1(+1/Vol) / V2(+|OFI|/Vol) / V3(both),
fit-2024 → OOS 2025+, 15m. BTC: V1 +0.0066, V2 +0.0098, V3 +0.0154
(≈additive — Case C shape). SOL: V1 −0.0068, V2 +0.0056, V3 −0.0038
(collinearity fight). Diagnostics over same base: raw OFI −0.001/+0.002
(nothing); Volume +0.007/+0.019; logVolume **+0.025/+0.044** (dwarfs
everything); 1/Vol mixed (+0.007/−0.007). Verdict: H4 CONFIRMED in
stronger-than-predicted form — not the denominator (1/Vol inconsistent)
but VOLUME ITSELF (logVol dominates both assets). |imb|-frac was an
inverted volume proxy (both spike on low-volume bars); its negative sign
(low fraction → ? no: HIGH fraction = LOW volume → calmer ahead) is the
volume effect wearing imbalance costume. Status: REPLICATED_STRUCTURE
→ **CONSTRUCTION_EXPLAINED**. The surviving discovery is
volume-state → future vol (+2–4pp over move-controls), not OFI. H1/H3
re-enter through the volume door (liquidity state was the right family;
imbalance was its proxy).

Volume-State Ladder (TOD-seasonal profile frozen from 2024-train;
fit-2024 → OOS 2025+): BTC raw logVol +0.025 / lagged +0.019 / surprise
+0.019; SOL +0.044 / +0.034 / +0.040. Surprise retains ~90% of raw —
NOT a clock artifact; unexpected volume carries the information. Lagged
works too (persistent activity state, not a one-bar blip). Status:
VOLUME-SURPRISE → future vol OOS-supported; mechanism (activity vs
liquidity vs participation) unresolved; source replication still absent.
(Correction: surprise retains 76% BTC / 91% SOL of raw, not ~90% both.)

Frozen volume-surprise replication (2024-fit never refit, quarterly
blocks 25Q1–26H2-YTD): BTC +0.019/+0.004/+0.013/+0.004/+0.016/+0.107,
SOL +0.015/+0.005/−0.062/+0.009/+0.096/+0.233 → 11/12 positive (SOL25Q3
the exception). 2026 block audit (V0/V3/Δ/Var/n, frozen b): target
variance does NOT explode in 2026 (BTC 26H2-YTD Var lowest of all blocks)
— but absolute MSE gains (≈Δ×Var) run only ~2× 25Q1, not 5×: escalation
partly metric mechanics (dead baseline + low variance inflating R²),
partly real. Relabeled 26H2-YTD (partial, data to Oct-2026). Status:
SIGN-STABLE 11/12, MAGNITUDE-REGIME-DEPENDENT, OOS-SUPPORTED;
2026-escalation = INVESTIGATE (audited: real-but-smaller than R² suggests).

Regime stratification (frozen spec; OOS 2025+ by trailing-vol terciles,
ΔMSE + ΔR²): BTC low dR²+0.107/dMSE+7e-8 (V0 −0.08 dead), mid −0.015,
high +0.015; SOL low +0.204/+3.7e-7 (V0 −0.11 dead), mid +0.018, high
+0.021. Usefulness concentrates in LOW-vol states where persistence
predicts continued calm and misses arriving turbulence — calm-before-
storm detection, not general vol tracking. 5/6 strata positive on both
metrics. Status: CONDITIONAL information structure (strongest mechanistic
characterization to date); universal-coefficient reading retired.

REVISED by conditional-structure challenge (rich HAR-style baseline
RV_short/med/long + |r| + Amihud; fit-2024 → OOS 2025+): M1(+VS) survives
(BTC +0.008, SOL +0.021 over M0=0.33) but M2(+VS×LowVol) adds nothing
under 33/34/33 or 20/60/20 thresholds (worse than M1 both), and
continuous VS×volStd interaction goes POSITIVE (t=+2.4/+13.2, opposite of
predicted negative). Verdict: stable ADDITIVE effect, no conditional
modulation — tercile pattern attributed to baseline-health/denominator
mechanics (low-vol strata: dead baselines inflate fixed absolute gains).
CONDITIONAL retired; STABLE-ADDITIVE stands (survived its strongest
baseline yet).

## 21. Explicit discovered equation (VS_t definition + frozen coefficients)

VS_t = (logVol_t − μ_q) / σ_q, where logVol_t = log bar quote-volume
(NaN if ≤0), q = 15-min quarter-of-day bucket (96/day) from bar open
timestamp, μ_q/σ_q frozen from 2024-train bars in bucket q (>20 obs).
Target RV = std of next-16-bar log-returns. HAR state: trailing 12/48/96.
All features standardized with frozen 2024-train (mu, sd); OLS fit-2024:

BTC: RV = 0.002333 + 0.00031·tv + (−0.00012)·med + 0.00043·lng
     + 0.00024·|r| + (−0.00002)·amih + 0.00021·VS
SOL: RV = 0.004240 + 0.00045·tv + 0.00002·med + 0.00070·lng
     + 0.00046·|r| + (−0.00012)·amih + 0.00030·VS
(standardized units; amih native scale ~1e-12.) Status: STABLE-ADDITIVE,
frozen-persistent, replicated BTC/SOL/ETH-transfer; mechanism open;
source replication absent; no regime interaction (rejected).

Canonical H-A (frozen, word-for-word — validation layers stay AROUND the
equation, never inside it):
  RV̂_{t+1:t+16|t} = β0 + β1·TV_t + β2·|r_t| + β3·Amihud_t
                    + β4·log(RollMean_40(V_t))
  BTC: β4=+0.00021; SOL: +0.00030 (standardized; fit-2024, never refit).
Layer 1 (aggregate OOS): ΔR²>0, 6 assets. Layer 2 (replication):
permutation 0/300s, 31/36 quarterly, 15-asset battery, window ablation
40>15 8/12. Layer 3 (trajectory): calm watch well-behaved (forecast
~2× realized, same neighborhood); storm watch (Oct-10 crash window)
forecast 0.002–0.006 vs realized 0.014–0.015 — blind to jumps by
construction. Scope constraint (not a term): validated for incremental
background-vol forecasting; NOT validated for jump/crash anticipation.
H-A forecasts weather, not earthquakes.

## 22. Grammar (a): rolling operators + VS rediscovery verdict

Implemented (approved (a) only; sweep.py, defaults, (b) untouched):
RollingOpNode (rollmean/rollstd, frozen ROLL_WINDOW=40, causal NaN-head,
dimension-preserving, non-nesting) + generation block 1c (leaves/depth-1
only, principled feature-constructor bound) + finite-overlap treatment at
three absolutist gates (fingerprint mask ≥8, Gate-A scaled min(10,size),
scoring overlap masks ≥10) — previously EVERY rolling form died silently
(32-pt probes < W=40; three all-finite gates). 4 unit tests green; L2
suites 30/31 (1 failure = concurrent session's pre-existing message
rename, verified against HEAD, untouched by this work); ruff clean.
Rediscovery (volume-only vocab, depth-4/c10, 2024 15m, target future
vol): exact VS form generated (1 node) and scorable (train 8e-6) but
NOT preferred — holdout 0.00252 vs top log(RollMean(V)^−2) 0.00096
(2.6× worse); verdict MULTI among better forms. Frozen OOS 2025+:
exact-VS −0.085 vs V0 +0.217 (Δ−0.30, catastrophic); raw-volume rolling
forms win in-discovery. Gate outcome per approval: VS = EXTERNALLY
VALIDATED FEATURE, not autonomously discovered structure. Note: first
rediscovery run accidentally searched double-log space (X=logVol);
diagnosed mid-flight, rerun correctly on raw volume — recorded so the
ledger shows the misstep. Uncommitted (commit never requested).

Synthetic control (planted z=(x−μ40)/σ40 truth, SNR~7, vocab
x/rollmean/rollstd/+−/, z withheld): exact-z ranked #1 of 1020, verdict
UNIQUE_SUPPORT. Engine CAN discover standardization when planted —
BTC/SOL non-selection is market-landscape selection (volume-level forms
fit better there), not search incapability. Boundary refined:
aggregation ✓, synthetic standard-recovery ✓; market VS-quotient not
preferred. (b) stays unimplemented: no evidence a TOD operator would be
exploited.

Generation+R² audit (2024 15m, OLS scale, train-70/holdout-30): H-A-core
log(rollmean(V)) train +0.197/hold +0.143; exact-VS quotient +0.024/+0.016;
raw rollmean(V) +0.212/+0.024 — best train, collapses OOS. H-A-core wins
holdout among generated forms: the log transform generalizes, raw level
overfits. Selection-vs-validity, demonstrated inside one table.

Window ablation H-A15 vs H-A40 (fit-2024 frozen, quarterly OOS ΔR²):
BTC 40 wins 4/6 (15 takes 25Q1/25Q4), SOL 40 wins 4/6 (15 takes 26H1/26H2)
→ 8/12 for 40. Both negative in 25Q3 (regime dominates window choice).
H-A₄₀ RETAINED; H-A₁₅ plausible but not superior on this evidence.

Head-to-head (engine pick H-A=log(RollMean(V)) vs exact rolling VS
quotient H-B, frozen-2024 OLS, quarterly OOS): H-A wins 11/12 blocks
(sole exception SOL25Q3, both negative there). Training preference and
OOS agree — engine was right. Precision: rejected object is the
ROLLING-40 quotient specifically; the validated TOD-seasonal VS is a
different construction and stands separately.

H-A full-protocol start: permutation 0/300 BTC + SOL (alignment matters);
own-2024 frozen → 2025+ deltas ETH +0.031, BTC +0.042, SOL +0.055 — all
positive, all LARGER than TOD-VS increments. Status: PREFERRED +
PERMUTATION + CROSS-ASSET-OOS; full battery (quarterly blocks, horizon,
move/Amihud controls) pending. Note: H-A may outperform the validated
structure — adjudicate, don't assume.

Joint model HAR+|r|+Amihud+H-A+VS^TOD (fit-2024 → OOS 2025+): t-stats
H-A +9.7/+9.5/+6.8 AND VS +15.6/+11.9/+12.9 (BTC/SOL/ETH) — both survive
jointly. OOS: BTC joint 0.3414 beats VS-alone 0.3397 and H-A-alone 0.3375
(complementary); SOL joint 0.3510 < VS-alone 0.3537 and ETH joint ties
VS (redundant there). H-A quarterly replication: 15/18 positive; 25Q3
misses on ALL THREE assets (BTC −0.031, SOL −0.057, ETH −0.008) — same
quarter VS missed (SOL −0.062): hostile-quarter effect, structure-wide.
Status: COMPLEMENTARITY-PARTIAL (BTC) / REDUNDANT (SOL/ETH); H-A
quarterly 15/18; 25Q3 filed as universal hostile block.
25Q3 diagnostic (candidate-independent descriptors): lowest return
dispersion, weakest vol clustering ac1(|r|) (BTC 0.22, SOL 0.18 vs 0.30+
elsewhere), lowest Amihud impact — calm efficient quarter hostile to ALL
history-based vol prediction, not to H-A/VS specifically. Descriptor only,
never a model term.

## 25. Descriptor battery (frozen for next flagged quarter)

Pre-registered instrument: volConc (top-decile bar share), xcorr (mean
pairwise 15m-return corr), Amihud median, volvol (std logVol), ac1r,
ac1v, market beta, idiosyncratic-vol share. 25Q3 calibration
(descriptive): BNB stands out — lowest xcorr (0.653), highest idio
(0.461), lowest beta ex-BTC (0.598); XRP highest ac1v (0.694);
Amihud-Q3: BTC 6.5e-11, SOL 3.5e-10, ETH 1.0e-10, BNB 8.0e-10, XRP
5.2e-10, DOGE 1.1e-9. No selection among descriptors until the next
flagged quarter — the battery waits frozen.

## 26. Minimum H-A grammar audit (for trading-sandbox integration)
Verified: discovery top pick log(RollMean(V)^−2) ≡ −2·log(RollMean(V)) to
7e-15 — one model, two spellings; canonical H-A = log(RollMean_40(V)).
Minimum grammar: {volume variable, rollmean(W=40, causal), log,
unary(rolling) composition, +−×/ search support}. NOT required: rollstd,
z-scores, TOD/group-by, regime terms, VS, cross-asset descriptors.
Gap audit: sandbox FIN_TERMS (12 mechanical state terms on x/v/u) has NO
volume variable and NO rolling operator — H-A inexpressible there today.
Proposed (not implemented, pending approval + owning session): add volume
channel + rollmean term to the sandbox slate ("vol", "rollmean_vol"?),
then let the Tier pipeline adjudicate rather than hand-bless. Rationale
recorded: engine preferred level-over-surprise from ordinary primitives,
which is the clean story — no market-specific operator needed.

IMPLEMENTED (approved): rollmean/rollstd added to BOTH open-discovery L2
operator sets (market/sweep.py, sandbox/real_market_l2_sweep.py);
multiterm branches untouched. FIN_TERMS slate deliberately NOT extended:
rolling terms are delay-differential (unintegrable as ODE RHS via
solve_ivp) and repo daily bars carry no volume channel — a volume term
there would be dead text. Rolling series CAN enter dynamics via the u
drive (existing u/sat_u terms) when data exists. Volume variable NOT
added (no data to supply it). Smoke: W3_COVID/sp500 end-to-end 15s,
verdict insufficient_evidence, top r_lag — honest non-selection.
Ruff clean. Uncommitted.

## 27. Volume sandbox wired to frontend (live)
`volume_market` registered in /api/sandbox/list (13 entries), runner in
/api/sandbox/run (asset passthrough, COMPOSITE→BTC), dashboard Sandbox
tab auto-populates + asset row shown. Server restarted; refresh the tab.

Promotion Gates block (auditable, per-gate ✓/✗ with details) + universe
note + "Promoted Structure / Model" rename (sandbox panel only).
Bug caught live: gate R² first computed uncentered (+0.69) contradicting
its own leaderboard row (−0.32) — fixed to centered target-variance
derivation identical to the rows. Refresh to see the gate table.

`volume_market` registered in /api/sandbox/list (13 entries), runner in
/api/sandbox/run (asset passthrough, COMPOSITE→BTC), dashboard Sandbox
tab auto-populates + asset row shown. Server restarted; live list
verified. First live-equivalent run (BTC): verdict INSUFFICIENT
(V1 RMSE 0.01206 vs V0 0.01159 on the DAILY spec — correctly refuses;
this is the daily-resolution volume spec, NOT the validated 15m
TOD-surprise — different resolution, different verdict, no contradiction).

Gate fix (post-first-live-run): adapter initially promoted on relative
improvement alone (MATIC: delta +0.25 with R² −0.57, whiteness FAIL →
wrongly PROMOTED). Rule strengthened: UNIQUE_SUPPORT requires delta>0
AND perm_p<0.05 AND V1 R²>0. MATIC/BTC now correctly refuse. Server
restarted live; refresh the tab.
Gate-selector fix: breakdown evaluated V1 (hardcoded) instead of the
best challenger — now selects lowest-RMSE non-baseline row with ITS
delta/perm/R² (BTC: V2 +0.001 / p=0.695 / −0.188 → refuse). Universe
note now boxed Requested-vs-Actual. Server restarted again; refresh.

CONTROL-VOLUME-01 (filed, not an H-A contradiction): dashboard volume
benchmark tests contemporaneous log(volume) vs frozen trailing-vol
baseline — no rolling-state operator in its candidates (V0: RMSE 0.0116,
R² −0.22; V1: 0.0121, −0.32; rival gap 0.00046 ≤ τ). Correct abstention;
volume-LEVEL ≠ volume-STATE. Standing decomposition: raw level → no
promotion; rolling state (H-A) → validated; TOD surprise → narrow
replicated; rolling surprise → rejected. Dashboard caption fixed per
benchmark (was leaking crypto-grammar text into volume runs).

## 28. Cross-market replication (PAXG gold, SPX/NDX 1h)

PAXG 15m (fit-2024 frozen → OOS 2025+): V0=0.281, H-A=0.122, Δ−0.159;
quarterly 1/6 (26H1 +0.014 only). Permutation recount 0/300. SPX 1h
(fit-25H1 → OOS): Δ−0.168. NDX 1h: Δ−0.022. Verdict: FAIL on gold and
equities — Outcome C (crypto-specific or crypto-regime-specific).
Nuance: PAXG real-column beats shuffled-column 300/300 despite Δ<0 —
alignment information with miscalibrated sign (parameter, not structural,
failure; cf. PARAMETER_UNSTABLE theme). FX spot untested (no centralized
volume); FX futures untested (no free intraday source).

Sign-reversal test (same H-A feature, sign free, fit-train → frozen OOS):
PAXG POS (t=+27.0) Δ−0.159; SPX POS (+4.2) Δ−0.168; NDX POS (+3.9)
Δ−0.022. REJECTED: no asset fits negative — direction universally
positive in-sample; failure is transfer, not sign. Mature-market
absorption (negative coefficient) is not what any fit produces.

## 29. H-A memory-scale transfer (W family, frozen protocol)

W ∈ {20,40,80,120,150,200}, fit-train frozen → OOS, ΔR² vs persistence:
BTC 15m: +0.040/+0.042/+0.046/+0.051/+0.043/+0.045 — flat plateau, no
window sensitivity (robust). SOL: +0.053/+0.055/+0.057/+0.056/+0.057/+0.061
— plateau. PAXG: −0.096/−0.159/−0.278/−0.375/−0.359/−0.422 — monotone
deterioration; longer memory strictly worse. SPX 1h: −0.453/−0.168/−0.014/
+0.016/+0.034/+0.081 — crosses zero 80–120, climbs to 200. NDX 1h:
−0.104/−0.022/+0.025/+0.070/+0.080/+0.106 — crosses zero 40–80.
Verdict: characteristic timescale τ_V ASSET-CLASS-DEPENDENT — crypto
~10h plateau (any W works), equities need 120–200h (~1–8 trading days),
gold fails at all W (different story, not a timescale problem).

Fine grid W=80–280 (1h): SPX −0.014/+0.004/+0.016/+0.017/+0.058/+0.081/
+0.081/+0.013/−0.184; NDX +0.025/+0.055/+0.070/+0.075/+0.105/+0.130/
+0.106/+0.008/−0.072. PEAK at ~180–200 bars (~1–1.5 trading weeks),
collapse beyond 240 — bounded characteristic timescale, not endless
climb. Taxonomy: crypto W*∈[20,200] plateau (40 canonical, unrevised);
equities W*≈180–200; gold none. H-A(W) family with asset-dependent W*.

## 30. W-shape × descriptor map (hypothesis-generating, n=9, no inference)

W-profiles (fit frozen → OOS): ETH flat +0.031..+0.037; BNB flat
+0.075..+0.084; XRP weak-decaying +0.009→+0.002; DOGE flat +0.065..+0.079.
2024 descriptors: flat-group ac1vol 0.80–0.81, volConc 0.34–0.43 (BTC/SOL/
ETH/BNB); XRP ac1vol 0.888/volConc 0.542; DOGE ac1vol 0.911/volConc
0.570/volvol 1.417 yet FLAT-positive; PAXG ac1vol 0.524 (lowest), ac1|r|
0.508 (highest), Amihud 3e-8 (highest). No single descriptor separates
flat/peaked/absent (DOGE breaks volConc, XRP breaks ac1vol); session
structure (24/7 vs 6.5h) separates peaked from flat but not PAXG.
Pre-registered determinant search awaits more assets/periods — map filed,
no pattern promoted.

§30 status PERMANENT NEGATIVE: no univariate separator (DOGE falsifies
ac1vol/volConc/volvol explanations; PAXG falsifies session structure;
XRP weak-decay fits none cleanly). Multivariate/interaction explanations
reserved for future preregistered data (new assets + periods) — never to
be fit on these nine points. Determinant of transfer: UNRESOLVED.
Determinant search STOPS here; PAXG retained as distinctive failure
specimen, not an explanation.

## 31. Canonical family (supersedes fixed-W=40 claim)

Final form — structural family, not single equation:
  RV̂_{t+1:t+H} = β0 + β1·TV_t + β2·|r_t| + β3·Amihud_t
                 + β4·log(RollMean_W(V_t)),  H_A(W) = log(RollMean_W(V)).
Canonical instantiations: crypto W≈20–200 plateau (W=40 canonical,
unrevised — plateau means no reason to move); SPX W≈160–200; NDX
W≈180–200; PAXG none supported. W is asset-class-dependent memory scale,
not universal constant. Status: BOUNDED ASSET-DEPENDENT MEMORY FAMILY;
universal law rejected; mechanism unresolved.

Novelty calibration (filed, not claimed): ingredients (HAR persistence,
|r|, Amihud, volume, rolling means, logs) are all standard finance —
NO claim of equation novelty without prior-art review. Defensible claim:
autonomous discovery (engine preferred H-A unprompted) + predefined
OOS/permutation/replication/horizon/window/cross-market falsification
sequence with domain of validity visible (crypto plateau, equities
W*≈180–200, gold absent). Novelty, if any, is in the validated
discovery trajectory, not the symbols.

Prior-art check (web, 2026-10-03): log-average-volume-augmented HAR
exists in literature (1/5/22-day log volume averages in HAR-RV, OOS on
41 equities; also volume→RV forecasting broadly). So the FORM
log(avg-volume)-in-volatility-model is published at daily scale.
H-A's remaining distinctive claims: autonomous (undirected) discovery,
intraday 15m resolution, per-asset-class W won by ablation, crypto
coverage, and the full falsification ledger. "Unknown equation" is OFF
the table permanently; "independently discovered + validated trajectory"
is the surviving novelty claim.

Family-as-specified test (joint HAR + H-A(W*) per asset): BTC +0.005,
SOL +0.008, ETH +0.006, BNB +0.020, DOGE +0.015 (W=40); XRP −0.014,
PAXG −0.031 (fail). SPX/NDX 1h at W=180: sick baselines (V0 −0.50/−0.12
on short-train/turbulent-test split) → BASELINE_UNHEALTHY, no inference
— setup mismatch with §29's cleaner V0, not a family verdict there.

## 32. Transformation battery (frozen H-A form)

OOS ΔR² over V0 (fit-2024 → 2025+, 15m): H_A linear +0.031/+0.036/+0.022
(BTC/SOL/ETH); H_A² −0.001/−0.062/−0.008; |H_A|,sqrt ≈ 0/neg; exp
+0.018/+0.005/+0.000 (weaker); sin +0.015..+0.018 all three; cos ~0/neg;
tan +0.019..+0.032 all three. Orthogonalized beyond linear H-A: sin/cos/
tan/sq ≈ 0 everywhere (BTC +0.0003 ×4 = collinearity dust; SOL sin −0.068
destabilizes; ETH ≤0) — periodic positives were linear-component
stowaways (sin≈x−x³/6 carries the linear term). Verdict: LINEAR H-A
carries everything; no saturating/periodic/quadratic structure beyond it.
Functional form closed for the preregistered/tested battery (not
universally — infinitely many transforms exist untested): plain
linear-in-log-volume-state.

## 33. H-A extension battery (all incremental over base+H-A, OOS 2025+)

ΔH-A transitions d1/d16/d96: all ≈0 or negative (best ETH-d1 +0.0015) —
LEVEL SUFFICIENT, transition adds nothing. Second scale hA180:
+0.005/+0.002/−0.002 — dead on crypto (plateau confirmed from the other
side). H-A×Amihud: +0.005/+0.015/+0.001 — SOL +0.015 noted as
unconfirmed single-asset hypothesis, not promoted. H-A×TV: ~0/neg.
Verdict: level-only H-A stands; no transition, second scale, or
interaction earns inclusion. SOL-HAxAmih filed as hypothesis.

## 29b. Single-name equities (AAPL/MSFT/NVDA 1h; Yahoo has 2025+ only)

W=40: MSFT V0=+0.044 healthy, Δ+0.036, perm 0/200 → SUPPORTED (weak);
AAPL V0=−0.255, NVDA V0=−0.411 → BASELINE_UNHEALTHY, no inference
(permutation inverted there: shuffled beats real — poison-point regime
break, same signature as PAXG sign case). W=180: all fail (AAPL −0.005,
MSFT −0.223, NVDA −2.069). Caveat: half-year train (Yahoo 1h starts
2025) vs full-2024 train behind §29 index results — train-length
confound on the single-name W=180 failure; index-level W*≈180 stands on
its own protocol. Single names do not extend it on this evidence. The
memory-scale hypothesis CONFIRMED where it predicted rescue (equities),
REFUTED where it predicted nothing new (crypto flat) or rescue (gold).

## 34. Memory-spread direction test (frozen long/flat on S>0, daily, 5bps)

S_t = H_A(40) − H_A(180) (scale-free log-ratio); long iff S>0 else flat;
2025+ unseen; new assets BNB/DOGE + PAXG control. BNB net −47.7% vs BH
+11.0% (Sharpe −0.83, perm 165/200 — systematically worse than chance);
DOGE −88.7% vs BH −122.9% (−0.85, 120/200); PAXG +9.1% vs +45.5% (+0.25,
84/200). Verdict: spread-as-direction FAILS — long on short>long memory
loses outright (BNB/PAXG) or revanche-beats a collapsed BH by holding
cash (DOGE). No inverse-signal rescue (post-hoc flip = p-hacking).
Memory term-structure is not a directional signal under this frozen rule.

## 35. Status labels (calibrated)

H-A = "Conditional volatility-state predictor — qualified empirical
support; no validated directional/trading content; no causal-mechanism
claim." Scope-bound: within tested asset classes/horizons/controls,
incremental over TV+|r|+Amihud. Single-scale claim narrowed: adequate
within crypto spec; window may vary by asset/regime/sampling/target.
Inverse-sign rescue archived as FUTURE-preregistered-only (never
post-hoc). Q3/BNB noted as regime-boundary evidence, not mechanism.

## 36. Horse race vs GARCH(1,1) (frozen, OOS 2025+, floored QLIKE)

MSE: H-A wins both (BTC 8.35e-11 vs 9.43e-11; SOL 1.35e-9 vs 1.48e-9).
QLIKE (5%-median floor; first attempt exploded on near-zero squared
forecasts — construction bug caught and fixed): H-A wins both (−11.35
vs −11.02; −10.07 vs −10.03). DM (loss_G − loss_HA, positive favors
H-A): +36.7 / +3.5 — no HAC correction (overlapping H=16 horizons
inflate t; treat as directional, not calibrated). Verdict: H-A beats the
industry workhorse on both proper scores; DM magnitude overstated,
sign robust.

## 37. Diagnostic dossier (exploratory, mined sample — not confirmatory)

Within-quarter demeaning: R² −0.77 (BTC) / −0.99 (SOL) — relationship
does NOT survive removal of quarterly level differences: BETWEEN-QUARTER
level effect, not within-regime co-movement. Reconciles with positive
quarterly frozen deltas (global coefs exploit cross-quarter alignment).
Lead-lag: corr(hA,past) +0.67/+0.65 > contemp +0.45/+0.49 > future
+0.44/+0.48 — strongest with PAST RV: lagging-state-proxy signature.
Matched placebos rank/z: +0.024/+0.031, +0.035/+0.036 ≈ H-A's own deltas
— ordinal/state information, NOT log-form-specific evidence. Tail AUC
(top-quartile vol) 0.764/0.774 — good upper-tail discrimination.
Target robustness (full-model R²): sum|r| +0.33/+0.35, max|r| +0.19/+0.20
— general across vol definitions (persistence included). Net: three
adverse (demeaning, lead-lag, placebo-equivalence) vs three supportive
(tail, targets, frozen replication). The between-regime confound now
bites the validated structure itself — filed as the leading rival
explanation, not a footnote.

## 38. Identification ladder (demeaned OOS; fit on demeaned train)

H3 (regime mean) = 0.000 by construction. BTC: H1-struct +0.164,
H2-rank +0.144. SOL: H1 +0.150, H2 +0.125. Ordering H3 < H2 < H1 on
both assets — structural form beats rank-only beats regime-only.
Reconciles §37 (global-frozen coefs fail demeaned test): pooled OLS
compromises between within-slope and between-level; refit within the
demeaned frame, the within relation is STRONG (+0.15). New structural
insight: two-component panel (regime level α_q + within slope β) —
pooled single-coefficient fits serve neither. Caveats: test-quarter
means used in demeaning (identification, not deployability); mined
sample (exploratory). Next: pre-register panel estimation on untouched
data with quarter effects + within slope jointly estimated.

Extended ladder: ETH H1 +0.152/H2 +0.135; BNB +0.157/+0.145; XRP
+0.188/+0.147; DOGE +0.155/+0.130 — H3<H2<H1 on all four, magnitudes
uniform (+0.13..+0.19). SPX H1 +0.003/H2 −0.098; NDX −0.010/−0.087 —
no within relation on equities. Two-component panel replicates 6/6
crypto, 0/2 equities: crypto-specific structure (or crypto-regime
persistence again — same qualifier as every crypto-only finding).

## 39. Causal quarter-effects ladder (expanding means, past-only)

M0=0.000 both. BTC: M1-rank +0.148, M2-struct +0.173, M3-both +0.185 —
ordering M0<M1<M2<M3, structural wins. SOL: M1 +0.080, M2 +0.054,
M3 −0.013 — rank wins, joint harmful. Verdict per decision tree: MIXED —
structural advantage replicates on BTC, fails on SOL. Rank-only is the
robust survivor across both; structural edge is BTC-specific.

## 40. Ladder extension (7 more alts, demeaned OOS)

ADA H1 +0.109/H2 +0.085; AVAX +0.077/+0.061; LINK +0.101/+0.090; NEAR
+0.236/+0.196; ATOM +0.092/+0.076; UNI +0.082/+0.066; LTC +0.135/+0.104
— H1>H2 on all seven (deltas +0.012..+0.040). Pooled: H1>H2 on 15/15
assets; H1>0 on 14/15 (only NDX −0.010 misses). Conditional reading:
given H2>0 (13 assets), H1>0 in 13/13; given H2<0 (SPX/NDX), 1/2. The
ordering claim is near-universal; zero-crossing tracks the H2 benchmark
level, not asset identity. MATIC excluded (delisted Sep-2024, no OOS).

## 41. Rank-primary ladder (causal quarter mean + trailing-252d rank)

M0 (prior-quarter mean RV): BTC −0.098, SOL −0.077, ETH −0.000 —
regime means ANTI-persist; level prediction dead. M1-rank: +0.144/+0.142/
+0.135. M2-struct: +0.182/+0.118/+0.156. M3-both: +0.151/−0.187/+0.124
(joint harmful on SOL). Verdict: rank model SUPPORTED strongly on all
three; structural beats rank on BTC/ETH (+0.03/+0.02), loses on SOL
(−0.02). Regime LEVEL carries nothing (M0 ≤ 0 everywhere) while
within-regime ORDER carries everything — level-vs-order separated
cleanly for the first time.

Extended §41 (6 more crypto + SPX/NDX 1h; increments over M0): BNB
M1 +0.223/M2 +0.350/M3 +0.335; XRP +0.187/+0.129/−0.038; DOGE
+0.240/+0.293/+0.245; ADA +0.098/+0.076/+0.053; AVAX +0.091/+0.129/
+0.121; LINK +0.115/+0.171/+0.153 — M2 best or near-best on 5/6 (XRP:
M1 wins, M3 harmful). SPX M1 +0.900/M2 +1.660/M3 +1.435 and NDX
+0.574/+0.953/+0.789 look spectacular until absolutes: M0 −3.59/−1.44,
so all models remain R²<0 — prior-month mean is disastrous at hourly
resolution, making any improvement look huge. Ordering M0<M1≤M2 holds
everywhere (11 assets); positive absolute levels only in crypto.
M0-dead + ordering-alive = regime means anti-predict universally.

## 42. Rank-primary rolling-forward (gate rule pre-registered)

Protocol: train 180d → OOS next 20d, 27 windows through 2025–2026H1,
M0(V)/M1(R)/M2(S)/M3(R+S); promote S iff mean Δ(M3−M1)>0 AND sign-p<0.05.
BTC: M0 −0.093/M1 −0.066/M2 −0.037/M3 −0.040, gate +0.025 p=0.70; SOL:
−0.112/−0.152/−0.085/−0.083, +0.069 p=0.12; ETH: −0.086/−0.097/−0.089/
−0.074, +0.023 p=0.12. GATE CLOSED on all three (positive means,
insignificant signs). Levels all-negative: short rolling windows at 15m
are a far harder task than full-year frozen fits — protocol difference,
not contradiction with §41. Rank-primary retained, S unpromoted. Status:
RANK-ONLY-ROBUST; STRUCT-ADVANTAGE-UNCONFIRMED (1/2).

## 43. Rolling gate extended (BNB/XRP/DOGE 15m, SPX/NDX 1h)

BNB (n=14): M0 −0.157/M1 −0.112/M2 −0.023/M3 −0.023, gate +0.089 p=1.00;
XRP: −0.095/−0.054/−0.040/−0.044, +0.011 p=0.79; DOGE: M0 +0.081 (only
positive baseline anywhere) with M1/M2 BELOW it (+0.021/+0.096) —
level beats rank and structure there. SPX/NDX 1h (n=11): all models
deeply negative (M0 −1.84/−2.48), gate +0.07/+0.17 p=0.55 — short-window
rolling at hourly resolution sees nothing at all. GATE CLOSED on all
five (8/8 total with BTC/SOL/ETH). Rank-primary retained everywhere; S
unpromoted everywhere.

## 44. Final equation (rank-primary, simplified — gates closed on S)

r_{t+1} = β0 + β1·R_t + ε, R_t = trailing-252d percentile rank of logVol
∈ [0,1]. Confirmatory run (fit-2024 frozen → OOS 2025+, 15m): BTC R²
+0.106, SOL +0.087, ETH +0.140 — all positive. Hierarchy final: M0
level REJECTED everywhere; M1 rank RETAINED everywhere tested; M2
structure NOT PROMOTED (8/8 gates closed); M3 gate closed. Session's
discovered equation is the rank law; structural term awaits an
independent pre-registered OOS opening that never came.

## 28. Rank equation in the sandbox (V2 third candidate, live)

V2 = 252-day percentile rank of daily RAW volume ∈ [0,1], causal —
distinct from §44's regime-order rank, H-A rank, and 15m rank
constructions; never conflate. run_volume_adjudication fields V0/V1/V2
with three-way gate (best challenger needs delta>0, own perm<0.05,
R²>0). Live: BTC all refuse (V2 ≈ V0, R²<0); MATIC refuses (V1 best
RMSE but R² −0.41). Server restarted; refresh to see three rows.
Rank-primary lives in the sandbox as a standing challenger,
unpromoted until it clears all three bars.

## 45. Vol-timing translation on 2026 (self-scaled target, rolling-180d)

First attempt hardcoded target 0.02 vs forecasts ~0.003 → degenerated to
fixed 2× leverage (turnover ~2, results ≈ 2×BH) — discarded as failed
execution, recorded so the ledger shows the misstep. Rerun with causal
trailing-median target: BTC net +3.5% vs BH −3.4% (Sharpe +0.12, maxDD
40%, turn 59); SOL −8.6% vs −4.4% (−0.20, 81%); ETH −4.0% vs −10.2%
(−0.10, 69%). Beats BH 2/3 on net, Sharpe noise-grade throughout.
Directional buy/sell untested by design (no validated directional
equation exists to translate). Translation: mixed-to-weak, not economic
validity. Status: VOL_TIMING_FEASIBLE, NOT VALIDATED — rank-vol forecast
converts to a causal exposure rule without degenerating (attempt-1
fixed-leverage failure discarded); 2/3 assets beat buy-hold with
noise-grade Sharpe and severe drawdowns. Architecture separation stands:
RV̂ → exposure sizing (this section) vs directional equation (absent).

## 46. Walk-forward QLIKE horse race (90d-train/30d-test, 2025+, n=51k)

V0 vs +logVol vs +rankVol vs +H-A, QLIKE (lower better): BTC V0 −5.268,
logV/rankV −5.279, H-A −5.273 (all beat V0; perm 0/200); SOL V0 −4.675,
logV/rankV −4.680, H-A −4.677 (perm 0/200). CONTRAST with frozen-2024
protocol (logVol/rank null there): walk-forward refit extracts value
where frozen fails → coefficients drift by regime; adaptivity helps AND
confounds (refit-chasing, cf. refit-artifact exhibit). Verdict:
volume-vol relation exists under adaptive estimation; frozen-transfer
remains the hard bar it keeps failing. Both true; protocol determines
conclusion — filed as estimation-dependence, not contradiction.

## 47. Overlap validation (non-overlapping 5-day targets, BTC)

Same V0/V1/V2 benchmark on every-5th-day targets: whiteness 0.050/0.098/
0.092 (PASS) vs 0.785/0.792/0.789 overlapping — the |ρ₁|≈0.79 failure was
overwhelmingly MECHANICAL (4/5 shared target days), not model
misspecification. Verdicts unchanged (V1/V2 still lose: R² −0.37/−0.23 vs
V0 −0.23) but now with white residuals: pure lack of signal, cleanly
measured. Rule filed: whiteness gate needs overlap-aware handling for
multi-day-horizon targets (compare non-overlapping control before
interpreting |ρ₁| as scientific evidence).

## 48. Walk-forward challenger separation (logV vs rankV vs H-A)

Prediction correlations: corr(logV-preds, rankV-preds) = 0.994 (BTC) /
0.996 (SOL); corr(logV, HA) = 0.94/0.96. Walk-forward QLIKE (n=40k):
all volume forms beat V0 by ~0.004–0.012; combined models add nothing
(logVrank ≈ logV). Verdict: logV ≡ rankV INFORMATIONALLY under refit —
monotone transform absorbed by OLS; no challenger separable, none
promoted. Adaptive family stands (all beat V0); single-challenger
promotion refused. §46 tie explained: structural equivalence, not
coincidence.
Qualified: logV≡rankV holds FOR this dataset/window/spec (0.994/0.996),
not as a general OLS identity. §46 equation = adaptive family with Z_t
∈ {logVol, rankVol, H-A}, refit per 90d window. Status:
ADAPTIVE_FAMILY / NO_UNIQUE_PROMOTION. Robust object may be the adaptive
volume-vol relationship itself, not any functional representation.

## 49. Vol-direction classification (26H2, adaptive family)

Up/down next-16b vol vs trailing vol, OOS 2026H2 (n≈9k each): BTC acc
0.644/bal 0.702 (base 0.518); SOL 0.673/0.717 (0.516); ETH 0.634/0.724
(0.518). Balanced accuracy +0.18–0.21 over base on all three — strongest
directional-adjacent result on file. Asymmetric skill: down-precision
~0.82 vs up-precision ~0.58 (better at predicting calm than storms).
Caveats: full adaptive model (H-A attribution not isolated — persistence
alone may carry it); mined sample (exploratory); predicts VOLATILITY
direction only — buy/sell RETURNS remain untested by design (no validated
directional equation exists).

## 50. Classifier ladder (C0 persistence / C1 V0 / C2 V0+H-A, 26H2 bal-acc)

BTC: C0 0.712 > C1 0.711 > C2 0.704. SOL: C0 0.724 > C1 0.657 > C2 0.630.
ETH: C0 0.700 ≈ C1 0.705 > C2 0.693. Adding controls then H-A
MONOTONICALLY DEGRADES the persistence-only classifier on all three —
§49's directional signal is persistence alone; H-A adds nothing to
direction (consistent with every directional null on file). Status:
PROMISING_EXPLORATORY for vol-direction classification generally;
H-A-SPECIFIC directional content: NOT SUPPORTED.

## 51. Classifier ladder extended (11 more assets, 26H2 bal-acc)

C0/C1/C2: BNB .681/.685/.678; XRP .643/.650/.607; DOGE .661/.621/.658;
ADA .668/.662/.667; AVAX .658/.620/.624; LINK .666/.663/.682; NEAR
.625/.643/.678; ATOM .647/.594/.703; UNI .668/.684/.695; LTC .657/.649/
.649; PAXG .671/.673/.718. C2 beats C0 on 5/11 (LINK/NEAR/ATOM/UNI/PAXG;
NEAR +0.053, ATOM +0.056, PAXG +0.047), loses-or-ties on 6/11. Combined
with §50 (0/3): H-A directional value is ASSET-HETEROGENEOUS — helps on
5/14, neutral-to-harmful on 9/14. Neither promotable universally nor
dismissible entirely; the heterogeneity itself is now the finding.

## 52. Cross-asset transfer matrix (TV+H-A coefs i→j, dBA vs C0_j)

Full-matrix: BTC row positive 12/13 (up to +0.215 NEAR); PAXG column ALL
negative (−0.03..−0.06, nothing transfers TO gold); PAXG row bimodal
(+0.20–0.23 to BTC/ETH/BNB/XRP/DOGE, −0.28–0.30 to ADA/AVAX/LINK/NEAR/
ATOM/UNI/LTC). Control (TV-only transfer): +0.154–0.201 for EVERY
source incl. PAXG — universal persistence floor. C_RV/C_HA with BTC:
0.54–0.89 (high common correlation). Verdict: transfer ≈ persistence
floor + deviations; H-A-specific transfer NOT isolated (full ≤ TV-only
on average); PAXG isolation is the robust cross-asset finding (nothing
in, erratic out). Shared-mechanism vs common-factor: unseparated —
persistence explains the floor, H-A adds no proven transfer increment.

## 53. Incremental transfer matrix (properly standardized; ΔΔBA)

dDBA = full-transfer minus TV-only-transfer per ordered pair (positive =
H-A adds beyond persistence). BTC row: positive 13/14 (+0.002..+0.055,
only self −0.007). ETH/BNB rows: mostly positive (+0.01..+0.06).
XRP/DOGE/ADA rows: negative on majors, mixed elsewhere. PAXG column:
BTC/ETH/BNB transfer IN positively (+0.04..+0.06), rest ~0/neg; PAXG row
erratic (+0.095 BTC, −0.500 SOL). Verdict: H-A-SPECIFIC transfer
SUPPORTED for BTC/ETH/BNB sources, absent for XRP/DOGE/ADA sources,
mixed for PAXG. (Prior unstandardized attempt produced −0.4 degeneracy
blocks — discarded as scale-mismatch artifact, recorded.) Transportability
is SOURCE-dependent: some assets' H-A coefficients travel, others' don't.

## 54. Source transportability scores + characteristics

T (mean dDBA) / P (frac>0) over 13 targets each: BTC +0.028/1.00, ETH
+0.023/1.00, BNB +0.022/0.77, SOL +0.002/0.69, ATOM +0.002/0.54;
XRP −0.117/0.00, DOGE −0.124/0.08, ADA −0.096/0.23, UNI −0.064/0.31,
LTC −0.041/0.23, PAXG −0.056/0.31, NEAR −0.040/0.62, AVAX −0.001/0.69,
LINK −0.002/0.62. Transportable: BTC/ETH (+BNB); non-transportable:
XRP/DOGE/ADA (P=0–0.23). Correlations: C_RV-meancorr +0.18 (weak),
beta −0.43 (lower-beta transports better — strongest lead), Amihud ~0.
−0.500 cells audited: genuine degenerate one-sided regimes
(all-UP predictions both sides; TV-only 0.74 vs full 0.24), NOT a code
floor — but the comparison they encode is real. Status: transportability
is source-conditional; beta-negative association is hypothesis-generating
(n=14), not a finding.

## 55. Beta-stratified prospective replication (fit-25H2 → test-26H1)

Beta groups frozen from train period only. LOW-beta sources: mean dDBA
−0.022, sign-p 0.92. HIGH-beta: −0.025, sign-p 0.03 (leans negative).
∂T/∂β<0 NOT established — neither group positive; low group indistinguishable
from chance. The §54 beta association (−0.43, n=14 exploratory) does NOT
survive prospective group testing. Status: BETA-TRANSPORT HYPOTHESIS
FAILED FORWARD. Transportability differences remain described
(BTC/ETH/BNB travel, XRP/DOGE/ADA don't) but unexplained — no verified
determinant.

## 56. Source-identity replication (fit-25H2 → test-26H1, all + non-degenerate)

REPLICATED: BTC (+0.030, P=0.92), SOL (+0.006, 0.69), ETH (+0.009, 0.85),
AVAX (+0.006, 0.69). NON-REPLICATED: rest incl. BNB (−0.034, was
transportable) and PAXG (−0.146, nondegen 0/13 — every PAXG transfer
one-sided). Non-degenerate split agrees directionally throughout (BTC
nondegen n=1/13 only — nearly all its transfers one-sided yet positive
mean; reported, not rescued). Verdict: source identity partially
replicates (BTC/ETH hold, SOL/AVAX join, BNB drops, PAXG sinks) —
transportability is itself regime-dependent, not a fixed source trait.

## 57. Regime-stratified transport (sync moderator test)

Monthly mean-dDBA (182 pairs) vs prior-month market sync, 21 months:
corr = +0.127 (predicted NEGATIVE) — hypothesis NOT supported.
Deeper problem: sync range is [0.60, 0.89] always-high; crypto 15m has
NO low-sync regime to test against (range restriction kills power).
Mean dDBA negative in 20/21 months. Verdict: sync-moderator rejected as
formulated; the test lacked a real low-sync condition. Sync hypothesis
unresolved-for-lack-of-variation, not confirmed-absent.

## 58. Stress-regime transfer (frozen terciles from 2024)

All-pairs dDBA: LOW −0.058 / NORMAL −0.051 / HIGH −0.017 (n=196 each)
— monotone toward zero but negative everywhere. BTC-source: LOW +0.013,
NORMAL −0.006, HIGH −0.001. Verdict: stress does NOT rescue transfer;
only BTC-in-LOW is positive (+0.013, marginal). No regime shows
systematic positive H-A transfer. Stress hypothesis FAILS as rescue;
trend regime next per sequence (then stop if it fails too).

## 59. Trend interaction (LAST moderator test — search stops after this)

Frozen 2024 terciles on trailing-20d return (q1=−0.047, q2=+0.071);
dDBA by regime, all pairs: DOWN −0.035 / FLAT −0.042 / UP −0.016
(n=196 each). All negative; UP least-negative by a margin that changes
nothing (still <0). No systematic trend interaction. MODERATOR SEARCH
CLOSED: beta ✗, sync untestable ✗, stress ✗, trend ✗. Standing result:
predictive structure real in places; transportability mechanism
unidentified — and stays unidentified until fresh regimes, not fresh
moderators, arrive.

## 60. tanh saturation (adapted — no cubic singularity exists in this ledger)

Note: proposed §60 targeted a different H-A ((P−P̄)/(P̄σ) z-score);
this ledger's H-A = log(RollMean(V)) has no poles/singularity. Tested
the transferable part (tanh saturation, the one transform §32 missed):
+tanh(H-A) vs linear H-A OOS — BTC −0.0087, SOL −0.0004, ETH −0.0028
(all ≤ linear). Saturation adds nothing; linear stands. H-A remains
linear-in-log-volume-state with no bounded-transform improvement.

## 61. Ex-ante orthogonalization (H_A residualized on V0, block-perm gate)

+H_A-perp deltas IDENTICAL to plain H-A (+0.0308/+0.0363/+0.0222 —
reparameterization invariance: same column space, same fit); block
permutation 0/200 all three (shuffled residual never wins). So
orthogonalization changes nothing numerically — it is interpretive
machinery (isolates the marginal component for attribution), not an
empirical improvement. Gate passes (perm + ΔR²>0). RLS/adaptive
estimation and semi-variance splits NOT run (no matching object in this
ledger: H-A has no σ³ denominator or signed-deviation form here).

## 62. Pole-pathology diagnostics (ma_dev/vol³ pair; H-A leverage)

ma_dev/vol³ range ±1e9 (poles both sides); rank-corr(f,1/f)=+0.49, NOT
−1 (sign break at MA crossings — the −1.0 claim needs positive domain).
First OLS attempt CRASHED SVD convergence (non-finite standardization
from pole values) — pathology manifesting as numerical breakdown.
Robust-scaled rerun: V0=0.2370, +f=0.2369, +finv=0.2370, winsorized
+0.000/−0.002 — pole pair contributes NOTHING once stabilized. H-A OLS
Cook's: max 0.029, 0.1% points >1% (n=35k) — no leverage problem.
Verdict: reciprocal test confirms pole pathology for denominator forms;
H-A clean. Pole-bearing equations rejected as numerics, not mechanisms.

## 63. Next program: epistemically-constrained open discovery (filed, pilot below)

Flagship: APEDP epistemically-constrained open discovery. Priority order:
(1) L2 genuine open discovery (truth outside initial grammar) + pilot
below; (2) Unknown-Unknown epistemic benchmark (recoverable/ambiguous/
misspecified/novel/non-mechanistic/adversarial tiers with status output);
(3) prediction-vs-validity tournament (PySR/AI-Feynman/LLM-SR/neural vs
APEDP on best-predictor≠best-hypothesis datasets); (4) causal dynamical
discovery (direction + confounders + interventions); (5) active experiment
selection (next most-discriminative measurement); (6) real physical/
materials system (cleaner claim than finance). Thesis: Discovery →
Validation → Adjudication → Uncertainty → Experiment → Revision (not
Data→Best Equation). No metric-optimization races.

### 63a. L2 pilot RESULT (L0 {x,xl,s}+{+,-,*}; truth y=xl/s+noise)
L0 verdict: INSUFFICIENT_EVIDENCE (correct abstention — division
inexpressible). L2 (+{/}): exact xl/s ranked #1 of scored (RMSE 0.0498
≈ noise floor 0.05); verdict MULTI_HYPOTHESIS (sign/junk-term twins tie
at identical RMSE — correct non-uniqueness, not a failure). Pilot
demonstrates the full escalation: abstain → discover exact truth →
refuse false uniqueness. Program proceeds on this evidence.

## 64. E2/E3 adversarial probes (design flaws found, both instructive)

E2 as run was DEGENERATE: v∈[0.5,1.5] all-positive makes v|v|≡v²
identically — no genuine rivalry constructed. Promoted v²+sqrt-junk
(RMSE≈noise): correct family, decorative terms. REDESIGN required:
signed domain + matched rival (e.g. c·v³ fit to domain center).
E3 (heavy noise, H1/H2/H3 present): xl/s-family dominates top-6 (H2/H3
absent — mechanism discrimination WORKS), but winner is xl+xl/s with
near-ties at 1e-5 and UNIQUE verdict — junk-decorated promotion the
verdict doesn't penalize. Finding: engine overfits additive decorative
terms; adjudication needs a parsimony/near-tie guard. Status: E2 VOID
(design flaw), E3 MIXED (discrimination ✓, promotion discipline ✗).

## 65. E2 signed-domain rerun (v|v| vs v², truth=v|v|)

Full [-1.5,1.5]: MULTI (v²±sqrt-debris twins). Neg-only: MULTI. Near-zero:
MULTI (bizarre forms). Pos-only: UNIQUE_SUPPORT with THREE forms tied at
RMSE 0.05091 — same numerical-tie structure as full-signed MULTI.
Inconsistency: identical ties → different verdicts (tie-handling depends
on non-tie factors; near-tie guard absent — corroborates §64 finding).
Truth v|v| never cleanly atop anywhere (v²+debris dominates). Status:
domain-rivalry constructed correctly this time; adjudication tie-
handling inconsistent; parsimony guard still missing (proposal only).

## 66. E3b decorative-term suite (A–F, frozen implementation)

A (necessary x³): top x^(5/2)±x, MULTI — truth family present but not
clean. B (decorative zero): x²±sqrt-debris, MULTI. C (tiny 0.01x³):
x²±sqrt-debris, MULTI. D (collinear x1=x,x2=2x): x1² AND x2² both top
(debris-attached), MULTI — spreads weight, picks neither. E (OOD
necessity): x³+x² present in-domain; OOD x²-only 1.29 vs x²+x³ 0.11
(12×) → RETAIN x³. F (sign-flipping coef): bizarre 1/sqrt forms,
RMSE 0.25 vs noise 0.1, MULTI — correctly no false UNIQUE.
Tie-invariance unit test: 6/6 order permutations → INSUFFICIENT
(order-invariant ✓; class is INSUFFICIENT not MULTI — record exact).
Ablation (drop debris): ΔRMSE −2.7e-04 on 0.10 (0.3%), coef −0.05 —
decorative by materiality. Status: discrimination works (E), debris
attaches everywhere, tie-handling invariant but INSUFFICIENT-leaning;
parsimony guard still unimplemented (awaiting approval).

## 67. E3b ablation profiles + defect localization (frozen impl)

OLS necessity (holdout Δ on removal): A +1.43 RETAIN; B +3.6e-07 REMOVE;
C +4.7e-03 BORDERLINE; D x1=x2=both EQUIV; F both ~1.53 (15× noise)
NEITHER-WORKS; E OOD 12× RETAIN (§66). Adjudicator kernel unit tests:
order-invariance ✓ (6/6 INSUFFICIENT), representation/order swap ✓
(MULTI both). Localization: adjudicate() itself invariant → §65
UNIQUE-vs-MULTI inconsistency originates UPSTREAM (generation→
ranking→selection feeds different candidate sets in). Missing stage is
term-necessity/parsimony filter, not tie logic. Explanations: B/C not
overfitting (fits don't improve on noise); debris = representation
degeneracy + absent necessity filter. Proposed contract (approval
required before any shared-code change): expanded grammar must not by
itself raise epistemic status; any UNIQUE requires ablation-passing
terms + near-tie equivalence class of size 1 + regression suite.

## 68. Guard implemented as standalone module + regression suite (no shared-code surgery)

New `apedp_dyn/epistemic/term_necessity.py` (post-hoc filter; adjudicate.py
untouched): drop-one-term OLS ablation (`ablate_terms`, holdout-scored;
RETAIN rel>5e-2 / BORDERLINE / REMOVE rel<1e-3, calibrated on E3b),
`near_tie_class`, `necessity_contract_check` (UNIQUE requires all-RETAIN +
singleton top class), `ood_necessity` (E3b-E pattern). New
`tests/test_dyn_term_necessity.py`: 9/9 pass in ~2s, ruff clean — pins
A-RETAIN, B-REMOVE, C-BORDERLINE, D-tied, E-OOD-12×, F-contract-fails,
tie-invariance, representation-invariance, contract gates. Shared
adjudicator still frozen; guard is opt-in post-processing until approved
for wiring. Status: Observe→Characterize→Specify done; Implement done
(additively); Regression-test done; awaiting approval to wire in.

## 69. Sensitivity sweep (c): monotone ladder + OOD ratchet (contract strengthened)

eps in {0,1e-4,1e-3,1e-2,1e-1,1} (noise 0.1): guard x3 REMOVE (3.9e-05),
REMOVE (-4.7e-04), REMOVE (9.75e-04), RETAIN (5.33e-02), RETAIN (2.22),
RETAIN (28.3); OOD ratio 1.0/1.0/1.0/1.1/4.4/40.4×. All engine verdicts
MULTI (no false UNIQUE); tops debris-decorated throughout (even eps=1:
x³+x²+1/sqrt(x)). Holdout-small/OOD-large dissociation remains E3b-E
(train |x|<1, OOD 12×) — now an explicit principle: `ood_necessity_check`
+ `apply_ood_upgrade` one-way ratchet (small effect != unnecessary;
OOD upgrades, never demotes). Suite 11/11, ruff clean. Milestone: the
L2 adversarial campaign localized false uniqueness not to adjudication
itself, but to uncontrolled hypothesis expansion; APEDP therefore
requires an explicit evidence-of-necessity layer between discovery and
epistemic promotion. Status: (c) PASSED — awaiting wire-in approval,
then full §63a–§68 regression, then E4 (Epistemic Recovery Under
Ambiguity: MULTI + maximally-discriminating measurement proposal).

## 70. Phase 1 wire-in + Phase 2 regression (guard at Step 5b, adjudicate.py untouched)

Wired `guard_selection` between Pareto selection and adjudication
(`l2_engine.py` Step 5b; purely restrictive: drops REMOVE-term carriers,
records top near-tie class, empty→NO_VALID; +`guard_report` result field).
Two wire-in bugs caught by existing suite: single-term expressions
ablated to REMOVE (vacuous → now RETAIN) and a circular import
(ast_generator→l2_engine; now deferred). After fixes: L2 suite 24/24,
necessity suite 11/11, ruff clean. Guarded reruns: debris-tops dropped
in 4/5 cases (E2-signed, E2-pos, E3b-B, eps1); eps0 clean/nothing dropped.
Caveats for the record: (a) §65 pos-only UNIQUE did not reproduce —
unguarded rerun gives clean-v² UNIQUE which is domain-correct (v|v|≡v²
for v>0); identical seeds gave different candidate sets → generation
bit-reproducibility gap flagged, not chased. (b) Verdict-level
before/after is confounded by that stochasticity; mechanism-level
(debris removal) verified, suite pins stand. (c) Leftover `stash@{1}
CHECKPOINT` duplicates current work — do not double-apply. Status:
Phase 1 DONE, Phase 2 DONE (modulo caveats); E4 NOT started.

## 71. E4 active disambiguation: max-D_sigma policy + abstention (synthetic first)

New `apedp_dyn/epistemic/active_disambiguation/` (additive; adjudicate.py
untouched): `Measurement`, noise-normalized `discrimination_table`
(D_sigma = min-pairwise-separation/sigma, P4), `select_measurement`
(max-D_sigma + abstain below k=3, P7), sequential `run_trial`
(select→observe→re-adjudicate through adjudicate(), P6; ≤3 rounds —
single shot cannot clear the sole-survivor τ≈0.0225, needs Δ>τ),
`compare_policies` harness (P5/P8). Frozen synthetic: H1=x²,H2=x²+0.1x³
(truth),H3=x²−0.1x³ on [-1,1] (MULTI pre, P1); queries at ±{2.5,2,1.5,0.3}.
P(UNIQUE≤3 rounds, n=40): active 1.000 = oracle 1.000 > random 0.050 >
worst 0.000; winner H2; flat-measurement set abstains
(INSUFFICIENT_DISCRIMINATION). Suite 7/7, ruff clean. Abstention chain
now: discovery → identification → experimental-selection. Status: E4
SYNTHETIC DONE; trading transfer NOT started.

## 72. E4-T1 trading transfer: volume-family 5d ambiguity (11 origins, frozen)

v1 abandoned (d1 target: H1 R² 0.03, near-constant predictors, D≤0.06σ —
ambiguity-from-weakness; July smoke origin discarded). v2: H1/H2/H3 fit
per origin on train non-overlapping 5d blocks (d5mean; H1 R² 0.38 vs H2
0.15 vs H3 0.11, block-MULTI pre on all origins); 8 measurements
(temporal/regime/transfer + d1 negative control); forward = origin-month
blocks; policies active/random/worst/persistence(m1)/oracle. Result
(11 origins × 5 policies = 55 trials): active abstains 11/11 (max D 0.29
< k=3); ALL non-abstain posts MULTI 44/44 — even oracle never promotes.
P5 on UNIQUE-rate degenerates (0 across the board): monthly forward
slices (~6 blocks) are too thin for any promotion, and H1 slope is
regime-unstable (0.0098 Aug22 → 0.0002 Jan23) while H2 persists (~0.008).
Honest reading: ambiguity ROBUST, not policy artifact; abstention works
in the wild; relationships rotate faster than the observation window.
Suite `test_dyn_trading_e4t1.py` 7/7 pins P1/abstention/oracle-cannot-
manufacture/determinism/time-order. No strategy claims. Status: E4-T1
DONE (negative result, recorded as such).

## 73. E5 stable vs drifting ambiguity (synthetic; classifier only, no new status)

New `active_disambiguation/drift_e5.py` (additive; adjudicate.py untouched):
H1=x²,H2=x²+0.45x³,H3=x²−0.45x³, 6×300 windows. A (truth H2 always) →
stable-identifiable; B (1e-6 gap) → stable-ambiguous (margins ~1e-9);
C (H2→H1→H3) → drifting (rank-1 rotation + margin 13× collapse/recovery,
local UNIQUEs but cumulative never promotes). Frozen classifier
(kernel-consistent τ-gate + sustained-collapse rule): 3-way separation on seeds
91–95 (5/5). Two sub-findings: (a) τ-knife-edge flicker (Δ=0.0506 vs
0.0491 vs τ=0.05) can fake UNIQUE-then-loss — sustained-collapse rule
required, noted as adjudication-hysteresis desideratum (not implemented,
kernel frozen); (b) rotation must be margin-gated or B's noise
coin-flips false-positive. Suite 11/11, ruff clean. Status: E5
SIGNATURES DONE; §72 plausibly Type B+C mix (flat margins + slope
rotation); no NONSTATIONARY_AMBIGUITY status implemented.

## 74. E5-market: BTC volume-family trajectory = drifting signature (diagnostic only)

`e5_market.py` on frozen §72 setup (11 origins, local=trailing-20-blocks,
cumulative=all-train, 10-field rows): rank-1 rotates H1→H2→H3→H1 across
origins (4 floor-gated changes), cumulative margins flat 0.0001–0.0008,
zero promotions, oracle MULTI on all 11. Null control (real features,
intercept-only matched-variance target, identical NaNs):
stable-ambiguous — its 5 raw flips all sub-floor, so the machinery does
NOT manufacture rotation. Classifier: market drifting ≠ null
stable-ambiguous (pinned, deterministic). Method note: raw-margin gate
(1e-4) failed the control (null also rotated); replaced by kernel-
consistent gate (rank change counts only above the window's own τ =
REL_FLOOR×RMS(y)) — synthetic 3-way separation re-verified 5/5 seeds
after the change; no post-hoc constant was fit to separate market from
null. Caveats: single asset/period, small absolute margins; formal
DRIFTING status NOT implemented (reviewer criteria 1–5 unmet); no
selection/strategy use. Status: E5-MARKET DONE — Outcome 3
(substantial drifting signature).

## 75. E5-R1 cross-asset replication (6 assets, per-asset nulls, frozen)

Assets pre-chosen (no PAXG in file): BTC ETH SOL BNB DOGE AVAX. Market:
6/6 drifting, zero promotions everywhere, oracle MULTI throughout.
Nulls @5150: 4/6 stable-ambiguous; ETH+BNB misclassify on that seed only
(pass 5151–53) → trajectory-level false-positive rate 3/24 ≈ 12.5%;
market 6/6 vs null background p≈4e-6 binomial — not plausibly all
flukes, reported with the rate not hidden. Calendar (descriptive, no
causality): rotations spread over all 10 boundaries — no single common
break; Nov-22 moves none; Mar–Jun-23 moves 3–4 assets each (mild
clustering, causes not inferred). Reading: asset-specific instability
with a possible common component in 2023H1 — characterization only.
Suite `test_dyn_e5_r1.py` 4/4 pins the table + null-rate + calendar
shape. No status, no selection, no strategy. Status: E5-R1 DONE.

## 77. Cross-asset H-A coupling replication (quad, 6 assets, frozen §76)

BTC stable-ambiguous (H-A#=1); ETH/SOL/BNB/DOGE/AVAX drifting
(H-A#=1/1/0/3/5; AVAX highest). Zero promotions on all six (verified). Outcome-matrix
reading: mostly-drifting + mixed-by-asset → H-A ambiguity remains
genuinely time-varying on 5/6 even after persistence control; BTC was
special. H-A never dominates rank-1 anywhere (max 5/11 AVAX, shared
with P/L). Ledger: predictive association supported conditionally;
unique structure rejected; stability not supported; single stationary
H-A law not promoted. H-A retained as observationally-similar member
of an unresolved class. Suite pins table in `test_dyn_ensemble.py`.
Status: COUPLING-R1 DONE.

## 78. M0–M3 ensemble: utility without mechanism (pooled OOS R², 11 origins)

Stacking-OLS, forward-month blocks, train-mean baseline (M_hist≡0).
M0=H-P, M1=+H-RV, M2=+H-A, M3=all:

| asset | M0 | M1 | M2 | M3 | ΔM3−M0 | win-rate |
|---|---|---|---|---|---|---|
| BTC | .192 | .202 | .330 | .328 | +.136 | .82 |
| ETH | .291 | .272 | .335 | .330 | +.039 | .36 |
| SOL | .276 | .291 | .341 | .333 | +.058 | .82 |
| BNB | .298 | .295 | .305 | .332 | +.034 | .73 |
| DOGE | .360 | .347 | .387 | .367 | +.006 | .64 |
| AVAX | .327 | .255 | .255 | .229 | −.098 | .27 |

M2≈M3 everywhere (rankV adds ~nothing once H-A present; overlap
handled by joint OLS, ETH-M1<M0 shows blind stacking can hurt). H-A
adds over H-P on BTC strongly, ETH/SOL/BNB modestly, DOGE ~nil, AVAX
negative (collinear overfit). Tension recorded: ETH pooled Δ>0 but
win-rate .36 (few origins dominate). Claim licensed: ensemble
predictive utility on 4/6, NOT mechanism — H-A rank frequency (AVAX
5/11) dissociates from incremental utility (AVAX −.098). Suite 4/4,
ruff clean. Status: ENSEMBLE DONE; options/futures vol-forecasting
is the sensible next application, directional trading is not.

## 79. Gated ensemble: prospectively gated, permutation-humbled (§78 downgraded)

M0=always H-P vs always-M2 (H-P+H-A) vs adaptive (train-only gate:
70/30 fit/val, admit iff val-MSE improves AND val block win-rate>1/2;
AVAX gate opens 2/11, BTC 11/11). Primary QLIKE (5th-pct train floor
after BNB-near-zero blowup caught pre-record), secondary R²,
month-block permutation (200 draws) on pooled ΔR²(M2−M0), per-origin
tables. QLIKE/R² agree: adaptive tracks best arm, avoids AVAX damage.
BUT perm_p = .58/.82/.24/.56/.38/.99 — pooled M2−M0 edge within
temporal-reshuffling noise on ALL six. §78 re-read: nominal pooled
gains (BTC +.136 etc.) are real numbers, not significant ones; ETH
.36 win-rate already hinted this. Licensed update: gate works
prospectively without hindsight; utility NOT established at
permutation level; no trading claim (forecasting≠trading retained).
Suite 4/4 (`test_dyn_gated_ensemble.py`), ruff clean. Status: §79
DONE — the control did its job against our own prior.

## 80. Incremental battery: A|P,RV nominal, permutation-null, HA suppressor

Six combos (P/RV/A/P+RV/P+A/P+RV+A), 6 assets × 11 origins, QLIKE
primary. Headline A|P,RV (QLIKE): BTC +.0287 ETH +.0093 SOL +.0078
BNB +.0053 DOGE −.0043 AVAX −.0071. Permutation (procedure rerun
inside): .60/.88/.62/.72/.94/1.0 — headline NULL everywhere. DM-HAC4
agrees except BTC (.036) and AVAX-worse (.016), both contradicted by
their own permutations (headline stays permutation, pre-registered).
Redundancy: corr(ŷP,ŷA) .72–.94; RV|P,A ≈ 0 all assets (RV dead once
P+A present); β_HA<0 in joint model on ~100% origins/assets
(suppressor, not independent driver); HA⊥ t negative (−2 to −6).
Wins concentrate P+A/P+RV+A (BTC 4+5) without significance. Decision
tree: Case 2/4 — apparent contribution largely redundant, not
prospectively significant; transfer-matrix + regime-ablation
explicitly DEFERRED. Sentence earned: mechanistic identification
failed AND incremental information not established at permutation
level — H-A ≈ persistence-conditioned association stands. Suite 4/4,
ruff clean. Status: §80 DONE.

### §80b — ERR-INC-80 erratum (2026-10-20, disclosed): re-derivation after the
### forward-overlap purge of the development harness

Commit `3419425` ("Purge forward-overlapping development targets") legiti-
mately removed the 5-day-ahead window from the *shared* development targets
(`trading_e4t1_v2.run_origin`: `tr_blocks = tr_idx[tr_idx + 5 < origin_idx]`),
fixing a real target-leakage flaw — but the §80 headline numbers above and the
pins in `tests/test_dyn_incremental.py` had been computed under the contamin-
ated harness and began failing. Per the honest-erratum discipline (same
seed 8080, same artifacts, no threshold moved, no re-registration beyond the
harness fix already merged), the values were RE-DERIVED from the same panels
under the purged harness (deterministic, verified twice) and the pins were
sharpened to exact per-asset values:

| Quantity | Old (overlapping harness) | Re-derived (purged) |
|---|---|---|
| A\|P,RV | .0287/.0093/.0078/.0053/−.0043/−.0071 | .013693/.017228/.016629/.018592/.000086/.000308 |
| perm p | .60/.88/.62/.72/.94/1.0 | .7164/.0896/.2189/.7363/.8557/.5622 |
| DM-HAC4 p | BTC .036*, AVAX .016* | BTC .0585, ETH .000557, SOL .000971, BNB .366, DOGE .966, AVAX .941 |
| corr(ŷP,ŷA) | .72–.94 (panel) | .630/.887/.863/.387/.741/.855 |
| BTC wins P+RV+A / P+A | 4/5 | 3/3 |

(* = DM-significant at .05; both contradicted by their own permutations.)

Subscribe unchanged conclusions: the permutation headline remains PRIMARY and
NULL everywhere (minETH = .0896 > alpha = .05); DM significant somewhere
(ETH/SOL) is again contradicted by the permutation null — same Case 2/4. The
suppressor pattern holds (RV\|P,A all dead, |x| < .008; HA sign-flips 100% on
every asset; redundancy weaker — BNB .387 — recorded, suppressor interpret-
ation unchanged). Thresholds and the §80 decision stand; only the frozen
magnitudes moved, and only because the harness leak they were computed under
was purged. Pins updated in `tests/test_dyn_incremental.py` with this
disclosure in the docstring.

## 81. Grammar Space executable in sandbox (spec §§1–19 as code)

New `apedp_dyn/sandbox/grammar_space.py` (additive): primitive/operator
registry with domain checks (log→x>0 noted vs spec's "log(x)>0"),
causal builders for H-P/H-RV/H-A/H-LOG, d5mean target, 7 combos,
incremental/redundancy equations, Level-0..5 `promote()` with the
no-rescue rule, frozen §80 evidence record, executable §17 table
(H-P→L4-held, H-RV/H-A→L1-held at L2-incremental, nothing promoted).
Cross-check caught a real divergence: my HA omitted v1's outer z
around the component sum — fixed to bit-identical (pinned), so
§§72–80 stand on one series; fidelity over elegance recorded.
Suite 6/6, ruff clean. Status: §81 DONE — spec and sandbox agree.

## 82. Volume-family generatable: rank/z-score ops in L2 (additive)

`RollingOpNode` gains causal `rollrank` (trailing percentile),
`rollzscore` (trailing standardized, W=40), and `expanding_zscore`
(min-history 60, unbounded causal history). All three are
dimensionless-out from any input dimension, no roll-of-roll, wired into
frontier + probe sizing (expanding form forces probe grid ≥68). No
default operator-set changes (physics sessions unaffected). H-P (`abs`)
was already generatable; H-RV/H-A now expressible compositionally.
Caveat recorded: bare-leaf rollrank collapses on monotonic probe grids
(constant fingerprint), so the frontier carries it on abs/log children —
identical on positive real volume.

**Fidelity fix (this pass):** frozen H-A standardizes with an *expanding*
z-score, which fixed-window rollzscore does not reproduce, so
`expanding_zscore` was added to close the gap. Its cumulative-sum window
was initially off by one (prepended-zero alignment) and was caught by
differential testing against `grammar_space.causal_zscore`; after the
`[1:]` correction it matches to 1.3e-15 including NaN-contaminated
series, with zero future leakage under perturbation.

Autonomous engine run, frozen BTC d5 panel, finance ops
(`+ - * / abs log rollmean rollrank rollzscore expanding_zscore`),
depth 3 / complexity 8, 29,789 scored candidates, verdict MULTI, guard
drops nothing:

| rank | holdout RMSE | generated equation |
|---|---|---|
| 1 | 0.006298 | `−r·ExpandingZscore(r,60) + RollMean(Abs(r),40)` |
| 2 | 0.006298 | `r·ExpandingZscore(r,60) + RollMean(Abs(r),40)` |
| 3 | 0.006322 | `RollMean(Abs(r),40) + Abs(r)` |
| 4 | 0.006322 | `−RollMean(Abs(r),40) + Abs(r)` |

Volume-bearing candidates are generated in bulk (26,327 of 29,789) but
the best volume form lands at RMSE 0.006413 (ranks 9–14, e.g.
`RollMean(Abs(r),40) + RollZscore(V,40)`), i.e. *behind* the pure
persistence family. Reading: with the volume-family grammar available,
open-ended discovery still converges on persistence and does not prefer
volume representations — independent agreement with §76/§80 from the
autonomous side. Suites 8/8 (rollrank/zscore pins) and 41/41 (L2 +
grammar-space + necessity) pass; ruff clean. Status: §82 DONE.

## 83. Full regression sweep + one disclosed test update

Sweep: 639 collected across `tests/test_dyn_*.py` (`-m "not slow"`), 633
passed, 6 failed. Baseline classification by reverting only
`ast_generator.py` + `l2_engine.py` to HEAD and re-running the six:
**5 of 6 fail identically at HEAD** (pre-existing, not introduced here) —
`test_stridge_recovers_duffing_truth` (FIN_TERMS vocabulary drift:
`u_relax` present, test expects `v`), `test_m1_single_mechanism_recovers_hall_petch`
(UNIQUE→MULTI, materials path does not use L2DiscoveryEngine),
`test_single_window_sp500_calm_sweep` and two `test_dyn_ui_studio.py`
cases (`multi_hypothesis` vs `insufficient_evidence`, `unique_support` vs
`multi_hypothesis`; live-API tests, consistent with the stale `:8000`
worker caveat).

**1 regression was mine and is fixed.** `test_transcendental_dimensional_type_enforcement`
passed at HEAD and failed on the working tree. Cause: the dimensionless gate
message had been generalized from `"Transcendental operator 'sin' requires…"`
to `"Operator 'sin' requires strictly dimensionless input [0, 0, ...], got
[L]"` when `sat`/`inv_quad` joined the dimensionless-gated set — the word
"Transcendental" had become false. The new message is strictly more
informative (it names the offending dimension), so the **test regex was
updated, not the message**; documented here rather than made silently.
`test_dyn_transcendentals_l2.py` is now 7/7 — the two long-standing
pre-existing failures in that file are also resolved.

Net: 634 passed, 5 pre-existing failures unchanged, 0 regressions
attributable to this work. Status: §83 DONE.

## 84. Bootstrap tiebreak contract defect + four stale tests (all green)

**Root-cause defect (real, not cosmetic).** `tiebreak` documents its inputs
as *per-timepoint squared errors*, but `resid_best`/`resid_rival` are
*signed* residuals — correctly so, since the sole-survivor whiteness guard
needs the signs. Differencing signed residuals and bootstrapping their
**mean** tests nothing about accuracy. Symptom: M1 Hall-Petch, ΔRMSE =
+70.8 with the point winner at RMSE 2.64 vs 73.43, returned
`rival-favored` on a CI of [-81.6, -54.2] and refused the win. Fix: the
conversion now lives in `adjudicate` (square at the tiebreak call site
only; whiteness still sees signed series). `tiebreak` keeps its documented
squared-error contract, with no auto-detection — auto-detecting by
sign would corrupt callers that legitimately pass negative differentials.

Corroboration: `qec_threshold.py` already carried a comment diagnosing
exactly this ("tiebreak contracts squared-error series, not signed
residuals. Passing (op - pred) here decouples the branch label from the
RMSE margin") and worked around it **locally**, leaving every other caller
(materials, lynx, the sweeps) broken. That workaround is removed so the
contract is uniform; a pre-squaring call site in
`test_dyn_zero_false_promotion.py` is likewise de-squared.

**Four stale assertions updated, each with the reason recorded:**

1. `test_stridge_recovers_duffing_truth` — the five terms added to
   `_TERM_BUILDERS` for the crypto work (`sat_x/sat_v/sat_u/inv_x2/u_relax`)
   are re-parameterizations of primitives already in the library
   (`u_relax == u - v` exactly). Threshold-based STRidge is unstable under
   that collinearity: it kept `u_relax` and dropped `v`. They stay in
   `_TERM_BUILDERS` (crypto FIN_TERMS, the IR family map, the UI, and an
   explicit test all reference them) but are out of the *default* sparse
   vocabulary, matching `DEFAULT_TOPOLOGY`, and remain reachable via an
   explicit `vocabulary=`.
2. `test_single_window_sp500_calm_sweep` + the API twin
   `test_market_discovery_calm_refuses_spurious_models` — expect
   `insufficient_evidence`/`out_of_grammar=True`; now
   `multi_hypothesis`/`out_of_grammar=False`. Once gate A tolerates
   causally honest NaN padding, the grammar *does* hold an adequate
   candidate (train RMSE 6.2e-03, holdout 4.7e-03 — it generalizes; whiteness
   passes), and the surviving rival `ma_dev**(-2)+sqrt(ma_dev)` is the same
   function reordered. Claiming "out of grammar" would be false; the
   refusal is representation-equivalence, which is what the tests now pin,
   with the anti-promotion intent kept explicit.
3. `test_registry_problem_discovery_endpoint` (coupled) — expects
   `multi_hypothesis`, now `unique_support` (Δ=0.656 vs τ=0.178, a clear
   3.7x separation) after the coupled `[0]→[idx]` drive fix. The winner's
   residuals are structured (|rho1|=0.998), which the pair path deliberately
   reports but does not enforce (see `_winner_structure_note`); the test now
   pins that the caveat survives into the response rather than being dropped.
4. `test_lynx_real_data` — its `multi_hypothesis` was pinned to the buggy
   differencing; corrected, the margin Δ=0.568 > τ=0.341 is confirmed and the
   verdict is `unique_support`. Flagged for review: this promotion is
   marginal (1.66x τ) with |rho1|=0.614.

Also corrected: `test_would_be_unique_flips_to_multi` and
`test_bootstrap_ci_straddling_zero_flips_to_multi` could no longer express
"a straddling CI" by passing a pre-differenced series (squares cannot
cancel); both now build the scenario from valid signed residuals. And one
self-inflicted error is on the record: a gate-4 "generalization guard" was
added on the belief that the sp500 candidate overfit 123x — that compared
train **MSE** against holdout **RMSE`; the true ratio is 0.76, the guard was
a no-op, and it was removed rather than left in on a false premise.

Sweep: **639 passed, 0 failed** across `tests/test_dyn_*.py -m "not slow"`;
ruff clean. Status: §84 DONE — the pre-existing-failure backlog is empty.

## 85. Frozen decision: 639/639 ACCEPT; two boundary cases labelled, gate unchanged

**Suite: 639 passed, 0 failed — ACCEPT.** The promotion gate was
deliberately **not** modified. Adding a margin requirement or enforcing
whiteness on the pair path would be post-hoc rule modification driven by
two inconvenient cases; the rule stays frozen and the *reporting* carries
the qualification.

**Central distinction made explicit:**

> Predictive Winner != Complete Dynamical Explanation

`delta > tau` answers "did this candidate beat that rival on unseen data?"
Residual whiteness asks "has it captured all systematically recoverable
structure?". These are not equivalent — `RMSE_A < RMSE_B` can hold while
`|rho1(eps_A)| ~ 1`. **Holdout superiority does not imply dynamical
completeness.** `coupled` is precisely the case that exposes it.

New status-preserving diagnostics on `Adjudication`
(`promotion_confidence`, `diagnostic_flags`), populated on both
`UNIQUE_SUPPORT` branches and surfaced by the API. They never alter
`status` — verified by re-running the full suite:

| case | delta/tau | winner \|rho1\| | verdict | confidence | flags |
|---|---|---|---|---|---|
| lynx_cycle | 1.66 | 0.614 | unique_support | low_confidence | WEAK_PROMOTION_MARGIN, RESIDUAL_STRUCTURE_UNRESOLVED |
| coupled | 3.67 | 0.998 | unique_support | (none) | RESIDUAL_STRUCTURE_UNRESOLVED |

**lynx_cycle — boundary-case promotion.** Formally promoted under the
frozen rule, but weak promotion margin and substantial residual
autocorrelation make this a low-confidence identification. Its promotion
establishes formal support under the current protocol, **not** strong
identification.

**coupled — predictive promotion only.** Formally promoted as a predictive
winner under the existing holdout-arbitration doctrine, but `|rho1| =
0.998` prevents interpreting the result as a complete dynamical
identification. Claim table: predicts better than rival **yes**; wins the
frozen holdout comparison **yes**; little residual dependence **no**;
complete dynamical explanation **not established**; predictive winner
**yes**; uniquely identified **no / conditional**.

Both flags are now asserted in tests, so neither caveat can be silently
dropped from a response later.

Sweep after adding diagnostics: **639 passed, 0 failed** (unchanged, as
required). Status: §85 DONE — gate frozen, boundary cases labelled.

## 76. H-A/E5 coupling: persistence absorbs the rotation (case study, BTC)

Frozen quad (H-A composite, rankV, logV, persistence-only) on d5 blocks,
11 origins: rank H-RV,H-RV,H-A,H-P×8 — H-A rank-1 once (Oct-22);
H-P modal rank-1 from Nov-22 without a single promotion; class
stable-ambiguous (rotations sub-floor); cumulative margins ≤0.0006.
Four answers: (1) H-A rank-1 persistence 1/11; (2) all margins
sub-floor; (3) early rotation settles into H-P plurality, not gated
drift; (4) zero consolidation. Interpretation (three-claim split):
predictive association — some support; unique structural mechanism —
not established; stable cross-time mechanism — challenged. Notable:
trio-drifting vs quad-stable — adding persistence absorbs the apparent
volume-model rotation into stable persistence-plurality. H-A reads as
one observationally-similar description among several, not a
consolidating law. Suite 4/4, ruff clean, no performance claims.
Status: H-A COUPLING DONE.

1. Selection metric ≠ scientific metric (reciprocal paradox).
2. Pooled significance can be a between-regime artifact — always split (BTC ×2, +0.29 ×1).
3. Rank the evidence hierarchy, never a trading leaderboard.
4. Pre-register family + endpoint + success criterion before evaluation.
5. A replicated null beats an unreplicated +144%.
6. Hypotheses graduate only on unseen data with controls pre-contact.
7. Report incremental ΔR² over persistence, never absolute R² as the discovery claim (task lives at 0.2–0.35 via clustering).
8. Absolute R² ≈ 0.7+ at this task = audit for leakage/target/overlap artifacts first, celebrate never.

## 14. T0→T3 nested ladder (vol-normalized returns + autonomous vol)

Frozen theories, fit H1-2023, autonomous 20d rollouts over 9 frozen H2
windows (model's own state feeds forward; signal = sign of cumulative
predicted return): BTC T0 0.01909 < T1/T2/T3 (all worse; 5/9 hits); SOL
T2 0.04641 ≈ T1 ≈ T0 ≈ T3 within 0.5% (8/9 hits — drift-up base rate);
ETH all tied ±0.0004 (3/9 hits). Normalization adds nothing anywhere
(BTC worse, SOL +0.1%, ETH tied). In-sample notes: BTC zAR +0.23
evaporates OOS; SOL GARCH ρ=0.88 / ETH ρ=0.79 (vol persistence real) but
buys zero OOS return prediction; BTC GARCH hits ρ=0 boundary (no
persistence at all). Outcome C on all three: fits, autonomous OOS
collapses to ≤ predicting zero. Volatility has persistent dynamics;
returns do not inherit predictability from them.

## 15. T4 cross-asset ladder + lead-lag matrix (fit H1, frozen, H2)

T4A (drift) → T4B (+lagged composite) → T4C (+idiosyncratic residual),
OOS RMSE: BTC T0 0.01996 best, each rung worse (R² −0.01/−0.02); ETH T0
best, rungs worse; SOL T4A 0.04925 vs T0 0.04969 (0.9%, noise). Pairwise
lead-lag matrix (all 9 ordered pairs): RMSE constant down columns
(BTC-tgt 0.01995–0.01998 regardless of source), hits 0.49–0.51 everywhere
— no triangular asymmetry, no direction anywhere. Referee verdict: no
incremental OOS information over T0 from self-history, market factor,
residuals, or cross-asset lags. Confusion matrix unneeded beyond hit≈coin
(base rates reported: the matrix IS the null). State-representation
hypothesis (relative dynamics) fails its first test; cross-asset channel
joins the documented nulls.

## 16. OFI-only state hypothesis (fit H1, frozen, H2; BTC/ETH/SOL)

OFI descriptives: same-day corr with returns +0.36..+0.51 (flow coincides
with the close, as expected); means ≈ 0, std 0.03–0.04. Lagged-OFI OOS:
BTC T0 0.01995 vs OFI 0.01988 (+0.35%, bal 0.52); ETH tied/worse, bal 0.24
(one-sided); SOL +0.8% with bal 0.25 (degenerate one-sided). Referee
verdict: NOT SUPPORTED — sub-1% edges, degenerate directionals.
Mechanism in one line: flow coincides with returns contemporaneously but
does not lead at daily resolution. OFI joins self/vol/market/cross-lag in
the documented nulls (as a return predictor; its vol-prediction role from
OF-01 is separate and stands).

## 17. Intraday OFI horizons (Binance 15m/1h bars, fit-2024 → OOS 2025–Oct26)

T0 (0) vs T1 (signed taker volume) vs T2 (imbalance fraction), BTC+SOL:
15m BTC T0/T1/T2 = 0.002376 ×3 (bal ≤0.514, n_te=61k); 1h 0.004685 vs
0.004688/0.004686; SOL 15m 0.004183 ×3; 1h 0.008237 vs 0.008246 ×2.
Identical to 6 decimals on 61k test bars — no practically meaningful
improvement demonstrated at the reported precision. (Effective-sample
caveats apply: intraday bars are serially dependent, so no formal power
claim is made; the strength is the flat zero, not a significance test.)
5m/30m remain untested — lower-priority given 15m/1h/1d flatness, not
presumed empty.

## 18. OFI→volatility intraday (separate hypothesis, OPEN)

Target = next-12-bar realized vol; baseline = trailing-vol persistence
(the correct baseline for vol, not zero); +|imbalance| + volume-z;
fit-2024 frozen → OOS 2025–Oct26: BTC15m Δ+0.0147 (|imb| t=−13.7), BTC1h
Δ+0.0151 (t=−3.9), SOL15m Δ+0.0099 (t=−13.2), SOL1h Δ+0.0114 (t=−5.9).
Consistent +1–1.5pp on 4/4 cells atop 12–23% persistence baselines —
Hypothesis B supported at intraday scale. Sign negative (one-sided flow
→ calmer ahead), mechanism uninterrogated. Magnitude without direction,
confirmed where direction died: the split the file preserves.

Attribution ladder (fit-2024 → OOS 2025+): V1(+volz) deltas +0.001/+0.004/
+0.000/−0.000 ≈ 0 on all 4 cells; V2(+|imb|) deltas +0.011/+0.011/+0.009/
+0.011 on all 4. Carrier isolated: |imb|, not volume normalization; no
suppression pattern (V3 ≈ V2). Frozen replication fit-2025 → 2026YTD
(n_te 26k/6.6k): 15m holds weakly (BTC +0.005, SOL +0.003), 1h fails
(BTC −0.003, SOL −0.014; SOL1h persistence itself collapses to 0.016).
Verdict: weak horizon-dependent replication — neither clean pass nor
clean kill; 15m structure partially survives calendar shift, 1h does not.

2026H1→2026H2 at 15m, four gates (n_te=9070): BTC base R²=0.003 →
BASELINE_UNHEALTHY, delta −0.0004 uninterpretable, sign neg (train
t=−8.3); SOL base 0.074 healthy, delta +0.0003 (vs history
+0.003..+0.015), sign neg (t=−8.4). Trajectory across forward windows:
+0.015 → +0.005 → +0.000 — diminishing toward zero, not stabilizing.
Status: effect dying on current evidence; 2026H2 short-window caveat
noted but does not reverse the trend. BASELINE_HEALTH gate worked as
designed (BTC): no inference from increments over a dead baseline.

SUPERSEDED by frozen-origin decay test: 2024-fit V3 NEVER refit,
quarterly blocks 25Q1–26H2 — deltas BTC
+0.013/+0.019/+0.040/+0.021/+0.010/+0.021, SOL
+0.001/+0.010/+0.020/+0.012/+0.015/+0.026. ALL TWELVE positive, no decay
trend; increments survive even dead baselines (25Q3 BTC V0=−0.36 yet
Δ=+0.040; 26H2 V0 negative both, Δ=+0.021/+0.026). Prior "fading" was
re-estimation artifact: refit versions decayed, frozen original persists.
Status upgraded: OOS-SUPPORTED-narrow, frozen-persistent 12/12 — first
genuinely replicating structure of the session. Mechanism unresolved.

Hostile suite: permutation (shuffled |imb|, 300 draws): 0/300 reach
observed delta on BTC and SOL (p<0.0033) — temporal alignment carries
it, not the distribution. Asset replication (pooled BTC+SOL-2024 frozen
spec → ETH/DOGE 2025+, n=61k each): ETH +0.0135 (clean pass), DOGE
+0.0017 (directional, negligible). Exchange/source replication BLOCKED:
no other venue exposes taker-split klines; documented, not substituted.
Net: survives permutation + ETH transfer; DOGE marginal; source
independence untested. Status: REPLICATED_STRUCTURE (narrow,
heterogeneous, mechanism open).

Mechanism battery (fit-2024 → OOS 2025+, 15m): horizon profile Δ(h) BTC
+0.003/+0.006/+0.009/+0.010/+0.012 and SOL
+0.002/+0.005/+0.007/+0.008/+0.010 for h=2/4/8/12/24 — monotonically
increasing, REJECTING H2-completed-event decay prediction; favors
persistent-state (H1/H3). Construction variants over persist+Amihud base:
imb-frac +0.011/+0.009 (survives liquidity proxy); raw signedVol
−0.002/+0.002 and trailing-norm −0.001/+0.001 (nothing) — effect exists
ONLY under contemporaneous-volume normalization. H4 constrained but open:
fraction-vs-ratio-artifact (1/volume-alone untested) remains the sharpest
unrun check; Amihud does not absorb the effect (partial H1/H3 split).

Mechanism discrimination (RV = persist + |imb| + volz + r_t + |r_t|,
fit-2024 → OOS 2025+, 15m): BTC Δ over no-|imb| +0.0108 with t(|imb|)=−12.8
beside t(|r|)=+37.9, t(r)=−7.4; SOL Δ+0.0075, t(|imb|)=−11.9 beside
t(|r|)=+40.8. |imb| survives move-magnitude controls — not a "big move"
proxy. Byproducts: |r_t| dominates (contemporaneous magnitude rules),
signed r_t negative-significant (intraday leverage effect). "Large
imbalance during large move" fails to absorb the effect; H1 (absorption)
vs H2 (completed event) vs H3 (liquidity) vs H4 (construction artifact)
all still open.

## 24. H-A horizon profile + trading translation (frozen, 2025+ unseen)

Horizon ΔR² (fit-2024 frozen, quarterly): grows monotonically 4→32 on
all assets (BTC 12/12 blocks positive; SOL/ETH 10/12, 25Q3-only misses)
— characteristic profile: longer-horizon vol more volume-predictable.
Other-asset battery (fit-2024 frozen → OOS 2025+, permutation 200):
BNB Δ+0.083, perm 0/200, quarterly 6/6 incl. Q3 +0.074 (strongest asset
yet); XRP Δ+0.008, perm 0/200, 4/6 with 25Q3 −0.414 (catastrophic miss);
DOGE Δ+0.073, perm 0/200, 5/6 (Q3 −0.043 only miss). Permutation clean
3/3. 25Q3 hostile to XRP/DOGE as to BTC/SOL/ETH; BNB immune — first
breach in Q3-universality (watch: single quarter).
Trading translation (vol-targeted long-only, daily rebalance, 5bps,
frozen H-A): BTC net +12.3% vs BH −9.9% (Sharpe +0.17, maxDD 64%);
SOL −9.3% vs −46.6% (−0.07, 141%); ETH +29.0% vs −22.0% (+0.30, 87%).
Beats buy-hold in a downtape (low bar) with uninvestable risk-adjusted
numbers. Verdict: predictive validity ≠ trading validity — economic
exploitability NOT established; translation recorded, not promoted.

Canonical H-A word-for-word (V0=TV+|r|+Amihud, +log(RollMean40), OLS
fit-2024 frozen, OOS 2025+, perm-300): BTC Δ+0.031 perm 0/300 quarterly
6/6 (+0.033/+0.019/+0.045/+0.008/+0.023/+0.140); SOL Δ+0.036 perm 0/300,
5/6 (Q3 −0.121 only miss); ETH Δ+0.022 perm 0/300, 5/6 (Q3 −0.037 only
miss). 16/18 quarterly positive; richer baseline flips BTC-Q3 positive
(Amihud absorbs the Q3 break there). Status: CANONICAL-VALIDATED.

## 23. Pre-registered clustering-regime rule (frozen, awaiting next quarter)

Descriptor: C = AC1(|r_15m|) per quarter. Rule (frozen 2026-10-03):
LOW-CLUSTERING if C < 0.25. Calibration (descriptive only): flags
BTC-25Q3, SOL-25Q2, SOL-25Q3 — retrodicts the observed weak/miss blocks
(BTC-Q3 miss, SOL-Q2 +0.005 weak, SOL-Q3 miss), which is coherence, not
confirmation. Protocol for next flagged quarter: frozen V3 and H-A specs,
ΔR² + ΔMSE vs healthy-baseline check; supports regime hypothesis iff
volume-family increments attenuate specifically there while direction
preserved elsewhere. Statuses: 25Q3 = REGIME_DIAGNOSTIC (low-clustering,
low-impact); mechanism = UNTESTED/PRE-REGISTERED; no Q3 term enters any
grammar.
