# Result: PREREG-GRAPHS-2026-11

**Run:** once, 2026-11-19, at commit `96124ee`, which froze the preregistration, the script, the
generator and the graph module (`preregistration/DIGEST-GRAPHS.json`; the script checks it before
running). Python 3.13.18, numpy 2.5.3, scipy 1.18.1, Tigramite's PCMCI+ unmodified. One CPU core,
under a minute. No deviation.

Outputs, unedited, in [`RESULT-GRAPHS-2026-11/`](RESULT-GRAPHS-2026-11/): `results.json` and `runs.json`.

## Outcome

60 simulated series with a known lagged graph, 922 true links in all. Precision is the share of
reported links that are true; recall is the share of true links that are reported.

| Series | Method | Precision | Recall | F1 |
| :--- | :--- | ---: | ---: | ---: |
| Linear and heavy-tailed (40) | PCMCI+ | **0.973** | **0.992** | 0.982 |
| | PCMCI+ with the even pass | 0.936 | 0.992 | 0.963 |
| | Vector autoregression | 0.905 | 1.000 | 0.950 |
| | Marginal correlation | 0.202 | 0.993 | 0.335 |
| Nonlinear (20) | PCMCI+ | 0.942 | 0.831 | 0.883 |
| | PCMCI+ with the even pass | 0.889 | **0.990** | **0.937** |
| | Vector autoregression | 0.891 | 0.834 | 0.862 |
| | Marginal correlation | 0.328 | 0.850 | 0.473 |
| All (60) | PCMCI+ | 0.963 | 0.937 | 0.950 |
| | PCMCI+ with the even pass | 0.920 | 0.991 | 0.954 |
| | Vector autoregression | 0.901 | 0.944 | 0.922 |
| | Marginal correlation | 0.229 | 0.945 | 0.368 |

Links that act through an even function of the source (58, all in the nonlinear series): PCMCI+
found 7, the vector autoregression 9, PCMCI+ with the added pass **57**.

## Against the expectations

1. **Accurate on linear data: held.** Precision 0.973 and recall 0.992 (limits 0.90).
2. **Better than the naive method everywhere: held.** F1 0.987 against 0.330 (linear), 0.977 against
   0.341 (heavy-tailed), 0.883 against 0.473 (nonlinear).
3. **Not more false links than a vector autoregression: held.** Precision 0.963 against 0.901: 33 false
   links against 96.
4. **The added pass finds what PCMCI+ cannot: held.** 57 of 58 even links (98 %; limit 80 %), at an
   overall precision of 0.920 (limit 0.85).

All four held.

## What this means

- The graph engine now has a measured result: on data that meet its assumptions it reports few false
  links and misses few true ones, and it reports a third as many false links as a vector
  autoregression at the same level.
- PCMCI+ alone does not see a link that acts through the size of a variable regardless of sign. The
  added pass does, and that lifts recall on the nonlinear series from 0.83 to 0.99.
- Marginal correlation reports four false links for every true one. Anything built on lagged
  correlations alone inherits that.

## After the run (not preregistered)

- **The added pass has a price on heavy-tailed data.** There it adds no true link (there are no even
  links to find) and adds false ones: 34 false links against 12 for PCMCI+ alone, precision 0.899
  against 0.962. Its test assumes Gaussian residuals, and squares of heavy-tailed variables break
  that. On market-like data the pass should be off, or its test replaced by a robust one.
- **Length barely matters in this range.** PCMCI+: precision 0.968 and recall 0.933 at 300 steps;
  0.958 and 0.941 at 1,000.
- **Cost.** A median of 0.17 s per series for PCMCI+ (0.4 s at most), 0.23 s with the pass.

## What it does not show

As disclosed before the run: every series has no hidden common cause, no link within a time step
and a fixed graph. Real market data have all three. This result is a floor for the method on clean
ground, not evidence that it finds causal structure in markets.
