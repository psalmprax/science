# PREREG-GRAPHS-2026-11: recovering a lagged causal graph

**Status:** frozen before the run by `preregistration/DIGEST-GRAPHS.json`. Runs once:
`python scripts/run_prereg_graphs.py --out preregistration/RESULT-GRAPHS-2026-11`.

## What is tested

`apedp_dyn/proposers/causal_graph.py`. The graph engine was an adapter around PCMCI+ that had never
been measured. Here it returns the whole lagged graph from one run (`discover_graph`), with an added
pass for links that act through an even function of the source, which partial correlation cannot
see (`discover_graph_with_even_links`). Two baselines show what the search is worth: a full vector
autoregression with t-tests, and marginal lagged correlation. All four use the same level (0.01,
the canonical Diebold-Mariano threshold).

## Data

Simulated series with a known graph (`scripts/causal_graph_worlds.py`): every variable depends on
its own previous value, and has about one incoming link from another variable at a lag of 1 to 3.

- `linear`: Gaussian noise, linear links.
- `heavy_tail`: linear links, Student-t noise with 3 degrees of freedom.
- `nonlinear`: each cross-link acts through the identity, `tanh`, a signed square, or a centred
  square (even).

5 and 10 variables; 300 and 1,000 time steps; five seeds each (3000 to 3004, not used before). 60
series. A method is scored on the lagged links it reports against the true ones; numbers are pooled
over series by counting links.

## Expectations, fixed now

1. **Accurate on linear data:** on the linear and heavy-tail series together, PCMCI+ has precision
   at least 0.90 and recall at least 0.90.
2. **Better than the naive method everywhere:** in each of the three kinds of series, PCMCI+ has a
   higher F1 than marginal correlation.
3. **Not more false links than a vector autoregression:** over all series, PCMCI+'s precision is at
   least the vector autoregression's.
4. **The added pass finds what PCMCI+ cannot:** on the nonlinear series, with the pass, at least 80 %
   of the links through an even function are found, and over all series its precision stays at
   least 0.85.
5. Reported without a criterion: recall against the vector autoregression, every size separately,
   and run times.

A miss on 1 to 4 is reported as a miss.

## Disclosed before freezing

- Development used seeds 700 to 703 with the same generator. Pooled there: PCMCI+ precision 0.94 to
  1.00 and recall 1.00 on linear and heavy-tail series; recall 0.86 to 0.87 on nonlinear series, where
  it found 3 of 18 even links; with the added pass 18 of 18, at precision 0.82 to 0.92 on the nonlinear
  series and 0.94 to 1.00 elsewhere. The vector autoregression had the same recall as PCMCI+ and
  lower precision (0.79 to 0.95). Marginal correlation had precision near 0.2.
- The added pass was written after seeing PCMCI+ miss the even links in that development run.
- **What this cannot show.** The series satisfy PCMCI+'s assumptions: no hidden common cause, no
  links within a time step, a fixed graph. Market data break all three. The vector autoregression is
  a strong baseline here because the systems are small and mostly linear; the comparison in
  expectation 3 is about false links, not about being a better method overall.
- The generator and the scoring are mine. PCMCI+ itself is the Tigramite implementation, unmodified.

## Deviation rule

The run is not repeated. A crash is fixed only if the fix cannot change a number, the fix is
recorded with its date in the result, and the digest is re-frozen with the old one kept. Any other
change makes the result a development result.
