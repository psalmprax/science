# H-LRF Status: DEFERRED — Data Requirement Unmet (audited, not approximated)

**Verdict:** H-LRF remains UNTESTED. Per its own proposal, a daily proxy
would be a misleading approximation, so none was run.

## Data audit (2026-10-11, committed record)

All local order-flow files are DAILY single-row-per-date:

| File | Rows | Granularity | Has aggressor vol? | Has depth snapshots? |
|---|---|---|---|---|
| crypto_top15_orderflow.csv | 1096 | daily | yes (TAKER_BUY, daily total) | NO |
| multi_asset_30/60_orderflow.csv | 754 | daily | yes (daily total) | NO |
| multi_asset_90_5yr_orderflow.csv | 1256 | daily | yes (daily total) | NO |

H-LRF requires, per event: one-minute aggressor volumes ($V^{buy}, V^{sell}$),
depth immediately before/after the shock ($D^{pre}$, $D^{post}$), depth one
minute later ($D^{refill}$), and minute mid-prices $M_{t+1} \ldots M_{t+6}$:

$$O_{i,t} = \frac{V^{buy}_{i,t} - V^{sell}_{i,t}}{V^{buy}_{i,t} + V^{sell}_{i,t} + \epsilon}, \qquad R_{i,t} = \operatorname{clip}\!\left(\frac{D^{refill}_{i,t} - D^{post}_{i,t}}{D^{pre}_{i,t} - D^{post}_{i,t}},\, 0,\, 1\right)$$

$$Y_{i,t} = \log\!\left(\frac{M_{t+6}}{M_{t+1}}\right), \qquad Y_{i,t} = \alpha + \beta_P O_{i,t}(1-R_{i,t}) + \beta_R O_{i,t}R_{i,t} + \boldsymbol{\theta}^{\top}X_{i,t} + \varepsilon_{i,t}$$

with the directional prediction $\boxed{\beta_P - \beta_R > 0}$.
None of $D^{pre}$, $D^{post}$, $D^{refill}$, minute $V^{buy}/V^{sell}$, or
minute $M$ exist locally. Daily TAKER_BUY/OFI cannot substitute (proposal
§4, frozen safeguard).

## What would unblock H-LRF

Historical L2 order-book snapshots + aggressor-flagged trades at minute (or
finer) resolution: exchange historical data products, commercial L2 vendors,
or self-captured colocation feeds. Public spot APIs expose current depth
only, not history.

## Disposition

H-LRF: DEFERRED (untestable ≠ rejected ≠ supported). No proxy test run, by
design. Revisit only when qualifying data exists; at that point it needs a
fresh preregistration, not this document.
