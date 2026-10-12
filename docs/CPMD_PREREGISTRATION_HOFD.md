# Preregistration: CPMD-HOFD — Daily Order-Flow Imbalance Reversal

**Document ID:** PREREG-CPMD-HOFD-2026-10-11
**Code:** `apedp_dyn/sandbox/cpmd_hofd.py` (committed SEPARATELY after)
**Standing:** NEW hypothesis (8th candidate). All prior verdicts stand.
  Uses available DAILY aggressor/OFI columns — a different mechanism and
  frequency from H-LRF (minute depth); no H-LRF rule is bent.
**Limitation recorded:** equity OFI is a vendor proxy (see PROVENANCE.json);
  crypto OFI is exchange taker data. Results labeled per panel.

## Hypothesis H-OFD (frozen)

Extreme daily aggressor imbalance mean-reverts overnight at the asset level:

$$s_{i,t} = \frac{OFI_{i,t} - \mu^{(60)}_{i,t}}{\sigma^{(60)}_{i,t} + \epsilon}, \qquad \boxed{\hat{r}_{i,t+1} = -\sigma_{i,t}\,\tanh(s_{i,t})}$$

$s$ = 60-day trailing z-scored daily OFI (past data only); $\sigma_{i,t}$ =
trailing-20 return std. Predicts reversal after one-sided aggressive flow.
Directional, falsifiable; novelty vs literature unclaimed.

## Controls (frozen)

MOM: $\hat{r} = \mathrm{sign}(\Sigma_5 r)·\sigma_t$. MARKET: $\hat{r} =
\hat{\beta}_{i,t}·m_t$ (past-100 beta × median standardized return).
Primary: pooled panel IC + circular-shift null (200/panel), $p < 0.05$,
MSE ratio < 1. Incremental: residualize on [MOM, MARKET], residual IC null
$p < 0.05$.

## Discovery / confirmation (no tuning anywhere)

- Discovery: CRYPTO panel (1095d × 15, OFI native). Pass = primary + incremental.
- Confirmation (ONLY if discovery passes): EQUITY last-40% + MULTI_ASSET_30
  panel (OFI columns never used by any prior test → genuine confirmation).
  Same per-panel gates. Any fail → H-OFD REJECTED.
- No trading evaluation regardless (needs its own prereg if prediction passes).
