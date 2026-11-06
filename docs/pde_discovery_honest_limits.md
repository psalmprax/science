# PDE/ODE discovery — honest limits

This document exists so the discovery layer is judged honestly, not on its cleanest
canonical demos alone.

## Verified strong points (this layer)

- Frozen-operator holdout in `apedp_dyn/epistemic/pde_discovery.py`: dev-discovered
  coefficients are scored on holdout residuals using mirrored stencils, not re-fitted
  on the holdout window.
- Scale-aware STRidge in the Rust kernel (`rime/pde.rs`): column normalization before the
  solve, coefficient rescaling back, relative pruning to the largest active contribution,
  and a significance rescue with a documented rationale.
- Degeneracy guards: constant-field R² handling, failed-linear-solve disables the
  significance rescue, unknown frozen-transfer terms raise.
- Nonuniform discovery path in Rust using Fornberg weights.
- Pinned claim integrity test that regenerates headline numbers and checks the artifact
  on disk says the same thing.

## Honest weaknesses (this layer)

- Default PDE config is brittle to small additive noise. On the clean canonical sims from
  the pinned tests, the kernel returns the right active terms and close coefficients. Once
  small noise is added, the default library + default sparsity threshold often pulls in
  spurious nonlinear/advection terms and the fit quality degrades. This is not a missing
  feature so much as an unfixed robustness issue in the default configuration.
- The pinned demos are clean, low-noise, well-resolved sims with very specific parameters.
  They prove the engine can nail those cases. They are not, by themselves, proof of strong
  noise robustness under default settings.
- “Exact recovery” is mostly a fit-quality gate. It does not, by itself, protect strongly
  against a decent-fit-but-wrong-structure outcome beyond relative thresholding.
- Library size is a real trade-off. The richer default library helps on clean cases and
  becomes an alias source in noise. Trimming the library reduces spurious terms but can
  also reduce fit on the noisy case — there is no free lunch under default settings.
- `integrated_trajectory_r2` can be `None` in some noisy ODE probes while `jet_rmse` stays
  finite. Callers must handle `None`, not assume a number.

## What this repo can and cannot claim about this layer

- It can honestly claim strong, well-engineered clean-case discovery with disciplined
  holdout and sparsity mechanics.
- It cannot honestly claim, without additional evidence, that the default PDE config is a
  strong general-purpose noise-robust PDE discovery method across equations and seeds.

## Further honest work not done here

- A fixed-config, multi-seed, multi-equation noise sweep with soft metrics and spurious-term
  accounting. A first version of that pursuit lives in the test added alongside this note.
- A comparison of default config against a deliberately trimmed/regularized config so the
  robustness trade-off is exposed as data rather than asserted.

## Relation to the blind-benchmark weak-form work done in this session

The blind-benchmark weak-form fitter is a different method from this repo’s PDE discovery
path. The stability-selection work there is not a fix to this layer. This document keeps that
boundary explicit.
