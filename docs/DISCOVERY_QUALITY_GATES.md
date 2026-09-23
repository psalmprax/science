# Discovery Quality Gates

`apedp.discovery_quality` wraps the existing symbolic search without changing its grammar or fitness function.

It adds three measurements:

- `train_mse`: fit on the supplied observations.
- `expanded_output_rms`: output scale on an unlabeled shell beyond the observed input range. This is an extrapolation safety diagnostic, not a holdout accuracy score.
- `stability_rate`: fraction of independent search seeds producing finite, domain-valid candidates.

Use `multi_seed_discovery()` when a single evolutionary seed would be misleading. Use `screen_candidate()` before promoting a candidate to prospective validation.

These gates do not establish physical truth. They reduce obvious extrapolation and seed-instability failures before the locked-holdout and falsification stages.
