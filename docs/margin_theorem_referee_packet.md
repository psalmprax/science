# Margin-theorem referee packet (adjudication-ready evidence pack)

Status: proposer-side evidence ONLY. The acceptance of the margin theorem
into the certified boundary is promotion-checklist item 2 of
[`docs/cbf_proximity_graph_backlog_gate.md`](cbf_proximity_graph_backlog_gate.md);
its DECISION belongs to the frozen referee protocol — a proposer never
decides its own acceptance (AGENTS.md §1.1). Machine artifact:
[`benchmarks/margin_theorem_referee_packet.json`](../benchmarks/margin_theorem_referee_packet.json),
regenerable, deterministic checks, builder exits nonzero on any failure:
`python3 benchmarks/run_margin_theorem_referee_packet.py`.

## What is adjudication-ready here

The theorem: under the certified module's EXACT assumption box
(`A1 |v_i| ≤ v_max`, `A2 γ·dt ≤ 0.5`, `A5 deterministic numerics`), the
worst-case in-box relative velocity saturates the paired barrier law
`hdot_ij ≥ −γ·h_ij` at exactly `h = h* := 2·v_max/γ`, never violates it —
so pairs at or beyond h* need NO explicit constraint element. The
restricted solver then constrains only `h_ij < h*` pairs and re-checks
ALL pairs fail-closed every release.

### Machine checks now live in the artifact (all passing, values recorded verbatim)

| Check | Result |
|---|---|
| `exact_saturation_at_h_star` | worst-case anti-aligned top-speed pair at h = h*: (v_i−v_j)·n = −2·v_max = −γ·h* to <1e-12 — SATURATION, not violation |
| `headroom_above_h_star` | strictly positive slack at h = 1.001·h* |
| `box_worst_case_bound_sampled` | min over 20k in-assumption (norm-ball) commands/orientations: −1.913 > −2·v_max (Cauchy–Schwarz floor is constructive, never crossed) |
| `restricted_solve_all_pairs_recheck_honesty` | 5 seeded grid fixtures: released commands pass the FULL all-pairs re-check (worst deficit ≤ tol) with constraint ratios far below 1 |
| `unsolved_nominal_refused_fail_closed` | max_iter=0 on a provably-violated near pair reaches the re-check and RAISES (never silently released) |
| `proximity_horizon_model_check` | 8-agent closing grid, 60 steps through the PRODUCTION arc stepper: min_h_observed = 2.28 > 0 (measured evidence, NOT a formal claim) |

### Why the box-worst-case check samples the NORM BALL (recorded correction)

The theorem's A1 is `|v_i| ≤ v_max` as MAGNITUDE; the certified solver
projects box-clipped commands onto the norm ball at release. Sampling the
per-axis box instead would sample OUTSIDE A1 (a (√2,√2)-magnitude vector
satisfies the box while violating the magnitude bound) and could falsely
"fail" −2·v_max. The packet records this correction openly.

### Claim boundary (unchanged from the pitch module)

- sufficient-condition argument under A1/A2/A5 — nothing stronger than the
  certified module's assumption box;
- the unicycle realization is MODEL-CHECKED evidence, never a formal claim;
- kernel parity for the restricted solver is NOT claimed (python mirror
  provenance);
- production wiring stays behind the N ≥ 256 real-time gate (default cap
  N ≤ 64 does not trigger it);
- the FULL all-pairs fail-closed re-check stays in EVERY release, forever —
  the compositional honesty floor.

## Verification pointers (in the artifact, with SHA-256s)

- pitch module `apedp_dyn/robotics/proximity_margin.py`,
- certified module `apedp_dyn/robotics/cbf_certificate.py` (the exact
  assumption box source),
- pin suite `tests/test_dyn_proximity_margin_pins.py` (10 pins),
- certificate battery artifact `benchmarks/cbf_certificate_report.json`,
- this packet's builder script (re-runnable, exit-code gated).

## Decision slot

`"status": "UNADJUDICATED"` — the artifact ends here DELIBERATELY. The
deciding layer (frozen referee/adjudication protocol) must record its
verdict, date, and evidence fingerprint in that slot. Until then, item 2
of the proximity-lock promotion checklist remains open, and the
restricted certified path stays `experimental`, wired to nothing.
