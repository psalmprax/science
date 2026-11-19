# Documentation "Verified (N/N)" rows — execution audit

Generated: 2026-11-08 06:13:51 CET
Harness: `scripts/audit_documentation_verified_rows.py` (audit-only; executions took 727.5s)
Registry: `docs/DOCUMENTATION.md` — **46 rows**, **58 distinct cited files**

Every cited file was executed with `pytest -q`. A row is `MATCH` only when
its stated count equals the measured pass count with zero failures. Rows
that cite no file, cite a missing file, or cite `cargo test` are reported
as unmeasurable rather than assumed correct.

## Summary

| Verdict | Rows |
|---|---|
| `MATCH` | 30 |
| `PARTIAL_MATCH` | 7 |
| `UNCOUNTED_CLAIM` | 5 |
| `MANUAL_REVIEW` | 2 |
| `NOT_AUTOMATED` | 2 |

## Per-file results

| Cited file | Status | Passed | Failed | Errors | Seconds |
|---|---|---|---|---|---|
| `benchmarks/run_bridge_fastpath_benchmark.py` | no_tests | 0 | 0 | 0 | 5.4 |
| `benchmarks/run_five_gate_validation.py` | no_tests | 0 | 0 | 0 | 5.9 |
| `benchmarks/run_jet_flow_video_benchmark.py` | no_tests | 0 | 0 | 0 | 9.2 |
| `benchmarks/run_jetflow_cpu_export_benchmark.py` | no_tests | 0 | 0 | 0 | 4.5 |
| `benchmarks/run_jetflow_distill_benchmark.py` | no_tests | 0 | 0 | 0 | 9.4 |
| `benchmarks/run_jetflow_vs_diffusion_matched.py` | no_tests | 0 | 0 | 0 | 8.6 |
| `benchmarks/run_population_extrapolation_rime_ad_vs_ga.py` | no_tests | 0 | 0 | 0 | 8.3 |
| `benchmarks/run_rime_microsample_adjudication.py` | no_tests | 0 | 0 | 0 | 4.4 |
| `benchmarks/run_vortex_lossy_regimes_refinetune.py` | no_tests | 0 | 0 | 0 | 9.3 |
| `benchmarks/run_vortex_slow_motion_demo.py` | no_tests | 0 | 0 | 0 | 9.0 |
| `tests/test_cross_implementation_conformance.py` | pass | 23 | 0 | 0 | 5.9 |
| `tests/test_dyn_all_subpackages.py` | pass | 10 | 0 | 0 | 11.1 |
| `tests/test_dyn_audio_field_transport.py` | pass | 6 | 0 | 0 | 9.5 |
| `tests/test_dyn_cluster_orchestration.py` | pass | 5 | 0 | 0 | 3.9 |
| `tests/test_dyn_coupled_and_2d_pde.py` | pass | 12 | 0 | 0 | 3.6 |
| `tests/test_dyn_cuda_batch_physics.py` | pass | 5 | 0 | 0 | 6.1 |
| `tests/test_dyn_dag_explorer.py` | pass | 4 | 0 | 0 | 5.0 |
| `tests/test_dyn_decoupled_continuous_transport.py` | pass | 6 | 0 | 0 | 11.1 |
| `tests/test_dyn_distributed_chunking.py` | pass | 6 | 0 | 0 | 9.1 |
| `tests/test_dyn_dmrg_quantum.py` | pass | 8 | 0 | 0 | 27.0 |
| `tests/test_dyn_five_gate_validation.py` | pass | 1 | 0 | 0 | 5.4 |
| `tests/test_dyn_fluid_domain.py` | pass | 4 | 0 | 0 | 12.2 |
| `tests/test_dyn_frontier_9_of_10.py` | pass | 5 | 0 | 0 | 17.9 |
| `tests/test_dyn_frontier_claims_integrity.py` | pass | 6 | 0 | 0 | 12.2 |
| `tests/test_dyn_high_dim_hybrid_pde.py` | pass | 33 | 0 | 0 | 7.9 |
| `tests/test_dyn_higher_order_pde_hypothesis_engine.py` | pass | 4 | 0 | 0 | 6.4 |
| `tests/test_dyn_integer_relation_methods.py` | pass | 8 | 0 | 0 | 3.0 |
| `tests/test_dyn_jet_flow_sota_path.py` | pass | 8 | 0 | 0 | 11.9 |
| `tests/test_dyn_jet_flow_video.py` | pass | 8 | 0 | 0 | 18.4 |
| `tests/test_dyn_jetflow_anchored_and_distill.py` | pass | 5 | 0 | 0 | 80.2 |
| `tests/test_dyn_jetflow_cpu_export.py` | pass | 3 | 0 | 0 | 10.8 |
| `tests/test_dyn_jetflow_perceptual_decoder.py` | pass | 3 | 0 | 0 | 14.4 |
| `tests/test_dyn_jetflow_video_package.py` | pass | 10 | 0 | 0 | 12.7 |
| `tests/test_dyn_landauer_report_fidelity.py` | pass | 4 | 0 | 0 | 3.8 |
| `tests/test_dyn_lob_transport.py` | pass | 5 | 0 | 0 | 4.4 |
| `tests/test_dyn_magic_state.py` | pass | 4 | 0 | 0 | 3.8 |
| `tests/test_dyn_molecular_se3_flow.py` | pass | 9 | 0 | 0 | 9.6 |
| `tests/test_dyn_multi_agent_robotics.py` | pass | 6 | 0 | 0 | 5.9 |
| `tests/test_dyn_ncm_proposer.py` | pass | 5 | 0 | 0 | 7.4 |
| `tests/test_dyn_notebook_adapter.py` | pass | 30 | 0 | 0 | 61.9 |
| `tests/test_dyn_number_theory_and_invariant_transfer.py` | pass | 5 | 0 | 0 | 5.4 |
| `tests/test_dyn_pde_ode_packages.py` | pass | 11 | 0 | 0 | 6.7 |
| `tests/test_dyn_pde_transfer.py` | pass | 4 | 0 | 0 | 3.1 |
| `tests/test_dyn_planck_limitation.py` | pass | 5 | 0 | 0 | 5.9 |
| `tests/test_dyn_population_extrapolation_benchmark.py` | pass | 3 | 0 | 0 | 1.7 |
| `tests/test_dyn_prior_guided_unsolicited.py` | pass | 7 | 0 | 0 | 45.2 |
| `tests/test_dyn_remote_storage.py` | pass | 11 | 0 | 0 | 4.5 |
| `tests/test_dyn_rime_microsample_adjudication.py` | pass | 24 | 0 | 0 | 10.0 |
| `tests/test_dyn_robotics_trajectory.py` | pass | 6 | 0 | 0 | 9.2 |
| `tests/test_dyn_rust_all_equations_benchmark.py` | pass | 6 | 0 | 0 | 17.1 |
| `tests/test_dyn_rust_koopman_vs_ga.py` | pass | 3 | 0 | 0 | 6.4 |
| `tests/test_dyn_rust_lie_symmetry.py` | pass | 3 | 0 | 0 | 1.5 |
| `tests/test_dyn_rust_parity_harness.py` | pass | 6 | 0 | 0 | 9.4 |
| `tests/test_dyn_shape_inference.py` | pass | 15 | 0 | 0 | 89.7 |
| `tests/test_dyn_video_eval_gate.py` | pass | 5 | 0 | 0 | 13.8 |
| `tests/test_dyn_video_metrics_kernel.py` | pass | 8 | 0 | 0 | 10.4 |
| `tests/test_rust_bridge_dispatch.py` | pass | 10 | 0 | 0 | 5.7 |
| `tests/test_rust_bridge_observability.py` | pass | 8 | 0 | 0 | 5.6 |

## Per-row verdicts

| DOCUMENTATION.md line | Verdict | Claim | Finding |
|---|---|---|---|
| 1962 | `MATCH` | 1/2, 1/4, 8/8, 8/10 | claimed 8/8, measured 8 passed (`tests/test_dyn_dmrg_quantum.py`) |
| 1964 | `MATCH` | 5/5 | claimed 5/5, measured 5 passed (`tests/test_dyn_lob_transport.py`) |
| 1965 | `MATCH` | 1/2, 6/6 | claimed 6/6, measured 6 passed (`tests/test_dyn_audio_field_transport.py`) |
| 1966 | `MATCH` | 9/9 | claimed 9/9, measured 9 passed (`tests/test_dyn_molecular_se3_flow.py`) |
| 1967 | `MATCH` | 6/6, 20/20 | claimed 6/6, measured 6 passed (`tests/test_dyn_robotics_trajectory.py`) |
| 2182 | `MATCH` | 7/7 | claimed 7/7, measured 7 passed (`tests/test_dyn_prior_guided_unsolicited.py`) |
| 2183 | `MATCH` | 4/4 | claimed 4/4, measured 4 passed (`tests/test_dyn_landauer_report_fidelity.py`) |
| 2184 | `MATCH` | 15/15 | claimed 15/15, measured 15 passed (`tests/test_dyn_shape_inference.py`) |
| 2188 | `MATCH` | 6/6 | claimed 6/6, measured 6 passed (`tests/test_dyn_rust_all_equations_benchmark.py`) |
| 2189 | `PARTIAL_MATCH` | 3/3 | claimed 3/3; per-file passes {'benchmarks/run_population_extrapolation_rime_ad_vs_ga.py': 0, 'tests/test_dyn_population_extrapolation_benchmark.py': 3}, aggregate 3 — the claim does not name which file it counts (`benchmarks/run_population_extrapolation_rime_ad_vs_ga.py`, `tests/test_dyn_population_extrapolation_benchmark.py`) |
| 2190 | `PARTIAL_MATCH` | 6/6 | claimed 6/6; per-file passes {'tests/test_dyn_rust_koopman_vs_ga.py': 3, 'tests/test_dyn_rust_lie_symmetry.py': 3}, aggregate 6 — the claim does not name which file it counts (`tests/test_dyn_rust_koopman_vs_ga.py`, `tests/test_dyn_rust_lie_symmetry.py`) |
| 2191 | `MANUAL_REVIEW` | 8/9 | counts [(8, 9)] are not of the N/N form (`benchmarks/run_five_gate_validation.py`, `tests/test_dyn_five_gate_validation.py`, `tests/test_dyn_planck_limitation.py`) |
| 2193 | `NOT_AUTOMATED` | 45/45, 26/26 | row also cites `cargo test`, which this harness does not run (`benchmarks/run_bridge_fastpath_benchmark.py`, `tests/test_cross_implementation_conformance.py`, `tests/test_rust_bridge_dispatch.py`, `tests/test_rust_bridge_observability.py`) |
| 2194 | `PARTIAL_MATCH` | 5/5, 4/4 | claimed 4/4; per-file passes {'tests/test_dyn_frontier_9_of_10.py': 5, 'tests/test_dyn_frontier_claims_integrity.py': 6, 'tests/test_dyn_pde_transfer.py': 4}, aggregate 15 — the claim does not name which file it counts (`tests/test_dyn_frontier_9_of_10.py`, `tests/test_dyn_frontier_claims_integrity.py`, `tests/test_dyn_pde_transfer.py`) |
| 2195 | `MATCH` | 5/5, 2/5, 49/20, 1/8 | claimed 5/5, measured 5 passed (`tests/test_dyn_number_theory_and_invariant_transfer.py`) |
| 2196 | `MATCH` | 12/12 | claimed 12/12, measured 12 passed (`tests/test_dyn_coupled_and_2d_pde.py`) |
| 2197 | `MATCH` | 33/33, 11/11, 0/300, 100/100 | claimed 33/33, measured 33 passed (`tests/test_dyn_high_dim_hybrid_pde.py`) |
| 2198 | `MATCH` | 10/10, 517/621 | claimed 10/10, measured 10 passed (`tests/test_dyn_jetflow_video_package.py`) |
| 2199 | `UNCOUNTED_CLAIM` | qualitative | row asserts a qualitative result; cited file(s) pass (16 passed) but no count is claimed (`tests/test_dyn_jet_flow_sota_path.py`, `tests/test_dyn_jet_flow_video.py`) |
| 2200 | `MATCH` | 5/5 | claimed 5/5, measured 5 passed (`tests/test_dyn_video_eval_gate.py`) |
| 2201 | `PARTIAL_MATCH` | 17/17, 5/5, 4/4, 8/8 | claimed 4/4; per-file passes {'tests/test_dyn_fluid_domain.py': 4, 'tests/test_dyn_video_eval_gate.py': 5, 'tests/test_dyn_video_metrics_kernel.py': 8}, aggregate 17 — the claim does not name which file it counts (`tests/test_dyn_fluid_domain.py`, `tests/test_dyn_video_eval_gate.py`, `tests/test_dyn_video_metrics_kernel.py`) |
| 2202 | `MATCH` | 4/4 | claimed 4/4, measured 4 passed (`tests/test_dyn_fluid_domain.py`) |
| 2203 | `PARTIAL_MATCH` | 11/11 | claimed 11/11; per-file passes {'benchmarks/run_jetflow_cpu_export_benchmark.py': 0, 'benchmarks/run_jetflow_distill_benchmark.py': 0, 'tests/test_dyn_jetflow_anchored_and_distill.py': 5, 'tests/test_dyn_jetflow_cpu_export.py': 3, 'tests/test_dyn_jetflow_perceptual_decoder.py': 3}, aggregate 11 — the claim does not name which file it counts (`benchmarks/run_jetflow_cpu_export_benchmark.py`, `benchmarks/run_jetflow_distill_benchmark.py`, `tests/test_dyn_jetflow_anchored_and_distill.py`, `tests/test_dyn_jetflow_cpu_export.py`, `tests/test_dyn_jetflow_perceptual_decoder.py`) |
| 2204 | `UNCOUNTED_CLAIM` | qualitative | row asserts a qualitative result; cited file(s) pass (11 passed) but no count is claimed (`tests/test_dyn_pde_ode_packages.py`) |
| 2205 | `MATCH` | 6/6 | claimed 6/6, measured 6 passed (`tests/test_dyn_decoupled_continuous_transport.py`) |
| 2206 | `PARTIAL_MATCH` | 10/10 | claimed 10/10; per-file passes {'benchmarks/run_jetflow_vs_diffusion_matched.py': 0, 'tests/test_dyn_jetflow_video_package.py': 10}, aggregate 10 — the claim does not name which file it counts (`benchmarks/run_jetflow_vs_diffusion_matched.py`, `tests/test_dyn_jetflow_video_package.py`) |
| 2207 | `UNCOUNTED_CLAIM` | qualitative | row asserts a qualitative result; cited file(s) pass (3 passed) but no count is claimed (`benchmarks/run_vortex_lossy_regimes_refinetune.py`, `tests/test_dyn_jetflow_perceptual_decoder.py`) |
| 2208 | `MANUAL_REVIEW` | 60/120 | counts [(60, 120)] are not of the N/N form (`benchmarks/run_jet_flow_video_benchmark.py`, `benchmarks/run_vortex_slow_motion_demo.py`, `tests/test_dyn_jet_flow_video.py`, `tests/test_dyn_jetflow_perceptual_decoder.py`) |
| 2209 | `MATCH` | 4/4 | claimed 4/4, measured 4 passed (`tests/test_dyn_higher_order_pde_hypothesis_engine.py`) |
| 2210 | `MATCH` | 6/6 | claimed 6/6, measured 6 passed (`tests/test_dyn_distributed_chunking.py`) |
| 2211 | `MATCH` | 6/6 | claimed 6/6, measured 6 passed (`tests/test_dyn_rust_parity_harness.py`) |
| 2212 | `PARTIAL_MATCH` | 30/30, 12/12 | claimed 30/30; per-file passes {'benchmarks/run_rime_microsample_adjudication.py': 0, 'tests/test_dyn_notebook_adapter.py': 30}, aggregate 30 — the claim does not name which file it counts (`benchmarks/run_rime_microsample_adjudication.py`, `tests/test_dyn_notebook_adapter.py`) |
| 2213 | `UNCOUNTED_CLAIM` | qualitative | row asserts a qualitative result; cited file(s) pass (10 passed) but no count is claimed (`tests/test_dyn_all_subpackages.py`) |
| 2214 | `MATCH` | 11/11 | claimed 11/11, measured 11 passed (`tests/test_dyn_remote_storage.py`) |
| 2215 | `UNCOUNTED_CLAIM` | qualitative | row asserts a qualitative result; cited file(s) pass (8 passed) but no count is claimed (`tests/test_dyn_integer_relation_methods.py`) |
| 2217 | `MATCH` | 5/5 | claimed 5/5, measured 5 passed (`tests/test_dyn_cuda_batch_physics.py`) |
| 2218 | `MATCH` | 5/5 | claimed 5/5, measured 5 passed (`tests/test_dyn_cluster_orchestration.py`) |
| 2219 | `MATCH` | 4/4 | claimed 4/4, measured 4 passed (`tests/test_dyn_dag_explorer.py`) |
| 2220 | `MATCH` | 4/4 | claimed 4/4, measured 4 passed (`tests/test_dyn_magic_state.py`) |
| 2221 | `MATCH` | 6/6 | claimed 6/6, measured 6 passed (`tests/test_dyn_multi_agent_robotics.py`) |
| 2222 | `MATCH` | 5/5 | claimed 5/5, measured 5 passed (`tests/test_dyn_ncm_proposer.py`) |
| 2223 | `NOT_AUTOMATED` | 24/24, 12/12, 186/186 | row also cites `cargo test`, which this harness does not run (`benchmarks/run_rime_microsample_adjudication.py`, `tests/test_dyn_rime_microsample_adjudication.py`) |
| 2224 | `MATCH` | 24/24, 11/11, 12/12 | claimed 24/24, measured 24 passed (`tests/test_dyn_rime_microsample_adjudication.py`) |
| 2225 | `MATCH` | 24/24, 11/11, 0/11 | claimed 24/24, measured 24 passed (`tests/test_dyn_rime_microsample_adjudication.py`) |
| 2226 | `MATCH` | 24/24, 11/11, 11/11 | claimed 24/24, measured 24 passed (`tests/test_dyn_rime_microsample_adjudication.py`) |
| 2227 | `MATCH` | 24/24, 12/12, 12/12 | claimed 24/24, measured 24 passed (`tests/test_dyn_rime_microsample_adjudication.py`) |

## How to read this

- `MATCH` — the row's count reproduces. The claim is still only as strong
  as the test it cites: a tautological assertion reproduces too.
- `PARTIAL_MATCH` — the count appears in the data but the row does not say
  which cited file it counts.
- `MISMATCH` / `MEASURED_FAILURE` — a documented result is not what the
  cited test currently does. Either the row or the test is wrong.
- `NO_CITED_FILE` / `NOT_AUTOMATED` — the row cannot be checked by running
  pytest. These are the rows a reader has to trust, and the ones a
  registry should either cite a runnable file for or stop counting.

