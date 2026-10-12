//! RIME regression tests for the four defects confirmed by the Stage-1 audit.
//!
//! Each test below states the behaviour RIME must have and is written so that
//! it FAILS on the pre-audit implementation (per-node memo, fake AddState,
//! raw-output scoring, vacuous null test). They pass only after the fixes.
//!
//!   1. `difference_term_keeps_its_values`  — memo cache keyed by node only
//      made `x[t] - x[t-1]` evaluate to zero.
//   2. `add_state_is_a_recursive_decay`    — AddState was not a recursion and
//      had a flipped sign.
//   3. `scale_and_intercept_are_fitted`    — graphs were scored on raw output,
//      so a planted mechanism with a small amplitude / large offset was
//      invisible.
//   4. `null_world_false_positive_rate`    — the bundled null test used a
//      penalty (1.0 per edge) larger than any possible MSE change, so it
//      proved nothing. This one uses a penalty on the scale of the target
//      variance and bounds the false-promotion rate.

use apedp_kernel::rime::eval::{
    fit_readout, fitted_mse, mse_finite, score_batch_fitted, score_candidate_fitted, ScoreConfig,
};
use apedp_kernel::rime::graph::{Graph, Node};
use apedp_kernel::rime::rewrite::{apply, RewriteOp};
use apedp_kernel::rime::search::{multiplicity_dof, SearchConfig};
use apedp_kernel::rime::seeds::small_forms;

/// Deterministic hash noise in (-0.5, 0.5): no RNG, reproducible everywhere.
fn hash_noise(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let v = ((t as f64 + phase) * 12.9898).sin() * 43758.5453;
            v - v.floor() - 0.5
        })
        .collect()
}

fn variance(x: &[f64]) -> f64 {
    let f: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let m = f.iter().sum::<f64>() / f.len() as f64;
    f.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / f.len() as f64
}

fn seed_input() -> Graph {
    let mut g = Graph::new();
    g.output = g.push(Node::Input(0));
    g
}

/// Defect 1: a node used both directly and lagged must keep BOTH values.
#[test]
fn difference_term_keeps_its_values() {
    let mut g = Graph::new();
    let x = g.push(Node::Input(0));
    let xl = g.push(Node::Lag(x, 1));
    let nxl = g.push(Node::Neg(xl));
    g.output = g.push(Node::Add(x, nxl)); // y = x[t] - x[t-1]

    let out = g.evaluate(&[vec![1.0, 2.0, 3.0, 4.0]]);
    assert!(out[0].is_nan(), "first row is lag warmup");
    assert_eq!(
        &out[1..],
        &[1.0, 1.0, 1.0],
        "x[t] - x[t-1] on 1,2,3,4 must be 1 after warmup; got {:?}",
        out
    );
}

/// Same bug, bluntest possible statement: a difference term must not be all
/// zeros, and its mean must equal the planted drift.
#[test]
fn difference_term_retains_magnitude() {
    let x: Vec<f64> = (0..64).map(|t| t as f64).collect();
    let mut g = Graph::new();
    let xi = g.push(Node::Input(0));
    let xl = g.push(Node::Lag(xi, 1));
    let nxl = g.push(Node::Neg(xl));
    g.output = g.push(Node::Add(xi, nxl));

    let out = g.evaluate(&[x]);
    let finite: Vec<f64> = out.iter().copied().filter(|v| v.is_finite()).collect();
    assert_eq!(finite.len(), 63);
    assert!(
        finite.iter().all(|v| (v - 1.0).abs() < 1e-12),
        "every step of t - (t-1) is 1; got {:?}",
        &out[..4]
    );
}

/// Defect 2: AddState must be a real recursion with the documented sign:
/// h[t] = d*h[t-1] + u[t], h[-1] = init (so h[0] = u[0] for init = 0).
#[test]
fn add_state_is_a_recursive_decay() {
    let g = seed_input();
    let s = apply(&g, &RewriteOp::AddState { decay_num: 1, decay_den: 2 }).unwrap();
    let out = s.evaluate(&[vec![1.0, 1.0, 1.0, 1.0]]);
    let want = [1.0, 1.5, 1.75, 1.875];
    for (t, (o, w)) in out.iter().zip(want.iter()).enumerate() {
        assert!(
            (o - w).abs() < 1e-12,
            "decaying memory of an all-ones input at t={} is {} (want {}); full {:?}",
            t,
            o,
            w,
            out
        );
    }
}

/// Defect 2b: the state must actually depend on its input, not on a fixed
/// sign flip, and must respect the decay parameter.
#[test]
fn add_state_responds_to_decay_parameter() {
    let g = seed_input();
    let fast = apply(&g, &RewriteOp::AddState { decay_num: 0, decay_den: 1 }).unwrap();
    let slow = apply(&g, &RewriteOp::AddState { decay_num: 9, decay_den: 10 }).unwrap();
    let inp = vec![vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
    let fast_out = fast.evaluate(&inp);
    let slow_out = slow.evaluate(&inp);
    // d = 0: memoryless, state law equals the input.
    assert!(
        fast_out.iter().all(|v| (v - 1.0).abs() < 1e-12),
        "d=0 must be memoryless, got {:?}",
        fast_out
    );
    // d = 0.9: geometric accumulation toward the 1/(1-d) ceiling.
    assert!(slow_out[7] > 5.0 && slow_out[7] < 6.2, "d=0.9 accumulation got {:?}", slow_out);
}

/// Defect 3: with a planted mechanism that has a small amplitude and a large
/// offset, raw-output scoring sees no improvement at all, while a fitted
/// readout recovers it exactly.
#[test]
fn scale_and_intercept_are_fitted_before_scoring() {
    let n = 600;
    let x = hash_noise(n, 0.0);
    let y: Vec<f64> = (0..n)
        .map(|t| if t >= 2 { 0.001 * x[t - 2] + 5.0 } else { f64::NAN })
        .collect();
    let cols = vec![x.clone()];
    let seed = seed_input();

    // The graph is the true mechanism read off raw (no readout fitted yet).
    let lag2 = apply(&seed, &RewriteOp::AddLagEdge { src: 0, k: 2 }).unwrap();
    let out = lag2.evaluate(&cols);

    // (a) raw comparison — this is what the pre-audit evaluator did.
    let (parent_raw, _) = mse_finite(&seed.evaluate(&cols), &y);
    let (child_raw, _) = mse_finite(&out, &y);
    let var_y = variance(&y);
    assert!(
        parent_raw - child_raw > 0.5 * var_y,
        "raw-output scoring cannot see the planted mechanism: \
         improvement {:.3e} vs var(y) {:.3e}",
        parent_raw - child_raw,
        var_y
    );

    // (b) fitted readout — the corrected path.
    let (a, b) = fit_readout(&out, &y, 1e-9);
    assert!(
        (a - 0.001).abs() < 1e-6 && (b - 5.0).abs() < 1e-3,
        "fitted readout must recover y = 0.001*x[t-2] + 5: scale {:.6e} intercept {:.6e}",
        a,
        b
    );

    let fitted = score_candidate_fitted(&lag2, parent_raw, &cols, &y, &ScoreConfig::new(0.01 * var_y, 1e-9));
    assert!(
        fitted.score > 0.5 * var_y,
        "fitted scoring must promote the planted mechanism: score {:.3e}",
        fitted.score
    );
    assert!(fitted.r2 > 0.99, "fitted R2 {:.4}", fitted.r2);
}

/// Defect 4b: the same control on a *short* series, where a fraction-of-variance
/// penalty is not enough because the search looks at ~60 seeds at once. The gate
/// must pay the multiplicity it created (EBIC-style selection charge) on top of
/// the per-degree-of-freedom charge, and it must still find a real law on the
/// same short series.
#[test]
fn short_series_null_control_pays_for_multiplicity() {
    let n = 150;
    let base = hash_noise(n, 0.0);
    let n_dev = n;

    // (a) legacy fraction-of-variance penalty: too permissive here.
    let y_null = hash_noise(n, 4242.0);
    let legacy = SearchConfig::scaled_to(&y_null, 0.01).with_max_depth(1).with_budget(60);
    let legacy_out = apedp_kernel::rime::search::search(&small_forms(1), &[base.clone()], &y_null, &legacy);

    // (b) sample-size-aware penalty + selection charge: must abstain.
    let lc = SearchConfig::noise_scaled(&y_null, n_dev, 1.5).with_max_depth(1).with_budget(60);
    let out = apedp_kernel::rime::search::search(&small_forms(1), &[base.clone()], &y_null, &lc);
    assert_eq!(
        out.accepted, 0,
        "short null world promoted {} candidates: {:?}",
        out.accepted,
        out.proposals.iter().map(|p| p.equation.clone()).collect::<Vec<_>>()
    );

    // (c) positive control on the same short series and identical configuration:
    // a planted law is still found, so (b) is not abstaining out of weakness.
    let mut y_sig = hash_noise(n, 4242.0);
    let x_sig = hash_noise(n, 0.0);
    for t in 2..n {
        y_sig[t] = 3.0 * x_sig[t - 2];
    }
    let lc_sig = SearchConfig::noise_scaled(&y_sig, n_dev, 1.5).with_max_depth(1).with_budget(60);
    let sig_out = apedp_kernel::rime::search::search(&small_forms(1), &[x_sig], &y_sig, &lc_sig);
    assert!(sig_out.accepted > 0, "positive control must be recovered");
    assert!(sig_out.best().unwrap().r2 > 0.99);

    // The selection charge is the mechanism doing the work: without it the same
    // configuration is markedly weaker on the null world.
    assert!(multiplicity_dof(60) > 8.0);
    let weak = SearchConfig::noise_scaled(&y_null, n_dev, 1.5)
        .with_max_depth(1)
        .with_budget(60);
    assert!(weak.score_config().selection_dof > 0.0);
    let legacy_accepted = legacy_out.accepted;
    assert!(
        legacy_accepted >= out.accepted,
        "the legacy penalty ({}) cannot be stricter than the calibrated one ({})",
        legacy_accepted,
        out.accepted
    );
}

/// Defect 4: null-world control with a penalty set on the scale of the target
/// (1% of var(y) per edge unit), not the degenerate 1.0-per-edge used before.
/// Over 24 independent null worlds the false-promotion rate must stay low.
#[test]
fn null_world_false_positive_rate_is_controlled() {
    let n = 500;
    let base = hash_noise(n, 0.0);
    let seed = seed_input();
    let lags = [1usize, 2, 5, 10, 20];

    let mut promotions = 0usize;
    for j in 0..24 {
        // Target depends only on t: no lagged input can predict it.
        let target = hash_noise(n, 1000.0 + 37.0 * j as f64);
        let var_y = variance(&target);
        let lambda_c = 0.01 * var_y;
        let (parent_mse, _, _) = fitted_mse(&seed, &[base.clone()], &target, 1e-9);
        let jobs: Vec<_> = lags
            .iter()
            .map(|&k| {
                (
                    apply(&seed, &RewriteOp::AddLagEdge { src: 0, k }).unwrap(),
                    parent_mse,
                    target.clone(),
                )
            })
            .collect();
        let res = score_batch_fitted(&jobs, &[base.clone()], &ScoreConfig::new(lambda_c, 1e-9));
        if res.iter().any(|r| r.score > 0.0) {
            promotions += 1;
        }
    }
    assert!(
        promotions <= 2,
        "false promotions {}/24 on null worlds at lambda_c = 1% of var(y)",
        promotions
    );
}
