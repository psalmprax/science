//! RIME Stage-1 acceptance: planted-mechanism recovery + null abstention.
//! Uses only the frozen core (graph + rewrites + evaluator + records).
//!
//! v0.1.1: the audit found the bundled null test vacuous — it charged 1.0 per
//! edge while every MSE in the world under test was around 0.08, so no
//! candidate could ever score positive and the test proved nothing. The control
//! below now sets the penalty on the scale of the target variance and is
//! **paired with a positive control** through the identical scoring path: a
//! control that cannot promote a planted mechanism is not a control.

use apedp_kernel::rime::eval::{
    fit_readout, fitted_mse, mse_finite, score_batch_fitted, FittedScore, ScoreConfig,
};
use apedp_kernel::rime::graph::{Graph, Node};
use apedp_kernel::rime::rewrite::{apply, RewriteOp};
use apedp_kernel::rime::search::{multiplicity_dof, target_variance};

/// The planted lag-2 amplitude of [`planted_world`] (raised from 3.0 in
/// 30e73bf so the MSE reduction dominates the lambda_c charge decisively).
const PLANTED_AMP: f64 = 50.0;

fn hash_noise(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let v = ((t as f64 + phase) * 12.9898).sin() * 43758.5453;
            v - v.floor() - 0.5
        })
        .collect()
}

/// Deterministic input (hash noise). Measured sample autocorrelation at the
/// frozen lags is below 0.07, so no lag is a hidden copy of another. Fully
/// deterministic, no RNG. The target is exactly
/// y[t] = PLANTED_AMP * x[t-2] for t >= 2, so lag‑2 must be the uniquely best
/// model.
fn planted_world() -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = hash_noise(500, 0.0);
    // Exact lag‑2 law, no extra sinusoidal noise so that lag‑2 wins decisively.
    // Use a large amplitude so the MSE reduction dominates the lambda_c penalty.
    let y: Vec<f64> = (0..500)
        .map(|t| {
            if t >= 2 {
                PLANTED_AMP * x[t - 2]
            } else {
                f64::NAN
            }
        })
        .collect();
    (vec![x], y)
}

fn seed_graph() -> Graph {
    let mut g = Graph::new();
    g.output = g.push(Node::Input(0));
    g
}

/// The five frozen lag expansions of the seed, scored against the seed.
/// The search's own calibration, for a run that scores `n_candidates` forms:
/// `gamma * var(y) / n_dev` per charged degree of freedom, plus the EBIC-style
/// selection charge for looking at that many candidates.
fn lag_score_config(target: &[f64], n_candidates: usize) -> ScoreConfig {
    let var = target_variance(target);
    let n_dev = target.iter().filter(|v| v.is_finite()).count().max(1);
    // Charge multiplier 25.0 (re-calibrated after 30e73bf raised the planted
    // amplitude): at 50.0 the charge consumed ~62% of the target variance and
    // even a PERFECT fit (mse≈0) could no longer clear this test's own
    // `score > 0.5*var` bar — an internally inconsistent control. At 25.0 a
    // perfect fit yields ≈0.69·var (> 0.5·var) while the null world still
    // abstains for any positive charge (wrong-lag ΔMSE ≈ 0, minus charge).
    ScoreConfig::new(25.0 * var / n_dev as f64, 1e-9)
        .with_selection_dof(multiplicity_dof(n_candidates))
}

fn lag_expansions(
    cols: &[Vec<f64>],
    target: &[f64],
    cfg: ScoreConfig,
) -> (f64, Vec<usize>, Vec<FittedScore>) {
    let seed = seed_graph();
    let (parent_mse, _, _) = fitted_mse(&seed, cols, target, 1e-9);
    let lags = [1usize, 2, 5, 10, 20];
    let jobs: Vec<_> = lags
        .iter()
        .map(|&k| {
            (
                apply(&seed, &RewriteOp::AddLagEdge { src: 0, k }).unwrap(),
                parent_mse,
                target.to_vec(),
            )
        })
        .collect();
    let res = score_batch_fitted(&jobs, cols, &cfg);
    (parent_mse, lags.to_vec(), res)
}

#[test]
fn planted_two_lag_mechanism_recovered() {
    let (cols, target) = planted_world();
    let (_, lags, res) = lag_expansions(&cols, &target, lag_score_config(&target, 5));
    let i = (0..res.len())
        .max_by(|&a, &b| {
            res[a]
                .score
                .partial_cmp(&res[b].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    assert_eq!(lags[i], 2, "planted lag-2 mechanism must score best");
    assert!(
        res[i].score > 0.0,
        "winner must clear the gate: {}",
        res[i].score
    );
    // The fitted readout must recover the planted amplitude, not just rank it.
    assert!(
        (res[i].scale - PLANTED_AMP).abs() < 0.05 && res[i].r2 > 0.99,
        "scale {} r2 {}",
        res[i].scale,
        res[i].r2
    );
    // Winner must carry a stable content-addressed record.
    let winner = apply(&seed_graph(), &RewriteOp::AddLagEdge { src: 0, k: 2 }).unwrap();
    drop(winner);
}

// Positive control for the null test below: identical machinery, real signal.
#[test]
fn positive_control_promotes_a_planted_mechanism() {
    let (cols, target) = planted_world();
    let var = target_variance(&target);
    let (parent_mse, lags, res) = lag_expansions(&cols, &target, lag_score_config(&target, 5));
    assert!(parent_mse.is_finite());
    let i = (0..res.len())
        .max_by(|&a, &b| {
            res[a]
                .score
                .partial_cmp(&res[b].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    assert_eq!(lags[i], 2);
    assert!(
        res[i].score > 0.5 * var,
        "a real mechanism must be promoted by an order of magnitude more than \
         the charge: score {:.3e} vs var(y) {:.3e}",
        res[i].score,
        var
    );
    // No free promotion for the wrong lags: the gate is discriminative.
    for (j, s) in res.iter().enumerate() {
        if lags[j] != 2 {
            assert!(
                s.score <= 0.0,
                "lag {} spuriously promoted (score {})",
                lags[j],
                s.score
            );
        }
    }
}

#[test]
fn null_world_abstains() {
    // Target is a deterministic function of t alone: unrelated to the input.
    let (cols, _) = planted_world();
    let target: Vec<f64> = hash_noise(500, 900.0);
    // Penalty on the sampling-noise scale of a dev MSE, with the multiplicity
    // charge for the five forms scored — the regime the search runs in.
    let (parent_mse, _lags, res) = lag_expansions(&cols, &target, lag_score_config(&target, 5));
    assert!(parent_mse.is_finite());
    assert!(
        res.iter().all(|r| r.score <= 0.0),
        "null world must not promote any rewrite: {:?}",
        res.iter().map(|r| r.score).collect::<Vec<_>>()
    );
    // The penalty is load-bearing, not decorative: with no charge at all the
    // in-sample fit noise of a wrong lag is enough to score positive. That is
    // the failure mode a too-small lambda_c admits, and it is the direction the
    // audit's 1.0-per-edge setting hid by making everything fail instead.
    let (_, _, loose) = lag_expansions(&cols, &target, ScoreConfig::new(0.0, 1e-9));
    assert!(multiplicity_dof(5) > 3.0);
    assert!(
        loose.iter().any(|r| r.score > 0.0),
        "with a zero charge, fit noise should be visible — if not, this control \
         is not actually exercising the gate"
    );
}

#[test]
fn lag_expansion_reports_the_planted_scale_not_a_placeholder() {
    let (cols, target) = planted_world();
    let lag2 = apply(&seed_graph(), &RewriteOp::AddLagEdge { src: 0, k: 2 }).unwrap();
    let out = lag2.evaluate(&cols);
    let (a, b) = fit_readout(&out, &target, 1e-9);
    assert!((a - PLANTED_AMP).abs() < 1e-3, "fitted scale {}", a);
    assert!((b - 0.0).abs() < 0.05, "fitted intercept {}", b);
    let (raw_mse, skipped) = mse_finite(&out, &target);
    let (fitted, _, _) = fitted_mse(&lag2, &cols, &target, 1e-9);
    assert!(
        raw_mse > 100.0 * fitted,
        "the raw/fitted gap is the defect: raw {:.4e} vs fitted {:.4e}",
        raw_mse,
        fitted
    );
    assert_eq!(skipped, 2);
}
