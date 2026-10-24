//! RIME regression tests – validated against the v0.1/v0.1.2 public API.
//!
//! The earlier versions of these tests referenced the removed
//! `SearchPacket` / JSON-value proposal surface and did not compile against
//! the current `SearchOutcome` / typed-`Proposal` API; they are rewritten as
//! true equivalents that compile and assert the same intents: null-world
//! abstention (spec §8), positive-control recovery, and budget/beam caps.
use apedp_kernel::rime::search::{search, SearchConfig};
use apedp_kernel::rime::seeds;

fn hash_noise(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let v = ((t as f64 + phase) * 12.9898).sin() * 43758.5453;
            v - v.floor() - 0.5
        })
        .collect()
}

#[test]
fn regression_search_null_world_abstains() {
    let n = 200;
    // A null world: the target is a white-noise series independent of the
    // (also white) input column. Spec §8: no candidate beats the mean
    // predictor by more than its complexity charge, so the search proposes
    // nothing.
    let y_null = hash_noise(n, 4242.0);
    let x = hash_noise(n, 900.0);
    let small = seeds::small_forms(1);
    let config = SearchConfig {
        budget: 60,
        beam: 5,
        max_depth: 1,
        lambda_c: 1.5 * variance(&y_null) / n as f64,
        ridge: 1e-9,
        seed: 42,
        static_ops: false,
        ..SearchConfig::default()
    };
    let legacy_out = search(&small, &[x], &y_null, &config);
    assert_eq!(
        legacy_out.accepted,
        0,
        "null world must abstain (spec §8), got: {:?}",
        legacy_out
            .proposals
            .iter()
            .map(|p| p.equation.clone())
            .collect::<Vec<_>>()
    );
}

fn variance(y: &[f64]) -> f64 {
    let n = y.len().max(1) as f64;
    let m = y.iter().sum::<f64>() / n;
    y.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n
}

#[test]
fn regression_search_positive_control() {
    let n = 200;
    // Planted lag-2 law with amplitude and offset on a real (hash-noise)
    // input column: the mechanism must be recovered through the fitted
    // readout, and the proposal's score must be finite.
    let x = hash_noise(n, 4242.0);
    let mut y_sig = vec![0.0; n];
    for t in 2..n {
        y_sig[t] = 3.0 * x[t - 2] + 1.0;
    }
    let small = seeds::small_forms(1);
    let config = SearchConfig {
        budget: 60,
        beam: 5,
        max_depth: 1,
        lambda_c: 1.5 * variance(&y_sig) / n as f64,
        ridge: 1e-9,
        seed: 42,
        static_ops: false,
        ..SearchConfig::default()
    };
    let sig_out = search(&small, &[x], &y_sig, &config);
    assert!(
        sig_out.accepted > 0,
        "positive control must recover a candidate"
    );
    let first = sig_out.proposals.first().expect("proposals non-empty");
    assert!(first.score.is_finite(), "proposal score should be finite");
    assert!(
        first.r2 > 0.99,
        "planted lag-2 law must be recovered: r2={} eq={}",
        first.r2,
        first.equation
    );
}

#[test]
fn regression_search_respects_budget_and_beam() {
    let n = 200;
    let x = hash_noise(n, 4242.0);
    let mut y_sig = vec![0.0; n];
    for t in 2..n {
        y_sig[t] = 3.0 * x[t - 2];
    }
    let small = seeds::small_forms(1);
    let config = SearchConfig {
        budget: 3,
        beam: 5,
        max_depth: 1,
        lambda_c: 1e-6,
        ridge: 1e-9,
        seed: 42,
        static_ops: false,
        ..SearchConfig::default()
    };
    let out = search(&small, &[x], &y_sig, &config);
    assert!(
        out.evaluated <= 3,
        "budget must cap scoring: {} scored",
        out.evaluated
    );
    assert!(out.proposals.len() <= 3, "should respect budget");
}
