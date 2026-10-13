//! Static search mode regression tests (v0.1.2 algebraic grammar extension).
//!
//! Scenario = the measured QEC capability boundary: a static threshold law
//! `y = C·p²` over a log-spaced, order-structured `p` axis. With the temporal
//! grammar enabled the search can lock onto grid order (`p[t-k]` fits adjacent
//! samples of an autocorrelated axis) — a dev-objective search cannot tell
//! grid order from physics, and the holdout correctly rejects what should
//! never have been proposed. Static mode (`static = true`) makes that failure
//! class structurally unrepresentable: time is removed from the grammar, so
//! algebraic recovery rides on the transformed input column (`p²`) the caller
//! supplies, mirroring the GA/OLS algebraic basis.
use apedp_kernel::rime::search::{search, SearchConfig, SearchOutcome};
use apedp_kernel::rime::seeds;

/// The QEC-style world: log-spaced p in [1e-4, 1e-3], y = 1000·p² + eps.
/// Returns (cols[p, p_sq], target) with deterministic tiny noise.
fn log_spaced_world(n: usize, with_sq: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
    let p: Vec<f64> = (0..n)
        .map(|i| 1e-4 * 10f64.powf(i as f64 / (n - 1) as f64))
        .collect();
    let sq: Vec<f64> = p.iter().map(|v| v * v).collect();
    let noise: Vec<f64> = (0..n)
        .map(|t| {
            let v = ((t as f64 + 31.0) * 12.9898).sin() * 43758.5453;
            (v - v.floor() - 0.5) * 1e-6
        })
        .collect();
    let y: Vec<f64> = (0..n).map(|t| 1000.0 * sq[t] + noise[t]).collect();
    let cols = if with_sq { vec![p, sq] } else { vec![p] };
    (cols, y)
}

fn static_cfg(y: &[f64], depth: usize) -> SearchConfig {
    // The proposer's calibration: gamma * var(y) / n_dev, safety factor 1.5.
    let finite: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    let n = finite.len().max(1);
    let mean = finite.iter().sum::<f64>() / n as f64;
    let var = finite.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    SearchConfig {
        lambda_c: 1.5 * var / n as f64,
        static_ops: true,
        ..SearchConfig::default()
    }
    .with_max_depth(depth)
}

#[test]
fn static_mode_recovers_quadratic_law_via_transformed_column() {
    let n = 300;
    let (cols, y) = log_spaced_world(n, true);
    let seeds = seeds::small_forms(2);
    let cfg = static_cfg(&y, 2);
    let out: SearchOutcome = search(&seeds, &cols, &y, &cfg);

    let best = out
        .best()
        .expect("static mode must promote the level law on the squared axis");
    assert!(
        best.r2 > 0.995,
        "quadratic law must be recovered exactly: r2={} eq={}",
        best.r2,
        best.equation
    );
    // The winner must read the SQUARED column (v1), not the raw axis (v0):
    // physics, not grid order.
    assert!(best.equation.contains("v1"), "winner must use the transformed column: {}", best.equation);
    // And no proposal anywhere may read time — the point of the mode.
    for p in &out.proposals {
        assert!(
            !p.equation.contains("[t-") && !p.equation.contains("state_"),
            "temporal form leaked into a static search: {}",
            p.equation
        );
        assert!(!p.graph.is_temporal(), "temporal graph in static packet: {}", p.canonical);
    }
}

#[test]
fn static_mode_without_the_transformed_column_cannot_claim_exact_recovery() {
    // Honest boundary control: with only the raw axis available, a level fit
    // is the best static grammar can do — and it must NOT reach the exact law.
    let n = 300;
    let (cols, y) = log_spaced_world(n, false);
    let seeds = seeds::small_forms(1);
    let cfg = static_cfg(&y, 2);
    let out = search(&seeds, &cols, &y, &cfg);
    if let Some(best) = out.best() {
        assert!(
            best.r2 < 0.95,
            "linear-in-p must not masquerade as the quadratic law: r2={} eq={}",
            best.r2,
            best.equation
        );
    }
    // Either way (promote a mediocre level or abstain), never claim exactness.
}

#[test]
fn temporal_mode_unchanged_by_static_flag_default() {
    // Default config keeps the frozen §7/§8 grammar: a lag law must still be
    // recovered with static_ops = false.
    let n = 200;
    let x: Vec<f64> = (0..n)
        .map(|t| {
            let v = ((t as f64) * 12.9898).sin() * 43758.5453;
            v - v.floor() - 0.5
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|t| if t >= 2 { 3.0 * x[t - 2] } else { f64::NAN })
        .collect();
    let seeds = seeds::small_forms(1);
    let cfg = SearchConfig { static_ops: false, ..SearchConfig::default() }.with_max_depth(1);
    let out = search(&seeds, &[x], &y, &cfg);
    let best = out
        .best()
        .expect("temporal default must still recover a lag-2 law");
    assert!(best.r2 > 0.99, "lag-2 recovery broken: eq={}", best.equation);
    assert!(best.equation.contains("[t-2]"), "lag not expressed: {}", best.equation);
}

