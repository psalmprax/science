//! Kernel phase-decomposition profile (v0.1.3 prerequisite; audit-only).
//!
//! Decomposes per-candidate time in the Rust search into:
//!   (1) graph evaluation (`Graph::evaluate`, series-at-a-time arena walk),
//!   (2) readout fit (`fit_readout` + residual statistics),
//!   (3) search bookkeeping = the REMAINDER of full-search wall time per
//!       candidate after (1)+(2) — an estimate labeled as such, because the
//!       search's exact candidate mix (seed batch + beam children) cannot be
//!       replayed bit-for-bit from outside the engine.
//!
//! Method: a planted lag-2 world at each series length; 7 repetitions per
//! measurement, MEDIAN reported (the v0.1.2 arithmetic-audit discipline).
//! Phases (1)/(2) are measured on the 60-form seed catalogue over the same
//! data, single-threaded, so the numbers are per-candidate phase costs, NOT
//! the rayon-batch wall time the production path enjoys.
//!
//! Writes `benchmarks/kernel_profile_results.json` next to the repo root and
//! prints the table for CLI capture.
use std::time::Instant;

use apedp_kernel::rime::eval::{fit_readout, residual_stats};
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn profile_size(n: usize) -> serde_json::Value {
    let x = hash_noise(n, 0.0);
    let mut y = vec![f64::NAN; n];
    for t in 2..n {
        y[t] = 3.0 * x[t - 2] + 1.0;
    }
    let cols = vec![x, hash_noise(n, 7.0)];
    let seeds = seeds::small_forms(1);
    let mut cfg =
        SearchConfig::default().with_max_depth(2).with_beam(8).with_budget(200);
    {
        // The proposer's calibration: 1.5 * var(y) / n_dev.
        let ndev = (n - 2) as f64;
        let m = y.iter().filter(|v| v.is_finite()).sum::<f64>() / ndev;
        let var = y
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| (v - m) * (v - m))
            .sum::<f64>()
            / ndev;
        cfg.lambda_c = 1.5 * var / ndev;
    }

    // (A) Full search: 7 reps, median wall + candidate count from the last run.
    let mut walls = Vec::new();
    let mut evaluated = 0usize;
    for _ in 0..7 {
        let t0 = Instant::now();
        let out = search(&seeds, &cols, &y, &cfg);
        walls.push(t0.elapsed().as_secs_f64() * 1e6);
        evaluated = out.evaluated;
    }
    let wall_us = median(walls);
    let per_candidate = wall_us / evaluated.max(1) as f64;

    // (B) Graph-evaluation phase: single-threaded over the seed catalogue.
    let mut eval_walls = Vec::new();
    for _ in 0..7 {
        let t0 = Instant::now();
        for g in &seeds {
            let _ = g.evaluate(&cols);
        }
        eval_walls.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let us_graph_eval = median(eval_walls) / seeds.len() as f64;

    // (C) Readout phase on PRECOMPUTED outputs: fit + residual stats, 7 reps.
    // black_box defeats dead-code elimination of the pure arithmetic (the
    // first cut of this harness measured a hard 0.0 µs — an artifact, not a
    // speed). A running checksum is fused at each step and released once.
    let precomputed: Vec<Vec<f64>> = seeds.iter().map(|g| g.evaluate(&cols)).collect();
    let mut fit_walls = Vec::new();
    let mut sink = 0.0f64;
    for _ in 0..7 {
        let t0 = Instant::now();
        for out in &precomputed {
            let (a, b) = fit_readout(out, &y, cfg.ridge);
            let (mse, _tss, _ne, _sk) = residual_stats(out, &y, a, b);
            sink += a * 0.0 + b * 0.0 + mse * 0.0 + 1.0;
        }
        fit_walls.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    std::hint::black_box(&mut sink);
    let us_fit = median(fit_walls) / seeds.len() as f64;

    let remainder = (per_candidate - us_graph_eval - us_fit).max(0.0);
    let round = |v: f64| (v * 10.0).round() / 10.0;
    serde_json::json!({
        "n": n,
        "evaluated": evaluated,
        "us_per_candidate": round(per_candidate),
        "us_graph_eval_phase": round(us_graph_eval),
        "us_readout_fit_phase": round(us_fit),
        "us_search_bookkeeping_remainder": round(remainder),
        "remainder_pct": round(100.0 * remainder / per_candidate),
    })
}

#[test]
fn writes_kernel_phase_profile() {
    let rows: Vec<serde_json::Value> =
        [150usize, 600, 1200, 3000, 12000].iter().map(|&n| profile_size(n)).collect();
    let method_note = "phase decomposition, median of 7 reps per measurement; phases "
        .to_string()
        + "measured SINGLE-THREADED on the seed catalogue against a planted "
        + "lag-2 world, while the production search wall is rayon-parallel — at "
        + "large n the phase sum can exceed us_per_candidate and the remainder "
        + "clamps to 0 (0 or a negative remainder means NO usable bookkeeping "
        + "estimate at that size). remainder_pct>0 rows: bookkeeping is the "
        + "dominant cost at small n (operational targeting for v0.2: enumeration "
        + "and readout batches, not arithmetic);";
    let artifact = serde_json::json!({
        "method": method_note,
        "kernel": "apedp-kernel rime, release profile",
        "rows": rows,
    });
    let path = format!(
        "{}/../../benchmarks/kernel_profile_results.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = serde_json::to_string_pretty(&artifact).unwrap();
    std::fs::write(&path, &text).unwrap_or_else(|e| panic!("write artifact {path}: {e}"));
    println!("kernel profile artifact: {path}");
    for r in &rows {
        println!("{}", serde_json::to_string(r).unwrap());
    }
}
