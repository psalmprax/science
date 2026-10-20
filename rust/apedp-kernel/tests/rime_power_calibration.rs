//! RIME power calibration (audit-only).
//!
//! Measures, for the REAL `search` engine with the calibration the Python
//! proposer uses (`noise_scaled(y, n_dev, gamma)`, depth 2, beam 8, budget 200):
//!
//! * **recovery rate** vs signal strength (correlation `rho` between the true
//!   noise-free mechanism output and the target) and sample size `n`;
//! * **false-promotion rate** on null worlds (any proposal at all);
//! * **wrong-promotion rate** on signal worlds (promoted, but the best proposal
//!   does not reproduce the planted mechanism);
//!
//! each with a Wilson 95% interval over independent replicates.
//!
//! Recovery is *functional*, never string matching: the top proposal's graph
//! is evaluated on a FRESH independent series from the same generator and must
//! correlate (|r| >= `RECOVERY_CORR`) with the noise-free planted mechanism.
//! That is grammar-agnostic (a different but equivalent graph counts) and a
//! graph that merely fits the development noise does not.
//!
//! Run (slow, prints the table):
//!   cargo test --release --no-default-features --test rime_power_calibration \
//!       -- --ignored --nocapture
//! Env: RIME_POWER_REPS (default 40).

use apedp_kernel::rime::additive::search_additive;
use apedp_kernel::rime::search::SearchConfig;
use apedp_kernel::rime::seeds::small_forms;

/// Recovery counts when the candidate tracks the true mechanism this closely
/// on fresh data.
const RECOVERY_CORR: f64 = 0.95;
/// Rows of the fresh independent evaluation series.
const FRESH_N: usize = 4000;
const GAMMA: f64 = 1.5;

// ---- deterministic RNG (SplitMix64 + Box-Muller): no external dependency ----

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn normals(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal()).collect()
    }
}

#[derive(Clone, Copy, Debug)]
enum Mech {
    Null,
    Lag2,
    Momentum,
    State80,
    Interaction,
    TanhLag5,
    StateOffGrid87,
    TwoLagSum,
    MixedSum,
}

impl Mech {
    fn name(&self) -> &'static str {
        match self {
            Mech::Null => "null",
            Mech::Lag2 => "lag2",
            Mech::Momentum => "momentum",
            Mech::State80 => "state0.80",
            Mech::Interaction => "z*x[t-1]",
            Mech::TanhLag5 => "tanh(x[t-5])",
            Mech::StateOffGrid87 => "state0.87(offgrid)",
            Mech::TwoLagSum => "x[t-1]+x[t-2]",
            Mech::MixedSum => "x[t-1]+0.5z[t-5]",
        }
    }

    /// Noise-free planted mechanism output (NaN on warm-up rows).
    fn feature(&self, x: &[f64], z: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|t| match self {
                Mech::Null | Mech::State80 | Mech::StateOffGrid87 => f64::NAN, // handled in `feature`
                Mech::Lag2 => {
                    if t >= 2 {
                        x[t - 2]
                    } else {
                        f64::NAN
                    }
                }
                Mech::Momentum => {
                    if t >= 1 {
                        x[t] - x[t - 1]
                    } else {
                        f64::NAN
                    }
                }
                Mech::Interaction => {
                    if t >= 1 {
                        z[t] * x[t - 1]
                    } else {
                        f64::NAN
                    }
                }
                Mech::TanhLag5 => {
                    if t >= 5 {
                        x[t - 5].tanh()
                    } else {
                        f64::NAN
                    }
                }
                Mech::TwoLagSum => {
                    if t >= 2 {
                        x[t - 1] + x[t - 2]
                    } else {
                        f64::NAN
                    }
                }
                Mech::MixedSum => {
                    if t >= 5 {
                        x[t - 1] + 0.5 * z[t - 5]
                    } else {
                        f64::NAN
                    }
                }
            })
            .collect()
    }
}

/// O(n) recursive state feature `h[t] = d*h[t-1] + x[t]`, `h[-1] = 0`.
fn state_feature(x: &[f64], d: f64) -> Vec<f64> {
    let mut h = 0.0;
    x.iter()
        .map(|v| {
            h = d * h + v;
            h
        })
        .collect()
}

fn feature(m: Mech, x: &[f64], z: &[f64]) -> Vec<f64> {
    match m {
        Mech::State80 => state_feature(x, 0.80),
        Mech::StateOffGrid87 => state_feature(x, 0.87),
        _ => m.feature(x, z),
    }
}

fn finite_corr(a: &[f64], b: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .filter(|(p, q)| p.is_finite() && q.is_finite())
        .map(|(p, q)| (*p, *q))
        .collect();
    let n = pairs.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let ma = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mb = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (p, q) in pairs {
        sab += (p - ma) * (q - mb);
        saa += (p - ma) * (p - ma);
        sbb += (q - mb) * (q - mb);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        0.0
    } else {
        sab / (saa.sqrt() * sbb.sqrt())
    }
}

/// Target with correlation `rho` to the planted feature: y = rho*f_std + sqrt(1-rho^2)*eps.
fn make_target(f: &[f64], rho: f64, rng: &mut Rng) -> Vec<f64> {
    let fin: Vec<f64> = f.iter().copied().filter(|v| v.is_finite()).collect();
    let n = fin.len().max(1) as f64;
    let m = fin.iter().sum::<f64>() / n;
    let sd = (fin.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n)
        .sqrt()
        .max(1e-12);
    let s = (1.0 - rho * rho).max(0.0).sqrt();
    f.iter()
        .map(|v| {
            if v.is_finite() {
                rho * (v - m) / sd + s * rng.normal()
            } else {
                f64::NAN
            }
        })
        .collect()
}

#[derive(Default, Clone, Copy)]
struct Tally {
    reps: usize,
    promoted: usize,
    recovered: usize,
    /// Best proposal is a multi-term (`AddTerm`) model.
    multi_term: usize,
}

fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.959_963_984_540_054_f64;
    let (k, n) = (k as f64, n as f64);
    let p = k / n;
    let den = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / den;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / den;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

fn run_cell(m: Mech, rho: f64, n: usize, reps: usize, base_seed: u64, terms: usize) -> Tally {
    let mut tally = Tally::default();
    let seeds = small_forms(2);
    for r in 0..reps {
        let mut rng = Rng(base_seed ^ ((r as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03)));
        let x = rng.normals(n);
        let z = rng.normals(n);
        let f = feature(m, &x, &z);
        let y = match m {
            Mech::Null => {
                // Null: target independent of every input.
                rng.normals(n)
            }
            _ => make_target(&f, rho, &mut rng),
        };
        let n_dev = y.iter().filter(|v| v.is_finite()).count();
        let cfg = SearchConfig::noise_scaled(&y, n_dev, GAMMA).with_max_depth(2);
        let out = search_additive(&seeds, &[x, z], &y, &cfg, terms);
        tally.reps += 1;
        if let Some(best) = out.best() {
            tally.promoted += 1;
            if best.rewrite_op == "AddTerm" {
                tally.multi_term += 1;
            }
            if !matches!(m, Mech::Null) {
                // Fresh independent series from the same generator.
                let mut rng2 =
                    Rng(base_seed ^ 0xA5A5_A5A5 ^ ((r as u64 + 7).wrapping_mul(0x9E37_79B9)));
                let xf = rng2.normals(FRESH_N);
                let zf = rng2.normals(FRESH_N);
                let ff = feature(m, &xf, &zf);
                let of = best.graph.evaluate(&[xf, zf]);
                if finite_corr(&of, &ff).abs() >= RECOVERY_CORR {
                    tally.recovered += 1;
                }
            }
        }
    }
    tally
}

fn reps() -> usize {
    std::env::var("RIME_POWER_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40)
}

/// Always-on smoke: the harness itself must separate a strong signal from a
/// null, or nothing it prints means anything.
#[test]
fn harness_positive_and_null_controls() {
    let strong = run_cell(Mech::Lag2, 0.80, 1500, 6, 11, 1);
    assert_eq!(
        strong.recovered, strong.reps,
        "harness cannot recover an easy lag-2: {:?}",
        strong.recovered
    );
    let null = run_cell(Mech::Null, 0.0, 1500, 12, 13, 1);
    assert!(
        null.promoted <= 2,
        "null world promoted {} of {}",
        null.promoted,
        null.reps
    );
}

#[test]
fn wilson_interval_sanity() {
    let (lo, hi) = wilson(0, 40);
    assert!(lo == 0.0 && hi > 0.05 && hi < 0.12, "{lo} {hi}");
    let (lo, hi) = wilson(20, 40);
    assert!(lo < 0.5 && hi > 0.5);
}

#[test]
#[ignore = "audit-only: slow; run with --release --ignored --nocapture"]
fn power_curves() {
    let reps = reps();
    let mechs = [
        Mech::Null,
        Mech::Lag2,
        Mech::Momentum,
        Mech::State80,
        Mech::StateOffGrid87,
        Mech::Interaction,
        Mech::TanhLag5,
        Mech::TwoLagSum,
        Mech::MixedSum,
    ];
    let rhos = [0.02, 0.05, 0.10, 0.15, 0.30];
    let ns = [1000usize, 5000, 15000];
    println!(
        "reps per cell = {reps}; recovery = |corr(best, truth)| >= {RECOVERY_CORR} on fresh data"
    );
    println!(
        "{:<20} {:>5} {:>6}  {:>22}  {:>22}",
        "mechanism", "rho", "n", "recovered [95% CI]", "promoted [95% CI]"
    );
    for m in mechs {
        let rho_list: &[f64] = if matches!(m, Mech::Null) {
            &[0.0]
        } else {
            &rhos
        };
        for &rho in rho_list {
            for &n in &ns {
                let t = run_cell(m, rho, n, reps, 0xC0FFEE ^ (n as u64), 1);
                let (rl, rh) = wilson(t.recovered, t.reps);
                let (pl, ph) = wilson(t.promoted, t.reps);
                println!(
                    "{:<20} {:>5.2} {:>6}  {:>3}/{:<3} [{:.2},{:.2}]    {:>3}/{:<3} [{:.2},{:.2}]",
                    m.name(),
                    rho,
                    n,
                    t.recovered,
                    t.reps,
                    rl,
                    rh,
                    t.promoted,
                    t.reps,
                    pl,
                    ph
                );
            }
        }
    }
}

/// Same worlds, same seeds, same gate: single-term `search` vs
/// `search_additive(max_terms = 3)`. The additive stage must (a) recover
/// multi-term laws, (b) add NO false promotion on nulls (it only runs after the
/// frozen gate promoted), and (c) not manufacture extra terms for single-term
/// laws more often than the multiplicity charge allows.
#[test]
#[ignore = "audit-only: slow; run with --release --ignored --nocapture"]
fn additive_vs_single_term() {
    let reps = reps();
    let mechs = [
        Mech::Null,
        Mech::Lag2,
        Mech::Momentum,
        Mech::TanhLag5,
        Mech::TwoLagSum,
        Mech::MixedSum,
    ];
    let rhos = [0.10, 0.15, 0.30];
    let ns = [1000usize, 5000, 15000];
    println!("reps per cell = {reps}");
    println!(
        "{:<18} {:>5} {:>6} | {:>15} {:>9} | {:>15} {:>9} {:>11}",
        "mechanism",
        "rho",
        "n",
        "single recov",
        "promoted",
        "additive recov",
        "promoted",
        "multi-term"
    );
    for m in mechs {
        let rho_list: &[f64] = if matches!(m, Mech::Null) {
            &[0.0]
        } else {
            &rhos
        };
        for &rho in rho_list {
            for &n in &ns {
                let seed = 0xC0FFEE ^ (n as u64);
                let a = run_cell(m, rho, n, reps, seed, 1);
                let b = run_cell(m, rho, n, reps, seed, 3);
                println!(
                    "{:<18} {:>5.2} {:>6} | {:>7}/{:<3}[{:.2}] {:>5}/{:<3} | {:>7}/{:<3}[{:.2}] {:>5}/{:<3} {:>7}/{:<3}",
                    m.name(), rho, n,
                    a.recovered, a.reps, wilson(a.recovered, a.reps).0, a.promoted, a.reps,
                    b.recovered, b.reps, wilson(b.recovered, b.reps).0, b.promoted, b.reps,
                    b.multi_term, b.reps
                );
            }
        }
    }
}
