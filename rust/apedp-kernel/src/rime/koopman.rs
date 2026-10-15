//! Delay-embedded linear spectral discovery (restricted Hankel-DMD).
//!
//! For a scalar, uniformly sampled trajectory `y[0..n)`, fits the affine
//! order-2 linear recurrence
//!     z[t] = a1 * z[t-1] + a2 * z[t-2] + b,   z = y - mean(y)
//! by least squares, then reports the companion-matrix eigenvalues (real or
//! complex-conjugate pair), the implied oscillation frequency, decay rate,
//! and spectral radius.
//!
//! Scope (v1, explicit): scalar trajectories, caller-supplied uniform `dt`,
//! two-mode dynamics. General EDMD with lifted observables and multi-mode
//! spectra is deferred. This module fits a candidate linear model and reports
//! diagnostics; it never declares a scientific discovery.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ComplexVal {
    pub re: f64,
    pub im: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KoopmanConfig {
    pub ridge: f64,
    pub min_samples: usize,
}

impl Default for KoopmanConfig {
    fn default() -> KoopmanConfig {
        KoopmanConfig {
            ridge: 1e-9,
            min_samples: 12,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoopmanResult {
    pub n_samples: usize,
    pub dt: f64,
    pub mean: f64,
    pub a1: f64,
    pub a2: f64,
    pub intercept: f64,
    pub eigenvalues: [ComplexVal; 2],
    /// Oscillation frequency |arg(λ)| / dt when the spectrum is a complex pair.
    pub frequency: Option<f64>,
    /// Growth/decay rate ln|λ| / dt of the dominant mode.
    pub decay_rate: f64,
    pub spectral_radius: f64,
    pub fit_rmse: f64,
    pub fit_r2: f64,
}

/// Solve a 3x3 linear system by Gaussian elimination with partial pivoting.
fn solve_3x3(mut s: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for row in (col + 1)..3 {
            if s[row][col].abs() > s[piv][col].abs() {
                piv = row;
            }
        }
        if !s[piv][col].is_finite() || s[piv][col].abs() <= 1e-18 {
            return None;
        }
        s.swap(col, piv);
        b.swap(col, piv);
        for row in (col + 1)..3 {
            let f = s[row][col] / s[col][col];
            for j in col..3 {
                s[row][j] -= f * s[col][j];
            }
            b[row] -= f * b[col];
        }
    }
    let mut w = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut acc = b[i];
        for j in (i + 1)..3 {
            acc -= s[i][j] * w[j];
        }
        if s[i][i].abs() <= 1e-18 {
            return None;
        }
        w[i] = acc / s[i][i];
    }
    Some(w)
}

/// Fit the order-2 delay-embedded linear model and report its spectrum.
pub fn discover_linear_spectrum(
    y: &[f64],
    dt: f64,
    config: &KoopmanConfig,
) -> Result<KoopmanResult, String> {
    let n = y.len();
    let need = config.min_samples.max(4);
    if n < need {
        return Err(format!(
            "Koopman discovery requires at least {need} samples, got {n}"
        ));
    }
    if !(dt.is_finite() && dt > 0.0) {
        return Err("dt must be finite and positive".to_string());
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err("trajectory contains non-finite values".to_string());
    }

    let mean = y.iter().sum::<f64>() / (n as f64);
    let z: Vec<f64> = y.iter().map(|v| v - mean).collect();

    // Affine normal equations for z[t] = a1*z[t-1] + a2*z[t-2] + b.
    // The intercept absorbs any residual mean so a finite sample window of
    // an oscillation cannot bias the homogeneous (spectral) coefficients.
    let mut s = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for t in 2..n {
        let u = [z[t - 1], z[t - 2], 1.0];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += u[i] * u[j];
            }
            b[i] += u[i] * z[t];
        }
    }
    let r = config.ridge.max(0.0);
    s[0][0] += r;
    s[1][1] += r;
    let w = solve_3x3(s, b).ok_or_else(|| {
        "Koopman discovery: degenerate regressor matrix".to_string()
    })?;
    let (a1, a2, intercept) = (w[0], w[1], w[2]);
    if !a1.is_finite() || !a2.is_finite() || !intercept.is_finite() {
        return Err("Koopman discovery: non-finite recurrence coefficients".to_string());
    }

    // In-sample fit statistics on the centred series.
    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    for t in 2..n {
        let pred = a1 * z[t - 1] + a2 * z[t - 2] + intercept;
        let e = z[t] - pred;
        ss_res += e * e;
        ss_tot += z[t] * z[t];
    }
    let m = (n - 2) as f64;
    let fit_rmse = (ss_res / m).sqrt();
    let fit_r2 = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 1.0 };

    // Eigenvalues of the companion matrix [[a1, a2], [1, 0]].
    let disc = a1 * a1 + 4.0 * a2;
    let (l1, l2) = if disc >= 0.0 {
        let s = disc.sqrt();
        (
            ComplexVal {
                re: (a1 + s) / 2.0,
                im: 0.0,
            },
            ComplexVal {
                re: (a1 - s) / 2.0,
                im: 0.0,
            },
        )
    } else {
        let s = (-disc).sqrt() / 2.0;
        (
            ComplexVal {
                re: a1 / 2.0,
                im: s,
            },
            ComplexVal {
                re: a1 / 2.0,
                im: -s,
            },
        )
    };
    let r1 = (l1.re * l1.re + l1.im * l1.im).sqrt();
    let r2v = (l2.re * l2.re + l2.im * l2.im).sqrt();
    let radius = r1.max(r2v);
    if !(radius > 0.0) {
        return Err("Koopman discovery: degenerate spectrum".to_string());
    }
    let frequency = if disc < 0.0 {
        Some((l1.im / l1.re).atan().abs() / dt)
    } else {
        None
    };
    let decay_rate = radius.ln() / dt;

    Ok(KoopmanResult {
        n_samples: n,
        dt,
        mean,
        a1,
        a2,
        intercept,
        eigenvalues: [l1, l2],
        frequency,
        decay_rate,
        spectral_radius: radius,
        fit_rmse,
        fit_r2,
    })
}

/// Roll out the fitted recurrence `steps` ahead from two centred states.
pub fn rollout(
    a1: f64,
    a2: f64,
    intercept: f64,
    mean: f64,
    z0: f64,
    z1: f64,
    steps: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(steps);
    let (mut p0, mut p1) = (z0, z1);
    for _ in 0..steps {
        let v = a1 * p1 + a2 * p0 + intercept;
        out.push(v + mean);
        p0 = p1;
        p1 = v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r2_score(yt: &[f64], yp: &[f64]) -> f64 {
        let mean = yt.iter().sum::<f64>() / (yt.len() as f64);
        let tot: f64 = yt.iter().map(|v| (v - mean) * (v - mean)).sum();
        let res: f64 = yt
            .iter()
            .zip(yp.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        1.0 - res / tot
    }

    #[test]
    fn test_koopman_recovers_harmonic_frequency() {
        // x(t) = cos(2t): exact AR(2) with a1 = 2cos(2dt), a2 = -1.
        let dt = 0.05f64;
        let n = 200usize;
        let y: Vec<f64> = (0..n).map(|i| (2.0 * dt * (i as f64)).cos()).collect();
        let res = discover_linear_spectrum(&y, dt, &KoopmanConfig::default()).unwrap();
        let freq = res.frequency.expect("harmonic spectrum must be a complex pair");
        assert!((freq - 2.0).abs() < 0.02, "frequency: {freq}");
        assert!(
            (res.spectral_radius - 1.0).abs() < 0.02,
            "radius: {}",
            res.spectral_radius
        );
        assert!(res.decay_rate.abs() < 0.05, "decay: {}", res.decay_rate);
        // Forward rollout from the last two centred states extrapolates.
        let m = y.len();
        let z0 = y[m - 2] - res.mean;
        let z1 = y[m - 1] - res.mean;
        let sim = rollout(res.a1, res.a2, res.intercept, res.mean, z0, z1, 100);
        let truth: Vec<f64> = (0..100)
            .map(|i| (2.0 * (dt * ((m + i) as f64))).cos())
            .collect();
        assert!(r2_score(&truth, &sim) > 0.999, "rollout R2 degraded");
    }

    #[test]
    fn test_koopman_recovers_damped_decay_and_frequency() {
        // x(t) = exp(-0.5t) cos(3t).
        let dt = 0.02f64;
        let n = 400usize;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = dt * (i as f64);
                (-0.5 * t).exp() * (3.0 * t).cos()
            })
            .collect();
        let res = discover_linear_spectrum(&y, dt, &KoopmanConfig::default()).unwrap();
        let freq = res.frequency.expect("damped spectrum must be a complex pair");
        assert!((freq - 3.0).abs() < 0.05, "frequency: {freq}");
        assert!(
            (res.decay_rate + 0.5).abs() < 0.05,
            "decay: {}",
            res.decay_rate
        );
    }

    #[test]
    fn test_koopman_rejects_bad_inputs() {
        let cfg = KoopmanConfig::default();
        assert!(discover_linear_spectrum(&[1.0, 2.0, 3.0], 0.05, &cfg).is_err());
        let y = vec![1.0; 50];
        assert!(discover_linear_spectrum(&y, 0.0, &cfg).is_err());
        assert!(discover_linear_spectrum(&y, f64::NAN, &cfg).is_err());
        // Constant series: zero-variance regressors are degenerate.
        assert!(discover_linear_spectrum(&y, 0.05, &cfg).is_err());
    }
}
