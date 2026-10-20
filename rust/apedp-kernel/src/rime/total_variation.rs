//! Variational Bounded Variation (BV) / Total Variation (TV) Denoising.
//!
//! Provides fast, exact 1D Total Variation regularization:
//!     min_u 0.5 * ||u - y||_2^2 + lambda * ||D u||_1
//! where ||D u||_1 = sum_{i=0}^{N-2} |u_{i+1} - u_i|.
//!
//! Used to separate discontinuous quantum staircases and jump phenomena
//! from thermal broadening and observation noise without assuming an explicit
//! integer `floor` operator up front.

use serde::{Deserialize, Serialize};

/// Fast 1D Total Variation denoising via Condat's direct O(N) algorithm
/// (Laurent Condat, IEEE Signal Processing Letters, 20(11):1054-1057, 2013).
pub fn tv1d_condat(y: &[f64], lambda: f64) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 || lambda <= 1e-15 {
        return y.to_vec();
    }

    let mut output = vec![0.0f64; n];
    let mut k = 0usize;
    let mut k0 = 0usize;
    let mut umin = lambda;
    let mut umax = -lambda;
    let mut vmin = y[0] - lambda;
    let mut vmax = y[0] + lambda;
    let mut kplus = 0usize;
    let mut kminus = 0usize;
    let twolambda = 2.0 * lambda;

    loop {
        while k == n - 1 {
            if umin < 0.0 {
                let val = vmin;
                while k0 <= kminus {
                    output[k0] = val;
                    k0 += 1;
                }
                if k0 >= n {
                    return output;
                }
                k = k0;
                kminus = k;
                kplus = k;
                vmin = y[k];
                vmax = y[k] + twolambda;
                umin = lambda;
                umax = -lambda;
            } else if umax > 0.0 {
                let val = vmax;
                while k0 <= kplus {
                    output[k0] = val;
                    k0 += 1;
                }
                if k0 >= n {
                    return output;
                }
                k = k0;
                kplus = k;
                kminus = k;
                vmax = y[k];
                vmin = y[k] - twolambda;
                umax = -lambda;
                umin = lambda;
            } else {
                let mean_val = vmin + umin / ((k - k0 + 1) as f64);
                while k0 <= k {
                    output[k0] = mean_val;
                    k0 += 1;
                }
                return output;
            }
        }

        let y_next = y[k + 1];
        umin += y_next - vmin;
        umax += y_next - vmax;

        if umin < -lambda {
            let val = vmin;
            while k0 <= kminus {
                output[k0] = val;
                k0 += 1;
            }
            k = k0;
            kminus = k;
            kplus = k;
            vmin = y[k];
            vmax = y[k] + twolambda;
            umin = lambda;
            umax = -lambda;
        } else if umax > lambda {
            let val = vmax;
            while k0 <= kplus {
                output[k0] = val;
                k0 += 1;
            }
            k = k0;
            kplus = k;
            kminus = k;
            vmax = y[k];
            vmin = y[k] - twolambda;
            umax = -lambda;
            umin = lambda;
        } else {
            k += 1;
            if umin >= lambda {
                vmin += (umin - lambda) / ((k - k0 + 1) as f64);
                umin = lambda;
                kminus = k;
            }
            if umax <= -lambda {
                vmax += (umax + lambda) / ((k - k0 + 1) as f64);
                umax = -lambda;
                kplus = k;
            }
        }
    }
}

/// Fallback / verification dual FISTA solver for 1D TV.
pub fn tv1d_dual_fista(y: &[f64], lambda: f64, max_iters: usize, tol: f64) -> Vec<f64> {
    let n = y.len();
    if n <= 1 || lambda <= 1e-15 {
        return y.to_vec();
    }

    let m = n - 1;
    let mut p = vec![0.0f64; m];
    let mut q = vec![0.0f64; m];
    let mut t = 1.0f64;
    let step = 0.25f64; // Lipschitz constant of D D^T is <= 4

    let mut dt_p = vec![0.0f64; n];
    let mut grad = vec![0.0f64; m];

    for _ in 0..max_iters {
        // D^T q: (D^T q)_0 = -q_0, (D^T q)_i = q_{i-1} - q_i, (D^T q)_{n-1} = q_{m-1}
        dt_p[0] = -q[0];
        for i in 1..m {
            dt_p[i] = q[i - 1] - q[i];
        }
        dt_p[n - 1] = q[m - 1];

        // u = y - D^T q
        // grad = D (D^T q - y) = - D u
        let mut max_change = 0.0f64;
        let mut p_next = vec![0.0f64; m];

        for i in 0..m {
            let u_i = y[i] - dt_p[i];
            let u_ip1 = y[i + 1] - dt_p[i + 1];
            grad[i] = u_i - u_ip1;
            let val = q[i] - step * grad[i];
            let clipped = val.clamp(-lambda, lambda);
            let change = (clipped - p[i]).abs();
            if change > max_change {
                max_change = change;
            }
            p_next[i] = clipped;
        }

        let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let factor = (t - 1.0) / t_next;
        for i in 0..m {
            q[i] = p_next[i] + factor * (p_next[i] - p[i]);
        }
        p = p_next;
        t = t_next;

        if max_change < tol {
            break;
        }
    }

    // u = y - D^T p
    dt_p[0] = -p[0];
    for i in 1..m {
        dt_p[i] = p[i - 1] - p[i];
    }
    dt_p[n - 1] = p[m - 1];

    y.iter()
        .zip(dt_p.iter())
        .map(|(yi, dti)| yi - dti)
        .collect()
}

/// A detected jump event in 1D data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpEvent {
    pub index: usize,
    pub x_location: f64,
    pub step_height: f64,
}

/// Complete report of TV denoising, plateau detection, and quantization analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TvDenoiseReport {
    pub n_samples: usize,
    pub lambda_applied: f64,
    pub denoised_y: Vec<f64>,
    pub jumps: Vec<JumpEvent>,
    pub discrete_plateaus: Vec<f64>,
    pub flat_fraction: f64,
    pub is_quantized_staircase: bool,
    pub estimated_quantum_step: f64,
    pub reconstruction_r2: f64,
}

/// Denoise data, extract jumps and detect if it forms a quantized staircase.
pub fn tv_denoise_and_analyze(
    x: &[f64],
    y: &[f64],
    lambda: f64,
    jump_threshold: f64,
) -> Result<TvDenoiseReport, String> {
    let n = y.len();
    if n < 4 {
        return Err("TV denoising requires at least 4 samples".to_string());
    }
    if x.len() != n {
        return Err("x and y must have the same length".to_string());
    }

    let u = tv1d_condat(y, lambda);

    // Compute jumps: |u_{i+1} - u_i| >= jump_threshold
    let mut jumps = Vec::new();
    let mut flat_count = 0usize;

    for i in 0..(n - 1) {
        let diff = u[i + 1] - u[i];
        let abs_diff = diff.abs();
        if abs_diff >= jump_threshold {
            let x_loc = 0.5 * (x[i] + x[i + 1]);
            jumps.push(JumpEvent {
                index: i,
                x_location: x_loc,
                step_height: diff,
            });
        } else {
            flat_count += 1;
        }
    }

    let flat_fraction = flat_count as f64 / ((n - 1) as f64);

    // Cluster discrete plateau values (rounded to 3 decimals)
    let mut rounded_levels: Vec<f64> = u.iter().map(|&v| (v * 1000.0).round() / 1000.0).collect();
    rounded_levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    rounded_levels.dedup();

    // Determine quantum step height from median jump magnitude
    let mut step_magnitudes: Vec<f64> = jumps.iter().map(|j| j.step_height.abs()).collect();
    step_magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let estimated_quantum_step = if !step_magnitudes.is_empty() {
        step_magnitudes[step_magnitudes.len() / 2]
    } else {
        0.0
    };

    let is_quantized = flat_fraction >= 0.35 && !jumps.is_empty() && rounded_levels.len() <= 20;

    // Reconstruction R^2 against original y
    let mean_y = y.iter().sum::<f64>() / (n as f64);
    let ss_tot = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>();
    let ss_res = y
        .iter()
        .zip(u.iter())
        .map(|(yi, ui)| (yi - ui) * (yi - ui))
        .sum::<f64>();
    let r2 = if ss_tot > 1e-15 {
        1.0 - (ss_res / ss_tot)
    } else {
        1.0
    };

    Ok(TvDenoiseReport {
        n_samples: n,
        lambda_applied: lambda,
        denoised_y: u,
        jumps,
        discrete_plateaus: rounded_levels,
        flat_fraction,
        is_quantized_staircase: is_quantized,
        estimated_quantum_step,
        reconstruction_r2: r2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tv_constant_signal() {
        let y = vec![5.0; 20];
        let u = tv1d_condat(&y, 1.0);
        for val in u {
            assert!((val - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tv_clean_staircase_preserves_steps() {
        // Step signal: 0, 0, 0, 1, 1, 1, 2, 2, 2
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let u = tv1d_condat(&y, 0.1);
        for (yi, ui) in y.iter().zip(u.iter()) {
            assert!((yi - ui).abs() < 0.05, "yi: {}, ui: {}", yi, ui);
        }
        // Jump transitions remain sharp
        assert!((u[3] - u[2] - 1.0).abs() < 0.1);
        assert!((u[6] - u[5] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_tv_noisy_staircase_recovers_plateaus() {
        // True steps at 0, 1, 2, with 5% noise added
        let true_y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let noise = [
            0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01, 0.01, -0.02, 0.02, -0.01,
        ];
        let y: Vec<f64> = true_y
            .iter()
            .zip(noise.iter())
            .map(|(a, b)| a + b)
            .collect();
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();

        let report = tv_denoise_and_analyze(&x, &y, 0.2, 0.5).unwrap();
        assert!(report.is_quantized_staircase);
        assert_eq!(report.jumps.len(), 2);
        assert!((report.jumps[0].step_height - 1.0).abs() < 0.1);
        assert!((report.jumps[1].step_height - 1.0).abs() < 0.1);
    }
}
