//! Differential Padé & Jet Space Relation Discovery.
//!
//! Discovers governing differential equations satisfied by observed data:
//!     P(x) * y' + Q(x) * y = 0
//! or nonlinear Riccati relations:
//!     P(x) * y' + Q(x) * y + R(x) * y^2 = 0
//! or second-order equations:
//!     S(x) * y'' + P(x) * y' + Q(x) * y = 0
//!
//! Rather than forcing essential singularities, tunneling terms, or non-linear
//! differential relations into primitive expression tree tokens, this module
//! discovers the exact polynomial differential relation satisfied on the Jet
//! bundle, and validates the candidate by forward numerical integration (RK4)
//! against actual observations.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPadeConfig {
    pub max_x_degree: usize,
    pub max_diff_order: usize,
    pub smooth_window: usize,
    pub sparsity_threshold: f64,
    pub rcond_threshold: f64,
    #[serde(default)]
    pub include_riccati: bool,
}

impl Default for DiffPadeConfig {
    fn default() -> DiffPadeConfig {
        DiffPadeConfig {
            max_x_degree: 2,
            max_diff_order: 1,
            smooth_window: 5,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
            include_riccati: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPadeResult {
    pub formula: String,
    pub p_coefficients: Vec<f64>, // Polynomial P(x) multiplying y'
    pub q_coefficients: Vec<f64>, // Polynomial Q(x) multiplying y
    #[serde(default)]
    pub r_coefficients: Vec<f64>, // Polynomial R(x) multiplying y^2 (Riccati)
    #[serde(default)]
    pub s_coefficients: Vec<f64>, // Polynomial S(x) multiplying y'' (2nd order)
    pub jet_rmse: f64,
    pub integrated_trajectory_r2: f64,
    pub integrated_y: Vec<f64>,
    pub is_exact_recovery: bool,
    #[serde(default)]
    pub snapped_terms: bool, // true when rational snapping was accepted
}

/// Compute numerical first derivative y' using 4th-order central difference and 2nd-order boundary stencils.
pub fn compute_derivatives(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut dy = vec![0.0f64; n];
    if n < 4 {
        if n >= 2 {
            for i in 0..(n - 1) {
                let dx = (x[i + 1] - x[i]).max(1e-12);
                dy[i] = (y[i + 1] - y[i]) / dx;
            }
            dy[n - 1] = dy[n - 2];
        }
        return dy;
    }

    let h = (x[1] - x[0]).max(1e-12);

    // 2nd-order one-sided at edges
    dy[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h);
    dy[1] = (y[2] - y[0]) / (2.0 * h);

    // 4th-order central differences on interior
    for i in 2..(n - 2) {
        dy[i] = (-y[i + 2] + 8.0 * y[i + 1] - 8.0 * y[i - 1] + y[i - 2]) / (12.0 * h);
    }

    // 2nd-order one-sided at end
    dy[n - 2] = (y[n - 1] - y[n - 3]) / (2.0 * h);
    dy[n - 1] = (3.0 * y[n - 1] - 4.0 * y[n - 2] + y[n - 3]) / (2.0 * h);

    dy
}

/// Compute numerical second derivative y'' using 4th-order central difference and boundary stencils.
pub fn compute_second_derivatives(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut d2y = vec![0.0f64; n];
    if n < 5 {
        return d2y;
    }

    let h = (x[1] - x[0]).max(1e-12);
    let h2 = h * h;

    d2y[0] = (2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]) / h2;
    d2y[1] = (y[2] - 2.0 * y[1] + y[0]) / h2;

    for i in 2..(n - 2) {
        d2y[i] =
            (-y[i + 2] + 16.0 * y[i + 1] - 30.0 * y[i] + 16.0 * y[i - 1] - y[i - 2]) / (12.0 * h2);
    }

    d2y[n - 2] = (y[n - 1] - 2.0 * y[n - 2] + y[n - 3]) / h2;
    d2y[n - 1] = (2.0 * y[n - 1] - 5.0 * y[n - 2] + 4.0 * y[n - 3] - y[n - 4]) / h2;

    d2y
}

/// Snap a coefficient to the nearest rational p/q with q <= max_den when the
/// match is within tol * (1 + |v|); otherwise return the value unchanged.
fn snap_rational(v: f64, max_den: usize, tol: f64) -> f64 {
    if !v.is_finite() || v == 0.0 {
        return v;
    }
    let bound = tol * (1.0 + v.abs());
    let mut best = v;
    let mut best_err = bound;
    for q in 1..=max_den {
        let p = (v * q as f64).round();
        if p == 0.0 {
            continue;
        }
        let cand = p / q as f64;
        let err = (cand - v).abs();
        if err < best_err {
            best_err = err;
            best = cand;
        }
    }
    best
}

/// Dense Gaussian elimination with partial pivoting. Returns false on singular.
fn solve_dense(a: &mut [f64], b: &mut [f64], k: usize) -> bool {
    for col in 0..k {
        let mut pivot = col;
        let mut best = a[col * k + col].abs();
        for row in (col + 1)..k {
            let v = a[row * k + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-14 {
            return false;
        }
        if pivot != col {
            for j in 0..k {
                a.swap(col * k + j, pivot * k + j);
            }
            b.swap(col, pivot);
        }
        let diag = a[col * k + col];
        for row in (col + 1)..k {
            let factor = a[row * k + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for j in col..k {
                a[row * k + j] -= factor * a[col * k + j];
            }
            b[row] -= factor * b[col];
        }
    }
    for row in (0..k).rev() {
        let mut acc = b[row];
        for j in (row + 1)..k {
            acc -= a[row * k + j] * b[j];
        }
        b[row] = acc / a[row * k + row];
    }
    true
}

/// STLSQ step: minimize ||sum_j C_j c_j||^2 over the free active coefficients
/// with the maximal-magnitude active coefficient held fixed at its current
/// value. Returns (index, value) updates for the free coefficients.
fn stlsq_hold_fixed(
    cols: &[Vec<f64>],
    edge: usize,
    n: usize,
    active: &[usize],
    fixed_idx: usize,
    fixed_val: f64,
) -> Vec<(usize, f64)> {
    let free: Vec<usize> = active.iter().copied().filter(|&j| j != fixed_idx).collect();
    let g = free.len();
    if g == 0 {
        return Vec::new();
    }
    let mut normal = vec![0.0f64; g * g];
    let mut rhs = vec![0.0f64; g];
    for i in 0..g {
        for j in i..g {
            let mut dot = 0.0f64;
            for row in edge..(n - edge) {
                dot += cols[free[i]][row] * cols[free[j]][row];
            }
            normal[i * g + j] = dot;
            normal[j * g + i] = dot;
        }
        let mut dot = 0.0f64;
        for row in edge..(n - edge) {
            dot += cols[free[i]][row] * cols[fixed_idx][row];
        }
        rhs[i] = -dot * fixed_val;
    }
    let mut trace = 0.0f64;
    for i in 0..g {
        trace += normal[i * g + i];
    }
    let ridge = 1e-10 * (trace / (g as f64)).max(1e-12);
    for i in 0..g {
        normal[i * g + i] += ridge;
    }
    if !solve_dense(&mut normal, &mut rhs, g) {
        return Vec::new();
    }
    free.iter().copied().zip(rhs).collect()
}

/// SR3-lite iterative sparse refinement: alternate least squares on the active
/// set (with scale anchoring) and re-thresholding until the support stabilizes.
fn refine_sparse_system(
    cols: &[Vec<f64>],
    edge: usize,
    n: usize,
    raw_c: &mut [f64],
    threshold: f64,
) {
    let k = raw_c.len();
    for _ in 0..4 {
        let active: Vec<usize> = (0..k).filter(|&j| raw_c[j].abs() > threshold).collect();
        if active.len() < 2 {
            return;
        }
        let mut fixed_idx = active[0];
        for &j in &active {
            if raw_c[j].abs() > raw_c[fixed_idx].abs() {
                fixed_idx = j;
            }
        }
        let updates = stlsq_hold_fixed(cols, edge, n, &active, fixed_idx, raw_c[fixed_idx]);
        let mut changed = false;
        for (j, v) in updates {
            if (v - raw_c[j]).abs() > 1e-9 {
                changed = true;
                raw_c[j] = v;
            }
        }
        if !changed {
            return;
        }
        for j in 0..k {
            if raw_c[j].abs() < threshold {
                raw_c[j] = 0.0;
            }
        }
    }
}

/// RK4-integrate the discovered relation and score the trajectory against data.
fn integrate_trajectory_r2(
    x_s: &[f64],
    y_s: &[f64],
    y2_0: f64,
    q_coeffs: &[f64],
    p_coeffs: &[f64],
    r_coeffs: &[f64],
    s_coeffs: &[f64],
) -> (Vec<f64>, f64) {
    let n = x_s.len();
    let mut y_sim = vec![0.0f64; n];
    y_sim[0] = y_s[0];

    let has_second_order = s_coeffs.iter().any(|&c| c.abs() > 1e-4);

    if has_second_order {
        // 2nd-order ODE: S(x) y'' + P(x) y' + Q(x) y = 0
        let mut y1 = y_s[0];
        let mut y2 = y2_0;

        let eval_ode_2nd = |xv: f64, y1_v: f64, y2_v: f64| -> f64 {
            let mut s_val = 0.0f64;
            let mut p_val = 0.0f64;
            let mut q_val = 0.0f64;
            for (p, &c) in s_coeffs.iter().enumerate() {
                s_val += c * xv.powi(p as i32);
            }
            for (p, &c) in p_coeffs.iter().enumerate() {
                p_val += c * xv.powi(p as i32);
            }
            for (p, &c) in q_coeffs.iter().enumerate() {
                q_val += c * xv.powi(p as i32);
            }
            if s_val.abs() > 1e-10 {
                (-p_val * y2_v - q_val * y1_v) / s_val
            } else {
                0.0
            }
        };

        for i in 0..(n - 1) {
            let h = x_s[i + 1] - x_s[i];
            let x_c = x_s[i];

            let k1_y1 = y2;
            let k1_y2 = eval_ode_2nd(x_c, y1, y2);

            let k2_y1 = y2 + 0.5 * h * k1_y2;
            let k2_y2 = eval_ode_2nd(x_c + 0.5 * h, y1 + 0.5 * h * k1_y1, y2 + 0.5 * h * k1_y2);

            let k3_y1 = y2 + 0.5 * h * k2_y2;
            let k3_y2 = eval_ode_2nd(x_c + 0.5 * h, y1 + 0.5 * h * k2_y1, y2 + 0.5 * h * k2_y2);

            let k4_y1 = y2 + h * k3_y2;
            let k4_y2 = eval_ode_2nd(x_c + h, y1 + h * k3_y1, y2 + h * k3_y2);

            y1 += (h / 6.0) * (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1);
            y2 += (h / 6.0) * (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2);
            y_sim[i + 1] = y1;
        }
    } else {
        // 1st-order ODE: P(x) y' + Q(x) y + R(x) y^2 = 0
        let eval_ode = |xv: f64, yv: f64| -> f64 {
            let mut q_val = 0.0f64;
            let mut p_val = 0.0f64;
            let mut r_val = 0.0f64;
            for (p, &c) in q_coeffs.iter().enumerate() {
                q_val += c * xv.powi(p as i32);
            }
            for (p, &c) in p_coeffs.iter().enumerate() {
                p_val += c * xv.powi(p as i32);
            }
            for (p, &c) in r_coeffs.iter().enumerate() {
                r_val += c * xv.powi(p as i32);
            }
            if p_val.abs() > 1e-10 {
                (-q_val * yv - r_val * yv * yv) / p_val
            } else {
                0.0
            }
        };

        for i in 0..(n - 1) {
            let h = x_s[i + 1] - x_s[i];
            let x_curr = x_s[i];
            let y_curr = y_sim[i];

            let k1 = eval_ode(x_curr, y_curr);
            let k2 = eval_ode(x_curr + 0.5 * h, y_curr + 0.5 * h * k1);
            let k3 = eval_ode(x_curr + 0.5 * h, y_curr + 0.5 * h * k2);
            let k4 = eval_ode(x_curr + h, y_curr + h * k3);

            y_sim[i + 1] = y_curr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    }

    let mean_y = y_s.iter().sum::<f64>() / (n as f64);
    let ss_tot = y_s.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>();
    let ss_res = y_s
        .iter()
        .zip(y_sim.iter())
        .map(|(yi, si)| (yi - si) * (yi - si))
        .sum::<f64>();
    let r2 = if ss_tot > 1e-15 {
        1.0 - (ss_res / ss_tot)
    } else {
        1.0
    };
    (y_sim, r2)
}

/// Discover differential equation in Jet space (Linear, Riccati, or 2nd-order).
pub fn discover_differential_relation(
    x: &[f64],
    y: &[f64],
    config: &DiffPadeConfig,
) -> Result<DiffPadeResult, String> {
    let n = x.len();
    if n < 12 {
        return Err("Differential discovery requires at least 12 observations".to_string());
    }
    if y.len() != n {
        return Err("x and y lengths do not match".to_string());
    }

    // Sort by x
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let x_s: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
    let y_s: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

    // Compute derivatives
    let dy_s = compute_derivatives(&x_s, &y_s);
    let d2y_s = if config.max_diff_order >= 2 {
        compute_second_derivatives(&x_s, &y_s)
    } else {
        vec![0.0f64; n]
    };

    let deg = config.max_x_degree;

    let mut cols = Vec::new();
    let mut names = Vec::new();

    // Group 1: Q(x) * y
    let q_start = cols.len();
    for p in 0..=deg {
        let col: Vec<f64> = x_s
            .iter()
            .zip(y_s.iter())
            .map(|(&xv, &yv)| xv.powi(p as i32) * yv)
            .collect();
        cols.push(col);
        names.push(if p == 0 {
            "y".to_string()
        } else {
            format!("x^{p}*y")
        });
    }
    let q_count = cols.len() - q_start;

    // Group 2: R(x) * y^2 (Riccati nonlinear term)
    let r_start = cols.len();
    let r_count = if config.include_riccati {
        let max_r_deg = deg.min(1);
        for p in 0..=max_r_deg {
            let col: Vec<f64> = x_s
                .iter()
                .zip(y_s.iter())
                .map(|(&xv, &yv)| xv.powi(p as i32) * yv * yv)
                .collect();
            cols.push(col);
            names.push(if p == 0 {
                "y^2".to_string()
            } else {
                format!("x^{p}*y^2")
            });
        }
        cols.len() - r_start
    } else {
        0
    };

    // Group 3: P(x) * y'
    let p_start = cols.len();
    for p in 0..=deg {
        let col: Vec<f64> = x_s
            .iter()
            .zip(dy_s.iter())
            .map(|(&xv, &dyv)| xv.powi(p as i32) * dyv)
            .collect();
        cols.push(col);
        names.push(if p == 0 {
            "y'".to_string()
        } else {
            format!("x^{p}*y'")
        });
    }
    let p_count = cols.len() - p_start;

    // Group 4: S(x) * y'' (2nd-order Jet terms)
    let s_start = cols.len();
    let s_count = if config.max_diff_order >= 2 {
        for p in 0..=deg {
            let col: Vec<f64> = x_s
                .iter()
                .zip(d2y_s.iter())
                .map(|(&xv, &d2yv)| xv.powi(p as i32) * d2yv)
                .collect();
            cols.push(col);
            names.push(if p == 0 {
                "y''".to_string()
            } else {
                format!("x^{p}*y''")
            });
        }
        cols.len() - s_start
    } else {
        0
    };

    let k = cols.len();

    // Exclude the boundary edges where finite differences are noisy
    let edge = 2usize;
    let interior_n = n.saturating_sub(2 * edge);
    if interior_n < 6 {
        return Err("Insufficient interior points after boundary guard".to_string());
    }

    // Normalize interior columns
    let mut norms = vec![0.0f64; k];
    for j in 0..k {
        let mut sum_sq = 0.0f64;
        for i in edge..(n - edge) {
            sum_sq += cols[j][i] * cols[j][i];
        }
        norms[j] = sum_sq.sqrt().max(1e-12);
    }

    // Gram matrix on interior
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let mut dot = 0.0f64;
            for row in edge..(n - edge) {
                dot += (cols[i][row] / norms[i]) * (cols[j][row] / norms[j]);
            }
            gram[i * k + j] = dot;
            gram[j * k + i] = dot;
        }
    }

    // Jacobi Eigendecomposition
    let (evals, evecs) = crate::rime::implicit::jacobi_eigen(&gram, k, 150, 1e-15);

    // Find smallest eigenvalue
    let mut min_idx = 0usize;
    let mut min_val = f64::INFINITY;
    for (idx, &val) in evals.iter().enumerate() {
        if val.abs() < min_val {
            min_val = val.abs();
            min_idx = idx;
        }
    }

    let mut raw_c = vec![0.0f64; k];
    for j in 0..k {
        raw_c[j] = evecs[j * k + min_idx] / norms[j];
    }

    // Normalize so highest magnitude is 1.0
    let max_abs = raw_c
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1e-15);
    for v in raw_c.iter_mut() {
        *v /= max_abs;
    }

    // Sparsity thresholding
    for v in raw_c.iter_mut() {
        if v.abs() < config.sparsity_threshold {
            *v = 0.0;
        }
    }

    // Extract coefficient blocks helper (reused across refinement stages)
    let extract = |c: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let q = c[q_start..(q_start + q_count)].to_vec();
        let r = if r_count > 0 {
            c[r_start..(r_start + r_count)].to_vec()
        } else {
            Vec::new()
        };
        let p = c[p_start..(p_start + p_count)].to_vec();
        let s = if s_count > 0 {
            c[s_start..(s_start + s_count)].to_vec()
        } else {
            Vec::new()
        };
        (q, r, p, s)
    };

    // Jet residual RMSE for an arbitrary coefficient vector
    let jet_of = |c: &[f64]| -> f64 {
        let mut sum_sq = 0.0f64;
        for row in edge..(n - edge) {
            let mut r = 0.0f64;
            for j in 0..k {
                r += cols[j][row] * c[j];
            }
            sum_sq += r * r;
        }
        (sum_sq / (interior_n as f64)).sqrt()
    };

    // Baseline trajectory score before refinement
    let blocks_before = extract(&raw_c);
    let (_, traj_before) = integrate_trajectory_r2(
        &x_s,
        &y_s,
        dy_s[0],
        &blocks_before.0,
        &blocks_before.2,
        &blocks_before.1,
        &blocks_before.3,
    );
    let jet_before = jet_of(&raw_c);

    // SR3-lite sparse refinement with revert guard (trajectory and jet fidelity)
    let pre_refine = raw_c.clone();
    refine_sparse_system(&cols, edge, n, &mut raw_c, config.sparsity_threshold);
    let max_abs_r = raw_c
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1e-15);
    for v in raw_c.iter_mut() {
        *v /= max_abs_r;
    }
    let mut blocks = extract(&raw_c);
    let (_, mut traj_r2) = integrate_trajectory_r2(
        &x_s, &y_s, dy_s[0], &blocks.0, &blocks.2, &blocks.1, &blocks.3,
    );
    let jet_mid = jet_of(&raw_c);
    let mut jet_current = jet_mid;
    if traj_r2 < traj_before - 1e-3 || jet_mid > 10.0 * jet_before.max(1e-12) {
        raw_c = pre_refine;
        blocks = blocks_before;
        traj_r2 = traj_before;
        jet_current = jet_before;
    }

    // Rational snapping with trajectory-validation guard
    let mut snapped_terms = false;
    let mut snap_c = raw_c.clone();
    let mut any_snapped = false;
    for v in snap_c.iter_mut() {
        if v.abs() > 1e-6 {
            let s = snap_rational(*v, 16, 0.01);
            if (s - *v).abs() > 1e-12 {
                any_snapped = true;
                *v = s;
            }
        }
    }
    if any_snapped {
        let snap_blocks = extract(&snap_c);
        let (_, traj_s) = integrate_trajectory_r2(
            &x_s,
            &y_s,
            dy_s[0],
            &snap_blocks.0,
            &snap_blocks.2,
            &snap_blocks.1,
            &snap_blocks.3,
        );
        let jet_s = jet_of(&snap_c);
        if traj_s >= traj_r2 - 5e-3 && jet_s <= (10.0 * jet_current).max(1e-9) {
            raw_c = snap_c;
            blocks = snap_blocks;
            traj_r2 = traj_s;
            snapped_terms = true;
        }
    }

    let (q_coeffs, r_coeffs, p_coeffs, s_coeffs) = blocks;

    // Final trajectory simulation for the accepted coefficients
    let (y_sim, _) = integrate_trajectory_r2(
        &x_s, &y_s, dy_s[0], &q_coeffs, &p_coeffs, &r_coeffs, &s_coeffs,
    );

    // Compute jet residual RMSE from final coefficients
    let jet_rmse = jet_of(&raw_c);

    let has_second_order = s_coeffs.iter().any(|&c| c.abs() > 1e-4);
    let has_riccati = r_coeffs.iter().any(|&c| c.abs() > 1e-4);

    // Format formula
    let mut p_str_terms = Vec::new();
    for (p, &c) in p_coeffs.iter().enumerate() {
        if c.abs() > 1e-4 {
            p_str_terms.push(if p == 0 {
                format!("{:.3}", c)
            } else {
                format!("{:.3}*x^{p}", c)
            });
        }
    }
    let mut q_str_terms = Vec::new();
    for (p, &c) in q_coeffs.iter().enumerate() {
        if c.abs() > 1e-4 {
            q_str_terms.push(if p == 0 {
                format!("{:.3}", c)
            } else {
                format!("{:.3}*x^{p}", c)
            });
        }
    }
    let mut r_str_terms = Vec::new();
    for (p, &c) in r_coeffs.iter().enumerate() {
        if c.abs() > 1e-4 {
            r_str_terms.push(if p == 0 {
                format!("{:.3}", c)
            } else {
                format!("{:.3}*x^{p}", c)
            });
        }
    }
    let mut s_str_terms = Vec::new();
    for (p, &c) in s_coeffs.iter().enumerate() {
        if c.abs() > 1e-4 {
            s_str_terms.push(if p == 0 {
                format!("{:.3}", c)
            } else {
                format!("{:.3}*x^{p}", c)
            });
        }
    }

    let p_formatted = if p_str_terms.is_empty() {
        "0".to_string()
    } else {
        p_str_terms.join(" + ")
    };
    let q_formatted = if q_str_terms.is_empty() {
        "0".to_string()
    } else {
        q_str_terms.join(" + ")
    };
    let r_formatted = if r_str_terms.is_empty() {
        "0".to_string()
    } else {
        r_str_terms.join(" + ")
    };
    let s_formatted = if s_str_terms.is_empty() {
        "0".to_string()
    } else {
        s_str_terms.join(" + ")
    };

    let formula = if has_second_order {
        format!(
            "({}) * y'' + ({}) * y' + ({}) * y = 0",
            s_formatted, p_formatted, q_formatted
        )
    } else if has_riccati {
        format!(
            "({}) * y' + ({}) * y + ({}) * y^2 = 0",
            p_formatted, q_formatted, r_formatted
        )
    } else {
        format!("({}) * y' + ({}) * y = 0", p_formatted, q_formatted)
    };

    // Exact recovery = near-perfect fit AND structural evidence of a law:
    // either coefficients rationalize to simple exact values (snapped_terms)
    // or the fit is at machine precision. Calibrated 2026-10-16 on verified
    // cases: Landau-Zener/BCS/Planck-Riccati snap exactly; harmonic hits
    // jet ~1e-14; while fitted non-laws (cubic-linear Planck radiance) keep
    // irrational coefficients and snapped_terms=false at jet ~1e-4.
    let is_exact = traj_r2 > 0.99 && jet_rmse < 0.05 && (snapped_terms || jet_rmse < 1e-9);

    Ok(DiffPadeResult {
        formula,
        p_coefficients: p_coeffs,
        q_coefficients: q_coeffs,
        r_coefficients: r_coeffs,
        s_coefficients: s_coeffs,
        jet_rmse,
        integrated_trajectory_r2: traj_r2,
        integrated_y: y_sim,
        is_exact_recovery: is_exact,
        snapped_terms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_bcs_energy_gap_recovery() {
        // y(g) = 3.52 * exp(-1/g) => g^2 * y' - 1 * y = 0
        let n = 50;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let g = 0.2 + 0.8 * (i as f64) / ((n - 1) as f64);
            let val = 3.52 * (-1.0 / g).exp();
            x.push(g);
            y.push(val);
        }

        let cfg = DiffPadeConfig {
            max_x_degree: 2,
            max_diff_order: 1,
            smooth_window: 5,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
            include_riccati: false,
        };

        let res = discover_differential_relation(&x, &y, &cfg).unwrap();
        assert!(
            res.integrated_trajectory_r2 > 0.99,
            "Trajectory R2: {}",
            res.integrated_trajectory_r2
        );
        assert!(res.is_exact_recovery);
    }

    #[test]
    fn test_diff_landau_zener_recovery() {
        // y(v) = exp(-0.75 / v) => v^2 * y' - 0.75 * y = 0
        let n = 60;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let v = 0.2 + 3.8 * (i as f64) / ((n - 1) as f64);
            let val = (-0.75 / v).exp();
            x.push(v);
            y.push(val);
        }

        let cfg = DiffPadeConfig {
            max_x_degree: 2,
            max_diff_order: 1,
            smooth_window: 5,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
            include_riccati: false,
        };

        let res = discover_differential_relation(&x, &y, &cfg).unwrap();
        assert!(
            res.integrated_trajectory_r2 > 0.99,
            "Trajectory R2: {}",
            res.integrated_trajectory_r2
        );
        assert!(res.is_exact_recovery);
    }

    #[test]
    fn test_planck_riccati_ode_recovery() {
        // u(v) = v^3 / (e^v - 1) => v^3 * u' + (v^3 - 3*v^2) * u + u^2 = 0
        let n = 80;
        let mut v = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        for i in 0..n {
            let val_v = 0.3 + 4.7 * (i as f64) / ((n - 1) as f64);
            let val_u = (val_v.powi(3)) / (val_v.exp() - 1.0);
            v.push(val_v);
            u.push(val_u);
        }

        let cfg = DiffPadeConfig {
            max_x_degree: 3,
            max_diff_order: 1,
            smooth_window: 5,
            sparsity_threshold: 0.02,
            rcond_threshold: 1e-12,
            include_riccati: true,
        };

        let res = discover_differential_relation(&v, &u, &cfg).unwrap();
        println!("PLANCK RICCATI FORMULA: {}", res.formula);
        println!("PLANCK RICCATI JET RMSE: {}", res.jet_rmse);
        println!("PLANCK RICCATI TRAJ R2: {}", res.integrated_trajectory_r2);
        assert!(
            res.formula.contains("y^2"),
            "Formula must have Riccati y^2: {}",
            res.formula
        );
        assert!(res.jet_rmse < 0.05, "Jet RMSE: {}", res.jet_rmse);
        assert!(
            res.integrated_trajectory_r2 > 0.99,
            "Trajectory R2: {}",
            res.integrated_trajectory_r2
        );
        assert!(res.is_exact_recovery, "Riccati recovery must report exact");
    }

    #[test]
    fn test_planck_cubic_linear_fit_must_not_claim_exact() {
        // u(v) = v^3 / (e^v - 1). No exact first-order LINEAR polynomial ODE
        // exists (Hermite: e^v is transcendental). A cubic-coefficient linear
        // fit reaches traj ~0.99999 / jet ~1e-4 on this split and must NOT be
        // flagged as exact recovery.
        let n = 60;
        let mut v = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        for i in 0..n {
            let val_v = 0.1 + 5.9 * (i as f64) / ((n - 1) as f64);
            let val_u = (val_v.powi(3)) / (val_v.exp() - 1.0);
            v.push(val_v);
            u.push(val_u);
        }

        let cfg = DiffPadeConfig {
            max_x_degree: 3,
            max_diff_order: 1,
            smooth_window: 5,
            sparsity_threshold: 0.02,
            rcond_threshold: 1e-12,
            include_riccati: false,
        };

        let res = discover_differential_relation(&v, &u, &cfg).unwrap();
        assert!(!res.formula.contains("y^2"));
        assert!(
            !res.is_exact_recovery,
            "False exact claim on Planck linear fit: traj={}, jet={}",
            res.integrated_trajectory_r2, res.jet_rmse
        );
    }

    #[test]
    fn test_harmonic_oscillator_second_order_ode() {
        // y(t) = sin(2t) => y'' + 4y = 0
        let n = 70;
        let mut t = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let val_t = 0.1 + 3.0 * (i as f64) / ((n - 1) as f64);
            let val_y = (2.0 * val_t).sin();
            t.push(val_t);
            y.push(val_y);
        }

        let cfg = DiffPadeConfig {
            max_x_degree: 0, // constant coefficients
            max_diff_order: 2,
            smooth_window: 5,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
            include_riccati: false,
        };

        let res = discover_differential_relation(&t, &y, &cfg).unwrap();
        assert!(
            res.formula.contains("y''"),
            "Formula must contain y'': {}",
            res.formula
        );
        assert!(
            res.integrated_trajectory_r2 > 0.99,
            "Trajectory R2: {}",
            res.integrated_trajectory_r2
        );
        assert!(res.is_exact_recovery);
    }
}
