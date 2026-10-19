//! Sparse Implicit Null-Space Discovery.
//!
//! Discovers algebraic and physical relations of the form F(x, y) = 0
//! by finding sparse null-space vectors of a generalized feature tensor:
//!     Theta(x, y) * c ~ 0
//!
//! Completely symmetric with respect to LHS and RHS, enabling exact recovery
//! of rational poles, implicit geometric manifolds, and conservation surfaces
//! without declaring an arbitrary target variable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitConfig {
    pub max_degree: usize,
    pub include_cross_terms: bool,
    pub include_rational_products: bool,
    pub sparsity_threshold: f64,
    pub rcond_threshold: f64,
}

impl Default for ImplicitConfig {
    fn default() -> ImplicitConfig {
        ImplicitConfig {
            max_degree: 3,
            include_cross_terms: true,
            include_rational_products: true,
            sparsity_threshold: 1e-3,
            rcond_threshold: 1e-12,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitResult {
    pub feature_names: Vec<String>,
    pub coefficients: Vec<f64>,
    pub formula: String,
    pub algebraic_rmse: f64,
    pub relative_null_error: f64,
    pub condition_number: f64,
    pub num_active_terms: usize,
    pub is_exact_recovery: bool,
}

/// SVD / Jacobi Eigendecomposition for symmetric K x K matrix.
pub fn jacobi_eigen(a: &[f64], k: usize, max_rotations: usize, tol: f64) -> (Vec<f64>, Vec<f64>) {
    let mut mat = a.to_vec();
    let mut v = vec![0.0f64; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }

    for _ in 0..max_rotations {
        let mut max_off = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;

        for i in 0..k {
            for j in (i + 1)..k {
                let off = mat[i * k + j].abs();
                if off > max_off {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < tol {
            break;
        }

        let app = mat[p * k + p];
        let aqq = mat[q * k + q];
        let apq = mat[p * k + q];

        let theta = 0.5 * (aqq - app) / apq;
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            -1.0 / (-theta + (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let tau = s / (1.0 + c);

        mat[p * k + p] = app - t * apq;
        mat[q * k + q] = aqq + t * apq;
        mat[p * k + q] = 0.0;
        mat[q * k + p] = 0.0;

        for i in 0..k {
            if i != p && i != q {
                let a_ip = mat[i * k + p];
                let a_iq = mat[i * k + q];
                mat[i * k + p] = a_ip - s * (a_iq + tau * a_ip);
                mat[p * k + i] = mat[i * k + p];
                mat[i * k + q] = a_iq + s * (a_ip - tau * a_iq);
                mat[q * k + i] = mat[i * k + q];
            }
        }

        for i in 0..k {
            let v_ip = v[i * k + p];
            let v_iq = v[i * k + q];
            v[i * k + p] = c * v_ip - s * v_iq;
            v[i * k + q] = s * v_ip + c * v_iq;
        }
    }

    let mut eigenvalues = vec![0.0f64; k];
    for i in 0..k {
        eigenvalues[i] = mat[i * k + i];
    }

    (eigenvalues, v)
}

/// Construct feature matrix Theta(x, y) up to a specified maximum degree.
fn build_features_for_degree(
    x: &[f64],
    y: &[f64],
    deg: usize,
    include_cross_terms: bool,
    include_rational_products: bool,
) -> (Vec<Vec<f64>>, Vec<String>) {
    let n = x.len();
    let mut cols = Vec::new();
    let mut names = Vec::new();

    // Constant 1
    cols.push(vec![1.0f64; n]);
    names.push("1".to_string());

    // Single variable powers x^p, y^q
    for p in 1..=deg {
        let col_x: Vec<f64> = x.iter().map(|&v| v.powi(p as i32)).collect();
        cols.push(col_x);
        names.push(if p == 1 { "x".to_string() } else { format!("x^{p}") });

        let col_y: Vec<f64> = y.iter().map(|&v| v.powi(p as i32)).collect();
        cols.push(col_y);
        names.push(if p == 1 { "y".to_string() } else { format!("y^{p}") });
    }

    // Cross terms x^p * y^q with total degree p + q <= deg
    if include_cross_terms {
        for p in 1..=deg {
            for q in 1..=deg {
                if p + q <= deg {
                    let col_xy: Vec<f64> = x.iter().zip(y.iter()).map(|(&xv, &yv)| {
                        xv.powi(p as i32) * yv.powi(q as i32)
                    }).collect();
                    cols.push(col_xy);
                    let str_x = if p == 1 { "x".to_string() } else { format!("x^{p}") };
                    let str_y = if q == 1 { "y".to_string() } else { format!("y^{q}") };
                    names.push(format!("{str_x}*{str_y}"));
                }
            }
        }
    }

    // Rational resonance cross-terms: x^2 * y (total degree 3, essential for Padé saturation & detuned Rabi)
    if include_rational_products && deg >= 2 && !names.contains(&"x^2*y".to_string()) {
        let col_x2y: Vec<f64> = x.iter().zip(y.iter()).map(|(&xv, &yv)| xv * xv * yv).collect();
        cols.push(col_x2y);
        names.push("x^2*y".to_string());
    }

    (cols, names)
}

fn solve_implicit_for_features(
    raw_cols: &[Vec<f64>],
    names: &[String],
    n: usize,
    deg: usize,
    config: &ImplicitConfig,
) -> Option<ImplicitResult> {
    let k = raw_cols.len();
    if k == 0 {
        return None;
    }

    // Column norms for scaling
    let mut norms = vec![0.0f64; k];
    for j in 0..k {
        let sum_sq: f64 = raw_cols[j].iter().map(|v| v * v).sum();
        norms[j] = sum_sq.sqrt().max(1e-12);
    }

    // Gram matrix M = (Theta_norm)^T * Theta_norm
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let mut dot = 0.0f64;
            for row in 0..n {
                dot += (raw_cols[i][row] / norms[i]) * (raw_cols[j][row] / norms[j]);
            }
            gram[i * k + j] = dot;
            gram[j * k + i] = dot;
        }
    }

    // Jacobi Eigendecomposition of Gram matrix with adequate sweep depth
    let max_rotations = (20 * k * (k - 1) / 2).max(600);
    let (evals, evecs) = jacobi_eigen(&gram, k, max_rotations, 1e-15);

    // Find smallest non-negative eigenvalue and its eigenvector
    let mut min_idx = 0usize;
    let mut min_val = f64::INFINITY;
    let mut max_val = 0.0f64;

    for (idx, &val) in evals.iter().enumerate() {
        let abs_val = val.abs();
        if abs_val < min_val {
            min_val = abs_val;
            min_idx = idx;
        }
        if abs_val > max_val {
            max_val = abs_val;
        }
    }

    let cond = if min_val > 1e-15 { max_val / min_val } else { f64::INFINITY };

    // Examine candidates in the numerical null space to choose the candidate
    // maximizing sparsity while maintaining minimal algebraic RMSE.
    let null_cutoff = (min_val * 50.0).max(config.rcond_threshold * max_val).max(1e-7);

    let mut best_c = vec![0.0f64; k];
    let mut best_sparse_c = vec![0.0f64; k];
    let mut best_score = f64::INFINITY;
    let mut best_rmse = f64::INFINITY;

    for (idx, &val) in evals.iter().enumerate() {
        if val.abs() > null_cutoff && idx != min_idx {
            continue;
        }
        let mut cand_c = vec![0.0f64; k];
        for j in 0..k {
            cand_c[j] = evecs[j * k + idx] / norms[j];
        }
        let max_c = cand_c.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-15);
        for v in cand_c.iter_mut() {
            *v /= max_c;
        }
        let mut cand_sparse = cand_c.clone();
        let mut active_count = 0usize;
        for v in cand_sparse.iter_mut() {
            if v.abs() < config.sparsity_threshold {
                *v = 0.0;
            } else {
                active_count += 1;
            }
        }
        if active_count == 0 {
            continue;
        }

        // Compute residual error: ||Theta * c|| / sqrt(N)
        let mut sum_sq_err = 0.0f64;
        for row in 0..n {
            let mut r = 0.0f64;
            for j in 0..k {
                r += raw_cols[j][row] * cand_sparse[j];
            }
            sum_sq_err += r * r;
        }
        let cand_rmse = (sum_sq_err / (n as f64)).sqrt();
        let score = cand_rmse + 0.005 * (active_count as f64) + 0.02 * (deg as f64);
        if score < best_score {
            best_score = score;
            best_rmse = cand_rmse;
            best_c = cand_c;
            best_sparse_c = cand_sparse;
        }
    }

    if best_score == f64::INFINITY {
        for j in 0..k {
            best_c[j] = evecs[j * k + min_idx] / norms[j];
        }
        let max_c = best_c.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-15);
        for v in best_c.iter_mut() {
            *v /= max_c;
        }
        best_sparse_c = best_c.clone();
        for v in best_sparse_c.iter_mut() {
            if v.abs() < config.sparsity_threshold {
                *v = 0.0;
            }
        }
        let mut sum_sq_err = 0.0f64;
        for row in 0..n {
            let mut r = 0.0f64;
            for j in 0..k {
                r += raw_cols[j][row] * best_sparse_c[j];
            }
            sum_sq_err += r * r;
        }
        best_rmse = (sum_sq_err / (n as f64)).sqrt();
    }

    let sparse_c = best_sparse_c;
    let rmse = best_rmse;
    let rel_null_err = (min_val / (n as f64).max(1.0)).sqrt();

    // Scale-aware exactness gate (external-review item on F17): an absolute
    // `rmse < 1e-5` is meaningless when the relation's terms live on a large
    // scale (e.g. QEC-style slopes ~1e5) and over-permissive for badly
    // conditioned small-scale data. Exactness is therefore measured RELATIVE
    // to the RMS magnitude of the residual's own terms (the design columns),
    // with the same 1e-5 relative tolerance the gate previously held in
    // absolute units. The all-zero pruned vector still fails on active_count,
    // independent of scale.

    // Construct human-readable formula
    let mut terms = Vec::new();
    let mut active_count = 0usize;
    for j in 0..k {
        let val = sparse_c[j];
        if val.abs() > 1e-6 {
            active_count += 1;
            let term_str = if names[j] == "1" {
                format!("{:.4}", val)
            } else if (val - 1.0).abs() < 1e-4 {
                names[j].clone()
            } else if (val + 1.0).abs() < 1e-4 {
                format!("-{}", names[j])
            } else {
                format!("{:.4}*{}", val, names[j])
            };
            terms.push(term_str);
        }
    }

    let formula = if terms.is_empty() {
        "0 = 0".to_string()
    } else {
        format!("{} = 0", terms.join(" + ").replace("+ -", "- "))
    };

    // A pruned-away candidate (active_count == 0) is the tautology "0 = 0":
    // its algebraic residual is identically 0.0, so an rmse-only check would
    // certify the empty relation as an exact physical recovery and let it
    // bypass every downstream sanity filter. Exactness also demands that the
    // discovered relation actually has content (>= 2 active terms; a single
    // term c*phi = 0 is a degenerate if non-empty edge case and is allowed).
    let mut sum_sq_scale = 0.0f64;
    let mut n_scale_entries = 0usize;
    for col in raw_cols.iter() {
        for &v in col.iter() {
            sum_sq_scale += v * v;
            n_scale_entries += 1;
        }
    }
    let term_scale = (sum_sq_scale / (n_scale_entries.max(1) as f64)).sqrt();
    let exact_rmse_ceiling = 1e-5 * term_scale.max(1e-12);
    let is_exact = rmse < exact_rmse_ceiling && active_count >= 2;

    Some(ImplicitResult {
        feature_names: names.to_vec(),
        coefficients: sparse_c,
        formula,
        algebraic_rmse: rmse,
        relative_null_error: rel_null_err,
        condition_number: cond,
        num_active_terms: active_count,
        is_exact_recovery: is_exact,
    })
}

/// Discover implicit algebraic relations F(x, y) = 0.
/// Searches incrementally across degrees 1..=max_degree to ensure minimal relations
/// (e.g. x^2 + y^2 - 1 = 0) are prioritized over higher-order algebraic multiples.
pub fn discover_implicit_relation(
    x: &[f64],
    y: &[f64],
    config: &ImplicitConfig,
) -> Result<ImplicitResult, String> {
    let n = x.len();
    if n < 10 {
        return Err("Implicit discovery requires at least 10 observations".to_string());
    }
    if y.len() != n {
        return Err("x and y lengths do not match".to_string());
    }

    let max_deg = config.max_degree.max(1);
    let mut best_result: Option<ImplicitResult> = None;
    let mut min_score = f64::INFINITY;

    for deg in 1..=max_deg {
        let (raw_cols, names) = build_features_for_degree(
            x,
            y,
            deg,
            config.include_cross_terms,
            config.include_rational_products,
        );

        if let Some(res) = solve_implicit_for_features(&raw_cols, &names, n, deg, config) {
            // If an exact minimal relation is recovered at this degree, stop search immediately.
            if res.is_exact_recovery && res.algebraic_rmse < 1e-6 && res.num_active_terms > 1 {
                return Ok(res);
            }

            let score = res.algebraic_rmse + 0.005 * (res.num_active_terms as f64) + 0.02 * (deg as f64);
            if score < min_score {
                min_score = score;
                best_result = Some(res);
            }
        }
    }

    best_result.ok_or_else(|| "Failed to fit implicit relation".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_circle_discovery() {
        // x^2 + y^2 = 1 => 1*x^2 + 1*y^2 - 1 = 0
        let n = 50;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            x.push(theta.cos());
            y.push(theta.sin());
        }

        let cfg = ImplicitConfig {
            max_degree: 2,
            include_cross_terms: true,
            include_rational_products: false,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
        };

        let res = discover_implicit_relation(&x, &y, &cfg).unwrap();
        assert!(res.algebraic_rmse < 1e-8, "RMSE: {}", res.algebraic_rmse);
        assert!(res.is_exact_recovery);
    }

    #[test]
    fn test_implicit_circle_discovery_max_degree_3() {
        // Must discover minimal relation x^2 + y^2 - 1 = 0, NOT x*y*(x^2 + y^2 - 1) = 0
        let n = 50;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            x.push(theta.cos());
            y.push(theta.sin());
        }

        let cfg = ImplicitConfig {
            max_degree: 3,
            include_cross_terms: true,
            include_rational_products: false,
            sparsity_threshold: 0.05,
            rcond_threshold: 1e-12,
        };

        let res = discover_implicit_relation(&x, &y, &cfg).unwrap();
        assert!(res.algebraic_rmse < 1e-8, "RMSE: {}", res.algebraic_rmse);
        assert!(res.is_exact_recovery);
        assert_eq!(res.num_active_terms, 3);
        // Formula should not contain x*y or degree 3/4 terms
        assert!(!res.formula.contains("x*y"), "Formula must be minimal, got: {}", res.formula);
    }

    #[test]
    fn test_implicit_lorentzian_pade_resonance() {
        // Detuned Rabi / Padé: y = 25 / (25 + x^2) => 25*y + x^2*y - 25 = 0
        let n = 60;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xv = -10.0 + 20.0 * (i as f64) / ((n - 1) as f64);
            let yv = 25.0 / (25.0 + xv * xv);
            x.push(xv);
            y.push(yv);
        }

        let cfg = ImplicitConfig {
            max_degree: 2,
            include_cross_terms: true,
            include_rational_products: true,
            sparsity_threshold: 0.02,
            rcond_threshold: 1e-12,
        };

        let res = discover_implicit_relation(&x, &y, &cfg).unwrap();
        assert!(res.algebraic_rmse < 1e-8, "RMSE: {}", res.algebraic_rmse);
        assert!(res.is_exact_recovery);
    }
}
