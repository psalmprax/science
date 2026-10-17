//! Numerical kernels: metrics, conditioning, and linear least squares.
//!
//! Every routine here mirrors the arithmetic of its counterpart in
//! `apedp/rust_bridge.py::_fallback_*` closely enough that the differential
//! suite's tolerances measure *implementation* differences rather than
//! accumulated rounding noise. Where the Python side has an explicit branch
//! (the `var_y > 1e-15` guard, the `sigma_min > 1e-15` guard), the same
//! branch and the same constant appear here; drifting from them would make a
//! parity failure ambiguous between "the implementations disagree" and "one of
//! them was edited".

/// Loss metrics plus the non-finite flag.
///
/// Mirrors `RustKernelBridge._fallback_metrics`, including its refusal to
/// report numbers derived from a non-finite residual: a caller that reads
/// `mse` as a quality verdict must never see a number computed from NaN.
pub fn metrics(y_true: &[f64], y_pred: &[f64]) -> (f64, f64, f64, f64, bool) {
    let nan_metrics = || (f64::NAN, f64::NAN, f64::NAN, f64::NAN, true);

    if y_true.is_empty() || y_true.len() != y_pred.len() {
        return nan_metrics();
    }

    let mut sum_sq = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    for i in 0..y_true.len() {
        let r = y_true[i] - y_pred[i];
        if !r.is_finite() {
            return nan_metrics();
        }
        sum_sq += r * r;
        let a = r.abs();
        sum_abs += a;
        if a > max_abs {
            max_abs = a;
        }
    }

    let n = y_true.len() as f64;
    let mse = sum_sq / n;
    let mae = sum_abs / n;

    let mean = y_true.iter().sum::<f64>() / n;
    let var_y = y_true.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;

    let r2 = if var_y > 1e-15 {
        1.0 - (mse / var_y)
    } else if mse < 1e-15 {
        1.0
    } else {
        0.0
    };

    (mse, mae, r2, max_abs, false)
}

/// Conditioning report for the columns of a design matrix.
///
/// `cols` are the columns; the matrix is `n_rows x n_cols`. Mirrors
/// `RustKernelBridge._fallback_conditioning`.
pub fn conditioning<V: crate::data::DataCol>(
    cols: &[V],
    rcond_threshold: f64,
) -> (f64, f64, f64, usize, bool, Vec<f64>) {
    let degenerate = || (f64::INFINITY, 0.0, 0.0, 0usize, false, Vec::new());

    if cols.is_empty() {
        return degenerate();
    }
    let n_rows = cols[0].row_len();
    if cols.iter().any(|c| c.row_len() != n_rows) {
        // Ragged columns have no matrix. numpy would raise here, and the
        // Python side catches it into the degenerate report.
        return degenerate();
    }

    let matrix = to_matrix(cols, n_rows, cols.len());
    if matrix.iter().any(|v| !v.is_finite()) {
        // SVD non-convergence *is* the degeneracy signal; a non-finite design
        // matrix (NaN Jacobian columns from sqrt of a negative, say) must
        // report ill-conditioned rather than raise.
        return degenerate();
    }

    let mut s = singular_values(&matrix, n_rows, cols.len());
    if s.is_empty() {
        return degenerate();
    }
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let sigma_max = s[0];
    let sigma_min = s[s.len() - 1];
    let cond = if sigma_min > 1e-15 {
        sigma_max / sigma_min
    } else {
        f64::INFINITY
    };
    let rank = s.iter().filter(|v| **v > sigma_max * rcond_threshold).count();
    let is_well = cond < (1.0 / rcond_threshold.max(1e-15));

    (cond, sigma_max, sigma_min, rank, is_well, s)
}

/// Least squares for the linear-in-parameters stage.
///
/// Mirrors `RustKernelBridge._fallback_linear_solver`: with `lambda_reg` above
/// `1e-15` it forms the regularised normal equations and solves them, and
/// reports the same MSE/MAE/R^2 plus the conditioning of the original basis.
pub fn linear_least_squares<V: crate::data::DataCol>(
    cols: &[V],
    target: &[f64],
    lambda_reg: f64,
) -> Result<(Vec<f64>, f64, f64, f64, f64, bool), String> {
    if cols.is_empty() {
        return Err("linear solve needs at least one basis column".to_string());
    }
    let n = cols[0].row_len();
    if n == 0 {
        return Err("linear solve needs at least one row".to_string());
    }
    if target.len() != n {
        return Err(format!(
            "target has length {} but the basis has {} rows",
            target.len(),
            n
        ));
    }
    for (i, c) in cols.iter().enumerate() {
        if c.row_len() != n {
            return Err(format!(
                "basis column {i} has length {} but column 0 has length {n}",
                c.row_len()
            ));
        }
    }
    let k = cols.len();
    let matrix = to_matrix(cols, n, k);

    let coeffs = if lambda_reg > 1e-15 {
        // (A^T A + lambda I) c = A^T y, solved by Cholesky. The regulariser
        // makes the system SPD, so Cholesky cannot fail here; a failure would
        // be a bug rather than a data condition.
        let mut ata = vec![0.0f64; k * k];
        let mut aty = vec![0.0f64; k];
        for a in 0..k {
            for b in 0..k {
                let mut acc = 0.0;
                for row in 0..n {
                    acc += matrix[row * k + a] * matrix[row * k + b];
                }
                ata[a * k + b] = acc;
            }
            ata[a * k + a] += lambda_reg;
            let mut acc = 0.0;
            for row in 0..n {
                acc += matrix[row * k + a] * target[row];
            }
            aty[a] = acc;
        }
        cholesky_solve(&ata, &aty, k)?
    } else {
        least_squares_qr(&matrix, n, k, target)
    };

    let mut sum_sq = 0.0f64;
    let mut sum_abs = 0.0f64;
    for row in 0..n {
        let mut pred = 0.0;
        for j in 0..k {
            pred += matrix[row * k + j] * coeffs[j];
        }
        let r = target[row] - pred;
        sum_sq += r * r;
        sum_abs += r.abs();
    }
    let nf = n as f64;
    let mse = sum_sq / nf;
    let mae = sum_abs / nf;

    let mean = target.iter().sum::<f64>() / nf;
    let var_y = target.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / nf;
    let r2 = if var_y > 1e-15 {
        1.0 - (mse / var_y)
    } else if mse < 1e-15 {
        1.0
    } else {
        0.0
    };

    let (cond, _, _, _, is_well, _) = conditioning(cols, 1e-12);

    Ok((coeffs, mse, mae, r2, cond, is_well))
}

/// Machine-epsilon identifiability boundary.
///
/// Mirrors `RustKernelBridge._fallback_precision_boundary`, including the
/// custom-precision fallback for any bit width other than 32 or 64.
pub fn precision_boundary(
    separation_delta: f64,
    condition_number: f64,
    precision_bits: i64,
) -> (String, f64, f64, bool, bool) {
    let (eps_mach, name) = match precision_bits {
        32 => (f32::EPSILON as f64, "Float32 (Single Precision)".to_string()),
        64 => (f64::EPSILON, "Float64 (Double Precision)".to_string()),
        other => (
            2.0f64.powi(-(other - 1) as i32),
            format!("CustomPrecision ({other}-bit)"),
        ),
    };
    let eps_star = (2.0 * eps_mach).sqrt();
    let effective_noise = eps_mach * condition_number.max(1.0);
    let resolved = separation_delta > effective_noise && separation_delta > eps_star;
    (name, eps_mach, eps_star, resolved, resolved)
}

#[inline]
fn to_matrix<V: crate::data::DataCol>(cols: &[V], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n_rows * n_cols];
    for c in 0..n_cols {
        for r in 0..n_rows {
            m[r * n_cols + c] = cols[c].row(r);
        }
    }
    m
}

/// Singular values of a row-major `m x n` matrix, via one-sided Jacobi.
///
/// One-sided Jacobi is chosen over a LAPACK-backed SVD because it needs no
/// external BLAS and is accurate to high relative precision on the small,
/// frequently near-singular design matrices this kernel is handed -- which is
/// exactly the regime where `effective_rank` has to be right.
///
/// Returns `min(m, n)` values. Does not check convergence, so callers that
/// need a verdict on conditioning read `sigma_min` against the same `1e-15`
/// floor the reference side uses.
pub fn singular_values(matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    if m == 0 || n == 0 {
        return Vec::new();
    }
    // One-sided Jacobi wants a tall matrix. For a wide one, the singular
    // values of A are those of A^T.
    if m < n {
        let mut t = vec![0.0f64; n * m];
        for r in 0..m {
            for c in 0..n {
                t[c * m + r] = matrix[r * n + c];
            }
        }
        return singular_values(&t, n, m);
    }

    let mut a = matrix.to_vec();
    let eps = f64::EPSILON;
    let max_sweeps = 60;

    for _ in 0..max_sweeps {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..m {
                    let ap = a[i * n + p];
                    let aq = a[i * n + q];
                    alpha += ap * ap;
                    beta += aq * aq;
                    gamma += ap * aq;
                }
                if gamma.abs() <= eps * (alpha * beta).sqrt() || gamma == 0.0 {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let ap = a[i * n + p];
                    let aq = a[i * n + q];
                    a[i * n + p] = c * ap - s * aq;
                    a[i * n + q] = s * ap + c * aq;
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }

    let mut s: Vec<f64> = (0..n)
        .map(|c| {
            let mut acc = 0.0;
            for i in 0..m {
                let v = a[i * n + c];
                acc += v * v;
            }
            acc.sqrt()
        })
        .collect();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    s
}

/// Solve a symmetric positive-definite system by Cholesky factorisation.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut acc = a[i * n + j];
            for k in 0..j {
                acc -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(acc > 0.0) {
                    return Err(format!(
                        "normal equations are not positive definite (pivot {i} = {acc}); \
                         the design matrix is degenerate"
                    ));
                }
                l[i * n + j] = acc.sqrt();
            } else {
                l[i * n + j] = acc / l[j * n + j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[i * n + k] * y[k];
        }
        y[i] = acc / l[i * n + i];
    }
    // Back substitution: L^T x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut acc = y[i];
        for k in (i + 1)..n {
            acc -= l[k * n + i] * x[k];
        }
        x[i] = acc / l[i * n + i];
    }
    Ok(x)
}

/// Unregularised least squares via Householder QR, used when
/// `lambda_reg <= 1e-15`.
fn least_squares_qr(matrix: &[f64], n: usize, k: usize, b: &[f64]) -> Vec<f64> {
    let mut a = matrix.to_vec();
    let mut rhs = b.to_vec();
    let cols = k.min(n);

    for j in 0..cols {
        // Householder vector for column j below the diagonal.
        let mut norm = 0.0;
        for i in j..n {
            norm += a[i * k + j] * a[i * k + j];
        }
        norm = norm.sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if a[j * k + j] > 0.0 { -norm } else { norm };
        let mut v = vec![0.0f64; n];
        for i in j..n {
            v[i] = a[i * k + j];
        }
        v[j] -= alpha;

        let mut vtv = 0.0;
        for i in j..n {
            vtv += v[i] * v[i];
        }
        if vtv == 0.0 {
            continue;
        }

        for c in j..k {
            let mut dot = 0.0;
            for i in j..n {
                dot += v[i] * a[i * k + c];
            }
            let factor = 2.0 * dot / vtv;
            for i in j..n {
                a[i * k + c] -= factor * v[i];
            }
        }
        let mut dot = 0.0;
        for i in j..n {
            dot += v[i] * rhs[i];
        }
        let factor = 2.0 * dot / vtv;
        for i in j..n {
            rhs[i] -= factor * v[i];
        }
    }

    // Back substitution on the upper triangle.
    let mut x = vec![0.0f64; k];
    for i in (0..cols).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..cols {
            acc -= a[i * k + j] * x[j];
        }
        let diag = a[i * k + i];
        x[i] = if diag.abs() > 1e-300 { acc / diag } else { 0.0 };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_on_a_perfect_fit() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let (mse, mae, r2, max_res, nan) = metrics(&y, &y);
        assert_eq!((mse, mae, max_res), (0.0, 0.0, 0.0));
        assert_eq!(r2, 1.0);
        assert!(!nan);
    }

    #[test]
    fn metrics_refuse_to_score_a_non_finite_residual() {
        let y = vec![1.0, 2.0, 3.0];
        let p = vec![1.0, f64::INFINITY, 3.0];
        let (_, _, _, _, nan) = metrics(&y, &p);
        assert!(nan, "a residual containing inf must not yield a verdict");
    }

    #[test]
    fn metrics_on_a_flat_target_are_not_a_nan_artefact() {
        let y = vec![5.0; 4];
        let (_, _, r2, _, nan) = metrics(&y, &y);
        assert!(!nan);
        assert_eq!(r2, 1.0, "zero-variance target with zero error is a perfect fit");
    }

    #[test]
    fn jacobi_finds_a_known_spectrum() {
        // Diagonal 3x3: singular values are exactly 3, 2, 1.
        let m = vec![3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        let s = singular_values(&m, 3, 3);
        assert_eq!(s.len(), 3);
        for (got, want) in s.iter().zip([3.0, 2.0, 1.0].iter()) {
            assert!((got - want).abs() < 1e-10, "{got} != {want}");
        }
    }

    #[test]
    fn collinear_columns_have_rank_one() {
        let cols = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];
        let (cond, _, _, rank, is_well, _) = conditioning(&cols, 1e-12);
        assert_eq!(rank, 1);
        assert!(!is_well);
        assert!(cond.is_infinite() || cond > 1e10, "cond = {cond}");
    }

    #[test]
    fn identity_is_perfectly_conditioned() {
        let cols = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (cond, _, _, rank, is_well, _) = conditioning(&cols, 1e-12);
        assert_eq!(rank, 3);
        assert!(is_well);
        assert!((cond - 1.0).abs() < 1e-10, "cond = {cond}");
    }

    #[test]
    fn non_finite_columns_report_ill_conditioned_instead_of_raising() {
        let cols = vec![vec![f64::NAN; 10], vec![1.0; 10]];
        let (cond, _, _, rank, is_well, _) = conditioning(&cols, 1e-12);
        assert_eq!(cond, f64::INFINITY);
        assert_eq!(rank, 0);
        assert!(!is_well);
    }

    #[test]
    fn regularised_solve_recovers_known_coefficients() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let ones = vec![1.0; 50];
        let target: Vec<f64> = x.iter().map(|v| 3.0 * v - 1.5).collect();
        let (c, mse, _, r2, _, is_well) =
            linear_least_squares(&[x, ones], &target, 1e-9).unwrap();
        assert!((c[0] - 3.0).abs() < 1e-4, "c0 = {}", c[0]);
        assert!((c[1] + 1.5).abs() < 1e-4, "c1 = {}", c[1]);
        assert!(mse < 1e-16, "mse = {mse}");
        assert!(r2 > 1.0 - 1e-12, "r2 = {r2}");
        assert!(is_well);
    }

    #[test]
    fn qr_solve_matches_the_regularised_path_on_a_well_conditioned_problem() {
        let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
        let ones = vec![1.0; 40];
        let target: Vec<f64> = x.iter().map(|v| 2.0 * v + 0.25).collect();
        let qr = linear_least_squares(&[x.clone(), ones.clone()], &target, 0.0).unwrap();
        assert!((qr.0[0] - 2.0).abs() < 1e-8, "c0 = {}", qr.0[0]);
        assert!((qr.0[1] - 0.25).abs() < 1e-8, "c1 = {}", qr.0[1]);
    }

    #[test]
    fn precision_boundary_scales_with_bit_width() {
        let (name, eps, _, resolved, _) = precision_boundary(1e-3, 1.0, 32);
        assert_eq!(name, "Float32 (Single Precision)");
        assert!((eps - f32::EPSILON as f64).abs() < 1e-18);
        assert!(resolved);

        let (_, _, _, resolved, _) = precision_boundary(1e-5, 1.0, 32);
        assert!(!resolved, "1e-5 is below the f32 threshold");

        let (_, _, _, resolved, _) = precision_boundary(1e-5, 1.0, 64);
        assert!(resolved, "1e-5 clears the f64 threshold");
    }

    #[test]
    fn precision_boundary_reports_both_flags_consistently() {
        // The bridge returns `resolved` under two keys; they must not disagree.
        for delta in [1e-8, 1e-5, 1e-1] {
            for bits in [32, 64, 128] {
                let (_, _, _, observed, resolved) = precision_boundary(delta, 1.0, bits);
                assert_eq!(observed, resolved, "delta={delta} bits={bits}");
            }
        }
    }
}

/// Error function via the Abramowitz & Stegun 7.1.26 five-term
/// approximation (|error| <= 1.5e-7). Odd, erf(±inf) = ±1, erf(NaN) = NaN.
pub fn erf_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        // signum(0.0) is 1.0 in Rust, so the generic path would return
        // ~1e-9 instead of exactly 0.
        return 0.0;
    }
    if x.is_infinite() {
        return x.signum();
    }
    let s = x.signum();
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    s * (1.0 - poly * (-ax * ax).exp())
}

/// Bessel function of the first kind, order 0, via the defining power
/// series with early termination. Accurate to ~1e-12 for |x| <= 8 and
/// degrading gracefully beyond (alternating cancellation); callers needing
/// far-field asymptotics should extend this, not silently accept drift.
pub fn bessel_j0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let z = (x * x) / 4.0;
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    for k in 1..100 {
        term *= -z / ((k * k) as f64);
        sum += term;
        if term.abs() < 1e-16 * sum.abs().max(1e-300) && k > 10 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod special_fn_tests {
    use super::{bessel_j0, erf_approx};

    #[test]
    fn erf_matches_reference_values() {
        // Abramowitz-Stegun 7.1.26 claims |error| <= 1.5e-7; pin at 1e-6.
        assert_eq!(erf_approx(0.0), 0.0);
        assert!((erf_approx(1.0) - 0.8427007929).abs() < 1e-6);
        assert!((erf_approx(-1.0) + 0.8427007929).abs() < 1e-6);
        assert!((erf_approx(2.0) - 0.9953222650).abs() < 1e-6);
        assert_eq!(erf_approx(f64::INFINITY), 1.0);
        assert_eq!(erf_approx(f64::NEG_INFINITY), -1.0);
        assert!(erf_approx(f64::NAN).is_nan());
    }

    #[test]
    fn bessel_j0_matches_reference_values() {
        // J0(0) = 1; first positive zero at 2.4048255576957728.
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-14);
        assert!((bessel_j0(2.4048255576957728)).abs() < 1e-9);
        assert!(bessel_j0(f64::NAN).is_nan());
        assert_eq!(bessel_j0(f64::INFINITY), 0.0);
    }
}
#[cfg(test)]
mod reference_grid_audit {
    use super::{bessel_j0, erf_approx, singular_values};

    /// 2026-10-17 numeric.rs audit: dense reference grids generated from
    /// Python 3.12 `math.erf` and `scipy.special.j0` (LAPACK-backed), pinned
    /// against the documented accuracy claims AFTER measuring the real
    /// maxima: erf_approx errs 1.388e-7 on [-10, 10] (A&S 7.1.26 claim
    /// 1.5e-7 — exactly the documented bound, so the pin is 1.45e-7);
    /// bessel_j0 errs 1.06e-14 absolute on [-8, 8] with 4.0e-11 relative
    /// near zeros (doc claims ~1e-12; the pin is absolute 1e-12). The
    /// linear-algebra set (metrics/conditioning/QR/Cholesky/precision) is
    /// differentially covered against numpy by tests/test_rust_bridge*.py.
    const ERF_GRID: &[(f64, f64)] = &[
        (-6.00000000000000000e+00,
         -1.00000000000000000e+00),
        (-5.95000000000000018e+00,
         -1.00000000000000000e+00),
        (-5.90000000000000036e+00,
         -9.99999999999999889e-01),
        (-5.85000000000000053e+00,
         -9.99999999999999889e-01),
        (-5.80000000000000071e+00,
         -9.99999999999999778e-01),
        (-5.75000000000000000e+00,
         -9.99999999999999556e-01),
        (-5.70000000000000018e+00,
         -9.99999999999999223e-01),
        (-5.65000000000000036e+00,
         -9.99999999999998668e-01),
        (-5.60000000000000053e+00,
         -9.99999999999997669e-01),
        (-5.55000000000000071e+00,
         -9.99999999999995781e-01),
        (-5.50000000000000000e+00,
         -9.99999999999992673e-01),
        (-5.45000000000000018e+00,
         -9.99999999999987121e-01),
        (-5.40000000000000036e+00,
         -9.99999999999977685e-01),
        (-5.35000000000000053e+00,
         -9.99999999999961475e-01),
        (-5.30000000000000071e+00,
         -9.99999999999933831e-01),
        (-5.25000000000000000e+00,
         -9.99999999999886868e-01),
        (-5.20000000000000018e+00,
         -9.99999999999807487e-01),
        (-5.15000000000000036e+00,
         -9.99999999999674039e-01),
        (-5.10000000000000053e+00,
         -9.99999999999450662e-01),
        (-5.05000000000000071e+00,
         -9.99999999999078737e-01),
        (-5.00000000000000000e+00,
         -9.99999999998462563e-01),
        (-4.95000000000000018e+00,
         -9.99999999997446931e-01),
        (-4.90000000000000036e+00,
         -9.99999999995781041e-01),
        (-4.85000000000000053e+00,
         -9.99999999993062438e-01),
        (-4.80000000000000071e+00,
         -9.99999999988647859e-01),
        (-4.75000000000000000e+00,
         -9.99999999981514898e-01),
        (-4.70000000000000018e+00,
         -9.99999999970047404e-01),
        (-4.65000000000000036e+00,
         -9.99999999951702967e-01),
        (-4.60000000000000053e+00,
         -9.99999999922503990e-01),
        (-4.54999999999999982e+00,
         -9.99999999876259538e-01),
        (-4.50000000000000000e+00,
         -9.99999999803383943e-01),
        (-4.45000000000000018e+00,
         -9.99999999689113683e-01),
        (-4.40000000000000036e+00,
         -9.99999999510828963e-01),
        (-4.35000000000000053e+00,
         -9.99999999234055581e-01),
        (-4.29999999999999982e+00,
         -9.99999998806528234e-01),
        (-4.25000000000000000e+00,
         -9.99999998149425862e-01),
        (-4.20000000000000018e+00,
         -9.99999997144505826e-01),
        (-4.15000000000000036e+00,
         -9.99999995615322934e-01),
        (-4.10000000000000053e+00,
         -9.99999993299972356e-01),
        (-4.04999999999999982e+00,
         -9.99999989811755086e-01),
        (-4.00000000000000000e+00,
         -9.99999984582742085e-01),
        (-3.95000000000000018e+00,
         -9.99999976783267797e-01),
        (-3.90000000000000036e+00,
         -9.99999965207751362e-01),
        (-3.85000000000000009e+00,
         -9.99999948113706627e-01),
        (-3.80000000000000027e+00,
         -9.99999922996072543e-01),
        (-3.75000000000000000e+00,
         -9.99999886272743388e-01),
        (-3.70000000000000018e+00,
         -9.99999832848942116e-01),
        (-3.65000000000000036e+00,
         -9.99999755517349431e-01),
        (-3.60000000000000009e+00,
         -9.99999644137007038e-01),
        (-3.55000000000000027e+00,
         -9.99999484516175374e-01),
        (-3.50000000000000000e+00,
         -9.99999256901627609e-01),
        (-3.45000000000000018e+00,
         -9.99998933948206492e-01),
        (-3.40000000000000036e+00,
         -9.99998478006637104e-01),
        (-3.35000000000000009e+00,
         -9.99997837523179944e-01),
        (-3.30000000000000027e+00,
         -9.99996942290203528e-01),
        (-3.25000000000000000e+00,
         -9.99995697220536273e-01),
        (-3.20000000000000018e+00,
         -9.99993974238848282e-01),
        (-3.15000000000000036e+00,
         -9.99991601788684736e-01),
        (-3.10000000000000009e+00,
         -9.99988351342632753e-01),
        (-3.05000000000000027e+00,
         -9.99983920174239871e-01),
        (-3.00000000000000000e+00,
         -9.99977909503001361e-01),
        (-2.95000000000000018e+00,
         -9.99969796957935864e-01),
        (-2.90000000000000036e+00,
         -9.99958902121900528e-01),
        (-2.85000000000000009e+00,
         -9.99944343720038598e-01),
        (-2.80000000000000027e+00,
         -9.99924986805334592e-01),
        (-2.75000000000000000e+00,
         -9.99899378077880319e-01),
        (-2.70000000000000018e+00,
         -9.99865667260059432e-01),
        (-2.65000000000000036e+00,
         -9.99821512247975996e-01),
        (-2.60000000000000009e+00,
         -9.99763965583470693e-01),
        (-2.55000000000000027e+00,
         -9.99689339657360843e-01),
        (-2.50000000000000000e+00,
         -9.99593047982554994e-01),
        (-2.45000000000000018e+00,
         -9.99469419887748955e-01),
        (-2.40000000000000036e+00,
         -9.99311486103354962e-01),
        (-2.35000000000000009e+00,
         -9.99110732967867565e-01),
        (-2.30000000000000027e+00,
         -9.98856823402643368e-01),
        (-2.25000000000000000e+00,
         -9.98537283413318821e-01),
        (-2.20000000000000018e+00,
         -9.98137153702018165e-01),
        (-2.14999999999999991e+00,
         -9.97638607037325320e-01),
        (-2.10000000000000009e+00,
         -9.97020533343667026e-01),
        (-2.05000000000000027e+00,
         -9.96258096044456920e-01),
        (-2.00000000000000000e+00,
         -9.95322265018952712e-01),
        (-1.95000000000000018e+00,
         -9.94179333592189129e-01),
        (-1.90000000000000013e+00,
         -9.92790429235257510e-01),
        (-1.85000000000000009e+00,
         -9.91111030056085696e-01),
        (-1.80000000000000004e+00,
         -9.89090501635730757e-01),
        (-1.75000000000000000e+00,
         -9.86671671219182422e-01),
        (-1.70000000000000018e+00,
         -9.83790458590774630e-01),
        (-1.65000000000000013e+00,
         -9.80375585023360308e-01),
        (-1.60000000000000009e+00,
         -9.76348383344644022e-01),
        (-1.55000000000000004e+00,
         -9.71622733262012539e-01),
        (-1.50000000000000000e+00,
         -9.66105146475310761e-01),
        (-1.45000000000000018e+00,
         -9.59695025637459231e-01),
        (-1.40000000000000013e+00,
         -9.52285119762648846e-01),
        (-1.35000000000000009e+00,
         -9.43762196122724051e-01),
        (-1.30000000000000004e+00,
         -9.34007944940652424e-01),
        (-1.25000000000000000e+00,
         -9.22900128256458174e-01),
        (-1.20000000000000018e+00,
         -9.10313978229635445e-01),
        (-1.15000000000000013e+00,
         -8.96123842936915049e-01),
        (-1.10000000000000009e+00,
         -8.80205069574081733e-01),
        (-1.05000000000000004e+00,
         -8.62436106090096688e-01),
        (-1.00000000000000000e+00,
         -8.42700792949714894e-01),
        (-9.50000000000000067e-01,
         -8.20890807273278011e-01),
        (-9.00000000000000022e-01,
         -7.96908212422832163e-01),
        (-8.50000000000000089e-01,
         -7.70668057608352552e-01),
        (-8.00000000000000044e-01,
         -7.42100964707660515e-01),
        (-7.50000000000000000e-01,
         -7.11155633653515085e-01),
        (-7.00000000000000067e-01,
         -6.77801193837418547e-01),
        (-6.50000000000000022e-01,
         -6.42029327355671864e-01),
        (-6.00000000000000089e-01,
         -6.03856090847926019e-01),
        (-5.50000000000000044e-01,
         -5.63323366325109043e-01),
        (-5.00000000000000000e-01,
         -5.20499877813046519e-01),
        (-4.50000000000000011e-01,
         -4.75481719786923662e-01),
        (-4.00000000000000022e-01,
         -4.28392355046668449e-01),
        (-3.50000000000000033e-01,
         -3.79382053562310362e-01),
        (-3.00000000000000044e-01,
         -3.28626759459127449e-01),
        (-2.50000000000000000e-01,
         -2.76326390168236902e-01),
        (-2.00000000000000011e-01,
         -2.22702589210478474e-01),
        (-1.50000000000000022e-01,
         -1.67995971427363527e-01),
        (-1.00000000000000006e-01,
         -1.12462916018284897e-01),
        (-5.00000000000000028e-02,
         -5.63719777970166300e-02),
        (0.00000000000000000e+00,
         0.00000000000000000e+00),
        (5.00000000000000028e-02,
         5.63719777970166300e-02),
        (1.00000000000000006e-01,
         1.12462916018284897e-01),
        (1.50000000000000022e-01,
         1.67995971427363527e-01),
        (2.00000000000000011e-01,
         2.22702589210478474e-01),
        (2.50000000000000000e-01,
         2.76326390168236902e-01),
        (3.00000000000000044e-01,
         3.28626759459127449e-01),
        (3.50000000000000033e-01,
         3.79382053562310362e-01),
        (4.00000000000000022e-01,
         4.28392355046668449e-01),
        (4.50000000000000011e-01,
         4.75481719786923662e-01),
        (5.00000000000000000e-01,
         5.20499877813046519e-01),
        (5.50000000000000044e-01,
         5.63323366325109043e-01),
        (6.00000000000000089e-01,
         6.03856090847926019e-01),
        (6.50000000000000022e-01,
         6.42029327355671864e-01),
        (7.00000000000000067e-01,
         6.77801193837418547e-01),
        (7.50000000000000000e-01,
         7.11155633653515085e-01),
        (8.00000000000000044e-01,
         7.42100964707660515e-01),
        (8.50000000000000089e-01,
         7.70668057608352552e-01),
        (9.00000000000000022e-01,
         7.96908212422832163e-01),
        (9.50000000000000067e-01,
         8.20890807273278011e-01),
        (1.00000000000000000e+00,
         8.42700792949714894e-01),
        (1.05000000000000004e+00,
         8.62436106090096688e-01),
        (1.10000000000000009e+00,
         8.80205069574081733e-01),
        (1.15000000000000013e+00,
         8.96123842936915049e-01),
        (1.20000000000000018e+00,
         9.10313978229635445e-01),
        (1.25000000000000000e+00,
         9.22900128256458174e-01),
        (1.30000000000000004e+00,
         9.34007944940652424e-01),
        (1.35000000000000009e+00,
         9.43762196122724051e-01),
        (1.40000000000000013e+00,
         9.52285119762648846e-01),
        (1.45000000000000018e+00,
         9.59695025637459231e-01),
        (1.50000000000000000e+00,
         9.66105146475310761e-01),
        (1.55000000000000004e+00,
         9.71622733262012539e-01),
        (1.60000000000000009e+00,
         9.76348383344644022e-01),
        (1.65000000000000013e+00,
         9.80375585023360308e-01),
        (1.70000000000000018e+00,
         9.83790458590774630e-01),
        (1.75000000000000000e+00,
         9.86671671219182422e-01),
        (1.80000000000000004e+00,
         9.89090501635730757e-01),
        (1.85000000000000009e+00,
         9.91111030056085696e-01),
        (1.90000000000000013e+00,
         9.92790429235257510e-01),
        (1.95000000000000018e+00,
         9.94179333592189129e-01),
        (2.00000000000000000e+00,
         9.95322265018952712e-01),
        (2.05000000000000027e+00,
         9.96258096044456920e-01),
        (2.10000000000000009e+00,
         9.97020533343667026e-01),
        (2.14999999999999991e+00,
         9.97638607037325320e-01),
        (2.20000000000000018e+00,
         9.98137153702018165e-01),
        (2.25000000000000000e+00,
         9.98537283413318821e-01),
        (2.30000000000000027e+00,
         9.98856823402643368e-01),
        (2.35000000000000009e+00,
         9.99110732967867565e-01),
        (2.40000000000000036e+00,
         9.99311486103354962e-01),
        (2.45000000000000018e+00,
         9.99469419887748955e-01),
        (2.50000000000000000e+00,
         9.99593047982554994e-01),
        (2.55000000000000027e+00,
         9.99689339657360843e-01),
        (2.60000000000000009e+00,
         9.99763965583470693e-01),
        (2.65000000000000036e+00,
         9.99821512247975996e-01),
        (2.70000000000000018e+00,
         9.99865667260059432e-01),
        (2.75000000000000000e+00,
         9.99899378077880319e-01),
        (2.80000000000000027e+00,
         9.99924986805334592e-01),
        (2.85000000000000009e+00,
         9.99944343720038598e-01),
        (2.90000000000000036e+00,
         9.99958902121900528e-01),
        (2.95000000000000018e+00,
         9.99969796957935864e-01),
        (3.00000000000000000e+00,
         9.99977909503001361e-01),
        (3.05000000000000027e+00,
         9.99983920174239871e-01),
        (3.10000000000000009e+00,
         9.99988351342632753e-01),
        (3.15000000000000036e+00,
         9.99991601788684736e-01),
        (3.20000000000000018e+00,
         9.99993974238848282e-01),
        (3.25000000000000000e+00,
         9.99995697220536273e-01),
        (3.30000000000000027e+00,
         9.99996942290203528e-01),
        (3.35000000000000009e+00,
         9.99997837523179944e-01),
        (3.40000000000000036e+00,
         9.99998478006637104e-01),
        (3.45000000000000018e+00,
         9.99998933948206492e-01),
        (3.50000000000000000e+00,
         9.99999256901627609e-01),
        (3.55000000000000027e+00,
         9.99999484516175374e-01),
        (3.60000000000000009e+00,
         9.99999644137007038e-01),
        (3.65000000000000036e+00,
         9.99999755517349431e-01),
        (3.70000000000000018e+00,
         9.99999832848942116e-01),
        (3.75000000000000000e+00,
         9.99999886272743388e-01),
        (3.80000000000000027e+00,
         9.99999922996072543e-01),
        (3.85000000000000009e+00,
         9.99999948113706627e-01),
        (3.90000000000000036e+00,
         9.99999965207751362e-01),
        (3.95000000000000018e+00,
         9.99999976783267797e-01),
        (4.00000000000000000e+00,
         9.99999984582742085e-01),
        (4.04999999999999982e+00,
         9.99999989811755086e-01),
        (4.10000000000000053e+00,
         9.99999993299972356e-01),
        (4.15000000000000036e+00,
         9.99999995615322934e-01),
        (4.20000000000000018e+00,
         9.99999997144505826e-01),
        (4.25000000000000000e+00,
         9.99999998149425862e-01),
        (4.29999999999999982e+00,
         9.99999998806528234e-01),
        (4.35000000000000053e+00,
         9.99999999234055581e-01),
        (4.40000000000000036e+00,
         9.99999999510828963e-01),
        (4.45000000000000018e+00,
         9.99999999689113683e-01),
        (4.50000000000000000e+00,
         9.99999999803383943e-01),
        (4.54999999999999982e+00,
         9.99999999876259538e-01),
        (4.60000000000000053e+00,
         9.99999999922503990e-01),
        (4.65000000000000036e+00,
         9.99999999951702967e-01),
        (4.70000000000000018e+00,
         9.99999999970047404e-01),
        (4.75000000000000000e+00,
         9.99999999981514898e-01),
        (4.80000000000000071e+00,
         9.99999999988647859e-01),
        (4.85000000000000053e+00,
         9.99999999993062438e-01),
        (4.90000000000000036e+00,
         9.99999999995781041e-01),
        (4.95000000000000018e+00,
         9.99999999997446931e-01),
        (5.00000000000000000e+00,
         9.99999999998462563e-01),
        (5.05000000000000071e+00,
         9.99999999999078737e-01),
        (5.10000000000000053e+00,
         9.99999999999450662e-01),
        (5.15000000000000036e+00,
         9.99999999999674039e-01),
        (5.20000000000000018e+00,
         9.99999999999807487e-01),
        (5.25000000000000000e+00,
         9.99999999999886868e-01),
        (5.30000000000000071e+00,
         9.99999999999933831e-01),
        (5.35000000000000053e+00,
         9.99999999999961475e-01),
        (5.40000000000000036e+00,
         9.99999999999977685e-01),
        (5.45000000000000018e+00,
         9.99999999999987121e-01),
        (5.50000000000000000e+00,
         9.99999999999992673e-01),
        (5.55000000000000071e+00,
         9.99999999999995781e-01),
        (5.60000000000000053e+00,
         9.99999999999997669e-01),
        (5.65000000000000036e+00,
         9.99999999999998668e-01),
        (5.70000000000000018e+00,
         9.99999999999999223e-01),
        (5.75000000000000000e+00,
         9.99999999999999556e-01),
        (5.80000000000000071e+00,
         9.99999999999999778e-01),
        (5.85000000000000053e+00,
         9.99999999999999889e-01),
        (5.90000000000000036e+00,
         9.99999999999999889e-01),
        (5.95000000000000018e+00,
         1.00000000000000000e+00),
        (6.00000000000000000e+00,
         1.00000000000000000e+00),
        (6.50000000000000000e+00,
         1.00000000000000000e+00),
        (7.00000000000000000e+00,
         1.00000000000000000e+00),
        (7.50000000000000000e+00,
         1.00000000000000000e+00),
        (8.00000000000000000e+00,
         1.00000000000000000e+00),
        (9.00000000000000000e+00,
         1.00000000000000000e+00),
        (1.00000000000000000e+01,
         1.00000000000000000e+00),
        (-6.50000000000000000e+00,
         -1.00000000000000000e+00),
        (-7.00000000000000000e+00,
         -1.00000000000000000e+00),
        (-7.50000000000000000e+00,
         -1.00000000000000000e+00),
        (-8.00000000000000000e+00,
         -1.00000000000000000e+00),
        (-9.00000000000000000e+00,
         -1.00000000000000000e+00),
        (-1.00000000000000000e+01,
         -1.00000000000000000e+00)
    ];
    const J0_GRID: &[(f64, f64)] = &[
        (-8.00000000000000000e+00,
         1.71650807137553901e-01),
        (-7.95000000000000018e+00,
         1.83199569087022313e-01),
        (-7.90000000000000036e+00,
         1.94361844841278192e-01),
        (-7.85000000000000053e+00,
         2.05107645401739064e-01),
        (-7.80000000000000071e+00,
         2.15407807746262753e-01),
        (-7.75000000000000000e+00,
         2.25234069120106684e-01),
        (-7.70000000000000018e+00,
         2.34559139586464377e-01),
        (-7.65000000000000036e+00,
         2.43356772659918202e-01),
        (-7.60000000000000053e+00,
         2.51601833849976197e-01),
        (-7.55000000000000071e+00,
         2.59270366946112163e-01),
        (-7.50000000000000000e+00,
         2.66339657880378389e-01),
        (-7.45000000000000018e+00,
         2.72788296008710573e-01),
        (-7.40000000000000036e+00,
         2.78596232657477627e-01),
        (-7.35000000000000053e+00,
         2.83744836787635568e-01),
        (-7.30000000000000071e+00,
         2.88216947635014265e-01),
        (-7.25000000000000000e+00,
         2.91996924191779028e-01),
        (-7.20000000000000018e+00,
         2.95070691400957874e-01),
        (-7.15000000000000036e+00,
         2.97425782943094852e-01),
        (-7.10000000000000053e+00,
         2.99051380501550113e-01),
        (-7.05000000000000071e+00,
         2.99938349400731497e-01),
        (-7.00000000000000000e+00,
         3.00079270519555574e-01),
        (-6.95000000000000018e+00,
         2.99468468390713671e-01),
        (-6.90000000000000036e+00,
         2.98102035404820109e-01),
        (-6.85000000000000053e+00,
         2.95977852047236034e-01),
        (-6.80000000000000071e+00,
         2.93095603104272739e-01),
        (-6.75000000000000000e+00,
         2.89456789784556578e-01),
        (-6.70000000000000018e+00,
         2.85064737710576055e-01),
        (-6.65000000000000036e+00,
         2.79924600744788477e-01),
        (-6.60000000000000053e+00,
         2.74043360624146359e-01),
        (-6.55000000000000071e+00,
         2.67429822386454630e-01),
        (-6.50000000000000000e+00,
         2.60094605581606464e-01),
        (-6.45000000000000018e+00,
         2.52050131270409317e-01),
        (-6.40000000000000036e+00,
         2.43310604823406634e-01),
        (-6.35000000000000053e+00,
         2.33891994541793713e-01),
        (-6.30000000000000071e+00,
         2.23812006132191210e-01),
        (-6.25000000000000000e+00,
         2.13090053076660763e-01),
        (-6.20000000000000018e+00,
         2.01747222948904320e-01),
        (-6.15000000000000036e+00,
         1.89806239737044802e-01),
        (-6.10000000000000053e+00,
         1.77291422242743690e-01),
        (-6.05000000000000071e+00,
         1.64228638635619706e-01),
        (-6.00000000000000000e+00,
         1.50645257250996950e-01),
        (-5.95000000000000018e+00,
         1.36570093727890779e-01),
        (-5.90000000000000036e+00,
         1.22033354592822815e-01),
        (-5.85000000000000053e+00,
         1.07066577403525140e-01),
        (-5.80000000000000071e+00,
         9.17025675748164293e-02),
        (-5.75000000000000000e+00,
         7.59753320169010843e-02),
        (-5.70000000000000018e+00,
         5.99200097240374721e-02),
        (-5.65000000000000036e+00,
         4.35727994589076364e-02),
        (-5.60000000000000053e+00,
         2.69708846851146669e-02),
        (-5.55000000000000071e+00,
         1.01523559069746413e-02),
        (-5.50000000000000000e+00,
         -6.84386941781918879e-03),
        (-5.45000000000000018e+00,
         -2.39781292210390316e-02),
        (-5.40000000000000036e+00,
         -4.12101012449911860e-02),
        (-5.35000000000000053e+00,
         -5.84988893592435499e-02),
        (-5.30000000000000071e+00,
         -7.58031115855839072e-02),
        (-5.25000000000000000e+00,
         -9.30809896393178793e-02),
        (-5.20000000000000018e+00,
         -1.10290439790986461e-01),
        (-5.15000000000000036e+00,
         -1.27389164848996650e-01),
        (-5.10000000000000053e+00,
         -1.44334747060500312e-01),
        (-5.05000000000000071e+00,
         -1.61084741725177311e-01),
        (-5.00000000000000000e+00,
         -1.77596771314338292e-01),
        (-4.95000000000000018e+00,
         -1.93828619886003861e-01),
        (-4.90000000000000036e+00,
         -2.09738327585326179e-01),
        (-4.85000000000000053e+00,
         -2.25284285018900743e-01),
        (-4.80000000000000071e+00,
         -2.40425327291183266e-01),
        (-4.75000000000000000e+00,
         -2.55120827491373936e-01),
        (-4.70000000000000018e+00,
         -2.69330789419752692e-01),
        (-4.65000000000000036e+00,
         -2.83015939343566114e-01),
        (-4.60000000000000053e+00,
         -2.96137816574141122e-01),
        (-4.54999999999999982e+00,
         -3.08658862658980904e-01),
        (-4.50000000000000000e+00,
         -3.20542508985121488e-01),
        (-4.45000000000000018e+00,
         -3.31753262593042053e-01),
        (-4.40000000000000036e+00,
         -3.42256790003885647e-01),
        (-4.35000000000000053e+00,
         -3.52019998866669404e-01),
        (-4.29999999999999982e+00,
         -3.61011117236535173e-01),
        (-4.25000000000000000e+00,
         -3.69199770299895424e-01),
        (-4.20000000000000018e+00,
         -3.76557054367567645e-01),
        (-4.15000000000000036e+00,
         -3.83055607962638156e-01),
        (-4.10000000000000053e+00,
         -3.88669679835853654e-01),
        (-4.04999999999999982e+00,
         -3.93375193747787899e-01),
        (-4.00000000000000000e+00,
         -3.97149809863847292e-01),
        (-3.95000000000000018e+00,
         -3.99972982615375450e-01),
        (-3.90000000000000036e+00,
         -4.01826014887639971e-01),
        (-3.85000000000000009e+00,
         -4.02692108403355287e-01),
        (-3.80000000000000027e+00,
         -4.02556410178564139e-01),
        (-3.75000000000000000e+00,
         -4.01406054936174372e-01),
        (-3.70000000000000018e+00,
         -3.99230203371191195e-01),
        (-3.65000000000000036e+00,
         -3.96020076170688351e-01),
        (-3.60000000000000009e+00,
         -3.91768983700797824e-01),
        (-3.55000000000000027e+00,
         -3.86472351282452442e-01),
        (-3.50000000000000000e+00,
         -3.80127739987263458e-01),
        (-3.45000000000000018e+00,
         -3.72734862894736729e-01),
        (-3.40000000000000036e+00,
         -3.64295596762000518e-01),
        (-3.35000000000000009e+00,
         -3.54813989067321311e-01),
        (-3.30000000000000027e+00,
         -3.44296260398884724e-01),
        (-3.25000000000000000e+00,
         -3.32750802170611593e-01),
        (-3.20000000000000018e+00,
         -3.20188169657123045e-01),
        (-3.15000000000000036e+00,
         -3.06621070350352298e-01),
        (-3.10000000000000009e+00,
         -2.92064347650697664e-01),
        (-3.05000000000000027e+00,
         -2.76534959915994694e-01),
        (-3.00000000000000000e+00,
         -2.60051954901933502e-01),
        (-2.95000000000000018e+00,
         -2.42636439637843176e-01),
        (-2.90000000000000036e+00,
         -2.24311545791968325e-01),
        (-2.85000000000000009e+00,
         -2.05102390590473832e-01),
        (-2.80000000000000027e+00,
         -1.85036033364387509e-01),
        (-2.75000000000000000e+00,
         -1.64141427808513707e-01),
        (-2.70000000000000018e+00,
         -1.42449370046011936e-01),
        (-2.65000000000000036e+00,
         -1.19992442601780699e-01),
        (-2.60000000000000009e+00,
         -9.68049543970383591e-02),
        (-2.55000000000000027e+00,
         -7.29228768864880611e-02),
        (-2.50000000000000000e+00,
         -4.83837764681980392e-02),
        (-2.45000000000000018e+00,
         -2.32267433047989025e-02),
        (-2.40000000000000036e+00,
         2.50768329724359923e-03),
        (-2.35000000000000009e+00,
         2.87775927956617390e-02),
        (-2.30000000000000027e+00,
         5.55397844456018142e-02),
        (-2.25000000000000000e+00,
         8.27498512887340082e-02),
        (-2.20000000000000018e+00,
         1.10362266922173796e-01),
        (-2.14999999999999991e+00,
         1.38330474864731601e-01),
        (-2.10000000000000009e+00,
         1.66606980331990256e-01),
        (-2.05000000000000027e+00,
         1.95143444226129226e-01),
        (-2.00000000000000000e+00,
         2.23890779141235619e-01),
        (-1.95000000000000018e+00,
         2.52799247180120423e-01),
        (-1.90000000000000013e+00,
         2.81818559374385380e-01),
        (-1.85000000000000009e+00,
         3.10897976495655426e-01),
        (-1.80000000000000004e+00,
         3.39986411042558279e-01),
        (-1.75000000000000000e+00,
         3.69032530185150753e-01),
        (-1.70000000000000018e+00,
         3.97984859446109396e-01),
        (-1.65000000000000013e+00,
         4.26791886896105699e-01),
        (-1.60000000000000009e+00,
         4.55402167639380628e-01),
        (-1.55000000000000004e+00,
         4.83764428364631127e-01),
        (-1.50000000000000000e+00,
         5.11827671735918144e-01),
        (-1.45000000000000018e+00,
         5.39541280398398215e-01),
        (-1.40000000000000013e+00,
         5.66855120374288535e-01),
        (-1.35000000000000009e+00,
         5.93719643625565729e-01),
        (-1.30000000000000004e+00,
         6.20085989561508955e-01),
        (-1.25000000000000000e+00,
         6.45906085271285235e-01),
        (-1.20000000000000018e+00,
         6.71132744264362424e-01),
        (-1.15000000000000013e+00,
         6.95719763504612954e-01),
        (-1.10000000000000009e+00,
         7.19622018527510843e-01),
        (-1.05000000000000004e+00,
         7.42795556433860082e-01),
        (-1.00000000000000000e+00,
         7.65197686557966494e-01),
        (-9.50000000000000067e-01,
         7.86787068613109875e-01),
        (-9.00000000000000022e-01,
         8.07523798122544822e-01),
        (-8.50000000000000089e-01,
         8.27369488950077225e-01),
        (-8.00000000000000044e-01,
         8.46287352750480060e-01),
        (-7.50000000000000000e-01,
         8.64242275166648533e-01),
        (-7.00000000000000067e-01,
         8.81200888607405197e-01),
        (-6.50000000000000022e-01,
         8.97131641447265538e-01),
        (-6.00000000000000089e-01,
         9.12004863497210660e-01),
        (-5.50000000000000044e-01,
         9.25792827603606372e-01),
        (-5.00000000000000000e-01,
         9.38469807240812970e-01),
        (-4.50000000000000011e-01,
         9.50012129971735786e-01),
        (-4.00000000000000022e-01,
         9.60398226659563425e-01),
        (-3.50000000000000033e-01,
         9.69608676323187479e-01),
        (-3.00000000000000044e-01,
         9.77626246538296106e-01),
        (-2.50000000000000000e-01,
         9.84435929295852774e-01),
        (-2.00000000000000011e-01,
         9.90024972239576306e-01),
        (-1.50000000000000022e-01,
         9.94382905214139967e-01),
        (-1.00000000000000006e-01,
         9.97501562066040015e-01),
        (-5.00000000000000028e-02,
         9.99375097649468525e-01),
        (0.00000000000000000e+00,
         1.00000000000000000e+00),
        (5.00000000000000028e-02,
         9.99375097649468525e-01),
        (1.00000000000000006e-01,
         9.97501562066040015e-01),
        (1.50000000000000022e-01,
         9.94382905214139967e-01),
        (2.00000000000000011e-01,
         9.90024972239576306e-01),
        (2.50000000000000000e-01,
         9.84435929295852774e-01),
        (3.00000000000000044e-01,
         9.77626246538296106e-01),
        (3.50000000000000033e-01,
         9.69608676323187479e-01),
        (4.00000000000000022e-01,
         9.60398226659563425e-01),
        (4.50000000000000011e-01,
         9.50012129971735786e-01),
        (5.00000000000000000e-01,
         9.38469807240812970e-01),
        (5.50000000000000044e-01,
         9.25792827603606372e-01),
        (6.00000000000000089e-01,
         9.12004863497210660e-01),
        (6.50000000000000022e-01,
         8.97131641447265538e-01),
        (7.00000000000000067e-01,
         8.81200888607405197e-01),
        (7.50000000000000000e-01,
         8.64242275166648533e-01),
        (8.00000000000000044e-01,
         8.46287352750480060e-01),
        (8.50000000000000089e-01,
         8.27369488950077225e-01),
        (9.00000000000000022e-01,
         8.07523798122544822e-01),
        (9.50000000000000067e-01,
         7.86787068613109875e-01),
        (1.00000000000000000e+00,
         7.65197686557966494e-01),
        (1.05000000000000004e+00,
         7.42795556433860082e-01),
        (1.10000000000000009e+00,
         7.19622018527510843e-01),
        (1.15000000000000013e+00,
         6.95719763504612954e-01),
        (1.20000000000000018e+00,
         6.71132744264362424e-01),
        (1.25000000000000000e+00,
         6.45906085271285235e-01),
        (1.30000000000000004e+00,
         6.20085989561508955e-01),
        (1.35000000000000009e+00,
         5.93719643625565729e-01),
        (1.40000000000000013e+00,
         5.66855120374288535e-01),
        (1.45000000000000018e+00,
         5.39541280398398215e-01),
        (1.50000000000000000e+00,
         5.11827671735918144e-01),
        (1.55000000000000004e+00,
         4.83764428364631127e-01),
        (1.60000000000000009e+00,
         4.55402167639380628e-01),
        (1.65000000000000013e+00,
         4.26791886896105699e-01),
        (1.70000000000000018e+00,
         3.97984859446109396e-01),
        (1.75000000000000000e+00,
         3.69032530185150753e-01),
        (1.80000000000000004e+00,
         3.39986411042558279e-01),
        (1.85000000000000009e+00,
         3.10897976495655426e-01),
        (1.90000000000000013e+00,
         2.81818559374385380e-01),
        (1.95000000000000018e+00,
         2.52799247180120423e-01),
        (2.00000000000000000e+00,
         2.23890779141235619e-01),
        (2.05000000000000027e+00,
         1.95143444226129226e-01),
        (2.10000000000000009e+00,
         1.66606980331990256e-01),
        (2.14999999999999991e+00,
         1.38330474864731601e-01),
        (2.20000000000000018e+00,
         1.10362266922173796e-01),
        (2.25000000000000000e+00,
         8.27498512887340082e-02),
        (2.30000000000000027e+00,
         5.55397844456018142e-02),
        (2.35000000000000009e+00,
         2.87775927956617390e-02),
        (2.40000000000000036e+00,
         2.50768329724359923e-03),
        (2.45000000000000018e+00,
         -2.32267433047989025e-02),
        (2.50000000000000000e+00,
         -4.83837764681980392e-02),
        (2.55000000000000027e+00,
         -7.29228768864880611e-02),
        (2.60000000000000009e+00,
         -9.68049543970383591e-02),
        (2.65000000000000036e+00,
         -1.19992442601780699e-01),
        (2.70000000000000018e+00,
         -1.42449370046011936e-01),
        (2.75000000000000000e+00,
         -1.64141427808513707e-01),
        (2.80000000000000027e+00,
         -1.85036033364387509e-01),
        (2.85000000000000009e+00,
         -2.05102390590473832e-01),
        (2.90000000000000036e+00,
         -2.24311545791968325e-01),
        (2.95000000000000018e+00,
         -2.42636439637843176e-01),
        (3.00000000000000000e+00,
         -2.60051954901933502e-01),
        (3.05000000000000027e+00,
         -2.76534959915994694e-01),
        (3.10000000000000009e+00,
         -2.92064347650697664e-01),
        (3.15000000000000036e+00,
         -3.06621070350352298e-01),
        (3.20000000000000018e+00,
         -3.20188169657123045e-01),
        (3.25000000000000000e+00,
         -3.32750802170611593e-01),
        (3.30000000000000027e+00,
         -3.44296260398884724e-01),
        (3.35000000000000009e+00,
         -3.54813989067321311e-01),
        (3.40000000000000036e+00,
         -3.64295596762000518e-01),
        (3.45000000000000018e+00,
         -3.72734862894736729e-01),
        (3.50000000000000000e+00,
         -3.80127739987263458e-01),
        (3.55000000000000027e+00,
         -3.86472351282452442e-01),
        (3.60000000000000009e+00,
         -3.91768983700797824e-01),
        (3.65000000000000036e+00,
         -3.96020076170688351e-01),
        (3.70000000000000018e+00,
         -3.99230203371191195e-01),
        (3.75000000000000000e+00,
         -4.01406054936174372e-01),
        (3.80000000000000027e+00,
         -4.02556410178564139e-01),
        (3.85000000000000009e+00,
         -4.02692108403355287e-01),
        (3.90000000000000036e+00,
         -4.01826014887639971e-01),
        (3.95000000000000018e+00,
         -3.99972982615375450e-01),
        (4.00000000000000000e+00,
         -3.97149809863847292e-01),
        (4.04999999999999982e+00,
         -3.93375193747787899e-01),
        (4.10000000000000053e+00,
         -3.88669679835853654e-01),
        (4.15000000000000036e+00,
         -3.83055607962638156e-01),
        (4.20000000000000018e+00,
         -3.76557054367567645e-01),
        (4.25000000000000000e+00,
         -3.69199770299895424e-01),
        (4.29999999999999982e+00,
         -3.61011117236535173e-01),
        (4.35000000000000053e+00,
         -3.52019998866669404e-01),
        (4.40000000000000036e+00,
         -3.42256790003885647e-01),
        (4.45000000000000018e+00,
         -3.31753262593042053e-01),
        (4.50000000000000000e+00,
         -3.20542508985121488e-01),
        (4.54999999999999982e+00,
         -3.08658862658980904e-01),
        (4.60000000000000053e+00,
         -2.96137816574141122e-01),
        (4.65000000000000036e+00,
         -2.83015939343566114e-01),
        (4.70000000000000018e+00,
         -2.69330789419752692e-01),
        (4.75000000000000000e+00,
         -2.55120827491373936e-01),
        (4.80000000000000071e+00,
         -2.40425327291183266e-01),
        (4.85000000000000053e+00,
         -2.25284285018900743e-01),
        (4.90000000000000036e+00,
         -2.09738327585326179e-01),
        (4.95000000000000018e+00,
         -1.93828619886003861e-01),
        (5.00000000000000000e+00,
         -1.77596771314338292e-01),
        (5.05000000000000071e+00,
         -1.61084741725177311e-01),
        (5.10000000000000053e+00,
         -1.44334747060500312e-01),
        (5.15000000000000036e+00,
         -1.27389164848996650e-01),
        (5.20000000000000018e+00,
         -1.10290439790986461e-01),
        (5.25000000000000000e+00,
         -9.30809896393178793e-02),
        (5.30000000000000071e+00,
         -7.58031115855839072e-02),
        (5.35000000000000053e+00,
         -5.84988893592435499e-02),
        (5.40000000000000036e+00,
         -4.12101012449911860e-02),
        (5.45000000000000018e+00,
         -2.39781292210390316e-02),
        (5.50000000000000000e+00,
         -6.84386941781918879e-03),
        (5.55000000000000071e+00,
         1.01523559069746413e-02),
        (5.60000000000000053e+00,
         2.69708846851146669e-02),
        (5.65000000000000036e+00,
         4.35727994589076364e-02),
        (5.70000000000000018e+00,
         5.99200097240374721e-02),
        (5.75000000000000000e+00,
         7.59753320169010843e-02),
        (5.80000000000000071e+00,
         9.17025675748164293e-02),
        (5.85000000000000053e+00,
         1.07066577403525140e-01),
        (5.90000000000000036e+00,
         1.22033354592822815e-01),
        (5.95000000000000018e+00,
         1.36570093727890779e-01),
        (6.00000000000000000e+00,
         1.50645257250996950e-01),
        (6.05000000000000071e+00,
         1.64228638635619706e-01),
        (6.10000000000000053e+00,
         1.77291422242743690e-01),
        (6.15000000000000036e+00,
         1.89806239737044802e-01),
        (6.20000000000000018e+00,
         2.01747222948904320e-01),
        (6.25000000000000000e+00,
         2.13090053076660763e-01),
        (6.30000000000000071e+00,
         2.23812006132191210e-01),
        (6.35000000000000053e+00,
         2.33891994541793713e-01),
        (6.40000000000000036e+00,
         2.43310604823406634e-01),
        (6.45000000000000018e+00,
         2.52050131270409317e-01),
        (6.50000000000000000e+00,
         2.60094605581606464e-01),
        (6.55000000000000071e+00,
         2.67429822386454630e-01),
        (6.60000000000000053e+00,
         2.74043360624146359e-01),
        (6.65000000000000036e+00,
         2.79924600744788477e-01),
        (6.70000000000000018e+00,
         2.85064737710576055e-01),
        (6.75000000000000000e+00,
         2.89456789784556578e-01),
        (6.80000000000000071e+00,
         2.93095603104272739e-01),
        (6.85000000000000053e+00,
         2.95977852047236034e-01),
        (6.90000000000000036e+00,
         2.98102035404820109e-01),
        (6.95000000000000018e+00,
         2.99468468390713671e-01),
        (7.00000000000000000e+00,
         3.00079270519555574e-01),
        (7.05000000000000071e+00,
         2.99938349400731497e-01),
        (7.10000000000000053e+00,
         2.99051380501550113e-01),
        (7.15000000000000036e+00,
         2.97425782943094852e-01),
        (7.20000000000000018e+00,
         2.95070691400957874e-01),
        (7.25000000000000000e+00,
         2.91996924191779028e-01),
        (7.30000000000000071e+00,
         2.88216947635014265e-01),
        (7.35000000000000053e+00,
         2.83744836787635568e-01),
        (7.40000000000000036e+00,
         2.78596232657477627e-01),
        (7.45000000000000018e+00,
         2.72788296008710573e-01),
        (7.50000000000000000e+00,
         2.66339657880378389e-01),
        (7.55000000000000071e+00,
         2.59270366946112163e-01),
        (7.60000000000000053e+00,
         2.51601833849976197e-01),
        (7.65000000000000036e+00,
         2.43356772659918202e-01),
        (7.70000000000000018e+00,
         2.34559139586464377e-01),
        (7.75000000000000000e+00,
         2.25234069120106684e-01),
        (7.80000000000000071e+00,
         2.15407807746262753e-01),
        (7.85000000000000053e+00,
         2.05107645401739064e-01),
        (7.90000000000000036e+00,
         1.94361844841278192e-01),
        (7.95000000000000018e+00,
         1.83199569087022313e-01),
        (8.00000000000000000e+00,
         1.71650807137553901e-01)
    ];
    /// [8, 16] band where the power series visibly degrades (measured max
    /// 1.48e-11 at x=15.65): pinned to stay BOUNDED, catching a broken
    /// termination gate (e.g. its running away or rounding off early).
    const J0_BEYOND_CLAIM: &[(f64, f64)] = &[
        (8.00000000000000000e+00,
         1.71650807137553901e-01),
        (8.50000000000000000e+00,
         4.19392518429344899e-02),
        (9.00000000000000000e+00,
         -9.03336111828759170e-02),
        (9.50000000000000000e+00,
         -1.93928747687422270e-01),
        (1.00000000000000000e+01,
         -2.45935764451348293e-01),
        (1.05000000000000000e+01,
         -2.36648194462347111e-01),
        (1.10000000000000000e+01,
         -1.71190300407196072e-01),
        (1.15000000000000000e+01,
         -6.76539481116651964e-02),
        (1.20000000000000000e+01,
         4.76893107968335492e-02),
        (1.25000000000000000e+01,
         1.46884054700421041e-01),
        (1.30000000000000000e+01,
         2.06926102377067850e-01),
        (1.35000000000000000e+01,
         2.14989165880400818e-01),
        (1.40000000000000000e+01,
         1.71073476110458644e-01),
        (1.45000000000000000e+01,
         8.75448680103762394e-02),
        (1.50000000000000000e+01,
         -1.42244728267807916e-02),
        (1.55000000000000000e+01,
         -1.09230650900050172e-01),
        (1.60000000000000000e+01,
         -1.74899073983629194e-01)
    ];

    #[test]
    fn erf_approx_tracks_the_reference_grid() {
        assert!(!ERF_GRID.is_empty());
        let mut worst: (f64, f64) = (0.0, 0.0);
        for &(x, y_ref) in ERF_GRID {
            let v = erf_approx(x);
            let err = (v - y_ref).abs();
            if err > worst.0 { worst = (err, x); }
            assert!(err <= 1.45e-7, "erf_approx({x}) = {v} vs erf({y_ref:.17e}) err {err}");
        }
        eprintln!("erf_approx max grid deviation {:.3e} at x={}", worst.0, worst.1);
    }

    #[test]
    fn bessel_j0_tracks_the_reference_grid() {
        assert!(!J0_GRID.is_empty());
        let mut worst: (f64, f64) = (0.0, 0.0);
        for &(x, y_ref) in J0_GRID {
            let v = bessel_j0(x);
            let err = (v - y_ref).abs();
            if err > worst.0 { worst = (err, x); }
            assert!(err <= 1e-12, "bessel_j0({x}) = {v} vs j0({y_ref:.17e}) err {err}");
        }
        for &(x, y_ref) in J0_BEYOND_CLAIM {
            let err = (bessel_j0(x) - y_ref).abs();
            assert!(err <= 2e-11, "bessel_j0 beyond-claim band broke at x={x} err {err}");
        }
        eprintln!("bessel_j0 max grid deviation {:.3e} at x={}", worst.0, worst.1);
    }

    /// Deterministic dense matrices (numpy RandomState(20260101), U[-3, 3]
    /// rounded to 12 dp) with numpy/LAPACK singular values as reference.
    /// Embedded, never regenerated in Rust, so the two readers cannot drift.
    const MATS: &[f64] = &[
        1.10179287811900006e+00,
        -7.77845749442000023e-01,
        -5.37803325683000044e-01,
        -1.09783171531400003e+00,
        -4.11732891328000006e-01,
        -1.09750473950600003e+00,
        5.74367128438999952e-01,
        -1.28451352292100007e+00,
        8.67741970550999997e-01,
        1.60573600060200006e+00,
        1.84191303894899994e+00,
        -1.44607687397399998e+00,
        1.21291050590000005e+00,
        -2.05521301253200006e+00,
        2.89309178090099994e+00,
        -2.43063091190800007e+00,
        -5.90390659720000044e-01,
        -2.01568474633200001e+00,
        3.01121558068000006e-01,
        -2.08141250237000003e-01,
        -2.00066803118799985e+00,
        6.55604572138000052e-01,
        -8.16769601388000033e-01,
        2.32515826351100019e+00,
        -1.65598670593800001e+00,
        -7.40995628405999973e-01,
        5.70082710971000051e-01,
        1.55583989773000003e-01,
        9.18881544267999950e-01,
        -1.71412628783700005e+00,
        1.89483252124900003e+00,
        3.83032484103999982e-01,
        1.20900600981600004e+00,
        1.08985398531899991e+00,
        1.62673209923099993e+00,
        -2.08353857681299992e+00,
        2.69634064584599997e+00,
        -2.55155570110800012e+00,
        1.73191515211599989e+00,
        -1.78082347322999990e-01,
        2.57195926124299978e+00,
        2.93389817658100016e+00,
        -9.31625874946999999e-01,
        1.73521881471000006e+00,
        -1.33969435376499990e+00,
        -1.58948492293400001e+00,
        -2.92149531224999981e-01,
        -1.87814640447899994e+00,
        2.94839191660499989e+00,
        5.95394414518000015e-01,
        2.09209958995400003e+00,
        -9.70481942956999966e-01,
        9.58825333150000042e-02,
        -2.29887660189999998e+00,
        1.93426048680400009e+00,
        1.57368523638600011e+00,
        1.12262458690300004e+00,
        -1.45488636836499996e+00,
        1.42782696448099999e+00,
        1.12685629083699990e+00,
        1.09586677819100009e+00,
        -2.80344372273299980e+00,
        -1.43231608650999986e-01,
        -2.94881572633999989e+00,
        2.42195342141399994e+00,
        1.64437316015199997e+00,
        2.83990008115799997e+00,
        -1.97008422751700008e+00,
        -1.51227801231799996e+00,
        -7.97542002593999988e-01,
        -2.52106472325300013e+00,
        -2.24729423337200007e+00,
        -7.87156755614999981e-01,
        -2.84378875585800017e+00,
        -4.54143782331999979e-01,
        1.21827565234800006e+00,
        2.24597767098899981e+00,
        8.98319152427999978e-01,
        -1.06809624259899993e+00,
        4.59428401019999988e-01,
        1.39292547844800008e+00,
        2.52946727734000021e+00,
        2.20797188218000012e-01,
        7.98983169787999992e-01,
        -7.26497201257999992e-01,
        3.31153189773999990e-01,
        2.47942221097999987e-01,
        3.41336455074000011e-01,
        -8.53699013956000008e-01,
        -2.71063392761399991e+00,
        -2.40490490248400013e+00,
        2.36782807824699981e+00,
        -2.34142396547900011e+00,
        -2.49382827670299978e+00,
        -2.00164059568799990e+00,
        -2.73549540083299991e+00,
        7.71726553505999946e-01,
        -1.72936231027899989e+00,
        2.71738027167099983e+00,
        -2.98757165699000016e-01,
        -2.14318376500200003e+00,
        2.58731979609699980e+00,
        2.47793866858499978e+00,
        2.97849100802599986e+00,
        3.51927634145000001e-01,
        2.87214174469700012e+00,
        -2.52860944216799988e+00,
        -5.01950866750000002e-02,
        -1.11618815580000003e-01,
        -1.50463018102800006e+00,
        -1.76781869558900007e+00,
        7.91768591866000016e-01,
        1.53829653433999991e+00,
        -1.79794360257199992e+00,
        -1.49412045816300010e+00,
        -2.06209296934600017e+00,
        7.43110391155999972e-01,
        2.68955436958299998e+00,
        1.18526633769299994e+00,
        2.17634650644400018e+00,
        -1.06783066491899992e+00,
        1.60286192360400004e+00,
        -1.99497057203799999e+00,
        2.29916689757400006e+00,
        -1.72723084254599990e+00,
        -2.86165845510199990e+00,
        -1.75290375611700000e+00,
        2.73074167371499987e+00,
        5.39321695128999967e-01,
        8.70033645792000043e-01,
        1.11066666260999991e+00,
        1.83085523015599994e+00,
        2.34122773948800011e+00,
        -1.36329724551000009e+00,
        2.04891416228299983e+00,
        -1.54561994354300003e+00,
        -2.25489954667999992e+00,
        1.60674006643800005e+00,
        -8.69601063479000014e-01,
        -1.03654074831100007e+00,
        -2.15873348676699983e+00,
        2.30668971997999989e-01,
        -2.12055462996399990e+00,
        2.03397877432100005e+00,
        1.79038989516100000e+00,
        2.19961288212000006e-01,
        -2.91877261550700018e+00,
        -3.44556951062999994e-01,
        -1.74975695078299998e+00,
        2.68858419427699991e+00,
        1.51841611484400008e+00,
        -2.70819319869100017e+00,
        5.54788960289999966e-02,
        1.01712042994099994e+00,
        1.27746485543900001e+00,
        1.16247709612900008e+00,
        -2.87039733484800008e+00,
        -8.94158349115000028e-01,
        9.70730846945000003e-01,
        2.23653807364400015e+00,
        -4.87217021550000018e-01,
        7.67498471126000004e-01,
        1.14883599455800001e+00,
        -2.40362727130999998e+00,
        3.30232500980000010e-02,
        2.42974749645799992e+00,
        2.62547425549000013e-01,
        -9.10832981859999946e-01,
        -1.12549761959999994e+00,
        2.75539949131999995e+00,
        -1.69094936003799989e+00,
        1.94828983650099996e+00,
        2.10363461084999992e+00,
        1.03451222539400001e+00,
        -8.14180479000999990e-01,
        -2.45087304316400001e+00,
        1.88427746596200008e+00,
        -1.64482954139000004e+00,
        -1.81545575929500003e+00,
        -2.37083806811999986e+00,
        2.86133564964999998e+00,
        -1.33496380505200007e+00,
        -4.99285456422999985e-01,
        -2.82399068634599981e+00,
        7.47409614931999977e-01,
        1.54702562752500006e+00,
        2.21532388550400006e+00,
        1.75567571330899996e+00,
        -1.47498135751200010e+00,
        -3.81553271563999985e-01,
        -4.72845025733000024e-01,
        -1.55449713313000004e-01,
        -7.51429502222999979e-01,
        1.83702948241300001e+00,
        -2.32291079612899987e+00,
        -5.12387914317000037e-01,
        2.71493903344700005e+00,
        -4.49916025203000003e-01,
        1.20852381506699991e+00,
        -2.02570456507000007e-01,
        -2.15255831286000010e-01,
        -2.98275678877799999e+00,
        2.82514383634000010e+00,
        3.85156375087000014e-01,
        1.46435041862599991e+00,
        1.61914714000799997e+00,
        -3.23830474782000000e-01,
        9.84586509376000008e-01,
        -1.27533747020499999e+00,
        1.05016389202200000e+00,
        -1.15281054238400005e+00,
        2.44277838103699985e+00,
        9.13706397422999972e-01,
        -1.06959753304599992e+00,

    ];
    const SIGMAS: &[f64] = &[
        4.54678310664618301e+00,
        2.58403531333034175e+00,
        1.48225342616509215e+00,
        5.91894276248138684e+00,
        3.55428814794760939e+00,
        3.03533316909464679e+00,
        1.28713195419258630e+00,
        7.76748735164001669e+00,
        6.60255331351550812e+00,
        3.62049939820380740e+00,
        3.03026125031437621e+00,
        1.85035157742011891e+00,
        2.96936904976749361e+00,
        1.08579992830928940e+00,
        2.36597878903852921e-01,
        6.03638350727299144e+00,
        5.34477906986479478e+00,
        7.31153454237850831e+00,
        6.13582499401461146e+00,
        4.22513129905785689e+00,
        3.84262609792106558e+00,
        7.57779777542142696e+00,
        6.87828630731245561e+00,
        5.50768757604846915e+00,
        4.72771831125260533e+00,
        3.08473898952475345e+00,
        2.35006028169376613e+00,
        3.91296732954948601e+00,
        3.34109333241731710e+00,
        2.79798357272608156e+00,
        1.45471429689657095e+00,

    ];
    /// (rows, cols, matrix_start, sigma_start)
    const SHAPES: &[(usize, usize, usize, usize)] = &[
        (5, 3, 0, 0),
        (6, 4, 15, 3),
        (8, 5, 39, 7),
        (3, 3, 79, 12),
        (7, 2, 88, 15),
        (9, 4, 102, 17),
        (10, 6, 138, 21),
        (4, 4, 198, 27)
    ];

    #[test]
    fn jacobi_singular_values_match_a_numpy_reference_grid() {
        for &(m, n, moff, soff) in SHAPES {
            let mat = &MATS[moff..moff + m * n];
            let s = singular_values(mat, m, n);
            let refs = &SIGMAS[soff..soff + n];
            assert_eq!(s.len(), refs.len(), "(m, n) = ({m}, {n}) count");
            let smax = refs[0];
            for (got, want) in s.iter().zip(refs.iter()) {
                let tol = 1e-10 * smax.max(1.0);
                assert!(
                    (got - want).abs() <= tol,
                    "Jacobi sigma {got} vs numpy {want} on ({m}, {n})"
                );
            }
        }
    }
}
