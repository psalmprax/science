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