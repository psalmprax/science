//! High-performance Linear Least Squares & Variable Projection (VarPro) linear solver using Faer.

use faer::Mat;
use faer::prelude::SolveLstsq;
use serde::{Deserialize, Serialize};
use crate::conditioning::MatrixConditioner;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearFitResult {
    pub coefficients: Vec<f64>,
    pub mse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub condition_number: f64,
    pub is_well_conditioned: bool,
}

pub struct VarProSolver;

impl VarProSolver {
    /// Solves linear least squares min_c ||A c - y||^2 + lambda ||c||^2 via QR.
    /// basis_cols: slice of K column slices, each of length N.
    /// target: target slice of length N.
    pub fn solve_linear_coefficients(
        basis_cols: &[&[f64]],
        target: &[f64],
        lambda: f64,
    ) -> Result<LinearFitResult, String> {
        let k = basis_cols.len();
        if k == 0 {
            return Err("No basis functions provided".to_string());
        }
        let n = target.len();
        if n == 0 {
            return Err("Target array is empty".to_string());
        }
        for (idx, col) in basis_cols.iter().enumerate() {
            if col.len() != n {
                return Err(format!("Basis column {} length {} != {}", idx, col.len(), n));
            }
        }

        // 1. Condition number diagnostics
        let cond_rep = MatrixConditioner::compute_conditioning(basis_cols, 1e-12)?;

        // 2. Build design matrix A (N x K) and target vector y (N x 1)
        let mut a = Mat::<f64>::zeros(n, k);
        let mut y = Mat::<f64>::zeros(n, 1);

        for i in 0..n {
            y[(i, 0)] = target[i];
            for j in 0..k {
                a[(i, j)] = basis_cols[j][i];
            }
        }

        // 3. Solve via column-pivoted QR (with augmented regularization if lambda > 0)
        let c_mat = if lambda > 1e-15 {
            let mut a_aug = Mat::<f64>::zeros(n + k, k);
            let mut y_aug = Mat::<f64>::zeros(n + k, 1);
            for i in 0..n {
                y_aug[(i, 0)] = target[i];
                for j in 0..k {
                    a_aug[(i, j)] = basis_cols[j][i];
                }
            }
            let sqrt_lambda = lambda.sqrt();
            for j in 0..k {
                a_aug[(n + j, j)] = sqrt_lambda;
            }
            let qr = a_aug.col_piv_qr();
            qr.solve_lstsq(&y_aug)
        } else {
            let qr = a.col_piv_qr();
            qr.solve_lstsq(&y)
        };

        let mut coeffs = vec![0.0; k];
        for j in 0..k {
            coeffs[j] = c_mat[(j, 0)];
        }

        // 4. Compute residuals and R^2
        let y_pred = &a * &c_mat;
        let mut sum_sq_err = 0.0;
        let mut sum_abs_err = 0.0;
        let mut sum_y = 0.0;

        for i in 0..n {
            let yt = target[i];
            let yp = y_pred[(i, 0)];
            let err = yt - yp;
            sum_sq_err += err * err;
            sum_abs_err += err.abs();
            sum_y += yt;
        }

        let mse = sum_sq_err / (n as f64);
        let mae = sum_abs_err / (n as f64);
        let mean_y = sum_y / (n as f64);

        let mut total_var = 0.0;
        for i in 0..n {
            let diff = target[i] - mean_y;
            total_var += diff * diff;
        }

        let r2 = if total_var > 1e-15 {
            1.0 - (sum_sq_err / total_var)
        } else if mse < 1e-15 {
            1.0
        } else {
            0.0
        };

        Ok(LinearFitResult {
            coefficients: coeffs,
            mse,
            mae,
            r_squared: r2,
            condition_number: cond_rep.condition_number,
            is_well_conditioned: cond_rep.is_well_conditioned,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varpro_linear_fit_exact() {
        // Target: y = 2.0 * x0 - 3.5 * x1 + 1.0
        let n = 100;
        let mut x0 = vec![0.0; n];
        let mut x1 = vec![0.0; n];
        let x_const = vec![1.0; n];
        let mut y = vec![0.0; n];

        for i in 0..n {
            x0[i] = i as f64 * 0.1;
            x1[i] = (i as f64 * 0.05).sin();
            y[i] = 2.0 * x0[i] - 3.5 * x1[i] + 1.0;
        }

        let res = VarProSolver::solve_linear_coefficients(
            &[&x0, &x1, &x_const],
            &y,
            1e-8,
        ).unwrap();

        assert!(res.mse < 1e-10, "MSE too high: {}", res.mse);
        assert!((res.r_squared - 1.0).abs() < 1e-8, "R2 not 1.0: {}", res.r_squared);
        assert!((res.coefficients[0] - 2.0).abs() < 1e-4);
        assert!((res.coefficients[1] - (-3.5)).abs() < 1e-4);
        assert!((res.coefficients[2] - 1.0).abs() < 1e-4);
        assert!(res.is_well_conditioned);
    }
}
