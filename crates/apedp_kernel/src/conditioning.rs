//! Design matrix conditioning and SVD diagnostics using Faer.

use faer::Mat;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditioningReport {
    pub condition_number: f64,
    pub max_singular_value: f64,
    pub min_singular_value: f64,
    pub effective_rank: usize,
    pub is_well_conditioned: bool,
    pub singular_values: Vec<f64>,
}

pub struct MatrixConditioner;

impl MatrixConditioner {
    /// Computes condition number kappa = sigma_max / sigma_min for design matrix A (N x K).
    /// data: slice of column slices, each of length N (K columns).
    pub fn compute_conditioning(cols: &[&[f64]], rcond_threshold: f64) -> Result<ConditioningReport, String> {
        let k = cols.len();
        if k == 0 {
            return Err("Matrix has 0 columns".to_string());
        }
        let n = cols[0].len();
        if n == 0 {
            return Err("Matrix has 0 rows".to_string());
        }
        for (idx, col) in cols.iter().enumerate() {
            if col.len() != n {
                return Err(format!("Column {} length {} != {}", idx, col.len(), n));
            }
        }

        // Build Faer matrix (N x K)
        let mut mat = Mat::<f64>::zeros(n, k);
        for j in 0..k {
            for i in 0..n {
                mat[(i, j)] = cols[j][i];
            }
        }

        // Compute SVD
        let svd = mat.svd().map_err(|e| format!("SVD computation failed: {:?}", e))?;
        let s_diag = svd.S(); // Diagonal of singular values
        let s_vec = s_diag.column_vector();

        let num_s = s_vec.nrows();
        if num_s == 0 {
            return Ok(ConditioningReport {
                condition_number: f64::INFINITY,
                max_singular_value: 0.0,
                min_singular_value: 0.0,
                effective_rank: 0,
                is_well_conditioned: false,
                singular_values: vec![],
            });
        }

        let sigma_max = s_vec[0];
        let sigma_min = s_vec[num_s - 1];

        let cutoff = sigma_max * rcond_threshold;
        let mut rank = 0;
        for i in 0..num_s {
            if s_vec[i] > cutoff {
                rank += 1;
            }
        }

        let cond = if sigma_min > 1e-15 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        };

        let mut s_vec_all = Vec::with_capacity(num_s);
        for i in 0..num_s {
            s_vec_all.push(s_vec[i]);
        }

        let is_well = cond < (1.0 / rcond_threshold.max(1e-15));

        Ok(ConditioningReport {
            condition_number: cond,
            max_singular_value: sigma_max,
            min_singular_value: sigma_min,
            effective_rank: rank,
            is_well_conditioned: is_well,
            singular_values: s_vec_all,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_conditioning() {
        let col0 = vec![1.0, 0.0, 0.0];
        let col1 = vec![0.0, 1.0, 0.0];
        let col2 = vec![0.0, 0.0, 1.0];
        let rep = MatrixConditioner::compute_conditioning(&[&col0, &col1, &col2], 1e-12).unwrap();
        assert!((rep.condition_number - 1.0).abs() < 1e-6);
        assert_eq!(rep.effective_rank, 3);
        assert!(rep.is_well_conditioned);
    }

    #[test]
    fn test_collinear_conditioning() {
        let col0 = vec![1.0, 2.0, 3.0];
        let col1 = vec![2.0, 4.0, 6.0]; // perfectly collinear
        let rep = MatrixConditioner::compute_conditioning(&[&col0, &col1], 1e-12).unwrap();
        assert!(rep.condition_number > 1e10 || rep.condition_number.is_infinite());
        assert_eq!(rep.effective_rank, 1);
        assert!(!rep.is_well_conditioned);
    }
}
