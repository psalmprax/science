//! Active singularity & pole stress testing engine in Rust.

use serde::{Deserialize, Serialize};
use crate::bytecode::CompiledExpression;
use crate::vm::ExpressionVM;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SingularityReport {
    pub is_sound: bool,
    pub has_pole: bool,
    pub pole_coordinate: Option<f64>,
    pub max_derivative: f64,
    pub has_nan_inf: bool,
    pub risk_type: String,
}

pub struct SingularityProber;

impl SingularityProber {
    /// Dense grid evaluation across domain bounds to actively detect unsampled poles.
    pub fn probe_expression_grid(
        expr: &CompiledExpression,
        domain_bounds: &[(f64, f64)],
        num_grid_points: usize,
    ) -> SingularityReport {
        if domain_bounds.is_empty() {
            return SingularityReport {
                is_sound: true,
                has_pole: false,
                pole_coordinate: None,
                max_derivative: 0.0,
                has_nan_inf: false,
                risk_type: "NONE".to_string(),
            };
        }

        let vm = ExpressionVM::new();
        let (min_x, max_x) = domain_bounds[0];
        let step = (max_x - min_x) / (num_grid_points as f64).max(1.0);

        let mut prev_y: Option<f64> = None;
        let mut max_deriv: f64 = 0.0;
        let mut detected_pole = false;
        let mut pole_coord: Option<f64> = None;
        let mut has_nan_inf = false;

        let mut vars = vec![0.0; expr.num_variables];
        // Set other variables to mid-domain
        for (i, &(_, _)) in domain_bounds.iter().enumerate().skip(1) {
            if i < vars.len() {
                vars[i] = (domain_bounds[i].0 + domain_bounds[i].1) / 2.0;
            }
        }

        for i in 0..=num_grid_points {
            let x = min_x + (i as f64) * step;
            vars[0] = x;

            let y = vm.evaluate_scalar(expr, &vars);

            if !y.is_finite() || y.abs() > 1e6 {
                detected_pole = true;
                if pole_coord.is_none() {
                    pole_coord = Some(x);
                }
                has_nan_inf = true;
                break;
            }

            if let Some(py) = prev_y {
                let deriv = (y - py).abs() / step;
                if deriv > max_deriv {
                    max_deriv = deriv;
                }
                // Asymptotic sign flip or massive derivative spike indicating a pole
                let sign_flip = py * y < 0.0 && (py.abs() > 50.0 || y.abs() > 50.0);
                if deriv > 1e4 && (sign_flip || py.abs() > 500.0 || y.abs() > 500.0) {
                    detected_pole = true;
                    if pole_coord.is_none() {
                        pole_coord = Some(x);
                    }
                }
            }

            prev_y = Some(y);
        }

        let is_sound = !detected_pole && !has_nan_inf;
        let risk_type = if detected_pole {
            "POLE_IN_DOMAIN".to_string()
        } else if has_nan_inf {
            "DOMAIN_VIOLATION".to_string()
        } else {
            "NONE".to_string()
        };

        SingularityReport {
            is_sound,
            has_pole: detected_pole,
            pole_coordinate: pole_coord,
            max_derivative: max_deriv,
            has_nan_inf,
            risk_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ExpressionCompiler;

    #[test]
    fn test_smooth_function_is_sound() {
        let expr = ExpressionCompiler::compile("x0**2 + 1.0", &["x0".to_string()]).unwrap();
        let rep = SingularityProber::probe_expression_grid(&expr, &[(-5.0, 5.0)], 500);
        assert!(rep.is_sound);
        assert!(!rep.has_pole);
    }

    #[test]
    fn test_pole_detection() {
        let expr = ExpressionCompiler::compile("1.0 / (x0 - 2.5)", &["x0".to_string()]).unwrap();
        let rep = SingularityProber::probe_expression_grid(&expr, &[(0.0, 5.0)], 1000);
        assert!(!rep.is_sound);
        assert!(rep.has_pole);
    }
}
