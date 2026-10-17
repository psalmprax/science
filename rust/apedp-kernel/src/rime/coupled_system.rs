//! Coupled Multi-Variable Dynamical System Discovery (SINDy-style Jet System).
//!
//! Extends RIME from scalar ODEs and PDEs to coupled multi-variable
//! non-linear dynamical systems:
//!     \dot{\mathbf{x}} = \mathbf{F}(\mathbf{x})
//! such as Lotka-Volterra, FitzHugh-Nagumo, coupled Hamiltonian phase spaces,
//! and Lorenz attractors.

use serde::{Deserialize, Serialize};
use super::pde::{run_stridge, PdeConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledConfig {
    pub max_polynomial_degree: usize,
    pub include_cross_interactions: bool,
    pub sparsity_threshold: f64,
    pub ridge_alpha: f64,
}

impl Default for CoupledConfig {
    fn default() -> CoupledConfig {
        CoupledConfig {
            max_polynomial_degree: 2,
            include_cross_interactions: true,
            sparsity_threshold: 0.02,
            ridge_alpha: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledEquation {
    pub state_variable: String,
    pub formula: String,
    pub active_terms: Vec<String>,
    pub coefficients: Vec<f64>,
    pub r2: f64,
    pub residual_rmse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledSystemResult {
    pub equations: Vec<CoupledEquation>,
    pub mean_system_r2: f64,
    pub worst_state_r2: f64,
    pub is_exact_recovery: bool,
}

/// Discovers coupled autonomous non-linear ODE system \dot{\mathbf{x}} = F(\mathbf{x}).
/// `series` has shape D (state variables) x N (time steps).
pub fn discover_coupled_system(
    series: &[Vec<f64>],
    dt: f64,
    labels: Option<&[String]>,
    config: &CoupledConfig,
) -> Result<CoupledSystemResult, String> {
    let d = series.len();
    if d < 2 {
        return Err("Coupled system requires at least 2 state variables".to_string());
    }
    let n = series[0].len();
    if n < 7 {
        return Err("Coupled system requires at least 7 time points".to_string());
    }
    for s in series {
        if s.len() != n {
            return Err("Inconsistent series lengths in coupled system".to_string());
        }
    }

    let default_labels: Vec<String> = (0..d).map(|i| format!("x{}", i)).collect();
    let state_labels = labels.unwrap_or(&default_labels);

    let interior_n = n - 4;
    let dt = dt.max(1e-12);

    // Compute 4th-order central time derivatives for each state variable on interior
    let mut d_dt_all = Vec::with_capacity(d);
    let mut state_interior = Vec::with_capacity(d);

    for s in series {
        let mut d_dt = Vec::with_capacity(interior_n);
        let mut vals = Vec::with_capacity(interior_n);
        for i in 2..(n - 2) {
            let deriv = (-s[i + 2] + 8.0 * s[i + 1] - 8.0 * s[i - 1] + s[i - 2]) / (12.0 * dt);
            d_dt.push(deriv);
            vals.push(s[i]);
        }
        d_dt_all.push(d_dt);
        state_interior.push(vals);
    }

    // Build multi-variable polynomial library Theta(x1, x2, ..., xD)
    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    // 1. Constant term 1
    term_names.push("1".to_string());
    columns.push(vec![1.0f64; interior_n]);

    // 2. Linear terms x_i
    for (i, lbl) in state_labels.iter().enumerate() {
        term_names.push(lbl.clone());
        columns.push(state_interior[i].clone());
    }

    // 3. Quadratic & cross-interaction terms: x_i * x_j
    if config.max_polynomial_degree >= 2 {
        for i in 0..d {
            let start_j = if config.include_cross_interactions { i } else { i };
            let end_j = if config.include_cross_interactions { d } else { i + 1 };
            for j in start_j..end_j {
                let name = if i == j {
                    format!("{}^2", state_labels[i])
                } else {
                    format!("{}*{}", state_labels[i], state_labels[j])
                };
                term_names.push(name);
                let col: Vec<f64> = (0..interior_n)
                    .map(|idx| state_interior[i][idx] * state_interior[j][idx])
                    .collect();
                columns.push(col);
            }
        }
    }

    // 4. Cubic self-powers x_i^3
    if config.max_polynomial_degree >= 3 {
        for i in 0..d {
            term_names.push(format!("{}^3", state_labels[i]));
            let col: Vec<f64> = (0..interior_n)
                .map(|idx| state_interior[i][idx].powi(3))
                .collect();
            columns.push(col);
        }
    }

    let pde_cfg = PdeConfig {
        max_spatial_order: 1,
        max_polynomial_degree: config.max_polynomial_degree,
        include_advection_terms: false,
        sparsity_threshold: config.sparsity_threshold,
        ridge_alpha: config.ridge_alpha,
    };

    let mut equations = Vec::with_capacity(d);
    let mut sum_r2 = 0.0;
    let mut worst_r2 = f64::INFINITY;

    for i in 0..d {
        let target_label = format!("d{}/dt", state_labels[i]);
        let res = run_stridge(&d_dt_all[i], &columns, &term_names, &pde_cfg, &target_label)?;

        sum_r2 += res.r2;
        if res.r2 < worst_r2 {
            worst_r2 = res.r2;
        }

        equations.push(CoupledEquation {
            state_variable: state_labels[i].clone(),
            formula: res.formula,
            active_terms: res.active_terms,
            coefficients: res.coefficients,
            r2: res.r2,
            residual_rmse: res.residual_rmse,
        });
    }

    let mean_system_r2 = sum_r2 / (d as f64);
    let is_exact = worst_r2 > 0.99;

    Ok(CoupledSystemResult {
        equations,
        mean_system_r2,
        worst_state_r2: worst_r2,
        is_exact_recovery: is_exact,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lotka_volterra_coupled_recovery() {
        // Lotka-Volterra Predator-Prey System:
        // dx/dt = 1.5 * x - 1.0 * x * y
        // dy/dt = -3.0 * y + 1.0 * x * y
        let dt = 0.005;
        let n = 200;
        let mut x = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];

        x[0] = 2.0;
        y[0] = 1.0;

        // Forward RK4 simulation
        for i in 0..(n - 1) {
            let f = |cx: f64, cy: f64| -> (f64, f64) {
                (1.5 * cx - 1.0 * cx * cy, -3.0 * cy + 1.0 * cx * cy)
            };
            let (k1_x, k1_y) = f(x[i], y[i]);
            let (k2_x, k2_y) = f(x[i] + 0.5 * dt * k1_x, y[i] + 0.5 * dt * k1_y);
            let (k3_x, k3_y) = f(x[i] + 0.5 * dt * k2_x, y[i] + 0.5 * dt * k2_y);
            let (k4_x, k4_y) = f(x[i] + dt * k3_x, y[i] + dt * k3_y);

            x[i + 1] = x[i] + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
            y[i + 1] = y[i] + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y);
        }

        let series = vec![x, y];
        let labels = vec!["x".to_string(), "y".to_string()];
        let cfg = CoupledConfig::default();

        let res = discover_coupled_system(&series, dt, Some(&labels), &cfg)
            .expect("Coupled discovery must succeed");

        assert!(res.is_exact_recovery, "System R2 = {}", res.mean_system_r2);
        assert_eq!(res.equations.len(), 2);

        // Equation 1: dx/dt = 1.5*x - 1.0*x*y
        let eq_x = &res.equations[0];
        assert!(eq_x.active_terms.contains(&"x".to_string()));
        assert!(eq_x.active_terms.contains(&"x*y".to_string()));
        let x_idx = eq_x.active_terms.iter().position(|t| t == "x").unwrap();
        let xy_idx = eq_x.active_terms.iter().position(|t| t == "x*y").unwrap();
        assert!((eq_x.coefficients[x_idx] - 1.5).abs() < 0.05);
        assert!((eq_x.coefficients[xy_idx] - (-1.0)).abs() < 0.05);

        // Equation 2: dy/dt = -3.0*y + 1.0*x*y
        let eq_y = &res.equations[1];
        assert!(eq_y.active_terms.contains(&"y".to_string()));
        assert!(eq_y.active_terms.contains(&"x*y".to_string()));
        let y_idx = eq_y.active_terms.iter().position(|t| t == "y").unwrap();
        let y_xy_idx = eq_y.active_terms.iter().position(|t| t == "x*y").unwrap();
        assert!((eq_y.coefficients[y_idx] - (-3.0)).abs() < 0.05);
        assert!((eq_y.coefficients[y_xy_idx] - 1.0).abs() < 0.05);
    }
}
