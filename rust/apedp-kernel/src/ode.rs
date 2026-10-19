//! High-performance native ODE integrator using Dormand-Prince 5(4) (RK45).
//!
//! Provides adaptive step-size integration for general first-order ODE systems:
//!     dy/dt = f(t, y)
//! where y in R^d.
//!
//! Evaluates compiled RHS expressions directly in native Rust without Python GIL
//! or NumPy call overhead during numerical integration steps.

use crate::parser::{self, Compiled};

/// Butcher tableau coefficients for Dormand-Prince 5(4).
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
const C6: f64 = 1.0;

// Runge-Kutta matrix A: a_{ij}
const A21: f64 = 1.0 / 5.0;

const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;

const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;

const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;

const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

// Order 5 weights (b_i)
const B1: f64 = 35.0 / 384.0;
const B2: f64 = 0.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;
// B7 is 0.0 for 5th order step

// Error estimation coefficients e_i = b_i - \hat{b}_i
const E1: f64 = 71.0 / 57600.0;
const E2: f64 = 0.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// Output of an ODE integration run.
#[derive(Debug, Clone)]
pub struct OdeResult {
    pub success: bool,
    pub detail: String,
    pub nfev: usize,
    pub t: Vec<f64>,
    /// State trajectories: outer length = num_states, inner length = num_eval_points.
    pub y: Vec<Vec<f64>>,
}

/// A compiled system of ordinary differential equations.
#[derive(Clone)]
pub struct OdeSystem {
    exprs: Vec<Compiled>,
    pub num_states: usize,
    pub num_params: usize,
    pub num_inputs: usize,
    pub state_names: Vec<String>,
    pub param_names: Vec<String>,
    pub input_names: Vec<String>,
    is_mechanical_2nd_order: bool,
}

impl OdeSystem {
    /// Compile a system of ODEs given equation strings, state names, parameter names, and optional input names.
    ///
    /// Variable resolution layout:
    /// `["t", state_0, ..., state_{d-1}, param_0, ..., param_{P-1}, input_0, ..., input_{M-1}]`
    pub fn compile(
        equations: &[String],
        state_names: &[String],
        param_names: &[String],
        input_names: &[String],
    ) -> Result<Self, String> {
        if equations.len() != state_names.len() {
            return Err(format!(
                "ODE arity mismatch: {} equation(s) provided for {} state(s)",
                equations.len(),
                state_names.len()
            ));
        }
        if state_names.is_empty() {
            return Err("ODE system must have at least one state variable".to_string());
        }

        let mut var_table = Vec::with_capacity(1 + state_names.len() + param_names.len() + input_names.len());
        var_table.push("t".to_string());
        var_table.extend_from_slice(state_names);
        var_table.extend_from_slice(param_names);
        var_table.extend_from_slice(input_names);

        let mut exprs = Vec::with_capacity(equations.len());
        for (i, eq) in equations.iter().enumerate() {
            let compiled = parser::compile(eq, &var_table)
                .map_err(|e| format!("failed to compile ODE equation for state '{}': {e}", state_names[i]))?;
            exprs.push(compiled);
        }

        let is_mechanical = state_names.len() == 2
            && (state_names[0] == "x" || state_names[0] == "q")
            && (state_names[1] == "v" || state_names[1] == "dq")
            && equations[0].trim() == state_names[1];

        Ok(Self {
            exprs,
            num_states: state_names.len(),
            num_params: param_names.len(),
            num_inputs: input_names.len(),
            state_names: state_names.to_vec(),
            param_names: param_names.to_vec(),
            input_names: input_names.to_vec(),
            is_mechanical_2nd_order: is_mechanical,
        })
    }

    /// Integrate the ODE system over `t_span` from initial condition `x0`, evaluating at `t_eval`.
    pub fn integrate(
        &self,
        t_span: (f64, f64),
        x0: &[f64],
        t_eval: &[f64],
        params: &[f64],
        inputs: &[f64],
        rtol: f64,
        atol: f64,
        max_steps: usize,
    ) -> OdeResult {
        let d = self.num_states;
        let empty_res = |msg: &str| OdeResult {
            success: false,
            detail: msg.to_string(),
            nfev: 0,
            t: t_eval.to_vec(),
            y: vec![Vec::new(); d],
        };

        if x0.len() != d {
            return empty_res(&format!("initial state x0 length {} != state count {}", x0.len(), d));
        }
        if params.len() != self.num_params {
            return empty_res(&format!("param count {} != expected {}", params.len(), self.num_params));
        }
        if inputs.len() != self.num_inputs {
            return empty_res(&format!("input count {} != expected {}", inputs.len(), self.num_inputs));
        }
        if t_eval.is_empty() {
            return empty_res("t_eval must not be empty");
        }
        if t_span.1 <= t_span.0 {
            return empty_res("t_span[1] must be strictly greater than t_span[0]");
        }

        let n_eval = t_eval.len();
        let mut y_out = vec![vec![0.0f64; n_eval]; d];

        // Prepare point buffer for expression evaluation
        let mut point = vec![0.0f64; 1 + d + self.num_params + self.num_inputs];
        point[1 + d..1 + d + self.num_params].copy_from_slice(params);
        point[1 + d + self.num_params..].copy_from_slice(inputs);

        let is_mech = self.is_mechanical_2nd_order;
        let exprs = &self.exprs;

        let mut eval_rhs = |t: f64, y: &[f64], out: &mut [f64]| -> bool {
            point[0] = t;
            point[1..1 + d].copy_from_slice(y);
            if is_mech {
                out[0] = y[1];
                out[1] = exprs[1].eval_point(&point);
                out[0].is_finite() && out[1].is_finite()
            } else {
                for i in 0..d {
                    out[i] = exprs[i].eval_point(&point);
                    if !out[i].is_finite() {
                        return false;
                    }
                }
                true
            }
        };

        let mut t_n = t_span.0;
        let t_end = t_span.1;
        let mut y_n = x0.to_vec();

        let mut k1 = vec![0.0f64; d];
        let mut k2 = vec![0.0f64; d];
        let mut k3 = vec![0.0f64; d];
        let mut k4 = vec![0.0f64; d];
        let mut k5 = vec![0.0f64; d];
        let mut k6 = vec![0.0f64; d];
        let mut k7 = vec![0.0f64; d];

        let mut y_tmp = vec![0.0f64; d];
        let mut y_next = vec![0.0f64; d];

        let mut nfev = 0usize;

        // Initial evaluation
        if !eval_rhs(t_n, &y_n, &mut k1) {
            return OdeResult {
                success: false,
                detail: "non-finite derivative at initial condition".to_string(),
                nfev: 1,
                t: t_eval.to_vec(),
                y: y_out,
            };
        }
        nfev += 1;

        let mut eval_idx = 0usize;
        while eval_idx < n_eval && t_eval[eval_idx] <= t_n + 1e-14 {
            for j in 0..d {
                y_out[j][eval_idx] = y_n[j];
            }
            eval_idx += 1;
        }

        // Initial step size heuristic
        let span_len = t_end - t_n;
        let mut h = (span_len / 100.0).clamp(1e-6, 1.0);
        let min_h = 10.0 * f64::EPSILON * span_len.max(1.0);
        let max_h = span_len;

        let mut step_count = 0usize;

        while t_n < t_end - 1e-14 && step_count < max_steps {
            step_count += 1;

            if t_n + h > t_end {
                h = t_end - t_n;
            }

            // Stage 2
            let t2 = t_n + C2 * h;
            for j in 0..d {
                y_tmp[j] = y_n[j] + h * (A21 * k1[j]);
            }
            if !eval_rhs(t2, &y_tmp, &mut k2) {
                h *= 0.2;
                if h < min_h {
                    return OdeResult {
                        success: false,
                        detail: "step size underflow: non-finite derivative".to_string(),
                        nfev,
                        t: t_eval.to_vec(),
                        y: y_out,
                    };
                }
                continue;
            }

            // Stage 3
            let t3 = t_n + C3 * h;
            for j in 0..d {
                y_tmp[j] = y_n[j] + h * (A31 * k1[j] + A32 * k2[j]);
            }
            if !eval_rhs(t3, &y_tmp, &mut k3) {
                h *= 0.2;
                if h < min_h {
                    return OdeResult {
                        success: false,
                        detail: "step size underflow: non-finite derivative".to_string(),
                        nfev,
                        t: t_eval.to_vec(),
                        y: y_out,
                    };
                }
                continue;
            }

            // Stage 4
            let t4 = t_n + C4 * h;
            for j in 0..d {
                y_tmp[j] = y_n[j] + h * (A41 * k1[j] + A42 * k2[j] + A43 * k3[j]);
            }
            if !eval_rhs(t4, &y_tmp, &mut k4) {
                h *= 0.2;
                if h < min_h {
                    return OdeResult {
                        success: false,
                        detail: "step size underflow: non-finite derivative".to_string(),
                        nfev,
                        t: t_eval.to_vec(),
                        y: y_out,
                    };
                }
                continue;
            }

            // Stage 5
            let t5 = t_n + C5 * h;
            for j in 0..d {
                y_tmp[j] = y_n[j] + h * (A51 * k1[j] + A52 * k2[j] + A53 * k3[j] + A54 * k4[j]);
            }
            if !eval_rhs(t5, &y_tmp, &mut k5) {
                h *= 0.2;
                if h < min_h {
                    return OdeResult {
                        success: false,
                        detail: "step size underflow: non-finite derivative".to_string(),
                        nfev,
                        t: t_eval.to_vec(),
                        y: y_out,
                    };
                }
                continue;
            }

            // Stage 6
            let t6 = t_n + C6 * h;
            for j in 0..d {
                y_tmp[j] = y_n[j] + h * (A61 * k1[j] + A62 * k2[j] + A63 * k3[j] + A64 * k4[j] + A65 * k5[j]);
            }
            if !eval_rhs(t6, &y_tmp, &mut k6) {
                h *= 0.2;
                if h < min_h {
                    return OdeResult {
                        success: false,
                        detail: "step size underflow: non-finite derivative".to_string(),
                        nfev,
                        t: t_eval.to_vec(),
                        y: y_out,
                    };
                }
                continue;
            }

            // Candidate 5th-order state
            let mut non_finite = false;
            for j in 0..d {
                let y5 = y_n[j] + h * (B1 * k1[j] + B2 * k2[j] + B3 * k3[j] + B4 * k4[j] + B5 * k5[j] + B6 * k6[j]);
                if !y5.is_finite() || y5.abs() > 1e12 {
                    non_finite = true;
                    break;
                }
                y_next[j] = y5;
            }

            if non_finite {
                h *= 0.2;
                if h < min_h {
                    return OdeResult {
                        success: false,
                        detail: "divergence: non-finite state encountered".to_string(),
                        nfev,
                        t: t_eval.to_vec(),
                        y: y_out,
                    };
                }
                continue;
            }

            // Stage 7 (FSAL for accepted steps)
            let t_next = t_n + h;
            if !eval_rhs(t_next, &y_next, &mut k7) {
                h *= 0.2;
                if h < min_h {
                    return OdeResult {
                        success: false,
                        detail: "step size underflow: non-finite derivative".to_string(),
                        nfev,
                        t: t_eval.to_vec(),
                        y: y_out,
                    };
                }
                continue;
            }
            nfev += 6;

            // Error estimation
            let mut err_sq = 0.0f64;
            for j in 0..d {
                let err_est = h * (E1 * k1[j] + E2 * k2[j] + E3 * k3[j] + E4 * k4[j] + E5 * k5[j] + E6 * k6[j] + E7 * k7[j]);
                let scale = atol + rtol * y_n[j].abs().max(y_next[j].abs());
                let ratio = err_est / scale;
                err_sq += ratio * ratio;
            }
            let err = (err_sq / d as f64).sqrt();

            if !err.is_finite() || err > 1.0 {
                // Step rejected
                let factor = if !err.is_finite() {
                    0.2
                } else {
                    (0.9 * err.powf(-0.2)).clamp(0.2, 0.9)
                };
                h *= factor;
                if h < min_h {
                    return OdeResult {
                        success: false,
                        detail: "step size underflow: tolerance could not be satisfied".to_string(),
                        nfev,
                        t: t_eval.to_vec(),
                        y: y_out,
                    };
                }
            } else {
                // Step accepted! Dense Hermite cubic interpolation for any points in (t_n, t_next]
                while eval_idx < n_eval && t_eval[eval_idx] <= t_next + 1e-14 {
                    let t_target = t_eval[eval_idx];
                    let theta = if h > 0.0 {
                        ((t_target - t_n) / h).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    let th2 = theta * theta;
                    let th3 = th2 * theta;
                    let h00 = 1.0 - 3.0 * th2 + 2.0 * th3;
                    let h01 = 3.0 * th2 - 2.0 * th3;
                    let h10 = theta - 2.0 * th2 + th3;
                    let h11 = -th2 + th3;

                    for j in 0..d {
                        y_out[j][eval_idx] = h00 * y_n[j] + h01 * y_next[j] + h * (h10 * k1[j] + h11 * k7[j]);
                    }
                    eval_idx += 1;
                }

                // Advance state
                t_n = t_next;
                y_n.copy_from_slice(&y_next);
                k1.copy_from_slice(&k7); // FSAL property: k7 becomes next step's k1

                // Next step size
                let factor = if err < 1e-10 {
                    2.0
                } else {
                    (0.9 * err.powf(-0.2)).clamp(0.2, 2.0)
                };
                h = (h * factor).clamp(min_h, max_h);
            }
        }

        // Fill in boundary if floating-point edge missed exact equality
        while eval_idx < n_eval && (t_eval[eval_idx] - t_end).abs() < 1e-4 {
            for j in 0..d {
                y_out[j][eval_idx] = y_n[j];
            }
            eval_idx += 1;
        }

        if eval_idx < n_eval {
            return OdeResult {
                success: false,
                detail: format!("span not fully covered (reached t={t_n:.6}, eval_idx={eval_idx}/{n_eval})"),
                nfev,
                t: t_eval.to_vec(),
                y: y_out,
            };
        }

        OdeResult {
            success: true,
            detail: "ok".to_string(),
            nfev,
            t: t_eval.to_vec(),
            y: y_out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonic_oscillator_matches_analytical() {
        // d^2x/dt^2 + omega^2 x = 0
        // dx/dt = v, dv/dt = -omega^2 * x
        // x(t) = cos(omega * t), v(t) = -omega * sin(omega * t)
        let eq = vec!["v".to_string(), "-4.0 * x".to_string()];
        let states = vec!["x".to_string(), "v".to_string()];
        let ode = OdeSystem::compile(&eq, &states, &[], &[]).unwrap();

        let n_eval = 101;
        let mut t_eval = Vec::with_capacity(n_eval);
        for i in 0..n_eval {
            t_eval.push(i as f64 * (2.0 * std::f64::consts::PI / 100.0));
        }

        let res = ode.integrate(
            (0.0, 2.0 * std::f64::consts::PI),
            &[1.0, 0.0],
            &t_eval,
            &[],
            &[],
            1e-8,
            1e-10,
            10_000,
        );

        assert!(res.success, "integration failed: {}", res.detail);
        assert_eq!(res.y.len(), 2);
        assert_eq!(res.y[0].len(), n_eval);

        let omega = 2.0;
        for i in 0..n_eval {
            let t = t_eval[i];
            let exact_x = (omega * t).cos();
            let exact_v = -omega * (omega * t).sin();
            let sim_x = res.y[0][i];
            let sim_v = res.y[1][i];

            assert!(
                (sim_x - exact_x).abs() < 1e-4,
                "x mismatch at t={t}: got {sim_x}, exact {exact_x}"
            );
            assert!(
                (sim_v - exact_v).abs() < 1e-4,
                "v mismatch at t={t}: got {sim_v}, exact {exact_v}"
            );
        }
    }

    #[test]
    fn parameter_and_nonlinear_damping_integration() {
        // dx/dt = v
        // dv/dt = -k * x - gamma * abs(v) * v
        let eq = vec!["v".to_string(), "-k * x - gamma * abs(v) * v".to_string()];
        let states = vec!["x".to_string(), "v".to_string()];
        let params = vec!["k".to_string(), "gamma".to_string()];
        let ode = OdeSystem::compile(&eq, &states, &params, &[]).unwrap();

        let mut t_eval = Vec::new();
        for i in 0..50 {
            t_eval.push(i as f64 * 0.1);
        }

        let res = ode.integrate(
            (0.0, 4.9),
            &[2.0, 0.0],
            &t_eval,
            &[5.0, 0.3],
            &[],
            1e-6,
            1e-8,
            5000,
        );

        assert!(res.success, "failed: {}", res.detail);
        // Energy must decay monotonically due to quadratic drag
        let x_end = res.y[0].last().unwrap().abs();
        assert!(x_end < 2.0, "amplitude should decrease: {x_end}");
    }

    #[test]
    fn divergent_rhs_fails_explicitly_without_burning_the_step_budget() {
        // Regression: stages 3-7 used to shrink h on a non-finite derivative
        // without the min_h check that stages 2 and the candidate-state guard
        // already had, so a divergent system could iterate until max_steps
        // instead of reporting an explicit step-control failure.
        let eq = vec!["1.0e300 * x * x".to_string()];
        let states = vec!["x".to_string()];
        let ode = OdeSystem::compile(&eq, &states, &[], &[]).unwrap();
        let res = ode.integrate(
            (0.0, 1.0),
            &[1.0],
            &[0.0, 0.5, 1.0],
            &[],
            &[],
            1e-8,
            1e-10,
            100_000,
        );
        assert!(!res.success, "divergent system must not succeed");
        assert!(
            res.detail.contains("underflow") || res.detail.contains("divergence"),
            "expected an explicit step-control failure, got: {}",
            res.detail
        );
    }
}
