//! Hybrid High-Dimensional PDE Discovery Engine (1D to 20D+).
//!
//! Solves the curse of dimensionality across continuous spatio-temporal PDE discovery:
//! - For D < 10: Eulerian Cartesian grid finite difference discretization
//!   (1D, 2D, 3D tensor partial derivatives and Cartesian Laplacian \nabla^2_D u = \sum \partial^2_{x_k x_k} u).
//! - For D >= 10 (up to 20D+): Bypasses the exponential N^D grid explosion via 4 high-dimensional methods:
//!   1. Mesh-Free Axis-Star Point Cloud Sampling: Lagrangian / Monte Carlo anchor probe sampling
//!      requiring only M * (2D + 1) points to evaluate the exact D-dimensional Laplacian.
//!   2. Hyperspherical Radial Invariant Reduction: Exact geometric reduction
//!      \nabla^2_D u = u_{rr} + ((D - 1) / r) * u_r, scaling in O(1) memory and time with respect to D.
//!   3. Separable / Tensor Coordinate Factors: Factorized 1D coordinate representations.
//!   4. Mesh-Free STRidge Sparse Regression: Sequentially Thresholded Ridge Regression
//!      directly on high-dimensional observation libraries.

use ndarray::{linalg::general_mat_mul, ArrayView2, ArrayViewMut2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::pde::{run_stridge, solve_linear_system, PdeConfig, PdeResult};

/// Second-order Taylor Jet `(val, d1, d2)` for single-pass exact value, gradient,
/// and curvature/Laplacian propagation without nested autodiff graph allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Jet2 {
    pub val: f64,
    pub d1: f64,
    pub d2: f64,
}

impl Jet2 {
    #[inline(always)]
    pub fn new(val: f64, d1: f64, d2: f64) -> Self {
        Self { val, d1, d2 }
    }

    /// Propagate a 2-jet through `tanh(z)` in a single register pass:
    /// tanh'(z) = 1 - tanh^2(z)
    /// tanh''(z) = -2 * tanh(z) * (1 - tanh^2(z))
    #[inline(always)]
    pub fn tanh(self) -> Self {
        let t = self.val.tanh();
        let dt = 1.0 - t * t;
        let d2t = -2.0 * t * dt;
        Self {
            val: t,
            d1: dt * self.d1,
            d2: d2t * (self.d1 * self.d1) + dt * self.d2,
        }
    }
}

/// Output of the fused Jet2 High-Dimensional Neural PDE Operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jet2PdeOperatorOutput {
    pub u: Vec<f64>,
    pub grad_norm_sq: Vec<f64>,
    pub laplacian: Vec<f64>,
    pub pde_residual: Vec<f64>,
}

/// Evaluate a 1-hidden-layer (`D -> H -> 1`, `tanh` activation) High-Dimensional Neural PDE Operator:
///   u(x) = \sum_{j=1}^H w2_j * tanh((W1 * x)_j + b1_j)
/// together with its exact D-dimensional gradient norm |\nabla_D u(x)|^2,
/// exact D-dimensional Laplacian \Delta_D u(x) = \sum_{k=1}^D \partial^2 u / \partial x_k^2,
/// and stationary viscous Hamilton-Jacobi-Bellman (HJB) spatial operator:
///   R_{HJB}(x) = -\nu * \Delta_D u(x) + 0.5 * |\nabla_D u(x)|^2
///
/// Uses L1/L2 cache-tiled AVX2 GEMM (`general_mat_mul`) + in-place fused 2-Jet Taylor
/// propagation across `rayon` worker threads:
/// - Compute Complexity: O(M * H * D) for this 1-hidden-layer architecture (vs O(M * H * D^2) full-Hessian autodiff).
/// - Scratch Memory Complexity: O(threads * TILE_M * (H + D)) in L1/L2 cache plus 4 * M * 8 bytes for outputs.
pub fn eval_jet2_neural_pde_operator(
    x_flat: &[f64],  // [M * D] row-major particle coordinates
    w1_flat: &[f64], // [H * D] row-major first-layer weights
    b1: &[f64],      // [H] first-layer bias
    w2: &[f64],      // [H] readout weights
    m: usize,
    d: usize,
    h: usize,
    viscosity_nu: f64,
) -> Result<Jet2PdeOperatorOutput, String> {
    if x_flat.len() != m * d {
        return Err(format!("x_flat length {} != m*d ({})", x_flat.len(), m * d));
    }
    if !viscosity_nu.is_finite() || viscosity_nu < 0.0 {
        // A negative viscosity turns the Laplacian term anti-diffusive (the
        // backward heat equation): high-frequency modes grow without bound, so
        // the operator is refused rather than silently exploding.
        return Err("viscosity_nu must be finite and non-negative".to_string());
    }
    if x_flat.iter().any(|&v| !v.is_finite()) {
        return Err("Non-finite (NaN or Inf) value found in x_flat".to_string());
    }
    if w1_flat.iter().any(|&v| !v.is_finite()) || b1.iter().any(|&v| !v.is_finite()) || w2.iter().any(|&v| !v.is_finite()) {
        return Err("Non-finite (NaN or Inf) value found in network parameters".to_string());
    }

    // Precompute row-wise squared norms S_j = \sum_{k=1}^D W1_{j,k}^2 and scaled curvature weights
    let mut w2_s = vec![0.0f64; h];
    for j in 0..h {
        let row = &w1_flat[j * d..(j + 1) * d];
        let mut s = 0.0f64;
        for &w in row {
            s += w * w;
        }
        w2_s[j] = w2[j] * s;
    }

    let w1_view = ArrayView2::from_shape((h, d), w1_flat)
        .map_err(|e| format!("w1 shape error: {}", e))?;

    let x_view = ArrayView2::from_shape((m, d), x_flat)
        .map_err(|e| format!("x shape error: {}", e))?;

    let mut u_out = vec![0.0f64; m];
    let mut grad_sq_out = vec![0.0f64; m];
    let mut lap_out = vec![0.0f64; m];
    let mut res_out = vec![0.0f64; m];

    // 1. Unified L3-resident AVX2/FMA GEMM: Z = X @ W1^T ([M, H])
    let mut z_mat = vec![0.0f64; m * h];
    {
        let mut z_view = ArrayViewMut2::from_shape((m, h), &mut z_mat).unwrap();
        general_mat_mul(1.0, &x_view, &w1_view.t(), 0.0, &mut z_view);
    }

    // 2. Parallel activation pass over rows (in-place transforming z_mat into alpha_mat)
    const CHUNK_SIZE: usize = 256;
    u_out
        .par_chunks_mut(CHUNK_SIZE)
        .zip(lap_out.par_chunks_mut(CHUNK_SIZE))
        .zip(z_mat.par_chunks_mut(CHUNK_SIZE * h))
        .for_each(|((u_chunk, lap_chunk), z_chunk)| {
            for (offset, (u_val, lap_val)) in u_chunk.iter_mut().zip(lap_chunk.iter_mut()).enumerate() {
                let base = offset * h;
                let mut u = 0.0f64;
                let mut lap = 0.0f64;
                for j in 0..h {
                    let z = z_chunk[base + j] + b1[j];
                    let t = z.tanh();
                    let dt = 1.0 - t * t;
                    let d2t = -2.0 * t * dt;
                    u += w2[j] * t;
                    lap += w2_s[j] * d2t;
                    // Overwrite with alpha_{r, j} = w2_j * dt
                    z_chunk[base + j] = w2[j] * dt;
                }
                *u_val = u;
                *lap_val = lap;
            }
        });

    // 3. Unified L3-resident AVX2/FMA GEMM: Grad = Alpha @ W1 ([M, D])
    let mut grad_mat = vec![0.0f64; m * d];
    {
        let alpha_view = ArrayView2::from_shape((m, h), &z_mat).unwrap();
        let mut grad_view = ArrayViewMut2::from_shape((m, d), &mut grad_mat).unwrap();
        general_mat_mul(1.0, &alpha_view, &w1_view, 0.0, &mut grad_view);
    }

    // 4. Parallel pass to compute grad_sq and residual
    grad_sq_out
        .par_chunks_mut(CHUNK_SIZE)
        .zip(res_out.par_chunks_mut(CHUNK_SIZE))
        .zip(lap_out.par_chunks(CHUNK_SIZE))
        .enumerate()
        .for_each(|(chunk_idx, ((gsq_chunk, res_chunk), lap_chunk))| {
            let start_r = chunk_idx * CHUNK_SIZE;
            for (offset, ((gsq_val, res_val), &lap_val)) in
                gsq_chunk.iter_mut().zip(res_chunk.iter_mut()).zip(lap_chunk.iter()).enumerate()
            {
                let r = start_r + offset;
                let g_row = &grad_mat[r * d..(r + 1) * d];
                let mut g_sq = 0.0f64;
                for &gk in g_row {
                    g_sq += gk * gk;
                }
                *gsq_val = g_sq;
                *res_val = -viscosity_nu * lap_val + 0.5 * g_sq;
            }
        });

    Ok(Jet2PdeOperatorOutput {
        u: u_out,
        grad_norm_sq: grad_sq_out,
        laplacian: lap_out,
        pde_residual: res_out,
    })
}

/// Evaluate the Ornstein-Uhlenbeck / Kolmogorov Fokker-Planck stationary operator:
///   R_{FP}(x) = -\nu * \Delta_D u(x) + \gamma * <x, \nabla_D u(x)>
/// where `<x, \nabla_D u(x)>` is the linear restoring drift term distinct from HJB's |\nabla u|^2.
pub fn eval_fokker_planck_ou_operator(
    x_flat: &[f64],
    w1_flat: &[f64],
    b1: &[f64],
    w2: &[f64],
    m: usize,
    d: usize,
    h: usize,
    diffusion_nu: f64,
    drift_gamma: f64,
) -> Result<Jet2PdeOperatorOutput, String> {
    if x_flat.len() != m * d || w1_flat.len() != h * d || b1.len() != h || w2.len() != h {
        return Err("Dimension mismatch in eval_fokker_planck_ou_operator".to_string());
    }
    if !diffusion_nu.is_finite() || diffusion_nu < 0.0 {
        return Err("diffusion_nu must be finite and non-negative".to_string());
    }
    if !drift_gamma.is_finite() {
        return Err("diffusion_nu and drift_gamma must be finite".to_string());
    }
    if x_flat.iter().any(|&v| !v.is_finite()) {
        return Err("Non-finite (NaN or Inf) value found in x_flat".to_string());
    }
    if w1_flat.iter().any(|&v| !v.is_finite()) || b1.iter().any(|&v| !v.is_finite()) || w2.iter().any(|&v| !v.is_finite()) {
        return Err("Non-finite (NaN or Inf) value found in network parameters".to_string());
    }

    let mut w2_s = vec![0.0f64; h];
    for j in 0..h {
        let row = &w1_flat[j * d..(j + 1) * d];
        let mut s = 0.0f64;
        for &w in row {
            s += w * w;
        }
        w2_s[j] = w2[j] * s;
    }

    let w1_view = ArrayView2::from_shape((h, d), w1_flat)
        .map_err(|e| format!("w1 shape error: {}", e))?;

    let x_view = ArrayView2::from_shape((m, d), x_flat)
        .map_err(|e| format!("x shape error: {}", e))?;

    let mut u_out = vec![0.0f64; m];
    let mut x_dot_grad_out = vec![0.0f64; m];
    let mut lap_out = vec![0.0f64; m];
    let mut res_out = vec![0.0f64; m];

    // 1. Unified L3-resident AVX2/FMA GEMM: Z = X @ W1^T ([M, H])
    let mut z_mat = vec![0.0f64; m * h];
    {
        let mut z_view = ArrayViewMut2::from_shape((m, h), &mut z_mat).unwrap();
        general_mat_mul(1.0, &x_view, &w1_view.t(), 0.0, &mut z_view);
    }

    // 2. Exact algebraic fusion: <x, \nabla u(x)> = \sum_{j=1}^H \alpha_j (z_{raw})_j
    // completely eliminating the M x D second GEMM!
    const CHUNK_SIZE: usize = 256;
    u_out
        .par_chunks_mut(CHUNK_SIZE)
        .zip(x_dot_grad_out.par_chunks_mut(CHUNK_SIZE))
        .zip(lap_out.par_chunks_mut(CHUNK_SIZE))
        .zip(res_out.par_chunks_mut(CHUNK_SIZE))
        .enumerate()
        .for_each(|(chunk_idx, (((u_chunk, xdg_chunk), lap_chunk), res_chunk))| {
            let start_r = chunk_idx * CHUNK_SIZE;
            for (offset, (((u_val, xdg_val), lap_val), res_val)) in
                u_chunk.iter_mut().zip(xdg_chunk.iter_mut()).zip(lap_chunk.iter_mut()).zip(res_chunk.iter_mut()).enumerate()
            {
                let r = start_r + offset;
                let base = r * h;
                let mut u = 0.0f64;
                let mut xdg = 0.0f64;
                let mut lap = 0.0f64;
                for j in 0..h {
                    let z_raw = z_mat[base + j];
                    let z = z_raw + b1[j];
                    let t = z.tanh();
                    let dt = 1.0 - t * t;
                    let d2t = -2.0 * t * dt;
                    let alpha = w2[j] * dt;
                    u += w2[j] * t;
                    lap += w2_s[j] * d2t;
                    xdg += alpha * z_raw;
                }
                *u_val = u;
                *xdg_val = xdg;
                *lap_val = lap;
                *res_val = -diffusion_nu * lap + drift_gamma * xdg;
            }
        });

    Ok(Jet2PdeOperatorOutput {
        u: u_out,
        grad_norm_sq: x_dot_grad_out,
        laplacian: lap_out,
        pde_residual: res_out,
    })
}

/// Output of a single 3-Jet analytical PDE training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jet3TrainStepOutput {
    pub loss: f64,
    pub grad_w1: Vec<f64>,
    pub grad_b1: Vec<f64>,
    pub grad_w2: Vec<f64>,
}

/// Compute the exact 3-Jet (`Jet3`) analytical parameter gradients of the second-order
/// high-dimensional PDE residual loss:
///   R_\theta(x_p) = -\nu * \Delta_D u_\theta(x_p) + 0.5 * \beta * |\nabla_D u_\theta(x_p)|^2 + \mu * u_\theta(x_p) - target_f[p]
///   L(\theta) = (1 / (2 * M)) * \sum_{p=1}^M R_\theta(x_p)^2
/// with respect to all network parameters `(W1, b1, w2)` in O(M * H * D) time!
///
/// Uses `tanh'''(z) = -2 * (1 - tanh^2(z)) * (1 - 3 * tanh^2(z))` and factors all
/// outer-product parameter gradients into AVX2 FMA matrix multiplications (`general_mat_mul`).
pub fn compute_jet3_pde_loss_and_grads(
    x_flat: &[f64],   // [M * D]
    target_f: &[f64], // [M]
    w1_flat: &[f64],  // [H * D]
    b1: &[f64],       // [H]
    w2: &[f64],       // [H]
    m: usize,
    d: usize,
    h: usize,
    viscosity_nu: f64,
    hjb_beta: f64,
    reaction_mu: f64,
) -> Result<Jet3TrainStepOutput, String> {
    if x_flat.len() != m * d || target_f.len() != m || w1_flat.len() != h * d || b1.len() != h || w2.len() != h {
        return Err("Dimension mismatch in compute_jet3_pde_loss_and_grads".to_string());
    }
    if !viscosity_nu.is_finite() || !hjb_beta.is_finite() || !reaction_mu.is_finite() {
        return Err("viscosity_nu, hjb_beta, and reaction_mu must be finite".to_string());
    }
    if x_flat.iter().any(|&v| !v.is_finite()) || target_f.iter().any(|&v| !v.is_finite()) {
        return Err("Non-finite (NaN or Inf) value found in x_flat or target_f".to_string());
    }
    if w1_flat.iter().any(|&v| !v.is_finite()) || b1.iter().any(|&v| !v.is_finite()) || w2.iter().any(|&v| !v.is_finite()) {
        return Err("Non-finite (NaN or Inf) value found in network parameters".to_string());
    }

    let mut s_norm_sq = vec![0.0f64; h];
    for j in 0..h {
        let row = &w1_flat[j * d..(j + 1) * d];
        let mut s = 0.0f64;
        for &w in row {
            s += w * w;
        }
        s_norm_sq[j] = s;
    }

    let x_view = ArrayView2::from_shape((m, d), x_flat).unwrap();
    let w1_view = ArrayView2::from_shape((h, d), w1_flat).unwrap();

    // 1. Z = X @ W1^T ([M, H])
    let mut z_mat = vec![0.0f64; m * h];
    {
        let mut z_view = ArrayViewMut2::from_shape((m, h), &mut z_mat).unwrap();
        general_mat_mul(1.0, &x_view, &w1_view.t(), 0.0, &mut z_view);
    }

    const CHUNK_SIZE: usize = 128;

    // Store t = tanh(z+b), dt = 1-t^2, d2t = -2*t*dt, d3t = -2*dt*(1-3*t^2)
    let mut t_mat = vec![0.0f64; m * h];
    let mut dt_mat = vec![0.0f64; m * h];
    let mut d2t_mat = vec![0.0f64; m * h];
    let mut d3t_mat = vec![0.0f64; m * h];
    let mut alpha_mat = vec![0.0f64; m * h];
    let mut u_vec = vec![0.0f64; m];
    let mut lap_vec = vec![0.0f64; m];

    t_mat
        .par_chunks_mut(CHUNK_SIZE * h)
        .zip(dt_mat.par_chunks_mut(CHUNK_SIZE * h))
        .zip(d2t_mat.par_chunks_mut(CHUNK_SIZE * h))
        .zip(d3t_mat.par_chunks_mut(CHUNK_SIZE * h))
        .zip(alpha_mat.par_chunks_mut(CHUNK_SIZE * h))
        .zip(u_vec.par_chunks_mut(CHUNK_SIZE))
        .zip(lap_vec.par_chunks_mut(CHUNK_SIZE))
        .zip(z_mat.par_chunks(CHUNK_SIZE * h))
        .for_each(|(((((((t_chk, dt_chk), d2t_chk), d3t_chk), a_chk), u_chk), lap_chk), z_chk)| {
            let n_pts = u_chk.len();
            for i in 0..n_pts {
                let base = i * h;
                let mut u_p = 0.0f64;
                let mut lap_p = 0.0f64;
                for j in 0..h {
                    let z = z_chk[base + j] + b1[j];
                    let t = z.tanh();
                    let t2 = t * t;
                    let dt = 1.0 - t2;
                    let d2t = -2.0 * t * dt;
                    let d3t = -2.0 * dt * (1.0 - 3.0 * t2);
                    t_chk[base + j] = t;
                    dt_chk[base + j] = dt;
                    d2t_chk[base + j] = d2t;
                    d3t_chk[base + j] = d3t;
                    a_chk[base + j] = w2[j] * dt;
                    u_p += w2[j] * t;
                    lap_p += w2[j] * d2t * s_norm_sq[j];
                }
                u_chk[i] = u_p;
                lap_chk[i] = lap_p;
            }
        });

    // 2. G = Alpha @ W1 ([M, D]) and C = G @ W1^T ([M, H]) if hjb_beta != 0
    let mut g_mat = vec![0.0f64; m * d];
    let mut c_mat = vec![0.0f64; m * h];
    let mut gsq_vec = vec![0.0f64; m];

    if hjb_beta.abs() > 0.0 {
        let alpha_view = ArrayView2::from_shape((m, h), &alpha_mat).unwrap();
        {
            let mut g_view = ArrayViewMut2::from_shape((m, d), &mut g_mat).unwrap();
            general_mat_mul(1.0, &alpha_view, &w1_view, 0.0, &mut g_view);
        }
        let g_view = ArrayView2::from_shape((m, d), &g_mat).unwrap();
        {
            let mut c_view = ArrayViewMut2::from_shape((m, h), &mut c_mat).unwrap();
            general_mat_mul(1.0, &g_view, &w1_view.t(), 0.0, &mut c_view);
        }
        gsq_vec
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, gsq_chk)| {
                let p_start = chunk_idx * CHUNK_SIZE;
                for (offset, val) in gsq_chk.iter_mut().enumerate() {
                    let p = p_start + offset;
                    let g_row = &g_mat[p * d..(p + 1) * d];
                    let mut gsq = 0.0f64;
                    for &gk in g_row {
                        gsq += gk * gk;
                    }
                    *val = gsq;
                }
            });
    }

    // 3. Accumulate loss and per-(p, j) sensitivity matrices Gamma ([M, H]) and Eta ([M, H])
    let inv_m = 1.0 / (m as f64);
    let mut gamma_mat = vec![0.0f64; m * h];
    let mut eta_mat = vec![0.0f64; m * h];

    let (total_loss, grad_b1, grad_w2, w1_row_scale) = gamma_mat
        .par_chunks_mut(CHUNK_SIZE * h)
        .zip(eta_mat.par_chunks_mut(CHUNK_SIZE * h))
        .zip(u_vec.par_chunks(CHUNK_SIZE))
        .zip(lap_vec.par_chunks(CHUNK_SIZE))
        .zip(gsq_vec.par_chunks(CHUNK_SIZE))
        .zip(target_f.par_chunks(CHUNK_SIZE))
        .zip(t_mat.par_chunks(CHUNK_SIZE * h))
        .zip(dt_mat.par_chunks(CHUNK_SIZE * h))
        .zip(d2t_mat.par_chunks(CHUNK_SIZE * h))
        .zip(d3t_mat.par_chunks(CHUNK_SIZE * h))
        .zip(c_mat.par_chunks(CHUNK_SIZE * h))
        .map(|((((((((((gamma_chk, eta_chk), u_chk), lap_chk), gsq_chk), tf_chk), t_chk), dt_chk), d2t_chk), d3t_chk), c_chk)| {
            let n_pts = u_chk.len();
            let mut local_loss = 0.0f64;
            let mut local_gb1 = vec![0.0f64; h];
            let mut local_gw2 = vec![0.0f64; h];
            let mut local_scale = vec![0.0f64; h];

            for i in 0..n_pts {
                let res_p = -viscosity_nu * lap_chk[i] + 0.5 * hjb_beta * gsq_chk[i] + reaction_mu * u_chk[i] - tf_chk[i];
                local_loss += 0.5 * inv_m * res_p * res_p;
                let delta_p = res_p * inv_m;
                let base = i * h;

                for j in 0..h {
                    let t = t_chk[base + j];
                    let dt = dt_chk[base + j];
                    let d2t = d2t_chk[base + j];
                    let d3t = d3t_chk[base + j];
                    let c_pj = c_chk[base + j];
                    let s_j = s_norm_sq[j];
                    let w2_j = w2[j];

                    local_gw2[j] += delta_p * (-viscosity_nu * d2t * s_j + hjb_beta * dt * c_pj + reaction_mu * t);

                    let gamma_pj = delta_p * w2_j * (-viscosity_nu * d3t * s_j + hjb_beta * d2t * c_pj + reaction_mu * dt);
                    gamma_chk[base + j] = gamma_pj;
                    local_gb1[j] += gamma_pj;

                    local_scale[j] += delta_p * (-2.0 * viscosity_nu * w2_j * d2t);
                    eta_chk[base + j] = delta_p * hjb_beta * w2_j * dt;
                }
            }
            (local_loss, local_gb1, local_gw2, local_scale)
        })
        .reduce(
            || (0.0f64, vec![0.0f64; h], vec![0.0f64; h], vec![0.0f64; h]),
            |mut a, b| {
                a.0 += b.0;
                for j in 0..h {
                    a.1[j] += b.1[j];
                    a.2[j] += b.2[j];
                    a.3[j] += b.3[j];
                }
                a
            },
        );

    // 4. grad_W1 = Gamma^T @ X + Eta^T @ G + diag(w1_row_scale) * W1
    let mut grad_w1 = vec![0.0f64; h * d];
    {
        let gamma_view = ArrayView2::from_shape((m, h), &gamma_mat).unwrap();
        let mut gw1_view = ArrayViewMut2::from_shape((h, d), &mut grad_w1).unwrap();
        general_mat_mul(1.0, &gamma_view.t(), &x_view, 0.0, &mut gw1_view);

        if hjb_beta.abs() > 0.0 {
            let eta_view = ArrayView2::from_shape((m, h), &eta_mat).unwrap();
            let g_view = ArrayView2::from_shape((m, d), &g_mat).unwrap();
            general_mat_mul(1.0, &eta_view.t(), &g_view, 1.0, &mut gw1_view);
        }
    }
    for j in 0..h {
        let scale = w1_row_scale[j];
        for k in 0..d {
            grad_w1[j * d + k] += scale * w1_flat[j * d + k];
        }
    }

    Ok(Jet3TrainStepOutput {
        loss: total_loss,
        grad_w1,
        grad_b1,
        grad_w2,
    })
}

/// Evolve N interacting high-dimensional particles/tokens X \in R^{N x D} across `steps`
/// layers/time-steps of the coupled McKean-Vlasov Self-Attention Transport PDE:
///
///   \partial_t X_i = \sum_{j=1}^N softmax_j(<X_i, X_j> / \sqrt{D}) * (X_j - X_i)
///                    - damping * X_i
///
/// Uses Block-Tiled FlashAttention-CPU (Online Softmax + AVX2 GEMM micro-kernels)
/// and Ping-Pong double buffering:
/// - Compute Complexity per step: O(N^2 * D) exact dense attention across all query-key pairs.
/// - Scratch Memory Complexity: O(N * D + threads * B_R * (B_C + D)) — never allocates the O(N^2) attention matrix in RAM.
pub fn step_vlasov_attention_pde_hd(
    x_init_flat: &[f64], // [N * D] initial particle/token states
    n: usize,
    d: usize,
    steps: usize,
    dt: f64,
    damping: f64,
) -> Result<Vec<f64>, String> {
    if x_init_flat.len() != n * d {
        return Err(format!("x_init_flat length {} != n*d ({})", x_init_flat.len(), n * d));
    }
    if n == 0 || d == 0 {
        return Err("n and d must be >= 1".to_string());
    }

    let inv_sqrt_d = 1.0 / (d as f64).sqrt();
    let mut buf_a = x_init_flat.to_vec();
    let mut buf_b = vec![0.0f64; n * d];

    const B_R: usize = 128;
    const B_C: usize = 128;

    let n_tiles = (n + B_R - 1) / B_R;
    // Pre-allocate L1/L2-resident scratch memory per query tile once outside the time loop
    let mut tile_scratches: Vec<Vec<f64>> = (0..n_tiles)
        .map(|tile_idx| {
            let br = ((tile_idx + 1) * B_R).min(n) - tile_idx * B_R;
            vec![0.0f64; br + br + br * d + br * B_C]
        })
        .collect();

    for _step in 0..steps {
        let src = &buf_a;
        buf_b
            .par_chunks_mut(B_R * d)
            .zip(tile_scratches.par_iter_mut())
            .enumerate()
            .for_each(|(q_tile_idx, (dst_tile, scratch))| {
                let q_start = q_tile_idx * B_R;
                let br = dst_tile.len() / d;
                let xq_slice = &src[q_start * d..(q_start + br) * d];
                let xq_view = ArrayView2::from_shape((br, d), xq_slice).unwrap();

                // Partition preallocated scratch buffer without heap allocations:
                // [0..br] -> m_row, [br..2*br] -> l_row, [2*br..2*br+br*d] -> acc_buf, [rest] -> scores_buf
                let (m_row, rest) = scratch[..br + br + br * d + br * B_C].split_at_mut(br);
                let (l_row, rest2) = rest.split_at_mut(br);
                let (acc_buf, scores_buf) = rest2.split_at_mut(br * d);

                m_row.fill(f64::NEG_INFINITY);
                l_row.fill(0.0f64);
                acc_buf.fill(0.0f64);

                let mut k_start = 0;
                while k_start < n {
                    let bc = (n - k_start).min(B_C);
                    let xk_slice = &src[k_start * d..(k_start + bc) * d];
                    let xk_view = ArrayView2::from_shape((bc, d), xk_slice).unwrap();

                    let s_slice = &mut scores_buf[..br * bc];
                    {
                        let mut s_view = ArrayViewMut2::from_shape((br, bc), s_slice).unwrap();
                        // Block-tiled Q_tile @ K_tile^T in AVX2 FMA registers
                        general_mat_mul(inv_sqrt_d, &xq_view, &xk_view.t(), 0.0, &mut s_view);
                    }

                    // Online Softmax rescaling on L1-resident tile [br, bc]
                    for r in 0..br {
                        let s_row = &mut s_slice[r * bc..(r + 1) * bc];
                        let mut row_max = m_row[r];
                        for &val in s_row.iter() {
                            if val > row_max {
                                row_max = val;
                            }
                        }
                        let exp_old = if m_row[r] == f64::NEG_INFINITY {
                            0.0
                        } else {
                            (m_row[r] - row_max).exp()
                        };

                        let mut block_sum = 0.0f64;
                        for val in s_row.iter_mut() {
                            let w = (*val - row_max).exp();
                            *val = w;
                            block_sum += w;
                        }

                        if exp_old != 1.0 {
                            let acc_row = &mut acc_buf[r * d..(r + 1) * d];
                            for a in acc_row.iter_mut() {
                                *a *= exp_old;
                            }
                        }

                        l_row[r] = l_row[r] * exp_old + block_sum;
                        m_row[r] = row_max;
                    }

                    {
                        let p_view = ArrayView2::from_shape((br, bc), s_slice).unwrap();
                        let mut acc_view = ArrayViewMut2::from_shape((br, d), &mut *acc_buf).unwrap();
                        // Block-tiled Acc_tile += P_tile @ V_tile in AVX2 FMA registers
                        general_mat_mul(1.0, &p_view, &xk_view, 1.0, &mut acc_view);
                    }

                    k_start += bc;
                }

                // Finalize Euler PDE step for query tile
                for r in 0..br {
                    let inv_l = 1.0 / l_row[r].max(1e-30);
                    let x_i = &xq_slice[r * d..(r + 1) * d];
                    let acc_i = &acc_buf[r * d..(r + 1) * d];
                    let dst_i = &mut dst_tile[r * d..(r + 1) * d];
                    for k in 0..d {
                        let attn_k = acc_i[k] * inv_l;
                        let drift_k = (attn_k - x_i[k]) - damping * x_i[k];
                        dst_i[k] = x_i[k] + dt * drift_k;
                    }
                }
            });

        std::mem::swap(&mut buf_a, &mut buf_b);
    }

    Ok(buf_a)
}

/// Zero-copy flat-slice implementation of D-dimensional Mesh-Free Point Cloud PDE Discovery.
/// Accepts contiguous flat slices directly from NumPy without JSON serialization overhead,
/// and evaluates the D-dimensional stencil in parallel across time and anchor points.
pub fn discover_pde_point_cloud_hd_flat(
    u_anchors_flat: &[f64],      // [N_t * M]
    u_probes_plus_flat: &[f64],  // [N_t * M * D]
    u_probes_minus_flat: &[f64], // [N_t * M * D]
    nt: usize,
    m: usize,
    dimension: usize,
    dt: f64,
    h: f64,
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    if nt < 5 {
        return Err("Point cloud HD PDE requires at least 5 time steps".to_string());
    }
    if m < 5 {
        return Err("Point cloud HD PDE requires at least 5 anchor points".to_string());
    }
    if u_anchors_flat.len() != nt * m {
        return Err("u_anchors_flat size mismatch".to_string());
    }
    if u_probes_plus_flat.len() != nt * m * dimension || u_probes_minus_flat.len() != nt * m * dimension {
        return Err("u_probes_flat size mismatch".to_string());
    }

    let dt = dt.max(1e-12);
    let h = h.max(1e-12);
    let inv_2dt = 1.0 / (2.0 * dt);
    let inv_2h = 1.0 / (2.0 * h);
    let inv_h_sq = 1.0 / (h * h);

    let n_interior_t = nt - 2;
    let total_pts = n_interior_t * m;

    let mut ut = vec![0.0f64; total_pts];
    let mut u_vals = vec![0.0f64; total_pts];
    let mut lap_hd = vec![0.0f64; total_pts];
    let mut grad_sq = vec![0.0f64; total_pts];
    let mut u_lap = vec![0.0f64; total_pts];

    ut.par_iter_mut()
        .zip(u_vals.par_iter_mut())
        .zip(lap_hd.par_iter_mut())
        .zip(grad_sq.par_iter_mut())
        .zip(u_lap.par_iter_mut())
        .enumerate()
        .for_each(|(idx, ((((ut_i, u_i), lap_i), gsq_i), ulap_i))| {
            let t = (idx / m) + 1;
            let p = idx % m;
            let tp_idx = t * m + p;

            let u_curr = u_anchors_flat[tp_idx];
            let d_t = (u_anchors_flat[(t + 1) * m + p] - u_anchors_flat[(t - 1) * m + p]) * inv_2dt;

            let probe_base = tp_idx * dimension;
            let plus_slice = &u_probes_plus_flat[probe_base..probe_base + dimension];
            let minus_slice = &u_probes_minus_flat[probe_base..probe_base + dimension];

            let mut lap = 0.0f64;
            let mut g_sq = 0.0f64;
            let two_u = 2.0 * u_curr;

            for k in 0..dimension {
                let u_p = plus_slice[k];
                let u_m = minus_slice[k];
                let d_xk = (u_p - u_m) * inv_2h;
                let d_xkxk = (u_p - two_u + u_m) * inv_h_sq;
                lap += d_xkxk;
                g_sq += d_xk * d_xk;
            }

            *ut_i = d_t;
            *u_i = u_curr;
            *lap_i = lap;
            *gsq_i = g_sq;
            *ulap_i = u_curr * lap;
        });

    let mut term_names = Vec::with_capacity(5);
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(5);

    term_names.push("u".to_string());
    if config.max_polynomial_degree >= 2 {
        let u_sq: Vec<f64> = u_vals.iter().map(|&v| v * v).collect();
        columns.push(u_vals);
        term_names.push("u^2".to_string());
        columns.push(u_sq);
    } else {
        columns.push(u_vals);
    }

    term_names.push(format!("laplacian_{}d(u)", dimension));
    columns.push(lap_hd);

    term_names.push("|grad_u|^2".to_string());
    columns.push(grad_sq);

    term_names.push(format!("u*laplacian_{}d(u)", dimension));
    columns.push(u_lap);

    let n_int_t = nt - 2;
    let method = format!("meshfree_point_cloud_{}d", dimension);
    run_stridge_with_frozen_holdout(&ut, &columns, &term_names, config, n_int_t, dimension, method)
}

/// Result of high-dimensional or hybrid PDE discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighDimPdeResult {
    pub formula: String,
    pub active_terms: Vec<String>,
    pub coefficients: Vec<f64>,
    pub residual_rmse: f64,
    pub r2: f64,
    pub is_exact_recovery: bool,
    pub dimension: usize,
    pub method_used: String,
    #[serde(default)]
    pub frozen_holdout_r2: f64,
    #[serde(default)]
    pub frozen_holdout_rmse: f64,
}

impl From<(PdeResult, usize, String)> for HighDimPdeResult {
    fn from((res, dim, method): (PdeResult, usize, String)) -> Self {
        let r2 = res.r2;
        let rmse = res.residual_rmse;
        HighDimPdeResult {
            formula: res.formula,
            active_terms: res.active_terms,
            coefficients: res.coefficients,
            residual_rmse: rmse,
            r2,
            is_exact_recovery: res.is_exact_recovery,
            dimension: dim,
            method_used: method,
            frozen_holdout_r2: r2,
            frozen_holdout_rmse: rmse,
        }
    }
}

impl From<(PdeResult, usize, &str)> for HighDimPdeResult {
    fn from((res, dim, method): (PdeResult, usize, &str)) -> Self {
        HighDimPdeResult::from((res, dim, method.to_string()))
    }
}

/// Fit STRidge strictly on the development time window and evaluate the FROZEN
/// discovered formula and coefficients out-of-sample on the future holdout time window.
pub fn run_stridge_with_frozen_holdout(
    ut: &[f64],
    columns: &[Vec<f64>],
    term_names: &[String],
    config: &PdeConfig,
    n_interior_t: usize,
    dimension: usize,
    method: String,
) -> Result<HighDimPdeResult, String> {
    let total = ut.len();
    if total == 0 || columns.is_empty() {
        return Err("Empty sample library for STRidge".to_string());
    }

    let pts_per_step = if n_interior_t > 0 && total % n_interior_t == 0 {
        total / n_interior_t
    } else {
        1
    };

    // Partition chronologically along interior time steps: first ~75% dev, remaining ~25% holdout
    let dev_t = if n_interior_t >= 4 {
        ((n_interior_t * 3) / 4).max(2)
    } else {
        n_interior_t
    };
    let n_dev = (dev_t * pts_per_step).min(total);

    let ut_dev = &ut[..n_dev];
    let cols_dev: Vec<Vec<f64>> = columns.iter().map(|c| c[..n_dev].to_vec()).collect();

    let raw_dev = run_stridge(ut_dev, &cols_dev, term_names, config, "u_t")?;

    // Evaluate frozen dev coefficients on held-out future steps [n_dev..total]
    let (frozen_holdout_r2, frozen_holdout_rmse) = if n_dev < total {
        let hold_len = total - n_dev;
        let hold_slice = &ut[n_dev..];
        let hold_mean = hold_slice.iter().sum::<f64>() / (hold_len as f64);
        let mut ss_res = 0.0f64;
        let mut ss_tot = 0.0f64;

        for s in n_dev..total {
            let mut pred = 0.0f64;
            for (idx, t_name) in raw_dev.active_terms.iter().enumerate() {
                if let Some(col_idx) = term_names.iter().position(|n| n == t_name) {
                    pred += raw_dev.coefficients[idx] * columns[col_idx][s];
                }
            }
            let diff = ut[s] - pred;
            ss_res += diff * diff;
            let dm = ut[s] - hold_mean;
            ss_tot += dm * dm;
        }
        let rmse = (ss_res / (hold_len as f64)).sqrt();
        let r2 = if ss_tot > 1e-30 {
            1.0 - (ss_res / ss_tot)
        } else if ss_res < 1e-24 {
            1.0
        } else {
            0.0
        };
        (r2, rmse)
    } else {
        (raw_dev.r2, raw_dev.residual_rmse)
    };

    Ok(HighDimPdeResult {
        formula: raw_dev.formula,
        active_terms: raw_dev.active_terms,
        coefficients: raw_dev.coefficients,
        residual_rmse: raw_dev.residual_rmse,
        r2: raw_dev.r2,
        is_exact_recovery: raw_dev.is_exact_recovery,
        dimension,
        method_used: method,
        frozen_holdout_r2,
        frozen_holdout_rmse,
    })
}

/// Discover 3D Cartesian Eulerian spatio-temporal PDE:
/// u_t = F(u, \nabla^2 u, u_x, u_y, u_z, ...).
/// Uses 4th-order 5-point central stencils when grid >= 7x7x7, and 2nd-order 3-point stencils for 5x5x5..6x6x6.
pub fn discover_pde_3d_cartesian(
    u_grid: &[Vec<Vec<Vec<f64>>>],
    dt: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    let nt = u_grid.len();
    if nt < 5 {
        return Err("3D PDE requires at least 5 time steps".to_string());
    }
    let nx = u_grid[0].len();
    let ny = u_grid[0][0].len();
    let nz = u_grid[0][0][0].len();
    if nx < 5 || ny < 5 || nz < 5 {
        return Err("3D PDE requires at least 5x5x5 spatial grid".to_string());
    }

    let dt = dt.max(1e-12);
    let dx = dx.max(1e-12);
    let dy = dy.max(1e-12);
    let dz = dz.max(1e-12);

    let use_fourth_order = nx >= 7 && ny >= 7 && nz >= 7;
    let pad = if use_fourth_order { 2 } else { 1 };

    let mut ut = Vec::new();
    let mut u_vals = Vec::new();
    let mut lap3d = Vec::new();
    let mut grad_sq = Vec::new();

    for t in 1..(nt - 1) {
        for i in pad..(nx - pad) {
            for j in pad..(ny - pad) {
                for k in pad..(nz - pad) {
                    let d_t = (u_grid[t + 1][i][j][k] - u_grid[t - 1][i][j][k]) / (2.0 * dt);
                    let u_val = u_grid[t][i][j][k];

                    let (d_x, d_xx, d_y, d_yy, d_z, d_zz) = if use_fourth_order {
                        let ux = (-u_grid[t][i + 2][j][k] + 8.0 * u_grid[t][i + 1][j][k]
                            - 8.0 * u_grid[t][i - 1][j][k] + u_grid[t][i - 2][j][k])
                            / (12.0 * dx);
                        let uxx = (-u_grid[t][i + 2][j][k] + 16.0 * u_grid[t][i + 1][j][k]
                            - 30.0 * u_val + 16.0 * u_grid[t][i - 1][j][k] - u_grid[t][i - 2][j][k])
                            / (12.0 * dx * dx);

                        let uy = (-u_grid[t][i][j + 2][k] + 8.0 * u_grid[t][i][j + 1][k]
                            - 8.0 * u_grid[t][i][j - 1][k] + u_grid[t][i][j - 2][k])
                            / (12.0 * dy);
                        let uyy = (-u_grid[t][i][j + 2][k] + 16.0 * u_grid[t][i][j + 1][k]
                            - 30.0 * u_val + 16.0 * u_grid[t][i][j - 1][k] - u_grid[t][i][j - 2][k])
                            / (12.0 * dy * dy);

                        let uz = (-u_grid[t][i][j][k + 2] + 8.0 * u_grid[t][i][j][k + 1]
                            - 8.0 * u_grid[t][i][j][k - 1] + u_grid[t][i][j][k - 2])
                            / (12.0 * dz);
                        let uzz = (-u_grid[t][i][j][k + 2] + 16.0 * u_grid[t][i][j][k + 1]
                            - 30.0 * u_val + 16.0 * u_grid[t][i][j][k - 1] - u_grid[t][i][j][k - 2])
                            / (12.0 * dz * dz);
                        (ux, uxx, uy, uyy, uz, uzz)
                    } else {
                        let ux = (u_grid[t][i + 1][j][k] - u_grid[t][i - 1][j][k]) / (2.0 * dx);
                        let uxx = (u_grid[t][i + 1][j][k] - 2.0 * u_val + u_grid[t][i - 1][j][k]) / (dx * dx);
                        let uy = (u_grid[t][i][j + 1][k] - u_grid[t][i][j - 1][k]) / (2.0 * dy);
                        let uyy = (u_grid[t][i][j + 1][k] - 2.0 * u_val + u_grid[t][i][j - 1][k]) / (dy * dy);
                        let uz = (u_grid[t][i][j][k + 1] - u_grid[t][i][j][k - 1]) / (2.0 * dz);
                        let uzz = (u_grid[t][i][j][k + 1] - 2.0 * u_val + u_grid[t][i][j][k - 1]) / (dz * dz);
                        (ux, uxx, uy, uyy, uz, uzz)
                    };

                    let lap = d_xx + d_yy + d_zz;
                    let g_sq = d_x * d_x + d_y * d_y + d_z * d_z;

                    ut.push(d_t);
                    u_vals.push(u_val);
                    lap3d.push(lap);
                    grad_sq.push(g_sq);
                }
            }
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push("laplacian(u)".to_string());
    columns.push(lap3d.clone());

    term_names.push("|grad_u|^2".to_string());
    columns.push(grad_sq);

    term_names.push("u*laplacian(u)".to_string());
    columns.push(u_vals.iter().zip(lap3d.iter()).map(|(&u, &l)| u * l).collect());

    run_stridge_with_frozen_holdout(
        &ut,
        &columns,
        &term_names,
        config,
        nt - 2,
        3,
        "eulerian_cartesian_3d".to_string(),
    )
}

/// General D-Dimensional Eulerian Cartesian Grid PDE Discovery for ALL D in 1..=9.
///
/// Accepts a flattened tensor `u_flat` of shape `[N_t, N_1, N_2, ..., N_D]` and coordinate
/// spacings `dxs` of length `D`. Along each spatial axis `k`, automatically applies the
/// 4th-order 5-point central difference stencil when `N_k >= 7`, or the 2nd-order 3-point
/// central difference stencil when `3 <= N_k < 7`.
pub fn discover_pde_nd_cartesian(
    u_flat: &[f64],
    grid_shape: &[usize], // [N_t, N_1, ..., N_D]
    dt: f64,
    dxs: &[f64],
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    if grid_shape.len() < 2 {
        return Err("grid_shape must have at least [N_t, N_1]".to_string());
    }
    let dimension = grid_shape.len() - 1;
    if dimension >= 10 {
        return Err(format!(
            "Eulerian Cartesian grid discovery is restricted to D < 10 (got D={}); use high-dimensional methods for D >= 10",
            dimension
        ));
    }
    if dxs.len() != dimension {
        return Err(format!("dxs length {} != dimension {}", dxs.len(), dimension));
    }
    let nt = grid_shape[0];
    if nt < 5 {
        return Err("Cartesian PDE discovery requires at least 5 time steps".to_string());
    }
    let expected_len: usize = grid_shape.iter().product();
    if u_flat.len() != expected_len {
        return Err(format!(
            "u_flat length {} does not match grid_shape product {}",
            u_flat.len(),
            expected_len
        ));
    }

    let spatial_dims = &grid_shape[1..];
    let mut pads = vec![1usize; dimension];
    for k in 0..dimension {
        if spatial_dims[k] < 3 {
            return Err(format!(
                "Spatial axis {} has size {} < 3 (minimum 3 required for central stencil)",
                k, spatial_dims[k]
            ));
        }
        pads[k] = if spatial_dims[k] >= 7 { 2 } else { 1 };
    }

    let mut strides = vec![1usize; dimension];
    for k in (0..(dimension - 1)).rev() {
        strides[k] = strides[k + 1] * spatial_dims[k + 1];
    }
    let time_stride = strides[0] * spatial_dims[0];

    // Enumerate all interior spatial offsets
    let mut interior_offsets = Vec::new();
    let mut current_idx = pads.clone();
    loop {
        let mut off = 0usize;
        for k in 0..dimension {
            off += current_idx[k] * strides[k];
        }
        interior_offsets.push(off);

        let mut carry = true;
        for k in (0..dimension).rev() {
            if carry {
                current_idx[k] += 1;
                if current_idx[k] < spatial_dims[k] - pads[k] {
                    carry = false;
                } else {
                    current_idx[k] = pads[k];
                }
            }
        }
        if carry {
            break;
        }
    }

    if interior_offsets.is_empty() {
        return Err("No interior spatial grid points available".to_string());
    }

    let dt = dt.max(1e-12);
    let n_int_t = nt - 2;
    let total_samples = n_int_t * interior_offsets.len();

    let mut ut = Vec::with_capacity(total_samples);
    let mut u_vals = Vec::with_capacity(total_samples);
    let mut lap_nd = Vec::with_capacity(total_samples);
    let mut grad_sq = Vec::with_capacity(total_samples);

    for t in 1..(nt - 1) {
        let t_base = t * time_stride;
        for &sp_off in &interior_offsets {
            let idx = t_base + sp_off;
            let u_val = u_flat[idx];
            let d_t = (u_flat[idx + time_stride] - u_flat[idx - time_stride]) / (2.0 * dt);

            let mut lap = 0.0f64;
            let mut g_sq = 0.0f64;

            for k in 0..dimension {
                let st = strides[k];
                let dx = dxs[k].max(1e-12);
                let (d_xk, d_xkxk) = if pads[k] == 2 {
                    let up2 = u_flat[idx + 2 * st];
                    let up1 = u_flat[idx + st];
                    let um1 = u_flat[idx - st];
                    let um2 = u_flat[idx - 2 * st];
                    let g1 = (-up2 + 8.0 * up1 - 8.0 * um1 + um2) / (12.0 * dx);
                    let g2 = (-up2 + 16.0 * up1 - 30.0 * u_val + 16.0 * um1 - um2) / (12.0 * dx * dx);
                    (g1, g2)
                } else {
                    let up1 = u_flat[idx + st];
                    let um1 = u_flat[idx - st];
                    let g1 = (up1 - um1) / (2.0 * dx);
                    let g2 = (up1 - 2.0 * u_val + um1) / (dx * dx);
                    (g1, g2)
                };
                lap += d_xkxk;
                g_sq += d_xk * d_xk;
            }

            ut.push(d_t);
            u_vals.push(u_val);
            lap_nd.push(lap);
            grad_sq.push(g_sq);
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push("laplacian(u)".to_string());
    columns.push(lap_nd.clone());

    term_names.push("|grad_u|^2".to_string());
    columns.push(grad_sq);

    term_names.push("u*laplacian(u)".to_string());
    columns.push(u_vals.iter().zip(lap_nd.iter()).map(|(&u, &l)| u * l).collect());

    let method = format!("eulerian_cartesian_{}d", dimension);
    run_stridge_with_frozen_holdout(&ut, &columns, &term_names, config, n_int_t, dimension, method)
}

/// Discover D-dimensional PDE via Hyperspherical Radial Invariant Reduction (Technique 2).
///
/// Reduces the D-dimensional Laplacian to:
/// \nabla^2_D u = u_{rr} + ((D - 1) / r) * u_r
/// Valid for radially symmetric fields (e.g. isotropic heat, waves, central potentials).
/// Operates on radial slices u(t, r) with O(1) complexity in spatial dimension D.
pub fn discover_pde_radial_hd(
    r_coords: &[f64],
    u_grid: &[Vec<f64>], // [N_t][N_r]
    dt: f64,
    dr: f64,
    dimension: usize,
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    let nt = u_grid.len();
    let nr = r_coords.len();
    if nt < 5 || nr < 7 {
        return Err(format!(
            "Radial HD PDE requires at least 5 time steps and 7 radial points (got nt={}, nr={})",
            nt, nr
        ));
    }
    if dimension < 1 {
        return Err("Dimension must be >= 1".to_string());
    }

    let dt = dt.max(1e-12);
    let dr = dr.max(1e-12);
    let d_minus_1 = (dimension - 1) as f64;

    let mut ut = Vec::new();
    let mut u_vals = Vec::new();
    let mut lap_hd = Vec::new();
    let mut ur_vals = Vec::new();
    let mut u_ur_vals = Vec::new();

    // 4th order central interior stencils in r
    for ti in 1..(nt - 1) {
        for ri in 2..(nr - 2) {
            let r_val = r_coords[ri];
            if r_val.abs() < 1e-9 {
                continue; // Avoid singularity at r = 0
            }

            let d_t = (u_grid[ti + 1][ri] - u_grid[ti - 1][ri]) / (2.0 * dt);
            let u_val = u_grid[ti][ri];

            // 4th order first radial derivative: (-u_{i+2} + 8 u_{i+1} - 8 u_{i-1} + u_{i-2}) / (12 dr)
            let ur = (-u_grid[ti][ri + 2] + 8.0 * u_grid[ti][ri + 1] - 8.0 * u_grid[ti][ri - 1] + u_grid[ti][ri - 2])
                / (12.0 * dr);

            // 4th order second radial derivative
            let urr = (-u_grid[ti][ri + 2] + 16.0 * u_grid[ti][ri + 1] - 30.0 * u_val
                + 16.0 * u_grid[ti][ri - 1] - u_grid[ti][ri - 2])
                / (12.0 * dr * dr);

            // Exact Hyperspherical D-dimensional Laplacian
            let lap = urr + (d_minus_1 / r_val) * ur;

            ut.push(d_t);
            u_vals.push(u_val);
            lap_hd.push(lap);
            ur_vals.push(ur);
            u_ur_vals.push(u_val * ur);
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push(format!("laplacian_{}d(u)", dimension));
    columns.push(lap_hd.clone());

    term_names.push("u_r".to_string());
    columns.push(ur_vals);

    if config.include_advection_terms {
        term_names.push("u*u_r".to_string());
        columns.push(u_ur_vals);
    }

    let method = format!("hyperspherical_radial_{}d", dimension);
    run_stridge_with_frozen_holdout(&ut, &columns, &term_names, config, nt - 2, dimension, method)
}

/// Discover D-dimensional PDE from a TRUE Scattered Point Cloud (X, U) via Multi-Scale
/// Gaussian Radial Basis Functions (RBFs) — Technique 1 (No finite-difference probes required!).
///
/// Given M scattered points `points_flat` ([M * D]) in R^D and scalar observations
/// `u_values_flat` ([N_t * M]), fits an overdetermined multi-scale Gaussian RBF expansion
/// \psi_{c, \epsilon}(x) = \exp(-\epsilon^2 ||x - c||^2) and evaluates the exact analytical
/// D-dimensional RBF Laplacian:
///   \nabla^2_D \psi_{c, \epsilon}(x) = 2 \epsilon^2 (2 \epsilon^2 ||x - c||^2 - D) \psi_{c, \epsilon}(x)
pub fn discover_pde_rbf_point_cloud_hd(
    points_flat: &[f64],   // [M * D] scattered coordinates in R^D
    u_values_flat: &[f64], // [N_t * M] scalar field values at scattered points
    nt: usize,
    m: usize,
    dimension: usize,
    dt: f64,
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    if nt < 5 {
        return Err("RBF point cloud HD PDE requires at least 5 time steps".to_string());
    }
    if m < dimension + 6 {
        return Err(format!(
            "RBF point cloud in {}D requires at least {} scattered points (got {})",
            dimension,
            dimension + 6,
            m
        ));
    }
    if points_flat.len() != m * dimension {
        return Err("points_flat length mismatch with m * dimension".to_string());
    }
    if u_values_flat.len() != nt * m {
        return Err("u_values_flat length mismatch with nt * m".to_string());
    }

    let dt = dt.max(1e-12);
    let dim_f = dimension as f64;

    // 1. Compute cloud centroid
    let mut centroid = vec![0.0f64; dimension];
    for p in 0..m {
        let pt = &points_flat[p * dimension..(p + 1) * dimension];
        for k in 0..dimension {
            centroid[k] += pt[k] / (m as f64);
        }
    }

    // 2. Estimate field stationary center at t=0 via log-quadratic least squares:
    //    ln(|u(0, x)| + 1e-12) ~ c0 + b^T x - a ||x||^2  =>  center = b / (2a)
    let p_quad = dimension + 2;
    let mut ztz = vec![0.0f64; p_quad * p_quad];
    let mut zty = vec![0.0f64; p_quad];
    for p in 0..m {
        let pt = &points_flat[p * dimension..(p + 1) * dimension];
        let r2: f64 = pt.iter().map(|&v| v * v).sum();
        let y_val = (u_values_flat[p].abs() + 1e-12).ln();

        let mut row = Vec::with_capacity(p_quad);
        row.push(1.0);
        row.extend_from_slice(pt);
        row.push(-r2);

        for i in 0..p_quad {
            zty[i] += row[i] * y_val;
            for j in 0..p_quad {
                ztz[i * p_quad + j] += row[i] * row[j];
            }
        }
    }
    for i in 0..p_quad {
        ztz[i * p_quad + i] += 1e-8;
    }

    let (est_center, est_eps) = if let Ok(sol) = solve_linear_system(&ztz, &zty, p_quad) {
        let a_pos = sol[dimension + 1].abs().max(1e-4);
        let a_denom = 2.0 * a_pos;
        let mut c = vec![0.0f64; dimension];
        for k in 0..dimension {
            c[k] = (sol[1 + k] / a_denom).clamp(-2.0, 2.0);
        }
        (c, a_pos.sqrt().clamp(0.05, 5.0))
    } else {
        // Retain the recorded approximation: changing it perturbs the search.
        #[allow(clippy::approx_constant)]
        let initial_scale = 0.70710678;
        (centroid.clone(), initial_scale)
    };

    // 3. Construct multi-scale Gaussian RBF dictionary around [est_center, centroid]
    // including adaptive scales anchored around `est_eps` plus a broad multi-scale grid
    let centers = [est_center, centroid];
    let scales = [
        est_eps * 0.94,
        est_eps * 0.97,
        est_eps,
        est_eps * 1.03,
        est_eps * 1.06,
        0.25,
        0.40,
        0.55,
        0.70,
        0.85,
        1.05,
        1.30,
    ];
    let n_scales = scales.len();
    let b_basis = centers.len() * n_scales;

    let mut phi_mat = vec![0.0f64; m * b_basis];
    let mut lap_op = vec![0.0f64; m * b_basis];
    let mut grad_op = vec![0.0f64; m * b_basis * dimension];

    for (c_idx, center) in centers.iter().enumerate() {
        for (s_idx, &eps) in scales.iter().enumerate() {
            let b = c_idx * n_scales + s_idx;
            let e2 = eps * eps;

            for p in 0..m {
                let pt = &points_flat[p * dimension..(p + 1) * dimension];
                let mut r2 = 0.0f64;
                for k in 0..dimension {
                    let d = pt[k] - center[k];
                    r2 += d * d;
                }
                let phi = (-e2 * r2).exp();
                let lap = 2.0 * e2 * (2.0 * e2 * r2 - dim_f) * phi;

                phi_mat[p * b_basis + b] = phi;
                lap_op[p * b_basis + b] = lap;

                let g_base = (p * b_basis + b) * dimension;
                for k in 0..dimension {
                    grad_op[g_base + k] = -2.0 * e2 * (pt[k] - center[k]) * phi;
                }
            }
        }
    }

    // Precompute regularized Gram matrix PtP = Phi^T Phi + 1e-9 I
    let mut ptp = vec![0.0f64; b_basis * b_basis];
    for i in 0..b_basis {
        for j in i..b_basis {
            let mut sum = 0.0f64;
            for p in 0..m {
                sum += phi_mat[p * b_basis + i] * phi_mat[p * b_basis + j];
            }
            if i == j {
                sum += 1e-9;
            }
            ptp[i * b_basis + j] = sum;
            ptp[j * b_basis + i] = sum;
        }
    }

    let n_int_t = nt - 2;
    let total_samples = n_int_t * m;
    let mut ut = Vec::with_capacity(total_samples);
    let mut u_vals = Vec::with_capacity(total_samples);
    let mut lap_hd = Vec::with_capacity(total_samples);
    let mut grad_sq = Vec::with_capacity(total_samples);

    for t in 1..(nt - 1) {
        let u_curr = &u_values_flat[t * m..(t + 1) * m];
        let u_prev = &u_values_flat[(t - 1) * m..t * m];
        let u_next = &u_values_flat[(t + 1) * m..(t + 2) * m];

        // Compute Phi^T u_curr
        let mut ptu = vec![0.0f64; b_basis];
        for b in 0..b_basis {
            let mut sum = 0.0f64;
            for p in 0..m {
                sum += phi_mat[p * b_basis + b] * u_curr[p];
            }
            ptu[b] = sum;
        }

        let w = solve_linear_system(&ptp, &ptu, b_basis)?;

        for p in 0..m {
            let d_t = (u_next[p] - u_prev[p]) / (2.0 * dt);
            let mut lap_p = 0.0f64;
            let mut g_vec = vec![0.0f64; dimension];
            for b in 0..b_basis {
                let wb = w[b];
                lap_p += lap_op[p * b_basis + b] * wb;
                let g_base = (p * b_basis + b) * dimension;
                for k in 0..dimension {
                    g_vec[k] += grad_op[g_base + k] * wb;
                }
            }
            let g_sq_p: f64 = g_vec.iter().map(|&gk| gk * gk).sum();

            ut.push(d_t);
            u_vals.push(u_curr[p]);
            lap_hd.push(lap_p);
            grad_sq.push(g_sq_p);
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push(format!("laplacian_{}d(u)", dimension));
    columns.push(lap_hd.clone());

    term_names.push("|grad_u|^2".to_string());
    columns.push(grad_sq);

    term_names.push(format!("u*laplacian_{}d(u)", dimension));
    columns.push(u_vals.iter().zip(lap_hd.iter()).map(|(&u, &l)| u * l).collect());

    let method = format!("meshfree_rbf_point_cloud_{}d", dimension);
    run_stridge_with_frozen_holdout(&ut, &columns, &term_names, config, n_int_t, dimension, method)
}

/// Discover D-dimensional PDE via Mesh-Free Axis-Star Point Cloud Sampling.
///
/// Uses M anchor points in R^D sampled via Lagrangian or Monte Carlo methods.
/// Around each anchor x_p, probe vectors along coordinate directions x_p \pm h * e_k
/// compute the exact partial derivatives \partial_{x_k} u and \partial^2_{x_k x_k} u,
/// evaluating \nabla^2_D u = \sum_{k=1}^D u_{x_k x_k} and |\nabla_D u|^2.
/// Requires only M * (2D + 1) points in total, completely bypassing Cartesian N^D grids.
pub fn discover_pde_point_cloud_hd(
    u_anchors: &[Vec<f64>],            // [N_t][M]
    u_probes_plus: &[Vec<Vec<f64>>],   // [N_t][M][D] (value at x_p + h * e_k)
    u_probes_minus: &[Vec<Vec<f64>>],  // [N_t][M][D] (value at x_p - h * e_k)
    dt: f64,
    h: f64,
    dimension: usize,
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    let nt = u_anchors.len();
    if nt < 5 {
        return Err("Point cloud HD PDE requires at least 5 time steps".to_string());
    }
    let m = u_anchors[0].len();
    if m < 5 {
        return Err("Point cloud HD PDE requires at least 5 anchor points".to_string());
    }

    let dt = dt.max(1e-12);
    let h = h.max(1e-12);
    let h_sq = h * h;

    let mut ut = Vec::new();
    let mut u_vals = Vec::new();
    let mut lap_hd = Vec::new();
    let mut grad_sq = Vec::new();

    for t in 1..(nt - 1) {
        for p in 0..m {
            let u_curr = u_anchors[t][p];
            let d_t = (u_anchors[t + 1][p] - u_anchors[t - 1][p]) / (2.0 * dt);

            let mut lap = 0.0f64;
            let mut g_sq = 0.0f64;

            for k in 0..dimension {
                let u_plus = u_probes_plus[t][p][k];
                let u_minus = u_probes_minus[t][p][k];

                let d_xk = (u_plus - u_minus) / (2.0 * h);
                let d_xkxk = (u_plus - 2.0 * u_curr + u_minus) / h_sq;

                lap += d_xkxk;
                g_sq += d_xk * d_xk;
            }

            ut.push(d_t);
            u_vals.push(u_curr);
            lap_hd.push(lap);
            grad_sq.push(g_sq);
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push(format!("laplacian_{}d(u)", dimension));
    columns.push(lap_hd.clone());

    term_names.push("|grad_u|^2".to_string());
    columns.push(grad_sq);

    term_names.push(format!("u*laplacian_{}d(u)", dimension));
    columns.push(u_vals.iter().zip(lap_hd.iter()).map(|(&u, &l)| u * l).collect());

    let method = format!("meshfree_point_cloud_{}d", dimension);
    run_stridge_with_frozen_holdout(&ut, &columns, &term_names, config, nt - 2, dimension, method)
}

/// Discover D-dimensional PDE via Low-Rank Tensor-Train (TT / MPS) Decomposition (Technique 3).
///
/// Represents the D-dimensional field as a rank-R Tensor Train / Matrix Product State:
///   u(t, x_1, ..., x_D) = \sum_{r=1}^R \prod_{k=1}^D \Phi^{(k)}_r(t, x_k)
/// where `tt_cores_flat` has shape `[N_t * R * D * N_x]`.
/// Memory scales linearly in dimension D as O(N_t * D * N_x * R) instead of exponential O(N_x^D).
/// Evaluates 4th-order 1D stencils on each core and contracts prefix/suffix products across
/// interior fiber samples to compute the exact D-dimensional Laplacian and gradient norm.
pub fn discover_pde_tensor_train_hd(
    tt_cores_flat: &[f64], // [N_t * R * D * N_x]
    nt: usize,
    bond_rank: usize,
    dimension: usize,
    nx: usize,
    dt: f64,
    dx: f64,
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    if nt < 5 {
        return Err("Tensor-Train HD PDE requires at least 5 time steps".to_string());
    }
    if bond_rank < 1 || dimension < 1 || nx < 7 {
        return Err("Tensor-Train HD PDE requires bond_rank >= 1, dimension >= 1, nx >= 7".to_string());
    }
    if tt_cores_flat.len() != nt * bond_rank * dimension * nx {
        return Err("tt_cores_flat length mismatch with nt * bond_rank * dimension * nx".to_string());
    }

    let dt = dt.max(1e-12);
    let dx = dx.max(1e-12);
    let n_int_x = nx - 4;

    // Build deterministic multi-index evaluation fibers across [2..nx-2]^D:
    // Includes diagonal fibers (i, i, ..., i) and shifted Halton-like multi-index fibers
    let mut fibers: Vec<Vec<usize>> = Vec::with_capacity(n_int_x * 2);
    for i in 2..(nx - 2) {
        fibers.push(vec![i; dimension]);
        let mut shifted = Vec::with_capacity(dimension);
        for k in 0..dimension {
            let idx = 2 + ((i - 2 + k * 3) % n_int_x);
            shifted.push(idx);
        }
        fibers.push(shifted);
    }

    let core_idx = |t: usize, r: usize, k: usize, ix: usize| -> usize {
        (((t * bond_rank + r) * dimension + k) * nx) + ix
    };

    let eval_u_at = |t: usize, fiber: &[usize]| -> f64 {
        let mut total = 0.0f64;
        for r in 0..bond_rank {
            let mut prod = 1.0f64;
            for k in 0..dimension {
                prod *= tt_cores_flat[core_idx(t, r, k, fiber[k])];
            }
            total += prod;
        }
        total
    };

    let n_int_t = nt - 2;
    let total_samples = n_int_t * fibers.len();
    let mut ut = Vec::with_capacity(total_samples);
    let mut u_vals = Vec::with_capacity(total_samples);
    let mut lap_hd = Vec::with_capacity(total_samples);
    let mut grad_sq = Vec::with_capacity(total_samples);

    for t in 1..(nt - 1) {
        for fiber in &fibers {
            let u_curr = eval_u_at(t, fiber);
            let u_prev = eval_u_at(t - 1, fiber);
            let u_next = eval_u_at(t + 1, fiber);
            let d_t = (u_next - u_prev) / (2.0 * dt);

            let mut lap_val = 0.0f64;
            let mut grad_vec = vec![0.0f64; dimension];

            for r in 0..bond_rank {
                // Prefix and suffix products of \Phi^{(k)}_r(t, i_k) avoid division by zero
                let mut vals = vec![0.0f64; dimension];
                let mut d1_vals = vec![0.0f64; dimension];
                let mut d2_vals = vec![0.0f64; dimension];

                for k in 0..dimension {
                    let ix = fiber[k];
                    let c0 = tt_cores_flat[core_idx(t, r, k, ix)];
                    let cp1 = tt_cores_flat[core_idx(t, r, k, ix + 1)];
                    let cp2 = tt_cores_flat[core_idx(t, r, k, ix + 2)];
                    let cm1 = tt_cores_flat[core_idx(t, r, k, ix - 1)];
                    let cm2 = tt_cores_flat[core_idx(t, r, k, ix - 2)];

                    vals[k] = c0;
                    d1_vals[k] = (-cp2 + 8.0 * cp1 - 8.0 * cm1 + cm2) / (12.0 * dx);
                    d2_vals[k] = (-cp2 + 16.0 * cp1 - 30.0 * c0 + 16.0 * cm1 - cm2) / (12.0 * dx * dx);
                }

                let mut prefix = vec![1.0f64; dimension];
                let mut suffix = vec![1.0f64; dimension];
                for k in 1..dimension {
                    prefix[k] = prefix[k - 1] * vals[k - 1];
                }
                for k in (0..(dimension - 1)).rev() {
                    suffix[k] = suffix[k + 1] * vals[k + 1];
                }

                for k in 0..dimension {
                    let prod_others = prefix[k] * suffix[k];
                    lap_val += d2_vals[k] * prod_others;
                    grad_vec[k] += d1_vals[k] * prod_others;
                }
            }

            let g_sq: f64 = grad_vec.iter().map(|&g| g * g).sum();
            ut.push(d_t);
            u_vals.push(u_curr);
            lap_hd.push(lap_val);
            grad_sq.push(g_sq);
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push(format!("laplacian_{}d(u)", dimension));
    columns.push(lap_hd);

    term_names.push("|grad_u|^2".to_string());
    columns.push(grad_sq);

    let method = format!("tensor_train_mps_{}d", dimension);
    run_stridge_with_frozen_holdout(&ut, &columns, &term_names, config, n_int_t, dimension, method)
}

/// Discover D-dimensional PDE via Deep BSDE / Neural PINN Jet2 Autodiff (Technique 4).
///
/// Evaluates a neural network parameterization u_\theta(t, x) = \sum_{j=1}^H w2_j(t) \tanh((W1 x)_j + b1_j)
/// at mesh-free collocation points X \in R^{M x D}. Uses Fused 2-Jet Taylor-Mode Automatic
/// Differentiation (`eval_jet2_neural_pde_operator`) to compute exact analytical spatial
/// gradients \nabla_x u_\theta and exact analytical D-dimensional Laplacians \Delta_D u_\theta
/// without constructing any spatial mesh, then applies STRidge to recover the governing PDE.
pub fn discover_pde_neural_pinn_hd(
    x_collocation_flat: &[f64], // [M * D] mesh-free collocation points in R^D
    w1_flat: &[f64],            // [H * D] hidden layer weights
    b1: &[f64],                 // [H] hidden layer biases
    w2_time_flat: &[f64],       // [N_t * H] time-indexed readout weights w2(t)
    nt: usize,
    m: usize,
    dimension: usize,
    h: usize,
    dt: f64,
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    if nt < 5 {
        return Err("Neural PINN HD PDE requires at least 5 time steps".to_string());
    }
    if m < 5 || dimension < 1 || h < 1 {
        return Err("Invalid (m, dimension, h) for Neural PINN HD PDE".to_string());
    }
    if x_collocation_flat.len() != m * dimension {
        return Err("x_collocation_flat length mismatch".to_string());
    }
    if w1_flat.len() != h * dimension || b1.len() != h {
        return Err("w1_flat / b1 length mismatch".to_string());
    }
    if w2_time_flat.len() != nt * h {
        return Err("w2_time_flat length mismatch with nt * h".to_string());
    }

    let dt = dt.max(1e-12);

    // Evaluate Jet2 operator across all time steps
    let mut snapshots = Vec::with_capacity(nt);
    for t in 0..nt {
        let w2_t = &w2_time_flat[t * h..(t + 1) * h];
        let snap = eval_jet2_neural_pde_operator(
            x_collocation_flat,
            w1_flat,
            b1,
            w2_t,
            m,
            dimension,
            h,
            0.0,
        )?;
        snapshots.push(snap);
    }

    let n_int_t = nt - 2;
    let total_samples = n_int_t * m;
    let mut ut = Vec::with_capacity(total_samples);
    let mut u_vals = Vec::with_capacity(total_samples);
    let mut lap_hd = Vec::with_capacity(total_samples);
    let mut grad_sq = Vec::with_capacity(total_samples);

    for t in 1..(nt - 1) {
        for p in 0..m {
            let d_t = (snapshots[t + 1].u[p] - snapshots[t - 1].u[p]) / (2.0 * dt);
            ut.push(d_t);
            u_vals.push(snapshots[t].u[p]);
            lap_hd.push(snapshots[t].laplacian[p]);
            grad_sq.push(snapshots[t].grad_norm_sq[p]);
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push(format!("laplacian_{}d(u)", dimension));
    columns.push(lap_hd.clone());

    term_names.push("|grad_u|^2".to_string());
    columns.push(grad_sq);

    term_names.push(format!("u*laplacian_{}d(u)", dimension));
    columns.push(u_vals.iter().zip(lap_hd.iter()).map(|(&u, &l)| u * l).collect());

    let method = format!("neural_pinn_bsde_jet2_{}d", dimension);
    run_stridge_with_frozen_holdout(&ut, &columns, &term_names, config, n_int_t, dimension, method)
}

/// Discover D-dimensional PDE via Separable Coordinate-Factored Decomposition.
///
/// For fields factored as u(t, x_1, ..., x_D) = \prod_{k=1}^D \phi(t, x_k),
/// the D-dimensional Laplacian is evaluated via 1D factor slices:
/// \nabla^2_D u = D * \phi^{D-1} * \phi_{xx}
/// Requires only O(D * N_x) points instead of N^D.
pub fn discover_pde_separable_hd(
    phi_grid: &[Vec<f64>], // [N_t][N_x] for 1D slice
    dt: f64,
    dx: f64,
    dimension: usize,
    config: &PdeConfig,
) -> Result<HighDimPdeResult, String> {
    let nt = phi_grid.len();
    if nt < 5 {
        return Err("Separable HD PDE requires at least 5 time steps".to_string());
    }
    let nx = phi_grid[0].len();
    if nx < 5 {
        return Err("Separable HD PDE requires at least 5 spatial points".to_string());
    }

    let dt = dt.max(1e-12);
    let dx = dx.max(1e-12);
    let dim_f = dimension as f64;

    let mut ut = Vec::new();
    let mut u_vals = Vec::new();
    let mut lap_hd = Vec::new();

    for t in 1..(nt - 1) {
        for i in 1..(nx - 1) {
            let phi_mid = phi_grid[t][i];
            if phi_mid.abs() < 1e-12 {
                continue;
            }

            let phi_t = (phi_grid[t + 1][i] - phi_grid[t - 1][i]) / (2.0 * dt);
            let phi_xx = (phi_grid[t][i + 1] - 2.0 * phi_mid + phi_grid[t][i - 1]) / (dx * dx);

            // u = phi^D
            let u_val = phi_mid.powi(dimension as i32);
            // u_t = D * phi^{D-1} * phi_t
            let u_t = dim_f * phi_mid.powi((dimension - 1) as i32) * phi_t;
            // \nabla^2_D u = D * phi^{D-1} * phi_xx
            let lap = dim_f * phi_mid.powi((dimension - 1) as i32) * phi_xx;

            ut.push(u_t);
            u_vals.push(u_val);
            lap_hd.push(lap);
        }
    }

    let mut term_names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    term_names.push("u".to_string());
    columns.push(u_vals.clone());

    if config.max_polynomial_degree >= 2 {
        term_names.push("u^2".to_string());
        columns.push(u_vals.iter().map(|&v| v * v).collect());
    }

    term_names.push(format!("laplacian_{}d(u)", dimension));
    columns.push(lap_hd);

    let method = format!("separable_tensor_{}d", dimension);
    run_stridge_with_frozen_holdout(&ut, &columns, &term_names, config, nt - 2, dimension, method)
}


/// Unified AVX2/Rayon SIMD-fused Jet2 Curvature Video Flow Step:
/// Computes total convective trajectory acceleration:
///   a_t = \partial_t v + (\nabla_z v) * v = (v_t_plus - v_t_minus) / dt_span + (v_spatial_plus - v_spatial_minus) / (2 * h_disp)
/// and applies the second-order Taylor ODE update:
///   z_{t + \Delta t} = z_t + \Delta t * v_t + 0.5 * (\Delta t)^2 * a_t
/// in a single parallelized pass over L1/L2 cache blocks.
pub fn jet2_video_flow_step(
    z_flat: &[f64],
    v_current: &[f64],
    v_t_plus: &[f64],
    v_t_minus: &[f64],
    v_spatial_plus: &[f64],
    v_spatial_minus: &[f64],
    dt: f64,
    dt_span: f64,
    two_h_disp: f64,
    order: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = z_flat.len();
    if v_current.len() != n {
        return Err(format!("v_current length {} != z_flat length {}", v_current.len(), n));
    }
    // dt is a *signed* step: backward trajectory inversion (dt < 0) must be
    // accepted. The step is nonlinear-in-dt through 0.5*dt^2 (sign-invariant)
    // and linear through dt*v, so only non-finite or zero dt is invalid.
    if !dt.is_finite() || dt == 0.0 {
        return Err("dt must be finite and non-zero".to_string());
    }

    if order == 1 {
        // First-order Euler step: z_{t+dt} = z_t + dt * v_t
        let mut z_next = vec![0.0f64; n];
        const CHUNK: usize = 1024;
        z_next
            .par_chunks_mut(CHUNK)
            .zip(z_flat.par_chunks(CHUNK))
            .zip(v_current.par_chunks(CHUNK))
            .for_each(|((zn_c, z_c), v_c)| {
                for i in 0..zn_c.len() {
                    zn_c[i] = z_c[i] + dt * v_c[i];
                }
            });
        return Ok((z_next, vec![0.0f64; n]));
    }

    // Second-order Jet2 Taylor step with acceleration
    if v_t_plus.len() != n || v_t_minus.len() != n || v_spatial_plus.len() != n || v_spatial_minus.len() != n {
        return Err("Directional velocity arrays must have length equal to z_flat".to_string());
    }
    // dt_span carries the SIGNED temporal probe separation and two_h_disp the
    // SIGNED spatial probe separation: a directional difference divided by its
    // (signed) span is the derivative in that direction, so rejecting one side
    // silently forbade backward flows. Only non-finite/zero spans are invalid.
    if !dt_span.is_finite() || dt_span == 0.0 || !two_h_disp.is_finite() || two_h_disp == 0.0 {
        return Err("dt_span and two_h_disp must be finite and non-zero".to_string());
    }

    let inv_dt_span = 1.0 / dt_span;
    let inv_two_h = 1.0 / two_h_disp;
    let half_dt_sq = 0.5 * dt * dt;

    let mut z_next = vec![0.0f64; n];
    let mut a_out = vec![0.0f64; n];

    const CHUNK: usize = 1024;
    z_next
        .par_chunks_mut(CHUNK)
        .zip(a_out.par_chunks_mut(CHUNK))
        .zip(z_flat.par_chunks(CHUNK))
        .zip(v_current.par_chunks(CHUNK))
        .zip(v_t_plus.par_chunks(CHUNK))
        .zip(v_t_minus.par_chunks(CHUNK))
        .zip(v_spatial_plus.par_chunks(CHUNK))
        .zip(v_spatial_minus.par_chunks(CHUNK))
        .for_each(|(((((((zn_c, a_c), z_c), vc_c), vtp_c), vtm_c), vsp_c), vsm_c)| {
            for i in 0..zn_c.len() {
                let a_temporal = (vtp_c[i] - vtm_c[i]) * inv_dt_span;
                let a_convective = (vsp_c[i] - vsm_c[i]) * inv_two_h;
                let a_tot = a_temporal + a_convective;
                a_c[i] = a_tot;
                zn_c[i] = z_c[i] + dt * vc_c[i] + half_dt_sq * a_tot;
            }
        });

    Ok((z_next, a_out))
}

/// Helmholtz-Hodge Divergence-Free Solenoidal Projection in 2D.
/// Decomposes any 2D vector field v = (vx, vy) into:
///   v = v_sol + \nabla \phi,  where \nabla \cdot v_sol = 0
/// by solving the Poisson equation \nabla^2 \phi = \nabla \cdot v
/// using Red-Black Gauss-Seidel relaxation.
pub fn helmholtz_hodge_div_free_2d(
    vx: &[f64],
    vy: &[f64],
    height: usize,
    width: usize,
    dx: f64,
    dy: f64,
    num_iter: usize,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let n = height * width;
    if vx.len() != n || vy.len() != n {
        return Err(format!("vx len {} and vy len {} must equal height * width = {}", vx.len(), vy.len(), n));
    }
    if height < 3 || width < 3 {
        return Err("Grid dimensions must be at least 3x3".to_string());
    }
    if dx <= 0.0 || dy <= 0.0 {
        return Err("dx and dy must be positive".to_string());
    }

    let inv_2dx = 1.0 / (2.0 * dx);
    let inv_2dy = 1.0 / (2.0 * dy);
    let dx_sq = dx * dx;
    let dy_sq = dy * dy;
    let denom = 2.0 * (1.0 / dx_sq + 1.0 / dy_sq);
    let inv_denom = 1.0 / denom;

    // 1. Compute divergence field: div = dvx/dx + dvy/dy
    let mut div = vec![0.0f64; n];
    let mut sum_div = 0.0f64;
    for y in 0..height {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };
        let row_curr = y * width;
        let row_m = ym * width;
        let row_p = yp * width;
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let xp = if x + 1 == width { 0 } else { x + 1 };

            let dvx = (vx[row_curr + xp] - vx[row_curr + xm]) * inv_2dx;
            let dvy = (vy[row_p + x] - vy[row_m + x]) * inv_2dy;
            let d = dvx + dvy;
            div[row_curr + x] = d;
            sum_div += d;
        }
    }
    // Zero-mean compatibility condition for Poisson solver on periodic domain
    let mean_div = sum_div / (n as f64);
    for d in div.iter_mut() {
        *d -= mean_div;
    }

    // 2. Solve Poisson equation \nabla^2 phi = div via Red-Black SOR (Successive Over-Relaxation)
    let pi = std::f64::consts::PI;
    let n_max = (height.max(width)) as f64;
    let omega = (2.0 / (1.0 + (pi / n_max).sin())).min(1.92);

    let mut phi = vec![0.0f64; n];

    for _ in 0..num_iter {
        // Red phase: (x + y) % 2 == 0
        for y in 0..height {
            let ym = if y == 0 { height - 1 } else { y - 1 };
            let yp = if y + 1 == height { 0 } else { y + 1 };
            let row_curr = y * width;
            let row_m = ym * width;
            let row_p = yp * width;
            for x in ((y % 2)..width).step_by(2) {
                let xm = if x == 0 { width - 1 } else { x - 1 };
                let xp = if x + 1 == width { 0 } else { x + 1 };

                let lap_neighbors = (phi[row_curr + xp] + phi[row_curr + xm]) / dx_sq
                    + (phi[row_p + x] + phi[row_m + x]) / dy_sq;
                let phi_target = (lap_neighbors - div[row_curr + x]) * inv_denom;
                phi[row_curr + x] = (1.0 - omega) * phi[row_curr + x] + omega * phi_target;
            }
        }
        // Black phase: (x + y) % 2 == 1
        for y in 0..height {
            let ym = if y == 0 { height - 1 } else { y - 1 };
            let yp = if y + 1 == height { 0 } else { y + 1 };
            let row_curr = y * width;
            let row_m = ym * width;
            let row_p = yp * width;
            for x in (((y + 1) % 2)..width).step_by(2) {
                let xm = if x == 0 { width - 1 } else { x - 1 };
                let xp = if x + 1 == width { 0 } else { x + 1 };

                let lap_neighbors = (phi[row_curr + xp] + phi[row_curr + xm]) / dx_sq
                    + (phi[row_p + x] + phi[row_m + x]) / dy_sq;
                let phi_target = (lap_neighbors - div[row_curr + x]) * inv_denom;
                phi[row_curr + x] = (1.0 - omega) * phi[row_curr + x] + omega * phi_target;
            }
        }
    }

    // 3. Subtract gradient of phi: v_sol = v - \nabla phi
    let mut vx_sol = vec![0.0f64; n];
    let mut vy_sol = vec![0.0f64; n];
    let mut rem_div_sum_sq = 0.0f64;

    for y in 0..height {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };
        let row_curr = y * width;
        let row_m = ym * width;
        let row_p = yp * width;
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let xp = if x + 1 == width { 0 } else { x + 1 };

            let grad_phi_x = (phi[row_curr + xp] - phi[row_curr + xm]) * inv_2dx;
            let grad_phi_y = (phi[row_p + x] - phi[row_m + x]) * inv_2dy;

            let vxs = vx[row_curr + x] - grad_phi_x;
            let vys = vy[row_curr + x] - grad_phi_y;
            vx_sol[row_curr + x] = vxs;
            vy_sol[row_curr + x] = vys;
        }
    }

    // 4. Compute remaining divergence norm
    for y in 0..height {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };
        let row_curr = y * width;
        let row_m = ym * width;
        let row_p = yp * width;
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let xp = if x + 1 == width { 0 } else { x + 1 };

            let rem_div_x = (vx_sol[row_curr + xp] - vx_sol[row_curr + xm]) * inv_2dx;
            let rem_div_y = (vy_sol[row_p + x] - vy_sol[row_m + x]) * inv_2dy;
            let rem_div = rem_div_x + rem_div_y;
            rem_div_sum_sq += rem_div * rem_div;
        }
    }

    let rem_div_norm = (rem_div_sum_sq / (n as f64)).sqrt();
    Ok((vx_sol, vy_sol, rem_div_norm))
}

/// Native 2D/3D Multi-Channel Laplacian Dissipation Step:
///   Z_{new} = Z + dt * nu * \nabla^2 Z
/// Parallelized over channels and row blocks using Rayon.
pub fn laplacian_dissipation_2d(
    field: &[f64],
    height: usize,
    width: usize,
    channels: usize,
    nu: f64,
    dt: f64,
    dx: f64,
    dy: f64,
) -> Result<Vec<f64>, String> {
    let plane_size = height * width;
    let total_len = channels * plane_size;
    if field.len() != total_len {
        return Err(format!("Field length {} != channels {} * height {} * width {}", field.len(), channels, height, width));
    }
    if height < 3 || width < 3 {
        return Err("Grid dimensions must be at least 3x3".to_string());
    }
    if dx <= 0.0 || dy <= 0.0 || dt <= 0.0 {
        return Err("dx, dy, dt must be positive".to_string());
    }
    // Negative viscosity turns the Laplacian term anti-diffusive: a backward
    // heat equation in which every high-frequency component grows without
    // bound. Non-finite nu guards NaN propagation through the grid as well.
    if !nu.is_finite() || nu < 0.0 {
        return Err("viscosity/diffusion nu must be finite and non-negative".to_string());
    }
    // Explicit explicit-scheme stability: for the discrete update
    // z += nu*dt*(lap_dx + lap_dy) the amplification magnitude stays <= 1
    // only under the diffusion CFL limit nu*dt*(1/dx^2 + 1/dy^2) <= 1/2.
    // Crossing it makes every step a noise amplifier instead of a smoother.
    let dx_sq = dx * dx;
    let dy_sq = dy * dy;
    let cfl = nu * dt * (1.0 / dx_sq + 1.0 / dy_sq);
    if cfl > 0.5 {
        return Err(format!(
            "diffusion CFL violated: nu*dt*(1/dx^2 + 1/dy^2) = {:.6} > 0.5 (reduce dt or nu, or increase dx/dy)",
            cfl
        ));
    }
    let factor = nu * dt;

    let mut out = vec![0.0f64; total_len];

    out.par_chunks_mut(plane_size)
        .zip(field.par_chunks(plane_size))
        .for_each(|(out_c, in_c)| {
            for y in 0..height {
                let ym = if y == 0 { height - 1 } else { y - 1 };
                let yp = if y + 1 == height { 0 } else { y + 1 };
                let row_curr = y * width;
                let row_m = ym * width;
                let row_p = yp * width;
                for x in 0..width {
                    let xm = if x == 0 { width - 1 } else { x - 1 };
                    let xp = if x + 1 == width { 0 } else { x + 1 };

                    let d2x = (in_c[row_curr + xp] - 2.0 * in_c[row_curr + x] + in_c[row_curr + xm]) / dx_sq;
                    let d2y = (in_c[row_p + x] - 2.0 * in_c[row_curr + x] + in_c[row_m + x]) / dy_sq;
                    let lap = d2x + d2y;
                    out_c[row_curr + x] = in_c[row_curr + x] + factor * lap;
                }
            }
        });

    Ok(out)
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_20d_hyperspherical_radial_diffusion() {
        let dim = 20;
        let alpha = 0.5;
        let nt = 25;
        let nr = 30;
        let dt = 0.005;
        let dr = 0.05;

        let mut r_coords = Vec::new();
        for ri in 0..nr {
            r_coords.push(0.2 + (ri as f64) * dr);
        }

        // True 20D heat diffusion Gaussian pulse: u(t, r) = (1/t)^(D/2) * exp(-r^2 / (4*alpha*t))
        let mut u_grid = vec![vec![0.0f64; nr]; nt];
        for ti in 0..nt {
            let t = 1.0 + (ti as f64) * dt;
            let scale = (1.0 / t).powf(dim as f64 / 2.0);
            for ri in 0..nr {
                let r = r_coords[ri];
                u_grid[ti][ri] = scale * (-r * r / (4.0 * alpha * t)).exp();
            }
        }

        let config = PdeConfig {
            sparsity_threshold: 0.01,
            ridge_alpha: 1e-6,
            ..Default::default()
        };

        let res = discover_pde_radial_hd(&r_coords, &u_grid, dt, dr, dim, &config)
            .expect("radial 20d discovery failed");

        assert!(res.r2 > 0.99, "20D radial R2 = {}", res.r2);
        assert!(res.is_exact_recovery);
        assert_eq!(res.dimension, 20);
        assert!(res.formula.contains("laplacian_20d(u)"));
        // Coeff should be close to alpha = 0.5
        let lap_idx = res.active_terms.iter().position(|t| t == "laplacian_20d(u)").unwrap();
        let lap_coeff = res.coefficients[lap_idx];
        assert!((lap_coeff - 0.5).abs() < 0.05, "Lap coeff = {}", lap_coeff);
    }

    #[test]
    fn test_20d_point_cloud_diffusion() {
        let dim = 20;
        let alpha = 0.5;
        let nt = 12;
        let m = 20; // 20 anchor points
        let dt = 0.005;
        let h = 0.04;

        // Generate synthetic anchors in 20D
        let mut anchors = vec![vec![0.0f64; dim]; m];
        for p in 0..m {
            for k in 0..dim {
                anchors[p][k] = ((p * 7 + k * 13) % 100) as f64 * 0.005 - 0.25;
            }
        }

        let eval_gaussian = |t: f64, pt: &[f64]| -> f64 {
            let scale = (1.0 / t).powf(dim as f64 / 2.0);
            let r2: f64 = pt.iter().map(|&x| x * x).sum();
            scale * (-r2 / (4.0 * alpha * t)).exp()
        };

        let mut u_anchors = vec![vec![0.0f64; m]; nt];
        let mut u_plus = vec![vec![vec![0.0f64; dim]; m]; nt];
        let mut u_minus = vec![vec![vec![0.0f64; dim]; m]; nt];

        for ti in 0..nt {
            let t = 1.0 + (ti as f64) * dt;
            for p in 0..m {
                u_anchors[ti][p] = eval_gaussian(t, &anchors[p]);

                for k in 0..dim {
                    let mut pt_plus = anchors[p].clone();
                    pt_plus[k] += h;
                    u_plus[ti][p][k] = eval_gaussian(t, &pt_plus);

                    let mut pt_minus = anchors[p].clone();
                    pt_minus[k] -= h;
                    u_minus[ti][p][k] = eval_gaussian(t, &pt_minus);
                }
            }
        }

        let config = PdeConfig {
            sparsity_threshold: 0.01,
            ridge_alpha: 1e-6,
            ..Default::default()
        };

        let res = discover_pde_point_cloud_hd(&u_anchors, &u_plus, &u_minus, dt, h, dim, &config)
            .expect("point cloud 20d discovery failed");

        assert!(res.r2 > 0.99, "20D point cloud R2 = {}", res.r2);
        assert!(res.is_exact_recovery);
        assert_eq!(res.dimension, 20);
        assert!(res.formula.contains("laplacian_20d(u)"));
        let lap_idx = res.active_terms.iter().position(|t| t == "laplacian_20d(u)").unwrap();
        let lap_coeff = res.coefficients[lap_idx];
        assert!((lap_coeff - 0.5).abs() < 0.05, "Lap coeff = {}", lap_coeff);
    }

    #[test]
    fn test_3d_cartesian_diffusion() {
        let alpha = 0.3;
        let nt = 10;
        let nx = 8;
        let ny = 8;
        let nz = 8;
        let dt = 0.005;
        let dx = 0.1;
        let dy = 0.1;
        let dz = 0.1;

        let mut u_grid = vec![vec![vec![vec![0.0f64; nz]; ny]; nx]; nt];
        for t in 0..nt {
            let decay = (-3.0 * alpha * (t as f64) * dt).exp();
            for i in 0..nx {
                let px = (i as f64) * dx;
                for j in 0..ny {
                    let py = (j as f64) * dy;
                    for k in 0..nz {
                        let pz = (k as f64) * dz;
                        u_grid[t][i][j][k] = decay * px.sin() * py.sin() * pz.sin();
                    }
                }
            }
        }

        let config = PdeConfig {
            sparsity_threshold: 0.01,
            ridge_alpha: 1e-6,
            ..Default::default()
        };

        let res = discover_pde_3d_cartesian(&u_grid, dt, dx, dy, dz, &config)
            .expect("3D cartesian discovery failed");

        assert!(res.r2 > 0.98, "3D cartesian R2 = {}", res.r2);
        assert_eq!(res.dimension, 3);
        assert_eq!(res.method_used, "eulerian_cartesian_3d");
    }

    #[test]
    fn test_20d_separable_diffusion() {
        let dim = 20;
        let alpha = 0.5;
        let nt = 15;
        let nx = 25;
        let dt = 0.005;
        let dx = 0.1;

        let mut phi_grid = vec![vec![0.0f64; nx]; nt];
        for ti in 0..nt {
            let t = 1.0 + (ti as f64) * dt;
            let scale = 1.0 / t.sqrt();
            for xi in 0..nx {
                let x = -1.2 + (xi as f64) * dx;
                phi_grid[ti][xi] = scale * (-x * x / (4.0 * alpha * t)).exp();
            }
        }

        let config = PdeConfig {
            sparsity_threshold: 0.01,
            ridge_alpha: 1e-6,
            ..Default::default()
        };

        let res = discover_pde_separable_hd(&phi_grid, dt, dx, dim, &config)
            .expect("separable 20d discovery failed");

        assert!(res.r2 > 0.99, "20D separable R2 = {}", res.r2);
        assert_eq!(res.dimension, 20);
        assert_eq!(res.method_used, "separable_tensor_20d");
        assert!(res.formula.contains("laplacian_20d(u)"));
    }

    #[test]
    fn test_jet2_20d_neural_pde_operator_exact_laplacian() {
        let m = 8;
        let d = 20;
        let h = 16;
        let nu = 0.25;

        let mut x_flat = vec![0.0f64; m * d];
        for i in 0..(m * d) {
            x_flat[i] = ((i * 17 + 5) % 97) as f64 * 0.01 - 0.48;
        }
        let mut w1_flat = vec![0.0f64; h * d];
        for i in 0..(h * d) {
            w1_flat[i] = ((i * 13 + 3) % 89) as f64 * 0.005 - 0.22;
        }
        let mut b1 = vec![0.0f64; h];
        let mut w2 = vec![0.0f64; h];
        for j in 0..h {
            b1[j] = (j as f64) * 0.03 - 0.2;
            w2[j] = ((j * 7 + 1) % 31) as f64 * 0.04 - 0.6;
        }

        let out = eval_jet2_neural_pde_operator(&x_flat, &w1_flat, &b1, &w2, m, d, h, nu)
            .expect("jet2 operator failed");

        // Verify against central finite differences on each coordinate k in 20D (eps=1e-4 balances O(eps^2) truncation and O(ulp/eps^2) cancellation)
        let eps = 1e-4;
        let eval_u = |pt: &[f64]| -> f64 {
            let mut sum = 0.0;
            for j in 0..h {
                let mut z = b1[j];
                for k in 0..d {
                    z += w1_flat[j * d + k] * pt[k];
                }
                sum += w2[j] * z.tanh();
            }
            sum
        };

        for p in 0..m {
            let base = &x_flat[p * d..(p + 1) * d];
            let u0 = eval_u(base);
            assert!((out.u[p] - u0).abs() < 1e-14);

            let mut fd_lap = 0.0;
            let mut fd_gsq = 0.0;
            for k in 0..d {
                let mut xp = base.to_vec();
                let mut xm = base.to_vec();
                xp[k] += eps;
                xm[k] -= eps;
                let up = eval_u(&xp);
                let um = eval_u(&xm);
                let gx = (up - um) / (2.0 * eps);
                let gxx = (up - 2.0 * u0 + um) / (eps * eps);
                fd_gsq += gx * gx;
                fd_lap += gxx;
            }
            assert!(
                (out.grad_norm_sq[p] - fd_gsq).abs() < 1e-7,
                "grad_sq mismatch: {} vs {}",
                out.grad_norm_sq[p],
                fd_gsq
            );
            assert!(
                (out.laplacian[p] - fd_lap).abs() < 1e-6,
                "laplacian mismatch: {} vs {}",
                out.laplacian[p],
                fd_lap
            );
        }
    }

    #[test]
    fn test_vlasov_attention_pde_hd_online_softmax() {
        let n = 16;
        let d = 20;
        let steps = 5;
        let dt = 0.05;
        let damping = 0.1;

        let mut x_init = vec![0.0f64; n * d];
        for i in 0..(n * d) {
            x_init[i] = ((i * 23 + 7) % 101) as f64 * 0.02 - 1.0;
        }

        let evolved = step_vlasov_attention_pde_hd(&x_init, n, d, steps, dt, damping)
            .expect("vlasov stepper failed");
        assert_eq!(evolved.len(), n * d);
        assert!(evolved.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_jet3_pde_loss_and_grads_exact_finite_diff() {
        let m = 6;
        let d = 8;
        let h = 5;
        let nu = 0.35;
        let beta = 0.6;
        let mu = 0.25;

        let mut x = vec![0.0f64; m * d];
        for i in 0..(m * d) {
            x[i] = ((i * 19 + 3) % 71) as f64 * 0.015 - 0.5;
        }
        let mut target_f = vec![0.0f64; m];
        for p in 0..m {
            target_f[p] = (p as f64) * 0.1 - 0.25;
        }
        let mut w1 = vec![0.0f64; h * d];
        for i in 0..(h * d) {
            w1[i] = ((i * 11 + 7) % 53) as f64 * 0.01 - 0.26;
        }
        let mut b1 = vec![0.0f64; h];
        let mut w2 = vec![0.0f64; h];
        for j in 0..h {
            b1[j] = (j as f64) * 0.05 - 0.1;
            w2[j] = ((j * 13 + 2) % 29) as f64 * 0.03 - 0.4;
        }

        let out = compute_jet3_pde_loss_and_grads(&x, &target_f, &w1, &b1, &w2, m, d, h, nu, beta, mu)
            .expect("jet3 grads failed");

        let eps = 1e-6;
        // Check every entry of W1
        for idx in 0..(h * d) {
            let mut w1_p = w1.clone();
            let mut w1_m = w1.clone();
            w1_p[idx] += eps;
            w1_m[idx] -= eps;
            let lp = compute_jet3_pde_loss_and_grads(&x, &target_f, &w1_p, &b1, &w2, m, d, h, nu, beta, mu).unwrap().loss;
            let lm = compute_jet3_pde_loss_and_grads(&x, &target_f, &w1_m, &b1, &w2, m, d, h, nu, beta, mu).unwrap().loss;
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (out.grad_w1[idx] - fd).abs() < 1e-8,
                "grad_w1[{}] mismatch: {} vs {}",
                idx,
                out.grad_w1[idx],
                fd
            );
        }
        // Check every entry of b1 and w2
        for j in 0..h {
            let mut b1_p = b1.clone();
            let mut b1_m = b1.clone();
            b1_p[j] += eps;
            b1_m[j] -= eps;
            let lp = compute_jet3_pde_loss_and_grads(&x, &target_f, &w1, &b1_p, &w2, m, d, h, nu, beta, mu).unwrap().loss;
            let lm = compute_jet3_pde_loss_and_grads(&x, &target_f, &w1, &b1_m, &w2, m, d, h, nu, beta, mu).unwrap().loss;
            let fd_b = (lp - lm) / (2.0 * eps);
            assert!((out.grad_b1[j] - fd_b).abs() < 1e-8);

            let mut w2_p = w2.clone();
            let mut w2_m = w2.clone();
            w2_p[j] += eps;
            w2_m[j] -= eps;
            let lp2 = compute_jet3_pde_loss_and_grads(&x, &target_f, &w1, &b1, &w2_p, m, d, h, nu, beta, mu).unwrap().loss;
            let lm2 = compute_jet3_pde_loss_and_grads(&x, &target_f, &w1, &b1, &w2_m, m, d, h, nu, beta, mu).unwrap().loss;
            let fd_w2 = (lp2 - lm2) / (2.0 * eps);
            assert!((out.grad_w2[j] - fd_w2).abs() < 1e-8);
        }
    }

    #[test]
    fn test_4d_eulerian_cartesian_diffusion() {
        let dim = 4;
        let alpha = 0.25;
        let nt = 10;
        let spatial_shape = vec![7usize, 7, 7, 7];
        let spatial_vol: usize = spatial_shape.iter().product();
        let dt = 0.005;
        let dxs = vec![0.15, 0.15, 0.15, 0.15];

        let mut u_flat = vec![0.0f64; nt * spatial_vol];
        for ti in 0..nt {
            let decay = (-(dim as f64) * alpha * (ti as f64) * dt).exp();
            for s in 0..spatial_vol {
                let mut rem = s;
                let mut prod_sin = 1.0;
                for k in (0..dim).rev() {
                    let idx_k = rem % spatial_shape[k];
                    rem /= spatial_shape[k];
                    let xk = 0.2 + (idx_k as f64) * dxs[k];
                    prod_sin *= xk.sin();
                }
                u_flat[ti * spatial_vol + s] = decay * prod_sin;
            }
        }

        let config = PdeConfig {
            sparsity_threshold: 0.01,
            ridge_alpha: 1e-6,
            ..Default::default()
        };

        let grid_shape = vec![nt, 7, 7, 7, 7];
        let res = discover_pde_nd_cartesian(&u_flat, &grid_shape, dt, &dxs, &config)
            .expect("4D cartesian discovery failed");
        assert!(res.r2 > 0.999, "4D cartesian R2 = {}", res.r2);
        assert!(res.frozen_holdout_r2 > 0.999, "4D holdout R2 = {}", res.frozen_holdout_r2);
        assert_eq!(res.dimension, 4);
        assert_eq!(res.method_used, "eulerian_cartesian_4d");
    }

    #[test]
    fn test_20d_true_scattered_rbf_point_cloud_diffusion() {
        let dim = 20;
        let alpha = 0.5;
        let nt = 16;
        let m = 60;
        let dt = 0.005;

        let mut points_flat = vec![0.0f64; m * dim];
        for p in 0..m {
            for k in 0..dim {
                // Deterministic pseudo-random scattered points in [-0.25, 0.25]^20
                let v = ((p * 37 + k * 53 + p * k * 7) % 199) as f64 / 199.0 - 0.5;
                points_flat[p * dim + k] = v * 0.5;
            }
        }

        let mut u_values_flat = vec![0.0f64; nt * m];
        for ti in 0..nt {
            let t = 1.0 + (ti as f64) * dt;
            let scale = (1.0 / t).powf(dim as f64 / 2.0);
            for p in 0..m {
                let pt = &points_flat[p * dim..(p + 1) * dim];
                let r2: f64 = pt.iter().map(|&x| x * x).sum();
                u_values_flat[ti * m + p] = scale * (-r2 / (4.0 * alpha * t)).exp();
            }
        }

        let config = PdeConfig {
            sparsity_threshold: 0.02,
            ridge_alpha: 1e-6,
            ..Default::default()
        };

        let res = discover_pde_rbf_point_cloud_hd(&points_flat, &u_values_flat, nt, m, dim, dt, &config)
            .expect("20D RBF point cloud discovery failed");
        assert!(res.r2 > 0.999, "20D RBF point cloud R2 = {}", res.r2);
        assert!(res.frozen_holdout_r2 > 0.999, "20D RBF holdout R2 = {}", res.frozen_holdout_r2);
        assert_eq!(res.method_used, "meshfree_rbf_point_cloud_20d");
        let lap_idx = res.active_terms.iter().position(|t| t == "laplacian_20d(u)").unwrap();
        assert!((res.coefficients[lap_idx] - 0.5).abs() < 0.02, "Lap coeff = {}", res.coefficients[lap_idx]);
    }

    #[test]
    fn test_20d_tensor_train_mps_rank2_diffusion() {
        let dim = 20;
        let bond_rank = 2;
        let alpha = 0.4;
        let nt = 16;
        let nx = 25;
        let dt = 0.004;
        let dx = 0.1;

        // Rank-2 sum of two distinct 20D Gaussian heat kernels centered at different offsets
        let mut tt_cores_flat = vec![0.0f64; nt * bond_rank * dim * nx];
        for ti in 0..nt {
            let t = 1.0 + (ti as f64) * dt;
            let scale = (1.0 / t).sqrt();
            for r in 0..bond_rank {
                let weight = if r == 0 { 1.0f64 } else { 0.85f64 };
                let w_factor = weight.powf(1.0 / (dim as f64));
                for k in 0..dim {
                    let center = if r == 0 {
                        0.05 * ((k % 5) as f64 - 2.0)
                    } else {
                        -0.04 * ((k % 4) as f64 - 1.5)
                    };
                    for ix in 0..nx {
                        let x = -1.2 + (ix as f64) * dx - center;
                        let idx = (((ti * bond_rank + r) * dim + k) * nx) + ix;
                        tt_cores_flat[idx] = w_factor * scale * (-x * x / (4.0 * alpha * t)).exp();
                    }
                }
            }
        }

        let config = PdeConfig {
            sparsity_threshold: 0.02,
            ridge_alpha: 1e-6,
            ..Default::default()
        };

        let res = discover_pde_tensor_train_hd(&tt_cores_flat, nt, bond_rank, dim, nx, dt, dx, &config)
            .expect("20D Rank-2 TT/MPS discovery failed");
        assert!(res.r2 > 0.999, "20D TT/MPS R2 = {}", res.r2);
        assert!(res.frozen_holdout_r2 > 0.999, "20D TT/MPS holdout R2 = {}", res.frozen_holdout_r2);
        assert_eq!(res.method_used, "tensor_train_mps_20d");
        let lap_idx = res.active_terms.iter().position(|t| t == "laplacian_20d(u)").unwrap();
        assert!((res.coefficients[lap_idx] - alpha).abs() < 0.02, "Lap coeff = {}", res.coefficients[lap_idx]);
    }

    #[test]
    fn test_20d_neural_pinn_bsde_jet2_discovery() {
        let dim = 20;
        let m = 40;
        let h = 12;
        let nt = 16;
        let dt = 0.005;
        let alpha = 0.35;

        let mut x_flat = vec![0.0f64; m * dim];
        for i in 0..(m * dim) {
            x_flat[i] = ((i * 29 + 11) % 103) as f64 * 0.01 - 0.51;
        }
        let mut w1_flat = vec![0.0f64; h * dim];
        let mut b1 = vec![0.0f64; h];
        let m_sq = 12;
        let x_sq = &x_flat[0..(m_sq * dim)];

        let mut phi = vec![vec![0.0f64; h]; m_sq];
        let mut lap_b = vec![vec![0.0f64; h]; m_sq];
        for j in 0..h {
            let mut w_norm_sq = 0.0;
            for k in 0..dim {
                let w = (((j * 19 + k * 23 + 5) % 83) as f64 - 41.0) * 0.015;
                w1_flat[j * dim + k] = w;
                w_norm_sq += w * w;
            }
            b1[j] = 0.1 * ((j as f64) - 5.5);
            for p in 0..m_sq {
                let mut z = b1[j];
                for k in 0..dim {
                    z += w1_flat[j * dim + k] * x_sq[p * dim + k];
                }
                let th = z.tanh();
                let sech2 = 1.0 - th * th;
                phi[p][j] = th;
                lap_b[p][j] = -2.0 * w_norm_sq * th * sech2;
            }
        }

        let mut w2_curr = vec![0.1f64; h];
        for j in 0..h {
            w2_curr[j] = ((j * 7 + 3) % 13) as f64 * 0.08 - 0.4;
        }

        let eval_dw2 = |w2_vec: &[f64]| -> Vec<f64> {
            let mut rhs = vec![0.0f64; m_sq];
            for p in 0..m_sq {
                let mut s = 0.0;
                for j in 0..h {
                    s += lap_b[p][j] * w2_vec[j];
                }
                rhs[p] = alpha * s;
            }
            let mut ata_flat = vec![0.0f64; h * h];
            let mut atb = vec![0.0f64; h];
            for i in 0..h {
                for j in 0..h {
                    let mut s = 0.0;
                    for p in 0..m_sq {
                        s += phi[p][i] * phi[p][j];
                    }
                    ata_flat[i * h + j] = s;
                }
                ata_flat[i * h + i] += 1e-12;
                let mut sb = 0.0;
                for p in 0..m_sq {
                    sb += phi[p][i] * rhs[p];
                }
                atb[i] = sb;
            }
            solve_linear_system(&ata_flat, &atb, h).expect("linear solve failed")
        };

        let mut w2_time_flat = Vec::with_capacity(nt * h);
        for _ti in 0..nt {
            w2_time_flat.extend_from_slice(&w2_curr);
            // RK4 step of size dt
            let k1 = eval_dw2(&w2_curr);
            let w_k2: Vec<f64> = w2_curr.iter().zip(k1.iter()).map(|(&w, &k)| w + 0.5 * dt * k).collect();
            let k2 = eval_dw2(&w_k2);
            let w_k3: Vec<f64> = w2_curr.iter().zip(k2.iter()).map(|(&w, &k)| w + 0.5 * dt * k).collect();
            let k3 = eval_dw2(&w_k3);
            let w_k4: Vec<f64> = w2_curr.iter().zip(k3.iter()).map(|(&w, &k)| w + dt * k).collect();
            let k4 = eval_dw2(&w_k4);
            for j in 0..h {
                w2_curr[j] += (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
        }

        let config = PdeConfig {
            sparsity_threshold: 0.02,
            ridge_alpha: 1e-6,
            ..Default::default()
        };

        let res = discover_pde_neural_pinn_hd(x_sq, &w1_flat, &b1, &w2_time_flat, nt, m_sq, dim, h, dt, &config)
            .expect("20D Neural PINN Jet2 discovery failed");
        assert!(res.r2 > 0.999, "20D Neural PINN R2 = {}", res.r2);
        assert!(res.frozen_holdout_r2 > 0.999, "20D Neural PINN holdout R2 = {}", res.frozen_holdout_r2);
        assert_eq!(res.method_used, "neural_pinn_bsde_jet2_20d");
        let lap_idx = res.active_terms.iter().position(|t| t == "laplacian_20d(u)").unwrap();
        assert!((res.coefficients[lap_idx] - alpha).abs() < 0.02, "Lap coeff = {}", res.coefficients[lap_idx]);
    }

    #[test]
    fn test_helmholtz_hodge_div_free_2d() {
        let h = 32;
        let w = 32;
        let dx = 1.0 / (w as f64);
        let dy = 1.0 / (h as f64);

        // Stream function: psi = sin(2*pi*x) * sin(2*pi*y)
        // Divergence-free part: vx = -d psi / dy, vy = d psi / dx
        // Irrotational part: phi = cos(2*pi*x) * cos(2*pi*y), grad_phi = (d phi / dx, d phi / dy)
        let mut vx = vec![0.0f64; h * w];
        let mut vy = vec![0.0f64; h * w];

        let pi = std::f64::consts::PI;
        for y in 0..h {
            let y_coord = (y as f64) * dy;
            for x in 0..w {
                let x_coord = (x as f64) * dx;
                let vx_sol = -2.0 * pi * (2.0 * pi * x_coord).sin() * (2.0 * pi * y_coord).cos();
                let vy_sol = 2.0 * pi * (2.0 * pi * x_coord).cos() * (2.0 * pi * y_coord).sin();

                let grad_phi_x = -2.0 * pi * (2.0 * pi * x_coord).sin() * (2.0 * pi * y_coord).cos();
                let grad_phi_y = -2.0 * pi * (2.0 * pi * x_coord).cos() * (2.0 * pi * y_coord).sin();

                // Mix solenoidal + irrotational
                vx[y * w + x] = vx_sol + 0.5 * grad_phi_x;
                vy[y * w + x] = vy_sol + 0.5 * grad_phi_y;
            }
        }

        let (vx_sol_res, vy_sol_res, div_norm) =
            helmholtz_hodge_div_free_2d(&vx, &vy, h, w, dx, dy, 50).expect("Helmholtz-Hodge projection failed");

        assert_eq!(vx_sol_res.len(), h * w);
        assert_eq!(vy_sol_res.len(), h * w);
        // The solenoidal projection should substantially reduce divergence
        assert!(div_norm < 1.0, "Remaining divergence norm should be small: {}", div_norm);
    }

    #[test]
    fn test_laplacian_dissipation_2d() {
        let h = 32;
        let w = 32;
        let c = 2;
        let dx = 1.0 / (w as f64);
        let dy = 1.0 / (h as f64);
        let nu = 0.001;
        let dt = 0.01;

        let mut field = vec![0.0f64; c * h * w];
        let pi = std::f64::consts::PI;
        for ch in 0..c {
            for y in 0..h {
                let y_coord = (y as f64) * dy;
                for x in 0..w {
                    let x_coord = (x as f64) * dx;
                    let val = (2.0 * pi * x_coord).sin() * (2.0 * pi * y_coord).sin() * ((ch + 1) as f64);
                    field[ch * h * w + y * w + x] = val;
                }
            }
        }

        let diss = laplacian_dissipation_2d(&field, h, w, c, nu, dt, dx, dy)
            .expect("Laplacian dissipation failed");

        assert_eq!(diss.len(), c * h * w);
        // Laplacian of sin(2*pi*x)*sin(2*pi*y) is -8*pi^2 * val
        // So diss = val * (1 - 8*pi^2 * nu * dt)
        let factor_expected = 1.0 - 8.0 * pi * pi * nu * dt;
        let idx = h / 4 * w + w / 4; // where sin is non-zero
        let orig_val = field[idx];
        let diss_val = diss[idx];
        let ratio = diss_val / orig_val;
        assert!((ratio - factor_expected).abs() < 0.02, "Decay ratio {} != expected {}", ratio, factor_expected);
    }

    #[test]
    fn laplacian_dissipation_refuses_negative_viscosity() {
        // nu < 0 turns the Laplacian term anti-diffusive (backward heat
        // equation): every high-frequency component grows without bound.
        let field = vec![0.5f64; 2 * 5 * 5];
        let err = laplacian_dissipation_2d(&field, 5, 5, 2, -0.01, 0.01, 1.0, 1.0)
            .err()
            .expect("negative nu must be rejected");
        assert!(err.contains("non-negative"), "unexpected error: {err}");

        let err = laplacian_dissipation_2d(&field, 5, 5, 2, f64::NAN, 0.01, 1.0, 1.0)
            .err()
            .expect("NaN nu must be rejected");
        assert!(err.contains("non-negative"), "unexpected error: {err}");
    }

    #[test]
    fn laplacian_dissipation_enforces_diffusion_cfl() {
        // nu*dt*(1/dx^2 + 1/dy^2) > 1/2 must be refused: the explicit update
        // becomes an instability amplifier.
        let field = vec![0.5f64; 2 * 5 * 5];
        // nu=0.01, dt=0.1, dx=dy=0.1 -> cfl = 0.01*0.1*(100+100) = 0.2 -> OK.
        assert!(laplacian_dissipation_2d(&field, 5, 5, 2, 0.01, 0.1, 0.1, 0.1).is_ok());
        // nu=0.05, dt=0.1, dx=dy=0.1 -> cfl = 1.0 > 0.5 -> refused.
        let err = laplacian_dissipation_2d(&field, 5, 5, 2, 0.05, 0.1, 0.1, 0.1)
            .err()
            .expect("CFL violation must be rejected");
        assert!(err.contains("CFL"), "unexpected error: {err}");
    }
}



#[cfg(test)]
mod nu_sign_tests {
    use super::*;

    #[test]
    fn negative_viscosity_is_refused() {
        // A negative nu is the backward heat equation (high-frequency modes grow
        // without bound): the operator must refuse it rather than explode.
        let (m, d, hidden) = (2usize, 2usize, 2usize);
        let x = vec![0.0f64, 0.5, 1.0, 1.5]; // m*d
        let w1 = vec![0.1f64; hidden * d];
        let b1 = vec![0.0f64; hidden];
        let w2 = vec![0.2f64; hidden];
        assert!(eval_jet2_neural_pde_operator(&x, &w1, &b1, &w2, m, d, hidden, -0.5).is_err());
        assert!(eval_jet2_neural_pde_operator(&x, &w1, &b1, &w2, m, d, hidden, 0.5).is_ok());
    }
}
