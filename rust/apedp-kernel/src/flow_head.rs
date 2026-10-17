//! Learned flow-head CNN for DC-STT (mirrors
//! `apedp_dyn/video/learned_flow_head.LearnedFlowHead` exactly).
//!
//! Architecture (planes are (C, H, W) row-major, single batch):
//!   input  : concat(z_prev, z_curr)               -> (2*C_in, H, W)
//!   body   : Conv(3x3, pad=1) + GELU x N_LAYERS
//!   output : Conv(hidden -> 2, 3x3, pad=1) * FLOW_SCALE * tanh
//!
//! The transport step reuses the verified kernels one crossing deeper:
//! [`crate::video::warp_bilinear_2d`] for the semi-Lagrangian backtrace and
//! [`crate::rime::high_dim_pde::helmholtz_hodge_div_free_2d` /
//!  `laplacian_dissipation_2d`] for the Leray projection / viscosity, so the
//! Rust rollout is the same operator chain the Python rollout already runs —
//! only the head's CNN (which Python ran in torch) moved into the kernel:
//! exactly the boundary hop this module is for.
//!
//! Weight layout (flat, layer-major, Conv2d row-major (o, i, ky, kx)):
//!   for l in 0..N:  W[l] (hidden x c_in_l x 9) then b[l] (hidden)
//!   then:           Wout (2 x hidden x 9), bout (2)
//! where c_in_l = 2*C_in for l=0 and `hidden` afterwards.
//!
//! Backprop notes (parity pinned in tests/test_dyn_flow_head_rust.py):
//! - GELU is the exact erf form; [`flow_head::erf_hp`] is pinned to a 397-point
//!   `math.erf` grid (|error| <= 8e-16), so the v-parity tolerance carries no
//!   erf term anymore.
//! - The self-supervised loss backpropagates through the warp's COORDINATE
//!   branch only (data planes carry no gradient) — matching the Python
//!   trainer.
//! - AdamW mirrors torch.optim.AdamW: decoupled decay applied first
//!   (`p *= 1 - lr*wd`), bias-corrected moments, betas 0.9/0.999, eps 1e-8.
use crate::rime::high_dim_pde::{helmholtz_hodge_div_free_2d, laplacian_dissipation_2d};
use crate::video::warp_bilinear_2d;
use ndarray::Array2;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct FlowHeadSpec {
    pub c_in: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub flow_scale: f64,
}

const SQRT2: f64 = std::f64::consts::SQRT_2;
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1/sqrt(2*pi) (gelu pdf)
const FRAC_2_SQRT_PI: f64 = 1.128_379_167_095_512_6; // 2/sqrt(pi) (erf Maclaurin)

/// Total flat parameter count for a spec.
pub fn weights_len(spec: FlowHeadSpec) -> usize {
    let mut n = 0usize;
    let mut c_prev = spec.c_in * 2;
    for _ in 0..spec.n_layers {
        n += spec.hidden * c_prev * 9 + spec.hidden;
        c_prev = spec.hidden;
    }
    n += 2 * spec.hidden * 9 + 2;
    n
}

/// Byte offset of body layer `l`'s weights in the flat layout.
fn layer_offset(spec: FlowHeadSpec, l: usize) -> (usize, usize) {
    let mut base = 0usize;
    let mut c_prev = spec.c_in * 2;
    for _li in 0..l {
        base += spec.hidden * c_prev * 9 + spec.hidden;
        c_prev = spec.hidden;
    }
    let c_l = if l == 0 { spec.c_in * 2 } else { spec.hidden };
    (base, spec.hidden * c_l * 9)
}

fn out_offset(spec: FlowHeadSpec) -> usize {
    let mut base = 0usize;
    let mut c_prev = spec.c_in * 2;
    for _ in 0..spec.n_layers {
        base += spec.hidden * c_prev * 9 + spec.hidden;
        c_prev = spec.hidden;
    }
    base
}

/// Zero-padded 3x3 convolution, stride 1, single sample via im2col + GEMM.
/// x: (C_in, H, W) row-major; w_mat: (C_out, C_in, 3, 3) row-major
/// ((o, i, ky, kx)); b: (C_out,). The im2col row order (i*9 + ky*3 + kx)
/// matches the weight layout ravel exactly.
fn conv3x3(x: &[f64], c_in: usize, h: usize, w_size: usize,
           w_mat: &[f64], b: &[f64], c_out: usize) -> Vec<f64> {
    let plane = h * w_size;
    let cols = im2col(x, c_in, h, w_size);
    let w2 = Array2::from_shape_vec((c_out, c_in * 9), w_mat.to_vec())
        .expect("conv weight shape");
    let c2 = Array2::from_shape_vec((c_in * 9, plane), cols)
        .expect("im2col shape");
    let mut out = w2.dot(&c2);
    for o in 0..c_out {
        let mut row = out.row_mut(o);
        for v in row.iter_mut() {
            *v += b[o];
        }
    }
    out.into_raw_vec()
}

/// im2col: (C_in*9, H*W) zero-padded 3x3 windows, row r = i*9 + ky*3 + kx.
fn im2col(x: &[f64], c_in: usize, h: usize, w_size: usize) -> Vec<f64> {
    let plane = h * w_size;
    let mut cols = vec![0.0f64; c_in * 9 * plane];
    for i in 0..c_in {
        for ky in 0..3usize {
            for kx in 0..3usize {
                // target (y, x_c) valid iff source (y+dy, x+kx) is inside the
                // plane: 0 <= y+dy <= h-1  =>  y in [max(0, -dy), h-1-dy]
                let dy = ky as isize - 1;
                let dx = kx as isize - 1;
                let r = i * 9 + ky * 3 + kx;
                let y0 = ((-dy).max(0)) as usize;
                let x0 = ((-dx).max(0)) as usize;
                let y1 = ((h as isize - dy) as usize).min(h).max(y0);
                let x1 = ((w_size as isize - dx) as usize).min(w_size).max(x0);
                if y1 <= y0 || x1 <= x0 {
                    continue;
                }
                for y in y0..y1 {
                    let sy = (y as isize + dy) as usize;
                    let src = i * plane + sy * w_size + ((x0 as isize + dx) as usize);
                    let dst = r * plane + y * w_size + x0;
                    cols[dst..dst + (x1 - x0)]
                        .copy_from_slice(&x[src..src + (x1 - x0)]);
                }
            }
        }
    }
    cols
}

/// Accumulate transposed-conv input gradients: din[i, y-1+ky, x-1+kx] += W[o,i,ky,kx]*G[o,y,x].
/// Implemented as GEMM (W^T @ G) over im2col layout + slot-wise col2im scatter
/// (rayon over the C_in disjoint output planes).
fn conv3x3_backward_input(x_grad: &mut [f64], g_out: &[f64], w_mat: &[f64],
                          c_in: usize, c_out: usize, h: usize, w_size: usize) {
    let plane = h * w_size;
    let w2 = Array2::from_shape_vec((c_out, c_in * 9), w_mat.to_vec())
        .expect("conv weight shape");
    let g2 = Array2::from_shape_vec((c_out, plane), g_out.to_vec())
        .expect("grad shape");
    let din = (w2.t().dot(&g2)).into_raw_vec(); // (c_in*9, plane)

    // slot-wise scatter: the C_in output planes are disjoint, so produce each
    // plane's contribution in parallel (map) and add them sequentially — exact
    // and contention-free (an Fn closure cannot mutate captured &mut state).
    let contribs: Vec<(usize, Vec<f64>)> = (0..c_in)
        .into_par_iter()
        .map(|i| {
            let mut diff = vec![0.0f64; plane];
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let dy = ky as isize - 1;
                    let dx = kx as isize - 1;
                    let r = (i * 9 + ky * 3 + kx) * plane;
                    let y0 = ((-dy).max(0)) as usize;
                    let x0 = ((-dx).max(0)) as usize;
                    let y1 = ((h as isize - dy) as usize).min(h).max(y0);
                    let x1 = ((w_size as isize - dx) as usize).min(w_size).max(x0);
                    if y1 <= y0 || x1 <= x0 {
                        continue;
                    }
                    for y in y0..y1 {
                        let sy = (y as isize + dy) as usize;
                        let src = &din[r + y * w_size + x0..r + y * w_size + x1];
                        let dst = sy * w_size + ((x0 as isize + dx) as usize);
                        for (k, s) in src.iter().enumerate() {
                            diff[dst + k] += *s;
                        }
                    }
                }
            }
            (i, diff)
        })
        .collect();
    for (i, diff) in contribs {
        for k in 0..plane {
            x_grad[i * plane + k] += diff[k];
        }
    }
}

/// Accumulate conv weight/bias gradients from (x, g_out): dW = G @ im2col(x)^T.
fn conv3x3_backward_weights(x: &[f64], g_out: &[f64], w_grad: &mut [f64], b_grad: &mut [f64],
                            c_in: usize, c_out: usize, h: usize, w_size: usize) {
    let plane = h * w_size;
    let cols = im2col(x, c_in, h, w_size);
    let g2 = Array2::from_shape_vec((c_out, plane), g_out.to_vec())
        .expect("grad shape");
    let c2 = Array2::from_shape_vec((c_in * 9, plane), cols)
        .expect("im2col shape");
    let dw = g2.dot(&c2.t()); // (c_out, c_in*9)
    for (wg, d) in w_grad.iter_mut().zip(dw.into_raw_vec().into_iter()) {
        *wg += d;
    }
    for o in 0..c_out {
        let s: f64 = g_out[o * plane..(o + 1) * plane].iter().sum();
        b_grad[o] += s;
    }
}

/// High-precision erf (verified 5.6e-16 max abs err on the [-8, 8] grid).
/// |x| < 1.5: Maclaurin (terms decay immediately, no large-scale cancellation).
/// |x| >= 1.5: Lentz continued fraction for Gamma(1/2, x^2)/sqrt(pi) (the
/// asymptotic erfc series is NOT used — it diverges catastrophically near
/// the crossover: measured 1e-20-scale partial sums off by 1e42 at x = 2.001).
fn erf_hp(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1.5 {
        let xx = x * x;
        let mut s = FRAC_2_SQRT_PI * x;
        let mut p = x * xx;
        let mut fact = 1.0f64;
        let mut n = 1.0f64;
        loop {
            let sign = if n as i32 % 2 == 1 { -1.0 } else { 1.0 };
            let next = FRAC_2_SQRT_PI * p / (fact * (2.0 * n + 1.0)) * sign;
            s += next;
            p *= xx;
            n += 1.0;
            fact *= n;
            if next.abs() < 1e-18 {
                break;
            }
        }
        s
    } else {
        // NR gcf: Gamma(a, z) = exp(-z + a ln z) * h, a = 1/2, z = ax^2
        let z = ax * ax;
        let a = 0.5f64;
        let tiny = 1e-300f64;
        let mut b = z + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..200u64 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = c * d;
            h *= delta;
            if (delta - 1.0).abs() < 2e-16 {
                break;
            }
        }
        let gamma = (-z + a * z.ln()).exp() * h;
        let erfc = gamma / std::f64::consts::PI.sqrt();
        if x >= 0.0 { 1.0 - erfc } else { erfc - 1.0 }
    }
}

fn gelu_forward(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf_hp(*v / SQRT2));
    }
}

/// d/dx [0.5*x*(1+erf(x/sqrt2))] = Phi(x) + x*phi(x).
fn gelu_backward_in_place(x_raw: &[f64], g: &mut [f64]) {
    for (g, x) in g.iter_mut().zip(x_raw.iter()) {
        let phi = FRAC_1_SQRT_2PI * (-0.5 * x * x).exp();
        let cdf = 0.5 * (1.0 + erf_hp(x / SQRT2));
        *g *= cdf + x * phi;
    }
}

/// One full head forward pass. Returns (vx, vy), each length h*w.
pub fn forward_v(
    z_prev: &[f64], z_curr: &[f64], spec: FlowHeadSpec, weights: &[f64], h: usize, w: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let plane = h * w;
    let c2 = spec.c_in * 2;
    if z_prev.len() != plane * spec.c_in || z_curr.len() != plane * spec.c_in {
        return Err(format!(
            "head planes expected (c_in={}, H, W) each; got prev.len()={} curr.len()={}",
            spec.c_in, z_prev.len(), z_curr.len()
        ));
    }
    if weights.len() != weights_len(spec) {
        return Err(format!(
            "weights length {} != expected {}",
            weights.len(),
            weights_len(spec)
        ));
    }
    let mut x = Vec::with_capacity(plane * c2);
    x.extend_from_slice(z_prev);
    x.extend_from_slice(z_curr);

    for l in 0..spec.n_layers {
        let (base, w_len) = layer_offset(spec, l);
        let c_l = if l == 0 { c2 } else { spec.hidden };
        let wm = &weights[base..base + w_len];
        let bm = &weights[base + w_len..base + w_len + spec.hidden];
        x = conv3x3(&x, c_l, h, w, wm, bm, spec.hidden);
        gelu_forward(&mut x);
    }
    let out0 = out_offset(spec);
    let w_len = 2 * spec.hidden * 9;
    let wm = &weights[out0..out0 + w_len];
    let bm = &weights[out0 + w_len..out0 + w_len + 2];
    let raw = conv3x3(&x, spec.hidden, h, w, wm, bm, 2);
    let mut vx = Vec::with_capacity(plane);
    let mut vy = Vec::with_capacity(plane);
    for idx in 0..plane {
        vx.push(spec.flow_scale * raw[idx].tanh());
        vy.push(spec.flow_scale * raw[plane + idx].tanh());
    }
    Ok((vx, vy))
}

/// Self-supervised flow loss for ONE transition and its gradient w.r.t. v.
///
/// Mirrors `flow_consistency_loss`: adv = mse(Warp(z_curr, -dt*v), z_next);
/// div = lambda_div * mean(D^2) on the (H-2, W-2) interior; tv = lambda_smooth
/// * (mean |dv/dx| + mean |dv/dy|) over torch.diff pair counts. Returns
/// (loss, v_grad (2*plane, [vx-block, vy-block])).
pub fn loss_and_v_grad(
    z_prev: &[f64], z_curr: &[f64], z_next: &[f64], spec: FlowHeadSpec,
    weights: &[f64], h: usize, w: usize,
    dt: f64, lambda_div: f64, lambda_smooth: f64,
) -> Result<(f64, Vec<f64>), String> {
    let plane = h * w;
    let c = spec.c_in;
    let (vx, vy) = forward_v(z_prev, z_curr, spec, weights, h, w)?;

    let disp_vx: Vec<f64> = vx.iter().map(|f| -dt * f).collect();
    let disp_vy: Vec<f64> = vy.iter().map(|f| -dt * f).collect();
    let z_pred = warp_bilinear_2d(z_curr, &disp_vx, &disp_vy, c, h, w)?;

    let n_all = (plane * c) as f64;
    let mut adv_loss = 0.0f64;
    let mut dz_pred = vec![0.0f64; plane * c];
    for idx in 0..plane * c {
        let d = z_pred[idx] - z_next[idx];
        adv_loss += d * d;
        dz_pred[idx] = 2.0 * d / n_all;
    }
    adv_loss /= n_all;

    // --- coordinate-branch backward through the warp ---
    // Same clamp+floor bilinear semantics as video::warp_bilinear_2d; positions
    // clamped against a border have zero coordinate gradient (matches
    // grid_sample padding_mode='border').
    let mut v_grad = vec![0.0f64; 2 * plane];
    for y in 0..h {
        for x_c in 0..w {
            let idx = y * w + x_c;
            let sx = (x_c as f64 - dt * vx[idx]).clamp(0.0, (w - 1) as f64);
            let sy = (y as f64 - dt * vy[idx]).clamp(0.0, (h - 1) as f64);
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let wx = sx - x0 as f64;
            let wy = sy - y0 as f64;
            // channel-summed dz_pred at the output lattice point (y, x_c)
            let mut g_pred = 0.0f64;
            for ch in 0..c {
                g_pred += dz_pred[ch * plane + idx];
            }
            let in_range_x = sx > 0.0 && sx < (w - 1) as f64;
            let in_range_y = sy > 0.0 && sy < (h - 1) as f64;
            if in_range_x {
                let d_sx = (1.0 - wy) * (z_curr[y1 * w + x0] - z_curr[y0 * w + x0])
                    + wy * (z_curr[y1 * w + x1] - z_curr[y0 * w + x1]);
                v_grad[idx] += -dt * g_pred * d_sx;
            }
            if in_range_y {
                let d_sy = (1.0 - wx) * (z_curr[y0 * w + x1] - z_curr[y0 * w + x0])
                    + wx * (z_curr[y1 * w + x1] - z_curr[y1 * w + x0]);
                v_grad[plane + idx] += -dt * g_pred * d_sy;
            }
        }
    }

    // --- divergence penalty on the interior grid (H-2, W-2) ---
    let mut div_loss = 0.0f64;
    if h > 2 && w > 2 && lambda_div != 0.0 {
        let n_int = ((h - 2) * (w - 2)) as f64;
        let coef = 2.0 * lambda_div / n_int;
        let mut sum_sq = 0.0f64;
        for hy in 1..h - 1 {
            for hx in 1..w - 1 {
                let d = (vx[hy * w + hx + 1] - vx[hy * w + hx - 1]) * 0.5
                    + (vy[(hy + 1) * w + hx] - vy[(hy - 1) * w + hx]) * 0.5;
                sum_sq += d * d;
                let g = coef * d;
                v_grad[hy * w + hx + 1] += 0.5 * g;
                v_grad[hy * w + hx - 1] -= 0.5 * g;
                v_grad[plane + (hy + 1) * w + hx] += 0.5 * g;
                v_grad[plane + (hy - 1) * w + hx] -= 0.5 * g;
            }
        }
        div_loss = lambda_div * sum_sq / n_int;
    }

    // --- TV smoothness (forward diffs, both channels, torch-mean semantics) ---
    // value = lambda * (mean|dv/dx| + mean|dv/dy|); gradient via sign scatter.
    let mut tv_loss = 0.0f64;
    if lambda_smooth != 0.0 && h > 1 && w > 1 {
        let cnt_h = (c * h * (w - 1)) as f64;
        let cnt_v = (c * (h - 1) * w) as f64;
        let mut h_sum = 0.0f64;
        let mut v_sum = 0.0f64;
        for ch in 0..c {
            let f0: &[f64] = if ch == 0 { &vx } else { &vy };
            for y in 0..h {
                for x_c in 0..w - 1 {
                    let d = f0[y * w + x_c + 1] - f0[y * w + x_c];
                    h_sum += d.abs();
                    let g = lambda_smooth / cnt_h * d.signum();
                    v_grad[ch * plane + y * w + x_c] -= g;
                    v_grad[ch * plane + y * w + x_c + 1] += g;
                }
            }
            for y_c in 0..h - 1 {
                for x_c in 0..w {
                    let d = f0[(y_c + 1) * w + x_c] - f0[y_c * w + x_c];
                    v_sum += d.abs();
                    let g = lambda_smooth / cnt_v * d.signum();
                    v_grad[ch * plane + y_c * w + x_c] -= g;
                    v_grad[ch * plane + (y_c + 1) * w + x_c] += g;
                }
            }
        }
        tv_loss = lambda_smooth * (h_sum / cnt_h + v_sum / cnt_v);
    }

    Ok((adv_loss + div_loss + tv_loss, v_grad))
}

/// torch.optim.AdamW with decoupled decay-first ordering.
struct AdamW {
    m: Vec<f64>,
    v: Vec<f64>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    wd: f64,
    step: u64,
}

impl AdamW {
    fn new(len: usize, lr: f64, wd: f64) -> Self {
        Self { m: vec![0.0; len], v: vec![0.0; len], lr, betas: (0.9, 0.999), eps: 1e-8, wd, step: 0 }
    }

    fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        self.step += 1;
        let (b1, b2) = self.betas;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grads[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            let decayed = params[i] * (1.0 - self.lr * self.wd);
            params[i] = decayed - self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Backpropagate v_grad through the head's conv stack into flat param grads.
fn backward_to_param_grads(
    z_prev: &[f64], z_curr: &[f64], spec: FlowHeadSpec, weights: &[f64],
    h: usize, w: usize, v_grad: &[f64],
) -> Result<Vec<f64>, String> {
    let plane = h * w;
    let c2 = spec.c_in * 2;

    // rebuild forward intermediates
    let mut x = Vec::with_capacity(plane * c2);
    x.extend_from_slice(z_prev);
    x.extend_from_slice(z_curr);
    let mut layer_inputs: Vec<Vec<f64>> = Vec::with_capacity(spec.n_layers + 1);
    let mut layer_raws: Vec<Vec<f64>> = Vec::with_capacity(spec.n_layers);
    layer_inputs.push(x.clone());
    for l in 0..spec.n_layers {
        let (base, w_len) = layer_offset(spec, l);
        let c_l = if l == 0 { c2 } else { spec.hidden };
        let wm = &weights[base..base + w_len];
        let bm = &weights[base + w_len..base + w_len + spec.hidden];
        let raw = conv3x3(&x, c_l, h, w, wm, bm, spec.hidden);
        layer_raws.push(raw.clone());
        let mut act = raw;
        gelu_forward(&mut act);
        x = act;
        layer_inputs.push(x.clone());
    }
    let out0 = out_offset(spec);
    let w_len = 2 * spec.hidden * 9;
    let wm = &weights[out0..out0 + w_len];
    let bm = &weights[out0 + w_len..out0 + w_len + 2];
    let raw_out = conv3x3(&x, spec.hidden, h, w, wm, bm, 2);

    // dL/draw = dL/dout * scale * (1 - tanh^2(raw))
    let mut g_out = vec![0.0f64; 2 * plane];
    for idx in 0..plane {
        let t0 = raw_out[idx].tanh();
        let t1 = raw_out[plane + idx].tanh();
        g_out[idx] = v_grad[idx] * spec.flow_scale * (1.0 - t0 * t0);
        g_out[plane + idx] = v_grad[plane + idx] * spec.flow_scale * (1.0 - t1 * t1);
    }

    let total = weights_len(spec);
    let mut grads = vec![0.0f64; total];

    // output conv grads
    {
        let (wg, bg) = grads.split_at_mut(out0 + w_len);
        let wg = &mut wg[out0..];
        conv3x3_backward_weights(
            layer_inputs[spec.n_layers].as_slice(),
            &g_out, wg, bg, spec.hidden, 2, h, w,
        );
        let mut gin = vec![0.0f64; plane * spec.hidden];
        conv3x3_backward_input(&mut gin, &g_out, &weights[out0..out0 + w_len],
                               spec.hidden, 2, h, w);
        let mut g = gin;
        for l in (0..spec.n_layers).rev() {
            gelu_backward_in_place(&layer_raws[l], &mut g);
            let (base, w_len) = layer_offset(spec, l);
            let c_l = if l == 0 { c2 } else { spec.hidden };
            {
                let (head, tail) = grads.split_at_mut(base + w_len + spec.hidden);
                let wg = &mut head[base..base + w_len];
                let bg = &mut tail[..spec.hidden];
                conv3x3_backward_weights(
                    layer_inputs[l].as_slice(), &g, wg, bg, c_l, spec.hidden, h, w,
                );
            }
            if l > 0 {
                let mut g_in = vec![0.0f64; plane * c_l];
                conv3x3_backward_input(&mut g_in, &g, &weights[base..base + w_len],
                                       c_l, spec.hidden, h, w);
                g = g_in;
            }
        }
    }
    Ok(grads)
}

/// Train the head self-supervised on (T, C_in, H, W) head planes.
///
/// `frame_triples[i] = (t_prev, t_curr, t_next)` selects transitions
/// explicitly: index selection/RNG parity is a Python-side concern and the
/// kernel stays pure. `deadline_ms` stops mid-loop (one-step overage bound).
#[allow(clippy::too_many_arguments)]
pub fn train(
    data: &[f64],
    frame_triples: &[[usize; 3]],
    spec: FlowHeadSpec,
    weights: &mut [f64],
    h: usize,
    w: usize,
    steps: usize,
    lr: f64,
    weight_decay: f64,
    lambda_div: f64,
    lambda_smooth: f64,
    dt: f64,
    deadline_ms: Option<u128>,
) -> Result<Vec<f64>, String> {
    let plane = h * w;
    let t_total = data.len() / (spec.c_in * plane);
    if t_total < 3 {
        return Err(format!("need >= 3 frames for transitions, got T={}", t_total));
    }
    if weights.len() != weights_len(spec) {
        return Err(format!(
            "weights length mismatch: {} vs expected {}",
            weights.len(),
            weights_len(spec)
        ));
    }
    if frame_triples.is_empty() {
        return Err("frame_triples must be non-empty".to_string());
    }
    let start = std::time::Instant::now();
    let mut opt = AdamW::new(weights_len(spec), lr, weight_decay);
    let mut losses: Vec<f64> = Vec::with_capacity(steps);
    for step in 0..steps {
        let trip = frame_triples[step % frame_triples.len()];
        let z_prev = &data[trip[0] * spec.c_in * plane..(trip[0] + 1) * spec.c_in * plane];
        let z_curr = &data[trip[1] * spec.c_in * plane..(trip[1] + 1) * spec.c_in * plane];
        let z_next = &data[trip[2] * spec.c_in * plane..(trip[2] + 1) * spec.c_in * plane];
        let (loss, v_grad) = loss_and_v_grad(
            z_prev, z_curr, z_next, spec, weights, h, w, dt, lambda_div, lambda_smooth,
        )?;
        let grads = backward_to_param_grads(z_prev, z_curr, spec, weights, h, w, &v_grad)?;
        opt.step(weights, &grads);
        losses.push(loss);
        if let Some(deadline) = deadline_ms {
            if start.elapsed().as_millis() >= deadline {
                break;
            }
        }
    }
    Ok(losses)
}

/// One full DC-STT transport step — head + (optional) Leray projection +
/// warp + (optional) viscosity — in ONE boundary crossing.
///
/// Mirrors `DecoupledSpatioTemporalTransport.transport_step`: v comes from the
/// head; `disp = -dt*v_eff`; next = Warp(state, disp); optionally dissipate.
/// `head_prev`/`head_curr` are the two (c_in, H, W) history planes the caller
/// keeps. Returns (z_next, vx_eff, vy_eff, rem_divergence; NaN when the
/// projection is disabled).
#[allow(clippy::too_many_arguments)]
pub fn transport_step(
    state: &[f64],
    head_prev: &[f64],
    head_curr: &[f64],
    spec: FlowHeadSpec,
    weights: &[f64],
    h: usize,
    w: usize,
    dt: f64,
    enforce_hh: bool,
    hh_iters: usize,
    nu: f64,
    use_dissipation: bool,
    dx: f64,
    dy: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64), String> {
    let c = state.len() / (h * w);
    if c * h * w != state.len() {
        return Err(format!(
            "state length {} does not decompose over (h={}, w={})",
            state.len(), h, w
        ));
    }
    let (vx_raw, vy_raw) = forward_v(head_prev, head_curr, spec, weights, h, w)?;
    let (vx_eff, vy_eff, rem_div) = if enforce_hh {
        let (hx, hy, rem) =
            helmholtz_hodge_div_free_2d(&vx_raw, &vy_raw, h, w, dx, dy, hh_iters)?;
        (hx, hy, rem)
    } else {
        (vx_raw.clone(), vy_raw.clone(), f64::NAN)
    };
    let disp_vx: Vec<f64> = vx_eff.iter().map(|f| -dt * f).collect();
    let disp_vy: Vec<f64> = vy_eff.iter().map(|f| -dt * f).collect();
    let mut z_next = warp_bilinear_2d(state, &disp_vx, &disp_vy, c, h, w)?;
    if use_dissipation && nu > 0.0 {
        z_next = laplacian_dissipation_2d(&z_next, h, w, c, nu, dt, dx, dy)?;
    }
    Ok((z_next, vx_eff, vy_eff, rem_div))
}

#[cfg(test)]
mod erf_hp_grid_tests {
    use super::erf_hp;

    /// Reference grid over [{-6, 6}] at step 0.05, a fine sweep across the
    /// Maclaurin/Lentz crossover [1.3, 1.7] at step 0.005 (both signs), and
    /// large-|x| saturation points out to ±12. Values are <f64-exact>
    /// `math.erf` from Python 3.12; the pin tolerance (8e-16) is chosen to
    /// sit above cross-library ulp noise (~2 ulp near |erf|=1) but six
    /// orders below any real branch defect — it caught the 2/sqrt(2*pi)
    /// vs 2/sqrt(pi) constant bug at ±0.28 error.
    const GRID: &[(f64, f64)] = &[
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
        (-1.69500000000000006e+00,
         -9.83474225648236944e-01),
        (-1.68999999999999995e+00,
         -9.83152586895026159e-01),
        (-1.68500000000000005e+00,
         -9.82825466278453463e-01),
        (-1.68000000000000016e+00,
         -9.82492787002464896e-01),
        (-1.67500000000000004e+00,
         -9.82154471526607065e-01),
        (-1.66999999999999993e+00,
         -9.81810441565126535e-01),
        (-1.66500000000000004e+00,
         -9.81460618086203329e-01),
        (-1.66000000000000014e+00,
         -9.81104921311322098e-01),
        (-1.65500000000000003e+00,
         -9.80743270714783955e-01),
        (-1.65000000000000013e+00,
         -9.80375585023360308e-01),
        (-1.64500000000000002e+00,
         -9.80001782216091133e-01),
        (-1.64000000000000012e+00,
         -9.79621779524232017e-01),
        (-1.63500000000000001e+00,
         -9.79235493431349524e-01),
        (-1.63000000000000012e+00,
         -9.78842839673570109e-01),
        (-1.62500000000000000e+00,
         -9.78443733239983682e-01),
        (-1.62000000000000011e+00,
         -9.78038088373203496e-01),
        (-1.61499999999999999e+00,
         -9.77625818570087124e-01),
        (-1.61000000000000010e+00,
         -9.77206836582618643e-01),
        (-1.60499999999999998e+00,
         -9.76781054418955685e-01),
        (-1.60000000000000009e+00,
         -9.76348383344644022e-01),
        (-1.59499999999999997e+00,
         -9.75908733884000235e-01),
        (-1.59000000000000008e+00,
         -9.75462015821667694e-01),
        (-1.58499999999999996e+00,
         -9.75008138204345953e-01),
        (-1.58000000000000007e+00,
         -9.74547009342696890e-01),
        (-1.57500000000000018e+00,
         -9.74078536813429707e-01),
        (-1.57000000000000006e+00,
         -9.73602627461567116e-01),
        (-1.56499999999999995e+00,
         -9.73119187402895380e-01),
        (-1.56000000000000005e+00,
         -9.72628122026600206e-01),
        (-1.55500000000000016e+00,
         -9.72129335998090482e-01),
        (-1.55000000000000004e+00,
         -9.71622733262012539e-01),
        (-1.54499999999999993e+00,
         -9.71108217045457689e-01),
        (-1.54000000000000004e+00,
         -9.70585689861363732e-01),
        (-1.53500000000000014e+00,
         -9.70055053512113963e-01),
        (-1.53000000000000003e+00,
         -9.69516209093335690e-01),
        (-1.52500000000000013e+00,
         -9.68969056997899592e-01),
        (-1.52000000000000002e+00,
         -9.68413496920123129e-01),
        (-1.51500000000000012e+00,
         -9.67849427860179023e-01),
        (-1.51000000000000001e+00,
         -9.67276748128711672e-01),
        (-1.50500000000000012e+00,
         -9.66695355351663066e-01),
        (-1.50000000000000000e+00,
         -9.66105146475310761e-01),
        (-1.49500000000000011e+00,
         -9.65506017771518676e-01),
        (-1.48999999999999999e+00,
         -9.64897864843204167e-01),
        (-1.48500000000000010e+00,
         -9.64280582630022365e-01),
        (-1.47999999999999998e+00,
         -9.63654065414268901e-01),
        (-1.47500000000000009e+00,
         -9.63018206827005230e-01),
        (-1.46999999999999997e+00,
         -9.62372899854405772e-01),
        (-1.46500000000000008e+00,
         -9.61718036844329327e-01),
        (-1.45999999999999996e+00,
         -9.61053509513118076e-01),
        (-1.45500000000000007e+00,
         -9.60379208952622854e-01),
        (-1.45000000000000018e+00,
         -9.59695025637459231e-01),
        (-1.44999999999999996e+00,
         -9.59695025637459231e-01),
        (-1.44500000000000006e+00,
         -9.59000849432494196e-01),
        (-1.43999999999999995e+00,
         -9.58296569600564752e-01),
        (-1.43500000000000005e+00,
         -9.57582074810432005e-01),
        (-1.43000000000000016e+00,
         -9.56857253144968833e-01),
        (-1.42500000000000004e+00,
         -9.56121992109586261e-01),
        (-1.41999999999999993e+00,
         -9.55376178640896190e-01),
        (-1.41500000000000004e+00,
         -9.54619699115614728e-01),
        (-1.41000000000000014e+00,
         -9.53852439359705429e-01),
        (-1.40500000000000003e+00,
         -9.53074284657764559e-01),
        (-1.40000000000000013e+00,
         -9.52285119762648846e-01),
        (-1.39500000000000002e+00,
         -9.51484828905346913e-01),
        (-1.39000000000000012e+00,
         -9.50673295805096519e-01),
        (-1.38500000000000001e+00,
         -9.49850403679745714e-01),
        (-1.38000000000000012e+00,
         -9.49016035256362578e-01),
        (-1.37500000000000000e+00,
         -9.48170072782090312e-01),
        (-1.37000000000000011e+00,
         -9.47312398035252023e-01),
        (-1.36499999999999999e+00,
         -9.46442892336702979e-01),
        (-1.36000000000000010e+00,
         -9.45561436561433100e-01),
        (-1.35499999999999998e+00,
         -9.44667911150418482e-01),
        (-1.35000000000000009e+00,
         -9.43762196122724051e-01),
        (-1.34499999999999997e+00,
         -9.42844171087855898e-01),
        (-1.34000000000000008e+00,
         -9.41913715258365314e-01),
        (-1.33499999999999996e+00,
         -9.40970707462703948e-01),
        (-1.33000000000000007e+00,
         -9.40015026158330214e-01),
        (-1.32499999999999996e+00,
         -9.39046549445066936e-01),
        (-1.32000000000000006e+00,
         -9.38065155078711355e-01),
        (-1.31499999999999995e+00,
         -9.37070720484895703e-01),
        (-1.31000000000000005e+00,
         -9.36063122773199474e-01),
        (-1.30499999999999994e+00,
         -9.35042238751513266e-01),
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
        (1.30499999999999994e+00,
         9.35042238751513266e-01),
        (1.31000000000000005e+00,
         9.36063122773199474e-01),
        (1.31499999999999995e+00,
         9.37070720484895703e-01),
        (1.32000000000000006e+00,
         9.38065155078711355e-01),
        (1.32499999999999996e+00,
         9.39046549445066936e-01),
        (1.33000000000000007e+00,
         9.40015026158330214e-01),
        (1.33499999999999996e+00,
         9.40970707462703948e-01),
        (1.34000000000000008e+00,
         9.41913715258365314e-01),
        (1.34499999999999997e+00,
         9.42844171087855898e-01),
        (1.35000000000000009e+00,
         9.43762196122724051e-01),
        (1.35499999999999998e+00,
         9.44667911150418482e-01),
        (1.36000000000000010e+00,
         9.45561436561433100e-01),
        (1.36499999999999999e+00,
         9.46442892336702979e-01),
        (1.37000000000000011e+00,
         9.47312398035252023e-01),
        (1.37500000000000000e+00,
         9.48170072782090312e-01),
        (1.38000000000000012e+00,
         9.49016035256362578e-01),
        (1.38500000000000001e+00,
         9.49850403679745714e-01),
        (1.39000000000000012e+00,
         9.50673295805096519e-01),
        (1.39500000000000002e+00,
         9.51484828905346913e-01),
        (1.40000000000000013e+00,
         9.52285119762648846e-01),
        (1.40500000000000003e+00,
         9.53074284657764559e-01),
        (1.41000000000000014e+00,
         9.53852439359705429e-01),
        (1.41500000000000004e+00,
         9.54619699115614728e-01),
        (1.41999999999999993e+00,
         9.55376178640896190e-01),
        (1.42500000000000004e+00,
         9.56121992109586261e-01),
        (1.43000000000000016e+00,
         9.56857253144968833e-01),
        (1.43500000000000005e+00,
         9.57582074810432005e-01),
        (1.43999999999999995e+00,
         9.58296569600564752e-01),
        (1.44500000000000006e+00,
         9.59000849432494196e-01),
        (1.44999999999999996e+00,
         9.59695025637459231e-01),
        (1.45000000000000018e+00,
         9.59695025637459231e-01),
        (1.45500000000000007e+00,
         9.60379208952622854e-01),
        (1.45999999999999996e+00,
         9.61053509513118076e-01),
        (1.46500000000000008e+00,
         9.61718036844329327e-01),
        (1.46999999999999997e+00,
         9.62372899854405772e-01),
        (1.47500000000000009e+00,
         9.63018206827005230e-01),
        (1.47999999999999998e+00,
         9.63654065414268901e-01),
        (1.48500000000000010e+00,
         9.64280582630022365e-01),
        (1.48999999999999999e+00,
         9.64897864843204167e-01),
        (1.49500000000000011e+00,
         9.65506017771518676e-01),
        (1.50000000000000000e+00,
         9.66105146475310761e-01),
        (1.50500000000000012e+00,
         9.66695355351663066e-01),
        (1.51000000000000001e+00,
         9.67276748128711672e-01),
        (1.51500000000000012e+00,
         9.67849427860179023e-01),
        (1.52000000000000002e+00,
         9.68413496920123129e-01),
        (1.52500000000000013e+00,
         9.68969056997899592e-01),
        (1.53000000000000003e+00,
         9.69516209093335690e-01),
        (1.53500000000000014e+00,
         9.70055053512113963e-01),
        (1.54000000000000004e+00,
         9.70585689861363732e-01),
        (1.54499999999999993e+00,
         9.71108217045457689e-01),
        (1.55000000000000004e+00,
         9.71622733262012539e-01),
        (1.55500000000000016e+00,
         9.72129335998090482e-01),
        (1.56000000000000005e+00,
         9.72628122026600206e-01),
        (1.56499999999999995e+00,
         9.73119187402895380e-01),
        (1.57000000000000006e+00,
         9.73602627461567116e-01),
        (1.57500000000000018e+00,
         9.74078536813429707e-01),
        (1.58000000000000007e+00,
         9.74547009342696890e-01),
        (1.58499999999999996e+00,
         9.75008138204345953e-01),
        (1.59000000000000008e+00,
         9.75462015821667694e-01),
        (1.59499999999999997e+00,
         9.75908733884000235e-01),
        (1.60000000000000009e+00,
         9.76348383344644022e-01),
        (1.60499999999999998e+00,
         9.76781054418955685e-01),
        (1.61000000000000010e+00,
         9.77206836582618643e-01),
        (1.61499999999999999e+00,
         9.77625818570087124e-01),
        (1.62000000000000011e+00,
         9.78038088373203496e-01),
        (1.62500000000000000e+00,
         9.78443733239983682e-01),
        (1.63000000000000012e+00,
         9.78842839673570109e-01),
        (1.63500000000000001e+00,
         9.79235493431349524e-01),
        (1.64000000000000012e+00,
         9.79621779524232017e-01),
        (1.64500000000000002e+00,
         9.80001782216091133e-01),
        (1.65000000000000013e+00,
         9.80375585023360308e-01),
        (1.65500000000000003e+00,
         9.80743270714783955e-01),
        (1.66000000000000014e+00,
         9.81104921311322098e-01),
        (1.66500000000000004e+00,
         9.81460618086203329e-01),
        (1.66999999999999993e+00,
         9.81810441565126535e-01),
        (1.67500000000000004e+00,
         9.82154471526607065e-01),
        (1.68000000000000016e+00,
         9.82492787002464896e-01),
        (1.68500000000000005e+00,
         9.82825466278453463e-01),
        (1.68999999999999995e+00,
         9.83152586895026159e-01),
        (1.69500000000000006e+00,
         9.83474225648236944e-01),
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
        (8.00000000000000000e+00,
         1.00000000000000000e+00),
        (9.00000000000000000e+00,
         1.00000000000000000e+00),
        (1.20000000000000000e+01,
         1.00000000000000000e+00),
        (-6.50000000000000000e+00,
         -1.00000000000000000e+00),
        (-7.00000000000000000e+00,
         -1.00000000000000000e+00),
        (-8.00000000000000000e+00,
         -1.00000000000000000e+00),
        (-9.00000000000000000e+00,
         -1.00000000000000000e+00),
        (-1.20000000000000000e+01,
         -1.00000000000000000e+00),
    ];

    #[test]
    fn erf_hp_matches_the_reference_grid() {
        assert!(!GRID.is_empty());
        let mut worst: (f64, f64, f64) = (0.0, 0.0, 0.0);
        for &(x, y_ref) in GRID {
            let v = erf_hp(x);
            assert!(v.is_finite(), "erf_hp({x}) not finite");
            let err = (v - y_ref).abs();
            if err > worst.0 {
                worst = (err, x, y_ref);
            }
            assert!(
                err <= 8e-16,
                "erf_hp({x}) = {v} deviates from reference {y_ref} by {err}"
            );
        }
        // Odd symmetry on the tested range, at machine precision.
        for &(x, _) in GRID {
            let s = erf_hp(x) + erf_hp(-x);
            assert!(s.abs() <= 4e-16, "erf_hp symmetry broken at {x}: {s}");
        }
        // Emitted once per run so regressions report the guilty point.
        eprintln!(
            "erf_hp max grid deviation {:.3e} at x={} (ref {:.17e})",
            worst.0, worst.1, worst.2
        );
    }
}
