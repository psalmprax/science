//! Native JetFlow kernels. These mirror the Python reference in
//! `packages/jetflow-video/jetflow_video/stepper.py` and `tensor_train.py`.
//!
//! The Python side keeps the probe evaluation (it calls the velocity model),
//! and passes the probe result here. The arithmetic is done natively.

use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::types::PyList;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Fused directional derivative: a = (v_probe - v_current) / h_eff.
pub fn acceleration(v_current: &[f64], v_probe: &[f64], h_eff: f64) -> Result<Vec<f64>, String> {
    if v_current.len() != v_probe.len() {
        return Err(format!("shape mismatch: {} vs {}", v_current.len(), v_probe.len()));
    }
    if h_eff.abs() <= 1e-9 {
        return Err("h_eff must be non-zero".to_string());
    }
    Ok(v_current
        .iter()
        .zip(v_probe)
        .map(|(v, p)| (p - v) / h_eff)
        .collect())
}

/// Second-order Taylor step: z + dt*v + 0.5*dt^2*a.
pub fn taylor_step(z: &[f64], v: &[f64], a: &[f64], dt: f64) -> Result<Vec<f64>, String> {
    if z.len() != v.len() || z.len() != a.len() {
        return Err("shape mismatch in taylor_step".to_string());
    }
    Ok((0..z.len())
        .map(|i| z[i] + dt * v[i] + 0.5 * dt * dt * a[i])
        .collect())
}

fn rotate_cols(cols: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64) {
    for i in 0..cols[p].len() {
        let x = cols[p][i];
        let y = cols[q][i];
        cols[p][i] = c * x - s * y;
        cols[q][i] = s * x + c * y;
    }
}

/// Thin SVD of a row-major m x n matrix by one-sided Jacobi rotations.
/// Returns (u: m x k, s: k descending, vt: k x n), with k = min(m, n).
pub fn thin_svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let transpose = m < n;
    let (rows, cols) = if transpose { (n, m) } else { (m, n) };
    // Work on the (rows x cols) matrix with rows >= cols, stored as columns.
    let mut wc: Vec<Vec<f64>> = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|i| if transpose { a[j * n + i] } else { a[i * n + j] })
                .collect()
        })
        .collect();
    let mut vv: Vec<Vec<f64>> = (0..cols)
        .map(|j| {
            let mut e = vec![0.0; cols];
            e[j] = 1.0;
            e
        })
        .collect();

    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..cols {
            for q in (p + 1)..cols {
                let alpha: f64 = wc[p].iter().map(|x| x * x).sum();
                let beta: f64 = wc[q].iter().map(|x| x * x).sum();
                let gamma: f64 = wc[p].iter().zip(&wc[q]).map(|(x, y)| x * y).sum();
                if gamma == 0.0 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                rotate_cols(&mut wc, p, q, c, s);
                rotate_cols(&mut vv, p, q, c, s);
            }
        }
        if !rotated {
            break;
        }
    }

    let sv: Vec<f64> = wc.iter().map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| sv[b].partial_cmp(&sv[a]).unwrap_or(std::cmp::Ordering::Equal));
    let k = cols;
    let s: Vec<f64> = order.iter().map(|&j| sv[j]).collect();
    let norm_col = |j: usize, i: usize| if sv[j] > 0.0 { wc[j][i] / sv[j] } else { 0.0 };

    if !transpose {
        // A = U S V^T, U from wc, V^T rows from vv.
        let mut u = vec![0.0; m * k];
        let mut vt = vec![0.0; k * n];
        for (jj, &j) in order.iter().enumerate() {
            for i in 0..m {
                u[i * k + jj] = norm_col(j, i);
            }
            for i in 0..n {
                vt[jj * n + i] = vv[j][i];
            }
        }
        (u, s, vt)
    } else {
        // A^T = U S V^T  =>  A = V S U^T.
        let mut u = vec![0.0; m * k];
        let mut vt = vec![0.0; k * n];
        for (jj, &j) in order.iter().enumerate() {
            for i in 0..m {
                u[i * k + jj] = vv[j][i];
            }
            for i in 0..n {
                vt[jj * n + i] = norm_col(j, i);
            }
        }
        (u, s, vt)
    }
}

/// TT-SVD of a (T, C, H, W) tensor, following `compress_video_latent_torch`.
/// Returns the four cores flattened row-major, and the bond ranks (R1, R2, R3).
/// Smallest rank r whose discarded tail energy sum(s[r:]^2) is <= delta2,
/// capped at max_rank. Same rule as `_budget_rank` in tensor_train.py.
fn budget_rank(s: &[f64], delta2: f64, max_rank: usize) -> usize {
    let mut tail = 0.0;
    let mut r = s.len();
    for k in (0..s.len()).rev() {
        let next_tail = tail + s[k] * s[k];
        if next_tail > delta2 {
            break;
        }
        tail = next_tail;
        r = k;
    }
    r.min(max_rank).min(s.len()).max(1)
}

pub fn tt_svd(
    x: &[f64],
    dims: [usize; 4],
    max_rank: usize,
    tolerance: f64,
) -> Result<(Vec<Vec<f64>>, [usize; 3]), String> {
    let total: usize = dims.iter().product();
    if x.len() != total {
        return Err(format!("expected {} elements, got {}", total, x.len()));
    }
    // Global TT-SVD error budget (Oseledets 2011): ||X - TT|| <= tolerance * ||X||_F.
    let norm2: f64 = x.iter().map(|v| v * v).sum();
    let delta2 = (tolerance * norm2.sqrt()).powi(2) / 3.0;
    let mut cores: Vec<Vec<f64>> = Vec::with_capacity(4);
    let mut ranks = [0usize; 3];
    let mut rem: Vec<f64> = x.to_vec();
    let mut left = 1usize;
    for k in 0..3 {
        let rows = left * dims[k];
        let cols = rem.len() / rows;
        let (u, s, vt) = thin_svd(&rem, rows, cols);
        let kk = s.len();
        let r = budget_rank(&s, delta2, max_rank).min(rows).min(cols).max(1);
        let mut core = vec![0.0; rows * r];
        for i in 0..rows {
            for j in 0..r {
                core[i * r + j] = u[i * kk + j];
            }
        }
        let mut next = vec![0.0; r * cols];
        for j in 0..r {
            for i in 0..cols {
                next[j * cols + i] = s[j] * vt[j * cols + i];
            }
        }
        cores.push(core);
        ranks[k] = r;
        rem = next;
        left = r;
    }
    cores.push(rem);
    Ok((cores, ranks))
}

#[pyfunction]
#[pyo3(signature = (v_current, v_probe, h_eff))]
pub fn jetflow_compute_acceleration<'py>(
    py: Python<'py>,
    v_current: PyReadonlyArray1<'py, f64>,
    v_probe: PyReadonlyArray1<'py, f64>,
    h_eff: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = acceleration(&v_current.as_array().to_vec(), &v_probe.as_array().to_vec(), h_eff)
        .map_err(PyValueError::new_err)?;
    Ok(a.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (z, v, a, dt))]
pub fn jetflow_step_integration<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out = taylor_step(
        &z.as_array().to_vec(),
        &v.as_array().to_vec(),
        &a.as_array().to_vec(),
        dt,
    )
    .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (latent, dims, max_rank=16, tolerance=1e-4))]
pub fn jetflow_tensor_train_compress<'py>(
    py: Python<'py>,
    latent: PyReadonlyArray1<'py, f64>,
    dims: [usize; 4],
    max_rank: usize,
    tolerance: f64,
) -> PyResult<(Bound<'py, PyList>, [usize; 3])> {
    let (cores, ranks) = tt_svd(&latent.as_array().to_vec(), dims, max_rank, tolerance)
        .map_err(PyValueError::new_err)?;
    let list = PyList::empty(py);
    for core in cores {
        list.append(core.into_pyarray(py))?;
    }
    Ok((list, ranks))
}

/// Spectral Leray projection of each (T, C, H, W) channel pair (2k, 2k+1) onto the
/// divergence-free subspace, matching `NoetherConservationEngine` in noether.py
/// (dx = dy = 1, exact central-difference symbols). Returns (projected, mean residual RMS).
#[pyfunction]
#[pyo3(signature = (velocity))]
pub fn jetflow_noether_project<'py>(
    py: Python<'py>,
    velocity: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, f64)> {
    let v = velocity.as_array();
    if v.ndim() != 4 {
        return Err(PyValueError::new_err("expected 4D (T, C, H, W)"));
    }
    let (t, c, h, w) = (v.shape()[0], v.shape()[1], v.shape()[2], v.shape()[3]);
    let mut out = v.to_owned();
    let mut total = 0.0;
    let mut pairs = 0usize;
    for ti in 0..t {
        for ci in (0..c.saturating_sub(1)).step_by(2) {
            let vx: Vec<f64> = (0..h * w).map(|i| v[[ti, ci, i / w, i % w]]).collect();
            let vy: Vec<f64> = (0..h * w).map(|i| v[[ti, ci + 1, i / w, i % w]]).collect();
            let (sx, sy, rem) = leray_project_2d(&vx, &vy, h, w);
            for i in 0..h * w {
                out[[ti, ci, i / w, i % w]] = sx[i];
                out[[ti, ci + 1, i / w, i % w]] = sy[i];
            }
            total += rem;
            pairs += 1;
        }
    }
    let residual = total / pairs.max(1) as f64;
    Ok((out.into_pyarray(py), residual))
}

/// Spectral Leray projection of one (H, W) field pair with unit grid spacing.
/// Returns (vx_sol, vy_sol, residual RMS of central-difference divergence).
fn leray_project_2d(vx: &[f64], vy: &[f64], h: usize, w: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let two_pi = 2.0 * std::f64::consts::PI;
    // Signed FFT frequency index k in (-n/2, n/2]; symbol sin(2*pi*k/n).
    let sym = |k: usize, n: usize| -> f64 {
        let ks = if k <= n / 2 { k as f64 } else { k as f64 - n as f64 };
        (two_pi * ks / n as f64).sin()
    };
    let sxk: Vec<f64> = (0..w).map(|kx| sym(kx, w)).collect();
    let syk: Vec<f64> = (0..h).map(|ky| sym(ky, h)).collect();
    // Naive separable DFT (small grids); fft is not a dependency of this crate.
    let dft = |x: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let (mut re, mut im) = (vec![0.0; h * w], vec![0.0; h * w]);
        for ky in 0..h {
            for kx in 0..w {
                let (mut a, mut b) = (0.0, 0.0);
                for y in 0..h {
                    for xx in 0..w {
                        let ang = -two_pi * ((ky * y) as f64 / h as f64 + (kx * xx) as f64 / w as f64);
                        a += x[y * w + xx] * ang.cos();
                        b += x[y * w + xx] * ang.sin();
                    }
                }
                re[ky * w + kx] = a;
                im[ky * w + kx] = b;
            }
        }
        (re, im)
    };
    let (vxr, vxi) = dft(vx);
    let (vyr, vyi) = dft(vy);
    let mut phr = vec![0.0; h * w];
    let mut phi_ = vec![0.0; h * w];
    for ky in 0..h {
        for kx in 0..w {
            let i = ky * w + kx;
            let (sx, sy) = (sxk[kx], syk[ky]);
            let s_sq = if i == 0 { 1.0 } else { sx * sx + sy * sy };
            let dr = sx * vxr[i] + sy * vyr[i];
            let di = sx * vxi[i] + sy * vyi[i];
            if i != 0 {
                phr[i] = dr / s_sq;
                phi_[i] = di / s_sq;
            }
        }
    }
    // Inverse DFT for the corrected field (real part only).
    let idft = |re: &[f64], im: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; h * w];
        for y in 0..h {
            for xx in 0..w {
                let mut acc = 0.0;
                for ky in 0..h {
                    for kx in 0..w {
                        let i = ky * w + kx;
                        let ang = two_pi * ((ky * y) as f64 / h as f64 + (kx * xx) as f64 / w as f64);
                        acc += re[i] * ang.cos() - im[i] * ang.sin();
                    }
                }
                out[y * w + xx] = acc / (h * w) as f64;
            }
        }
        out
    };
    let mut cxr = vec![0.0; h * w];
    let mut cxi = vec![0.0; h * w];
    let mut cyr = vec![0.0; h * w];
    let mut cyi = vec![0.0; h * w];
    for ky in 0..h {
        for kx in 0..w {
            let i = ky * w + kx;
            let (sx, sy) = (sxk[kx], syk[ky]);
            cxr[i] = vxr[i] - sx * phr[i];
            cxi[i] = vxi[i] - sx * phi_[i];
            cyr[i] = vyr[i] - sy * phr[i];
            cyi[i] = vyi[i] - sy * phi_[i];
        }
    }
    let vx_sol = idft(&cxr, &cxi);
    let vy_sol = idft(&cyr, &cyi);
    let mut sq = 0.0;
    for y in 0..h {
        for xx in 0..w {
            let yp = (y + 1) % h;
            let ym = (y + h - 1) % h;
            let xp = (xx + 1) % w;
            let xm = (xx + w - 1) % w;
            let dvx = (vx_sol[y * w + xp] - vx_sol[y * w + xm]) / 2.0;
            let dvy = (vy_sol[yp * w + xx] - vy_sol[ym * w + xx]) / 2.0;
            sq += (dvx + dvy) * (dvx + dvy);
        }
    }
    (vx_sol, vy_sol, (sq / (h * w) as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    fn reconstruct_svd(u: &[f64], s: &[f64], vt: &[f64], m: usize, n: usize) -> Vec<f64> {
        let k = s.len();
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = (0..k).map(|q| u[i * k + q] * s[q] * vt[q * n + j]).sum();
            }
        }
        out
    }

    #[test]
    fn svd_reconstructs_tall_and_wide_matrices() {
        let tall: Vec<f64> = (0..12).map(|i| ((i * 7 % 5) as f64) - 1.5).collect();
        let (u, s, vt) = thin_svd(&tall, 4, 3);
        assert!(max_abs_diff(&reconstruct_svd(&u, &s, &vt, 4, 3), &tall) < 1e-10);
        let wide: Vec<f64> = (0..12).map(|i| ((i * 3 % 7) as f64) * 0.5).collect();
        let (u, s, vt) = thin_svd(&wide, 3, 4);
        assert!(max_abs_diff(&reconstruct_svd(&u, &s, &vt, 3, 4), &wide) < 1e-10);
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn acceleration_is_forward_difference_of_velocity() {
        let a = acceleration(&[1.0, 2.0], &[1.5, 1.0], 0.5).unwrap();
        assert_eq!(a, vec![1.0, -2.0]);
        assert!(acceleration(&[1.0], &[1.0], 0.0).is_err());
    }

    #[test]
    fn taylor_step_matches_formula() {
        let z = taylor_step(&[1.0], &[2.0], &[4.0], 0.5).unwrap();
        // 1 + 0.5*2 + 0.5*0.25*4 = 2.5
        assert!((z[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn tt_full_rank_reconstructs_exactly() {
        let dims = [3usize, 2, 2, 3];
        let x: Vec<f64> = (0..36).map(|i| ((i * 13 % 11) as f64) - 5.0).collect();
        let (cores, ranks) = tt_svd(&x, dims, 64, 0.0).unwrap();
        let [r1, r2, r3] = ranks;
        let (t, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let mut dense = vec![0.0; t * c * h * w];
        for ti in 0..t { for ci in 0..c { for hi in 0..h { for wi in 0..w {
            let mut acc = 0.0;
            for a in 0..r1 { for b in 0..r2 { for d in 0..r3 {
                acc += cores[0][ti * r1 + a]
                    * cores[1][(a * c + ci) * r2 + b]
                    * cores[2][(b * h + hi) * r3 + d]
                    * cores[3][d * w + wi];
            }}}
            dense[((ti * c + ci) * h + hi) * w + wi] = acc;
        }}}}
        assert!(max_abs_diff(&dense, &x) < 1e-8);
    }
}
