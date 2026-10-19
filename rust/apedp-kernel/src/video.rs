//! Video evaluation metrics (kernel first-line port of the reference
//! implementations in `apedp_dyn/video/eval_gate.py`).
//!
//! These are *metric definitions*, not referee thresholds: every constant
//! below (the PSNR non-finite floor, the SSIM window/contrast constants)
//! mirrors the reference implementation bit-for-bit in intent, so the two
//! engines certify the same number. None of them decides pass/fail — the
//! certification API keeps thresholds caller-owned.
//!
//! Data crossing: frame rows and feature rows arrive as borrowed
//! [`crate::data::DataCol`] views (owned `Vec<f64>` for the internal callers,
//! zero-copy `ArrayView1` straight out of NumPy buffers for the pyo3
//! surface), following the same pattern as the parser/genetic surfaces.
use crate::data::DataCol;
use rayon::prelude::*;

/// Peak SNR in dB between two flat sample arrays in [0, 1].
///
/// Mirrors `compute_psnr`: a mean squared error at or below `1e-12` reports
/// the 100 dB ceiling (an identical-or-near-identical pair would otherwise
/// be a log10 of a denormal-ish number).
pub fn psnr_series(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("Shape mismatch: {} vs {}", a.len(), b.len()));
    }
    if a.is_empty() {
        return Err("PSNR needs at least one sample".to_string());
    }
    let mut sum_sq = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum_sq += d * d;
    }
    let mse = sum_sq / a.len() as f64;
    if mse <= 1e-12 {
        return Ok(100.0);
    }
    Ok(10.0 * (1.0 / mse).log10())
}

/// Mean **windowed** SSIM over one grayscale frame.
///
/// `a_rows`/`b_rows` are the frame given as `h` row views of width `w`
/// (zero-copy NumPy rows on the Python side). Windows are the non-overlapping
/// `win x win` blocks the reference steps through; a frame smaller than the
/// window yields `NaN`, exactly like `np.mean([])` on the reference side
/// (an empty-signal pair is a data condition, not a verdict).
///
/// Constants are supplied by callers. The canonical values for unit-range
/// inputs are the SQUARED stabilizers `c1 = (K1*L)^2 = 1e-4`,
/// `c2 = (K2*L)^2 = 9e-4` (K1 = 0.01, K2 = 0.03, L = 1); `win = 8` default.
pub fn ssim_frame<V: DataCol>(a_rows: &[V], b_rows: &[V], win: usize, c1: f64, c2: f64) -> f64 {
    if a_rows.is_empty() || b_rows.is_empty() || a_rows.len() != b_rows.len() {
        return f64::NAN;
    }
    let h = a_rows.len();
    let w = a_rows[0].row_len();
    // `win == 0` must be rejected explicitly: `h < win` / `w < win` cannot be
    // true for an unsigned zero, so the window loops below would increment by
    // zero forever (100% CPU, unrecoverable from Python through PyO3).
    if win == 0 || w == 0 || h < win || w < win {
        return f64::NAN;
    }

    let mut vals: Vec<f64> = Vec::new();
    let mut y = 0usize;
    while y + win <= h {
        let mut x = 0usize;
        while x + win <= w {
            // Window statistics over the block rows.
            let mut ma = 0.0f64;
            let mut mb = 0.0f64;
            for r in 0..win {
                for col in x..x + win {
                    ma += a_rows[y + r].row(col);
                    mb += b_rows[y + r].row(col);
                }
            }
            let win_f = (win * win) as f64;
            ma /= win_f;
            mb /= win_f;
            let mut va = 0.0f64;
            let mut vb = 0.0f64;
            let mut cov = 0.0f64;
            for r in 0..win {
                for col in x..x + win {
                    let da = a_rows[y + r].row(col) - ma;
                    let db = b_rows[y + r].row(col) - mb;
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            va /= win_f;
            vb /= win_f;
            cov /= win_f;
            let num = (2.0 * ma * mb + c1) * (2.0 * cov + c2);
            let den = (ma * ma + mb * mb + c1) * (va + vb + c2);
            vals.push(num / den);
            x += win;
        }
        y += win;
    }

    if vals.is_empty() {
        f64::NAN
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

/// Sliced Wasserstein-2 distance between two point clouds, given explicit
/// projection directions.
///
/// x/y arrive as `n` row views of width `d` (zero-copy NumPy rows), and
/// `dir_rows` as `p` unit-normalized direction rows of the same width `d` —
/// the projection matrix is generated numpy-side (`RandomState(seed)`) so
/// both engines score identical projections and the distances are provably
/// equal, not merely close. Mirrors `sliced_wasserstein`: per direction,
/// quantile-matched sorted projections over an equal-length grid, mean of
/// squared grid differences, averaged over directions, then sqrt.
pub fn sliced_wasserstein2<V: DataCol>(
    x_rows: &[V],
    y_rows: &[V],
    dir_rows: &[V],
) -> Result<f64, String> {
    if x_rows.is_empty() || y_rows.is_empty() || dir_rows.is_empty() {
        return Err("Need rows for x, y and at least one projection direction".to_string());
    }
    let d = x_rows[0].row_len();
    if d == 0 {
        return Err("Point clouds need at least one feature dimension".to_string());
    }
    for r in x_rows.iter().chain(y_rows.iter()) {
        if r.row_len() != d {
            return Err("Feature dimension mismatch".to_string());
        }
    }
    for dir in dir_rows {
        if dir.row_len() != d {
            return Err("Projection direction width mismatch".to_string());
        }
    }

    let project = |rows: &[V], dir: &V| -> Vec<f64> {
        let mut out: Vec<f64> = Vec::with_capacity(rows.len());
        for r in rows {
            let mut acc = 0.0f64;
            for k in 0..d {
                acc += r.row(k) * dir.row(k);
            }
            out.push(acc);
        }
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out
    };

    let n_grid = x_rows.len().min(y_rows.len());
    let mut total = 0.0f64;
    for dir in dir_rows {
        let px = project(x_rows, dir);
        let py = project(y_rows, dir);
        let mut sq = 0.0f64;
        for j in 0..n_grid {
            // numpy: np.interp(linspace(0, 1, n), linspace(0, 1, len(px)), px)
            let u = if n_grid == 1 {
                0.0
            } else {
                j as f64 / (n_grid - 1) as f64
            };
            let vx = quantile_at(&px, u);
            let vy = quantile_at(&py, u);
            let diff = vx - vy;
            sq += diff * diff;
        }
        total += sq / n_grid as f64;
    }
    Ok((total / dir_rows.len() as f64).sqrt())
}

/// numpy.interp over the sorted quantiles of one projection: the sorted
/// array is the piecewise-linear function of the unit interval taken at
/// evenly spaced nodes.
fn quantile_at(sorted: &[f64], u: f64) -> f64 {
    let m = sorted.len();
    if m == 0 {
        return f64::NAN;
    }
    if m == 1 || u <= 0.0 {
        return sorted[0];
    }
    if u >= 1.0 {
        return sorted[m - 1];
    }
    // Node positions i/(m-1); find the bracketing nodes for u.
    let pos = u * (m - 1) as f64;
    let i = pos.floor() as usize;
    let i = i.min(m - 2);
    let frac = pos - i as f64;
    sorted[i] * (1.0 - frac) + sorted[i + 1] * frac
}

/// 2D Bilinear Sub-Pixel Coordinate Warping of a multi-channel image/latent tensor:
///   I_out(c, y, x) = I_in(c, y + vy(y, x), x + vx(y, x))
/// using border clamping and Rayon parallelization across channels and planes.
pub fn warp_bilinear_2d(
    input: &[f64],
    vx: &[f64],
    vy: &[f64],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f64>, String> {
    let plane_len = height * width;
    if input.len() != channels * plane_len {
        return Err(format!(
            "Input length {} != channels {} * height {} * width {}",
            input.len(),
            channels,
            height,
            width
        ));
    }
    if vx.len() != plane_len || vy.len() != plane_len {
        return Err(format!(
            "Flow lengths ({}, {}) != plane_size {}",
            vx.len(),
            vy.len(),
            plane_len
        ));
    }
    if height < 2 || width < 2 {
        return Err("Dimensions must be at least 2x2".to_string());
    }

    let mut output = vec![0.0f64; input.len()];
    let max_x = (width - 1) as f64;
    let max_y = (height - 1) as f64;

    output
        .par_chunks_mut(plane_len)
        .enumerate()
        .for_each(|(c, out_plane)| {
            let in_plane = &input[c * plane_len..(c + 1) * plane_len];
            for y in 0..height {
                let row_idx = y * width;
                for x in 0..width {
                    let idx = row_idx + x;
                    // Source sample position
                    let sx = ((x as f64) + vx[idx]).clamp(0.0, max_x);
                    let sy = ((y as f64) + vy[idx]).clamp(0.0, max_y);

                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = (x0 + 1).min(width - 1);
                    let y1 = (y0 + 1).min(height - 1);

                    let wx = sx - (x0 as f64);
                    let wy = sy - (y0 as f64);

                    let top_left = in_plane[y0 * width + x0];
                    let top_right = in_plane[y0 * width + x1];
                    let bottom_left = in_plane[y1 * width + x0];
                    let bottom_right = in_plane[y1 * width + x1];

                    let top = top_left * (1.0 - wx) + top_right * wx;
                    let bottom = bottom_left * (1.0 - wx) + bottom_right * wx;

                    out_plane[idx] = top * (1.0 - wy) + bottom * wy;
                }
            }
        });

    Ok(output)
}

/// 2D Bilinear Sub-Pixel Coordinate Warping with PERIODIC (wrap-around)
/// boundary conditions:
///   I_out(c, y, x) = I_in(c, wrap(y + vy), wrap(x + vx))
///
/// Indexing mirrors the fluid-domain generator's semi-Lagrangian backtrace
/// exactly: the backtrace coordinate is floored FIRST, the integer index is
/// wrapped with `rem_euclid` (numpy `%` semantics for negative coordinates),
/// and the fractional weight is `q - floor(q)`. Floor-before-wrap matters:
/// wrapping the float coordinate first loses exactness for tiny negative
/// backtraces (a rounded `rem_euclid` can land on `size` instead of `0`).
pub fn warp_bilinear_2d_periodic(
    input: &[f64],
    vx: &[f64],
    vy: &[f64],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f64>, String> {
    let plane_len = height * width;
    if input.len() != channels * plane_len {
        return Err(format!(
            "Input length {} != channels {} * height {} * width {}",
            input.len(),
            channels,
            height,
            width
        ));
    }
    if vx.len() != plane_len || vy.len() != plane_len {
        return Err(format!(
            "Flow lengths ({}, {}) != plane_size {}",
            vx.len(),
            vy.len(),
            plane_len
        ));
    }
    if height < 2 || width < 2 {
        return Err("Dimensions must be at least 2x2".to_string());
    }

    let mut output = vec![0.0f64; input.len()];
    let w_i = width as i64;
    let h_i = height as i64;

    output
        .par_chunks_mut(plane_len)
        .enumerate()
        .for_each(|(c, out_plane)| {
            let in_plane = &input[c * plane_len..(c + 1) * plane_len];
            for y in 0..height {
                let row_idx = y * width;
                for x in 0..width {
                    let idx = row_idx + x;
                    // Backtrace source position (unclamped, un-wrapped).
                    let qx = (x as f64) + vx[idx];
                    let qy = (y as f64) + vy[idx];
                    // Floor first, then wrap the integer index (numpy `%`).
                    let fx = qx.floor();
                    let fy = qy.floor();
                    let x0 = (fx as i64).rem_euclid(w_i) as usize;
                    let y0 = (fy as i64).rem_euclid(h_i) as usize;
                    let x1 = (x0 + 1) % width;
                    let y1 = (y0 + 1) % height;

                    let wx = qx - fx;
                    let wy = qy - fy;

                    let top_left = in_plane[y0 * width + x0];
                    let top_right = in_plane[y0 * width + x1];
                    let bottom_left = in_plane[y1 * width + x0];
                    let bottom_right = in_plane[y1 * width + x1];

                    let top = top_left * (1.0 - wx) + top_right * wx;
                    let bottom = bottom_left * (1.0 - wx) + bottom_right * wx;

                    out_plane[idx] = top * (1.0 - wy) + bottom * wy;
                }
            }
        });

    Ok(output)
}

/// Compute spatial optical flow divergence:
///   div(v) = (vx(x+1, y) - vx(x-1, y)) / (2*dx) + (vy(x, y+1) - vy(x, y-1)) / (2*dy)
pub fn compute_optical_flow_divergence(
    vx: &[f64],
    vy: &[f64],
    height: usize,
    width: usize,
    dx: f64,
    dy: f64,
) -> Result<Vec<f64>, String> {
    let n = height * width;
    if vx.len() != n || vy.len() != n {
        return Err(format!(
            "Velocity lengths ({}, {}) != {}",
            vx.len(),
            vy.len(),
            n
        ));
    }
    if height < 3 || width < 3 {
        return Err("Grid dimensions must be at least 3x3".to_string());
    }
    if dx <= 0.0 || dy <= 0.0 {
        return Err("dx and dy must be positive".to_string());
    }

    let inv_2dx = 1.0 / (2.0 * dx);
    let inv_2dy = 1.0 / (2.0 * dy);
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;
    let mut div = vec![0.0f64; n];

    for y in 0..height {
        let ym = if y == 0 { 0 } else { y - 1 };
        let yp = if y + 1 >= height { height - 1 } else { y + 1 };
        let row_curr = y * width;
        let row_m = ym * width;
        let row_p = yp * width;
        // Border rows use a ONE-SIDED difference: clamping the neighbour index
        // at the border (xm = 0 when x = 0) makes the numerator span a single
        // cell while the divisor stays 2*dx, halving every border derivative.
        let (scale_y, inv_dy_eff) = if y == 0 || y + 1 >= height {
            (1.0, inv_dy)
        } else {
            (2.0, inv_2dy)
        };
        for x in 0..width {
            let xm = if x == 0 { 0 } else { x - 1 };
            let xp = if x + 1 >= width { width - 1 } else { x + 1 };
            let border_x = x == 0 || x + 1 >= width;
            let inv_dx_eff = if border_x { inv_dx } else { inv_2dx };
            let _ = scale_y;

            let dvx = (vx[row_curr + xp] - vx[row_curr + xm]) * inv_dx_eff;
            let dvy = (vy[row_p + x] - vy[row_m + x]) * inv_dy_eff;
            div[row_curr + x] = dvx + dvy;
        }
    }
    Ok(div)
}

/// Detect disocclusion mask based on positive flow divergence:
/// Mask > 0 where new surfaces/occlusions are exposed, triggering residual injection s(t).
pub fn detect_disocclusion_mask(div_flow: &[f64], divergence_threshold: f64) -> Vec<f64> {
    div_flow
        .iter()
        .map(|&d| {
            if d > divergence_threshold {
                (d - divergence_threshold).min(1.0)
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayView1;

    fn view(v: &[f64]) -> ArrayView1<'_, f64> {
        ArrayView1::from(v)
    }

    #[test]
    fn psnr_matches_reference_semantics() {
        let a = vec![0.1, 0.2, 0.3];
        assert_eq!(psnr_series(&a, &a).unwrap(), 100.0);
        let b = vec![0.15, 0.25, 0.35];
        let mse = (0.05f64.powi(2) * 3.0) / 3.0;
        let expect = 10.0 * (1.0 / mse).log10();
        assert!((psnr_series(&a, &b).unwrap() - expect).abs() < 1e-12);
        assert!(psnr_series(&a, &b[..2]).is_err());
        assert!(psnr_series(&a, &[]).is_err());
    }

    #[test]
    fn divergence_border_uses_one_sided_difference() {
        // vx = x (so d(vx)/dx == 1 everywhere, borders included), vy = 0.
        // The old clamped-neighbour central difference halved the border values.
        let (h, w) = (5usize, 6usize);
        let mut vx = vec![0.0f64; h * w];
        let vy = vec![0.0f64; h * w];
        for y in 0..h {
            for x in 0..w {
                vx[y * w + x] = x as f64;
            }
        }
        let div = compute_optical_flow_divergence(&vx, &vy, h, w, 1.0, 1.0).unwrap();
        for (i, d) in div.iter().enumerate() {
            assert!(
                (d - 1.0).abs() < 1e-12,
                "divergence at {i} = {d}, expected 1.0"
            );
        }

        // Same in y for a vy = y field.
        let vx0 = vec![0.0f64; h * w];
        let mut vy = vec![0.0f64; h * w];
        for y in 0..h {
            for x in 0..w {
                vy[y * w + x] = y as f64;
            }
        }
        let div_y = compute_optical_flow_divergence(&vx0, &vy, h, w, 1.0, 1.0).unwrap();
        for (i, d) in div_y.iter().enumerate() {
            assert!(
                (d - 1.0).abs() < 1e-12,
                "y-divergence at {i} = {d}, expected 1.0"
            );
        }
    }

    #[test]
    fn ssim_frame_zero_window_returns_nan_instead_of_hanging() {
        // win == 0 used to pass the unsigned size guard (`h < 0` is never true)
        // and then spin forever in `x += win`. The call must return promptly.
        let rows: Vec<Vec<f64>> = (0..8).map(|r| vec![0.2 + r as f64 * 0.01; 8]).collect();
        let rows_view: Vec<ArrayView1<f64>> = rows
            .iter()
            .map(|r| ArrayView1::from(r.as_slice()))
            .collect();
        assert!(ssim_frame(&rows_view, &rows_view, 0, 1e-4, 9e-4).is_nan());
    }

    #[test]
    fn ssim_frame_is_identity_one_and_nan_when_too_small() {
        let rows: Vec<Vec<f64>> = (0..16).map(|_| vec![0.3; 16]).collect();
        let rows_view: Vec<ArrayView1<f64>> = rows
            .iter()
            .map(|r| ArrayView1::from(r.as_slice()))
            .collect();
        let s = ssim_frame(&rows_view, &rows_view, 8, 1e-4, 9e-4);
        assert!((s - 1.0).abs() < 1e-12, "{s}");
        // A 4x8 frame cannot host one 8x8 window in h.
        let small: Vec<Vec<f64>> = (0..4).map(|_| vec![0.5; 8]).collect();
        let sv: Vec<ArrayView1<f64>> = small
            .iter()
            .map(|r| ArrayView1::from(r.as_slice()))
            .collect();
        assert!(ssim_frame(&sv, &sv, 8, 1e-4, 9e-4).is_nan());
    }

    #[test]
    fn sliced_wasserstein2_zero_for_identical_grows_for_shifted() {
        // 30 samples of width 2; directions must carry the cloud width.
        let make_rows = |shift: f64| -> Vec<Vec<f64>> {
            (0..30)
                .map(|i| {
                    let fi = i as f64;
                    vec![fi * 0.1 - 1.5 + shift, ((i * 7919) % 13) as f64 * 0.05]
                })
                .collect()
        };
        let x = make_rows(0.0);
        let xs: Vec<ArrayView1<f64>> = x.iter().map(|r| ArrayView1::from(r.as_slice())).collect();
        let dir = vec![1.0, 0.0];
        let dir_view = view(&dir);
        let s_same = sliced_wasserstein2(&xs, &xs, &[dir_view]).unwrap();
        assert!(s_same < 1e-12);
        // Shift the first dimension: distance must be positive.
        let y = make_rows(0.7);
        let ys: Vec<ArrayView1<f64>> = y.iter().map(|r| ArrayView1::from(r.as_slice())).collect();
        let s_shift = sliced_wasserstein2(&xs, &ys, &[dir_view]).unwrap();
        assert!(s_shift > 0.5, "{s_shift}");
        assert!(s_shift > 0.5, "{s_shift}");
        assert!(sliced_wasserstein2(&xs, &ys, &[]).is_err());
    }

    #[test]
    fn quantile_interp_matches_numpy_semantics() {
        let sorted = vec![1.0, 2.0, 4.0];
        // Nodes sit at u = 0, 0.5, 1.0 -> np.interp(0.5) reads the middle node.
        assert_eq!(quantile_at(&sorted, 0.0), 1.0);
        assert_eq!(quantile_at(&vec![7.0], 0.5), 7.0); // single node: constant
        assert!((quantile_at(&sorted, 1.0) - 4.0).abs() < 1e-15);
        assert!((quantile_at(&sorted, 0.5) - 2.0).abs() < 1e-15);
        // Between nodes: halfway between u=0.5 and u=1.0 nodes.
        assert!((quantile_at(&sorted, 0.75) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn warp_bilinear_2d_zero_flow_is_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let vx = vec![0.0; 9];
        let vy = vec![0.0; 9];
        let warped = warp_bilinear_2d(&input, &vx, &vy, 1, 3, 3).unwrap();
        assert_eq!(warped, input);
    }

    #[test]
    fn warp_bilinear_2d_translation_and_clamping() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Shift x by +0.5 (half pixel right)
        let vx = vec![0.5; 4];
        let vy = vec![0.0; 4];
        let warped = warp_bilinear_2d(&input, &vx, &vy, 1, 2, 2).unwrap();
        // (0,0) samples halfway between 1.0 and 2.0 -> 1.5
        assert!((warped[0] - 1.5).abs() < 1e-12);
        // (1,0) samples beyond right border, clamped at 2.0 -> 2.0
        assert!((warped[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn warp_bilinear_2d_periodic_zero_flow_is_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let vx = vec![0.0; 9];
        let vy = vec![0.0; 9];
        let warped = warp_bilinear_2d_periodic(&input, &vx, &vy, 1, 3, 3).unwrap();
        assert_eq!(warped, input);
    }

    #[test]
    fn warp_bilinear_2d_periodic_wraps_across_border() {
        // Row 0: 1 2, row 1: 3 4. Shift x by +1 with wrap:
        // out(1,0) = in(0,0) = 1.0 (instead of clamping to 2.0).
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let vx = vec![1.0; 4];
        let vy = vec![0.0; 4];
        let warped = warp_bilinear_2d_periodic(&input, &vx, &vy, 1, 2, 2).unwrap();
        assert!((warped[0] - 2.0).abs() < 1e-15); // out(0,0) = in(1,0)
        assert!((warped[1] - 1.0).abs() < 1e-15); // out(1,0) = in(0,0) wrapped
        assert!((warped[2] - 4.0).abs() < 1e-15);
        assert!((warped[3] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn warp_bilinear_2d_periodic_negative_backtrace_matches_floor_mod() {
        // q = -0.5 must sample between the last and first columns with
        // weight 0.5 (numpy floor-mod semantics), NOT clamp.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let vx = vec![-0.5; 4];
        let vy = vec![0.0; 4];
        let warped = warp_bilinear_2d_periodic(&input, &vx, &vy, 1, 2, 2).unwrap();
        // out(0,0): 0.5 * in(1,0) + 0.5 * in(0,0) = 0.5*2 + 0.5*1 = 1.5
        assert!((warped[0] - 1.5).abs() < 1e-15);
        assert!((warped[1] - 1.5).abs() < 1e-15);
    }

    #[test]
    fn optical_flow_divergence_and_disocclusion_mask() {
        let h = 4;
        let w = 4;
        // Divergent flow: vx expands along x, vy expands along y
        let mut vx = vec![0.0; h * w];
        let mut vy = vec![0.0; h * w];
        for y in 0..h {
            for x in 0..w {
                vx[y * w + x] = x as f64 * 0.2;
                vy[y * w + x] = y as f64 * 0.2;
            }
        }
        let div = compute_optical_flow_divergence(&vx, &vy, h, w, 1.0, 1.0).unwrap();
        // Divergence should be positive everywhere in the interior (approx 0.2 + 0.2 = 0.4)
        assert!(div[1 * w + 1] > 0.3);
        let mask = detect_disocclusion_mask(&div, 0.2);
        assert!(mask[1 * w + 1] > 0.1);
        assert_eq!(mask.len(), h * w);
    }
}
