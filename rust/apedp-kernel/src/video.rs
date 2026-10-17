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

/// Peak SNR in dB between two flat sample arrays in [0, 1].
///
/// Mirrors `compute_psnr`: a mean squared error at or below `1e-12` reports
/// the 100 dB ceiling (an identical-or-near-identical pair would otherwise
/// be a log10 of a denormal-ish number).
pub fn psnr_series(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Shape mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
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
/// Constants mirror the reference call signature: `c1 = 0.01`, `c2 = 0.03`
/// (the squared luminance/contrast stabilizers), `win = 8` default.
pub fn ssim_frame<V: DataCol>(a_rows: &[V], b_rows: &[V], win: usize, c1: f64, c2: f64) -> f64 {
    if a_rows.is_empty() || b_rows.is_empty() || a_rows.len() != b_rows.len() {
        return f64::NAN;
    }
    let h = a_rows.len();
    let w = a_rows[0].row_len();
    if w == 0 || h < win || w < win {
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
            let u = if n_grid == 1 { 0.0 } else { j as f64 / (n_grid - 1) as f64 };
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
    fn ssim_frame_is_identity_one_and_nan_when_too_small() {
        let rows: Vec<Vec<f64>> = (0..16).map(|r| vec![0.3; 16]).collect();
        let rows_view: Vec<ArrayView1<f64>> = rows.iter().map(|r| ArrayView1::from(r.as_slice())).collect();
        let s = ssim_frame(&rows_view, &rows_view, 8, 0.01, 0.03);
        assert!((s - 1.0).abs() < 1e-12, "{s}");
        // A 4x8 frame cannot host one 8x8 window in h.
        let small: Vec<Vec<f64>> = (0..4).map(|_| vec![0.5; 8]).collect();
        let sv: Vec<ArrayView1<f64>> = small.iter().map(|r| ArrayView1::from(r.as_slice())).collect();
        assert!(ssim_frame(&sv, &sv, 8, 0.01, 0.03).is_nan());
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
}
