//! The first pass of the law search, fast: every enumerated expression fitted
//! to the rows with a scale and an offset, and sums of two of them.
//!
//! The reference is Python: `apedp_dyn/discovery_l2/exhaustive_search.py`
//! (`evaluate`, `_fit`) and `guided_search.py` (`residual_structure`). Those
//! files are frozen by preregistrations and are the definition of the
//! result. This module computes the same quantities in one pass over the
//! expression table, sharing every sub-expression's column, and it is *not*
//! bit-identical to NumPy and LAPACK: transcendental functions may differ in
//! the last digit and the least squares is solved by orthogonalisation, not by
//! a singular value decomposition.
//!
//! So nothing here decides. The caller
//! (`apedp_dyn/discovery_l2/fast_first_pass.py`) uses these numbers to find
//! the few candidates a decision can turn on and recomputes those with the
//! reference. To let it do that safely each fit comes with a `condition`
//! number, from which the caller bounds how far the reference's value can be
//! from this one, and with a status that says when this module would rather
//! not judge (`NEAR_RANK_THRESHOLD`).

/// Operation codes of the expression table. `VAR` reads column `left` of the rows.
pub const VAR: u8 = 0;
pub const SQRT: u8 = 1;
pub const EXP: u8 = 2;
pub const LOG: u8 = 3;
pub const SIN: u8 = 4;
pub const COS: u8 = 5;
pub const INV: u8 = 6;
pub const SQ: u8 = 7;
pub const ADD: u8 = 8;
pub const SUB: u8 = 9;
pub const MUL: u8 = 10;
pub const DIV: u8 = 11;

/// The fit exists.
pub const FITTED: u8 = 0;
/// A value of the column is not finite: the reference returns no fit.
pub const NOT_FINITE: u8 = 1;
/// The column is constant to working precision: the reference returns no fit.
pub const RANK_DEFICIENT: u8 = 2;
/// Too close to the reference's rank threshold to call here; the caller asks the reference.
pub const NEAR_RANK_THRESHOLD: u8 = 3;

/// Share of the rank threshold inside which this module does not judge.
const BAND: f64 = 0.1;

/// What `singles` returns, one entry per expression.
pub struct Singles {
    pub rss: Vec<f64>,
    pub slope: Vec<f64>,
    pub intercept: Vec<f64>,
    /// Leave-one-out R^2 of each residual from its neighbours' residuals (`residual_structure`).
    pub structure: Vec<f64>,
    /// How ill-conditioned the fit is: the larger of the design's condition number and of the
    /// column's size over its variation. Bounds the difference to the reference's value.
    pub condition: Vec<f64>,
    pub status: Vec<u8>,
}

/// What `pairs` returns, row-major over (first, partner).
pub struct Pairs {
    pub rss: Vec<f64>,
    pub condition: Vec<f64>,
    pub status: Vec<u8>,
}

fn apply_unary(op: u8, child: &[f64], out: &mut [f64]) {
    match op {
        SQRT => out.iter_mut().zip(child).for_each(|(o, &a)| *o = a.sqrt()),
        EXP => out.iter_mut().zip(child).for_each(|(o, &a)| *o = a.exp()),
        LOG => out.iter_mut().zip(child).for_each(|(o, &a)| *o = a.ln()),
        SIN => out.iter_mut().zip(child).for_each(|(o, &a)| *o = a.sin()),
        COS => out.iter_mut().zip(child).for_each(|(o, &a)| *o = a.cos()),
        INV => out.iter_mut().zip(child).for_each(|(o, &a)| *o = 1.0 / a),
        _ => out.iter_mut().zip(child).for_each(|(o, &a)| *o = a * a),
    }
}

fn apply_binary(op: u8, a: &[f64], b: &[f64], out: &mut [f64]) {
    match op {
        ADD => out
            .iter_mut()
            .zip(a.iter().zip(b))
            .for_each(|(o, (&p, &q))| *o = p + q),
        SUB => out
            .iter_mut()
            .zip(a.iter().zip(b))
            .for_each(|(o, (&p, &q))| *o = p - q),
        MUL => out
            .iter_mut()
            .zip(a.iter().zip(b))
            .for_each(|(o, (&p, &q))| *o = p * q),
        _ => out
            .iter_mut()
            .zip(a.iter().zip(b))
            .for_each(|(o, (&p, &q))| *o = p / q),
    }
}

/// NumPy's threshold for `lstsq(rcond=None)`: machine epsilon times the larger dimension.
fn rank_threshold(rows: usize, columns: usize) -> f64 {
    f64::EPSILON * (rows.max(columns) as f64)
}

/// Fit `y ~ slope * column + intercept` on centred targets `yc` (mean `y_mean`), then the residual check.
#[allow(clippy::too_many_arguments)]
fn fit_one(
    column: &[f64],
    yc: &[f64],
    y_mean: f64,
    nn: &[u32],
    k: usize,
    floor: f64,
    resid: &mut [f64],
) -> (u8, f64, f64, f64, f64, f64) {
    let n = column.len();
    let nf = n as f64;
    let mut sum = 0.0;
    let mut norm2 = 0.0;
    for &c in column {
        if !c.is_finite() {
            return (NOT_FINITE, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }
        sum += c;
        norm2 += c * c;
    }
    let mean = sum / nf;
    let mut d2 = 0.0;
    let mut sxy = 0.0;
    for (&c, &t) in column.iter().zip(yc) {
        let v = c - mean;
        d2 += v * v;
        sxy += v * t;
    }
    // Singular values of [column, 1] from its triangular factor [[sqrt(n), sum/sqrt(n)], [0, sqrt(d2)]].
    let t = nf + norm2;
    let smax2 = 0.5 * (t + (t * t - 4.0 * nf * d2).max(0.0).sqrt());
    let ratio = if smax2 > 0.0 {
        (nf * d2).sqrt() / smax2
    } else {
        0.0
    };
    let threshold = rank_threshold(n, 2);
    if !(ratio > (1.0 - BAND) * threshold) {
        return (
            RANK_DEFICIENT,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        );
    }
    let status = if ratio < (1.0 + BAND) * threshold {
        NEAR_RANK_THRESHOLD
    } else {
        FITTED
    };
    let slope = sxy / d2;
    let mut rss = 0.0;
    let mut resid_sum = 0.0;
    for ((r, &c), &t) in resid.iter_mut().zip(column).zip(yc) {
        *r = t - slope * (c - mean);
        rss += *r * *r;
        resid_sum += *r;
    }
    let resid_mean = resid_sum / nf;
    let mut total = 0.0;
    for &r in resid.iter() {
        total += (r - resid_mean) * (r - resid_mean);
    }
    let structure = if total <= 0.0 || k == 0 {
        -1.0
    } else {
        let mut unexplained = 0.0;
        for (i, &r) in resid.iter().enumerate() {
            let mut near = 0.0;
            for &j in &nn[i * k..(i + 1) * k] {
                near += resid[j as usize];
            }
            let gap = r - near / (k as f64);
            unexplained += gap * gap;
        }
        1.0 - unexplained / total
    };
    let condition = (1.0 / ratio).max((norm2 / d2).sqrt());
    (
        status,
        rss.max(floor),
        slope,
        y_mean - slope * mean,
        structure,
        condition,
    )
}

/// Every expression of the table fitted to `y` with a scale and an offset.
///
/// `ops`, `left`, `right` describe the expressions in enumeration order; a child always comes
/// before its parent. `x` holds the rows, row-major, `d` columns. `nn` holds each row's `k`
/// nearest neighbours, row-major. `floor` is the least residual sum of squares counted.
#[allow(clippy::too_many_arguments)]
pub fn singles(
    ops: &[u8],
    left: &[i32],
    right: &[i32],
    x: &[f64],
    d: usize,
    y: &[f64],
    nn: &[u32],
    k: usize,
    floor: f64,
) -> Result<Singles, String> {
    let count = ops.len();
    let n = y.len();
    if left.len() != count || right.len() != count {
        return Err("ops, left and right must have one entry per expression".into());
    }
    if n == 0 || d == 0 || x.len() != n * d {
        return Err("x must hold len(y) rows of d columns, and there must be rows".into());
    }
    if nn.len() != n * k || nn.iter().any(|&j| j as usize >= n) {
        return Err("nn must hold k row indices for every row".into());
    }
    // Which expressions are somebody's child: only their columns are kept.
    let mut slot: Vec<usize> = vec![usize::MAX; count];
    let mut kept = 0usize;
    for i in 0..count {
        let (children, arity): ([i32; 2], usize) = match ops[i] {
            VAR => ([0, 0], 0),
            SQRT..=SQ => ([left[i], 0], 1),
            ADD..=DIV => ([left[i], right[i]], 2),
            other => return Err(format!("unknown operation code {other}")),
        };
        if ops[i] == VAR && (left[i] < 0 || left[i] as usize >= d) {
            return Err(format!(
                "expression {i} reads a column the rows do not have"
            ));
        }
        for &child in &children[..arity] {
            if child < 0 || child as usize >= i {
                return Err(format!(
                    "expression {i} has a child that does not come before it"
                ));
            }
            if slot[child as usize] == usize::MAX {
                slot[child as usize] = kept;
                kept += 1;
            }
        }
    }
    let mut store = vec![0.0f64; kept * n];
    let mut scratch = vec![0.0f64; n];
    let mut resid = vec![0.0f64; n];
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let yc: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();
    let mut out = Singles {
        rss: vec![f64::NAN; count],
        slope: vec![f64::NAN; count],
        intercept: vec![f64::NAN; count],
        structure: vec![f64::NAN; count],
        condition: vec![f64::NAN; count],
        status: vec![NOT_FINITE; count],
    };
    for i in 0..count {
        // A child's column lives in an earlier part of the store than its parent's, so the store
        // can be split there and the parent's column written while the children's are read.
        let (before, after) = if slot[i] == usize::MAX {
            (&store[..], &mut scratch[..])
        } else {
            let (head, tail) = store.split_at_mut(slot[i] * n);
            (&*head, &mut tail[..n])
        };
        match ops[i] {
            VAR => {
                let column = left[i] as usize;
                for (row, value) in after.iter_mut().enumerate() {
                    *value = x[row * d + column];
                }
            }
            SQRT..=SQ => {
                let child = slot[left[i] as usize] * n;
                apply_unary(ops[i], &before[child..child + n], after);
            }
            _ => {
                let a = slot[left[i] as usize] * n;
                let b = slot[right[i] as usize] * n;
                apply_binary(ops[i], &before[a..a + n], &before[b..b + n], after);
            }
        }
        let (status, rss, slope, intercept, structure, condition) =
            fit_one(after, &yc, y_mean, nn, k, floor, &mut resid);
        out.status[i] = status;
        out.rss[i] = rss;
        out.slope[i] = slope;
        out.intercept[i] = intercept;
        out.structure[i] = structure;
        out.condition[i] = condition;
    }
    Ok(out)
}

/// Singular values of an upper-triangular 3x3 matrix by one-sided Jacobi rotations: largest and smallest.
fn extreme_singular_values(r: [[f64; 3]; 3]) -> (f64, f64) {
    let mut a = r;
    for _ in 0..30 {
        let mut off = 0.0f64;
        for p in 0..2 {
            for q in (p + 1)..3 {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for row in &a {
                    alpha += row[p] * row[p];
                    beta += row[q] * row[q];
                    gamma += row[p] * row[q];
                }
                if gamma == 0.0 {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt().max(f64::MIN_POSITIVE));
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for row in a.iter_mut() {
                    let (u, v) = (row[p], row[q]);
                    row[p] = c * u - s * v;
                    row[q] = s * u + c * v;
                }
            }
        }
        if off < 1e-15 {
            break;
        }
    }
    let mut norms = [0.0f64; 3];
    for (j, norm) in norms.iter_mut().enumerate() {
        *norm = a.iter().map(|row| row[j] * row[j]).sum::<f64>().sqrt();
    }
    (
        norms.iter().cloned().fold(0.0, f64::max),
        norms.iter().cloned().fold(f64::INFINITY, f64::min),
    )
}

/// `y ~ a * first + b * partner + c` for every (first, partner): residual sums of squares.
///
/// `first` and `partners` hold columns of `len(y)` rows each, one after another.
pub fn pairs(first: &[f64], partners: &[f64], y: &[f64], floor: f64) -> Result<Pairs, String> {
    let n = y.len();
    if n == 0 || first.len() % n != 0 || partners.len() % n != 0 {
        return Err("first and partners must hold whole columns of len(y) rows".into());
    }
    let (firsts, others) = (first.len() / n, partners.len() / n);
    let nf = n as f64;
    let y_mean = y.iter().sum::<f64>() / nf;
    let yc: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();
    let threshold = rank_threshold(n, 3);
    let mut out = Pairs {
        rss: vec![f64::NAN; firsts * others],
        condition: vec![f64::NAN; firsts * others],
        status: vec![NOT_FINITE; firsts * others],
    };
    // Each partner centred once.
    let mut centred = vec![0.0f64; partners.len()];
    let mut partner_sum = vec![0.0f64; others];
    let mut partner_norm2 = vec![0.0f64; others];
    let mut partner_finite = vec![true; others];
    for p in 0..others {
        let column = &partners[p * n..(p + 1) * n];
        partner_finite[p] = column.iter().all(|v| v.is_finite());
        if !partner_finite[p] {
            continue;
        }
        partner_sum[p] = column.iter().sum();
        partner_norm2[p] = column.iter().map(|v| v * v).sum();
        let mean = partner_sum[p] / nf;
        for (c, &v) in centred[p * n..(p + 1) * n].iter_mut().zip(column) {
            *c = v - mean;
        }
    }
    let mut v1 = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];
    for f in 0..firsts {
        let column = &first[f * n..(f + 1) * n];
        if !column.iter().all(|v| v.is_finite()) {
            continue;
        }
        let sum1: f64 = column.iter().sum();
        let norm1: f64 = column.iter().map(|v| v * v).sum();
        let mean1 = sum1 / nf;
        let mut d1 = 0.0;
        let mut y1 = 0.0;
        for ((v, &c), &t) in v1.iter_mut().zip(column).zip(&yc) {
            *v = c - mean1;
            d1 += *v * *v;
            y1 += *v * t;
        }
        for p in 0..others {
            let at = f * others + p;
            if !partner_finite[p] {
                continue;
            }
            let v2 = &centred[p * n..(p + 1) * n];
            let g: f64 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
            let proj = if d1 > 0.0 { g / d1 } else { 0.0 };
            let mut d2 = 0.0;
            let mut y2 = 0.0;
            for (((o, &a), &b), &t) in w.iter_mut().zip(&v1).zip(v2).zip(&yc) {
                *o = b - proj * a;
                d2 += *o * *o;
                y2 += *o * t;
            }
            let (r11, r22, r33) = (nf.sqrt(), d1.sqrt(), d2.sqrt());
            let r23 = if r22 > 0.0 { g / r22 } else { 0.0 };
            let (smax, smin) = extreme_singular_values([
                [r11, sum1 / r11, partner_sum[p] / r11],
                [0.0, r22, r23],
                [0.0, 0.0, r33],
            ]);
            let ratio = if smax > 0.0 { smin / smax } else { 0.0 };
            if !(ratio > (1.0 - BAND) * threshold) {
                out.status[at] = RANK_DEFICIENT;
                continue;
            }
            out.status[at] = if ratio < (1.0 + BAND) * threshold {
                NEAR_RANK_THRESHOLD
            } else {
                FITTED
            };
            let (a1, a2) = (y1 / d1, y2 / d2);
            let mut rss = 0.0;
            for ((&t, &a), &b) in yc.iter().zip(&v1).zip(w.iter()) {
                let r = t - a1 * a - a2 * b;
                rss += r * r;
            }
            out.rss[at] = rss.max(floor);
            out.condition[at] = (1.0 / ratio)
                .max((norm1 / d1).sqrt())
                .max((partner_norm2[p] / d2).sqrt());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> (Vec<f64>, Vec<f64>, Vec<u32>) {
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v.sqrt() - 2.0).collect();
        let nn: Vec<u32> = (0..n)
            .flat_map(|i| [((i + 1) % n) as u32, ((i + n - 1) % n) as u32])
            .collect();
        (x, y, nn)
    }

    #[test]
    fn the_law_is_fitted_exactly_and_shares_its_child() {
        let (x, y, nn) = rows();
        // x0, sqrt(x0), exp(sqrt(x0)), x0 + sqrt(x0)
        let fits = singles(
            &[VAR, SQRT, EXP, ADD],
            &[0, 0, 1, 0],
            &[-1, -1, -1, 1],
            &x,
            1,
            &y,
            &nn,
            2,
            0.0,
        )
        .unwrap();
        assert_eq!(fits.status, vec![FITTED; 4]);
        assert!(fits.rss[1] < 1e-24, "{}", fits.rss[1]);
        assert!((fits.slope[1] - 3.0).abs() < 1e-12 && (fits.intercept[1] + 2.0).abs() < 1e-12);
        assert!(fits.rss[0] > 1e-3 && fits.rss[2] > 1e-3 && fits.rss[3] > 1e-6);
        assert!(
            fits.structure[0] > 0.5,
            "a wrong law leaves structure: {}",
            fits.structure[0]
        );
    }

    #[test]
    fn constants_and_undefined_values_are_not_fitted() {
        let (x, y, nn) = rows();
        // x0 - x0 is constant; log(x0 - x0) is not finite
        let fits = singles(
            &[VAR, SUB, LOG],
            &[0, 0, 1],
            &[-1, 0, -1],
            &x,
            1,
            &y,
            &nn,
            2,
            0.0,
        )
        .unwrap();
        assert_eq!(fits.status, vec![FITTED, RANK_DEFICIENT, NOT_FINITE]);
    }

    #[test]
    fn the_floor_is_the_least_sum_of_squares_counted() {
        let (x, y, nn) = rows();
        let fits = singles(&[VAR, SQRT], &[0, 0], &[-1, -1], &x, 1, &y, &nn, 2, 1e-6).unwrap();
        assert_eq!(fits.rss[1], 1e-6);
    }

    #[test]
    fn a_table_out_of_order_is_refused() {
        let (x, y, nn) = rows();
        assert!(singles(&[SQRT, VAR], &[1, 0], &[-1, -1], &x, 1, &y, &nn, 2, 0.0).is_err());
        assert!(singles(&[VAR], &[3], &[-1], &x, 1, &y, &nn, 2, 0.0).is_err());
        assert!(singles(&[99], &[0], &[-1], &x, 1, &y, &nn, 2, 0.0).is_err());
    }

    #[test]
    fn a_sum_of_two_terms_is_fitted_and_a_repeated_term_is_not() {
        let n = 50;
        let a: Vec<f64> = (0..n).map(|i| 1.0 + 0.07 * i as f64).collect();
        let b: Vec<f64> = a.iter().map(|v| (1.3 * v).sin()).collect();
        let y: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(p, q)| 2.0 * p - 0.5 * q + 1.0)
            .collect();
        let partners: Vec<f64> = b.iter().chain(a.iter()).cloned().collect();
        let fits = pairs(&a, &partners, &y, 0.0).unwrap();
        assert_eq!(fits.status, vec![FITTED, RANK_DEFICIENT]);
        assert!(fits.rss[0] < 1e-22, "{}", fits.rss[0]);
    }

    #[test]
    fn singular_values_of_a_triangle() {
        let (largest, smallest) =
            extreme_singular_values([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]]);
        assert!((largest - 3.0).abs() < 1e-14 && (smallest - 0.5).abs() < 1e-14);
        let (largest, smallest) =
            extreme_singular_values([[1.0, 1.0, 0.0], [0.0, 1e-9, 0.0], [0.0, 0.0, 1.0]]);
        assert!(
            (largest - 2.0f64.sqrt()).abs() < 1e-9
                && (smallest - 1e-9 / 2.0f64.sqrt()).abs() < 1e-18
        );
    }
}
