//! LLL-based integer-relation search (PSLQ-class) with exact verification.
//!
//! `number_theory::find_integer_relation` is a branch-and-bound over a box: exact
//! inside `[-max_coeff, max_coeff]^n`, exponential in `n`, and it refuses `n > 8`.
//! This module is the other half of that trade — lattice reduction of the
//! embedding `[ I | round(x * 10^digits) ]`, so cost is polynomial in `n` for a
//! fixed precision and the coefficient range is not a box.
//!
//! Two things keep it from overclaiming:
//!
//! * Every candidate is **verified by substitution in exact rational arithmetic**
//!   on the exact values of the input `f64`s — never a float dot product. A
//!   relation that exists only as a float coincidence fails that check instead of
//!   being returned. The check is on the numbers it was given: an `f64` is a
//!   rational, so "holds" means "holds for these rationals", and
//!   [`PslqResult::digits_held`] reports how many relative digits it really holds
//!   to.
//! * Precision is escalated in decimal digits and the answer carries the precision
//!   that produced it. With `f64` inputs no relation can be distinguished beyond
//!   ~16 significant digits *of the inputs*; escalation buys coefficient range,
//!   not information.
//!
//! The method name is [`INTEGER_RELATION_METHOD_LLL`]; nothing here returns a
//! relation without a substitution record.

use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Pow, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};

/// What [`find_integer_relation_lll`] runs, exposed so a caller cannot describe
/// the result as a branch-and-bound finding or as a PSLQ certificate.
pub const INTEGER_RELATION_METHOD_LLL: &str = "lll_integer_relation_pslq";

/// Exhaustive search cost is `O((2*max_coeff+1)^n)`; reduction cost is roughly
/// `O(n^4 log^2 B)` for a basis of bit-length `B`. This is that same trade in one
/// place: nothing here is cheap for both.
pub const INTEGER_RELATION_COMPLEXITY: &str =
    "polynomial in n for fixed precision; coefficient range is not bounded by a box";

/// A candidate must be this many times shorter than `10^(digits/n)` — the size a
/// *random* relation has in an `n`-row embedding carrying `digits` digits. The
/// constant column is rounded, so the lattice always contains some vector with a
/// zero last coordinate whose size is `10^(digits/n)`; equivalently, any ten
/// twenty-digit integers have a small integer relation. Without this margin the
/// reduction "finds" a relation in every set of numbers, and the exact residual
/// check cannot catch it because the residual really is tiny — it is tiny because
/// the embedding rounded it there, not because the inputs obey a law.
pub const SPURIOUS_SIZE_GUARD: f64 = 10.0;

/// How far above the embedding's own rounding noise a candidate's residual must
/// sit to count as resolved rather than manufactured. A vector whose last
/// coordinate is exactly zero has a residual no larger than
/// `0.5 * sum|m_i| * 10^-digits`, so at high precision the *rounding* is what made
/// it look related; `4x` that floor is the smallest margin that separates a
/// relation the inputs support from one the embedding invented.
pub const RESOLUTION_GUARD: f64 = 4.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PslqOptions {
    /// `0` means unconstrained, which is the default and the point of the
    /// reduction: a positive value reintroduces the box the branch-and-bound
    /// lives in, because a candidate beyond it is discarded rather than
    /// rescaled. Use it as a sanity bound, not as the search space.
    pub max_coeff: i64,
    /// Relative tolerance: `|m·x| / sum |m_i x_i| <= eps`.
    pub eps: f64,
    /// First decimal precision of the embedding, and the ladder's top.
    pub start_digits: u32,
    pub max_digits: u32,
    /// LLL iteration cap, so a pathological basis is reported as truncated
    /// rather than hanging.
    pub max_iterations: u32,
}

impl Default for PslqOptions {
    fn default() -> Self {
        Self {
            max_coeff: 0,
            eps: 1e-12,
            // 17 digits is what an f64 actually carries. Escalating beyond it buys
            // coefficient range but no information, so the ladder starts at the
            // input's own precision and the residual guards above keep the extra
            // digits from manufacturing answers.
            start_digits: 17,
            max_digits: 100,
            max_iterations: 20_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PslqResult {
    pub coefficients: Vec<i64>,
    pub formula: String,
    /// `m·x` evaluated in `f64` for readability only.
    pub residual: f64,
    /// `|m·x| / sum |m_i x_i|`, computed in exact rational arithmetic.
    pub relative_residual: f64,
    /// Relative digits the relation holds to: `-log10(relative_residual)`.
    pub digits_held: f64,
    /// Decimal precision of the embedding that produced the candidate.
    pub working_digits: u32,
    /// Total LLL iterations across the precision ladder.
    pub iterations: u32,
    /// Rows the reduction presented with an exactly zero constant coordinate.
    pub candidates_examined: usize,
    /// Candidates rejected by the coefficient bound or `eps`.
    pub candidates_rejected: usize,
    /// Largest `|m_i|` in the accepted vector.
    pub max_coefficient: i64,
    /// `10^(working_digits / n)`: the size a random relation has at that
    /// precision. [`SPURIOUS_SIZE_GUARD`] divides it into the evidence region and
    /// the arithmetic region.
    pub generic_coefficient_bound: f64,
    /// `true` only after the exact substitution check passed.
    pub verified: bool,
    pub method: String,
    pub statements: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LllOutcome {
    pub iterations: u32,
    pub completed: bool,
    pub delta: f64,
}

/// Exact rational value of an `f64`, from its IEEE-754 fields.
fn exact_rational(value: f64) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(BigRational::zero());
    }
    let bits = value.to_bits();
    let negative = (bits >> 63) != 0;
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    let (mantissa, exp) = if exponent == 0 {
        (fraction, -1074i64)
    } else {
        (fraction | 0x0010_0000_0000_0000, exponent - 1075)
    };
    let mut numerator = BigInt::from(mantissa);
    if negative {
        numerator = -numerator;
    }
    let denominator = if exp >= 0 {
        numerator <<= exp as u32;
        BigInt::one()
    } else {
        BigInt::one() << (-exp) as u32
    };
    Some(BigRational::new(numerator, denominator))
}

/// Nearest integer, ties away from zero. `mu` is only ever used as a reduction
/// coefficient, so any consistent rounding rule gives a valid LLL run.
fn round_half_away(value: &BigRational) -> BigInt {
    let (quotient, remainder) = value.numer().div_rem(value.denom());
    let twice = remainder.abs() << 1u32;
    if twice >= *value.denom() {
        if value.numer().sign() == Sign::Minus {
            quotient - 1
        } else {
            quotient + 1
        }
    } else {
        quotient
    }
}

fn dot_int_rat(a: &[BigInt], b: &[BigRational]) -> BigRational {
    let mut acc = BigRational::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        if x.is_zero() || y.is_zero() {
            continue;
        }
        acc += BigRational::from_integer(x.clone()) * y;
    }
    acc
}

/// Exact Gram-Schmidt: returns `mu` and `|b*_i|^2`. Any dependent row makes the
/// decomposition undefined, which the LLL loop reports as truncated.
fn gram_schmidt(basis: &[Vec<BigInt>]) -> Option<(Vec<Vec<BigRational>>, Vec<BigRational>)> {
    let m = basis.len();
    let mut mu = vec![vec![BigRational::zero(); m]; m];
    let mut norm = vec![BigRational::zero(); m];
    let mut bstar: Vec<Vec<BigRational>> = Vec::with_capacity(m);
    for i in 0..m {
        let mut v: Vec<BigRational> = basis[i]
            .iter()
            .map(|c| BigRational::from_integer(c.clone()))
            .collect();
        for j in 0..i {
            if norm[j].is_zero() {
                return None;
            }
            let m_ij = dot_int_rat(&basis[i], &bstar[j]) / norm[j].clone();
            if !m_ij.is_zero() {
                for (slot, bs) in v.iter_mut().zip(bstar[j].iter()) {
                    *slot -= &m_ij * bs;
                }
            }
            mu[i][j] = m_ij;
        }
        let mut nrm = BigRational::zero();
        for c in &v {
            nrm += c * c;
        }
        if nrm.is_zero() {
            return None;
        }
        norm[i] = nrm;
        bstar.push(v);
    }
    Some((mu, norm))
}

/// LLL reduction in exact rational arithmetic (Lovász condition with `delta`).
///
/// Size reduction uses incremental `mu` updates, which are exact here; a swap
/// recomputes the decomposition because the Gram-Schmidt basis genuinely changes.
pub fn lll_reduce(basis: &mut [Vec<BigInt>], delta: &BigRational, max_iterations: u32) -> LllOutcome {
    let m = basis.len();
    let delta_f = delta.to_f64().unwrap_or(0.75);
    if m < 2 {
        return LllOutcome {
            iterations: 0,
            completed: true,
            delta: delta_f,
        };
    }
    let Some((mut mu, mut norm)) = gram_schmidt(basis) else {
        return LllOutcome {
            iterations: 0,
            completed: false,
            delta: delta_f,
        };
    };
    let mut iterations: u32 = 0;
    let mut k = 1usize;
    while k < m {
        if iterations >= max_iterations {
            return LllOutcome {
                iterations,
                completed: false,
                delta: delta_f,
            };
        }
        iterations += 1;
        for j in (0..k).rev() {
            let q = round_half_away(&mu[k][j]);
            if q.is_zero() {
                continue;
            }
            let bj = basis[j].clone();
            for (target, source) in basis[k].iter_mut().zip(bj.iter()) {
                *target -= &q * source;
            }
            let q_rat = BigRational::from_integer(q.clone());
            for l in 0..j {
                let sub = q_rat.clone() * mu[j][l].clone();
                mu[k][l] -= sub;
            }
            mu[k][j] -= BigRational::from_integer(q);
        }
        let lhs = norm[k].clone() + mu[k][k - 1].clone() * mu[k][k - 1].clone() * norm[k - 1].clone();
        if lhs >= delta.clone() * norm[k - 1].clone() {
            k += 1;
            continue;
        }
        basis.swap(k, k - 1);
        let Some((fresh_mu, fresh_norm)) = gram_schmidt(basis) else {
            return LllOutcome {
                iterations,
                completed: false,
                delta: delta_f,
            };
        };
        mu = fresh_mu;
        norm = fresh_norm;
        k = if k > 1 { k - 1 } else { 1 };
    }
    LllOutcome {
        iterations,
        completed: true,
        delta: delta_f,
    }
}

// ---------------------------------------------------------------------------
// Fast path: floating-point sweep, certified by periodic exact re-orthogonalisation
// ---------------------------------------------------------------------------

/// Gram-Schmidt in `f64`. It only *steers* the reduction — the basis it produces is
/// certified afterwards in exact arithmetic — so its worst failure mode is a missed
/// relation, never a wrong one.
struct FloatGso {
    mu: Vec<Vec<f64>>,
    norm: Vec<f64>,
}

fn as_f64(value: &BigInt) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn float_gram_schmidt(basis: &[Vec<BigInt>]) -> Option<FloatGso> {
    let m = basis.len();
    let width = basis.first().map(|r| r.len()).unwrap_or(0);
    let mut mu = vec![vec![0.0f64; m]; m];
    let mut norm = vec![0.0f64; m];
    let mut bstar: Vec<Vec<f64>> = Vec::with_capacity(m);
    for i in 0..m {
        let mut v: Vec<f64> = basis[i].iter().map(as_f64).collect();
        for j in 0..i {
            if !(norm[j] > 0.0) || !norm[j].is_finite() {
                return None;
            }
            let mut numerator = 0.0;
            for t in 0..width {
                numerator += as_f64(&basis[i][t]) * bstar[j][t];
            }
            let m_ij = numerator / norm[j];
            if !m_ij.is_finite() {
                return None;
            }
            for (slot, bs) in v.iter_mut().zip(bstar[j].iter()) {
                *slot -= m_ij * bs;
            }
            mu[i][j] = m_ij;
        }
        let mut squared = 0.0;
        for value in &v {
            squared += value * value;
        }
        if !(squared > 0.0) || !squared.is_finite() {
            return None;
        }
        norm[i] = squared;
        bstar.push(v);
    }
    Some(FloatGso { mu, norm })
}

/// One LLL pass in `f64`, keeping the decomposition in place.
///
/// The decomposition is built once per pass and updated by the standard size-
/// reduction identities (`mu[k][l] -= q*mu[j][l]`, `mu[k][j] -= q`; `norm[k]` is
/// untouched because `b*_k` does not move) and rebuilt only after a swap. Rebuilding
/// it per step — as the first version of this function did, twice, from `BigInt` —
/// made it *slower* than the exact path it was meant to replace, because the
/// per-step `BigInt` size reduction dominates and the route through `f64` never got
/// to save anything.
fn float_sweep(basis: &mut [Vec<BigInt>], delta: f64, budget: &mut u32) -> bool {
    let m = basis.len();
    let Some(mut gso) = float_gram_schmidt(basis) else {
        return false;
    };
    let mut k = 1usize;
    while k < m {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let (head, tail) = basis.split_at_mut(k);
        for j in (0..k).rev() {
            let q = gso.mu[k][j].round();
            if q == 0.0 || !q.is_finite() || q.abs() > 1e15 {
                continue;
            }
            let qi = BigInt::from(q as i64);
            for (target, source) in tail[0].iter_mut().zip(head[j].iter()) {
                *target -= &qi * source;
            }
            for l in 0..j {
                gso.mu[k][l] -= q * gso.mu[j][l];
            }
            gso.mu[k][j] -= q;
        }
        let lhs = gso.norm[k] + gso.mu[k][k - 1] * gso.mu[k][k - 1] * gso.norm[k - 1];
        if lhs >= delta * gso.norm[k - 1] {
            k += 1;
        } else {
            basis.swap(k, k - 1);
            let Some(fresh) = float_gram_schmidt(basis) else {
                return false;
            };
            gso = fresh;
            k = if k > 1 { k - 1 } else { 1 };
        }
    }
    true
}

/// First index whose exact LLL conditions fail, if any.
fn exact_violation(basis: &[Vec<BigInt>], delta: &BigRational) -> Option<usize> {
    let (mu, norm) = gram_schmidt(basis)?;
    let half = BigRational::new(BigInt::one(), BigInt::from(2));
    for i in 1..basis.len() {
        for j in 0..i {
            if mu[i][j].abs() > half {
                return Some(i);
            }
        }
    }
    for k in 1..basis.len() {
        let lhs =
            norm[k].clone() + mu[k][k - 1].clone() * mu[k][k - 1].clone() * norm[k - 1].clone();
        if lhs < delta.clone() * norm[k - 1].clone() {
            return Some(k);
        }
    }
    None
}

/// One exact size-reduction (and a swap when Lovász fails) at index `k`.
fn exact_one_step(basis: &mut [Vec<BigInt>], delta: &BigRational, k: usize) {
    if k == 0 || k >= basis.len() {
        return;
    }
    let Some((mu, _)) = gram_schmidt(basis) else {
        return;
    };
    for j in (0..k).rev() {
        let q = round_half_away(&mu[k][j]);
        if q.is_zero() {
            continue;
        }
        let bj = basis[j].clone();
        for (target, source) in basis[k].iter_mut().zip(bj.iter()) {
            *target -= &q * source;
        }
    }
    let Some((mu, norm)) = gram_schmidt(basis) else {
        return;
    };
    let lhs = norm[k].clone() + mu[k][k - 1].clone() * mu[k][k - 1].clone() * norm[k - 1].clone();
    if lhs < delta.clone() * norm[k - 1].clone() {
        basis.swap(k, k - 1);
    }
}

/// Float-driven reduction, certified in exact arithmetic before it returns.
///
/// The sweep does the bulk of the work at one `f64` Gram-Schmidt per step; the exact
/// pass then asks whether the resulting basis satisfies the LLL conditions in
/// rationals, and takes one exact step where it does not. `completed == false` means
/// the budget ran out: the caller then redoes the basis with [`lll_reduce`] instead
/// of trusting a basis that was never certified.
pub fn lll_reduce_fast(
    basis: &mut [Vec<BigInt>],
    delta: &BigRational,
    max_iterations: u32,
) -> LllOutcome {
    let delta_f = delta.to_f64().unwrap_or(0.75);
    if basis.len() < 2 {
        return LllOutcome {
            iterations: 0,
            completed: true,
            delta: delta_f,
        };
    }
    let mut budget = max_iterations;
    let mut exact_steps = 0u32;
    loop {
        if !float_sweep(basis, delta_f, &mut budget) {
            return LllOutcome {
                iterations: max_iterations - budget,
                completed: false,
                delta: delta_f,
            };
        }
        match exact_violation(basis, delta) {
            None => {
                return LllOutcome {
                    iterations: max_iterations - budget,
                    completed: true,
                    delta: delta_f,
                }
            }
            Some(k) => {
                if budget == 0 || exact_steps >= 64 {
                    return LllOutcome {
                        iterations: max_iterations - budget,
                        completed: false,
                        delta: delta_f,
                    };
                }
                budget -= 1;
                exact_steps += 1;
                exact_one_step(basis, delta, k);
            }
        }
    }
}

fn canonicalise(mut coeffs: Vec<i64>) -> Option<Vec<i64>> {
    let first_nonzero = coeffs.iter().find(|&&c| c != 0).copied()?;
    if first_nonzero < 0 {
        for c in coeffs.iter_mut() {
            *c = -(*c);
        }
    }
    let mut g: i64 = 0;
    for &c in &coeffs {
        g = gcd_i64(g, c.abs());
    }
    if g > 1 {
        for c in coeffs.iter_mut() {
            *c /= g;
        }
    }
    Some(coeffs)
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd_i64(b, a % b)
    }
}

fn format_relation(coeffs: &[i64], labels: Option<&[String]>) -> String {
    let mut terms: Vec<String> = Vec::new();
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let lbl = labels
            .and_then(|l| l.get(i).cloned())
            .unwrap_or_else(|| format!("x{}", i));
        if terms.is_empty() {
            terms.push(match c {
                1 => lbl,
                -1 => format!("-{}", lbl),
                _ => format!("{}*{}", c, lbl),
            });
        } else if c > 0 {
            terms.push(if c == 1 {
                format!(" + {}", lbl)
            } else {
                format!(" + {}*{}", c, lbl)
            });
        } else {
            terms.push(if c == -1 {
                format!(" - {}", lbl)
            } else {
                format!(" - {}*{}", c.abs(), lbl)
            });
        }
    }
    format!("{} = 0", terms.concat())
}

/// Exact substitution check on the inputs' exact rational values.
struct Substitution {
    /// `|m·x| / sum|m_i x_i|`, computed in rationals.
    relative: f64,
    /// `m·x` as an `f64`, for display only.
    absolute: f64,
    /// `sum|m_i x_i|` as an `f64`, the normaliser the embedding noise is measured against.
    scale: f64,
    /// The residual itself, so a caller can ask whether it is exactly zero.
    exact: BigRational,
}

fn exact_substitution(coeffs: &[i64], exact_x: &[BigRational]) -> Substitution {
    let mut residual = BigRational::zero();
    let mut scale = BigRational::zero();
    for (&m, x) in coeffs.iter().zip(exact_x.iter()) {
        if m == 0 {
            continue;
        }
        let term = BigRational::from_integer(BigInt::from(m)) * x;
        scale += term.abs();
        residual += term;
    }
    // A zero residual holds at every tolerance; a nonzero one over a zero scale
    // cannot be normalised and is reported as no evidence rather than as perfect.
    let relative = if residual.is_zero() {
        0.0
    } else if scale.is_zero() {
        f64::INFINITY
    } else {
        (residual.clone().abs() / scale.clone())
            .to_f64()
            .unwrap_or(f64::INFINITY)
    };
    Substitution {
        relative,
        absolute: residual.to_f64().unwrap_or(f64::NAN),
        scale: scale.to_f64().unwrap_or(f64::INFINITY),
        exact: residual,
    }
}

/// Searches for `m in Z^n`, `m != 0`, with `|m·x| / sum|m_i x_i| <= eps`.
///
/// Ladder: embed at `start_digits`, reduce, verify every row whose constant
/// coordinate vanished, and escalate by ten digits when nothing survives. A
/// candidate is returned only if the exact substitution check passes at the
/// requested tolerance; `None` means no relation was found at any precision the
/// ladder reached, which is not the same as "no relation exists".
pub fn find_integer_relation_lll(
    x: &[f64],
    labels: Option<&[String]>,
    opts: &PslqOptions,
) -> Option<PslqResult> {
    let n = x.len();
    if n < 2 || x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let exact_x: Vec<BigRational> = x
        .iter()
        .map(|&v| exact_rational(v))
        .collect::<Option<Vec<_>>>()?;
    let norm_inf = x.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    if norm_inf <= 0.0 {
        return None;
    }

    let base_statements = vec![
        format!("method: {}", INTEGER_RELATION_METHOD_LLL),
        "reduction: LLL on [ I | round(x * 10^digits) ], f64 Gram-Schmidt with periodic \
         exact re-orthogonalisation and an exact certification pass"
            .to_string(),
        "verification: m·x evaluated in exact rational arithmetic on the f64 inputs".to_string(),
    ];

    // An input that is exactly zero needs no search: the unit vector is a relation
    // whose exact residual is zero, which is the strongest form available.
    if let Some(idx) = exact_x.iter().position(|v| v.is_zero()) {
        let mut coeffs = vec![0i64; n];
        coeffs[idx] = 1;
        let sub = exact_substitution(&coeffs, &exact_x);
        return Some(PslqResult {
            formula: format_relation(&coeffs, labels),
            coefficients: coeffs,
            residual: sub.absolute,
            relative_residual: sub.relative,
            digits_held: f64::INFINITY,
            working_digits: 0,
            iterations: 0,
            candidates_examined: 1,
            candidates_rejected: 0,
            max_coefficient: 1,
            generic_coefficient_bound: 0.0,
            verified: true,
            method: INTEGER_RELATION_METHOD_LLL.to_string(),
            statements: base_statements,
        });
    }

    let delta = BigRational::new(BigInt::from(3), BigInt::from(4));
    let mut ladder: Vec<u32> = Vec::new();
    let mut digits = opts.start_digits.max(8);
    let floor = digits;
    let ceiling = opts.max_digits.max(floor);
    loop {
        ladder.push(digits);
        if digits >= ceiling {
            break;
        }
        digits = (digits + 10).min(ceiling);
    }

    let mut total_iterations: u32 = 0;
    let mut examined: usize = 0;
    let mut rejected: usize = 0;
    let mut truncated = false;

    for working_digits in ladder {
        let scale = BigInt::from(10u32).pow(working_digits);
        let mut basis: Vec<Vec<BigInt>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![BigInt::zero(); n + 1];
            row[i] = BigInt::one();
            let scaled = &exact_x[i] * BigRational::from_integer(scale.clone());
            row[n] = round_half_away(&scaled);
            basis.push(row);
        }
        let untouched = basis.clone();
        let mut outcome = lll_reduce_fast(&mut basis, &delta, opts.max_iterations);
        if !outcome.completed {
            basis = untouched;
            outcome = lll_reduce(&mut basis, &delta, opts.max_iterations);
        }
        total_iterations += outcome.iterations;
        if !outcome.completed {
            truncated = true;
        }

        let mut best: Option<PslqResult> = None;
        for row in &basis {
            // Every reduced row is a candidate: rounding the constant column means a
            // genuine relation usually leaves a small nonzero last coordinate, so a
            // `== 0` filter would discard exactly the vectors worth checking. The
            // exact substitution below is what decides, and it cannot be fooled by
            // that rounding.
            examined += 1;
            let raw: Vec<i64> = match row[..n].iter().map(|v| v.to_i64()).collect::<Option<Vec<_>>>() {
                Some(v) => v,
                None => {
                    rejected += 1;
                    continue;
                }
            };
            let Some(coeffs) = canonicalise(raw) else {
                rejected += 1;
                continue;
            };
            let max_abs = coeffs.iter().map(|c| c.abs()).max().unwrap_or(1);
            if opts.max_coeff > 0 && max_abs > opts.max_coeff {
                rejected += 1;
                continue;
            }
            let sub = exact_substitution(&coeffs, &exact_x);
            let exact_zero = sub.exact.is_zero();
            let generic = 10f64.powf(working_digits as f64 / n as f64);
            if !exact_zero {
                // An approximate relation has to clear three bars: the requested
                // tolerance, the size a random relation has at this precision, and
                // the embedding's own rounding floor.
                let l1_abs: f64 = coeffs.iter().map(|c| (*c as f64).abs()).sum();
                let noise = if sub.scale > 0.0 {
                    0.5 * l1_abs * 10f64.powi(-(working_digits as i32)) / sub.scale
                } else {
                    f64::INFINITY
                };
                let size_ok = (max_abs as f64) * SPURIOUS_SIZE_GUARD <= generic;
                let resolved = sub.relative > RESOLUTION_GUARD * noise;
                if !(size_ok && resolved && sub.relative <= opts.eps) {
                    rejected += 1;
                    continue;
                }
            }
            let relative = sub.relative;
            let residual = sub.absolute;
            let digits_held = if relative == 0.0 {
                f64::INFINITY
            } else {
                -relative.log10()
            };
            let l1: i64 = coeffs.iter().map(|c| c.abs()).sum();
            let better = match &best {
                None => true,
                Some(prev) => {
                    let prev_l1: i64 = prev.coefficients.iter().map(|c| c.abs()).sum();
                    l1 < prev_l1 || (l1 == prev_l1 && digits_held > prev.digits_held)
                }
            };
            if better {
                let mut statements = base_statements.clone();
                statements.push(format!(
                    "embedding precision: 10^-{} ({})",
                    working_digits,
                    if outcome.completed {
                        "reduction completed"
                    } else {
                        "reduction truncated at the iteration cap"
                    }
                ));
                statements.push(format!(
                    "exact residual holds to {:.1} relative digits",
                    digits_held
                ));
                statements.push(format!(
                    "size margin: |m|_inf = {} against a generic bound of {:.3e}",
                    max_abs, generic
                ));
                best = Some(PslqResult {
                    formula: format_relation(&coeffs, labels),
                    coefficients: coeffs,
                    residual,
                    relative_residual: relative,
                    digits_held,
                    working_digits,
                    iterations: total_iterations,
                    candidates_examined: examined,
                    candidates_rejected: rejected,
                    max_coefficient: max_abs,
                    generic_coefficient_bound: generic,
                    verified: true,
                    method: INTEGER_RELATION_METHOD_LLL.to_string(),
                    statements,
                });
            }
        }
        if best.is_some() {
            return best;
        }
        if truncated {
            break;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pslq(x: &[f64]) -> Option<PslqResult> {
        find_integer_relation_lll(x, None, &PslqOptions::default())
    }

    #[test]
    fn recovers_a_planted_three_term_relation() {
        let root2 = 2.0f64.sqrt();
        let root3 = 3.0f64.sqrt();
        let res = pslq(&[root2, root3, root2 + root3]).expect("planted relation");
        assert_eq!(res.coefficients, vec![1, 1, -1]);
        assert!(res.verified);
        assert!(res.relative_residual <= 1e-12, "{}", res.relative_residual);
        assert!(res.digits_held >= 12.0, "{}", res.digits_held);
        assert!(res.formula.contains("= 0"));
    }

    #[test]
    fn reaches_past_the_branch_and_bound_eight_input_cap() {
        // n = 10: the first nine primes and their sum. The relation is exact in
        // these inputs (integers, so the residual is zero), which is the only kind
        // of ten-term relation an f64 carries: an approximate one needs
        // `n * log10(max|m|) + digits_of_residual` digits, and f64 has 17.
        const PRIMES: [u32; 9] = [2, 3, 5, 7, 11, 13, 17, 19, 23];
        let mut x: Vec<f64> = PRIMES.iter().map(|&p| p as f64).collect();
        x.push(PRIMES.iter().map(|&p| p as f64).sum::<f64>());
        assert_eq!(x.len(), 10);
        let res = pslq(&x).expect("ten-input relation");
        assert!(res.verified);
        assert_eq!(res.relative_residual, 0.0, "the input relation is exact");
        // Any exact relation among these values is a true statement; the point of
        // the test is that n = 10 was searched at all, not which short vector.
        let annihilates: f64 = res
            .coefficients
            .iter()
            .zip(x.iter())
            .map(|(m, v)| *m as f64 * v)
            .sum();
        assert_eq!(annihilates, 0.0);
    }

    #[test]
    fn an_exactly_zero_input_yields_a_unit_relation() {
        let res = pslq(&[1.5, 0.0, 2.5]).expect("unit relation");
        assert_eq!(res.coefficients, vec![0, 1, 0]);
        assert_eq!(res.relative_residual, 0.0);
        assert!(res.verified);
    }

    #[test]
    fn a_tolerance_below_the_exact_residual_is_refused() {
        // Three independent irrationals carry no small-coefficient relation, and
        // exact arithmetic cannot be talked into one by rounding.
        let x = [2.0f64.sqrt(), 3.0f64.sqrt(), 5.0f64.sqrt()];
        let opts = PslqOptions {
            eps: 1e-300,
            max_coeff: 1_000,
            max_digits: 40,
            ..PslqOptions::default()
        };
        assert!(
            find_integer_relation_lll(&x, None, &opts).is_none(),
            "a candidate whose exact residual is nonzero must not be returned"
        );
    }

    #[test]
    fn the_float_sweep_reaches_the_same_relations_as_the_exact_path() {
        let root2 = 2.0f64.sqrt();
        let root3 = 3.0f64.sqrt();
        let cases: Vec<(Vec<f64>, Vec<i64>)> = vec![
            (vec![root2, root3, root2 + root3], vec![1, 1, -1]),
            (vec![60.0, 90.0], vec![3, -2]),
            (vec![2.016, 31.998, 18.015], vec![2, 1, -2]),
        ];
        for (x, expected) in cases {
            let res = find_integer_relation_lll(&x, None, &PslqOptions::default())
                .expect("float path lost a relation the exact path finds");
            assert_eq!(res.coefficients, expected, "inputs {:?}", x);
        }
    }

    #[test]
    fn the_float_path_leaves_a_basis_the_exact_conditions_accept() {
        let mut basis = vec![
            vec![BigInt::from(10), BigInt::from(11), BigInt::from(12)],
            vec![BigInt::from(11), BigInt::from(13), BigInt::from(12)],
            vec![BigInt::from(12), BigInt::from(11), BigInt::from(15)],
        ];
        let delta = BigRational::new(BigInt::from(3), BigInt::from(4));
        let outcome = lll_reduce_fast(&mut basis, &delta, 1_000);
        assert!(outcome.completed);
        assert!(
            exact_violation(&basis, &delta).is_none(),
            "the float path returned an uncertified basis: {:?}",
            basis
        );
    }

    #[test]
    fn lll_shortens_a_two_dimensional_basis() {
        let mut basis = vec![
            vec![BigInt::from(10), BigInt::from(11)],
            vec![BigInt::from(11), BigInt::from(12)],
        ];
        let outcome = lll_reduce(
            &mut basis,
            &BigRational::new(BigInt::from(3), BigInt::from(4)),
            1_000,
        );
        assert!(outcome.completed);
        let shortest = basis
            .iter()
            .map(|row| row.iter().map(|c| c.clone() * c.clone()).fold(BigInt::zero(), |a, b| a + b))
            .min()
            .unwrap();
        assert!(
            shortest <= BigInt::from(2),
            "LLL must find the (1, 1) and (1, 2) short vectors, got {:?}",
            basis
        );
    }

    /// `[ I | round(x_i * 10^digits) ]` for twelve wobbly constants, with no short
    /// relation, so the reduction has to do the work instead of stopping early.
    fn hard_basis(n: usize, digits: u32) -> Vec<Vec<BigInt>> {
        let scale = 10f64.powi(digits as i32);
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let phi = 1.618_033_988_749_895;
            let x = ((i as f64) * phi + 0.314_159_265_358_979).sin().abs() + 0.123_456_7;
            let mut row = vec![BigInt::zero(); n + 1];
            row[i] = BigInt::one();
            let scaled = format!("{:.0}", x * scale);
            row[n] = BigInt::parse_bytes(scaled.as_bytes(), 10).expect("scaled constant");
            rows.push(row);
        }
        rows
    }

    #[test]
    fn the_float_sweep_shortens_a_hard_basis_without_losing_certification() {
        // Same inputs, same delta, same iteration cap: the exact path rebuilds a
        // `BigRational` Gram-Schmidt after every size reduction, the float path
        // builds one `f64` Gram-Schmidt per sweep and certifies in exact arithmetic
        // afterwards. Timings are printed, not asserted — wall clock is not a
        // property of the code — but both bases are asserted to satisfy the same
        // exact LLL conditions afterwards, which is what the speedup may not cost.
        let delta = BigRational::new(BigInt::from(3), BigInt::from(4));
        let cap = 20_000;
        let n = 8;
        let digits = 20;

        let mut exact_basis = hard_basis(n, digits);
        let started = std::time::Instant::now();
        let exact = lll_reduce(&mut exact_basis, &delta, cap);
        let exact_ms = started.elapsed().as_secs_f64() * 1e3;

        let mut fast_basis = hard_basis(n, digits);
        let started = std::time::Instant::now();
        let fast = lll_reduce_fast(&mut fast_basis, &delta, cap);
        let fast_ms = started.elapsed().as_secs_f64() * 1e3;

        println!(
            "{} rows x {} digits: exact Gram-Schmidt {:.1} ms ({} iterations), \
             float sweep + exact certification {:.1} ms ({} iterations), {:.1}x",
            n,
            digits,
            exact_ms,
            exact.iterations,
            fast_ms,
            fast.iterations,
            exact_ms / fast_ms
        );

        assert!(exact.completed, "the exact path must finish on this input");
        assert!(fast.completed, "the float path must finish within the same cap");
        assert!(
            exact_violation(&fast_basis, &delta).is_none(),
            "the float path returned an uncertified basis"
        );
        assert!(
            exact_violation(&exact_basis, &delta).is_none(),
            "the exact path returned an uncertified basis"
        );
    }
}
