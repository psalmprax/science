//! Discrete & Algebraic Number Theory Engine for Scientific Discovery
//!
//! Provides:
//! 1. Continued Fraction Expansion & Farey rational approximation for exact fraction recovery.
//! 2. Integer Relation Discovery (PSLQ/Lattice null-vector search) for physical conservation laws.
//! 3. Modular Lattice & Periodic Structure Discovery for quantum and discrete spatial lattices.

use serde::{Deserialize, Serialize};

/// Canonical irreducible rational number p/q with q > 0 and gcd(p, q) == 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    pub fn new(mut num: i64, mut den: i64) -> Result<Self, &'static str> {
        if den == 0 {
            return Err("Denominator cannot be zero");
        }
        if den < 0 {
            num = -num;
            den = -den;
        }
        let g = gcd(num.abs(), den);
        Ok(Self {
            num: num / g,
            den: den / g,
        })
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

pub fn vector_gcd(vec: &[i64]) -> i64 {
    let mut g = 0;
    for &x in vec {
        if x != 0 {
            g = if g == 0 { x.abs() } else { gcd(g, x.abs()) };
        }
    }
    if g == 0 { 1 } else { g }
}

/// Continued fraction expansion and convergent generator.
///
/// Recovers the simplest irreducible fraction p/q with q <= max_denom
/// that approximates `val` within `eps`.
pub fn farey_rational(val: f64, max_denom: i64, eps: f64) -> Option<Rational> {
    if !val.is_finite() {
        return None;
    }
    let sign = if val < 0.0 { -1 } else { 1 };
    let x0 = val.abs();

    // Check integer first
    let round_int = x0.round() as i64;
    if (x0 - round_int as f64).abs() <= eps {
        return Rational::new(sign * round_int, 1).ok();
    }

    let mut p0: i64 = 0;
    let mut q0: i64 = 1;
    let mut p1: i64 = 1;
    let mut q1: i64 = 0;

    let mut x = x0;
    let max_steps = 30;

    let mut best_rat: Option<Rational> = None;
    let mut best_err = (x0 - round_int as f64).abs();
    if best_err <= eps {
        best_rat = Rational::new(sign * round_int, 1).ok();
    }

    for _ in 0..max_steps {
        let a = x.floor() as i64;
        let p2 = a * p1 + p0;
        let q2 = a * q1 + q0;

        if q2 > max_denom {
            // Check semiconvergents: (c * p1 + p0) / (c * q1 + q0)
            let c_max = (max_denom - q0) / q1;
            for c in (1..=c_max).rev() {
                let pc = c * p1 + p0;
                let qc = c * q1 + q0;
                let err = (x0 - (pc as f64 / qc as f64)).abs();
                if err <= eps && err < best_err {
                    best_err = err;
                    best_rat = Rational::new(sign * pc, qc).ok();
                }
            }
            break;
        }

        let rat = Rational::new(sign * p2, q2).ok()?;
        let err = (x0 - (p2 as f64 / q2 as f64)).abs();
        if err <= eps && (best_rat.is_none() || err < best_err) {
            best_err = err;
            best_rat = Some(rat);
        }

        if err < 1e-12 {
            break;
        }

        let frac = x - a as f64;
        if frac.abs() < 1e-12 {
            break;
        }
        x = 1.0 / frac;

        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
    }

    best_rat
}

/// Discovers an integer relation: finds m in Z^n (non-zero, |m_i| <= max_coeff)
/// such that |sum m_i x_i| < eps.
///
/// Recovers conservation laws, stoichiometry, and harmonic ratio identities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegerRelationResult {
    pub coefficients: Vec<i64>,
    pub residual: f64,
    pub formula: String,
    pub is_exact: bool,
}

pub fn find_integer_relation(
    x: &[f64],
    max_coeff: i64,
    eps: f64,
    labels: Option<&[String]>,
) -> Option<IntegerRelationResult> {
    let n = x.len();
    if n < 2 || n > 8 {
        return None;
    }
    for &v in x {
        if !v.is_finite() {
            return None;
        }
    }

    let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm_x < 1e-12 {
        return None;
    }

    // Branch and bound lattice search for minimal integer vector m with |m.x| < eps
    let mut best_m: Option<Vec<i64>> = None;
    let mut best_residual = f64::INFINITY;
    let mut best_l1_norm = i64::MAX;

    let mut current_m = vec![0i64; n];

    fn search(
        idx: usize,
        n: usize,
        x: &[f64],
        max_coeff: i64,
        eps: f64,
        current_m: &mut [i64],
        best_m: &mut Option<Vec<i64>>,
        best_residual: &mut f64,
        best_l1_norm: &mut i64,
    ) {
        if idx == n {
            let is_all_zero = current_m.iter().all(|&c| c == 0);
            if is_all_zero {
                return;
            }
            // Enforce canonical sign: first non-zero coefficient is positive
            let first_nonzero = current_m.iter().find(|&&c| c != 0).copied().unwrap_or(0);
            if first_nonzero < 0 {
                return;
            }

            let g = vector_gcd(current_m);
            if g > 1 {
                return; // Only consider irreducible integer relations
            }

            let mut sum = 0.0;
            let mut l1: i64 = 0;
            for i in 0..n {
                sum += (current_m[i] as f64) * x[i];
                l1 += current_m[i].abs();
            }
            let res = sum.abs();

            if res <= eps {
                if l1 < *best_l1_norm || (l1 == *best_l1_norm && res < *best_residual) {
                    *best_residual = res;
                    *best_l1_norm = l1;
                    *best_m = Some(current_m.to_vec());
                }
            }
            return;
        }

        // Prioritize smaller coefficients: 0, 1, -1, 2, -2, ...
        let mut ordered_coeffs = Vec::with_capacity((2 * max_coeff + 1) as usize);
        ordered_coeffs.push(0i64);
        for c in 1..=max_coeff {
            ordered_coeffs.push(c);
            ordered_coeffs.push(-c);
        }

        for c in ordered_coeffs {
            current_m[idx] = c;
            search(idx + 1, n, x, max_coeff, eps, current_m, best_m, best_residual, best_l1_norm);
        }
    }

    search(
        0,
        n,
        x,
        max_coeff,
        eps,
        &mut current_m,
        &mut best_m,
        &mut best_residual,
        &mut best_l1_norm,
    );

    let m = best_m?;
    let default_labels: Vec<String> = (0..n).map(|i| format!("x{}", i)).collect();
    let var_labels = labels.unwrap_or(&default_labels);

    let mut terms = Vec::new();
    for i in 0..n {
        let c = m[i];
        if c == 0 {
            continue;
        }
        let lbl = if i < var_labels.len() { &var_labels[i] } else { "x" };
        if terms.is_empty() {
            if c == 1 {
                terms.push(lbl.to_string());
            } else if c == -1 {
                terms.push(format!("-{}", lbl));
            } else {
                terms.push(format!("{}*{}", c, lbl));
            }
        } else if c > 0 {
            if c == 1 {
                terms.push(format!(" + {}", lbl));
            } else {
                terms.push(format!(" + {}*{}", c, lbl));
            }
        } else {
            if c == -1 {
                terms.push(format!(" - {}", lbl));
            } else {
                terms.push(format!(" - {}*{}", c.abs(), lbl));
            }
        }
    }
    let formula = format!("{} = 0", terms.concat());

    Some(IntegerRelationResult {
        coefficients: m,
        residual: best_residual,
        formula,
        is_exact: best_residual < 1e-10,
    })
}

/// Modular Lattice & Periodic Structure Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModularLatticeResult {
    pub spacing: f64,
    pub offset: f64,
    pub residual_rmse: f64,
    pub integer_indices: Vec<i64>,
    pub formula: String,
}

/// Detects if points x_i fall on a 1D discrete periodic lattice:
/// x_i = offset + k_i * spacing + noise, with k_i in Z.
pub fn detect_modular_lattice(
    data: &[f64],
    min_spacing: f64,
    max_spacing: f64,
    max_rmse: f64,
) -> Option<ModularLatticeResult> {
    if data.len() < 4 {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute consecutive pairwise differences
    let mut diffs: Vec<f64> = sorted
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .filter(|&d| d >= min_spacing && d.is_finite())
        .collect();

    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Candidate spacings from smallest non-zero differences
    let mut candidate_spacings = Vec::new();
    let min_diff = diffs[0];
    if min_diff <= max_spacing {
        candidate_spacings.push(min_diff);
    }
    // Also try submultiples and medians
    let median_diff = diffs[diffs.len() / 2];
    if median_diff <= max_spacing {
        candidate_spacings.push(median_diff);
    }

    let mut best_result: Option<ModularLatticeResult> = None;
    let mut best_rmse = f64::INFINITY;

    for &spacing in &candidate_spacings {
        if spacing < min_spacing || spacing > max_spacing {
            continue;
        }
        // Test fine-grid around spacing
        for factor in &[0.5, 1.0, 2.0] {
            let test_spacing = spacing * factor;
            if test_spacing < min_spacing || test_spacing > max_spacing {
                continue;
            }

            // Estimate optimal offset: median of (x_i mod test_spacing)
            let mut mods: Vec<f64> = sorted
                .iter()
                .map(|&v| {
                    let m = v % test_spacing;
                    if m < 0.0 { m + test_spacing } else { m }
                })
                .collect();
            mods.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let offset = mods[mods.len() / 2];

            // Evaluate RMSE of lattice alignment
            let mut sq_err = 0.0;
            let mut indices = Vec::with_capacity(sorted.len());
            for &v in &sorted {
                let k = ((v - offset) / test_spacing).round() as i64;
                let pred = offset + (k as f64) * test_spacing;
                sq_err += (v - pred).powi(2);
                indices.push(k);
            }
            let rmse = (sq_err / sorted.len() as f64).sqrt();

            if rmse <= max_rmse {
                let is_better = match &best_result {
                    None => true,
                    Some(prev) => {
                        if (rmse - best_rmse).abs() < 1e-5 {
                            test_spacing > prev.spacing // Prefer larger (fundamental) period on equal fit
                        } else {
                            rmse < best_rmse
                        }
                    }
                };
                if is_better {
                    best_rmse = rmse;
                    best_result = Some(ModularLatticeResult {
                        spacing: test_spacing,
                        offset,
                        residual_rmse: rmse,
                        integer_indices: indices,
                        formula: format!("x = {:.4} + k * {:.4} (k in Z)", offset, test_spacing),
                    });
                }
            }
        }
    }

    best_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_farey_recovers_exact_rational_fractions() {
        // Quantum Hall fractions & physical coefficients
        let cases = [
            (0.28, 7, 25),             // 7/25
            (0.3333333333, 1, 3),      // 1/3
            (0.6666666667, 2, 3),      // 2/3
            (0.125, 1, 8),             // 1/8
            (0.40, 2, 5),              // 2/5 (Laughlin)
            (0.714285714, 5, 7),       // 5/7
            (2.45, 49, 20),            // 49/20
            (-0.75, -3, 4),            // -3/4
        ];

        for (val, exp_num, exp_den) in cases {
            let rat = farey_rational(val, 100, 1e-4).expect("Should recover rational");
            assert_eq!(rat.num, exp_num, "Value {} expected num {}", val, exp_num);
            assert_eq!(rat.den, exp_den, "Value {} expected den {}", val, exp_den);
        }
    }

    #[test]
    fn test_integer_relation_chemical_stoichiometry() {
        // Water formation: 2 H2 + 1 O2 -> 2 H2O
        // Relation: 2*x0 + 1*x1 - 2*x2 = 0
        let mole_weights = [2.016, 31.998, 18.015]; // weights of H2, O2, H2O
        // 2*2.016 + 1*31.998 - 2*18.015 = 4.032 + 31.998 - 36.030 = 0
        let labels = vec!["H2".to_string(), "O2".to_string(), "H2O".to_string()];
        let res = find_integer_relation(&mole_weights, 3, 1e-2, Some(&labels))
            .expect("Should discover stoichiometric balance");
        assert_eq!(res.coefficients, vec![2, 1, -2]);
        assert!(res.formula.contains("2*H2 + O2 - 2*H2O = 0") || res.formula.contains("2*H2 + 1*O2 - 2*H2O = 0"));
    }

    #[test]
    fn test_integer_relation_harmonic_resonance() {
        // Resonant frequency coupling: 3*w1 - 2*w2 = 0
        let freqs = [60.0, 90.0];
        let labels = vec!["w1".to_string(), "w2".to_string()];
        let res = find_integer_relation(&freqs, 5, 1e-5, Some(&labels))
            .expect("Should discover frequency resonance");
        assert_eq!(res.coefficients, vec![3, -2]);
    }

    #[test]
    fn test_modular_lattice_discovery() {
        // Discrete crystal lattice or quantum staircase steps
        // Spacing = 1.5, offset = 0.5
        let data = vec![0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5];
        let res = detect_modular_lattice(&data, 0.5, 3.0, 1e-4)
            .expect("Should discover modular lattice");
        assert!((res.spacing - 1.5).abs() < 1e-4);
        assert!((res.offset - 0.5).abs() < 1e-4);
        assert_eq!(res.integer_indices, vec![0, 1, 2, 3, 4, 5, 6]);
    }
}
