//! Lie point-symmetry discovery for first-order polynomial ODEs y' = f(x, y).
//!
//! Given `f` as explicit polynomial terms, seeks infinitesimal generators
//! `(ξ(x, y), η(x, y))` with polynomial components of bounded degree that
//! satisfy the linearised symmetry condition:
//! ```text
//! η_x + (η_y − ξ_x) f − ξ_y f² − ξ f_x − η f_y = 0.
//! ```
//! Exact bivariate polynomial algebra turns the condition into a homogeneous
//! linear system for the generator coefficients; the numerical nullspace
//! (Gram matrix + Jacobi eigendecomposition) then yields a symmetry basis.
//!
//! Scope (v1, explicit): first-order ODEs with polynomial `f`. This module
//! reports the nullspace basis and residuals; physical interpretation of any
//! symmetry stays upstream.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub type Mono = (usize, usize);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LieConfig {
    pub ansatz_degree: usize,
    pub null_tol: f64,
    pub sparsity_threshold: f64,
}

impl Default for LieConfig {
    fn default() -> LieConfig {
        LieConfig {
            ansatz_degree: 2,
            null_tol: 1e-8,
            sparsity_threshold: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LieTerm {
    pub px: usize,
    pub py: usize,
    pub coeff: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LieGenerator {
    pub xi: Vec<LieTerm>,
    pub eta: Vec<LieTerm>,
    pub residual: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LieResult {
    pub ansatz_degree: usize,
    pub n_unknowns: usize,
    pub n_equations: usize,
    pub nullity: usize,
    pub generators: Vec<LieGenerator>,
    pub max_basis_residual: f64,
    pub min_eigenvalue: f64,
    pub max_eigenvalue: f64,
}

/// Exact bivariate polynomial with f64 coefficients.
#[derive(Debug, Clone, Default)]
struct Poly(BTreeMap<Mono, f64>);

fn clean(map: &mut BTreeMap<Mono, f64>) {
    map.retain(|_, c| c.abs() > 1e-300);
}

impl Poly {
    fn term(px: usize, py: usize, c: f64) -> Poly {
        let mut m = BTreeMap::new();
        if c.abs() > 1e-300 {
            m.insert((px, py), c);
        }
        Poly(m)
    }

    fn add(&self, o: &Poly) -> Poly {
        let mut m = self.0.clone();
        for (k, c) in &o.0 {
            *m.entry(*k).or_insert(0.0) += c;
        }
        clean(&mut m);
        Poly(m)
    }

    fn mul(&self, o: &Poly) -> Poly {
        let mut m = BTreeMap::new();
        for ((i1, j1), c1) in &self.0 {
            for ((i2, j2), c2) in &o.0 {
                *m.entry((i1 + i2, j1 + j2)).or_insert(0.0) += c1 * c2;
            }
        }
        clean(&mut m);
        Poly(m)
    }

    fn dx(&self) -> Poly {
        let mut m = BTreeMap::new();
        for ((i, j), c) in &self.0 {
            if *i > 0 {
                m.insert((i - 1, *j), (*i as f64) * c);
            }
        }
        Poly(m)
    }

    fn dy(&self) -> Poly {
        let mut m = BTreeMap::new();
        for ((i, j), c) in &self.0 {
            if *j > 0 {
                m.insert((*i, j - 1), (*j as f64) * c);
            }
        }
        Poly(m)
    }
}

/// Polynomial whose coefficients are linear forms in the K unknowns.
#[derive(Debug, Clone, Default)]
struct LinPoly(BTreeMap<Mono, Vec<f64>>);

impl LinPoly {
    fn add(&self, o: &LinPoly) -> LinPoly {
        let mut m = self.0.clone();
        for (k, v) in &o.0 {
            let e = m.entry(*k).or_insert_with(|| vec![0.0; v.len()]);
            for (a, b) in e.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        LinPoly(m)
    }

    fn sub(&self, o: &LinPoly) -> LinPoly {
        let mut m = self.0.clone();
        for (k, v) in &o.0 {
            let e = m.entry(*k).or_insert_with(|| vec![0.0; v.len()]);
            for (a, b) in e.iter_mut().zip(v.iter()) {
                *a -= b;
            }
        }
        LinPoly(m)
    }

    /// Multiply by a plain polynomial (distributes over monomials).
    fn mul_plain(&self, p: &Poly) -> LinPoly {
        let mut m: BTreeMap<Mono, Vec<f64>> = BTreeMap::new();
        for (k1, v) in &self.0 {
            for (k2, c) in &p.0 {
                let key = (k1.0 + k2.0, k1.1 + k2.1);
                let e = m.entry(key).or_insert_with(|| vec![0.0; v.len()]);
                for (a, b) in e.iter_mut().zip(v.iter()) {
                    *a += b * c;
                }
            }
        }
        LinPoly(m)
    }

    fn dx(&self) -> LinPoly {
        let mut m: BTreeMap<Mono, Vec<f64>> = BTreeMap::new();
        for ((i, j), v) in &self.0 {
            if *i > 0 {
                let s = *i as f64;
                let e = m.entry((i - 1, *j)).or_insert_with(|| vec![0.0; v.len()]);
                for (a, b) in e.iter_mut().zip(v.iter()) {
                    *a += s * b;
                }
            }
        }
        LinPoly(m)
    }

    fn dy(&self) -> LinPoly {
        let mut m: BTreeMap<Mono, Vec<f64>> = BTreeMap::new();
        for ((i, j), v) in &self.0 {
            if *j > 0 {
                let s = *j as f64;
                let e = m.entry((*i, j - 1)).or_insert_with(|| vec![0.0; v.len()]);
                for (a, b) in e.iter_mut().zip(v.iter()) {
                    *a += s * b;
                }
            }
        }
        LinPoly(m)
    }
}

fn monomials_upto(d: usize) -> Vec<Mono> {
    let mut out = Vec::new();
    for i in 0..=d {
        for j in 0..=(d - i) {
            out.push((i, j));
        }
    }
    out
}

fn unit_vec(k: usize, idx: usize) -> Vec<f64> {
    let mut v = vec![0.0; k];
    v[idx] = 1.0;
    v
}

/// Build the determining-equation design matrix: each row is one monomial of
/// the symmetry condition, each column one generator coefficient.
fn build_design_matrix(
    f_terms: &[(usize, usize, f64)],
    ansatz_degree: usize,
) -> Result<(Vec<Vec<f64>>, usize), String> {
    if ansatz_degree > 6 {
        return Err("ansatz_degree above 6 is not supported".to_string());
    }
    let mut f = Poly::default();
    for (px, py, c) in f_terms {
        if !c.is_finite() {
            return Err("non-finite coefficient in f".to_string());
        }
        f = f.add(&Poly::term(*px, *py, *c));
    }

    let monos = monomials_upto(ansatz_degree);
    let nm = monos.len();
    let k = 2 * nm;
    let mut xi = LinPoly::default();
    let mut eta = LinPoly::default();
    for (i, m) in monos.iter().enumerate() {
        xi.0.insert(*m, unit_vec(k, i));
        eta.0.insert(*m, unit_vec(k, nm + i));
    }

    let f2 = f.mul(&f);
    let cond = eta
        .dx()
        .add(&eta.dy().sub(&xi.dx()).mul_plain(&f))
        .sub(&xi.dy().mul_plain(&f2))
        .sub(&xi.mul_plain(&f.dx()))
        .sub(&eta.mul_plain(&f.dy()));

    let rows: Vec<Vec<f64>> = cond.0.values().cloned().collect();
    Ok((rows, k))
}

/// Discover point symmetries of y' = f(x, y) with polynomial f.
pub fn discover_point_symmetries(
    f_terms: &[(usize, usize, f64)],
    config: &LieConfig,
) -> Result<LieResult, String> {
    let (rows, k) = build_design_matrix(f_terms, config.ansatz_degree)?;
    let n_eq = rows.len();

    // Gram matrix G = MᵀM; its near-null eigenvectors span the symmetries.
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let mut s = 0.0f64;
            for r in &rows {
                s += r[i] * r[j];
            }
            gram[i * k + j] = s;
            gram[j * k + i] = s;
        }
    }
    let (evals, evecs) = crate::rime::implicit::jacobi_eigen(&gram, k, 150, 1e-15);

    // FLAW-15: the nullspace gate must be RELATIVE to the operator norm.
    // Eigenvalues of MᵀM scale quadratically with data amplitude and linearly
    // with sample count, so an unnormalized absolute tolerance missed genuine
    // symmetries on large/high-amplitude data and fabricated them on
    // small-amplitude data. Normalize: null ⇔ |λ| ≤ tol · λ_max. When every
    // eigenvalue is 0 (all-zero Gram matrix) the singular value structure is
    // degenerate and all directions lie in the nullspace.
    let max_eval = evals.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let tol = config.null_tol.max(0.0);
    let mut null_idx = Vec::new();
    for (idx, &v) in evals.iter().enumerate() {
        if v.abs() <= tol * max_eval.max(1e-300) {
            null_idx.push(idx);
        }
    }

    let min_eval = evals.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);

    let monos = monomials_upto(config.ansatz_degree);
    let nm = monos.len();
    let denom = (n_eq.max(1) as f64).sqrt();
    let mut generators = Vec::new();
    let mut max_res = 0.0f64;
    for &idx in &null_idx {
        let c: Vec<f64> = (0..k).map(|j| evecs[j * k + idx]).collect();
        let mut res_sq = 0.0f64;
        for r in &rows {
            let d: f64 = r.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            res_sq += d * d;
        }
        let residual = (res_sq).sqrt() / denom;
        max_res = max_res.max(residual);

        let thresh = config.sparsity_threshold.max(0.0);
        let mut xi = Vec::new();
        let mut eta = Vec::new();
        for (i, m) in monos.iter().enumerate() {
            if c[i].abs() > thresh {
                xi.push(LieTerm {
                    px: m.0,
                    py: m.1,
                    coeff: c[i],
                });
            }
            if c[nm + i].abs() > thresh {
                eta.push(LieTerm {
                    px: m.0,
                    py: m.1,
                    coeff: c[nm + i],
                });
            }
        }
        generators.push(LieGenerator { xi, eta, residual });
    }

    Ok(LieResult {
        ansatz_degree: config.ansatz_degree,
        n_unknowns: k,
        n_equations: n_eq,
        nullity: null_idx.len(),
        generators,
        max_basis_residual: max_res,
        min_eigenvalue: if min_eval.is_finite() { min_eval } else { 0.0 },
        max_eigenvalue: max_eval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual_of_known(
        f_terms: &[(usize, usize, f64)],
        degree: usize,
        xi: &[(usize, usize, f64)],
        eta: &[(usize, usize, f64)],
    ) -> f64 {
        let (rows, k) = build_design_matrix(f_terms, degree).unwrap();
        let monos = monomials_upto(degree);
        let nm = monos.len();
        let mut c = vec![0.0; k];
        for (px, py, v) in xi {
            let i = monos.iter().position(|m| m == &(*px, *py)).unwrap();
            c[i] = *v;
        }
        for (px, py, v) in eta {
            let i = monos.iter().position(|m| m == &(*px, *py)).unwrap();
            c[nm + i] = *v;
        }
        let mut res_sq = 0.0f64;
        for r in &rows {
            let d: f64 = r.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            res_sq += d * d;
        }
        res_sq.sqrt() / ((rows.len().max(1) as f64).sqrt())
    }

    #[test]
    fn test_scaling_symmetry_of_exponential_ode() {
        // y' = y admits the scaling symmetry (ξ, η) = (0, y).
        let f = [(0usize, 1usize, 1.0f64)];
        assert!(residual_of_known(&f, 2, &[], &[(0, 1, 1.0)]) < 1e-9);
        let res = discover_point_symmetries(&f, &LieConfig::default()).unwrap();
        assert!(res.nullity >= 1, "nullity: {}", res.nullity);
        assert!(
            res.max_basis_residual < 1e-6,
            "residual: {}",
            res.max_basis_residual
        );
    }

    #[test]
    fn test_trivial_ode_admits_exact_translation_algebra() {
        // y' = 0: every ξ(x, y) plus every x-independent η(y) is a symmetry.
        // Degree 2: 6 ξ monomials + 3 η monomials = 9.
        let res = discover_point_symmetries(&[], &LieConfig::default()).unwrap();
        assert_eq!(res.nullity, 9);
        assert!(res.max_basis_residual < 1e-9);
    }

    #[test]
    fn test_y_translation_symmetry_when_f_depends_only_on_x() {
        // y' = x admits (ξ, η) = (0, 1).
        let f = [(1usize, 0usize, 1.0f64)];
        assert!(residual_of_known(&f, 2, &[], &[(0, 0, 1.0)]) < 1e-9);
        let res = discover_point_symmetries(&f, &LieConfig::default()).unwrap();
        assert!(res.nullity >= 1);
    }

    #[test]
    fn test_rejects_bad_inputs() {
        let bad_cfg = LieConfig {
            ansatz_degree: 7,
            ..LieConfig::default()
        };
        assert!(discover_point_symmetries(&[(0, 0, 1.0)], &bad_cfg).is_err());
        assert!(discover_point_symmetries(&[(0, 0, f64::NAN)], &LieConfig::default()).is_err());
    }
}
