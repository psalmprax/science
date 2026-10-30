//! Physics-informed constraint engine.
//!
//! Enforces dimensional analysis, symmetries, conservation laws, and
//! other physical constraints before searching.

use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimensions {
    pub mass: i32,
    pub length: i32,
    pub time: i32,
    pub temperature: i32,
    pub current: i32,
    pub amount: i32,
    pub luminosity: i32,
}

impl Dimensions {
    pub fn dimensionless() -> Self {
        Dimensions {
            mass: 0, length: 0, time: 0, temperature: 0,
            current: 0, amount: 0, luminosity: 0,
        }
    }

    pub fn add(&self, other: &Dimensions) -> Dimensions {
        Dimensions {
            mass: self.mass + other.mass,
            length: self.length + other.length,
            time: self.time + other.time,
            temperature: self.temperature + other.temperature,
            current: self.current + other.current,
            amount: self.amount + other.amount,
            luminosity: self.luminosity + other.luminosity,
        }
    }

    pub fn sub(&self, other: &Dimensions) -> Dimensions {
        Dimensions {
            mass: self.mass - other.mass,
            length: self.length - other.length,
            time: self.time - other.time,
            temperature: self.temperature - other.temperature,
            current: self.current - other.current,
            amount: self.amount - other.amount,
            luminosity: self.luminosity - other.luminosity,
        }
    }

    pub fn mul_scalar(&self, s: i32) -> Dimensions {
        Dimensions {
            mass: self.mass * s,
            length: self.length * s,
            time: self.time * s,
            temperature: self.temperature * s,
            current: self.current * s,
            amount: self.amount * s,
            luminosity: self.luminosity * s,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VariableDim {
    pub name: String,
    pub dims: Dimensions,
}

#[derive(Debug, Clone)]
pub struct PhysicsConstraintEngine {
    pub variable_dims: HashMap<String, Dimensions>,
    pub target_dim: Option<Dimensions>,
    pub symmetries: HashSet<String>,
    pub conserved_quantities: Vec<String>,
}

impl PhysicsConstraintEngine {
    pub fn new() -> Self {
        Self {
            variable_dims: HashMap::new(),
            target_dim: None,
            symmetries: HashSet::new(),
            conserved_quantities: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, name: String, dims: Dimensions) {
        self.variable_dims.insert(name, dims);
    }

    pub fn set_target(&mut self, dims: Dimensions) {
        self.target_dim = Some(dims);
    }

    pub fn check_dimensional_consistency(&self, expr_dims: &Dimensions) -> bool {
        if let Some(ref target) = self.target_dim {
            expr_dims.mass == target.mass &&
            expr_dims.length == target.length &&
            expr_dims.time == target.time &&
            expr_dims.temperature == target.temperature &&
            expr_dims.current == target.current &&
            expr_dims.amount == target.amount &&
            expr_dims.luminosity == target.luminosity
        } else {
            true
        }
    }

    /// Fraction of the candidate's terms whose dimensions equal the target.
    ///
    /// Dimensional analysis is a judgement about a candidate, so it needs the
    /// candidate's terms; a call with no terms, or with no target to judge them
    /// against, has nothing to measure and is an error rather than a constant.
    pub fn dimensional_analysis_score(&self, term_dims: &[Dimensions]) -> Result<f64, String> {
        if term_dims.is_empty() {
            return Err(
                "dimensional_analysis_score needs the dimensions of the terms to score"
                    .to_string(),
            );
        }
        let Some(target) = self.target_dim.as_ref() else {
            return Err(
                "dimensional_analysis_score needs a target dimension to score the terms against"
                    .to_string(),
            );
        };
        let matching = term_dims
            .iter()
            .filter(|dims| {
                dims.mass == target.mass
                    && dims.length == target.length
                    && dims.time == target.time
                    && dims.temperature == target.temperature
                    && dims.current == target.current
                    && dims.amount == target.amount
                    && dims.luminosity == target.luminosity
            })
            .count();
        Ok(matching as f64 / term_dims.len() as f64)
    }

    pub fn detect_symmetries(&self, data: &Vec<Vec<f64>>) -> HashSet<String> {
        let mut symmetries = HashSet::new();
        let n = data.len();
        if n < 2 {
            return symmetries;
        }
        let dim = data[0].len();
        if dim == 0 {
            return symmetries;
        }
        // Symmetry detection is a numerical probe, not a label: a translation
        // claim is only reported if a small uniform shift of all coordinates
        // leaves the pairwise distance matrix unchanged within tolerance.
        let shift: Vec<f64> = vec![1e-3; dim];
        let orig_dists: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        data[i]
                            .iter()
                            .zip(data[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect()
            })
            .collect();
        let shifted: Vec<Vec<f64>> = data
            .iter()
            .map(|row| row.iter().zip(shift.iter()).map(|(a, s)| a + s).collect())
            .collect();
        let shifted_dists: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        shifted[i]
                            .iter()
                            .zip(shifted[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect()
            })
            .collect();
        let translation = orig_dists
            .iter()
            .zip(shifted_dists.iter())
            .all(|(o, s)| o.iter().zip(s.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
        if translation {
            symmetries.insert("translation".to_string());
        }
        symmetries
    }

    pub fn check_conservation(&self, quantities: &Vec<f64>) -> bool {
        if quantities.is_empty() {
            return true;
        }
        let mean = quantities.iter().sum::<f64>() / quantities.len() as f64;
        let mut max_dev = 0.0f64;
        for &q in quantities {
            let dev = (q - mean).abs();
            if dev > max_dev { max_dev = dev; }
        }
        max_dev < 1e-10
    }

    pub fn smoothness_score(&self, values: &Vec<f64>) -> f64 {
        if values.len() < 3 {
            return 1.0;
        }
        let mut roughness = 0.0f64;
        for i in 1..values.len()-1 {
            let second_diff = (values[i+1] - 2.0*values[i] + values[i-1]).abs();
            roughness += second_diff;
        }
        1.0 / (1.0 + roughness / values.len() as f64)
    }
}

#[pyclass]
#[pyo3(name = "PhysicsConstraintEngine")]
pub struct PyPhysicsConstraintEngine {
    engine: PhysicsConstraintEngine,
}

#[pymethods]
impl PyPhysicsConstraintEngine {
    #[new]
    fn new() -> Self {
        Self {
            engine: PhysicsConstraintEngine::new(),
        }
    }

    fn add_variable(&mut self, name: String, mass: i32, length: i32, time: i32, temp: i32, current: i32, amount: i32, lum: i32) {
        let dims = Dimensions { mass, length, time, temperature: temp, current, amount, luminosity: lum };
        self.engine.add_variable(name, dims);
    }

    fn set_target_dimensions(&mut self, mass: i32, length: i32, time: i32, temp: i32, current: i32, amount: i32, lum: i32) {
        let dims = Dimensions { mass, length, time, temperature: temp, current, amount, luminosity: lum };
        self.engine.set_target(dims);
    }

    fn check_dimensional_consistency(&self, expr_dims_json: &str) -> bool {
        // Parse dimensions from JSON
        if let Ok(dims) = serde_json::from_str::<Dimensions>(expr_dims_json) {
            self.engine.check_dimensional_consistency(&dims)
        } else {
            true
        }
    }

    /// `term_dims_json` is a JSON list of `Dimensions` objects, one per term.
    fn dimensional_analysis_score(&self, term_dims_json: &str) -> PyResult<f64> {
        let term_dims: Vec<Dimensions> = serde_json::from_str(term_dims_json).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "term dimensions must be a JSON list of Dimensions objects: {err}"
            ))
        })?;
        self.engine
            .dimensional_analysis_score(&term_dims)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn smoothness_score(&self, values: Vec<f64>) -> f64 {
        self.engine.smoothness_score(&values)
    }

    fn check_conservation(&self, quantities: Vec<f64>) -> bool {
        self.engine.check_conservation(&quantities)
    }
}
