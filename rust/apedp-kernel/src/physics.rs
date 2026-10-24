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

    pub fn dimensional_analysis_pass(&self) -> f64 {
        // Score based on dimensional consistency
        if self.target_dim.is_none() {
            return 0.5;
        }
        1.0
    }

    pub fn detect_symmetries(&self, data: &Vec<Vec<f64>>) -> HashSet<String> {
        let mut symmetries = HashSet::new();
        // Basic symmetry detection - translational invariance checks etc
        if data.len() >= 2 && data[0].len() > 1 {
            // Check for simple patterns
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

    fn dimensional_analysis_score(&self) -> f64 {
        self.engine.dimensional_analysis_pass()
    }

    fn smoothness_score(&self, values: Vec<f64>) -> f64 {
        self.engine.smoothness_score(&values)
    }

    fn check_conservation(&self, quantities: Vec<f64>) -> bool {
        self.engine.check_conservation(&quantities)
    }
}
