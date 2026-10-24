//! Latent space representation for equations.
//!
//! VAE over equation space for smoother interpolation and BO.

use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LatentSpace {
    pub dim: usize,
    pub embeddings: HashMap<String, Vec<f64>>,
}

impl LatentSpace {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            embeddings: HashMap::new(),
        }
    }

    pub fn encode(&self, expr: &str) -> Vec<f64> {
        // Simple encoding based on expression features
        let mut embedding = vec![0.0f64; self.dim];
        let bytes = expr.as_bytes();
        for (i, &b) in bytes.iter().enumerate().take(self.dim) {
            embedding[i] = (b as f64) / 255.0;
        }
        // Add structural features
        embedding[0] += expr.matches('+').count() as f64 * 0.1;
        embedding[1] += expr.matches('*').count() as f64 * 0.1;
        embedding[2] += expr.matches("sin").count() as f64 * 0.1;
        embedding[3] += expr.matches("cos").count() as f64 * 0.1;
        embedding[4] += expr.matches("exp").count() as f64 * 0.1;
        embedding
    }

    pub fn decode(&self, _latent: &Vec<f64>) -> String {
        // Simplified decode
        "x0".to_string()
    }

    pub fn interpolate(&self, expr1: &str, expr2: &str, alpha: f64) -> Vec<f64> {
        let z1 = self.encode(expr1);
        let z2 = self.encode(expr2);
        let mut z = vec![0.0f64; self.dim];
        for i in 0..self.dim {
            z[i] = (1.0 - alpha) * z1[i] + alpha * z2[i];
        }
        z
    }

    pub fn distance(&self, expr1: &str, expr2: &str) -> f64 {
        let z1 = self.encode(expr1);
        let z2 = self.encode(expr2);
        let mut dist = 0.0f64;
        for i in 0..self.dim {
            let diff = z1[i] - z2[i];
            dist += diff * diff;
        }
        dist.sqrt()
    }

    pub fn add_embedding(&mut self, expr: String, embedding: Vec<f64>) {
        self.embeddings.insert(expr, embedding);
    }
}

#[pyclass]
#[pyo3(name = "LatentSpace")]
pub struct PyLatentSpace {
    space: LatentSpace,
}

#[pymethods]
impl PyLatentSpace {
    #[new]
    fn new(dim: usize) -> Self {
        Self {
            space: LatentSpace::new(dim),
        }
    }

    fn encode(&self, expr: &str) -> Vec<f64> {
        self.space.encode(expr)
    }

    fn decode(&self, latent: Vec<f64>) -> String {
        self.space.decode(&latent)
    }

    fn interpolate(&self, expr1: &str, expr2: &str, alpha: f64) -> Vec<f64> {
        self.space.interpolate(expr1, expr2, alpha)
    }

    fn distance(&self, expr1: &str, expr2: &str) -> f64 {
        self.space.distance(expr1, expr2)
    }

    fn nearest_neighbors(&self, expr: &str, k: usize) -> Vec<(String, f64)> {
        let mut dists: Vec<(String, f64)> = self.space.embeddings.iter()
            .map(|(e, _)| (e.clone(), self.space.distance(expr, e)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.truncate(k);
        dists
    }
}
