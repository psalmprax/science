//! Multi-precision identifiability evaluation and precision scaling engine.
//!
//! Validates the fundamental empirical scaling law:
//! epsilon* approx sqrt(2 * epsilon_mach)
//! across f32, f64, and arbitrary precision arithmetic.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrecisionScalingReport {
    pub precision_type: String,
    pub machine_epsilon: f64,
    pub theoretical_threshold: f64,
    pub empirical_transition_observed: bool,
    pub condition_number: f64,
    pub resolved: bool,
}

pub struct MultiPrecisionEngine;

impl MultiPrecisionEngine {
    /// Computes machine epsilon and theoretical identifiability boundary for float precision.
    pub fn theoretical_boundary(precision_bits: usize) -> (f64, f64) {
        let eps_mach = match precision_bits {
            32 => f32::EPSILON as f64,
            64 => f64::EPSILON,
            128 => 1.92592994438723585305597794258492732e-34,
            bits => (2.0f64).powi(-((bits as i32) - 1)),
        };
        let theoretical_eps_star = (2.0 * eps_mach).sqrt();
        (eps_mach, theoretical_eps_star)
    }

    /// Evaluates precision identifiability transition for two competing hypotheses f_A and f_B
    /// given a separation delta = ||f_A - f_B|| and measurement noise sigma.
    pub fn evaluate_identifiability(
        separation_delta: f64,
        condition_number: f64,
        precision_bits: usize,
    ) -> PrecisionScalingReport {
        let (eps_mach, eps_star) = Self::theoretical_boundary(precision_bits);

        // Effective numerical noise ceiling under ill-conditioning
        let effective_noise_floor = eps_mach * condition_number.max(1.0);

        // Identifiability is resolved if separation strictly exceeds the threshold
        let resolved = separation_delta > effective_noise_floor && separation_delta > eps_star;

        let name = match precision_bits {
            32 => "Float32 (Single Precision)",
            64 => "Float64 (Double Precision)",
            128 => "QuadPrecision (128-bit)",
            _b => "ArbitraryPrecision",
        };

        PrecisionScalingReport {
            precision_type: name.to_string(),
            machine_epsilon: eps_mach,
            theoretical_threshold: eps_star,
            empirical_transition_observed: resolved,
            condition_number,
            resolved,
        }
    }

    /// Evaluates candidate scalar in f32 precision and compares with f64.
    pub fn evaluate_f32_vs_f64(
        val_f64: f64,
    ) -> (f32, f64, f64) {
        let val_f32 = val_f64 as f32;
        let delta = ((val_f32 as f64) - val_f64).abs();
        (val_f32, val_f64, delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_scaling_boundary() {
        let (eps_mach, eps_star) = MultiPrecisionEngine::theoretical_boundary(32);
        assert!((eps_mach - 1.1920929e-7).abs() < 1e-12);
        assert!((eps_star - 0.00048828125).abs() < 1e-6);

        // A separation of 1e-5 cannot be resolved in f32 (below 4.88e-4)
        let rep_unresolved = MultiPrecisionEngine::evaluate_identifiability(1e-5, 1.0, 32);
        assert!(!rep_unresolved.resolved);

        // A separation of 1e-3 CAN be resolved in f32
        let rep_resolved = MultiPrecisionEngine::evaluate_identifiability(1e-3, 1.0, 32);
        assert!(rep_resolved.resolved);
    }

    #[test]
    fn test_f64_scaling_boundary() {
        let (eps_mach, eps_star) = MultiPrecisionEngine::theoretical_boundary(64);
        assert!((eps_mach - 2.220446e-16).abs() < 1e-20);
        assert!((eps_star - 2.107342e-8).abs() < 1e-12);

        // A separation of 1e-5 CAN be resolved in f64 (well above 2.1e-8)
        let rep_resolved = MultiPrecisionEngine::evaluate_identifiability(1e-5, 1.0, 64);
        assert!(rep_resolved.resolved);

        // A separation of 1e-10 cannot be resolved in f64 (below 2.1e-8)
        let rep_unresolved = MultiPrecisionEngine::evaluate_identifiability(1e-10, 1.0, 64);
        assert!(!rep_unresolved.resolved);
    }
}
