//! RIME search (Stage‑2 scaffolding, spec §8/§12).
//! Built only from the frozen v0.1 primitives – no new node types, no rewrite
//! extensions.  Used by the APEDP‑X proposer → referee pipeline.

use serde_json::Value;

use super::graph::Graph;

/* Data carried for one env‑cell in the search. */
#[derive(Clone, Debug)]
pub struct EnvData {
    /// Design matrix rows (samples × variables).
    pub cols: Vec<Vec<f64>>,
    /// Target vector (length = number of samples).
    pub target: Vec<f64>,
}

/* Search configuration (all fields have sensible defaults). */
#[derive(Clone, Debug, Default)]
pub struct SearchConfig {
    pub max_depth: usize,
    pub beam: usize,
    pub budget: usize,
    /// Penalty per edge added to the dev‑MSE score.
    pub lambda_c: f64,
    pub ridge: f64,
    pub seed: u64,
}

/// Aggregation rule for combining per‑env scores.
#[derive(Clone, Debug, PartialEq)]
pub enum EnvAggregation {
    /// Take the median of the worst‑gated per‑env scores.
    MedianWorstGated,
    /// Simple arithmetic mean per env.
    MeanPerEnv,
}

/// Compute the (population) variance of a slice of f64.
pub fn target_variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / n as f64
}

/// One RIME search run over the frozen rewrites.
///
/// Returns a `SearchPacket` that the caller can serialize to JSON.
pub fn search_envs(
    _seeds: &[Graph],
    _envs: &[EnvData],
    _cfg: &SearchConfig,
    _agg: EnvAggregation,
    _gamma_gate: f64,
) -> SearchPacket {
    // -----------------------------------------------------------------
    // TODO: real candidate enumeration + evaluation uses the frozen rewrites
    // (graph.rs / rewrite.rs) and the evaluator (eval.rs).  For v0.1 we emit
    // an empty packet so the pipeline does not crash; the APEDP‑X referee will
    // reject the empty candidacy.
    // -----------------------------------------------------------------
    SearchPacket {
        proposals: vec![],
        accepted: true,
    }
}

/// Run one full RIME search over the frozen rewrites for the given data.
///
/// * `seeds`   – candidate graphs (typically from `rime::seeds::small_forms`).
/// * `columns` – design matrix (samples × variables).
/// * `target`  – target vector.
/// * `cfg`     – search configuration (lambda_c, beam, budget, etc.).
pub fn search(seeds: &[Graph], columns: &[Vec<f64>], target: &[f64], cfg: &SearchConfig) -> SearchPacket {
    // TODO: real enumeration + evaluation using the frozen rewrites (graph/rewrite/eval).
    // For v0.1 we emit an empty packet; the APEDP‑X referee will reject the empty candidacy.
    SearchPacket { proposals: vec![], accepted: true }
}

/* A tiny packet that can be turned into JSON for the Python referee. */
#[derive(Clone, Debug)]
pub struct SearchPacket {
    /// Candidate proposals produced by the search (empty in v0.1).
    pub proposals: Vec<Value>,
    /// Acceptance flag – `true` when the referee deems the candidacy viable.
    pub accepted: bool,
}

impl SearchPacket {
    /// Serialize to a `serde_json::Value` (used by the pyfunction `rime_search_json`).
    pub fn to_json(&self) -> Value {
        // Minimal JSON – the Python side expects at least a "proposals" key.
        serde_json::json!({ "proposals": self.proposals })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rime::seeds;

    #[test]
    fn target_variance_basic() {
        let d = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = target_variance(&d);
        // population variance of that sample is 4.0
        assert!((v - 4.0).abs() < 1e-6);
    }

    #[test]
    fn small_forms_catalogue_size() {
        let forms = seeds::small_forms(2);
        assert!(forms.len() <= 60, "catalogue exceeds budget");
    }

    #[test]
    fn search_packet_json_has_proposals() {
        let pkt = SearchPacket { proposals: vec![serde_json::json!({ "equation": "v0" })], accepted: true };
        let j = pkt.to_json();
        assert!(j.get("proposals").is_some());
    }
}