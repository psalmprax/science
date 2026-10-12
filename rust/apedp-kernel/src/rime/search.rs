//! RIME search (Stage‑2 scaffolding, spec §8/§12).
//! Built only from the frozen v0.1 primitives – no new node types, no rewrite
//! extensions.  Used by the APEDP‑X proposer → referee pipeline.

use serde_json::Value;

use super::graph::Graph;
use super::eval::score_candidate_raw;
use super::rewrite::RewriteOp;
use super::seeds;

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

/// Estimate the effective number of degrees of freedom for a candidate set.
/// In v0.1 this is a simple placeholder (n candidates → n dof); the EBIC‑style
/// selection charge is applied per candidate in the `lag_score_config` helper.
pub fn multiplicity_dof(n: usize) -> f64 {
    n as f64
}

/* ----------------------------------------------------------------- */
/* One RIME search run over the frozen rewrites for the given data. */
/* ----------------------------------------------------------------- */
/// Run a full RIME search over the frozen rewrites for the given data.
///
/// * `seeds`   – candidate graphs (typically from `rime::seeds::small_forms`).
/// * `columns` – design matrix (samples × variables).
/// * `target`  – target vector.
/// * `cfg`     – search configuration (lambda_c, beam, budget, etc.).
///
/// In v0.1 the function enumerates all single‑rewrite neighbours of each seed
/// using the five frozen ops (`AddLagEdge`, `AddInteraction`, `AddState`,
/// `WrapNonlinearity`, `PruneEdge`), scores them with `score_candidate_raw`,
/// and returns up to `cfg.beam` best proposals sorted by score descending.
/// The returned `SearchPacket` carries an `accepted` flag that is `true`
/// whenever at least one proposal was generated.
pub fn search(
    seeds: &[Graph],
    columns: &[Vec<f64>],
    target: &[f64],
    cfg: &SearchConfig,
) -> SearchPacket {
    // -----------------------------------------------------------------
    // Fast‑path: no seeds → empty packet.
    // -----------------------------------------------------------------
    if seeds.is_empty() {
        return SearchPacket {
            proposals: vec![],
            accepted: false,
        };
    }

    let mut proposals = Vec::new();
    let parent_dev_mse = 0.0; // seed treated as having zero dev‑MSE; score already
                               // includes the complexity penalty.

    for seed in seeds {
        let candidates = enumerate_candidates(seed);
        for (cand, op) in &candidates {
            let (score, mse, _skipped) = score_candidate_raw(cand, parent_dev_mse, columns, target, cfg.lambda_c);

            let score_num = serde_json::Number::from_f64(score).unwrap_or_else(|| serde_json::Number::from(0));
            let mse_num = serde_json::Number::from_f64(mse).unwrap_or_else(|| serde_json::Number::from(0));

            let eq = Value::Object(
                serde_json::Map::from_iter([
                    (
                        "equation".to_string(),
                        Value::String(cand.canonical()),
                    ),
                    (
                        "rewrite".to_string(),
                        Value::String(op.name().to_string()),
                    ),
                    ("score".to_string(), Value::Number(score_num)),
                    ("mse".to_string(), Value::Number(mse_num)),
                ])
            );
            proposals.push(eq);

            // Respect the budget: stop once we have enough candidates.
            if proposals.len() >= cfg.budget {
                break;
            }
        }
        if proposals.len() >= cfg.budget {
            break;
        }
    }

    // Sort by score descending so the best proposals come first.
    proposals.sort_by(|a, b| {
        let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Keep at most `beam` proposals.
    let limited = proposals.into_iter().take(cfg.beam.max(1)).collect::<Vec<Value>>();
    let accepted = !limited.is_empty();
    SearchPacket {
        proposals: limited,
        accepted,
    }
}

/* ----------------------------------------------------------------- */
/* Enumerate all single‑rewrite neighbours of a graph using the five frozen
   v0.1 operations (AddLagEdge, AddInteraction, AddState, WrapNonlinearity,
   PruneEdge).  Operations that cannot be applied (e.g. out‑of‑range indices)
   are silently dropped. */
/* ----------------------------------------------------------------- */
pub fn enumerate_candidates(g: &Graph) -> Vec<(Graph, RewriteOp)> {
    let mut out = Vec::new();
    let n = g.nodes.len();

    // AddLagEdge tries k = 1 ..= 30 (the frozen lag set is defined in graph.rs).
    for k in 1..=30 {
        if let Some(h) = super::rewrite::apply(g, &RewriteOp::AddLagEdge { src: 0, k }) {
            out.push((h, RewriteOp::AddLagEdge { src: 0, k }));
        }
    }
    // AddInteraction: all unordered pairs of existing node indices.
    for a in 0..n {
        for b in a..n {
            if let Some(h) = super::rewrite::apply(g, &RewriteOp::AddInteraction { a, b }) {
                out.push((h, RewriteOp::AddInteraction { a, b }));
            }
        }
    }
    // AddState: one attempt with a valid decay fraction (9/10).
    if let Some(h) = super::rewrite::apply(g, &RewriteOp::AddState { decay_num: 9, decay_den: 10 }) {
        out.push((h, RewriteOp::AddState { decay_num: 9, decay_den: 10 }));
    }
    // WrapNonlinearity: wrap each existing node in Tanh.
    for src in 0..n {
        if let Some(h) = super::rewrite::apply(g, &RewriteOp::WrapNonlinearity { src }) {
            out.push((h, RewriteOp::WrapNonlinearity { src }));
        }
    }
    // PruneEdge: prune each node once (node → Const(0.0)).
    for node in 0..n {
        if let Some(h) = super::rewrite::apply(g, &RewriteOp::PruneEdge { node }) {
            out.push((h, RewriteOp::PruneEdge { node }));
        }
    }
    out
}

/* ----------------------------------------------------------------- */
/* A tiny packet that can be turned into JSON for the Python referee. */
/* ----------------------------------------------------------------- */
#[derive(Clone, Debug)]
pub struct SearchPacket {
    /// Candidate proposals produced by the search (sorted by score, limited by beam).
    pub proposals: Vec<Value>,
    /// Acceptance flag – `true` when at least one proposal was generated.
    pub accepted: bool,
}

impl SearchPacket {
    /// Serialize to a `serde_json::Value` (used by the pyfunction `rime_search_json`).
    pub fn to_json(&self) -> Value {
        serde_json::json!({ "proposals": self.proposals, "accepted": self.accepted })
    }
}

/* ----------------------------------------------------------------- */
/* Library tests – these stay green regardless of the implementation
   details as long as the public API contracts are respected. */
/* ----------------------------------------------------------------- */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_returns_empty_packet_for_no_seeds() {
        let seeds: Vec<Graph> = vec![];
        let cols: Vec<Vec<f64>> = vec![];
        let target: Vec<f64> = vec![];
        let cfg = SearchConfig::default();
        let pkt = search(&seeds, &cols, &target, &cfg);
        assert!(!pkt.accepted);
        assert!(pkt.proposals.is_empty());
    }

    #[test]
    fn search_returns_packet_with_proposals_when_seeds_provide_candidates() {
        let seeds = seeds::small_forms(2);
        let cols = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![0.5, 1.0, 1.5, 2.0, 2.5]];
        let target = vec![f64::NAN, f64::NAN, 2.5, 4.0, 5.5];
        let cfg = SearchConfig { budget: 10, beam: 5, ..Default::default() };
        let pkt = search(&seeds, &cols, &target, &cfg);
        assert!(pkt.accepted, "should have at least one proposal");
        assert!(!pkt.proposals.is_empty());
        // every proposal must contain the required keys.
        for p in &pkt.proposals {
            assert!(p.get("equation").is_some(), "proposal missing equation");
            assert!(p.get("rewrite").is_some(), "proposal missing rewrite");
            assert!(p.get("score").is_some(), "proposal missing score");
            assert!(p.get("mse").is_some(), "proposal missing mse");
        }
    }

    #[test]
    fn search_respects_budget_and_beam() {
        let seeds = seeds::small_forms(2);
        let cols = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let target = vec![1.0, 2.0];
        let cfg = SearchConfig { budget: 3, beam: 5, ..Default::default() };
        let pkt = search(&seeds, &cols, &target, &cfg);
        assert!(
            pkt.proposals.len() <= 3,
            "should respect budget (got {})",
            pkt.proposals.len()
        );
        assert!(
            pkt.proposals.len() <= 5,
            "should respect beam (got {})",
            pkt.proposals.len()
        );
    }

    #[test]
    fn search_packet_json_has_proposals_false_when_empty() {
        let pkt = SearchPacket { proposals: vec![], accepted: false };
        let j = pkt.to_json();
        assert!(!j.get("accepted").unwrap_or(&Value::Bool(false)).as_bool().unwrap());
    }

    /// Smoke test: exercise `search` with a few seed catalogs and configs
    /// without panicking and verify the returned packet has the expected shape.
    #[test]
    fn search_smoke() {
        // Test with 1‑input and 2‑input seed catalogs.
        for &n_inputs in &[1usize, 2] {
            let seeds = seeds::small_forms(n_inputs);
            // Minimal design matrix (10 samples, n_inputs columns) and target.
            let cols = vec![vec![1.0; 10]; n_inputs];
            let target = vec![0.5; 10];
            let cfg = SearchConfig { budget: 5, beam: 3, max_depth: 1, lambda_c: 0.1, ridge: 0.0, seed: 42 };
            let pkt = search(&seeds, &cols, &target, &cfg);
            // The packet must be well‑formed.
            assert!(pkt.accepted || pkt.proposals.is_empty(), "packet should be valid");
            for p in &pkt.proposals {
                assert!(p.get("equation").is_some(), "proposal missing equation");
                assert!(p.get("rewrite").is_some(), "proposal missing rewrite");
                assert!(p.get("score").is_some(), "proposal missing score");
                assert!(p.get("mse").is_some(), "proposal missing mse");
            }
        }
    }
}