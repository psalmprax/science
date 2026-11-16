//! RIME referee (spec §9, §10).
//!
//! Adjudicates candidate proposals returned by `crate::rime::search` and
//! classifies a [`SearchOutcome`] (or [`SearchEnvsOutcome`] / JSON search packet)
//! into one of the four frozen taxonomy outcomes:
//!
//! - `NO_RECOVERY`: no valid proposal clears the gate (or inputs/threshold are invalid).
//! - `PREDICTIVE_RECOVERY`: exactly one valid proposal clears the gate with a
//!   positive score after complexity penalty.
//! - `MULTI_HYPOTHESIS`: more than one valid proposal clears the gate.
//! - `UNIQUE_IDENTIFICATION`: exactly one valid proposal clears the gate, its
//!   score exceeds the caller threshold, and its arena graph passes structural
//!   topological-DAG validation.
//!
//! The referee is purely functional and mutates no ledger or search state.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::graph::{Graph, LAG_SET};
use super::search::{EnvAggregation, Proposal, SearchEnvsOutcome, SearchOutcome};

/// Frozen v0.1 / v0.1.2 referee outcomes.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefereeOutcome {
    /// No valid proposal cleared the gate (or inputs/threshold failed validation).
    NO_RECOVERY,
    /// Exactly one valid proposal cleared the gate with a positive score.
    PREDICTIVE_RECOVERY,
    /// More than one valid proposal cleared the gate.
    MULTI_HYPOTHESIS,
    /// Exactly one valid proposal cleared the gate, score > threshold, and the
    /// graph is structurally topological (acyclic DAG).
    UNIQUE_IDENTIFICATION,
}

impl RefereeOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            RefereeOutcome::NO_RECOVERY => "NO_RECOVERY",
            RefereeOutcome::PREDICTIVE_RECOVERY => "PREDICTIVE_RECOVERY",
            RefereeOutcome::MULTI_HYPOTHESIS => "MULTI_HYPOTHESIS",
            RefereeOutcome::UNIQUE_IDENTIFICATION => "UNIQUE_IDENTIFICATION",
        }
    }
}

/// Explicit referee validation / invocation failure for fallible callers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RefereeError {
    /// `unique_score_threshold` was NaN, infinite, or negative.
    InvalidThreshold,
    /// `baseline_mse` was NaN, infinite, or non-positive when proposals were claimed.
    InvalidBaselineMse,
    /// Multi-env `worst_env_mse` was invalid when proposals were claimed.
    InvalidWorstEnvMse,
    /// Packet JSON could not be parsed or lacked required structure.
    InvalidJson(String),
}

impl std::fmt::Display for RefereeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefereeError::InvalidThreshold => {
                write!(f, "unique_score_threshold must be finite and >= 0.0")
            }
            RefereeError::InvalidBaselineMse => {
                write!(f, "baseline_mse must be finite and > 0.0")
            }
            RefereeError::InvalidWorstEnvMse => {
                write!(f, "worst_env_mse must be finite and > 0.0")
            }
            RefereeError::InvalidJson(msg) => write!(f, "invalid referee JSON packet: {}", msg),
        }
    }
}

impl std::error::Error for RefereeError {}

/// Mode controlling how `Proposal.graph` is validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphValidationMode {
    /// Validate live `p.graph` when populated, or validate canonical + SHA-256
    /// integrity when `p.graph` is empty due to `#[serde(skip)]` deserialization.
    /// An empty `"O0"` canonical form is always rejected.
    LiveOrDeserialized,
    /// Require an in-memory populated `p.graph` passing `Graph::validate()` and
    /// matching metadata.
    RequireLiveGraph,
}

fn sha256_hex(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    h.finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

fn parse_two_operands(rest: &str, node_idx: usize) -> bool {
    let Some((a_str, b_str)) = rest.split_once(':') else {
        return false;
    };
    let (Ok(a), Ok(b)) = (a_str.parse::<usize>(), b_str.parse::<usize>()) else {
        return false;
    };
    a < node_idx && b < node_idx
}

/// Validate a serialized `Graph::canonical()` string, its SHA-256 `graph_id`,
/// topological ordering, frozen lags, finite constants, and `edge_count`.
///
/// Used when `Proposal.graph` was skipped during JSON deserialization
/// (`#[serde(skip)]`). Rejects empty graphs (`"O0"`).
pub fn validate_canonical(canonical: &str, expected_graph_id: &str, expected_edges: usize) -> bool {
    if canonical.is_empty() || sha256_hex(canonical) != expected_graph_id {
        return false;
    }
    let parts: Vec<&str> = canonical.split(';').collect();
    if parts.len() < 2 {
        return false;
    }
    let Some((out_part, node_parts)) = parts.split_last() else {
        return false;
    };
    let Some(out_str) = out_part.strip_prefix('O') else {
        return false;
    };
    let Ok(out_idx) = out_str.parse::<usize>() else {
        return false;
    };
    if out_idx >= node_parts.len() {
        return false;
    }

    let mut edges = 0usize;
    for (idx, frag) in node_parts.iter().enumerate() {
        if let Some(rest) = frag.strip_prefix("SI") {
            let Ok(a) = rest.parse::<usize>() else {
                return false;
            };
            if a >= idx {
                return false;
            }
            edges += 1;
        } else if let Some(rest) = frag.strip_prefix("CO") {
            let Ok(a) = rest.parse::<usize>() else {
                return false;
            };
            if a >= idx {
                return false;
            }
            edges += 1;
        } else if let Some(rest) = frag.strip_prefix("EX") {
            let Ok(a) = rest.parse::<usize>() else {
                return false;
            };
            if a >= idx {
                return false;
            }
            edges += 1;
        } else if let Some(rest) = frag.strip_prefix("LG") {
            let Ok(a) = rest.parse::<usize>() else {
                return false;
            };
            if a >= idx {
                return false;
            }
            edges += 1;
        } else if let Some(rest) = frag.strip_prefix("DF") {
            let Ok(a) = rest.parse::<usize>() else {
                return false;
            };
            if a >= idx {
                return false;
            }
            edges += 1;
        } else if let Some(rest) = frag.strip_prefix('I') {
            if rest.parse::<usize>().is_err() {
                return false;
            }
        } else if let Some(rest) = frag.strip_prefix('K') {
            let Ok(c) = rest.parse::<f64>() else {
                return false;
            };
            if !c.is_finite() {
                return false;
            }
        } else if let Some(rest) = frag.strip_prefix('A') {
            if !parse_two_operands(rest, idx) {
                return false;
            }
            edges += 2;
        } else if let Some(rest) = frag.strip_prefix('M') {
            if !parse_two_operands(rest, idx) {
                return false;
            }
            edges += 2;
        } else if let Some(rest) = frag.strip_prefix('D') {
            if !parse_two_operands(rest, idx) {
                return false;
            }
            edges += 2;
        } else if let Some(rest) = frag
            .strip_prefix('N')
            .or_else(|| frag.strip_prefix('T'))
            .or_else(|| frag.strip_prefix('S'))
            .or_else(|| frag.strip_prefix('B'))
        {
            let Ok(a) = rest.parse::<usize>() else {
                return false;
            };
            if a >= idx {
                return false;
            }
            edges += 1;
        } else if let Some(rest) = frag.strip_prefix('L') {
            let Some((a_str, k_str)) = rest.split_once(':') else {
                return false;
            };
            let (Ok(a), Ok(k)) = (a_str.parse::<usize>(), k_str.parse::<usize>()) else {
                return false;
            };
            if a >= idx || !LAG_SET.contains(&k) {
                return false;
            }
            edges += 1;
        } else if let Some(rest) = frag.strip_prefix('P') {
            let mut segs = rest.split(':');
            let (Some(src_s), Some(num_s), Some(den_s), None) =
                (segs.next(), segs.next(), segs.next(), segs.next())
            else {
                return false;
            };
            let (Ok(src), Ok(_p_num), Ok(p_den)) = (
                src_s.parse::<usize>(),
                num_s.parse::<i32>(),
                den_s.parse::<u32>(),
            ) else {
                return false;
            };
            if src >= idx || p_den == 0 || p_den > 10 {
                return false;
            }
            edges += 1;
        } else if let Some(rest) = frag.strip_prefix('X') {
            let mut segs = rest.split(':');
            let (Some(src_s), Some(decay_s), Some(init_s), None) =
                (segs.next(), segs.next(), segs.next(), segs.next())
            else {
                return false;
            };
            let (Ok(src), Ok(decay), Ok(init)) = (
                src_s.parse::<usize>(),
                decay_s.parse::<f64>(),
                init_s.parse::<f64>(),
            ) else {
                return false;
            };
            if src >= idx || !decay.is_finite() || !init.is_finite() {
                return false;
            }
            edges += 1;
        } else {
            return false;
        }
    }

    edges == expected_edges
}

fn valid_proposal(p: &Proposal, mode: GraphValidationMode) -> bool {
    let numerics_ok = p.score.is_finite()
        && p.score > 0.0
        && p.ref_score.is_finite()
        && p.ref_score > 0.0
        && p.mse.is_finite()
        && p.mse >= 0.0
        && p.r2.is_finite()
        && p.r2 <= 1.0 + 1e-12
        && p.scale.is_finite()
        && p.intercept.is_finite()
        && p.complexity.is_finite()
        && p.complexity >= 0.0
        && p.selection_dof.is_finite()
        && p.selection_dof >= 0.0
        && p.n_eff > 0
        && !p.graph_id.is_empty()
        && !p.canonical.is_empty();
    if !numerics_ok {
        return false;
    }

    if !p.graph.nodes.is_empty() {
        p.graph.validate().is_ok()
            && p.graph.id() == p.graph_id
            && p.graph.canonical() == p.canonical
            && p.graph.edge_count() == p.edges
    } else {
        match mode {
            GraphValidationMode::RequireLiveGraph => false,
            GraphValidationMode::LiveOrDeserialized => {
                validate_canonical(&p.canonical, &p.graph_id, p.edges)
            }
        }
    }
}

fn valid_positive_proposals(outcome: &SearchOutcome, mode: GraphValidationMode) -> Vec<&Proposal> {
    outcome
        .proposals
        .iter()
        .filter(|p| valid_proposal(p, mode))
        .collect()
}

fn valid_threshold(unique_score_threshold: f64) -> bool {
    unique_score_threshold.is_finite() && unique_score_threshold >= 0.0
}

fn valid_positive_mse(mse: f64) -> bool {
    mse.is_finite() && mse > 0.0
}

/// Fallible referee classification with optional `unique_score_threshold`.
pub fn try_classify_with_mode(
    outcome: &SearchOutcome,
    unique_score_threshold: Option<f64>,
    mode: GraphValidationMode,
) -> Result<(RefereeOutcome, &'static str), RefereeError> {
    if let Some(th) = unique_score_threshold {
        if !valid_threshold(th) {
            return Err(RefereeError::InvalidThreshold);
        }
    }
    if outcome.accepted == 0 || outcome.proposals.is_empty() {
        return Ok((
            RefereeOutcome::NO_RECOVERY,
            outcome_reason(&RefereeOutcome::NO_RECOVERY),
        ));
    }
    if !valid_positive_mse(outcome.baseline_mse) {
        return Err(RefereeError::InvalidBaselineMse);
    }

    let valid_pos = valid_positive_proposals(outcome, mode);
    if valid_pos.is_empty() {
        return Ok((
            RefereeOutcome::NO_RECOVERY,
            "no proposal passed structural and numerical validation",
        ));
    }
    if valid_pos.len() > 1 {
        return Ok((
            RefereeOutcome::MULTI_HYPOTHESIS,
            outcome_reason(&RefereeOutcome::MULTI_HYPOTHESIS),
        ));
    }
    let best = valid_pos[0];
    if let Some(th) = unique_score_threshold {
        if best.score > th {
            return Ok((
                RefereeOutcome::UNIQUE_IDENTIFICATION,
                outcome_reason(&RefereeOutcome::UNIQUE_IDENTIFICATION),
            ));
        }
    }
    Ok((
        RefereeOutcome::PREDICTIVE_RECOVERY,
        outcome_reason(&RefereeOutcome::PREDICTIVE_RECOVERY),
    ))
}

/// Fallible referee classification (`LiveOrDeserialized` mode).
pub fn try_classify(
    outcome: &SearchOutcome,
) -> Result<(RefereeOutcome, &'static str), RefereeError> {
    try_classify_with_mode(outcome, None, GraphValidationMode::LiveOrDeserialized)
}

/// Fallible referee classification with explicit unique-identification threshold.
pub fn try_classify_with_threshold(
    outcome: &SearchOutcome,
    unique_score_threshold: f64,
) -> Result<(RefereeOutcome, &'static str), RefereeError> {
    try_classify_with_mode(
        outcome,
        Some(unique_score_threshold),
        GraphValidationMode::LiveOrDeserialized,
    )
}

/// Classify a [`SearchOutcome`] into one of the frozen referee outcomes.
///
/// Fails closed (`NO_RECOVERY`) when `accepted == 0`, `baseline_mse` is
/// invalid, or any numerical/graph invariant fails. Handles `#[serde(skip)]`
/// deserialized outcomes by verifying canonical graph grammar and SHA-256 IDs.
pub fn classify(outcome: &SearchOutcome) -> RefereeOutcome {
    classify_with_reason(outcome).0
}

/// Classify a [`SearchOutcome`] strictly requiring in-memory populated graphs.
pub fn classify_strict(outcome: &SearchOutcome) -> RefereeOutcome {
    try_classify_with_mode(outcome, None, GraphValidationMode::RequireLiveGraph)
        .map(|(o, _)| o)
        .unwrap_or(RefereeOutcome::NO_RECOVERY)
}

/// Classify a [`SearchOutcome`] with an explicit score margin threshold for
/// [`RefereeOutcome::UNIQUE_IDENTIFICATION`].
///
/// Fails closed (`NO_RECOVERY`) if `unique_score_threshold` is non-finite or
/// negative, or if `baseline_mse` / proposal invariants are invalid.
pub fn classify_with_threshold(
    outcome: &SearchOutcome,
    unique_score_threshold: f64,
) -> RefereeOutcome {
    classify_with_threshold_and_reason(outcome, unique_score_threshold).0
}

/// Return a short static diagnostic phrase corresponding to an outcome.
pub fn outcome_reason(outcome: &RefereeOutcome) -> &'static str {
    match outcome {
        RefereeOutcome::NO_RECOVERY => "no proposal cleared the gate",
        RefereeOutcome::PREDICTIVE_RECOVERY => "exactly one proposal cleared the gate",
        RefereeOutcome::MULTI_HYPOTHESIS => "multiple proposals cleared the gate",
        RefereeOutcome::UNIQUE_IDENTIFICATION => {
            "exactly one proposal cleared the gate and the graph is acyclic"
        }
    }
}

/// Convenience: classify [`SearchOutcome`] and pair it with a human-readable reason.
pub fn classify_with_reason(outcome: &SearchOutcome) -> (RefereeOutcome, &'static str) {
    match try_classify_with_mode(outcome, None, GraphValidationMode::LiveOrDeserialized) {
        Ok(res) => res,
        Err(RefereeError::InvalidThreshold) => (
            RefereeOutcome::NO_RECOVERY,
            "invalid unique identification threshold",
        ),
        Err(RefereeError::InvalidBaselineMse) => {
            (RefereeOutcome::NO_RECOVERY, "invalid baseline mse")
        }
        Err(RefereeError::InvalidWorstEnvMse) => {
            (RefereeOutcome::NO_RECOVERY, "invalid worst_env_mse")
        }
        Err(RefereeError::InvalidJson(_)) => {
            (RefereeOutcome::NO_RECOVERY, "invalid referee packet")
        }
    }
}

/// Convenience: classify [`SearchOutcome`] with unique-identification threshold
/// and pair it with a human-readable reason.
pub fn classify_with_threshold_and_reason(
    outcome: &SearchOutcome,
    unique_score_threshold: f64,
) -> (RefereeOutcome, &'static str) {
    match try_classify_with_mode(
        outcome,
        Some(unique_score_threshold),
        GraphValidationMode::LiveOrDeserialized,
    ) {
        Ok(res) => res,
        Err(RefereeError::InvalidThreshold) => (
            RefereeOutcome::NO_RECOVERY,
            "invalid unique identification threshold",
        ),
        Err(RefereeError::InvalidBaselineMse) => {
            (RefereeOutcome::NO_RECOVERY, "invalid baseline mse")
        }
        Err(RefereeError::InvalidWorstEnvMse) => {
            (RefereeOutcome::NO_RECOVERY, "invalid worst_env_mse")
        }
        Err(RefereeError::InvalidJson(_)) => {
            (RefereeOutcome::NO_RECOVERY, "invalid referee packet")
        }
    }
}

/// Classify a multi-environment [`SearchEnvsOutcome`].
pub fn classify_envs(outcome: &SearchEnvsOutcome) -> (RefereeOutcome, &'static str) {
    classify_envs_with_threshold(outcome, None)
}

/// Classify a multi-environment [`SearchEnvsOutcome`] with optional unique threshold.
pub fn classify_envs_with_threshold(
    outcome: &SearchEnvsOutcome,
    unique_score_threshold: Option<f64>,
) -> (RefereeOutcome, &'static str) {
    if let Some(th) = unique_score_threshold {
        if !valid_threshold(th) {
            return (
                RefereeOutcome::NO_RECOVERY,
                "invalid unique identification threshold",
            );
        }
    }
    if outcome.accepted == 0 || outcome.proposals.is_empty() {
        return (
            RefereeOutcome::NO_RECOVERY,
            outcome_reason(&RefereeOutcome::NO_RECOVERY),
        );
    }
    if matches!(outcome.aggregation, EnvAggregation::MedianWorstGated) && !outcome.worst_env_passed
    {
        return (
            RefereeOutcome::NO_RECOVERY,
            "worst-environment invariance gate failed",
        );
    }
    if !valid_positive_mse(outcome.baseline_mse) {
        return (RefereeOutcome::NO_RECOVERY, "invalid baseline mse");
    }
    if !valid_positive_mse(outcome.worst_env_mse) {
        return (RefereeOutcome::NO_RECOVERY, "invalid worst_env_mse");
    }
    let valid_pos: Vec<&Proposal> = outcome
        .proposals
        .iter()
        .filter(|p| valid_proposal(p, GraphValidationMode::LiveOrDeserialized))
        .collect();
    if valid_pos.is_empty() {
        return (
            RefereeOutcome::NO_RECOVERY,
            "no proposal passed structural and numerical validation",
        );
    }
    if valid_pos.len() > 1 {
        return (
            RefereeOutcome::MULTI_HYPOTHESIS,
            outcome_reason(&RefereeOutcome::MULTI_HYPOTHESIS),
        );
    }
    if let Some(th) = unique_score_threshold {
        if valid_pos[0].score > th {
            return (
                RefereeOutcome::UNIQUE_IDENTIFICATION,
                outcome_reason(&RefereeOutcome::UNIQUE_IDENTIFICATION),
            );
        }
    }
    (
        RefereeOutcome::PREDICTIVE_RECOVERY,
        outcome_reason(&RefereeOutcome::PREDICTIVE_RECOVERY),
    )
}

/// Classify a JSON packet emitted by `SearchOutcome::to_json()` or
/// `SearchEnvsOutcome::to_json()`, deserializing and validating every embedded
/// `graph_json` DAG and `dev_evidence` metric.
pub fn classify_packet_json(
    packet_json: &str,
    unique_score_threshold: Option<f64>,
) -> Result<(RefereeOutcome, &'static str), RefereeError> {
    if let Some(th) = unique_score_threshold {
        if !valid_threshold(th) {
            return Err(RefereeError::InvalidThreshold);
        }
    }
    let root: serde_json::Value =
        serde_json::from_str(packet_json).map_err(|e| RefereeError::InvalidJson(e.to_string()))?;
    let obj = root
        .as_object()
        .ok_or_else(|| RefereeError::InvalidJson("packet root must be a JSON object".into()))?;
    let accepted = obj
        .get("accepted")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| RefereeError::InvalidJson("missing integer field 'accepted'".into()))?
        as usize;
    let proposals = obj
        .get("proposals")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RefereeError::InvalidJson("missing array field 'proposals'".into()))?;

    if accepted == 0 || proposals.is_empty() {
        return Ok((
            RefereeOutcome::NO_RECOVERY,
            outcome_reason(&RefereeOutcome::NO_RECOVERY),
        ));
    }

    if let Some(agg) = obj.get("aggregation").and_then(|v| v.as_str()) {
        if agg == "median_worst_gated" || agg == "MedianWorstGated" {
            let passed = obj
                .get("worst_env_passed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !passed {
                return Ok((
                    RefereeOutcome::NO_RECOVERY,
                    "worst-environment invariance gate failed",
                ));
            }
        }
    }
    if let Some(w_mse) = obj.get("worst_env_mse") {
        let Some(v) = w_mse.as_f64() else {
            return Err(RefereeError::InvalidWorstEnvMse);
        };
        if !valid_positive_mse(v) {
            return Err(RefereeError::InvalidWorstEnvMse);
        }
    }

    let baseline_mse = obj
        .get("baseline_mse")
        .and_then(|v| v.as_f64())
        .ok_or(RefereeError::InvalidBaselineMse)?;
    if !valid_positive_mse(baseline_mse) {
        return Err(RefereeError::InvalidBaselineMse);
    }

    let mut valid_scores: Vec<f64> = Vec::new();
    for item in proposals {
        let Some(pobj) = item.as_object() else {
            continue;
        };
        let Some(graph_json) = pobj.get("graph_json").and_then(|v| v.as_str()) else {
            continue;
        };
        let Ok(graph) = serde_json::from_str::<Graph>(graph_json) else {
            continue;
        };
        let Some(graph_id) = pobj.get("graph_id").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(canonical) = pobj.get("canonical").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(ev) = pobj.get("dev_evidence").and_then(|v| v.as_object()) else {
            continue;
        };
        let (
            Some(score),
            Some(ref_score),
            Some(mse),
            Some(r2),
            Some(scale),
            Some(intercept),
            Some(complexity),
            Some(edges),
            Some(selection_dof),
            Some(n_eff),
        ) = (
            ev.get("score").and_then(|v| v.as_f64()),
            ev.get("ref_score").and_then(|v| v.as_f64()),
            ev.get("mse").and_then(|v| v.as_f64()),
            ev.get("r2").and_then(|v| v.as_f64()),
            ev.get("scale").and_then(|v| v.as_f64()),
            ev.get("intercept").and_then(|v| v.as_f64()),
            ev.get("complexity").and_then(|v| v.as_f64()),
            ev.get("edges").and_then(|v| v.as_u64()),
            ev.get("selection_dof").and_then(|v| v.as_f64()),
            ev.get("n_eff").and_then(|v| v.as_u64()),
        )
        else {
            continue;
        };
        let prop = Proposal {
            graph_id: graph_id.to_string(),
            parent_id: pobj
                .get("parent_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            rewrite_op: pobj
                .get("rewrite_op")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            depth: pobj.get("depth").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            equation: pobj
                .get("equation")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            canonical: canonical.to_string(),
            score,
            ref_score,
            mse,
            r2,
            scale,
            intercept,
            complexity,
            edges: edges as usize,
            selection_dof,
            n_eff: n_eff as usize,
            seed: pobj.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
            graph,
        };
        if valid_proposal(&prop, GraphValidationMode::RequireLiveGraph) {
            valid_scores.push(score);
        }
    }

    if valid_scores.is_empty() {
        return Ok((
            RefereeOutcome::NO_RECOVERY,
            "no proposal passed structural and numerical validation",
        ));
    }
    if valid_scores.len() > 1 {
        return Ok((
            RefereeOutcome::MULTI_HYPOTHESIS,
            outcome_reason(&RefereeOutcome::MULTI_HYPOTHESIS),
        ));
    }
    if let Some(th) = unique_score_threshold {
        if valid_scores[0] > th {
            return Ok((
                RefereeOutcome::UNIQUE_IDENTIFICATION,
                outcome_reason(&RefereeOutcome::UNIQUE_IDENTIFICATION),
            ));
        }
    }
    Ok((
        RefereeOutcome::PREDICTIVE_RECOVERY,
        outcome_reason(&RefereeOutcome::PREDICTIVE_RECOVERY),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rime::graph::{Graph, Node};
    use crate::rime::search::{search, SearchConfig};
    use crate::rime::seeds::small_forms;

    fn valid_lag_graph(lag_k: usize) -> Graph {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        let l = g.push(Node::Lag(x, lag_k));
        g.output = l;
        g
    }

    fn make_proposal(score: f64, graph: Graph) -> Proposal {
        Proposal {
            graph_id: graph.id(),
            parent_id: String::new(),
            rewrite_op: "seed".to_string(),
            depth: 0,
            equation: graph.describe(),
            canonical: graph.canonical(),
            score,
            ref_score: score,
            mse: 0.01,
            r2: 0.9,
            scale: 1.0,
            intercept: 0.0,
            complexity: 1.0,
            edges: graph.edge_count(),
            selection_dof: 0.0,
            n_eff: 100,
            seed: 0,
            graph,
        }
    }

    fn make_outcome(proposals: Vec<Proposal>, accepted: usize) -> SearchOutcome {
        SearchOutcome {
            evaluated: proposals.len(),
            accepted,
            rejected_score: 0,
            dropped_dedup: 0,
            dropped_scope: 0,
            baseline_mse: 1.0,
            baseline_graph_id: "null".to_string(),
            proposals,
        }
    }

    #[test]
    fn classify_no_proposals() {
        let out = make_outcome(vec![], 0);
        assert_eq!(classify(&out), RefereeOutcome::NO_RECOVERY);
    }

    #[test]
    fn classify_one_positive() {
        let out = make_outcome(vec![make_proposal(0.5, valid_lag_graph(1))], 1);
        assert_eq!(classify(&out), RefereeOutcome::PREDICTIVE_RECOVERY);
    }

    #[test]
    fn classify_multi_positive() {
        let out = make_outcome(
            vec![
                make_proposal(0.5, valid_lag_graph(1)),
                make_proposal(0.3, valid_lag_graph(2)),
            ],
            2,
        );
        assert_eq!(classify(&out), RefereeOutcome::MULTI_HYPOTHESIS);
    }

    #[test]
    fn classify_rejects_non_topological_graph() {
        let mut cyclic = Graph::new();
        cyclic.output = cyclic.push(Node::Neg(0));
        let out = make_outcome(vec![make_proposal(0.8, cyclic)], 1);
        assert_eq!(classify(&out), RefereeOutcome::NO_RECOVERY);
        assert_eq!(
            classify_with_threshold(&out, 0.2),
            RefereeOutcome::NO_RECOVERY
        );
    }

    #[test]
    fn classify_with_threshold_promotes_unique_identification() {
        let out = make_outcome(vec![make_proposal(0.85, valid_lag_graph(2))], 1);
        let (outcome, reason) = classify_with_threshold_and_reason(&out, 0.5);
        assert_eq!(outcome, RefereeOutcome::UNIQUE_IDENTIFICATION);
        assert!(reason.contains("acyclic"));
        assert_eq!(
            classify_with_threshold(&out, 1.0),
            RefereeOutcome::PREDICTIVE_RECOVERY
        );
    }

    #[test]
    fn classify_handles_serde_roundtrip_with_skipped_graph() {
        let out = make_outcome(vec![make_proposal(0.85, valid_lag_graph(2))], 1);
        let serialized = serde_json::to_string(&out).expect("serialize");
        let deserialized: SearchOutcome = serde_json::from_str(&serialized).expect("deserialize");
        assert!(deserialized.proposals[0].graph.nodes.is_empty());

        // LiveOrDeserialized validates canonical grammar + SHA-256 graph_id
        assert_eq!(classify(&deserialized), RefereeOutcome::PREDICTIVE_RECOVERY);
        assert_eq!(
            classify_with_threshold(&deserialized, 0.5),
            RefereeOutcome::UNIQUE_IDENTIFICATION
        );
        // Strict mode still insists on an in-memory Graph arena.
        assert_eq!(classify_strict(&deserialized), RefereeOutcome::NO_RECOVERY);

        // Tampered canonical string or graph_id must fail validation.
        let mut tampered = deserialized.clone();
        tampered.proposals[0].canonical = "I0;L0:999;O1".to_string();
        assert_eq!(classify(&tampered), RefereeOutcome::NO_RECOVERY);

        // Empty Graph::new() ("O0") must always be rejected even via LiveOrDeserialized.
        let empty_prop = make_proposal(0.85, Graph::new());
        let empty_out = make_outcome(vec![empty_prop], 1);
        assert_eq!(classify(&empty_out), RefereeOutcome::NO_RECOVERY);
    }

    #[test]
    fn classify_rejects_corrupt_numerics_and_baseline() {
        let mut bad_mse = make_proposal(0.5, valid_lag_graph(1));
        bad_mse.mse = -0.01;
        assert_eq!(
            classify(&make_outcome(vec![bad_mse], 1)),
            RefereeOutcome::NO_RECOVERY
        );

        let mut bad_ref_score = make_proposal(0.5, valid_lag_graph(1));
        bad_ref_score.ref_score = -0.1;
        assert_eq!(
            classify(&make_outcome(vec![bad_ref_score], 1)),
            RefereeOutcome::NO_RECOVERY
        );

        let mut zero_neff = make_proposal(0.5, valid_lag_graph(1));
        zero_neff.n_eff = 0;
        assert_eq!(
            classify(&make_outcome(vec![zero_neff], 1)),
            RefereeOutcome::NO_RECOVERY
        );

        let mut bad_r2 = make_proposal(0.5, valid_lag_graph(1));
        bad_r2.r2 = 1.5;
        assert_eq!(
            classify(&make_outcome(vec![bad_r2], 1)),
            RefereeOutcome::NO_RECOVERY
        );

        let mut bad_base = make_outcome(vec![make_proposal(0.5, valid_lag_graph(1))], 1);
        bad_base.baseline_mse = 0.0;
        assert_eq!(classify(&bad_base), RefereeOutcome::NO_RECOVERY);
        assert_eq!(
            try_classify(&bad_base),
            Err(RefereeError::InvalidBaselineMse)
        );
    }

    #[test]
    fn classify_fails_closed_on_invalid_threshold() {
        let out = make_outcome(vec![make_proposal(0.85, valid_lag_graph(2))], 1);
        for bad_t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
            let (outcome, reason) = classify_with_threshold_and_reason(&out, bad_t);
            assert_eq!(outcome, RefereeOutcome::NO_RECOVERY);
            assert_eq!(reason, "invalid unique identification threshold");
            assert_eq!(
                try_classify_with_threshold(&out, bad_t),
                Err(RefereeError::InvalidThreshold)
            );
        }
    }

    #[test]
    fn classify_packet_json_validates_embedded_graph() {
        let out = make_outcome(vec![make_proposal(0.85, valid_lag_graph(2))], 1);
        let pkt_str = serde_json::to_string(&out.to_json()).unwrap();
        let (o, _) = classify_packet_json(&pkt_str, Some(0.5)).unwrap();
        assert_eq!(o, RefereeOutcome::UNIQUE_IDENTIFICATION);

        let mut cyclic = Graph::new();
        cyclic.output = cyclic.push(Node::Neg(0));
        let bad_pkt = serde_json::json!({
            "baseline_mse": 1.0,
            "accepted": 1,
            "proposals": [{
                "graph_json": serde_json::to_string(&cyclic).unwrap(),
                "graph_id": cyclic.id(),
                "canonical": cyclic.canonical(),
                "dev_evidence": {
                    "score": 0.9, "ref_score": 0.9, "mse": 0.01, "r2": 0.9,
                    "scale": 1.0, "intercept": 0.0, "complexity": 1.0,
                    "edges": 1, "selection_dof": 0.0, "n_eff": 100
                }
            }]
        });
        let (o_bad, _) = classify_packet_json(&bad_pkt.to_string(), Some(0.1)).unwrap();
        assert_eq!(o_bad, RefereeOutcome::NO_RECOVERY);
    }

    #[test]
    fn classify_integrates_with_search() {
        let seeds: Vec<Graph> = vec![];
        let cols: Vec<Vec<f64>> = vec![];
        let target: Vec<f64> = vec![];
        let cfg = SearchConfig::default();
        let out = search(&seeds, &cols, &target, &cfg);
        let (outcome, reason) = classify_with_reason(&out);
        assert_eq!(outcome, RefereeOutcome::NO_RECOVERY);
        assert_eq!(reason, "no proposal cleared the gate");

        let n = 400;
        let x: Vec<f64> = (0..n).map(|t| (t as f64 * 0.13).sin()).collect();
        let mut y = vec![f64::NAN; n];
        for (t, slot) in y.iter_mut().enumerate().skip(2) {
            *slot = 3.0 * x[t - 2] + 1.0;
        }
        let cfg_sig = SearchConfig::noise_scaled(&y, n - 2, 1.5)
            .with_max_depth(1)
            .with_budget(60);
        let out_sig = search(&small_forms(1), &[x], &y, &cfg_sig);
        let sig_outcome = classify(&out_sig);
        assert!(
            matches!(
                sig_outcome,
                RefereeOutcome::PREDICTIVE_RECOVERY | RefereeOutcome::MULTI_HYPOTHESIS
            ),
            "expected signal recovery, got {:?}",
            sig_outcome
        );
    }
}
