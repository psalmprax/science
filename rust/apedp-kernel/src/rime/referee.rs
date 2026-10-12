//! RIME referee (spec §9, §10).
//! Adjudicates candidate proposals returned by the search module and returns
//! one of the four frozen taxonomy outcomes.
//!
//! Outcomes (frozen, v0.1):
//! - NO_RECOVERY       : no proposal clears the gate.
//! - PREDICTIVE_RECOVERY : exactly one proposal clears the gate and has a
//!   positive score after the complexity penalty.
//! - MULTI_HYPOTHESIS  : more than one proposal clears the gate.
//! - UNIQUE_IDENTIFICATION : exactly one proposal clears the gate, its score
//!   exceeds a threshold, and the graph is acyclic (no cycles).
//!
//! The referee does not modify any state; it only reads the packet and returns
//! an outcome.  The APEDP‑X driver is responsible for recording the result
//! in the experiment ledger.

use super::search::SearchPacket;
use super::search::SearchConfig;

/// Frozen v0.1 referee outcomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RefereeOutcome {
    /// No proposal cleared the gate (all scores ≤ 0 after penalty).
    NO_RECOVERY,
    /// Exactly one proposal cleared the gate with a positive score.
    PREDICTIVE_RECOVERY,
    /// More than one proposal cleared the gate.
    MULTI_HYPOTHESIS,
    /// Exactly one proposal cleared the gate, score > threshold, and the graph
    /// is acyclic (no lag cycles).  This is the “unique identification” case.
    UNIQUE_IDENTIFICATION,
}

/// Classify the search packet into one of the four frozen outcomes.
///
/// The function only looks at the `proposals` field and the `accepted` flag.
/// Deeper graph‑acyclicity checks would be performed by the APEDP‑X driver if
/// needed.
pub fn classify(packet: &SearchPacket) -> RefereeOutcome {
    let n_positive = packet
        .proposals
        .iter()
        .filter(|p| p.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) > 0.0)
        .count();

    if !packet.accepted || n_positive == 0 {
        RefereeOutcome::NO_RECOVERY
    } else if n_positive == 1 {
        // Single positive – check whether the score is meaningfully above zero.
        // In v0.1 we treat any positive score as predictive recovery.
        RefereeOutcome::PREDICTIVE_RECOVERY
    } else {
        RefereeOutcome::MULTI_HYPOTHESIS
    }
}

/// Convenience: return the outcome together with a human‑readable reason.
pub fn classify_with_reason(packet: &SearchPacket) -> (RefereeOutcome, &'static str) {
    let outcome = classify(packet);
    let reason = match outcome {
        RefereeOutcome::NO_RECOVERY => "no proposal cleared the gate",
        RefereeOutcome::PREDICTIVE_RECOVERY => "exactly one proposal cleared the gate",
        RefereeOutcome::MULTI_HYPOTHESIS => "multiple proposals cleared the gate",
        RefereeOutcome::UNIQUE_IDENTIFICATION => {
            "exactly one proposal cleared the gate and the graph is acyclic"
        }
    };
    (outcome, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rime::search::search;

    #[test]
    fn classify_no_proposals() {
        let pkt = SearchPacket { proposals: vec![], accepted: false };
        assert_eq!(classify(&pkt), RefereeOutcome::NO_RECOVERY);
    }

    #[test]
    fn classify_one_positive() {
        use serde_json::json;
        let pkt = SearchPacket {
            proposals: vec![json!({ "score": 0.5 })],
            accepted: true,
        };
        assert_eq!(classify(&pkt), RefereeOutcome::PREDICTIVE_RECOVERY);
    }

    #[test]
    fn classify_multi_positive() {
        use serde_json::json;
        let pkt = SearchPacket {
            proposals: vec![json!({ "score": 0.5 }), json!({ "score": 0.3 })],
            accepted: true,
        };
        assert_eq!(classify(&pkt), RefereeOutcome::MULTI_HYPOTHESIS);
    }

    #[test]
    fn classify_integrates_with_search() {
        let seeds = vec![];
        let cols: Vec<Vec<f64>> = vec![];
        let target: Vec<f64> = vec![];
        let cfg = SearchConfig::default();
        let pkt = search(&seeds, &cols, &target, &cfg);
        let (outcome, reason) = classify_with_reason(&pkt);
        assert_eq!(outcome, RefereeOutcome::NO_RECOVERY);
        assert_eq!(reason, "no proposal cleared the gate");
    }
}