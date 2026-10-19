//! RIME immutable candidate records (spec §7, §12).
//!
//! A record binds (graph id, parent id, rewrite op, seed, dev score). Records
//! are content-addressed: the record id is SHA-256 over the canonical graph
//! plus the seed, so identical (graph, seed) pairs always share an id.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::graph::Graph;

/// An immutable candidate record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub record_id: String,
    pub graph_id: String,
    pub parent_id: Option<String>,
    pub rewrite_op: String,
    pub seed: u64,
    pub dev_score: f64,
    pub canonical: String,
}

impl CandidateRecord {
    pub fn new(
        graph: &Graph,
        parent_id: Option<String>,
        rewrite_op: &str,
        seed: u64,
        dev_score: f64,
    ) -> Self {
        let canonical = graph.canonical();
        let graph_id = graph.id();
        let mut h = Sha256::new();
        h.update(graph_id.as_bytes());
        h.update(seed.to_le_bytes());
        let record_id = h
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();
        CandidateRecord {
            record_id,
            graph_id,
            parent_id,
            rewrite_op: rewrite_op.to_string(),
            seed,
            dev_score,
            canonical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::graph::Node;
    use super::*;

    #[test]
    fn record_ids_deterministic() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = x;
        let a = CandidateRecord::new(&g, None, "seed", 42, 1.5);
        let b = CandidateRecord::new(&g, None, "seed", 42, 1.5);
        assert_eq!(a, b);
        assert_eq!(a.record_id.len(), 64);
    }

    #[test]
    fn seed_changes_record_not_graph() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = x;
        let a = CandidateRecord::new(&g, None, "seed", 1, 0.0);
        let b = CandidateRecord::new(&g, None, "seed", 2, 0.0);
        assert_eq!(a.graph_id, b.graph_id);
        assert_ne!(a.record_id, b.record_id);
    }

    #[test]
    fn parent_link_preserved() {
        let mut g = Graph::new();
        let x = g.push(Node::Input(0));
        g.output = x;
        let p = CandidateRecord::new(&g, None, "seed", 0, 0.0);
        let c = CandidateRecord::new(&g, Some(p.graph_id.clone()), "AddLagEdge", 0, 0.5);
        assert_eq!(c.parent_id, Some(p.graph_id));
        assert_eq!(c.rewrite_op, "AddLagEdge");
    }
}
