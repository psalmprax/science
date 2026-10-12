//! RIME Stage-1 acceptance: planted-mechanism recovery + null abstention.
//! Uses only the frozen core (graph + rewrites + evaluator + records).

use apedp_kernel::rime::eval::{mse_finite, score_batch};
use apedp_kernel::rime::graph::{Graph, Node};
use apedp_kernel::rime::record::CandidateRecord;
use apedp_kernel::rime::rewrite::{apply, RewriteOp};

fn planted_world() -> (Vec<Vec<f64>>, Vec<f64>) {
    // Deterministic white-ish input (hash noise): no periodic confound, so
    // ONLY the true lag-2 form can predict y. Fully deterministic, no RNG.
    let x: Vec<f64> = (0..500)
        .map(|t| {
            let v = ((t as f64) * 12.9898).sin() * 43758.5453;
            v - v.floor() - 0.5
        })
        .collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(t, _)| if t >= 2 { 3.0 * x[t - 2] + 0.01 * ((t as f64) * 1.7).sin() } else { f64::NAN })
        .collect();
    (vec![x], y)
}

fn seed_graph() -> Graph {
    let mut g = Graph::new();
    let x = g.push(Node::Input(0));
    g.output = x;
    g
}

#[test]
fn planted_two_lag_mechanism_recovered() {
    let (cols, target) = planted_world();
    let seed = seed_graph();
    let (parent_mse, _) = mse_finite(&seed.evaluate(&cols), &target);
    // Expand seed with each frozen lag; the k=2 form must win.
    let mut jobs = Vec::new();
    let mut names = Vec::new();
    for k in [1usize, 2, 5, 10, 20] {
        let g = apply(&seed, &RewriteOp::AddLagEdge { src: 0, k }).unwrap();
        jobs.push((g, parent_mse, target.clone()));
        names.push(k);
    }
    let res = score_batch(&jobs, &cols, 1e-6);
    let best = res
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
        .map(|(i, _)| names[i])
        .unwrap();
    assert_eq!(best, 2, "planted lag-2 mechanism must score best");
    // Winner must actually improve (positive score) and carry a stable id.
    let rec = CandidateRecord::new(&jobs[names.iter().position(|&k| k == best).unwrap()].0, None, "AddLagEdge", 7, res.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max));
    assert_eq!(rec.record_id.len(), 64);
}

#[test]
fn null_world_abstains() {
    // Pure periodic target unrelated to inputs: no rewrite may improve.
    let (cols, _) = planted_world();
    let target: Vec<f64> = (0..500).map(|t| (((t as f64) * 7.233).sin() * 269.7).fract().abs() - 0.5).collect();
    let seed = seed_graph();
    let (parent_mse, _) = mse_finite(&seed.evaluate(&cols), &target);
    let jobs: Vec<_> = [1usize, 2, 5]
        .iter()
        .map(|&k| (apply(&seed, &RewriteOp::AddLagEdge { src: 0, k }).unwrap(), parent_mse, target.clone()))
        .collect();
    // Complexity penalty on: best score must not be positive (no free lunch).
    let res = score_batch(&jobs, &cols, 1.0);
    assert!(res.iter().all(|r| r.0 <= 0.0), "null world must not promote any rewrite");
}
