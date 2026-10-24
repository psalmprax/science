//! Neural Guided Tree Search (NGTS) for symbolic regression.
//!
//! Improved search using tree search with learned priors and value functions.

use std::collections::HashMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchNode {
    pub id: usize,
    pub parent: Option<usize>,
    pub expression: String,
    pub visits: usize,
    pub value_sum: f64,
    pub prior: f64,
    pub children: Vec<usize>,
    pub depth: usize,
}

#[derive(Debug, Clone)]
pub struct NGTSConfig {
    pub exploration_constant: f64,
    pub num_simulations: usize,
    pub max_depth: usize,
    pub temperature: f64,
    pub use_uct: bool,
    pub use_puct: bool,
    pub alpha: f64,
}

impl Default for NGTSConfig {
    fn default() -> Self {
        Self {
            exploration_constant: 1.414, // sqrt(2)
            num_simulations: 1000,
            max_depth: 10,
            temperature: 1.0,
            use_uct: true,
            use_puct: false,
            alpha: 0.3,
        }
    }
}

#[derive(Debug)]
pub struct NGTS {
    pub config: NGTSConfig,
    pub root: SearchNode,
    pub nodes: HashMap<usize, SearchNode>,
    pub node_counter: usize,
}

impl NGTS {
    pub fn new(root_expr: String) -> Self {
        let root = SearchNode {
            id: 0,
            parent: None,
            expression: root_expr,
            visits: 0,
            value_sum: 0.0,
            prior: 1.0,
            children: Vec::new(),
            depth: 0,
        };
        let mut nodes = HashMap::new();
        nodes.insert(0, root.clone());
        Self {
            config: NGTSConfig::default(),
            root,
            nodes,
            node_counter: 1,
        }
    }

    pub fn uct_value(&self, node_id: usize, parent_visits: usize) -> f64 {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return f64::NEG_INFINITY,
        };
        if node.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = node.value_sum / node.visits as f64;
        let exploration = self.config.exploration_constant * 
            (parent_visits as f64).sqrt() / (1.0 + node.visits as f64);
        exploitation + exploration
    }

    pub fn puct_value(&self, node_id: usize, parent_visits: usize) -> f64 {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return f64::NEG_INFINITY,
        };
        let exploitation = if node.visits == 0 { 0.0f64 } else { node.value_sum / node.visits as f64 };
        let exploration = self.config.exploration_constant * node.prior * 
            (parent_visits as f64).sqrt() / (1.0 + node.visits as f64);
        exploitation + exploration
    }

    pub fn select(&self, node_id: usize) -> usize {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n.clone(),
            None => return node_id,
        };
        if node.children.is_empty() || node.visits == 0 {
            return node_id;
        }
        let mut best_child = node.children[0];
        let mut best_value = if self.config.use_puct { 
            self.puct_value(best_child, node.visits) 
        } else { 
            self.uct_value(best_child, node.visits) 
        };
        for &child_id in &node.children[1..] {
            let val = if self.config.use_puct {
                self.puct_value(child_id, node.visits)
            } else {
                self.uct_value(child_id, node.visits)
            };
            if val > best_value {
                best_value = val;
                best_child = child_id;
            }
        }
        if self.nodes.get(&best_child).map_or(false, |c| c.depth < self.config.max_depth) {
            self.select(best_child)
        } else {
            best_child
        }
    }

    pub fn expand(&mut self, node_id: usize, actions: Vec<(String, f64)>) {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n.clone(),
            None => return,
        };
        if node.depth >= self.config.max_depth {
            return;
        }
        for (expr, prior) in actions {
            let child_id = self.node_counter;
            self.node_counter += 1;
            let child = SearchNode {
                id: child_id,
                parent: Some(node_id),
                expression: expr,
                visits: 0,
                value_sum: 0.0,
                prior,
                children: Vec::new(),
                depth: node.depth + 1,
            };
            self.nodes.insert(child_id, child);
            if let Some(parent_node) = self.nodes.get_mut(&node_id) {
                parent_node.children.push(child_id);
            }
        }
    }

    pub fn backpropagate(&mut self, mut node_id: usize, value: f64) {
        loop {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.visits += 1;
                node.value_sum += value;
            }
            match self.nodes.get(&node_id).and_then(|n| n.parent) {
                Some(parent_id) => node_id = parent_id,
                None => break,
            }
        }
    }

    pub fn best_child(&self) -> Option<SearchNode> {
        let root_children = self.nodes.get(&0)?.children.clone();
        if root_children.is_empty() {
            return self.nodes.get(&0).cloned();
        }
        let mut best_id = root_children[0];
        let mut best_score = self.nodes.get(&best_id).map_or(0.0, |n| {
            if n.visits == 0 { -1e18 } else { n.value_sum / n.visits as f64 }
        });
        for &child_id in &root_children[1..] {
            if let Some(child) = self.nodes.get(&child_id) {
                let score = if child.visits == 0 { -1e18f64 } else { child.value_sum / child.visits as f64 };
                if score > best_score {
                    best_score = score;
                    best_id = child_id;
                }
            }
        }
        self.nodes.get(&best_id).cloned()
    }

    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("total_nodes".to_string(), self.nodes.len() as f64);
        stats.insert("root_visits".to_string(), self.nodes.get(&0).map_or(0.0, |n| n.visits as f64));
        stats.insert("num_children".to_string(), self.nodes.get(&0).map_or(0.0, |n| n.children.len() as f64));
        stats
    }
}

#[pyclass]
#[pyo3(name = "NGTSSearch")]
pub struct PyNGTSSearch {
    ngts: NGTS,
}

#[pymethods]
impl PyNGTSSearch {
    #[new]
    fn new(root_expr: String) -> Self {
        Self {
            ngts: NGTS::new(root_expr),
        }
    }

    fn set_config(&mut self, exploration_constant: f64, num_simulations: usize, max_depth: usize) {
        self.ngts.config.exploration_constant = exploration_constant;
        self.ngts.config.num_simulations = num_simulations;
        self.ngts.config.max_depth = max_depth;
    }

    fn select(&self) -> usize {
        self.ngts.select(0)
    }

    fn expand(&mut self, node_id: usize, actions_json: &str) -> PyResult<()> {
        // actions_json format: [["expr1", prior1], ["expr2", prior2]]
        if let Ok(actions) = serde_json::from_str::<Vec<Vec<serde_json::Value>>>(actions_json) {
            let mut action_list = Vec::new();
            for a in actions {
                if a.len() >= 2 {
                    if let Some(expr) = a[0].as_str() {
                        if let Some(prior) = a[1].as_f64() {
                            action_list.push((expr.to_string(), prior));
                        }
                    }
                }
            }
            self.ngts.expand(node_id, action_list);
        }
        Ok(())
    }

    fn backpropagate(&mut self, node_id: usize, value: f64) {
        self.ngts.backpropagate(node_id, value);
    }

    fn best_expression(&self) -> Option<String> {
        self.ngts.best_child().map(|n| n.expression)
    }

    fn best_score(&self) -> f64 {
        if let Some(best) = self.ngts.best_child() {
            if best.visits == 0 { 0.0 } else { best.value_sum / best.visits as f64 }
        } else {
            0.0
        }
    }

    fn statistics(&self) -> String {
        let stats = self.ngts.get_statistics();
        serde_json::to_string(&stats).unwrap_or_default()
    }

    fn run_simulations(&mut self, num_sims: usize) -> usize {
        let mut completed = 0;
        for _ in 0..num_sims {
            let selected = self.ngts.select(0);
            // Expand if needed
            if let Some(node) = self.ngts.nodes.get(&selected) {
                if node.children.is_empty() && node.depth < self.ngts.config.max_depth {
                    // Add some default actions
                    let actions = vec![
                        (format!("{} + 1", node.expression), 0.1),
                        (format!("{} * x0", node.expression), 0.1),
                        (format!("sin({})", node.expression), 0.05),
                    ];
                    self.ngts.expand(selected, actions);
                    if let Some(child) = self.ngts.nodes.get(&selected).and_then(|n| n.children.first().cloned()) {
                        // Simulate
                        let value = 0.5 + (rand_simple() * 0.5);
                        self.ngts.backpropagate(child, value);
                        completed += 1;
                        continue;
                    }
                }
            }
            // Just backpropagate
            let value = rand_simple() * 0.5;
            self.ngts.backpropagate(selected, value);
            completed += 1;
        }
        completed
    }
}

fn rand_simple() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    (hasher.finish() as f64) / (u64::MAX as f64)
}
