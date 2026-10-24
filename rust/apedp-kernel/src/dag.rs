//! DAG (Directed Acyclic Graph) representation for expressions.
//!
//! Better than tree representation for expressions with common subexpressions
//! and for handling commutativity/associativity.

use std::collections::HashMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: usize,
    pub op: DagOp,
    pub children: Vec<usize>,
    pub value: Option<f64>,
    pub var_idx: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DagOp {
    Const,
    Var,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sin,
    Cos,
    Exp,
    Log,
    Sqrt,
    Abs,
    Neg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionDag {
    pub nodes: Vec<DagNode>,
    pub root: usize,
    pub variables: Vec<String>,
}

impl ExpressionDag {
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            nodes: Vec::new(),
            root: 0,
            variables,
        }
    }

    pub fn add_node(&mut self, op: DagOp, children: Vec<usize>, value: Option<f64>, var_idx: Option<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(DagNode { id, op, children, value, var_idx });
        id
    }

    pub fn to_string(&self, node_id: Option<usize>) -> String {
        let nid = node_id.unwrap_or(self.root);
        if nid >= self.nodes.len() {
            return "0".to_string();
        }
        let node = &self.nodes[nid];
        match node.op {
            DagOp::Const => {
                if let Some(v) = node.value {
                    format!("{}", v)
                } else {
                    "0".to_string()
                }
            }
            DagOp::Var => {
                if let Some(idx) = node.var_idx {
                    if idx < self.variables.len() {
                        self.variables[idx].clone()
                    } else {
                        format!("x{}", idx)
                    }
                } else {
                    "x".to_string()
                }
            }
            DagOp::Add => {
                let left = self.to_string(Some(node.children[0]));
                let right = self.to_string(Some(node.children[1]));
                format!("({} + {})", left, right)
            }
            DagOp::Sub => {
                let left = self.to_string(Some(node.children[0]));
                let right = self.to_string(Some(node.children[1]));
                format!("({} - {})", left, right)
            }
            DagOp::Mul => {
                let left = self.to_string(Some(node.children[0]));
                let right = self.to_string(Some(node.children[1]));
                format!("({} * {})", left, right)
            }
            DagOp::Div => {
                let left = self.to_string(Some(node.children[0]));
                let right = self.to_string(Some(node.children[1]));
                format!("({} / {})", left, right)
            }
            DagOp::Pow => {
                let left = self.to_string(Some(node.children[0]));
                let right = self.to_string(Some(node.children[1]));
                format!("({} ** {})", left, right)
            }
            DagOp::Sin => format!("sin({})", self.to_string(Some(node.children[0]))),
            DagOp::Cos => format!("cos({})", self.to_string(Some(node.children[0]))),
            DagOp::Exp => format!("exp({})", self.to_string(Some(node.children[0]))),
            DagOp::Log => format!("log({})", self.to_string(Some(node.children[0]))),
            DagOp::Sqrt => format!("sqrt({})", self.to_string(Some(node.children[0]))),
            DagOp::Abs => format!("abs({})", self.to_string(Some(node.children[0]))),
            DagOp::Neg => format!("(-{})", self.to_string(Some(node.children[0]))),
        }
    }

    pub fn canonicalize(&self) -> String {
        // Canonical form for DAG to enable deduplication
        self.to_canonical(self.root)
    }

    fn to_canonical(&self, node_id: usize) -> String {
        if node_id >= self.nodes.len() {
            return "0".to_string();
        }
        let node = &self.nodes[node_id];
        match node.op {
            DagOp::Const => {
                if let Some(v) = node.value {
                    // Normalize float representation
                    format!("{:.15e}", v)
                } else {
                    "0".to_string()
                }
            }
            DagOp::Var => {
                if let Some(idx) = node.var_idx {
                    format!("x{}", idx)
                } else {
                    "x".to_string()
                }
            }
            DagOp::Add | DagOp::Mul => {
                // Commutative - sort children
                let mut child_strs: Vec<String> = node.children.iter()
                    .map(|&c| self.to_canonical(c))
                    .collect();
                child_strs.sort();
                let op_str = if matches!(node.op, DagOp::Add) { "+" } else { "*" };
                format!("({})", child_strs.join(op_str))
            }
            DagOp::Sub => {
                let left = self.to_canonical(node.children[0]);
                let right = self.to_canonical(node.children[1]);
                format!("({}-{})", left, right)
            }
            DagOp::Div => {
                let left = self.to_canonical(node.children[0]);
                let right = self.to_canonical(node.children[1]);
                format!("({}/{})", left, right)
            }
            DagOp::Pow => {
                let left = self.to_canonical(node.children[0]);
                let right = self.to_canonical(node.children[1]);
                format!("({}^{})", left, right)
            }
            DagOp::Sin => format!("sin({})", self.to_canonical(node.children[0])),
            DagOp::Cos => format!("cos({})", self.to_canonical(node.children[0])),
            DagOp::Exp => format!("exp({})", self.to_canonical(node.children[0])),
            DagOp::Log => format!("log({})", self.to_canonical(node.children[0])),
            DagOp::Sqrt => format!("sqrt({})", self.to_canonical(node.children[0])),
            DagOp::Abs => format!("abs({})", self.to_canonical(node.children[0])),
            DagOp::Neg => format!("neg({})", self.to_canonical(node.children[0])),
        }
    }

    pub fn complexity(&self) -> usize {
        self.nodes.len()
    }

    pub fn depth(&self, node_id: Option<usize>) -> usize {
        let nid = node_id.unwrap_or(self.root);
        if nid >= self.nodes.len() {
            return 0;
        }
        let node = &self.nodes[nid];
        if node.children.is_empty() {
            return 1;
        }
        1 + node.children.iter().map(|&c| self.depth(Some(c))).max().unwrap_or(0)
    }
}

#[pyclass]
#[pyo3(name = "DagExpression")]
pub struct PyDagExpression {
    #[pyo3(get)]
    pub dag_json: String,
    #[pyo3(get)]
    pub expression: String,
    #[pyo3(get)]
    pub canonical: String,
    #[pyo3(get)]
    pub complexity: usize,
    #[pyo3(get)]
    pub depth: usize,
}

#[pymethods]
impl PyDagExpression {
    #[new]
    fn new(variables: Vec<String>) -> Self {
        let dag = ExpressionDag::new(variables.clone());
        Self {
            dag_json: serde_json::to_string(&dag).unwrap_or_default(),
            expression: "0".to_string(),
            canonical: "0".to_string(),
            complexity: 0,
            depth: 0,
        }
    }

    fn from_expression(&mut self, expr: &str, variables: Vec<String>) -> PyResult<()> {
        // Simple parser - in practice would use the existing parser
        let dag = ExpressionDag::new(variables.clone());
        self.dag_json = serde_json::to_string(&dag).unwrap_or_default();
        self.expression = expr.to_string();
        self.canonical = expr.to_string();
        self.complexity = 1;
        self.depth = 1;
        Ok(())
    }

    fn canonicalize(&self) -> String {
        self.canonical.clone()
    }
}
