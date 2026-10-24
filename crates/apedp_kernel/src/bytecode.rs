//! Flat bytecode instruction set and compiled expression representation.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum OpCode {
    LoadVar(u16),
    LoadConst(f64),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Pow,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Tan,
    Tanh,
    SanityCheck,
    Halt,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledExpression {
    pub opcodes: Vec<OpCode>,
    pub num_variables: usize,
    pub variable_names: Vec<String>,
}

impl CompiledExpression {
    pub fn new(opcodes: Vec<OpCode>, variable_names: Vec<String>) -> Self {
        let num_variables = variable_names.len();
        Self {
            opcodes,
            num_variables,
            variable_names,
        }
    }
}
