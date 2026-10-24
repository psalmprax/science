//! High-performance stack-based Expression Virtual Machine.

use crate::bytecode::{CompiledExpression, OpCode};

#[derive(Default, Clone, Copy)]
pub struct ExpressionVM;

impl ExpressionVM {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }

    /// Evaluates compiled expression on a single scalar point.
    #[inline]
    pub fn evaluate_scalar(
        &self,
        expr: &CompiledExpression,
        vars: &[f64],
    ) -> f64 {
        let mut stack: [f64; 32] = [0.0; 32];
        let mut sp = 0;

        for &op in &expr.opcodes {
            match op {
                OpCode::LoadVar(idx) => {
                    let val = vars.get(idx as usize).copied().unwrap_or(f64::NAN);
                    stack[sp] = val;
                    sp += 1;
                }
                OpCode::LoadConst(c) => {
                    stack[sp] = c;
                    sp += 1;
                }
                OpCode::Add => {
                    if sp < 2 { return f64::NAN; }
                    sp -= 1;
                    let b = stack[sp];
                    stack[sp - 1] += b;
                }
                OpCode::Sub => {
                    if sp < 2 { return f64::NAN; }
                    sp -= 1;
                    let b = stack[sp];
                    stack[sp - 1] -= b;
                }
                OpCode::Mul => {
                    if sp < 2 { return f64::NAN; }
                    sp -= 1;
                    let b = stack[sp];
                    stack[sp - 1] *= b;
                }
                OpCode::Div => {
                    if sp < 2 { return f64::NAN; }
                    sp -= 1;
                    let b = stack[sp];
                    stack[sp - 1] /= b;
                }
                OpCode::Neg => {
                    if sp < 1 { return f64::NAN; }
                    stack[sp - 1] = -stack[sp - 1];
                }
                OpCode::Pow => {
                    if sp < 2 { return f64::NAN; }
                    sp -= 1;
                    let b = stack[sp];
                    let a = stack[sp - 1];
                    stack[sp - 1] = a.powf(b);
                }
                OpCode::Sqrt => {
                    if sp < 1 { return f64::NAN; }
                    let a = stack[sp - 1];
                    stack[sp - 1] = if a >= 0.0 { a.sqrt() } else { f64::NAN };
                }
                OpCode::Exp => {
                    if sp < 1 { return f64::NAN; }
                    stack[sp - 1] = stack[sp - 1].exp();
                }
                OpCode::Ln => {
                    if sp < 1 { return f64::NAN; }
                    let a = stack[sp - 1];
                    stack[sp - 1] = if a > 0.0 { a.ln() } else { f64::NAN };
                }
                OpCode::Sin => {
                    if sp < 1 { return f64::NAN; }
                    stack[sp - 1] = stack[sp - 1].sin();
                }
                OpCode::Cos => {
                    if sp < 1 { return f64::NAN; }
                    stack[sp - 1] = stack[sp - 1].cos();
                }
                OpCode::Tan => {
                    if sp < 1 { return f64::NAN; }
                    stack[sp - 1] = stack[sp - 1].tan();
                }
                OpCode::Tanh => {
                    if sp < 1 { return f64::NAN; }
                    stack[sp - 1] = stack[sp - 1].tanh();
                }
                OpCode::SanityCheck => {
                    if sp < 1 || !stack[sp - 1].is_finite() {
                        return f64::NAN;
                    }
                }
                OpCode::Halt => break,
            }
        }

        if sp > 0 {
            stack[0]
        } else {
            f64::NAN
        }
    }

    /// Evaluates compiled expression over an array of observations using Rayon parallel chunks.
    pub fn evaluate_batch(
        &mut self,
        expr: &CompiledExpression,
        data: &[&[f64]], // one slice per variable
        output: &mut [f64],
    ) -> Result<(), String> {
        let n = output.len();
        if n == 0 {
            return Ok(());
        }

        for (var_idx, slice) in data.iter().enumerate() {
            if slice.len() != n {
                return Err(format!(
                    "Variable slice {} length {} != output length {}",
                    var_idx,
                    slice.len(),
                    n
                ));
            }
        }

        let num_vars = data.len();

        if n < 4096 {
            // Sequential fast-path for small batches
            if num_vars <= 8 {
                let mut vars = [0.0; 8];
                for i in 0..n {
                    for v in 0..num_vars {
                        vars[v] = data[v][i];
                    }
                    output[i] = self.evaluate_scalar(expr, &vars[..num_vars]);
                }
            } else {
                let mut vars = vec![0.0; num_vars];
                for i in 0..n {
                    for v in 0..num_vars {
                        vars[v] = data[v][i];
                    }
                    output[i] = self.evaluate_scalar(expr, &vars);
                }
            }
        } else {
            // Parallel chunked execution via Rayon for large datasets
            use rayon::prelude::*;
            let chunk_size = 4096;

            output
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let offset = chunk_idx * chunk_size;
                    let m = chunk.len();
                    let vm = ExpressionVM::new();

                    if num_vars <= 8 {
                        let mut vars = [0.0; 8];
                        for j in 0..m {
                            let idx = offset + j;
                            for v in 0..num_vars {
                                vars[v] = data[v][idx];
                            }
                            chunk[j] = vm.evaluate_scalar(expr, &vars[..num_vars]);
                        }
                    } else {
                        let mut vars = vec![0.0; num_vars];
                        for j in 0..m {
                            let idx = offset + j;
                            for v in 0..num_vars {
                                vars[v] = data[v][idx];
                            }
                            chunk[j] = vm.evaluate_scalar(expr, &vars);
                        }
                    }
                });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ExpressionCompiler;

    #[test]
    fn test_scalar_evaluation() {
        let expr = ExpressionCompiler::compile("x0 * 2.0 + 3.0", &["x0".to_string()]).unwrap();
        let vm = ExpressionVM::new();
        let res = vm.evaluate_scalar(&expr, &[4.0]);
        assert_eq!(res, 11.0);
    }

    #[test]
    fn test_batch_evaluation() {
        let expr = ExpressionCompiler::compile("x0**2 + 1.0", &["x0".to_string()]).unwrap();
        let mut vm = ExpressionVM::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        vm.evaluate_batch(&expr, &[&data], &mut out).unwrap();
        assert_eq!(out, vec![2.0, 5.0, 10.0, 17.0]);
    }

    #[test]
    fn test_complex_expression() {
        let expr = ExpressionCompiler::compile("sin(x0) + exp(x1)", &["x0".to_string(), "x1".to_string()]).unwrap();
        let vm = ExpressionVM::new();
        let res = vm.evaluate_scalar(&expr, &[0.0, 0.0]);
        assert!((res - 1.0).abs() < 1e-10);
    }
}
