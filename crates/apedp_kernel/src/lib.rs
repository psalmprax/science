//! APEDP Kernel: High-Performance Rust Numerical & Validation Kernel.

pub mod ast;
pub mod bytecode;
pub mod conditioning;
pub mod dimensions;
pub mod metrics;
pub mod precision;
pub mod singularity;
pub mod varpro;
pub mod vm;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::ast::ExpressionCompiler;
use crate::bytecode::CompiledExpression;
use crate::conditioning::MatrixConditioner;
use crate::dimensions::UnitVector;
use crate::metrics::MetricsEngine;
use crate::precision::MultiPrecisionEngine;
use crate::singularity::SingularityProber;
use crate::varpro::VarProSolver;
use crate::vm::ExpressionVM;

#[pyclass(name = "UnitVector")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PyUnitVector {
    pub inner: UnitVector,
}

#[pymethods]
impl PyUnitVector {
    #[new]
    #[pyo3(signature = (length=0, mass=0, time=0, current=0, temp=0, amount=0, luminosity=0))]
    pub fn new(
        length: i8,
        mass: i8,
        time: i8,
        current: i8,
        temp: i8,
        amount: i8,
        luminosity: i8,
    ) -> Self {
        Self {
            inner: UnitVector {
                length,
                mass,
                time,
                current,
                temp,
                amount,
                luminosity,
            },
        }
    }

    #[staticmethod]
    pub fn dimensionless() -> Self {
        Self {
            inner: UnitVector::DIMENSIONLESS,
        }
    }

    #[staticmethod]
    pub fn parse(s: &str) -> PyResult<Self> {
        let inner = UnitVector::parse(s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { inner })
    }

    pub fn mul(&self, rhs: &PyUnitVector) -> Self {
        Self {
            inner: self.inner.mul(rhs.inner),
        }
    }

    pub fn div(&self, rhs: &PyUnitVector) -> Self {
        Self {
            inner: self.inner.div(rhs.inner),
        }
    }

    pub fn pow(&self, power: i8) -> Self {
        Self {
            inner: self.inner.pow(power),
        }
    }

    pub fn is_dimensionless(&self) -> bool {
        self.inner.is_dimensionless()
    }

    pub fn as_tuple(&self) -> (i8, i8, i8, i8, i8, i8, i8) {
        (
            self.inner.length,
            self.inner.mass,
            self.inner.time,
            self.inner.current,
            self.inner.temp,
            self.inner.amount,
            self.inner.luminosity,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "UnitVector(L={}, M={}, T={}, I={}, Theta={}, N={}, J={})",
            self.inner.length,
            self.inner.mass,
            self.inner.time,
            self.inner.current,
            self.inner.temp,
            self.inner.amount,
            self.inner.luminosity
        )
    }
}

#[pyclass(name = "CompiledExpression")]
#[derive(Clone)]
pub struct PyCompiledExpression {
    pub inner: CompiledExpression,
}

#[pymethods]
impl PyCompiledExpression {
    #[new]
    pub fn new(expr_str: &str, variable_names: Vec<String>) -> PyResult<Self> {
        let inner = ExpressionCompiler::compile(expr_str, &variable_names)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { inner })
    }

    pub fn num_variables(&self) -> usize {
        self.inner.num_variables
    }

    pub fn variable_names(&self) -> Vec<String> {
        self.inner.variable_names.clone()
    }
}

#[pyfunction]
pub fn compile(expr_str: &str, variable_names: Vec<String>) -> PyResult<PyCompiledExpression> {
    PyCompiledExpression::new(expr_str, variable_names)
}

#[pyfunction]
pub fn evaluate_batch<'py>(
    py: Python<'py>,
    expr: &PyCompiledExpression,
    data: Vec<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Data cannot be empty"));
    }
    let data_slices: Vec<&[f64]> = data
        .iter()
        .map(|arr| arr.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())))
        .collect::<PyResult<Vec<_>>>()?;

    let n = data_slices[0].len();
    for slice in &data_slices {
        if slice.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All variable arrays must have identical length",
            ));
        }
    }

    let mut out_vec = vec![0.0; n];

    let mut vm = ExpressionVM::new();
    vm.evaluate_batch(&expr.inner, &data_slices, &mut out_vec)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(out_vec.into_pyarray_bound(py))
}

#[pyfunction]
pub fn evaluate_scalar(
    expr: &PyCompiledExpression,
    vars: Vec<f64>,
) -> f64 {
    let vm = ExpressionVM::new();
    vm.evaluate_scalar(&expr.inner, &vars)
}

#[pyfunction]
pub fn compute_metrics(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
) -> PyResult<(f64, f64, f64, f64, bool)> {
    let yt = y_true.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let yp = y_pred.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let res = MetricsEngine::compute_metrics(yt, yp);
    Ok((res.mse, res.mae, res.r_squared, res.max_residual, res.has_nan_or_inf))
}

#[pyfunction]
pub fn probe_singularity(
    expr: &PyCompiledExpression,
    domain_bounds: Vec<(f64, f64)>,
    num_grid_points: usize,
) -> PyResult<(bool, bool, Option<f64>, f64, String)> {
    let res = SingularityProber::probe_expression_grid(&expr.inner, &domain_bounds, num_grid_points);
    Ok((res.is_sound, res.has_pole, res.pole_coordinate, res.max_derivative, res.risk_type))
}

#[pyfunction]
pub fn score_candidates_parallel<'py>(
    py: Python<'py>,
    candidates: Vec<PyRef<'py, PyCompiledExpression>>,
    data: Vec<PyReadonlyArray1<'py, f64>>,
    target: PyReadonlyArray1<'py, f64>,
) -> PyResult<Vec<(usize, f64, f64, f64, f64, bool)>> {
    let target_slice = target
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let data_slices: Vec<&[f64]> = data
        .iter()
        .map(|arr| arr.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())))
        .collect::<PyResult<Vec<_>>>()?;

    let raw_candidates: Vec<CompiledExpression> = candidates.iter().map(|c| c.inner.clone()).collect();

    let scored = py.allow_threads(|| {
        MetricsEngine::score_candidates_parallel(&raw_candidates, &data_slices, target_slice)
    });

    let results = scored
        .into_iter()
        .map(|sc| {
            (
                sc.candidate_id,
                sc.metrics.mse,
                sc.metrics.mae,
                sc.metrics.r_squared,
                sc.metrics.max_residual,
                sc.metrics.has_nan_or_inf,
            )
        })
        .collect();

    Ok(results)
}

#[pyfunction]
pub fn compute_matrix_conditioning<'py>(
    cols: Vec<PyReadonlyArray1<'py, f64>>,
    rcond_threshold: f64,
) -> PyResult<(f64, f64, f64, usize, bool, Vec<f64>)> {
    let col_slices: Vec<&[f64]> = cols
        .iter()
        .map(|arr| arr.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())))
        .collect::<PyResult<Vec<_>>>()?;

    let rep = MatrixConditioner::compute_conditioning(&col_slices, rcond_threshold)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok((
        rep.condition_number,
        rep.max_singular_value,
        rep.min_singular_value,
        rep.effective_rank,
        rep.is_well_conditioned,
        rep.singular_values,
    ))
}

#[pyfunction]
pub fn solve_linear_coefficients<'py>(
    basis_cols: Vec<PyReadonlyArray1<'py, f64>>,
    target: PyReadonlyArray1<'py, f64>,
    lambda: f64,
) -> PyResult<(Vec<f64>, f64, f64, f64, f64, bool)> {
    let target_slice = target
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let col_slices: Vec<&[f64]> = basis_cols
        .iter()
        .map(|arr| arr.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())))
        .collect::<PyResult<Vec<_>>>()?;

    let res = VarProSolver::solve_linear_coefficients(&col_slices, target_slice, lambda)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok((
        res.coefficients,
        res.mse,
        res.mae,
        res.r_squared,
        res.condition_number,
        res.is_well_conditioned,
    ))
}

#[pyfunction]
pub fn evaluate_precision_boundary(
    separation_delta: f64,
    condition_number: f64,
    precision_bits: usize,
) -> PyResult<(String, f64, f64, bool, bool)> {
    let rep = MultiPrecisionEngine::evaluate_identifiability(separation_delta, condition_number, precision_bits);
    Ok((
        rep.precision_type,
        rep.machine_epsilon,
        rep.theoretical_threshold,
        rep.empirical_transition_observed,
        rep.resolved,
    ))
}

#[pymodule]
fn apedp_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompiledExpression>()?;
    m.add_class::<PyUnitVector>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(compute_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(score_candidates_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(probe_singularity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_matrix_conditioning, m)?)?;
    m.add_function(wrap_pyfunction!(solve_linear_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_precision_boundary, m)?)?;
    Ok(())
}
