//! The `apedp_kernel` Python extension module.
//!
//! This is the native first-line engine behind `apedp/rust_bridge.py`. The
//! bridge calls the eight entry points below and falls back to its NumPy
//! reference path when any of them raises, so the shapes here are a contract
//! with that file rather than an internal detail: `rust_bridge.py` unpacks
//! fixed-arity tuples in a specific order, and `tests/test_rust_bridge_dispatch.py`
//! pins those orders.
//!
//! Nothing in this module decides a verdict about a candidate. A compile
//! failure means "the fast path could not evaluate this expression", never
//! "this expression is bad"; the bridge turns it into a per-candidate fallback
//! and keeps going.

pub mod numeric;
pub mod ode;
pub mod parser;
pub mod units;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use ode::OdeSystem;
use parser::Compiled;
use units::UnitVector as UnitVec;

/// A compiled expression, opaque to Python.
#[pyclass(module = "apedp_kernel")]
#[derive(Clone)]
pub struct CompiledExpression {
    inner: Compiled,
}

#[pymethods]
impl CompiledExpression {
    fn __repr__(&self) -> String {
        format!(
            "CompiledExpression({:?}, variables={:?})",
            self.inner.source, self.inner.variables
        )
    }

    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    #[getter]
    fn variables(&self) -> Vec<String> {
        self.inner.variables.clone()
    }

    #[getter]
    fn num_variables(&self) -> usize {
        self.inner.num_vars()
    }
}

/// Copy a list of 1-D float64 arrays into owned vectors.
fn extract_columns(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<Vec<f64>>> {
    let list = obj
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err(format!("{what} must be a list of 1-D float64 arrays")))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let arr = item.extract::<PyReadonlyArray1<f64>>().map_err(|e| {
            PyValueError::new_err(format!("{what} must contain 1-D float64 arrays: {e}"))
        })?;
        if arr.as_array().len() == 0 {
            return Err(PyValueError::new_err(format!("{what} must not be empty")));
        }
        out.push(arr.as_array().iter().copied().collect());
    }
    Ok(out)
}

fn extract_vec1(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    let arr = obj.extract::<PyReadonlyArray1<f64>>().map_err(|e| {
        PyValueError::new_err(format!("{what} must be a 1-D float64 array: {e}"))
    })?;
    Ok(arr.as_array().iter().copied().collect())
}

#[pyfunction]
#[pyo3(signature = (expr, variables))]
fn compile(expr: &str, variables: Vec<String>) -> PyResult<CompiledExpression> {
    let inner = parser::compile(expr, &variables).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(CompiledExpression { inner })
}

#[pyfunction]
#[pyo3(signature = (compiled, data))]
fn evaluate_batch<'py>(
    py: Python<'py>,
    compiled: &CompiledExpression,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cols = extract_columns(data, "data")?;
    let values = compiled
        .inner
        .evaluate(&cols)
        .map_err(PyValueError::new_err)?;
    Ok(values.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn compute_metrics(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, f64, f64, bool)> {
    let yt = extract_vec1(y_true, "y_true")?;
    let yp = extract_vec1(y_pred, "y_pred")?;
    Ok(numeric::metrics(&yt, &yp))
}

/// Score compiled candidates against one target, in parallel.
///
/// Returns one row per candidate as
/// `(id, mse, mae, r2, max_residual, has_nan_or_inf)`, where `id` indexes the
/// *compiled* list, which is what the bridge maps back through its own
/// surviving-candidate index.
#[pyfunction]
#[pyo3(signature = (compiled, data, target))]
fn score_candidates_parallel(
    py: Python<'_>,
    compiled: Vec<CompiledExpression>,
    data: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<Vec<(usize, f64, f64, f64, f64, bool)>> {
    use rayon::prelude::*;

    let cols = extract_columns(data, "data")?;
    let target_vec = extract_vec1(target, "target")?;

    // Release the GIL for the fan-out: the whole point of this entry point is
    // using more than one core, and holding the GIL would serialise it.
    let rows = py.allow_threads(|| {
        compiled
            .par_iter()
            .enumerate()
            .map(|(id, c)| {
                // A candidate whose evaluation fails scores as non-finite
                // rather than propagating an error: the bridge would rather
                // score it than lose the whole batch, and it reports the
                // non-finite flag in the same row.
                match c.inner.evaluate(&cols) {
                    Ok(pred) => {
                        let (mse, mae, r2, max_res, has_nan) =
                            numeric::metrics(&target_vec, &pred);
                        (id, mse, mae, r2, max_res, has_nan)
                    }
                    Err(_) => (id, f64::NAN, f64::NAN, f64::NAN, f64::NAN, true),
                }
            })
            .collect()
    });

    Ok(rows)
}

/// Probe a compiled expression for poles across its first variable's bounds.
///
/// The remaining variables are held at the midpoint of their bounds, which is
/// the reference implementation's convention; matching it is what lets the
/// two engines be compared on the same function rather than two different ones.
///
/// Returns `(is_sound, has_pole, pole_coordinate, max_derivative, risk_type)`.
#[pyfunction]
#[pyo3(signature = (compiled, bounds, num_grid_points=2000))]
fn probe_singularity(
    compiled: &CompiledExpression,
    bounds: Vec<(f64, f64)>,
    num_grid_points: usize,
) -> PyResult<(bool, bool, Option<f64>, f64, String)> {
    if bounds.is_empty() {
        return Ok((true, false, None, 0.0, "NONE".to_string()));
    }
    if bounds.len() != compiled.inner.num_vars() {
        return Err(PyValueError::new_err(format!(
            "expression '{}' has {} variable(s) but {} bound pair(s) were supplied",
            compiled.inner.source,
            compiled.inner.num_vars(),
            bounds.len()
        )));
    }

    let (min_x, max_x) = bounds[0];
    let n = num_grid_points + 1;
    let step = (max_x - min_x) / num_grid_points.max(1) as f64;

    // numpy.linspace semantics, including the exact stop value in the last
    // slot: matching it is what makes the two engines report the same pole
    // coordinate rather than coordinates a grid step apart.
    let mut grid = vec![0.0f64; n];
    for (i, slot) in grid.iter_mut().enumerate() {
        *slot = min_x + step * i as f64;
    }
    grid[n - 1] = max_x;

    let mut cols = vec![grid.clone()];
    for (lo, hi) in &bounds[1..] {
        cols.push(vec![(lo + hi) / 2.0; n]);
    }

    let y = compiled
        .inner
        .evaluate(&cols)
        .map_err(|e| PyValueError::new_err(e))?;

    // A non-finite value or a magnitude blow-up is the pole signal.
    let mut has_pole = false;
    let mut pole_coord: Option<f64> = None;
    for (i, v) in y.iter().enumerate() {
        if !v.is_finite() || v.abs() > 1e6 {
            has_pole = true;
            pole_coord = Some(grid[i]);
            break;
        }
    }

    let diffs: Vec<f64> = y.windows(2).map(|w| (w[1] - w[0]).abs() / step).collect();
    let max_deriv = diffs
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0f64, f64::max);

    if !has_pole {
        for i in 0..diffs.len() {
            let d = diffs[i];
            if d.is_finite() && d > 1e4 {
                let sign_flip = y[i] * y[i + 1] < 0.0
                    && (y[i].abs() > 50.0 || y[i + 1].abs() > 50.0);
                if sign_flip || y[i].abs() > 500.0 || y[i + 1].abs() > 500.0 {
                    has_pole = true;
                    pole_coord = Some(grid[i]);
                    break;
                }
            }
        }
    }

    Ok((
        !has_pole,
        has_pole,
        pole_coord,
        max_deriv,
        if has_pole {
            "POLE_IN_DOMAIN".to_string()
        } else {
            "NONE".to_string()
        },
    ))
}

/// Conditioning of a design matrix given as a list of column arrays.
///
/// Returns `(condition_number, max_singular_value, min_singular_value,
/// effective_rank, is_well_conditioned, singular_values)`.
#[pyfunction]
#[pyo3(signature = (cols, rcond_threshold=1e-12))]
fn compute_matrix_conditioning(
    cols: &Bound<'_, PyAny>,
    rcond_threshold: f64,
) -> PyResult<(f64, f64, f64, usize, bool, Vec<f64>)> {
    let columns = extract_columns(cols, "cols")?;
    Ok(numeric::conditioning(&columns, rcond_threshold))
}

/// Regularised linear least squares over a list of basis column arrays.
///
/// Returns `(coefficients, mse, mae, r2, condition_number,
/// is_well_conditioned)`.
#[pyfunction]
#[pyo3(signature = (cols, target, lambda_reg=1e-9))]
fn solve_linear_coefficients(
    cols: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    lambda_reg: f64,
) -> PyResult<(Vec<f64>, f64, f64, f64, f64, bool)> {
    let columns = extract_columns(cols, "cols")?;
    let target_vec = extract_vec1(target, "target")?;
    numeric::linear_least_squares(&columns, &target_vec, lambda_reg)
        .map_err(PyValueError::new_err)
}

/// Identifiability boundary under machine precision.
///
/// Returns `(precision_type, machine_epsilon, theoretical_threshold,
/// empirical_transition_observed, resolved)`.
#[pyfunction]
#[pyo3(signature = (separation_delta, condition_number=1.0, precision_bits=64))]
fn evaluate_precision_boundary(
    separation_delta: f64,
    condition_number: f64,
    precision_bits: i64,
) -> (String, f64, f64, bool, bool) {
    numeric::precision_boundary(separation_delta, condition_number, precision_bits)
}

/// A 7-D SI unit vector.
#[pyclass(module = "apedp_kernel")]
#[derive(Clone)]
pub struct UnitVector {
    inner: UnitVec,
}

#[pymethods]
impl UnitVector {
    #[staticmethod]
    fn dimensionless() -> UnitVector {
        UnitVector {
            inner: UnitVec::dimensionless(),
        }
    }

    /// Resolve a unit name or a compound such as `"kg*m/s**2"`.
    #[staticmethod]
    fn parse(text: &str) -> PyResult<UnitVector> {
        let inner = UnitVec::parse(text).map_err(|e| PyValueError::new_err(e.0))?;
        Ok(UnitVector { inner })
    }

    fn as_tuple(&self) -> (i32, i32, i32, i32, i32, i32, i32) {
        self.inner.as_tuple()
    }

    fn is_dimensionless(&self) -> bool {
        self.inner.is_dimensionless()
    }

    fn mul(&self, other: &UnitVector) -> UnitVector {
        UnitVector {
            inner: self.inner.mul(&other.inner),
        }
    }

    fn div(&self, other: &UnitVector) -> UnitVector {
        UnitVector {
            inner: self.inner.div(&other.inner),
        }
    }

    fn pow(&self, n: i32) -> UnitVector {
        UnitVector {
            inner: self.inner.pow(n),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("UnitVector({:?})", self.inner.as_tuple())
    }

    fn __eq__(&self, other: &UnitVector) -> bool {
        self.inner == other.inner
    }
}

/// A compiled ODE system for high-performance simulation.
#[pyclass(module = "apedp_kernel")]
#[derive(Clone)]
pub struct CompiledOdeSystem {
    inner: OdeSystem,
}

#[pymethods]
impl CompiledOdeSystem {
    #[new]
    #[pyo3(signature = (equations, state_names, param_names, input_names=None))]
    fn new(
        equations: Vec<String>,
        state_names: Vec<String>,
        param_names: Vec<String>,
        input_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let inames = input_names.unwrap_or_default();
        let inner = OdeSystem::compile(&equations, &state_names, &param_names, &inames)
            .map_err(PyValueError::new_err)?;
        Ok(CompiledOdeSystem { inner })
    }

    #[getter]
    fn state_names(&self) -> Vec<String> {
        self.inner.state_names.clone()
    }

    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.inner.param_names.clone()
    }

    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.inner.input_names.clone()
    }

    #[getter]
    fn num_states(&self) -> usize {
        self.inner.num_states
    }

    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params
    }

    #[getter]
    fn num_inputs(&self) -> usize {
        self.inner.num_inputs
    }

    /// Simulate the ODE system using native Dormand-Prince 5(4) (RK45).
    ///
    /// Releases the Python GIL during integration steps.
    /// Returns `(success, detail, nfev, t, y)` where `y` is a 2D float64 array of shape `(num_states, n_eval)`.
    #[pyo3(signature = (t_span, x0, t_eval, param_values, input_values=None, rtol=1e-3, atol=1e-6, max_steps=100_000))]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        t_span: (f64, f64),
        x0: Vec<f64>,
        t_eval: Vec<f64>,
        param_values: Vec<f64>,
        input_values: Option<Vec<f64>>,
        rtol: f64,
        atol: f64,
        max_steps: usize,
    ) -> PyResult<(bool, String, usize, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
        let in_vals = input_values.unwrap_or_default();
        let res = py.allow_threads(|| {
            self.inner.integrate(t_span, &x0, &t_eval, &param_values, &in_vals, rtol, atol, max_steps)
        });

        let n_states = self.inner.num_states;
        let n_eval = res.t.len();
        let mut arr = Array2::<f64>::zeros((n_states, n_eval));
        if res.success && res.y.len() == n_states {
            for i in 0..n_states {
                for j in 0..n_eval {
                    arr[[i, j]] = res.y[i][j];
                }
            }
        }

        let py_t = res.t.into_pyarray_bound(py);
        let py_y = arr.into_pyarray_bound(py);
        Ok((res.success, res.detail, res.nfev, py_t, py_y))
    }
}

/// Compile and simulate an ODE system in a single native call.
#[pyfunction]
#[pyo3(signature = (
    equations,
    state_names,
    param_names,
    param_values,
    t_span,
    x0,
    t_eval,
    input_names=None,
    input_values=None,
    rtol=1e-3,
    atol=1e-6,
    max_steps=100_000
))]
fn simulate_ode_rk45<'py>(
    py: Python<'py>,
    equations: Vec<String>,
    state_names: Vec<String>,
    param_names: Vec<String>,
    param_values: Vec<f64>,
    t_span: (f64, f64),
    x0: Vec<f64>,
    t_eval: Vec<f64>,
    input_names: Option<Vec<String>>,
    input_values: Option<Vec<f64>>,
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> PyResult<(bool, String, usize, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let inames = input_names.unwrap_or_default();
    let ivals = input_values.unwrap_or_default();
    let sys = OdeSystem::compile(&equations, &state_names, &param_names, &inames)
        .map_err(PyValueError::new_err)?;
    let res = py.allow_threads(|| {
        sys.integrate(t_span, &x0, &t_eval, &param_values, &ivals, rtol, atol, max_steps)
    });

    let n_states = sys.num_states;
    let n_eval = res.t.len();
    let mut arr = Array2::<f64>::zeros((n_states, n_eval));
    if res.success && res.y.len() == n_states {
        for i in 0..n_states {
            for j in 0..n_eval {
                arr[[i, j]] = res.y[i][j];
            }
        }
    }

    let py_t = res.t.into_pyarray_bound(py);
    let py_y = arr.into_pyarray_bound(py);
    Ok((res.success, res.detail, res.nfev, py_t, py_y))
}

#[pymodule]
fn apedp_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(score_candidates_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(probe_singularity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_matrix_conditioning, m)?)?;
    m.add_function(wrap_pyfunction!(solve_linear_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_precision_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_ode_rk45, m)?)?;
    m.add_class::<CompiledExpression>()?;
    m.add_class::<CompiledOdeSystem>()?;
    m.add_class::<UnitVector>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}