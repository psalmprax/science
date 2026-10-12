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
pub mod genetic;
pub mod rime;

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

/// Native Genetic Algorithm search over dataset: evolves equation trees in parallel.
/// Returns list of (formula_str, mse, complexity, scale_a, bias_b) and total evaluations.
#[pyfunction]
#[pyo3(signature = (
    data,
    variable_names,
    target,
    population_size=100,
    generations=25,
    islands=4,
    migration_interval=5,
    max_depth=4,
    tournament_size=4,
    mutation_rate=0.35,
    crossover_rate=0.70,
    allow_trans=true,
    seed=42
))]
fn evolve_native<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    variable_names: Vec<String>,
    target: &Bound<'py, PyAny>,
    population_size: usize,
    generations: usize,
    islands: usize,
    migration_interval: usize,
    max_depth: usize,
    tournament_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    allow_trans: bool,
    seed: u64,
) -> PyResult<(Vec<(String, f64, usize, f64, f64)>, usize)> {
    let columns = extract_columns(data, "data")?;
    let target_vec = extract_vec1(target, "target")?;
    let config = genetic::NativeSearchConfig {
        population_size,
        generations,
        islands,
        migration_interval,
        max_depth,
        tournament_size,
        mutation_rate,
        crossover_rate,
        allow_trans,
        seed,
    };

    let (pareto, total_evals) = py.allow_threads(|| {
        genetic::run_native_evolutionary_search(&columns, &target_vec, &variable_names, &config)
    });

    let res: Vec<(String, f64, usize, f64, f64)> = pareto
        .into_iter()
        .map(|p| (p.formula, p.mse, p.complexity, p.scale_a, p.bias_b))
        .collect();

    Ok((res, total_evals))
}

/// Evaluate one graph against dev data on its **fitted** readout.
///
/// Returns (graph_id, score, mse, skipped, scale, intercept, r2, complexity).
/// `score` is the spec §8 quantity: ΔMSE vs `parent_mse` minus the complexity
/// charge (edges + the two readout parameters).
#[pyfunction]
#[pyo3(signature = (graph_json, data, target, parent_mse, lambda_c, ridge=1e-9))]
fn rime_evaluate_json(
    py: Python<'_>,
    graph_json: &str,
    data: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    parent_mse: f64,
    lambda_c: f64,
    ridge: f64,
) -> PyResult<(String, f64, f64, usize, f64, f64, f64, f64)> {
    let graph: rime::graph::Graph = serde_json::from_str(graph_json)
        .map_err(|e| PyValueError::new_err(format!("rime: bad graph JSON: {}", e)))?;
    graph
        .validate()
        .map_err(|e| PyValueError::new_err(format!("rime: invalid graph: {}", e)))?;
    let columns = extract_columns(data, "data")?;
    let target_vec = extract_vec1(target, "target")?;
    // Single-candidate evaluation: no selection charge is added here, because
    // the caller knows how many candidates it looked at; the search sets it.
    let score_cfg = rime::eval::ScoreConfig::new(lambda_c, ridge);
    let s = py.allow_threads(|| {
        rime::eval::score_candidate_fitted(&graph, parent_mse, &columns, &target_vec, &score_cfg)
    });
    Ok((graph.id(), s.score, s.mse, s.skipped, s.scale, s.intercept, s.r2, s.complexity))
}

/// Evaluate a graph's output series (spec §12 `evaluate(graph, data)`).
///
/// Proposers need the signal itself, not only its score, so the fitted readout
/// can be applied by the caller. An invalid graph raises instead of returning
/// NaNs: a non-topological arena is a bug in the caller, not a data gap.
#[pyfunction]
fn rime_evaluate_series_json(
    py: Python<'_>,
    graph_json: &str,
    data: &Bound<'_, PyAny>,
) -> PyResult<Vec<f64>> {
    let graph: rime::graph::Graph = serde_json::from_str(graph_json)
        .map_err(|e| PyValueError::new_err(format!("rime: bad graph JSON: {}", e)))?;
    let columns = extract_columns(data, "data")?;
    py.allow_threads(|| graph.evaluate_checked(&columns))
        .map_err(|e| PyValueError::new_err(format!("rime: invalid graph: {}", e)))
}

/// Multi-environment RIME search (spec §8 with per-environment scoring).
///
/// `envs` is a JSON array of `{"cols": [[...]], "target": [...]}` objects, one
/// per environment. `config_json` additionally accepts `"aggregation"`
/// (`"mean"` | `"median_worst_gated"`) and `"gamma_gate"`. The aggregation rule
/// is part of the frozen protocol and is reported back in the packet.
#[pyfunction]
#[pyo3(signature = (envs_json, config_json, variable_names=None))]
fn rime_search_envs_json(
    py: Python<'_>,
    envs_json: &str,
    config_json: &str,
    variable_names: Option<Vec<String>>,
) -> PyResult<String> {
    let raw: serde_json::Value = serde_json::from_str(envs_json)
        .map_err(|e| PyValueError::new_err(format!("rime: bad envs JSON: {}", e)))?;
    let arr = raw
        .as_array()
        .ok_or_else(|| PyValueError::new_err("rime: envs must be a JSON array"))?;
    if arr.is_empty() {
        return Err(PyValueError::new_err("rime: at least one environment is required"));
    }
    let mut envs = Vec::with_capacity(arr.len());
    for (k, e) in arr.iter().enumerate() {
        let cols_v = e
            .get("cols")
            .ok_or_else(|| PyValueError::new_err(format!("rime: env {k} missing cols")))?;
        let tgt_v = e
            .get("target")
            .ok_or_else(|| PyValueError::new_err(format!("rime: env {k} missing target")))?;
        let mut cols: Vec<Vec<f64>> = Vec::new();
        let arr_cols = cols_v.as_array().ok_or_else(|| {
            PyValueError::new_err(format!("rime: env {k} cols must be an array"))
        })?;
        for c in arr_cols {
            let vals: Vec<f64> = c
                .as_array()
                .ok_or_else(|| {
                    PyValueError::new_err(format!("rime: env {k} cols entries must be arrays"))
                })?
                .iter()
                .map(|v| {
                    v.as_f64().ok_or_else(|| {
                        PyValueError::new_err(format!("rime: env {k} cols must be numeric"))
                    })
                })
                .collect::<PyResult<Vec<f64>>>()?;
            cols.push(vals);
        }
        let target: Vec<f64> = tgt_v
            .as_array()
            .ok_or_else(|| PyValueError::new_err(format!("rime: env {k} target must be an array")))?
            .iter()
            .map(|v| {
                v.as_f64().ok_or_else(|| {
                    PyValueError::new_err(format!("rime: env {k} target must be numeric"))
                })
            })
            .collect::<PyResult<Vec<f64>>>()?;
        envs.push(rime::search::EnvData { cols, target });
    }
    let cfg_raw: serde_json::Value = if config_json.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("rime: bad config JSON: {}", e)))?
    };
    let mut cfg = rime::search::SearchConfig::default();
    if let Some(v) = cfg_raw.get("max_depth").and_then(|v| v.as_u64()) {
        cfg.max_depth = v as usize;
    }
    if let Some(v) = cfg_raw.get("beam").and_then(|v| v.as_u64()) {
        cfg.beam = (v as usize).max(1);
    }
    if let Some(v) = cfg_raw.get("budget").and_then(|v| v.as_u64()) {
        cfg.budget = (v as usize).max(1);
    }
    if let Some(v) = cfg_raw.get("seed").and_then(|v| v.as_u64()) {
        cfg.seed = v;
    }
    if let Some(v) = cfg_raw.get("ridge").and_then(|v| v.as_f64()) {
        cfg.ridge = v;
    }
    if let Some(v) = cfg_raw.get("gamma").and_then(|v| v.as_f64()) {
        // Σ over envs of their finite rows, calibrated on the pooled scale.
        let mut all: Vec<f64> = Vec::new();
        for e in &envs {
            all.extend(e.target.iter().copied().filter(|t| t.is_finite()));
        }
        let n_pooled = all.len().max(1);
        cfg.lambda_c = v * rime::search::target_variance(&all) / n_pooled as f64;
    } else if let Some(v) = cfg_raw.get("lambda_c").and_then(|v| v.as_f64()) {
        cfg.lambda_c = v;
    }
    let agg = match cfg_raw.get("aggregation").and_then(|v| v.as_str()) {
        Some("median_worst_gated") => rime::search::EnvAggregation::MedianWorstGated,
        _ => rime::search::EnvAggregation::MeanPerEnv,
    };
    let gamma_gate = cfg_raw.get("gamma_gate").and_then(|v| v.as_f64()).unwrap_or(1.5);

    let seeds = rime::seeds::small_forms(envs.iter().map(|e| e.cols.len()).max().unwrap_or(1));
    let outcome =
        py.allow_threads(|| rime::search::search_envs(&seeds, &envs, &cfg, agg, gamma_gate));
    let mut packet = outcome.to_json();
    if let Some(names) = variable_names.filter(|n| !n.is_empty()) {
        if let Some(list) = packet.get_mut("proposals").and_then(|v| v.as_array_mut()) {
            for p in list.iter_mut() {
                if let Some(eq) = p.get("equation").and_then(|v| v.as_str()).map(|s| s.to_string()) {
                    let mut e = eq;
                    for (i, name) in names.iter().enumerate() {
                        e = e.replace(&format!("v{}", i), name);
                    }
                    p["equation"] = serde_json::Value::String(e);
                }
            }
        }
    }
    serde_json::to_string(&packet)
        .map_err(|e| PyValueError::new_err(format!("rime: serialize failed: {}", e)))
}

/// One RIME search (spec §8/§12) over the frozen rewrites.
///
/// `config_json` fields: max_depth, beam, budget, lambda_c, ridge, seed, and
/// optionally `rel_penalty` to scale lambda_c to the target variance. Returns
/// the full search packet as JSON (proposals + dev evidence).
#[pyfunction]
#[pyo3(signature = (data, target, config_json, variable_names=None))]
fn rime_search_json(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    config_json: &str,
    variable_names: Option<Vec<String>>,
) -> PyResult<String> {
    let columns = extract_columns(data, "data")?;
    let target_vec = extract_vec1(target, "target")?;
    let requested: Vec<String> = variable_names.unwrap_or_default();

    let mut cfg = rime::search::SearchConfig::default();
    let raw: serde_json::Value = if config_json.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("rime: bad config JSON: {}", e)))?
    };
    if let Some(v) = raw.get("max_depth").and_then(|v| v.as_u64()) {
        cfg.max_depth = v as usize;
    }
    if let Some(v) = raw.get("beam").and_then(|v| v.as_u64()) {
        cfg.beam = (v as usize).max(1);
    }
    if let Some(v) = raw.get("budget").and_then(|v| v.as_u64()) {
        cfg.budget = (v as usize).max(1);
    }
    if let Some(v) = raw.get("seed").and_then(|v| v.as_u64()) {
        cfg.seed = v;
    }
    if let Some(v) = raw.get("ridge").and_then(|v| v.as_f64()) {
        cfg.ridge = v;
    }
    if let Some(v) = raw.get("gamma").and_then(|v| v.as_f64()) {
        // Sample-size-aware calibration: gamma * var(y) / n_dev.
        let n_dev = target_vec.iter().filter(|t| t.is_finite()).count();
        cfg.lambda_c = v * rime::search::target_variance(&target_vec) / n_dev.max(1) as f64;
    } else if let Some(v) = raw.get("rel_penalty").and_then(|v| v.as_f64()) {
        cfg.lambda_c = v * rime::search::target_variance(&target_vec);
    } else if let Some(v) = raw.get("lambda_c").and_then(|v| v.as_f64()) {
        cfg.lambda_c = v;
    }

    let seeds = rime::seeds::small_forms(columns.len());
    let outcome = py.allow_threads(|| rime::search::search(&seeds, &columns, &target_vec, &cfg));
    let mut packet = outcome.to_json();
    // Name the variables in every equation so the referee packet is readable.
    if !requested.is_empty() {
        let names: Vec<&str> = requested.iter().map(|s| s.as_str()).collect();
        if let Some(list) = packet.get_mut("proposals").and_then(|v| v.as_array_mut()) {
            for p in list.iter_mut() {
                if let Some(eq) = p.get("equation").and_then(|v| v.as_str()).map(|s| s.to_string()) {
                    let mut e = eq;
                    for (i, name) in names.iter().enumerate() {
                        e = e.replace(&format!("v{}", i), name);
                    }
                    p["equation"] = serde_json::Value::String(e);
                }
            }
        }
    }
    serde_json::to_string(&packet)
        .map_err(|e| PyValueError::new_err(format!("rime: serialize failed: {}", e)))
}

#[pyfunction]
fn rime_rewrite_json(graph_json: &str, op_json: &str) -> PyResult<String> {
    let graph: rime::graph::Graph = serde_json::from_str(graph_json)
        .map_err(|e| PyValueError::new_err(format!("rime: bad graph JSON: {}", e)))?;
    let op: rime::rewrite::RewriteOp = serde_json::from_str(op_json)
        .map_err(|e| PyValueError::new_err(format!("rime: bad op JSON: {}", e)))?;
    match rime::rewrite::apply(&graph, &op) {
        Some(h) => serde_json::to_string(&h)
            .map_err(|e| PyValueError::new_err(format!("rime: serialize failed: {}", e))),
        None => Err(PyValueError::new_err("rime: rewrite rejected (bad index or lag)")),
    }
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
    m.add_function(wrap_pyfunction!(evolve_native, m)?)?;
    m.add_class::<CompiledExpression>()?;
    m.add_class::<CompiledOdeSystem>()?;
    m.add_function(wrap_pyfunction!(rime_evaluate_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_rewrite_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_search_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_search_envs_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_evaluate_series_json, m)?)?;
    m.add_class::<UnitVector>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}