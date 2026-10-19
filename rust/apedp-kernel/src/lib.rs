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

pub mod data;
pub mod numeric;
pub mod video;
pub mod flow_head;
pub mod ode;
pub mod parser;
pub mod units;
pub mod genetic;
pub mod rime;

pub use data::DataCol;

use ndarray::ArrayView1;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
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

/// Guard-backed zero-copy columns extracted from a Python list of NumPy arrays.
///
/// The [`PyReadonlyArray1`] guards pin the Python-side buffers alive, so this
/// struct must live as long as any views materialized from it. The views
/// ([`PythonColumns::views`]) alias the NumPy memory directly: the previous
/// `Vec<Vec<f64>>` deep-copy re-copied the same dataset on every
/// per-candidate call, which is the cost this replaces.
struct PythonColumns<'py> {
    sources: Vec<PyReadonlyArray1<'py, f64>>,
}

impl<'py> PythonColumns<'py> {
    fn extract(obj: &'py Bound<'_, PyAny>, what: &str) -> PyResult<Self> {
        let list = obj
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err(format!("{what} must be a list of 1-D float64 arrays")))?;
        let mut sources = Vec::with_capacity(list.len());
        for item in list.iter() {
            let arr: PyReadonlyArray1<f64> = item.extract().map_err(|e| {
                PyValueError::new_err(format!("{what} must contain 1-D float64 arrays: {e}"))
            })?;
            if arr.as_array().len() == 0 {
                return Err(PyValueError::new_err(format!("{what} must not be empty")));
            }
            sources.push(arr);
        }
        Ok(Self { sources })
    }

    /// Materialize zero-copy borrowed views over the held NumPy buffers.
    fn views(&self) -> Vec<ArrayView1<'_, f64>> {
        self.sources.iter().map(|a| a.as_array()).collect()
    }
}

/// Guard-backed zero-copy 1-D vector extracted from a NumPy array.
///
/// Mirrors [`PythonColumns`] for the single-array case (targets, true/pred
/// pairs). The [`PyReadonlyArray1`] guard pins the Python buffer alive; a
/// contiguous buffer is handed out as a borrowed [`PyReadonlyArray::as_slice`]
/// into Python-side memory with no copy, and a strided buffer falls back to
/// one owned copy so the [`GuardedVector::slice`] contract (`&[f64]`) always
/// holds.
struct GuardedVector<'py> {
    source: PyReadonlyArray1<'py, f64>,
    /// Empty when the source buffer is contiguous; otherwise the one-time
    /// copy that every subsequent consumer reads.
    owned: Vec<f64>,
}

impl<'py> GuardedVector<'py> {
    fn extract(obj: &'py Bound<'_, PyAny>, what: &str) -> PyResult<Self> {
        let arr: PyReadonlyArray1<f64> = obj.extract().map_err(|e| {
            PyValueError::new_err(format!("{what} must be a 1-D float64 array: {e}"))
        })?;
        if arr.as_slice().is_ok() {
            Ok(Self { source: arr, owned: Vec::new() })
        } else {
            // Strided buffer: one-time contiguous copy; every subsequent
            // consumer reads that instead of re-copying per row.
            let owned: Vec<f64> = arr.as_array().iter().copied().collect();
            Ok(Self { source: arr, owned })
        }
    }

    /// Zero-copy view of the extracted vector (one-time copy when strided).
    ///
    /// The returned borrow is tied to `&self` (the guard), and the guard pins
    /// a Python reference — so callers that release the GIL must hoist this
    /// call *outside* the closure. The `&[f64]` slice it yields is
    /// `Send + Sync` and safe to hold across `detach`.
    fn slice(&self) -> &[f64] {
        if !self.owned.is_empty() {
            return &self.owned;
        }
        self.source
            .as_slice()
            .expect("contiguity verified at extraction")
    }
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
    let cols_guard = PythonColumns::extract(data, "data")?;
    let cols = cols_guard.views();
    let values = compiled
        .inner
        .evaluate(&cols)
        .map_err(PyValueError::new_err)?;
    Ok(values.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn compute_metrics(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, f64, f64, bool)> {
    let yt_guard = GuardedVector::extract(y_true, "y_true")?;
    let yp_guard = GuardedVector::extract(y_pred, "y_pred")?;
    Ok(numeric::metrics(yt_guard.slice(), yp_guard.slice()))
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

    let cols_guard = PythonColumns::extract(data, "data")?;
    let cols = cols_guard.views();
    let target_guard = GuardedVector::extract(target, "target")?;
    let target_slice = target_guard.slice();

    // Release the GIL for the fan-out: the whole point of this entry point is
    // using more than one core, and holding the GIL would serialise it.
    let rows = py.detach(|| {
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
                            numeric::metrics(target_slice, &pred);
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
    let columns_guard = PythonColumns::extract(cols, "cols")?;
    let columns = columns_guard.views();
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
    let columns_guard = PythonColumns::extract(cols, "cols")?;
    let columns = columns_guard.views();
    let target_guard = GuardedVector::extract(target, "target")?;
    numeric::linear_least_squares(&columns, target_guard.slice(), lambda_reg)
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
        let res = py.detach(|| {
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

        let py_t = res.t.into_pyarray(py);
        let py_y = arr.into_pyarray(py);
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
    let res = py.detach(|| {
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

    let py_t = res.t.into_pyarray(py);
    let py_y = arr.into_pyarray(py);
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
    allow_exp=true,
    allow_misc=true,
    allow_advanced=true,
    polish_constants=true,
    polish_top_k=3,
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
    allow_exp: bool,
    allow_misc: bool,
    allow_advanced: bool,
    polish_constants: bool,
    polish_top_k: usize,
    seed: u64,
) -> PyResult<(Vec<(String, f64, usize, f64, f64)>, usize)> {
    let columns_guard = PythonColumns::extract(data, "data")?;
    let columns = columns_guard.views();
    let target_guard = GuardedVector::extract(target, "target")?;
    let target_slice = target_guard.slice();
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
        allow_exp,
        allow_misc,
        allow_advanced,
        polish_constants,
        polish_top_k,
        seed,
    };

    let (pareto, total_evals) = py.detach(|| {
        genetic::run_native_evolutionary_search(&columns, target_slice, &variable_names, &config)
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
    let columns_guard = PythonColumns::extract(data, "data")?;
    let columns = columns_guard.views();
    let target_guard = GuardedVector::extract(target, "target")?;
    let target_slice = target_guard.slice();
    // Single-candidate evaluation: no selection charge is added here, because
    // the caller knows how many candidates it looked at; the search sets it.
    let score_cfg = rime::eval::ScoreConfig::new(lambda_c, ridge);
    let s = py.detach(|| {
        rime::eval::score_candidate_fitted(&graph, parent_mse, &columns, target_slice, &score_cfg)
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
    let columns_guard = PythonColumns::extract(data, "data")?;
    let columns = columns_guard.views();
    py.detach(|| graph.evaluate_checked(&columns))
        .map_err(|e| PyValueError::new_err(format!("rime: invalid graph: {}", e)))
}

/// Multi-environment RIME search (spec §8 with per-environment scoring).
///
/// `envs` is a JSON array of `{"cols": [[...]], "target": [...]}` objects, one
/// per environment. `config_json` additionally accepts `"aggregation"`
/// (`"mean"` | `"median_worst_gated"`), `"gamma_gate"`, and `"static"` (disable
/// temporal primitives for order-structured static laws). The aggregation rule
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
    if let Some(v) = cfg_raw.get("static").and_then(|v| v.as_bool()) {
        cfg.static_ops = v;
    }
    if let Some(v) = cfg_raw.get("seed_order").and_then(|v| v.as_str()) {
        cfg.seed_order = if v == "coverage_first" {
            rime::seeds::SeedOrder::CoverageFirst
        } else if v == "legacy" {
            rime::seeds::SeedOrder::Legacy
        } else {
            return Err(PyValueError::new_err(format!(
                "rime: bad seed_order {:?} (want \"legacy\" or \"coverage_first\")", v)));
        };
    }
    if let Some(v) = cfg_raw.get("square_seeds").and_then(|v| v.as_bool()) {
        cfg.square_seeds = v;
    }
    if let Some(v) = cfg_raw.get("rational_ops").and_then(|v| v.as_bool()) {
        cfg.rational_ops = v;
    }
    if let Some(v) = cfg_raw.get("stem_ops").and_then(|v| v.as_bool()) {
        cfg.stem_ops = v;
    }
    if let Some(v) = cfg_raw.get("power_ops").and_then(|v| v.as_bool()) {
        cfg.power_ops = v;
    }
    if let Some(v) = cfg_raw.get("diverse_beam").and_then(|v| v.as_bool()) {
        cfg.diverse_beam = v;
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
    if let Some(v) = cfg_raw.get("max_seeds").and_then(|v| v.as_u64()) {
        cfg.max_seeds = Some(v as usize);
    }
    if let Some(v) = cfg_raw.get("lagged_products").and_then(|v| v.as_bool()) {
        cfg.lagged_products = v;
    }
    if let Some(v) = cfg_raw.get("harmonic_seeds").and_then(|v| v.as_bool()) {
        cfg.harmonic_seeds = v;
    }

    let seeds = rime::seeds::small_forms_extended(
        envs.iter().map(|e| e.cols.len()).max().unwrap_or(1),
        cfg.seed_order,
        cfg.square_seeds,
        cfg.power_ops,
        cfg.max_seeds,
        cfg.lagged_products,
        cfg.harmonic_seeds,
    );
    let outcome =
        py.detach(|| rime::search::search_envs(&seeds, &envs, &cfg, agg, gamma_gate));
    let mut packet = outcome.to_json();
    if let Some(names) = variable_names.filter(|n| !n.is_empty()) {
        if let Some(list) = packet.get_mut("proposals").and_then(|v| v.as_array_mut()) {
            for p in list.iter_mut() {
                if let Some(eq) = p.get("equation").and_then(|v| v.as_str()).map(|s| s.to_string()) {
                    // Two-pass substitution, longest index first: a naive
                    // ascending replace() corrupts multi-digit indices twice —
                    // once ("v13" contains "v1") and again inside placeholders.
                    // Descending order plus control-char placeholders (which
                    // describe() output never contains) makes it exact.
                    let mut e = eq;
                    for i in (0..names.len()).rev() {
                        e = e.replace(&format!("v{}", i), &format!("\u{0}{}\u{1}", i));
                    }
                    for (i, name) in names.iter().enumerate() {
                        e = e.replace(&format!("\u{0}{}\u{1}", i), name);
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
/// optionally `rel_penalty` to scale lambda_c to the target variance, and
/// optionally `static` = true to disable temporal primitives entirely (seeds
/// and rewrites: no lags, no memory cells) for order-structured static laws.
#[pyfunction]
#[pyo3(signature = (data, target, config_json, variable_names=None))]
fn rime_search_json(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    config_json: &str,
    variable_names: Option<Vec<String>>,
) -> PyResult<String> {
    let columns_guard = PythonColumns::extract(data, "data")?;
    let columns = columns_guard.views();
    let target_guard = GuardedVector::extract(target, "target")?;
    let target_slice = target_guard.slice();
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
    if let Some(v) = raw.get("static").and_then(|v| v.as_bool()) {
        cfg.static_ops = v;
    }
    if let Some(v) = raw.get("seed_order").and_then(|v| v.as_str()) {
        cfg.seed_order = if v == "coverage_first" {
            rime::seeds::SeedOrder::CoverageFirst
        } else if v == "legacy" {
            rime::seeds::SeedOrder::Legacy
        } else {
            return Err(PyValueError::new_err(format!(
                "rime: bad seed_order {:?} (want \"legacy\" or \"coverage_first\")", v)));
        };
    }
    if let Some(v) = raw.get("square_seeds").and_then(|v| v.as_bool()) {
        cfg.square_seeds = v;
    }
    if let Some(v) = raw.get("rational_ops").and_then(|v| v.as_bool()) {
        cfg.rational_ops = v;
    }
    if let Some(v) = raw.get("stem_ops").and_then(|v| v.as_bool()) {
        cfg.stem_ops = v;
    }
    if let Some(v) = raw.get("power_ops").and_then(|v| v.as_bool()) {
        cfg.power_ops = v;
    }
    if let Some(v) = raw.get("diverse_beam").and_then(|v| v.as_bool()) {
        cfg.diverse_beam = v;
    }
    if let Some(v) = raw.get("max_seeds").and_then(|v| v.as_u64()) {
        cfg.max_seeds = Some(v as usize);
    }
    if let Some(v) = raw.get("lagged_products").and_then(|v| v.as_bool()) {
        cfg.lagged_products = v;
    }
    if let Some(v) = raw.get("harmonic_seeds").and_then(|v| v.as_bool()) {
        cfg.harmonic_seeds = v;
    }
    if let Some(v) = raw.get("gamma").and_then(|v| v.as_f64()) {
        // Sample-size-aware calibration: gamma * var(y) / n_dev.
        let n_dev = target_guard.slice().iter().filter(|t| t.is_finite()).count();
        cfg.lambda_c = v * rime::search::target_variance(target_guard.slice()) / n_dev.max(1) as f64;
    } else if let Some(v) = raw.get("rel_penalty").and_then(|v| v.as_f64()) {
        cfg.lambda_c = v * rime::search::target_variance(target_guard.slice());
    } else if let Some(v) = raw.get("lambda_c").and_then(|v| v.as_f64()) {
        cfg.lambda_c = v;
    }

    let seeds = rime::seeds::small_forms_extended(
        columns.len(),
        cfg.seed_order,
        cfg.square_seeds,
        cfg.power_ops,
        cfg.max_seeds,
        cfg.lagged_products,
        cfg.harmonic_seeds,
    );
    // Opt-in multi-term refinement (`additive_terms` >= 2). Absent, 0 or 1 is
    // exactly the frozen single-term `search`, so existing callers see no change.
    let additive_terms = raw.get("additive_terms").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let outcome = py.detach(|| {
        rime::additive::search_additive(&seeds, &columns, target_slice, &cfg, additive_terms)
    });
    let mut packet = outcome.to_json();
    // Name the variables in every equation so the referee packet is readable.
    if !requested.is_empty() {
        let names: Vec<&str> = requested.iter().map(|s| s.as_str()).collect();
        if let Some(list) = packet.get_mut("proposals").and_then(|v| v.as_array_mut()) {
            for p in list.iter_mut() {
                if let Some(eq) = p.get("equation").and_then(|v| v.as_str()).map(|s| s.to_string()) {
                    // Two-pass substitution, longest index first: a naive
                    // ascending replace() corrupts multi-digit indices twice —
                    // once ("v13" contains "v1") and again inside placeholders.
                    // Descending order plus control-char placeholders (which
                    // describe() output never contains) makes it exact.
                    let mut e = eq;
                    for i in (0..names.len()).rev() {
                        e = e.replace(&format!("v{}", i), &format!("\u{0}{}\u{1}", i));
                    }
                    for (i, name) in names.iter().enumerate() {
                        e = e.replace(&format!("\u{0}{}\u{1}", i), name);
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

#[pyfunction]
fn rime_total_variation_json(x_json: &str, y_json: &str, lambda: f64, jump_tol: f64) -> PyResult<String> {
    let x: Vec<f64> = serde_json::from_str(x_json)
        .map_err(|e| PyValueError::new_err(format!("bad x JSON: {}", e)))?;
    let y: Vec<f64> = serde_json::from_str(y_json)
        .map_err(|e| PyValueError::new_err(format!("bad y JSON: {}", e)))?;
    let report = rime::total_variation::tv_denoise_and_analyze(&x, &y, lambda, jump_tol)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_implicit_discover_json(x_json: &str, y_json: &str, config_json: &str) -> PyResult<String> {
    let x: Vec<f64> = serde_json::from_str(x_json)
        .map_err(|e| PyValueError::new_err(format!("bad x JSON: {}", e)))?;
    let y: Vec<f64> = serde_json::from_str(y_json)
        .map_err(|e| PyValueError::new_err(format!("bad y JSON: {}", e)))?;
    let config: rime::implicit::ImplicitConfig = if config_json.is_empty() {
        rime::implicit::ImplicitConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::implicit::discover_implicit_relation(&x, &y, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_diff_pade_discover_json(x_json: &str, y_json: &str, config_json: &str) -> PyResult<String> {
    let x: Vec<f64> = serde_json::from_str(x_json)
        .map_err(|e| PyValueError::new_err(format!("bad x JSON: {}", e)))?;
    let y: Vec<f64> = serde_json::from_str(y_json)
        .map_err(|e| PyValueError::new_err(format!("bad y JSON: {}", e)))?;
    let config: rime::diff_pade::DiffPadeConfig = if config_json.is_empty() {
        rime::diff_pade::DiffPadeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::diff_pade::discover_differential_relation(&x, &y, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_enumerate_search_json(
    vars_json: &str,
    target_json: &str,
    var_names_json: &str,
    include_trig: bool,
    include_floor: bool,
    include_exp: bool,
    include_extended: bool,
    include_advanced: bool,
    max_depth: usize,
) -> PyResult<String> {
    let vars: Vec<Vec<f64>> = serde_json::from_str(vars_json)
        .map_err(|e| PyValueError::new_err(format!("bad vars JSON: {}", e)))?;
    let target: Vec<f64> = serde_json::from_str(target_json)
        .map_err(|e| PyValueError::new_err(format!("bad target JSON: {}", e)))?;
    let var_names: Vec<String> = serde_json::from_str(var_names_json)
        .map_err(|e| PyValueError::new_err(format!("bad var_names JSON: {}", e)))?;
    let candidates = rime::enumerate::search_enumerated_shared(
        &vars,
        &target,
        &var_names,
        include_trig,
        include_floor,
        include_exp,
        include_extended,
        include_advanced,
        max_depth,
    );
    serde_json::to_string(&candidates)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_rational_search_json(
    vars_json: &str,
    target_json: &str,
    var_names_json: &str,
    include_trig: bool,
    include_floor: bool,
    include_exp: bool,
    include_advanced: bool,
    max_depth: usize,
    top_k: usize,
) -> PyResult<String> {
    let vars: Vec<Vec<f64>> = serde_json::from_str(vars_json)
        .map_err(|e| PyValueError::new_err(format!("bad vars JSON: {}", e)))?;
    let target: Vec<f64> = serde_json::from_str(target_json)
        .map_err(|e| PyValueError::new_err(format!("bad target JSON: {}", e)))?;
    let var_names: Vec<String> = serde_json::from_str(var_names_json)
        .map_err(|e| PyValueError::new_err(format!("bad var_names JSON: {}", e)))?;
    let candidates = rime::enumerate::search_rational_pairs(
        &vars,
        &target,
        &var_names,
        include_trig,
        include_floor,
        include_exp,
        include_advanced,
        max_depth,
        top_k,
    );
    serde_json::to_string(&candidates)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_koopman_discover_json(
    series_json: &str,
    dt: f64,
    config_json: &str,
) -> PyResult<String> {
    let series: Vec<f64> = serde_json::from_str(series_json)
        .map_err(|e| PyValueError::new_err(format!("bad series JSON: {}", e)))?;
    let config: rime::koopman::KoopmanConfig = if config_json.is_empty() {
        rime::koopman::KoopmanConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::koopman::discover_linear_spectrum(&series, dt, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_lie_symmetry_json(f_terms_json: &str, config_json: &str) -> PyResult<String> {
    let f_terms: Vec<(usize, usize, f64)> = serde_json::from_str(f_terms_json)
        .map_err(|e| PyValueError::new_err(format!("bad f_terms JSON: {}", e)))?;
    let config: rime::lie_symmetry::LieConfig = if config_json.is_empty() {
        rime::lie_symmetry::LieConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::lie_symmetry::discover_point_symmetries(&f_terms, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_pde_discover_json(
    u_grid_json: &str,
    dt: f64,
    dx: f64,
    config_json: &str,
) -> PyResult<String> {
    let u_grid: Vec<Vec<f64>> = serde_json::from_str(u_grid_json)
        .map_err(|e| PyValueError::new_err(format!("bad u_grid JSON: {}", e)))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::pde::discover_pde_1d(&u_grid, dt, dx, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_pde_discover_2d_json(
    u_grid_json: &str,
    dt: f64,
    dx: f64,
    dy: f64,
    config_json: &str,
) -> PyResult<String> {
    let u_grid: Vec<Vec<Vec<f64>>> = serde_json::from_str(u_grid_json)
        .map_err(|e| PyValueError::new_err(format!("bad u_grid 2D JSON: {}", e)))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::pde::discover_pde_2d(&u_grid, dt, dx, dy, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_pde_discover_nonuniform_json(
    u_grid_json: &str,
    t_coords: Vec<f64>,
    x_coords: Vec<f64>,
    config_json: &str,
) -> PyResult<String> {
    let u_grid: Vec<Vec<f64>> = serde_json::from_str(u_grid_json)
        .map_err(|e| PyValueError::new_err(format!("bad u_grid JSON: {}", e)))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::pde::discover_pde_nonuniform(&u_grid, &t_coords, &x_coords, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_pde_discover_3d_json(
    u_grid_json: &str,
    dt: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    config_json: &str,
) -> PyResult<String> {
    let u_grid: Vec<Vec<Vec<Vec<f64>>>> = serde_json::from_str(u_grid_json)
        .map_err(|e| PyValueError::new_err(format!("bad u_grid 3D JSON: {}", e)))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::high_dim_pde::discover_pde_3d_cartesian(&u_grid, dt, dx, dy, dz, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_high_dim_pde_radial_json(
    r_coords: Vec<f64>,
    u_grid_json: &str,
    dt: f64,
    dr: f64,
    dimension: usize,
    config_json: &str,
) -> PyResult<String> {
    let u_grid: Vec<Vec<f64>> = serde_json::from_str(u_grid_json)
        .map_err(|e| PyValueError::new_err(format!("bad u_grid radial JSON: {}", e)))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::high_dim_pde::discover_pde_radial_hd(&r_coords, &u_grid, dt, dr, dimension, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_high_dim_pde_point_cloud_json(
    u_anchors_json: &str,
    u_probes_plus_json: &str,
    u_probes_minus_json: &str,
    dt: f64,
    h: f64,
    dimension: usize,
    config_json: &str,
) -> PyResult<String> {
    let u_anchors: Vec<Vec<f64>> = serde_json::from_str(u_anchors_json)
        .map_err(|e| PyValueError::new_err(format!("bad u_anchors JSON: {}", e)))?;
    let u_probes_plus: Vec<Vec<Vec<f64>>> = serde_json::from_str(u_probes_plus_json)
        .map_err(|e| PyValueError::new_err(format!("bad u_probes_plus JSON: {}", e)))?;
    let u_probes_minus: Vec<Vec<Vec<f64>>> = serde_json::from_str(u_probes_minus_json)
        .map_err(|e| PyValueError::new_err(format!("bad u_probes_minus JSON: {}", e)))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::high_dim_pde::discover_pde_point_cloud_hd(
        &u_anchors,
        &u_probes_plus,
        &u_probes_minus,
        dt,
        h,
        dimension,
        &config,
    ).map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_high_dim_pde_separable_json(
    phi_grid_json: &str,
    dt: f64,
    dx: f64,
    dimension: usize,
    config_json: &str,
) -> PyResult<String> {
    let phi_grid: Vec<Vec<f64>> = serde_json::from_str(phi_grid_json)
        .map_err(|e| PyValueError::new_err(format!("bad phi_grid JSON: {}", e)))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let res = rime::high_dim_pde::discover_pde_separable_hd(&phi_grid, dt, dx, dimension, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
#[pyo3(signature = (series_json, dt, labels=None, config_json=""))]
fn rime_coupled_discover_json(
    series_json: &str,
    dt: f64,
    labels: Option<Vec<String>>,
    config_json: &str,
) -> PyResult<String> {
    let series: Vec<Vec<f64>> = serde_json::from_str(series_json)
        .map_err(|e| PyValueError::new_err(format!("bad series JSON: {}", e)))?;
    let config: rime::coupled_system::CoupledConfig = if config_json.is_empty() {
        rime::coupled_system::CoupledConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };
    let lbl_refs = labels.as_ref().map(|l| l.as_slice());
    let res = rime::coupled_system::discover_coupled_system(&series, dt, lbl_refs, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_number_theory_farey_json(val: f64, max_denom: i64, eps: f64) -> PyResult<String> {
    let res = rime::number_theory::farey_rational(val, max_denom, eps);
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_number_theory_integer_relation_json(
    x: Vec<f64>,
    max_coeff: i64,
    eps: f64,
    labels: Option<Vec<String>>,
) -> PyResult<String> {
    let lbl_refs = labels.as_ref().map(|l| l.as_slice());
    let res = rime::number_theory::find_integer_relation(&x, max_coeff, eps, lbl_refs);
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_number_theory_modular_lattice_json(
    data: Vec<f64>,
    min_spacing: f64,
    max_spacing: f64,
    max_rmse: f64,
) -> PyResult<String> {
    let res = rime::number_theory::detect_modular_lattice(&data, min_spacing, max_spacing, max_rmse);
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_invariant_transfer_evaluate_json(
    candidate_name: String,
    formula: String,
    envs_json: String,
    coeffs: Vec<f64>,
    model_type: String,
    min_worst_r2: f64,
    max_r2_dispersion: f64,
    extrap_margin_ratio: f64,
    max_gradient_ratio: f64,
) -> PyResult<String> {
    let envs: Vec<rime::invariant_transfer::EnvironmentData> = serde_json::from_str(&envs_json)
        .map_err(|e| PyValueError::new_err(format!("bad envs JSON: {}", e)))?;

    let eval_fn = move |x: f64| -> f64 {
        match model_type.as_str() {
            "poly" => {
                let mut sum = 0.0;
                let mut x_pow = 1.0;
                for &c in &coeffs {
                    sum += c * x_pow;
                    x_pow *= x;
                }
                sum
            }
            "rational" => {
                let mid = coeffs.len() / 2;
                let num_coeffs = &coeffs[0..mid];
                let den_coeffs = &coeffs[mid..];
                let mut num = 0.0;
                let mut xp = 1.0;
                for &c in num_coeffs {
                    num += c * xp;
                    xp *= x;
                }
                let mut den = 1.0;
                let mut xd = x;
                for &d in den_coeffs {
                    den += d * xd;
                    xd *= x;
                }
                if den.abs() < 1e-12 { f64::NAN } else { num / den }
            }
            _ => {
                // Default linear: c0 + c1 * x
                let c0 = coeffs.get(0).copied().unwrap_or(0.0);
                let c1 = coeffs.get(1).copied().unwrap_or(1.0);
                c0 + c1 * x
            }
        }
    };

    let res = rime::invariant_transfer::evaluate_invariant_transfer(
        &candidate_name,
        &formula,
        &envs,
        eval_fn,
        min_worst_r2,
        max_r2_dispersion,
        extrap_margin_ratio,
        max_gradient_ratio,
    );

    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_jet2_neural_pde_operator<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w1: PyReadonlyArray2<'py, f64>,
    b1: PyReadonlyArray1<'py, f64>,
    w2: PyReadonlyArray1<'py, f64>,
    viscosity_nu: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let x_shape = x.shape();
    let w1_shape = w1.shape();
    let m = x_shape[0];
    let d = x_shape[1];
    let h = w1_shape[0];
    if w1_shape[1] != d {
        return Err(PyValueError::new_err("w1 column dimension must match x dimension D"));
    }

    let x_slice = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let w1_slice = w1
        .as_slice()
        .map_err(|_| PyValueError::new_err("w1 must be C-contiguous"))?;
    let b1_slice = b1
        .as_slice()
        .map_err(|_| PyValueError::new_err("b1 must be C-contiguous"))?;
    let w2_slice = w2
        .as_slice()
        .map_err(|_| PyValueError::new_err("w2 must be C-contiguous"))?;

    let out = py
        .detach(|| {
            rime::high_dim_pde::eval_jet2_neural_pde_operator(
                x_slice,
                w1_slice,
                b1_slice,
                w2_slice,
                m,
                d,
                h,
                viscosity_nu,
            )
        })
        .map_err(|e| PyValueError::new_err(e))?;

    Ok((
        out.u.into_pyarray(py),
        out.grad_norm_sq.into_pyarray(py),
        out.laplacian.into_pyarray(py),
        out.pde_residual.into_pyarray(py),
    ))
}

#[pyfunction]
fn rime_vlasov_attention_pde_step<'py>(
    py: Python<'py>,
    x_init: PyReadonlyArray2<'py, f64>,
    steps: usize,
    dt: f64,
    damping: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = x_init.shape();
    let n = shape[0];
    let d = shape[1];
    let x_slice = x_init
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_init must be C-contiguous"))?;

    let evolved = py
        .detach(|| {
            rime::high_dim_pde::step_vlasov_attention_pde_hd(x_slice, n, d, steps, dt, damping)
        })
        .map_err(|e| PyValueError::new_err(e))?;

    let arr = Array2::from_shape_vec((n, d), evolved)
        .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
fn rime_high_dim_pde_point_cloud_flat_json<'py>(
    py: Python<'py>,
    u_anchors: PyReadonlyArray2<'py, f64>,
    u_probes_plus: PyReadonlyArray3<'py, f64>,
    u_probes_minus: PyReadonlyArray3<'py, f64>,
    dt: f64,
    h: f64,
    dimension: usize,
    config_json: &str,
) -> PyResult<String> {
    let anc_shape = u_anchors.shape();
    let nt = anc_shape[0];
    let m = anc_shape[1];
    let anc_slice = u_anchors
        .as_slice()
        .map_err(|_| PyValueError::new_err("u_anchors must be C-contiguous"))?;
    let plus_slice = u_probes_plus
        .as_slice()
        .map_err(|_| PyValueError::new_err("u_probes_plus must be C-contiguous"))?;
    let minus_slice = u_probes_minus
        .as_slice()
        .map_err(|_| PyValueError::new_err("u_probes_minus must be C-contiguous"))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };

    let res = py
        .detach(|| {
            rime::high_dim_pde::discover_pde_point_cloud_hd_flat(
                anc_slice,
                plus_slice,
                minus_slice,
                nt,
                m,
                dimension,
                dt,
                h,
                &config,
            )
        })
        .map_err(|e| PyValueError::new_err(e))?;

    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_fokker_planck_ou_operator<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w1: PyReadonlyArray2<'py, f64>,
    b1: PyReadonlyArray1<'py, f64>,
    w2: PyReadonlyArray1<'py, f64>,
    diffusion_nu: f64,
    drift_gamma: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let x_shape = x.shape();
    let w1_shape = w1.shape();
    let m = x_shape[0];
    let d = x_shape[1];
    let h = w1_shape[0];
    if w1_shape[1] != d {
        return Err(PyValueError::new_err("w1 column dimension must match x dimension D"));
    }

    let x_slice = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let w1_slice = w1.as_slice().map_err(|_| PyValueError::new_err("w1 must be C-contiguous"))?;
    let b1_slice = b1.as_slice().map_err(|_| PyValueError::new_err("b1 must be C-contiguous"))?;
    let w2_slice = w2.as_slice().map_err(|_| PyValueError::new_err("w2 must be C-contiguous"))?;

    let out = py
        .detach(|| {
            rime::high_dim_pde::eval_fokker_planck_ou_operator(
                x_slice,
                w1_slice,
                b1_slice,
                w2_slice,
                m,
                d,
                h,
                diffusion_nu,
                drift_gamma,
            )
        })
        .map_err(|e| PyValueError::new_err(e))?;

    Ok((
        out.u.into_pyarray(py),
        out.grad_norm_sq.into_pyarray(py),
        out.laplacian.into_pyarray(py),
        out.pde_residual.into_pyarray(py),
    ))
}

#[pyfunction]
fn rime_jet3_pde_loss_and_grads<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    target_f: PyReadonlyArray1<'py, f64>,
    w1: PyReadonlyArray2<'py, f64>,
    b1: PyReadonlyArray1<'py, f64>,
    w2: PyReadonlyArray1<'py, f64>,
    viscosity_nu: f64,
    hjb_beta: f64,
    reaction_mu: f64,
) -> PyResult<(
    f64,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let x_shape = x.shape();
    let w1_shape = w1.shape();
    let m = x_shape[0];
    let d = x_shape[1];
    let h = w1_shape[0];
    if w1_shape[1] != d {
        return Err(PyValueError::new_err("w1 column dimension must match x dimension D"));
    }

    let x_slice = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let f_slice = target_f.as_slice().map_err(|_| PyValueError::new_err("target_f must be C-contiguous"))?;
    let w1_slice = w1.as_slice().map_err(|_| PyValueError::new_err("w1 must be C-contiguous"))?;
    let b1_slice = b1.as_slice().map_err(|_| PyValueError::new_err("b1 must be C-contiguous"))?;
    let w2_slice = w2.as_slice().map_err(|_| PyValueError::new_err("w2 must be C-contiguous"))?;

    let out = py
        .detach(|| {
            rime::high_dim_pde::compute_jet3_pde_loss_and_grads(
                x_slice,
                f_slice,
                w1_slice,
                b1_slice,
                w2_slice,
                m,
                d,
                h,
                viscosity_nu,
                hjb_beta,
                reaction_mu,
            )
        })
        .map_err(|e| PyValueError::new_err(e))?;

    let gw1_arr = Array2::from_shape_vec((h, d), out.grad_w1)
        .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;

    Ok((
        out.loss,
        gw1_arr.into_pyarray(py),
        out.grad_b1.into_pyarray(py),
        out.grad_w2.into_pyarray(py),
    ))
}

#[pyfunction]
fn rime_pde_discover_nd_cartesian_json<'py>(
    py: Python<'py>,
    u_flat: PyReadonlyArray1<'py, f64>,
    grid_shape: Vec<usize>,
    dt: f64,
    dxs: Vec<f64>,
    config_json: &str,
) -> PyResult<String> {
    let u_slice = u_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("u_flat must be C-contiguous"))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };

    let res = py
        .detach(|| {
            rime::high_dim_pde::discover_pde_nd_cartesian(u_slice, &grid_shape, dt, &dxs, &config)
        })
        .map_err(|e| PyValueError::new_err(e))?;

    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_high_dim_pde_rbf_point_cloud_json<'py>(
    py: Python<'py>,
    points_flat: PyReadonlyArray1<'py, f64>,
    u_values_flat: PyReadonlyArray1<'py, f64>,
    nt: usize,
    m: usize,
    dimension: usize,
    dt: f64,
    config_json: &str,
) -> PyResult<String> {
    let pts_slice = points_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("points_flat must be C-contiguous"))?;
    let u_slice = u_values_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("u_values_flat must be C-contiguous"))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };

    let res = py
        .detach(|| {
            rime::high_dim_pde::discover_pde_rbf_point_cloud_hd(
                pts_slice, u_slice, nt, m, dimension, dt, &config,
            )
        })
        .map_err(|e| PyValueError::new_err(e))?;

    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_high_dim_pde_tensor_train_json<'py>(
    py: Python<'py>,
    tt_cores_flat: PyReadonlyArray1<'py, f64>,
    nt: usize,
    bond_rank: usize,
    dimension: usize,
    nx: usize,
    dt: f64,
    dx: f64,
    config_json: &str,
) -> PyResult<String> {
    let cores_slice = tt_cores_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("tt_cores_flat must be C-contiguous"))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };

    let res = py
        .detach(|| {
            rime::high_dim_pde::discover_pde_tensor_train_hd(
                cores_slice,
                nt,
                bond_rank,
                dimension,
                nx,
                dt,
                dx,
                &config,
            )
        })
        .map_err(|e| PyValueError::new_err(e))?;

    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_high_dim_pde_neural_pinn_json<'py>(
    py: Python<'py>,
    x_collocation_flat: PyReadonlyArray1<'py, f64>,
    w1_flat: PyReadonlyArray1<'py, f64>,
    b1: PyReadonlyArray1<'py, f64>,
    w2_time_flat: PyReadonlyArray1<'py, f64>,
    nt: usize,
    m: usize,
    dimension: usize,
    h: usize,
    dt: f64,
    config_json: &str,
) -> PyResult<String> {
    let x_slice = x_collocation_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_collocation_flat must be C-contiguous"))?;
    let w1_slice = w1_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("w1_flat must be C-contiguous"))?;
    let b1_slice = b1
        .as_slice()
        .map_err(|_| PyValueError::new_err("b1 must be C-contiguous"))?;
    let w2_slice = w2_time_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("w2_time_flat must be C-contiguous"))?;
    let config: rime::pde::PdeConfig = if config_json.is_empty() {
        rime::pde::PdeConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config JSON: {}", e)))?
    };

    let res = py
        .detach(|| {
            rime::high_dim_pde::discover_pde_neural_pinn_hd(
                x_slice, w1_slice, b1_slice, w2_slice, nt, m, dimension, h, dt, &config,
            )
        })
        .map_err(|e| PyValueError::new_err(e))?;

    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {}", e)))
}

#[pyfunction]
fn rime_jet2_video_flow_step<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    v_current: PyReadonlyArray1<'py, f64>,
    v_t_plus: PyReadonlyArray1<'py, f64>,
    v_t_minus: PyReadonlyArray1<'py, f64>,
    v_spatial_plus: PyReadonlyArray1<'py, f64>,
    v_spatial_minus: PyReadonlyArray1<'py, f64>,
    dt: f64,
    dt_span: f64,
    two_h_disp: f64,
    order: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let z_slice = z
        .as_slice()
        .map_err(|_| PyValueError::new_err("z must be C-contiguous"))?;
    let vc_slice = v_current
        .as_slice()
        .map_err(|_| PyValueError::new_err("v_current must be C-contiguous"))?;
    let vtp_slice = v_t_plus
        .as_slice()
        .map_err(|_| PyValueError::new_err("v_t_plus must be C-contiguous"))?;
    let vtm_slice = v_t_minus
        .as_slice()
        .map_err(|_| PyValueError::new_err("v_t_minus must be C-contiguous"))?;
    let vsp_slice = v_spatial_plus
        .as_slice()
        .map_err(|_| PyValueError::new_err("v_spatial_plus must be C-contiguous"))?;
    let vsm_slice = v_spatial_minus
        .as_slice()
        .map_err(|_| PyValueError::new_err("v_spatial_minus must be C-contiguous"))?;

    let (z_next, a_out) = py
        .detach(|| {
            rime::high_dim_pde::jet2_video_flow_step(
                z_slice,
                vc_slice,
                vtp_slice,
                vtm_slice,
                vsp_slice,
                vsm_slice,
                dt,
                dt_span,
                two_h_disp,
                order,
            )
        })
        .map_err(|e| PyValueError::new_err(e))?;

    Ok((z_next.into_pyarray(py), a_out.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (vx, vy, height, width, dx=1.0, dy=1.0, num_iter=50))]
fn rime_helmholtz_hodge_div_free_2d<'py>(
    py: Python<'py>,
    vx: PyReadonlyArray1<'py, f64>,
    vy: PyReadonlyArray1<'py, f64>,
    height: usize,
    width: usize,
    dx: f64,
    dy: f64,
    num_iter: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let vx_slice = vx.as_slice().map_err(|_| PyValueError::new_err("vx must be C-contiguous"))?;
    let vy_slice = vy.as_slice().map_err(|_| PyValueError::new_err("vy must be C-contiguous"))?;

    let (vx_sol, vy_sol, rem_div) = py
        .detach(|| {
            rime::high_dim_pde::helmholtz_hodge_div_free_2d(vx_slice, vy_slice, height, width, dx, dy, num_iter)
        })
        .map_err(|e| PyValueError::new_err(e))?;

    Ok((vx_sol.into_pyarray(py), vy_sol.into_pyarray(py), rem_div))
}

#[pyfunction]
#[pyo3(signature = (field, height, width, channels, nu, dt, dx=1.0, dy=1.0))]
fn rime_laplacian_dissipation_2d<'py>(
    py: Python<'py>,
    field: PyReadonlyArray1<'py, f64>,
    height: usize,
    width: usize,
    channels: usize,
    nu: f64,
    dt: f64,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let field_slice = field.as_slice().map_err(|_| PyValueError::new_err("field must be C-contiguous"))?;

    let out = py
        .detach(|| {
            rime::high_dim_pde::laplacian_dissipation_2d(field_slice, height, width, channels, nu, dt, dx, dy)
        })
        .map_err(|e| PyValueError::new_err(e))?;

    Ok(out.into_pyarray(py))
}

/// First-line port of `apedp_dyn/video/eval_gate.py` frame metrics.
///
/// Data enters as borrowed NumPy views through the same zero-copy path as
/// the parser surface. Failures raise; a raise is "this input could not be
/// scored", never a verdict, and the bridge falls back to the NumPy
/// reference exactly like every other kernel-backed method.
#[pyfunction]
fn video_psnr(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let a_guard = GuardedVector::extract(a, "a")?;
    let b_guard = GuardedVector::extract(b, "b")?;
    video::psnr_series(a_guard.slice(), b_guard.slice()).map_err(PyValueError::new_err)
}

/// Mean windowed SSIM over one grayscale frame given as row views.
#[pyfunction]
fn video_ssim_frame(
    a_rows: &Bound<'_, PyAny>,
    b_rows: &Bound<'_, PyAny>,
    win: usize,
    c1: f64,
    c2: f64,
) -> PyResult<f64> {
    let a_guard = PythonColumns::extract(a_rows, "a_rows")?;
    let b_guard = PythonColumns::extract(b_rows, "b_rows")?;
    let a = a_guard.views();
    let b = b_guard.views();
    Ok(video::ssim_frame(&a, &b, win, c1, c2))
}

/// Sliced Wasserstein-2 over explicit (numpy-generated) projection rows.
#[pyfunction]
fn video_sliced_wasserstein2(
    x_rows: &Bound<'_, PyAny>,
    y_rows: &Bound<'_, PyAny>,
    dir_rows: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let x_guard = PythonColumns::extract(x_rows, "x_rows")?;
    let y_guard = PythonColumns::extract(y_rows, "y_rows")?;
    let d_guard = PythonColumns::extract(dir_rows, "dir_rows")?;
    let x = x_guard.views();
    let y = y_guard.views();
    let dirs = d_guard.views();
    video::sliced_wasserstein2(&x, &y, &dirs).map_err(PyValueError::new_err)
}

/// 2D Bilinear Sub-Pixel Coordinate Warping of an image or latent field.
#[pyfunction]
#[pyo3(signature = (input, vx, vy, channels, height, width))]
fn video_warp_bilinear_2d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, f64>,
    vx: PyReadonlyArray1<'py, f64>,
    vy: PyReadonlyArray1<'py, f64>,
    channels: usize,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let in_slice = input.as_slice().map_err(|_| PyValueError::new_err("input must be C-contiguous"))?;
    let vx_slice = vx.as_slice().map_err(|_| PyValueError::new_err("vx must be C-contiguous"))?;
    let vy_slice = vy.as_slice().map_err(|_| PyValueError::new_err("vy must be C-contiguous"))?;

    let out = py
        .detach(|| {
            video::warp_bilinear_2d(in_slice, vx_slice, vy_slice, channels, height, width)
        })
        .map_err(PyValueError::new_err)?;

    Ok(out.into_pyarray(py))
}

/// 2D Bilinear Sub-Pixel Coordinate Warping with periodic (wrap-around)
/// boundary conditions (floor-then-mod backtrace indexing, numpy `%`
/// semantics); the fluid-domain generator's advection kernel.
#[pyfunction]
#[pyo3(signature = (input, vx, vy, channels, height, width))]
fn video_warp_bilinear_2d_periodic<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, f64>,
    vx: PyReadonlyArray1<'py, f64>,
    vy: PyReadonlyArray1<'py, f64>,
    channels: usize,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let in_slice = input.as_slice().map_err(|_| PyValueError::new_err("input must be C-contiguous"))?;
    let vx_slice = vx.as_slice().map_err(|_| PyValueError::new_err("vx must be C-contiguous"))?;
    let vy_slice = vy.as_slice().map_err(|_| PyValueError::new_err("vy must be C-contiguous"))?;

    let out = py
        .detach(|| {
            video::warp_bilinear_2d_periodic(in_slice, vx_slice, vy_slice, channels, height, width)
        })
        .map_err(PyValueError::new_err)?;

    Ok(out.into_pyarray(py))
}

/// Compute spatial optical flow divergence: div(v) = d(vx)/dx + d(vy)/dy.
#[pyfunction]
#[pyo3(signature = (vx, vy, height, width, dx=1.0, dy=1.0))]
fn video_compute_optical_flow_divergence<'py>(
    py: Python<'py>,
    vx: PyReadonlyArray1<'py, f64>,
    vy: PyReadonlyArray1<'py, f64>,
    height: usize,
    width: usize,
    dx: f64,
    dy: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vx_slice = vx.as_slice().map_err(|_| PyValueError::new_err("vx must be C-contiguous"))?;
    let vy_slice = vy.as_slice().map_err(|_| PyValueError::new_err("vy must be C-contiguous"))?;

    let out = py
        .detach(|| {
            video::compute_optical_flow_divergence(vx_slice, vy_slice, height, width, dx, dy)
        })
        .map_err(PyValueError::new_err)?;

    Ok(out.into_pyarray(py))
}

/// Detect disocclusion mask where flow divergence exceeds threshold.
#[pyfunction]
#[pyo3(signature = (div_flow, divergence_threshold=0.0))]
fn video_detect_disocclusion_mask<'py>(
    py: Python<'py>,
    div_flow: PyReadonlyArray1<'py, f64>,
    divergence_threshold: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let div_slice = div_flow.as_slice().map_err(|_| PyValueError::new_err("div_flow must be C-contiguous"))?;

    let out = py
        .detach(|| {
            video::detect_disocclusion_mask(div_slice, divergence_threshold)
        });

    Ok(out.into_pyarray(py))
}

/// Learned flow-head forward pass: v = head(z_prev, z_curr).
#[pyfunction]
#[pyo3(signature = (z_prev, z_curr, channel_count, hidden, n_layers, flow_scale, height, width, weights))]
fn flow_head_forward<'py>(
    py: Python<'py>,
    z_prev: PyReadonlyArray1<'py, f64>,
    z_curr: PyReadonlyArray1<'py, f64>,
    channel_count: usize,
    hidden: usize,
    n_layers: usize,
    flow_scale: f64,
    height: usize,
    width: usize,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let spec = flow_head::FlowHeadSpec { c_in: channel_count, hidden, n_layers, flow_scale };
    let zp = z_prev.as_slice().map_err(|_| PyValueError::new_err("z_prev must be C-contiguous"))?;
    let zc = z_curr.as_slice().map_err(|_| PyValueError::new_err("z_curr must be C-contiguous"))?;
    let wt = weights.as_slice().map_err(|_| PyValueError::new_err("weights must be C-contiguous"))?;
    let (vx, vy) = py
        .detach(|| flow_head::forward_v(zp, zc, spec, wt, height, width))
        .map_err(PyValueError::new_err)?;
    Ok((vx.into_pyarray(py), vy.into_pyarray(py)))
}

/// Self-supervised flow-head training over explicit frame triples.
/// `frame_triples` is the flattened [t_prev, t_curr, t_next] * n list.
/// Returns (losses, trained_weights) — the caller re-injects the weights
/// (no unsafe in-place aliasing across the boundary).
#[pyfunction]
#[pyo3(signature = (data, frame_triples, channel_count, hidden, n_layers, flow_scale, height, width, weights, steps, lr, weight_decay, lambda_div, lambda_smooth, dt, deadline_ms=None))]
#[allow(clippy::too_many_arguments)]
fn flow_head_train<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    frame_triples: PyReadonlyArray1<'py, u64>,
    channel_count: usize,
    hidden: usize,
    n_layers: usize,
    flow_scale: f64,
    height: usize,
    width: usize,
    weights: PyReadonlyArray1<'py, f64>,
    steps: usize,
    lr: f64,
    weight_decay: f64,
    lambda_div: f64,
    lambda_smooth: f64,
    dt: f64,
    deadline_ms: Option<u128>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let spec = flow_head::FlowHeadSpec { c_in: channel_count, hidden, n_layers, flow_scale };
    let d = data.as_slice().map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
    let tr = frame_triples.as_slice().map_err(|_| PyValueError::new_err("frame_triples must be C-contiguous"))?;
    if tr.len() % 3 != 0 {
        return Err(PyValueError::new_err("frame_triples must pack groups of three"));
    }
    let triples: Vec<[usize; 3]> = tr
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();
    let w0 = weights.as_slice().map_err(|_| PyValueError::new_err("weights must be C-contiguous"))?.to_vec();
    let (losses, w_out) = py
        .detach(move || {
            let mut w_mut = w0;
            let res = flow_head::train(
                d, &triples, spec, &mut w_mut, height, width, steps, lr, weight_decay,
                lambda_div, lambda_smooth, dt, deadline_ms,
            );
            res.map(|losses| (losses, w_mut))
        })
        .map_err(PyValueError::new_err)?;
    Ok((losses.into_pyarray(py), w_out.into_pyarray(py)))
}

/// One full DC-STT transport step (head + optional Leray + warp + optional
/// viscosity) in a single Python<->Rust crossing.
#[pyfunction]
#[pyo3(signature = (state, head_prev, head_curr, channel_count, hidden, n_layers, flow_scale, height, width, weights, dt, enforce_hh, hh_iters, nu, use_dissipation, dx, dy))]
#[allow(clippy::too_many_arguments)]
fn flow_head_transport_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    head_prev: PyReadonlyArray1<'py, f64>,
    head_curr: PyReadonlyArray1<'py, f64>,
    channel_count: usize,
    hidden: usize,
    n_layers: usize,
    flow_scale: f64,
    height: usize,
    width: usize,
    weights: PyReadonlyArray1<'py, f64>,
    dt: f64,
    enforce_hh: bool,
    hh_iters: usize,
    nu: f64,
    use_dissipation: bool,
    dx: f64,
    dy: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let spec = flow_head::FlowHeadSpec { c_in: channel_count, hidden, n_layers, flow_scale };
    let st = state.as_slice().map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
    let hp = head_prev.as_slice().map_err(|_| PyValueError::new_err("head_prev must be C-contiguous"))?;
    let hc = head_curr.as_slice().map_err(|_| PyValueError::new_err("head_curr must be C-contiguous"))?;
    let wt = weights.as_slice().map_err(|_| PyValueError::new_err("weights must be C-contiguous"))?;
    let (z_next, vx, vy, rem) = py
        .detach(|| {
            flow_head::transport_step(
                st, hp, hc, spec, wt, height, width, dt, enforce_hh, hh_iters, nu,
                use_dissipation, dx, dy,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((z_next.into_pyarray(py), vx.into_pyarray(py), vy.into_pyarray(py), rem))
}

#[pymodule]
fn apedp_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(video_psnr, m)?)?;
    m.add_function(wrap_pyfunction!(video_ssim_frame, m)?)?;
    m.add_function(wrap_pyfunction!(video_sliced_wasserstein2, m)?)?;
    m.add_function(wrap_pyfunction!(video_warp_bilinear_2d, m)?)?;
    m.add_function(wrap_pyfunction!(video_warp_bilinear_2d_periodic, m)?)?;
    m.add_function(wrap_pyfunction!(video_compute_optical_flow_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(video_detect_disocclusion_mask, m)?)?;
    m.add_function(wrap_pyfunction!(flow_head_forward, m)?)?;
    m.add_function(wrap_pyfunction!(flow_head_train, m)?)?;
    m.add_function(wrap_pyfunction!(flow_head_transport_step, m)?)?;
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
    m.add_function(wrap_pyfunction!(rime_total_variation_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_implicit_discover_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_diff_pade_discover_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_enumerate_search_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_rational_search_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_koopman_discover_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_lie_symmetry_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_pde_discover_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_pde_discover_2d_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_pde_discover_nonuniform_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_pde_discover_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_pde_discover_nd_cartesian_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_high_dim_pde_radial_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_high_dim_pde_point_cloud_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_high_dim_pde_point_cloud_flat_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_high_dim_pde_rbf_point_cloud_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_high_dim_pde_tensor_train_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_high_dim_pde_neural_pinn_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_high_dim_pde_separable_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_jet2_neural_pde_operator, m)?)?;
    m.add_function(wrap_pyfunction!(rime_jet2_video_flow_step, m)?)?;
    m.add_function(wrap_pyfunction!(rime_helmholtz_hodge_div_free_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rime_laplacian_dissipation_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rime_fokker_planck_ou_operator, m)?)?;
    m.add_function(wrap_pyfunction!(rime_jet3_pde_loss_and_grads, m)?)?;
    m.add_function(wrap_pyfunction!(rime_vlasov_attention_pde_step, m)?)?;
    m.add_function(wrap_pyfunction!(rime_coupled_discover_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_number_theory_farey_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_number_theory_integer_relation_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_number_theory_modular_lattice_json, m)?)?;
    m.add_function(wrap_pyfunction!(rime_invariant_transfer_evaluate_json, m)?)?;
    m.add_class::<UnitVector>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}