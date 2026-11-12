//! SSM/Mamba operations for efficient temporal modeling
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (x, a_log, dt, b, c, d))]
pub fn ssm_scan<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f64>,
    a_log: PyReadonlyArray2<'py, f64>,
    dt: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray3<'py, f64>,
    c: PyReadonlyArray3<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    // Keep the Python keyword interface, but never return placeholder science.
    let _ = (py, x, a_log, dt, b, c, d);
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "ssm_scan has no implemented native recurrence; use a validated reference implementation",
    ))
}

#[pyfunction]
pub fn mamba2_selective_scan<'py>(
    py: Python<'py>,
    u: PyReadonlyArray3<'py, f64>,
    delta: PyReadonlyArray3<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray3<'py, f64>,
    c: PyReadonlyArray3<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let _ = (py, u, delta, a, b, c, d);
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "mamba2_selective_scan has no implemented native recurrence; use a validated reference implementation",
    ))
}
