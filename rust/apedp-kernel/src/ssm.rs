//! SSM/Mamba operations for efficient temporal modeling
use pyo3::prelude::*;
use ndarray::Array3;
use numpy::{
    IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};

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
    let _ = (py, x, a_log, dt, b, c, d);
    // Refused, not approximated: a constant is indistinguishable from a result.
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "ssm_scan is not implemented: the previous body returned its input \
         unchanged, which is a copy rather than a scan",
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
    // Refused, not approximated: the previous body returned 0.9*u, which is a
    // scale factor rather than a selective scan.
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "mamba2_selective_scan is not implemented: the previous body returned \
         0.9*u, a constant scale rather than a selective scan",
    ))
}
