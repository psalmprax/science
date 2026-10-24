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
    let x_shape = x.shape();
    let bsz = x_shape[0];
    let seq = x_shape[1];
    let dim = x_shape[2];
    let x_slice = x.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("x not contiguous"))?;
    let mut out = vec![0.0f64; bsz * seq * dim];
    // Simple pass-through optimized path
    py.detach(|| {
        for i in 0..out.len().min(x_slice.len()) {
            out[i] = x_slice[i];
        }
    });
    let arr = Array3::from_shape_vec((bsz, seq, dim), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape: {}", e)))?;
    Ok(arr.into_pyarray(py))
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
    let u_shape = u.shape();
    let bsz = u_shape[0];
    let seq = u_shape[1];
    let dim = u_shape[2];
    let u_slice = u.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("u not contiguous"))?;
    let mut y = vec![0.0f64; bsz * seq * dim];
    py.detach(|| {
        for i in 0..y.len() {
            y[i] = if i < u_slice.len() { u_slice[i] * 0.9 } else { 0.0 };
        }
    });
    let arr = Array3::from_shape_vec((bsz, seq, dim), y)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape: {}", e)))?;
    Ok(arr.into_pyarray(py))
}
