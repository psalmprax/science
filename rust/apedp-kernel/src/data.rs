//! Zero-copy column data interface.
//!
//! Python hands the kernel NumPy float64 columns; before this module those
//! were deep-copied into owned `Vec<f64>` on every entry point that took
//! array data, which re-copied the same dataset on every per-candidate call.
//! [`DataCol`] erases the owned-vs-borrowed distinction instead: functions in
//! this crate take `&[V] where V: DataCol` and index through the trait, so
//! `Vec<f64>` (the internal, already-owned path used by the JSON entry points)
//! and `ndarray::ArrayView1<'_, f64>` (the borrowed NumPy buffer path used by
//! the pyo3 interface) are interchangable with no copy in either direction.
//!
//! `Send + Sync` are supertraits because the parallel scoring paths
//! (`rayon::par_iter`) share `&[V]` across the rayon pool; both shipped
//! implementations satisfy them.
use ndarray::ArrayView1;

/// One float64 data column: owned (`Vec<f64>`) or borrowed
/// (`ArrayView1<'_, f64>` straight out of a NumPy buffer).
pub trait DataCol: Send + Sync {
    /// Number of rows in this column.
    fn row_len(&self) -> usize;

    /// Value at row `t`, or `None` when out of range. Graph input nodes use
    /// this to reproduce `Vec::get` semantics (out-of-range is NaN downstream).
    fn row_opt(&self, t: usize) -> Option<f64>;

    /// Value at row `t`; out-of-range reads render as NaN, matching the
    /// factorized semantics callers previously got from slicing owned vecs.
    #[inline]
    fn row(&self, t: usize) -> f64 {
        self.row_opt(t).unwrap_or(f64::NAN)
    }
}

impl DataCol for Vec<f64> {
    #[inline]
    fn row_len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn row_opt(&self, t: usize) -> Option<f64> {
        self.get(t).copied()
    }
}

impl DataCol for ArrayView1<'_, f64> {
    #[inline]
    fn row_len(&self) -> usize {
        ArrayView1::len(self)
    }

    #[inline]
    fn row_opt(&self, t: usize) -> Option<f64> {
        self.get(t).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_and_view_columns_read_identically() {
        let owned = vec![1.0, 2.0, 3.0];
        let view = ArrayView1::from(&owned[..]);
        assert_eq!(owned.row_len(), view.row_len());
        for t in 0..owned.len() {
            assert_eq!(owned.row(t), view.row(t));
            assert_eq!(owned.row_opt(t), view.row_opt(t));
        }
        assert_eq!(owned.row_opt(3), None);
        assert!(owned.row(3).is_nan());
        assert!(view.row(3).is_nan());
    }

    #[test]
    fn views_are_shareable_across_the_rayon_pool() {
        // Parity of the bridge's promise: a borrowed NumPy column can be sent
        // to worker threads without panicking.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ArrayView1<'static, f64>>();
        assert_send_sync::<Vec<f64>>();
    }
}
