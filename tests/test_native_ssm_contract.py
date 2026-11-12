"""Unimplemented native scans must not return plausible-looking placeholder data."""
import numpy as np
import pytest

from apedp.rust_bridge import is_rust_kernel_available


@pytest.mark.skipif(not is_rust_kernel_available(), reason="native kernel is not built")
def test_native_scan_placeholders_fail_explicitly():
    import apedp_kernel

    volume = np.zeros((1, 2, 3), dtype=np.float64)
    matrix = np.zeros((1, 3), dtype=np.float64)
    vector = np.zeros(3, dtype=np.float64)
    with pytest.raises(NotImplementedError, match="no implemented native recurrence"):
        apedp_kernel.ssm_scan(
            x=volume, a_log=matrix, dt=matrix, b=volume, c=volume, d=matrix
        )
    with pytest.raises(NotImplementedError, match="no implemented native recurrence"):
        apedp_kernel.mamba2_selective_scan(
            u=volume, delta=volume, a=vector, b=volume, c=volume, d=matrix
        )
