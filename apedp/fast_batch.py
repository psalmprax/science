"""Fast batch data processing with Rust-first operations."""
from __future__ import annotations

import numpy as np


class FastBatchProcessor:
    """Fast batch operations on data."""
    
    def normalize(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Fast normalization."""
        try:
            from apedp.rust_bridge import get_default_bridge
            bridge = get_default_bridge()
            if bridge.use_rust and hasattr(bridge, 'kernel'):
                kernel = bridge.kernel
                if hasattr(kernel, 'batch_normalize'):
                    return kernel.batch_normalize(X)
        except Exception:
            pass
        
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-12
        X_norm = (X - mean) / std
        return X_norm, {"mean": mean, "std": std}
    
    def batch_apply(self, func, data: np.ndarray, batch_size: int = 1000):
        """Apply function in batches for memory efficiency."""
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            results.append(func(batch))
        return np.concatenate(results) if results else np.array([])


_default_batch = None

def get_fast_batch() -> FastBatchProcessor:
    global _default_batch
    if _default_batch is None:
        _default_batch = FastBatchProcessor()
    return _default_batch
