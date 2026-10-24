"""Fast expression canonicalization with Rust-first backend."""
from __future__ import annotations

import hashlib
from typing import Any

import sympy as sp

from apedp.safety import safe_parse_expr


class FastCanonicalizer:
    """Fast canonicalization and deduplication."""
    
    def canonicalize(self, expr_str: str) -> str:
        """Get canonical form of expression."""
        try:
            from apedp.rust_bridge import get_default_bridge
            bridge = get_default_bridge()
            if bridge.use_rust and hasattr(bridge, 'kernel'):
                kernel = bridge.kernel
                if hasattr(kernel, 'canonicalize_expr'):
                    return kernel.canonicalize_expr(expr_str)
        except Exception:
            pass
        
        return self._python_canonicalize(expr_str)
    
    def _python_canonicalize(self, expr_str: str) -> str:
        """Python fallback canonicalization."""
        try:
            expr = safe_parse_expr(expr_str)
            # Use sympy's canonicalization
            expr_canon = sp.simplify(expr)
            return str(expr_canon)
        except Exception:
            return expr_str
    
    def deduplicate(self, expressions: list[str]) -> list[str]:
        """Remove duplicate expressions."""
        seen = set()
        unique = []
        for expr in expressions:
            canon = self.canonicalize(expr)
            key = self._hash_expr(canon)
            if key not in seen:
                seen.add(key)
                unique.append(expr)
        return unique
    
    def _hash_expr(self, expr_str: str) -> str:
        return hashlib.md5(expr_str.encode()).hexdigest()


_default_canon = None

def get_fast_canonicalizer() -> FastCanonicalizer:
    global _default_canon
    if _default_canon is None:
        _default_canon = FastCanonicalizer()
    return _default_canon
