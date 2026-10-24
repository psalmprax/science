"""Fast grammar enumeration and pruning with Rust-first backend."""
from __future__ import annotations

from typing import Any


class FastGrammar:
    """Fast grammar operations."""
    
    def enumerate(self, variables: list[str], max_terms: int = 4) -> list[dict[str, Any]]:
        """Enumerate grammar candidates."""
        try:
            from apedp.rust_bridge import get_default_bridge
            bridge = get_default_bridge()
            if bridge.use_rust and hasattr(bridge, 'kernel'):
                kernel = bridge.kernel
                if hasattr(kernel, 'grammar_enumerate'):
                    return kernel.grammar_enumerate(variables, max_terms)
        except Exception:
            pass
        
        return self._python_enumerate(variables, max_terms)
    
    def _python_enumerate(self, variables: list[str], max_terms: int) -> list[dict[str, Any]]:
        """Python fallback enumeration."""
        from apedp_dyn.grammar.mechanical_grammar import enumerate_candidates
        try:
            cands = enumerate_candidates(required=("u",) if "u" in variables else (variables[0] if variables else "x",),
                                        optional=tuple(v for v in variables if v not in ("u",)),
                                        max_terms=max_terms)
            return [{"expression": str(c)} for c in cands[:100]]
        except Exception:
            return []


_default_grammar = None

def get_fast_grammar() -> FastGrammar:
    global _default_grammar
    if _default_grammar is None:
        _default_grammar = FastGrammar()
    return _default_grammar
