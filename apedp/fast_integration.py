"""Fast ODE integration with Rust-first backend and Python fallback."""
from __future__ import annotations

from typing import Any

import numpy as np

from apedp.rust_bridge import RustKernelBridge, get_default_bridge


class FastODEIntegrator:
    """Rust-first ODE integrator with fallback."""
    
    def __init__(self, bridge: RustKernelBridge | None = None):
        self.bridge = bridge or get_default_bridge()
    
    def integrate(self, equations: list[str], state_names: list[str], 
                  param_names: list[str], param_values: list[float],
                  t_span: tuple[float, float], x0: list[float], 
                  t_eval: list[float], **kwargs) -> dict[str, Any]:
        """Integrate ODE system with Rust acceleration."""
        try:
            if hasattr(self.bridge, 'simulate_ode') or hasattr(getattr(self.bridge, 'kernel', None), 'simulate_ode_rk45'):
                kernel = getattr(self.bridge, 'kernel', None)
                if kernel and hasattr(kernel, 'simulate_ode_rk45'):
                    res = kernel.simulate_ode_rk45(
                        equations, state_names, param_names, param_values,
                        t_span, x0, t_eval, **kwargs
                    )
                    return {
                        "success": res[0],
                        "message": res[1],
                        "nfev": res[2],
                        "t": np.array(res[3]),
                        "y": np.array(res[4]),
                        "kernel_path": "rust"
                    }
        except Exception:
            pass
        
        return self._python_fallback(equations, state_names, param_names, 
                                    param_values, t_span, x0, t_eval)
    
    def _python_fallback(self, equations, state_names, param_names, 
                        param_values, t_span, x0, t_eval):
        """Fallback using scipy."""
        try:
            from scipy.integrate import solve_ivp
            import sympy as sp
            
            # Build RHS
            syms = {name: sp.Symbol(name) for name in state_names + param_names}
            eqs = [sp.sympify(eq, locals=syms) for eq in equations]
            funcs = [sp.lambdify([sp.Symbol(s) for s in state_names] + [sp.Symbol(p) for p in param_names], eq, 'numpy') for eq in eqs]
            
            def rhs(t, y):
                args = list(y) + list(param_values)
                return [f(*args) for f in funcs]
            
            sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method='RK45')
            return {
                "success": sol.success,
                "message": sol.message,
                "nfev": sol.nfev,
                "t": sol.t,
                "y": sol.y,
                "kernel_path": "fallback"
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e),
                "nfev": 0,
                "t": np.array(t_eval),
                "y": np.zeros((len(state_names), len(t_eval))),
                "kernel_path": "fallback"
            }


_default_integrator = None

def get_fast_integrator() -> FastODEIntegrator:
    global _default_integrator
    if _default_integrator is None:
        _default_integrator = FastODEIntegrator()
    return _default_integrator


def fast_integrate(*args, **kwargs) -> dict[str, Any]:
    return get_fast_integrator().integrate(*args, **kwargs)
