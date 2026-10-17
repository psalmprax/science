# APEDP-Continuum: Autonomous High-Dimensional Continuous PDE & Coupled ODE Discovery Platform

`apedp-continuum` is a standalone, lightweight scientific discovery engine for continuous ordinary and partial differential equations across arbitrary spatial dimensions ($1\text{D}$ to $100\text{D}+$):

- **Continuous PDE Discovery ($1\text{D}$ to $100\text{D}+$):**
  - **Eulerian Cartesian Grids ($D < 10$):** 4th-order central difference stencils and Sequentially Thresholded Ridge Regression (STRidge).
  - **Mesh-Free Point Cloud Sampling ($D \ge 10$):** Scattered RBF Gaussian collocation with exact analytical $D$-dimensional Laplacians and axis-star finite difference probe stencils.
  - **Hyperspherical Radial Symmetry Reduction:** Exact $O(1)$ dimension-invariant radial coordinates.
  - **Tensor-Train / Matrix Product States (TT / MPS):** Low-rank tensor core contractions bypassing $O(N^D)$ exponential grid scaling.
  - **Neural PINNs / Deep BSDE:** Fused Jet2/Jet3 Taylor-mode automatic differentiation.
- **Continuous Non-Linear Coupled ODE Systems:**
  - Automated multi-variable dynamical system recovery (e.g. Lotka-Volterra, FitzHugh-Nagumo, SIR epidemiology).
  - 1st-order rational ODE discovery via Differential Padé and Jet algebra.
- **Strict Epistemic Referee Adjudication:**
  - Formal 5-verdict epistemic taxonomy (`IDENTIFIED_CAUSAL_LAW`, `PREDICTIVE_WRONG_STRUCTURE`, etc.).
  - Out-of-sample frozen model-transfer $R^2$ validation.
  - Second-law thermodynamic arrow of time verification ($\alpha > 0$).
  - Residual whiteness autocorrelation screening ($|\rho_1| < 0.50$).

---

## Installation

```bash
# Direct local pip install
pip install ./packages/apedp-continuum

# Editable development mode
pip install -e ./packages/apedp-continuum
```

---

## Quickstart

### 1. Spatio-Temporal PDE Discovery
```python
from apedp_continuum import discover_pde, PdeGridData

# Discover 2D Heat Diffusion from grid u(t, x, y)
res = discover_pde(u_grid, dt=0.01, dxs=0.05)
print(f"Discovered PDE: {res.formula}")
print(f"Active Terms: {res.active_terms}, Coefficients: {res.coefficients}")
```

### 2. High-Dimensional Mesh-Free Point Clouds (20D to 100D+)
```python
from apedp_continuum import discover_pde, PdePointCloudData

# Discover 20D diffusion from scattered point cloud
cloud = PdePointCloudData.from_axis_star(u_anc, u_plus, u_minus, dt=0.005, h=0.04, dimension=20)
res = discover_pde(cloud)
print(f"Discovered {res.formula} with holdout R^2: {res.r2_dev:.6f}")
```

### 3. Coupled Non-Linear Dynamical Systems (ODEs)
```python
from apedp_continuum import discover_coupled_ode

# Discover Lotka-Volterra predator-prey dynamics
data = {"x": prey_series, "y": predator_series}
res = discover_coupled_ode(data, dt=0.05)
for eq in res.equations:
    print(f"d{eq.state_variable}/dt = {eq.formula} (R^2 = {eq.r2:.4f})")
```
