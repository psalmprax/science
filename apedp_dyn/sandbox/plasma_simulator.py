"""Synthetic plasma thermal sandbox for APEDP research.

Models a simplified tokamak plasma energy balance:
    dT/dt = P_heat - P_loss(T, n, B)
    dn/dt = S_fuel - L(n, T)

where:
    P_loss = alpha * T^(7/2) / (n * tau_E)  (Spitzer-like conduction)
    tau_E = confinement time
    n = plasma density
    T = temperature (keV)

The generating relationships are HIDDEN from APEDP.
APEDP only sees: (T, n, B, P_heat) -> sigma_y (analogous to some material property
or plasma response).

Truth used for synthesis (not exposed to APEDP):
    alpha = 1.0, n0 = 1.0e20, B0 = 2.5, tau0 = 1.0
    P_loss = alpha * T^(7/2) / (n * tau_E)
    tau_E = tau0 * (T/T0)^(3/2) * (n0/n)^(1/2) * (B/B0)^(-1)  # scaling

APEDP observes noisy measurements of the system response.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# === Physical constants (fixed, not exposed to APEDP) ===
ALPHA = 1.0       # loss coefficient
TAU0 = 1.0        # baseline confinement time [s]
N0 = 1.0e20       # baseline density [m^-3]
B0 = 2.5          # baseline magnetic field [T]
T0 = 1.0          # baseline temperature [keV]
# Power law exponents in the TRUE generating model:
GAMMA_T = 3.5     # T exponent in P_loss
GAMMA_n = -1.0    # n exponent in P_loss  
GAMMA_B = -1.0    # B exponent in confinement time

@dataclass
class PlasmaState:
    """Plasma state at a given time."""
    T: float       # temperature [keV]
    n: float       # density [m^-3]
    B: float       # magnetic field [T]
    t: float       # time [s]

@dataclass
class PlasmaConfig:
    """Configuration driving the plasma."""
    P_heat: float   # heating power [MW]
    duration: float # simulation duration [s]
    dt: float       # timestep [s]
    noise_sigma: float = 0.05  # measurement noise level (relative)

def _P_loss(T: float, n: float, **kwargs) -> float:
    """True power loss (hidden from APEDP).
    
    P_loss = ALPHA * T^(gamma_T) / (n * tau_E)
    """
    # True confinement time scaling
    tau_E = TAU0 * (T/T0)**1.5 * (N0/n)**0.5 * (B0/kwargs.get('B', B0))**(-1)
    return float(ALPHA * T**GAMMA_T / (n * tau_E))

def _dTdt(T: float, n: float, P_heat: float, **kwargs) -> float:
    """Temperature derivative per the true model."""
    return P_heat - _P_loss(T, n, **kwargs)

def _dn_dt(T: float, n: float, **kwargs) -> float:
    """Density derivative (simplified fueling/loss)."""
    # Simple fueling balanced by thermal loss
    S_fuel = 0.1 * N0 / 10.0  # slow fueling
    L = 0.01 * n / max(T, 0.1)  # loss proportional to density, inverse T
    return S_fuel - L

def simulate_plasma(config: PlasmaConfig,
                    state0: PlasmaState | None = None) -> tuple[list[PlasmaState], np.ndarray]:
    """Simulate plasma evolution.
    
    Returns:
        states: list of PlasmaState at each timestep
        observations: noisy measurements [T_meas, n_meas, B, P_fus_meas]
    """
    if state0 is None:
        state0 = PlasmaState(T=0.5, n=N0, B=B0, t=0.0)
    
    dt = config.dt
    t_final = config.duration
    n_steps = int(t_final / dt) + 1
    
    states = []
    observations = np.zeros((n_steps, 4))  # T, n, B, P_fus
    
    T, n, B, t = state0.T, state0.n, state0.B, state0.t
    
    nrng = np.random.default_rng(42)  # fixed seed for reproducibility
    
    for i in range(n_steps):
        # Store state
        states.append(PlasmaState(T=T, n=n, B=B, t=t))
        
        # Compute true fusion power (hidden from APEDP but included in observations)
        P_fus_true = max(0.0, 1e-3 * n * T**2)  # simplified Lawson-like
        
        # Noisy observations APEDP would see
        T_meas = T + nrng.normal(0, config.noise_sigma * max(T, 0.1))
        n_meas = n + nrng.normal(0, config.noise_sigma * max(n, N0*1e-5))
        B_meas = B + nrng.normal(0, config.noise_sigma * max(B, B0*1e-3))
        P_fus_meas = P_fus_true + nrng.normal(0, config.noise_sigma * max(P_fus_true, 1e-3))
        
        observations[i] = [T_meas, n_meas, B_meas, P_fus_meas]
        
        # Advance timestep using true derivatives
        dT = _dTdt(T, n, config.P_heat) * dt
        dn = _dn_dt(T, n) * dt
        
        T = max(0.01, T + dT)   # temperature can't go negative
        n = max(1e-5, n + dn)   # density can't go negative
        B = B  # constant field for simplicity
        t += dt
    
    return states, observations

def generate_synthetic_dataset(n_conditions: int = 20,
                              noise_levels: tuple[float, ...] = (0.03, 0.05, 0.10)) -> dict:
    """Generate multiple synthetic datasets with varying noise levels."""
    datasets = {}
    
    for noise_sigma in noise_levels:
        config = PlasmaConfig(
            P_heat=1.5,       # 1.5 MW heating
            duration=20.0,    # 20 seconds
            dt=0.05,
            noise_sigma=noise_sigma
        )
        
        # Different initial conditions
        state0 = PlasmaState(
            T=0.5 + 0.3 * np.random.random(),
            n=(1.0 + 0.3 * np.random.random()) * N0,
            B=(2.5 + 0.3 * np.random.random()),
            t=0.0
        )
        
        states, obs = simulate_plasma(config, state0)
        
        datasets[f"noise_{noise_sigma}"] = {
            'config': config,
            'states': states,
            'observations': obs,
            'noise_sigma': noise_sigma,
            'n_points': len(states)
        }
    
    return datasets
