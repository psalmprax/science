"""APEDP platform integrity audit suite.

Runs dimensional and unit-consistency tests against a ClosedLoopAssuranceSystem instance.
"""


import logging

import sympy as sp

from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine

logger = logging.getLogger(__name__)


def build_demo_platform() -> tuple[DimensionalAssuranceEngine, ClosedLoopAssuranceSystem, sp.Expr, sp.Expr]:
    """Build a demo platform with Einstein energy and Newton gravity models."""
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)

    m_phys = engine.register_variable('m_phys', 'kilogram')
    engine.register_variable('v_phys', 'meter / second')
    r_dist = engine.register_variable('r_dist', 'meter')
    m_target = engine.register_variable('m_target', 'kilogram')

    # Einstein energy: E = m c^2
    einstein_expr = m_phys * sp.Symbol('c_light')**2
    platform.register_model_spec("Einstein_Energy", einstein_expr, "joule")

    # Newton gravity: F = G m1 m2 / r^2
    grav_force_expr = sp.Symbol('G_grav') * m_phys * m_target / (r_dist**2)
    platform.register_model_spec("Newton_Gravity", grav_force_expr, "newton")

    # Ideal gas law: n = P V / (R T)
    P_press = engine.register_variable('P_press', 'pascal')
    V_vol = engine.register_variable('V_vol', 'meter**3')
    T_temp = engine.register_variable('T_temp', 'kelvin')
    R_gas = sp.Symbol('R_gas')

    ideal_moles_expr = (P_press * V_vol) / (R_gas * T_temp)
    platform.register_model_spec("Ideal_Gas_Moles", ideal_moles_expr, "mole")

    return engine, platform, grav_force_expr, einstein_expr


def run_platform_audit_suite() -> None:
    """Run audit tests and print outcomes."""
    _engine, platform_assurance, grav_force_expr, einstein_expr = build_demo_platform()
    failures = 0
    passes = 0

    # Test 1: Detect dimensional violation (Force + Energy)
    print("[Test 1] Registering illegal equation (Force + Energy)...")
    try:
        illegal_expr = grav_force_expr + einstein_expr
        platform_assurance.register_model_spec("Illegal_Force_Energy_Add", illegal_expr, "newton")
        print("-> FAIL: System accepted adding incompatible dimensions (Force + Energy).")
        failures += 1
    except TypeError as e:
        print(f"-> PASS: System correctly blocked invalid operation. Error caught: {e}")
        passes += 1

    # Test 2: Unit mismatch on model definition
    print("\n[Test 2] Registering model with mismatched expected output unit...")
    try:
        platform_assurance.register_model_spec("Mismatched_Einstein_Moles", einstein_expr, "mole")
        print("-> FAIL: System accepted Joules as Moles.")
        failures += 1
    except TypeError as e:
        print(f"-> PASS: System correctly blocked mismatched return unit. Error caught: {e}")
        passes += 1

    # Test 3: Numerical edge case (near-vacuum pressure)
    print("\n[Test 3] Chemistry evaluation with near-vacuum pressure value...")
    try:
        vacuum_inputs = {
            'P_press': 1e-10,  # Pa
            'V_vol': 1.0,      # m^3
            'T_temp': 300.0,   # K
        }
        vacuum_moles = platform_assurance.evaluate_model("Ideal_Gas_Moles", vacuum_inputs)
        print(f"-> PASS: Solved vacuum scenario. Moles in vacuum: {vacuum_moles:.4e}")
        passes += 1
    except Exception as e:  # noqa: BLE001 - the audit reports failures, it does not raise
        logger.warning("Boundary value calculation failed", exc_info=True)
        print(f"-> FAIL: Exception during boundary value calculation: {e}")
        failures += 1

    print("\n=== AUDIT RESULTS ===")
    print(f"Passed Tests: {passes}")
    print(f"Failed Tests: {failures}")
    assert failures == 0, f"Audit suite had {failures} failures!"


def test_platform_audit_suite() -> None:
    """Pytest entrypoint for platform integrity audit suite."""
    run_platform_audit_suite()


if __name__ == '__main__':
    run_platform_audit_suite()

