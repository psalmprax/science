"""APEDP Applied Fusion Energy & Plasma Physics Module.

Closed-form plasma and reactor relations. SI units unless a name says otherwise.
Reactor-level outputs are first-order estimates from the inputs given; they are
not design results.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

MU0 = 4e-7 * math.pi
EPS0 = 8.854e-12
E_CHARGE = 1.602e-19
K_B = 1.381e-23


class PlasmaPhysicsAnalyzer:
    """Plasma physics analysis and confinement metrics."""

    @staticmethod
    def lawson_criterion(n_i: float, tau_e: float, temperature_keV: float) -> dict[str, Any]:
        """Fusion triple product n * T * tau_E.

        Units: n_i in m^-3, temperature in keV, tau_e in s, so the product is in
        m^-3 keV s. The ignition reference (~3e21 m^-3 keV s) is the commonly
        quoted D-T value for ignition-level triple product.
        """
        if n_i <= 0 or tau_e <= 0 or temperature_keV <= 0:
            raise ValueError("n_i, tau_e and temperature must be positive")
        triple_product = n_i * temperature_keV * tau_e
        threshold = 3e21
        return {
            "triple_product_m3_keV_s": triple_product,
            "ignition_reference_m3_keV_s": threshold,
            "meets_ignition_reference": triple_product >= threshold,
            "margin": triple_product / threshold,
            "basis": "closed_form",
        }

    @staticmethod
    def beta_limit(B_field_T: float, pressure_Pa: float) -> dict[str, Any]:
        """Plasma beta = 2 mu0 p / B^2 (ratio of plasma to magnetic pressure)."""
        if B_field_T <= 0:
            raise ValueError("B_field_T must be positive")
        b_pressure = (B_field_T ** 2) / (2 * MU0)
        beta = pressure_Pa / b_pressure
        # Rough reference only: the Troyon limit is beta_N = beta[%] * a * B / I, not a fixed beta.
        troyon_limit = 0.04
        return {
            "beta": beta,
            "beta_percent": beta * 100,
            "troyon_limit_reference": troyon_limit,
            "within_reference_limit": beta < troyon_limit,
            "margin_to_reference": (troyon_limit - beta) / troyon_limit,
            "basis": "closed_form",
        }

    @staticmethod
    def debye_length(n_e: float, temperature_eV: float) -> float:
        """Electron Debye length lambda_D = sqrt(eps0 kT / (n e^2)), in metres.

        Temperature is converted from eV to K with 1 eV = 11604.5 K.
        """
        if n_e <= 0 or temperature_eV <= 0:
            raise ValueError("n_e and temperature must be positive")
        T_K = temperature_eV * 11604.5
        return math.sqrt(EPS0 * K_B * T_K / (n_e * E_CHARGE ** 2))


class FusionReactorDesigner:
    """Reactor concept lookup plus first-order power balance."""

    REACTOR_CONCEPTS: ClassVar[dict[str, dict[str, Any]]] = {
        'tokamak': {'type': 'magnetic_confinement', 'maturity': 'high'},
        'stellarator': {'type': 'magnetic_confinement', 'maturity': 'medium'},
        'spherical_tokamak': {'type': 'magnetic_confinement', 'maturity': 'medium'},
        'inertial_confinement': {'type': 'inertial', 'maturity': 'medium'},
        'field_reversed_configuration': {'type': 'magnetic', 'maturity': 'low'},
        'z_pinch': {'type': 'magnetic', 'maturity': 'low'},
    }

    @classmethod
    def design_reactor(cls, concept: str = 'tokamak', power_output_MW: float = 500,
                       fusion_power_MW: float | None = None, heating_power_MW: float = 50.0) -> dict[str, Any]:
        """Fusion gain Q = P_fusion / P_heating, computed from the given powers.

        If fusion_power_MW is not supplied, it is set to the target output, so
        Q is the gain the heating system would need for that output.
        """
        if concept not in cls.REACTOR_CONCEPTS:
            raise ValueError(f"unknown concept: {concept}")
        if heating_power_MW <= 0:
            raise ValueError("heating_power_MW must be positive")
        config = cls.REACTOR_CONCEPTS[concept]
        p_fus = power_output_MW if fusion_power_MW is None else fusion_power_MW
        q = p_fus / heating_power_MW
        return {
            'concept': concept,
            'type': config['type'],
            'target_power_MW': power_output_MW,
            'fusion_power_MW': p_fus,
            'heating_power_MW': heating_power_MW,
            'fusion_gain_Q': round(q, 3),
            'fuel': 'D-T',
            'tritium_breeding_required': True,
            'note': 'Power-balance ratio only; no plasma transport or engineering model.',
            'basis': 'closed_form',
        }

    @classmethod
    def optimize_confinement(cls, parameters: dict[str, Any]) -> dict[str, Any]:
        """Not implemented. Raises instead of returning fixed 'optimal' values."""
        raise NotImplementedError("confinement optimisation requires a transport model; not implemented")


class FusionFuelCycle:
    """Tritium fuel cycle and sustainability."""

    @staticmethod
    def tritium_breeding_ratio(breeder_material: str = 'LiPb', neutron_multiplier: str = 'Be') -> dict[str, Any]:
        """Looks up literature-typical TBR values for named breeders (not computed)."""
        tbr_values = {'LiPb': 1.15, 'Li4SiO4': 1.05, 'FLiBe': 1.12, 'Li2TiO3': 1.08}
        if breeder_material not in tbr_values:
            raise ValueError(f"no reference TBR for breeder: {breeder_material}")
        tbr = tbr_values[breeder_material]
        return {
            'breeder_material': breeder_material,
            'tritium_breeding_ratio_reference': tbr,
            'self_sufficient_reference': tbr > 1.0,
            'margin': tbr - 1.0,
            'source': 'literature-typical value, not computed',
            'basis': 'literature_lookup',
        }

    @staticmethod
    def deuterium_availability() -> dict[str, Any]:
        """Deuterium abundance: ~1 D atom per 6420 H atoms in seawater (literature value)."""
        ratio = 1 / 6420
        return {
            'source': 'Seawater (heavy water)',
            'd_to_h_atom_ratio': ratio,
            'basis': 'literature_lookup',
        }


class FusionMaterials:
    """Plasma-facing material reference data (literature-typical, not computed)."""

    MATERIALS_DB: ClassVar[dict[str, dict[str, Any]]] = {
        'tungsten': {'melting_point_K': 3695, 'thermal_conductivity_W_mK': 173},
        'beryllium': {'melting_point_K': 1560, 'thermal_conductivity_W_mK': 200},
        'carbon_fiber_composite': {'melting_point_K': 3800, 'thermal_conductivity_W_mK': 400},
        'liquid_lithium': {'melting_point_K': 453, 'thermal_conductivity_W_mK': 85},
    }

    @classmethod
    def evaluate_pfm(cls, material: str = 'tungsten') -> dict[str, Any]:
        if material not in cls.MATERIALS_DB:
            raise ValueError(f"no reference data for material: {material}")
        return {'material': material, **cls.MATERIALS_DB[material],
                'source': 'literature-typical values', 'basis': 'literature_lookup'}
