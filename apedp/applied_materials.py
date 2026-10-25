"""APEDP Applied Materials Science Module.

Computed screening helpers and literature reference data. Nothing here performs
materials discovery or first-principles prediction.
"""

from __future__ import annotations

from typing import Any, ClassVar

ROOM_TEMP_K = 293.0


class SuperconductorDiscovery:
    """Classify a superconductor by critical temperature and reference data."""

    REFERENCE_CANDIDATES: ClassVar[list[dict[str, Any]]] = [
        {'material': 'LaH10', 'Tc_K': 250, 'pressure_GPa': 170, 'type': 'hydride'},
        {'material': 'YBCO', 'Tc_K': 93, 'pressure_GPa': 0, 'type': 'cuprate'},
        {'material': 'MgB2', 'Tc_K': 39, 'pressure_GPa': 0, 'type': 'conventional'},
        {'material': 'H3S', 'Tc_K': 203, 'pressure_GPa': 150, 'type': 'hydride'},
    ]

    @staticmethod
    def analyze_superconductor(critical_temp_K: float, critical_field_T: float = 0,
                               type_sc: str = 'conventional') -> dict[str, Any]:
        """Classifies by temperature only. Does not predict Tc or assess feasibility."""
        if critical_temp_K < 0:
            raise ValueError("critical_temp_K must be non-negative")
        return {
            'critical_temperature_K': critical_temp_K,
            'critical_field_T': critical_field_T,
            'type': type_sc,
            'above_liquid_nitrogen_77K': critical_temp_K > 77,
            'above_room_temperature_reference': critical_temp_K > ROOM_TEMP_K,
            'cooper_pair_mechanism_reference': 'BCS' if type_sc == 'conventional' else 'unconventional',
        }

    @classmethod
    def search_candidates(cls) -> list[dict[str, Any]]:
        """Returns a fixed list of literature reference materials. Not a search."""
        return [dict(c) for c in cls.REFERENCE_CANDIDATES]


class CatalystDiscovery:
    """Literature-typical catalyst materials per reaction (reference data only)."""

    REFERENCE: ClassVar[dict[str, list[str]]] = {
        'water_splitting': ['Pt', 'IrO2', 'NiFe-LDH'],
        'co2_reduction': ['Cu', 'Ag', 'Au'],
        'ammonia_synthesis': ['Fe', 'Ru'],
        'oxygen_reduction': ['Pt/C', 'Fe-N-C'],
    }

    @staticmethod
    def design_catalyst(reaction: str, target_activity: float = 0.9) -> dict[str, Any]:
        """Returns reference materials for a reaction. No activity is predicted."""
        if reaction not in CatalystDiscovery.REFERENCE:
            raise ValueError(f"no reference catalysts for reaction: {reaction}")
        materials = CatalystDiscovery.REFERENCE[reaction]
        return {
            'reaction': reaction,
            'reference_catalysts': list(materials),
            'target_activity': target_activity,
            'cost_flag_precious_metal': any(m.startswith(('Pt', 'Ir', 'Ru', 'Au', 'Ag')) for m in materials),
            'source': 'literature reference list, not computed',
        }


class EnergyStorageMaterials:
    """Reference energy density from specific capacity and voltage, computed."""

    @staticmethod
    def analyze_battery_material(cathode_capacity_mAh_g: float = 200.0, anode_capacity_mAh_g: float = 372.0,
                                 mean_voltage_V: float = 3.7, cycle_life: int = 1000) -> dict[str, Any]:
        """Theoretical gravimetric energy from the cathode capacity and mean voltage.

        Energy (Wh/kg) = capacity (mAh/g) * voltage (V) / 1000. Anode capacity is
        used only as a reported input; it does not limit the cathode-based value.
        """
        if cathode_capacity_mAh_g <= 0 or mean_voltage_V <= 0:
            raise ValueError("capacity and voltage must be positive")
        energy_wh_kg = cathode_capacity_mAh_g * mean_voltage_V / 1000.0
        return {
            'cathode_capacity_mAh_g': cathode_capacity_mAh_g,
            'anode_capacity_mAh_g': anode_capacity_mAh_g,
            'mean_voltage_V': mean_voltage_V,
            'cathode_energy_Wh_kg': round(energy_wh_kg, 3),
            'cycle_life_input': cycle_life,
            'note': 'Active-material energy only; excludes packaging and electrolyte.',
        }


class MetamaterialDesigner:
    """Transformation-optics cloak sizing from wavelength (closed form)."""

    @staticmethod
    def design_cloak(frequency_hz: float = 1e9, diameter_m: float = 0.1) -> dict[str, Any]:
        """Reports the wavelength and size ratio d / lambda. No cloaking efficiency is claimed."""
        if frequency_hz <= 0 or diameter_m <= 0:
            raise ValueError("frequency and diameter must be positive")
        c = 299_792_458.0
        wavelength = c / frequency_hz
        return {
            'operating_frequency_Hz': frequency_hz,
            'target_size_m': diameter_m,
            'wavelength_m': wavelength,
            'diameter_over_wavelength': diameter_m / wavelength,
            'note': 'Geometric ratio only; cloaking performance is not modelled.',
        }
