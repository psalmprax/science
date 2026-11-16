"""APEDP Applied Climate Science & Carbon Capture Module.

Every number here is a published central estimate or an explicitly assumed
scaling; nothing is a climate or cost model. The return records say which key is
which, so a reference figure cannot be read as a projection.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict


class _SorbentSpec(TypedDict):
    energy_kwh_tco2: int
    cost_usd_tco2: int
    maturity: str


class _MineralSpec(TypedDict):
    co2_capacity_kg_kg: float
    reactivity: str
    availability: str


_ScenarioSpec = TypedDict(
    "_ScenarioSpec",
    {"2100_temp_increase_c": float, "description": str},
)


class DirectAirCapture:
    @staticmethod
    def analyze_dac_technology(
        sorbent: str = "amine_based",
        capture_scale_tco2_yr: float | None = None,
    ) -> dict[str, Any]:
        sorbents: dict[str, _SorbentSpec] = {
            "amine_based": {"energy_kwh_tco2": 1500, "cost_usd_tco2": 600, "maturity": "commercial"},
            "solid_sorbent": {"energy_kwh_tco2": 1000, "cost_usd_tco2": 400, "maturity": "demonstration"},
            "alkaline_solution": {"energy_kwh_tco2": 2000, "cost_usd_tco2": 800, "maturity": "commercial"},
            "metal_organic_frameworks": {"energy_kwh_tco2": 800, "cost_usd_tco2": 300, "maturity": "research"},
        }
        if sorbent not in sorbents:
            raise ValueError(f"unknown sorbent {sorbent!r}; expected one of {sorted(sorbents)}")
        if capture_scale_tco2_yr is not None and (
            not math.isfinite(capture_scale_tco2_yr) or capture_scale_tco2_yr <= 0
        ):
            raise ValueError("capture_scale_tco2_yr must be finite and positive")
        info = sorbents[sorbent]
        # Minimum isothermal Gibbs unmixing work for 400 ppm CO2 at 298.15 K
        x_co2 = 400e-6
        r_gas = 8.314462618
        temp_k = 298.15
        molar_mass_co2 = 0.04401  # kg/mol
        min_j_mol = -(r_gas * temp_k / x_co2) * (
            x_co2 * math.log(x_co2) + (1.0 - x_co2) * math.log(1.0 - x_co2)
        )
        thermo_min_kwh_tco2 = (min_j_mol / molar_mass_co2) / 3600.0
        second_law_eff = thermo_min_kwh_tco2 / info["energy_kwh_tco2"]
        extra_scale = (
            {"annual_energy_MWh": round(capture_scale_tco2_yr * info["energy_kwh_tco2"] / 1000.0, 3)}
            if capture_scale_tco2_yr is not None
            else {}
        )
        return {
            "sorbent_technology": sorbent,
            "energy_requirement_kwh_tCO2": info["energy_kwh_tco2"],
            "thermodynamic_minimum_kwh_tCO2": round(thermo_min_kwh_tco2, 2),
            "second_law_efficiency": round(second_law_eff, 4),
            "cost_usd_per_tCO2": info["cost_usd_tco2"],
            "technology_readiness_level": info["maturity"],
            "scalability": "high" if info["maturity"] == "commercial" else "medium",
            "path_to_cost_parity": "2030-2035" if info["cost_usd_tco2"] < 500 else "2035-2040",
            "co2_removal_potential_reference_gigatons_yr": 10,
            "co2_removal_potential_is_computed": False,
            **extra_scale,
            "basis": "reference_spec",
            "evidence": "reference_spec",
            "not_modelled": [
                "lifecycle_assessment",
                "cost_curve",
                "deployment_rate",
                "co2_removal_potential",
            ],
        }


class CarbonMineralization:
    @staticmethod
    def analyze_mineralization(
        mineral: str = "olivine",
        rock_mass_kg: float | None = None,
    ) -> dict[str, Any]:
        minerals: dict[str, _MineralSpec] = {
            "olivine": {"co2_capacity_kg_kg": 0.5, "reactivity": "high", "availability": "abundant"},
            "basalt": {"co2_capacity_kg_kg": 0.1, "reactivity": "moderate", "availability": "very_abundant"},
            "serpentine": {"co2_capacity_kg_kg": 0.4, "reactivity": "moderate", "availability": "abundant"},
        }
        if mineral not in minerals:
            raise ValueError(f"unknown mineral {mineral!r}; expected one of {sorted(minerals)}")
        if rock_mass_kg is not None and (not math.isfinite(rock_mass_kg) or rock_mass_kg < 0):
            raise ValueError("rock_mass_kg must be finite and non-negative")
        info = minerals[mineral]
        extra_stoich = (
            {"stoichiometric_co2_uptake_kg": round(rock_mass_kg * info["co2_capacity_kg_kg"], 3)}
            if rock_mass_kg is not None
            else {}
        )
        return {
            "mineral": mineral,
            "co2_sequestration_capacity_kg_per_kg": info["co2_capacity_kg_kg"],
            "reactivity": info["reactivity"],
            "global_availability": info["availability"],
            "permanence": "permanent",
            "energy_requirement": "low",
            "cost_usd_tco2_reference": 50,
            "cost_usd_tco2_is_computed": False,
            "scale_potential_gigatons_yr": 2,
            **extra_stoich,
            "basis": "literature_lookup",
            "evidence": "literature_lookup",
            "not_modelled": [
                "cost_usd_tco2",
                "reaction_kinetics",
                "lifecycle_assessment",
                "scale_potential",
            ],
        }


class ClimateModeling:
    @staticmethod
    def project_temperature(scenario: str = "ssp245") -> dict[str, Any]:
        projections: dict[str, _ScenarioSpec] = {
            "ssp119": {"2100_temp_increase_c": 1.0, "description": "Very low emissions"},
            "ssp126": {"2100_temp_increase_c": 1.5, "description": "Low emissions"},
            "ssp245": {"2100_temp_increase_c": 2.4, "description": "Intermediate"},
            "ssp370": {"2100_temp_increase_c": 3.6, "description": "High emissions"},
            "ssp585": {"2100_temp_increase_c": 4.4, "description": "Very high emissions"},
        }
        if scenario not in projections:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {sorted(projections)}")
        info = projections[scenario]
        warming = info["2100_temp_increase_c"]
        return {
            "scenario": scenario,
            **info,
            "sea_level_rise_assumed_m": warming * 0.2,
            "sea_level_rise_is_modelled": False,
            "sea_level_rise_basis": (
                "assumed linear scaling of 0.2 m per degree of 2100 warming; "
                "not a calibrated ice-sheet model"
            ),
            "risk_level": "low" if warming < 2 else "medium" if warming < 3 else "high",
            "tipping_points_at_risk": warming > 1.5,
            "basis": "literature_lookup",
            "evidence": "literature_lookup",
            "not_modelled": [
                "sea_level",
                "ice_sheet_dynamics",
                "regional_patterns",
                "carbon_cycle_feedback",
            ],
        }
