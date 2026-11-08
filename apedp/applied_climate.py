"""APEDP Applied Climate Science & Carbon Capture Module.

Every number here is a published central estimate or an explicitly assumed
scaling; nothing is a climate or cost model. The return records say which key is
which, so a reference figure cannot be read as a projection.
"""

from __future__ import annotations

from typing import Any


class DirectAirCapture:
    @staticmethod
    def analyze_dac_technology(sorbent: str = "amine_based") -> dict[str, Any]:
        sorbents = {
            "amine_based": {"energy_kwh_tco2": 1500, "cost_usd_tco2": 600, "maturity": "commercial"},
            "solid_sorbent": {"energy_kwh_tco2": 1000, "cost_usd_tco2": 400, "maturity": "demonstration"},
            "alkaline_solution": {"energy_kwh_tco2": 2000, "cost_usd_tco2": 800, "maturity": "commercial"},
            "metal_organic_frameworks": {"energy_kwh_tco2": 800, "cost_usd_tco2": 300, "maturity": "research"},
        }
        info = sorbents.get(sorbent, sorbents["amine_based"])
        return {
            "sorbent_technology": sorbent,
            "energy_requirement_kwh_tCO2": info["energy_kwh_tco2"],
            "cost_usd_per_tCO2": info["cost_usd_tco2"],
            "technology_readiness_level": info["maturity"],
            "scalability": "high" if info["maturity"] == "commercial" else "medium",
            "path_to_cost_parity": "2030-2035" if info["cost_usd_tco2"] < 500 else "2035-2040",
            "co2_removal_potential_reference_gigatons_yr": 10,
            "co2_removal_potential_is_computed": False,
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
    def analyze_mineralization(mineral: str = "olivine") -> dict[str, Any]:
        minerals = {
            "olivine": {"co2_capacity_kg_kg": 0.5, "reactivity": "high", "availability": "abundant"},
            "basalt": {"co2_capacity_kg_kg": 0.1, "reactivity": "moderate", "availability": "very_abundant"},
            "serpentine": {"co2_capacity_kg_kg": 0.4, "reactivity": "moderate", "availability": "abundant"},
        }
        info = minerals.get(mineral, minerals["olivine"])
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
        projections = {
            "ssp119": {"2100_temp_increase_c": 1.0, "description": "Very low emissions"},
            "ssp126": {"2100_temp_increase_c": 1.5, "description": "Low emissions"},
            "ssp245": {"2100_temp_increase_c": 2.4, "description": "Intermediate"},
            "ssp370": {"2100_temp_increase_c": 3.6, "description": "High emissions"},
            "ssp585": {"2100_temp_increase_c": 4.4, "description": "Very high emissions"},
        }
        info = projections.get(scenario, projections["ssp245"])
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
            "evidence": "literature_lookup",
            "not_modelled": [
                "sea_level",
                "ice_sheet_dynamics",
                "regional_patterns",
                "carbon_cycle_feedback",
            ],
        }
