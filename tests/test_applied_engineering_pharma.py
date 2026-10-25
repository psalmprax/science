"""Tests for the applied biopharma, electrical, fusion and materials modules.

These check that outputs follow from inputs, match closed-form formulas, and do
not depend on Python's per-process hash seed.
"""

import math
import os
import subprocess
import sys

import pytest

from apedp.applied_biopharma import (
    ADMETPredictor,
    CombinationTrialSimulator,
    DrugRepurposingEngine,
)
from apedp.applied_electrical import (
    AnalogRF,
    ControlSystems,
    EmbeddedSystems,
    PowerElectronics,
    PowerSystems,
    Semiconductors,
)
from apedp.applied_fusion import FusionReactorDesigner, PlasmaPhysicsAnalyzer
from apedp.applied_materials import (
    CatalystDiscovery,
    EnergyStorageMaterials,
    MetamaterialDesigner,
    SuperconductorDiscovery,
)

# ---------- biopharma ----------

def test_admet_verdict_depends_on_inputs():
    good = ADMETPredictor.predict_properties(300, 2.0, 2, 4, 3, 60)
    poor = ADMETPredictor.predict_properties(600, 6.0, 6, 11, 12, 150)
    assert good["drug_likeness"] == "good"
    assert poor["drug_likeness"] == "poor"
    assert good["lipinski_violations"] == 0 and poor["lipinski_violations"] == 4
    assert poor["veber_violations"] == 2


def test_admet_tpsa_and_rotatable_bonds_now_affect_result():
    base = ADMETPredictor.predict_properties(300, 2.0, 2, 4, 3, 60)
    high_tpsa = ADMETPredictor.predict_properties(300, 2.0, 2, 4, 3, 160)
    assert base["drug_likeness"] != high_tpsa["drug_likeness"]


def test_repurposing_typo_fixed_and_alias_kept():
    cands = DrugRepurposingEngine.discover_repurposing_candidates("oncology")
    assert cands
    assert DrugRepurposingEngine.discover_repuposing_candidates("oncology") == cands


def test_combination_trial_is_deterministic_across_processes():
    code = (
        "from apedp.applied_biopharma import CombinationTrialSimulator as C;"
        "print(C.simulate_combination_trial(['A','B'],'oncology')['results']['combination_response_rate'])"
    )
    outs = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        outs.add(r.stdout.strip())
    assert len(outs) == 1


def test_combination_trial_no_p_value_claim():
    res = CombinationTrialSimulator.simulate_combination_trial(["A", "B"], "oncology")
    assert "p_value" not in res["results"]
    assert res["results"]["absolute_risk_reduction"] == pytest.approx(
        res["results"]["combination_response_rate"] - res["results"]["control_response_rate"], abs=1e-3
    )


def test_repurposing_efficacy_rejects_bad_mechanism_match():
    with pytest.raises(ValueError):
        CombinationTrialSimulator.simulate_repuposing_efficacy("X", "Y", mechanism_match=1.5)


# ---------- electrical ----------

def test_grid_fault_mva_and_drop_follow_inputs():
    a = PowerSystems.analyze_grid(source_impedance_pu=0.1)
    b = PowerSystems.analyze_grid(source_impedance_pu=0.2)
    assert a["three_phase_fault_mva"] == pytest.approx(1000.0)
    assert b["three_phase_fault_mva"] == pytest.approx(500.0)
    assert b["voltage_drop_pu"] > a["voltage_drop_pu"]


def test_grid_reactive_power_matches_power_factor():
    res = PowerSystems.analyze_grid(load_mw=100, load_pf=0.8, source_impedance_pu=0.1)
    q_pu = math.tan(math.acos(0.8))
    assert res["voltage_drop_pu"] == pytest.approx(q_pu * 0.1, rel=1e-5)


@pytest.mark.parametrize("ctype,expected", [("buck", 5 / 12), ("boost", 1 - 12 / 24), ("buck_boost", 5 / 17)])
def test_converter_duty_cycle_formulas(ctype, expected):
    v_in = 12.0
    v_out = 24.0 if ctype == "boost" else 5.0
    res = PowerElectronics.design_converter(ctype, v_in=v_in, v_out=v_out)
    if ctype == "buck_boost":
        assert res["duty_cycle"] == pytest.approx(v_out / (v_in + v_out), rel=1e-5)
    elif ctype == "boost":
        assert res["duty_cycle"] == pytest.approx(expected, rel=1e-5)
    else:
        assert res["duty_cycle"] == pytest.approx(expected, rel=1e-5)


def test_converter_rejects_unsupported_type():
    with pytest.raises(ValueError):
        PowerElectronics.design_converter("inverter")


def test_amplifier_noise_factor_from_nf():
    res = AnalogRF.design_circuit("amplifier", noise_figure_db=3.0)
    assert res["noise_factor"] == pytest.approx(10 ** 0.3, rel=1e-5)


def test_filter_attenuation_increases_with_order():
    lo = AnalogRF.design_circuit("filter", order=2)
    hi = AnalogRF.design_circuit("filter", order=6)
    assert hi["attenuation_at_2x_cutoff_db"] > lo["attenuation_at_2x_cutoff_db"]


def test_embedded_clock_period_from_clock():
    assert EmbeddedSystems.design_system(clock_mhz=100)["clock_period_ns"] == pytest.approx(10.0)
    assert EmbeddedSystems.design_system(clock_mhz=200)["clock_period_ns"] == pytest.approx(5.0)


def test_semiconductor_fom_is_product():
    res = Semiconductors.analyze_device(rdson_mohm=25, qg_nc=4)
    assert res["figure_of_merit_mohm_nc"] == pytest.approx(100.0)


def test_pid_gains_follow_ziegler_nichols():
    res = ControlSystems.design_controller(plant_gain=2.0, plant_time_constant_s=4.0, dead_time_s=0.5)
    kp = 1.2 * 4.0 / (2.0 * 0.5)
    assert res["kp"] == pytest.approx(kp)
    assert res["ki"] == pytest.approx(kp / (2 * 0.5))
    assert res["kd"] == pytest.approx(kp * 0.5 * 0.5)


# ---------- fusion ----------

def test_lawson_triple_product_and_units():
    res = PlasmaPhysicsAnalyzer.lawson_criterion(n_i=1e20, tau_e=3.0, temperature_keV=10.0)
    assert res["triple_product_m3_keV_s"] == pytest.approx(3e21)
    assert res["meets_ignition_reference"] is True


def test_beta_formula():
    B = 5.0
    p = 1e5
    res = PlasmaPhysicsAnalyzer.beta_limit(B, p)
    assert res["beta"] == pytest.approx(2 * (4e-7 * math.pi) * p / B ** 2, rel=1e-9)


def test_debye_length_scales_as_sqrt_t_over_n():
    a = PlasmaPhysicsAnalyzer.debye_length(1e19, 10.0)
    b = PlasmaPhysicsAnalyzer.debye_length(1e19, 40.0)
    assert b / a == pytest.approx(2.0, rel=1e-6)


def test_reactor_q_computed_from_powers():
    res = FusionReactorDesigner.design_reactor(power_output_MW=500, heating_power_MW=50)
    assert res["fusion_gain_Q"] == pytest.approx(10.0)
    res2 = FusionReactorDesigner.design_reactor(power_output_MW=500, heating_power_MW=100)
    assert res2["fusion_gain_Q"] == pytest.approx(5.0)


def test_confinement_optimizer_not_faked():
    with pytest.raises(NotImplementedError):
        FusionReactorDesigner.optimize_confinement({})


# ---------- materials ----------

def test_superconductor_classification_by_temperature():
    assert SuperconductorDiscovery.analyze_superconductor(93)["above_liquid_nitrogen_77K"] is True
    assert SuperconductorDiscovery.analyze_superconductor(39)["above_liquid_nitrogen_77K"] is False


def test_battery_energy_computed_from_capacity_and_voltage():
    res = EnergyStorageMaterials.analyze_battery_material(cathode_capacity_mAh_g=200, mean_voltage_V=3.7)
    # 200 mAh/g * 3.7 V = 740 mWh/g = 0.74 Wh/kg
    assert res["cathode_energy_Wh_kg"] == pytest.approx(0.74)


def test_cloak_wavelength_from_frequency():
    c = 299_792_458.0
    res = MetamaterialDesigner.design_cloak(frequency_hz=c, diameter_m=1.0)
    assert res["wavelength_m"] == pytest.approx(1.0, rel=1e-9)


def test_catalyst_unknown_reaction_raises():
    with pytest.raises(ValueError):
        CatalystDiscovery.design_catalyst("unknown")
