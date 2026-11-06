"""APEDP Applied Electrical & Electronics Engineering Module.

Closed-form engineering calculations. Each function computes its outputs from
its inputs using standard textbook formulas. Nothing is looked up from a
hardcoded result table.
"""

from __future__ import annotations

import math
from typing import Any


class PowerSystems:
    @staticmethod
    def analyze_grid(grid_type: str = "transmission", voltage_kv: float | None = None,
                     load_mw: float = 100.0, source_impedance_pu: float = 0.1,
                     load_pf: float = 0.9) -> dict[str, Any]:
        """Per-unit fault level and voltage-drop estimate for a single bus.

        Base: 100 MVA. Fault MVA = base / Z_source(pu); voltage drop uses the
        approximation dV ~= (P*R + Q*X) / V for the source impedance taken as
        purely reactive (X = Z_source), so dV_pu = Q_pu * X_pu.
        """
        if not 0.0 < load_pf <= 1.0:
            raise ValueError("load_pf must be in (0, 1]")
        if source_impedance_pu <= 0:
            raise ValueError("source_impedance_pu must be positive")
        if voltage_kv is None:
            voltage_kv = 230.0 if grid_type == "transmission" else 13.8
        base_mva = 100.0
        p_pu = load_mw / base_mva
        q_pu = p_pu * math.tan(math.acos(load_pf))
        fault_mva = base_mva / source_impedance_pu
        voltage_drop_pu = q_pu * source_impedance_pu
        return {
            "grid_type": grid_type,
            "voltage_kv": voltage_kv,
            "frequency_hz": 60,
            "load_mw": load_mw,
            "load_pf": load_pf,
            "source_impedance_pu": source_impedance_pu,
            "three_phase_fault_mva": round(fault_mva, 3),
            "voltage_drop_pu": round(voltage_drop_pu, 6),
            "voltage_regulation_ok": voltage_drop_pu <= 0.05,
            "note": "Single-bus approximation; not a load-flow solution.",
        }


class PowerElectronics:
    @staticmethod
    def design_converter(converter_type: str = "buck", v_in: float = 12.0, v_out: float = 5.0,
                         i_out: float = 2.0, f_sw_khz: float = 100.0,
                         inductance_uh: float = 10.0) -> dict[str, Any]:
        """Ideal-duty-cycle and ripple for DC-DC converters (CCM, ideal switches).

        buck: D = Vout/Vin; boost: D = 1 - Vin/Vout; buck_boost: D = Vout/(Vin+Vout).
        Inverter is not modelled here and raises ValueError.
        """
        if converter_type == "buck":
            duty = v_out / v_in
        elif converter_type == "boost":
            duty = 1.0 - v_in / v_out
        elif converter_type == "buck_boost":
            duty = v_out / (v_in + v_out)
        else:
            raise ValueError(f"unsupported converter_type: {converter_type}")
        if not 0.0 < duty < 1.0:
            raise ValueError("operating point outside the converter's duty-cycle range")
        f_sw = f_sw_khz * 1e3
        L = inductance_uh * 1e-6
        if converter_type == "buck":
            delta_i = (v_in - v_out) * duty / (L * f_sw)
        elif converter_type == "boost":
            delta_i = v_in * duty / (L * f_sw)
        else:
            delta_i = v_in * duty / (L * f_sw)
        return {
            "converter_type": converter_type,
            "duty_cycle": round(duty, 6),
            "v_in": v_in, "v_out": v_out, "i_out": i_out,
            "switching_frequency_khz": f_sw_khz,
            "inductor_ripple_current_a": round(delta_i, 6),
            "control_method": "PWM",
            "note": "Ideal-switch CCM estimate; efficiency not modelled.",
        }


class AnalogRF:
    @staticmethod
    def design_circuit(circuit_type: str = "amplifier", gain_db: float = 20.0,
                       bandwidth_mhz: float = 100.0, noise_figure_db: float = 2.0,
                       center_freq_mhz: float = 1000.0, cutoff_mhz: float = 100.0,
                       order: int = 4) -> dict[str, Any]:
        """Closed-form RF/analog figures.

        amplifier: gain is passed through and the noise factor (linear) of this
        single stage is derived from the noise figure. filter: Butterworth magnitude at the
        cutoff and the stopband attenuation at 2x cutoff. oscillator: Leeson
        phase-noise is not modelled, so only the frequency is reported.
        """
        if circuit_type == "amplifier":
            f = 10 ** (noise_figure_db / 10)
            return {
                "circuit_type": circuit_type,
                "gain_db": gain_db,
                "bandwidth_mhz": bandwidth_mhz,
                "noise_figure_db": noise_figure_db,
                "noise_factor": round(f, 6),
                "simulation_method": "closed-form",
            }
        if circuit_type == "filter":
            if order < 1 or cutoff_mhz <= 0:
                raise ValueError("filter order and cutoff must be positive")
            atten_2x_db = 10 * math.log10(1 + (2.0) ** (2 * order))
            return {
                "circuit_type": circuit_type,
                "order": order,
                "cutoff_mhz": cutoff_mhz,
                "attenuation_at_2x_cutoff_db": round(atten_2x_db, 3),
                "simulation_method": "closed-form",
            }
        if circuit_type == "oscillator":
            return {
                "circuit_type": circuit_type,
                "frequency_mhz": center_freq_mhz,
                "simulation_method": "closed-form",
                "note": "Phase noise not modelled.",
            }
        raise ValueError(f"unsupported circuit_type: {circuit_type}")


class EmbeddedSystems:
    @staticmethod
    def design_system(mcu_family: str = "ARM_Cortex_M", clock_mhz: float = 168.0,
                      flash_kb: float = 512.0, ram_kb: float = 192.0,
                      rtos: str | None = None) -> dict[str, Any]:
        """Echoes the caller-supplied resources and derives the clock period.

        This does not know any real MCU's specifications; the caller supplies them.
        """
        if clock_mhz <= 0 or flash_kb <= 0 or ram_kb <= 0:
            raise ValueError("clock, flash and RAM must be positive")
        return {
            "mcu_family": mcu_family,
            "clock_mhz": clock_mhz,
            "flash_kb": flash_kb,
            "ram_kb": ram_kb,
            "rtos": rtos,
            "clock_period_ns": round(1e3 / clock_mhz, 6),
        }


class Semiconductors:
    @staticmethod
    def analyze_device(device_type: str = "MOSFET", rdson_mohm: float = 10.0,
                       qg_nc: float = 20.0) -> dict[str, Any]:
        """Conduction-loss and gate-charge figure of merit from caller-supplied values.

        FOM = Rds(on) * Qg (mOhm * nC). Lower is better for a given voltage class.
        """
        if rdson_mohm <= 0 or qg_nc <= 0:
            raise ValueError("rdson_mohm and qg_nc must be positive")
        return {
            "device_type": device_type,
            "rdson_mohm": rdson_mohm,
            "qg_nc": qg_nc,
            "figure_of_merit_mohm_nc": round(rdson_mohm * qg_nc, 3),
        }


class ControlSystems:
    @staticmethod
    def design_controller(controller_type: str = "PID", plant_gain: float = 1.0,
                          plant_time_constant_s: float = 1.0, dead_time_s: float = 0.1) -> dict[str, Any]:
        """Ziegler-Nichols PID tuning from a first-order-plus-dead-time (FOPDT) model.

        Ti = 2L, Td = 0.5L, Kp = 1.2 * tau / (K * L) (Ziegler-Nichols reaction curve).
        Only PID is supported; other types raise ValueError.
        """
        if controller_type != "PID":
            raise ValueError("only PID tuning is implemented")
        if plant_gain == 0 or plant_time_constant_s <= 0 or dead_time_s <= 0:
            raise ValueError("plant gain must be nonzero and time constants positive")
        L = dead_time_s
        tau = plant_time_constant_s
        kp = 1.2 * tau / (plant_gain * L)
        ti = 2.0 * L
        td = 0.5 * L
        return {
            "controller_type": controller_type,
            "tuning_method": "Ziegler_Nichols_reaction_curve",
            "kp": round(kp, 6),
            "ki": round(kp / ti, 6),
            "kd": round(kp * td, 6),
        }


# Aliases
PowerSystemDesigner = PowerSystems
PowerElectronicsDesigner = PowerElectronics
RFAnalogDesigner = AnalogRF
EmbeddedSystemsDesigner = EmbeddedSystems
SemiconductorDesigner = Semiconductors
ControlSystemsDesigner = ControlSystems
ElectronicsDesign = PowerElectronics
