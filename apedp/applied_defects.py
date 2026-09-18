"""APEDP Applied Science Equation Defect Auditor & Regularization Engine.

Systematically captures physical, chemical, material, and nanoscale equation defects
where textbook mathematical models break down when deployed in real-world applications:
1. Applied Physics: Aerodynamics, continuum breakdown (Knudsen), compressibility singularities.
2. Applied Chemistry: Electrochemical overpotentials, mass-transfer limits, thermal runaways.
3. Applied Material Science: Hall-Petch nanoscale softening, crack tip singularities, creep limits.
4. Applied Nanotechnology: Gibbs-Thomson negative melting, Casimir stiction, quantum resistance.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar

from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine


class AppliedPhysicsDefectAuditor:
    """Audits and regularizes applied physics continuum, aerodynamic, and thermodynamic equations."""

    @staticmethod
    def audit_prandtl_glauert_compressibility(
        mach_number: float,
        cp_incompressible: float = -0.5,
        boundary_relief_eps: float = 0.04,
        airfoil_thickness_ratio: float | None = None
    ) -> dict[str, Any]:
        """Audits Prandtl-Glauert aerodynamic compressibility rule: Cp = Cp_0 / sqrt(1 - M^2).

        Defect: Diverges to infinity as Mach M -> 1 (transonic singularity), predicting infinite lift/drag.
        Regularization: Karman-Tsien rule or Van Dyke transonic softening: Cp_soft = Cp_0 / ((1 - M^2)^2 + eps^2)^(1/4).
        Boundary Relief Parameter eps:
            - eps=0.15 gives significant pre-shock operational distortion (-3.9% at M=0.8, -11.4% at M=0.9).
            - eps=0.04 (calibrated) drops distortion to <0.2% at M=0.8 and 1.1% at M=0.9, while keeping finite transonic ceiling.
            - If airfoil_thickness_ratio tau is provided, eps = tau^(2/3) (transonic similarity parameter scaling).
        """
        is_singular = mach_number >= 1.0 or abs(1.0 - mach_number) < 1e-4
        defects: list[str] = []

        if is_singular:
            defects.append(
                f"TRANSONIC_SINGULARITY: Mach number M={mach_number:.3f} causes division by zero in "
                f"Prandtl-Glauert factor 1/sqrt(1 - M^2). Real aerodynamics undergoes shock wave formation."
            )

        # ``float(...)`` here is not cosmetic: mypy types ``float ** float`` as
        # ``Any``, so an un-converted ``eps`` would propagate an unchecked value
        # into the distortion metrics below (and out of an annotated ``-> float``).
        if airfoil_thickness_ratio is not None and airfoil_thickness_ratio > 0:
            eps = float(airfoil_thickness_ratio ** (2.0 / 3.0))
        else:
            eps = float(boundary_relief_eps)

        # Regularized model: Cp_reg = Cp_0 / ((1 - M^2)^2 + eps^2)^(1/4)
        denom_sq = (1.0 - mach_number**2)**2 + eps**2
        cp_soft = cp_incompressible / (denom_sq ** 0.25)

        # Operational pre-singularity distortion metrics at M = 0.8 and M = 0.9
        def _calc_distortion(m: float) -> float:
            c_class = cp_incompressible / math.sqrt(max(1e-12, 1.0 - m**2))
            # Same ``Any``-typed ``**`` as above; coerce before the subtraction so
            # the return is a checked float rather than an unchecked value.
            c_reg = float(cp_incompressible / (((1.0 - m**2)**2 + eps**2) ** 0.25))
            return abs(c_reg - c_class) / abs(c_class)

        distortion_m08 = _calc_distortion(0.8)
        distortion_m09 = _calc_distortion(0.9)

        return {
            'mach_number': mach_number,
            'cp_incompressible': cp_incompressible,
            'boundary_relief_eps': eps,
            'defect_detected': is_singular,
            'defects': defects,
            'regularized_cp': cp_soft,
            'distortion_at_m08': distortion_m08,
            'distortion_at_m09': distortion_m09,
            'recommended_regularization': 'Van Dyke Transonic Softening: Cp = Cp_0 / ((1 - M^2)^2 + eps^2)^(1/4)'
        }

    @staticmethod
    def audit_knudsen_continuum_breakdown(
        mean_free_path_m: float,
        characteristic_length_m: float
    ) -> dict[str, Any]:
        """Audits Navier-Stokes no-slip continuum validity via Knudsen number Kn = lambda / L.

        Defect: For Kn > 0.01 (slip flow) and Kn > 0.1 (transition / ballistic flow in micro/nano channels),
        continuum Navier-Stokes and standard Hagen-Poiseuille equations fail completely.
        """
        if characteristic_length_m <= 0:
            raise ValueError("Characteristic length L must be strictly positive.")

        knudsen_number = mean_free_path_m / characteristic_length_m
        is_continuum_valid = knudsen_number < 0.01
        is_slip_flow = 0.01 <= knudsen_number < 0.1
        is_free_molecular = knudsen_number >= 0.1

        defects: list[str] = []
        if is_slip_flow:
            defects.append(
                f"SLIP_FLOW_REGIME: Knudsen number Kn={knudsen_number:.3e} exceeds no-slip threshold (0.01). "
                "Navier-Stokes requires Maxwell-Smoluchowski slip velocity boundary condition."
            )
        elif is_free_molecular:
            defects.append(
                f"CONTINUUM_BREAKDOWN: Knudsen number Kn={knudsen_number:.3e} exceeds continuum limit (0.1). "
                "Continuum Navier-Stokes fails. Requires Boltzmann kinetic transport / DSMC."
            )

        # Poiseuille correction factor (Knudsen correction): f_slip = 1 + 6 * C * Kn
        poiseuille_permeability_correction = 1.0 + 6.0 * knudsen_number

        return {
            'mean_free_path_m': mean_free_path_m,
            'characteristic_length_m': characteristic_length_m,
            'knudsen_number': knudsen_number,
            'defect_detected': not is_continuum_valid,
            'defects': defects,
            'regime': 'Continuum' if is_continuum_valid else ('Slip Flow' if is_slip_flow else 'Transitional/Ballistic'),
            'poiseuille_permeability_correction': poiseuille_permeability_correction
        }


class AppliedChemistryDefectAuditor:
    """Audits chemical kinetics, mass transport, and electrochemistry equations for real-world limits."""

    @staticmethod
    def audit_butler_volmer_electrochemical_saturation(
        overpotential_volts: float,
        exchange_current_density_a_m2: float = 10.0,
        diffusion_limiting_current_a_m2: float = 1000.0,
        temperature_kelvin: float = 298.15,
        alpha_transfer: float = 0.5
    ) -> dict[str, Any]:
        """Audits Butler-Volmer electrochemical current density: j = j_0 * [exp(alpha*F*eta/RT) - exp(-(1-alpha)*F*eta/RT)].

        Defect: Predicts exponential current density j -> inf as overpotential eta increases,
        ignoring Fickian diffusion limit (mass-transport concentration polarization).
        """
        F = 96485.33212  # Faraday constant, C/mol
        R = 8.314462618   # Gas constant, J/(mol*K)
        f_factor = F / (R * temperature_kelvin)

        # Raw Butler-Volmer current density
        eta = overpotential_volts
        j_raw = exchange_current_density_a_m2 * (
            math.exp(min(50.0, alpha_transfer * f_factor * eta))
            - math.exp(max(-50.0, -(1.0 - alpha_transfer) * f_factor * eta))
        )

        exceeds_mass_transfer = abs(j_raw) > diffusion_limiting_current_a_m2
        defects: list[str] = []
        if exceeds_mass_transfer:
            defects.append(
                f"MASS_TRANSPORT_VIOLATION: Raw Butler-Volmer current ({j_raw:.2e} A/m^2) exceeds Fickian "
                f"diffusion limiting current ({diffusion_limiting_current_a_m2:.2e} A/m^2). "
                "Electrode surface concentration of reactants has dropped to zero."
            )

        # Regularized model: mixed kinetic-diffusion mass transfer (Koutecky-Levich saturation)
        # 1 / j_total = 1 / j_BV + 1 / j_lim
        if j_raw >= 0:
            j_soft = (j_raw * diffusion_limiting_current_a_m2) / (diffusion_limiting_current_a_m2 + j_raw)
        else:
            j_soft = (j_raw * diffusion_limiting_current_a_m2) / (diffusion_limiting_current_a_m2 - j_raw)

        return {
            'overpotential_volts': overpotential_volts,
            'raw_current_density_a_m2': j_raw,
            'diffusion_limit_a_m2': diffusion_limiting_current_a_m2,
            'defect_detected': exceeds_mass_transfer,
            'defects': defects,
            'regularized_current_density_a_m2': j_soft,
            'recommended_regularization': 'Koutecky-Levich Mass Transfer Saturation: j = (j_BV * j_lim) / (j_lim + |j_BV|)'
        }

    @staticmethod
    def audit_semenov_thermal_runaway(
        heat_of_reaction_joules: float,
        activation_energy_joules_per_mole: float,
        reactor_temp_kelvin: float,
        coolant_temp_kelvin: float,
        heat_transfer_coefficient_ua: float,
        reaction_rate_mol_per_sec: float
    ) -> dict[str, Any]:
        """Audits Semenov criterion for exothermic chemical reactor thermal explosion.

        Defect: When d(Q_generated)/dT > d(Q_removed)/dT, stable pseudo-steady state equations fail;
        reactor enters exponential thermal runaway.
        """
        R = 8.314462618
        T = reactor_temp_kelvin
        Tc = coolant_temp_kelvin

        q_gen = abs(heat_of_reaction_joules) * reaction_rate_mol_per_sec
        q_rem = heat_transfer_coefficient_ua * (T - Tc)

        # Semenov derivatives with respect to temperature T
        dq_gen_dt = q_gen * (activation_energy_joules_per_mole / (R * (T ** 2)))
        dq_rem_dt = heat_transfer_coefficient_ua

        is_thermal_runaway = dq_gen_dt > dq_rem_dt
        defects: list[str] = []
        if is_thermal_runaway:
            ratio = dq_gen_dt / dq_rem_dt
            defects.append(
                f"SEMENOV_THERMAL_RUNAWAY: Heat generation rate slope ({dq_gen_dt:.2f} W/K) exceeds heat removal slope "
                f"({dq_rem_dt:.2f} W/K) by {ratio:.2f}x. Steady-state kinetic equations are unstable; reactor will explode."
            )

        # Critical maximum cooling temperature difference: Delta_T_crit = R * T^2 / Ea
        delta_t_crit = (R * (T ** 2)) / activation_energy_joules_per_mole

        return {
            'reactor_temp_k': T,
            'coolant_temp_k': Tc,
            'heat_generated_watts': q_gen,
            'heat_removed_watts': q_rem,
            'dq_gen_dt': dq_gen_dt,
            'dq_rem_dt': dq_rem_dt,
            'defect_detected': is_thermal_runaway,
            'defects': defects,
            'critical_temperature_rise_k': delta_t_crit,
            'recommended_remediation': 'Limit feed rate, increase cooling UA, or operate below Semenov critical temperature.'
        }


class AppliedMaterialScienceDefectAuditor:
    """Audits structural materials, fracture mechanics, and metallurgy equations."""

    MATERIAL_PROFILES: ClassVar[dict[str, dict[str, float]]] = {
        "ferritic_steel": {
            "sigma_0_friction_stress_pa": 70e6,
            "k_hall_petch_pa_m_half": 0.74e6,
            "critical_grain_size_meters": 10e-9,
            "theoretical_shear_strength_pa": 3.0e9,
        },
        "copper_annealed": {
            "sigma_0_friction_stress_pa": 25e6,
            "k_hall_petch_pa_m_half": 0.36e6,
            "critical_grain_size_meters": 15e-9,
            "theoretical_shear_strength_pa": 1.5e9,
        },
        "titanium_pure": {
            "sigma_0_friction_stress_pa": 120e6,
            "k_hall_petch_pa_m_half": 0.40e6,
            "critical_grain_size_meters": 8e-9,
            "theoretical_shear_strength_pa": 4.0e9,
        },
    }

    @classmethod
    def audit_hall_petch_nanoscale_breakdown(
        cls,
        grain_size_meters: float,
        sigma_0_friction_stress_pa: float = 70e6,
        k_hall_petch_pa_m_half: float = 0.74e6,
        critical_grain_size_meters: float = 10e-9,
        theoretical_shear_strength_pa: float = 3e9,
        material_name: str | None = None
    ) -> dict[str, Any]:
        """Audits Hall-Petch equation: sigma_y = sigma_0 + k_HP / sqrt(d).

        Defect: As d -> 0, sigma_y -> inf.
        Physical Reality: Below ~10 nm, dislocations can no longer pile up; grain boundary sliding sets in
        causing Inverse Hall-Petch softening, and strength can never exceed theoretical lattice shear strength (~G/10).
        Concrete material parameters can be calibrated via material_name (e.g. 'ferritic_steel', 'copper_annealed', 'titanium_pure').
        """
        d = grain_size_meters
        if d <= 0:
            raise ValueError("Grain size d must be strictly positive.")

        if material_name is not None:
            norm_name = material_name.lower().replace(" ", "_").replace("-", "_")
            if norm_name in cls.MATERIAL_PROFILES:
                prof = cls.MATERIAL_PROFILES[norm_name]
                sigma_0_friction_stress_pa = prof["sigma_0_friction_stress_pa"]
                k_hall_petch_pa_m_half = prof["k_hall_petch_pa_m_half"]
                critical_grain_size_meters = prof["critical_grain_size_meters"]
                theoretical_shear_strength_pa = prof["theoretical_shear_strength_pa"]

        sigma_raw = sigma_0_friction_stress_pa + (k_hall_petch_pa_m_half / math.sqrt(d))

        is_nanoscale_breakdown = d < critical_grain_size_meters
        exceeds_theoretical_shear = sigma_raw > theoretical_shear_strength_pa

        defects: list[str] = []
        if is_nanoscale_breakdown:
            defects.append(
                f"INVERSE_HALL_PETCH_BREAKDOWN: Grain size d={d*1e9:.2f} nm is below critical dislocation pile-up "
                f"cutoff ({critical_grain_size_meters*1e9:.1f} nm). Grain boundary sliding occurs; material softens."
            )
        if exceeds_theoretical_shear:
            defects.append(
                f"THEORETICAL_LATTICE_STRENGTH_VIOLATION: Predicted yield strength ({sigma_raw/1e9:.2f} GPa) exceeds "
                f"theoretical atomic shear limit ({theoretical_shear_strength_pa/1e9:.2f} GPa)."
            )

        # Regularized model: Smooth transition from Hall-Petch (d > dc) to Inverse Hall-Petch (d < dc)
        # sigma_reg = sigma_0 + (k_HP / sqrt(d + d0)) * [1 / (1 + (dc / d)^2)]
        d_eff = max(d, 1e-11)
        softening_factor = 1.0 / (1.0 + (critical_grain_size_meters / d_eff) ** 2)
        sigma_soft = sigma_0_friction_stress_pa + (k_hall_petch_pa_m_half / math.sqrt(d_eff + 1e-10)) * softening_factor
        sigma_bounded = min(sigma_soft, theoretical_shear_strength_pa)

        return {
            'grain_size_m': d,
            'material_profile': material_name,
            'raw_hall_petch_yield_stress_pa': sigma_raw,
            'defect_detected': is_nanoscale_breakdown or exceeds_theoretical_shear,
            'defects': defects,
            'regularized_yield_stress_pa': sigma_bounded,
            'theoretical_limit_respected': sigma_bounded <= theoretical_shear_strength_pa,
            'recommended_regularization': 'Constitutive Inverse Hall-Petch Transition: sigma = sigma_0 + (k / sqrt(d)) * [1 / (1 + (dc/d)^2)]'
        }

    @staticmethod
    def audit_griffith_crack_tip_singularity(
        distance_from_crack_tip_meters: float,
        stress_intensity_factor_k1: float = 30e6,
        yield_strength_pa: float = 500e6
    ) -> dict[str, Any]:
        """Audits linear elastic fracture mechanics (LEFM) crack tip stress: sigma = K_I / sqrt(2*pi*r).

        Defect: Blows up to infinity as r -> 0.
        Physical Reality: Real materials yield at sigma_yield; plastic zone (Irwin/Dugdale) redistributes stress.
        """
        r = distance_from_crack_tip_meters
        if r <= 0:
            raise ValueError("Distance from crack tip r must be strictly positive.")

        sigma_raw = stress_intensity_factor_k1 / math.sqrt(2.0 * math.pi * r)
        exceeds_yield = sigma_raw > yield_strength_pa

        defects: list[str] = []
        if exceeds_yield:
            defects.append(
                f"CRACK_TIP_SINGULARITY: Linear elastic stress ({sigma_raw/1e6:.1f} MPa) exceeds yield strength "
                f"({yield_strength_pa/1e6:.1f} MPa) at r={r*1e6:.1f} um. LEFM assumptions violated; plastic zone forms."
            )

        # Irwin Plastic Zone Radius: r_p = (1 / (2 * pi)) * (K_I / sigma_yield)^2
        plastic_zone_radius_m = (1.0 / (2.0 * math.pi)) * ((stress_intensity_factor_k1 / yield_strength_pa) ** 2)

        # Bounded Dugdale cohesive stress: sigma_soft = min(sigma_raw, yield_strength)
        sigma_soft = min(sigma_raw, yield_strength_pa)

        return {
            'distance_r_m': r,
            'raw_stress_pa': sigma_raw,
            'yield_strength_pa': yield_strength_pa,
            'plastic_zone_radius_m': plastic_zone_radius_m,
            'defect_detected': exceeds_yield,
            'defects': defects,
            'regularized_stress_pa': sigma_soft,
            'recommended_regularization': 'Irwin/Dugdale Cohesive Zone Regularization: sigma = min(K_I / sqrt(2*pi*r), sigma_yield)'
        }


class AppliedNanotechnologyDefectAuditor:
    """Audits nanoscale thermodynamics, Casimir stiction, and quantum transport equations."""

    @staticmethod
    def audit_gibbs_thomson_melting_depression(
        nanoparticle_radius_meters: float,
        bulk_melting_temp_kelvin: float = 1337.0,  # Gold (Au)
        solid_liquid_surface_energy_j_m2: float = 0.27,
        solid_density_kg_m3: float = 19300.0,
        latent_heat_of_fusion_j_kg: float = 64000.0
    ) -> dict[str, Any]:
        """Audits Gibbs-Thomson melting temperature depression: Delta_T_m / T_m0 = (2 * gamma_sl) / (rho_s * L_f * r).

        Defect: As r -> 0, Delta_T_m -> inf, predicting melting temperature T_m < 0 K (violates 3rd law of thermodynamics!).
        Physical Reality: At r < 1 nm, atomic clusters form discrete magic-number shells; T_m saturates above 0 K.
        """
        r = nanoparticle_radius_meters
        if r <= 0:
            raise ValueError("Nanoparticle radius r must be strictly positive.")

        constant_c = (2.0 * solid_liquid_surface_energy_j_m2) / (solid_density_kg_m3 * latent_heat_of_fusion_j_kg)
        delta_tm_raw = bulk_melting_temp_kelvin * (constant_c / r)
        predicted_tm_raw = bulk_melting_temp_kelvin - delta_tm_raw

        violates_third_law = predicted_tm_raw <= 0.0
        is_subnanometer = r < 1e-9
        defects: list[str] = []
        if violates_third_law:
            defects.append(
                f"UNPHYSICAL_NEGATIVE_TEMPERATURE: Gibbs-Thomson predicts melting point T_m={predicted_tm_raw:.1f} K "
                f"<= 0 K for nanoparticle radius r={r*1e9:.2f} nm. Violates the Third Law of Thermodynamics."
            )
        elif is_subnanometer:
            defects.append(
                f"NANOCLUSTER_DISCRETE_SHELL_BREAKDOWN: Cluster radius r={r*1e9:.2f} nm is in atomic shell regime. "
                "Continuum Gibbs-Thomson thermodynamic relations fail."
            )

        # Regularized model: Smooth exponential decay or atomic core offset r0 (e.g. atomic radius ~0.3 nm)
        # T_m,soft(r) = T_m0 * exp(-C / (r + r0))
        r0 = 3e-10  # Atomic cluster core cutoff (0.3 nm)
        predicted_tm_soft = bulk_melting_temp_kelvin * math.exp(-constant_c / (r + r0))

        return {
            'radius_m': r,
            'bulk_melting_temp_k': bulk_melting_temp_kelvin,
            'raw_predicted_melting_temp_k': predicted_tm_raw,
            'defect_detected': len(defects) > 0,
            'defects': defects,
            'regularized_melting_temp_k': predicted_tm_soft,
            'recommended_regularization': 'Atomic Shell Cluster Exponential Envelope: T_m = T_m0 * exp(-C / (r + r0))'
        }

    @staticmethod
    def audit_casimir_nems_stiction_divergence(
        separation_distance_meters: float,
        plate_area_m2: float = 1e-8,
        hamaker_constant_joules: float = 1e-19,
        interatomic_cutoff_z0_meters: float = 3e-10
    ) -> dict[str, Any]:
        """Audits Casimir / van der Waals attraction in NEMS switches: F_vdW = (A_H * Area) / (6 * pi * z^3).

        Defect: Blows up to infinity as gap z -> 0, predicting infinite stiction force and unphysical collapse.
        Physical Reality: At sub-nanometer gaps, electron cloud Pauli repulsion (Born steric repulsion) arrests the collapse.
        Regularization: Integrated Lennard-Jones 12-6 / Born steric cutoff balance:
            F_reg = [A_H * Area / (6 * pi * (z + z0)^3)] * [1 - (z0 / (z + z0))^6].
        """
        z = separation_distance_meters
        if z < 0:
            raise ValueError("Separation distance z must be non-negative.")

        # Raw van der Waals attraction force
        f_vdw_raw = (hamaker_constant_joules * plate_area_m2) / (6.0 * math.pi * (max(z, 1e-18) ** 3))

        # At z < interatomic cutoff, unphysical continuum divergence
        is_stiction_singularity = z < interatomic_cutoff_z0_meters
        defects: list[str] = []
        if is_stiction_singularity:
            defects.append(
                f"CASIMIR_STICTION_DIVERGENCE: Sub-nanometer gap z={z*1e9:.3f} nm causes continuum 1/z^3 blowup "
                f"({f_vdw_raw:.2e} N). Ignores Born steric atomic repulsion."
            )

        # Regularized Lennard-Jones style potential force:
        # Conservative surface potential and coordinate normal force convention:
        # Potential: U(z) = (A_H * Area / (12*pi)) * [-1/(z+z0)^2 + z0^6 / (4*(z+z0)^8)]
        # Coordinate normal force: F_z(z) = -dU/dz = [A_H * Area / (6*pi*(z+z0)^3)] * [(z0/(z+z0))^6 - 1]
        # For z > 0: F_z < 0 (attractive force directed inward towards contact)
        # For z < 0: F_z > 0 (mathematical continuation / effective repulsive branch representing surrogate penetration coordinate)
        # At contact z = 0: F_z(0) = 0 (contact force equilibrium)
        # Restoring contact stiffness:
        # k_eff(0) = -dF_z/dz(0) = d^2U/dz^2(0) = (A_H * Area) / (pi * z0^4) > 0.
        # Attractive force magnitude: F_attraction = -F_z = [A_H * Area / (6*pi*(z+z0)^3)] * [1 - (z0/(z+z0))^6]
        z0 = interatomic_cutoff_z0_meters
        z_eff = z + z0
        f_attraction_soft = (hamaker_constant_joules * plate_area_m2) / (6.0 * math.pi * (z_eff ** 3))
        repulsion_balance = max(0.0, 1.0 - (z0 / z_eff) ** 6)
        f_net_soft = f_attraction_soft * repulsion_balance
        k_eff_contact = (hamaker_constant_joules * plate_area_m2) / (math.pi * (z0 ** 4))
        is_contact_stable = k_eff_contact > 0.0

        return {
            'separation_z_m': z,
            'raw_attractive_force_n': f_vdw_raw,
            'defect_detected': is_stiction_singularity,
            'defects': defects,
            'regularized_force_n': f_net_soft,
            'attractive_envelope_n': f_attraction_soft,
            'equilibrium_contact_arrest': (z == 0.0 or f_net_soft < f_attraction_soft),
            'contact_stiffness_n_per_m': k_eff_contact,
            'potential_curvature_d2u_dz2': k_eff_contact,
            'contact_mechanical_stability': is_contact_stable,
            'recommended_regularization': 'Bounded short-range continuation: F_attract = [A_H * Area / (6 * pi * (z + z0)^3)] * [1 - (z0 / (z + z0))^6]'
        }


class UniversalAppliedDefectEngine:
    """Unified multi-domain auditor running cross-disciplinary defect scans on applied science models."""

    def __init__(self, assurance_system: ClosedLoopAssuranceSystem | None = None):
        if assurance_system is None:
            engine = DimensionalAssuranceEngine()
            assurance_system = ClosedLoopAssuranceSystem(engine)
        self.system = assurance_system

    def audit_applied_equation(
        self,
        domain: str,
        equation_name: str,
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Runs the specialized domain auditor on a proposed applied science equation."""
        domain_norm = domain.lower()
        eq_norm = equation_name.lower()

        if any(w in domain_norm for w in ["physics", "aero", "fluid"]):
            if any(w in eq_norm for w in ["prandtl", "glauert", "compress", "mach"]) or any(k in inputs for k in ["mach_number", "mach"]):
                return AppliedPhysicsDefectAuditor.audit_prandtl_glauert_compressibility(
                    mach_number=inputs.get('mach_number', inputs.get('mach', 1.0)),
                    cp_incompressible=inputs.get('cp_incompressible', -0.5),
                    boundary_relief_eps=inputs.get('boundary_relief_eps', 0.04),
                    airfoil_thickness_ratio=inputs.get('airfoil_thickness_ratio', None)
                )
            if any(w in eq_norm for w in ["knudsen", "continuum", "navier"]) or any(k in inputs for k in ["mean_free_path_m", "knudsen"]):
                return AppliedPhysicsDefectAuditor.audit_knudsen_continuum_breakdown(
                    mean_free_path_m=inputs.get('mean_free_path_m', 68e-9),
                    characteristic_length_m=inputs.get('characteristic_length_m', 50e-9)
                )

        elif any(w in domain_norm for w in ["chem", "electro", "reaction"]):
            if any(w in eq_norm for w in ["butler", "volmer", "electro"]) or any(k in inputs for k in ["overpotential_volts", "overpotential"]):
                return AppliedChemistryDefectAuditor.audit_butler_volmer_electrochemical_saturation(
                    overpotential_volts=inputs.get('overpotential_volts', inputs.get('overpotential', 0.8)),
                    exchange_current_density_a_m2=inputs.get('exchange_current_density_a_m2', 10.0),
                    diffusion_limiting_current_a_m2=inputs.get('diffusion_limiting_current_a_m2', 500.0)
                )
            if any(w in eq_norm for w in ["semenov", "runaway", "thermal", "reactor"]):
                return AppliedChemistryDefectAuditor.audit_semenov_thermal_runaway(
                    heat_of_reaction_joules=inputs.get('heat_of_reaction_joules', -100000.0),
                    activation_energy_joules_per_mole=inputs.get('activation_energy_joules_per_mole', 80000.0),
                    reactor_temp_kelvin=inputs.get('reactor_temp_kelvin', 380.0),
                    coolant_temp_kelvin=inputs.get('coolant_temp_kelvin', 300.0),
                    heat_transfer_coefficient_ua=inputs.get('heat_transfer_coefficient_ua', 250.0),
                    reaction_rate_mol_per_sec=inputs.get('reaction_rate_mol_per_sec', 0.05)
                )

        elif any(w in domain_norm for w in ["material", "metallurgy", "solid"]):
            if any(w in eq_norm for w in ["hall", "petch", "grain"]) or any(k in inputs for k in ["grain_size_meters", "grain"]):
                return AppliedMaterialScienceDefectAuditor.audit_hall_petch_nanoscale_breakdown(
                    grain_size_meters=inputs.get('grain_size_meters', inputs.get('grain', 5e-9)),
                    material_name=inputs.get('material_name', None)
                )
            if any(w in eq_norm for w in ["griffith", "crack", "fracture"]) or any(k in inputs for k in ["distance_from_crack_tip_meters", "crack"]):
                return AppliedMaterialScienceDefectAuditor.audit_griffith_crack_tip_singularity(
                    distance_from_crack_tip_meters=inputs.get('distance_from_crack_tip_meters', inputs.get('crack', 1e-7))
                )

        elif any(w in domain_norm for w in ["nano", "quantum"]):
            if any(w in eq_norm for w in ["gibbs", "thomson", "melting"]) or any(k in inputs for k in ["nanoparticle_radius_meters", "radius"]):
                return AppliedNanotechnologyDefectAuditor.audit_gibbs_thomson_melting_depression(
                    nanoparticle_radius_meters=inputs.get('nanoparticle_radius_meters', inputs.get('radius', 0.3e-9))
                )
            if any(w in eq_norm for w in ["casimir", "stiction", "vdw"]) or any(k in inputs for k in ["separation_distance_meters", "separation"]):
                return AppliedNanotechnologyDefectAuditor.audit_casimir_nems_stiction_divergence(
                    separation_distance_meters=inputs.get('separation_distance_meters', inputs.get('separation', 1e-10)),
                    plate_area_m2=inputs.get('plate_area_m2', 1e-8),
                    hamaker_constant_joules=inputs.get('hamaker_constant_joules', 1e-19),
                    interatomic_cutoff_z0_meters=inputs.get('interatomic_cutoff_z0_meters', 3e-10)
                )

        raise ValueError(f"Unknown applied domain '{domain}' or equation specification '{equation_name}'")
