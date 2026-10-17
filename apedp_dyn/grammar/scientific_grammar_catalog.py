"""APEDP Scientific Grammar Space: Physics, Chemistry, and Biology.

Comprehensive domain-specific discovery grammar implementing:
  1. Shared Scientific Discovery Core:
     - Causal data, candidate equations, uncertainty, stability, abstention, provenance.
     - Shared operators: algebraic, nonlinear, derivatives, integration, spatial gradients,
       delays, thresholds, Hill/Michaelis-Menten saturation, network couplings.
  2. APEDP-PHYSICS:
     - Mechanics, Thermodynamics, Electromagnetism, Waves & Oscillations,
       Fluids & Transport, Materials, Quantum & Astrophysics.
     - Physics Validity Gate: SI dimensional homogeneity, conservation laws, symmetries.
  3. APEDP-CHEMISTRY:
     - Composition & Concentration, Reaction Kinetics, Chemical Thermodynamics,
       Electrochemistry, Molecular Descriptors, Spectroscopy, Catalysis.
     - Chemistry Validity Gate: Stoichiometric consistency, mass/atom balance,
       non-negativity of concentrations, thermodynamic bounds (dG <= 0).
  4. APEDP-BIOLOGY:
     - Molecular & Gene Regulation, Enzyme Kinetics, Cell Biology, Physiology,
       Microbiology & Epidemiology, Ecology & Populations, Evolution & Genetics,
       Neuroscience, Developmental & Spatial Biology, Omics.
     - Biology Validity Gate: Biological context required (organism/cell type),
       carrying capacity bounds, delay positivity, non-negative populations.
  5. Typed Schema & Fingerprint:
     - Machine-readable ScientificVariableSpec matching Spec §6.
     - ScientificGrammarCatalog query engine and SHA-256 integrity verification.
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Scientific Domain, Family, and Type Enums
# ─────────────────────────────────────────────────────────────────────────────

class ScientificDomain(str, enum.Enum):
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ASTROPHYSICS = "astrophysics"
    MATHEMATICS = "mathematics"


class ScientificVariableKind(str, enum.Enum):
    MEASURED_STATE = "measured_state"
    DERIVED_STATE = "derived_state"
    PARAMETER = "parameter"
    ENVIRONMENTAL_INPUT = "environmental_input"
    LATENT_STATE = "latent_state"
    BOUNDARY_CONDITION = "boundary_condition"


class ValueType(str, enum.Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"
    FIELD = "field"
    NETWORK_NODE = "network_node"
    CATEGORICAL_STATE = "categorical_state"


class PhysicsFamily(str, enum.Enum):
    MECHANICS_GEOMETRY = "mechanics_geometry"
    MECHANICS_MOTION = "mechanics_motion"
    MECHANICS_INERTIA = "mechanics_inertia"
    MECHANICS_FORCE_MOMENTUM = "mechanics_force_momentum"
    MECHANICS_ENERGY_WORK = "mechanics_energy_work"
    MECHANICS_CONTACT_DEFORMATION = "mechanics_contact_deformation"
    THERMODYNAMICS = "thermodynamics"
    STATISTICAL_PHYSICS = "statistical_physics"
    ELECTROMAGNETISM = "electromagnetism"
    CIRCUITS_OSCILLATORS = "circuits_oscillators"
    WAVES_OPTICS_OSCILLATIONS = "waves_optics_oscillations"
    FLUIDS_TRANSPORT = "fluids_transport"
    MATERIALS_SOLIDS = "materials_solids"
    QUANTUM_ATOMIC_ASTRO = "quantum_atomic_astro"
    QUANTUM_MODERN = "quantum_modern"
    ASTROPHYSICS_GRAVITATION = "astrophysics_gravitation"


class ChemistryFamily(str, enum.Enum):
    COMPOSITION_CONCENTRATION = "composition_concentration"
    REACTION_STATE = "reaction_state"
    REACTION_PROGRESS = "reaction_progress"
    REACTION_RATES = "reaction_rates"
    KINETIC_PARAMETERS = "kinetic_parameters"
    REACTION_CONDITIONS = "reaction_conditions"
    NETWORK_STRUCTURE = "network_structure"
    THERMODYNAMICS_EQUILIBRIUM = "thermodynamics_equilibrium"
    SOLUTION_EQUILIBRIUM = "solution_equilibrium"
    THERMOCHEMISTRY = "thermochemistry"
    ELECTROCHEMISTRY = "electrochemistry"
    MOLECULAR_DESCRIPTORS = "molecular_descriptors"
    ANALYTICAL_SPECTROSCOPY = "analytical_spectroscopy"
    CATALYSIS_SURFACES = "catalysis_surfaces"


class BiologyFamily(str, enum.Enum):
    MOLECULAR_GENE_REGULATION = "molecular_gene_regulation"
    ENZYMES_BIOCHEMICAL = "enzymes_biochemical"
    CELL_BIOLOGY = "cell_biology"
    PHYSIOLOGY_ORGANISM = "physiology_organism"
    MICROBIOLOGY_EPIDEMIOLOGY = "microbiology_epidemiology"
    EPIDEMIOLOGY_COMPARTMENTS = "epidemiology_compartments"
    ECOLOGY_POPULATIONS = "ecology_populations"
    EVOLUTION_GENETICS = "evolution_genetics"
    NEUROSCIENCE_BEHAVIOR = "neuroscience_behavior"
    ELECTROPHYSIOLOGY = "electrophysiology"
    BIOENERGETICS_METABOLISM = "bioenergetics_metabolism"
    DEVELOPMENTAL_SPATIAL = "developmental_spatial"
    OMICS_HIGH_DIMENSIONAL = "omics_high_dimensional"


class MathematicsFamily(str, enum.Enum):
    DYNAMICAL_SYSTEMS = "dynamical_systems"
    CHAOS_BIFURCATION = "chaos_bifurcation"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Machine-Readable Variable Specification (Spec §6)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScientificVariableSpec:
    """Standardized machine-readable scientific variable definition matching Spec §6."""
    variable_id: str
    domain: ScientificDomain
    family: str  # Member of PhysicsFamily, ChemistryFamily, or BiologyFamily
    kind: ScientificVariableKind = ScientificVariableKind.MEASURED_STATE
    value_type: ValueType = ValueType.SCALAR
    unit: str = "dimensionless"
    # SI Base Dimensions: (Mass M, Length L, Time T, Temperature Theta, Substance N, Current I, Luminosity J)
    dimensions: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    source: str = "measurement_or_derived"
    availability: str = "timestamped"
    uncertainty: str = "required"
    entity: str | None = None  # Specific species, molecule, or gene
    context_required: tuple[str, ...] = ()  # E.g. ('organism', 'cell_type', 'assay')
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["domain"] = self.domain.value
        d["kind"] = self.kind.value
        d["value_type"] = self.value_type.value
        d["dimensions"] = {
            "mass": self.dimensions[0],
            "length": self.dimensions[1],
            "time": self.dimensions[2],
            "temperature": self.dimensions[3],
            "substance": self.dimensions[4],
            "current": self.dimensions[5],
            "luminosity": self.dimensions[6],
        }
        return d


# ─────────────────────────────────────────────────────────────────────────────
# 3. Master Scientific Feature Registry
# ─────────────────────────────────────────────────────────────────────────────

def _build_master_scientific_catalog() -> dict[str, ScientificVariableSpec]:
    cat: dict[str, ScientificVariableSpec] = {}

    def reg(spec: ScientificVariableSpec) -> None:
        cat[spec.variable_id] = spec

    # ═════════════════════════════════════════════════════════════════════════
    # 1. PHYSICS GRAMMAR (Spec §1)
    # ═════════════════════════════════════════════════════════════════════════
    # A. Mechanics & Motion
    reg(ScientificVariableSpec("POSITION_X", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_GEOMETRY.value, unit="m", dimensions=(0, 1, 0, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("POSITION_Y", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_GEOMETRY.value, unit="m", dimensions=(0, 1, 0, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("POSITION_Z", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_GEOMETRY.value, unit="m", dimensions=(0, 1, 0, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("DISTANCE", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_GEOMETRY.value, unit="m", dimensions=(0, 1, 0, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("DISPLACEMENT", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_GEOMETRY.value, unit="m", dimensions=(0, 1, 0, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("ANGLE", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_GEOMETRY.value, unit="rad", dimensions=(0, 0, 0, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("AREA", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_GEOMETRY.value, unit="m^2", dimensions=(0, 2, 0, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("VOLUME", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_GEOMETRY.value, unit="m^3", dimensions=(0, 3, 0, 0, 0, 0, 0), constraints={"nonnegative": True}))

    reg(ScientificVariableSpec("VELOCITY", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_MOTION.value, value_type=ValueType.VECTOR, unit="m/s", dimensions=(0, 1, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("SPEED", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_MOTION.value, unit="m/s", dimensions=(0, 1, -1, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("ACCELERATION", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_MOTION.value, value_type=ValueType.VECTOR, unit="m/s^2", dimensions=(0, 1, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("JERK", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_MOTION.value, unit="m/s^3", dimensions=(0, 1, -3, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("ANGULAR_VELOCITY", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_MOTION.value, unit="rad/s", dimensions=(0, 0, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("ANGULAR_ACCELERATION", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_MOTION.value, unit="rad/s^2", dimensions=(0, 0, -2, 0, 0, 0, 0)))

    reg(ScientificVariableSpec("MASS", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_INERTIA.value, kind=ScientificVariableKind.PARAMETER, unit="kg", dimensions=(1, 0, 0, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("DENSITY", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_INERTIA.value, unit="kg/m^3", dimensions=(1, -3, 0, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("MOMENT_OF_INERTIA", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_INERTIA.value, unit="kg*m^2", dimensions=(1, 2, 0, 0, 0, 0, 0), constraints={"positive": True}))

    reg(ScientificVariableSpec("FORCE", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_FORCE_MOMENTUM.value, value_type=ValueType.VECTOR, unit="N", dimensions=(1, 1, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("TORQUE", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_FORCE_MOMENTUM.value, unit="N*m", dimensions=(1, 2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("LINEAR_MOMENTUM", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_FORCE_MOMENTUM.value, value_type=ValueType.VECTOR, unit="kg*m/s", dimensions=(1, 1, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("ANGULAR_MOMENTUM", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_FORCE_MOMENTUM.value, unit="kg*m^2/s", dimensions=(1, 2, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("IMPULSE", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_FORCE_MOMENTUM.value, unit="N*s", dimensions=(1, 1, -1, 0, 0, 0, 0)))

    reg(ScientificVariableSpec("KINETIC_ENERGY", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_ENERGY_WORK.value, unit="J", dimensions=(1, 2, -2, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("POTENTIAL_ENERGY", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_ENERGY_WORK.value, unit="J", dimensions=(1, 2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("WORK", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_ENERGY_WORK.value, unit="J", dimensions=(1, 2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("POWER", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_ENERGY_WORK.value, unit="W", dimensions=(1, 2, -3, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("DISSIPATION_RATE", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_ENERGY_WORK.value, unit="W", dimensions=(1, 2, -3, 0, 0, 0, 0), constraints={"nonnegative": True}))

    reg(ScientificVariableSpec("FRICTION_COEFFICIENT", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_CONTACT_DEFORMATION.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("CONTACT_FORCE", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_CONTACT_DEFORMATION.value, unit="N", dimensions=(1, 1, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("STRAIN", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_CONTACT_DEFORMATION.value, unit="dimensionless"))
    reg(ScientificVariableSpec("STRESS", ScientificDomain.PHYSICS, PhysicsFamily.MECHANICS_CONTACT_DEFORMATION.value, unit="Pa", dimensions=(1, -1, -2, 0, 0, 0, 0)))

    # B. Thermodynamics & Statistical Physics
    reg(ScientificVariableSpec("TEMPERATURE", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="K", dimensions=(0, 0, 0, 1, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("PRESSURE", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="Pa", dimensions=(1, -1, -2, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("INTERNAL_ENERGY", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="J", dimensions=(1, 2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("ENTHALPY", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="J", dimensions=(1, 2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("ENTROPY", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="J/K", dimensions=(1, 2, -2, -1, 0, 0, 0)))
    reg(ScientificVariableSpec("HELMHOLTZ_FREE_ENERGY", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="J", dimensions=(1, 2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("GIBBS_FREE_ENERGY", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="J", dimensions=(1, 2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("HEAT_CAPACITY", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="J/K", dimensions=(1, 2, -2, -1, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("THERMAL_CONDUCTIVITY", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="W/(m*K)", dimensions=(1, 1, -3, -1, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("HEAT_FLUX", ScientificDomain.PHYSICS, PhysicsFamily.THERMODYNAMICS.value, unit="W/m^2", dimensions=(1, 0, -3, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("PARTICLE_NUMBER", ScientificDomain.PHYSICS, PhysicsFamily.STATISTICAL_PHYSICS.value, unit="count", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("NUMBER_DENSITY", ScientificDomain.PHYSICS, PhysicsFamily.STATISTICAL_PHYSICS.value, unit="1/m^3", dimensions=(0, -3, 0, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("CHEMICAL_POTENTIAL", ScientificDomain.PHYSICS, PhysicsFamily.STATISTICAL_PHYSICS.value, unit="J/mol", dimensions=(1, 2, -2, 0, -1, 0, 0)))
    reg(ScientificVariableSpec("RELAXATION_TIME", ScientificDomain.PHYSICS, PhysicsFamily.STATISTICAL_PHYSICS.value, unit="s", dimensions=(0, 0, 1, 0, 0, 0, 0), constraints={"positive": True}))

    # C. Electricity & Magnetism
    reg(ScientificVariableSpec("ELECTRIC_CHARGE", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, unit="C", dimensions=(0, 0, 1, 0, 0, 1, 0)))
    reg(ScientificVariableSpec("CURRENT", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, unit="A", dimensions=(0, 0, 0, 0, 0, 1, 0)))
    reg(ScientificVariableSpec("VOLTAGE", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, unit="V", dimensions=(1, 2, -3, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("RESISTANCE", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, kind=ScientificVariableKind.PARAMETER, unit="ohm", dimensions=(1, 2, -3, 0, 0, -2, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("CAPACITANCE", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, kind=ScientificVariableKind.PARAMETER, unit="F", dimensions=(-1, -2, 4, 0, 0, 2, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("INDUCTANCE", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, kind=ScientificVariableKind.PARAMETER, unit="H", dimensions=(1, 2, -2, 0, 0, -2, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("ELECTRIC_FIELD", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, value_type=ValueType.VECTOR, unit="V/m", dimensions=(1, 1, -3, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("MAGNETIC_FIELD", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, value_type=ValueType.VECTOR, unit="T", dimensions=(1, 0, -2, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("MAGNETIC_FLUX", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, unit="Wb", dimensions=(1, 2, -2, 0, 0, -1, 0)))

    # D. Waves & Oscillations
    reg(ScientificVariableSpec("AMPLITUDE", ScientificDomain.PHYSICS, PhysicsFamily.WAVES_OPTICS_OSCILLATIONS.value, unit="arbitrary"))
    reg(ScientificVariableSpec("FREQUENCY", ScientificDomain.PHYSICS, PhysicsFamily.WAVES_OPTICS_OSCILLATIONS.value, unit="Hz", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("ANGULAR_FREQUENCY", ScientificDomain.PHYSICS, PhysicsFamily.WAVES_OPTICS_OSCILLATIONS.value, unit="rad/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("WAVELENGTH", ScientificDomain.PHYSICS, PhysicsFamily.WAVES_OPTICS_OSCILLATIONS.value, unit="m", dimensions=(0, 1, 0, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("WAVE_SPEED", ScientificDomain.PHYSICS, PhysicsFamily.WAVES_OPTICS_OSCILLATIONS.value, unit="m/s", dimensions=(0, 1, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("DAMPING_COEFFICIENT", ScientificDomain.PHYSICS, PhysicsFamily.WAVES_OPTICS_OSCILLATIONS.value, kind=ScientificVariableKind.PARAMETER, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("QUALITY_FACTOR", ScientificDomain.PHYSICS, PhysicsFamily.WAVES_OPTICS_OSCILLATIONS.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"positive": True}))

    # E. Fluid Mechanics & Transport
    reg(ScientificVariableSpec("FLUID_DENSITY", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="kg/m^3", dimensions=(1, -3, 0, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("PRESSURE_GRADIENT", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="Pa/m", dimensions=(1, -2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("VISCOSITY", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, kind=ScientificVariableKind.PARAMETER, unit="Pa*s", dimensions=(1, -1, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("KINEMATIC_VISCOSITY", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="m^2/s", dimensions=(0, 2, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("REYNOLDS_NUMBER", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="dimensionless", constraints={"positive": True}))
    reg(ScientificVariableSpec("PECLET_NUMBER", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="dimensionless", constraints={"positive": True}))
    reg(ScientificVariableSpec("DIFFUSION_COEFFICIENT", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, kind=ScientificVariableKind.PARAMETER, unit="m^2/s", dimensions=(0, 2, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("DRAG_COEFFICIENT", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"nonnegative": True}))

    # ═════════════════════════════════════════════════════════════════════════
    # 2. CHEMISTRY GRAMMAR (Spec §2)
    # ═════════════════════════════════════════════════════════════════════════
    # A. Chemical Composition & Concentration
    reg(ScientificVariableSpec("SPECIES_CONCENTRATION", ScientificDomain.CHEMISTRY, ChemistryFamily.COMPOSITION_CONCENTRATION.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True, "species_identity_required": True}))
    reg(ScientificVariableSpec("MOLE_FRACTION", ScientificDomain.CHEMISTRY, ChemistryFamily.COMPOSITION_CONCENTRATION.value, unit="dimensionless", constraints={"range": [0.0, 1.0], "species_identity_required": True}))
    reg(ScientificVariableSpec("MASS_FRACTION", ScientificDomain.CHEMISTRY, ChemistryFamily.COMPOSITION_CONCENTRATION.value, unit="dimensionless", constraints={"range": [0.0, 1.0], "species_identity_required": True}))
    reg(ScientificVariableSpec("ACTIVITY", ScientificDomain.CHEMISTRY, ChemistryFamily.COMPOSITION_CONCENTRATION.value, unit="dimensionless", constraints={"nonnegative": True, "species_identity_required": True}))
    reg(ScientificVariableSpec("ACTIVITY_COEFFICIENT", ScientificDomain.CHEMISTRY, ChemistryFamily.COMPOSITION_CONCENTRATION.value, unit="dimensionless", constraints={"positive": True}))
    reg(ScientificVariableSpec("PARTIAL_PRESSURE", ScientificDomain.CHEMISTRY, ChemistryFamily.COMPOSITION_CONCENTRATION.value, unit="Pa", dimensions=(1, -1, -2, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("IONIC_STRENGTH", ScientificDomain.CHEMISTRY, ChemistryFamily.COMPOSITION_CONCENTRATION.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("PH", ScientificDomain.CHEMISTRY, ChemistryFamily.COMPOSITION_CONCENTRATION.value, unit="dimensionless", constraints={"range": [0.0, 14.0]}))

    # B. Reaction & Kinetics Variables
    reg(ScientificVariableSpec("REACTANT_CONCENTRATION", ScientificDomain.CHEMISTRY, ChemistryFamily.REACTION_STATE.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("PRODUCT_CONCENTRATION", ScientificDomain.CHEMISTRY, ChemistryFamily.REACTION_STATE.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("REACTION_EXTENT", ScientificDomain.CHEMISTRY, ChemistryFamily.REACTION_PROGRESS.value, unit="mol", dimensions=(0, 0, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("CONVERSION", ScientificDomain.CHEMISTRY, ChemistryFamily.REACTION_PROGRESS.value, unit="dimensionless", constraints={"range": [0.0, 1.0]}))
    reg(ScientificVariableSpec("YIELD", ScientificDomain.CHEMISTRY, ChemistryFamily.REACTION_PROGRESS.value, unit="dimensionless", constraints={"range": [0.0, 1.0]}))
    reg(ScientificVariableSpec("REACTION_RATE", ScientificDomain.CHEMISTRY, ChemistryFamily.REACTION_RATES.value, unit="mol/(m^3*s)", dimensions=(0, -3, -1, 0, 1, 0, 0)))
    reg(ScientificVariableSpec("FORMATION_RATE", ScientificDomain.CHEMISTRY, ChemistryFamily.REACTION_RATES.value, unit="mol/(m^3*s)", dimensions=(0, -3, -1, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("CONSUMPTION_RATE", ScientificDomain.CHEMISTRY, ChemistryFamily.REACTION_RATES.value, unit="mol/(m^3*s)", dimensions=(0, -3, -1, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("RATE_CONSTANT", ScientificDomain.CHEMISTRY, ChemistryFamily.KINETIC_PARAMETERS.value, kind=ScientificVariableKind.PARAMETER, unit="variable_by_order", constraints={"positive": True}))
    reg(ScientificVariableSpec("ACTIVATION_ENERGY", ScientificDomain.CHEMISTRY, ChemistryFamily.KINETIC_PARAMETERS.value, kind=ScientificVariableKind.PARAMETER, unit="J/mol", dimensions=(1, 2, -2, 0, -1, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("STOICHIOMETRIC_COEFFICIENT", ScientificDomain.CHEMISTRY, ChemistryFamily.NETWORK_STRUCTURE.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless"))

    # C. Chemical Thermodynamics & Equilibrium
    reg(ScientificVariableSpec("EQUILIBRIUM_CONSTANT", ScientificDomain.CHEMISTRY, ChemistryFamily.THERMODYNAMICS_EQUILIBRIUM.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"positive": True}))
    reg(ScientificVariableSpec("REACTION_QUOTIENT", ScientificDomain.CHEMISTRY, ChemistryFamily.THERMODYNAMICS_EQUILIBRIUM.value, unit="dimensionless", constraints={"positive": True}))
    reg(ScientificVariableSpec("STANDARD_REACTION_GIBBS_ENERGY", ScientificDomain.CHEMISTRY, ChemistryFamily.THERMODYNAMICS_EQUILIBRIUM.value, unit="J/mol", dimensions=(1, 2, -2, 0, -1, 0, 0)))
    reg(ScientificVariableSpec("SOLUBILITY", ScientificDomain.CHEMISTRY, ChemistryFamily.THERMODYNAMICS_EQUILIBRIUM.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))

    # D. Electrochemistry & Catalysis
    reg(ScientificVariableSpec("ELECTRODE_POTENTIAL", ScientificDomain.CHEMISTRY, ChemistryFamily.ELECTROCHEMISTRY.value, unit="V", dimensions=(1, 2, -3, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("OVERPOTENTIAL", ScientificDomain.CHEMISTRY, ChemistryFamily.ELECTROCHEMISTRY.value, unit="V", dimensions=(1, 2, -3, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("FARADAIC_EFFICIENCY", ScientificDomain.CHEMISTRY, ChemistryFamily.ELECTROCHEMISTRY.value, unit="dimensionless", constraints={"range": [0.0, 1.0]}))
    reg(ScientificVariableSpec("CATALYST_LOADING", ScientificDomain.CHEMISTRY, ChemistryFamily.CATALYSIS_SURFACES.value, kind=ScientificVariableKind.PARAMETER, unit="kg/m^3", dimensions=(1, -3, 0, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("TURNOVER_FREQUENCY", ScientificDomain.CHEMISTRY, ChemistryFamily.CATALYSIS_SURFACES.value, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"positive": True}))

    # ═════════════════════════════════════════════════════════════════════════
    # 3. BIOLOGY GRAMMAR (Spec §3)
    # ═════════════════════════════════════════════════════════════════════════
    # A. Molecular Biology & Gene Regulation
    reg(ScientificVariableSpec("GENE_EXPRESSION", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, unit="transcripts_per_million", context_required=("organism", "cell_type", "assay"), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("MRNA_ABUNDANCE", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, unit="molecules/cell", context_required=("organism", "cell_type"), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("TRANSCRIPTION_RATE", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, unit="molecules/(cell*s)", context_required=("organism", "cell_type"), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("RNA_DEGRADATION_RATE", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, kind=ScientificVariableKind.PARAMETER, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("PROTEIN_ABUNDANCE", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, unit="molecules/cell", context_required=("organism", "cell_type"), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("TRANSLATION_RATE", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, unit="molecules/(cell*s)", context_required=("organism", "cell_type"), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("TRANSCRIPTION_FACTOR_ABUNDANCE", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, unit="molecules/cell", context_required=("organism", "cell_type"), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("BINDING_AFFINITY", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, kind=ScientificVariableKind.PARAMETER, unit="M", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"positive": True}))

    # B. Enzymes & Biochemical Reactions
    reg(ScientificVariableSpec("ENZYME_CONCENTRATION", ScientificDomain.BIOLOGY, BiologyFamily.ENZYMES_BIOCHEMICAL.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("SUBSTRATE_CONCENTRATION", ScientificDomain.BIOLOGY, BiologyFamily.ENZYMES_BIOCHEMICAL.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("MICHAELIS_CONSTANT", ScientificDomain.BIOLOGY, BiologyFamily.ENZYMES_BIOCHEMICAL.value, kind=ScientificVariableKind.PARAMETER, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("TURNOVER_NUMBER_KCAT", ScientificDomain.BIOLOGY, BiologyFamily.ENZYMES_BIOCHEMICAL.value, kind=ScientificVariableKind.PARAMETER, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("INHIBITOR_CONCENTRATION", ScientificDomain.BIOLOGY, BiologyFamily.ENZYMES_BIOCHEMICAL.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))

    # C. Cell Biology & Physiology
    reg(ScientificVariableSpec("CELL_COUNT", ScientificDomain.BIOLOGY, BiologyFamily.CELL_BIOLOGY.value, unit="cells", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("CELL_DENSITY", ScientificDomain.BIOLOGY, BiologyFamily.CELL_BIOLOGY.value, unit="cells/m^3", dimensions=(0, -3, 0, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("CELL_GROWTH_RATE", ScientificDomain.BIOLOGY, BiologyFamily.CELL_BIOLOGY.value, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("APOPTOSIS_RATE", ScientificDomain.BIOLOGY, BiologyFamily.CELL_BIOLOGY.value, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("HEART_RATE", ScientificDomain.BIOLOGY, BiologyFamily.PHYSIOLOGY_ORGANISM.value, unit="1/min", context_required=("species",), constraints={"positive": True}))
    reg(ScientificVariableSpec("BLOOD_PRESSURE", ScientificDomain.BIOLOGY, BiologyFamily.PHYSIOLOGY_ORGANISM.value, unit="mmHg", context_required=("species",), constraints={"positive": True}))
    reg(ScientificVariableSpec("BODY_TEMPERATURE", ScientificDomain.BIOLOGY, BiologyFamily.PHYSIOLOGY_ORGANISM.value, unit="K", dimensions=(0, 0, 0, 1, 0, 0, 0), context_required=("species",), constraints={"positive": True}))

    # D. Microbiology, Ecology & Populations
    reg(ScientificVariableSpec("PATHOGEN_LOAD", ScientificDomain.BIOLOGY, BiologyFamily.MICROBIOLOGY_EPIDEMIOLOGY.value, unit="copies/mL", context_required=("pathogen_species",), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("TRANSMISSION_RATE", ScientificDomain.BIOLOGY, BiologyFamily.MICROBIOLOGY_EPIDEMIOLOGY.value, kind=ScientificVariableKind.PARAMETER, unit="1/(person*day)", constraints={"positive": True}))
    reg(ScientificVariableSpec("RECOVERY_RATE", ScientificDomain.BIOLOGY, BiologyFamily.MICROBIOLOGY_EPIDEMIOLOGY.value, kind=ScientificVariableKind.PARAMETER, unit="1/day", constraints={"positive": True}))
    reg(ScientificVariableSpec("POPULATION_SIZE", ScientificDomain.BIOLOGY, BiologyFamily.ECOLOGY_POPULATIONS.value, unit="individuals", context_required=("species", "habitat"), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("BIRTH_RATE", ScientificDomain.BIOLOGY, BiologyFamily.ECOLOGY_POPULATIONS.value, unit="1/year", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("DEATH_RATE", ScientificDomain.BIOLOGY, BiologyFamily.ECOLOGY_POPULATIONS.value, unit="1/year", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("PREDATOR_DENSITY", ScientificDomain.BIOLOGY, BiologyFamily.ECOLOGY_POPULATIONS.value, unit="individuals/km^2", context_required=("species",), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("PREY_DENSITY", ScientificDomain.BIOLOGY, BiologyFamily.ECOLOGY_POPULATIONS.value, unit="individuals/km^2", context_required=("species",), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("CARRYING_CAPACITY", ScientificDomain.BIOLOGY, BiologyFamily.ECOLOGY_POPULATIONS.value, kind=ScientificVariableKind.PARAMETER, unit="individuals", context_required=("species", "habitat"), constraints={"positive": True}))
    reg(ScientificVariableSpec("SPECIES_INTERACTION_STRENGTH", ScientificDomain.BIOLOGY, BiologyFamily.ECOLOGY_POPULATIONS.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless"))

    # E. Neuroscience & Behavior
    reg(ScientificVariableSpec("MEMBRANE_POTENTIAL", ScientificDomain.BIOLOGY, BiologyFamily.NEUROSCIENCE_BEHAVIOR.value, unit="V", dimensions=(1, 2, -3, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("SPIKE_RATE", ScientificDomain.BIOLOGY, BiologyFamily.NEUROSCIENCE_BEHAVIOR.value, unit="Hz", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("SYNAPTIC_STRENGTH", ScientificDomain.BIOLOGY, BiologyFamily.NEUROSCIENCE_BEHAVIOR.value, unit="S", dimensions=(-1, -2, 3, 0, 0, 2, 0), constraints={"nonnegative": True}))

    # ═════════════════════════════════════════════════════════════════════════
    # 4. ASTROPHYSICS & GRAVITATION GRAMMAR
    # ═════════════════════════════════════════════════════════════════════════
    reg(ScientificVariableSpec("GRAVITATIONAL_POTENTIAL", ScientificDomain.ASTROPHYSICS, PhysicsFamily.ASTROPHYSICS_GRAVITATION.value, unit="m^2/s^2", dimensions=(0, 2, -2, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("ECCENTRICITY", ScientificDomain.ASTROPHYSICS, PhysicsFamily.ASTROPHYSICS_GRAVITATION.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"range": [0.0, 1.0]}))
    reg(ScientificVariableSpec("SEMI_MAJOR_AXIS", ScientificDomain.ASTROPHYSICS, PhysicsFamily.ASTROPHYSICS_GRAVITATION.value, unit="m", dimensions=(0, 1, 0, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("ORBITAL_PERIOD", ScientificDomain.ASTROPHYSICS, PhysicsFamily.ASTROPHYSICS_GRAVITATION.value, unit="s", dimensions=(0, 0, 1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("COSMOLOGICAL_SCALE_FACTOR", ScientificDomain.ASTROPHYSICS, PhysicsFamily.ASTROPHYSICS_GRAVITATION.value, unit="dimensionless", constraints={"positive": True}))
    reg(ScientificVariableSpec("HUBBLE_PARAMETER", ScientificDomain.ASTROPHYSICS, PhysicsFamily.ASTROPHYSICS_GRAVITATION.value, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("LUMINOSITY", ScientificDomain.ASTROPHYSICS, PhysicsFamily.ASTROPHYSICS_GRAVITATION.value, unit="W", dimensions=(1, 2, -3, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("REDSHIFT", ScientificDomain.ASTROPHYSICS, PhysicsFamily.ASTROPHYSICS_GRAVITATION.value, unit="dimensionless", constraints={"nonnegative": True}))

    # ═════════════════════════════════════════════════════════════════════════
    # 5. EXPANDED PHYSICS: CIRCUITS, FLUIDS, QUANTUM
    # ═════════════════════════════════════════════════════════════════════════
    # Circuits & Oscillators
    reg(ScientificVariableSpec("INDUCTOR_CURRENT", ScientificDomain.PHYSICS, PhysicsFamily.CIRCUITS_OSCILLATORS.value, unit="A", dimensions=(0, 0, 0, 0, 0, 1, 0)))
    reg(ScientificVariableSpec("IMPEDANCE", ScientificDomain.PHYSICS, PhysicsFamily.CIRCUITS_OSCILLATORS.value, unit="ohm", dimensions=(1, 2, -3, 0, 0, -2, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("ELECTRIC_DISPLACEMENT", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, value_type=ValueType.VECTOR, unit="C/m^2", dimensions=(0, -2, 1, 0, 0, 1, 0)))
    reg(ScientificVariableSpec("MAGNETIC_VECTOR_POTENTIAL", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, value_type=ValueType.VECTOR, unit="T*m", dimensions=(1, 1, -2, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("PERMITTIVITY", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, kind=ScientificVariableKind.PARAMETER, unit="F/m", dimensions=(-1, -3, 4, 0, 0, 2, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("PERMEABILITY", ScientificDomain.PHYSICS, PhysicsFamily.ELECTROMAGNETISM.value, kind=ScientificVariableKind.PARAMETER, unit="H/m", dimensions=(1, 1, -2, 0, 0, -2, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("MEMRISTANCE", ScientificDomain.PHYSICS, PhysicsFamily.CIRCUITS_OSCILLATORS.value, unit="ohm", dimensions=(1, 2, -3, 0, 0, -2, 0), constraints={"positive": True}))

    # Advanced Fluid Dynamics
    reg(ScientificVariableSpec("VORTICITY", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, value_type=ValueType.VECTOR, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("STREAM_FUNCTION", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="m^2/s", dimensions=(0, 2, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("MACH_NUMBER", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="dimensionless", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("RAYLEIGH_NUMBER", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="dimensionless", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("FROUDE_NUMBER", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="dimensionless", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("KNUDSEN_NUMBER", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="dimensionless", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("SHEAR_RATE", ScientificDomain.PHYSICS, PhysicsFamily.FLUIDS_TRANSPORT.value, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"nonnegative": True}))

    # Quantum & Modern Physics
    reg(ScientificVariableSpec("PLANCK_CONSTANT", ScientificDomain.PHYSICS, PhysicsFamily.QUANTUM_MODERN.value, kind=ScientificVariableKind.PARAMETER, unit="J*s", dimensions=(1, 2, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("PROBABILITY_DENSITY", ScientificDomain.PHYSICS, PhysicsFamily.QUANTUM_MODERN.value, unit="1/m^3", dimensions=(0, -3, 0, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("SPIN", ScientificDomain.PHYSICS, PhysicsFamily.QUANTUM_MODERN.value, unit="J*s", dimensions=(1, 2, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("HAMILTONIAN", ScientificDomain.PHYSICS, PhysicsFamily.QUANTUM_MODERN.value, unit="J", dimensions=(1, 2, -2, 0, 0, 0, 0)))

    # ═════════════════════════════════════════════════════════════════════════
    # 6. EXPANDED CHEMISTRY: KINETICS, EQUILIBRIUM, THERMOCHEMISTRY
    # ═════════════════════════════════════════════════════════════════════════
    reg(ScientificVariableSpec("ARRHENIUS_PRE_EXPONENTIAL", ScientificDomain.CHEMISTRY, ChemistryFamily.KINETIC_PARAMETERS.value, kind=ScientificVariableKind.PARAMETER, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("REACTION_ORDER", ScientificDomain.CHEMISTRY, ChemistryFamily.KINETIC_PARAMETERS.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless"))
    reg(ScientificVariableSpec("HALF_LIFE", ScientificDomain.CHEMISTRY, ChemistryFamily.KINETIC_PARAMETERS.value, unit="s", dimensions=(0, 0, 1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("ACID_DISSOCIATION_CONSTANT", ScientificDomain.CHEMISTRY, ChemistryFamily.SOLUTION_EQUILIBRIUM.value, kind=ScientificVariableKind.PARAMETER, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("BASE_DISSOCIATION_CONSTANT", ScientificDomain.CHEMISTRY, ChemistryFamily.SOLUTION_EQUILIBRIUM.value, kind=ScientificVariableKind.PARAMETER, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("SOLUBILITY_PRODUCT", ScientificDomain.CHEMISTRY, ChemistryFamily.SOLUTION_EQUILIBRIUM.value, kind=ScientificVariableKind.PARAMETER, unit="(mol/m^3)^2", constraints={"positive": True}))
    reg(ScientificVariableSpec("STANDARD_REACTION_ENTHALPY", ScientificDomain.CHEMISTRY, ChemistryFamily.THERMOCHEMISTRY.value, unit="J/mol", dimensions=(1, 2, -2, 0, -1, 0, 0)))
    reg(ScientificVariableSpec("STANDARD_REACTION_ENTROPY", ScientificDomain.CHEMISTRY, ChemistryFamily.THERMOCHEMISTRY.value, unit="J/(mol*K)", dimensions=(1, 2, -2, -1, -1, 0, 0)))
    reg(ScientificVariableSpec("FARADAY_CONSTANT", ScientificDomain.CHEMISTRY, ChemistryFamily.ELECTROCHEMISTRY.value, kind=ScientificVariableKind.PARAMETER, unit="C/mol", dimensions=(0, 0, 1, 0, -1, 1, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("NERNST_POTENTIAL", ScientificDomain.CHEMISTRY, ChemistryFamily.ELECTROCHEMISTRY.value, unit="V", dimensions=(1, 2, -3, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("TRANSFER_COEFFICIENT", ScientificDomain.CHEMISTRY, ChemistryFamily.ELECTROCHEMISTRY.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"range": [0.0, 1.0]}))
    reg(ScientificVariableSpec("MOLECULAR_WEIGHT", ScientificDomain.CHEMISTRY, ChemistryFamily.MOLECULAR_DESCRIPTORS.value, kind=ScientificVariableKind.PARAMETER, unit="kg/mol", dimensions=(1, 0, 0, 0, -1, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("ELECTRONEGATIVITY", ScientificDomain.CHEMISTRY, ChemistryFamily.MOLECULAR_DESCRIPTORS.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"positive": True}))

    # ═════════════════════════════════════════════════════════════════════════
    # 7. EXPANDED BIOLOGY: EPIDEMIOLOGY SEIR, GENETICS, METABOLISM, NEURO
    # ═════════════════════════════════════════════════════════════════════════
    reg(ScientificVariableSpec("SUSCEPTIBLE_POPULATION", ScientificDomain.BIOLOGY, BiologyFamily.EPIDEMIOLOGY_COMPARTMENTS.value, unit="individuals", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("EXPOSED_POPULATION", ScientificDomain.BIOLOGY, BiologyFamily.EPIDEMIOLOGY_COMPARTMENTS.value, unit="individuals", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("INFECTIOUS_POPULATION", ScientificDomain.BIOLOGY, BiologyFamily.EPIDEMIOLOGY_COMPARTMENTS.value, unit="individuals", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("RECOVERED_POPULATION", ScientificDomain.BIOLOGY, BiologyFamily.EPIDEMIOLOGY_COMPARTMENTS.value, unit="individuals", constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("BASIC_REPRODUCTION_NUMBER", ScientificDomain.BIOLOGY, BiologyFamily.EPIDEMIOLOGY_COMPARTMENTS.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"nonnegative": True}))

    reg(ScientificVariableSpec("HILL_COEFFICIENT", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless", constraints={"positive": True}))
    reg(ScientificVariableSpec("PROMOTER_ACTIVITY", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("PROTEIN_DEGRADATION_RATE", ScientificDomain.BIOLOGY, BiologyFamily.MOLECULAR_GENE_REGULATION.value, kind=ScientificVariableKind.PARAMETER, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0), constraints={"positive": True}))

    reg(ScientificVariableSpec("ATP_CONCENTRATION", ScientificDomain.BIOLOGY, BiologyFamily.BIOENERGETICS_METABOLISM.value, unit="mol/m^3", dimensions=(0, -3, 0, 0, 1, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("ENERGY_CHARGE", ScientificDomain.BIOLOGY, BiologyFamily.BIOENERGETICS_METABOLISM.value, unit="dimensionless", constraints={"range": [0.0, 1.0]}))
    reg(ScientificVariableSpec("BIOMASS", ScientificDomain.BIOLOGY, BiologyFamily.BIOENERGETICS_METABOLISM.value, unit="kg/m^3", dimensions=(1, -3, 0, 0, 0, 0, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("METABOLIC_FLUX", ScientificDomain.BIOLOGY, BiologyFamily.BIOENERGETICS_METABOLISM.value, unit="mol/(kg*s)", dimensions=(-1, 0, -1, 0, 1, 0, 0)))

    reg(ScientificVariableSpec("ACTION_POTENTIAL_THRESHOLD", ScientificDomain.BIOLOGY, BiologyFamily.ELECTROPHYSIOLOGY.value, kind=ScientificVariableKind.PARAMETER, unit="V", dimensions=(1, 2, -3, 0, 0, -1, 0)))
    reg(ScientificVariableSpec("REFRACTORY_PERIOD", ScientificDomain.BIOLOGY, BiologyFamily.ELECTROPHYSIOLOGY.value, kind=ScientificVariableKind.PARAMETER, unit="s", dimensions=(0, 0, 1, 0, 0, 0, 0), constraints={"positive": True}))
    reg(ScientificVariableSpec("IONIC_CONDUCTANCE", ScientificDomain.BIOLOGY, BiologyFamily.ELECTROPHYSIOLOGY.value, unit="S/m^2", dimensions=(-1, -4, 3, 0, 0, 2, 0), constraints={"nonnegative": True}))
    reg(ScientificVariableSpec("REVERSAL_POTENTIAL", ScientificDomain.BIOLOGY, BiologyFamily.ELECTROPHYSIOLOGY.value, kind=ScientificVariableKind.PARAMETER, unit="V", dimensions=(1, 2, -3, 0, 0, -1, 0)))

    # ═════════════════════════════════════════════════════════════════════════
    # 8. MATHEMATICS & DYNAMICAL SYSTEMS GRAMMAR
    # ═════════════════════════════════════════════════════════════════════════
    reg(ScientificVariableSpec("BIFURCATION_PARAMETER", ScientificDomain.MATHEMATICS, MathematicsFamily.CHAOS_BIFURCATION.value, kind=ScientificVariableKind.PARAMETER, unit="dimensionless"))
    reg(ScientificVariableSpec("LYAPUNOV_EXPONENT", ScientificDomain.MATHEMATICS, MathematicsFamily.CHAOS_BIFURCATION.value, unit="1/s", dimensions=(0, 0, -1, 0, 0, 0, 0)))
    reg(ScientificVariableSpec("ORDER_PARAMETER", ScientificDomain.MATHEMATICS, MathematicsFamily.DYNAMICAL_SYSTEMS.value, unit="dimensionless", constraints={"range": [0.0, 1.0]}))
    reg(ScientificVariableSpec("PHASE_SPACE_COORDINATE", ScientificDomain.MATHEMATICS, MathematicsFamily.DYNAMICAL_SYSTEMS.value, unit="arbitrary"))
    reg(ScientificVariableSpec("TIME_DELAY", ScientificDomain.MATHEMATICS, MathematicsFamily.DYNAMICAL_SYSTEMS.value, kind=ScientificVariableKind.PARAMETER, unit="s", dimensions=(0, 0, 1, 0, 0, 0, 0), constraints={"positive": True}))

    return cat


MASTER_SCIENTIFIC_CATALOG: dict[str, ScientificVariableSpec] = _build_master_scientific_catalog()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Shared Scientific Operator Library (Spec §4)
# ─────────────────────────────────────────────────────────────────────────────

def op_michaelis_menten(s: np.ndarray | float, vmax: float, km: float) -> np.ndarray:
    """Enzyme saturation rate law: v = V_max * S / (K_m + S)."""
    s_arr = np.maximum(np.asarray(s, dtype=float), 0.0)
    km_eff = max(km, 1e-12)
    return vmax * s_arr / (km_eff + s_arr)


def op_hill_activation(x: np.ndarray | float, k: float, n: float = 2.0) -> np.ndarray:
    """Hill activation function: f(x) = x^n / (K^n + x^n)."""
    x_arr = np.maximum(np.asarray(x, dtype=float), 0.0)
    k_eff = max(k, 1e-12)
    xn = np.power(x_arr, n)
    kn = np.power(k_eff, n)
    return xn / (kn + xn)


def op_hill_repression(x: np.ndarray | float, k: float, n: float = 2.0) -> np.ndarray:
    """Hill repression function: f(x) = K^n / (K^n + x^n)."""
    x_arr = np.maximum(np.asarray(x, dtype=float), 0.0)
    k_eff = max(k, 1e-12)
    xn = np.power(x_arr, n)
    kn = np.power(k_eff, n)
    return kn / (kn + xn)


def op_temporal_derivative(x: np.ndarray, dt: float) -> np.ndarray:
    """Causal numerical derivative dx/dt using backward difference."""
    x_arr = np.asarray(x, dtype=float)
    T = len(x_arr)
    out = np.full(T, np.nan)
    dt_eff = max(dt, 1e-12)
    if T > 1:
        out[1:] = (x_arr[1:] - x_arr[:-1]) / dt_eff
    return out


def op_temporal_integral(x: np.ndarray, dt: float, init: float = 0.0) -> np.ndarray:
    """Causal trapezoidal accumulation: int_0^t x(s) ds."""
    x_arr = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    out = np.zeros(len(x_arr), dtype=float)
    cur = init
    out[0] = cur
    for t in range(1, len(x_arr)):
        cur += 0.5 * (x_arr[t] + x_arr[t - 1]) * dt
        out[t] = cur
    return out


def op_delay(x: np.ndarray, delay_steps: int) -> np.ndarray:
    """Causal delayed state: x(t - tau)."""
    x_arr = np.asarray(x, dtype=float)
    out = np.full_like(x_arr, np.nan)
    if delay_steps < len(x_arr):
        out[delay_steps:] = x_arr[:-delay_steps]
    return out


def op_threshold_activation(x: np.ndarray, theta: float) -> np.ndarray:
    """Heaviside step activation: 1(x > theta)."""
    return np.where(np.asarray(x, dtype=float) > theta, 1.0, 0.0)


def op_lotka_volterra_step(
    prey: float,
    pred: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    dt: float,
) -> tuple[float, float]:
    """Causal discrete update step for Lotka-Volterra predator-prey dynamics."""
    d_prey = (alpha * prey - beta * prey * pred) * dt
    d_pred = (delta * prey * pred - gamma * pred) * dt
    new_prey = max(prey + d_prey, 0.0)
    new_pred = max(pred + d_pred, 0.0)
    return new_prey, new_pred


# ─────────────────────────────────────────────────────────────────────────────
# 5. Specialized Domain Validity Gates (Spec §5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    domain: ScientificDomain
    gate_name: str
    reasons: tuple[str, ...] = ()


class PhysicsValidityGate:
    """Validates physics candidates against SI dimensions, energy bounds, and symmetries."""

    @staticmethod
    def check_dimensional_homogeneity(
        dim_left: tuple[int, int, int, int, int, int, int],
        dim_right: tuple[int, int, int, int, int, int, int],
    ) -> GateVerdict:
        if dim_left == dim_right:
            return GateVerdict(True, ScientificDomain.PHYSICS, "dimensional_homogeneity")
        return GateVerdict(
            False,
            ScientificDomain.PHYSICS,
            "dimensional_homogeneity",
            (f"Dimensional mismatch: Left {dim_left} != Right {dim_right}",),
        )

    @staticmethod
    def check_energy_conservation(kinetic_energy: np.ndarray, potential_energy: np.ndarray, tol: float = 0.05) -> GateVerdict:
        """Verify total mechanical energy E = T + V is conserved in conservative systems."""
        ke = np.asarray(kinetic_energy, dtype=float)
        pe = np.asarray(potential_energy, dtype=float)
        total_e = ke + pe
        rel_drift = (np.max(total_e) - np.min(total_e)) / max(np.mean(np.abs(total_e)), 1e-12)
        if rel_drift <= tol:
            return GateVerdict(True, ScientificDomain.PHYSICS, "energy_conservation")
        return GateVerdict(
            False,
            ScientificDomain.PHYSICS,
            "energy_conservation",
            (f"Total mechanical energy drifted by {rel_drift:.2%}, exceeding tolerance {tol:.2%}",),
        )


class ChemistryValidityGate:
    """Validates reaction equations against stoichiometry, non-negativity, and thermodynamic bounds."""

    @staticmethod
    def check_concentration_positivity(concentrations: np.ndarray) -> GateVerdict:
        c = np.asarray(concentrations, dtype=float)
        if np.all(c[np.isfinite(c)] >= -1e-9):
            return GateVerdict(True, ScientificDomain.CHEMISTRY, "concentration_positivity")
        min_c = float(np.min(c))
        return GateVerdict(
            False,
            ScientificDomain.CHEMISTRY,
            "concentration_positivity",
            (f"Concentration violated positivity constraint: min concentration = {min_c:.4e}",),
        )

    @staticmethod
    def check_mass_balance(
        stoichiometric_matrix: np.ndarray,
        reaction_rates: np.ndarray,
        species_rates: np.ndarray,
        tol: float = 1e-6,
    ) -> GateVerdict:
        """Verify dc/dt = S * r holds within experimental tolerance."""
        predicted_rates = stoichiometric_matrix @ reaction_rates
        residual = float(np.max(np.abs(predicted_rates - species_rates)))
        if residual <= tol:
            return GateVerdict(True, ScientificDomain.CHEMISTRY, "mass_balance")
        return GateVerdict(
            False,
            ScientificDomain.CHEMISTRY,
            "mass_balance",
            (f"Mass balance violated: max residual {residual:.4e} > {tol:.4e}",),
        )


class BiologyValidityGate:
    """Validates biological equations against population non-negativity, capacity limits, and context."""

    @staticmethod
    def check_population_bounds(
        populations: np.ndarray,
        carrying_capacity: float | None = None,
    ) -> GateVerdict:
        p = np.asarray(populations, dtype=float)
        if np.any(p[np.isfinite(p)] < 0.0):
            return GateVerdict(False, ScientificDomain.BIOLOGY, "population_bounds", ("Negative population encountered",))
        if carrying_capacity is not None and np.any(p[np.isfinite(p)] > 2.0 * carrying_capacity):
            return GateVerdict(
                False,
                ScientificDomain.BIOLOGY,
                "population_bounds",
                (f"Population exceeded 2x carrying capacity {carrying_capacity}",),
            )
        return GateVerdict(True, ScientificDomain.BIOLOGY, "population_bounds")

    @staticmethod
    def check_biological_context(spec: ScientificVariableSpec, provided_context: dict[str, str]) -> GateVerdict:
        """Verify essential biological context tags (organism, cell_type, species) are supplied."""
        missing = [ctx for ctx in spec.context_required if ctx not in provided_context]
        if not missing:
            return GateVerdict(True, ScientificDomain.BIOLOGY, "biological_context")
        return GateVerdict(
            False,
            ScientificDomain.BIOLOGY,
            "biological_context",
            (f"Variable '{spec.variable_id}' missing required biological context: {missing}",),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Scientific Discovery Packages & Query Engine (Spec §7)
# ─────────────────────────────────────────────────────────────────────────────

class ScientificGrammarCatalog:
    """Master registry and verification engine for APEDP's multi-domain scientific grammar."""

    def __init__(self, catalog: dict[str, ScientificVariableSpec] | None = None):
        self._catalog = catalog if catalog is not None else MASTER_SCIENTIFIC_CATALOG

    @property
    def total_variables(self) -> int:
        return len(self._catalog)

    def get_variable(self, variable_id: str) -> ScientificVariableSpec:
        if variable_id not in self._catalog:
            raise KeyError(f"Scientific variable '{variable_id}' not found in catalogue.")
        return self._catalog[variable_id]

    def filter_by_domain(self, domain: ScientificDomain) -> list[ScientificVariableSpec]:
        return [v for v in self._catalog.values() if v.domain == domain]

    def filter_by_family(self, family: str) -> list[ScientificVariableSpec]:
        return [v for v in self._catalog.values() if v.family == family]

    def list_variables(self) -> list[str]:
        return sorted(self._catalog.keys())

    def get_package_variables(self, package_name: str) -> list[str]:
        pkg = package_name.upper().replace("-", "_")
        if pkg in ("APEDP_PHYSICS", "PHYSICS"):
            return [v.variable_id for v in self.filter_by_domain(ScientificDomain.PHYSICS)]
        elif pkg in ("APEDP_CHEMISTRY", "CHEMISTRY"):
            return [v.variable_id for v in self.filter_by_domain(ScientificDomain.CHEMISTRY)]
        elif pkg in ("APEDP_BIOLOGY", "BIOLOGY"):
            return [v.variable_id for v in self.filter_by_domain(ScientificDomain.BIOLOGY)]
        elif pkg in ("APEDP_ASTROPHYSICS", "ASTROPHYSICS"):
            return [v.variable_id for v in self.filter_by_domain(ScientificDomain.ASTROPHYSICS)]
        elif pkg in ("APEDP_MATHEMATICS", "MATHEMATICS", "DYNAMICS"):
            return [v.variable_id for v in self.filter_by_domain(ScientificDomain.MATHEMATICS)]
        raise ValueError(f"Unknown scientific package '{package_name}'. Valid: APEDP-PHYSICS, APEDP-CHEMISTRY, APEDP-BIOLOGY, APEDP-ASTROPHYSICS, APEDP-MATHEMATICS.")

    def to_json(self, indent: int = 2) -> str:
        serialized = {k: v.to_dict() for k, v in self._catalog.items()}
        return json.dumps(serialized, indent=indent)

    def compute_catalog_fingerprint(self) -> str:
        """SHA-256 cryptographic provenance hash."""
        canonical = sorted([
            (k, v.domain.value, v.family, v.unit, v.dimensions)
            for k, v in self._catalog.items()
        ])
        raw = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
