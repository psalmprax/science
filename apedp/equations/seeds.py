"""Curated seed list: named laws -> source pages. No network at import."""
# license recorded per source for FAIR compliance.

SEED_LAWS: list = [
    {"name": "Newton's second law", "page": "Newton's laws of motion",
     "domain": "mechanics", "license": "CC-BY-SA-4.0"},
    {"name": "Coulomb's law", "page": "Coulomb's law",
     "domain": "electromagnetism", "license": "CC-BY-SA-4.0"},
    {"name": "Hooke's law", "page": "Hooke's law",
     "domain": "mechanics", "license": "CC-BY-SA-4.0"},
    {"name": "Ohm's law", "page": "Ohm's law",
     "domain": "electromagnetism", "license": "CC-BY-SA-4.0"},
    {"name": "Fourier's law", "page": "Thermal conduction",
     "domain": "thermodynamics", "license": "CC-BY-SA-4.0"},
    {"name": "Fick's laws", "page": "Fick's laws of diffusion",
     "domain": "mass-transfer", "license": "CC-BY-SA-4.0"},
    {"name": "Bernoulli's principle", "page": "Bernoulli's principle",
     "domain": "fluids", "license": "CC-BY-SA-4.0"},
    {"name": "Ideal gas law", "page": "Ideal gas law",
     "domain": "thermodynamics", "license": "CC-BY-SA-4.0"},
    {"name": "Stefan-Boltzmann law", "page": "Stefan–Boltzmann law",
     "domain": "thermodynamics", "license": "CC-BY-SA-4.0"},
    {"name": "Maxwell equations", "page": "Maxwell's equations",
     "domain": "electromagnetism", "license": "CC-BY-SA-4.0"},
]


def seed_pages() -> list:
    return [{"page": s["page"], "domain": s["domain"],
             "license": s["license"], "name": s["name"]} for s in SEED_LAWS]
