"""APEDP variable semantics: symbol meaning disambiguation.

Bare symbols (n, T, tau) are ambiguous. This module carriers explicit
VariableSpec records so PVE checks semantics, not just dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariableSpec:
    symbol: str
    name: str
    unit: str
    meaning: str
    domain: str = ""
    species: str = ""
    frame: str = ""
    scale: str = ""
    assumptions: tuple = ()
    valid_range: str = ""


FUSION_VARIABLES: dict[str, VariableSpec] = {
    "n_e": VariableSpec("n_e", "electron density", "1/meter**3",
                        "electron particle density", "fusion plasma", "electron", "", "", (), "1e19-1e21 m^-3"),
    "n_i": VariableSpec("n_i", "ion density", "1/meter**3",
                        "ion particle density", "fusion plasma", "ion", "", "", (), "1e19-1e21 m^-3"),
    "n_D": VariableSpec("n_D", "deuterium density", "1/meter**3",
                        "deuterium fuel density", "fusion plasma", "D", "", "", (), ""),
    "n_T": VariableSpec("n_T", "tritium density", "1/meter**3",
                        "tritium fuel density", "fusion plasma", "T", "", "", (), ""),
    "T_e": VariableSpec("T_e", "electron temperature", "keV",
                        "electron temperature", "fusion plasma", "electron", "lab", "", (), "5-20 keV"),
    "T_i": VariableSpec("T_i", "ion temperature", "keV",
                        "ion temperature", "fusion plasma", "ion", "lab", "", (), "5-20 keV"),
    "tau_E": VariableSpec("tau_E", "energy confinement time", "second",
                          "energy confinement time", "fusion plasma", "", "lab", "", ("steady_state",), ""),
    "tau_p": VariableSpec("tau_p", "particle confinement time", "second",
                          "particle confinement time", "fusion plasma", "", "lab", "", ("steady_state",), ""),
    "sigma_v_DT": VariableSpec("sigma_v_DT", "D-T reactivity", "meter**3/second",
                               "fusion reactivity <sigma*v>", "fusion plasma", "D-T", "CM", "", ("Maxwellian",), ""),
    "E_alpha": VariableSpec("E_alpha", "alpha energy", "joule",
                            "alpha particle energy per D-T reaction", "D-T", "alpha", "lab", "", (), "~3.5 MeV"),
    "E_n": VariableSpec("E_n", "neutron energy", "joule",
                        "neutron energy per D-T reaction (not neutrino)", "D-T", "neutron", "lab", "", (), "~14.1 MeV"),
}

# Bare symbols that must trigger a semantics warning.
AMBIGUOUS_BARE_SYMBOLS = {"n", "T", "tau"}


def check_variable_semantics(symbols: set[str]) -> dict:
    """Return resolved specs plus warnings for bare ambiguous symbols."""
    resolved = {s: FUSION_VARIABLES[s] for s in symbols if s in FUSION_VARIABLES}
    ambiguous = sorted(s for s in symbols if s in AMBIGUOUS_BARE_SYMBOLS)
    return {"resolved": resolved, "ambiguous_bare": ambiguous}


class VariableRegistry:
    """Domain-agnostic variable registry. Fusion is one domain plugin."""

    def __init__(self) -> None:
        self._specs: dict[str, VariableSpec] = dict(FUSION_VARIABLES)
        self._ambiguous: set[str] = set(AMBIGUOUS_BARE_SYMBOLS)

    def register(self, spec: VariableSpec, ambiguous_bare: tuple = ()) -> None:
        self._specs[spec.symbol] = spec
        for b in ambiguous_bare:
            self._ambiguous.add(b)

    def check(self, symbols: set[str]) -> dict:
        resolved = {s: self._specs[s] for s in symbols if s in self._specs}
        ambiguous = sorted(s for s in symbols if s in self._ambiguous)
        unresolved = sorted(s for s in symbols if s not in self._specs and s not in self._ambiguous)
        return {"resolved": resolved, "ambiguous_bare": ambiguous, "unresolved": unresolved}


DEFAULT_REGISTRY = VariableRegistry()


@dataclass
class AssumptionSet:
    items: dict[str, str] = field(default_factory=dict)

    def add(self, key: str, value: str) -> None:
        self.items[key] = value

    def missing(self, required: list) -> list:
        return [k for k in required if k not in self.items]


@dataclass
class ApplicabilityEnvelope:
    ranges: dict[str, str] = field(default_factory=dict)
    tested: dict[str, str] = field(default_factory=dict)

    def covers(self, var: str) -> bool:
        return var in self.ranges
