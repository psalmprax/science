"""APEDP Core: Dimensional Ontology, Invariants, Buckingham-Pi, and Model Assurance."""
from __future__ import annotations

import logging
from typing import Any, ClassVar, TypeAlias

import numpy as np
import pint
import sympy as sp
from pint.facets.plain.unit import PlainUnit

# Global Pint unit registry
ureg = pint.UnitRegistry()

logger = logging.getLogger(__name__)
if hasattr(ureg, 'formatter'):
    ureg.formatter.default_format = "~"
else:
    ureg.default_format = "~"

NumberLike: TypeAlias = int | float | np.number | sp.Integer | sp.Float | sp.Rational | sp.Number
QuantityLike: TypeAlias = NumberLike | pint.Quantity
# Pint produces three distinct unit-like types depending on the operation:
# attribute access (Unit), string parsing/multiplication (Quantity), and
# float exponentiation (PlainUnit). This alias covers all three.
DimensionalUnit = pint.Unit | pint.Quantity | PlainUnit


class PhysicalConstant:
    """Represents a fundamental physical constant with exact value, units, and description."""
    def __init__(self, name: str, symbol: sp.Symbol, value: float, unit_str: str, description: str):
        self.name = name
        self.symbol = symbol
        self.value = value
        self.unit_str = unit_str
        self.unit = ureg(unit_str)
        self.description = description

    def __repr__(self) -> str:
        return f"PhysicalConstant({self.name}={self.value} [{self.unit_str}])"


class FundamentalConstantsRegistry:
    """Authoritative registry for SI fundamental physical constants."""
    def __init__(self):
        self._constants: dict[str, PhysicalConstant] = {
            "c_light": PhysicalConstant("c_light", sp.Symbol("c_light"), 299792458.0, "meter / second", "Speed of light in vacuum"),
            "G_grav": PhysicalConstant("G_grav", sp.Symbol("G_grav"), 6.67430e-11, "meter**3 / (kilogram * second**2)", "Newtonian gravitational constant"),
            "h_bar": PhysicalConstant("h_bar", sp.Symbol("h_bar"), 1.054571817e-34, "joule * second", "Reduced Planck constant"),
            "k_B": PhysicalConstant("k_B", sp.Symbol("k_B"), 1.380649e-23, "joule / kelvin", "Boltzmann constant"),
            "e_charge": PhysicalConstant("e_charge", sp.Symbol("e_charge"), 1.602176634e-19, "coulomb", "Elementary charge"),
            "eps_0": PhysicalConstant("eps_0", sp.Symbol("eps_0"), 8.8541878128e-12, "farad / meter", "Vacuum electric permittivity"),
            "mu_0": PhysicalConstant("mu_0", sp.Symbol("mu_0"), 1.25663706212e-6, "newton / ampere**2", "Vacuum magnetic permeability"),
            "sigma_sb": PhysicalConstant("sigma_sb", sp.Symbol("sigma_sb"), 5.670374419e-8, "watt / (meter**2 * kelvin**4)", "Stefan-Boltzmann constant"),
            "R_gas": PhysicalConstant("R_gas", sp.Symbol("R_gas"), 8.314462618, "joule / (mole * kelvin)", "Universal gas constant"),
            "N_A": PhysicalConstant("N_A", sp.Symbol("N_A"), 6.02214076e23, "1 / mole", "Avogadro constant"),
            "g_earth": PhysicalConstant("g_earth", sp.Symbol("g_earth"), 9.80665, "meter / second**2", "Standard acceleration of gravity on Earth")
        }

    def register_constant(self, name: str, value: float, unit_str: str, description: str = "") -> PhysicalConstant:
        const = PhysicalConstant(name, sp.Symbol(name), value, unit_str, description)
        self._constants[name] = const
        return const

    def get_constant(self, name: str) -> PhysicalConstant:
        if name not in self._constants:
            raise KeyError(f"Constant '{name}' not found in registry.")
        return self._constants[name]

    def values_dict(self) -> dict[str, float]:
        return {k: c.value for k, c in self._constants.items()}

    def symbols_dict(self) -> dict[str, sp.Symbol]:
        return {k: c.symbol for k, c in self._constants.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._constants

    def __getitem__(self, name: str) -> PhysicalConstant:
        return self.get_constant(name)

    def __iter__(self):
        return iter(self._constants)

    def __len__(self) -> int:
        return len(self._constants)

    def get(self, name: str, default: Any = None) -> Any:
        return self._constants.get(name, default)

    def keys(self):
        return self._constants.keys()

    def values(self):
        return self._constants.values()

    def items(self):
        return self._constants.items()


PHYSICAL_CONSTANTS = FundamentalConstantsRegistry()


class ConservationInvariantRegistry:
    """Registry and verification engine for Noether invariants and fundamental conservation laws."""

    @staticmethod
    def audit_energy_conservation(Q_dot: float, W_dot: float, dE_dt: float, tolerance: float = 1e-9) -> tuple[bool, float]:
        """First Law of Thermodynamics rate form: dE/dt = Q_dot - W_dot (Watts)."""
        expected_dE_dt = Q_dot - W_dot
        residual = abs(dE_dt - expected_dE_dt)
        is_conserved = residual <= tolerance
        return is_conserved, residual

    @staticmethod
    def audit_momentum_conservation(d_momentum_dt: float, sum_forces: float, tolerance: float = 1e-9) -> tuple[bool, float]:
        """Newtonian momentum conservation: dP/dt = sum(F_ext) (Newtons)."""
        residual = abs(d_momentum_dt - sum_forces)
        is_conserved = residual <= tolerance
        return is_conserved, residual

    @staticmethod
    def audit_bernoulli_invariant(
        P1: float, v1: float, h1: float,
        P2: float, v2: float, h2: float,
        rho: float, g: float = 9.80665,
        tolerance: float = 1e-6
    ) -> tuple[bool, float]:
        """Bernoulli stream-tube invariant: P + 0.5*rho*v^2 + rho*g*h = constant (Pascals)."""
        E1 = P1 + 0.5 * rho * (v1 ** 2) + rho * g * h1
        E2 = P2 + 0.5 * rho * (v2 ** 2) + rho * g * h2
        residual = abs(E1 - E2)
        is_conserved = residual <= tolerance
        return is_conserved, residual

    @staticmethod
    def audit_continuity_mass(A1: float, v1: float, A2: float, v2: float, tolerance: float = 1e-9) -> tuple[bool, float]:
        """Incompressible mass continuity: A1*v1 = A2*v2 (m^3/s)."""
        Q1 = A1 * v1
        Q2 = A2 * v2
        residual = abs(Q1 - Q2)
        return residual <= tolerance, residual


class DimensionalAssuranceEngine:
    """Dimensional ontology engine: tracks units and enforces strict homogeneity."""

    def __init__(self, registry: pint.UnitRegistry | None = None):
        self.registry = registry or ureg
        self.symbol_units: dict[sp.Symbol, DimensionalUnit] = {}
        # Pre-register all standard physical constants
        for const in PHYSICAL_CONSTANTS._constants.values():
            self.symbol_units[const.symbol] = const.unit

    def register_variable(self, symbol_name: str, unit_str: str | pint.Unit) -> sp.Symbol:
        sym = sp.Symbol(symbol_name)
        unit: DimensionalUnit
        if isinstance(unit_str, pint.Unit):
            unit = unit_str
        else:
            unit = self.registry(str(unit_str))
        self.symbol_units[sym] = unit
        return sym

    def _get_unit(self, sym: sp.Symbol) -> DimensionalUnit:
        if sym in self.symbol_units:
            return self.symbol_units[sym]
        raise ValueError(f"Symbol '{sym}' is not registered with any physical unit.")

    def check_expression_consistency(self, expr: sp.Expr) -> DimensionalUnit:
        """Computes Pint Unit for a SymPy expression or raises TypeError on dimensional violations."""
        if isinstance(expr, sp.Symbol):
            return self._get_unit(expr)

        if isinstance(expr, (int, float, sp.Integer, sp.Float, sp.Rational)):
            return self.registry.dimensionless

        if isinstance(expr, sp.Add):
            terms = expr.args
            first_unit = self.check_expression_consistency(terms[0])
            for term in terms[1:]:
                term_unit = self.check_expression_consistency(term)
                if first_unit.dimensionality != term_unit.dimensionality:
                    raise TypeError(
                        f"Dimensional Error: Cannot add terms with incompatible dimensions: {first_unit} vs {term_unit} in expression '{expr}'"
                    )
            return first_unit

        if isinstance(expr, sp.Mul):
            result_unit = self.check_expression_consistency(expr.args[0])
            for arg in expr.args[1:]:
                result_unit = result_unit * self.check_expression_consistency(arg)
            return result_unit

        if isinstance(expr, sp.Pow):
            base, exponent = expr.args
            base_unit = self.check_expression_consistency(base)
            if isinstance(exponent, (int, float, sp.Integer, sp.Float, sp.Rational)):
                return base_unit ** float(exponent)
            exp_unit = self.check_expression_consistency(exponent)
            if exp_unit.dimensionality != self.registry.dimensionless.dimensionality:
                raise TypeError(f"Dimensional Error: Exponent '{exponent}' must be dimensionless, got {exp_unit}.")
            if base_unit.dimensionality == self.registry.dimensionless.dimensionality:
                return self.registry.dimensionless
            raise NotImplementedError(f"Variable non-numeric exponent '{exponent}' for dimensioned base '{base}' is not supported.")

        if hasattr(expr, 'func'):
            func = expr.func
            if func == sp.sqrt:
                arg_unit = self.check_expression_consistency(expr.args[0])
                return arg_unit ** 0.5

            if func in (sp.Abs, sp.re, sp.im, sp.conjugate):
                return self.check_expression_consistency(expr.args[0])

            if func in (sp.sign, sp.Heaviside):
                self.check_expression_consistency(expr.args[0])
                return self.registry.dimensionless

            if func in (sp.Min, sp.Max):
                args = expr.args
                first_unit = self.check_expression_consistency(args[0])
                for arg in args[1:]:
                    arg_unit = self.check_expression_consistency(arg)
                    if first_unit.dimensionality != arg_unit.dimensionality:
                        raise TypeError(
                            f"Dimensional Error: Incompatible units in {func.__name__}: {first_unit} vs {arg_unit}"
                        )
                return first_unit

            if func == sp.atan2:
                u1 = self.check_expression_consistency(expr.args[0])
                u2 = self.check_expression_consistency(expr.args[1])
                if u1.dimensionality != u2.dimensionality:
                    raise TypeError(f"Dimensional Error: Arguments to atan2 must have matching dimensions, got {u1} and {u2}.")
                return self.registry.dimensionless

            bessel_family = {sp.besselj, sp.bessely, sp.besseli, sp.besselk}
            if func in bessel_family:
                for arg in expr.args:
                    arg_unit = self.check_expression_consistency(arg)
                    if arg_unit.dimensionality != self.registry.dimensionless.dimensionality:
                        raise TypeError(f"Dimensional Error: Arguments to {func.__name__} must be dimensionless, got {arg_unit}.")
                return self.registry.dimensionless

            transcendental = {
                sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
                sp.sinh, sp.cosh, sp.tanh, sp.asin, sp.acos, sp.atan,
                sp.erf, sp.erfc, sp.gamma, sp.zeta,
            }
            if func in transcendental:
                if len(expr.args) != 1:
                    raise TypeError(f"Transcendental function {func.__name__} must have exactly one argument.")
                arg_unit = self.check_expression_consistency(expr.args[0])
                if arg_unit.dimensionality != self.registry.dimensionless.dimensionality:
                    raise TypeError(f"Dimensional Error: Argument to {func.__name__} must be dimensionless, got {arg_unit}.")
                return self.registry.dimensionless

            for arg in expr.args:
                arg_unit = self.check_expression_consistency(arg)
                if arg_unit.dimensionality != self.registry.dimensionless.dimensionality:
                    raise TypeError(f"Dimensional Error: Generic function argument '{arg}' must be dimensionless.")
            return self.registry.dimensionless

        raise NotImplementedError(f"Expression type {type(expr)} is not supported.")

    def validate_homogeneity(self, lhs_expr: sp.Expr, rhs_expr: sp.Expr) -> bool:
        """Verifies that LHS and RHS of an equation have identical physical dimensionality."""
        lhs_unit = self.check_expression_consistency(lhs_expr)
        rhs_unit = self.check_expression_consistency(rhs_expr)
        if lhs_unit.dimensionality != rhs_unit.dimensionality:
            raise TypeError(f"Homogeneity Violation: LHS is {lhs_unit}, RHS is {rhs_unit}.")
        return True


class BuckinghamPiEngine:
    """Computes dimensionless Pi-groups using exact rational nullspace reduction of dimensional matrix."""

    @staticmethod
    def _nullspace_powers(
        variables: dict[str, dict[str, int]],
    ) -> tuple[list[str], list[list[int]]]:
        """Shared nullspace -> minimal coprime integer exponent vectors."""
        var_names = list(variables.keys())
        if not var_names:
            return [], []
        all_dim_keys: set[str] = set()
        for v in var_names:
            all_dim_keys.update(variables[v].keys())
        standard_dims = ['M', 'L', 'T', 'Theta', 'N', 'I', 'J']
        active_dims = [d for d in standard_dims if d in all_dim_keys and any(variables[v].get(d, 0) != 0 for v in var_names)]
        custom_dims = sorted([d for d in all_dim_keys if d not in standard_dims and any(variables[v].get(d, 0) != 0 for v in var_names)])
        active_dims.extend(custom_dims)
        if not active_dims:
            return var_names, [[1 if j == i else 0 for j in range(len(var_names))] for i in range(len(var_names))]
        dim_matrix = [[variables[v].get(d, 0) for v in var_names] for d in active_dims]
        nullspace = sp.Matrix(dim_matrix).nullspace()
        if not nullspace:
            return var_names, []
        powers: list[list[int]] = []
        for vec in nullspace:
            sym_powers = [sp.Rational(vec[j]) for j in range(len(var_names))]
            lcm_den = 1
            for sp_val in sym_powers:
                lcm_den = sp.lcm(lcm_den, sp_val.q)
            int_powers = [int(sp_val.p * (lcm_den // sp_val.q)) for sp_val in sym_powers]
            non_zeros = [abs(p) for p in int_powers if p != 0]
            if non_zeros:
                common_gcd = non_zeros[0]
                for p in non_zeros[1:]:
                    common_gcd = sp.gcd(common_gcd, p)
                int_powers = [p // common_gcd for p in int_powers]
            first_nz = next((p for p in int_powers if p != 0), None)
            if first_nz is not None and first_nz < 0:
                int_powers = [-p for p in int_powers]
            powers.append(int_powers)
        return var_names, powers

    @staticmethod
    def compute_pi_groups(variables: dict[str, dict[str, int]]) -> list[str]:
        """
        variables: dict of var_name -> {'M': a, 'L': b, 'T': c, 'Theta': d, 'N': e, 'I': f, 'J': g}
        Returns list of string representations of independent dimensionless Pi-groups.
        """
        var_names, powers = BuckinghamPiEngine._nullspace_powers(variables)
        if not var_names:
            return []
        if powers and len(powers) == len(var_names) and all(sum(1 for p in row if p) == 1 for row in powers):
            # All variables already dimensionless.
            return [f"Pi_{i+1} = {v}" for i, v in enumerate(var_names)]
        if not powers:
            return []

        pi_groups: list[str] = []
        for i, int_powers in enumerate(powers):

            num_terms = []
            den_terms = []
            for j in range(len(var_names)):
                power = int_powers[j]
                if power == 0:
                    continue
                var = var_names[j]
                if power > 0:
                    if power == 1:
                        num_terms.append(var)
                    else:
                        num_terms.append(f"{var}^{power}")
                else:
                    pos_power = -power
                    if pos_power == 1:
                        den_terms.append(var)
                    else:
                        den_terms.append(f"{var}^{pos_power}")

            numerator_str = " * ".join(num_terms) if num_terms else "1"
            if den_terms:
                denominator_str = " * ".join(den_terms)
                pi_str = f"({numerator_str}) / ({denominator_str})" if len(num_terms) > 1 or len(den_terms) > 1 else f"{numerator_str} / {denominator_str}"
            else:
                pi_str = numerator_str
            pi_groups.append(f"Pi_{i+1} = {pi_str}")

        return pi_groups

    @staticmethod
    def compute_pi_groups_symbolic(variables: dict[str, dict[str, int]]) -> list[sp.Expr]:
        """Same nullspace as :meth:`compute_pi_groups` but as symbolic products.

        Returns e.g. ``[T*sqrt(g/L)]`` instead of ``["Pi_1 = ..."]`` so callers
        can substitute, differentiate, or check dimensionality directly.
        """
        var_names, powers = BuckinghamPiEngine._nullspace_powers(variables)
        if not powers:
            return []
        syms = {name: sp.Symbol(name) for name in var_names}
        groups: list[sp.Expr] = []
        for int_powers in powers:
            expr: sp.Expr = sp.Integer(1)
            for name, power in zip(var_names, int_powers):
                if power:
                    expr = expr * sp.Pow(syms[name], int(power))
            groups.append(expr)
        return groups


class VectorizedDimensionalEngine:
    """High-throughput vectorized dimensional engine supporting arbitrary basis dimensions and domains."""

    # 7 SI Base Dimensions: [Mass, Length, Time, Temperature, Amount, Current, Luminous Intensity]
    #
    # Exponents are stored as int64, not int8: with int8 a dimension raised to a
    # power above 127 wrapped silently (``np.int8(100) * np.int8(2) == -56``), so a
    # large exponent could make an inconsistent model pass the homogeneity check.
    # Exponents are small in practice, but the failure mode is silent, which is
    # exactly the failure mode worth paying 8 extra bytes per vector to remove.
    DIM_MAP: ClassVar[dict[str, np.ndarray]] = {
        'dimensionless': np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        'kilogram':      np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        'meter':         np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.int64),
        'second':        np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.int64),
        'kelvin':        np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int64),
        'mole':          np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.int64),
        'ampere':        np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.int64),
        'candela':       np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.int64),
    }

    # Common derived unit dimension vectors
    DIM_MAP['newton']  = DIM_MAP['kilogram'] + DIM_MAP['meter'] - 2 * DIM_MAP['second']
    DIM_MAP['joule']   = DIM_MAP['kilogram'] + 2 * DIM_MAP['meter'] - 2 * DIM_MAP['second']
    DIM_MAP['watt']    = DIM_MAP['kilogram'] + 2 * DIM_MAP['meter'] - 3 * DIM_MAP['second']
    DIM_MAP['pascal']  = DIM_MAP['kilogram'] - DIM_MAP['meter'] - 2 * DIM_MAP['second']
    DIM_MAP['coulomb'] = DIM_MAP['second'] + DIM_MAP['ampere']
    DIM_MAP['volt']    = DIM_MAP['joule'] - DIM_MAP['coulomb']
    DIM_MAP['farad']   = DIM_MAP['coulomb'] - DIM_MAP['volt']
    DIM_MAP['hertz']   = -DIM_MAP['second']

    def __init__(
        self,
        basis_dimensions: list[str] | None = None,
        unit_map: dict[str, np.ndarray] | None = None
    ):
        if basis_dimensions is None:
            self.basis_dimensions = ['Mass', 'Length', 'Time', 'Temperature', 'Amount', 'Current', 'Luminous Intensity']
            self.unit_map = dict(self.DIM_MAP)
        else:
            self.basis_dimensions = list(basis_dimensions)
            self.unit_map = dict(unit_map or {})
            n = len(self.basis_dimensions)
            if 'dimensionless' not in self.unit_map:
                self.unit_map['dimensionless'] = np.zeros(n, dtype=np.int64)
            for idx, dim in enumerate(self.basis_dimensions):
                d_key = dim.lower()
                if d_key not in self.unit_map:
                    vec = np.zeros(n, dtype=np.int64)
                    vec[idx] = 1
                    self.unit_map[d_key] = vec

    def register_unit(self, name: str, vector: np.ndarray) -> None:
        arr = np.asarray(vector, dtype=np.int64)
        if arr.shape[-1] != len(self.basis_dimensions):
            raise ValueError(
                f"Vector dimension {arr.shape[-1]} does not match basis dimension count {len(self.basis_dimensions)}"
            )
        self.unit_map[name.lower()] = arr

    def get_unit_vector(self, name: str) -> np.ndarray:
        name_lower = name.lower()
        if name_lower in self.unit_map:
            return self.unit_map[name_lower]
        if len(self.basis_dimensions) == 7 and name_lower in self.DIM_MAP:
            return self.DIM_MAP[name_lower]
        raise KeyError(f"Unit '{name}' is not registered in this dimensional basis: {self.basis_dimensions}")

    @classmethod
    def create_si_engine(cls) -> VectorizedDimensionalEngine:
        return cls()

    @classmethod
    def create_economics_engine(cls) -> VectorizedDimensionalEngine:
        engine = cls(basis_dimensions=['Currency', 'Time', 'Quantity'])
        engine.register_unit('dollar', np.array([1, 0, 0], dtype=np.int64))
        engine.register_unit('second', np.array([0, 1, 0], dtype=np.int64))
        engine.register_unit('item', np.array([0, 0, 1], dtype=np.int64))
        engine.register_unit('price', np.array([1, 0, -1], dtype=np.int64))
        engine.register_unit('revenue_rate', np.array([1, -1, 0], dtype=np.int64))
        return engine

    @classmethod
    def create_computer_science_engine(cls) -> VectorizedDimensionalEngine:
        engine = cls(basis_dimensions=['Data', 'Packet', 'Time', 'Compute'])
        engine.register_unit('bit', np.array([1, 0, 0, 0], dtype=np.int64))
        engine.register_unit('byte', np.array([1, 0, 0, 0], dtype=np.int64))
        engine.register_unit('packet', np.array([0, 1, 0, 0], dtype=np.int64))
        engine.register_unit('second', np.array([0, 0, 1, 0], dtype=np.int64))
        engine.register_unit('flop', np.array([0, 0, 0, 1], dtype=np.int64))
        engine.register_unit('bps', np.array([1, 0, -1, 0], dtype=np.int64))
        engine.register_unit('flops', np.array([0, 0, -1, 1], dtype=np.int64))
        return engine

    @classmethod
    def batch_validate_multiplication(cls, dim_a: np.ndarray, dim_b: np.ndarray) -> np.ndarray:
        result: np.ndarray = dim_a + dim_b
        return result

    @classmethod
    def batch_validate_division(cls, dim_a: np.ndarray, dim_b: np.ndarray) -> np.ndarray:
        result: np.ndarray = dim_a - dim_b
        return result

    @classmethod
    def batch_validate_addition(cls, dim_a: np.ndarray, dim_b: np.ndarray) -> np.ndarray:
        dim_a = np.asarray(dim_a)
        dim_b = np.asarray(dim_b)
        mismatch = np.any(dim_a != dim_b, axis=-1)
        if np.any(mismatch):
            count = int(np.sum(mismatch)) if mismatch.ndim > 0 else 1
            raise TypeError(f"Dimensional mismatch detected in {count} batch entries!")
        return dim_a

    @classmethod
    def batch_validate_subtraction(cls, dim_a: np.ndarray, dim_b: np.ndarray) -> np.ndarray:
        return cls.batch_validate_addition(dim_a, dim_b)

    @classmethod
    def batch_validate_power(cls, dim_a: np.ndarray, power: float) -> np.ndarray:
        return np.asarray(dim_a) * power

    @classmethod
    def batch_check_homogeneity(cls, lhs_dims: np.ndarray, rhs_dims: np.ndarray) -> np.ndarray:
        """Returns boolean mask of matching dimensions across batches."""
        lhs_dims = np.asarray(lhs_dims)
        rhs_dims = np.asarray(rhs_dims)
        return np.asarray(np.all(lhs_dims == rhs_dims, axis=-1))


class ClosedLoopAssuranceSystem:
    """Model registry, unit gatekeeper, and safe numerical evaluator."""

    def __init__(
        self,
        assurance_engine: DimensionalAssuranceEngine,
        strict_units: bool = False,
    ):
        """
        Parameters
        ----------
        strict_units:
            When True, a bare number supplied for a variable whose registered unit
            is not dimensionless is refused instead of being interpreted in that
            unit. The default keeps the permissive behaviour for compatibility
            (every internal call site passes bare numbers), but the assumption is
            no longer silent: it is logged once per variable and recorded in
            :attr:`unit_assumptions`, so a wrong-unit evaluation leaves a trace
            instead of quietly producing a result.
        """
        self.engine = assurance_engine
        self.registered_models: dict[str, dict[str, Any]] = {}
        self.constants_values: dict[str, float] = dict(PHYSICAL_CONSTANTS.values_dict())
        self.strict_units = strict_units
        #: ``{variable: interpreted_unit}`` for every bare number interpreted in a
        #: registered unit. Read-only evidence of what was assumed, not guessed.
        self.unit_assumptions: dict[str, str] = {}

    def register_model_spec(
        self,
        model_name: str,
        expression: sp.Expr,
        expected_unit_str: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        computed_unit = self.engine.check_expression_consistency(expression)
        target_unit = self.engine.registry(expected_unit_str)
        if computed_unit.dimensionality != target_unit.dimensionality:
            raise TypeError(
                f"Model '{model_name}' dimensional mismatch: computed {computed_unit} vs expected {target_unit}"
            )
        self.registered_models[model_name] = {
            'expression': expression,
            'computed_unit': computed_unit,
            'expected_unit': target_unit,
            'metadata': metadata or {}
        }

    def _normalize_variable_value(self, name: str, value: QuantityLike) -> float:
        sym = sp.Symbol(name)
        if sym not in self.engine.symbol_units:
            raise ValueError(f"Variable '{name}' is not registered with any unit.")
        unit = self.engine.symbol_units[sym]
        if isinstance(value, pint.Quantity):
            return float(value.to(unit).magnitude)
        if isinstance(value, (int, float, np.number, sp.Integer, sp.Float, sp.Rational, sp.Number)):
            if unit.dimensionality != ureg.dimensionless.dimensionality:
                # A bare number carries no unit. Interpreting it in the registered
                # unit is a guess about the caller's intent, so it is refused under
                # strict mode and recorded otherwise -- never silent.
                if self.strict_units:
                    raise TypeError(
                        f"Variable '{name}' is registered as '{unit}' but a bare number was "
                        f"supplied ({value!r}); pass a pint quantity such as "
                        f"ureg.Quantity({value!r}, '{unit}'), or construct the system with "
                        "strict_units=False to interpret bare numbers in the registered unit."
                    )
                if name not in self.unit_assumptions:
                    logger.warning(
                        "Interpreting bare number %r for variable '%s' as '%s'; pass a pint "
                        "quantity to state the unit explicitly.",
                        value,
                        name,
                        unit,
                    )
                self.unit_assumptions[name] = str(unit)
            return float(value)
        raise TypeError(f"Unsupported value type for '{name}': {type(value)}")

    def evaluate_model(self, model_name: str, variable_values: dict[str, QuantityLike]) -> float:
        if model_name not in self.registered_models:
            raise ValueError(f"Model '{model_name}' is not registered.")
        model_spec = self.registered_models[model_name]
        expr = model_spec['expression']

        full_context = dict(self.constants_values)
        for name, val in variable_values.items():
            full_context[name] = self._normalize_variable_value(name, val)

        sub_map = {sp.Symbol(k): v for k, v in full_context.items()}
        missing = [s for s in expr.free_symbols if s not in sub_map]
        if missing:
            raise ValueError(
                f"Missing required variable(s) to evaluate model '{model_name}': {[s.name for s in missing]}"
            )

        numeric_expr = expr.subs(sub_map)
        if numeric_expr in (sp.zoo, sp.oo, -sp.oo, sp.nan):
            raise ArithmeticError(f"Model evaluated to non-finite value: {numeric_expr}")

        try:
            val = complex(numeric_expr)
        except TypeError as e:
            raise ArithmeticError(f"Failed to evaluate expression to numerical value: {numeric_expr}") from e

        if abs(val.imag) > 1e-12:
            raise ArithmeticError(f"Model evaluated to non-real complex output: {val}")

        real_val = float(val.real)
        if np.isnan(real_val) or np.isinf(real_val):
            raise ArithmeticError(f"Model evaluated to invalid numeric output: {real_val}")

        return real_val
