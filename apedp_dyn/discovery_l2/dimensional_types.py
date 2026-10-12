"""Physically Typed Dimensional Metrology for L2 Open Discovery.

Enforces SI dimensional homogeneity and type calculus during expression construction:
  - Multiplication: [u] * [v] = [u + v] in dimensional exponents
  - Division: [u] / [v] = [u - v] in dimensional exponents
  - Addition / Subtraction: [u] + [v] and [u] - [v] require [u] == [v]
  - Exponentiation: [u]^p requires constant rational exponent p
  - Transcendental (exp, log): requires [u] == [0, 0, 0, ...] (strictly dimensionless)
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from fractions import Fraction
from typing import Any


class DimensionalError(TypeError):
    """Raised when an operation violates physical dimensional homogeneity."""


def _to_fraction(val: int | float | str | Fraction) -> Fraction:
    """Convert scalar power to exact rational Fraction."""
    if isinstance(val, Fraction):
        return val
    if isinstance(val, int):
        return Fraction(val, 1)
    if isinstance(val, float):
        return Fraction(val).limit_denominator(1000)
    if isinstance(val, str):
        return Fraction(val)
    raise TypeError(f"Cannot convert {type(val).__name__} to dimensional exponent Fraction")


class Dimension:
    """Represents an SI physical dimension vector with exact rational exponents.

    Seven SI base dimensions:
      - m:  Mass (kg)
      - l:  Length (m)
      - t:  Time (s)
      - i:  Electric current (A)
      - th: Thermodynamic temperature (K)
      - n:  Amount of substance (mol)
      - j:  Luminous intensity (cd)
    """

    __slots__ = ("_i", "_j", "_l", "_m", "_n", "_p", "_q", "_t", "_th")

    # Explicit annotations so static analysis sees the ``__slots__`` attributes
    # that are populated via ``object.__setattr__`` in ``__init__``.
    _i: Fraction
    _j: Fraction
    _l: Fraction
    _m: Fraction
    _n: Fraction
    _p: Fraction
    _q: Fraction
    _t: Fraction
    _th: Fraction

    def __init__(
        self,
        m: int | float | str | Fraction = 0,
        l: int | float | str | Fraction = 0,  # noqa: E741
        t: int | float | str | Fraction = 0,
        i: int | float | str | Fraction = 0,
        th: int | float | str | Fraction = 0,
        n: int | float | str | Fraction = 0,
        j: int | float | str | Fraction = 0,
        p: int | float | str | Fraction = 0,  # Price / Currency [$]
        q: int | float | str | Fraction = 0,  # Quantity / Volume [Q]
        length: int | float | str | Fraction | None = None,
        price: int | float | str | Fraction | None = None,
        quantity: int | float | str | Fraction | None = None,
    ) -> None:
        actual_l = l if length is None else length
        actual_p = p if price is None else price
        actual_q = q if quantity is None else quantity
        object.__setattr__(self, "_m", _to_fraction(m))
        object.__setattr__(self, "_l", _to_fraction(actual_l))
        object.__setattr__(self, "_t", _to_fraction(t))
        object.__setattr__(self, "_i", _to_fraction(i))
        object.__setattr__(self, "_th", _to_fraction(th))
        object.__setattr__(self, "_n", _to_fraction(n))
        object.__setattr__(self, "_j", _to_fraction(j))
        object.__setattr__(self, "_p", _to_fraction(actual_p))
        object.__setattr__(self, "_q", _to_fraction(actual_q))

    @property
    def p(self) -> Fraction:
        return self._p

    @property
    def price(self) -> Fraction:
        return self._p

    @property
    def q(self) -> Fraction:
        return self._q

    @property
    def quantity(self) -> Fraction:
        return self._q

    @property
    def m(self) -> Fraction:
        return self._m

    @property
    def l(self) -> Fraction:  # noqa: E743
        return self._l

    @property
    def length(self) -> Fraction:
        return self._l

    @property
    def t(self) -> Fraction:
        return self._t

    @property
    def i(self) -> Fraction:
        return self._i

    @property
    def th(self) -> Fraction:
        return self._th

    @property
    def n(self) -> Fraction:
        return self._n

    @property
    def j(self) -> Fraction:
        return self._j

    def is_dimensionless(self) -> bool:
        """Return True if all dimensional powers are exactly zero."""
        return (
            self._m == 0
            and self._l == 0
            and self._t == 0
            and self._i == 0
            and self._th == 0
            and self._n == 0
            and self._j == 0
            and self._p == 0
            and self._q == 0
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return False
        return (
            self._m == other._m
            and self._l == other._l
            and self._t == other._t
            and self._i == other._i
            and self._th == other._th
            and self._n == other._n
            and self._j == other._j
            and self._p == other._p
            and self._q == other._q
        )

    def __hash__(self) -> int:
        sig = f"{self._m},{self._l},{self._t},{self._i},{self._th},{self._n},{self._j},{self._p},{self._q}"
        return int(zlib.crc32(sig.encode()))

    def __mul__(self, other: Dimension) -> Dimension:
        """Multiplication of physical quantities: dimensional exponents add ([u] * [v] = [u + v])."""
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(
            m=self._m + other._m,
            l=self._l + other._l,
            t=self._t + other._t,
            i=self._i + other._i,
            th=self._th + other._th,
            n=self._n + other._n,
            j=self._j + other._j,
            p=self._p + other._p,
            q=self._q + other._q,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        """Division of physical quantities: dimensional exponents subtract ([u] / [v] = [u - v])."""
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(
            m=self._m - other._m,
            l=self._l - other._l,
            t=self._t - other._t,
            i=self._i - other._i,
            th=self._th - other._th,
            n=self._n - other._n,
            j=self._j - other._j,
            p=self._p - other._p,
            q=self._q - other._q,
        )

    def __pow__(self, exponent: int | float | str | Fraction) -> Dimension:
        """Power of a physical quantity: dimensional exponents scale by rational p ([u^p] = p * [u])."""
        p = _to_fraction(exponent)
        return Dimension(
            m=self._m * p,
            l=self._l * p,
            t=self._t * p,
            i=self._i * p,
            th=self._th * p,
            n=self._n * p,
            j=self._j * p,
            p=self._p * p,
            q=self._q * p,
        )

    def check_compatible(self, other: Dimension, op_name: str = "+") -> None:
        """Raise DimensionalError if dimensions are not identical for addition/subtraction."""
        if self != other:
            raise DimensionalError(
                f"Cannot perform '{op_name}' on incompatible dimensions: {self} vs {other}"
            )

    def __add__(self, other: Dimension) -> Dimension:
        """Addition of physical quantities requires identical dimensions: [u] == [v]."""
        if not isinstance(other, Dimension):
            return NotImplemented
        self.check_compatible(other, "+")
        return self

    def __sub__(self, other: Dimension) -> Dimension:
        """Subtraction of physical quantities requires identical dimensions: [u] == [v]."""
        if not isinstance(other, Dimension):
            return NotImplemented
        self.check_compatible(other, "-")
        return self

    @classmethod
    def dimensionless(cls) -> Dimension:
        """Construct dimensionless dimension."""
        return cls()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Dimension:
        """Construct Dimension from dictionary of powers (supports case-insensitive symbols/names)."""
        mapping: dict[str, Any] = {}
        alias_map = {
            "m": "m",
            "mass": "m",
            "kg": "m",
            "l": "l",
            "length": "l",
            "meter": "l",
            "metre": "l",
            "t": "t",
            "time": "t",
            "sec": "t",
            "second": "t",
            "i": "i",
            "current": "i",
            "ampere": "i",
            "a": "i",
            "th": "th",
            "theta": "th",
            "temp": "th",
            "temperature": "th",
            "k": "th",
            "kelvin": "th",
            "n": "n",
            "amount": "n",
            "mol": "n",
            "mole": "n",
            "j": "j",
            "intensity": "j",
            "cd": "j",
            "candela": "j",
            "p": "p",
            "price": "p",
            "usd": "p",
            "currency": "p",
            "quote": "p",
            "q": "q",
            "quantity": "q",
            "vol": "q",
            "volume": "q",
            "shares": "q",
            "contracts": "q",
            "base": "q",
        }
        for k, v in d.items():
            norm_k = alias_map.get(k.lower())
            if norm_k:
                mapping[norm_k] = v
        return cls(**mapping)

    def __repr__(self) -> str:
        non_zero = []
        for name in ("m", "l", "t", "i", "th", "n", "j", "p", "q"):
            val = getattr(self, f"_{name}")
            if val != 0:
                val_str = (
                    str(val.numerator)
                    if val.denominator == 1
                    else f"{val.numerator}/{val.denominator}"
                )
                non_zero.append(f"{name}={val_str}")
        if not non_zero:
            return "Dimension.dimensionless()"
        return f"Dimension({', '.join(non_zero)})"

    def __str__(self) -> str:
        if self.is_dimensionless():
            return "1"
        parts = []
        labels = [
            ("M", self._m),
            ("L", self._l),
            ("T", self._t),
            ("I", self._i),
            ("Theta", self._th),
            ("N", self._n),
            ("J", self._j),
            ("P", self._p),
            ("Q", self._q),
        ]
        for label, p in labels:
            if p != 0:
                p_str = str(p.numerator) if p.denominator == 1 else f"{p.numerator}/{p.denominator}"
                parts.append(f"{label}^{p_str}" if p != 1 else label)
        return " ".join(parts)


DIMENSIONLESS = Dimension()
BaseDimension = Dimension

# Semantic Financial and Market Dimension Constants
PRICE_DIMENSION = Dimension(p=1)
LOG_PRICE_STATE = DIMENSIONLESS
RETURN_VELOCITY = Dimension(t=-1)
ACCELERATION_RATE = Dimension(t=-2)
VOLATILITY = Dimension(t=Fraction(-1, 2))
VOLUME_QUANTITY = Dimension(q=1)
ORDER_FLOW = Dimension(q=1, t=-1)
DOLLAR_FLOW = Dimension(p=1, q=1, t=-1)
SPREAD_DIMENSION = Dimension(p=1)

