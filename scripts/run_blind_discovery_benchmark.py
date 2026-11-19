"""The blind discovery benchmark frozen by preregistration/PREREG-BLIND-L2-2026-11.md.

Does the open discovery engine (``L2DiscoveryEngine``) find laws, and does it
find laws nobody could have memorised as readily as textbook ones?

    python scripts/run_blind_discovery_benchmark.py --out <new dir>

Two families of laws, generated and judged identically:

- **known**: twelve textbook laws (F = m a, E = m g h, T = sqrt(L / g), ...).
- **invented**: twenty-four laws drawn by a seeded generator over variables
  with meaningless names (``q1``, ``w7``) and randomly assigned SI
  dimensions, in the same grammar class as the known ones (products of
  powers from the engine's power set, and sums of two such terms). A model
  that had read the physics literature has no advantage on these.

Each law gets one irrelevant variable with its own dimension, 120 samples
with every variable uniform on [1, 10], and multiplicative noise at each of
``NOISE_LEVELS``, over ``SEEDS``. The engine receives only the data, the
names, the SI dimensions and the target's dimension.

A run counts as

- **recovered** if the engine promotes (``UNIQUE_SUPPORT``) a law whose
  additive terms, with numeric coefficients removed, are exactly the true
  law's;
- **false discovery** if it promotes anything else;
- **abstention** if it promotes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.discovery_l2.dimensional_types import Dimension  # noqa: E402
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine  # noqa: E402
from apedp_dyn.epistemic.adjudicate import EpistemicStatus  # noqa: E402

PREREGISTRATION_ID = "PREREG-BLIND-L2-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-BLIND-L2.json"
FROZEN = ("preregistration/PREREG-BLIND-L2-2026-11.md", "scripts/run_blind_discovery_benchmark.py")

OPERATORS = ("+", "-", "*", "/", "pow")
NOISE_LEVELS = (0.0, 0.01, 0.05, 0.10)
SEEDS = (0, 1)
N_SAMPLES = 120
N_INVENTED = 24
#: The powers the invented generator may use: the engine's own set, so that
#: the two families differ in familiarity and not in what the grammar allows.
INVENTED_POWERS = (Fraction(-2), Fraction(-1), Fraction(-1, 2), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(3))


@dataclass(frozen=True)
class Law:
    law_id: str
    family: str  # "known" | "invented"
    dimensions: dict[str, Dimension]  # relevant variables only
    target_dimension: Dimension
    expression: str  # sympy-parsable, numeric coefficients included

    def sympy(self) -> sp.Expr:
        return sp.sympify(self.expression, locals={name: sp.Symbol(name) for name in self.dimensions})


def known_laws() -> list[Law]:
    mass, length, time_ = Dimension(m=1), Dimension(l=1), Dimension(t=1)
    velocity, accel = Dimension(l=1, t=-1), Dimension(l=1, t=-2)
    force, area, volume = Dimension(m=1, l=1, t=-2), Dimension(l=2), Dimension(l=3)
    current, resistance = Dimension(i=1), Dimension(m=1, l=2, t=-3, i=-2)
    voltage = Dimension(m=1, l=2, t=-3, i=-1)
    energy = Dimension(m=1, l=2, t=-2)
    return [
        Law("newton_second", "known", {"mass": mass, "accel": accel}, force, "mass*accel"),
        Law("momentum", "known", {"mass": mass, "vel": velocity}, Dimension(m=1, l=1, t=-1), "mass*vel"),
        Law("kinetic_energy", "known", {"mass": mass, "vel": velocity}, energy, "0.5*mass*vel**2"),
        Law("power", "known", {"force": force, "vel": velocity}, Dimension(m=1, l=2, t=-3), "force*vel"),
        Law("work", "known", {"force": force, "dist": length}, energy, "force*dist"),
        Law("density", "known", {"mass": mass, "vol": volume}, Dimension(m=1, l=-3), "mass/vol"),
        Law("pressure", "known", {"force": force, "area": area}, Dimension(m=1, l=-1, t=-2), "force/area"),
        Law("ohm", "known", {"current": current, "resist": resistance}, voltage, "current*resist"),
        Law("angular_rate", "known", {"vel": velocity, "radius": length}, Dimension(t=-1), "vel/radius"),
        Law("centripetal", "known", {"vel": velocity, "radius": length}, accel, "vel**2/radius"),
        Law("potential_energy", "known", {"mass": mass, "grav": accel, "height": length}, energy, "mass*grav*height"),
        Law("pendulum_period", "known", {"length": length, "grav": accel}, time_, "6.283185307*(length/grav)**(1/2)"),
    ]


def _random_dimension(rng: np.random.Generator) -> Dimension:
    while True:
        exps = {k: int(rng.integers(-2, 3)) for k in ("m", "l", "t")}
        if any(exps.values()):
            return Dimension(**exps)


def _pow_str(name: str, p: Fraction) -> str:
    return name if p == 1 else f"{name}**({p.numerator}/{p.denominator})"


def invented_laws(n: int = N_INVENTED, seed: int = 20261119) -> list[Law]:
    """Laws over meaningless names, dimensionally consistent by construction.

    Three shapes, in rotation: c·x^p, c·x^p·y^q, and c1·x^p + c2·y^q, where
    the second term's variable is given whatever dimension makes the two
    terms add. Coefficients are dimensionless, uniform on [0.5, 3].
    """
    rng = np.random.default_rng(seed)
    letters = "bcdfghjkpqrsvwxz"
    laws: list[Law] = []
    for k in range(n):
        names = [f"{letters[int(rng.integers(len(letters)))]}{int(rng.integers(1, 10))}" for _ in range(2)]
        while names[1] == names[0]:
            names[1] = f"{letters[int(rng.integers(len(letters)))]}{int(rng.integers(1, 10))}"
        x, y = names
        dx = _random_dimension(rng)
        p = INVENTED_POWERS[int(rng.integers(len(INVENTED_POWERS)))]
        q = INVENTED_POWERS[int(rng.integers(len(INVENTED_POWERS)))]
        c1, c2 = (round(float(rng.uniform(0.5, 3.0)), 3) for _ in range(2))
        shape = k % 3
        if shape == 0:
            dims = {x: dx}
            expr, target = f"{c1}*{_pow_str(x, p)}", dx ** p
        elif shape == 1:
            dy = _random_dimension(rng)
            while dy == dx:
                dy = _random_dimension(rng)
            dims = {x: dx, y: dy}
            expr, target = f"{c1}*{_pow_str(x, p)}*{_pow_str(y, q)}", (dx ** p) * (dy ** q)
        else:
            target = dx ** p
            dy = target ** (1 / q)  # y^q must carry the first term's dimension
            dims = {x: dx, y: dy}
            expr = f"{c1}*{_pow_str(x, p)} + {c2}*{_pow_str(y, q)}"
        laws.append(Law(f"invented_{k:02d}", "invented", dims, target, expr))
    return laws


def term_exponents(term: sp.Expr) -> dict[str, Fraction]:
    """One additive term as {variable: exponent}, its numeric coefficient dropped.

    ``(x/y)**(1/2)``, ``sqrt(x)/sqrt(y)`` and ``x**(1/2) * y**(-1/2)`` are the
    same term; splitting every power of a product into powers of its factors
    (``expand_power_base(force=True)``; all benchmark variables are positive)
    is what makes them compare equal.
    """
    exponents: dict[str, Fraction] = {}
    split = sp.expand_power_base(sp.powdenest(term, force=True), force=True)
    for factor in sp.Mul.make_args(split):
        base, exp = factor.as_base_exp()
        if base.is_number:
            continue
        if not isinstance(base, sp.Symbol):
            raise ValueError(f"term {term} has a non-monomial factor {factor}")
        exponents[str(base)] = exponents.get(str(base), Fraction(0)) + Fraction(str(exp))
    return {k: v for k, v in exponents.items() if v != 0}


def monomials(expr: sp.Expr) -> frozenset[frozenset[tuple[str, Fraction]]]:
    """The law's additive terms, each as variable exponents, coefficients removed."""
    return frozenset(
        frozenset(term_exponents(term).items())
        for term in sp.Add.make_args(sp.expand(sp.expand_power_base(expr, force=True)))
    )


def _distractor(law: Law, rng: np.random.Generator) -> tuple[str, Dimension]:
    name = "z_other"
    while True:
        dim = _random_dimension(rng)
        if dim != law.target_dimension and all(dim != d for d in law.dimensions.values()):
            return name, dim


def judge_run(law: Law, noise: float, seed: int, engine: L2DiscoveryEngine) -> dict[str, Any]:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{PREREGISTRATION_ID}|{law.law_id}|{noise}|{seed}".encode()).digest()[:8], "big"))
    distractor, ddim = _distractor(law, rng)
    names = [*law.dimensions, distractor]
    data = {name: rng.uniform(1.0, 10.0, N_SAMPLES) for name in names}
    fn = sp.lambdify([sp.Symbol(n) for n in law.dimensions], law.sympy(), "numpy")
    clean = np.asarray(fn(*[data[n] for n in law.dimensions]), dtype=float)
    y = clean * (1.0 + noise * rng.standard_normal(N_SAMPLES)) if noise > 0 else clean
    dims = {**law.dimensions, distractor: ddim}
    started = time.perf_counter()
    result = engine.discover(
        X=data, Y=y, dimensions=dims, target_dimension=law.target_dimension,
        target_name="y", noise_sigma=(noise * float(np.mean(np.abs(y)))) if noise > 0 else None, seed=seed,
    )
    elapsed = time.perf_counter() - started
    status = result.adjudication.status
    promoted = result.adjudication.promoted
    found = None
    if promoted is not None:
        node = promoted.candidate.metadata.get("ast")
        found = node.to_sympy() if node is not None else None
    truth = monomials(law.sympy())
    recovered = status is EpistemicStatus.UNIQUE_SUPPORT and found is not None and monomials(found) == truth
    return {
        "law_id": law.law_id, "family": law.family, "noise": noise, "seed": seed,
        "truth": law.expression, "distractor": distractor,
        "status": status.name, "promoted": str(found) if found is not None else None,
        "outcome": "recovered" if recovered else ("false_discovery" if promoted is not None else "abstention"),
        "candidates_generated": len(result.generated_nodes), "seconds": round(elapsed, 3),
    }


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {}
    for family in ("known", "invented", "all"):
        for noise in (*NOISE_LEVELS, "all"):
            sel = [r for r in rows if (family == "all" or r["family"] == family) and (noise == "all" or r["noise"] == noise)]
            n = len(sel)
            counts = {o: sum(r["outcome"] == o for r in sel) for o in ("recovered", "false_discovery", "abstention")}
            cells[f"{family}|{noise}"] = {
                "n": n, **counts,
                "recovery_rate": counts["recovered"] / n if n else None,
                "recovery_interval_95": clopper_pearson(counts["recovered"], n),
                "false_discovery_rate": counts["false_discovery"] / n if n else None,
                "false_discovery_interval_95": clopper_pearson(counts["false_discovery"], n),
            }
    return cells


def frozen_digests(root: Path = ROOT) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FROZEN}


def check_frozen(root: Path = ROOT, digest_path: Path = DIGEST) -> None:
    recorded = json.loads(digest_path.read_text(encoding="utf-8"))["sha256"]
    if recorded != frozen_digests(root):
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")


def run(out: Path, laws: list[Law] | None = None, noise_levels: tuple[float, ...] = NOISE_LEVELS,
        seeds: tuple[int, ...] = SEEDS, check: bool = True) -> dict[str, Any]:
    if check:
        check_frozen()
    out.mkdir(parents=True, exist_ok=False)
    laws = laws if laws is not None else [*known_laws(), *invented_laws()]
    engine = L2DiscoveryEngine(operators=OPERATORS)
    rows = []
    with (out / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for law in laws:
            for noise in noise_levels:
                for seed in seeds:
                    row = judge_run(law, noise, seed, engine)
                    rows.append(row)
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
    results = {
        "preregistration": PREREGISTRATION_ID,
        "laws": [{"law_id": law.law_id, "family": law.family, "expression": law.expression,
                  "dimensions": {k: str(v) for k, v in law.dimensions.items()},
                  "target_dimension": str(law.target_dimension)} for law in laws],
        "summary": summarise(rows),
        "frozen": frozen_digests() if check else None,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = run(args.out)
    for key, cell in results["summary"].items():
        lo, hi = cell["recovery_interval_95"]
        print(f"{key:16s} n={cell['n']:3d} recovered={cell['recovered']:3d} ({cell['recovery_rate']:.0%}, 95% {lo:.0%}-{hi:.0%}) "
              f"false={cell['false_discovery']:3d} abstain={cell['abstention']:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
