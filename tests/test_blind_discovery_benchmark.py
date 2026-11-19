"""The blind discovery benchmark's machinery (preregistration/PREREG-BLIND-L2-2026-11.md).

None of these runs the engine on a benchmark law: the laws are for the one
preregistered run. The end-to-end test uses a toy law that is not in either
family.
"""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_blind_discovery_benchmark as bench

from apedp_dyn.discovery_l2.dimensional_types import Dimension


def _term_dimension(term: sp.Expr, dims: dict[str, Dimension]) -> Dimension:
    total = Dimension()
    for name, exp in bench.term_exponents(term).items():
        total = total * (dims[name] ** exp)
    return total


@pytest.mark.parametrize("law", [*bench.known_laws(), *bench.invented_laws()], ids=lambda law: law.law_id)
def test_every_law_is_dimensionally_consistent(law):
    """Each additive term carries exactly the target's dimension, so the engine
    is never asked for a law its own type system forbids."""
    for term in sp.Add.make_args(sp.expand(sp.expand_power_base(law.sympy(), force=True))):
        assert _term_dimension(term, law.dimensions) == law.target_dimension, (law.law_id, term)


def test_the_families_have_the_declared_sizes_and_disjoint_names():
    known, invented = bench.known_laws(), bench.invented_laws()
    assert len(known) == 12 and len(invented) == bench.N_INVENTED == 24
    assert {law.family for law in known} == {"known"} and {law.family for law in invented} == {"invented"}
    physics_words = {name for law in known for name in law.dimensions}
    assert not physics_words & {name for law in invented for name in law.dimensions}
    # Seeded: the same draw every time.
    assert [law.expression for law in invented] == [law.expression for law in bench.invented_laws()]


def test_invented_laws_use_only_the_engines_powers():
    from apedp_dyn.discovery_l2.ast_generator import DEFAULT_POWERS

    allowed = {*DEFAULT_POWERS, Fraction(1)}
    for law in bench.invented_laws():
        for term in sp.Add.make_args(sp.expand(law.sympy())):
            for name, exp in bench.term_exponents(term).items():
                assert exp in allowed, (law.law_id, name, exp)


def test_recovery_is_judged_up_to_coefficients_and_not_beyond():
    x, y = sp.symbols("x y")
    assert bench.monomials(2.5 * x * y) == bench.monomials(x * y)
    assert bench.monomials(3 * x**2 + 0.1 * y) == bench.monomials(x**2 + y)
    assert bench.monomials(sp.sqrt(x / y)) == bench.monomials(x**sp.Rational(1, 2) * y**sp.Rational(-1, 2))
    assert bench.monomials(x * y) != bench.monomials(x * y + x)  # an extra term is not the law
    assert bench.monomials(x**2) != bench.monomials(x**3)


def test_end_to_end_on_a_toy_law_outside_both_families(tmp_path):
    toy = bench.Law("toy", "known", {"alpha": Dimension(l=1), "beta": Dimension(t=-1)},
                    Dimension(l=1, t=-1), "alpha*beta")
    # 1 % noise, not 0: on noise-free data the referee refuses an exact fit
    # whenever the residuals come out exactly zero, which depends on the
    # platform's floating point (M-01 in docs/SYSTEM_FLAWS_AUDIT.md).
    results = bench.run(tmp_path / "out", laws=[toy], noise_levels=(0.01,), seeds=(0,), check=False)
    rows = [json.loads(line) for line in (tmp_path / "out" / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "recovered", rows[0]
    assert rows[0]["distractor"] == "z_other"
    cell = results["summary"]["known|0.01"]
    assert cell["n"] == 1 and cell["recovered"] == 1 and cell["recovery_rate"] == 1.0


def test_it_refuses_to_reuse_an_output_directory(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        bench.run(tmp_path / "out", laws=[], check=False)


def test_it_refuses_to_run_once_a_frozen_file_has_changed(tmp_path):
    for name in bench.FROZEN:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    digest = tmp_path / "DIGEST.json"
    digest.write_text(json.dumps({"sha256": bench.frozen_digests(tmp_path)}))
    bench.check_frozen(tmp_path, digest)
    (tmp_path / bench.FROZEN[0]).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="changed since it was frozen"):
        bench.check_frozen(tmp_path, digest)


def test_the_frozen_files_match_the_recorded_digest():
    recorded = json.loads(bench.DIGEST.read_text())["sha256"]
    assert recorded == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in bench.FROZEN}
