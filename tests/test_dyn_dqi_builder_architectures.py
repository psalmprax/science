"""The DQI builder's architecture registry may list only architectures that exist.

Status: audit-only

``AVAILABLE_ARCHITECTURES`` is the gate between a caller and a circuit build, so every
name in it is a claim that the architecture is buildable. It used to be a tuple subclass
whose ``__contains__`` answered ``True`` for ``single_array_schur`` while its ``__eq__``
still reported the value as ``("proos_zalka", "gluzdov_v6")``; the test that shipped with
it asserted both, and both passed. These pins replace that test with the honest one:

1. the registry is a *plain* tuple naming exactly the architectures with code behind it;
2. no ``__contains__``/``__eq__`` override can re-admit a name the tuple does not hold;
3. every retired alias is refused by ``build_candidate`` rather than dispatched;
4. the retired single-array stub is not importable from the package — it lives in the
   untracked ``build/apedp_stubs/``, where nothing in ``apedp_dyn`` advertises it.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from apedp_dyn.quantum_compilation import dqi_builder

RETIRED_NAMES = ("single_array_schur", "single_array", "schur_forney")


def test_the_registry_is_a_plain_tuple_of_buildable_architectures():
    """Exactly the two architectures the adapter dispatches to, as builtin-tuple semantics."""
    registry = dqi_builder.AVAILABLE_ARCHITECTURES
    assert type(registry) is tuple, (
        f"the registry must be a plain tuple, not {type(registry).__name__}; a subclass "
        "can answer __contains__ differently from what it holds"
    )
    assert registry == ("proos_zalka", "gluzdov_v6")
    assert dqi_builder.DEFAULT_ARCHITECTURE in registry

    # Every advertised name resolves to code: the V6 scaffold in the package, the reference
    # decoder in the package or in the competition bundle the sweep runner puts on the path.
    root = pathlib.Path(dqi_builder.__file__).resolve().parent
    bundle = (
        root.parent / "sandbox" / "google-verifiable-quantum-circuit-compilation"
        / "verifiers" / "dqi" / "dqi_reference_circuit.py"
    )
    assert (root / "dqi_gluzdov_v6.py").is_file(), "gluzdov_v6 is advertised without a module"
    assert (root / "dqi_reference_circuit.py").is_file() or bundle.is_file(), (
        "proos_zalka is advertised without a reference decoder in either location"
    )


def test_no_override_can_re_admit_a_name_the_tuple_does_not_hold():
    """`in` and `==` must be the builtin tuple operations, so both answer the same set."""
    registry = dqi_builder.AVAILABLE_ARCHITECTURES
    assert type(registry).__contains__ is tuple.__contains__
    assert type(registry).__eq__ is tuple.__eq__
    for name in RETIRED_NAMES:
        assert name not in registry
        assert not any(name in entry for entry in registry)


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_retired_aliases_are_refused_not_dispatched(name):
    """The refusal happens before any backend import, so it needs no kickmix."""
    with pytest.raises(ValueError, match="Unknown architecture"):
        dqi_builder.build_candidate(n_iters=2, architecture=name)


def test_the_retired_single_array_stub_is_not_importable_from_the_package():
    """A retired stub must not resolve on the package path, or the registry could grow back."""
    assert importlib.util.find_spec(
        "apedp_dyn.quantum_compilation.dqi_schur_forney"
    ) is None
    assert importlib.util.find_spec(
        "apedp_dyn.quantum_compilation.dfthcsos_riemannian_15batch"
    ) is None
