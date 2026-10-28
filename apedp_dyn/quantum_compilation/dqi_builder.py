"""Repository adapter for building the official Proos–Zalka DQI decoder.

This adapter deliberately does not reimplement the reference decoder. It
instantiates the existing ``dqi_reference_circuit.ProosAndZalkaDecoder`` and
makes ``n_iters`` explicit so the official verifier can adjudicate each
compiled circuit.

An alternative architecture is selectable by name:

``architecture="proos_zalka"`` (default)
    The official reference decoder. This is the only architecture backed by a
    sealed, verifier-receipted baseline row (``n_iters=222``), so it stays the
    default and the sweep harness keeps reproducing that row.

``architecture="gluzdov_v6"``
    The degree-register-recycling scaffold in
    ``apedp_dyn.quantum_compilation.dqi_gluzdov_v6``. Selecting it changes the
    circuit that gets compiled, but it does NOT establish the 14.396x target:
    that factor is a projection over ``Q`` and ``T_mean`` and only a verifier
    receipt on a compiled ``.kmb`` can confirm it. Any receipt produced from
    this architecture must be treated as a new candidate, never as a
    replacement for the pinned baseline row.

Run from a science checkout with its root on PYTHONPATH, and with the official
competition verifier directory exposed by scripts/sweep_dqi_eea_bound.py.
"""
from __future__ import annotations

from typing import Any

#: Architectures this adapter can build, in descending order of evidentiary
#: strength (receipted first).
AVAILABLE_ARCHITECTURES: tuple[str, ...] = ("proos_zalka", "gluzdov_v6")

DEFAULT_ARCHITECTURE: str = "proos_zalka"


def build_candidate(*, n_iters: int, architecture: str = DEFAULT_ARCHITECTURE) -> Any:
    """Build a DQI circuit for the requested EEA iteration cap.

    The 70 syndrome registers and 4,095 error-mask registers match the benchmark
    instance used in the repository's documented DQI reference build.
    """
    if isinstance(n_iters, bool) or not isinstance(n_iters, int) or n_iters <= 0:
        raise ValueError("n_iters must be a positive integer")
    if architecture not in AVAILABLE_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; "
            f"expected one of {AVAILABLE_ARCHITECTURES}"
        )

    if architecture == "gluzdov_v6":
        # Imported lazily so the reference path does not depend on the V6
        # module (and its numpy import) when only the baseline is built.
        from apedp_dyn.quantum_compilation import dqi_gluzdov_v6

        return dqi_gluzdov_v6.build_candidate(n_iters=n_iters)

    # Imported inside the function because the sweep runner adds the competition
    # bundle's kickmix/ and verifiers/ directories to sys.path before loading us.
    import kickmix as km

    # Prefer the project's own reference generator when it exists. The fallback
    # supports checkouts where the generator is supplied by the official bundle.
    try:
        from apedp_dyn.quantum_compilation import dqi_reference_circuit as DR
    except (ImportError, ModuleNotFoundError):
        from dqi import dqi_reference_circuit as DR

    field = km.GF2Field(12)
    builder = km.CircuitBuilder()
    syndromes = [
        builder.create_quantum_register(12, name=f"S{i}")
        for i in range(70)
    ]
    error_masks = [
        builder.create_classical_register(12, name=f"E{j}")
        for j in range(4095)
    ]

    decoder = DR.ProosAndZalkaDecoder(70, field)
    decoder.build_syndrome_decoding_mbuc(
        builder,
        syndromes=syndromes,
        error_mbuc=error_masks,
        n_iters=n_iters,
    )
    return builder.finish_circuit()
