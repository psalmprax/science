"""Repository adapter for building the official Proos–Zalka DQI decoder.

This adapter deliberately does not implement a new decoder. It instantiates the
existing dqi_reference_circuit.ProosAndZalkaDecoder and makes n_iters explicit so
the official verifier can adjudicate each compiled circuit.

Run from a science checkout with its root on PYTHONPATH, and with the official
competition verifier directory exposed by scripts/sweep_dqi_eea_bound.py.
"""
from __future__ import annotations

from typing import Any


def build_candidate(*, n_iters: int) -> Any:
    """Build a DQI circuit for the requested EEA iteration cap.

    The 70 syndrome registers and 4,095 error-mask registers match the benchmark
    instance used in the repository's documented DQI reference build.
    """
    if isinstance(n_iters, bool) or not isinstance(n_iters, int) or n_iters <= 0:
        raise ValueError("n_iters must be a positive integer")

    # Imported inside the function because the sweep runner adds the competition
    # bundle's kickmix/ and verifiers/ directories to sys.path before loading us.
    import kickmix as km

    # Prefer the project's own reference generator when it exists. The fallback
    # supports checkouts where the generator is supplied by the official bundle.
    try:
        import apedp_dyn.quantum_compilation.dqi_reference_circuit as DR
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "apedp_dyn.quantum_compilation.dqi_reference_circuit",
            "apedp_dyn.quantum_compilation",
        }:
            raise
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
