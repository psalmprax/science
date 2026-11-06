"""Correctness checks for exact GF(2) weight-space transformations."""
from __future__ import annotations

import numpy as np

from gf2mul_llm_adjustable import (
    GF2FieldConfig,
    LLMTunableGF2Multiplier,
    MMLMMultiplier,
    TensorDecomposition,
    WSPConfig,
    WSPLayer,
    build_multiplication_tensor,
    build_schoolbook_decomposition,
)


def _redundant_exact_decomposition():
    tensor = np.ones((1, 1, 1), dtype=np.uint8)
    decomposition = TensorDecomposition(
        rank=2,
        U=np.array([[1], [0]], dtype=np.uint8),
        V=np.ones((2, 1), dtype=np.uint8),
        W=np.ones((2, 1), dtype=np.uint8),
    )
    return tensor, decomposition


def test_importance_pruning_only_removes_exactly_redundant_atoms():
    tensor, decomposition = _redundant_exact_decomposition()
    layer = WSPLayer(WSPConfig(weight_scheme="importance", sparsity_target=0.5))

    optimized, diagnostics = layer.apply_wsp(decomposition, tensor)

    assert optimized.rank == 1
    assert optimized.reconstruct(tensor)[0]
    assert diagnostics["pruned_atoms_removed"] == 1
    assert diagnostics["sparsity_achieved"] == 0.5


def test_inexact_input_is_not_modified_or_accepted():
    tensor = np.ones((1, 1, 1), dtype=np.uint8)
    decomposition = TensorDecomposition(
        rank=1,
        U=np.zeros((1, 1), dtype=np.uint8),
        V=np.ones((1, 1), dtype=np.uint8),
        W=np.ones((1, 1), dtype=np.uint8),
    )
    layer = WSPLayer(WSPConfig(weight_scheme="llm_tuned"))

    result, diagnostics = layer.apply_wsp(decomposition, tensor)

    assert result is decomposition
    assert not diagnostics["input_exact"]
    assert not diagnostics["accepted"]


def test_inexact_llm_transform_falls_back_to_original_decomposition():
    tensor, decomposition = _redundant_exact_decomposition()
    layer = WSPLayer(WSPConfig(
        weight_scheme="llm_tuned",
        llm_adjustable_params={"transform_type": "scale", "scale_u": 0},
    ))

    result, diagnostics = layer.apply_wsp(decomposition, tensor)

    assert result is decomposition
    assert decomposition.reconstruct(tensor)[0]
    assert diagnostics["candidate_exact"] is False
    assert not diagnostics["accepted"]


def test_invertible_gf2_mix_preserves_exactness():
    tensor, decomposition = _redundant_exact_decomposition()
    layer = WSPLayer(WSPConfig(
        weight_scheme="llm_tuned",
        llm_adjustable_params={
            "transform_type": "mix",
            "mix_matrix": [[1, 1], [0, 1]],
        },
    ))

    result, diagnostics = layer.apply_wsp(decomposition, tensor)

    assert result.reconstruct(tensor)[0]
    assert diagnostics["accepted"]
    assert diagnostics["transform_type"] == "mix"


def test_unsupported_new_atoms_and_unreachable_rank_fail_closed():
    tensor, decomposition = _redundant_exact_decomposition()
    new_atoms, diagnostics = WSPLayer(WSPConfig(
        weight_scheme="llm_tuned",
        llm_adjustable_params={"transform_type": "new_atoms"},
    )).apply_wsp(decomposition, tensor)
    assert new_atoms is decomposition
    assert "unsupported" in diagnostics["rejection_reason"]

    rank_limited, rank_diagnostics = WSPLayer(WSPConfig(
        weight_scheme="llm_tuned",
        llm_adjustable_params={"transform_type": "swap", "target_rank": 1},
    )).apply_wsp(decomposition, tensor)
    assert rank_limited.rank == 1
    assert rank_limited.reconstruct(tensor)[0]
    assert rank_diagnostics["target_rank_reached"]


def test_target_rank_above_current_rank_does_not_corrupt_rank_metadata():
    tensor = np.ones((1, 1, 1), dtype=np.uint8)
    decomposition = TensorDecomposition(
        rank=1,
        U=np.ones((1, 1), dtype=np.uint8),
        V=np.ones((1, 1), dtype=np.uint8),
        W=np.ones((1, 1), dtype=np.uint8),
    )
    layer = WSPLayer(WSPConfig(
        weight_scheme="llm_tuned",
        llm_adjustable_params={"transform_type": "swap", "target_rank": 3},
    ))

    result, diagnostics = layer.apply_wsp(decomposition, tensor)

    assert result.rank == 1
    assert result.reconstruct(tensor)[0]
    assert diagnostics["target_rank_reached"]


def test_schoolbook_baseline_is_exact_and_multiplier_fails_safe_on_placeholder_rank():
    degree = 3
    modulus = 0b1011
    tensor = build_multiplication_tensor(degree, modulus)
    baseline = build_schoolbook_decomposition(degree, modulus)
    assert baseline.reconstruct(tensor)[0]

    multiplier = MMLMMultiplier(
        degree=degree,
        field_config=GF2FieldConfig(
            degree=degree,
            irreducible_poly=modulus,
            modulus_int=modulus,
        ),
    )
    result = multiplier.get_decomposition(rank=1)
    assert result.reconstruct(tensor)[0]
    assert result.metadata["fallback_reason"].startswith("proposed rank-54")
    assert not result.metadata["rank_request_diagnostics"]["target_rank_reached"]


def test_llm_tuning_replaces_frozen_configs_without_losing_exactness():
    multiplier = LLMTunableGF2Multiplier(
        degree=3,
        field_config=GF2FieldConfig(
            degree=3,
            irreducible_poly=0b1011,
            modulus_int=0b1011,
        ),
    )

    report = multiplier.tune(
        primitive_element=3,
        weight_scheme="importance",
        sparsity_target=0.25,
    )

    assert multiplier.field_config.primitive_element == 3
    assert multiplier.wsp_config.weight_scheme == "importance"
    assert multiplier.wsp_config.sparsity_target == 0.25
    assert report["reconstruction_exact"]
