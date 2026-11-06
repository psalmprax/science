"""Pins for the leftover scaffolds: placeholders that now refuse or measure.

Each test here corresponds to a module that presented a placeholder as a
result, and moves the knob to show the behaviour is tied to the input:

- the mechanical grammar's placeholder terms (a constant constraint basis, free
  symbols for spatial derivatives and trading indicators),
- the Triton backend claim in the batch steppers,
- the picture-to-video region masks that were fixed image coordinates,
- the hashed text encoder that was described as an embedding model,
- the derived sizing-curve rows that could reach a GENERATION verdict,
- the Tesla/energy constants that were hidden inside the arithmetic,
- the neural symbol regressor's teacher-forcing no-op and non-jittable step.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

GRAMMAR_TERMS = (
    "c1", "c2", "grad_x", "laplacian_x",
    "zscore_x", "ma_x", "mom_x", "rsi_like", "bollinger", "macd_like",
)


# ---------------------------------------------------------------- mechanical grammar


def test_every_placeholder_grammar_term_refuses_with_a_reason():
    from apedp_dyn.grammar import mechanical_grammar

    UNAVAILABLE_TERMS = mechanical_grammar.UNAVAILABLE_TERMS
    UnavailableGrammarTerm = mechanical_grammar.UnavailableGrammarTerm
    builders = mechanical_grammar._TERM_BUILDERS

    for term in GRAMMAR_TERMS:
        assert term in UNAVAILABLE_TERMS, term
        assert UNAVAILABLE_TERMS[term].strip(), term
        builder = builders[term]
        with pytest.raises(UnavailableGrammarTerm) as exc:
            builder(None, None, None, None, None)
        # The message names the term and says what is missing, so a caller does
        # not have to read the grammar to understand the refusal.
        assert term in str(exc.value)
        assert UNAVAILABLE_TERMS[term][:20] in str(exc.value)


def test_a_placeholder_term_cannot_be_materialised_into_a_model():
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics, Term
    from apedp_dyn.grammar.mechanical_grammar import (
        UnavailableGrammarTerm,
        materialize,
    )
    from apedp_dyn.problems.coupled import load as load_coupled

    problem = load_coupled()
    candidate = CandidateDynamics(
        lhs="xddot",
        terms=(Term("x", "a0"), Term("grad_x", "a1")),
        metadata={"family": "mechanical_2nd_order", "order": 2,
                  "state_variables": ["x", "v"], "input_variables": ["u"]},
        candidate_id="grad-x-probe",
    )
    with pytest.raises(UnavailableGrammarTerm):
        materialize(problem, candidate)


def test_a_placeholder_candidate_never_survives_pruning():
    """The slate still enumerates the names, but no placeholder reaches a fit.

    Enumeration is unchanged (the vocabulary keeps its names, so counts and
    canonical forms do not move); materialisation refuses, `prune_admissible`
    catches the refusal as a rejection reason, and nothing with a placeholder
    term is returned as a survivor.
    """
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics, Term
    from apedp_dyn.grammar.mechanical_grammar import prune_admissible
    from apedp_dyn.problems.coupled import load as load_coupled

    candidate = CandidateDynamics(
        lhs="xddot",
        terms=(Term("u", "a0"), Term("bollinger", "a1")),
        metadata={"family": "mechanical_2nd_order", "order": 2,
                  "state_variables": ["x", "v"], "input_variables": ["u"]},
        candidate_id="bollinger-probe",
    )
    survivors = prune_admissible(load_coupled(), [candidate])
    kept_terms = {
        term.name
        for scored in survivors
        for term in scored.candidate.terms
    }
    assert not kept_terms & set(GRAMMAR_TERMS), kept_terms


# ---------------------------------------------------------------- batch steppers


def test_triton_flag_cannot_claim_a_kernel_that_does_not_exist(monkeypatch):
    from apedp_dyn.cuda import batch_physics_steppers as steppers

    assert steppers.triton_kernels() == ()
    # Simulate the worst case: Triton imports and CUDA reports available.
    monkeypatch.setattr(steppers, "HAS_TRITON", True)
    monkeypatch.setattr(steppers.torch.cuda, "is_available", lambda: True)
    assert steppers.is_triton_available() is False

    # And it does answer True once a kernel is registered.
    monkeypatch.setattr(steppers, "triton_kernels", lambda: ("probe",))
    assert steppers.is_triton_available() is True


# ---------------------------------------------------------------- grammar bindings
#
# Three places still let a symbol into an emitted expression that no caller
# binds. Each one now refuses, or binds, at build time instead of handing the
# failure to a gate that would report it as bad caller data.


def test_single_dof_materialization_refuses_a_remote_term():
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics, Term
    from apedp_dyn.grammar.mechanical_grammar import (
        UnavailableGrammarTerm,
        materialize,
    )
    from apedp_dyn.problems import duffing

    candidate = CandidateDynamics(
        lhs="xddot", terms=(Term("xr", "a0"),), candidate_id="remote-in-1dof")
    with pytest.raises(UnavailableGrammarTerm, match="remote-position"):
        materialize(duffing.load(), candidate)


def test_library_column_is_never_a_scaled_term_with_an_unbound_scale():
    """`gL` used to reach lambdify unbound: a NameError, not a refusal."""
    from apedp_dyn.grammar.proposer import _library_matrix
    from apedp_dyn.problems import duffing

    cols, _target, skipped = _library_matrix(
        duffing.load(), ("x", "langmuir_bind", "hill2_bind"))
    assert set(cols) == {"x"}
    reasons = " ".join(skipped)
    assert "langmuir_bind" in reasons and "unbound" in reasons
    # The surviving column is the state itself, i.e. evaluable, not a stub.
    truth = duffing.load().train["truth_y"]
    np.testing.assert_allclose(cols["x"], truth[:, 0])


def test_two_dof_materialization_binds_a_scaled_term_to_a_fitted_scale():
    """`gL{i}` was passed to the builder and never substituted."""
    from apedp_dyn.grammar.candidate_ir import (
        CandidateDynamics,
        SystemCandidate,
        Term,
    )
    from apedp_dyn.grammar.coupled import materialize_system
    from apedp_dyn.problems import coupled

    term = Term("langmuir_bind", "a0", scale_parameter="s0")
    eq = CandidateDynamics(lhs="vddot", terms=(term,), candidate_id="dof")
    system = SystemCandidate(equations=(eq, eq), candidate_id="scaled-2dof")

    model = materialize_system(coupled.load(), system)
    expressions = [e.expression for e in model.equations]
    assert not any("gL" in expr for expr in expressions), expressions
    # Skip the kinematic equations (state = velocity), which carry no term.
    force = [expr for expr in expressions if "a0" in expr]
    assert len(force) == 2, expressions
    assert all("s0" in expr for expr in force), force
    scales = [p for p in model.parameters if getattr(p, "role", None) == "scale"]
    assert [p.name for p in scales] == ["s0"]
    assert scales[0].units == "meter"


# ---------------------------------------------------------------- video region masks


def _latent(height: int, width: int, *, textured_rows: slice | None = None) -> np.ndarray:
    arr = np.zeros((3, height, width), dtype=np.float64)
    if textured_rows is not None:
        cols = np.arange(width, dtype=np.float64)
        arr[:, textured_rows, :] = np.sin(cols * 1.7)[None, None, :]
    return arr


def test_region_masks_are_derived_from_the_picture_not_from_fixed_rows():
    from apedp_dyn.video.region_masks import derive_region_masks

    # Texture only in the top fifth: the split must follow the picture.
    top = derive_region_masks(_latent(64, 64, textured_rows=slice(0, 12)))
    bottom = derive_region_masks(_latent(64, 64, textured_rows=slice(52, 64)))

    assert top["boundaries"] != bottom["boundaries"]
    assert top["boundaries"]["upper_to_middle"] != 0.40
    assert top["method"] == "row_texture_energy_argmax_steps"
    # The band assignment follows texture energy, so the busy band differs.
    assert top["energy"]["upper_band_mean"] > top["energy"]["lower_band_mean"]
    assert bottom["energy"]["lower_band_mean"] > bottom["energy"]["upper_band_mean"]


def test_region_masks_are_a_partition_and_are_deterministic():
    from apedp_dyn.video.region_masks import derive_region_masks

    latent = _latent(48, 40, textured_rows=slice(30, 48))
    first = derive_region_masks(latent)
    second = derive_region_masks(latent)
    total = sum(first["masks"][name] for name in ("upper", "middle", "lower"))
    np.testing.assert_allclose(total, 1.0, atol=1e-9)
    assert first["boundaries"] == second["boundaries"]
    for name, mask in first["masks"].items():
        assert mask.shape == (48, 40), name
        np.testing.assert_array_equal(mask, second["masks"][name])


# ---------------------------------------------------------------- text encoder


def test_text_encoder_declares_itself_as_feature_hashing():
    from jetflow_video import text_encoder

    assert text_encoder.EMBEDDING_METHOD == "feature_hashing+hand_written_concept_ontology"
    assert text_encoder.PRETRAINED_SEMANTICS is False
    source = Path(inspect.getfile(text_encoder)).read_text()
    assert "hashlib.md5" not in source, "MD5 was replaced by a labelled 64-bit digest"


# ---------------------------------------------------------------- QD provenance


def test_derived_sizing_rows_cannot_reach_a_generation_verdict():
    from apedp.qd_generation import (
        DERIVED_FROM_POLYNOMIAL,
        MEASURED_TEM_PROVENANCE,
        adjudicate_verdict,
        yu_cdse_rows,
    )

    rows = yu_cdse_rows()
    assert rows and all(row["independent_measurement"] is False for row in rows)
    assert all(row["provenance"] == DERIVED_FROM_POLYNOMIAL for row in rows)

    withheld = adjudicate_verdict({"verdict": "GENERATION"}, DERIVED_FROM_POLYNOMIAL)
    assert withheld["verdict"] == "INCONCLUSIVE"
    assert withheld["withheld_verdict"] == "GENERATION"
    assert "derived" in withheld["reason"]

    kept = adjudicate_verdict({"verdict": "GENERATION"}, MEASURED_TEM_PROVENANCE)
    assert kept["verdict"] == "GENERATION"

    # A CONFIRMATION is allowed on derived rows: agreeing with the source curve
    # is exactly what the numbers support.
    confirmed = adjudicate_verdict({"verdict": "CONFIRMATION"}, DERIVED_FROM_POLYNOMIAL)
    assert confirmed["verdict"] == "CONFIRMATION"


# ---------------------------------------------------------------- experimental physics


def test_motor_slip_is_an_input_and_the_evidence_is_labelled():
    from apedp.experimental_physics import TeslaInventionSimulator

    slow = TeslaInventionSimulator.polyphase_induction_motor(slip=0.02)
    fast = TeslaInventionSimulator.polyphase_induction_motor(slip=0.05)
    assert slow["rotor_speed_rpm"] != fast["rotor_speed_rpm"]
    assert slow["evidence"] == "closed_form_textbook"
    assert any("slip" in note for note in slow["assumptions"])


def test_coil_output_scales_with_the_stated_primary_voltage():
    from apedp.experimental_physics import TeslaInventionSimulator

    base = TeslaInventionSimulator.tesla_coil_analysis(primary_voltage_V=1000.0)
    doubled = TeslaInventionSimulator.tesla_coil_analysis(primary_voltage_V=2000.0)
    assert doubled["output_voltage_estimate_V"] == pytest.approx(
        2 * base["output_voltage_estimate_V"])
    assert base["evidence"] == "closed_form_textbook"


def test_collector_names_its_irradiance():
    from apedp.experimental_physics import ExperimentalEnergyConcepts

    dark = ExperimentalEnergyConcepts.radiative_collection_theory(irradiance_W_m2=0.0)
    bright = ExperimentalEnergyConcepts.radiative_collection_theory(irradiance_W_m2=1000.0)
    assert dark["available_power_W"] == 0.0
    assert bright["available_power_W"] == pytest.approx(1000.0)
    assert bright["evidence"] == "closed_form_textbook"


# ---------------------------------------------------------------- neural symbolic


def test_neural_symbolic_teacher_forcing_is_implemented_not_a_pass():
    source = (ROOT / "apedp/neural_discovery/symbolic.py").read_text()
    assert "Teacher forcing would go here" not in source
    tree = ast.parse(source)
    decode = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "decode"
    )
    assert "targets" in [arg.arg for arg in decode.args.args]
    # The training step is eager by construction and says so.
    module_assign = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert module_assign["TRAIN_STEP_JITTABLE"].value is False
    decorated = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.decorator_list
        and any(getattr(d, "id", None) == "jit" for d in node.decorator_list)
    ]
    assert "train_step" not in decorated


def test_abandoned_kernel_tree_is_deleted_and_stays_out_of_the_workspace():
    """`crates/apedp_kernel/` was a grave, not a crate: cargo never saw it.

    It held ten modules abandoned when development moved to `rust/apedp-kernel`,
    was never a workspace member, and was deleted on 2026-11-02. Both halves are
    pinned: the manifest still selects only the active kernel, and the tree does
    not come back.
    """
    cargo = (ROOT / "Cargo.toml").read_text()
    # The path appears in the explanatory comment above `members`, so read the
    # member list itself rather than grepping the file.
    members = cargo.split("members = [", 1)[1].split("]", 1)[0]
    assert "crates/apedp_kernel" not in members, members
    assert "rust/apedp-kernel" in members, members
    assert not (ROOT / "crates").exists(), (
        "the abandoned kernel tree is back on disk; it was deleted on purpose "
        "(see docs/SCAFFOLD_INVENTORY.md)"
    )


SHIM_PACKAGES = (
    "apedp-audio", "apedp-continuum", "apedp-distributed",
    "apedp-molecular", "apedp-quantum", "apedp-robotics",
)


def test_packaging_shims_are_deleted_not_reimplemented():
    """The six `packages/apedp-*` re-export shells are gone.

    Each held no implementation: its `__init__.py` inserted the repo root on
    `sys.path` and re-exported `apedp_dyn`, and `apedp-continuum` added two
    forwarding module files. The behaviour lives in `apedp_dyn` and is tested
    there, so the shells were a second name for one implementation. `packages/`
    now holds `jetflow-video` only, which is standalone by contract.
    """
    remaining = sorted(p.name for p in (ROOT / "packages").iterdir() if p.is_dir())
    assert remaining == ["jetflow-video"], remaining
    for name in SHIM_PACKAGES:
        assert not (ROOT / "packages" / name).exists(), (
            f"{name} is back; the behaviour belongs in apedp_dyn, not in a wheel"
        )


def test_symbolic_module_imports_only_with_jax():
    pytest.importorskip("jax")
    from apedp.neural_discovery import symbolic

    assert symbolic.TRAIN_STEP_JITTABLE is False
    assert "targets" in inspect.signature(
        symbolic.NeuralSymbolicRegressor.decode).parameters
