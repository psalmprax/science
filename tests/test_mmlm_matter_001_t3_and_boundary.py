"""T3 topology adjudication, the runtime boundary guard and T2 loop closure.

The four remaining limits are pinned here:

* a T3 answer is judged on its topology as well as its dimensions, and an
  unverifiable structure is refused rather than passed on its metadata;
* the proposer/referee boundary is enforced at run time, not only by a lint rule;
* a T2 pass gets the same independent loop-closure second opinion as T1;
* a trace follows one physical circuit by default.
"""

from __future__ import annotations

import pytest

from mmlm_matter_001.kinematics import (
    BoundaryError,
    FourBar,
    check_function,
    check_path,
    check_topology,
    planar_revolute_dof,
)
from mmlm_matter_001.pipeline import run_gate
from mmlm_matter_001.protocol import Phase0Preregistration
from mmlm_matter_001.schemas import PublicTaskView, TopologyEntry

CRANK_ROCKER = FourBar(
    ground_mm=40.0, input_mm=12.0, coupler_mm=45.0, output_mm=30.0,
    coupler_point_mm=(20.0, 0.0))


def _preregistration(**overrides) -> Phase0Preregistration:
    values = {
        "grashof_margin_mm": 0.5,
        "function_tolerance_deg": 0.25,
        "rationale": "Phase 0 thresholds and measurement settings.",
        "path_samples": 37,
    }
    values.update(overrides)
    return Phase0Preregistration(**values)


def _path_task(*, tolerance: float = 2.0) -> PublicTaskView:
    trace = CRANK_ROCKER.trace((0.0, 360.0), samples=37)
    return PublicTaskView(
        task_id="T3-path", family="T3",
        target_kind="ordered planar coupler path", tolerance=tolerance,
        input_range_deg=(0.0, 360.0), target_points=trace.points_mm)


def _function_task(*, tolerance: float = 0.25) -> PublicTaskView:
    samples = tuple(
        (float(angle), CRANK_ROCKER.solve(float(angle), branch="open").theta4_deg)
        for angle in range(100, 141, 5)
    )
    return PublicTaskView(
        task_id="T2-function", family="T2",
        target_kind="input-output angle function", tolerance=tolerance,
        input_range_deg=(100.0, 140.0), function_samples=samples)


def _topology(**overrides) -> TopologyEntry:
    values = {
        "topology_id": "T4-000",
        "link_count": 4,
        "joint_count": 4,
        "inversion": "crank-rocker",
        "ground_link": "ground",
        "input_link": "input",
        "output_link": "output",
        "parent_chain_id": "4R-0001",
        "graph_hash": "a" * 64,
    }
    values.update(overrides)
    return TopologyEntry(**values)


# --------------------------------------------------------------- T3 topology


def test_the_degree_of_freedom_formula_matches_a_four_bar():
    assert planar_revolute_dof(link_count=4, joint_count=4) == 1
    assert planar_revolute_dof(link_count=6, joint_count=7) == 1
    assert planar_revolute_dof(link_count=5, joint_count=6) == 0


def test_a_matching_topology_and_dimensions_pass():
    verdict = run_gate(
        CRANK_ROCKER, _path_task(), preregistration=_preregistration(),
        topology=_topology())

    assert verdict.passed is True, verdict.reasons
    assert verdict.topology is not None
    assert verdict.topology["passed"] is True
    assert verdict.topology["dof"] == 1
    assert verdict.topology["achieved_inversion"] == "crank-rocker"
    assert verdict.path is not None


def test_a_declared_inversion_the_dimensions_do_not_produce_is_refused():
    verdict = run_gate(
        CRANK_ROCKER, _path_task(), preregistration=_preregistration(),
        topology=_topology(inversion="double-rocker"))

    assert verdict.passed is False
    assert "topology_inversion_mismatch" in verdict.reasons
    assert verdict.topology["achieved_inversion"] == "crank-rocker"


def test_an_unverifiable_structure_is_refused_not_passed():
    verdict = run_gate(
        CRANK_ROCKER, _path_task(), preregistration=_preregistration(),
        topology=_topology(
            topology_id="T6-000", link_count=6, joint_count=7,
            inversion="six-bar"))

    assert verdict.passed is False
    assert "topology_unsupported_structure" in verdict.reasons
    assert verdict.topology["passed"] is False


def test_a_t3_answer_without_a_topology_is_refused():
    verdict = run_gate(
        CRANK_ROCKER, _path_task(), preregistration=_preregistration())

    assert verdict.passed is False
    assert "topology_missing" in verdict.reasons


def test_check_topology_reports_the_inversion_it_actually_achieved():
    check = check_topology(_topology(), CRANK_ROCKER, margin_mm=0.5)

    assert check.passed is True
    assert check.declared_inversion == check.achieved_inversion
    assert check.record()["dof"] == 1


# ---------------------------------------------------------- runtime boundary


def test_a_proposer_calling_a_checker_is_refused_at_run_time():
    namespace = {
        "__name__": "mmlm_matter_001.proposers.p1_llm_topology",
        "check_path": check_path,
        "check_function": check_function,
        "run_gate": run_gate,
        "mechanism": CRANK_ROCKER,
        "task": _path_task(),
        "preregistration": _preregistration(),
    }

    with pytest.raises(BoundaryError):
        exec("check_path(mechanism, task, samples=5)", namespace)  # noqa: S102
    with pytest.raises(BoundaryError):
        exec(  # noqa: S102
            "run_gate(mechanism, task, preregistration=preregistration)", namespace)


def test_a_model_adapter_calling_the_function_metric_is_refused():
    namespace = {
        "__name__": "mmlm_matter_001.models.outputs",
        "check_function": check_function,
        "mechanism": CRANK_ROCKER,
        "task": _function_task(),
    }

    with pytest.raises(BoundaryError):
        exec("check_function(mechanism, task)", namespace)  # noqa: S102


def test_referee_and_driver_callers_are_allowed():
    check = check_path(CRANK_ROCKER, _path_task(), samples=13)

    assert check.frechet_mm is not None


# ------------------------------------------------------- T2 loop closure


def test_a_t2_pass_gets_a_loop_closure_second_opinion():
    task = _function_task()

    check = check_function(CRANK_ROCKER, task, verify_loop_closure=True)

    assert check.passed is True, check.reasons
    assert check.loop_closure_verified is True
    assert check.loop_closure_checked == len(task.function_samples)
    assert check.max_loop_residual_mm is not None
    assert check.max_loop_residual_mm <= 1e-8
    assert check.record()["loop_closure_verified"] is True


def test_the_gate_records_the_t2_second_opinion():
    verdict = run_gate(
        CRANK_ROCKER, _function_task(), preregistration=_preregistration())

    assert verdict.passed is True, verdict.reasons
    assert verdict.loop_closure is not None
    assert verdict.loop_closure["verified"] is True
    assert verdict.loop_closure["checked"] == 9


# ------------------------------------------------------- continuity default


def test_tracing_follows_one_circuit_by_default():
    trace = CRANK_ROCKER.trace((0.0, 360.0), samples=13)

    assert trace.continuity_tracked is True
    assert trace.complete is True
    assert len(trace.poses) == 13


def test_the_label_sweep_is_still_available_and_labelled_as_such():
    forensic = CRANK_ROCKER.trace((0.0, 360.0), samples=13, circuit=False)

    assert forensic.continuity_tracked is False
    assert forensic.complete is True
