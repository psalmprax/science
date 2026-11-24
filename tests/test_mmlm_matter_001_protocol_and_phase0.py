"""Frozen protocol, offline dry run and the Phase 0 four-bar checker.

Three separate claims are pinned here:

* a protocol bundle cannot be edited, silently replaced or run against a
  drifted prompt/schema (§12.6);
* the P1 pipeline runs end to end with no provider, writing a real
  ``budget_ledger.jsonl`` whose cost arithmetic is exercised rather than faked;
* the four-bar checker implements §5.2–§5.6 — strict Grashof with a declared
  margin, both closed-form assembly branches, circuit continuity, toggle
  classification and the ordered discrete Fréchet path metric.
"""

from __future__ import annotations

import json
import math

import pytest

from mmlm_matter_001.kinematics import (
    FourBar,
    LoopClosureError,
    MechanismError,
    UnreachableError,
    angle_difference_deg,
    check_function,
    check_path,
    circuit_switches,
    cross_check,
    discrete_frechet,
    poses_share_circuit,
    random_grashof_mechanisms,
    score_function_samples,
    solve_numerically,
)
from mmlm_matter_001.models import (
    ModelContract,
    P1SelectionOutput,
    P2DesignOutput,
    PromptTemplate,
    p1_prompt,
    p2_prompt,
)
from mmlm_matter_001.pipeline import (
    STUB_MODEL,
    StubRankingAdapter,
    build_stub_contract,
    run_dry_run,
    stub_catalogue,
    stub_task,
)
from mmlm_matter_001.pipeline.dry_run import main
from mmlm_matter_001.protocol import (
    ArmPin,
    FrozenProtocol,
    ProtocolDriftError,
    ProtocolError,
    ProtocolIntegrityError,
    freeze_protocol,
    load_protocol,
)
from mmlm_matter_001.schemas import PublicTaskView

CRANK_ROCKER = FourBar(ground_mm=40.0, input_mm=12.0, coupler_mm=45.0, output_mm=30.0,
                       coupler_point_mm=(20.0, 0.0))
NON_GRASHOF = FourBar(ground_mm=20.0, input_mm=10.0, coupler_mm=11.0, output_mm=30.0)


def _p1_contract(**overrides) -> ModelContract:
    values = {
        "provider": "stub",
        "model": STUB_MODEL,
        "prompt": p1_prompt(),
        "schema": P1SelectionOutput,
        "max_output_tokens": 512,
        "price_input_usd_per_mtok": 0.0,
        "price_output_usd_per_mtok": 0.0,
    }
    values.update(overrides)
    return ModelContract.freeze(**values)


def _p2_contract(**overrides) -> ModelContract:
    values = {
        "provider": "stub",
        "model": STUB_MODEL,
        "prompt": p2_prompt(),
        "schema": P2DesignOutput,
        "max_output_tokens": 512,
    }
    values.update(overrides)
    return ModelContract.freeze(**values)


def _function_task(
    samples,
    *,
    tolerance: float = 1e-6,
    input_range_deg: tuple[float, float] = (20.0, 200.0),
    task_id: str = "T2-fixture",
) -> PublicTaskView:
    return PublicTaskView(
        task_id=task_id,
        family="T2",
        target_kind="input/output angle relationship",
        tolerance=tolerance,
        input_range_deg=input_range_deg,
        function_samples=tuple(samples),
    )


def _declared_function(samples: int = 19) -> list[tuple[float, float]]:
    """The mechanism's own input/output relationship over 20..200 degrees."""
    low, high = 20.0, 200.0
    inputs = [low + (high - low) * index / (samples - 1) for index in range(samples)]
    return [(angle, CRANK_ROCKER.solve(angle).theta4_deg) for angle in inputs]


def _path_task(points, *, tolerance: float = 0.5, task_id: str = "T1-fixture") -> PublicTaskView:
    return PublicTaskView(
        task_id=task_id,
        family="T1",
        target_kind="ordered planar coupler path",
        tolerance=tolerance,
        input_range_deg=(0.0, 360.0),
        target_points=tuple(points),
    )


# --------------------------------------------------------------------- protocol


def test_the_protocol_bundle_pins_both_arms_and_reloads_identically(tmp_path):
    path = tmp_path / "protocol.json"
    p1, p2 = _p1_contract(), _p2_contract()
    frozen = freeze_protocol(path, {"P1": p1, "P2": p2})

    assert set(frozen.arms) == {"P1", "P2"}
    assert len(frozen.protocol_hash) == 64
    assert frozen.arm("P1").prompt_hash == p1_prompt().prompt_hash
    assert frozen.arm("P2").prompt_id == p2_prompt().prompt_id
    assert frozen.arm("P1").contract_hash == p1.digest
    assert frozen.arm("P2").contract_hash == p2.digest

    reloaded = load_protocol(path)
    assert reloaded.protocol_hash == frozen.protocol_hash
    assert reloaded.record()["arms"]["P1"]["schema_hash"] == frozen.arm("P1").schema_hash
    assert reloaded.contract("P1") == p1


def test_the_bundle_is_refused_when_a_hash_is_edited_by_hand(tmp_path):
    path = tmp_path / "protocol.json"
    freeze_protocol(path, {"P1": _p1_contract(), "P2": _p2_contract()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arms"]["P1"]["prompt_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProtocolIntegrityError, match="edited"):
        FrozenProtocol.load(path)


def test_a_bundle_whose_prompt_hash_does_not_describe_the_checkout_refuses(tmp_path):
    drifted = PromptTemplate(
        prompt_id="p1_topology_ranking_v1",
        arm="P1",
        system=p1_prompt().system,
        template=p1_prompt().template + "\nrank the easiest one first",
    )
    contract = _p1_contract(prompt=drifted)
    path = tmp_path / "protocol.json"

    with pytest.raises(ProtocolDriftError, match="drifted"):
        freeze_protocol(path, {"P1": contract, "P2": _p2_contract()})
    assert not path.exists(), "a bundle that does not describe the checkout must not survive"


def test_verification_names_every_drifted_arm(tmp_path):
    path = tmp_path / "protocol.json"
    freeze_protocol(path, {"P1": _p1_contract(), "P2": _p2_contract()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    for arm in ("P1", "P2"):
        payload["arms"][arm]["schema_hash"] = "b" * 64
    payload.pop("protocol_hash")
    from mmlm_matter_001.models import canonical_json, sha256_text

    payload["protocol_hash"] = sha256_text(canonical_json(payload))
    path.write_text(json.dumps(payload), encoding="utf-8")

    frozen = FrozenProtocol.load(path)
    with pytest.raises(ProtocolDriftError) as caught:
        frozen.verify_checkout()
    message = str(caught.value)
    assert "P1" in message and "P2" in message and "schema" in message


def test_a_retargeted_contract_is_refused_when_the_model_is_pinned(tmp_path):
    contract = _p1_contract()
    frozen = freeze_protocol(
        tmp_path / "protocol.json", {"P1": contract, "P2": _p2_contract()})
    frozen.check_contract("P1", contract)

    retargeted = contract.retarget(model="some/other-model")
    with pytest.raises(ProtocolDriftError, match="frozen protocol"):
        frozen.check_contract("P1", retargeted)


def test_an_unpinned_model_bundle_allows_the_model_to_be_chosen_later(tmp_path):
    frozen = freeze_protocol(
        tmp_path / "protocol.json",
        {"P1": _p1_contract(), "P2": _p2_contract()},
        pin_models=False,
    )
    assert frozen.pin_models is False
    assert frozen.arm("P1").contract_hash is None
    assert not frozen.arm("P1").pins_model

    chosen = _p1_contract(provider="openrouter", model="vendor/model-v2")
    frozen.check_contract("P1", chosen)

    wrong_prompt = _p1_contract(prompt=p1_prompt().__class__(
        prompt_id=p1_prompt().prompt_id,
        arm="P1",
        system=p1_prompt().system,
        template=p1_prompt().template + " and be brief",
    ))
    with pytest.raises(ProtocolDriftError, match="prompt hash"):
        frozen.check_contract("P1", wrong_prompt)


def test_both_arms_are_required_and_an_existing_bundle_is_not_silently_replaced(tmp_path):
    path = tmp_path / "protocol.json"
    with pytest.raises(ProtocolError, match="exactly the arms"):
        freeze_protocol(path, {"P1": _p1_contract()})

    freeze_protocol(path, {"P1": _p1_contract(), "P2": _p2_contract()})
    with pytest.raises(ProtocolError, match="already exists"):
        freeze_protocol(path, {"P1": _p1_contract(), "P2": _p2_contract()})
    replaced = freeze_protocol(
        path, {"P1": _p1_contract(), "P2": _p2_contract()}, overwrite=True,
        notes="replacement recorded deliberately")
    assert replaced.notes == "replacement recorded deliberately"


def test_the_loader_refuses_missing_files_and_unknown_arms(tmp_path):
    with pytest.raises(ProtocolError, match="does not exist"):
        FrozenProtocol.load(tmp_path / "absent.json")

    frozen = freeze_protocol(tmp_path / "protocol.json", {"P1": _p1_contract(),
                                                          "P2": _p2_contract()})
    with pytest.raises(ProtocolError, match="no arm"):
        frozen.arm("P3")


def test_an_arm_pin_requires_real_digests():
    with pytest.raises(ProtocolIntegrityError, match="prompt_hash"):
        ArmPin(
            arm="P1",
            prompt_id="p1_topology_ranking_v1",
            prompt_hash="not-a-digest",
            schema_id="p1_topology_selection_v1",
            schema_hash="a" * 64,
        )


# ---------------------------------------------------------------------- dry run


def test_the_dry_run_writes_one_ledger_row_per_task(tmp_path):
    catalogue = stub_catalogue(5)
    tasks = (stub_task("T1-stub-a"), stub_task("T1-stub-b"))
    result = run_dry_run(
        tasks=tasks, catalogue=catalogue, ledger_path=tmp_path / "budget_ledger.jsonl")

    rows = [json.loads(line) for line in
            (tmp_path / "budget_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert [row["sequence"] for row in rows] == [1, 2]
    assert {row["arm"] for row in rows} == {"P1"}
    assert {row["provider"] for row in rows} == {"stub"}
    assert {row["model"] for row in rows} == {STUB_MODEL}
    assert all(row["cost_basis"] == "priced" for row in rows)
    assert all(row["cost_usd"] == 0.0 for row in rows)

    assert result.calls == 2 and result.tasks == 2
    assert result.aggregate["cost_complete"] is True
    assert result.aggregate["cost_usd"] == 0.0
    assert result.aggregate["calls"] == 2
    assert result.aggregate["by_arm"]["P1"]["input_tokens_complete"] is True
    assert result.protocol_hash is None
    for proposal in result.proposals:
        assert 1 <= len(proposal.topology_ids) <= 3
        assert set(proposal.topology_ids) <= set(result.catalogue_ids)


def test_the_stub_is_deterministic_across_runs(tmp_path):
    first = run_dry_run(ledger_path=tmp_path / "a.jsonl")
    second = run_dry_run(ledger_path=tmp_path / "b.jsonl")
    assert [p.topology_ids for p in first.proposals] == [p.topology_ids for p in second.proposals]
    assert [p.request_hash for p in first.proposals] == [p.request_hash for p in second.proposals]


def test_the_dry_run_never_touches_the_network(tmp_path, monkeypatch):
    import httpx

    def explode(*_args, **_kwargs):
        raise AssertionError("the dry run must not reach a provider")

    monkeypatch.setattr(httpx, "post", explode)
    result = run_dry_run(tasks=(stub_task("T1-offline"),), ledger_path=tmp_path / "l.jsonl")
    assert result.calls == 1


def test_a_drifted_protocol_refuses_before_the_first_call(tmp_path):
    frozen = freeze_protocol(
        tmp_path / "protocol.json", {"P1": _p1_contract(), "P2": _p2_contract()},
        pin_models=False)
    drifted = build_stub_contract().retarget(temperature=0.5)
    ledger_path = tmp_path / "budget_ledger.jsonl"

    with pytest.raises(ProtocolDriftError, match="temperature"):
        run_dry_run(tasks=(stub_task("T1-refused"),), ledger_path=ledger_path,
                    protocol=frozen, contract=drifted)
    assert not ledger_path.exists(), "a refused protocol must not write a partial ledger"


def test_the_dry_run_refuses_empty_inputs(tmp_path):
    with pytest.raises(ValueError, match="catalogue"):
        run_dry_run(catalogue=(), ledger_path=tmp_path / "l.jsonl")
    with pytest.raises(ValueError, match="task"):
        run_dry_run(tasks=(), ledger_path=tmp_path / "l.jsonl")


def test_the_stub_refuses_a_prompt_without_a_catalogue():
    from mmlm_matter_001.models import ModelRequest

    request = ModelRequest(
        contract=_p1_contract(), system_prompt="s", user_prompt="no catalogue here")
    with pytest.raises(RuntimeError, match="catalogue"):
        StubRankingAdapter().complete(request)


def test_the_command_line_entry_writes_a_ledger(tmp_path, capsys):
    ledger = tmp_path / "cli_ledger.jsonl"
    assert main(["--ledger", str(ledger), "--tasks", "2", "--catalogue", "4"]) == 0
    rows = ledger.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["calls"] == 2
    assert printed["aggregate"]["cost_usd"] == 0.0


# -------------------------------------------------------------- four-bar: Grashof


def test_grashof_classifies_the_inversion_by_which_link_is_shortest():
    crank_rocker = CRANK_ROCKER.check_grashof()
    assert crank_rocker.is_grashof is True
    assert crank_rocker.inversion == "crank-rocker"
    assert crank_rocker.shortest_link == "input"
    assert crank_rocker.full_rotation_input is True
    assert crank_rocker.slack_mm == pytest.approx(13.0)

    double_crank = FourBar(
        ground_mm=10.0, input_mm=30.0, coupler_mm=20.0, output_mm=28.0).check_grashof()
    assert double_crank.inversion == "double-crank"
    assert double_crank.shortest_link == "ground"
    assert double_crank.full_rotation_input is True

    double_rocker = FourBar(
        ground_mm=30.0, input_mm=25.0, coupler_mm=10.0, output_mm=26.0).check_grashof()
    assert double_rocker.inversion == "double-rocker"
    assert double_rocker.shortest_link == "coupler"
    assert double_rocker.full_rotation_input is False


def test_non_grashof_and_equality_are_rejected():
    non_grashof = NON_GRASHOF.check_grashof()
    assert non_grashof.is_grashof is False
    assert non_grashof.inversion == "non-grashof"
    assert non_grashof.reason == "non_grashof"

    equality = FourBar(
        ground_mm=30.0, input_mm=10.0, coupler_mm=10.0, output_mm=30.0).check_grashof()
    assert equality.slack_mm == pytest.approx(0.0)
    assert equality.inversion == "change-point"
    assert equality.is_grashof is False


def test_the_declared_margin_makes_a_borderline_mechanism_fail():
    assert CRANK_ROCKER.check_grashof(margin_mm=12.5).is_grashof is True
    borderline = CRANK_ROCKER.check_grashof(margin_mm=13.0)
    assert borderline.is_grashof is False
    assert borderline.reason == "grashof_margin_not_met"
    with pytest.raises(MechanismError, match="margin_mm"):
        CRANK_ROCKER.check_grashof(margin_mm=-1.0)


def test_a_degenerate_link_is_refused():
    with pytest.raises(MechanismError, match="input_mm"):
        FourBar(ground_mm=40.0, input_mm=0.0, coupler_mm=45.0, output_mm=30.0)
    with pytest.raises(MechanismError, match="coupler_point_mm"):
        FourBar(ground_mm=40.0, input_mm=12.0, coupler_mm=45.0, output_mm=30.0,
                coupler_point_mm=(float("nan"), 0.0))


# ------------------------------------------------- four-bar: closed form & branch


def test_the_closed_form_closes_the_loop_at_every_input_angle():
    for branch in ("open", "crossed"):
        for theta2_deg in (0.0, 45.0, 137.0, 250.0, 359.0):
            pose = CRANK_ROCKER.solve(theta2_deg, branch=branch)
            a, b, c, d = 12.0, 45.0, 30.0, 40.0
            from math import cos, radians, sin

            t2, t3, t4 = radians(pose.theta2_deg), radians(pose.theta3_deg), radians(pose.theta4_deg)
            residual = (
                a * cos(t2) + b * cos(t3) - c * cos(t4) - d,
                a * sin(t2) + b * sin(t3) - c * sin(t4),
            )
            assert max(abs(value) for value in residual) < 1e-9
            assert math.dist(pose.pivot_b_mm, (40.0, 0.0)) == pytest.approx(30.0)
            assert math.dist(pose.pivot_b_mm, pose.pivot_a_mm) == pytest.approx(45.0)


def test_both_assembly_branches_exist_and_differ():
    open_pose = CRANK_ROCKER.solve(30.0, branch="open")
    crossed_pose = CRANK_ROCKER.solve(30.0, branch="crossed")
    assert open_pose.branch == "open"
    assert crossed_pose.branch == "crossed"
    assert open_pose.coupler_point_mm != pytest.approx(crossed_pose.coupler_point_mm)
    assert open_pose.coupler_point_mm[1] > 0 > crossed_pose.coupler_point_mm[1]


def test_a_requested_branch_that_does_not_exist_is_refused():
    with pytest.raises(MechanismError, match="branch"):
        CRANK_ROCKER.solve(0.0, branch="upside-down")


def test_a_non_grashof_mechanism_has_unreachable_inputs():
    with pytest.raises(UnreachableError):
        NON_GRASHOF.trace((0.0, 360.0), samples=181)


def test_a_crank_rocker_traverses_a_full_rotation_on_one_circuit():
    trace = CRANK_ROCKER.trace((0.0, 360.0), samples=181)
    assert trace.complete is True
    assert len(trace.poses) == 181
    assert poses_share_circuit(trace.poses) is True
    assert circuit_switches(trace.poses) == 0
    assert trace.points_mm[0] == pytest.approx(trace.points_mm[-1])


def test_a_mixed_branch_sequence_is_not_one_circuit():
    open_pose = CRANK_ROCKER.solve(30.0, branch="open")
    crossed_pose = CRANK_ROCKER.solve(30.0, branch="crossed")
    assert poses_share_circuit((open_pose, crossed_pose)) is False
    assert circuit_switches((open_pose, crossed_pose)) == 1


def test_input_and_output_dead_centres_are_classified_separately():
    toggles = CRANK_ROCKER.toggles()
    assert toggles.input_dead_centre_deg == pytest.approx(
        (-47.3396043848, -30.0019058153, 30.0019058153, 47.3396043848), abs=1e-6)
    assert toggles.input_locked is False
    assert any("pass-through" in note for note in toggles.notes)

    locked = NON_GRASHOF.toggles()
    assert locked.input_locked is True


# ------------------------------------------- independent numerical loop closure


def test_the_independent_solver_agrees_with_the_closed_form():
    for theta2_deg in (10.0, 75.0, 200.0, 300.0):
        numerical = solve_numerically(CRANK_ROCKER, theta2_deg)
        assert {solution.branch for solution in numerical} == {"open", "crossed"}
        for solution in numerical:
            closed = CRANK_ROCKER.solve(theta2_deg, branch=solution.branch)
            assert abs(angle_difference_deg(
                closed.theta3_deg, solution.theta3_deg)) < 1e-6
            assert abs(angle_difference_deg(
                closed.theta4_deg, solution.theta4_deg)) < 1e-6
            assert solution.residual_mm < 1e-9
            assert solution.converged is True


def test_the_numerical_solver_returns_no_unconverged_roots():
    numerical = solve_numerically(CRANK_ROCKER, 137.0)
    assert numerical, "the fixture has two assemblies at this angle"
    assert all(solution.converged for solution in numerical)
    assert all(not solution.singular for solution in numerical)
    assert all(solution.residual_mm <= 1e-10 for solution in numerical)


def test_the_cross_check_agrees_across_random_feasible_mechanisms():
    report = cross_check(mechanism_count=6, angles=6, seed=7)

    assert report.passed is True
    assert report.missing_branches == 0
    # Two branches per angle, unless a dead centre merges them into one pose.
    assert 6 * 6 <= report.comparisons <= 6 * 6 * 2
    assert report.max_theta3_error_deg < 1e-6
    assert report.max_theta4_error_deg < 1e-6
    assert report.max_residual_mm < 1e-9
    assert report.details == ()
    assert report.record()["passed"] is True


def test_the_random_mechanisms_are_full_rotation_crank_rockers():
    mechanisms = random_grashof_mechanisms(5, seed=3)
    assert len(mechanisms) == 5
    for mechanism in mechanisms:
        grashof = mechanism.check_grashof(margin_mm=1.0)
        assert grashof.is_grashof is True
        assert grashof.inversion == "crank-rocker"
        assert grashof.shortest_link == "input"


def test_the_cross_check_is_deterministic_for_a_seed():
    first = cross_check(mechanism_count=3, angles=4, seed=11)
    second = cross_check(mechanism_count=3, angles=4, seed=11)
    assert first.record() == second.record()


def test_an_impossible_sampling_request_is_refused_loudly():
    with pytest.raises(LoopClosureError, match="crank-rockers"):
        random_grashof_mechanisms(1, length_range_mm=(1.0, 1.05), attempts_per_mechanism=20)
    with pytest.raises(LoopClosureError, match="FourBar"):
        solve_numerically("not a mechanism", 0.0)


# --------------------------------------------------------------- Fréchet metric


def test_the_frechet_metric_is_zero_only_for_matching_ordered_paths():
    path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert discrete_frechet(path, path) == 0.0
    shifted = [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
    assert discrete_frechet(path, shifted) == pytest.approx(1.0)


def test_the_metric_preserves_order_where_hausdorff_would_not():
    forward = [(0.0, 0.0), (0.0, 1.0)]
    backward = [(0.0, 1.0), (0.0, 0.0)]
    assert discrete_frechet(forward, backward) == pytest.approx(1.0)


def test_the_metric_is_symmetric_and_refuses_empty_sequences():
    first = [(0.0, 0.0), (2.0, 0.0), (2.0, 3.0)]
    second = [(0.5, 0.0), (2.0, 1.0), (2.0, 3.5)]
    assert discrete_frechet(first, second) == pytest.approx(discrete_frechet(second, first))
    with pytest.raises(ValueError, match="non-empty"):
        discrete_frechet([], second)


# ---------------------------------------------------------------- T1 path check


def test_a_mechanism_passes_against_its_own_coupler_path():
    trace = CRANK_ROCKER.trace((0.0, 360.0), samples=25)
    task = _path_task(trace.points_mm, tolerance=1e-6)
    check = check_path(CRANK_ROCKER, task, samples=25)

    assert check.passed is True
    assert check.reasons == ()
    assert check.frechet_mm == pytest.approx(0.0)
    assert check.circuit_continuous is True
    assert check.order_ok is True
    assert check.target_mm == task.target_points
    assert check.record()["frechet_mm"] == pytest.approx(0.0)


def test_a_path_shifted_past_the_task_tolerance_fails():
    trace = CRANK_ROCKER.trace((0.0, 360.0), samples=25)
    shifted = [(x + 5.0, y) for x, y in trace.points_mm]
    check = check_path(CRANK_ROCKER, _path_task(shifted, tolerance=0.5), samples=25)

    assert check.passed is False
    assert "frechet_exceeds_tolerance" in check.reasons
    assert check.frechet_mm == pytest.approx(5.0)
    assert check.tolerance_mm == 0.5


def test_a_task_without_target_points_is_refused_not_passed():
    check = check_path(CRANK_ROCKER, _path_task([], tolerance=0.5))
    assert check.passed is False
    assert check.reasons == ("no_target_points",)
    assert check.frechet_mm is None


def test_an_unreachable_input_range_is_reported_as_such():
    task = _path_task(CRANK_ROCKER.trace((0.0, 360.0), samples=12).points_mm, tolerance=1.0)
    check = check_path(NON_GRASHOF, task, samples=181)
    assert check.passed is False
    assert "unreachable_input" in check.reasons


def test_a_mechanism_passes_its_own_declared_function_relationship():
    task = _function_task(_declared_function())
    check = check_function(CRANK_ROCKER, task)

    assert check.passed is True
    assert check.reasons == ()
    assert check.max_error_deg == pytest.approx(0.0)
    assert check.rms_error_deg == pytest.approx(0.0)
    assert check.reachable_samples == 19
    assert check.sample_count == 19
    assert check.circuit_continuous is True
    assert check.branch_switches == 0
    assert check.range_covered is True
    assert check.record()["max_error_deg"] == pytest.approx(0.0)


def test_a_function_error_past_the_preregistered_tolerance_fails():
    shifted = [(angle, output + 5.0) for angle, output in _declared_function()]
    check = check_function(CRANK_ROCKER, _function_task(shifted, tolerance=0.5))

    assert check.passed is False
    assert check.max_error_deg == pytest.approx(5.0)
    assert check.rms_error_deg == pytest.approx(5.0)
    assert "function_error_exceeds_tolerance" in check.reasons
    assert check.tolerance_deg == 0.5


def test_angular_error_is_wrapped_not_taken_literally():
    samples = [(angle, (output + 359.5) % 360.0) for angle, output in _declared_function()]
    passes = check_function(CRANK_ROCKER, _function_task(samples, tolerance=0.75))
    fails = check_function(CRANK_ROCKER, _function_task(samples, tolerance=0.25))

    assert passes.max_error_deg == pytest.approx(0.5)
    assert passes.passed is True
    assert fails.passed is False
    assert "function_error_exceeds_tolerance" in fails.reasons


def test_unreachable_function_inputs_are_a_rejection_not_a_large_error():
    angles = [0.0 + 10.0 * index for index in range(36)]
    samples = [(angle, 45.0) for angle in angles]
    check = check_function(NON_GRASHOF, _function_task(samples, tolerance=5.0))

    assert check.passed is False
    assert "unreachable_input" in check.reasons
    assert check.reachable_samples < check.sample_count
    assert any(point.achieved_output_deg is None for point in check.points)


def test_a_branch_change_across_the_samples_is_rejected():
    samples = _declared_function(samples=6)
    poses = [
        CRANK_ROCKER.solve(angle, branch="open" if index % 2 == 0 else "crossed")
        for index, (angle, _) in enumerate(samples)
    ]
    check = score_function_samples(_function_task(samples), poses)

    assert check.passed is False
    assert "branch_change" in check.reasons
    assert check.circuit_continuous is False
    assert check.branch_switches >= 1


def test_samples_that_do_not_cover_the_declared_range_are_refused():
    samples = [(angle, output) for angle, output in _declared_function()
               if 30.0 <= angle <= 170.0]
    check = check_function(
        CRANK_ROCKER, _function_task(samples, input_range_deg=(20.0, 200.0)))

    assert check.passed is False
    assert "range_not_covered" in check.reasons
    assert check.range_covered is False
    assert check.max_error_deg == pytest.approx(0.0)


def test_the_requested_branch_actually_selects_the_output_angle():
    task = _function_task(_declared_function())
    open_check = check_function(CRANK_ROCKER, task, branch="open")
    crossed_check = check_function(CRANK_ROCKER, task, branch="crossed")

    assert open_check.max_error_deg == pytest.approx(0.0)
    assert crossed_check.max_error_deg > 1.0
    assert crossed_check.circuit_continuous is True
    assert crossed_check.branch == "crossed"


def test_a_task_without_function_samples_is_refused_not_passed():
    task = _function_task([], tolerance=1.0)
    check = check_function(CRANK_ROCKER, task)

    assert check.passed is False
    assert check.reasons == ("no_function_samples",)
    assert check.max_error_deg is None


def test_scoring_requires_one_pose_per_declared_sample():
    task = _function_task(_declared_function(samples=4))
    with pytest.raises(ValueError, match="declared samples"):
        score_function_samples(task, [CRANK_ROCKER.solve(20.0)])


def test_the_reversed_orientation_is_only_accepted_when_the_task_allows_it():
    trace = CRANK_ROCKER.trace((0.0, 360.0), samples=25)
    backwards = tuple(reversed(trace.points_mm))
    task = _path_task(backwards, tolerance=1e-6)

    refused = check_path(CRANK_ROCKER, task, samples=25)
    assert refused.passed is False
    assert "target_order" in refused.reasons

    allowed = check_path(CRANK_ROCKER, task, samples=25, allow_reversed=True)
    assert allowed.passed is True
    assert allowed.orientation_reversed is True
