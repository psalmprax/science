"""MMLM-POC-001 with a real language model choosing the moves: tested with a scripted stand-in for the model."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.quantum import llm_decomposition_arm as arm
from apedp_dyn.quantum.field_tensor_search import (
    build_54_reference,
    build_multiplication_tensor,
    diagnose_gf2_bilinear_system,
)

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> object:
    for folder in (str(ROOT), str(ROOT / "scripts")):
        if folder not in sys.path:
            sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _script("run_mmlm_poc_001_llm")
T = build_multiplication_tensor()


def _scripted(replies: list[str], usage: dict | None = None) -> object:
    def client(prompt: str) -> str:
        client.prompts.append(prompt)                                                                 # type: ignore[attr-defined]
        if usage is not None:
            client.last = {"usage": dict(usage)}                                                      # type: ignore[attr-defined]
        reply = replies[min(len(client.prompts), len(replies)) - 1]                                   # type: ignore[attr-defined]
        if reply == "RAISE":
            raise RuntimeError("HTTP 429")
        return reply

    client.prompts = []                                                                               # type: ignore[attr-defined]
    client.model = "scripted"                                                                         # type: ignore[attr-defined]
    return client


def test_only_a_valid_move_is_read_from_a_reply() -> None:
    assert arm.parse_move('I would try this.\n{"move": "set_term", "term": 3, "u": [0, 11, 0], "v": [5]}', 50) == {
        "move": "set_term", "term": 3, "u": [0, 11], "v": [5]}
    assert arm.parse_move('{"move": "xor_terms", "into": 1, "from": 2, "bits": [0, 1, 2]}', 50)["from"] == 2
    assert arm.parse_move('first {"move": "frobenius", "term": 1} then {"move": "frobenius", "term": 7}', 50) == {
        "move": "frobenius", "term": 7}                                                               # the last one in the text
    for bad in ('{"move": "set_term", "term": 50, "u": [0], "v": [0]}',                               # no such product
                '{"move": "set_term", "term": 1, "u": [12], "v": [0]}',                               # no such bit
                '{"move": "set_term", "term": 1, "u": [], "v": [0]}',                                 # an empty factor
                '{"move": "xor_terms", "into": 2, "from": 2, "bits": [0]}',                           # into itself
                '{"move": "delete_everything"}', '{"move": "frobenius", "term": true}', "not json at all", ""):
        assert arm.parse_move(bad, 50) is None, bad


def test_a_move_changes_the_products_as_it_says_and_leaves_the_inputs_alone() -> None:
    U = np.zeros((3, 12), dtype=np.uint8)
    V = np.zeros((3, 12), dtype=np.uint8)
    U[1, [0, 1]] = 1
    V[1, [2]] = 1
    set_U, set_V = arm.apply_move(U, V, {"move": "set_term", "term": 0, "u": [3, 4], "v": [5]})
    assert arm.bits_of(set_U[0]) == [3, 4] and arm.bits_of(set_V[0]) == [5] and not U[0].any()
    xor_U, xor_V = arm.apply_move(set_U, set_V, {"move": "xor_terms", "into": 0, "from": 1, "bits": [0, 3]})
    assert arm.bits_of(xor_U[0]) == [0, 3, 4] and arm.bits_of(xor_V[0]) == [5]                        # only the named bits move
    x = np.zeros(12, dtype=np.uint8)
    x[1] = 1                                                                                          # the element x; its square is x^2
    assert arm.bits_of(arm.frobenius(x)) == [2]
    top = np.zeros(12, dtype=np.uint8)
    top[6] = 1                                                                                        # x^6 squared is x^12 = x^6 + x^4 + x + 1
    assert arm.bits_of(arm.frobenius(top)) == [0, 1, 4, 6]


def test_the_model_proposes_and_the_exact_checker_decides() -> None:
    reference = build_54_reference()
    U, V = reference.U[:50].copy(), reference.V[:50].copy()
    start, _w = diagnose_gf2_bilinear_system(U, V, T)
    lost = {"u": arm.bits_of(reference.U[50]), "v": arm.bits_of(reference.V[50])}                     # a product of the 54 the start lacks
    replies = ["I think product 0 could go.",                                                         # not a move
               json.dumps({"move": "set_term", "term": 0, "u": [0], "v": [0]}),                       # changes nothing: rejected
               json.dumps({"move": "set_term", "term": 0, **lost}),                                   # judged by the checker alone
               "RAISE"]
    client = _scripted(replies, usage={"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920})
    got = arm.run_llm_arm(T, client, U, V, max_steps=10)
    assert got["start_defect"] == start.defect_count == 64 and got["arm"] == "language model (scripted)"
    assert [entry["outcome"] for entry in got["log"][:2]] == ["invalid reply", "rejected"]
    third = got["log"][2]
    moved_U, moved_V = arm.apply_move(U, V, third["move"])
    truth, _w = diagnose_gf2_bilinear_system(moved_U, moved_V, T)
    assert third["after"] == truth.defect_count and third["outcome"] == ("kept" if truth.defect_count < 64 else "rejected")
    assert got["best_defect"] == min(64, truth.defect_count) and got["invalid_replies"] == 1
    assert got["calls"] == 3 and got["stopped"] == "RuntimeError: HTTP 429"                           # the fourth call failed: the arm ends
    assert got["tokens"] == {"prompt_tokens": 2700, "completion_tokens": 60, "total_tokens": 2760}    # the service's counts, summed
    assert "wrong entries 64 to 64, invalid reply" in client.prompts[1]                               # it is told what its move did
    assert "Current products" in client.prompts[0] and "49: " in client.prompts[0]
    assert np.array_equal(U, reference.U[:50])                                                        # the start is not changed

    silent = arm.run_llm_arm(T, _scripted(["{}"]), U, V, max_steps=2)                                 # a client with no token counts
    assert silent["tokens"] is None and silent["calls_without_a_token_count"] == 2 and silent["invalid_replies"] == 2


def test_every_arm_starts_from_the_same_products_and_has_the_same_number_of_moves() -> None:
    start = runner.start_products()
    reference = build_54_reference()
    assert np.array_equal(start[0], reference.U[:50]) and np.array_equal(start[1], reference.V[:50])
    client = _scripted([json.dumps({"move": "frobenius", "term": 4})])
    record = runner.compare(client, seeds=2, steps=5)
    assert record["start_defect"] == 64 and len(record["runs"]) == 2
    for run in record["runs"]:
        assert run["language_model"]["calls"] == 5 and run["heuristic"]["steps"] == 5 and run["random_walk"]["steps"] == 5
        assert run["heuristic"]["tokens"] is None and run["random_walk"]["tokens"] is None            # no tokens are made up
    summary = record["summary"]
    assert summary["runs"] == 2 and summary["language_model_calls"] == 10 and summary["language_model_tokens"] is None
    assert set(summary["language_model_against_heuristic"]) == {"better", "worse", "same", "sign_test_p_one_sided"}
    assert sum(summary["language_model_against_heuristic"][k] for k in ("better", "worse", "same")) == 2


def test_gate_zero_stops_the_run_unless_it_is_called_development(tmp_path: Path) -> None:
    gate = runner.gate_zero(T)
    assert gate["cleared"] is False and "no copy of the published 51-product decomposition" in gate["why"]
    with pytest.raises(SystemExit, match=r"Gate 0 is not cleared: .* Nothing was run"):
        runner.main(["--out", str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_the_first_runner_no_longer_calls_its_heuristic_a_language_model() -> None:
    text = (ROOT / "scripts" / "run_mmlm_poc_001.py").read_text(encoding="utf-8")
    assert '"Arm B: Frontier LLM Agent"' not in text and "No language model runs in it" in text         # the label, not the correction
    assert "diagnostic-guided heuristic (no language model)" in text and "load_published_51_reference`` returns" in text
