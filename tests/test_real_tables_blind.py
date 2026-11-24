"""The blind test's script, exercised on made-up tables and on a development table: it runs once, so it must not fail."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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


blind = _script("run_real_tables_blind")


def test_the_two_published_formulas_are_what_they_say_and_agree_to_a_few_percent() -> None:
    reynolds = np.array([1e4, 1e5, 1e6, 1e7])
    roughness = np.array([1e-3, 1e-3, 1e-4, 1e-5])
    colebrook = blind.colebrook(reynolds, roughness)
    assert 1 / np.sqrt(colebrook) == pytest.approx(-2 * np.log10(roughness / 3.7 + 2.51 / (reynolds * np.sqrt(colebrook))), abs=1e-10)
    assert blind.haaland(reynolds, roughness) == pytest.approx(colebrook, rel=0.02)
    assert blind.haaland(np.array([1e5]), np.array([1e-9]))[0] == pytest.approx(0.018, rel=0.03)      # a smooth pipe at Re 1e5


def test_whole_pipes_are_left_out_by_the_rule_fixed_beforehand() -> None:
    reynolds = np.tile(np.logspace(4, 6, 30), 7)
    seven = np.repeat([1e-4, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.3e-2, 6.6e-2], 30)
    test, split = blind.pipes_left_out(seven, reynolds)
    assert split["left_out"] == [0.002, 0.016] and int(test.sum()) == 60                              # the second and fifth smoothest
    four = np.repeat([1e-4, 2e-3, 4e-3, 8e-3], 30)
    test, split = blind.pipes_left_out(four, reynolds[:120])
    assert split["left_out"] == [0.004] and int(test.sum()) == 30
    spread = np.linspace(1e-4, 1e-2, 90)                                                              # no roughness repeats: no pipes
    test, split = blind.pipes_left_out(spread, reynolds[:90])
    assert split["rule"] == "every third row by Reynolds number" and int(test.sum()) == 30


def _pipes(fanning: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    roughness = np.repeat([1e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.3e-2], 12)
    reynolds = np.tile(np.logspace(4, 6, 12), 7)
    friction = blind.colebrook(reynolds, roughness) * (1 + 0.01 * rng.standard_normal(len(reynolds)))
    return {"Re": reynolds, "eD": roughness, "f": friction / 4 if fanning else friction}


def test_the_pipe_arm_runs_end_to_end_and_scores_every_formula_on_the_pipes_left_out(monkeypatch: pytest.MonkeyPatch,
                                                                                    tmp_path: Path) -> None:
    monkeypatch.setattr(blind, "held_back", lambda key, cache: {"columns": _pipes()})
    monkeypatch.setattr(blind, "READ_SECONDS", 60.0)
    out = blind.pipe_friction(0.0, tmp_path)
    assert out["rows_left_out"] == 24 and out["rows_work"] == 60 and out["factor_applied_to_f"] == 1.0
    errors = out["error_on_pipes_left_out"]
    assert set(errors) == {"the reading's law", "the route in force's law", "Haaland, published constants",
                           "Colebrook and White, published constants", "Haaland's form, constants refitted on the work rows"}
    assert errors["Colebrook and White, published constants"] < 0.01                                  # the data were made with it
    assert errors["Haaland, published constants"] < 0.02
    assert out["reading"]["action"] == "read" and out["reading"]["stability"]["rows_removed"] == 10   # 60 rows: folds


def test_a_table_of_fanning_factors_is_put_on_the_darcy_scale_by_the_declared_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    columns = _pipes(fanning=True)
    test, _split = blind.pipes_left_out(columns["eD"], columns["Re"])
    ratio = float(np.median(columns["f"][~test] / blind.haaland(columns["Re"][~test], columns["eD"][~test])))
    assert 0.125 <= ratio < 0.5                                                                       # a quarter: the rule multiplies by four


def test_the_expansion_arm_runs_end_to_end_in_units_of_each_rows_stated_error(monkeypatch: pytest.MonkeyPatch,
                                                                             tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    z = np.sort(rng.uniform(0.05, 2.0, 30))
    sigma = rng.uniform(5.0, 25.0, 30)
    rate = 70.0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7) + sigma * rng.standard_normal(30)
    monkeypatch.setattr(blind, "held_back", lambda key, cache: {"columns": {"z": z, "H": rate, "sigma_H": sigma}})
    monkeypatch.setattr(blind, "READ_SECONDS", 60.0)
    out = blind.expansion_rate(0.0, tmp_path)
    assert out["rows_left_out"] == 10 and out["rows_work"] == 20
    assert out["reading"]["errors"] == "in units of each row's stated error"
    chi2 = out["chi2_per_row_left_out"]
    assert len(chi2) == 3 and all(value is None or value >= 0 for value in chi2.values())
    assert chi2["the standard model's form (flat, matter and a constant), fitted on the work rows"] < 4.0


def test_the_known_law_arm_runs_end_to_end_on_a_development_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("z3")
    loaded = blind.real_data.load("langmuir_table_ix")
    monkeypatch.setattr(blind, "held_back", lambda key, cache: {"table": loaded["table"], "columns": loaded["columns"]})
    monkeypatch.setattr(blind, "READ_SECONDS", 60.0)
    table = blind.known_law_table("langmuir_table_ix", 0.0, tmp_path)
    assert table["rows"] == 10 and set(table["readings"]) == {"informed_with_searches", "informed", "plain"}
    assert table["readings"]["informed"]["judged"]["same_form_as_the_known_law"] is True
    assert table["readings"]["plain"]["judged"]["same_form_as_the_known_law"] is False
    assert table["product"]["action"] == "not_run"                                                    # ten rows: the product refuses
    summary = blind.summarise_known([table])
    assert summary["informed"]["known_form"] == 1 and summary["informed"]["reported_without_the_known_form"] == 0
    assert summary["plain"]["known_form"] == 0 and summary["product"]["ran"] == 0


def test_what_is_known_is_stated_for_every_table_and_names_only_its_inputs() -> None:
    import json

    stated = json.loads(blind.KNOWN.read_text(encoding="utf-8"))["tables"]
    tables = blind.real_data.registry()
    assert set(stated) == set(tables)
    for key, entry in stated.items():
        for statement in entry["known"]:
            assert statement["kind"] in blind.small_table.known_beforehand.KINDS, key
            assert statement.get("input") is None or statement["input"] in tables[key]["inputs"], key
            assert statement["why"], key
    assert all(key in tables and tables[key]["split"] == "held_back" for key in blind.KNOWN_LAW)


def test_the_generated_arm_draws_laws_the_development_tables_did_not_use() -> None:
    laws = blind.blind_laws()
    development = {blind.generated.curriculum.structure(blind.sp.sympify(law["text"])) for law in blind.generated.draw_laws(120)}
    assert len(laws) == blind.BLIND_TABLES and all(law["tag"] == "small-tables-blind" for law in laws)
    assert not {blind.generated.curriculum.structure(blind.sp.sympify(law["text"])) for law in laws} & development
    a, b = blind.generated.table_of(laws[0]), blind.generated.table_of({**laws[0], "tag": "small-tables-dev"})
    assert not np.array_equal(a["X"], b["X"])                                                         # its own rows too


def test_nothing_runs_if_a_frozen_file_has_changed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(blind, "frozen_files_changed", lambda: ["scripts/run_real_tables_blind.py"])
    for command in ("real", "generated", "collect", "check"):
        with pytest.raises(SystemExit, match="frozen files changed"):
            blind.main([command, "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
