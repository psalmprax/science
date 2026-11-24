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
    assert summary["informed"]["known_form"] == 1 and summary["informed"]["wrong_reports"] == 0
    assert summary["plain"]["known_form"] == 0 and summary["product"]["ran"] == 0
    decoy = table["decoy"]                                                                            # the same inputs, another law
    assert decoy["law_of_the_decoy"] == "a*(1 - exp(-b*p))" and decoy["reading"]["action"] == "read"
    assert set(decoy) >= {"judged_against_the_known_law", "judged_against_the_decoys_law"}


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
        if tables[key]["kind"] == "known_law":                                                        # a decoy: another law, same inputs
            decoy = blind.sp.sympify(entry["decoy"]["law"])
            assert {str(s) for s in decoy.free_symbols} == set(tables[key]["inputs"]) | set(entry["decoy"]["parameters"]), key
            assert entry["decoy"]["law"] != tables[key]["law"] and entry["decoy"]["why"], key
    assert all(key in tables and tables[key]["split"] == "held_back" for key in blind.KNOWN_LAW)
    assert set(blind.OPEN.values()) | set(blind.KNOWN_LAW) == {k for k, t in tables.items() if t["split"] == "held_back"}
    assert blind.needed_columns("cosmic_chronometers") == ["z", "H", "sigma_H"]


def test_the_generated_arm_draws_laws_the_development_tables_did_not_use() -> None:
    laws = blind.blind_laws()
    development = {blind.generated.curriculum.structure(blind.sp.sympify(law["text"])) for law in blind.generated.draw_laws(120)}
    assert len(laws) == blind.BLIND_TABLES and all(law["tag"] == "small-tables-blind" for law in laws)
    assert not {blind.generated.curriculum.structure(blind.sp.sympify(law["text"])) for law in laws} & development
    a, b = blind.generated.table_of(laws[0]), blind.generated.table_of({**laws[0], "tag": "small-tables-dev"})
    assert not np.array_equal(a["X"], b["X"])                                                         # its own rows too


def test_nothing_runs_if_a_frozen_file_has_changed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(blind, "frozen_files_changed", lambda: ["scripts/run_real_tables_blind.py"])
    for command in ("real", "generated", "traps", "collect", "check"):
        with pytest.raises(SystemExit, match="frozen files changed"):
            blind.main([command, "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def _checked(monkeypatch: pytest.MonkeyPatch, rows: int = 5, lines: int = 6) -> list[str]:
    """Stubs for the check before reading: every table loads with ``rows`` usable rows of a file of ``lines`` lines."""
    asked: list[str] = []

    def loading(key: str, cache: Path) -> dict:
        asked.append(key)
        return {"table": {"key": key, "lines": lines, "format": "csv", "inputs": ["x"], "target": "y"},
                "columns": {"x": np.arange(1.0, rows + 1), "y": np.arange(1.0, rows + 1)}, "rows_dropped": 0}

    monkeypatch.setattr(blind, "frozen_files_changed", lambda: [])
    monkeypatch.setattr(blind, "held_back", loading)
    monkeypatch.setattr(blind, "fitted_forms", lambda table, stated, X, y: [{"fitted": 1, "r2_all_rows": 1.0, "counted": True}])
    monkeypatch.setattr(blind, "known_law_readings", lambda *a: pytest.fail("a table was read after the check failed"))
    return asked


def test_nothing_is_read_if_one_table_cannot_be_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every table is fetched, checked and parsed before a method reads any: a held-back table is read once."""
    asked = _checked(monkeypatch)
    loading = blind.held_back

    def one_fails(key: str, cache: Path) -> dict:
        if key == "nikuradse_superpipe":
            asked.append(key)
            raise ValueError("the downloaded file is not the pinned bytes")
        return loading(key, cache)

    monkeypatch.setattr(blind, "held_back", one_fails)
    with pytest.raises(SystemExit, match=r"nikuradse_superpipe could not be loaded .* Nothing was run"):
        blind.main(["real", "--out", str(tmp_path), "--searches", "0"])
    assert asked == [*blind.KNOWN_LAW, "nikuradse_superpipe"] and not (tmp_path / "real.json").exists()
    with pytest.raises(SystemExit, match="not parts of the run"):
        blind.main(["real", "--out", str(tmp_path), "--searches", "0", "--only", "hubble"])


def test_nothing_is_read_if_a_table_lost_rows_in_parsing_or_the_judges_own_fit_fails(monkeypatch: pytest.MonkeyPatch,
                                                                                    tmp_path: Path) -> None:
    _checked(monkeypatch, rows=5, lines=13)                                                           # twelve rows promised, five usable
    with pytest.raises(SystemExit, match=r"has 5 usable rows where its file's 13 lines promise about 12\. Nothing was run"):
        blind.main(["real", "--out", str(tmp_path), "--searches", "0"])
    _checked(monkeypatch, rows=11, lines=13)                                                          # one short of the promise: allowed
    monkeypatch.setattr(blind, "fitted_forms", lambda table, stated, X, y: [{"error": "RuntimeError: no fit", "counted": False}])
    with pytest.raises(SystemExit, match=r"the known law could not be fitted to the table .* Nothing was run"):
        blind.main(["real", "--out", str(tmp_path), "--searches", "0"])
    _checked(monkeypatch, rows=11, lines=13)
    monkeypatch.setattr(blind, "decoy_target", lambda *a, **k: (_ for _ in ()).throw(ValueError("the decoy's values are not finite")))
    with pytest.raises(SystemExit, match=r"its decoy could not be built .* Nothing was run"):
        blind.main(["real", "--out", str(tmp_path), "--searches", "0"])
    assert not (tmp_path / "real.json").exists()


def test_a_table_is_fetched_again_on_failure_and_rows_missing_a_needed_value_are_dropped(monkeypatch: pytest.MonkeyPatch,
                                                                                        tmp_path: Path) -> None:
    calls = {"n": 0}

    def fetch(key: str, *, preregistration: str, into: Path) -> Path:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection reset")
        into.mkdir(parents=True, exist_ok=True)
        path = into / f"{key}.csv"
        path.write_text("D,v,name\n1.0,500,a\n2.0,,b\n3.0,1500,c\n", encoding="utf-8")
        return path

    monkeypatch.setattr(blind.real_data, "fetch", fetch)
    monkeypatch.setattr(blind.time, "sleep", lambda seconds: None)
    loaded = blind.held_back("hubble_1929", tmp_path)
    assert calls["n"] == 3 and loaded["rows_dropped"] == 1
    assert loaded["columns"]["D"].tolist() == [1.0, 3.0] and loaded["columns"]["v"].tolist() == [500.0, 1500.0]
    assert "name" not in loaded["columns"]                                                            # only the columns the law uses
    monkeypatch.setattr(blind.real_data, "fetch", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(RuntimeError, match="not fetched in 3 attempts"):
        blind.held_back("tully_fisher", tmp_path)


def test_an_error_in_one_reading_is_recorded_and_the_other_arms_still_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A held-back table is read once: one failing arm must not lose the others."""
    pytest.importorskip("z3")
    loaded = blind.real_data.load("langmuir_table_ix")
    monkeypatch.setattr(blind, "held_back", lambda key, cache: {"table": loaded["table"], "columns": loaded["columns"]})
    monkeypatch.setattr(blind, "READ_SECONDS", 60.0)
    read = blind.small_table.read

    def failing(X: object, y: object, **options: object) -> dict:
        if not options.get("known"):
            raise RuntimeError("made to fail")
        return read(X, y, **options)

    monkeypatch.setattr(blind.small_table, "read", failing)
    table = blind.known_law_table("langmuir_table_ix", 0.0, tmp_path)
    assert table["readings"]["plain"] == {"action": "error", "error": "plain: RuntimeError: made to fail",
                                         "judged": {"law": None}}
    assert table["readings"]["informed"]["judged"]["same_form_as_the_known_law"] is True
    summary = blind.summarise_known([table])
    assert summary["plain"]["errors"] == 1 and summary["plain"]["read"] == 0 and summary["informed"]["known_form"] == 1


def _stub_table(key: str, found: bool = True, on_decoy: bool = False, reported: bool = True) -> dict:
    return {"key": key, "rows": 9,
            "readings": {"informed_with_searches": {"action": "read", "reported": reported, "judged": {"same_form_as_the_known_law": found}}},
            "direct": {"action": "widen_search"}, "product": {"action": "not_run"},
            "decoy": {"reading": {"action": "read"}, "judged_against_the_known_law": {"same_form_as_the_known_law": on_decoy}}}


def test_the_real_run_writes_after_every_part_and_survives_a_part_that_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    monkeypatch.setattr(blind, "frozen_files_changed", lambda: [])
    monkeypatch.setattr(blind, "preflight", lambda parts, cache: {part: {"rows": 9} for part in parts})
    seen: list[Path] = []
    order: list[str] = []

    def readings(key: str, searches: float, cache: Path, limit: float) -> dict:
        seen.append(cache)
        order.append(f"reading {key}")
        if key == "time_dilation":
            raise ValueError("a bug in the judging")
        table = _stub_table(key, found=key != "hubble_1929")
        return {k: v for k, v in table.items() if k not in ("direct", "product")}

    def rivals(key: str, searches: float, cache: Path, allowance: object) -> dict:
        order.append(f"route {key}")
        if key == "tully_fisher":
            raise RuntimeError("the route broke")
        return {arm: {"action": "widen_search", "judged": {"same_form_as_the_known_law": key == "hubble_1929"}} for arm in blind.ROUTE_ARMS}

    monkeypatch.setattr(blind, "known_law_readings", readings)
    monkeypatch.setattr(blind, "known_law_rivals", rivals)
    monkeypatch.setattr(blind, "pipe_friction", lambda searches, cache, limit, route_limit: order.append("pipes") or {"rows": 100})
    monkeypatch.setattr(blind, "expansion_rate", lambda searches, cache, limit: (_ for _ in ()).throw(RuntimeError("no rows")))
    assert blind.main(["real", "--out", str(tmp_path), "--searches", "0"]) == 0
    record = json.loads((tmp_path / "real.json").read_text(encoding="utf-8"))
    assert [t["key"] for t in record["known_law"]] == list(blind.KNOWN_LAW) and record["search_stage"] == "off"
    first_route = next(i for i, step in enumerate(order) if step.startswith("route"))
    assert all(step.startswith("reading") or step == "pipes" for step in order[:first_route])          # the reading on every table first
    assert first_route == len(blind.KNOWN_LAW) + 1 and "route time_dilation" not in order               # a table not read has no rivals
    failed = next(t for t in record["known_law"] if t["key"] == "time_dilation")
    assert failed["action"] == "error" and "a bug in the judging" in failed["error"] and failed["readings"] == {}
    assert record["summary"]["informed_with_searches"] == {
        "read": 5, "errors": 0, "no_answer": 0, "known_form": 4, "reported": 5, "wrong_reports": 1,
        "known_form_on": [k for k in blind.KNOWN_LAW if k not in ("time_dilation", "hubble_1929")]}
    assert record["summary"]["direct"]["known_form_on"] == ["hubble_1929"]
    broken = next(t for t in record["known_law"] if t["key"] == "tully_fisher")
    assert broken["direct"]["action"] == "error" and broken["readings"]["informed_with_searches"]["action"] == "read"
    assert record["pipe_friction"] == {"rows": 100} and record["expansion_rate"]["action"] == "error"
    assert not seen[0].exists() and sorted(p.name for p in tmp_path.iterdir()) == ["real.json"]       # the held-back files are gone

    # the deviation rule: the two failed parts alone; a part that was read is not replaced
    monkeypatch.setattr(blind, "known_law_readings", lambda key, searches, cache, limit: {
        k: v for k, v in _stub_table(key, found=key == "time_dilation").items() if k not in ("direct", "product")})
    monkeypatch.setattr(blind, "known_law_rivals", lambda key, searches, cache, allowance: {arm: {"action": "widen_search"} for arm in blind.ROUTE_ARMS})
    monkeypatch.setattr(blind, "expansion_rate", lambda searches, cache, limit: {"rows": 31})
    assert blind.main(["real", "--out", str(tmp_path), "--searches", "0", "--only", "time_dilation,hubble_1929,expansion_rate"]) == 0
    alone = json.loads((tmp_path / "real_only_time_dilation_and_hubble_1929_and_expansion_rate.json").read_text(encoding="utf-8"))
    assert alone["only"] == ["time_dilation", "hubble_1929", "expansion_rate"] and "pipe_friction" not in alone
    merged = blind.merged_real(tmp_path)
    assert [(r["part"], r["taken"]) for r in merged["run_alone"]] == [("time_dilation", True), ("hubble_1929", False),
                                                                      ("expansion_rate", True)]
    assert [t["key"] for t in merged["known_law"]] == list(blind.KNOWN_LAW)
    assert merged["summary"]["informed_with_searches"]["known_form"] == 5                              # hubble's first reading stands
    assert "hubble_1929" not in merged["summary"]["informed_with_searches"]["known_form_on"]
    assert merged["expansion_rate"] == {"rows": 31} and merged["pipe_friction"] == {"rows": 100}


def test_collecting_survives_a_file_cut_short(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    monkeypatch.setattr(blind, "frozen_files_changed", lambda: [])
    (tmp_path / "real.json").write_text('{"known_law": [{"key": "hubble', encoding="utf-8")            # a job killed while writing
    row = {"index": 0, "rows": 8, "noise": 0.0, "inputs": 1,
           "small": {"law": "x0", "reported": True, "accurate": True, "true_form": True},
           "direct": {"law": None, "reported": False, "accurate": False, "true_form": False}}
    (tmp_path / "rows_0.jsonl").write_text(json.dumps(row) + "\n" + '{"index": 1, "rows"', encoding="utf-8")
    assert blind.main(["collect", "--out", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert "real" not in results and results["generated"]["tables"] == 1 and results["generated"]["lines_cut_short"] == 1
    assert results["expectations"]["D1"]["holds"] is None and results["expectations"]["D1"]["complete"] is False


def test_the_traps_are_four_kinds_the_reading_was_not_developed_on() -> None:
    laws = blind.trap_laws()
    grammar = blind.generated.curriculum
    assert [law["index"] for law in laws] == list(range(4 * blind.TRAP_TABLES))
    by_kind = {kind: [law for law in laws if law["kind"] == kind] for kind in blind.TRAPS}
    assert all(len(group) == blind.TRAP_TABLES for group in by_kind.values())
    for group in by_kind.values():                                                                    # every kind over the same sizes
        assert sorted({law["rows"] for law in group}) == list(blind.TRAP_ROWS)
        assert all(sum(law["rows"] == rows for law in group) == blind.TRAP_TABLES // 3 for rows in blind.TRAP_ROWS)
    used = {grammar.structure(blind.sp.sympify(law["text"])) for law in [*blind.generated.draw_laws(120), *blind.blind_laws()]}
    fresh = {grammar.structure(blind.sp.sympify(law["text"])) for kind in ("outside_the_grammar", "harder_laws") for law in by_kind[kind]}
    assert len(fresh) == 2 * blind.TRAP_TABLES and not fresh & used                                   # new structures, each once
    assert {law["inputs"] for law in by_kind["harder_laws"]} == {3} and {law["level"] for law in by_kind["harder_laws"]} == {4, 5}

    outside = [blind.sp.sympify(law["text"]) for law in by_kind["outside_the_grammar"]]
    assert all(blind.outside_the_grammar(expr) for expr in outside)                                   # a function of a function, or of two inputs
    drawn_by_the_grammar = [*blind.generated.draw_laws(120), *blind.blind_laws(), *by_kind["harder_laws"]]
    assert not any(blind.outside_the_grammar(blind.sp.sympify(law["text"])) for law in drawn_by_the_grammar)
    assert {law["inputs"] for law in by_kind["outside_the_grammar"]} == {1, 2}
    assert {law["noise"] for law in by_kind["one_gross_error"]} == set(blind.NOISE_TRAP_NOISES)
    assert {law.get("trap") for law in by_kind["one_gross_error"]} == {"one_gross_error"}
    assert {law.get("trap") for law in by_kind["noise_in_proportion"]} == {"relative_noise"}
    assert not {law["text"] for law in by_kind["one_gross_error"]} & {law["text"] for law in by_kind["noise_in_proportion"]}

    gross = by_kind["one_gross_error"][0]
    plain = {k: v for k, v in gross.items() if k != "trap"}
    with_error, without = blind.generated.table_of(gross), blind.generated.table_of(plain)
    off = np.abs(with_error["y"] - without["y"])
    size = float(np.sqrt(np.mean(without["test_clean"] ** 2)))
    assert int((off > 0).sum()) == 1 and off.max() > 5 * gross["noise"] * size                         # one row, far outside the noise

    relative = by_kind["noise_in_proportion"][0]
    table = blind.generated.table_of(relative)
    truth = blind.sp.lambdify([blind.sp.Symbol(n) for n in table["names"]], table["truth"], "numpy")(*table["X"].T)
    assert np.std(table["y"] / truth - 1.0) == pytest.approx(relative["noise"], rel=0.6)              # in proportion to each row's value
    a = blind.generated.table_of(blind.blind_laws()[0])
    b = blind.generated.table_of({**blind.blind_laws()[0], "tag": "small-tables-trap-noise_in_proportion"})
    assert not np.array_equal(a["X"], b["X"])                                                         # a trap's rows are its own


def test_the_known_form_needs_exactly_the_forms_constants() -> None:
    rng = np.random.default_rng(5)
    distance = rng.uniform(0.2, 2.0, 30)
    X = {"D": distance}
    speed = 450.0 * distance + 60.0 * rng.standard_normal(30)
    forms = blind.fitted_forms({"key": "made_up", "law": "a*D", "law_parameters": ["a"]}, {}, X, speed)
    assert blind.fitted_constants("450.0*D + 12.0", ["D"]) == 2 and blind.fitted_constants("0.36*D**1.5", ["D"]) == 1
    assert blind.fitted_constants("2.0*D/(D + 3.0) + 2.0", ["D"]) == 3                                # each occurrence counts
    line_through_zero = blind.beside_the_known_law("441.0*D", forms, X, speed)
    assert line_through_zero["same_form_as_the_known_law"] is True and line_through_zero["form_matched"] == "a*D"
    with_an_offset = blind.beside_the_known_law("450.0*D + 12.0", forms, X, speed)                    # refits to a*D, but claims an offset
    assert with_an_offset["refits_to_the_form"] is True and with_an_offset["same_form_as_the_known_law"] is False
    assert blind.beside_the_known_law("300.0*D**2", forms, X, speed)["same_form_as_the_known_law"] is False
    assert blind.beside_the_known_law(None, forms, X, speed) == {"law": None}

    width = rng.uniform(100.0, 500.0, 18)                                                             # fewer constants is another claim too
    magnitude = -7.0 * np.log(width) + 20.0 + 0.2 * rng.standard_normal(18)
    line_in_the_log = blind.fitted_forms({"key": "made_up", "law": "a*log(DV) + b", "law_parameters": ["a", "b"]}, {}, {"DV": width}, magnitude)
    assert blind.beside_the_known_law("-7.1*log(DV) + 20.4", line_in_the_log, {"DV": width}, magnitude)["same_form_as_the_known_law"] is True
    without_the_offset = blind.beside_the_known_law("-3.04*log(DV)", line_in_the_log, {"DV": width}, magnitude)
    assert without_the_offset["refits_to_the_form"] is True and without_the_offset["same_form_as_the_known_law"] is False


def test_the_readings_own_count_of_constants_is_used_not_the_factor_it_writes_for_the_units() -> None:
    """Found by review: the reading writes its law back in the table's units, and a form with no leading constant
    gains a written factor it never fitted. Counted from the text, Langmuir's own form had three constants."""
    rng = np.random.default_rng(6)
    pressure = np.linspace(0.5, 60.0, 17)
    X = {"p": pressure}
    uptake = 6.1 * pressure / (pressure + 13.9) * (1 + 0.003 * rng.standard_normal(17))
    forms = blind.fitted_forms({"key": "made_up", "law": "a*p/(1 + b*p)", "law_parameters": ["a", "b"]}, {}, X, uptake)
    as_written = "4.5348354*p/(0.74447959*p + 10.356215)"                                             # p/(C*p + C), times the target's size
    assert blind.fitted_constants(as_written, ["p"]) == 3
    assert blind.beside_the_known_law(as_written, forms, X, uptake)["same_form_as_the_known_law"] is False
    judged = blind.beside_the_known_law(as_written, forms, X, uptake, constants=2)                    # what the reading fitted
    assert judged["same_form_as_the_known_law"] is True and judged["fitted_constants"] == 2

    reading = blind.small_table.read(X, uptake * 1e6, memory=False)                                   # and end to end, in other units
    assert reading["action"] == "read"
    counted = blind.beside_the_known_law(reading["law"], blind.fitted_forms(
        {"key": "made_up", "law": "a*p/(1 + b*p)", "law_parameters": ["a", "b"]}, {}, X, uptake * 1e6), X, uptake * 1e6,
        reading["chosen"]["constants"])
    assert counted["fitted_constants"] == reading["chosen"]["constants"]


def test_a_form_declared_beside_the_known_law_counts_only_if_the_table_cannot_tell_them_apart() -> None:
    rng = np.random.default_rng(3)
    star, distance = rng.uniform(0.5, 2.0, 12), rng.uniform(0.05, 5.0, 12)
    table = {"key": "made_up", "law": "a*sqrt(d**3/(m1 + m2))", "law_parameters": ["a"]}
    stated = {"also_accepted": [{"law": "a*sqrt(d**3/m1)", "parameters": ["a"], "why": "the planet weighs nothing beside the star"}]}

    planets = {"m1": star, "m2": rng.uniform(1e-4, 3e-3, 12), "d": distance}
    period = np.sqrt(distance ** 3 / (star + planets["m2"])) * (1 + 0.002 * rng.standard_normal(12))
    forms = blind.fitted_forms(table, stated, planets, period)
    assert [form["counted"] for form in forms] == [True, True]
    without_the_planet = blind.beside_the_known_law("1.0*d**1.5/m1**0.5", forms, planets, period)
    assert without_the_planet["same_form_as_the_known_law"] is True and without_the_planet["form_matched"] == "a*sqrt(d**3/m1)"
    assert blind.beside_the_known_law("0.9*d**1.5/(m1 + m2)**0.5", forms, planets, period)["form_matched"] == "a*sqrt(d**3/(m1 + m2))"

    companions = {"m1": star, "m2": rng.uniform(0.5, 2.0, 12), "d": distance}                          # two stars: the second mass matters
    period = np.sqrt(distance ** 3 / (star + companions["m2"])) * (1 + 0.002 * rng.standard_normal(12))
    forms = blind.fitted_forms(table, stated, companions, period)
    assert [form["counted"] for form in forms] == [True, False]
    assert blind.beside_the_known_law("1.0*d**1.5/m1**0.5", forms, companions, period)["same_form_as_the_known_law"] is False


def test_a_table_counts_for_a1_only_if_the_known_form_is_not_given_on_its_decoy_too() -> None:
    tables = [_stub_table("kepler_exoplanets"), _stub_table("langmuir_sun_et_al", on_decoy=True),
              _stub_table("hubble_1929", found=False), _stub_table("tully_fisher")]
    tables[3]["decoy"] = {"action": "error", "error": "decoy: ValueError"}                             # a decoy not read counts against
    summary = blind.summarise_known(tables)
    assert summary["informed_with_searches"]["known_form"] == 3
    assert summary["known_form_and_not_on_the_decoy"] == ["kepler_exoplanets"]
    assert summary["decoys"]["known_form_on_the_decoy"] == ["langmuir_sun_et_al"] and summary["decoys"]["read"] == 3
    assert summary["decoys"]["known_form_given_and_decoy_not_read"] == ["tully_fisher"]
    assert summary["informed_with_searches"]["wrong_reports"] == 1                                    # hubble: reported, not the known form


def test_the_decoy_carries_the_tables_scatter_in_the_readings_own_measure() -> None:
    rng = np.random.default_rng(8)
    d = np.array([0.39, 0.72, 1.0, 1.52, 5.2, 9.54, 19.2, 30.1])
    period = 0.3644 * d ** 1.5 * (1 + 0.01 * rng.standard_normal(8))
    table = {"key": "made_up", "law": "a*d**1.5", "law_parameters": ["a"]}
    stated = {"decoy": {"law": "a*d**2", "parameters": ["a"]}}
    forms = blind.fitted_forms(table, stated, {"d": d}, period)
    known = blind.law_on(str(forms[0]["fitted"]), ["d"], d.reshape(-1, 1))
    in_proportion, fitted = blind.decoy_target(table, stated, forms, {"d": d}, period, relative=True)
    values = blind.law_on(str(fitted), ["d"], d.reshape(-1, 1))
    assert in_proportion / values == pytest.approx(period / known)                                    # each row as far off, in proportion
    as_a_difference, _fitted = blind.decoy_target(table, stated, forms, {"d": d}, period, relative=False)
    assert as_a_difference - values == pytest.approx(period - known)
    assert np.std(in_proportion / values) < 0.05 < np.max(np.abs(as_a_difference / values - 1))       # the difference swamps the small rows


def test_the_expectations_are_computed_as_the_plan_defines_them() -> None:
    def cell(reported: int, wrong: int) -> dict:
        return {"reported": reported, "wrong_reports": wrong}

    def real(found: tuple[str, ...], on_decoy: tuple[str, ...] = (), reported: bool = True, failed: str | None = None) -> dict:
        tables = [_stub_table(key, found=key in found, on_decoy=key in on_decoy, reported=reported and key in found)
                  for key in blind.KNOWN_LAW]
        for table in tables:
            if table["key"] == "kepler_binary_stars":
                table.update({"rows": 4, "readings": {"informed_with_searches": {"action": "not_run"}}, "decoy": {}})
            if table["key"] == failed:
                table.update({"action": "error", "readings": {}, "decoy": {}})
        return {"known_law": tables}

    three = ("kepler_exoplanets", "langmuir_sun_et_al", "hubble_1929")
    got = blind.expectations({"real": real(three)})
    assert got["A1"] == {"known_form": 3, "of_those_not_on_the_decoy": 3, "needed": 3, "tables_with_enough_rows": 5,
                         "known_form_given_and_decoy_not_read": [], "no_answer_from_the_arm_under_test": 0,
                         "tables_not_read": [], "holds": True}
    assert got["A2"]["holds"] is True
    assert got["A3"] == {"reading": 3, "route_in_force": 0, "route_arms_with_no_answer": dict.fromkeys(blind.ROUTE_ARMS, 0),
                         "holds": True}
    assert blind.expectations({"real": real(three, on_decoy=("hubble_1929",))})["A1"]["holds"] is False   # one came from the default
    nothing_reported = blind.expectations({"real": real(three, reported=False)})
    assert nothing_reported["A1"]["holds"] is True and nothing_reported["A2"]["holds"] is None            # no report: A2 says nothing
    lost = blind.expectations({"real": real(three, failed="tully_fisher")})
    assert lost["A1"]["holds"] is None and lost["A3"]["holds"] is None and lost["A1"]["tables_not_read"] == ["tully_fisher"]
    wrong = real(three)
    wrong["known_law"][5]["readings"]["informed_with_searches"] = {"action": "read", "reported": True,
                                                                  "judged": {"same_form_as_the_known_law": False}}
    assert blind.expectations({"real": wrong})["A2"]["holds"] is False                                    # a wrong report is a miss
    stopped = real(three)                                                                                 # the arm under test gave no answer
    stopped["known_law"][0]["readings"]["informed_with_searches"] = {"action": "no_answer"}
    assert blind.expectations({"real": stopped})["A1"] | {"holds": None} == got["A1"] | {
        "known_form": 2, "of_those_not_on_the_decoy": 2, "no_answer_from_the_arm_under_test": 1, "holds": None}
    assert blind.expectations({"real": stopped})["A1"]["holds"] is False                                  # and the table does not count

    part_of_it = {"known_law": wrong["known_law"][3:]}                                                    # a job that died half way
    partial = blind.expectations({"real": part_of_it})
    assert partial["A1"]["holds"] is None and partial["A1"]["tables_not_read"] == list(blind.KNOWN_LAW[:3])
    assert partial["A2"]["holds"] is False                                                                # a wrong report still misses
    assert blind.expectations({"real": {"known_law": []}}) == {}

    def rows(count: int) -> list[dict]:
        return [{"index": i} for i in range(count)]

    generated = {"rows": rows(blind.BLIND_TABLES), "summary": {"small": {"all": cell(40, 2)},
                                                               "paired": {"true_form_gained_by_small": 30, "true_form_lost_by_small": 1}}}
    traps = {"rows": rows(4 * blind.TRAP_TABLES), "summary": {"small": {"all": cell(19, 0)}}}
    got = blind.expectations({"generated": generated, "traps": traps})
    assert got["D1"]["holds"] is True and got["D1"]["allowed"] == pytest.approx(2.0)                      # 2 of 40 is 5 %
    assert got["D2"]["holds"] is True
    assert got["E1"]["holds"] is None                                                                     # under 20 reported: decides nothing
    generated["summary"]["small"]["all"] = cell(40, 3)
    assert blind.expectations({"generated": generated})["D1"]["holds"] is False
    generated["rows"] = rows(blind.BLIND_TABLES - 5)                                                      # a shard lost: not decided
    lost = blind.expectations({"generated": generated})
    assert lost["D1"]["holds"] is None and lost["D2"]["holds"] is None
    generated["rows"] = [*rows(240), *rows(120)]                                                          # 360 rows, a third of them twice
    assert blind.expectations({"generated": generated})["D1"]["holds"] is None


def test_the_generated_summary_has_every_arm_by_size_noise_and_kind_with_a_bound_on_the_wrong_share() -> None:
    def arm(reported: bool, accurate: bool, form: bool) -> dict:
        return {"law": "x0", "reported": reported, "accurate": accurate, "true_form": form, "action": "read"}

    rows = [{"index": i, "rows": 12 if i % 2 else 30, "noise": 0.0, "inputs": 1, "kind": "harder_laws" if i < 30 else "outside_the_grammar",
             "small": arm(True, i != 3, i % 3 == 0), "direct": arm(False, False, False)} for i in range(60)]
    summary = blind.summarise_generated(rows)
    assert set(summary) == {"small", "direct", "paired"}                                                  # no arm that was not run
    assert summary["small"]["all"]["reported"] == 60 and summary["small"]["all"]["wrong_reports"] == 1
    assert summary["small"]["all"]["wrong_share_upper_bound_95"] == pytest.approx(0.0767, abs=0.002)       # 1 of 60: under 7.7 %
    assert set(summary["small"]["by_kind"]) == {"harder_laws", "outside_the_grammar"}
    assert summary["small"]["by_kind"]["harder_laws"]["wrong_reports"] == 1
    assert set(summary["small"]["by_rows"]) == {"12", "30"} and summary["paired"]["true_form_gained_by_small"] == 20
    assert blind.cell_of([], "small")["wrong_share_upper_bound_95"] is None
    all_wrong = [{"small": arm(True, False, False), "direct": arm(False, False, False)} for _ in range(3)]
    assert blind.cell_of(all_wrong, "small")["wrong_share_upper_bound_95"] == 1.0

    laws = [{"index": i, "level": i % 3, "rows": (8, 12, 20, 30)[(i // 3) % 4]} for i in range(360)]     # shards: across a fixed shuffle
    shards = [blind.shard_of(laws, k, 3) for k in range(3)]
    assert sorted(law["index"] for shard in shards for law in shard) == list(range(360))
    assert all(len({law["level"] for law in shard}) == 3 and len({law["rows"] for law in shard}) == 4 for shard in shards)
    assert blind.shard_of(laws, 1, 3) == shards[1]


def test_the_plans_development_numbers_are_the_records_and_the_frozen_files_are_unchanged() -> None:
    import json

    plan = (ROOT / "preregistration" / "PREREG-REAL-TABLES-2026-11.md").read_text(encoding="utf-8")
    record = json.loads((ROOT / "benchmarks" / "small_tables" / "dev_comparison.json").read_text(encoding="utf-8"))
    summary = record["second_run_as_frozen"]["summary"]
    reading, first_pass = summary["small"]["all"], summary["direct"]["all"]
    for label, key in (("the true form", "true_form"), ("an accurate law", "accurate"), ("laws reported", "reported"),
                       ("of those, wrong", "wrong_reports"), ("of those, the true form", "reported_true_form")):
        assert f"| {label} | {reading[key]} | {first_pass[key]} |" in plan, label
    assert f"true form on {summary['paired']['true_form_gained_by_small']} tables where the first pass had not" in plan
    assert summary["paired"]["true_form_lost_by_small"] == 0
    before = record["first_run_before_the_rule_of_five_rows"]["summary"]["small"]["all"]
    assert f"reported {before['reported']} laws and {before['wrong_reports']} were wrong" in plan
    from_memory = record["second_run_as_frozen"]["true_forms_by_the_source_of_the_chosen_form"]["memory of confirmed laws"]
    assert from_memory == 7 and "Seven of its 17 true forms were chosen from the memory" in plan
    assert record["second_run_as_frozen"]["tables_whose_outcome_differs_from_the_first_run"] == 4
    assert "of 120 development tables read twice, 4 came out" in " ".join(plan.split())

    arm_a = json.loads((ROOT / "benchmarks" / "real_data" / "dev_arm_a_as_frozen.json").read_text(encoding="utf-8"))
    counted = blind.summarise_known(arm_a["tables"])
    assert counted["informed_with_searches"]["known_form_on"] == ["kepler_solar_system", "langmuir_table_ix"]
    assert counted["decoys"]["known_form_on_the_decoy"] == ["langmuir_table_ix"]
    assert counted["known_form_and_not_on_the_decoy"] == ["kepler_solar_system"] and "one development table of three counts" in plan
    assert blind.frozen_files_changed() == []                                                         # the digest is of these files
    for constant in ("360", "2026112477", "2026112478"):
        assert constant in plan
    assert blind.BLIND_TABLES == 360 and blind.TRAP_TABLES * len(blind.TRAPS) == 360 and blind.KNOWN_FORM_NEEDED == 3


def test_a_part_with_the_search_stage_runs_in_a_process_of_its_own_and_is_stopped_at_its_limit(tmp_path: Path) -> None:
    """Found the day of freezing: the route with its search stage went silent for two hours. An error can be caught; silence cannot."""
    rng = np.random.default_rng(4)
    x = rng.uniform(1.0, 5.0, 10)
    job = {"what": "reading", "X": {"x0": [float(v) for v in x]}, "y": [float(v) for v in 2.0 * x + 3.0], "searches": 0.0,
           "options": {"noise": "absolute"}}
    here = blind.run_job(job)
    assert here["action"] == "read" and here["reported"] is True
    answer = blind.in_a_process(job, 240.0, tmp_path)                                                 # the same job, as a child
    assert answer["action"] == "read" and answer["law"] == here["law"]
    stopped = blind.in_a_process(job, 0.2, tmp_path)                                                  # no time to answer
    assert stopped == {"action": "no_answer", "error": "no answer within 0.2 s; stopped"}
    broken = blind.in_a_process({**job, "what": "something else"}, 240.0, tmp_path)
    assert broken["action"] == "error" and "not a job" in broken["error"]                               # the child says what went wrong
    assert not list(tmp_path.iterdir())                                                               # no job file is left behind
    assert blind.part("reading", {"x0": x}, 2.0 * x + 3.0, 0.0, tmp_path, 60.0, noise="absolute")["law"] == here["law"]
    assert blind.part("nonsense", {"x0": x}, x, 0.0, tmp_path, 60.0)["action"] == "error"             # without a search stage: here, guarded


def test_the_routes_arms_get_their_full_limit_or_are_not_started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    x = np.linspace(1.0, 5.0, 12)
    monkeypatch.setattr(blind, "_loaded", lambda key, cache: ({}, {}, {"x": x}, 2.0 * x, []))
    calls: list[tuple[str, float, float]] = []

    def part(what: str, X: object, y: object, searches: float, folder: Path, limit: float, **options: object) -> dict:
        calls.append((what, searches, limit))
        return {"action": "widen_search"}

    monkeypatch.setattr(blind, "part", part)
    out_of_time = blind.known_law_rivals("made_up", 90.0, tmp_path, lambda: None)
    assert out_of_time["direct"]["error"] == "the run's time was used up before this arm" and out_of_time["product"]["action"] == "not_run"
    assert out_of_time["direct_first_pass"]["action"] == "widen_search"                               # the first pass alone always runs
    assert calls == [("direct", 0.0, blind.CHILD_LIMIT), ("product", 0.0, blind.CHILD_LIMIT)]
    calls.clear()
    in_time = blind.known_law_rivals("made_up", 90.0, tmp_path, lambda: 700.0)
    assert [c[:2] for c in calls] == [("direct", 90.0), ("product", 90.0), ("direct", 0.0), ("product", 0.0)]
    assert calls[0][2] == 700.0 and set(in_time) == set(blind.ROUTE_ARMS)
    assert blind.READ_SECONDS + 390.0 < blind.CHILD_LIMIT                                             # room for a reading's own time and the stage's


def test_a_rehearsal_reads_development_and_made_up_tables_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    def no_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("a rehearsal fetches nothing")

    monkeypatch.setattr(blind.real_data, "fetch", no_network)
    for key in blind.REHEARSAL:
        assert blind.real_data.registry()[key]["split"] == "development"
        assert blind.rehearsal_tables(key, tmp_path)["columns"].keys() == blind.real_data.load(key)["columns"].keys()
    pipes = blind.rehearsal_tables("nikuradse_superpipe", tmp_path)["columns"]
    assert len(pipes["Re"]) == 400 and blind.pipes_left_out(pipes["eD"], pipes["Re"])[1]["rule"] == "whole pipes"
    assert len(blind.rehearsal_tables("cosmic_chronometers", tmp_path)["columns"]["z"]) == 32
    with pytest.raises(blind.real_data.HeldBack):
        blind.rehearsal_tables("hubble_1929", tmp_path)

    held_back = blind.held_back
    done: list[str] = []
    monkeypatch.setattr(blind, "known_law_readings", lambda key, searches, cache, limit: done.append(key) or {"key": key, "readings": {}})
    monkeypatch.setattr(blind, "known_law_rivals", lambda key, searches, cache, allowance: {})
    monkeypatch.setattr(blind, "pipe_friction", lambda searches, cache, limit, route_limit: {"rows": len(blind.held_back("nikuradse_superpipe", cache)["columns"]["f"])})
    monkeypatch.setattr(blind, "expansion_rate", lambda searches, cache, limit: {"limit": limit})
    monkeypatch.setattr(blind, "frozen_files_changed", lambda: pytest.fail("a rehearsal needs no digest"))
    try:
        assert blind.main(["rehearse", "--out", str(tmp_path / "r"), "--searches", "0", "--limit", "123"]) == 0
    finally:
        blind.held_back = held_back                                                                   # the rehearsal replaced it in the module
    record = json.loads((tmp_path / "r" / "rehearsal.json").read_text(encoding="utf-8"))
    assert done == list(blind.REHEARSAL) and record["rehearsal"] is True and "preregistration" not in record
    assert record["pipe_friction"] == {"rows": 400} and record["expansion_rate"] == {"limit": 123.0}
