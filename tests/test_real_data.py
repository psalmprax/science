"""Tables of real measurements: the pins, the development copies, and the lock on the held-back tables."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "benchmarks" / "real_data"


def _load(name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


real_data = _load("real_data")
TABLES = real_data.registry()


def test_every_table_is_pinned_by_source_commit_and_hash() -> None:
    assert len(TABLES) == 11
    for key, table in TABLES.items():
        assert table["split"] in real_data.SPLITS and table["kind"] in ("known_law", "open"), key
        assert re.fullmatch(r"[0-9a-f]{40}", table["source"]["commit"]) and re.fullmatch(r"[0-9a-f]{64}", table["sha256"]), key
        assert table["source"]["repo"].count("/") == 1 and table["source"]["path"] and table["source"]["licence"], key
        assert table["inputs"] and table["target"] not in table["inputs"], key
        assert (table["law"] is not None) == (table["kind"] == "known_law"), key
    assert sorted(k for k, t in TABLES.items() if t["split"] == "development") == [
        "kepler_solar_system", "langmuir_table_ix", "leavitt"]


def test_the_development_copies_are_the_pinned_bytes_and_no_held_back_table_is_in_the_repository() -> None:
    held_back = {t["sha256"]: key for key, t in TABLES.items() if t["split"] == "held_back"}
    for key, table in TABLES.items():
        path = real_data.development_path(key)
        if table["split"] == "development":
            assert hashlib.sha256(path.read_bytes()).hexdigest() == table["sha256"], key
        else:
            assert not path.exists(), key
    for path in HOME.rglob("*"):
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() not in held_back, path
    assert real_data.main(["check"]) == 0


def test_development_code_cannot_read_a_held_back_table() -> None:
    for key, table in TABLES.items():
        if table["split"] == "held_back":
            with pytest.raises(real_data.HeldBack):
                real_data.load(key)


def test_a_held_back_table_is_fetched_only_for_a_preregistration_that_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("nothing may be downloaded without a preregistration")

    monkeypatch.setattr(real_data.urllib.request, "urlopen", no_network)
    for name in ("PREREG-DOES-NOT-EXIST-2026-11", "development", ""):
        with pytest.raises(real_data.HeldBack):
            real_data.fetch("hubble_1929", preregistration=name, into=tmp_path)
    assert not list(tmp_path.iterdir())


def test_the_development_tables_parse_to_their_inputs_and_target() -> None:
    kepler, langmuir, leavitt = (real_data.load(k) for k in ("kepler_solar_system", "langmuir_table_ix", "leavitt"))
    assert list(kepler["X"]) == ["d"] and len(kepler["y"]) == 8 and kepler["X"]["d"][2] == 1.0       # the Earth, at one unit
    assert set(kepler["columns"]) == {"p", "m1", "m2", "d"} and np.all(kepler["columns"]["m1"] == 1.0)
    assert list(langmuir["X"]) == ["p"] and len(langmuir["y"]) == 10                                # the file starts with a byte-order mark
    assert list(leavitt["X"]) == ["logP"] and len(leavitt["y"]) == 26
    for loaded in (kepler, langmuir, leavitt):
        assert all(np.all(np.isfinite(column)) for column in loaded["columns"].values())


def test_a_scale_in_the_registry_is_applied_and_is_the_units_conversion_it_says() -> None:
    table = dict(TABLES["kepler_exoplanets"])
    assert table["scale"]["m2"] == pytest.approx(1.898e27 / 1.9885e30, rel=1e-5)                     # Jupiter masses to solar masses
    columns = real_data.parse(table, "p m1 m2 d\n1 2 1000 4 # made up\n")
    assert columns["m2"][0] == pytest.approx(0.954488) and columns["m1"][0] == 2.0
    bare = real_data.parse(dict(TABLES["cosmic_chronometers"]), "0.5\t90\t10\n1.0\t120\t15\n")
    assert list(bare) == ["z", "H", "sigma_H"] and len(bare["z"]) == 2


def test_the_runner_reads_only_development_tables_and_tells_the_known_form_from_another() -> None:
    runner = _load("run_real_data")
    assert set(runner.DEVELOPMENT) == {k for k, t in TABLES.items() if t["split"] == "development"}
    loaded = real_data.load("kepler_solar_system")
    truth, score = runner.fitted_law(loaded["table"], loaded["X"], loaded["y"])
    assert score is not None and score > 0.9999 and "d**1.5" in str(truth)
    same = runner.judged("0.3632*d**1.5", truth, loaded["X"], loaded["y"])
    other = runner.judged("2.0*d + 0.1", truth, loaded["X"], loaded["y"])
    assert same["same_form_as_the_known_law"] is True and other["same_form_as_the_known_law"] is False
    assert runner.judged(None, truth, loaded["X"], loaded["y"]) == {"law": None}


def test_the_recorded_first_look_reported_nothing_and_says_what_the_product_refused() -> None:
    record = json.loads((HOME / "dev_first_pass_alone.json").read_text(encoding="utf-8"))
    tables = {t["key"]: t for t in record["record"]["tables"]}
    assert all(t["summary"]["product_reported"] == 0 and t["summary"]["direct_reported"] == 0 for t in tables.values())
    assert tables["kepler_solar_system"]["summary"]["product_refused"] == 5
    assert tables["kepler_solar_system"]["summary"]["direct_gave_the_law_tentatively"] == 5
    assert tables["langmuir_table_ix"]["summary"]["product_refused"] == 5
    assert tables["leavitt"]["summary"]["product_refused"] == 0
    assert tables["leavitt"]["summary"]["product_gave_the_law_tentatively"] == 0
