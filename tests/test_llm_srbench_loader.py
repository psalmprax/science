"""The LLM-SRBench loader on a stand-in dataset of the same layout: it checks every mapping and leaks nothing."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

h5py = pytest.importorskip("h5py")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]


def _loader():
    spec = importlib.util.spec_from_file_location("llm_srbench", ROOT / "scripts" / "llm_srbench.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_srbench"] = module
    spec.loader.exec_module(module)
    return module


def _stand_in(tmp_path: Path) -> Path:
    """Two problems with real-looking rows and one whose expression does not match its rows."""
    rng = np.random.default_rng(0)
    (tmp_path / "data").mkdir()
    rows = []
    with h5py.File(tmp_path / "lsr_bench_data.hdf5", "w") as handle:
        for name, expression, symbols, properties, f in (
            ("logistic", "dP_dt = 0.8*P*(1 - P/30)", ["dP_dt", "P"], ["O", "V"], lambda P: 0.8 * P * (1 - P / 30)),
            ("rate", "A*exp(-k*T)", ["r", "T", "A", "k"], ["O", "V", "C", "C"], None),
            ("wrong", "x + 1", ["y", "x"], ["O", "V"], lambda x: x ** 2),
        ):
            rows.append({"name": name, "symbols": symbols, "symbol_descs": ["d"] * len(symbols),
                         "symbol_properties": properties, "expression": expression})
            group = handle.create_group(f"/lsr_synth/bio_pop_growth/{name}")
            for split, n in (("train", 400), ("test", 100), ("ood_test", 50)):
                x = rng.uniform(1.0, 5.0, n)
                if name == "rate":                       # constants are columns too: the arrays have every symbol
                    A, k = np.full(n, 2.0), np.full(n, 0.3)
                    group[split] = np.column_stack([A * np.exp(-k * x), x, A, k])
                else:
                    group[split] = np.column_stack([f(x), x])
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "lsr_synth_bio_pop_growth-00000-of-00001.parquet")
    return tmp_path


def test_it_loads_checked_problems_and_counts_the_rest(tmp_path) -> None:
    loader = _loader()
    problems, left_out = loader.load(_stand_in(tmp_path), splits=("lsr_synth_bio_pop_growth",))
    assert len(problems) == 2 and left_out == {"the expression does not reproduce the output column": 1}
    x0 = sp.Symbol("x0")
    for problem in problems:                     # constants fixed in the rows take their values; inputs become x0, ...
        assert problem.inputs == 1 and problem.truth.free_symbols == {x0}
        assert problem.train.shape == (400, 2) and problem.test.shape == (100, 2) and problem.ood.shape == (50, 2)
        assert len(problem.key) == 12 and "logistic" not in problem.key


def test_its_schema_says_nothing_of_the_contents(tmp_path) -> None:
    loader = _loader()
    described = json.dumps(loader.schema(_stand_in(tmp_path)))
    assert "lsr_synth_bio_pop_growth-00000-of-00001.parquet" in described and "expression" in described
    for secret in ("0.8*P", "dP_dt", "logistic", "exp(-k*T)", "rate"):
        assert secret not in described


def test_it_refuses_to_run_without_a_token(monkeypatch) -> None:
    loader = _loader()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="gated"):
        loader.download()
