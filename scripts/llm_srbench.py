#!/usr/bin/env python3
"""Loader for LLM-SRBench (``nnheui/llm-srbench`` on the Hugging Face hub), built to keep its content private.

**The dataset is gated.** It is read with the token in ``HF_TOKEN`` of an account
that has been granted access, and nothing here works around that. This
repository is public, and so are its workflow logs and artifacts. So:

* nothing in this module prints, writes or returns for publication an
  equation, a symbol name or a value from the dataset;
* a problem is referred to by ``problem_key``, a hash of its name;
* ``schema`` describes the files by column names, row counts and array shapes
  only, never by their contents.

**The layout assumed**, from the benchmark's own loader. One parquet table per
split (``lsr_transform``, ``lsr_synth_bio_pop_growth``, ``lsr_synth_chem_react``,
``lsr_synth_matsci``, ``lsr_synth_phys_osc``) has the columns ``name``,
``symbols``, ``symbol_descs``, ``symbol_properties`` and ``expression``. The rows
are in ``lsr_bench_data.hdf5`` under ``/lsr_transform/<name>`` or
``/lsr_synth/<domain>/<name>``, with arrays ``train``, ``test`` and, for the
synthetic domains, ``ood_test``. Each array has one column per symbol; the
output symbol's property is ``O``. An expression may write a variable as a function
of another, ``P(t)``; that is read as the variable ``P``.

**Nothing is taken on trust.** Each problem's mapping is checked against its
own rows: the expression, evaluated on the input columns, must reproduce the
output column (R² of at least 0.999 on the training rows). A constant whose
column is the same in every row takes that value. A problem that does not
check out is left out and counted, never guessed at.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

REPO = "nnheui/llm-srbench"
SPLITS = ("lsr_transform", "lsr_synth_bio_pop_growth", "lsr_synth_chem_react", "lsr_synth_matsci", "lsr_synth_phys_osc")
DATA_FILE = "lsr_bench_data.hdf5"
CHECK_R2 = 0.999
OUTPUT, VARIABLE = "O", "V"


@dataclass
class Problem:
    """One problem, with its inputs renamed ``x0``, ``x1``, ...; its texts stay inside the process."""

    key: str                      #: a hash of its name: what may be printed
    split: str
    inputs: int
    truth: sp.Expr                #: the law over x0, x1, ...: never printed
    train: np.ndarray             #: (rows, inputs + 1): inputs, then the output
    test: np.ndarray
    ood: np.ndarray | None


def problem_key(split: str, name: str) -> str:
    return hashlib.sha256(f"{split}|{name}".encode()).hexdigest()[:12]


def _group(split: str, name: str) -> str:
    if split == "lsr_transform":
        return f"/lsr_transform/{name}"
    return f"/lsr_synth/{split.removeprefix('lsr_synth_')}/{name}"


def _rhs(expression: str, symbols: Sequence[str] = ()) -> str:
    """The right-hand side, with a variable written as a function of another (``P(t)``) written as itself (``P``)."""
    text = expression.split("=", 1)[1] if "=" in expression else expression
    for name in sorted(symbols, key=len, reverse=True):
        text = re.sub(rf"(?<![\w.]){re.escape(name)}\(\s*[A-Za-z_]\w*\s*\)", name, text)
    return text


def _r2(y: np.ndarray, f: np.ndarray) -> float:
    keep = np.isfinite(f) & np.isfinite(y)
    if keep.sum() < 10:
        return -np.inf
    y, f = y[keep], f[keep]
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - f) ** 2)) / total if total > 0 else -np.inf


def build(split: str, row: dict[str, Any], arrays: dict[str, np.ndarray]) -> Problem | str:
    """A checked problem, or the reason (with no content of the dataset in it) it is left out."""
    symbols = [str(s) for s in row["symbols"]]
    properties = [str(p) for p in row["symbol_properties"]]
    if len(symbols) != len(properties) or properties.count(OUTPUT) != 1:
        return "not exactly one output symbol"
    train = np.asarray(arrays.get("train"), dtype=float)
    if train.ndim != 2:
        return "no training array"
    columns = [i for i, p in enumerate(properties) if p in (OUTPUT, VARIABLE)]
    if train.shape[1] == len(symbols):
        columns = list(range(len(symbols)))
    elif train.shape[1] != len(columns):
        return "the arrays do not have one column per symbol"
    names = [symbols[i] for i in columns]
    out_at = names.index(symbols[properties.index(OUTPUT)])
    inputs = [i for i in range(len(names)) if i != out_at and properties[columns[i]] == VARIABLE]
    local = {name: sp.Symbol(f"s{i}") for i, name in enumerate(names)}
    try:
        expr = sp.sympify(_rhs(str(row["expression"]), names), locals=local)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return "the expression could not be read"
    renamed: dict[sp.Symbol, Any] = {local[names[i]]: sp.Symbol(f"x{k}") for k, i in enumerate(inputs)}
    for i, name in enumerate(names):                   # a constant held fixed in the rows takes its value
        if properties[columns[i]] not in (OUTPUT, VARIABLE) and i != out_at:
            column = train[:, i]
            if np.all(np.isfinite(column)) and float(np.ptp(column)) == 0.0:
                renamed[local[name]] = sp.Float(float(column[0]))
    truth = expr.xreplace(renamed)
    if truth.free_symbols - {sp.Symbol(f"x{k}") for k in range(len(inputs))}:
        return "the expression uses symbols that are not inputs"
    xs = [sp.Symbol(f"x{k}") for k in range(len(inputs))]
    fn = sp.lambdify(xs, truth, "numpy")
    with np.errstate(all="ignore"):
        values = np.asarray(fn(*train[:, inputs].T), dtype=float) * np.ones(len(train))
    if _r2(train[:, out_at], values) < CHECK_R2:
        return "the expression does not reproduce the output column"

    def ordered(a: Any) -> np.ndarray | None:
        if a is None:
            return None
        a = np.asarray(a, dtype=float)
        return np.column_stack([a[:, inputs], a[:, out_at]]) if a.ndim == 2 and a.shape[1] == train.shape[1] else None

    train_rows, test_rows = ordered(train), ordered(arrays.get("test"))
    if train_rows is None or test_rows is None:
        return "no test array of the training array's shape"
    return Problem(problem_key(split, str(row["name"])), split, len(inputs), truth, train_rows, test_rows,
                   ordered(arrays.get("ood_test")))


def download(token: str | None = None) -> Path:
    """The dataset's files, fetched with ``HF_TOKEN``; fails plainly if the account has no access."""
    from huggingface_hub import snapshot_download

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set: LLM-SRBench is gated and needs a token of an account granted access")
    # The commit is not known until the account has access; LLM_SRBENCH_REVISION pins it once it is. The
    # snapshot's directory is named by the commit it resolved to (``revision_of``), so every run records it,
    # and nothing in the files is executed: they are read as tables and arrays and each problem is checked.
    revision = os.environ.get("LLM_SRBENCH_REVISION", "main")
    return Path(snapshot_download(REPO, repo_type="dataset", token=token, revision=revision))


def revision_of(root: Path) -> str:
    """The commit a downloaded snapshot is of: its directory's name."""
    return root.name


def load(root: Path, splits: Sequence[str] = SPLITS) -> tuple[list[Problem], dict[str, int]]:
    """Every problem that passes its check, and how many were left out for each reason."""
    import h5py
    import pandas as pd

    problems: list[Problem] = []
    left_out: dict[str, int] = {}
    with h5py.File(root / DATA_FILE, "r") as handle:
        for split in splits:
            tables = sorted((root / "data").glob(f"{split}-*.parquet"))
            for table in tables:
                for row in pd.read_parquet(table).to_dict("records"):
                    group = _group(split, str(row["name"]))
                    if group not in handle:
                        left_out["no rows in the data file"] = left_out.get("no rows in the data file", 0) + 1
                        continue
                    arrays = {k: v[...] for k, v in handle[group].items()}
                    made = build(split, row, arrays)
                    if isinstance(made, str):
                        left_out[made] = left_out.get(made, 0) + 1
                    else:
                        problems.append(made)
    return problems, left_out


def schema(root: Path) -> dict[str, Any]:
    """Column names, row counts and array shapes; no value of the dataset."""
    import h5py
    import pandas as pd

    out: dict[str, Any] = {"tables": {}, "groups": 0, "array_names": set(), "shapes": set()}
    for table in sorted((root / "data").glob("*.parquet")):
        frame = pd.read_parquet(table)
        out["tables"][table.name] = {"rows": len(frame), "columns": list(frame.columns)}
    with h5py.File(root / DATA_FILE, "r") as handle:
        def visit(name: str, item: Any) -> None:
            if hasattr(item, "shape"):
                out["array_names"].add(name.rsplit("/", 1)[-1])
                out["shapes"].add(len(item.shape))
            else:
                out["groups"] += 1
        handle.visititems(visit)
    out["array_names"] = sorted(out["array_names"])
    out["shapes"] = sorted(out["shapes"])
    return out
