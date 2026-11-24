#!/usr/bin/env python3
"""PREREG-REAL-TABLES-2026-11: the small-table reading on real tables it was not developed on, once.

Frozen with its plan by ``preregistration/DIGEST-REAL-TABLES.json``; this script refuses to run if a frozen file
has changed, and reads a held-back table only through ``real_data.fetch``, which needs the preregistration to
exist. Five arms (the plan says what each is for and what is expected):

* **A, known laws:** six held-back tables with a known law. The reading with what is known stated and the search
  stage on (the arm under test), without the searches, with nothing stated, and beside them the route in force
  called directly and the product's own reading. Each law given is compared with the known form. A **decoy**
  of each table (another law on the same inputs, with the table's own scatter) is read by the arm under test,
  to show whether its answer depends on the rows.
* **B, pipe friction:** two whole pipes are left out by a rule fixed before the file was read; a law is read from
  the other rows and scored on the pipes left out, beside Haaland's and Colebrook's published formulas.
* **C, the expansion rate:** every third measurement is left out; a law is read from the rest, in units of each
  measurement's stated error, and scored on those left out beside the standard model's form.
* **D, generated small tables:** 360 tables drawn after freezing from a seed used nowhere else: the reading, the
  reading without the memory of confirmed laws, and the first pass of the route in force called directly.
* **E, traps:** 360 more generated tables of four kinds the reading was not developed on: laws the development
  grammar cannot draw, laws with more inputs and parts, tables with one gross error, and noise in proportion.

``rehearse`` runs the whole of ``real`` on the three development tables and on made-up tables for B and C, with
the search stage on: every path of the run, without a held-back row. It needs no digest and fetches nothing.

A held-back table is read once. So before anything is read by a method every table is fetched, checked against
its pin and parsed (``preflight``), and the run stops there if one fails. After that an error in one reading, arm
or table is recorded in its place and the run goes on, and the record is written after every part.

**Hard limits.** The day this was frozen, the route in force with its search stage went silent for two hours on a
table of six rows (GitHub run 45434942785, cancelled by its job's limit with nothing written). An error can be
caught; silence cannot. So every part that runs the search stage runs in a process of its own and is stopped after
``CHILD_LIMIT`` seconds, recorded as no answer. The arm under test is read on every table first; the route in
force's arms come after, while the run's time lasts.

    python scripts/run_real_tables_blind.py real --out DIR --searches 90 [--only PART[,PART]]
    python scripts/run_real_tables_blind.py generated --out DIR [--shard 0 --shards 3]
    python scripts/run_real_tables_blind.py traps --out DIR [--shard 0 --shards 3]
    python scripts/run_real_tables_blind.py collect --out DIR
    python scripts/run_real_tables_blind.py rehearse --out DIR --searches 90 [--limit 600]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import signal
import subprocess  # nosec B404
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import real_data  # noqa: E402
import run_real_data as dev  # noqa: E402
import run_small_tables as generated  # noqa: E402

from apedp_dyn.conf.canonical_spec import REFEREE_MAX_FPR  # noqa: E402
from apedp_dyn.discovery_l2 import small_table  # noqa: E402
from apedp_dyn.ui import route_endpoint  # noqa: E402

PREREGISTRATION = "PREREG-REAL-TABLES-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-REAL-TABLES.json"
KNOWN = ROOT / "benchmarks" / "real_data" / "known_beforehand.json"
KNOWN_LAW = ("kepler_exoplanets", "kepler_binary_stars", "langmuir_sun_et_al", "time_dilation", "hubble_1929", "tully_fisher")
OPEN = {"pipe_friction": "nikuradse_superpipe", "expansion_rate": "cosmic_chronometers"}
PARTS = (*KNOWN_LAW, *OPEN)
EXTRA_COLUMNS = {"cosmic_chronometers": ("sigma_H",)}
FETCH_ATTEMPTS = 3
SEARCH_CORES = 4             # the search stage in force runs its four searches only where four cores are free
BLIND_SEED = 2026112477
BLIND_TABLES = 360           # ten for each size, noise and level; about 440 structures are left to draw from
TRAP_SEED = 2026112478
TRAP_TABLES = 90             # of each kind
TRAPS = ("outside_the_grammar", "harder_laws", "one_gross_error", "noise_in_proportion")
TRAP_ROWS = (12, 20, 30)     # sizes at which the reading reports at all: at 8 rows it reports only one-constant laws
LAW_TRAP_NOISES = (0.0, 0.01)
NOISE_TRAP_NOISES = (0.002, 0.01)    # at 5 % noise the reading reported nothing in development: a trap there is inert
HARDER_LEVELS = (4, 5)       # three inputs with three and with four parts; development stopped at two inputs and three parts
READINGS = ("informed_with_searches", "informed", "plain")
UNDER_TEST = "informed_with_searches"
KNOWN_FORM_NEEDED = 3        # of the tables with enough rows to read (five are expected)
MIN_REPORTS = 20             # with fewer laws reported than this, a share of wrong ones decides nothing
GENERATED_ARMS = ("small", "small_without_memory", "direct")
SEED = 0
MIN_PIPE_ROWS = 10           # a roughness with fewer rows than this is not a pipe for the split
READ_SECONDS = 600.0
CHILD_LIMIT = 1500.0         # a part that runs the search stage has a process of its own and this long, then is stopped:
                             # 600 s to score, the stage's own 390 s, and room to start Julia
RUN_BUDGET = 340 * 60.0      # one of the route in force's arms is started only if its full limit fits in this (the step has 350 minutes)
ROWS_MISSING_ALLOWED = 2     # usable rows a table may fall short of its file's lines, beyond the header
REHEARSAL = ("kepler_solar_system", "langmuir_table_ix", "leavitt")
ROUTE_ARMS = ("direct", "product", "direct_first_pass", "product_first_pass")


def frozen_files_changed() -> list[str]:
    recorded = json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"]
    return [name for name, digest in recorded.items()
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]


def needed_columns(key: str) -> list[str]:
    table = real_data.registry()[key]
    return [*table["inputs"], table["target"], *EXTRA_COLUMNS.get(key, ())]


def held_back(key: str, cache: Path) -> dict[str, Any]:
    """A held-back table's needed columns, rows with a missing value dropped. Fetched once and kept in ``cache``."""
    table = real_data.registry()[key]
    path = cache / f"{key}.{'csv' if table['format'] == 'csv' else 'dat'}"
    failure: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        if path.exists():
            break
        try:
            real_data.fetch(key, preregistration=PREREGISTRATION, into=cache)
        except real_data.HeldBack:
            raise
        except Exception as exc:  # noqa: BLE001 - a download that fails is tried again; the last failure is raised below
            failure = exc
            time.sleep(5.0 * (attempt + 1))
    if not path.exists():
        raise RuntimeError(f"{key}: not fetched in {FETCH_ATTEMPTS} attempts ({type(failure).__name__}: {failure})")
    columns = real_data.parse(table, path.read_text(encoding="utf-8"))
    names = needed_columns(key)
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"{key}: no column named {missing}")
    keep = np.all(np.isfinite(np.column_stack([columns[name] for name in names])), axis=1)
    return {"table": table, "columns": {name: columns[name][keep] for name in names}, "rows_dropped": int((~keep).sum())}


def preflight(parts: Sequence[str], cache: Path) -> dict[str, Any]:
    """Every table of ``parts`` fetched, checked against its pin and parsed, before a method reads any of them."""
    report: dict[str, Any] = {}
    for part in parts:
        key = OPEN.get(part, part)
        try:
            loaded = held_back(key, cache)
        except Exception as exc:  # noqa: BLE001 - whatever stops a table loading ends the run here, before any reading
            raise SystemExit(f"{key} could not be loaded ({type(exc).__name__}: {exc}). Nothing was run.") from exc
        rows = len(next(iter(loaded["columns"].values())))
        table = loaded["table"]
        expected = int(table["lines"]) - (0 if table["format"] == "whitespace_no_header" else 1)
        if rows < 1 or rows < expected - ROWS_MISSING_ALLOWED:
            raise SystemExit(f"{key} has {rows} usable rows where its file's {table['lines']} lines promise about {expected}. Nothing was run.")
        report[key] = {"rows": rows, "rows_dropped_for_a_missing_value": loaded["rows_dropped"]}
        if part in KNOWN_LAW:                                 # the judge's own fits must work before anything is read
            stated = json.loads(KNOWN.read_text(encoding="utf-8"))["tables"][key]
            X = {name: loaded["columns"][name] for name in table["inputs"]}
            y = loaded["columns"][table["target"]]
            forms = fitted_forms(table, stated, X, y)
            if "fitted" not in forms[0] or forms[0].get("r2_all_rows") is None:
                raise SystemExit(f"{key}: the known law could not be fitted to the table ({forms[0].get('error')}). Nothing was run.")
            report[key]["known_law_r2_all_rows"] = forms[0]["r2_all_rows"]
            if rows >= small_table.MIN_ROWS and stated.get("decoy"):
                try:
                    decoy_target(table, stated, forms, X, y, relative=False)
                except Exception as exc:  # noqa: BLE001 - a decoy that cannot be built ends the run here, before any reading
                    raise SystemExit(f"{key}: its decoy could not be built ({type(exc).__name__}: {exc}). Nothing was run.") from exc
    return report


def guarded(what: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """One part of the run. An error is recorded in its place, so that it does not lose the tables still to be read."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - recorded as this part's outcome; a held-back table cannot be read a second time
        return {"action": "error", "error": f"{what}: {type(exc).__name__}: {exc}"[:300]}


def stage(searches: float) -> Any:
    return route_endpoint._proposer(searches)[0] if searches > 0 else None


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """One reading or one of the route in force's arms, from a job as written by ``part``."""
    X = {name: np.asarray(values, dtype=float) for name, values in job["X"].items()}
    y = np.asarray(job["y"], dtype=float)
    searches = float(job["searches"])
    if job["what"] == "reading":
        options = dict(job["options"])
        sigma = options.pop("sigma", None)
        return short(small_table.read(X, y, proposer=stage(searches), seed=SEED, seconds=READ_SECONDS,
                                      sigma=None if sigma is None else np.asarray(sigma, dtype=float), **options))
    if job["what"] == "direct":
        direct: dict[str, Any] = dev.direct(X, y, SEED, searches)
        return direct
    if job["what"] == "product":
        product: dict[str, Any] = dev.keep(route_endpoint.route_reading(X, y, search_seconds=searches, seed=SEED))
        return product
    raise ValueError(f"not a job: {job['what']!r}")


def in_a_process(job: dict[str, Any], limit: float, folder: Path) -> dict[str, Any]:
    """A job in a process of its own, stopped with everything it started if it has not answered within ``limit`` seconds."""
    path = folder / f"job_{uuid.uuid4().hex}.json"
    answer = path.with_suffix(".out.json")
    path.write_text(json.dumps(job), encoding="utf-8")
    # subprocess starts this script again as a child, with a fixed argument list and no shell
    command = [sys.executable, str(Path(__file__).resolve()), "child", "--out", str(folder), "--job", str(path)]
    process = subprocess.Popen(command, start_new_session=True)  # noqa: S603 - a fixed command: this script, as a child  # nosec B603
    answered = True
    try:
        process.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        answered = False
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)            # the searches run in processes of their own: stop the whole group,
        except (ProcessLookupError, PermissionError):         # after an answer too, so that nothing it started lives on
            pass
        process.wait()
        path.unlink(missing_ok=True)
    if not answered:
        answer.unlink(missing_ok=True)
        return {"action": "no_answer", "error": f"no answer within {limit:g} s; stopped"}
    got = read_json(answer)
    answer.unlink(missing_ok=True)
    if got is None:
        return {"action": "error", "error": f"its process ended with code {process.returncode} and no record"}
    return got


def part(what: str, X: dict[str, np.ndarray], y: np.ndarray, searches: float, folder: Path, limit: float,
         **options: Any) -> dict[str, Any]:
    """A reading or one of the route's arms: in a process of its own with a hard limit if it runs the search stage."""
    job = {"what": what, "X": {name: [float(v) for v in values] for name, values in X.items()}, "y": [float(v) for v in y],
           "searches": float(searches), "options": options}
    if searches > 0:
        return in_a_process(job, limit, folder)
    return guarded(what, lambda: run_job(job))


def short(reading: dict[str, Any]) -> dict[str, Any]:
    keep = ("action", "law", "reported", "tentative", "tentative_reasons", "rows", "errors", "chosen", "class_size", "class",
            "best_by_left_out_error", "stability", "accuracy", "known_checks_on_the_law", "candidates", "error", "seconds")
    out = {k: reading[k] for k in keep if k in reading}
    out["set_aside_by_what_is_known"] = (reading.get("set_aside_by_what_is_known") or {}).get("forms")
    plain: dict[str, Any] = json.loads(json.dumps(out, default=str))
    return plain


def rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if np.all(ok) else None


def law_on(law: str | None, names: list[str], X: np.ndarray) -> np.ndarray | None:
    if not law:
        return None
    symbols = [sp.Symbol(n) for n in names]
    try:
        model = sp.sympify(str(law), locals=dict(zip(names, symbols, strict=True)))
        with np.errstate(all="ignore"):
            out: np.ndarray = np.asarray(sp.lambdify(symbols, model, "numpy")(*X.T), dtype=float) * np.ones(len(X))
    except Exception:  # noqa: BLE001 - a law that cannot be evaluated scores nothing
        return None
    return out


# ------------------------------------------------------------------ arm A
def fitted_constants(law: str, names: Sequence[str]) -> int:
    """The constants a law fits: every number written in it outside an exponent, each occurrence counted."""
    def count(node: sp.Basic, in_exponent: bool) -> int:
        if isinstance(node, sp.Float):
            return 0 if in_exponent else 1
        if isinstance(node, sp.Pow):
            return count(node.base, in_exponent) + count(node.exp, True)
        return sum(count(arg, in_exponent) for arg in node.args)

    return count(sp.sympify(str(law), locals={n: sp.Symbol(n) for n in names}), False)


def fitted_forms(table: dict[str, Any], stated: dict[str, Any], X: dict[str, np.ndarray], y: np.ndarray) -> list[dict[str, Any]]:
    """The known law, then any form the plan counts as the known law on this table, each fitted to the table.

    A form declared beside the known law counts only if the table cannot tell the two apart: fitted to the table,
    they differ on its rows by no more than the measurements scatter about the known law.
    """
    forms = [{"form": table["law"], "parameters": table["law_parameters"]},
             *({"form": extra["law"], "parameters": extra["parameters"]} for extra in stated.get("also_accepted", ()))]
    names = list(X)
    columns = np.column_stack([X[n] for n in names])
    out: list[dict[str, Any]] = []
    for position, form in enumerate(forms):
        try:
            fitted, score = dev.fitted_law({**table, "law": form["form"], "law_parameters": form["parameters"]}, X, y)
        except Exception as exc:  # noqa: BLE001 - a form that cannot be fitted is recorded as such and judges nothing
            out.append({**form, "counted": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        entry = {**form, "fitted": fitted, "r2_all_rows": None if score is None else round(score, 6), "counted": position == 0}
        if position > 0 and out and "fitted" in out[0]:
            known, other = law_on(str(out[0]["fitted"]), names, columns), law_on(str(fitted), names, columns)
            if known is not None and other is not None and np.all(np.isfinite(known)) and np.all(np.isfinite(other)):
                apart, scatter = float(np.sqrt(np.mean((other - known) ** 2))), float(np.sqrt(np.mean((y - known) ** 2)))
                rounding = 1e-9 * float(np.sqrt(np.mean(known ** 2)))    # on exact data both are rounding: not told apart
                entry.update({"rms_difference_from_the_known_law": apart, "rms_scatter_about_the_known_law": scatter,
                              "counted": bool(apart <= scatter or apart <= rounding)})
        out.append(entry)
    return out


def beside_the_known_law(law: Any, forms: list[dict[str, Any]], X: dict[str, np.ndarray], y: np.ndarray,
                         constants: int | None = None) -> dict[str, Any]:
    """A law given, beside the known one: whether it has a form counted as the known law.

    It has the form if, with its constants refitted and a scale and an offset allowed, it reproduces the fitted
    form on the table's rows (``symbolic_recovery``), **and it fits exactly as many constants as the form has**.
    Without the second condition ``a*D + b`` passes for ``a*D``, since the offset refits to nothing, and ``a*log(x)``
    for ``a*log(x) + b``. ``constants`` is the reading's own count of what it fitted; the law's text is counted only
    where there is none (the route's laws), because the reading writes its law back in the table's units and a
    form with no leading constant gains a written factor it never fitted. How far the law is from the known one
    on the table's rows, beside the measurements' scatter, is recorded and decides nothing.
    """
    usable = [form for form in forms if form.get("counted") and "fitted" in form]
    if not law or not usable:
        return {"law": str(law) if law else None}
    names = list(X)
    fits = int(constants) if constants is not None else fitted_constants(str(law), names)
    out: dict[str, Any] = {}
    for form in usable:
        entry = dev.judged(str(law), form["fitted"], X, y)
        refits_to_it = bool(entry.get("same_form_as_the_known_law"))
        entry.update({"refits_to_the_form": refits_to_it, "fitted_constants": fits,
                      "same_form_as_the_known_law": refits_to_it and fits == len(form["parameters"])})
        if not out:
            out = dict(entry)
        if entry["same_form_as_the_known_law"]:
            out = {**entry, "form_matched": form["form"]}
            break
    columns = np.column_stack([X[n] for n in names])
    known, given = law_on(str(usable[0]["fitted"]), names, columns), law_on(str(law), names, columns)
    if known is not None and given is not None and np.all(np.isfinite(known)) and np.all(np.isfinite(given)):
        out.update({"rms_difference_from_the_known_law": float(np.sqrt(np.mean((given - known) ** 2))),
                    "rms_scatter_about_the_known_law": float(np.sqrt(np.mean((y - known) ** 2)))})
    return out


def decoy_target(table: dict[str, Any], stated: dict[str, Any], forms: list[dict[str, Any]], X: dict[str, np.ndarray],
                 y: np.ndarray, *, relative: bool) -> tuple[np.ndarray, sp.Expr]:
    """The decoy law's values on the table's inputs with the table's own scatter about the known law.

    The scatter is carried over as each row's ratio to the known law where the reading of the real table counted
    errors in proportion (``relative``), and as each row's difference otherwise, so that the decoy is as noisy in the
    reading's own measure as the table is.
    """
    decoy = stated["decoy"]
    names = list(X)
    columns = np.column_stack([X[n] for n in names])
    known = law_on(str(forms[0]["fitted"]), names, columns)
    fitted, _score = dev.fitted_law({**table, "law": decoy["law"], "law_parameters": decoy["parameters"]}, X, y)
    values = law_on(str(fitted), names, columns)
    if known is None or values is None or not np.all(np.isfinite(known)) or not np.all(np.isfinite(values)):
        raise ValueError("the decoy law or the known law could not be evaluated on the table's inputs")
    with np.errstate(all="ignore"):
        target: np.ndarray = values * (y / known) if relative and np.all(known != 0) else values + (y - known)
    if not np.all(np.isfinite(target)):
        raise ValueError("the decoy's values are not finite")
    return target, fitted


def decoy_reading(table: dict[str, Any], stated: dict[str, Any], forms: list[dict[str, Any]], X: dict[str, np.ndarray],
                  y: np.ndarray, statements: list[dict[str, Any]], searches: float, folder: Path, limit: float,
                  relative: bool = False) -> dict[str, Any]:
    """The arm under test on the same inputs with another law's values and the table's own scatter.

    The decoy law satisfies everything stated about the table. If the reading gives the known form on it too, its
    answer on the real table did not come from the rows.
    """
    decoy = stated["decoy"]
    target, fitted = decoy_target(table, stated, forms, X, y, relative=relative)
    reading = part("reading", X, target, searches, folder, limit, units=stated["units"], known=statements, noise=stated["noise"])
    decoy_form = [{"form": decoy["law"], "parameters": decoy["parameters"], "fitted": fitted, "counted": True}]
    fits = (reading.get("chosen") or {}).get("constants")
    return {"law_of_the_decoy": decoy["law"], "fitted": str(fitted), "scatter_carried_over": "in proportion" if relative else "as a difference",
            "reading": reading,
            "judged_against_the_known_law": guarded("comparison", lambda: beside_the_known_law(reading.get("law"), forms, X, target, fits)),
            "judged_against_the_decoys_law": guarded("comparison", lambda: beside_the_known_law(reading.get("law"), decoy_form, X, target, fits))}


def _loaded(key: str, cache: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    loaded = held_back(key, cache)
    table, columns = loaded["table"], loaded["columns"]
    stated = json.loads(KNOWN.read_text(encoding="utf-8"))["tables"][key]
    X = {name: columns[name] for name in table["inputs"]}
    y = columns[table["target"]]
    return table, stated, X, y, fitted_forms(table, stated, X, y)


def known_law_readings(key: str, searches: float, cache: Path, limit: float = CHILD_LIMIT) -> dict[str, Any]:
    """The reading's three readings of a table and the arm under test on its decoy. The route's arms come later."""
    table, stated, X, y, forms = _loaded(key, cache)
    statements = [{k: v for k, v in s.items() if k != "why"} for s in stated["known"]]
    out: dict[str, Any] = {"key": key, "rows": len(y), "inputs": table["inputs"], "known_law": table["law"]}
    out["forms_counted_as_the_known_law"] = [{k: (str(v) if k == "fitted" else v) for k, v in form.items()} for form in forms]
    arms = {
        "informed_with_searches": (searches, {"units": stated["units"], "known": statements, "noise": stated["noise"]}),
        "informed": (0.0, {"units": stated["units"], "known": statements, "noise": stated["noise"]}),
        "plain": (0.0, {"noise": "auto"}),
    }
    out["readings"] = {}

    def judged(entry: dict[str, Any]) -> dict[str, Any]:
        fits = (entry.get("chosen") or {}).get("constants")
        return guarded("comparison with the known law", lambda: beside_the_known_law(entry.get("law"), forms, X, y, fits))

    for name, (budget, options) in arms.items():
        entry = part("reading", X, y, budget, cache, limit, **options)
        if entry.get("action") == "error" and "error" in entry and not entry["error"].startswith(name):
            entry["error"] = f"{name}: {entry['error'].split(': ', 1)[-1]}"
        out["readings"][name] = {**entry, "judged": judged(entry)}
    if len(y) >= small_table.MIN_ROWS and "fitted" in forms[0] and stated.get("decoy"):
        relative = out["readings"][UNDER_TEST].get("errors") == "relative"
        out["decoy"] = guarded("decoy", lambda: decoy_reading(table, stated, forms, X, y, statements, searches, cache, limit, relative))
    return out


def known_law_rivals(key: str, searches: float, cache: Path,
                     allowance: Callable[[], float | None] = lambda: CHILD_LIMIT) -> dict[str, Any]:
    """The route in force on a table: called directly and through the product, with its search stage and without.

    ``allowance`` gives an arm with the search stage its full limit, or ``None`` if the run has no time left for one.
    """
    _table, _stated, X, y, forms = _loaded(key, cache)
    out: dict[str, Any] = {}

    def judged(law: Any) -> dict[str, Any]:
        return guarded("comparison with the known law", lambda: beside_the_known_law(law, forms, X, y))

    for arm, what, budget in (("direct", "direct", searches), ("product", "product", searches),
                              ("direct_first_pass", "direct", 0.0), ("product_first_pass", "product", 0.0)):
        if what == "direct" and len(y) < dev.MIN_DIRECT_ROWS:
            entry: dict[str, Any] = {"action": "not_run", "error": "too few rows to split"}
        elif searches == 0.0 and arm.endswith("first_pass"):
            entry = dict(out[what])                              # no search stage anywhere: the same call
        else:
            limit = allowance() if budget > 0 else CHILD_LIMIT
            entry = ({"action": "not_run", "error": "the run's time was used up before this arm"} if limit is None
                     else part(what, X, y, budget, cache, limit))
        out[arm] = {**entry, "judged": judged(entry.get("law_as_confirmed"))}
    return out


def known_law_table(key: str, searches: float, cache: Path) -> dict[str, Any]:
    """Both passes on one table (the run itself makes the first pass on every table before the second on any)."""
    return {**known_law_readings(key, searches, cache), **known_law_rivals(key, searches, cache)}


# ------------------------------------------------------------------ arm B
def haaland(reynolds: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    """Haaland (1983): 1/sqrt(f) = -1.8 log10((e/D / 3.7)**1.11 + 6.9/Re), the Darcy friction factor."""
    out: np.ndarray = (-1.8 * np.log10((roughness / 3.7) ** 1.11 + 6.9 / reynolds)) ** -2
    return out


def colebrook(reynolds: np.ndarray, roughness: np.ndarray, steps: int = 60) -> np.ndarray:
    """Colebrook and White (1939): 1/sqrt(f) = -2 log10(e/D / 3.7 + 2.51 / (Re sqrt(f))), solved by iteration."""
    inverse_root = 1.0 / np.sqrt(haaland(reynolds, roughness))
    for _ in range(steps):
        inverse_root = -2.0 * np.log10(roughness / 3.7 + 2.51 * inverse_root / reynolds)
    out: np.ndarray = inverse_root ** -2
    return out


def pipes_left_out(roughness: np.ndarray, reynolds: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """The rows of the pipes left out, by the rule fixed before the file was read."""
    with np.errstate(all="ignore"):
        rounded = np.array([float(f"{value:.3g}") for value in roughness])
    values, counts = np.unique(rounded, return_counts=True)
    pipes = [float(v) for v, c in zip(values, counts, strict=True) if c >= MIN_PIPE_ROWS]
    if len(pipes) >= 5:
        chosen = [pipes[1], pipes[4]]                       # the second and the fifth smoothest
    elif len(pipes) >= 3:
        chosen = [pipes[len(pipes) // 2]]                   # the second of three, the third of four
    else:                                                    # no pipes to speak of: every third row by Reynolds number
        test: np.ndarray = np.zeros(len(roughness), dtype=bool)
        test[np.argsort(reynolds)[1::3]] = True
        return test, {"rule": "every third row by Reynolds number", "pipes": pipes}
    return np.isin(rounded, chosen), {"rule": "whole pipes", "pipes": pipes, "left_out": chosen}


def pipe_friction(searches: float, cache: Path, limit: float = CHILD_LIMIT, route_limit: float | None = CHILD_LIMIT) -> dict[str, Any]:
    columns = held_back("nikuradse_superpipe", cache)["columns"]
    reynolds, roughness, friction = columns["Re"], columns["eD"], columns["f"]
    usable = (reynolds > 0) & (roughness > 0) & (friction > 0)
    reynolds, roughness, friction = reynolds[usable], roughness[usable], friction[usable]
    test, split = pipes_left_out(roughness, reynolds)
    work = ~test
    ratio = float(np.median(friction[work] / haaland(reynolds[work], roughness[work])))
    factor = 4.0 if 0.125 <= ratio < 0.5 else 1.0             # a Fanning factor is a quarter of the Darcy one
    friction = friction * factor
    names = ["lRe", "leD"]
    logs = np.column_stack([np.log10(reynolds), np.log10(roughness)])
    target = np.log10(friction)
    out: dict[str, Any] = {"rows": len(target), "rows_not_positive_and_dropped": int((~usable).sum()),
                           "rows_work": int(work.sum()), "rows_left_out": int(test.sum()), "split": split,
                           "median_ratio_to_haaland_on_work_rows": round(ratio, 4), "factor_applied_to_f": factor}
    work_rows = {n: logs[work, i] for i, n in enumerate(names)}
    reading = part("reading", work_rows, target[work], searches, cache, limit, noise="absolute")
    out["reading"] = reading
    if route_limit is None:
        product: dict[str, Any] = {"action": "not_run", "error": "the run's time was used up before this arm"}
    else:
        product = part("product", work_rows, target[work], searches, cache, route_limit)
    out["product"] = product

    def scores() -> dict[str, Any]:
        def refit_resid(theta: np.ndarray) -> np.ndarray:
            with np.errstate(all="ignore"):
                inside = (roughness[work] / theta[1]) ** 1.11 + theta[2] / reynolds[work]
                value = np.log10((-theta[0] * np.log10(inside)) ** -2)
            res: np.ndarray = np.where(np.isfinite(value), value - target[work], 1e3)
            return res

        refit = least_squares(refit_resid, np.array([1.8, 3.7, 6.9]), max_nfev=2000)
        with np.errstate(all="ignore"):
            refitted = np.log10((-refit.x[0] * np.log10((roughness / refit.x[1]) ** 1.11 + refit.x[2] / reynolds)) ** -2)
        predictions = {
            "the reading's law": law_on(reading.get("law"), names, logs),
            "the route in force's law": law_on(product.get("law_as_confirmed"), names, logs),
            "Haaland, published constants": np.log10(haaland(reynolds, roughness)),
            "Colebrook and White, published constants": np.log10(colebrook(reynolds, roughness)),
            "Haaland's form, constants refitted on the work rows": refitted,
        }
        return {"error_on_pipes_left_out": {name: None if pred is None else rmse(pred[test], target[test])
                                            for name, pred in predictions.items()},
                "error_on_work_rows": {name: None if pred is None else rmse(pred[work], target[work])
                                       for name, pred in predictions.items()},
                "haaland_refitted_constants": [float(v) for v in refit.x]}

    scored = guarded("scores", scores)                        # the readings above are kept whatever the scoring does
    out.update(scored if "error" not in scored else {"scores_error": scored["error"]})
    return out


# ------------------------------------------------------------------ arm C
def expansion_rate(searches: float, cache: Path, limit: float = CHILD_LIMIT) -> dict[str, Any]:
    columns = held_back("cosmic_chronometers", cache)["columns"]
    z, rate, sigma = columns["z"], columns["H"], columns["sigma_H"]
    order = np.argsort(z, kind="stable")
    test: np.ndarray = np.zeros(len(z), dtype=bool)
    test[order[1::3]] = True                                  # the second, fifth, eighth ... by redshift
    work = ~test
    stated = json.loads(KNOWN.read_text(encoding="utf-8"))["tables"]["cosmic_chronometers"]
    statements = [{k: v for k, v in s.items() if k != "why"} for s in stated["known"]]
    reading = part("reading", {"z": z[work]}, rate[work], searches, cache, limit, sigma=[float(v) for v in sigma[work]],
                   known=statements)
    out: dict[str, Any] = {"rows": len(z), "rows_work": int(work.sum()), "rows_left_out": int(test.sum()), "reading": reading}

    def scores() -> dict[str, Any]:
        def fitted(model: Any, start: list[float]) -> np.ndarray:
            got = least_squares(lambda theta: (model(theta, z[work]) - rate[work]) / sigma[work], np.array(start), max_nfev=2000)
            values: np.ndarray = model(got.x, z)
            return values

        with np.errstate(all="ignore"):
            predictions = {
                "the reading's law": law_on(reading.get("law"), ["z"], z.reshape(-1, 1)),
                "the standard model's form (flat, matter and a constant), fitted on the work rows":
                    fitted(lambda t, x: t[0] * np.sqrt(np.abs(t[1]) * (1 + x) ** 3 + 1 - np.abs(t[1])), [70.0, 0.3]),
                "a straight line, fitted on the work rows": fitted(lambda t, x: t[0] * (1 + t[1] * x), [70.0, 1.0]),
            }

        def chi2(pred: np.ndarray | None, rows: np.ndarray) -> float | None:
            if pred is None or not np.all(np.isfinite(pred[rows])):
                return None
            return float(np.mean(((pred[rows] - rate[rows]) / sigma[rows]) ** 2))

        return {"chi2_per_row_left_out": {name: chi2(pred, test) for name, pred in predictions.items()},
                "chi2_per_row_work": {name: chi2(pred, work) for name, pred in predictions.items()}}

    scored = guarded("scores", scores)
    out.update(scored if "error" not in scored else {"scores_error": scored["error"]})
    return out


# ------------------------------------------------------------------ arms D and E
def blind_laws() -> list[dict[str, Any]]:
    """Laws drawn from a seed used nowhere else, leaving out every structure the development tables used."""
    used = frozenset(generated.curriculum.structure(sp.sympify(law["text"])) for law in generated.draw_laws(120))
    laws = generated.draw_laws(BLIND_TABLES, seed=BLIND_SEED, leave_out=used)
    return [{**law, "tag": "small-tables-blind"} for law in laws]


def outside_law(rng: np.random.Generator) -> tuple[sp.Expr, int]:
    """A law the development grammar cannot draw: an exponential, logarithm, sine, tanh or root of something that is
    not a line in one input (a function of a function, or two inputs joined inside a function).

    The grammar joins parts by sums, products and ratios, and each part's function is taken of a line in one input.
    The reading's candidates exclude one function inside another, so many of these are outside what it can choose.
    """
    constants = generated.curriculum.CONSTANTS
    x0, x1 = sp.Symbol("x0"), sp.Symbol("x1")

    def c() -> sp.Float:
        return sp.Float(float(rng.choice(constants)))

    if int(rng.integers(0, 2)) == 0:
        inner = (sp.sqrt(x0), x0 ** 2, sp.log(1 + c() * x0), sp.exp(-c() * x0), 1 / (c() + x0), sp.sin(c() * x0))
        inputs = 1
    else:
        inner = (x0 * x1, x0 + c() * x1, x0 / x1, x0 ** 2 + x1 ** 2, x0 * x1 / (x0 + x1), (x0 - x1) ** 2)
        inputs = 2
    u = inner[int(rng.integers(0, len(inner)))]
    outer = (sp.exp(-c() * u), sp.log(1 + c() * u), sp.sin(c() * u), sp.sqrt(c() + u), sp.tanh(c() * u))
    law = c() * outer[int(rng.integers(0, len(outer)))]
    if int(rng.integers(0, 2)) == 0:
        law = law + c()
    return law, inputs


def outside_the_grammar(law: sp.Expr) -> bool:
    """Whether a law has an exponential, logarithm, sine, tanh or root of something that is not a line in one input."""
    for node in sp.preorder_traversal(law):
        if isinstance(node, (sp.exp, sp.log, sp.sin, sp.tanh)):
            inside = node.args[0]
        elif isinstance(node, sp.Pow) and not node.exp.is_Integer:
            inside = node.base
        else:
            continue
        symbols = sorted(inside.free_symbols, key=str)
        if symbols and (len(symbols) > 1 or not inside.is_polynomial(*symbols) or sp.Poly(inside, *symbols).total_degree() > 1):
            return True
    return False


def trap_laws() -> list[dict[str, Any]]:
    """Arm E: ``TRAP_TABLES`` tables of each kind in ``TRAPS``, none of them a kind the reading was developed on."""
    grammar = generated.curriculum
    blind = blind_laws()
    seen = {grammar.structure(sp.sympify(law["text"])) for law in [*generated.draw_laws(120), *blind]}
    rng = np.random.default_rng(TRAP_SEED)

    def cell(position: int, noises: tuple[float, ...]) -> dict[str, Any]:
        return {"rows": TRAP_ROWS[position % len(TRAP_ROWS)], "noise": noises[(position // len(TRAP_ROWS)) % len(noises)]}

    outside: list[dict[str, Any]] = []
    for _ in range(200_000):
        if len(outside) == TRAP_TABLES:
            break
        expr, inputs = outside_law(rng)
        shape = grammar.structure(expr)
        if shape in seen or not outside_the_grammar(expr) or not grammar.acceptable(expr, inputs, rng):
            continue
        seen.add(shape)
        outside.append({"text": str(expr), "inputs": inputs, "level": 0, **cell(len(outside), LAW_TRAP_NOISES)})
    if len(outside) != TRAP_TABLES:
        raise SystemExit(f"only {len(outside)} laws outside the grammar")
    mixed = [int(i) for i in np.random.default_rng(TRAP_SEED + 3).permutation(TRAP_TABLES)]    # so that level, size and noise cross
    harder = [{**law, **cell(mixed[position], LAW_TRAP_NOISES)} for position, law in enumerate(
        generated.draw_laws(TRAP_TABLES, seed=TRAP_SEED + 1, leave_out=frozenset(seen), levels=HARDER_LEVELS))]
    order = [int(i) for i in np.random.default_rng(TRAP_SEED + 2).permutation(len(blind))]
    again = [blind[i] for i in order]                         # the blind arm's laws again, on rows of their own
    kinds = {"outside_the_grammar": outside, "harder_laws": harder,
             "one_gross_error": [{**law, **cell(position, NOISE_TRAP_NOISES)} for position, law in enumerate(again[:TRAP_TABLES])],
             "noise_in_proportion": [{**law, **cell(position, NOISE_TRAP_NOISES)}
                                     for position, law in enumerate(again[TRAP_TABLES:2 * TRAP_TABLES])]}
    table_kind = {"one_gross_error": "one_gross_error", "noise_in_proportion": "relative_noise"}
    laws = [{**law, "kind": kind, "tag": f"small-tables-trap-{kind}", **({"trap": table_kind[kind]} if kind in table_kind else {})}
            for kind in TRAPS for law in kinds[kind]]
    return [{**law, "index": index} for index, law in enumerate(laws)]


def shard_of(laws: list[dict[str, Any]], shard: int, shards: int) -> list[dict[str, Any]]:
    """One shard of the laws, taken across a fixed shuffle, so that no shard is one level, size or kind."""
    order = [int(i) for i in np.random.default_rng(BLIND_SEED + len(laws)).permutation(len(laws))]
    return [laws[i] for i in order[shard::shards]]


def one_table(law: dict[str, Any]) -> dict[str, Any]:
    """One generated table, or its error in its place: one table must not end a shard."""
    try:
        return _one_table(law)
    except Exception as exc:  # noqa: BLE001 - recorded as this table's outcome: nothing given by either arm
        nothing = {"law": None, "reported": False, "accurate": False, "true_form": False, "action": "error",
                   "error": f"{type(exc).__name__}: {exc}"[:200]}
        return {**{k: law[k] for k in ("index", "inputs", "level", "rows", "noise", "trap", "kind") if k in law},
                "small": dict(nothing), "direct": dict(nothing)}


def _one_table(law: dict[str, Any]) -> dict[str, Any]:
    """One generated table: the development harness's two arms, and for arm D the reading without the memory."""
    row: dict[str, Any] = generated.one((law, 0.0))
    if "kind" in law:
        row["kind"] = law["kind"]
    else:
        table = generated.table_of(law)
        started = time.perf_counter()
        reading = small_table.read({n: table["X"][:, i] for i, n in enumerate(table["names"])}, table["y"], memory=False,
                                   seed=law["index"])
        row["small_without_memory"] = {"reported": bool(reading["reported"]), "action": reading["action"],
                                       **generated.judged(reading.get("law"), table),
                                       "source": (reading.get("chosen") or {}).get("source"),
                                       "constants": (reading.get("chosen") or {}).get("constants"),
                                       "reasons": reading.get("tentative_reasons"), "error": reading.get("error"),
                                       "seconds": round(time.perf_counter() - started, 1)}
    return row


def cell_of(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    got = [r[arm] for r in rows if arm in r]
    reported = sum(bool(g.get("reported")) for g in got)
    wrong = sum(bool(g.get("reported")) and not g.get("accurate") for g in got)
    return {"tables": len(got), "law_given": sum(bool(g.get("law")) for g in got),
            "true_form": sum(bool(g.get("true_form")) for g in got), "accurate": sum(bool(g.get("accurate")) for g in got),
            "reported": reported, "reported_true_form": sum(bool(g.get("reported")) and bool(g.get("true_form")) for g in got),
            "wrong_reports": wrong, "not_run": sum(g.get("action") == "not_run" for g in got),
            "wrong_share_upper_bound_95": None if not reported else (
                float(beta.ppf(0.95, wrong + 1, reported - wrong)) if reported > wrong else 1.0)}


def summarise_generated(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every arm present, in all and by size, noise, inputs and kind of trap; the reading against the first pass, paired."""
    arms = [arm for arm in GENERATED_ARMS if any(arm in r for r in rows)]
    out: dict[str, Any] = {arm: {"all": cell_of(rows, arm)} for arm in arms}
    for arm in arms:
        for name, key in (("by_rows", "rows"), ("by_noise", "noise"), ("by_inputs", "inputs"), ("by_kind", "kind")):
            values = sorted({r[key] for r in rows if key in r}, key=str)
            if values:
                out[arm][name] = {str(v): cell_of([r for r in rows if r.get(key) == v], arm) for v in values}
    both = [r for r in rows if "small" in r and "direct" in r]
    out["paired"] = {"true_form_gained_by_small": sum(bool(r["small"].get("true_form")) and not r["direct"].get("true_form") for r in both),
                     "true_form_lost_by_small": sum(bool(r["direct"].get("true_form")) and not r["small"].get("true_form") for r in both)}
    return out


# ------------------------------------------------------------------ the record
def summarise_known(tables: list[dict[str, Any]]) -> dict[str, Any]:
    def form(entry: dict[str, Any]) -> bool:
        return bool((entry.get("judged") or {}).get("same_form_as_the_known_law"))

    def on_the_decoy(table: dict[str, Any]) -> bool | None:
        """Whether the arm under test gave the known form on the decoy too. None if the decoy was not read."""
        decoy = table.get("decoy") or {}
        if (decoy.get("reading") or {}).get("action") != "read":
            return None
        return bool((decoy.get("judged_against_the_known_law") or {}).get("same_form_as_the_known_law"))

    out: dict[str, Any] = {"tables": len(tables)}
    for arm in READINGS:
        got = [(t.get("readings") or {}).get(arm) or {} for t in tables]
        out[arm] = {"read": sum(g.get("action") == "read" for g in got), "errors": sum(g.get("action") == "error" for g in got),
                    "no_answer": sum(g.get("action") == "no_answer" for g in got), "known_form": sum(form(g) for g in got),
                    "reported": sum(bool(g.get("reported")) for g in got),
                    "wrong_reports": sum(bool(g.get("reported")) and not form(g) for g in got),
                    "known_form_on": [t["key"] for t, g in zip(tables, got, strict=True) if form(g)]}
    for arm in ROUTE_ARMS:
        got = [t.get(arm) or {} for t in tables]
        out[arm] = {"ran": sum(g.get("action") not in (None, "not_run", "error", "no_answer") for g in got),
                    "no_answer": sum(g.get("action") == "no_answer" for g in got), "known_form": sum(form(g) for g in got),
                    "reported": sum(bool(g.get("reported")) for g in got),
                    "wrong_reports": sum(bool(g.get("reported")) and not form(g) for g in got),
                    "known_form_on": [t["key"] for t, g in zip(tables, got, strict=True) if form(g)]}
    under_test = [(t.get("readings") or {}).get(UNDER_TEST) or {} for t in tables]
    out["decoys"] = {"read": sum(on_the_decoy(t) is not None for t in tables),
                     "known_form_given_and_decoy_not_read": [t["key"] for t, g in zip(tables, under_test, strict=True)
                                                             if form(g) and on_the_decoy(t) is None],
                     "known_form_on_the_decoy": [t["key"] for t in tables if on_the_decoy(t) is True],
                     "the_decoys_own_form": [t["key"] for t in tables if bool(
                         ((t.get("decoy") or {}).get("judged_against_the_decoys_law") or {}).get("same_form_as_the_known_law"))]}
    # a table counts for A1 if the known form is given on it and is not given on its decoy (a decoy not read counts against)
    out["known_form_and_not_on_the_decoy"] = [t["key"] for t, g in zip(tables, under_test, strict=True)
                                              if form(g) and on_the_decoy(t) is False]
    return out


def share_of_wrong(cell: dict[str, Any], complete: bool) -> dict[str, Any]:
    reported, wrong = int(cell["reported"]), int(cell["wrong_reports"])
    decided = complete and reported >= MIN_REPORTS
    return {"reported": reported, "wrong_reports": wrong, "allowed": REFEREE_MAX_FPR * reported, "complete": complete,
            "wrong_share_upper_bound_95": cell.get("wrong_share_upper_bound_95"),
            "holds": (wrong <= REFEREE_MAX_FPR * reported) if decided else None}


def whole(rows: list[dict[str, Any]], count: int) -> bool:
    """Whether the rows are exactly the tables 0 to ``count`` - 1, each once."""
    return sorted(r.get("index", -1) for r in rows) == list(range(count))


def expectations(record: dict[str, Any]) -> dict[str, Any]:
    """Each expectation of the plan, computed as the plan defines it. ``holds`` is None where the plan says it decides nothing."""
    out: dict[str, Any] = {}
    tables = [t for t in (record.get("real") or {}).get("known_law") or [] if t.get("key") in KNOWN_LAW]
    if tables:
        summary = summarise_known(tables)
        missing = [key for key in KNOWN_LAW if key not in {t["key"] for t in tables}]
        not_read = [*missing, *(t["key"] for t in tables if t.get("action") == "error")]
        found, counted = summary[UNDER_TEST]["known_form"], len(summary["known_form_and_not_on_the_decoy"])
        out["A1"] = {"known_form": found, "of_those_not_on_the_decoy": counted, "needed": KNOWN_FORM_NEEDED,
                     "known_form_given_and_decoy_not_read": summary["decoys"]["known_form_given_and_decoy_not_read"],
                     "no_answer_from_the_arm_under_test": summary[UNDER_TEST]["no_answer"],
                     "tables_with_enough_rows": sum(int(t.get("rows") or 0) >= small_table.MIN_ROWS for t in tables),
                     "tables_not_read": not_read, "holds": None if not_read else counted >= KNOWN_FORM_NEEDED}
        reported = {arm: summary[arm]["reported"] for arm in READINGS}
        wrong = {arm: summary[arm]["wrong_reports"] for arm in READINGS}
        undecided = bool(not_read) or sum(reported.values()) == 0
        out["A2"] = {"reported": reported, "wrong_reports": wrong,
                     "holds": False if sum(wrong.values()) else (None if undecided else True)}
        rival = max(summary[arm]["known_form"] for arm in ROUTE_ARMS)
        out["A3"] = {"reading": found, "route_in_force": rival,
                     "route_arms_with_no_answer": {arm: summary[arm]["no_answer"] for arm in ROUTE_ARMS},
                     "holds": None if not_read else found > rival}
    generated_arm = record.get("generated")
    if generated_arm:
        complete = whole(generated_arm["rows"], BLIND_TABLES)
        out["D1"] = share_of_wrong(generated_arm["summary"]["small"]["all"], complete)
        paired = generated_arm["summary"]["paired"]
        out["D2"] = {**paired, "complete": complete,
                     "holds": (paired["true_form_gained_by_small"] > paired["true_form_lost_by_small"]) if complete else None}
    traps = record.get("traps")
    if traps:
        out["E1"] = share_of_wrong(traps["summary"]["small"]["all"], whole(traps["rows"], len(TRAPS) * TRAP_TABLES))
    return out


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if isinstance(got, dict) else None


def read_rows(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], int]:
    """The rows of every shard file; a line cut short by a job that died is skipped and counted."""
    rows, cut = [], 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                cut += 1
                continue
            if isinstance(row, dict) and isinstance(row.get("index"), int) and "small" in row and "direct" in row:
                rows.append(row)
            else:
                cut += 1
    return sorted(rows, key=lambda r: r["index"]), cut


def merged_real(folder: Path) -> dict[str, Any] | None:
    """The real tables' record, with any part run alone under the deviation rule put in the place of a part that failed.

    A part that was read is never replaced: a table is read once.
    """
    base = read_json(folder / "real.json")
    alone = [got for got in (read_json(path) for path in sorted(folder.glob("real_only*.json"))) if got]
    alone.sort(key=lambda got: str(got.get("date_utc")))      # the earlier of two runs of a part is the one that stands
    if base is None and not alone:
        return None
    record: dict[str, Any] = base or {"preregistration": PREREGISTRATION}
    record["known_law"] = [t for t in record.get("known_law") or [] if isinstance(t, dict) and t.get("key") in KNOWN_LAW]
    record["run_alone"] = []
    for extra in alone:
        for part in extra.get("only") or []:
            if part in KNOWN_LAW:
                new = next((t for t in extra.get("known_law", []) if t.get("key") == part), None)
                old = next((t for t in record["known_law"] if t.get("key") == part), None)
            else:
                new, old = extra.get(part), record.get(part)
            taken = isinstance(new, dict) and (not isinstance(old, dict) or old.get("action") == "error")
            record["run_alone"].append({"part": part, "date_utc": extra.get("date_utc"), "commit": extra.get("commit"), "taken": taken})
            if not taken:
                continue
            if part in KNOWN_LAW:
                record["known_law"] = [t for t in record["known_law"] if t.get("key") != part] + [new]
            else:
                record[part] = new
    record["known_law"].sort(key=lambda t: KNOWN_LAW.index(t["key"]) if t.get("key") in KNOWN_LAW else len(KNOWN_LAW))
    record["summary"] = summarise_known(record["known_law"]) if record["known_law"] else None
    return record


def rehearsal_tables(key: str, cache: Path) -> dict[str, Any]:
    """In place of ``held_back`` for a rehearsal: the development tables, and made-up tables for arms B and C."""
    if key in REHEARSAL:
        loaded = real_data.load(key)
        return {"table": loaded["table"], "columns": loaded["columns"], "rows_dropped": 0}
    rng = np.random.default_rng(0)
    if key == "nikuradse_superpipe":                          # ten made-up pipes on Colebrook's formula, 1 % scatter
        roughness: np.ndarray = np.repeat([1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.3e-2, 5e-2], 40)
        reynolds = np.tile(np.logspace(4, 6, 40), 10)
        friction = colebrook(reynolds, roughness) * (1 + 0.01 * rng.standard_normal(len(reynolds)))
        return {"table": {}, "columns": {"Re": reynolds, "eD": roughness, "f": friction}, "rows_dropped": 0}
    if key == "cosmic_chronometers":                          # 32 made-up measurements on the standard model's form
        z = np.sort(rng.uniform(0.05, 2.0, 32))
        sigma = rng.uniform(5.0, 25.0, 32)
        rate = 70.0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7) + sigma * rng.standard_normal(32)
        return {"table": {}, "columns": {"z": z, "H": rate, "sigma_H": sigma}, "rows_dropped": 0}
    raise real_data.HeldBack(f"{key} is not a table of the rehearsal")


def run_real(parts: Sequence[str], known: Sequence[str], searches: float, target: Path, record: dict[str, Any], cache: Path,
             child_limit: float) -> None:
    """The run on tables already checked: the reading on every table, then arms C and B, then the route in force."""
    started = time.perf_counter()

    def write() -> None:
        record["seconds"] = round(time.perf_counter() - started, 1)
        scratch = target.with_suffix(".part")
        scratch.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(scratch, target)                           # a job killed while writing leaves the last whole record

    def allowance() -> float | None:
        """A full limit for one of the route's arms, or None if the run has no time left for one."""
        return child_limit if time.perf_counter() - started + child_limit <= RUN_BUDGET else None

    write()
    for key in (p for p in parts if p in known):              # first the reading, on every table
        table = guarded(key, lambda key=key: known_law_readings(key, searches, cache, child_limit))  # type: ignore[misc]
        record["known_law"].append({"key": key, "readings": {}, **table})
        record["summary"] = summarise_known(record["known_law"])
        write()
    if "expansion_rate" in parts:
        record["expansion_rate"] = guarded("expansion rate", lambda: expansion_rate(searches, cache, child_limit))
        write()
    if "pipe_friction" in parts:                              # its product arm is the route's: it runs if time is left
        record["pipe_friction"] = guarded("pipe friction", lambda: pipe_friction(searches, cache, child_limit, allowance()))
        write()
    for entry in record["known_law"]:                         # then the route in force, while the run's time lasts
        if entry.get("action") == "error":
            continue
        rivals = guarded(entry["key"], lambda entry=entry: known_law_rivals(entry["key"], searches, cache, allowance))  # type: ignore[misc]
        entry.update(rivals if "error" not in rivals else {arm: dict(rivals) for arm in ROUTE_ARMS})
        record["summary"] = summarise_known(record["known_law"])
        write()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("real", "generated", "traps", "collect", "check", "child", "rehearse"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--searches", type=float, default=90.0)
    parser.add_argument("--only", default="", help="deviation rule: the parts to run alone, comma-separated")
    parser.add_argument("--job", type=Path, default=None, help="for the child command: the job's file")
    parser.add_argument("--limit", type=float, default=CHILD_LIMIT, help="for a rehearsal only: a child's limit in seconds")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    if args.command == "child":                               # started by in_a_process; the run that started it checked the digest
        job = json.loads(args.job.read_text(encoding="utf-8"))
        answer = guarded(str(job.get("what")), lambda: run_job(job))
        args.job.with_suffix(".out.json").write_text(json.dumps(answer, default=str), encoding="utf-8")
        return 0
    if args.command == "rehearse":                            # every path of the real run, without a held-back row or a digest
        globals()["held_back"] = rehearsal_tables
        args.out.mkdir(parents=True, exist_ok=True)
        description = str(route_endpoint._proposer(args.searches)[1]) if args.searches > 0 else "off"
        rehearsal: dict[str, Any] = {"rehearsal": True, "search_seconds": args.searches, "search_stage": description,
                                     "cores": os.cpu_count(), "commit": os.environ.get("GITHUB_SHA"), "child_limit": args.limit,
                                     "known_law": []}
        with tempfile.TemporaryDirectory() as scratch_folder:
            run_real((*REHEARSAL, *OPEN), REHEARSAL, args.searches, args.out / "rehearsal.json", rehearsal, Path(scratch_folder),
                     args.limit)
        print(json.dumps(rehearsal.get("summary"), indent=1))
        return 0
    changed = frozen_files_changed()
    if changed:
        raise SystemExit(f"frozen files changed since the digest was written: {changed}. Nothing was run.")
    if args.command == "check":
        print(f"{PREREGISTRATION}: every frozen file matches its digest")
        return 0
    parts = tuple(p for p in args.only.split(",") if p) or PARTS
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise SystemExit(f"not parts of the run: {unknown}. Nothing was run.")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        record: dict[str, Any] = {"preregistration": PREREGISTRATION, "commit": os.environ.get("GITHUB_SHA")}
        real = merged_real(args.out)
        if real is not None:
            record["real"] = real
        rows, cut = read_rows(sorted(args.out.glob("rows_*.jsonl")))
        if rows:
            record["generated"] = {"tables": len(rows), "lines_cut_short": cut, "summary": summarise_generated(rows), "rows": rows}
        traps, cut = read_rows(sorted(args.out.glob("traps_*.jsonl")))
        if traps:
            record["traps"] = {"tables": len(traps), "lines_cut_short": cut, "summary": summarise_generated(traps), "rows": traps}
        record["expectations"] = expectations(record)
        (args.out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        of_d = (record.get("generated") or {}).get("summary") or {}
        of_e = (record.get("traps") or {}).get("summary") or {}
        print(json.dumps({"expectations": record["expectations"], "known": (record.get("real") or {}).get("summary"),
                          "generated": {arm: of_d[arm]["all"] for arm in GENERATED_ARMS if arm in of_d},
                          "traps": (of_e.get("small") or {}).get("by_kind")}, indent=1))
        return 0
    if args.command in ("generated", "traps"):
        laws = shard_of(blind_laws() if args.command == "generated" else trap_laws(), args.shard, args.shards)
        path = args.out / f"{'rows' if args.command == 'generated' else 'traps'}_{args.shard}.jsonl"
        with path.open("w", encoding="utf-8") as handle, mp.get_context("fork").Pool(max(args.workers, 1), maxtasksperchild=8) as pool:
            for row in pool.imap(one_table, laws, chunksize=1):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
        return 0
    description = "off"
    if args.searches > 0:
        try:
            import pysr  # noqa: F401 - the arm under test has the search stage; without PySR it would silently be another arm
        except Exception as exc:  # noqa: BLE001 - whatever stops PySR loading ends the run with its reason, before any table is read
            raise SystemExit(f"PySR cannot be imported ({type(exc).__name__}: {exc}). Nothing was run.") from exc
        if (os.cpu_count() or 1) < SEARCH_CORES:
            raise SystemExit(f"{os.cpu_count()} cores: the search stage in force needs {SEARCH_CORES} for its four searches. Nothing was run.")
        description = str(route_endpoint._proposer(args.searches)[1])
    record = {"preregistration": PREREGISTRATION, "search_seconds": args.searches, "search_stage": description,
              "cores": os.cpu_count(), "commit": os.environ.get("GITHUB_SHA"), "only": list(parts) if args.only else None,
              "date_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "known_law": []}
    target = args.out / (f"real_only_{'_and_'.join(parts)}.json" if args.only else "real.json")
    with tempfile.TemporaryDirectory() as held:              # the held-back files are never beside the results
        cache = Path(held)
        record["preflight"] = preflight(parts, cache)
        run_real(parts, KNOWN_LAW, args.searches, target, record, cache, CHILD_LIMIT)
    print(json.dumps(record.get("summary"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
