#!/usr/bin/env python3
"""Fault-tolerant preparation of the Steane code's logical zero, for PREREG-FT-PREP-2026-11.

Part A, any pair of qubits may interact: every shortest encoder (up to the
code's symmetries and the order of gates on disjoint qubits), every weight-three
check, every order of its CNOTs; each candidate certified by enumeration.

Part B, a 2-by-4 grid with nearest-neighbour CNOTs: seven data qubits and one
ancilla on a site with three neighbours, over every inequivalent way of laying
the code on the grid.

Part C, an outside simulator: selected circuits are sampled in Stim under the
same noise model and the rates compared with the exact coefficients.

    python scripts/run_prereg_ft_prep.py             # the preregistered run, once
    python scripts/run_prereg_ft_prep.py --development --encoders 2000 --shots 2000000
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.vqc import fault_tolerant_prep as ft  # noqa: E402

PREREGISTRATION_ID = "PREREG-FT-PREP-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-FT-PREP.json"
FROZEN = ("preregistration/PREREG-FT-PREP-2026-11.md", "scripts/run_prereg_ft_prep.py",
          "apedp_dyn/vqc/fault_tolerant_prep.py")
OUT = ROOT / "preregistration" / "RESULT-FT-PREP-2026-11"

P_CHECK = 0.001
SHOTS = 200_000_000
SHOTS_BARE = 20_000_000
GRID_ROWS, GRID_COLS = 2, 4
GRID_ANCILLA_SITE = 1          # a site with three neighbours; the four such sites are equivalent under the grid's symmetries
GRID_ENCODER_LIMIT = 20_000    # raw gate sequences per way of laying the code on the grid, before duplicates are removed
WORKERS = 2
CHUNK = 2_000


def circuit_record(circuit: ft.PrepCircuit, certificate: ft.Certificate) -> dict[str, Any]:
    return {"plus": list(circuit.plus), "cnots": [list(g) for g in circuit.cnots], "ancillas": circuit.n_anc,
            "cnot_count": len(circuit.cnots), "c1": certificate.c1, "c2": certificate.c2,
            "rejection_c1": certificate.rejection_c1, "fault_tolerant": certificate.fault_tolerant,
            "prepares_state": certificate.prepares_state, "single_faults": certificate.single_faults,
            "fault_pairs": certificate.fault_pairs, "sha256_of_ir": circuit.to_circuit().sha256()}


def survey_chunk(job: tuple[ft.CssCode, list[ft.PrepCircuit], list[int] | None]) -> dict[str, Any]:
    """Certify one chunk of encoders and their completions; keep counts, every c2, and one circuit per distinct c2."""
    code, encoders, support = job
    bare_ft = completed = 0
    c2: list[float] = []
    example: dict[float, tuple[ft.PrepCircuit, ft.Certificate]] = {}
    costs: dict[int, int] = {}
    for encoder in encoders:
        bare_ft += ft.certify(encoder, code).fault_tolerant
        found = ft.verified(encoder, code, max_ancillas=1, allowed_support=support)
        completed += bool(found)
        for circuit, certificate in found:
            c2.append(certificate.c2)
            costs[len(circuit.cnots)] = costs.get(len(circuit.cnots), 0) + 1
            key = round(certificate.c2, 9)
            if key not in example:
                example[key] = (circuit, certificate)
    return {"encoders": len(encoders), "bare_fault_tolerant": bare_ft, "encoders_completed": completed,
            "c2": np.array(c2, dtype=np.float64), "example": example, "costs": costs}


def survey(code: ft.CssCode, encoders: list[ft.PrepCircuit], support: list[int] | None, pool: Any) -> dict[str, Any]:
    jobs = [(code, encoders[i: i + CHUNK], support) for i in range(0, len(encoders), CHUNK)]
    parts = pool.map(survey_chunk, jobs) if pool is not None else [survey_chunk(job) for job in jobs]
    c2 = np.concatenate([part["c2"] for part in parts]) if parts else np.array([])
    example: dict[float, tuple[ft.PrepCircuit, ft.Certificate]] = {}
    costs: dict[int, int] = {}
    for part in parts:
        for key, value in part["example"].items():
            example.setdefault(key, value)
        for key, value in part["costs"].items():
            costs[key] = costs.get(key, 0) + value
    return {"encoders": sum(part["encoders"] for part in parts),
            "bare_fault_tolerant": sum(part["bare_fault_tolerant"] for part in parts),
            "encoders_completed": sum(part["encoders_completed"] for part in parts),
            "fault_tolerant_circuits": len(c2), "c2": c2, "example": example, "costs": costs}


def c2_summary(c2: np.ndarray) -> dict[str, Any]:
    if not len(c2):
        return {"count": 0}
    values, counts = np.unique(np.round(c2, 9), return_counts=True)
    return {"count": len(c2), "min": float(c2.min()), "median": float(np.median(c2)), "max": float(c2.max()),
            "quartiles": [float(v) for v in np.percentile(c2, [25, 50, 75])],
            "distinct_values": len(values), "histogram": {f"{v:.6f}": int(k) for v, k in zip(values, counts, strict=True)}}


def grid_layout() -> tuple[list[tuple[int, int]], int]:
    """Data-data edges of the grid in data-qubit numbering, and the mask of data qubits next to the ancilla."""
    sites = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    index = {site: k for k, site in enumerate(sites)}
    edges = [(index[(r, c)], index[(r, c + 1)]) for r in range(GRID_ROWS) for c in range(GRID_COLS - 1)]
    edges += [(index[(r, c)], index[(r + 1, c)]) for r in range(GRID_ROWS - 1) for c in range(GRID_COLS)]
    data_sites = [k for k in range(len(sites)) if k != GRID_ANCILLA_SITE]
    number = {site: q for q, site in enumerate(data_sites)}
    data_edges = [(number[a], number[b]) for a, b in edges if GRID_ANCILLA_SITE not in (a, b)]
    support = sum(1 << number[b if a == GRID_ANCILLA_SITE else a] for a, b in edges if GRID_ANCILLA_SITE in (a, b))
    return data_edges, support


def layings(code: ft.CssCode) -> list[ft.CssCode]:
    """The code under every relabelling of its qubits, one per distinct code."""
    seen: dict[frozenset[int], ft.CssCode] = {}
    for order in itertools.permutations(range(code.n)):
        hx = tuple(sorted(ft._permuted(v, order) for v in code.hx))
        hz = tuple(sorted(ft._permuted(v, order) for v in code.hz))
        relabelled = ft.CssCode(f"{code.name}@{''.join(map(str, order))}", code.n, hx, hz)
        seen.setdefault(frozenset(relabelled.x_space), relabelled)
    return list(seen.values())


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--development", action="store_true", help="a cut-down run on other settings; writes nothing to the repository")
    parser.add_argument("--encoders", type=int, default=None, help="development only: how many encoders to survey")
    parser.add_argument("--shots", type=int, default=None, help="development only: Stim shots per circuit")
    parser.add_argument("--p", type=float, default=None, help="development only: the noise level of the outside check")
    args = parser.parse_args(argv)
    if not args.development:
        if not DIGEST.exists() or json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
            raise SystemExit(f"{PREREGISTRATION_ID} is not frozen, or changed since it was frozen")
        OUT.mkdir(exist_ok=False)
    p = P_CHECK if not args.development else (args.p or 0.01)
    shots = SHOTS if not args.development else (args.shots or 2_000_000)
    shots_bare = SHOTS_BARE if not args.development else max(shots // 10, 100_000)
    started = time.time()
    code = ft.steane()
    results: dict[str, Any] = {"preregistration": PREREGISTRATION_ID, "development": bool(args.development)}
    selected: dict[str, tuple[ft.PrepCircuit, ft.Certificate, ft.CssCode]] = {}
    with mp.get_context("fork").Pool(WORKERS) as pool:
        # ---- Part A: any pair may interact
        encoders = ft.shortest_encoders(code, limit=args.encoders if args.development else None)
        a = survey(code, encoders, None, pool)
        ordered = sorted(a["example"].items())
        a_summary = c2_summary(a["c2"])
        results["all_to_all"] = {"encoder_cnots": len(encoders[0].cnots), "encoders": a["encoders"],
                                 "bare_fault_tolerant": a["bare_fault_tolerant"], "encoders_completed": a["encoders_completed"],
                                 "fault_tolerant_circuits": a["fault_tolerant_circuits"], "cnot_counts": a["costs"], "c2": a_summary}
        if ordered:
            median_key = min(a["example"], key=lambda k: (abs(k - a_summary["median"]), k))
            selected["all_to_all_lowest_c2"] = (*ordered[0][1], code)
            selected["all_to_all_median_c2"] = (*a["example"][median_key], code)
            selected["all_to_all_highest_c2"] = (*ordered[-1][1], code)
        print(f"part A: {a['encoders']} encoders, {a['fault_tolerant_circuits']} fault-tolerant circuits, {time.time() - started:.0f} s", flush=True)
        # ---- Part B: the grid
        edges, support = grid_layout()
        per_laying = []
        best: tuple[int, float, ft.PrepCircuit, ft.Certificate, ft.CssCode] | None = None
        for laid in layings(code):
            listed = ft.shortest_encoders(laid, edges, limit=GRID_ENCODER_LIMIT if not args.development else 300)
            if not listed:
                per_laying.append({"laying": laid.name, "encoder_cnots": None, "encoders": 0, "fault_tolerant_circuits": 0})
                continue
            b = survey(laid, listed, [support], pool)
            entry = {"laying": laid.name, "encoder_cnots": len(listed[0].cnots), "encoders": b["encoders"],
                     "limit_reached": len(listed) >= (GRID_ENCODER_LIMIT if not args.development else 300),
                     "fault_tolerant_circuits": b["fault_tolerant_circuits"], "cnot_counts": b["costs"],
                     "c2_min": float(b["c2"].min()) if len(b["c2"]) else None}
            per_laying.append(entry)
            if b["example"]:
                key, (circuit, certificate) = min(b["example"].items(), key=lambda kv: (len(kv[1][0].cnots), kv[0]))
                if best is None or (len(circuit.cnots), key) < (best[0], best[1]):
                    best = (len(circuit.cnots), key, circuit, certificate, laid)
        results["grid"] = {"rows": GRID_ROWS, "cols": GRID_COLS, "ancilla_site": GRID_ANCILLA_SITE, "data_edges": edges,
                           "ancilla_neighbours_mask": support, "layings": per_laying,
                           "layings_with_a_fault_tolerant_circuit": sum(e["fault_tolerant_circuits"] > 0 for e in per_laying)}
        if best is not None:
            selected["grid_best"] = (best[2], best[3], best[4])
        print(f"part B: {len(per_laying)} layings, {results['grid']['layings_with_a_fault_tolerant_circuit']} with a fault-tolerant circuit, {time.time() - started:.0f} s", flush=True)
    # ---- Part C: the outside simulator
    checks = {}
    for seed, (name, (circuit, certificate, its_code)) in enumerate(selected.items()):
        sampled = ft.stim_logical_error(circuit, its_code, p, shots, seed=1000 + seed)
        checks[name] = {"circuit": circuit_record(circuit, certificate), "laying": its_code.name, "p": p,
                        "predicted": certificate.logical_error_rate(p), "stim": sampled,
                        "ratio": sampled["rate"] / certificate.logical_error_rate(p)}
        print(f"part C: {name}: predicted {checks[name]['predicted']:.4e}, sampled {sampled['rate']:.4e} ± {sampled['standard_error']:.1e}", flush=True)
    if "all_to_all_lowest_c2" in selected:
        circuit, certificate, its_code = selected["all_to_all_lowest_c2"]
        bare = ft.PrepCircuit(circuit.n_data, 0, circuit.plus, circuit.cnots[: results["all_to_all"]["encoder_cnots"]])
        bare_certificate = ft.certify(bare, its_code)
        sampled = ft.stim_logical_error(bare, its_code, p, shots_bare, seed=2000)
        checks["bare_encoder_of_lowest_c2"] = {"circuit": circuit_record(bare, bare_certificate), "p": p,
                                               "predicted": bare_certificate.c1 * p, "stim": sampled,
                                               "ratio": sampled["rate"] / (bare_certificate.c1 * p)}
    results["outside_check"] = checks
    results["seconds"] = round(time.time() - started, 1)
    text = json.dumps(results, indent=1, sort_keys=True, default=str) + "\n"
    if args.development:
        print(text[:3000])
    else:
        results["frozen"] = frozen_digests()
        (OUT / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"wrote {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
