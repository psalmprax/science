"""End-to-end corpus pipeline: seeds -> fetch -> normalize -> canonicalize -> classify -> emit.

fetch_fn(page) -> list[str] injectable for offline tests. Default uses the
existing Wikipedia harvester (network, with timeout/quarantine).
"""
from __future__ import annotations

import logging

from apedp.equations.canonicalize import group_representations
from apedp.equations.classify import classify_group
from apedp.equations.emit import build_graph, build_records, yield_metrics
from apedp.equations.normalize import clean_latex, latex_to_sympy_str, split_alternatives
from apedp.equations.seeds import seed_pages

logger = logging.getLogger(__name__)


#: Failures a fetch stage may legitimately hit: the harvester extra is absent,
#: the network failed, or the page returned something unusable. Anything else
#: is a defect and must propagate (see docs/INDEPENDENT_REVIEW_2026-09-23.md
#: §4.2).
_FETCH_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError,
                 KeyError)


def _default_fetch(page: str) -> list:
    try:
        from apedp.ingestion.ingestion_web_equations import harvest_wikipedia_equations
        return harvest_wikipedia_equations(page)
    except _FETCH_ERRORS as exc:
        logger.debug("harvest failed for %r: %s", page, exc)
        return []


def run_pipeline(seeds: list | None = None, fetch_fn=None,
                 corpus: dict | None = None) -> dict:
    seeds = seeds if seeds is not None else seed_pages()
    fetch_fn = fetch_fn or _default_fetch
    all_records, all_edges = [], []
    fetched = parsed = quarantined = 0

    for seed in seeds:
        try:
            raw_list = fetch_fn(seed["page"])
        except _FETCH_ERRORS as exc:
            logger.debug("fetch_fn failed for %r: %s", seed["page"], exc)
            raw_list = []
        fetched += len(raw_list)
        exprs = []
        for raw in raw_list:
            for alt in split_alternatives(raw):
                cleaned = clean_latex(alt)
                if not cleaned:
                    continue
                ok, expr_str, _ = latex_to_sympy_str(cleaned)
                if ok:
                    exprs.append(expr_str)
                    parsed += 1
                else:
                    quarantined += 1
        groups = group_representations(exprs)
        classes = [classify_group(g["canonical"], g["members"], corpus) for g in groups]
        records = build_records(groups, seed, classes)
        all_records.extend(records)
        all_edges.extend(build_graph(records))

    metrics = yield_metrics(fetched, parsed, len(all_records), quarantined)
    return {"records": all_records, "edges": all_edges, "metrics": metrics}
