"""Spread discovery jobs over processes or machines without changing a single verdict.

A queue is a directory, so any machines that share a filesystem can work on
it; nothing else is needed (no broker, no GPU).

    python -m apedp_dyn.discovery_l2.work_queue worker --root /shared/queue

Layout::

    <root>/pending/chunk-00000.json   jobs not yet claimed
    <root>/claimed/chunk-00000.json.<worker>   claimed, being computed
    <root>/results/chunk-00000.json   finished

A worker claims a chunk by renaming it out of ``pending/`` (atomic on one
filesystem, so two workers cannot claim the same chunk), computes every job in
it, writes the results with an atomic replace, and removes its claim. A chunk
whose worker died is put back with :meth:`WorkQueue.requeue_stale`.

**One writer.** Workers never touch the trial ledger. The coordinator's
:func:`collect` reads every result in job order and is the only code that
records trials, so the ledger's contents do not depend on how many workers
ran, which finished first, or on which machine. Each job carries its own seed;
:func:`discover_job` uses nothing else that varies between workers. The
verdict-identity test runs the same jobs on one worker and on several and
requires the same candidates, verdicts and trial counts
(``tests/test_discovery_work_queue.py``). Timestamps differ, so ledger files
are compared record by record without them, not byte for byte.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_referee.trial_ledger import CandidateTrialLedger

DIMENSION_FIELDS = ("m", "l", "t", "i", "th", "n", "j", "p", "q")
JobFn = Callable[[dict[str, Any]], dict[str, Any]]


def dimension_to_json(dim: Dimension) -> dict[str, str]:
    return {k: str(getattr(dim, k)) for k in DIMENSION_FIELDS if getattr(dim, k) != 0}


def dimension_from_json(data: dict[str, str]) -> Dimension:
    return Dimension(**{k: v for k, v in data.items() if k in DIMENSION_FIELDS})


def _atomic_write(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class WorkQueue:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.pending, self.claimed, self.results = (self.root / d for d in ("pending", "claimed", "results"))

    @classmethod
    def create(cls, root: Path | str, jobs: Iterable[dict[str, Any]], chunk_size: int = 1) -> WorkQueue:
        """Write the jobs as chunks. Job ids must be unique; order is kept."""
        queue = cls(root)
        if queue.root.exists() and any(queue.root.iterdir()):
            raise FileExistsError(f"{queue.root} is not empty; a queue is created once")
        for d in (queue.pending, queue.claimed, queue.results):
            d.mkdir(parents=True, exist_ok=True)
        jobs = list(jobs)
        ids = [job["job_id"] for job in jobs]
        if len(set(ids)) != len(ids):
            raise ValueError("job ids must be unique")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        manifest = {"jobs": ids, "chunks": []}
        for n, start in enumerate(range(0, len(jobs), chunk_size)):
            name = f"chunk-{n:05d}.json"
            _atomic_write(queue.pending / name, jobs[start:start + chunk_size])
            manifest["chunks"].append(name)
        _atomic_write(queue.root / "manifest.json", manifest)
        return queue

    def manifest(self) -> dict[str, Any]:
        return dict(json.loads((self.root / "manifest.json").read_text(encoding="utf-8")))

    def claim(self, worker_id: str) -> Path | None:
        for chunk in sorted(self.pending.glob("chunk-*.json")):
            target = self.claimed / f"{chunk.name}.{worker_id}"
            try:
                os.rename(chunk, target)
            except FileNotFoundError:
                continue  # another worker took it first
            return target
        return None

    def complete(self, claim: Path, results: list[dict[str, Any]]) -> None:
        chunk_name = claim.name.split(".json.", 1)[0] + ".json"
        _atomic_write(self.results / chunk_name, results)
        claim.unlink()

    def requeue_stale(self, older_than_seconds: float) -> list[str]:
        """Put back chunks claimed longer ago than this (their worker is presumed dead)."""
        now, back = time.time(), []
        for claim in sorted(self.claimed.glob("chunk-*.json.*")):
            chunk_name = claim.name.split(".json.", 1)[0] + ".json"
            if (self.results / chunk_name).exists():
                claim.unlink(missing_ok=True)
                continue
            if now - claim.stat().st_mtime > older_than_seconds:
                try:
                    os.rename(claim, self.pending / chunk_name)
                    back.append(chunk_name)
                except FileNotFoundError:
                    pass
        return back

    def done(self) -> bool:
        return all((self.results / name).exists() for name in self.manifest()["chunks"])


def run_worker(root: Path | str, fn: JobFn, worker_id: str | None = None) -> int:
    """Claim and compute chunks until none are pending; returns the number of jobs done."""
    queue = WorkQueue(root)
    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    done = 0
    while (claim := queue.claim(worker_id)) is not None:
        jobs = json.loads(claim.read_text(encoding="utf-8"))
        queue.complete(claim, [fn(job) for job in jobs])
        done += len(jobs)
    return done


def collect(
    root: Path | str,
    ledger: CandidateTrialLedger | None = None,
    to_trial: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Every result in job order; the only place trials are recorded."""
    queue = WorkQueue(root)
    manifest = queue.manifest()
    missing = [name for name in manifest["chunks"] if not (queue.results / name).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} chunk(s) not finished: {missing[:5]}")
    by_id: dict[str, dict[str, Any]] = {}
    for name in manifest["chunks"]:
        for result in json.loads((queue.results / name).read_text(encoding="utf-8")):
            by_id[result["job_id"]] = result
    if set(by_id) != set(manifest["jobs"]):
        raise RuntimeError("results do not match the jobs the queue was created with")
    ordered = [by_id[job_id] for job_id in manifest["jobs"]]
    if ledger is not None:
        convert = to_trial or discovery_trial
        for result in ordered:
            ledger.record_trial(**convert(result))
    return ordered


def discover_job(job: dict[str, Any]) -> dict[str, Any]:
    """One open-discovery call; everything it depends on is in ``job``.

    ``job``: ``job_id``, ``data`` (an ``.npz`` path with one array per variable
    and ``y``), ``dimensions`` and ``target_dimension`` (as
    :func:`dimension_to_json`), ``seed``, optional ``noise_sigma``,
    ``operators`` and ``engine`` (constructor options).
    """
    from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine

    arrays = np.load(job["data"])
    dims = {k: dimension_from_json(v) for k, v in job["dimensions"].items()}
    X = {k: np.asarray(arrays[k], dtype=float) for k in dims}
    engine = L2DiscoveryEngine(operators=tuple(job.get("operators", ("+", "-", "*", "/", "pow"))),
                               **job.get("engine", {}))
    result = engine.discover(
        X=X, Y=np.asarray(arrays["y"], dtype=float), dimensions=dims,
        target_dimension=dimension_from_json(job["target_dimension"]), target_name="y",
        noise_sigma=job.get("noise_sigma"), seed=int(job["seed"]),
    )
    adj = result.adjudication
    return {
        "job_id": job["job_id"],
        "status": str(getattr(adj.status, "value", adj.status)),
        "attribution": str(adj.epistemic_attribution),
        "promoted": adj.promoted.candidate.candidate_id if adj.promoted is not None else None,
        "ranking": [c.candidate.candidate_id for c in adj.ranking],
        "generated": len(result.generated_nodes),
    }


def discovery_trial(result: dict[str, Any]) -> dict[str, Any]:
    """A :func:`discover_job` result as ``record_trial`` keyword arguments."""
    return {
        "candidate_id": f"{result['job_id']}:{result['promoted'] or 'none'}",
        "proposer_id": "l2_discovery_queue",
        "formula": result["promoted"] or "",
        "strategy_archetype": "equation_discovery",
        "in_sample_sharpe": 0.0,
        "status": result["status"],
        "selection_statistic": float("nan"),
        "metadata": {"job_id": result["job_id"], "attribution": result["attribution"],
                     "search_id": result["job_id"], "selection_pool_size": max(1, int(result["generated"]))},
    }


def _resolve(spec: str) -> JobFn:
    module, _, name = spec.partition(":")
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker", help="claim and compute chunks until none are pending")
    w.add_argument("--root", type=Path, required=True)
    w.add_argument("--job-fn", default="apedp_dyn.discovery_l2.work_queue:discover_job")
    w.add_argument("--worker-id")
    r = sub.add_parser("requeue", help="put back chunks whose worker stopped")
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--older-than", type=float, default=3600.0)
    args = parser.parse_args(argv)
    if args.command == "worker":
        print(f"{run_worker(args.root, _resolve(args.job_fn), args.worker_id)} jobs done")
    else:
        print("requeued:", ", ".join(WorkQueue(args.root).requeue_stale(args.older_than)) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
